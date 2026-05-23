"""ST-QNet mechanisms on canonical input (cnxxlquat 79.46 warm-start).

Implements the three quaternion components from the journal paper:
  1. Cycle consistency loss (Sec 4.4): dedicated Mamba head + manifold
     projection -> Q_pred at 3 temporal anchors, ground-truth Q_act from
     per-frame inertia covariance (simplified vs paper's per-point), enforce
     forward+backward cycle composition close to identity.
  2. Bearing quaternion rigidity descriptor (Sec 4.6): per-point bearing
     quaternion (rotation from north pole to direction-from-bbox-centroid),
     frame-to-frame angular consistency -> rigidity score r_i in [0, 1],
     learned projection f_theta (init 0) -> multiplicative feature modulation.
  3. Reconstruction grounding loss (Sec 4.7): pairwise rotation Q_pred must
     align centred source point cloud to centred target -- prevents cycle
     collapse to identity.

Loss: L_total = L_cls + lambda_cycle * (L_cycle + L_recon), lambda_cycle=0.8.
Aux loss exposed via self.aux_loss; main.py adds to total loss.

Warm-start preserved: all new learnable modules with non-zero gradient flow
either gated by lambda_cycle weight or zero-initialized (rigidity_proj).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.motion_cleanest_quat_head import MotionCleanestLinXLQuatHead
from models.motion_cleanest_quat import _inertia_quat


def _hamilton(q1, q2):
    """Quaternion product q1 * q2 in (w, x, y, z), broadcast-friendly."""
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    return torch.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], dim=-1)


def _conj(q):
    w, x, y, z = q.unbind(-1)
    return torch.stack([w, -x, -y, -z], dim=-1)


def _exp_map(v):
    """Axis-angle vector v in R^3 -> unit quaternion via exponential map.
    v: (..., 3). q = [cos(|v|/2), sin(|v|/2) * v/|v|].
    """
    norm = v.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    half = norm * 0.5
    w = torch.cos(half)
    xyz = torch.sin(half) * (v / norm)
    return torch.cat([w, xyz], dim=-1)


def _quat_rotate(q, p):
    """Rotate 3D points p (..., 3) by unit quaternion q (..., 4). Vectorized."""
    qw = q[..., 0:1]
    qv = q[..., 1:4]
    # q * p (treating p as pure quat) * q*
    # Standard Rodrigues form
    t = 2.0 * torch.cross(qv, p, dim=-1)
    return p + qw * t + torch.cross(qv, t, dim=-1)


def _bearing_quat(xyz):
    """xyz: (..., 3). Returns unit quaternion rotating north-pole [0,1,0]
    to the unit direction of xyz from origin. Bearing quaternion per Sec 4.6.
    """
    north = torch.tensor([0.0, 1.0, 0.0], device=xyz.device, dtype=xyz.dtype)
    d = F.normalize(xyz, dim=-1, eps=1e-6)
    # axis = north x d; angle = acos(north . d)
    cos_theta = (d * north).sum(dim=-1).clamp(-1 + 1e-6, 1 - 1e-6)
    theta = torch.acos(cos_theta)
    axis = torch.cross(north.expand_as(d), d, dim=-1)
    axis = F.normalize(axis, dim=-1, eps=1e-6)
    half = (theta * 0.5).unsqueeze(-1)
    return torch.cat([torch.cos(half), torch.sin(half) * axis], dim=-1)


class MotionCleanestLinXLSTQNet(MotionCleanestLinXLQuatHead):
    def __init__(self, *args,
                 lambda_cycle=0.8,
                 cycle_hidden=128,
                 fea3_channels=260,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.lambda_cycle = float(lambda_cycle)

        # Dedicated cycle-head Mamba: takes per-frame features, outputs 3D
        # axis-angle vector per frame.
        self.cycle_proj_in = nn.Linear(fea3_channels, cycle_hidden)
        self.cycle_temporal = nn.GRU(  # lightweight alt to a second Mamba
            cycle_hidden, cycle_hidden, num_layers=1,
            bidirectional=True, batch_first=True,
        )
        self.cycle_proj_out = nn.Linear(2 * cycle_hidden, 3)
        with torch.no_grad():
            # Init cycle_proj_out small so initial Q_pred is near identity.
            nn.init.zeros_(self.cycle_proj_out.weight)
            nn.init.zeros_(self.cycle_proj_out.bias)

        # Bearing rigidity projection: scalar rigidity score per point ->
        # per-channel modulation factor for fea3. Init zero: model starts
        # identical to baseline.
        self.rigidity_proj = nn.Linear(1, fea3_channels, bias=True)
        with torch.no_grad():
            nn.init.zeros_(self.rigidity_proj.weight)
            nn.init.zeros_(self.rigidity_proj.bias)

        self.aux_loss = None

    def no_decay_param_names(self):
        return [
            'cycle_proj_out.weight', 'cycle_proj_out.bias',
            'rigidity_proj.weight', 'rigidity_proj.bias',
        ]

    def _cycle_consistency_loss(self, per_frame_feat, coords):
        """Per-frame Q_pred from feature stream; Q_act from coords inertia.

        per_frame_feat: (B, T, F)  (mean-pooled fea3 over points)
        coords: (B, 4, T, P)  raw input coords (xyz + t)
        """
        B, T, F_dim = per_frame_feat.shape
        # Cycle head: temporal GRU -> per-frame 3D axis-angle -> exp map -> quat
        h = self.cycle_proj_in(per_frame_feat)
        h, _ = self.cycle_temporal(h)
        v = self.cycle_proj_out(h)                                      # (B, T, 3)
        Q_pred = _exp_map(v)                                            # (B, T, 4)

        # Q_act: per-frame inertia quat (paper uses per-point, we use per-frame).
        Q_act = _inertia_quat(coords[:, :3]).reshape(B, T, 4)

        # 3 temporal anchors at thirds of T.
        t_a = 0
        t_b = T // 3 + (T % 3 > 0)
        t_c = (2 * T) // 3 + (2 * T % 3 > 0)
        t_c = min(t_c, T - 1)

        Qp_a, Qp_b, Qp_c = Q_pred[:, t_a], Q_pred[:, t_b], Q_pred[:, t_c]
        Qa_a, Qa_b, Qa_c = Q_act[:, t_a], Q_act[:, t_b], Q_act[:, t_c]

        # Forward cycle: q_fwd = (Q_b * Q_a^-1) o (Q_c * Q_b^-1)
        q_hop1_pred = _hamilton(Qp_b, _conj(Qp_a))
        q_hop2_pred = _hamilton(Qp_c, _conj(Qp_b))
        q_fwd_pred = _hamilton(q_hop2_pred, q_hop1_pred)

        q_hop1_act = _hamilton(Qa_b, _conj(Qa_a))
        q_hop2_act = _hamilton(Qa_c, _conj(Qa_b))
        q_fwd_act = _hamilton(q_hop2_act, q_hop1_act)

        # Cycle residual: q_fwd_pred should match q_fwd_act. Use double-cover-
        # invariant L2.
        identity = torch.zeros_like(q_fwd_pred); identity[..., 0] = 1.0
        q_fwd_residual = _hamilton(q_fwd_pred, _conj(q_fwd_act))
        eps_fwd = torch.minimum(
            (q_fwd_residual - identity).pow(2).sum(-1),
            (q_fwd_residual + identity).pow(2).sum(-1),
        )

        # Backward cycle: same with q^-1 ordering
        q_hop1_bwd = _hamilton(Qp_b, _conj(Qp_c))
        q_hop2_bwd = _hamilton(Qp_a, _conj(Qp_b))
        q_bwd_pred = _hamilton(q_hop2_bwd, q_hop1_bwd)
        q_hop1_bwd_a = _hamilton(Qa_b, _conj(Qa_c))
        q_hop2_bwd_a = _hamilton(Qa_a, _conj(Qa_b))
        q_bwd_act = _hamilton(q_hop2_bwd_a, q_hop1_bwd_a)
        q_bwd_residual = _hamilton(q_bwd_pred, _conj(q_bwd_act))
        eps_bwd = torch.minimum(
            (q_bwd_residual - identity).pow(2).sum(-1),
            (q_bwd_residual + identity).pow(2).sum(-1),
        )

        cycle_loss = (eps_fwd + eps_bwd).mean()

        # Unit-norm constraint (Sec 4.4 L_unit).
        unit_loss = ((Q_pred.norm(dim=-1) - 1.0).pow(2)).mean()

        # Reconstruction grounding (Sec 4.7).
        # For pair (a, c): q must align centred coords at frame a to frame c.
        xyz = coords[:, :3]  # (B, 3, T, P)
        xyz = xyz.permute(0, 2, 3, 1)  # (B, T, P, 3)
        Xa = xyz[:, t_a]; Xc = xyz[:, t_c]
        # Centre
        Xa_c = Xa - Xa.mean(dim=1, keepdim=True)
        Xc_c = Xc - Xc.mean(dim=1, keepdim=True)
        # Rotate Xa_c by predicted hop-from-a-to-c quat
        q_ac_pred = _hamilton(Qp_c, _conj(Qp_a))                        # (B, 4)
        # Broadcast quat over P points
        q_ac_b = q_ac_pred.unsqueeze(1).expand(-1, Xa_c.shape[1], -1)   # (B, P, 4)
        Xa_rot = _quat_rotate(q_ac_b, Xa_c)
        recon_loss = (Xa_rot - Xc_c).pow(2).sum(-1).mean()

        return cycle_loss + 0.1 * unit_loss + recon_loss

    def _bearing_rigidity_modulation(self, fea3, coords):
        """Compute per-point rigidity from coords, multiplicatively modulate fea3.

        fea3: (B, C, T, P_eff)
        coords: (B, 4, T, P)
        Returns modulated fea3 (zero-init projection -> baseline preserved).
        """
        B, C, T, P_eff = fea3.shape
        xyz = coords[:, :3]                                              # (B, 3, T, P)
        # Center by bbox centroid per frame.
        centroid = xyz.mean(dim=-1, keepdim=True)                        # (B, 3, T, 1)
        d = xyz - centroid                                                # (B, 3, T, P)
        d_t = d.permute(0, 2, 3, 1)                                       # (B, T, P, 3)
        # Bearing quat per point per frame.
        bq = _bearing_quat(d_t)                                            # (B, T, P, 4)
        # Frame-to-frame angular change: q_t+1 ⊗ q_t^-1.
        bq_a = bq[:, 1:]                                                  # (B, T-1, P, 4)
        bq_b = bq[:, :-1]
        delta = _hamilton(bq_a, _conj(bq_b))                              # (B, T-1, P, 4)
        # Geodesic distance to identity: 2 * acos(|w|).
        w = delta[..., 0].abs().clamp(max=1 - 1e-6)
        dgeo = 2.0 * torch.acos(w)                                         # (B, T-1, P)
        # Rigidity score per point: r_p = exp(-mean_t(dgeo) / median).
        mean_dgeo = dgeo.mean(dim=1)                                       # (B, P)
        med = mean_dgeo.median(dim=-1, keepdim=True).values.clamp(min=1e-6)
        r = torch.exp(-mean_dgeo / med)                                    # (B, P)

        # Downsample r to match fea3's P_eff via linear interpolation (or
        # take first P_eff if same point order). The spatial pipeline does
        # learned downsampling on point indices, so we approximate by simply
        # averaging consecutive points until we hit P_eff.
        if r.shape[-1] != P_eff:
            ratio = r.shape[-1] // P_eff
            if ratio > 0:
                r = r[:, :P_eff * ratio].reshape(B, P_eff, ratio).mean(-1)
            else:
                r = F.interpolate(r.unsqueeze(1), size=P_eff, mode='linear',
                                  align_corners=False).squeeze(1)
        # f_theta(r): (B, P_eff, C)
        mod = self.rigidity_proj(r.unsqueeze(-1))                          # (B, P_eff, C)
        mod = mod.permute(0, 2, 1).unsqueeze(2).expand(B, C, T, P_eff)     # (B, C, T, P_eff)
        return fea3 * (1.0 + mod)

    def forward(self, inputs):
        if isinstance(inputs, dict):
            inputs = inputs['points']
        coords = self._sample_points(inputs)                               # (B, 4, T, P)
        B = coords.shape[0]; T = coords.shape[2]

        fea3 = self._encode_sampled_points(coords)                         # (B, C, T, P_eff)
        # Apply bearing rigidity modulation (zero-init projection at start).
        fea3 = self._bearing_rigidity_modulation(fea3, coords)

        # Per-frame feature for cycle head (mean-pool over points).
        per_frame_feat = fea3.mean(dim=-1).transpose(1, 2)                 # (B, T, C)

        # Main classifier path.
        output = self.stage5(fea3)
        output = self.pool5(output)
        output = self.global_bn(output)
        main_features = output.flatten(1)
        main_logits = self.classify_features(main_features)

        # Inertia quat-head aux (the existing parent head).
        quat_traj = _inertia_quat(coords[:, :3]).reshape(B, T * 4)
        inertia_logits = self.quat_head(quat_traj)

        # ST-QNet cycle + recon aux loss.
        if self.training and self.lambda_cycle > 0:
            cyc_recon = self._cycle_consistency_loss(per_frame_feat, coords)
            self.aux_loss = self.lambda_cycle * cyc_recon
        else:
            self.aux_loss = None

        return main_logits + self.quat_head_scale * inertia_logits

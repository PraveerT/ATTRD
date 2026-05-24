"""ST-QNet-C1: cluster-rotation cycle consistency.

Hand motion decomposes into K quasi-rigid segments (palm + 5 fingers).
For each segment k:
  Q_act^k(t) = inertia quat of cluster k at frame t (from weighted cov)
  Q_pred^k(t) = predicted via per-cluster temporal head

Mechanism:
  1. Per-point soft cluster assignment alpha_p^k(t) via attention on
     post-spatial features (B, T, P_eff, K) -> softmax.
  2. Per-cluster centroid c^k(t) and weighted covariance C^k(t) from
     downsampled coords. Q_act^k(t) = principal axis quat via eigh.
  3. Per-cluster feature aggregate -> bidir GRU on time -> 3D axis-angle
     -> exp map -> Q_pred^k(t).
  4. Direct supervision L_cycle = double-cover-invariant ||Q_pred - Q_act||^2
     averaged over clusters and time.
  5. Reconstruction grounding L_recon: for anchor pair (a, c), per cluster,
     rotating centered cluster points at frame a by predicted hop quat
     q_ac^k = Q_pred^k(c) (X) Q_pred^k(a)^{-1} should match centered
     cluster points at frame c, weighted by alpha^k(c).
  6. Load balancing: maximize entropy of mean cluster usage to prevent
     collapse to one cluster.

Why this works where paper fails: per-segment rigid motion holds, so the
quaternion math is physically meaningful. Articulation is handled by the
soft cluster assignment naturally.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.motion_cleanest_quat_head import MotionCleanestLinXLQuatHead
from models.motion_cleanest_quat import _inertia_quat


def _hamilton(q1, q2):
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
    norm = v.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    half = norm * 0.5
    return torch.cat([torch.cos(half), torch.sin(half) * (v / norm)], dim=-1)


def _quat_rotate(q, p):
    qw = q[..., 0:1]; qv = q[..., 1:4]
    t = 2.0 * torch.cross(qv, p, dim=-1)
    return p + qw * t + torch.cross(qv, t, dim=-1)


class MotionCleanestLinXLSTQNetC1(MotionCleanestLinXLQuatHead):
    def __init__(self, *args,
                 K=6,
                 lambda_cycle=0.1,
                 lambda_recon=0.05,
                 lambda_balance=0.01,
                 cluster_feat_dim=256,
                 cluster_hidden=64,
                 cluster_alpha_mode='per_frame',
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.K = int(K)
        self.lambda_cycle = float(lambda_cycle)
        self.lambda_recon = float(lambda_recon)
        self.lambda_balance = float(lambda_balance)
        # 'per_frame' (default): alpha computed independently per frame.
        # 'time_stable': alpha computed from time-pooled features once per
        # sample then broadcast over T. Needed when input lacks point
        # correspondence (e.g. raw NvidiaLoader vs canonical).
        self.cluster_alpha_mode = str(cluster_alpha_mode)

        # Cluster assignment head: per-point feature (256-d post-Mamba) -> K logits.
        self.cluster_head = nn.Sequential(
            nn.Linear(cluster_feat_dim, cluster_hidden),
            nn.GELU(),
            nn.Linear(cluster_hidden, self.K),
        )

        # Per-cluster cycle head: GRU on cluster-aggregate features over time.
        self.cycle_gru = nn.GRU(
            cluster_feat_dim, cluster_hidden,
            num_layers=1, batch_first=True, bidirectional=True,
        )
        self.cycle_proj = nn.Linear(2 * cluster_hidden, 3)
        with torch.no_grad():
            # Zero-init so initial Q_pred ~ identity -> warm-start preserved.
            nn.init.zeros_(self.cycle_proj.weight)
            nn.init.zeros_(self.cycle_proj.bias)

        self.aux_loss = None

    def no_decay_param_names(self):
        # cluster_head dies if subjected to weight decay -- its only gradient
        # path is through softmax + cycle loss, magnitudes too small to
        # counteract Adam's wd-driven zeroing. Exempt entirely.
        return [
            'cycle_proj.weight', 'cycle_proj.bias',
            'cluster_head.0.weight', 'cluster_head.0.bias',
            'cluster_head.2.weight', 'cluster_head.2.bias',
        ]

    def _compute_cluster_quats(self, fea3):
        """fea3: (B, 4+F, T, P_eff). Returns (alpha, xyz_centered, Q_act, Q_pred).
        alpha: (B, T, P_eff, K), xyz_centered: (B, T, P_eff, K, 3),
        Q_act/Q_pred: (B, T, K, 4).
        """
        B, _, T, P_eff = fea3.shape
        coords_eff = fea3[:, :4]                            # (B, 4, T, P_eff)
        feat_eff = fea3[:, 4:]                              # (B, F, T, P_eff)
        F_dim = feat_eff.shape[1]

        # Per-point features (B, T, P_eff, F)
        feat_pp = feat_eff.permute(0, 2, 3, 1).contiguous()
        if self.cluster_alpha_mode == 'time_stable':
            # Time-pooled assignment: same cluster IDs at every frame.
            feat_pp_time = feat_pp.mean(dim=1)                         # (B, P_eff, F)
            cluster_logits_static = self.cluster_head(feat_pp_time)    # (B, P_eff, K)
            alpha_static = torch.softmax(cluster_logits_static, dim=-1)
            alpha = alpha_static.unsqueeze(1).expand(B, T, P_eff, self.K).contiguous()
        else:
            cluster_logits = self.cluster_head(feat_pp)      # (B, T, P_eff, K)
            alpha = torch.softmax(cluster_logits, dim=-1)
        mass = alpha.sum(dim=2).clamp(min=1e-6)              # (B, T, K)

        # xyz downsampled (B, T, P_eff, 3)
        xyz_eff = coords_eff[:, :3].permute(0, 2, 3, 1).contiguous()

        # Per-cluster centroid (B, T, K, 3)
        centroid = torch.einsum('btpd,btpk->btkd', xyz_eff, alpha) / mass.unsqueeze(-1)

        # Centered coords per cluster (B, T, P_eff, K, 3)
        xyz_centered = xyz_eff.unsqueeze(-2) - centroid.unsqueeze(2)

        # Per-cluster covariance via weighted outer products (B, T, K, 3, 3)
        weighted = xyz_centered * alpha.unsqueeze(-1)        # (B, T, P, K, 3)
        cov = torch.einsum('btpkd,btpke->btkde', weighted, xyz_centered) \
              / mass.unsqueeze(-1).unsqueeze(-1)

        # Principal axis + angle from eigh
        try:
            eigvals, eigvecs = torch.linalg.eigh(cov)         # ascending
        except RuntimeError:
            # Degenerate cov -> use identity quat
            B_, T_, K_ = mass.shape
            Q_act = torch.zeros(B_, T_, K_, 4, device=fea3.device, dtype=fea3.dtype)
            Q_act[..., 0] = 1.0
            return alpha, xyz_centered, Q_act, Q_act
        axis = eigvecs[..., -1]                              # (B, T, K, 3)
        angle = eigvals[..., -1].clamp(min=0).sqrt()         # (B, T, K)
        half = (angle * 0.5).unsqueeze(-1)
        Q_act = torch.cat([torch.cos(half), torch.sin(half) * axis], dim=-1)

        # Per-cluster feature aggregate (B, T, K, F)
        feat_per_cluster = torch.einsum('btpf,btpk->btkf', feat_pp, alpha) \
                           / mass.unsqueeze(-1)
        # Reshape for GRU: (B*K, T, F)
        feat_kbf = feat_per_cluster.permute(0, 2, 1, 3).reshape(B * self.K, T, F_dim)
        gru_out, _ = self.cycle_gru(feat_kbf)                # (B*K, T, 2H)
        v = self.cycle_proj(gru_out)                          # (B*K, T, 3)
        Q_pred_flat = _exp_map(v)                             # (B*K, T, 4)
        Q_pred = Q_pred_flat.reshape(B, self.K, T, 4).permute(0, 2, 1, 3)

        return alpha, xyz_centered, Q_act, Q_pred

    def _cluster_aux_loss(self, alpha, xyz_centered, Q_act, Q_pred):
        """Compute cycle + recon + balance losses per cluster."""
        B, T, K, _ = Q_act.shape

        # Direct cycle supervision (double-cover invariant).
        diff_pos = (Q_pred - Q_act).pow(2).sum(-1)            # (B, T, K)
        diff_neg = (Q_pred + Q_act).pow(2).sum(-1)
        L_cycle = torch.minimum(diff_pos, diff_neg).mean()

        # Reconstruction grounding for anchor pair (a=0, c=T-1).
        t_a = 0; t_c = T - 1
        Qa = Q_pred[:, t_a]; Qc = Q_pred[:, t_c]              # (B, K, 4)
        q_ac = _hamilton(Qc, _conj(Qa))                       # (B, K, 4)
        Xa_c = xyz_centered[:, t_a]                           # (B, P, K, 3)
        Xc_c = xyz_centered[:, t_c]
        P_eff = Xa_c.shape[1]
        q_b = q_ac.unsqueeze(1).expand(-1, P_eff, -1, -1)     # (B, P, K, 4)
        Xa_rot = _quat_rotate(q_b, Xa_c)                       # (B, P, K, 3)
        alpha_c = alpha[:, t_c]                               # (B, P, K)
        sq = ((Xa_rot - Xc_c) ** 2).sum(-1)                   # (B, P, K)
        L_recon = (sq * alpha_c).sum() / (alpha_c.sum() + 1e-6)

        # Load balancing: maximize entropy of mean cluster usage.
        mean_usage = alpha.mean(dim=(1, 2)).clamp(min=1e-6)   # (B, K)
        entropy = -(mean_usage * mean_usage.log()).sum(-1)    # (B,)
        L_balance = -entropy.mean()                           # negate -> maximize

        return (self.lambda_cycle * L_cycle
                + self.lambda_recon * L_recon
                + self.lambda_balance * L_balance)

    def forward(self, inputs):
        if isinstance(inputs, dict):
            inputs = inputs['points']
        coords = self._sample_points(inputs)                  # (B, 4, T, P)
        B = coords.shape[0]; T = coords.shape[2]

        fea3 = self._encode_sampled_points(coords)            # (B, 260, T, P_eff)

        # Main classifier path (unchanged from parent).
        output = self.stage5(fea3)
        output = self.pool5(output)
        output = self.global_bn(output)
        main_features = output.flatten(1)
        main_logits = self.classify_features(main_features)

        # Inertia quat-head aux (the parent's 91.08 head).
        quat_traj = _inertia_quat(coords[:, :3]).reshape(B, T * 4)
        inertia_logits = self.quat_head(quat_traj)

        # Cluster-rotation cycle consistency aux.
        if self.training and (self.lambda_cycle > 0 or self.lambda_recon > 0):
            alpha, xyz_centered, Q_act, Q_pred = self._compute_cluster_quats(fea3)
            self.aux_loss = self._cluster_aux_loss(alpha, xyz_centered, Q_act, Q_pred)
        else:
            self.aux_loss = None

        return main_logits + self.quat_head_scale * inertia_logits

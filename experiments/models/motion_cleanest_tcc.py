"""CN-XXL + Temporal Cycle Consistency aux loss (Dwibedi et al. CVPR 2019).

Mechanism:
- Spatial pipeline produces per-frame embeddings (B, T, D) via mean-pool over N.
- Within each batch, samples are paired by random permutation.
- For each pair (u, v), per-frame in u soft-NN to v (softmax over -L2^2), then
  soft-NN back to u. Cycle-back-regression: fit Gaussian to the round-trip
  index distribution and penalize deviation from the original index:
      L = (i - mu)^2 / sigma^2 + lambda * log sigma          (Dwibedi eq.5)
- TCC loss exposed via self.aux_loss; main.py adds it to total loss.
- Inertia quat head and main classifier path preserved from the parent so
  warm-start from the cnxxlquat 91.08 ckpt is bit-compatible.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.motion_cleanest_quat_head import MotionCleanestLinXLQuatHead
from models.motion_cleanest_quat import _inertia_quat


def _tcc_cycle_loss(emb_a, emb_b, lambda_log=0.001):
    """(B, T, D) x (B, T, D) -> scalar TCC cycle-back-regression loss."""
    B, T, _ = emb_a.shape
    dists_ab = -torch.cdist(emb_a, emb_b) ** 2                     # (B, T, T)
    alpha = F.softmax(dists_ab, dim=2)
    v_tilde = torch.bmm(alpha, emb_b)                               # (B, T, D)
    dists_back = -torch.cdist(v_tilde, emb_a) ** 2                  # (B, T, T)
    beta = F.softmax(dists_back, dim=2)
    indices = torch.arange(T, device=emb_a.device, dtype=emb_a.dtype)
    mu = (beta * indices.view(1, 1, T)).sum(dim=2)                  # (B, T)
    var = (beta * (indices.view(1, 1, T) - mu.unsqueeze(-1)) ** 2).sum(dim=2)
    sigma_sq = var.clamp(min=1e-3)
    log_sigma = 0.5 * torch.log(sigma_sq)
    target = indices.view(1, T).expand(B, T)
    loss = ((target - mu) ** 2 / sigma_sq + lambda_log * log_sigma).mean()
    return loss


class MotionCleanestLinXLQuatHeadTCC(MotionCleanestLinXLQuatHead):
    def __init__(self, *args,
                 tcc_weight=0.1,
                 tcc_lambda_log=0.001,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.tcc_weight = float(tcc_weight)
        self.tcc_lambda_log = float(tcc_lambda_log)
        self.aux_loss = None

    def forward(self, inputs):
        if isinstance(inputs, dict):
            inputs = inputs['points']
        coords = self._sample_points(inputs)                        # (B, 4, T, P)
        B = coords.shape[0]; T = coords.shape[2]

        # Spatial + Mamba (parent path); fea3 keeps T dim for TCC.
        fea3 = self._encode_sampled_points(coords)                  # (B, C, T, P_eff)
        # Per-frame embedding: mean-pool over points within each frame.
        per_frame = fea3.mean(dim=-1).transpose(1, 2)               # (B, T, C)

        # Main classification: continue parent path (stage5 -> pool -> bn -> stage6).
        output = self.stage5(fea3)
        output = self.pool5(output)
        output = self.global_bn(output)
        main_features = output.flatten(1)
        main_logits = self.classify_features(main_features)

        # Inertia quat-head aux (the existing 91.08 quat_head).
        quat = _inertia_quat(coords[:, :3]).reshape(B, T * 4)
        inertia_logits = self.quat_head(quat)

        # TCC aux loss (train mode, batch >= 2).
        if self.training and B >= 2 and self.tcc_weight > 0:
            perm = torch.randperm(B, device=per_frame.device)
            # Avoid self-pairing if any fixed points.
            for _ in range(3):
                if (perm == torch.arange(B, device=perm.device)).any():
                    perm = torch.randperm(B, device=per_frame.device)
                else:
                    break
            partner = per_frame[perm]
            tcc = _tcc_cycle_loss(per_frame, partner, self.tcc_lambda_log)
            self.aux_loss = self.tcc_weight * tcc
        else:
            self.aux_loss = None

        return main_logits + self.quat_head_scale * inertia_logits

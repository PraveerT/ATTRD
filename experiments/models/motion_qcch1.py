"""
MotionQCCH1 — true cycle-consistency aux loss.

Architecture:
  - Backbone (Motion) produces 1024-d global feature.
  - Aux head: Linear(1024, num_transitions * 4) → (B, T-1, 4) per-frame
    forward quaternions.
  - Each predicted quat is L2-normalized (unit quaternion).

Loss:
  aux = predict_w · MSE(pred_quats, target_quats)
      + cycle_w · MSE(cycle_product(pred_quats), identity)

The cycle term forces the predicted quaternion sequence to compose to
identity (true QCC). The predict term anchors the chain so it learns
something gesture-meaningful (otherwise model trivially outputs identity
for every t).

Inputs (from NvidiaPts10QccSeqLoader): (B, T, P, 26)
  channels [0:22]   = stage1 input (pts10 + qcycprod)
  channels [22:26]  = per-frame q_combined target, broadcast across P
                      (last frame [T-1] is the zero pad)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .motion import Motion


def _quat_mul_batch(a, b):
    # a, b: (B, 4)  (w, x, y, z)
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], dim=-1)


class MotionQCCH1(Motion):
    def __init__(self, num_classes=25, pts_size=256, knn=(32, 24, 48, 24),
                 topk=8, downsample=(2, 2, 2),
                 coord_channels=4, stage1_in_channels=22,
                 num_transitions=31,
                 predict_weight=0.05,
                 cycle_weight=0.5,
                 **kwargs):
        super().__init__(
            num_classes=num_classes, pts_size=pts_size, knn=knn,
            topk=topk, downsample=downsample,
            coord_channels=coord_channels, stage1_in_channels=stage1_in_channels,
        )
        self.num_transitions = int(num_transitions)
        self.predict_weight = float(predict_weight)
        self.cycle_weight = float(cycle_weight)
        self.aux_head = nn.Linear(self.feature_dim, self.num_transitions * 4)
        self._latest_pred_quats = None
        self._latest_target_quats = None

    def forward(self, inputs):
        # inputs: (B, T, P, stage1_in_channels + 4)
        target_dim = 4
        total = self.stage1_in_channels + target_dim
        if inputs.shape[-1] != total:
            raise ValueError(f"expected last-dim={total}, got {inputs.shape[-1]}")
        main_input = inputs[..., :self.stage1_in_channels].contiguous()
        # take target sequence from frame 0..T-1 at point 0 (it's broadcast across P)
        # only the first num_transitions frames carry target (last is zero pad)
        target = inputs[:, :self.num_transitions, 0, self.stage1_in_channels:]  # (B, T-1, 4)

        features = self.extract_features(main_input)  # (B, 1024)
        pred_flat = self.aux_head(features)  # (B, (T-1)*4)
        pred = pred_flat.view(-1, self.num_transitions, 4)  # (B, T-1, 4)
        # unit-normalize predicted quats
        pred = pred / pred.norm(dim=-1, keepdim=True).clamp(min=1e-8)

        self._latest_pred_quats = pred
        self._latest_target_quats = target

        return self.classify_features(features)

    def get_auxiliary_loss(self):
        if self._latest_pred_quats is None:
            return None
        pred = self._latest_pred_quats        # (B, T-1, 4)
        target = self._latest_target_quats    # (B, T-1, 4)

        # predict (anchor) loss
        predict_loss = F.mse_loss(pred, target)

        # cycle loss: compose pred quats over T-1 transitions
        Q = pred[:, 0]
        for t in range(1, self.num_transitions):
            Q = _quat_mul_batch(Q, pred[:, t])
        identity = torch.zeros_like(Q)
        identity[:, 0] = 1.0
        cycle_loss = F.mse_loss(Q, identity)

        return self.predict_weight * predict_loss + self.cycle_weight * cycle_loss

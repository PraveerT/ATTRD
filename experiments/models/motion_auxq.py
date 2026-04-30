"""
MotionAuxQ — Motion backbone + auxiliary quaternion head trained to
predict Q_combined_cycle from the 1024-d global feature.

Aux loss = ‖predicted_Q − target_Q‖²

Loader is expected to return inputs of shape (B, T, P, 26):
  channels [0:22] = main input fed to stage1 (pts10 + qcycprod broadcast)
  channels [22:26] = target Q_combined_cycle, broadcast to (T, P, 4)

Model:
  - splits inputs
  - backbone (Motion.extract_features) on main_input
  - aux_head: Linear(1024, 4) on backbone features → predicted quaternion
  - main classifier as usual
  - get_auxiliary_loss() returns aux_weight · MSE(pred, target)

main.py picks up the aux loss via the existing
`if hasattr(model_ref, 'get_auxiliary_loss'): aux_loss = model_ref.get_auxiliary_loss()` hook.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .motion import Motion


class MotionAuxQ(Motion):
    def __init__(self, num_classes=25, pts_size=256, knn=(32, 24, 48, 24),
                 topk=8, downsample=(2, 2, 2),
                 coord_channels=4, stage1_in_channels=22,
                 aux_weight=0.1, target_dim=4, **kwargs):
        super().__init__(
            num_classes=num_classes, pts_size=pts_size, knn=knn,
            topk=topk, downsample=downsample,
            coord_channels=coord_channels, stage1_in_channels=stage1_in_channels,
        )
        self.aux_weight = float(aux_weight)
        self.target_dim = int(target_dim)
        self.aux_head = nn.Linear(self.feature_dim, self.target_dim)
        self._latest_aux_pred = None
        self._latest_target = None

    def forward(self, inputs):
        # inputs: (B, T, P, stage1_in_channels + target_dim)
        total = self.stage1_in_channels + self.target_dim
        if inputs.shape[-1] != total:
            raise ValueError(
                f"expected last-dim={total} (stage1_in={self.stage1_in_channels} + target={self.target_dim}), got {inputs.shape[-1]}"
            )
        main_input = inputs[..., :self.stage1_in_channels].contiguous()
        target = inputs[:, 0, 0, self.stage1_in_channels:].contiguous()  # (B, target_dim)

        features = self.extract_features(main_input)  # (B, 1024)
        pred = self.aux_head(features)  # (B, target_dim)
        self._latest_aux_pred = pred
        self._latest_target = target

        return self.classify_features(features)

    def get_auxiliary_loss(self):
        if self._latest_aux_pred is None or self._latest_target is None:
            return None
        return self.aux_weight * F.mse_loss(self._latest_aux_pred, self._latest_target)

"""
TwoStreamMotion — two parallel Motion encoders, one on raw uvd-t (4-ch) and
one on lattice octo (x, y, z, vx, vy, vz, 6-ch). Both produce 1024-d global
features which are concatenated and pushed through a small fusion head.

Input: tensor (B, T, P=512, 14) where channels [0:8] = raw, [8:14] = octo.
Output: (B, num_classes) logits.
"""

import torch
import torch.nn as nn

from .motion import Motion


class TwoStreamMotion(nn.Module):
    def __init__(self, num_classes=25, pts_size=256, knn=(32, 24, 48, 24),
                 topk=8, hidden=512, dropout=0.1, downsample=(2, 2, 2)):
        super().__init__()
        self.raw_branch = Motion(
            num_classes=num_classes, pts_size=pts_size, knn=knn,
            topk=topk, downsample=downsample, coord_channels=4,
        )
        self.octo_branch = Motion(
            num_classes=num_classes, pts_size=pts_size, knn=knn,
            topk=topk, downsample=downsample, coord_channels=6,
        )
        feat_dim = self.raw_branch.feature_dim + self.octo_branch.feature_dim
        self.fuse = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, inputs):
        # inputs: (B, T, P, 14)  — split along channel dim
        raw = inputs[..., :8].contiguous()       # raw uvd-t-xyz-t (model takes :coord internally)
        octo = inputs[..., 8:14].contiguous()    # x, y, z, vx, vy, vz
        f_raw = self.raw_branch.extract_features(raw)    # (B, 1024)
        f_octo = self.octo_branch.extract_features(octo)  # (B, 1024)
        return self.fuse(torch.cat([f_raw, f_octo], dim=1))

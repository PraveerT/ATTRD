"""
TwoStreamAvg — both branches produce logits independently; final prediction
is a weighted average controlled by a learnable alpha (initialized to favor
the raw branch so warmstart from a baseline checkpoint is preserved).

raw_branch: Motion(coord_channels=4) — uses raw uvd-t (8-ch input slice [0:8])
octo_branch: Motion(coord_channels=6) — uses lattice octo (slice [8:14])

Each branch has its own classifier (Motion.stage6 / classify_features).

Forward:
  logits_raw = raw_branch(raw_input)
  logits_octo = octo_branch(octo_input)
  alpha = sigmoid(self.alpha_logit)
  final = alpha * logits_raw + (1 - alpha) * logits_octo

Warmstart: load baseline pmamba_branch e110 into raw_branch.* (raw_branch
already has all 142 keys including stage6 classifier). octo_branch starts
random. alpha_logit init at +5 → sigmoid ~ 0.993 → at init final ≈ raw_branch
output ≈ baseline 89.83. Training nudges alpha toward 0.5 as octo_branch
learns; final prediction is weighted ensemble.
"""

import torch
import torch.nn as nn

from .motion import Motion


class TwoStreamAvg(nn.Module):
    def __init__(self, num_classes=25, pts_size=256, knn=(32, 24, 48, 24),
                 topk=8, downsample=(2, 2, 2),
                 alpha_init=5.0, freeze_raw=False, **kwargs):
        super().__init__()
        self.raw_branch = Motion(
            num_classes=num_classes, pts_size=pts_size, knn=knn,
            topk=topk, downsample=downsample, coord_channels=4,
        )
        self.octo_branch = Motion(
            num_classes=num_classes, pts_size=pts_size, knn=knn,
            topk=topk, downsample=downsample, coord_channels=6,
        )
        # alpha_logit initialized large positive so sigmoid -> ~1, weighting
        # the warm-started raw branch heavily at start.
        self.alpha_logit = nn.Parameter(torch.tensor(float(alpha_init)))

        if freeze_raw:
            for p in self.raw_branch.parameters():
                p.requires_grad = False
            self._raw_frozen = True
        else:
            self._raw_frozen = False

    def train(self, mode: bool = True):
        super().train(mode)
        # When raw_branch is frozen we always keep its BN layers in eval
        # mode so running stats don't drift during fine-tuning.
        if getattr(self, "_raw_frozen", False):
            self.raw_branch.eval()
        return self

    def forward(self, inputs):
        # inputs (B, T, P, 14): [0:8] raw, [8:14] octo
        raw = inputs[..., :8].contiguous()
        octo = inputs[..., 8:14].contiguous()
        logits_raw = self.raw_branch(raw)         # (B, num_classes)
        logits_octo = self.octo_branch(octo)
        alpha = torch.sigmoid(self.alpha_logit)
        return alpha * logits_raw + (1 - alpha) * logits_octo

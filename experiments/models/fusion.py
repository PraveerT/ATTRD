import torch
import torch.nn as nn
import torch.nn.functional as F

from models.motion import Motion
from models.reqnn_motion import BearingQCCFeatureMotion


class MotionDualBranchFusion(nn.Module):
    """Learned fusion of PMamba temporal branch and quaternion spatial branch.

    Each branch is frozen and provides features.  The fusion head concatenates
    projected branch features and classifies from the joint representation.
    Auxiliary per-branch classification losses keep each branch's signal sharp.
    """

    def __init__(
        self,
        num_classes,
        pts_size,
        temporal_model_args=None,
        spatial_model_args=None,
        fusion_dim=256,
        dropout=0.25,
        aux_weight=0.3,
        branch_prior=(0.75, 0.25),  # kept for config compat, unused
    ):
        super().__init__()
        self.num_classes = num_classes
        self.pts_size = pts_size
        self.aux_weight = aux_weight

        # --- Temporal branch (PMamba) ---
        t_args = temporal_model_args or {}
        self.temporal_branch = Motion(
            num_classes=num_classes, pts_size=pts_size, **t_args,
        )
        self.temporal_feat_dim = self.temporal_branch.feature_dim  # 1024

        # --- Spatial branch (Quaternion + Bearing QCC) ---
        s_args = spatial_model_args or {}
        self.spatial_branch = BearingQCCFeatureMotion(
            num_classes=num_classes, pts_size=pts_size, **s_args,
        )
        self.spatial_feat_dim = self.spatial_branch.feature_dim  # 512

        # --- Fusion layers ---
        # Project both branches to same dimension
        self.temporal_proj = nn.Sequential(
            nn.Linear(self.temporal_feat_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
        )
        self.spatial_proj = nn.Sequential(
            nn.Linear(self.spatial_feat_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
        )

        # Classifier on concatenated features (2 * fusion_dim -> num_classes)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(fusion_dim, num_classes),
        )

        # Lightweight auxiliary branch classifiers (frozen branch features -> logits)
        self.temporal_aux_cls = nn.Linear(self.temporal_feat_dim, num_classes)
        self.spatial_aux_cls = nn.Linear(self.spatial_feat_dim, num_classes)

        # Store branch logits for monitoring
        self.temporal_logits = None
        self.spatial_logits = None

    def _unpack_inputs(self, inputs):
        """Handle both dict and tensor inputs."""
        if isinstance(inputs, dict):
            return inputs['points'], inputs
        return inputs, None

    def extract_features(self, inputs):
        points, aux_input = self._unpack_inputs(inputs)

        # Temporal branch expects raw tensor
        t_feat = self.temporal_branch.extract_features(points)

        # Spatial branch expects dict or tensor with aux
        s_feat = self.spatial_branch.extract_features(points, aux_input=aux_input)

        # Auxiliary branch logits (for aux loss during training)
        self.temporal_logits = self.temporal_aux_cls(t_feat)
        self.spatial_logits = self.spatial_aux_cls(s_feat)

        # Project to fusion dimension
        t_proj = self.temporal_proj(t_feat)
        s_proj = self.spatial_proj(s_feat)

        # Concatenate both projections
        fused = torch.cat([t_proj, s_proj], dim=1)  # (batch, fusion_dim * 2)

        return fused

    def get_auxiliary_loss(self):
        """Return auxiliary branch classification losses."""
        if not self.training or self.temporal_logits is None:
            return None
        # We don't have labels here — main.py computes aux loss from
        # self.temporal_logits / self.spatial_logits via the branch loss monitor.
        # Return None and let main.py handle it.
        return None

    def forward(self, inputs):
        fused = self.extract_features(inputs)
        return self.classifier(fused)

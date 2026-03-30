import torch
import torch.nn as nn
import torch.nn.functional as F

from models.motion import Motion
from models.reqnn_motion import BearingQCCFeatureMotion


class MotionDualBranchFusion(nn.Module):
    """Learned fusion of PMamba temporal branch and quaternion spatial branch.

    Each branch is frozen and provides features. A learned gate decides
    per-sample how much to trust each branch, and a shared classifier
    produces final predictions from the fused representation.
    """

    def __init__(
        self,
        num_classes,
        pts_size,
        temporal_model_args=None,
        spatial_model_args=None,
        fusion_dim=512,
        dropout=0.15,
        branch_prior=(0.75, 0.25),
    ):
        super().__init__()
        self.num_classes = num_classes
        self.pts_size = pts_size

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
        total_feat_dim = self.temporal_feat_dim + self.spatial_feat_dim

        # Project both branches to same dimension
        self.temporal_proj = nn.Sequential(
            nn.Linear(self.temporal_feat_dim, fusion_dim),
            nn.BatchNorm1d(fusion_dim),
            nn.GELU(),
        )
        self.spatial_proj = nn.Sequential(
            nn.Linear(self.spatial_feat_dim, fusion_dim),
            nn.BatchNorm1d(fusion_dim),
            nn.GELU(),
        )

        # Learned gate: takes concatenated features, outputs per-branch weight
        self.gate = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, 2),
        )
        # Initialize gate bias toward the branch prior
        with torch.no_grad():
            prior = torch.tensor(branch_prior, dtype=torch.float32)
            self.gate[-1].bias.copy_(torch.log(prior / (1 - prior + 1e-8)))

        # Classifier on fused features
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, num_classes),
        )

        # Freeze branches — weights will be loaded after init but
        # requires_grad=False persists through load_state_dict
        self.freeze_branches()

    def freeze_branches(self):
        """Freeze both branch parameters so only fusion layers train."""
        for param in self.temporal_branch.parameters():
            param.requires_grad = False
        for param in self.spatial_branch.parameters():
            param.requires_grad = False
        self.temporal_branch.eval()
        self.spatial_branch.eval()

    def train(self, mode=True):
        """Override to keep branches in eval mode even when fusion trains."""
        super().train(mode)
        # Always keep branches in eval (frozen BN, no dropout)
        self.temporal_branch.eval()
        self.spatial_branch.eval()
        return self

    def _unpack_inputs(self, inputs):
        """Handle both dict and tensor inputs."""
        if isinstance(inputs, dict):
            return inputs['points'], inputs
        return inputs, None

    def extract_features(self, inputs):
        points, aux_input = self._unpack_inputs(inputs)

        # Temporal branch expects raw tensor
        with torch.no_grad():
            t_feat = self.temporal_branch.extract_features(points)

        # Spatial branch expects dict or tensor with aux
        with torch.no_grad():
            s_feat = self.spatial_branch.extract_features(points, aux_input=aux_input)

        # Project to fusion dimension
        t_proj = self.temporal_proj(t_feat)
        s_proj = self.spatial_proj(s_feat)

        # Learned gating
        gate_input = torch.cat([t_proj, s_proj], dim=1)
        gate_weights = torch.softmax(self.gate(gate_input), dim=1)  # (batch, 2)

        # Weighted fusion
        fused = gate_weights[:, 0:1] * t_proj + gate_weights[:, 1:2] * s_proj

        return fused, gate_weights

    def forward(self, inputs):
        fused, _ = self.extract_features(inputs)
        return self.classifier(fused)

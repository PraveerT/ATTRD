"""No-temporal-encoder ablation for NVGesture.

Replaces self.mamba (the recurrent temporal encoder) with a pure channel
projection — no temporal mixing at all. Each frame is processed independently.
Tests whether the temporal layer is actually contributing, or if all the
"recurrent architecture" variants we've tested were just adding capacity to
something the downstream classifier could equally well do with no temporal
mixing.

If solo accuracy drops to ~70% without temporal, the recurrent layer is
critical and architectural differences matter. If solo stays at ~85%, every
recurrent variant we've tested was studying noise around a data-bound ceiling.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.motion import Motion


class IdentityTemporalEncoder(nn.Module):
    """Channel projection only — no temporal mixing.

    Each frame at each spatial point is processed independently by a linear
    projection (in_channels -> output_dim). LayerNorm and dropout match the
    plumbing of the recurrent encoders for fair-ish comparison; nothing
    mixes across the T axis.
    """
    def __init__(self, in_channels, hidden_dim=128, output_dim=None,
                 dropout=0.3, **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim if output_dim is not None else hidden_dim
        self.input_proj = nn.Linear(in_channels, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, self.output_dim)
        self.dropout = nn.Dropout(dropout)
        self.gelu = nn.GELU()

    def forward(self, x):
        # x: (B, C, T, N) — apply 2-layer MLP per (frame, point); no T mixing
        B, C, T, N = x.shape
        x = x.permute(0, 2, 3, 1).reshape(B * T * N, C)   # B*T*N, C
        x = self.input_proj(x)
        x = self.gelu(x)
        x = self.norm(x)
        x = self.dropout(x)
        x = self.output_proj(x)
        x = x.reshape(B, T, N, self.output_dim).permute(0, 3, 1, 2)  # B, out, T, N
        return x


class MotionNoTemporal(Motion):
    """PMamba with the temporal encoder replaced by a per-frame MLP.

    Tests whether the entire family of recurrent variants we've explored
    (RD, BD-N, AttRD, BRD, ...) is adding real value or whether the
    spatial PointNet stages already determine NVGesture accuracy.
    """
    def __init__(self, *args, nt_hidden_dim=128, nt_dropout=0.3, **kwargs):
        super().__init__(*args, **kwargs)
        old = self.mamba
        in_c = old.in_channels
        out_d = old.output_dim
        self.mamba = IdentityTemporalEncoder(
            in_channels=in_c, hidden_dim=nt_hidden_dim, output_dim=out_d,
            dropout=nt_dropout,
        )

"""PMamba with the real Mamba2 SSD layer as the temporal encoder.

Drops in for self.mamba (the per-point temporal scan after the spatial pyramid).
Uses state-spaces/mamba_ssm 2.x (Mamba2 + causal_conv1d).

Constraints:
- d_model (hidden_dim) must be a multiple of headdim*8 / expand for the kernel
  alignment requirement. headdim=64, expand=2 -> d_model multiple of 256.
"""
import torch
import torch.nn as nn

from models.motion import Motion
from mamba_ssm.modules.mamba2 import Mamba2


class Mamba2TemporalEncoder(nn.Module):
    """Bidirectional Mamba2 encoder; matches RealDeltaNetTemporalEncoder API."""
    def __init__(self, in_channels, hidden_dim=256, output_dim=None, num_layers=2,
                 d_state=64, d_conv=4, expand=2, headdim=64, dropout=0.3,
                 bidirectional=True):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim if output_dim is not None else hidden_dim
        self.bidirectional = bidirectional
        self.input_proj = nn.Linear(in_channels, hidden_dim)

        def make_layers(n):
            return nn.ModuleList([
                Mamba2(d_model=hidden_dim, d_state=d_state, d_conv=d_conv,
                       expand=expand, headdim=headdim)
                for _ in range(n)
            ])

        self.fwd_blocks = make_layers(num_layers)
        self.fwd_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        if bidirectional:
            self.bwd_blocks = make_layers(num_layers)
            self.bwd_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, self.output_dim)

    def _stack(self, x, layers, norms):
        for blk, norm in zip(layers, norms):
            residual = x
            x = norm(x)
            x = blk(x)
            x = self.dropout(x)
            x = x + residual
        return x

    def forward(self, x):
        Bz, C, T, N = x.shape
        x = x.permute(0, 3, 2, 1).contiguous().reshape(Bz * N, T, C)   # (B*N, T, C)
        x = self.input_proj(x)
        fwd = self._stack(x, self.fwd_blocks, self.fwd_norms)
        out = fwd
        if self.bidirectional:
            bwd = self._stack(x.flip(1), self.bwd_blocks, self.bwd_norms).flip(1)
            out = out + bwd
        out = self.final_norm(out)
        out = self.output_proj(out)
        out = out.reshape(Bz, N, T, self.output_dim).permute(0, 3, 2, 1).contiguous()
        return out


class MotionMamba2(Motion):
    """PMamba with real Mamba2 temporal encoder."""
    def __init__(self, *args, m2_hidden_dim=256, m2_num_layers=2, m2_d_state=64,
                 m2_d_conv=4, m2_expand=2, m2_headdim=64, m2_dropout=0.3,
                 m2_bidirectional=True, **kwargs):
        super().__init__(*args, **kwargs)
        old = self.mamba
        self.mamba = Mamba2TemporalEncoder(
            in_channels=old.in_channels, hidden_dim=m2_hidden_dim,
            output_dim=old.output_dim, num_layers=m2_num_layers,
            d_state=m2_d_state, d_conv=m2_d_conv, expand=m2_expand,
            headdim=m2_headdim, dropout=m2_dropout, bidirectional=m2_bidirectional,
        )

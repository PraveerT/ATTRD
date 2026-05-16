"""Buffered DeltaNet for NVGesture — port of BD-N from the MQAR experiments.

BD-N combines a short attention KV-buffer (FIFO, capacity W) with a long-term
DeltaNet state. On each token, the buffer's oldest entry is ejected into the
delta state via the standard rank-1 delta-rule write; the new token's (k, v)
enters the buffer. Read = buffer-attention + delta-read (sum).

Tested on MQAR: 7/7 cells significantly beat plain DeltaNet (p<0.001 in 4
cells, p<0.01 in 3 cells) at exact param-match. Hypothesis here: the same
hybrid mechanism transfers to NVGesture's "register configuration → query
later" structure in gesture recognition.

Architecture mirrors models.motion_realdeltanet to preserve the PMamba plumbing
(point-cloud (B, C, T, N) input → (B*N, T, C) reshape → fwd+bwd encoder).
We use a non-quaternion real-valued block here (BD-N's MQAR design): drop-in
replacement for RealDeltaNetBlock with the buffer/eject augmentation.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.motion import Motion


class BDeltaBlock(nn.Module):
    """Buffered Delta block: short attention buffer + long-term delta state.

    Args:
        d_model: input/output dim.
        num_heads, head_dim: head config (d_inner = num_heads * head_dim).
        buffer_size: FIFO buffer capacity (in tokens). Must be < T to ensure
            ejection happens during the productive part of the sequence.
        use_short_conv: if True, depthwise conv over qkv (matches RD plumbing).
    """
    def __init__(self, d_model, num_heads=4, head_dim=32, buffer_size=4,
                 dropout=0.1, use_short_conv=True, conv_size=4,
                 max_seq_len=512, rope_base=10000.0):
        super().__init__()
        self.H = num_heads
        self.D = head_dim
        self.W = buffer_size
        d_inner = num_heads * head_dim
        self.q_proj = nn.Linear(d_model, d_inner, bias=False)
        self.k_proj = nn.Linear(d_model, d_inner, bias=False)
        self.v_proj = nn.Linear(d_model, d_inner, bias=False)
        self.beta_proj = nn.Linear(d_model, num_heads)

        self.use_short_conv = use_short_conv
        if use_short_conv:
            ch = 3 * d_inner
            self.short_conv = nn.Conv1d(ch, ch, kernel_size=conv_size,
                                        padding=conv_size - 1, groups=ch)

        self.dropout = nn.Dropout(dropout)
        self.attn_dropout = nn.Dropout(dropout)
        self.o_proj = nn.Linear(d_inner, d_model, bias=False)

        # RoPE cache for buffer-attention q/k positional encoding.
        # head_dim must be even.
        assert head_dim % 2 == 0, "RoPE needs even head_dim"
        inv_freq = 1.0 / (rope_base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        pos = torch.arange(max_seq_len).float()
        freqs = torch.einsum('i,j->ij', pos, inv_freq)       # (T_max, D/2)
        self.register_buffer('rope_cos', freqs.cos(), persistent=False)
        self.register_buffer('rope_sin', freqs.sin(), persistent=False)

    def _rope(self, x, pos):
        """Apply RoPE to x at absolute position(s) `pos`. x: (..., D)."""
        cos = self.rope_cos[pos]     # (D/2,)
        sin = self.rope_sin[pos]
        x1, x2 = x[..., 0::2], x[..., 1::2]
        rot1 = x1 * cos - x2 * sin
        rot2 = x1 * sin + x2 * cos
        return torch.stack([rot1, rot2], dim=-1).flatten(-2)

    def forward(self, x):
        B, T, D = x.shape
        H, Dh, W = self.H, self.D, self.W
        q = self.q_proj(x); k = self.k_proj(x); v = self.v_proj(x)
        if self.use_short_conv:
            qkv = torch.cat([q, k, v], dim=-1).transpose(1, 2)
            qkv = self.short_conv(qkv)[..., :T].transpose(1, 2)
            s1 = H * Dh
            q, k, v = qkv[..., :s1], qkv[..., s1:2*s1], qkv[..., 2*s1:]

        q = q.view(B, T, H, Dh)
        k = F.normalize(k.view(B, T, H, Dh), dim=-1)
        v = v.view(B, T, H, Dh)
        beta = torch.sigmoid(self.beta_proj(x)).view(B, T, H, 1)
        q = F.silu(q)

        # Long-term delta state S (B, H, Dh, Dh)
        S = torch.zeros(B, H, Dh, Dh, device=x.device, dtype=x.dtype)
        # Buffer entries store (k_rotated_at_absolute_pos, v, abs_pos).
        K_buf = []
        V_buf = []
        P_buf = []     # absolute positions of each buffer entry
        ys = []
        for t in range(T):
            kt = k[:, t]; vt = v[:, t]; bt = beta[:, t]
            # RoPE-rotate the new key at its absolute position before storing.
            kt_rot = self._rope(kt, t)
            # Eject oldest into delta state on buffer overflow
            if len(K_buf) >= W:
                kt_old = K_buf.pop(0); vt_old = V_buf.pop(0); P_buf.pop(0)
                # delta-rule write uses the (already-rotated) old key
                Sk = torch.einsum('bhij,bhj->bhi', S, kt_old)
                err = vt_old - Sk
                S = S + bt.unsqueeze(-1) * torch.einsum('bhi,bhj->bhij', err, kt_old)
            K_buf.append(kt_rot); V_buf.append(vt); P_buf.append(t)
            # Buffer attention with RoPE-rotated q at position t.
            q_rot = self._rope(q[:, t], t)
            K_stack = torch.stack(K_buf, dim=2)
            V_stack = torch.stack(V_buf, dim=2)
            scores = torch.einsum('bhd,bhld->bhl', q_rot, K_stack) / math.sqrt(Dh)
            attn = self.attn_dropout(F.softmax(scores, dim=-1))
            buf_out = torch.einsum('bhl,bhld->bhd', attn, V_stack)
            # Delta read with (un-rotated) q  -- delta state keeps absolute keys.
            delta_out = torch.einsum('bhij,bhj->bhi', S, q[:, t])
            yt = buf_out + delta_out
            ys.append(yt)
        y = torch.stack(ys, dim=1).reshape(B, T, H * Dh)
        return self.o_proj(self.dropout(y))


class BDeltaTemporalEncoder(nn.Module):
    """Bidirectional BD-N encoder. Mirrors RealDeltaNetTemporalEncoder."""
    def __init__(self, in_channels, hidden_dim=128, output_dim=None, num_layers=2,
                 num_heads=4, head_dim=32, buffer_size=4, dropout=0.3,
                 bidirectional=True):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim if output_dim is not None else hidden_dim
        self.bidirectional = bidirectional
        self.input_proj = nn.Linear(in_channels, hidden_dim)
        self.fwd_blocks = nn.ModuleList([
            BDeltaBlock(hidden_dim, num_heads, head_dim, buffer_size, dropout)
            for _ in range(num_layers)
        ])
        self.fwd_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        if bidirectional:
            self.bwd_blocks = nn.ModuleList([
                BDeltaBlock(hidden_dim, num_heads, head_dim, buffer_size, dropout)
                for _ in range(num_layers)
            ])
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
        x = x.permute(0, 3, 2, 1).reshape(Bz * N, T, C)
        x = self.input_proj(x)
        fwd = self._stack(x, self.fwd_blocks, self.fwd_norms)
        out = fwd
        if self.bidirectional:
            bwd = self._stack(x.flip(1), self.bwd_blocks, self.bwd_norms)
            out = out + bwd
        out = self.final_norm(out)
        out = self.output_proj(out)
        out = out.reshape(Bz, N, T, self.output_dim).permute(0, 3, 2, 1)
        return out


class MotionBDelta(Motion):
    """PMamba with Buffered DeltaNet replacing the Mamba temporal encoder."""
    def __init__(self, *args, bdn_hidden_dim=128, bdn_num_layers=2, bdn_num_heads=4,
                 bdn_head_dim=32, bdn_buffer_size=4, bdn_dropout=0.3,
                 bdn_bidirectional=True, **kwargs):
        super().__init__(*args, **kwargs)
        old = self.mamba
        in_c = old.in_channels
        out_d = old.output_dim
        self.mamba = BDeltaTemporalEncoder(
            in_channels=in_c, hidden_dim=bdn_hidden_dim, output_dim=out_d,
            num_layers=bdn_num_layers, num_heads=bdn_num_heads,
            head_dim=bdn_head_dim, buffer_size=bdn_buffer_size,
            dropout=bdn_dropout, bidirectional=bdn_bidirectional,
        )

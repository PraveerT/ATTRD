"""BD-RD: Buffered RealDeltaNet for NVGesture.

Takes the working RealDeltaNet block (alpha+beta gates, 4-component quaternion-
equivalent structure, parallel chunkwise scan that achieves 88.59% train-best
solo on NVGesture) and ADDS a buffer-attention head on top.

Architecture:
  RD path (unchanged):  y_rd = q^T B_acc[t]   (parallel scan; alpha-gated delta)
  Buffer path (new):    y_buf = softmax(q_flat . K_buf / sqrt(D)) V_buf
  Combined:             y = y_rd + W_combine * y_buf

The buffer is FIFO with capacity W; on overflow the oldest entry is dropped
(unlike pure BD-N where it would write into the delta state — here the delta
state is managed by RD's parallel scan, so we just drop). RoPE on q/k for
buffer attention to give it positional info.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.motion import Motion
from models.motion_realdeltanet import (
    RealDeltaNetBlock, RealDeltaNetTemporalEncoder, MotionRealDeltaNet,
    emul, econj, rmatmul,
)


class BufferedRealDeltaBlock(RealDeltaNetBlock):
    """RealDeltaNet + buffer-attention augment.

    Reuses RD's q, k, v projections (quaternion-equivalent, 4-component).
    Adds a separate buffer-attention head with its own o_proj that reads
    from a FIFO buffer of recent (k, v) entries via softmax attention with
    RoPE-rotated q/k for positional sensitivity.
    """
    def __init__(self, d_model, num_heads=4, n_q=4, n_v=8, buffer_size=4,
                 dropout=0.1, use_short_conv=True, conv_size=4,
                 max_seq_len=512, rope_base=10000.0):
        super().__init__(d_model, num_heads, n_q, n_v, dropout,
                         use_short_conv, conv_size)
        self.W = buffer_size
        d_q_flat = n_q * 4
        d_v_flat = n_v * 4
        H = num_heads
        # Buffer-side learned read scale + output projection (small)
        self.buf_o_proj = nn.Linear(H * d_v_flat, d_model, bias=False)
        # Mixing scalar for buffer output (init small so RD dominates at start)
        self.buf_mix = nn.Parameter(torch.zeros(1))
        # RoPE cache for q/k (flattened to d_q_flat)
        assert d_q_flat % 2 == 0
        inv_freq = 1.0 / (rope_base ** (torch.arange(0, d_q_flat, 2).float() / d_q_flat))
        pos = torch.arange(max_seq_len).float()
        freqs = torch.einsum('i,j->ij', pos, inv_freq)
        self.register_buffer('rope_cos', freqs.cos(), persistent=False)
        self.register_buffer('rope_sin', freqs.sin(), persistent=False)
        self.attn_dropout = nn.Dropout(dropout)

    def _rope(self, x, pos):
        cos = self.rope_cos[pos]
        sin = self.rope_sin[pos]
        x1, x2 = x[..., 0::2], x[..., 1::2]
        rot1 = x1 * cos - x2 * sin
        rot2 = x1 * sin + x2 * cos
        return torch.stack([rot1, rot2], dim=-1).flatten(-2)

    def forward(self, x):
        B, T, D = x.shape
        H, n_q, n_v = self.num_heads, self.n_q, self.n_v

        # Re-derive q, k, v just like parent (we need them for the buffer path too).
        q_proj = self.q_proj(x); k_proj = self.k_proj(x); v_proj = self.v_proj(x)
        if self.use_short_conv:
            qkv = torch.cat([q_proj, k_proj, v_proj], dim=-1).transpose(1, 2)
            qkv = self.short_conv(qkv)[..., :T].transpose(1, 2)
            s1 = H * n_q * 4
            s2 = s1 + H * n_q * 4
            q_proj, k_proj, v_proj = qkv[..., :s1], qkv[..., s1:s2], qkv[..., s2:]

        q = q_proj.view(B, T, H, n_q, 4)
        k = k_proj.view(B, T, H, n_q, 4)
        v = v_proj.view(B, T, H, n_v, 4)
        k = k / (k.norm(dim=-1, keepdim=True) + 1e-9)
        q_silu = F.silu(q)

        # --- RD path: parallel chunkwise scan (copied from parent) ---
        beta = torch.sigmoid(self.beta_proj(x)).view(B, T, H, n_q)
        alpha = torch.sigmoid(self.alpha_proj(x)).view(B, T, H, n_q)

        k_c = econj(k)
        M_kk = emul(k.unsqueeze(-2), k_c.unsqueeze(-3))
        eye_q = torch.zeros(n_q, n_q, 4, device=x.device, dtype=x.dtype)
        eye_q[torch.arange(n_q), torch.arange(n_q), 0] = 1.0
        eye_q = eye_q.expand(B, T, H, n_q, n_q, 4)
        beta_i = beta.unsqueeze(-1).unsqueeze(-1)
        alpha_i = alpha.unsqueeze(-1).unsqueeze(-1)
        A = alpha_i * (eye_q - beta_i * M_kk)

        v_c = econj(v)
        kv = emul(k.unsqueeze(-2), v_c.unsqueeze(-3))
        B_acc = beta.unsqueeze(-1).unsqueeze(-1) * kv
        A_acc = A
        ident_A = eye_q[:, :1]
        zero_B = torch.zeros_like(B_acc[:, :1])
        log_T = max(1, math.ceil(math.log2(max(T, 2))))
        for level in range(log_T):
            step = 1 << level
            if step >= T:
                break
            earlier_A = torch.cat([ident_A.expand(-1, step, -1, -1, -1, -1),
                                    A_acc[:, :T-step]], dim=1)
            earlier_B = torch.cat([zero_B.expand(-1, step, -1, -1, -1, -1),
                                    B_acc[:, :T-step]], dim=1)
            A_new = rmatmul(A_acc, earlier_A)
            B_new = rmatmul(A_acc, earlier_B) + B_acc
            A_acc, B_acc = A_new, B_new

        Y_rd = emul(q_silu.unsqueeze(-2), B_acc).sum(dim=-3)   # B,T,H,n_v,4
        y_rd = Y_rd.reshape(B, T, H * n_v * 4)
        y_rd = self.o_proj(self.dropout(y_rd))

        # --- Buffer path: flatten 4-component to a single vector, then softmax attn ---
        # q_flat / k_flat: (B, T, H, n_q*4); v_flat: (B, T, H, n_v*4)
        q_flat = q_silu.reshape(B, T, H, n_q * 4)
        k_flat = k.reshape(B, T, H, n_q * 4)
        v_flat = v.reshape(B, T, H, n_v * 4)

        d_q = n_q * 4
        W = self.W
        K_buf = []; V_buf = []
        y_buf_list = []
        for t in range(T):
            kt_rot = self._rope(k_flat[:, t], t)
            qt_rot = self._rope(q_flat[:, t], t)
            if len(K_buf) >= W:
                K_buf.pop(0); V_buf.pop(0)
            K_buf.append(kt_rot); V_buf.append(v_flat[:, t])
            K_stack = torch.stack(K_buf, dim=2)
            V_stack = torch.stack(V_buf, dim=2)
            scores = torch.einsum('bhd,bhld->bhl', qt_rot, K_stack) / math.sqrt(d_q)
            attn = self.attn_dropout(F.softmax(scores, dim=-1))
            yt = torch.einsum('bhl,bhld->bhd', attn, V_stack)   # B,H,n_v*4
            y_buf_list.append(yt)
        y_buf = torch.stack(y_buf_list, dim=1).reshape(B, T, H * n_v * 4)
        y_buf = self.buf_o_proj(self.dropout(y_buf))

        return y_rd + self.buf_mix * y_buf


class BufferedRDTemporalEncoder(RealDeltaNetTemporalEncoder):
    """RealDeltaNetTemporalEncoder with BufferedRealDeltaBlock blocks."""
    def __init__(self, in_channels, hidden_dim=128, output_dim=None, num_layers=2,
                 num_heads=4, n_q=4, n_v=8, buffer_size=4, dropout=0.3,
                 bidirectional=True):
        nn.Module.__init__(self)
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim if output_dim is not None else hidden_dim
        self.bidirectional = bidirectional
        self.input_proj = nn.Linear(in_channels, hidden_dim)
        self.fwd_blocks = nn.ModuleList([
            BufferedRealDeltaBlock(hidden_dim, num_heads, n_q, n_v, buffer_size, dropout)
            for _ in range(num_layers)
        ])
        self.fwd_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        if bidirectional:
            self.bwd_blocks = nn.ModuleList([
                BufferedRealDeltaBlock(hidden_dim, num_heads, n_q, n_v, buffer_size, dropout)
                for _ in range(num_layers)
            ])
            self.bwd_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, self.output_dim)


class MotionBDRD(MotionRealDeltaNet):
    """PMamba with RealDeltaNet + buffer attention augment."""
    def __init__(self, *args, rd_hidden_dim=128, rd_num_layers=2, rd_num_heads=4,
                 rd_n_q=4, rd_n_v=8, rd_dropout=0.3, rd_bidirectional=True,
                 rd_buffer_size=4, **kwargs):
        # Call grandparent (Motion) directly to avoid MotionRealDeltaNet's mamba reset
        Motion.__init__(self, *args, **kwargs)
        old = self.mamba
        in_c = old.in_channels
        out_d = old.output_dim
        self.mamba = BufferedRDTemporalEncoder(
            in_channels=in_c, hidden_dim=rd_hidden_dim, output_dim=out_d,
            num_layers=rd_num_layers, num_heads=rd_num_heads, n_q=rd_n_q,
            n_v=rd_n_v, buffer_size=rd_buffer_size, dropout=rd_dropout,
            bidirectional=rd_bidirectional,
        )

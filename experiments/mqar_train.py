"""MQAR (Multi-Query Associative Recall) — head-to-head AttRD vs DeltaNet.

Task: a sequence of (key, value) pairs followed by queries; predict the value
for each query. Standard benchmark from Zoology / DeltaNet / Mamba-2 papers.

Log format matches tg_messages.sh parser:
    Training epoch: N
    Mean training acc: X
    Mean training loss: X
    Test, Evaluation: Epoch N ... prec1 X, prec5 Y
"""
import argparse, math, os, sys, time, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# MQAR data generator
# -----------------------------------------------------------------------------
def make_mqar_batch(B, T, vocab, num_kv, num_q, device, rng):
    """One batch of MQAR.

    Layout per row: [k1 v1 k2 v2 ... k_n v_n  q1 ? q2 ? ... q_m ?]
        - keys/values drawn iid from [2, vocab-1]; 0 = pad, 1 = ?-placeholder
        - queries are sampled WITH replacement from existing keys
        - target at the '?' position is the value for that key
        - loss only on '?' positions
    """
    assert T >= 2 * num_kv + 2 * num_q
    x = torch.zeros(B, T, dtype=torch.long, device=device)
    y = torch.full((B, T), -100, dtype=torch.long, device=device)
    QP = 1  # query placeholder token
    for b in range(B):
        keys = rng.choice(vocab - 2, num_kv, replace=False) + 2
        vals = rng.integers(2, vocab, num_kv)
        kv_seq = np.empty(2 * num_kv, dtype=np.int64)
        kv_seq[0::2] = keys
        kv_seq[1::2] = vals
        # Random pad before kv block
        kv_start = 0
        x[b, kv_start:kv_start + 2 * num_kv] = torch.from_numpy(kv_seq).to(device)
        # Queries: sample with replacement
        idx = rng.integers(0, num_kv, num_q)
        q_start = 2 * num_kv
        for j, ki in enumerate(idx):
            pos = q_start + 2 * j
            x[b, pos]     = int(keys[ki])
            x[b, pos + 1] = QP
            y[b, pos + 1] = int(vals[ki])
    return x, y


# -----------------------------------------------------------------------------
# DeltaNet block (real-valued, canonical Schlag/Yang formulation, chunkwise scan)
# -----------------------------------------------------------------------------
class DeltaNetBlock(nn.Module):
    def __init__(self, d_model, num_heads=4, head_dim=32, dropout=0.1):
        super().__init__()
        self.H = num_heads
        self.D = head_dim
        d_inner = num_heads * head_dim
        self.q = nn.Linear(d_model, d_inner, bias=False)
        self.k = nn.Linear(d_model, d_inner, bias=False)
        self.v = nn.Linear(d_model, d_inner, bias=False)
        self.beta = nn.Linear(d_model, num_heads)
        self.o = nn.Linear(d_inner, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, _ = x.shape
        H, D = self.H, self.D
        q = self.q(x).view(B, T, H, D)
        k = F.normalize(self.k(x).view(B, T, H, D), dim=-1)
        v = self.v(x).view(B, T, H, D)
        beta = torch.sigmoid(self.beta(x)).view(B, T, H, 1)
        q = F.silu(q)

        # DeltaRule: S_t = S_{t-1} (I - beta_t k_t k_t^T) + beta_t k_t v_t^T
        # Causal sequential scan (slow but correct). T ~ 64-256 fine.
        S = torch.zeros(B, H, D, D, device=x.device, dtype=x.dtype)
        ys = []
        for t in range(T):
            kt = k[:, t]          # B,H,D
            vt = v[:, t]          # B,H,D
            bt = beta[:, t]       # B,H,1
            # Sk: (B,H,D) <- (B,H,D,D) @ k_t
            Sk = torch.einsum('bhij,bhj->bhi', S, kt)
            err = vt - Sk
            S = S + bt.unsqueeze(-1) * torch.einsum('bhi,bhj->bhij', err, kt)
            yt = torch.einsum('bhij,bhj->bhi', S, q[:, t])
            ys.append(yt)
        y = torch.stack(ys, dim=1).reshape(B, T, H * D)
        return self.o(self.dropout(y))


# =============================================================================
# Stage-3 novel architectures (designed to beat DN on MQAR)
# =============================================================================

class TwoPhaseDeltaBlock(nn.Module):
    """TP-DN: explicit write/query phase gate per step.

    Standard DN writes at every step with sigmoid β. TP-DN multiplies β by an
    additional sigmoid gate ψ that the model can learn to drive to 0 during
    query positions — so registration writes are clean and queries don't
    overwrite past KV pairs.

    Write: S_t = S_{t-1} + (ψ_t · β_t) · (v_t - S_{t-1} k_t) k_t^T
    Read:  y_t = q_t^T S_t                                  (same as DN)
    """
    def __init__(self, d_model, num_heads=4, head_dim=32, dropout=0.1):
        super().__init__()
        self.H = num_heads
        self.D = head_dim
        d_inner = num_heads * head_dim
        self.q = nn.Linear(d_model, d_inner, bias=False)
        self.k = nn.Linear(d_model, d_inner, bias=False)
        self.v = nn.Linear(d_model, d_inner, bias=False)
        self.beta = nn.Linear(d_model, num_heads)
        self.phase = nn.Linear(d_model, num_heads)        # write-mode gate
        self.o = nn.Linear(d_inner, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, _ = x.shape
        H, D = self.H, self.D
        q = self.q(x).view(B, T, H, D)
        k = F.normalize(self.k(x).view(B, T, H, D), dim=-1)
        v = self.v(x).view(B, T, H, D)
        beta = torch.sigmoid(self.beta(x)).view(B, T, H, 1)
        psi  = torch.sigmoid(self.phase(x)).view(B, T, H, 1)
        q = F.silu(q)

        S = torch.zeros(B, H, D, D, device=x.device, dtype=x.dtype)
        ys = []
        for t in range(T):
            kt = k[:, t]; vt = v[:, t]
            bt = beta[:, t] * psi[:, t]    # write-mode gated rate
            Sk = torch.einsum('bhij,bhj->bhi', S, kt)
            err = vt - Sk
            S = S + bt.unsqueeze(-1) * torch.einsum('bhi,bhj->bhij', err, kt)
            yt = torch.einsum('bhij,bhj->bhi', S, q[:, t])
            ys.append(yt)
        y = torch.stack(ys, dim=1).reshape(B, T, H * D)
        return self.o(self.dropout(y))


class AdaptiveBetaDeltaBlock(nn.Module):
    """AdaB-DN: β adapts to whether the state already contains the current key.

    β_t = σ(W·x_t + γ · ||S_{t-1} k_t||)
    When state already contains k_t direction → high ||S k|| → high β → overwrite.
    When k_t is novel → low ||S k|| → β follows the input-driven term.

    Gives the model a content-aware overwrite rate, in contrast to DN's
    input-only β.
    """
    def __init__(self, d_model, num_heads=4, head_dim=32, dropout=0.1):
        super().__init__()
        self.H = num_heads
        self.D = head_dim
        d_inner = num_heads * head_dim
        self.q = nn.Linear(d_model, d_inner, bias=False)
        self.k = nn.Linear(d_model, d_inner, bias=False)
        self.v = nn.Linear(d_model, d_inner, bias=False)
        self.beta_lin = nn.Linear(d_model, num_heads)
        self.gamma = nn.Parameter(torch.zeros(num_heads))     # learned scale, init 0 (= plain DN)
        self.o = nn.Linear(d_inner, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, _ = x.shape
        H, D = self.H, self.D
        q = self.q(x).view(B, T, H, D)
        k = F.normalize(self.k(x).view(B, T, H, D), dim=-1)
        v = self.v(x).view(B, T, H, D)
        beta_inp = self.beta_lin(x).view(B, T, H)             # pre-sigmoid
        q = F.silu(q)
        gamma = self.gamma.view(1, H)                          # 1,H

        S = torch.zeros(B, H, D, D, device=x.device, dtype=x.dtype)
        ys = []
        for t in range(T):
            kt = k[:, t]; vt = v[:, t]
            Sk = torch.einsum('bhij,bhj->bhi', S, kt)         # B,H,D
            Sk_norm = Sk.norm(dim=-1)                          # B,H — how loaded state is along k_t
            bt = torch.sigmoid(beta_inp[:, t] + gamma * Sk_norm).unsqueeze(-1)  # B,H,1
            err = vt - Sk
            S = S + bt.unsqueeze(-1) * torch.einsum('bhi,bhj->bhij', err, kt)
            yt = torch.einsum('bhij,bhj->bhi', S, q[:, t])
            ys.append(yt)
        y = torch.stack(ys, dim=1).reshape(B, T, H * D)
        return self.o(self.dropout(y))


class ForgetDeltaBlock(nn.Module):
    """FD-N: explicit forget op along k_t before the rank-1 write.

    S_t = (I - m_t k_t k_t^T) S_{t-1} + β_t (v_t - S_{t-1} k_t) k_t^T
    The (I - m_t k_t k_t^T) is a soft projection: subtracts contribution along
    the k_t direction with magnitude m_t ∈ [0,1]. m_t=0 → plain DN; m_t=1 →
    fully clears the k_t component before writing.

    Surgical forgetting vs DN's "overwrite-only" semantics.
    """
    def __init__(self, d_model, num_heads=4, head_dim=32, dropout=0.1):
        super().__init__()
        self.H = num_heads
        self.D = head_dim
        d_inner = num_heads * head_dim
        self.q = nn.Linear(d_model, d_inner, bias=False)
        self.k = nn.Linear(d_model, d_inner, bias=False)
        self.v = nn.Linear(d_model, d_inner, bias=False)
        self.beta = nn.Linear(d_model, num_heads)
        self.forget = nn.Linear(d_model, num_heads)
        self.o = nn.Linear(d_inner, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, _ = x.shape
        H, D = self.H, self.D
        q = self.q(x).view(B, T, H, D)
        k = F.normalize(self.k(x).view(B, T, H, D), dim=-1)
        v = self.v(x).view(B, T, H, D)
        beta = torch.sigmoid(self.beta(x)).view(B, T, H, 1)
        mt = torch.sigmoid(self.forget(x)).view(B, T, H, 1)
        q = F.silu(q)

        S = torch.zeros(B, H, D, D, device=x.device, dtype=x.dtype)
        ys = []
        for t in range(T):
            kt = k[:, t]; vt = v[:, t]
            # Subtract m_t * k_t k_t^T S along k_t direction:
            Sk = torch.einsum('bhij,bhj->bhi', S, kt)
            S = S - mt[:, t].unsqueeze(-1) * torch.einsum('bhi,bhj->bhij', Sk, kt)
            # Now standard delta write:
            Sk = torch.einsum('bhij,bhj->bhi', S, kt)
            err = vt - Sk
            S = S + beta[:, t].unsqueeze(-1) * torch.einsum('bhi,bhj->bhij', err, kt)
            yt = torch.einsum('bhij,bhj->bhi', S, q[:, t])
            ys.append(yt)
        y = torch.stack(ys, dim=1).reshape(B, T, H * D)
        return self.o(self.dropout(y))


class SlotHopfieldBlock(nn.Module):
    """SH-Mem: Slot-based content-addressable memory + Hopfield read.

    Fundamentally different substrate from rank-1 delta accumulation:
      State = N "slots", each a (key, value) pair (B, N, H, D)
      Write: content-addressable. Softmax over N slots gives soft assignment;
             each token's (k, v) is blended into the most-similar slot.
      Read:  Modern Hopfield retrieval. Softmax(β · q · slot_k / √D) over slots,
             with learned inverse temperature β. High β → hard pattern retrieval.

    Hypothesis for MQAR: bounded N slots = N independent KV pairs, no rank-1
    collisions. Different memory primitive than every prior variant.
    """
    def __init__(self, d_model, num_heads=4, head_dim=32,
                 num_slots=16, dropout=0.1):
        super().__init__()
        self.H = num_heads
        self.D = head_dim
        self.N = num_slots
        d_inner = num_heads * head_dim
        self.q = nn.Linear(d_model, d_inner, bias=False)
        self.k = nn.Linear(d_model, d_inner, bias=False)
        self.v = nn.Linear(d_model, d_inner, bias=False)
        self.write_addr = nn.Linear(d_model, num_heads * num_slots, bias=False)
        # Hopfield inverse temperature (learned, init to 1/sqrt(D) like standard attention)
        self.log_beta = nn.Parameter(torch.zeros(num_heads))
        self.o = nn.Linear(d_inner, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, _ = x.shape
        H, D, N = self.H, self.D, self.N
        q = self.q(x).view(B, T, H, D)
        k = F.normalize(self.k(x).view(B, T, H, D), dim=-1)
        v = self.v(x).view(B, T, H, D)
        q = F.silu(q)
        # B,T,H,N soft write addresses
        w_logits = self.write_addr(x).view(B, T, H, N)
        w_addr = F.softmax(w_logits, dim=-1)
        # Slot memory: (B, H, N, D)
        slot_k = torch.zeros(B, H, N, D, device=x.device, dtype=x.dtype)
        slot_v = torch.zeros(B, H, N, D, device=x.device, dtype=x.dtype)
        beta = (1.0 / math.sqrt(D)) * torch.exp(self.log_beta).view(1, H, 1)  # 1,H,1

        ys = []
        for t in range(T):
            kt = k[:, t]; vt = v[:, t]; wa = w_addr[:, t]   # wa: B,H,N
            # Soft write: blend (kt, vt) into slots by wa
            # New slot = (1 - wa) * old + wa * new_value
            wa_exp = wa.unsqueeze(-1)                        # B,H,N,1
            slot_k = (1 - wa_exp) * slot_k + wa_exp * kt.unsqueeze(-2)
            slot_v = (1 - wa_exp) * slot_v + wa_exp * vt.unsqueeze(-2)
            # Hopfield read: softmax(β · q · slot_k) over N slots
            scores = torch.einsum('bhd,bhnd->bhn', q[:, t], slot_k) * beta  # B,H,N
            attn = self.attn_dropout(F.softmax(scores, dim=-1))
            yt = torch.einsum('bhn,bhnd->bhd', attn, slot_v)
            ys.append(yt)
        y = torch.stack(ys, dim=1).reshape(B, T, H * D)
        return self.o(self.dropout(y))


class TTTDeltaBlock(nn.Module):
    """TTT-DN: Test-Time Training applied to the delta state.

    Standard DeltaNet does ONE gradient step per token on the inner loss
    ||S k_t - v_t||² (with step size β). TTT-DN takes K inner steps per token
    with a learned per-step LR. State refinement is more thorough; small writes
    accumulate to higher recall.

    For K=1 with input-driven step, equals plain DN. For K>1 it's strictly
    a generalisation. Mechanism resembles TTT (Sun 2024) but applied to a
    delta-rule outer-product state rather than an MLP.
    """
    def __init__(self, d_model, num_heads=4, head_dim=32, K=3, dropout=0.1):
        super().__init__()
        self.H = num_heads
        self.D = head_dim
        self.K_steps = K
        d_inner = num_heads * head_dim
        self.q = nn.Linear(d_model, d_inner, bias=False)
        self.k = nn.Linear(d_model, d_inner, bias=False)
        self.v = nn.Linear(d_model, d_inner, bias=False)
        # K learned step sizes per head, all softplus-positive
        self.step_log = nn.Parameter(torch.zeros(K, num_heads))
        # Plus input-dep gate scaling on the first step (like β)
        self.beta = nn.Linear(d_model, num_heads)
        self.o = nn.Linear(d_inner, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, _ = x.shape
        H, D = self.H, self.D
        K = self.K_steps
        q = self.q(x).view(B, T, H, D)
        k = F.normalize(self.k(x).view(B, T, H, D), dim=-1)
        v = self.v(x).view(B, T, H, D)
        beta = torch.sigmoid(self.beta(x)).view(B, T, H, 1, 1)
        q = F.silu(q)
        steps = F.softplus(self.step_log)                                   # K,H

        S = torch.zeros(B, H, D, D, device=x.device, dtype=x.dtype)
        ys = []
        for t in range(T):
            kt = k[:, t]; vt = v[:, t]
            for ki in range(K):
                eta = steps[ki].view(1, H, 1, 1)                            # 1,H,1,1
                Sk = torch.einsum('bhij,bhj->bhi', S, kt)
                err = vt - Sk
                # 1st step also modulated by β (input-driven gate)
                if ki == 0:
                    eta = eta * beta[:, t]
                S = S + eta * torch.einsum('bhi,bhj->bhij', err, kt)
            yt = torch.einsum('bhij,bhj->bhi', S, q[:, t])
            ys.append(yt)
        y = torch.stack(ys, dim=1).reshape(B, T, H * D)
        return self.o(self.dropout(y))


class Mamba2Block(nn.Module):
    """Mamba-2 / SSD: selective scan with diagonal scalar A.

    State recurrence per head:
        h_t = a_t * h_{t-1} + b_t * v_t · k_t^T          (rank-1 add, scalar gate)
        y_t = q_t^T h_t

    a_t = exp(-softplus(W_a x_t))   scalar in (0,1) (forget gate per head)
    b_t = σ(W_b x_t)                scalar in (0,1) (input gate per head)
    Selective: a_t and b_t depend on x_t.

    This is the SSD (state-space duality) form: A is diagonal scalar per head
    so the scan can be matmul-ified. We do a sequential scan here for
    simplicity at T=64-128 sizes.
    """
    def __init__(self, d_model, num_heads=4, head_dim=32, dropout=0.1):
        super().__init__()
        self.H = num_heads
        self.D = head_dim
        d_inner = num_heads * head_dim
        self.q = nn.Linear(d_model, d_inner, bias=False)
        self.k = nn.Linear(d_model, d_inner, bias=False)
        self.v = nn.Linear(d_model, d_inner, bias=False)
        self.a_proj = nn.Linear(d_model, num_heads)
        self.b_proj = nn.Linear(d_model, num_heads)
        self.o = nn.Linear(d_inner, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, _ = x.shape
        H, D = self.H, self.D
        q = self.q(x).view(B, T, H, D)
        k = self.k(x).view(B, T, H, D)
        v = self.v(x).view(B, T, H, D)
        # Discretized A and B; A in (0,1), B in (0,1) via gates.
        a = torch.exp(-F.softplus(self.a_proj(x))).view(B, T, H, 1, 1)   # forget
        b = torch.sigmoid(self.b_proj(x)).view(B, T, H, 1, 1)             # write
        q = F.silu(q)

        h = torch.zeros(B, H, D, D, device=x.device, dtype=x.dtype)
        ys = []
        for t in range(T):
            kt = k[:, t]; vt = v[:, t]
            kv = torch.einsum('bhi,bhj->bhij', vt, kt)    # rank-1 outer
            h = a[:, t] * h + b[:, t] * kv
            yt = torch.einsum('bhij,bhj->bhi', h, q[:, t])
            ys.append(yt)
        y = torch.stack(ys, dim=1).reshape(B, T, H * D)
        return self.o(self.dropout(y))


class MoEDeltaBlock(nn.Module):
    """MoE-DN: Mixture-of-Experts DeltaNet.

    Each token is routed (top-k softmax over E experts) into k of E independent
    delta heads. Each expert maintains its own state S^e ∈ R^{D×D}. The router
    is a linear classifier on x_t. Read combines top-k experts weighted by
    router scores.

    Hypothesis: MQAR's K distinct KV pairs can be stored in K distinct experts
    without rank-1 interference. Standard DN crams them all into one S.
    """
    def __init__(self, d_model, num_heads=4, head_dim=32,
                 num_experts=8, top_k=2, dropout=0.1):
        super().__init__()
        self.H = num_heads
        self.D = head_dim
        self.E = num_experts
        self.K = top_k
        d_inner = num_heads * head_dim
        self.q = nn.Linear(d_model, d_inner, bias=False)
        self.k = nn.Linear(d_model, d_inner, bias=False)
        self.v = nn.Linear(d_model, d_inner, bias=False)
        self.beta = nn.Linear(d_model, num_heads)
        self.router = nn.Linear(d_model, num_experts, bias=False)
        self.o = nn.Linear(d_inner, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, _ = x.shape
        H, D, E, K = self.H, self.D, self.E, self.K
        q = self.q(x).view(B, T, H, D)
        k = F.normalize(self.k(x).view(B, T, H, D), dim=-1)
        v = self.v(x).view(B, T, H, D)
        beta = torch.sigmoid(self.beta(x)).view(B, T, H, 1)
        q = F.silu(q)
        # Router: B,T,E logits → top-k indices + weights
        router_logits = self.router(x)                       # B,T,E
        top_w, top_idx = router_logits.topk(K, dim=-1)        # both B,T,K
        top_w = F.softmax(top_w, dim=-1)                      # normalize within top-k

        # Per-expert states: (B, E, H, D, D). Allocating all is fine since E small.
        S = torch.zeros(B, E, H, D, D, device=x.device, dtype=x.dtype)
        ys = []
        for t in range(T):
            kt = k[:, t]; vt = v[:, t]; bt = beta[:, t]
            idx_t = top_idx[:, t]                              # B,K
            w_t = top_w[:, t]                                  # B,K
            # For each of the K routed experts, write update on their state.
            # We do this in a loop over K (small) for clarity.
            for ki in range(K):
                ei = idx_t[:, ki]                              # B
                Se = S[torch.arange(B), ei]                    # B,H,D,D
                Sk = torch.einsum('bhij,bhj->bhi', Se, kt)
                err = vt - Sk
                S[torch.arange(B), ei] = Se + (w_t[:, ki:ki+1].unsqueeze(-1).unsqueeze(-1) *
                                                bt.unsqueeze(-1) *
                                                torch.einsum('bhi,bhj->bhij', err, kt))
            # Read: weighted sum over routed experts
            y_t = torch.zeros(B, H, D, device=x.device, dtype=x.dtype)
            for ki in range(K):
                ei = idx_t[:, ki]
                Se = S[torch.arange(B), ei]
                y_t = y_t + w_t[:, ki:ki+1].unsqueeze(-1) * torch.einsum('bhij,bhj->bhi', Se, q[:, t])
            ys.append(y_t)
        y = torch.stack(ys, dim=1).reshape(B, T, H * D)
        return self.o(self.dropout(y))


class BufferedDeltaBlock(nn.Module):
    """BD-N: short attention KV-cache for recent + delta state for distant.

    Maintains a sliding window of (k, v) pairs (size W). At time t, output is
    formed by:
      buf-attention: softmax(q_t · K_buf / sqrt(D)) · V_buf
      delta-read:    q_t^T S_t                        (S = long-term delta state)
      y_t = buf_out + delta_out

    On each step the new (k_t, v_t) enters the buffer; the *oldest* entry
    rolls into the delta state via the standard rank-1 delta update.

    Hypothesis: registration of W consecutive distinct KV pairs lives cleanly
    in the buffer (no rank-1 collisions); distant context still accessible
    via S. Directly addresses MQAR's KV-then-query structure.
    """
    def __init__(self, d_model, num_heads=4, head_dim=32, buffer_size=16, dropout=0.1):
        super().__init__()
        self.H = num_heads
        self.D = head_dim
        self.W = buffer_size
        d_inner = num_heads * head_dim
        self.q = nn.Linear(d_model, d_inner, bias=False)
        self.k = nn.Linear(d_model, d_inner, bias=False)
        self.v = nn.Linear(d_model, d_inner, bias=False)
        self.beta = nn.Linear(d_model, num_heads)
        self.o = nn.Linear(d_inner, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, _ = x.shape
        H, D, W = self.H, self.D, self.W
        q = self.q(x).view(B, T, H, D)
        k = F.normalize(self.k(x).view(B, T, H, D), dim=-1)
        v = self.v(x).view(B, T, H, D)
        beta = torch.sigmoid(self.beta(x)).view(B, T, H, 1)
        q = F.silu(q)

        # Long-term delta state
        S = torch.zeros(B, H, D, D, device=x.device, dtype=x.dtype)
        # Buffer of recent (k, v); start empty. Use lists for simplicity.
        K_buf = []      # entries are tensors B,H,D
        V_buf = []
        ys = []
        for t in range(T):
            kt = k[:, t]; vt = v[:, t]; bt = beta[:, t]
            # Eject oldest entry from buffer into delta state (if buffer full)
            if len(K_buf) >= W:
                kt_old = K_buf.pop(0)
                vt_old = V_buf.pop(0)
                Sk = torch.einsum('bhij,bhj->bhi', S, kt_old)
                err = vt_old - Sk
                S = S + bt.unsqueeze(-1) * torch.einsum('bhi,bhj->bhij', err, kt_old)
            K_buf.append(kt); V_buf.append(vt)
            # Buffer attention (causal: only over current buffer contents)
            K_stack = torch.stack(K_buf, dim=2)   # B,H,L,D where L=len(K_buf)
            V_stack = torch.stack(V_buf, dim=2)
            scores = torch.einsum('bhd,bhld->bhl', q[:, t], K_stack) / math.sqrt(D)
            attn = self.attn_dropout(F.softmax(scores, dim=-1))
            buf_out = torch.einsum('bhl,bhld->bhd', attn, V_stack)        # B,H,D
            # Delta read of long-term state
            delta_out = torch.einsum('bhij,bhj->bhi', S, q[:, t])
            yt = buf_out + delta_out
            ys.append(yt)
        y = torch.stack(ys, dim=1).reshape(B, T, H * D)
        return self.o(self.dropout(y))


# -----------------------------------------------------------------------------
# AttRD block: same DeltaRule write, attention read over the full {S_τ} sequence
# -----------------------------------------------------------------------------
class AttRDBlock(nn.Module):
    def __init__(self, d_model, num_heads=4, head_dim=32, d_read=32, dropout=0.1):
        super().__init__()
        self.H = num_heads
        self.D = head_dim
        self.d_read = d_read
        d_inner = num_heads * head_dim
        self.q = nn.Linear(d_model, d_inner, bias=False)
        self.k = nn.Linear(d_model, d_inner, bias=False)
        self.v = nn.Linear(d_model, d_inner, bias=False)
        self.beta = nn.Linear(d_model, num_heads)
        self.read_q = nn.Linear(d_model, num_heads * d_read, bias=False)
        self.read_k = nn.Linear(d_model, num_heads * d_read, bias=False)
        self.o = nn.Linear(d_inner, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, _ = x.shape
        H, D, d_r = self.H, self.D, self.d_read
        q = self.q(x).view(B, T, H, D)
        k = F.normalize(self.k(x).view(B, T, H, D), dim=-1)
        v = self.v(x).view(B, T, H, D)
        beta = torch.sigmoid(self.beta(x)).view(B, T, H, 1)
        q = F.silu(q)

        # Same DeltaRule, but stash every S_t  (B, T, H, D, D)
        S = torch.zeros(B, H, D, D, device=x.device, dtype=x.dtype)
        S_seq = []
        for t in range(T):
            kt = k[:, t]; vt = v[:, t]; bt = beta[:, t]
            Sk = torch.einsum('bhij,bhj->bhi', S, kt)
            err = vt - Sk
            S = S + bt.unsqueeze(-1) * torch.einsum('bhi,bhj->bhij', err, kt)
            S_seq.append(S)
        S_seq = torch.stack(S_seq, dim=1)   # B,T,H,D,D

        # V_pair[b,t,s,h,d] = q_t^T S_s  →  (B,T,T,H,D)
        V_pair = torch.einsum('bthd,bshde->btshe', q, S_seq)

        rq = self.read_q(x).view(B, T, H, d_r)
        rk = self.read_k(x).view(B, T, H, d_r)
        scores = torch.einsum('bthd,bshd->bths', rq, rk) / math.sqrt(d_r)

        # Causal mask: query at t can only read S_s for s <= t. scores: B,T_q,H,T_s
        mask = torch.ones(T, T, device=x.device).tril().bool()
        scores = scores.masked_fill(~mask.view(1, T, 1, T), float('-inf'))
        attn = self.attn_dropout(F.softmax(scores, dim=-1))   # softmax over s

        Y = torch.einsum('bths,btshe->bthe', attn, V_pair)
        Y = Y.reshape(B, T, H * D)
        return self.o(self.dropout(Y))


# -----------------------------------------------------------------------------
# Transformer block: vanilla causal multi-head softmax attention + MLP
# Reference baseline ("SOTA on MQAR" — attention is what MQAR rewards).
# -----------------------------------------------------------------------------
class TransformerBlock(nn.Module):
    """Causal multi-head softmax attention + MLP with RoPE positional encoding.

    RoPE is essential for MQAR — without positional information, attention is
    permutation-equivariant and cannot distinguish position-1 key from
    position-3 query.
    """
    def __init__(self, d_model, num_heads=4, head_dim=32, mlp_ratio=2, dropout=0.1,
                 max_seq_len=4096, rope_base=10000.0):
        super().__init__()
        self.H = num_heads
        self.D = head_dim
        d_inner = num_heads * head_dim
        self.qkv = nn.Linear(d_model, 3 * d_inner, bias=False)
        self.o = nn.Linear(d_inner, d_model, bias=False)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        d_ffn = mlp_ratio * d_model
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ffn), nn.GELU(),
            nn.Linear(d_ffn, d_model),
        )
        self.dropout = nn.Dropout(dropout)
        self.attn_dropout = nn.Dropout(dropout)
        # RoPE cache: cos/sin tables for pair-rotation in head_dim halves
        assert head_dim % 2 == 0, "RoPE needs even head_dim"
        inv_freq = 1.0 / (rope_base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        pos = torch.arange(max_seq_len).float()
        freqs = torch.einsum('i,j->ij', pos, inv_freq)         # (T_max, head_dim/2)
        self.register_buffer('cos_cache', freqs.cos(), persistent=False)
        self.register_buffer('sin_cache', freqs.sin(), persistent=False)

    def _apply_rope(self, x):
        # x: B,T,H,D
        T = x.shape[1]
        cos = self.cos_cache[:T].view(1, T, 1, -1)   # 1,T,1,D/2
        sin = self.sin_cache[:T].view(1, T, 1, -1)
        x1, x2 = x[..., 0::2], x[..., 1::2]
        rot1 = x1 * cos - x2 * sin
        rot2 = x1 * sin + x2 * cos
        out = torch.stack([rot1, rot2], dim=-1).flatten(-2)
        return out

    def forward(self, x):
        B, T, _ = x.shape
        H, D = self.H, self.D
        h = self.ln1(x)
        qkv = self.qkv(h).view(B, T, 3, H, D)
        q, k, v = qkv.unbind(dim=2)             # each B,T,H,D
        q = self._apply_rope(q)
        k = self._apply_rope(k)
        scores = torch.einsum('bthd,bshd->bths', q, k) / math.sqrt(D)
        mask = torch.ones(T, T, device=x.device).tril().bool()
        scores = scores.masked_fill(~mask.view(1, T, 1, T), float('-inf'))
        attn = self.attn_dropout(F.softmax(scores, dim=-1))
        Y = torch.einsum('bths,bshd->bthd', attn, v).reshape(B, T, H * D)
        x = x + self.dropout(self.o(Y))
        x = x + self.dropout(self.mlp(self.ln2(x)))
        return x


# -----------------------------------------------------------------------------
# Common wrapper: embed → stack of blocks (residual + norm) → LM head
# -----------------------------------------------------------------------------
class SeqModel(nn.Module):
    def __init__(self, arch, vocab, d_model=128, num_layers=2, num_heads=4,
                 head_dim=32, d_read=32, mlp_ratio=2, buffer_size=16, dropout=0.1):
        super().__init__()
        self.arch = arch
        self.emb = nn.Embedding(vocab, d_model)
        blocks = []
        for _ in range(num_layers):
            if arch == 'deltanet':
                blocks.append(DeltaNetBlock(d_model, num_heads, head_dim, dropout))
            elif arch == 'attrd':
                blocks.append(AttRDBlock(d_model, num_heads, head_dim, d_read, dropout))
            elif arch == 'transformer':
                blocks.append(TransformerBlock(d_model, num_heads, head_dim, mlp_ratio, dropout))
            elif arch == 'tpdn':
                blocks.append(TwoPhaseDeltaBlock(d_model, num_heads, head_dim, dropout))
            elif arch == 'adabdn':
                blocks.append(AdaptiveBetaDeltaBlock(d_model, num_heads, head_dim, dropout))
            elif arch == 'fdn':
                blocks.append(ForgetDeltaBlock(d_model, num_heads, head_dim, dropout))
            elif arch == 'bdn':
                blocks.append(BufferedDeltaBlock(d_model, num_heads, head_dim,
                                                   buffer_size=buffer_size, dropout=dropout))
            elif arch == 'moedn':
                blocks.append(MoEDeltaBlock(d_model, num_heads, head_dim, dropout=dropout))
            elif arch == 'mamba2':
                blocks.append(Mamba2Block(d_model, num_heads, head_dim, dropout=dropout))
            elif arch == 'tttdn':
                blocks.append(TTTDeltaBlock(d_model, num_heads, head_dim, dropout=dropout))
            elif arch == 'slothop':
                blocks.append(SlotHopfieldBlock(d_model, num_heads, head_dim, dropout=dropout))
            else:
                raise ValueError(arch)
        self.blocks = nn.ModuleList(blocks)
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])
        self.final_norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)
        self.head.weight = self.emb.weight  # tied

    def forward(self, x):
        h = self.emb(x)
        for blk, norm in zip(self.blocks, self.norms):
            if self.arch == 'transformer':
                # Transformer block has its own internal residuals + pre-norms
                h = blk(h)
            else:
                h = h + blk(norm(h))
        return self.head(self.final_norm(h))


# -----------------------------------------------------------------------------
# Train / eval
# -----------------------------------------------------------------------------
def run(arch, args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    model = SeqModel(arch, args.vocab, args.d_model, args.layers,
                     args.heads, args.head_dim, args.d_read, dropout=0.1).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'run: mqar_{arch}', flush=True)
    print(f'arch: {arch}  params: {n_params/1e6:.2f}M  T={args.T} kv={args.kv} q={args.q} vocab={args.vocab}', flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best_p1 = 0.0; best_ep = 0
    for ep in range(args.epochs):
        model.train()
        tot_loss = 0.0; tot_correct = 0; tot_tokens = 0
        print(f'Training epoch: {ep}', flush=True)
        for step in range(args.steps_per_epoch):
            x, y = make_mqar_batch(args.bs, args.T, args.vocab, args.kv, args.q, device, rng)
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, args.vocab), y.reshape(-1), ignore_index=-100)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            with torch.no_grad():
                pred = logits.argmax(-1)
                m = (y != -100)
                tot_correct += ((pred == y) & m).sum().item()
                tot_tokens  += m.sum().item()
                tot_loss    += loss.item()
        sched.step()
        tr_acc = 100 * tot_correct / max(1, tot_tokens)
        tr_loss = tot_loss / args.steps_per_epoch
        print(f'Mean training acc: {tr_acc:.4f}', flush=True)
        print(f'Mean training loss: {tr_loss:.4f}', flush=True)

        # Eval
        model.eval()
        p1 = 0; p5 = 0; ntot = 0
        rng_eval = np.random.default_rng(123456 + ep)  # held-out seed family
        with torch.no_grad():
            for _ in range(args.eval_steps):
                x, y = make_mqar_batch(args.bs, args.T, args.vocab, args.kv, args.q, device, rng_eval)
                logits = model(x)
                m = (y != -100)
                top5 = logits.topk(5, dim=-1).indices       # B,T,5
                correct1 = (top5[..., 0] == y) & m
                correct5 = (top5 == y.unsqueeze(-1)).any(-1) & m
                p1 += correct1.sum().item()
                p5 += correct5.sum().item()
                ntot += m.sum().item()
        p1_pct = 100 * p1 / max(1, ntot)
        p5_pct = 100 * p5 / max(1, ntot)
        print(f'Test, Evaluation: Epoch {ep} prec1 {p1_pct:.4f}, prec5 {p5_pct:.4f}', flush=True)
        if p1_pct > best_p1:
            best_p1 = p1_pct; best_ep = ep
        print(f'best: ep {best_ep} p1={best_p1:.2f}%', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--arch', choices=['deltanet', 'attrd'], required=True)
    p.add_argument('--vocab', type=int, default=256)
    p.add_argument('--T', type=int, default=64)
    p.add_argument('--kv', type=int, default=8)
    p.add_argument('--q', type=int, default=16)
    p.add_argument('--d_model', type=int, default=128)
    p.add_argument('--layers', type=int, default=2)
    p.add_argument('--heads', type=int, default=4)
    p.add_argument('--head_dim', type=int, default=32)
    p.add_argument('--d_read', type=int, default=32)
    p.add_argument('--bs', type=int, default=64)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--epochs', type=int, default=60)
    p.add_argument('--steps_per_epoch', type=int, default=50)
    p.add_argument('--eval_steps', type=int, default=20)
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()
    run(args.arch, args)

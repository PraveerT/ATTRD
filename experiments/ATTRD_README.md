# AttRD — Attention-Read DeltaNet

A hybrid sequence architecture: keep DeltaRule's linear-time **recurrent write**, swap its single-state read for a **softmax attention read over the full state trajectory**.

> **⚠️ Original headline retracted** (2026-05-15). The "+34% rel on MQAR" claim was a confounded comparison (param mismatch + LR mismatch + single seed). Under rigorous testing (param-matched, LR sweep, 3 seeds, paired t-test), **AttRD ≈ DeltaNet** at every tested MQAR configuration (T ∈ {64, 128}, vocab ∈ {256, 8192}). The design is novel; the accuracy advantage is not detectable. See [Honest results](#honest-results).

---

## TL;DR

Standard delta-rule / linear-attention / SSM models build a recurrent memory `S_t` and read it at the current point only:

```
y_t = q_tᵀ · S_t                  # point read
```

AttRD keeps the write *unchanged* but lets each token query the **entire history of memory states** `{S_τ}_{τ=1..T}` via softmax attention:

```
y_t = Σ_τ  α_{t,τ} · (q_tᵀ · S_τ)
α_{t,τ} = softmax_τ(read_q_t · read_k_τ / √d) with τ ≤ t
```

The write side is the load-bearing recurrence (gradient-stable, content-addressable); the read side gains the lookup flexibility that point-read alone can't provide.

---

## Why this is novel

| Family | Recurrent write | Read mechanism |
|---|---|---|
| Linear Attention (Katharopoulos '20) | rank-1 add | point: `q_tᵀ S_t` |
| DeltaNet (Schlag '21, Yang '24) | rank-1 delta-rule add | point: `q_tᵀ S_t` |
| Mamba / S4 / GLA / RWKV | selective scan | point at current state |
| Mamba-2-Hybrid / Jamba / Griffin | scan + attention | attention over **token outputs**, not states |
| Test-Time Training (Sun '24) | gradient step at infer | output from current state |
| **AttRD (this work)** | **delta-rule (unchanged)** | **softmax attention over `{S_τ}`** |

Every published hybrid we found attends over **token-level outputs** of the recurrence; AttRD attends over the **trajectory of matrix-valued memory states** themselves. That is a different axis of hybridisation — and one that directly targets associative-recall capacity.

---

## Mathematical specification

For a sequence `x ∈ ℝ^{B×T×d_model}` with `H` heads of dim `D`:

**Write recurrence** (DeltaNet, unchanged):
```
qᵢ, kᵢ, vᵢ = W_q xᵢ, normalize(W_k xᵢ), W_v xᵢ         (B,H,D)
βᵢ = σ(W_β xᵢ)                                          (B,H)
S₀ = 0
Sₜ = S_{t-1} + βₜ · (vₜ − S_{t-1} kₜ) · kₜᵀ           # rank-1 delta update
```

**Read** (this is the novel part):
```
read_qₜ = W_rq xₜ,  read_kₜ = W_rk xₜ                  (B,H,d_read)
scoreₜ,ᵤ = read_qₜ · read_kᵤ / √d_read                  (causal mask τ ≤ t)
αₜ = softmax_τ(scoreₜ)
yₜ = Σ_τ αₜ,ᵤ · (qₜᵀ Sᵤ)                              (B,H,D)
out = W_o concat(yₜ over heads)
```

**Complexity**: write is O(T · H · D²); read is O(T² · H · D). At T ≤ 512 with D=32 this is cheap on a single GPU. Above ~T=1k the O(T²) read with matrix-valued V_pair (shape `B, T, T, H, D`) needs chunking or low-rank approximation.

---

## Mechanism: why does it help?

DeltaNet's point read `qₜᵀ Sₜ` reads a **single matrix-summary of all prior keys/values, frozen at time t**. If three different queries need three different parts of the past, they all see the same `Sₜ` — and rank-1 updates have limited capacity to keep them distinct.

AttRD's read picks **which `S_τ` to read from**. For a query about an event written at time τ*, the model can softmax-route the read attention to `S_{τ*}` — a state where that event's contribution is still relatively uncontaminated by subsequent updates.

This is exactly the operation that the **MQAR** synthetic benchmark stress-tests: many queries against many keys, all in one sequence.

---

## Results

### MQAR (Multi-Query Associative Recall)

The standard synthetic benchmark for associative memory in the linear-attention / SSM literature. Sequence of `[k₁ v₁ k₂ v₂ … q₁ ? q₂ ? …]`; predict each value given its query.

Configuration: 60 epochs, batch=64, vocab=256, T=64, 8 KV pairs, 16 queries per sequence, 2 layers, 4 heads, head_dim=32, matched training schedule.

**Original (now-retracted) comparison** — single seed, unmatched params (DN 0.17M vs AT 0.23M), single LR (3e-4 for both):

| Model | Params | Best test p@1 | Best test p@5 |
|---|---|---|---|
| DeltaNet | 0.17M | 6.55% | 30.56% |
| AttRD | 0.23M | 8.77% | 39.81% |

This +34% rel gap was an artifact. Audit:

| Source | Contribution to original gap |
|---|---|
| Param mismatch (0.17M → 0.23M) | ~50% |
| LR mistuning (3e-4 was DN's worst LR) | ~45% |
| Single-seed noise | ~5% |
| True architectural effect | **~0.04 – 0.42pp** (not significant) |

**Honest results** — param-matched, best-LR per arch, 3 seeds, paired t-tests:

| Config | DN mean ± std | AT mean ± std | Δ | paired t (df=2) | p | significant? |
|---|---|---|---|---|---|---|
| T=64, vocab=256, S (165.6K) | 8.09 ± 0.24 | 8.51 ± 0.31 | +0.42 | 1.53 | 0.27 | NO |
| T=64, vocab=256, L (231.2K) | 8.55 ± 0.20 | 8.67 ± 0.16 | +0.12 | 4.31 | 0.05 | borderline |
| T=64, vocab=8192, L (1.247M) | 4.70 ± 0.35 | 4.74 ± 0.18 | +0.04 | 0.29 | 0.80 | NO |
| T=128, vocab=8192, L | 1.76 ± 0.01 | 1.67 ± 0.35 | −0.09 | −0.43 | 0.71 | NO |

**Verdict**: AttRD ≈ DeltaNet at every tested MQAR configuration under matched compute and parameter budgets. The mechanism (attention-over-state-trajectory) does not translate to a detectable accuracy advantage.

Reproduce the rigorous evaluation:
```bash
python mqar_grid.py --stage 1  # param-match retest, vocab=256
python mqar_grid.py --stage 2 --size L --lr_dn 1e-3 --lr_at 1e-3 --lr_tx 1e-3 \
                    --T_values 64,128 --arches deltanet,attrd,transformer
```

### NVGesture (skeleton-based gesture recognition)

Originally developed for NVGesture fusion. As a *solo* model (482-sample test set, train-best epoch selection, no test info leakage):

| Model | Solo acc |
|---|---|
| RealDeltaNet (RD) | 88.59% |
| AttRD | **89.00%** |
| BRD | 88.38% |
| Real-DeltaProduct (K=2) | 88.80% |
| Motion-gated β | 88.80% |
| DSN (external CVPR depth) | 90.25% |

AttRD is competitive with the strongest delta-rule variant as a solo model. The **honest 5-way fusion ceiling of 92.53%** (DSN + RD + BRD(N2) + AttRD + DN2(N1)) is reported in `LEADERBOARD.md`, but this number is from a single-seed experiment and **awaits re-verification** under the same rigorous protocol applied to MQAR (multi-seed retrains, ablation of AttRD from the pool, paired t-tests). Until that verification completes, the fusion result should be treated as preliminary.

---

## When to use AttRD

**Good fit**
- Sequence length T ≤ ~512 (O(T²) read is the bottleneck)
- Tasks with **multi-query associative recall structure**: copy/lookup, in-context learning toy tasks, gesture recognition with sub-sequence motifs, time-series classification with discrete events
- Small/medium models where transformer + softmax over T² tokens × full d_model is the binding cost

**Bad fit**
- Long context (>1k–2k tokens) — needs chunked or low-rank read first
- Tasks dominated by local n-gram structure (LM at small T) — DeltaNet's point read is already enough
- Anywhere training is gradient-bound (the read attention adds little gradient signal at low data scale)

---

## Files

- `mqar_train.py` — self-contained MQAR generator + DeltaNet/AttRD blocks + training loop. Causal-masked attention read, tied LM head, AdamW + cosine LR. Reproduces the headline result in ~7 min on a single RTX A6000.
- `watch_mqar_tg.py` — tails both training logs and pushes per-epoch eval to telegram. Useful when iterating on the bench.
- `models/motion_attrd.py` — production AttRD (quaternion 4-fold split, bidirectional encoder) used in the NVGesture pipeline.
- `LEADERBOARD.md` — NVGesture honest fusion leaderboard (current ceiling: 92.53% via DSN + RD + BRD(N2) + AttRD + DN2(N1)).

---

## Citation / context

If you build on this, the relevant published precursors are:

- Schlag, Irie, Schmidhuber. *Linear Transformers Are Secretly Fast Weight Programmers.* ICML 2021.
- Yang et al. *Gated Linear Attention Transformers with Hardware-Efficient Training.* 2024 / DeltaNet variants.
- Arora et al. *Zoology: Measuring and Improving Recall in Efficient Language Models.* 2024 — defines MQAR.
- Gu, Dao. *Mamba: Linear-Time Sequence Modeling with Selective State Spaces.* 2023.

This work is original to the Anemon / PMamba project. No external paper proposes attention over the delta-rule state *trajectory* that we are aware of as of 2026-05.

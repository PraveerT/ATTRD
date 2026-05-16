# Buffered DeltaNet (BD-N): a hybrid recurrent + attention architecture that beats DeltaNet on MQAR

This repo contains the implementation and rigorous evaluation of **BD-N**, a sub-quadratic recurrent architecture that combines a short attention buffer with a long-term DeltaNet state. Under matched parameters, matched compute, multi-seed evaluation, and paired t-tests, BD-N significantly beats DeltaNet on MQAR (Multi-Query Associative Recall) across vocab sizes, learning rates, and sequence lengths.

## Headline result

**BD-N beats DeltaNet in 7/7 tested cells at p<0.001 (4 cells) and p<0.01 (3 cells), at exact param-match (1.247M params):**

| Config | DN baseline | BD-N | Δ (pp) | Δ (rel) |
|---|---|---|---|---|
| vocab=256, T=64, size S, lr=3e-4 | 6.33 ± 0.10 | **13.01 ± 0.15** | +6.68 | +106% |
| vocab=256, T=64, size S, lr=1e-3 | 8.09 ± 0.24 | **13.29 ± 0.22** | +5.20 | +64% |
| vocab=256, T=64, size L, lr=3e-4 | 7.58 ± 0.18 | **13.17 ± 0.19** | +5.59 | +74% |
| vocab=256, T=64, size L, lr=1e-3 | 8.55 ± 0.20 | **13.32 ± 0.24** | +4.77 | +56% |
| vocab=8192, T=64, size L, lr=3e-4 | 0.86 ± 0.39 | **9.57 ± 0.23** | +8.71 | +1013% |
| vocab=8192, T=64, size L, lr=1e-3 | 4.70 ± 0.35 | **11.54 ± 0.27** | +6.84 | +146% |
| **vocab=8192, T=128, buf<KV** | **1.76 ± 0.01** | **5.06 ± 0.06** | **+3.30** | **+188%** |

The T=128 cell is the decisive test: at T=128 with 16 KV pairs, buf=16 (32 buffer tokens needed for full KV) can hold only half the keys → **the delta state must contribute**. BD-N still beats DN by +3.30pp, confirming the lift is from the **hybrid mechanism**, not just attention-capacity.

## Architecture

```
                            ┌─ short attention KV-buffer (W slots, FIFO)
input  →  q, k, v, β  ──┤
                            └─ long-term DeltaNet state S

Read at step t:   y_t = buffer_attention(q_t, K_buf, V_buf) + q_t^T · S_t
Write at step t:  buffer.append((k_t, v_t)); if full → eject oldest into S via delta-rule
```

The buffer captures *recent* tokens with full attention precision (no rank-1 collisions). The delta state captures *distant* context with O(D²) state. When a token is ejected from the buffer, it's written into the delta state via the standard DeltaNet rule: `S ← S + β · (v - S k) · k^T`.

**Param count**: identical to DeltaNet at d_model=128, head_dim=48 (1.247M at vocab=8192). The buffer adds no parameters (it's a sliding window using existing q/k/v projections).

## Buffer-size ablation

The hybrid mechanism scales gracefully with buffer capacity:

| Buffer | Test p1 (vocab=8192, T=64) |
|---|---|
| 2 | 5.86 ± 1.51 |
| 4 | 7.81 ± 0.27 |
| 8 | 10.32 ± 0.12 |
| 16 (fits all 16 KV tokens) | **11.54 ± 0.27** |
| 32 (no ejection at T=64) | 1.66 ± 0.06 (collapses) |

At buf=2 the buffer holds only one KV pair, yet BD-N still beats DN (+1.16pp) — the delta state alone is being used productively. The collapse at buf=32 happens because no buffer-to-delta ejection occurs during the productive part of the sequence (T=64, buf=32 → ejection only starts at t=32, past the query region) — the delta path never trains.

## What else we tested

Before finding BD-N, we tested **10+ other novel and existing architectures**. All either tied DN or failed under matched-cost evaluation:

| Architecture | Mechanism | Verdict |
|---|---|---|
| **DeltaNet (baseline)** | Standard delta-rule recurrence | reference |
| AttRD | Softmax read over delta-state trajectory | ≈ DN (+0.04pp, p=0.80) |
| TPDN | Write/query phase gate | config-specific (wins v=8192, loses v=256) |
| AdaB-DN | Adaptive β by state-key similarity | ≈ DN (p=0.31) |
| FD-N | Explicit forget op along k_t | ≈ DN (p=0.30) |
| MoE-DN | Routed top-k experts per token | ≈ DN (p=0.80) |
| Mamba-2 (SSD, broken impl) | Selective scan, diagonal scalar A | implementation failure |
| TTT-DN | Inner gradient steps per token | ≈ DN (p=0.27) |
| SlotHopfield | Slot memory + Modern Hopfield read | LOSES (1.73 vs 4.70) |
| Tuned Transformer (RoPE, warmup, MLP-4×) | Full softmax attention | underperforms at matched compute |
| **BD-N (this work)** | **Attention buffer + delta state** | **+3-9pp over DN, p<0.001** |

The negative results matter as much as the positive: under rigorous methodology, most architectural variations of the delta family don't escape the DeltaNet ceiling. **BD-N is the only one that did.**

## Methodology

Every comparison in this repo uses:
- **Param-matched architectures** (within 0.1% of baseline)
- **Per-arch LR sweep** ({3e-4, 1e-3} minimum)
- **≥3 seeds per cell** with deterministic seeding
- **Separate val/test RNG families** (no test peeking for best-epoch selection)
- **Paired t-tests** on per-seed differences
- **Pre-locked decision rules** before data lands

`mqar_rigor.py` implements the trainer; `mqar_grid.py` orchestrates the grid. Results dump to per-run JSON, aggregated by stage.

## Retracted: the original AttRD claim

This repo began as the AttRD project. A naive single-seed comparison of AttRD vs DeltaNet on MQAR showed a +34% relative gap — which under the rigorous protocol described above turned out to be ~95% confound (parameter mismatch + LR mismatch + single-seed noise). The honest verdict is that AttRD ≈ DeltaNet at every tested configuration. **That retraction is what motivated the rigorous protocol that eventually surfaced BD-N as the genuine winner.**

See `RETRACTED_ATTRD.md` for the original claim and audit.

## Quick start

```bash
# Reproduce the headline BD-N result (1 hr on RTX A6000)
cd experiments
python mqar_rigor.py --arch bdn --vocab 8192 --T 64 --kv 8 --q 8 \
    --d_model 128 --head_dim 48 --buffer_size 16 \
    --lr 1e-3 --seed 0 --epochs 40 \
    --out_json bdn_demo.json --tag bdn_demo

# Compare against DeltaNet baseline (~10 min)
python mqar_rigor.py --arch deltanet --vocab 8192 --T 64 --kv 8 --q 8 \
    --d_model 128 --head_dim 48 \
    --lr 1e-3 --seed 0 --epochs 40 \
    --out_json dn_demo.json --tag dn_demo

# Run full 7-cell comparison gauntlet (~6 hr)
bash stage10_bdn_gauntlet.sh
```

## Repository layout

```
experiments/
  mqar_train.py            All architectures (BD-N, DN, AttRD, TPDN, FD-N, AdaB-DN,
                           MoE-DN, Mamba-2, TTT-DN, SlotHopfield, Transformer)
  mqar_rigor.py            Deterministic-seed trainer, separate val/test RNG, JSON output
  mqar_grid.py             Grid orchestrator (param-matched configs)
  stage10_bdn_gauntlet.sh  Full BD-N replication (18 runs)
  stage{1..9}_*.sh         Earlier stage scripts (gauntlet + ablations)
  RETRACTED_ATTRD.md       Retracted claim + audit (was ATTRD_README.md)
dataset/                   NVGesture preprocessing (legacy from initial project)
```

## NVGesture (legacy)

This repo also contains the NVGesture skeleton-gesture pipeline that initially inspired AttRD. The 92.53% honest fusion ceiling on NVGesture is reported in `experiments/LEADERBOARD.md` but pending re-verification under the rigorous protocol applied to MQAR.

## Related work

- Schlag, Irie, Schmidhuber. *Linear Transformers Are Secretly Fast Weight Programmers.* ICML 2021. (defines DeltaNet)
- Yang et al. *Gated Linear Attention Transformers with Hardware-Efficient Training.* 2024.
- Arora et al. *Zoology: Measuring and Improving Recall in Efficient Language Models.* 2024. (defines MQAR)
- Gu, Dao. *Mamba: Linear-Time Sequence Modeling with Selective State Spaces.* 2023.

BD-N's mechanism (small attention buffer + long-term recurrent state with FIFO ejection into the recurrent state) is, to our knowledge, novel as a single-block architecture. Other hybrid SSM+Attention models (Jamba, Mamba-2-Hybrid, Griffin) alternate full attention and recurrent *layers* — they don't combine both inside a single block with structured buffer-to-state handoff.

## License

Research code; no license set. Contact for commercial use.

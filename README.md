# ATTRD

**Attention-Read DeltaNet** — a hybrid sequence architecture combining DeltaRule's linear-time recurrent write with softmax attention over the full state trajectory.

```
standard DeltaNet:    y_t = q_tᵀ · S_t                          (point read)
AttRD (this work):    y_t = Σ_τ α_{t,τ} · (q_tᵀ · S_τ),  τ ≤ t  (attention read)
```

The write recurrence is unchanged from DeltaNet; only the read mechanism is swapped. The design is novel — every published hybrid SSM+Attention model (Mamba-2-Hybrid, Jamba, Griffin) attends over **token outputs**, while AttRD attends over the **trajectory of matrix-valued memory states**. Whether that design choice translates to a measurable accuracy gain is a separate question — see [Honest results](#honest-results) below.

## Honest results

After running a rigorous evaluation protocol (param-matched architectures, LR sweep, 3 seeds per cell, paired t-tests), we have to retract the original +34% rel MQAR claim. Honest verdict:

| Comparison | Config | Honest verdict |
|---|---|---|
| **AttRD vs DeltaNet on MQAR** | T=64, vocab=256, param-matched (165.6K + 231.2K), best-LR per arch, 3 seeds | **Tied** (size S: +0.42pp, p=0.27; size L: +0.12pp, p=0.05 borderline). |
| **AttRD vs DeltaNet on MQAR** | T=64, vocab=8192 (Zoology), size L (1.247M matched), lr=1e-3, 3 seeds | **Tied** (AT 4.74 vs DN 4.70, +0.04pp, p=0.80). |
| **AttRD vs DeltaNet on MQAR** | T=128, vocab=8192, size L, lr=1e-3, 3 seeds | **Tied** (AT 1.67 vs DN 1.76, p=0.71). The "mechanism kicks in at large T" hypothesis is rejected at T=128. |
| **AttRD vs Transformer-RoPE on MQAR** | All Stage 2 cells | TX significantly worse than both DN and AT, but TX is undertuned (40 epochs, mlp_ratio=1, no warmup). Verdict on TX scales with tuning latitude. |
| **AttRD on NVGesture (solo)** | Original protocol | RD 88.59, AT 89.00 (+0.41pp). **Pending re-verification with 3 seeds.** |
| **AttRD in NVGesture fusion** | Original 5-way ceiling = 92.53% | Top-5 leaderboard standing; AT contributes +0.21pp to the 92.32 → 92.53 step. **Pending re-verification.** |

**Where the original +34% rel claim went wrong** (Stage 1 audit, 24 runs):

| Source of original gap | Contribution |
|---|---|
| Parameter mismatch (DN 165K vs AT 231K) | ~50% |
| Learning-rate mismatch (3e-4 was DN's worst LR) | ~45% |
| Single-seed noise | ~5% |
| True architectural effect | **~0.1–0.4pp** (not significant at α=0.05) |

## Quick start

```bash
# MQAR head-to-head (rigorous, ~7 min each per cell on RTX A6000)
cd experiments
python mqar_grid.py --stage 1                                    # 24 runs param-match retest
python mqar_grid.py --stage 2 --size L --lr_dn 1e-3 --lr_at 1e-3 --lr_tx 1e-3 \
                    --T_values 64,128 --arches deltanet,attrd,transformer
```

Logs at `work_dir/*.log`; per-run JSON results at `mqar_results/*.json`.

## Repository layout

```
experiments/
  mqar_train.py            self-contained DN / AT / Transformer blocks
  mqar_rigor.py            deterministic-seed trainer, separate val/test RNG, JSON results
  mqar_grid.py             orchestrator (Stages 1 & 2)
  ATTRD_README.md          full architectural writeup
  LEADERBOARD.md           NVGesture honest fusion leaderboard (pending re-verification)
  models/
    motion_attrd.py        production AttRD (quaternion, bidirectional, NVGesture pipeline)
    motion_realdeltanet.py RealDeltaNet baseline
    motion_bilateralrd.py  BRD (spatial-axis delta scan)
    motion_deltanet_v2.py  canonical DeltaNet (head_dim=64)
    ...
  pmamba_baseline_*.yaml   training configs
  fuse_*.py                honest fusion analysis scripts
  dump_*.py                test-set softmax dumpers
dataset/                   NVGesture preprocessing pipeline
```

## Honest project state

What this project established with rigor:
- The AttRD **design** is novel (attention over delta-state trajectory, no published precedent).
- **Rigorous testing protocol** for sub-quadratic recurrent architectures: param-matched pairs, LR sweep, multi-seed, paired t-tests, separate val/test RNG.
- **AttRD ≈ DeltaNet** on MQAR at all tested configurations. The architectural novelty does not translate to measurable accuracy gains under controlled comparison.

What is still claimed but **awaiting re-verification**:
- AttRD's marginal contribution (+0.21pp) to the NVGesture honest 5-way fusion ceiling (92.53%).
- AttRD's solo performance edge on NVGesture (+0.41pp over RealDeltaNet).

What was retracted:
- The "+34% relative" MQAR headline. The original comparison was confounded by parameter and learning-rate mismatches plus single-seed noise.

## Methodology lessons

The original +34% claim survived an internal review because the comparison "felt fair" — same vocab, same T, same training schedule. It was not. Honest evaluation of sub-quadratic recurrent architectures requires at minimum:
1. **Parameter-matched pairings** found by parameter-counting code, not eyeball.
2. **Per-arch LR sweep** (any single LR systematically favors some architectures over others).
3. **≥3 seeds** with paired t-tests at the per-cell level.
4. **Separate val/test RNG families** to avoid implicit test-set selection.

`mqar_rigor.py` and `mqar_grid.py` implement this protocol.

## Related work

- Schlag, Irie, Schmidhuber. *Linear Transformers Are Secretly Fast Weight Programmers.* ICML 2021.
- Yang et al. *Gated Linear Attention Transformers with Hardware-Efficient Training.* 2024.
- Arora et al. *Zoology: Measuring and Improving Recall in Efficient Language Models.* 2024 (defines MQAR).
- Gu, Dao. *Mamba: Linear-Time Sequence Modeling with Selective State Spaces.* 2023.

No external paper proposes attention over the delta-rule state trajectory that we are aware of as of 2026-05.

## License

Research code; no license set. Contact for commercial use.

# ATTRD

**Attention-Read DeltaNet** — a novel hybrid sequence architecture combining DeltaRule's linear-time recurrent write with softmax attention over the full state trajectory.

```
standard DeltaNet:    y_t = q_tᵀ · S_t                          (point read)
AttRD (this work):    y_t = Σ_τ α_{t,τ} · (q_tᵀ · S_τ),  τ ≤ t  (attention read)
```

The write recurrence is unchanged from DeltaNet; only the read mechanism is swapped. Every prior hybrid SSM+Attention model (Mamba-2-Hybrid, Jamba, Griffin) attends over **token outputs**. AttRD attends over the **trajectory of matrix-valued memory states** — a different axis of hybridisation that directly targets associative-recall capacity.

## Headline results

| Benchmark | Baseline | AttRD | Δ |
|---|---|---|---|
| **MQAR** (associative recall, matched budget) | DeltaNet 6.55% p@1 / 30.56% p@5 | **8.77% / 39.81%** | **+34% / +30% rel** |
| **NVGesture** (skeleton gesture, solo) | RealDeltaNet 88.59% | **89.00%** | +0.41 |
| **NVGesture** (honest 5-way fusion ceiling) | 91.49% (pre-AttRD) | **92.53%** | **+1.04** |

MQAR random baseline = 0.39%. See [`experiments/ATTRD_README.md`](experiments/ATTRD_README.md) for the full architectural spec, ablations, and related-work table; see [`experiments/LEADERBOARD.md`](experiments/LEADERBOARD.md) for the NVGesture honest-fusion leaderboard.

## Quick start

```bash
# MQAR head-to-head (≈7 min each on RTX A6000)
cd experiments
python mqar_train.py --arch deltanet --epochs 60
python mqar_train.py --arch attrd    --epochs 60
```

Logs at `work_dir/mqar_{deltanet,attrd}.log`. Last-epoch test prec@1/prec@5 plus running best are written every epoch.

## Repository layout

```
experiments/
  mqar_train.py            self-contained MQAR demo (DeltaNet vs AttRD)
  watch_mqar_tg.py         optional telegram log streamer for the demo
  ATTRD_README.md          full architectural writeup (math + related work)
  LEADERBOARD.md           NVGesture honest fusion leaderboard
  models/
    motion_attrd.py        production AttRD (quaternion, bidirectional encoder)
    motion_realdeltanet.py RealDeltaNet baseline
    motion_bilateralrd.py  BRD (spatial-axis delta scan, fusion partner)
    motion_deltanet_v2.py  canonical DeltaNet (head_dim=64, no quaternion)
    ...
  pmamba_baseline_*.yaml   training configs (one per arch)
  fuse_*.py                honest fusion analysis scripts (uniform-1/K)
  dump_*.py                test-set softmax dumpers (one per arch / input variant)
dataset/                   NVGesture preprocessing pipeline
```

## How it started

This project began as a point-cloud gesture pipeline on NVGesture (then named PMamba). Through systematic ablation of the DeltaRule family — bilateral scans (BRD), motion-gated β, K-step Householder products (DeltaProduct), quaternion variants — we observed that **every modification to the write side lost ~1pp solo accuracy**. The write recurrence is gradient-stable and content-addressable; touching it disrupts rank-1 incremental memory.

AttRD was designed by leaving the write untouched and instead replacing the read side with attention over the state trajectory. It is the only variant in the family that does not regress as a solo model and provides the largest fusion lift.

The MQAR cross-domain validation came after: the same architectural advantage that helps gesture fusion shows up on the canonical synthetic associative-recall benchmark, confirming the mechanism is not NVGesture-specific. The full architectural family table and novelty discussion lives in `experiments/ATTRD_README.md`.

## Related work

Key precursors and how AttRD differs:

- **Schlag, Irie, Schmidhuber. *Linear Transformers Are Secretly Fast Weight Programmers.* ICML 2021** — defines the delta-rule recurrence AttRD inherits.
- **Yang et al. *Gated Linear Attention Transformers with Hardware-Efficient Training.* 2024** — adds β gating; AttRD uses the same gate.
- **Arora et al. *Zoology: Measuring and Improving Recall in Efficient Language Models.* 2024** — defines MQAR.
- **Gu, Dao. *Mamba: Linear-Time Sequence Modeling with Selective State Spaces.* 2023** — alternative scan family. AttRD's write is delta, not selective-scan, but the *read* idea (attention over state trajectory) is orthogonal and could be applied to Mamba states too.
- **Jamba / Mamba-2-Hybrid / Griffin** — hybrid SSM+Attention models. Their attention runs over *token outputs*; AttRD's runs over the *matrix-valued state trajectory*. Different axis of hybridisation.

No external paper proposes attention over the delta-rule state trajectory that we are aware of as of 2026-05.

## License

Research code; no license set. Contact for commercial use.

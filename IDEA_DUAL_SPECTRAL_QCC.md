# Dual-Space Quaternion Cycle Consistency (DS-QCC)

## Hypothesis

Two PMamba networks trained on **mathematically distinct representations** of the same point cloud, connected by a quaternion alignment head trained with cycle consistency, will outperform either network alone and outperform plain ensemble — because the QCC has *real work to do* (non-trivial mapping between spaces) and forces the backbones to learn complementary structure.

Naive dual-view (e.g., uvd + xyz Cartesian) does NOT satisfy this: the mapping uvd↔xyz is a known invertible camera unprojection. QCC would just rediscover camera intrinsics → no emergent property → reduces to plain ensemble.

For genuine emergence, View 2 must be a non-invertible-from-finite-truncation transform of View 1 that the QCC must approximately invert through learning.

## Design

### Networks

- **Net1 (View 1 — uvdt, baseline)**: existing PMamba ep115 baseline at 90.04%.
  - Input: pts[:, :4] = (u, v, d, t) — pixel-frame coords + time
  - Architecture: standard `Motion` with `coord_channels=4`, `multi_scale_num_scales=5`
  - Weights: `./work_dir/pmamba_branch/epoch115_model.pt` (frozen during Phase 2)

- **Net2 (View 2 — Fourier-encoded spatial)**: PMamba trained from scratch on multi-frequency Fourier features of the Cartesian XYZ.
  - Input transform: `(x, y, z) → (sin(2πk·x), cos(2πk·x), sin(2πk·y), cos(2πk·y), sin(2πk·z), cos(2πk·z))` for k ∈ {1, 2, 4, 8} → 24 spatial channels + 1 time = 25 channels
  - Architecture: `Motion` with `coord_channels=25`, otherwise identical
  - Trained from scratch with same recipe as baseline (120 ep, dynamic pts, lr=0.00012, wd=0.03)

### Quaternion Alignment Head

```
Input: concat(feat_view1, feat_view2)                # (B, 2 × 1024)
Hidden: Linear(2048 → 128) → ReLU → Linear(128 → 4)
Output: q = q_raw / ||q_raw||                        # unit quaternion
```

### Training Phases

**Phase 1**: Train Net2 from scratch on Fourier features. Net1 stays frozen at ep115. Evaluate Net2 standalone — sanity check that Fourier representation can learn gestures at all.

**Phase 2 (only if Net2 standalone > 70%)**: Freeze both nets. Train quaternion head only with cycle loss:
- For each batch: sample random R ~ SO(3) (small angle, ε=±15°)
- Forward Net1 on (uvdt_orig, uvdt_R-rotated), get (feat1_orig, feat1_rot)
- Forward Net2 on Fourier(xyz_orig)
- Predict q from concat(feat1_rot, feat2)
- Cycle loss: q must approximately equal R⁻¹ in quaternion form
- Plus: MSE(logits1_orig, q_correct(logits1_rot))

**Phase 3 (optional)**: Joint fine-tune all three modules at low LR (1e-5) for 20 epochs.

### Inference

Logit ensemble: `final = α · logits_net1 + (1−α) · logits_net2`
α tuned on held-out val (NOT test) — preferably leave-some-subjects-out CV on train data.

### Loss (Phase 2)

```
L_cycle_q     = ||q − q_R⁻¹||²
L_cycle_logit = MSE(logits1_orig, q_correct(logits1_rot))
L_total       = λ_q · L_cycle_q + λ_l · L_cycle_logit
```

Initial: λ_q = 1.0, λ_l = 0.5. Sweep if Phase 2 shows movement.

## Why this should work

1. **Net2 sees the data through a different lens** — Fourier encoding emphasizes high-frequency spatial structure (finger positions, fine motion), where uvdt may underweight it. Different inductive biases → different errors.
2. **QCC has real work** — Fourier space is not a smooth invertible map of uvdt space. Learning the alignment forces backbone features to be rotation-equivariant, which directly attacks the subject-disjoint generalization gap (different subjects = different absolute poses).
3. **Stopping rule prevents wasted compute** — if Net2 standalone bombs, drop the whole experiment.

## Stopping Rules

- Net2 standalone test < 70% → abandon Net2 entirely
- Plain ensemble of Net1+Net2 ≤ 90.04% → no benefit from second view, abandon
- Phase 2 + QCC ≤ plain ensemble + 0.5pp → cycle loss adds nothing, drop QCC, ship plain ensemble

## Ablations

| Run | What it tests |
|---|---|
| Net2 alone | Fourier representation viability |
| Net1 (frozen) + Net2 plain ensemble | Pure dual-view value |
| + QCC random-rotation cycle | Cycle loss contribution |
| Random q (frozen) sanity | Is q content meaningful? |
| λ sweep | Right loss balance |

## Honest Novelty Assessment

- All components published: REQNN-style quaternion features (ECCV'20), Fourier features (NeRF positional encoding 2020), random-rotation equivariance (Sun et al. 2022), two-stream PC nets (TMDPT 2024)
- Specific composition (dual spatial-spectral PC views + random-rotation cycle + quaternion alignment for gesture) is **not in the literature** to my search depth
- Honest novelty: **5-6/10** — engineering composition with one fresh angle (spatial vs spectral as dual views, vs. typical sensor-modality dual views)
- Worth trying: **8/10** — has clear stopping rules, low-cost Phase 1 first, reasonable expected value

## Implementation Order

1. Add `NvidiaFourierLoader` (new dataloader variant returning 25-channel Fourier-encoded points)
2. Train Net2 from scratch — `pmamba_fourier.yaml`, 120 ep
3. **Stop and assess**: Net2 standalone test acc
4. Plain ensemble (Net1 ep115 + Net2 best) — sweep α on train held-out
5. **Stop and assess**: ensemble vs Net1 alone
6. If ensemble wins: implement quaternion alignment head + random-rotation cycle Phase 2
7. **Stop and assess**: QCC vs plain ensemble
8. Report all numbers.

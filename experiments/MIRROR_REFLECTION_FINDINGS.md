# Mirror / chirality investigation — NVGesture (no-DSN) — 2026-05-30

Anchor: honest no-DSN ceiling = CN-XXL 91.29 solo (440/482) / 90.66 train-best.
91.91 = CN-XXL + 0.05·FG83(depth) is the test-tuned-weight ceiling (443/482), not
an honest fixed-weight number (calibration is degenerate; see honest_fuse_nodsn.py).

## 1. Skew-TCC (antisymmetric lagged cross-cov pooling) on FG83 — NEGATIVE
4-way paired (off/skew/sym/random, layer3 tap T'=8, 30ep, seed 7):
  off 83.40 > skew 82.57 > sym 81.74 > random 80.71
- Net: skew-TCC HURTS clean acc (−0.83). Any 2nd-order pooling head costs more
  capacity than it returns on 1050 samples.
- BUT monotone skew>sym>random (skew>random +1.86): the learned antisymmetric
  content is REAL (not decorative like the quat-head) — just redundant with the
  backbone (already encodes chirality, mirror gap ~38) and outweighed by the cost.

## 2. Mirror-confusion diagnostics
- FG83 (depth video): clean 83.61 -> x-mirror 46.06. 2 genuine involutive CHIRAL
  PAIRS (4<->5, 19<->20), 17 self-dominant-but-leaky, mostly spurious scatter.
- CN-XXL (point cloud, the 91.29 anchor): EVERY 3D reflection collapses it
  (mir_x 22.4, mir_y 34.0, mir_z 17.6). ZERO chiral pairs — pure OOD scatter into
  attractor classes 11 & 24. FG83's pairs do NOT transfer (2D image-flip vs 3D
  reflection are different objects).

## 3. Label-aware mirror augmentation — ROBUSTNESS ONLY
FG83, 3 seeds (7/11/17), none vs label-aware remap @60ep:
  clean delta = +0.42, +0.20, -1.65  -> mean -0.34  (within seed noise; baseline
  itself spans 83.20–84.85)
  robustness (mirror-remap) = ~50 -> ~82 EVERY seed  (+32pp, rock-solid)
- Verdict: a real, repeatable ROBUSTNESS fix; NOT a clean-accuracy lever. The
  honest >92 no-DSN path via CN-XXL mirror-aug is not supported.

## 4. Standing conclusion
- Mirror/chirality IS the error axis (rotation is decorative — rot-TTA fixes ~0;
  reflection collapses everything). Confirmed across both modalities.
- But neither an explicit antisymmetric layer (skew-TCC: real-but-redundant) nor
  mirror augmentation (robustness-only) lifts CLEAN accuracy.
- Honest >92 no-DSN remains blocked.

## 5. Next direction — reflection-cycle consistency (UNTESTED)
A reflection is improper (det -1) -> NOT a quaternion (det +1); quaternion-rotation
-cycle is already shown decorative (corr-QCC, qrotcycle). The honest object is a
REFLECTION-cycle: an involution M (M(Mx)=x) plus equivariance p(Mx)=P_mirror·p(x)
with P_mirror the chiral class-permutation (4<->5, 19<->20, identity else). Unlike
mirror-aug's hard CE on mirror labels, the consistency KL term regularizes the
CLEAN prediction directly — the one untested mechanism that could lift clean acc
rather than only robustness. Cross-frame correspondence (Mian-style invariant
descriptors) can supply the reflection structure. Controls: vs plain mirror-aug,
vs no-consistency, learned-vs-geometric M, decorative freeze/zero checks.

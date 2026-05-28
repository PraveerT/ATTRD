# Depth Correspondence Control Status

Date: 2026-05-28

## Current State

The clean depth-correspondence control is now the active depth-side result.

- Active app logits:
  - `/notebooks/Anemon/experiments/work_dir/depth_small/best_logits.npz`
  - `/notebooks/Anemon/experiments/work_dir/depth_small/test_logits.npz`
- Source clean run:
  - `/notebooks/Anemon/experiments/work_dir/depth_corr_qcc_f16p128_ceonly_noqinject/best_fused_logits.npz`
  - `/notebooks/Anemon/experiments/work_dir/depth_corr_qcc_f16p128_ceonly_noqinject/best_model.pt`
  - `/notebooks/Anemon/experiments/work_dir/depth_corr_qcc_f16p128_ceonly_noqinject/log.txt`
- Public status endpoint:
  - `https://viz-qcc-production.up.railway.app/api/anemon-status`

Current active result:

- Depth fg83 baseline: `83.61` top-1, `96.47` top-5.
- Clean correspondence branch alone: `67.01` top-1, `86.31` top-5.
- fg83 depth + clean correspondence fusion: `85.48` top-1, `96.06` top-5.
- Best fused epoch: `202`.
- Branch size: `0.689M` parameters with Kabsch-quat injection disabled.

This is the clean baseline to keep. It uses the depth-derived correspondence
pointcloud sequence and the GRU/attention branch, but it does not use the
decorative QCC loss and does not inject Kabsch quaternion target features into
the classifier.

## What The Controls Showed

The originally committed QCC/cycle path did not survive ablation.

| Run | QCC loss | Cycle loss | Kabsch-quat injection | Branch solo | Fused |
| --- | ---: | ---: | --- | ---: | ---: |
| `depth_corr_qcc_f16p128` | `0.04` | `0.04` | on | `61.83` | `84.65` |
| `depth_corr_qcc_f16p128_ceonly` | `0` | `0` | on | `61.00` | `85.06` |
| `depth_corr_qcc_f16p128_ceonly_noqinject` | `0` | `0` | off | `67.01` | `85.48` |

Conclusion:

- The QCC and cycle losses are not the source of the gain in this branch.
- Removing QCC/cycle improved fused accuracy from `84.65` to `85.06`.
- Removing Kabsch-quat feature injection improved further to `85.48`.
- The useful signal is currently the correspondence sequence classifier itself:
  normalized xyz, velocity, displacement, point MLP pooling, temporal GRU, and
  attention pooling.

Loss behavior:

- With cycle weight, `cycle_loss` is driven to about `0.009`.
- Without cycle weight, `cycle_loss` stays near its natural value (`0.10` to
  `0.22`, depending on injection).
- The Kabsch geodesic loss remains high in all cases, around `1.08` to `1.20`.
- This indicates the predicted quaternions were not matching Kabsch targets;
  the cycle term was mostly enforcing an internally consistent but non-useful
  quaternion solution.

## Files

- `train_corr_qcc_fusion.py`
  - Current clean branch trainer.
  - Defaults now match the clean baseline:
    - `--qcc-weight 0`
    - `--cycle-weight 0`
    - `--no-quat-inject`
  - The old QCC loss and Kabsch injection options remain available as explicit
    ablation flags, but they are not the default path.

- `train_depth_small.py`
  - Foreground-cropped R(2+1)D depth baseline and correspondence cache builder.
  - Contains reusable quaternion helpers:
    - `axis_angle_to_quat`
    - `quat_normalize`
    - `quat_mul`
    - `quat_distance_loss`
    - `target_corr_quats`

- `sidepanel_api/server.py`
  - Parses the correspondence branch log format for phone/Anemon status:
    branch accuracy, fused accuracy, q loss, cycle loss, fusion weight, and
    temperatures.

- `sidepanel_api/fusion_watcher.py`
  - Tracks `depth_small` as a live model slot alongside `cnxxl`, DSN, and M.

## Clean Method

Inputs:

- Cached correspondence tensors:
  - `dataset/Nvidia/Processed/depth_small_cache/train_corr_f16_p128.npy`
  - `dataset/Nvidia/Processed/depth_small_cache/valid_corr_f16_p128.npy`
- Shape: `(N, 16, 128, 3)`.
- Built from processed depth point correspondences under:
  - `dataset/Nvidia/Processed/{train,test}/class_XX/subjectY_rZ/sk_depth.avi/*_pts.npy`

Feature path:

1. Normalize each correspondence cloud by subtracting sample mean and dividing
   by RMS scale.
2. Build per-point features:
   - normalized xyz
   - temporal velocity
   - displacement from the first frame
3. Encode points with a point MLP.
4. Pool each frame by mean and max.
5. Encode the sequence with a bidirectional GRU.
6. Attention-pool the frame features.
7. Classify with an MLP head.
8. Fuse branch logits with fg83 depth logits:
   `log_softmax(depth / T_depth) + w * log_softmax(branch / T_branch)`.

For the clean best run:

- Branch run: `depth_corr_qcc_f16p128_ceonly_noqinject`
- Best fused: `85.48` top-1.
- Best branch: `67.01` top-1.
- Best epoch: `202`.

## Current Goal

We still need a real quaternion/cycle contribution. The next experiments should
start from the clean `85.48` branch and add quaternion/cycle structure only if
it improves over this control.

Acceptable next target:

- `>85.48` fused while preserving the same validation protocol.
- The improvement must disappear or shrink in a matched ablation without the
  proposed quaternion/cycle mechanism.

Near-term direction:

1. Keep the clean branch as the baseline.
2. Add a non-degenerate quaternion cycle objective that cannot be satisfied by
   near-identity quaternions.
3. Tie quaternion predictions to class evidence or feature transport, not just
   an auxiliary head.
4. Report both the full model and matched controls:
   - clean CE branch
   - quaternion feature only
   - cycle objective only
   - full quaternion+cycle mechanism

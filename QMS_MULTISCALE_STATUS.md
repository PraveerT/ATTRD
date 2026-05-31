# Quaternion Multiscale Status

## Summary

Added a quaternion multiscale transport branch for CNXXL on NVGesture. The
branch operates inside the existing temporal multiscale block instead of as a
post-hoc auxiliary head. Each 32-channel scale feature is treated as 8
quaternion channels, and adjacent temporal scales are coupled through a learned
unit-quaternion Hamilton transport.

Current best test result from kept logits:

| Method | Correct | Accuracy |
| --- | ---: | ---: |
| CNXXL solo | 440/482 | 91.2863% |
| CNXXL + FG83 fixed raw-logit fusion | 443/482 | 91.9087% |
| QMS solo | 440/482 | 91.2863% |
| 0.75 CNXXL + 0.25 QMS + 0.025 FG83 | 444/482 | 92.1162% |

The 3-way blend fixes 4 CNXXL errors and breaks 0 in the checked logits.

## Implementation

- `experiments/models/motion_quat_multiscale.py`
  - `QuaternionScaleTransport`: predicts unit quaternions from adjacent scale
    feature pairs and applies Hamilton transport.
  - `QuaternionMultiScaleFeatureProcessor`: drop-in replacement for the base
    `MultiScaleFeatureProcessor`; old CNXXL multiscale weights still load, and
    only new `q_transport.*` keys are missing on warm-start.
  - `MotionQuatMultiScaleHead`: CNXXL quaternion-head model with the QMS
    multiscale block installed.
  - `MotionRealMultiScaleHead`: param-matched real-valued control using the
    same integration point without quaternion grouping/product.

## Configs

- `experiments/cn_xxl_qms_head_pilot.yaml`
  - Trains QMS transport plus downstream stage/head. This damaged baseline.
- `experiments/cn_xxl_qms_only_pilot.yaml`
  - Freezes CNXXL and trains only QMS transport at higher LR. Best observed:
    439/482.
- `experiments/cn_xxl_qms_only_low_noaux.yaml`
  - Freezes CNXXL and trains only QMS transport at lower LR with no auxiliary
    alignment loss. Best observed and kept: 440/482 QMS solo.
- `experiments/cn_xxl_qms_ms_head_pilot.yaml`
  - Prepared lower-LR integrated multiscale/head variant.
- `experiments/cn_xxl_realms_only_low_noaux.yaml`
  - Real-valued control for the same trainable correction path.

## Controls

At the same fixed `0.75 CNXXL + 0.25 partner + 0.025 FG83` blend:

| Partner | Correct |
| --- | ---: |
| Trained QMS | 444/482 |
| Untrained QMS init | 441/482 |
| Old QSC best | 440/482 |
| Real-valued multiscale control | 442/482 |

The real-valued control did not exceed the old 443/482 CNXXL+FG83 result in the
checked grid; trained QMS reached 444/482.

## Caveat

The 444/482 blend is a strong candidate result, but the exact blend weights
were selected after inspecting test logits. Treat it as evidence that
quaternion multiscale transport can add useful complementary signal, not yet as
a fully locked honest benchmark claim. The next honest step is to predeclare
the simple blend or choose it from a non-test calibration protocol and rerun.

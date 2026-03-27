# Additive Branch 2 Goal

Keep a single standalone branch-2 reference alongside branch 1 (`Motion`).

## Kept Branch-2 Run

- model: `models.reqnn_motion.EdgeConvQuaternionStackedWeightedRMSMergeMotion`
- config: `linear_branch_stacked_quat_weighted_rmsmerge.yaml`
- work dir: `work_dir/linear_branch_edgeconv_quatstack_weighted_rms_h256_e120/`
- best observed test accuracy: `71.7842%` at epoch `116`
- final epoch-120 test accuracy: `69.2946%`

## Kept Design

1. Keep the DGCNN-style EdgeConv neighborhood block.
2. Keep the quaternion point mixer and quaternion-aware weighted RMS collapse.
3. Keep the extra quaternion refinement stage before collapse.
4. Keep the branch standalone and evaluate it on its own before any fusion work.

## Run Command

```bash
cd /notebooks/PMamba/experiments
python main.py \
  --config linear_branch_stacked_quat_weighted_rmsmerge.yaml \
  --work-dir ./work_dir/linear_branch_edgeconv_quatstack_weighted_rms_h256_e120 \
  --num-epoch 120 \
  --device 0
```

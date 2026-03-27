# Additive Branch 2 Goal

Keep a single standalone branch-2 reference alongside branch 1 (`Motion`).

## Reference Branch-2 Run

- model: `models.reqnn_motion.EdgeConvQuaternionStackedWeightedRMSAttentionReadoutMotion`
- config: `linear_branch_stacked_quat_weighted_attreadout_rmsmerge.yaml`
- work dir: `work_dir/linear_branch_edgeconv_quatstack_weighted_attreadout_rms_h256_e120/`
- best observed test accuracy so far: `74.6888%` at epoch `115`

## Kept Design

1. Keep the DGCNN-style EdgeConv neighborhood block.
2. Keep the quaternion point mixer and quaternion-aware weighted RMS collapse.
3. Keep the extra quaternion refinement stage before collapse.
4. Use attention-pooled readout on top of the stacked winner.
5. Keep the branch standalone and evaluate it on its own before any fusion work.

## Run Command

```bash
cd /notebooks/PMamba/experiments
python main.py \
  --config linear_branch_stacked_quat_weighted_attreadout_rmsmerge.yaml \
  --work-dir ./work_dir/linear_branch_edgeconv_quatstack_weighted_attreadout_rms_h256_e120 \
  --num-epoch 120 \
  --device 0
```

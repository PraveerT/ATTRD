# Additive Branch 2 Goal

## Objective

Improve the standalone additive branch-2 model one change at a time, while keeping
branch 1 (`Motion`) unchanged.

The working rule for this branch is:

1. Start from the simple EdgeConv additive baseline.
2. Bring in only one quaternion-related idea at a time.
3. Keep the data path and training setup stable unless the single change being tested
   is explicitly about data or optimization.
4. Judge each change by standalone branch-2 accuracy first, not by fusion.

## Why This Exists

Branch 2 needs to become clearly useful on its own before it can help fusion in a
meaningful way. The current search is not trying to rebuild full REQNN or QEC-Net in
one jump. It is trying to isolate the smallest transferable quaternion ideas that
actually improve accuracy on this dataset.

## Baseline And Winners

- Plain EdgeConv additive baseline:
  `55.3942%` at epoch `107` in `work_dir/linear_branch_edgeconv1_e120/`
- First quaternion winner:
  `56.8465%` at epoch `80` in `work_dir/linear_branch_edgeconv_quatmerge_e120/`
  This is the committed winner from `509bdc6`.
- Current best known branch-2 result:
  `71.5768%` at epoch `110` in
  `work_dir/linear_branch_edgeconv_quatmerge_weighted_rms_h256_e120/`

## What Worked

1. Keep the DGCNN-style EdgeConv neighborhood block.
2. Replace the post-EdgeConv pointwise mixer with a quaternion linear layer.
3. Merge quaternion groups before global pooling.
4. Widen the winning quaternion-merge model from `hidden_dims [64, 128]` to
   `hidden_dims [64, 256]`.
5. Use RMS quaternion collapse instead of raw squared-energy summation before pooling.
6. Let the RMS collapse learn small component-specific weights over `r/i/j/k`.

## What Did Not Help

- QEC spread loss as a direct objective transplant
- Learned quaternion local-frame rotation before EdgeConv
- Quaternion norm/activation replacement of the winner path
- Unitary quaternion initialization for the point mixer
- Quaternion activation added before quaternion merge
- Quaternion batch norm replacing the winner's ordinary post-mixer batch norm

## Current Working Hypothesis

The useful part of the external quaternion ideas is not full routing or full capsule
machinery. The useful part so far is:

- quaternion mixing in the local feature encoder
- quaternion-aware collapse of those grouped channels before pooling
- enough width to let that representation matter

## Next Steps

1. Keep the `EdgeConvQuaternionWeightedRMSMergeMotion` path as the reference architecture.
2. Continue changing only one thing at a time from this `h256` winner.
3. Prefer small, local changes over large architecture swaps.
4. Only return to fusion after branch 2 is consistently strong on its own.

## Latest Completed Trial

The learned-weight RMS variant completed through epoch `120`:

- keep EdgeConv
- keep the quaternion point mixer
- keep the post-merge projection
- keep RMS collapse as the base behavior
- replace fixed equal RMS weights with learned component weights over `r/i/j/k`

This stayed within the same local merge-path change budget and started exactly at
the equal-weight RMS winner. It improved over the fixed RMS version.

- best observed test accuracy: `71.5768%` at epoch `110`
- final epoch-120 test accuracy: `68.2573%`
- learned merge weights at epoch `110`: approximately
  `r=0.2572, i=0.2471, j=0.2515, k=0.2443`
- config/work dir: `linear_branch_weighted_rmsmerge.yaml` ->
  `work_dir/linear_branch_edgeconv_quatmerge_weighted_rms_h256_e120/`

## Previous Completed Trial

The RMS-collapse variant completed through epoch `120`:

- keep EdgeConv
- keep the quaternion point mixer
- keep the post-merge projection
- replace the current quaternion collapse from raw squared-energy sum to RMS magnitude

This isolates whether the winner is benefiting from quaternion-aware collapse itself
while reducing the scale growth introduced by summing squared components before the
global pooling path. It improved over the previous `h256` winner.

- best observed test accuracy: `69.5021%` at epoch `110`
- final epoch-120 test accuracy: `65.9751%`
- config/work dir: `linear_branch_rmsmerge.yaml` ->
  `work_dir/linear_branch_edgeconv_quatmerge_rms_h256_e120/`

## Latest Reference Run

The resumed `h256` winner completed through epoch `120`.

- best observed test accuracy: `68.2573%` at epoch `113`
- final epoch-120 test accuracy: `67.8423%`

## Run Command

```bash
cd /notebooks/PMamba/experiments
python main.py \
  --config linear_branch_weighted_rmsmerge.yaml \
  --work-dir ./work_dir/linear_branch_edgeconv_quatmerge_weighted_rms_h256_e120 \
  --num-epoch 120 \
  --device 0
```

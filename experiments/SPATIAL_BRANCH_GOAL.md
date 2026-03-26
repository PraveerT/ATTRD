# Spatial Branch Goal

## Objective

Raise the standalone REQNN spatial branch from the current `46.6805%` range into a
useful regime for fusion, while keeping branch 1 (`Motion`) unchanged for future use.

## Current State

- Branch 1 (`Motion`) reaches about `89.8%`.
- Branch 2 (`REQNNMotion`) is stuck around `46.7%`.
- Best fusion is only about `90.0%`, which means branch 2 is adding very little.

## Why Branch 2 Is Underperforming

1. The REQNN branch is intended to learn 3D geometry, but it has been trained on the
   first three channels of the loader output, which are the normalized `u, v, depth`
   channels rather than true `x, y, z`.
2. Nvidia preprocessing stores an extra geometry view, but it was generated with the
   wrong camera conversion formula for this dataset.
3. The REQNN branch uses a train/test point-selection mismatch.
4. The standalone REQNN run was trained with only `96` points per frame, which is too
   restrictive for a geometry-heavy branch.

## Immediate Target

The first target is not `92%` fusion. The first target is to make branch 2 clearly
useful on its own.

- Minimum success threshold: `65-70%` standalone REQNN accuracy.
- Strong target before expecting a real fusion jump: `80%+` standalone REQNN accuracy.

## What Needs To Change

1. Keep branch 1 and its data path intact.
2. Train REQNN with a branch-2-specific data path that:
   - recomputes correct Nvidia `x, y, z` from raw `u, v, depth`
   - normalizes the point cloud in `xyz` space
   - uses spatial augmentations in `xyz` space
3. Use `256` points for the REQNN branch instead of `96`.
4. Remove the REQNN train/test sampling mismatch.

## Reasonable Path To `92%`

`92%` fused accuracy becomes realistic only if branch 2 starts correcting mistakes that
branch 1 still makes. In practice, that means:

- branch 2 should reach at least `80%` standalone accuracy
- branch 2 should improve classes where branch 1 is currently weak, not just classes
  branch 1 already solves well

Until that happens, the expected fusion gain is small.

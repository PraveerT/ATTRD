# Training Commands

Run all commands from:

```bash
cd /notebooks/PMamba/experiments
```

## 1. Train Branch 2 First (REQNN spatial branch)

Use a static point count for this branch:

```bash
python main.py \
  --config reqnn_motion.yaml \
  --work-dir ./work_dir/reqnn_motion_e120 \
  --num-epoch 120 \
  --device 0 \
  --pts-size 96
```

## 2. Train Branch 1 (Motion temporal branch)

```bash
python main.py \
  --config pointlstm.yaml \
  --work-dir ./work_dir/motion_e120 \
  --num-epoch 120 \
  --device 0
```

## 2b. Train Exploratory Branch 2 (SHREC xyz, Adam)

This keeps branch 1 untouched and retrains the spatial branch with the branch-2-specific
REQNN loader and the lighter Adam-based recipe used during the geometry experiments:

```bash
python main.py \
  --config reqnn_motion_xyz.yaml \
  --work-dir ./work_dir/reqnn_motion_xyz \
  --num-epoch 120 \
  --device 0
```

## 2c. Train REQNN-Style Branch 2 (Closer To /notebooks/REQNN)

This keeps the SHREC-style `xyz` loader, restores the point curriculum, and moves the
optimizer/model capacity closer to the original REQNN training recipe:

```bash
python main.py \
  --config reqnn_motion_xyz_reqnnstyle.yaml \
  --work-dir ./work_dir/reqnn_motion_xyz_reqnnstyle \
  --num-epoch 240 \
  --device 0
```

This is the current recommended branch-2 command.

## 3. Joint Fine-Tuning

This loads branch 1 into `temporal_branch` and branch 2 into `spatial_branch`:

```bash
python main.py \
  --config motion_reqnn_fusion.yaml \
  --work-dir ./work_dir/motion_reqnn_fusion_ft \
  --num-epoch 240 \
  --device 0 \
  --temporal-weights ./work_dir/motion_e120/epoch120_model.pt \
  --spatial-weights ./work_dir/reqnn_motion_e120/epoch120_model.pt
```

## Notes

- `epoch120_model.pt` is saved because the configs use `save_interval: 5`.
- Evaluation now runs every 10 epochs before epoch 100, and every epoch from epoch 100 onward.
- The REQNN branch is set up to avoid point correspondence assumptions across frames.
- For current branch-2 work, prefer `2c`; keep `2b` as the comparison run.

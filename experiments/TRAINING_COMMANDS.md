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

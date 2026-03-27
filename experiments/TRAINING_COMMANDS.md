# Training Commands

Run all commands from:

```bash
cd /notebooks/PMamba/experiments
```

## Train Branch 1 (Motion temporal branch)

```bash
python main.py \
  --config pointlstm.yaml \
  --work-dir ./work_dir/motion_e120 \
  --num-epoch 120 \
  --device 0
```

## Train Branch 2 (Additive stage 1qm-h256: wider quaternion merge winner)

```bash
python main.py \
  --config linear_branch.yaml \
  --work-dir ./work_dir/linear_branch_edgeconv_quatmerge_h256_e120 \
  --num-epoch 120 \
  --device 0
```

## Train Branch 2 (Additive stage 1qm-h256-rmsmerge previous winner)

```bash
python main.py \
  --config linear_branch_rmsmerge.yaml \
  --work-dir ./work_dir/linear_branch_edgeconv_quatmerge_rms_h256_e120 \
  --num-epoch 120 \
  --device 0
```

## Train Branch 2 (Additive stage 1qm-h256-weighted-rmsmerge current winner)

```bash
python main.py \
  --config linear_branch_weighted_rmsmerge.yaml \
  --work-dir ./work_dir/linear_branch_edgeconv_quatmerge_weighted_rms_h256_e120 \
  --num-epoch 120 \
  --device 0
```

## Notes

- `epoch120_model.pt` is saved because the configs use `save_interval: 5`.
- Evaluation now runs every 10 epochs before epoch 100, and every epoch from epoch 100 onward.

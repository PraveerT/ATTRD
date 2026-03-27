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

## Train Branch 2 (Standalone winner, 71.7842)

```bash
python main.py \
  --config linear_branch_stacked_quat_weighted_rmsmerge.yaml \
  --work-dir ./work_dir/linear_branch_edgeconv_quatstack_weighted_rms_h256_e120 \
  --num-epoch 120 \
  --device 0
```

## Train Branch 2 (Current best so far, stacked attention readout)

```bash
python main.py \
  --config linear_branch_stacked_quat_weighted_attreadout_rmsmerge.yaml \
  --work-dir ./work_dir/linear_branch_edgeconv_quatstack_weighted_attreadout_rms_h256_e120 \
  --num-epoch 120 \
  --device 0
```

## Train Branch 2 (Next trial: attention readout with lower `dropout`)

```bash
python main.py \
  --config linear_branch_stacked_quat_weighted_attreadout_rmsmerge_drop005.yaml \
  --work-dir ./work_dir/linear_branch_edgeconv_quatstack_weighted_attreadout_rms_drop005_h256_e120 \
  --num-epoch 120 \
  --device 0
```

## Notes

- `epoch120_model.pt` is saved because the configs use `save_interval: 5`.
- Evaluation now runs every 10 epochs before epoch 100, and every epoch from epoch 100 onward.

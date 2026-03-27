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

## Fine-Tune Branch 1 + Branch 2 Fusion (Motion + weighted-rms winner)

```bash
python main.py \
  --config motion_reqnn_fusion.yaml \
  --work-dir ./work_dir/motion_weighted_rms_fusion_e240_ft \
  --temporal-weights ./work_dir/motion_e120/epoch110_model.pt \
  --spatial-weights ./work_dir/linear_branch_edgeconv_quatmerge_weighted_rms_h256_e120/epoch110_model.pt \
  --num-epoch 240 \
  --device 0
```

## Train Branch 2 (Additive stage 1qm-h256-weighted-gate-rmsmerge recent non-winner)

```bash
python main.py \
  --config linear_branch_weighted_gate_rmsmerge.yaml \
  --work-dir ./work_dir/linear_branch_edgeconv_quatmerge_weighted_gate_rms_h256_e120 \
  --num-epoch 120 \
  --device 0
```

## Train Branch 2 (Additive stage 1qm-h256-dualquat-weighted-rmsmerge earlier non-winner)

```bash
python main.py \
  --config linear_branch_dualquat_weighted_rmsmerge.yaml \
  --work-dir ./work_dir/linear_branch_edgeconv_dualquat_weighted_rms_h256_e120 \
  --num-epoch 120 \
  --device 0
```

## Train Branch 2 (Additive stage 1qm-h256-weighted-similarity-rot-rmsmerge earlier non-winner)

```bash
python main.py \
  --config linear_branch_weighted_similarity_rot_rmsmerge.yaml \
  --work-dir ./work_dir/linear_branch_edgeconv_quatmerge_weighted_similarity_rot_rms_h256_e120 \
  --num-epoch 120 \
  --device 0
```

## Train Branch 2 (Additive stage 1qm-h256-quatstack-weighted-rmsmerge next trial)

```bash
python main.py \
  --config linear_branch_stacked_quat_weighted_rmsmerge.yaml \
  --work-dir ./work_dir/linear_branch_edgeconv_quatstack_weighted_rms_h256_e120 \
  --num-epoch 120 \
  --device 0
```

## Train Branch 2 (Additive stage qec-cache-localframe-weighted-rmsmerge next trial)

```bash
python main.py \
  --config linear_branch_qec_cache_weighted_rmsmerge.yaml \
  --work-dir ./work_dir/linear_branch_edgeconv_qec_cache_weighted_rms_h256_e120 \
  --num-epoch 120 \
  --device 0
```

## Notes

- `epoch120_model.pt` is saved because the configs use `save_interval: 5`.
- Evaluation now runs every 10 epochs before epoch 100, and every epoch from epoch 100 onward.
- For late-epoch evaluation of dynamic-`pts_size` runs, pass `--pts-size 256` to match the
  in-training setting used from epoch 101 onward.

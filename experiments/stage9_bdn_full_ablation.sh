#!/bin/bash
# Stage 9: complete BD-N buffer-size ablation.
# Already have: buf=2 (3 seeds = 5.86 ± 1.51), buf=16 (3 seeds = 11.54 ± 0.27).
# Need: buf=4, buf=8, buf=32 × 3 seeds each = 9 new runs.
cd /notebooks/PMamba/experiments
mkdir -p mqar_results

for bs in 4 8 32; do
  for s in 0 1 2; do
    tag="stage9_bdn_buf${bs}_T64_v8192_L_lr1e-03_s${s}"
    out="mqar_results/${tag}.json"
    if [ -f "$out" ]; then echo "[skip] $tag"; continue; fi
    log="work_dir/${tag}.log"
    echo "[run] $tag"
    python mqar_rigor.py --arch bdn --vocab 8192 --T 64 --kv 8 --q 8 \
      --d_model 128 --head_dim 48 --d_read 32 --buffer_size $bs \
      --lr 1e-3 --seed $s --epochs 40 \
      --out_json "$out" --tag "$tag" > "$log" 2>&1
  done
done
echo "DONE stage9 BDN full ablation"

#!/bin/bash
# Stage 7 direct launch (no Stage 6 wait).
cd /notebooks/PMamba/experiments
mkdir -p mqar_results
for arch in moedn mamba2 tttdn; do
  for s in 0 1 2; do
    tag="stage7_T64_v8192_L_${arch}_lr1e-3_s${s}"
    out="mqar_results/${tag}.json"
    if [ -f "$out" ]; then echo "[skip] $tag"; continue; fi
    log="work_dir/${tag}.log"
    echo "[run] $tag"
    python mqar_rigor.py --arch $arch --vocab 8192 --T 64 --kv 8 --q 8 \
      --d_model 128 --head_dim 48 --d_read 32 \
      --lr 1e-3 --seed $s --epochs 40 \
      --out_json "$out" --tag "$tag" > "$log" 2>&1
  done
done
echo "DONE stage7"

#!/bin/bash
# Stage 4: Test 4 novel architectures (TP-DN, AdaB-DN, FD-N, BD-N) on MQAR.
# Config: vocab=8192 T=64 size L (~1.247M params, exact match with DN/AT).
# Phase A: lr=1e-3 (best for DN/AT) × 3 seeds = 12 runs (~1 hr at T=64).
# Phase B: LR sweep on best Phase-A architecture (added later if Phase A finds a winner).
cd /notebooks/PMamba/experiments
mkdir -p mqar_results

# Use the same d=128 hd=48 as DN size L (exact param match for novel variants)
for arch in tpdn adabdn fdn bdn; do
  for s in 0 1 2; do
    tag="stage4_T64_v8192_L_${arch}_lr1e-03_s${s}"
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
echo "DONE stage4 novel-arch lr=1e-3 phase"

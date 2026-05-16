#!/bin/bash
# Stage 8: SlotHopfield — bounded-slot content-addressable memory + Hopfield read.
# Different memory substrate than rank-1 delta: state = N (k,v) slots, soft-write
# by content addressing, Hopfield-style softmax read over slots.
#
# Param match: d=128 hd=44 → 1.246M (vs DN 1.247M, 0.08% off).
# Config: vocab=8192, T=64, lr=1e-3, 3 seeds = 3 runs (~30 min).
cd /notebooks/PMamba/experiments
mkdir -p mqar_results

# Wait for Stage 7 to finish first
echo "[stage8] waiting for stage 7 to finish..."
while true; do
  c=$(ls mqar_results/stage7_*.json 2>/dev/null | wc -l)
  if [ "$c" -ge 9 ]; then break; fi
  sleep 60
done
echo "[stage8] stage 7 complete, launching SlotHopfield."

for s in 0 1 2; do
  tag="stage8_T64_v8192_L_slothop_lr1e-3_s${s}"
  out="mqar_results/${tag}.json"
  if [ -f "$out" ]; then echo "[skip] $tag"; continue; fi
  log="work_dir/${tag}.log"
  echo "[run] $tag"
  python mqar_rigor.py --arch slothop --vocab 8192 --T 64 --kv 8 --q 8 \
    --d_model 128 --head_dim 44 --d_read 32 \
    --lr 1e-3 --seed $s --epochs 40 \
    --out_json "$out" --tag "$tag" > "$log" 2>&1
done
echo "DONE stage8 SlotHopfield"

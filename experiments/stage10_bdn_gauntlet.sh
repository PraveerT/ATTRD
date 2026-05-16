#!/bin/bash
# Stage 10: BD-N full gauntlet (mirrors AttRD Stage 1 + TPDN Stage 6 protocol).
# Tests whether BD-N's +6.84pp lift at vocab=8192 T=64 buf=16 generalizes.
#
# Stage 10A: vocab=256, sizes S (hd=32) and L (hd=48), LRs {3e-4, 1e-3},
#   3 seeds × buf=16 (same buffer that won at vocab=8192). = 12 runs.
# Stage 10B: vocab=8192 T=64 size L lr=3e-4 buf=16, 3 seeds = 3 runs.
# Stage 10C: vocab=8192 T=128 size L lr=1e-3 buf=16, 3 seeds = 3 runs.
#   NB: at T=128 kv=16, buf=16 holds only half the KVs (16 tokens vs 32),
#   forcing delta-state usage. Crucial test of hybrid mechanism vs capacity.
# Total: 18 new BD-N runs.
cd /notebooks/PMamba/experiments
mkdir -p mqar_results

# --- 10A: vocab=256 grid ---
for SIZE in S L; do
  if [ "$SIZE" = "S" ]; then HD=32; else HD=48; fi
  for LR in 3e-4 1e-3; do
    for s in 0 1 2; do
      tag="stage10a_T64_v256_${SIZE}_bdn_buf16_lr${LR}_s${s}"
      out="mqar_results/${tag}.json"
      if [ -f "$out" ]; then echo "[skip] $tag"; continue; fi
      log="work_dir/${tag}.log"
      echo "[run] $tag"
      python mqar_rigor.py --arch bdn --vocab 256 --T 64 --kv 8 --q 16 \
        --d_model 128 --head_dim $HD --d_read 32 --buffer_size 16 \
        --lr $LR --seed $s --epochs 60 \
        --out_json "$out" --tag "$tag" > "$log" 2>&1
    done
  done
done

# --- 10B: vocab=8192 T=64 lr=3e-4 (lr=1e-3 already in stage 4) ---
for s in 0 1 2; do
  tag="stage10b_T64_v8192_L_bdn_buf16_lr3e-4_s${s}"
  out="mqar_results/${tag}.json"
  if [ -f "$out" ]; then echo "[skip] $tag"; continue; fi
  log="work_dir/${tag}.log"
  echo "[run] $tag"
  python mqar_rigor.py --arch bdn --vocab 8192 --T 64 --kv 8 --q 8 \
    --d_model 128 --head_dim 48 --d_read 32 --buffer_size 16 \
    --lr 3e-4 --seed $s --epochs 40 \
    --out_json "$out" --tag "$tag" > "$log" 2>&1
done

# --- 10C: vocab=8192 T=128 lr=1e-3 ---
for s in 0 1 2; do
  tag="stage10c_T128_v8192_L_bdn_buf16_lr1e-3_s${s}"
  out="mqar_results/${tag}.json"
  if [ -f "$out" ]; then echo "[skip] $tag"; continue; fi
  log="work_dir/${tag}.log"
  echo "[run] $tag"
  python mqar_rigor.py --arch bdn --vocab 8192 --T 128 --kv 16 --q 16 \
    --d_model 128 --head_dim 48 --d_read 32 --buffer_size 16 \
    --lr 1e-3 --seed $s --epochs 40 \
    --out_json "$out" --tag "$tag" > "$log" 2>&1
done

echo "DONE stage10 BD-N gauntlet"

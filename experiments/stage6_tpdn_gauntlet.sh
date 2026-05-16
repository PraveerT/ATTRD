#!/bin/bash
# Stage 6: TPDN full gauntlet (mirrors AttRD Stage 1 + Stage 2 protocol).
# Verifies whether TPDN's +0.40pp lift (p=0.10) at vocab=8192 T=64 size L lr=1e-3
# survives the same battery that exposed AttRD's headline as confound.
#
# Stage 6A (mirrors AttRD Stage 1): vocab=256, sizes S (hd=32) and L (hd=48),
#   LRs {3e-4, 1e-3}, 3 seeds = 12 runs.
# Stage 6B (mirrors AttRD Stage 2A lr=3e-4): vocab=8192 size L lr=3e-4, 3 seeds
#   (lr=1e-3 already done in stage 4) = 3 new runs.
# Stage 6C (mirrors AttRD Stage 2B): vocab=8192 size L T=128 lr=1e-3, 3 seeds = 3 runs.
# Total: 18 new TPDN runs.
#
# Params (matched to DN within ~1K / 0.5%):
#   vocab=256  size S: hd=32  →  166.7K (vs DN 165.6K)
#   vocab=256  size L: hd=48  →  232.2K (vs DN 231.2K)
#   vocab=8192 size L: hd=48  →  1.248M (vs DN 1.247M)
cd /notebooks/PMamba/experiments
mkdir -p mqar_results

# --- Stage 6A: vocab=256 grid, two sizes, two LRs ---
for SIZE in S L; do
  if [ "$SIZE" = "S" ]; then HD=32; else HD=48; fi
  for LR in 3e-4 1e-3; do
    for s in 0 1 2; do
      tag="stage6a_T64_v256_${SIZE}_tpdn_lr${LR}_s${s}"
      out="mqar_results/${tag}.json"
      if [ -f "$out" ]; then echo "[skip] $tag"; continue; fi
      log="work_dir/${tag}.log"
      echo "[run] $tag"
      python mqar_rigor.py --arch tpdn --vocab 256 --T 64 --kv 8 --q 16 \
        --d_model 128 --head_dim $HD --d_read 32 \
        --lr $LR --seed $s --epochs 60 \
        --out_json "$out" --tag "$tag" > "$log" 2>&1
    done
  done
done

# --- Stage 6B: vocab=8192 T=64 size L, lr=3e-4 (lr=1e-3 already in stage4) ---
for s in 0 1 2; do
  tag="stage6b_T64_v8192_L_tpdn_lr3e-4_s${s}"
  out="mqar_results/${tag}.json"
  if [ -f "$out" ]; then echo "[skip] $tag"; continue; fi
  log="work_dir/${tag}.log"
  echo "[run] $tag"
  python mqar_rigor.py --arch tpdn --vocab 8192 --T 64 --kv 8 --q 8 \
    --d_model 128 --head_dim 48 --d_read 32 \
    --lr 3e-4 --seed $s --epochs 40 \
    --out_json "$out" --tag "$tag" > "$log" 2>&1
done

# --- Stage 6C: vocab=8192 T=128 size L, lr=1e-3 ---
for s in 0 1 2; do
  tag="stage6c_T128_v8192_L_tpdn_lr1e-3_s${s}"
  out="mqar_results/${tag}.json"
  if [ -f "$out" ]; then echo "[skip] $tag"; continue; fi
  log="work_dir/${tag}.log"
  echo "[run] $tag"
  python mqar_rigor.py --arch tpdn --vocab 8192 --T 128 --kv 16 --q 16 \
    --d_model 128 --head_dim 48 --d_read 32 \
    --lr 1e-3 --seed $s --epochs 40 \
    --out_json "$out" --tag "$tag" > "$log" 2>&1
done

echo "DONE stage6 TPDN gauntlet"

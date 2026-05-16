#!/bin/bash
# Stage 3: Try to beat the 4.74 DN/AT MQAR ceiling with tuned Transformer + variants.
# vocab=8192 T=64 size L (~1.247M params, exact-ish match), 3 seeds each.
cd /notebooks/PMamba/experiments
mkdir -p mqar_results

# Tuned Transformer: mlp_ratio=4, d=112, hd=32 (1.236M), warmup=0.1, lr in {1e-3, 3e-4, 1e-4}
for lr in 1e-3 3e-4 1e-4; do
  for s in 0 1 2; do
    tag="stage3_T64_v8192_TXtuned_lr${lr}_s${s}"
    out="mqar_results/${tag}.json"
    if [ -f "$out" ]; then echo "[skip] $tag"; continue; fi
    log="work_dir/${tag}.log"
    echo "[run] $tag"
    python mqar_rigor.py --arch transformer --vocab 8192 --T 64 --kv 8 --q 8 \
      --d_model 112 --head_dim 32 --d_read 32 --mlp_ratio 4 \
      --lr $lr --seed $s --warmup_frac 0.1 --epochs 40 \
      --out_json "$out" --tag "$tag" > "$log" 2>&1
  done
done
echo "DONE stage3 TX-tuned"

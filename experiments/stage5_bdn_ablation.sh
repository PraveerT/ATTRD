#!/bin/bash
# Stage 5: BD-N buffer-size ablation. Diagnostic test for whether the +146% rel
# lift at buffer_size=16 was "attention solving MQAR" or genuine mechanism.
#
# Config: vocab=8192 T=64 kv=8 q=8 size L (d=128 hd=48 ~1.247M params).
# Buffer sizes tested: 2, 4, 8, 16, 32 × 3 seeds = 15 runs.
# kv=8 means 16 KV-section tokens. Critical thresholds:
#   buffer=2,4,8 → buffer < 16 tokens → can't hold all KVs in attention
#   buffer=16    → exactly fits KV section → "attention solves it" hypothesis
#   buffer=32    → larger than needed; extra capacity shouldn't help if 16 is enough
cd /notebooks/PMamba/experiments
mkdir -p mqar_results

for bs in 2 4 8 32; do  # skip 16 since already done in stage4
  for s in 0 1 2; do
    tag="stage5_bdn_buf${bs}_T64_v8192_L_lr1e-03_s${s}"
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
echo "DONE stage5 BDN buffer-size ablation"

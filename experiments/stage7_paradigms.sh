#!/bin/bash
# Stage 7: 3 genuinely different paradigms vs DN baseline on MQAR.
# - MoE-DN: routed-expert DeltaNet (per-token top-k routing to E experts)
# - Mamba-2: SSD form selective scan SSM (diagonal scalar A)
# - TTT-DN: K inner gradient steps per token, learned step size
#
# Waits for Stage 6 TPDN gauntlet to finish before starting (avoids GPU contention).
# Config: vocab=8192, T=64, size L (~1.247M params exact match), lr=1e-3, 3 seeds.
# = 3 arches × 3 seeds = 9 runs (~1-2 hr at single-GPU).
cd /notebooks/PMamba/experiments
mkdir -p mqar_results

# Wait until stage 6 is fully complete (18 jsons)
echo "[stage7] waiting for stage 6 to finish..."
while true; do
  c=$(ls mqar_results/stage6[abc]_*.json 2>/dev/null | wc -l)
  if [ "$c" -ge 18 ]; then break; fi
  sleep 60
done
echo "[stage7] stage 6 complete, launching paradigm tests."

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
echo "DONE stage7 paradigms"

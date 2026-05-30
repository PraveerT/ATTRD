#!/bin/bash
# Confirm the +0.42 clean lift isn't seed noise: paired none vs remap @60ep, seeds 11 & 17.
cd /notebooks/Anemon
LOG=/notebooks/Anemon/experiments/work_dir/mirroraug_seeds_queue.log
echo "===== QUEUE START $(date) =====" >> "$LOG"
for S in 11 17; do
  for AUG in none remap; do
    WD=/notebooks/Anemon/experiments/work_dir/mirroraug_s${S}_${AUG}
    echo "===== START seed${S} ${AUG} -> ${WD} $(date) =====" >> "$LOG"
    python experiments/train_depth_mirroraug.py --aug "${AUG}" --epochs 60 --bs 8 \
        --seed "${S}" --workdir "${WD}" >> "$LOG" 2>&1
    echo "===== END seed${S} ${AUG} (exit $?) $(date) =====" >> "$LOG"
  done
done
echo "===== QUEUE ALL DONE $(date) =====" >> "$LOG"

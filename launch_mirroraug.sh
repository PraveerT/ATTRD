#!/bin/bash
# Label-aware mirror-aug: paired queue none -> plain -> remap, identical recipe/seed.
cd /notebooks/Anemon
LOG=/notebooks/Anemon/experiments/work_dir/mirroraug_queue.log
echo "===== QUEUE START $(date) =====" >> "$LOG"
for AUG in none plain remap; do
  WD=/notebooks/Anemon/experiments/work_dir/mirroraug_${AUG}
  echo "===== START ${AUG} -> ${WD} $(date) =====" >> "$LOG"
  python experiments/train_depth_mirroraug.py --aug "${AUG}" --epochs 30 --bs 8 \
      --workdir "${WD}" >> "$LOG" 2>&1
  echo "===== END ${AUG} (exit $?) $(date) =====" >> "$LOG"
done
echo "===== QUEUE ALL DONE $(date) =====" >> "$LOG"

#!/bin/bash
# Confound check: does mirror-aug's clean-acc cost survive longer training?
# Paired none vs remap at 60 epochs (aug converges slower than baseline).
cd /notebooks/Anemon
LOG=/notebooks/Anemon/experiments/work_dir/mirroraug60_queue.log
echo "===== QUEUE START $(date) =====" >> "$LOG"
for AUG in none remap; do
  WD=/notebooks/Anemon/experiments/work_dir/mirroraug60_${AUG}
  echo "===== START ${AUG} (60ep) -> ${WD} $(date) =====" >> "$LOG"
  python experiments/train_depth_mirroraug.py --aug "${AUG}" --epochs 60 --bs 8 \
      --workdir "${WD}" >> "$LOG" 2>&1
  echo "===== END ${AUG} (exit $?) $(date) =====" >> "$LOG"
done
echo "===== QUEUE ALL DONE $(date) =====" >> "$LOG"

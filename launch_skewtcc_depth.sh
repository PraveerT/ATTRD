#!/bin/bash
# Skew-TCC on the FG83 depth R(2+1)D model: paired control queue.
# off=recipe-matched baseline, skew=contribution, sym=symmetric-bilinear control,
# random=frozen-projector (structural-presence) control. Identical recipe + seed.
cd /notebooks/Anemon
LOG=/notebooks/Anemon/experiments/work_dir/skewtcc_depth_queue.log
echo "===== QUEUE START $(date) =====" >> "$LOG"
for MODE in off skew sym random; do
  WD=/notebooks/Anemon/experiments/work_dir/skewtcc_depth_${MODE}
  echo "===== START ${MODE} -> ${WD} $(date) =====" >> "$LOG"
  python experiments/train_depth_skewtcc.py --mode "${MODE}" --tap layer3 --bs 8 \
      --epochs 30 --workdir "${WD}" >> "$LOG" 2>&1
  echo "===== END ${MODE} (exit $?) $(date) =====" >> "$LOG"
done
echo "===== QUEUE ALL DONE $(date) =====" >> "$LOG"

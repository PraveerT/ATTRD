#!/usr/bin/env bash
# Re-launch the fusion watcher whenever it exits.
cd /notebooks/Anemon/sidepanel_api
while true; do
  python3 -u fusion_watcher.py 60 >> state/fusion_watcher.log 2>&1
  echo "[watchdog] $(date '+%F %T') fusion_watcher exited, restarting in 10s..." >> state/fusion_watcher.log
  sleep 10
done

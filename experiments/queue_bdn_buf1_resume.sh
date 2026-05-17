#!/bin/bash
# Wait for BDN-Q to finish, then resume BDN buf=1 from epoch80 ckpt.
cd /notebooks/PMamba/experiments
echo "[queue] waiting for BDN-Q to finish..."
while ps -ef | grep -v grep | grep -q 'main.py --config pmamba_baseline_bdnq.yaml'; do
  sleep 60
done
echo "[queue] BDN-Q done at $(date), resuming BDN buf=1 from ep80..."
nohup python main.py --config pmamba_baseline_bdn_buf1.yaml > work_dir/bdn_buf1_resume.log 2>&1 &
echo "[queue] launched pid=$!"

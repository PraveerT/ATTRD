#!/bin/bash
# Wait for buf=4 BD-N to finish, then launch buf=2 variant.
cd /notebooks/PMamba/experiments
echo "[queue] waiting for buf=4 BD-N to finish..."
while ps -ef | grep -v grep | grep -q 'main.py --config pmamba_baseline_bdn.yaml'; do
  sleep 60
done
echo "[queue] buf=4 done, launching buf=2..."
nohup python main.py --config pmamba_baseline_bdn_buf2.yaml > work_dir/bdn_buf2_train.log 2>&1 &
echo "[queue] launched pid=$!"

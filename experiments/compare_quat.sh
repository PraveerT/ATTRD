#!/bin/bash
# Compare best test acc across baseline + quaternion variants.
cd /notebooks/Anemon/experiments

printf '%-25s %-15s %-15s %-15s\n' RUN BEST_TEST_ACC BEST_EPOCH FINAL_EPOCH

for run in cn_xxl cn_xxl_quat cn_xxl_quat_head; do
    log=work_dir/${run}/log.txt
    if [ ! -f "$log" ]; then
        printf '%-25s %-15s %-15s %-15s\n' "$run" "no_log" "-" "-"
        continue
    fi
    # Best test acc line + epoch
    best_line=$(grep 'Saved new best' "$log" | tail -1)
    best_acc=$(echo "$best_line" | grep -oE 'prec1=[0-9.]+' | head -1 | cut -d= -f2)
    # Find epoch from preceding context
    best_epoch=$(grep -B1 'Saved new best' "$log" | grep -oE 'epoch [0-9]+, Test' | tail -1 | grep -oE '[0-9]+')
    # Final epoch reached
    final_ep=$(grep 'Training epoch:' "$log" | tail -1 | grep -oE 'epoch: [0-9]+' | grep -oE '[0-9]+')
    printf '%-25s %-15s %-15s %-15s\n' "$run" "${best_acc:-pending}" "${best_epoch:-?}" "${final_ep:-?}"
done

echo
echo "All Test, Evaluation lines for last 5 epochs of each run:"
for run in cn_xxl cn_xxl_quat cn_xxl_quat_head; do
    log=work_dir/${run}/log.txt
    [ -f "$log" ] || continue
    echo "=== $run ==="
    grep -E 'Test, Evaluation' "$log" | tail -5
done

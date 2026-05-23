"""Oracle + fusion: cnxxlquat 91.08 vs perpoint_frozen 91.08.

Both models tie at 91.08 on the 482-sample NV test. If their errors are
decorrelated, an oracle (perfect router) goes way higher and softmax fusion
should lift too. If they share errors, both numbers stay at 91.08.
"""
import os, re, numpy as np

A_PATH = '/notebooks/Anemon/experiments/work_dir/cn_xxl_quat_head/test_logits.npz'
B_PATH = '/notebooks/Anemon/experiments/work_dir/cn_xxl_quat_head_perpoint_frozen/test_logits.npz'

def softmax(x, ax=-1):
    x = x - x.max(ax, keepdims=True); e = np.exp(x)
    return e / e.sum(ax, keepdims=True)

A = np.load(A_PATH, allow_pickle=True)
B = np.load(B_PATH, allow_pickle=True)
a_log, a_lab, a_sigs = A['logits'], A['labels'], A['sigs']
b_log, b_lab, b_sigs = B['logits'], B['labels'], B['sigs']

# Align B to A's sig order.
b_by_sig = {s: i for i, s in enumerate(b_sigs)}
order = np.array([b_by_sig[s] for s in a_sigs])
b_log = b_log[order]
b_lab = b_lab[order]
assert (a_lab == b_lab).all(), 'label mismatch!'

a_pred = a_log.argmax(1); b_pred = b_log.argmax(1)
a_acc  = (a_pred == a_lab).mean() * 100
b_acc  = (b_pred == a_lab).mean() * 100
print(f'cnxxlquat solo:        {a_acc:.2f}%')
print(f'perpoint_frozen solo:  {b_acc:.2f}%')

# Error analysis.
a_wrong = a_pred != a_lab
b_wrong = b_pred != a_lab
shared_wrong = a_wrong & b_wrong
disagree    = a_pred != b_pred
both_correct = ~a_wrong & ~b_wrong
print(f'\nA wrong: {a_wrong.sum()}/482   ({100-a_acc:.2f}%)')
print(f'B wrong: {b_wrong.sum()}/482   ({100-b_acc:.2f}%)')
print(f'shared wrong (both): {shared_wrong.sum()}')
print(f'A wrong only: {(a_wrong & ~b_wrong).sum()}')
print(f'B wrong only: {(b_wrong & ~a_wrong).sum()}')
print(f'A != B (any disagreement): {disagree.sum()}/482')

# Oracle: at least one is correct -> correct.
either_correct = ~shared_wrong
oracle = either_correct.mean() * 100
print(f'\nORACLE (perfect router): {oracle:.2f}%   (+{oracle - max(a_acc,b_acc):.2f} over best solo)')

# Honest fusions.
sm_a = softmax(a_log); sm_b = softmax(b_log)
def report(name, p):
    acc = (p.argmax(1) == a_lab).mean() * 100
    print(f'  {name}: {acc:.2f}%')

print('\n=== honest fusion (uniform 1/2) ===')
report('softmax avg',  (sm_a + sm_b) / 2)
report('logit avg  ',  (a_log + b_log) / 2)

# Weight sweep on softmax fusion (diagnostic).
print('\n=== softmax weight sweep (diagnostic) ===')
for w in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    fused = (1-w) * sm_a + w * sm_b
    print(f'  w_B={w:.2f}  acc={(fused.argmax(1)==a_lab).mean()*100:.2f}%')

# Logit temperature sweep on B (in case different calibration helps).
print('\n=== softmax sweep with B temperature scaling ===')
for T in [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]:
    sm_bT = softmax(b_log * T)
    fused = (sm_a + sm_bT) / 2
    print(f'  T_B={T:.2f}  fused={(fused.argmax(1)==a_lab).mean()*100:.2f}%')

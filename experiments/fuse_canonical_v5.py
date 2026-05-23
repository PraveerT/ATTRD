"""Oracle + fusion: cnxxlquat 91.08 vs cn_xxl_canonical v5 ep84 (79.46)."""
import re, numpy as np

A_PATH = '/notebooks/Anemon/experiments/work_dir/cn_xxl_quat_head/test_logits.npz'
B_PATH = '/notebooks/Anemon/experiments/work_dir/cn_xxl_canonical/test_logits.npz'

def softmax(x, ax=-1):
    x = x - x.max(ax, keepdims=True); e = np.exp(x)
    return e / e.sum(ax, keepdims=True)

A = np.load(A_PATH, allow_pickle=True)
B = np.load(B_PATH, allow_pickle=True)
a_log, a_lab, a_sigs = A['logits'], A['labels'], A['sigs']
b_log, b_lab, b_sigs = B['logits'], B['labels'], B['sigs']
b_by_sig = {s: i for i, s in enumerate(b_sigs)}
order = np.array([b_by_sig[s] for s in a_sigs])
b_log = b_log[order]; b_lab = b_lab[order]
assert (a_lab == b_lab).all(), 'label mismatch'

a_pred = a_log.argmax(1); b_pred = b_log.argmax(1)
a_acc = (a_pred == a_lab).mean() * 100
b_acc = (b_pred == a_lab).mean() * 100
print(f'cnxxlquat solo:          {a_acc:.2f}%')
print(f'cn_xxl_canonical solo:   {b_acc:.2f}%')

a_wrong = a_pred != a_lab
b_wrong = b_pred != a_lab
shared_wrong = a_wrong & b_wrong
print(f'\nA wrong: {a_wrong.sum()}/482')
print(f'B wrong: {b_wrong.sum()}/482')
print(f'shared wrong (both): {shared_wrong.sum()}')
print(f'A wrong only: {(a_wrong & ~b_wrong).sum()}  (canonical saves these)')
print(f'B wrong only: {(b_wrong & ~a_wrong).sum()}')
print(f'A != B disagreements: {(a_pred != b_pred).sum()}/482')

oracle = (~shared_wrong).mean() * 100
print(f'\nORACLE (perfect router): {oracle:.2f}%   (+{oracle - max(a_acc, b_acc):.2f} over best solo)')

sm_a = softmax(a_log); sm_b = softmax(b_log)
def rep(name, p):
    print(f'  {name}: {(p.argmax(1)==a_lab).mean()*100:.2f}%')

print('\n=== honest fusion (uniform 1/2) ===')
rep('softmax avg', (sm_a + sm_b) / 2)
rep('logit avg  ', (a_log + b_log) / 2)

print('\n=== softmax weight sweep (diagnostic) ===')
for w in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    acc = (((1-w)*sm_a + w*sm_b).argmax(1)==a_lab).mean()*100
    print(f'  w_B={w:.2f}  acc={acc:.2f}%')

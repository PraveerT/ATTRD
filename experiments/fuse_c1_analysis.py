"""Compare C1 errors vs canonical baseline + cnxxlquat. Oracle + fusion."""
import re, numpy as np

CN_QUAT = '/notebooks/Anemon/experiments/work_dir/cn_xxl_quat_head/test_logits.npz'
CN_CANON = '/notebooks/Anemon/experiments/work_dir/cn_xxl_canonical/test_logits.npz'
CN_C1 = '/notebooks/Anemon/experiments/work_dir/cn_xxl_canonical_stqnet_c1/test_logits.npz'
DSN = '/notebooks/Anemon/dsn_official_valid_logits.npz'

def softmax(x, ax=-1):
    x = x - x.max(ax, keepdims=True); e = np.exp(x)
    return e / e.sum(ax, keepdims=True)

def sig_of(p):
    m = re.search(r'class_(\d+)/subject(\d+)_r(\d+)', p)
    return f'class_{m.group(1)}/subject{m.group(2)}_r{m.group(3)}'

def load(p):
    A = np.load(p, allow_pickle=True)
    sigs = [s if 'class_' in str(s)[:7] else sig_of(str(s)) for s in A['sigs']] if 'sigs' in A.files else \
           [sig_of(s) for s in A['paths']]
    return A['logits'], A['labels'], np.array(sigs)

q_log, q_lab, q_sigs = load(CN_QUAT)
c_log, c_lab, c_sigs = load(CN_CANON)
c1_log, c1_lab, c1_sigs = load(CN_C1)
dsn_log, dsn_lab, dsn_sigs = load(DSN)

# Align everything to q_sigs order.
def align(log, lab, sigs, ref_sigs):
    by_sig = {s: i for i, s in enumerate(sigs)}
    order = np.array([by_sig[s] for s in ref_sigs])
    return log[order], lab[order]

c_log, c_lab = align(c_log, c_lab, c_sigs, q_sigs)
c1_log, c1_lab = align(c1_log, c1_lab, c1_sigs, q_sigs)
dsn_log, dsn_lab = align(dsn_log, dsn_lab, dsn_sigs, q_sigs)
assert (c_lab == q_lab).all() and (c1_lab == q_lab).all() and (dsn_lab == q_lab).all()

lab = q_lab

def report(name, log):
    pred = log.argmax(1)
    acc = (pred == lab).mean() * 100
    wrong = (pred != lab)
    return pred, wrong, acc

q_pred, q_wrong, q_acc = report('cnxxlquat', q_log)
c_pred, c_wrong, c_acc = report('canonical', c_log)
c1_pred, c1_wrong, c1_acc = report('C1', c1_log)
d_pred, d_wrong, d_acc = report('dsn', dsn_log)

print(f'cnxxlquat   solo: {q_acc:.2f}%   wrong: {q_wrong.sum()}/482')
print(f'canonical   solo: {c_acc:.2f}%   wrong: {c_wrong.sum()}/482')
print(f'C1 (cluster) solo: {c1_acc:.2f}%   wrong: {c1_wrong.sum()}/482')
print(f'dsn         solo: {d_acc:.2f}%   wrong: {d_wrong.sum()}/482')

print('\n=== C1 vs canonical baseline (same 79.46 ckpt warm-start) ===')
print(f'shared wrong: {(c_wrong & c1_wrong).sum()}')
print(f'canonical-only wrong (C1 SAVES): {(c_wrong & ~c1_wrong).sum()}')
print(f'C1-only wrong: {(c1_wrong & ~c_wrong).sum()}')
print(f'argmax disagreements: {(c_pred != c1_pred).sum()}/482')
print(f'oracle(canon, C1): {(~(c_wrong & c1_wrong)).mean()*100:.2f}%')

print('\n=== C1 vs cnxxlquat 91.08 ===')
print(f'shared wrong: {(q_wrong & c1_wrong).sum()}')
print(f'cnxxlquat-only wrong (C1 SAVES): {(q_wrong & ~c1_wrong).sum()}')
print(f'C1-only wrong: {(c1_wrong & ~q_wrong).sum()}')
print(f'argmax disagreements: {(q_pred != c1_pred).sum()}/482')
print(f'oracle(cnxxl, C1): {(~(q_wrong & c1_wrong)).mean()*100:.2f}%')

print('\n=== 3-way oracle (cnxxl, canon, C1) ===')
all_wrong3 = q_wrong & c_wrong & c1_wrong
print(f'shared wrong (all 3): {all_wrong3.sum()}')
print(f'3-way oracle: {(~all_wrong3).mean()*100:.2f}%')

print('\n=== 4-way oracle (cnxxl, canon, C1, DSN) ===')
all_wrong4 = q_wrong & c_wrong & c1_wrong & d_wrong
print(f'shared wrong (all 4): {all_wrong4.sum()}')
print(f'4-way oracle: {(~all_wrong4).mean()*100:.2f}%')

# Honest fusion experiments.
print('\n=== honest fusion (uniform softmax) ===')
sm_q = softmax(q_log); sm_c = softmax(c_log); sm_c1 = softmax(c1_log)
sm_d = softmax(dsn_log * 9.5)  # DSN T=9.5 calibration

def f(name, p):
    print(f'  {name}: {(p.argmax(1)==lab).mean()*100:.2f}%')

f('cnxxlquat+C1                /2', (sm_q + sm_c1) / 2)
f('cnxxlquat+canonical         /2', (sm_q + sm_c) / 2)
f('cnxxlquat+DSN(T=9.5)        /2', (sm_q + sm_d) / 2)
f('cnxxlquat+C1+canonical      /3', (sm_q + sm_c1 + sm_c) / 3)
f('cnxxlquat+DSN+C1            /3', (sm_q + sm_d + sm_c1) / 3)
f('cnxxlquat+DSN+canonical     /3', (sm_q + sm_d + sm_c) / 3)
f('cnxxlquat+DSN+canonical+C1  /4', (sm_q + sm_d + sm_c + sm_c1) / 4)

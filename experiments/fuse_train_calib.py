"""Honest fusion: calibrate DSN T_dsn so its TRAIN avg max-prob matches cnxxl's.
Then optimize fusion weight w using leave-one-subject-out CE on train (since
DSN is 100% on train; accuracy gives no signal but CE-rank does).
Apply final (T, w) to test.
"""
import os, re, numpy as np
from scipy.optimize import brentq, minimize_scalar

CN_TRAIN = '/notebooks/Anemon/experiments/work_dir/cn_xxl_quat_head/train_logits.npz'
DSN_TRAIN = '/notebooks/Anemon/dsn_official_train_logits.npz'
CN_TEST = '/notebooks/Anemon/experiments/work_dir/cn_xxl_quat_head/test_logits.npz'
DSN_TEST = '/notebooks/Anemon/dsn_official_valid_logits.npz'

def sig_of(p):
    m = re.search(r'class_(\d+)/subject(\d+)_r(\d+)', p)
    return f'class_{m.group(1)}/subject{m.group(2)}_r{m.group(3)}'

def subj_of(p):
    m = re.search(r'subject(\d+)_r', p)
    return int(m.group(1))

def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True); e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)

def load_pair(cn_path, dsn_path):
    A = np.load(cn_path, allow_pickle=True)
    B = np.load(dsn_path, allow_pickle=True)
    cn_log, cn_lab, cn_sigs = A['logits'], A['labels'], A['sigs']
    dsn_log, dsn_lab, dsn_paths = B['logits'], B['labels'], B['paths']
    dsn_by_sig = {sig_of(p): dsn_log[i] for i, p in enumerate(dsn_paths)}
    dsn_log_aligned = np.stack([dsn_by_sig[s] for s in cn_sigs])
    subjects = np.array([subj_of(s) for s in cn_sigs])
    return cn_log, dsn_log_aligned, cn_lab, cn_sigs, subjects

cn_tr, dsn_tr, lab_tr, sig_tr, subj_tr = load_pair(CN_TRAIN, DSN_TRAIN)
cn_te, dsn_te, lab_te, sig_te, subj_te = load_pair(CN_TEST, DSN_TEST)
print(f'train: cn_acc={(cn_tr.argmax(1)==lab_tr).mean()*100:.2f}% dsn_acc={(dsn_tr.argmax(1)==lab_tr).mean()*100:.2f}%')
print(f'test:  cn_acc={(cn_te.argmax(1)==lab_te).mean()*100:.2f}% dsn_acc={(dsn_te.argmax(1)==lab_te).mean()*100:.2f}%')
print(f'train subjects: {sorted(set(subj_tr))}')
print(f'test subjects:  {sorted(set(subj_te))}')

# 1) Calibrate T_dsn by matching TRAIN avg max-prob.
cn_avg_max = softmax(cn_tr).max(1).mean()
print(f'\ncn train avg max-prob: {cn_avg_max:.4f}')

def dsn_avg_max(T):
    return softmax(dsn_tr * T).max(1).mean()

# Find T such that dsn_avg_max(T) = cn_avg_max.
T_cal = brentq(lambda T: dsn_avg_max(T) - cn_avg_max, 0.1, 100)
print(f'T_dsn matching cn: {T_cal:.3f}  (memory: 9.5)')

# 2) Optimize fusion weight w via leave-one-subject-out CE.
def fused(cn_log, dsn_log, T, w):
    sm_cn = softmax(cn_log)
    sm_dsn = softmax(dsn_log * T)
    return w * sm_dsn + (1 - w) * sm_cn

def loo_loss(w, T):
    """Sum of CE on each held-out subject, fusion built from non-S samples does
    not matter here because the fusion weights are global. Just compute fusion
    CE per-subject and sum -- equivalent to global CE on whole train."""
    f = fused(cn_tr, dsn_tr, T, w)
    p_true = f[np.arange(len(lab_tr)), lab_tr]
    return -np.log(p_true.clip(1e-9)).mean()

def loo_acc(w, T):
    f = fused(cn_tr, dsn_tr, T, w)
    return (f.argmax(1) == lab_tr).mean()

# Skip CE (DSN at 100% saturates) -- use TRAIN-MISCLASSIFIED-only mean CE
mistakes = np.where(cn_tr.argmax(1) != lab_tr)[0]
print(f'\ntrain samples cn-wrong: {len(mistakes)}')
def loss_on_cn_mistakes(w, T):
    f = fused(cn_tr[mistakes], dsn_tr[mistakes], T, w)
    p_true = f[np.arange(len(mistakes)), lab_tr[mistakes]]
    return -np.log(p_true.clip(1e-9)).mean()

# Sweep
print('\n=== train sweep at T=T_cal (CE on cn-wrong train samples) ===')
for w in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
    print(f'  w={w:.2f}  train CE(cn-wrong)={loss_on_cn_mistakes(w, T_cal):.4f}  '
          f'train acc={loo_acc(w, T_cal)*100:.2f}%')

# 3) Optimize w by minimizing CE on cn-wrong train samples
res = minimize_scalar(lambda w: loss_on_cn_mistakes(w, T_cal), bounds=(0.05, 0.95), method='bounded')
w_opt = res.x
print(f'\noptimal w (train-CE-on-cn-wrong, T=T_cal): {w_opt:.3f}')

# 4) Apply (T_cal, w_opt) to test.
print('\n=== test results ===')
def report(name, T, w):
    f = fused(cn_te, dsn_te, T, w)
    acc = (f.argmax(1) == lab_te).mean() * 100
    print(f'  {name}: T={T:.3f} w={w:.3f}  acc={acc:.2f}%')

report('cnxxl solo',  1.0, 0.0)
report('dsn solo',    T_cal, 1.0)
report('uniform 1/2 (memory)', 9.5, 0.5)
report('calibrated T, uniform w=0.5', T_cal, 0.5)
report('calibrated T, optimized w on train', T_cal, w_opt)

# 5) Bonus: optimize T too via joint train-CE-on-cn-wrong
from scipy.optimize import minimize
def joint_loss(x):
    T, w = x
    return loss_on_cn_mistakes(w, T)

res = minimize(joint_loss, x0=[T_cal, w_opt], method='Nelder-Mead',
               options={'xatol':1e-3, 'fatol':1e-5})
T_jopt, w_jopt = res.x
print(f'\njoint-optimal (T,w) on train-cn-wrong CE: T={T_jopt:.3f} w={w_jopt:.3f}')
report('joint-opt (T,w)', T_jopt, w_jopt)

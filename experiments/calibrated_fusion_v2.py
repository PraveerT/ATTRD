"""Temperature-calibrated 2-way fusion with proper protocol:
  - hold out 20% of train as calibration set
  - fit DSN temperature T via NLL on held-out
  - evaluate fusion on test (untouched)
"""
import os, sys
import numpy as np

os.chdir('/notebooks/Anemon/experiments')


def softmax_np(x, T=1.0):
    x = x / T
    x = x - x.max(-1, keepdims=True); e = np.exp(x); return e / e.sum(-1, keepdims=True)


def nll(logits, labels, T=1.0):
    """Standard temperature-scaling NLL."""
    sm = softmax_np(logits, T)
    p_y = sm[np.arange(len(labels)), labels].clip(1e-12, 1.0)
    return -np.log(p_y).mean()


def load(name, path):
    z = np.load(path, allow_pickle=True)
    if 'logits' in z.files: l = z['logits']
    elif 'pred_logits' in z.files: l = z['pred_logits']
    else: raise SystemExit(f'{name}: keys {z.files}')
    return l, z['labels']


cn_tr_l, lab = load('cnxxl train', './work_dir/cn_xxl_quat_head/train_logits.npz')
dsn_tr_l, lab2 = load('dsn train',  '/notebooks/Anemon/dsn_official_train_logits.npz')
assert (lab == lab2).all()

cn_te_l, te_lab = load('cnxxl test', './work_dir/cn_xxl_quat_head/test_logits.npz')
dsn_te_l, te_lab2 = load('dsn test', '/notebooks/Anemon/dsn_official_valid_logits.npz')
assert (te_lab == te_lab2).all()

N = len(lab)
rng = np.random.RandomState(0)
perm = rng.permutation(N)
n_cal = int(N * 0.20)
cal_idx = perm[:n_cal]
trn_idx = perm[n_cal:]

cn_cal = cn_tr_l[cal_idx]; dsn_cal = dsn_tr_l[cal_idx]; lab_cal = lab[cal_idx]
print(f'cal split: {len(cal_idx)} samples')
print(f'  cnxxl cal acc: {(cn_cal.argmax(1) == lab_cal).mean() * 100:.2f}%')
print(f'  dsn   cal acc: {(dsn_cal.argmax(1) == lab_cal).mean() * 100:.2f}%')

print('\n==== T tuning on calibration split (NLL objective) ====')
print('  T values are evaluated on CAL set only:')
Ts = [0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0]

# 1) Tune DSN T via NLL on cal
best_nll_T = None; best_nll = float('inf')
for T in Ts:
    n_cnxxl = nll(cn_cal, lab_cal, T=1.0)
    n_dsn = nll(dsn_cal, lab_cal, T=T)
    # Joint NLL of fusion
    fuse_cal = 0.5 * softmax_np(cn_cal, T=1.0) + 0.5 * softmax_np(dsn_cal, T=T)
    pf = fuse_cal[np.arange(len(lab_cal)), lab_cal].clip(1e-12, 1.0)
    n_fuse = -np.log(pf).mean()
    a_fuse = (fuse_cal.argmax(1) == lab_cal).mean() * 100
    print(f'  T={T:.3f}  dsn_NLL={n_dsn:.3f}  fuse_NLL={n_fuse:.3f}  fuse_acc={a_fuse:.2f}%')
    if n_fuse < best_nll:
        best_nll = n_fuse; best_nll_T = T

print(f'\n  best T (fusion NLL min on cal): {best_nll_T}  (NLL={best_nll:.3f})')

# 2) Also try matching mean max-softmax of cnxxl
cn_cal_sm = softmax_np(cn_cal, T=1.0)
target_mean_max = cn_cal_sm.max(1).mean()
print(f'\n  cnxxl cal mean max-softmax: {target_mean_max:.4f}')
print('  Searching DSN T to match this mean max:')
best_match_T = None; best_diff = float('inf')
for T in Ts:
    dsn_sm = softmax_np(dsn_cal, T=T)
    m = dsn_sm.max(1).mean()
    diff = abs(m - target_mean_max)
    print(f'    T={T:.3f}  dsn mean max={m:.4f}  |diff|={diff:.4f}')
    if diff < best_diff:
        best_diff = diff; best_match_T = T
print(f'\n  best T (mean-max match on cal): {best_match_T}')

# Apply both choices to test, compare to T=1 baseline
print('\n==== TEST evaluation ====')
cn_te_sm = softmax_np(cn_te_l, T=1.0)
results = []

def report(name, T):
    dsn_sm = softmax_np(dsn_te_l, T=T)
    fuse = 0.5 * cn_te_sm + 0.5 * dsn_sm
    a = (fuse.argmax(1) == te_lab).mean() * 100
    results.append((name, T, a))
    print(f'  {name:30s} T={T:.3f}  test_fused_acc={a:.2f}%')

report('cnxxl solo (no DSN)',            float('inf'))  # use T=inf effectively
# proper cnxxl solo: just argmax of cn_te_l
print(f'  cnxxl solo argmax:               {(cn_te_l.argmax(1) == te_lab).mean()*100:.2f}%')
print(f'  dsn solo argmax:                 {(dsn_te_l.argmax(1) == te_lab).mean()*100:.2f}%')

# Reset and report properly
results = []
for T_choice, name in [(1.0, 'T=1 (no calibration)'),
                        (best_nll_T, f'cal NLL T={best_nll_T}'),
                        (best_match_T, f'cal mean-max T={best_match_T}')]:
    dsn_sm = softmax_np(dsn_te_l, T=T_choice)
    fuse = 0.5 * cn_te_sm + 0.5 * dsn_sm
    a = (fuse.argmax(1) == te_lab).mean() * 100
    results.append((name, T_choice, a))
    print(f'  {name:30s} test_fused={a:.2f}%')

# Test-only oracle sweep (informational)
print('\n  (test-only sweep, informational not used to pick):')
for T in Ts:
    a = ((0.5 * cn_te_sm + 0.5 * softmax_np(dsn_te_l, T=T)).argmax(1) == te_lab).mean() * 100
    print(f'    T={T:.3f}  test_fused={a:.2f}%')

# Best weighted fusion at chosen T
print(f'\n  Weighted fusion at chosen T={best_nll_T}:')
dsn_sm_chosen = softmax_np(dsn_te_l, T=best_nll_T)
for w in [0.3, 0.4, 0.5, 0.6, 0.7]:
    fuse = (1 - w) * cn_te_sm + w * dsn_sm_chosen
    a = (fuse.argmax(1) == te_lab).mean() * 100
    print(f'    DSN weight {w:.2f}: {a:.2f}%')

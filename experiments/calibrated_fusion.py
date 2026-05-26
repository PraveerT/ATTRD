"""Temperature-calibrated 2-way fusion: cnxxl + DSN.

Fit DSN temperature on TRAIN logits only (memory: never tune on test).
Search T in a grid to maximize train fusion accuracy. Then evaluate on test.
"""
import os, sys
import numpy as np

os.chdir('/notebooks/Anemon/experiments')

def softmax_np(x, T=1.0):
    x = x / T
    x = x - x.max(-1, keepdims=True); e = np.exp(x); return e / e.sum(-1, keepdims=True)

def load(name, path):
    z = np.load(path, allow_pickle=True)
    if 'logits' in z.files: l = z['logits']
    elif 'pred_logits' in z.files: l = z['pred_logits']
    else: raise SystemExit(f'{name}: no logits key in {z.files}')
    lab = z['labels']
    return l, lab

# Train
print('==== TRAIN logits (for T tuning) ====')
cn_tr_l, cn_tr_lab = load('cnxxl train', './work_dir/cn_xxl_quat_head/train_logits.npz')
dsn_tr_l, dsn_tr_lab = load('dsn train', '/notebooks/Anemon/dsn_official_train_logits.npz')
print(f'  cnxxl train: {cn_tr_l.shape}, dsn train: {dsn_tr_l.shape}')
print(f'  label match: {(cn_tr_lab == dsn_tr_lab).all()}')
if not (cn_tr_lab == dsn_tr_lab).all():
    raise SystemExit('train label order mismatch')

cn_tr_p = cn_tr_l.argmax(1)
dsn_tr_p = dsn_tr_l.argmax(1)
print(f'  cnxxl train acc: {(cn_tr_p == cn_tr_lab).mean()*100:.2f}%')
print(f'  dsn   train acc: {(dsn_tr_p == dsn_tr_lab).mean()*100:.2f}%')

cn_tr_sm = softmax_np(cn_tr_l, T=1.0)

print('\n==== Tune DSN temperature on TRAIN ====')
best_T = 1.0
best_acc = 0.0
results = []
# search T in [0.05, 5.0] — small T sharpens DSN logits (we want to BOOST DSN confidence since it was uniformly low)
for T in [0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0]:
    dsn_sm = softmax_np(dsn_tr_l, T=T)
    fused = 0.5 * cn_tr_sm + 0.5 * dsn_sm
    acc = (fused.argmax(1) == cn_tr_lab).mean() * 100
    results.append((T, acc))
    if acc > best_acc:
        best_acc = acc; best_T = T
    print(f'  T={T:.3f}  train_fused={acc:.2f}%')
print(f'\n  best T on train: {best_T}  (train fused acc {best_acc:.2f}%)')

# Test
print('\n==== TEST evaluation with calibrated T ====')
cn_l, cn_lab = load('cnxxl test', './work_dir/cn_xxl_quat_head/test_logits.npz')
dsn_l, dsn_lab = load('dsn test', '/notebooks/Anemon/dsn_official_valid_logits.npz')
assert (cn_lab == dsn_lab).all(), 'test label mismatch'

cn_te_sm = softmax_np(cn_l, T=1.0)
dsn_te_sm_orig = softmax_np(dsn_l, T=1.0)
dsn_te_sm_cal  = softmax_np(dsn_l, T=best_T)

cn_acc  = (cn_te_sm.argmax(1) == cn_lab).mean() * 100
dsn_acc = (dsn_te_sm_orig.argmax(1) == cn_lab).mean() * 100
print(f'  cnxxl solo:                              {cn_acc:.2f}%')
print(f'  dsn   solo:                              {dsn_acc:.2f}%')

# Uniform fusion at T=1
fuse_t1 = 0.5 * cn_te_sm + 0.5 * dsn_te_sm_orig
print(f'  uniform fusion (T=1):                    {(fuse_t1.argmax(1)==cn_lab).mean()*100:.2f}%')

# Uniform fusion with calibrated DSN
fuse_tcal = 0.5 * cn_te_sm + 0.5 * dsn_te_sm_cal
print(f'  uniform fusion (DSN T={best_T}):             {(fuse_tcal.argmax(1)==cn_lab).mean()*100:.2f}%')

# Best-T sweep on TEST too (informational only, NOT for picking)
print('\n  (test-only sweep — informational, NOT used to pick T):')
best_test = 0; best_test_T = 1
for T, _ in results:
    sm = softmax_np(dsn_l, T=T)
    a = ((0.5 * cn_te_sm + 0.5 * sm).argmax(1) == cn_lab).mean() * 100
    print(f'    T={T:.3f}  test_fused={a:.2f}%')
    if a > best_test: best_test, best_test_T = a, T
print(f'  best on test would be T={best_test_T} -> {best_test:.2f}% (DO NOT USE)')

# Also test a few weighted fusions with calibrated DSN
print(f'\n  Weighted fusion with T={best_T} on DSN:')
for w in [0.3, 0.4, 0.5, 0.6, 0.7]:
    fuse = (1 - w) * cn_te_sm + w * dsn_te_sm_cal
    a = (fuse.argmax(1) == cn_lab).mean() * 100
    print(f'    DSN weight {w:.2f}: {a:.2f}%')

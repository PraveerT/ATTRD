"""Error decorrelation: cnxxl vs DSN."""
import os, sys
import numpy as np

os.chdir('/notebooks/Anemon/experiments')

def softmax_np(x):
    x = x - x.max(-1, keepdims=True); e = np.exp(x); return e / e.sum(-1, keepdims=True)

def load(name, path):
    z = np.load(path, allow_pickle=True)
    if 'logits' in z.files: l = z['logits']
    elif 'pred_logits' in z.files: l = z['pred_logits']
    else: raise SystemExit(f'{name}: no logits in {z.files}')
    lab = z['labels']
    return l, l.argmax(1), lab

CN_PATH = './work_dir/cn_xxl_quat_head/test_logits.npz'
DSN_PATH = '/notebooks/Anemon/dsn_official_valid_logits.npz'

cn_l, cn_p, lab = load('cnxxl', CN_PATH)
dsn_l, dsn_p, lab2 = load('dsn', DSN_PATH)

print(f'cnxxl: {(cn_p == lab).sum()}/{len(lab)} = {(cn_p == lab).mean()*100:.2f}%')
print(f'dsn:   {(dsn_p == lab2).sum()}/{len(lab2)} = {(dsn_p == lab2).mean()*100:.2f}%')

# DSN logits may not be aligned to cnxxl test order. Check label sequence.
if len(lab) == len(lab2) and (lab == lab2).all():
    print('  label orders match.')
else:
    print('  WARNING: label orders differ. need to align via dsn_perm.npy')
    PERM = './work_dir/cn_xxl_quat_head/dsn_perm.npy'
    if os.path.exists(PERM):
        perm = np.load(PERM)
        dsn_l = dsn_l[perm]
        dsn_p = dsn_p[perm]
        lab2 = lab2[perm]
        ok = (lab == lab2).all()
        print(f'  applied dsn_perm.npy: aligned ok={ok}')

cn_err = (cn_p != lab); dsn_err = (dsn_p != lab)
both_wrong = cn_err & dsn_err
cn_only = cn_err & ~dsn_err
dsn_only = ~cn_err & dsn_err
both_right = ~cn_err & ~dsn_err

print(f'\n==== Pairwise error overlap (482 test samples) ====')
print(f'  both right:         {both_right.sum():3d}   ({both_right.mean()*100:.2f}%)')
print(f'  cnxxl wrong only:   {cn_only.sum():3d}   (dsn rescues)')
print(f'  dsn wrong only:     {dsn_only.sum():3d}   (cnxxl rescues)')
print(f'  both wrong:         {both_wrong.sum():3d}   (no rescue possible)')
print(f'  total cnxxl errors: {cn_err.sum():3d}')
print(f'  total dsn errors:   {dsn_err.sum():3d}')

union = (cn_err | dsn_err).sum()
inter = (cn_err & dsn_err).sum()
print(f'\n  error Jaccard:      {inter/union:.3f}')
print(f'  ORACLE upper bound: {(~(cn_err & dsn_err)).mean()*100:.2f}%')

cn_sm = softmax_np(cn_l); dsn_sm = softmax_np(dsn_l)
fused = (cn_sm + dsn_sm) / 2
fused_acc = (fused.argmax(1) == lab).mean() * 100
print(f'  2-way uniform fusion: {fused_acc:.2f}%')

# Per-class hot spots
print(f'\n==== Per-class disagreement ====')
print(f'  cls |  N |   cnxxl_err |   dsn_err  | unique_cn | unique_dsn | both_wrong')
for c in range(int(lab.max()) + 1):
    mask = (lab == c)
    n = mask.sum()
    if n == 0: continue
    ce = (cn_err & mask).sum()
    de = (dsn_err & mask).sum()
    uc = (cn_only & mask).sum()
    ud = (dsn_only & mask).sum()
    bw = (both_wrong & mask).sum()
    if ce > 0 or de > 0:
        print(f'  {c:3d} | {n:2d} |     {ce:3d} ({ce/n*100:5.1f}%) | {de:3d} ({de/n*100:5.1f}%) |     {uc:2d}    |      {ud:2d}    |    {bw:2d}')

# Confidence on disagreement
cn_conf = cn_sm.max(1); dsn_conf = dsn_sm.max(1)
print(f'\n==== Confidence on mixed-outcome samples ({(cn_only | dsn_only).sum()} total) ====')
print(f'  cnxxl conf when right (dsn wrong): {cn_conf[dsn_only].mean():.3f}')
print(f'  dsn   conf when right (cnxxl wrong): {dsn_conf[cn_only].mean():.3f}')
print(f'  cnxxl conf when wrong (dsn right):  {cn_conf[cn_only].mean():.3f}')
print(f'  dsn   conf when wrong (cnxxl right): {dsn_conf[dsn_only].mean():.3f}')

"""Fuse cnxxlquat 91.08 with I3DWTrans depth (DSN K-output.pth).

Both are evaluated on the 482-sample NV test set. Match by class/subject/rep
signature parsed from sample paths.

Reports:
  - solo accs (cnxxlquat, DSN)
  - softmax mean (uniform)
  - logit mean (uniform)
  - softmax with DSN scale sweep (no test tuning -- sweep is diagnostic)
"""
import os, re, numpy as np, torch

CNXXL = '/notebooks/Anemon/experiments/work_dir/cn_xxl_quat_head/test_logits.npz'
DSN   = '/notebooks/MotionRGBD/Checkpoints/I3DWTrans-NvGesture-K-20260514-012223/K-output.pth'

def sig_from_path(p):
    m = re.search(r'class_(\d+)/subject(\d+)_r(\d+)', p)
    return f'class_{m.group(1)}/subject{m.group(2)}_r{m.group(3)}'

def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)

# Load cnxxlquat dump.
A = np.load(CNXXL, allow_pickle=True)
cn_logits = A['logits']             # (482, 25)
cn_labels = A['labels']             # (482,)
cn_sigs   = A['sigs']               # (482,)
print(f'cnxxl: {cn_logits.shape} acc={(cn_logits.argmax(1)==cn_labels).mean()*100:.2f}%')

# Load DSN dump.
D = torch.load(DSN, map_location='cpu')
dsn_logits_by_sig = {}
for path, logits_t in D.items():
    s = sig_from_path(path)
    dsn_logits_by_sig[s] = logits_t.numpy()
print(f'dsn: {len(dsn_logits_by_sig)} samples')

# Align DSN logits to cnxxl order.
missing = [s for s in cn_sigs if s not in dsn_logits_by_sig]
print(f'missing in dsn: {len(missing)}')
dsn_logits = np.stack([dsn_logits_by_sig[s] for s in cn_sigs])    # (482, 25)

# Use Anemon dataset labels (cn_labels) throughout. 8/482 path-class labels
# disagree with dataset labels; trust the dataset since cnxxl was scored on it.
path_labels = np.array([int(re.search(r'class_(\d+)', s).group(1)) - 1 for s in cn_sigs])
n_disagree = int((path_labels != cn_labels).sum())
print(f'label disagreement (path vs dataset): {n_disagree}/482')

print(f'dsn solo acc (dataset labels): {(dsn_logits.argmax(1)==cn_labels).mean()*100:.2f}%')
print(f'dsn solo acc (path labels):    {(dsn_logits.argmax(1)==path_labels).mean()*100:.2f}%')

# Uniform softmax fusion.
sm_cn  = softmax(cn_logits)
sm_dsn = softmax(dsn_logits)
fused_sm = (sm_cn + sm_dsn) / 2
print(f'softmax uniform: {(fused_sm.argmax(1)==cn_labels).mean()*100:.2f}%')

# Uniform logit fusion.
fused_lg = (cn_logits + dsn_logits) / 2
print(f'logit uniform:   {(fused_lg.argmax(1)==cn_labels).mean()*100:.2f}%')

# DSN scale sweep on softmax (diagnostic; don't pick best for honest report).
print('\nDSN scale sweep (softmax = sm_cn + scale*sm_dsn, NORMALIZED):')
for sc in [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0]:
    f = sm_cn + sc * sm_dsn
    acc = (f.argmax(1)==cn_labels).mean()*100
    print(f'  scale={sc:>5.2f}  acc={acc:.2f}%')

# Logit scale sweep.
print('\nlogit scale sweep (cn + scale*dsn):')
for sc in [0.1, 0.25, 0.5, 0.75, 1.0]:
    f = cn_logits + sc * dsn_logits
    acc = (f.argmax(1)==cn_labels).mean()*100
    print(f'  scale={sc:>5.2f}  acc={acc:.2f}%')

# Save aligned dumps for further analysis.
np.savez('/notebooks/Anemon/experiments/work_dir/cn_xxl_quat_head/fusion_dsn.npz',
         cn_logits=cn_logits, dsn_logits=dsn_logits,
         labels=cn_labels, sigs=cn_sigs)
print('\nsaved fusion_dsn.npz')

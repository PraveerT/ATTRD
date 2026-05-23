"""Fuse cnxxlquat 91.08 with FRESH I3DW depth eval (79.67% solo)."""
import os, re, numpy as np

CNXXL = '/notebooks/Anemon/experiments/work_dir/cn_xxl_quat_head/test_logits.npz'
DSN   = '/notebooks/Anemon/dsn_official_valid_logits.npz'

def sig_of(p):
    m = re.search(r'class_(\d+)/subject(\d+)_r(\d+)', p)
    return f'class_{m.group(1)}/subject{m.group(2)}_r{m.group(3)}'

def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True); e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)

A = np.load(CNXXL, allow_pickle=True)
B = np.load(DSN, allow_pickle=True)
cn_log, cn_lab, cn_sigs = A['logits'], A['labels'], A['sigs']
dsn_log_raw, dsn_lab_raw, dsn_paths = B['logits'], B['labels'], B['paths']

dsn_by_sig  = {sig_of(p): dsn_log_raw[i] for i, p in enumerate(dsn_paths)}
dsn_lab_sig = {sig_of(p): int(dsn_lab_raw[i]) for i, p in enumerate(dsn_paths)}
dsn_log = np.stack([dsn_by_sig[s] for s in cn_sigs])
dsn_lab = np.array([dsn_lab_sig[s] for s in cn_sigs])

print(f'cnxxl solo (dataset lab): {(cn_log.argmax(1)==cn_lab).mean()*100:.2f}%')
print(f'dsn   solo (dataset lab): {(dsn_log.argmax(1)==cn_lab).mean()*100:.2f}%')
print(f'dsn   solo (dsn-lab)    : {(dsn_log.argmax(1)==dsn_lab).mean()*100:.2f}%')
print(f'label disagreement cn vs dsn: {int((cn_lab!=dsn_lab).sum())}/482')

sm_cn  = softmax(cn_log)
sm_dsn = softmax(dsn_log)

print('\n=== honest fusion (dataset labels) ===')
print(f'softmax uniform:                 {(((sm_cn+sm_dsn)/2).argmax(1)==cn_lab).mean()*100:.2f}%')
print(f'logit uniform:                   {(((cn_log+dsn_log)/2).argmax(1)==cn_lab).mean()*100:.2f}%')

# Per memory: train-calibrated DSN scale. We don't have train softmaxes yet,
# so sweep on TEST as diagnostic only -- do NOT pick best as the result.
print('\n=== DSN temperature sweep (scale logits before softmax) ===')
print('   sm_cn solo:', f'{(sm_cn.argmax(1)==cn_lab).mean()*100:.2f}%')
best = (0, 0)
for T in [3, 5, 7, 9, 9.5, 10, 12, 15, 20, 25, 30, 40, 50]:
    sm_dsn_T = softmax(dsn_log * T)
    fused = (sm_cn + sm_dsn_T) / 2
    acc = (fused.argmax(1)==cn_lab).mean()*100
    if acc > best[1]: best = (T, acc)
    print(f'  T={T:>5.1f}  dsn-top1={sm_dsn_T.max(1).mean():.3f}  fused acc={acc:.2f}%')
print(f'best (test-tuned): T={best[0]} acc={best[1]:.2f}%')

print('\n=== fixed T=9.5 (per memory), weight sweep ===')
sm_dsn_T = softmax(dsn_log * 9.5)
for w in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
    fused = (1-w) * sm_cn + w * sm_dsn_T
    print(f'  w_dsn={w:>4.2f}  acc={(fused.argmax(1)==cn_lab).mean()*100:.2f}%')

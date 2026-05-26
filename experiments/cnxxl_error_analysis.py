"""Characterize what cnxxl gets wrong on NvGesture test set.

Outputs:
  - per-class error rate
  - confusion matrix restricted to wrong samples (predicted vs true)
  - confidence distribution: max softmax on right vs wrong
  - confidence ranking of errors (the most-wrong / least-wrong)
  - simple per-sample point-cloud stats (extent, num-points, motion magnitude)
    to see if errors correlate with input characteristics
"""
import os, sys
import numpy as np

os.chdir('/notebooks/Anemon/experiments')


def softmax(x):
    x = x - x.max(-1, keepdims=True); e = np.exp(x); return e / e.sum(-1, keepdims=True)


z = np.load('./work_dir/cn_xxl_quat_head/test_logits.npz', allow_pickle=True)
logits = z['logits']; labels = z['labels']
preds = logits.argmax(1)
sm = softmax(logits)
maxp = sm.max(1)
true_p = sm[np.arange(len(labels)), labels]  # probability assigned to true class

err = preds != labels
err_idx = np.where(err)[0]
print(f'cnxxl test acc: {(preds == labels).mean()*100:.2f}%  ({(~err).sum()}/{len(labels)})')
print(f'errors: {err.sum()}\n')

# 1) Per-class error rate
print('==== Per-class error rate (sorted by rate) ====')
rows = []
for c in range(int(labels.max()) + 1):
    mask = labels == c
    n = mask.sum()
    e = (mask & err).sum()
    rows.append((c, n, e, e / max(n, 1)))
for c, n, e, r in sorted(rows, key=lambda x: -x[3]):
    if e > 0:
        print(f'  class {c:2d}:  {e}/{n}  = {r*100:5.1f}%')

# 2) Confusion: predicted class for each error
print('\n==== Predicted-vs-True for errors (only most common confusions) ====')
from collections import Counter
conf = Counter()
for i in err_idx:
    conf[(int(labels[i]), int(preds[i]))] += 1
for (t, p), n in sorted(conf.items(), key=lambda x: -x[1])[:15]:
    print(f'  true={t:2d}  pred={p:2d}  n={n}  true_prob={sm[err_idx, t][labels[err_idx] == t].mean():.3f}')

# 3) Confidence distribution
print('\n==== Confidence stats ====')
print(f'  correct  max-softmax mean={maxp[~err].mean():.3f}  median={np.median(maxp[~err]):.3f}')
print(f'  error    max-softmax mean={maxp[err].mean():.3f}  median={np.median(maxp[err]):.3f}')
print(f'  correct  true_prob   mean={true_p[~err].mean():.3f}')
print(f'  error    true_prob   mean={true_p[err].mean():.3f}  (true class barely gets prob mass)')

# 4) Top-5 hit on errors? maybe true class is 2nd/3rd most likely
top5 = np.argsort(-logits, 1)[:, :5]
top5_hit_on_err = (top5[err_idx] == labels[err_idx, None]).any(1).mean()
print(f'  top-5 hits true class on errors: {top5_hit_on_err*100:.1f}%')
# Which top-k contains the true class?
ranks = []
for i in err_idx:
    rank = np.where(np.argsort(-logits[i]) == labels[i])[0][0]
    ranks.append(rank)
ranks = np.array(ranks)
print(f'  rank of true class on errors: median={int(np.median(ranks))}, mean={ranks.mean():.1f}')

# 5) The 5 most-confident wrong predictions
print('\n==== Top 10 most-confident errors (model is very wrong) ====')
err_with_conf = [(i, maxp[i], labels[i], preds[i], true_p[i]) for i in err_idx]
for i, c, t, p, tp in sorted(err_with_conf, key=lambda x: -x[1])[:10]:
    print(f'  sample {i:3d}  true={t:2d}  pred={p:2d}  pred_conf={c:.3f}  true_prob={tp:.3f}')

# 6) The 5 least-confident wrong predictions (model knows it's uncertain)
print('\n==== Top 10 least-confident errors (model is uncertain) ====')
for i, c, t, p, tp in sorted(err_with_conf, key=lambda x: x[1])[:10]:
    print(f'  sample {i:3d}  true={t:2d}  pred={p:2d}  pred_conf={c:.3f}  true_prob={tp:.3f}')

# 7) Try to load point cloud and compute per-sample stats
print('\n==== Sample-level point-cloud stats on errors ====')
try:
    sys.path.insert(0, '/notebooks/Anemon/experiments')
    from nvidia_dataloader import NvidiaLoader
    ds = NvidiaLoader(phase='test', framerate=32)
    print(f'  loaded test set: {len(ds)} samples')

    def stats(idx):
        pts, lbl, _ = ds[idx]
        # pts shape: (T, N, 4) — first 3 are xyz
        pts = pts[..., :3]
        T = pts.shape[0]
        # extent (bbox volume cube-root)
        extent = (pts.max(axis=(0, 1)) - pts.min(axis=(0, 1)))
        bbox_vol = float(np.prod(extent))
        # motion magnitude: avg per-point std over time (correspondence-free)
        # use centroid drift
        cent = pts.mean(axis=1)  # (T, 3)
        cent_disp = np.linalg.norm(cent[-1] - cent[0])
        # spread variance: how spread points are
        spread = float(np.linalg.norm(pts.reshape(-1, 3).std(0)))
        # n_unique_points (non-trivial?)
        return dict(bbox_extent=float(np.linalg.norm(extent)), bbox_vol=bbox_vol,
                    cent_disp=float(cent_disp), spread=spread, T=T)

    err_stats = [stats(i) for i in err_idx]
    right_stats = [stats(i) for i in np.where(~err)[0][:80]]

    def report(label, S):
        print(f'  {label} (n={len(S)})')
        for k in S[0].keys():
            vals = np.array([s[k] for s in S])
            print(f'    {k:13s}: mean={vals.mean():.3f}  median={np.median(vals):.3f}  std={vals.std():.3f}')

    report('errors', err_stats)
    report('correct (sample of 80)', right_stats)
except Exception as e:
    print(f'  (skipping sample-level stats: {e})')

"""Confidence-gated TTA on cnxxl.

For each test sample:
  - run cnxxl at K (angle, speed) variants
  - select the variant where cnxxl has the highest max-softmax
  - use that variant's prediction

If invariance is the limiting factor, this should beat baseline 91.29%.
Also reports confidence-WEIGHTED averaging as an alternative aggregation.
"""
import os, sys, math
import numpy as np
import torch
import yaml

sys.path.insert(0, '/notebooks/Anemon/experiments')
os.chdir('/notebooks/Anemon/experiments')

from nvidia_dataloader import NvidiaLoader
from models.motion_cleanest_quat_head import MotionCleanestLinXLQuatHead
from torch.utils.data import DataLoader

CKPT = './work_dir/cn_xxl_quat_head/best_model.pt'
CFG_PATH = './cn_xxl_quat_head.yaml'

# Smaller grid focused near baseline.
ANGLES_DEG = [-15, -5, 0, 5, 15]
SPEEDS = [0.9, 1.0, 1.1]


def rotate_z(c, deg):
    th = math.radians(deg); cs, sn = math.cos(th), math.sin(th)
    out = c.clone()
    out[..., 0] = cs * c[..., 0] - sn * c[..., 1]
    out[..., 1] = sn * c[..., 0] + cs * c[..., 1]
    return out


def speed_resample(pts, speed):
    if abs(speed - 1.0) < 1e-6: return pts
    T = pts.shape[1]
    idx = (torch.arange(T, device=pts.device, dtype=torch.float32) * speed).round().long().clamp(0, T - 1)
    return pts[:, idx]


with open(CFG_PATH) as f:
    cfg = yaml.safe_load(f)
model = MotionCleanestLinXLQuatHead(**cfg['model_args']).cuda()
sd = torch.load(CKPT, map_location='cpu')
sd = sd.get('model_state_dict', sd.get('model', sd))
sd = {(k[7:] if k.startswith('module.') else k): v for k, v in sd.items()}
model.load_state_dict(sd, strict=False)
model.eval()

ds = NvidiaLoader(phase='test', framerate=32)
loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=4)
print(f'N={len(ds)}, {len(ANGLES_DEG)}x{len(SPEEDS)}={len(ANGLES_DEG)*len(SPEEDS)} variants per sample')

# Run all variants on all samples; collect softmax per variant.
all_softmax = []   # list over batches of (V, B, C)
all_labels = []
variant_keys = [(a, s) for a in ANGLES_DEG for s in SPEEDS]

with torch.no_grad():
    for batch in loader:
        pts, lbl, _ = batch
        pts = pts.cuda(non_blocking=True).float()
        lbl = lbl.numpy() if hasattr(lbl, 'numpy') else np.asarray(lbl)
        all_labels.append(lbl)
        v_softmax = []
        for ang, sp in variant_keys:
            cur = pts.clone()
            if ang != 0: cur[..., :3] = rotate_z(pts[..., :3], ang)
            if abs(sp - 1.0) > 1e-6: cur = speed_resample(cur, sp)
            sm = model(cur).softmax(-1).cpu().numpy()
            v_softmax.append(sm)
        all_softmax.append(np.stack(v_softmax))

all_softmax = np.concatenate(all_softmax, axis=1)   # (V, N, C)
all_labels = np.concatenate(all_labels)             # (N,)
V, N, C = all_softmax.shape
print(f'softmax tensor: {all_softmax.shape}')

baseline_idx = variant_keys.index((0, 1.0))
baseline_sm = all_softmax[baseline_idx]
baseline_pred = baseline_sm.argmax(1)
baseline_acc = (baseline_pred == all_labels).mean() * 100
print(f'\nbaseline (0, 1.0): {baseline_acc:.2f}%')

# Strategy 1: per-sample, pick the variant with highest max-softmax confidence
max_conf_per_variant = all_softmax.max(-1)          # (V, N)
best_variant_per_sample = max_conf_per_variant.argmax(0)  # (N,)
confgate_sm = all_softmax[best_variant_per_sample, np.arange(N)]   # (N, C)
confgate_pred = confgate_sm.argmax(1)
confgate_acc = (confgate_pred == all_labels).mean() * 100
print(f'confidence-gated (argmax variant):  {confgate_acc:.2f}%  ({confgate_acc - baseline_acc:+.2f})')

# Strategy 2: confidence-weighted softmax average
weights = max_conf_per_variant                                  # (V, N)
weights = weights / weights.sum(0, keepdims=True)
confweighted_sm = (weights[..., None] * all_softmax).sum(0)     # (N, C)
confweighted_pred = confweighted_sm.argmax(1)
confweighted_acc = (confweighted_pred == all_labels).mean() * 100
print(f'confidence-weighted softmax avg:    {confweighted_acc:.2f}%  ({confweighted_acc - baseline_acc:+.2f})')

# Strategy 3: per-class confidence-weighted -- weight each variant by its
# softmax mass on the class it's predicting (sharpened by exp)
sharp = np.exp(5.0 * max_conf_per_variant)
sharp = sharp / sharp.sum(0, keepdims=True)
sharp_sm = (sharp[..., None] * all_softmax).sum(0)
sharp_pred = sharp_sm.argmax(1)
sharp_acc = (sharp_pred == all_labels).mean() * 100
print(f'sharpened conf-weight (T=5):        {sharp_acc:.2f}%  ({sharp_acc - baseline_acc:+.2f})')

# Strategy 4: take the variant where the model is most CONSISTENT with neighbours
# i.e., variant whose argmax matches the most other variants
preds_per_variant = all_softmax.argmax(-1)   # (V, N)
consistency = np.zeros((V, N), dtype=np.int32)
for v in range(V):
    consistency[v] = (preds_per_variant == preds_per_variant[v:v+1]).sum(0)
best_consistent_variant = consistency.argmax(0)
consistent_pred = preds_per_variant[best_consistent_variant, np.arange(N)]
consistent_acc = (consistent_pred == all_labels).mean() * 100
print(f'most-consistent-variant prediction: {consistent_acc:.2f}%  ({consistent_acc - baseline_acc:+.2f})')

# Strategy 5: combine -- confidence-gated, but only if it changed the prediction
# AND its confidence is above some absolute threshold (gates flipping spurious)
for thresh in [0.3, 0.4, 0.5, 0.6]:
    cg = baseline_pred.copy()
    flipped = 0
    for i in range(N):
        bv = best_variant_per_sample[i]
        if bv == baseline_idx: continue
        new_p = preds_per_variant[bv, i]
        new_c = max_conf_per_variant[bv, i]
        old_c = max_conf_per_variant[baseline_idx, i]
        if new_c > thresh and new_c > old_c + 0.1:
            cg[i] = new_p
            flipped += 1
    acc = (cg == all_labels).mean() * 100
    print(f'confgate threshold {thresh}, +0.1 margin: {acc:.2f}%  ({acc - baseline_acc:+.2f})  (flipped {flipped})')

# How many baseline-correct samples does each strategy preserve / break?
def report(name, pred):
    fixed = ((baseline_pred != all_labels) & (pred == all_labels)).sum()
    broken = ((baseline_pred == all_labels) & (pred != all_labels)).sum()
    print(f'  {name:35s} fixed={fixed:3d}  broken={broken:3d}  net={fixed-broken:+3d}')

print('\n==== Fixes vs breakages ====')
report('confidence-gated (argmax variant)', confgate_pred)
report('confidence-weighted softmax avg', confweighted_pred)
report('sharpened conf-weight T=5', sharp_pred)
report('most-consistent-variant', consistent_pred)

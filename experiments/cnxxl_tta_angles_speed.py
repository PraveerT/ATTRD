"""Test-time augmentation on cnxxl: rotate every test sample around z-axis at
several angles AND temporally re-sample at different speeds. Originals are NEVER
modified -- transforms are applied on-the-fly to GPU tensors.

For each (angle, speed) variant we record the predicted class for every test
sample. Then we report:

  1. Prediction stability per sample (does cnxxl change its mind under
     small perturbation?)
  2. Per-variant test accuracy
  3. TTA ensemble accuracy (majority vote + softmax avg over all variants)
  4. The featured sample (sample 363, true=18, pred=9) followed across
     variants to see if any transform flips the prediction toward the truth.
"""
import os, sys, math
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import yaml

sys.path.insert(0, '/notebooks/Anemon/experiments')
os.chdir('/notebooks/Anemon/experiments')

from nvidia_dataloader import NvidiaLoader
from models.motion_cleanest_quat_head import MotionCleanestLinXLQuatHead

CKPT = './work_dir/cn_xxl_quat_head/best_model.pt'
CFG_PATH = './cn_xxl_quat_head.yaml'

FEATURED = 363   # sample index; cnxxl true=18, pred=9, conf=0.552

# Angle and speed grids (kept small enough to finish in a few minutes).
ANGLES_DEG = [-30, -20, -10, 0, 10, 20, 30]    # rotation around z (depth axis)
SPEEDS = [0.5, 0.75, 1.0, 1.25, 1.5]            # 1.0 = no change


def rotate_z(coords, angle_deg):
    """coords: (..., 3 or 4). Rotate xyz around z by angle_deg.
    4th channel (intensity) is preserved.
    """
    th = math.radians(angle_deg)
    c, s = math.cos(th), math.sin(th)
    out = coords.clone()
    x = coords[..., 0]; y = coords[..., 1]
    out[..., 0] = c * x - s * y
    out[..., 1] = s * x + c * y
    return out


def speed_resample(pts, speed):
    """pts: (B, T, N, 4). speed > 1: stretch (slow down) -> use fewer frames.
    speed < 1: compress (speed up) -> repeat frames.
    Always output T_orig frames by nearest-neighbour over the warped index.
    """
    B, T, N, C = pts.shape
    if abs(speed - 1.0) < 1e-6:
        return pts
    # New time indices: t' = t * speed for t in [0, T-1]
    t_orig = torch.arange(T, device=pts.device, dtype=torch.float32)
    t_warped = t_orig * speed
    idx = t_warped.round().long().clamp(0, T - 1)
    return pts[:, idx]


print('==== loading cnxxl model from best_model.pt ====')
with open(CFG_PATH) as f:
    cfg = yaml.safe_load(f)
m_args = cfg['model_args']
model = MotionCleanestLinXLQuatHead(**m_args).cuda()
ckpt = torch.load(CKPT, map_location='cpu')
sd = ckpt.get('model_state_dict', ckpt.get('model', ckpt))
sd = {(k[7:] if k.startswith('module.') else k): v for k, v in sd.items()}
res = model.load_state_dict(sd, strict=False)
print(f'  ckpt ep={ckpt.get("epoch")} missing={len(res.missing_keys)} unexpected={len(res.unexpected_keys)}')
model.eval()

print('\n==== loading test set ====')
ds = NvidiaLoader(phase='test', framerate=32)
loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=4)
print(f'  N={len(ds)} samples')


# Pre-flight: featured sample
print(f'\n==== Featured sample: idx={FEATURED} (cnxxl true=18 pred=9) ====')
featured_pts, featured_lbl, _ = ds[FEATURED]
print(f'  raw shape: {tuple(featured_pts.shape)}, label={int(featured_lbl)}')


# ---- TTA grid ----
all_labels = []
# per_variant_preds: dict variant_name -> list of preds
per_variant_preds = {}
# per_variant_softmax: variant_name -> (N, 25) array
per_variant_softmax = {}
featured_track = []   # list of (angle, speed, pred_class, softmax_vec)

with torch.no_grad():
    for batch in loader:
        pts_batch, lbl_batch, _ = batch
        pts_batch = pts_batch.cuda(non_blocking=True).float()
        lbl_batch = lbl_batch.numpy() if hasattr(lbl_batch, 'numpy') else np.asarray(lbl_batch)
        all_labels.append(lbl_batch)
        for ang in ANGLES_DEG:
            for sp in SPEEDS:
                # rotate xyz, leave intensity
                if ang != 0:
                    rotated = pts_batch.clone()
                    rotated[..., :3] = rotate_z(pts_batch[..., :3], ang)
                else:
                    rotated = pts_batch
                # temporal resample
                warped = speed_resample(rotated, sp) if abs(sp - 1.0) > 1e-6 else rotated
                logits = model(warped)
                sm = logits.softmax(-1).cpu().numpy()
                preds = sm.argmax(1)
                key = f'a{ang:+04d}_s{sp:.2f}'
                per_variant_preds.setdefault(key, []).append(preds)
                per_variant_softmax.setdefault(key, []).append(sm)

all_labels = np.concatenate(all_labels)
N = len(all_labels)
for k in per_variant_preds:
    per_variant_preds[k] = np.concatenate(per_variant_preds[k])
    per_variant_softmax[k] = np.concatenate(per_variant_softmax[k])

baseline_key = 'a+000_s1.00'
baseline_pred = per_variant_preds[baseline_key]
baseline_acc = (baseline_pred == all_labels).mean() * 100
print(f'\n==== Baseline (angle=0, speed=1.0) accuracy: {baseline_acc:.2f}% ====')


# 1) Per-variant accuracy table
print(f'\n==== Per-variant accuracy ({len(ANGLES_DEG)*len(SPEEDS)} variants) ====')
print(f'  angle      ' + '  '.join(f'sp{s:.2f}' for s in SPEEDS))
for ang in ANGLES_DEG:
    row = f'  {ang:+4d}deg   '
    for sp in SPEEDS:
        key = f'a{ang:+04d}_s{sp:.2f}'
        acc = (per_variant_preds[key] == all_labels).mean() * 100
        row += f'  {acc:5.2f}'
    print(row)

# 2) Stability: how often does each sample's predicted class change across variants?
print(f'\n==== Prediction stability across all {len(per_variant_preds)} variants ====')
all_preds_stack = np.stack(list(per_variant_preds.values()))     # (V, N)
unique_per_sample = np.array([len(set(all_preds_stack[:, i])) for i in range(N)])
print(f'  samples with stable prediction (1 unique class):  {(unique_per_sample == 1).sum()}/{N}  ({(unique_per_sample == 1).mean()*100:.1f}%)')
print(f'  samples with 2 different classes:                 {(unique_per_sample == 2).sum()}/{N}')
print(f'  samples with 3+ different classes:                {(unique_per_sample >= 3).sum()}/{N}')

baseline_err = (baseline_pred != all_labels)
print(f'  on baseline-correct samples: stability mean: {unique_per_sample[~baseline_err].mean():.2f} unique classes')
print(f'  on baseline-wrong samples:   stability mean: {unique_per_sample[baseline_err].mean():.2f} unique classes')

# 3) TTA ensemble: softmax avg + majority vote
sm_avg = np.stack(list(per_variant_softmax.values())).mean(0)
tta_pred = sm_avg.argmax(1)
tta_acc = (tta_pred == all_labels).mean() * 100
print(f'\n==== TTA softmax-average ====')
print(f'  TTA accuracy: {tta_acc:.2f}%   (baseline {baseline_acc:.2f}, delta {tta_acc - baseline_acc:+.2f})')

# Majority vote
from scipy.stats import mode
vote_pred = mode(all_preds_stack, axis=0).mode.squeeze() if hasattr(mode(all_preds_stack, axis=0), 'mode') else np.array([np.bincount(all_preds_stack[:, i]).argmax() for i in range(N)])
vote_pred = np.array([np.bincount(all_preds_stack[:, i]).argmax() for i in range(N)])  # robust
vote_acc = (vote_pred == all_labels).mean() * 100
print(f'  majority-vote accuracy: {vote_acc:.2f}%  (baseline {baseline_acc:.2f}, delta {vote_acc - baseline_acc:+.2f})')

# 4) Rescued and lost samples
baseline_correct = (baseline_pred == all_labels)
tta_correct = (tta_pred == all_labels)
rescued = (~baseline_correct & tta_correct).sum()
lost = (baseline_correct & ~tta_correct).sum()
print(f'  rescues (baseline wrong -> TTA right): {rescued}')
print(f'  losses  (baseline right -> TTA wrong): {lost}')

# 5) Featured sample tracking
print(f'\n==== Featured sample {FEATURED} (true=18) prediction grid ====')
print(f'  angle      ' + '  '.join(f'sp{s:.2f}' for s in SPEEDS))
for ang in ANGLES_DEG:
    row = f'  {ang:+4d}deg   '
    for sp in SPEEDS:
        key = f'a{ang:+04d}_s{sp:.2f}'
        p = per_variant_preds[key][FEATURED]
        sm = per_variant_softmax[key][FEATURED]
        conf = sm[p]
        true_p = sm[18]
        marker = '*' if p == 18 else ' '
        row += f'  {p:2d}({conf:.2f}){marker}'
    print(row)
print(f'  TTA softmax-avg prediction for sample {FEATURED}: {tta_pred[FEATURED]}  (true=18, baseline=9)')

# 6) Did TTA rescue any of cnxxl's 42 baseline errors?
baseline_err_idx = np.where(~baseline_correct)[0]
print(f'\n==== TTA effect on the 42 baseline errors ====')
fixed = baseline_err_idx[tta_correct[baseline_err_idx]]
still_wrong = baseline_err_idx[~tta_correct[baseline_err_idx]]
print(f'  rescued: {len(fixed)}/{len(baseline_err_idx)} errors fixed by TTA')
if len(fixed) > 0:
    print(f'  fixed sample indices: {list(fixed[:20])}')

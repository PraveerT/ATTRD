"""TTA on the 42 cnxxl baseline errors using **centroid-centered rotation**
(not buggy world-origin rotation). Re-evaluates whether the model's
fragility under rotation was a real invariance failure or an artifact of
rotation + translation drift.
"""
import os, sys, math
import numpy as np
import torch
import yaml

sys.path.insert(0, '/notebooks/Anemon/experiments')
os.chdir('/notebooks/Anemon/experiments')

from nvidia_dataloader import NvidiaLoader
from models.motion_cleanest_quat_head import MotionCleanestLinXLQuatHead

CKPT = './work_dir/cn_xxl_quat_head/best_model.pt'
CFG_PATH = './cn_xxl_quat_head.yaml'

ANGLES_DEG = [-30, -20, -10, 0, 10, 20, 30]
SPEEDS = [0.5, 0.75, 1.0, 1.25, 1.5]


def rotate_z_centered(coords, deg):
    """coords: (1, T, N, 4) tensor. Rotate xyz around the per-sample
    xy-centroid (not world origin). z is the rotation axis.
    """
    if coords.dim() != 4:
        raise ValueError(f'unexpected coords shape {coords.shape}')
    if abs(deg) < 1e-6:
        return coords
    th = math.radians(deg)
    cs, sn = math.cos(th), math.sin(th)
    cx = coords[..., 0].mean(dim=(1, 2), keepdim=True)
    cy = coords[..., 1].mean(dim=(1, 2), keepdim=True)
    x = coords[..., 0] - cx
    y = coords[..., 1] - cy
    out = coords.clone()
    out[..., 0] = cs * x - sn * y + cx
    out[..., 1] = sn * x + cs * y + cy
    return out


def speed_resample(pts, speed):
    if abs(speed - 1.0) < 1e-6:
        return pts
    T = pts.shape[1]
    idx = (torch.arange(T, device=pts.device, dtype=torch.float32) * speed).round().long().clamp(0, T - 1)
    return pts[:, idx]


# Load model
with open(CFG_PATH) as f:
    cfg = yaml.safe_load(f)
model = MotionCleanestLinXLQuatHead(**cfg['model_args']).cuda()
sd = torch.load(CKPT, map_location='cpu')
sd = sd.get('model_state_dict', sd.get('model', sd))
sd = {(k[7:] if k.startswith('module.') else k): v for k, v in sd.items()}
model.load_state_dict(sd, strict=False)
model.eval()

# Load test set + baseline preds
ds = NvidiaLoader(phase='test', framerate=32)
z = np.load('./work_dir/cn_xxl_quat_head/test_logits.npz', allow_pickle=True)
labels = z['labels']
baseline_preds = z['logits'].argmax(1)
err_idx = np.where(baseline_preds != labels)[0]
print(f'baseline acc: {(baseline_preds == labels).mean()*100:.2f}%  errors: {len(err_idx)}')
print(f'rotation: centroid-centered (FIXED).  speed: temporal subsample.\n')


# Also compute global invariance probe with centered rotation (small fast sweep)
print('==== Global invariance probe (all 482 samples, centroid-centered rotation) ====')
all_pts_correct = {}
loader_batch = 16
all_pred_at = {}
with torch.no_grad():
    for ang in [-30, -15, -5, 0, 5, 15, 30]:
        ok = 0; tot = 0
        for i in range(0, len(ds), loader_batch):
            chunk = [ds[j] for j in range(i, min(i + loader_batch, len(ds)))]
            pts = torch.from_numpy(np.stack([c[0].numpy() for c in chunk])).cuda().float()
            lbls = np.array([c[1] for c in chunk])
            cur = pts.clone()
            cur = rotate_z_centered(pts, ang)
            pred = model(cur).argmax(-1).cpu().numpy()
            ok += (pred == lbls).sum()
            tot += len(lbls)
        print(f'  angle {ang:+d}: {ok/tot*100:.2f}%')


# Run all 42 errors x full grid
result = {}
softmax_grid = {}
for i in err_idx:
    pts, lbl, _ = ds[i]
    true_class = int(lbl)
    pts = torch.as_tensor(np.asarray(pts)).unsqueeze(0).cuda().float()
    pred_grid = np.zeros((len(ANGLES_DEG), len(SPEEDS)), dtype=int)
    true_p_grid = np.zeros((len(ANGLES_DEG), len(SPEEDS)))
    with torch.no_grad():
        for a_i, ang in enumerate(ANGLES_DEG):
            for s_i, sp in enumerate(SPEEDS):
                cur = pts.clone()
                if ang != 0:
                    cur = rotate_z_centered(pts, ang)
                if abs(sp - 1.0) > 1e-6:
                    cur = speed_resample(cur, sp)
                logits = model(cur)
                sm = logits.softmax(-1)[0].cpu().numpy()
                pred_grid[a_i, s_i] = int(sm.argmax())
                true_p_grid[a_i, s_i] = float(sm[true_class])
    result[i] = pred_grid
    softmax_grid[i] = true_p_grid

print('\n==== Summary: errors fixable by AT LEAST ONE (angle, speed) variant ====')
fixable = 0
total_flip_count = 0
for i in err_idx:
    pg = result[i]
    flips = (pg == labels[i]).sum()
    total_flip_count += flips
    if flips > 0:
        fixable += 1
print(f'  fixable: {fixable}/{len(err_idx)}')
print(f'  mean flips-to-true across errors: {total_flip_count / len(err_idx):.1f} (out of 35 variants)')

print('\n==== Per-variant rescue count (out of 42 baseline errors) ====')
print('  angle      ' + '  '.join(f'sp{s:.2f}' for s in SPEEDS))
for a_i, ang in enumerate(ANGLES_DEG):
    row = f'  {ang:+4d}deg   '
    for s_i, _ in enumerate(SPEEDS):
        fixes = sum(1 for i in err_idx if result[i][a_i, s_i] == labels[i])
        row += f'  {fixes:3d} '
    print(row)

# Compare with previous (buggy) result
print('\n==== Per-sample detail for fixable errors ====')
for i in err_idx:
    pg = result[i]
    flips = (pg == labels[i]).sum()
    if flips == 0:
        continue
    print(f'\n sample {i:3d} true={labels[i]:2d} baseline_pred={baseline_preds[i]:2d}  flips={flips}/35')
    tp = softmax_grid[i]
    print('  angle      ' + '  '.join(f'sp{s:.2f}' for s in SPEEDS))
    for a_i, ang in enumerate(ANGLES_DEG):
        row = f'  {ang:+4d}deg   '
        for s_i, _ in enumerate(SPEEDS):
            p = pg[a_i, s_i]
            mark = '*' if p == labels[i] else ' '
            row += f'  {p:2d}({tp[a_i, s_i]:.2f}){mark}'
        print(row)

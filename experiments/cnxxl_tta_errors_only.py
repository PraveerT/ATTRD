"""Take ONLY the 42 baseline-wrong samples and re-run cnxxl across the
angle x speed grid. For each error, find which variants (if any) flip the
prediction to the true class.
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
labels = z['labels']; baseline_preds = z['logits'].argmax(1)
err_idx = np.where(baseline_preds != labels)[0]
print(f'baseline acc: {(baseline_preds == labels).mean()*100:.2f}%   errors: {len(err_idx)}')
print(f'error sample indices: {list(err_idx)}\n')

# For each error sample, run all 35 variants
result = {}   # idx -> matrix (len(ANGLES), len(SPEEDS)) of predicted class
softmax_grid = {}  # idx -> matrix with softmax of true class
for i in err_idx:
    pts, lbl, _ = ds[i]
    true_class = int(lbl)
    pts = torch.as_tensor(np.asarray(pts)).unsqueeze(0).cuda().float()  # (1, T, N, 4)
    pred_grid = np.zeros((len(ANGLES_DEG), len(SPEEDS)), dtype=int)
    true_p_grid = np.zeros((len(ANGLES_DEG), len(SPEEDS)))
    pred_conf_grid = np.zeros((len(ANGLES_DEG), len(SPEEDS)))
    with torch.no_grad():
        for a_i, ang in enumerate(ANGLES_DEG):
            for s_i, sp in enumerate(SPEEDS):
                cur = pts.clone()
                if ang != 0:
                    cur[..., :3] = rotate_z(pts[..., :3], ang)
                if abs(sp - 1.0) > 1e-6:
                    cur = speed_resample(cur, sp)
                logits = model(cur)
                sm = logits.softmax(-1)[0].cpu().numpy()
                pred_grid[a_i, s_i] = int(sm.argmax())
                true_p_grid[a_i, s_i] = float(sm[true_class])
                pred_conf_grid[a_i, s_i] = float(sm.max())
    result[i] = pred_grid
    softmax_grid[i] = (true_p_grid, pred_conf_grid)

# Summary
print('==== Summary: how many variants flip each error to correct ====')
fixable = 0; partial = 0
for i in err_idx:
    pg = result[i]
    flips = (pg == labels[i]).sum()
    if flips > 0: fixable += 1
    if flips >= 5: partial += 1
    print(f'  sample {i:3d}  true={labels[i]:2d}  baseline_pred={baseline_preds[i]:2d}  '
          f'variants_correct={flips}/{pg.size}')
print(f'\n  errors fixable by AT LEAST ONE variant: {fixable}/{len(err_idx)}')
print(f'  errors fixable by 5+ variants:           {partial}/{len(err_idx)}')

# Show per-sample detail for fixable errors
print('\n==== Detail for errors fixable by some variant ====')
for i in err_idx:
    pg = result[i]
    flips = (pg == labels[i]).sum()
    if flips == 0: continue
    tp, pc = softmax_grid[i]
    print(f'\nsample {i:3d}  true={labels[i]}  baseline={baseline_preds[i]}  '
          f'(true_p baseline={tp[3,2]:.3f})   {flips}/{pg.size} variants flip to true class')
    print('  pred grid (* = flipped to true):')
    print('  angle      ' + '  '.join(f'sp{s:.2f}' for s in SPEEDS))
    for a_i, ang in enumerate(ANGLES_DEG):
        row = f'  {ang:+4d}deg   '
        for s_i, _ in enumerate(SPEEDS):
            p = pg[a_i, s_i]
            mark = '*' if p == labels[i] else ' '
            row += f'  {p:2d}({tp[a_i, s_i]:.2f}){mark}'
        print(row)

# For each variant: how many of the 42 errors does it fix?
print('\n==== Per-variant rescue count (out of 42 baseline errors) ====')
print('  angle      ' + '  '.join(f'sp{s:.2f}' for s in SPEEDS))
for a_i, ang in enumerate(ANGLES_DEG):
    row = f'  {ang:+4d}deg   '
    for s_i, _ in enumerate(SPEEDS):
        fixes = sum(1 for i in err_idx if result[i][a_i, s_i] == labels[i])
        row += f'  {fixes:3d} '
    print(row)

"""TTA on cnxxl's 42 baseline errors with **true 3D rotation** pipeline:
  1. un-normalize z-score
  2. un-project pixel(u,v,d) -> physical (X,Y,Z) using camera intrinsics
  3. rotate in 3D around per-sample 3D centroid
  4. re-project (X,Y,Z) -> (u,v,d)
  5. re-normalize

This replaces the earlier image-plane rotation that was masquerading as
"z-rotation invariance" because the loader's 4 channels are
(pixel_row, pixel_col, depth, time), not a 3D point cloud.
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

# Camera intrinsics used by nvidia_process.py / uvd2xyz_sherc.
FX, FY = 463.889, 463.889
CX, CY = 320.0, 240.0   # principal point in (col, row)

# Normalization stats (from nvidia_dataset_stats.npy).
X_MEAN, X_STD = 143.5320921018914, 37.762996875345834
Y_MEAN, Y_STD = 197.01543121736293, 52.412147141177215
Z_MEAN, Z_STD = 131.22534211559645, 34.754814250125044
T_MEAN, T_STD = 15.438825881201044, 9.25209971363254


def rotate_3d_z(coords, theta_deg):
    """coords: (B, T, N, 4) normalized (pixel_row, pixel_col, depth, time).
    Rotate by theta_deg around the 3D camera-Z axis (depth direction)
    through the per-sample 3D centroid.

      pixel_row (ch0)  -> physical Y  (vertical)
      pixel_col (ch1)  -> physical X  (horizontal)
      depth     (ch2)  -> physical Z  (into scene)

    Rotation around Z mixes physical X and Y, then re-projects back to
    pixel coords through the pinhole model.
    """
    if abs(theta_deg) < 1e-6:
        return coords

    # 1. Un-normalize
    row = coords[..., 0] * X_STD + X_MEAN     # pixel row    (Nv: 0..479)
    col = coords[..., 1] * Y_STD + Y_MEAN     # pixel col    (Nv: 0..639)
    dep = coords[..., 2] * Z_STD + Z_MEAN     # depth value

    # 2. Un-project to 3D (per uvd2xyz_sherc convention)
    #    X = (col - CX) * d / FX      (physical horizontal)
    #    Y = (row - CY) * d / FY      (physical vertical)
    #    Z = d                         (depth)
    X = (col - CX) * dep / FX
    Y = (row - CY) * dep / FY
    Z = dep

    # 3. 3D centroid per sample (mean over T, N)
    Xc = X.mean(dim=(1, 2), keepdim=True)
    Yc = Y.mean(dim=(1, 2), keepdim=True)

    # 4. Rotate around Z axis through (Xc, Yc, anything-Z)
    th = math.radians(theta_deg)
    cs, sn = math.cos(th), math.sin(th)
    Xr = X - Xc
    Yr = Y - Yc
    X_new = cs * Xr - sn * Yr + Xc
    Y_new = sn * Xr + cs * Yr + Yc
    Z_new = Z

    # 5. Re-project to pixel coords
    eps = 1e-3
    col_new = X_new * FX / (Z_new + eps) + CX
    row_new = Y_new * FY / (Z_new + eps) + CY
    dep_new = Z_new

    # 6. Re-normalize
    out = coords.clone()
    out[..., 0] = (row_new - X_MEAN) / X_STD
    out[..., 1] = (col_new - Y_MEAN) / Y_STD
    out[..., 2] = (dep_new - Z_MEAN) / Z_STD
    return out


def speed_resample(pts, speed):
    if abs(speed - 1.0) < 1e-6:
        return pts
    T = pts.shape[1]
    idx = (torch.arange(T, device=pts.device, dtype=torch.float32) * speed).round().long().clamp(0, T - 1)
    return pts[:, idx]


# Sanity check the pipeline: 0deg should give identity.
def sanity_check(ds):
    pts, _, _ = ds[0]
    pts = torch.as_tensor(np.asarray(pts)).unsqueeze(0).float()
    out = rotate_3d_z(pts, 0.0)
    diff = (out - pts).abs().max().item()
    print(f'sanity: 0deg rotation max diff = {diff:.2e}  (should be ~0)')
    # Also verify 360deg gives identity (modulo floating point).
    out360 = rotate_3d_z(pts, 360.0)
    diff360 = (out360 - pts).abs().max().item()
    print(f'sanity: 360deg rotation max diff = {diff360:.2e}  (should be small)')


with open(CFG_PATH) as f:
    cfg = yaml.safe_load(f)
model = MotionCleanestLinXLQuatHead(**cfg['model_args']).cuda()
sd = torch.load(CKPT, map_location='cpu')
sd = sd.get('model_state_dict', sd.get('model', sd))
sd = {(k[7:] if k.startswith('module.') else k): v for k, v in sd.items()}
model.load_state_dict(sd, strict=False)
model.eval()

ds = NvidiaLoader(phase='test', framerate=32)
z = np.load('./work_dir/cn_xxl_quat_head/test_logits.npz', allow_pickle=True)
labels = z['labels']
baseline_preds = z['logits'].argmax(1)
err_idx = np.where(baseline_preds != labels)[0]
print(f'baseline acc: {(baseline_preds == labels).mean()*100:.2f}%  errors: {len(err_idx)}')
print(f'rotation: TRUE 3D z-rotation (un-project, rotate around 3D centroid, re-project)\n')

sanity_check(ds)


# Global invariance probe (all 482 samples).
print('\n==== Global invariance probe (482 samples, TRUE 3D z-rotation) ====')
loader_batch = 16
with torch.no_grad():
    for ang in [-30, -15, -5, 0, 5, 15, 30]:
        ok = 0; tot = 0
        for i in range(0, len(ds), loader_batch):
            chunk = [ds[j] for j in range(i, min(i + loader_batch, len(ds)))]
            pts = torch.from_numpy(np.stack([c[0].numpy() for c in chunk])).cuda().float()
            lbls = np.array([c[1] for c in chunk])
            cur = rotate_3d_z(pts, ang) if ang != 0 else pts
            pred = model(cur).argmax(-1).cpu().numpy()
            ok += (pred == lbls).sum()
            tot += len(lbls)
        print(f'  angle {ang:+d}: {ok/tot*100:.2f}%')


# TTA grid on the 42 errors.
ANGLES = [-30, -20, -10, 0, 10, 20, 30]
SPEEDS = [0.5, 0.75, 1.0, 1.25, 1.5]

result = {}
softmax_grid = {}
for i in err_idx:
    pts, lbl, _ = ds[i]
    true_class = int(lbl)
    pts = torch.as_tensor(np.asarray(pts)).unsqueeze(0).cuda().float()
    pg = np.zeros((len(ANGLES), len(SPEEDS)), dtype=int)
    tp = np.zeros((len(ANGLES), len(SPEEDS)))
    with torch.no_grad():
        for a_i, ang in enumerate(ANGLES):
            for s_i, sp in enumerate(SPEEDS):
                cur = rotate_3d_z(pts, ang) if ang != 0 else pts
                if abs(sp - 1.0) > 1e-6:
                    cur = speed_resample(cur, sp)
                sm = model(cur).softmax(-1)[0].cpu().numpy()
                pg[a_i, s_i] = int(sm.argmax())
                tp[a_i, s_i] = float(sm[true_class])
    result[i] = pg
    softmax_grid[i] = tp

# Summary
fixable = sum(1 for i in err_idx if (result[i] == labels[i]).any())
total_flips = sum((result[i] == labels[i]).sum() for i in err_idx)
print(f'\n==== Summary ====')
print(f'  errors fixable by AT LEAST ONE variant: {fixable}/{len(err_idx)}')
print(f'  mean flips-to-true across errors: {total_flips / len(err_idx):.1f} (of 35 variants)')

print('\n==== Per-variant rescue count (out of 42 baseline errors) ====')
print('  angle      ' + '  '.join(f'sp{s:.2f}' for s in SPEEDS))
for a_i, ang in enumerate(ANGLES):
    row = f'  {ang:+4d}deg   '
    for s_i, _ in enumerate(SPEEDS):
        fixes = sum(1 for i in err_idx if result[i][a_i, s_i] == labels[i])
        row += f'  {fixes:3d} '
    print(row)

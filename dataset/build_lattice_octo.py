"""
Build corresponded lattice + octonion sidecar files.

For each existing `<stub>_pts.npy` (shape (T, N, 8) — uvd-t + xyz-t), compute:
  - canonical Halton(2,3,5) lattice in [-1,1]^3 (512 points, fixed forever)
  - per-frame bbox of raw xyz
  - per-frame voxel occupancy (16^3 inside the static bbox of the sample)
  - per-frame lattice positions: warp canonical via per-frame bbox, then snap
    to the nearest occupied voxel center + sub-voxel jitter
  - forward-difference velocity (last frame = 0)
  - octonion = (1, x, y, z, 1, vx, vy, vz)

Writes `<stub>_octo.npy` shape (T, 512, 8) float32.
Existing `<stub>_pts.npy` is untouched.

Run from /notebooks/PMamba/dataset/ (same cwd convention as nvidia_process.py).
"""

import os
import re
import sys
import numpy as np
from multiprocessing import Pool, cpu_count

LATTICE_K = 512
VOXEL_N = 16
JITTER = 0.45  # fraction of voxel size used for sub-voxel offset


def halton(i, base):
    f, r, n = 1.0, 0.0, i
    while n > 0:
        f /= base
        r += f * (n % base)
        n //= base
    return r


def make_canonical_lattice(k=LATTICE_K):
    pts = np.zeros((k, 3), dtype=np.float32)
    for i in range(1, k + 1):
        pts[i - 1, 0] = 2 * halton(i, 2) - 1
        pts[i - 1, 1] = 2 * halton(i, 3) - 1
        pts[i - 1, 2] = 2 * halton(i, 5) - 1
    return pts


CANONICAL = make_canonical_lattice()


def process_sample(args):
    npy_id, pts_path = args
    if not os.path.exists(pts_path):
        return f"miss {pts_path}"

    out_path = pts_path[:-len("_pts.npy")] + "_octo.npy"
    if os.path.exists(out_path):
        return f"skip {npy_id} (exists)"

    raw = np.load(pts_path)  # (T, N, 8)
    if raw.ndim != 3 or raw.shape[2] < 7:
        return f"bad-shape {pts_path} {raw.shape}"

    pts = raw[..., 4:7].astype(np.float32)  # xyz channels
    T, N, _ = pts.shape

    # static bbox over all frames + points
    flat = pts.reshape(-1, 3)
    bb_min = flat.min(axis=0)
    bb_max = flat.max(axis=0)
    bb_size = np.maximum(bb_max - bb_min, 1e-6)
    voxel = bb_size / VOXEL_N

    # per-frame bbox center / half-extent
    pf_min = pts.min(axis=1)             # (T, 3)
    pf_max = pts.max(axis=1)             # (T, 3)
    pf_center = (pf_min + pf_max) * 0.5
    pf_half = (pf_max - pf_min) * 0.5

    lattice_pos = np.zeros((T, LATTICE_K, 3), dtype=np.float32)

    for t in range(T):
        # build occupancy + cell centers
        rel = (pts[t] - bb_min) / voxel
        idx = np.clip(rel.astype(np.int32), 0, VOXEL_N - 1)
        keys = idx[:, 0] + idx[:, 1] * VOXEL_N + idx[:, 2] * VOXEL_N * VOXEL_N
        occ_keys = np.unique(keys)
        ix = occ_keys % VOXEL_N
        iy = (occ_keys // VOXEL_N) % VOXEL_N
        iz = occ_keys // (VOXEL_N * VOXEL_N)
        centers = np.stack([
            bb_min[0] + (ix + 0.5) * voxel[0],
            bb_min[1] + (iy + 0.5) * voxel[1],
            bb_min[2] + (iz + 0.5) * voxel[2],
        ], axis=1).astype(np.float32)    # (M, 3)

        # candidate via per-frame bbox warp
        targets = pf_center[t][None, :] + CANONICAL * pf_half[t][None, :]   # (K, 3)

        if centers.shape[0] > 0:
            # nearest cell center per target (vectorized)
            d2 = ((targets[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
            nn = np.argmin(d2, axis=1)
            snapped = centers[nn]
            jitter = CANONICAL * voxel[None, :] * JITTER
            lattice_pos[t] = snapped + jitter
        else:
            lattice_pos[t] = targets

    # forward-difference velocity (last frame = 0)
    lattice_vel = np.zeros_like(lattice_pos)
    lattice_vel[:-1] = lattice_pos[1:] - lattice_pos[:-1]

    # octonion = (1, x, y, z, 1, vx, vy, vz)
    octo = np.zeros((T, LATTICE_K, 8), dtype=np.float32)
    octo[..., 0] = 1.0
    octo[..., 1:4] = lattice_pos
    octo[..., 4] = 1.0
    octo[..., 5:8] = lattice_vel

    np.save(out_path, octo)
    return f"ok {npy_id} {out_path} {octo.shape}"


def main():
    prefix = "./Nvidia"
    r = re.compile(r"[ \t\n\r:]+")
    train_path = f"{prefix}/Processed/train_depth_list.txt"
    test_path = f"{prefix}/Processed/test_depth_list.txt"

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print(f"missing list files; cwd={os.getcwd()}", file=sys.stderr)
        sys.exit(1)

    total_list = open(test_path).readlines() + open(train_path).readlines()

    pts_paths = []
    for line in total_list:
        parts = r.split(line)
        if len(parts) < 2:
            continue
        depth_npy = parts[1]
        pts_paths.append(depth_npy[:-4] + "_pts.npy")

    args_list = list(enumerate(pts_paths))
    print(f"processing {len(args_list)} samples ...")

    n_workers = min(cpu_count(), 16)
    with Pool(processes=n_workers) as pool:
        for i, msg in enumerate(pool.imap_unordered(process_sample, args_list)):
            if i % 200 == 0 or msg.startswith(("bad", "miss")):
                print(f"[{i}/{len(args_list)}] {msg}")

    np.save(f"{prefix}/canonical_lattice_512.npy", CANONICAL)
    print("done. saved canonical_lattice_512.npy")


if __name__ == "__main__":
    main()

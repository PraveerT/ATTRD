"""
Build `<stub>_pts10.npy` — 10-channel single-stream input that fuses raw
random-sampled point cloud with the corresponded lattice/octo features.

Per raw point at frame t:
  channels [0:4] = raw uvd-t  (from <stub>_pts.npy)
  channels [4:10] = lattice octo of NEAREST lattice cell, by xyz distance:
                   (x, y, z, vx, vy, vz)

Nearest lookup is per-frame: for each raw point at frame t, find the lattice
point at frame t with minimum ||raw_xyz - lattice_xyz||² and copy that
lattice's six octo channels. Lattice positions come from `<stub>_octo.npy`
(channels 1:4 = lattice xyz, channels 5:8 = velocity).

Result shape per sample: (T, 512, 10) float32.
Existing `<stub>_pts.npy` and `<stub>_octo.npy` are untouched.
"""

import os
import re
import sys
import numpy as np
from multiprocessing import Pool, cpu_count


def process_sample(args):
    npy_id, pts_path = args
    octo_path = pts_path[:-len("_pts.npy")] + "_octo.npy"
    out_path = pts_path[:-len("_pts.npy")] + "_pts10.npy"
    if not os.path.exists(pts_path) or not os.path.exists(octo_path):
        return f"miss {pts_path}"
    if os.path.exists(out_path):
        return f"skip {npy_id} (exists)"

    raw = np.load(pts_path).astype(np.float32)        # (T, N, 8)
    octo = np.load(octo_path).astype(np.float32)      # (T, K, 8)
    if raw.ndim != 3 or octo.ndim != 3:
        return f"bad-shape {pts_path}"

    T, N, _ = raw.shape
    K = octo.shape[1]

    raw_uvdt = raw[..., :4]            # (T, N, 4) — first 4 channels of raw
    raw_xyz = raw[..., 4:7]            # (T, N, 3) — real-world xyz (8-ch raw)
    lat_xyz = octo[..., 1:4]           # (T, K, 3)
    lat_vel = octo[..., 5:8]           # (T, K, 3)
    lat_feats = np.concatenate([lat_xyz, lat_vel], axis=-1)  # (T, K, 6)

    out = np.zeros((T, N, 10), dtype=np.float32)
    out[..., :4] = raw_uvdt

    # per-frame nearest lattice
    for t in range(T):
        # squared distance: (N, K)
        diff = raw_xyz[t][:, None, :] - lat_xyz[t][None, :, :]
        d2 = (diff * diff).sum(axis=-1)
        nn = np.argmin(d2, axis=-1)               # (N,)
        out[t, :, 4:10] = lat_feats[t, nn]

    np.save(out_path, out)
    return f"ok {npy_id} {out_path} {out.shape}"


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
    print("done.")


if __name__ == "__main__":
    main()

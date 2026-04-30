"""
Build `<stub>_qcyc.npy` sidecar: per-frame quaternion product q_fwd_raw ·
q_fwd_lat for use as input feature in the QCC + lattice fusion experiments.

For each sample (T frames):
  1. Per frame t, compute orientation rotation R_raw[t] via centroid + SVD on
     raw point cloud (channels [4:7] of <stub>_pts.npy).
  2. Per frame t, compute orientation rotation R_lat[t] same way on lattice
     positions (channels [1:4] of <stub>_octo.npy).
  3. Sign-stabilize axes against the previous frame so successive
     orientations form a smooth chain (otherwise SVD axes flip arbitrarily).
  4. Convert to quaternions q_raw[t], q_lat[t].
  5. Forward transition quats per frame transition:
     q_fwd_raw[t] = q_raw[t+1] · q_raw[t]^*
     q_fwd_lat[t] = q_lat[t+1] · q_lat[t]^*
  6. Combined: q_comb[t] = q_fwd_raw[t] · q_fwd_lat[t]
  7. Last frame (t = T-1) padded with zero quaternion (no transition).

Output shape: (T, 4) float32, stored at <stub>_qcyc.npy.
Existing <stub>_pts.npy / <stub>_octo.npy / <stub>_pts10.npy untouched.
"""

import os
import re
import sys
import numpy as np
from multiprocessing import Pool, cpu_count


def _quat_mul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], dtype=np.float64)


def _quat_conj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


def _rotation_to_quat(R):
    # standard conversion (right-handed, R real)
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float64)
    return q / max(np.linalg.norm(q), 1e-12)


def _cloud_axes(pts):
    # pts: (N, 3). Returns 3x3 rotation (columns = principal axes).
    c = pts.mean(axis=0)
    centered = pts - c
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    R = Vt.T  # columns = principal axes
    if np.linalg.det(R) < 0:
        R[:, 2] *= -1
    return R


def _stabilize(prev_R, cur_R):
    flipped = cur_R.copy()
    for i in range(3):
        if np.dot(prev_R[:, i], flipped[:, i]) < 0:
            flipped[:, i] *= -1
    if np.linalg.det(flipped) < 0:
        flipped[:, 2] *= -1
    return flipped


def _orientations(pts_seq):
    # pts_seq: (T, N, 3). Returns (T, 4) quaternions, sign-stabilized.
    T = pts_seq.shape[0]
    R_seq = []
    for t in range(T):
        R = _cloud_axes(pts_seq[t])
        if t > 0:
            R = _stabilize(R_seq[-1], R)
        R_seq.append(R)
    qs = np.stack([_rotation_to_quat(R) for R in R_seq], axis=0)  # (T, 4)
    return qs


def process_sample(args):
    npy_id, pts_path = args
    octo_path = pts_path[:-len("_pts.npy")] + "_octo.npy"
    out_path = pts_path[:-len("_pts.npy")] + "_qcyc.npy"
    if not os.path.exists(pts_path) or not os.path.exists(octo_path):
        return f"miss {pts_path}"
    if os.path.exists(out_path):
        return f"skip {npy_id}"

    raw = np.load(pts_path).astype(np.float32)
    octo = np.load(octo_path).astype(np.float32)
    if raw.ndim != 3 or octo.ndim != 3 or raw.shape[2] < 7:
        return f"bad {pts_path}"

    raw_xyz = raw[..., 4:7]      # (T, N, 3)
    lat_xyz = octo[..., 1:4]     # (T, K, 3)
    T = raw_xyz.shape[0]

    q_raw = _orientations(raw_xyz)   # (T, 4)
    q_lat = _orientations(lat_xyz)   # (T, 4)

    out = np.zeros((T, 4), dtype=np.float32)
    for t in range(T - 1):
        q_fwd_raw = _quat_mul(q_raw[t + 1], _quat_conj(q_raw[t]))
        q_fwd_lat = _quat_mul(q_lat[t + 1], _quat_conj(q_lat[t]))
        q_comb = _quat_mul(q_fwd_raw, q_fwd_lat)
        n = max(np.linalg.norm(q_comb), 1e-12)
        out[t] = (q_comb / n).astype(np.float32)
    # last row stays zero (no transition)

    np.save(out_path, out)
    return f"ok {npy_id} {out.shape}"


def main():
    prefix = "./Nvidia"
    r = re.compile(r"[ \t\n\r:]+")
    train = f"{prefix}/Processed/train_depth_list.txt"
    test = f"{prefix}/Processed/test_depth_list.txt"
    if not (os.path.exists(train) and os.path.exists(test)):
        print(f"missing list files; cwd={os.getcwd()}", file=sys.stderr)
        sys.exit(1)

    total = open(test).readlines() + open(train).readlines()
    pts_paths = []
    for line in total:
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

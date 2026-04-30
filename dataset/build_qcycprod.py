"""
Build `<stub>_qcycprod.npy` sidecar: per-sample 12-dim global feature
  [Q_raw_cycle (4), Q_lat_cycle (4), Q_combined_cycle (4)]

Re-uses orientation/quaternion machinery from build_qcyc.py. For each sample:
  - q_raw[t], q_lat[t] from cloud SVD orientations (sign-stabilized)
  - q_fwd_raw[t] = q_raw[t+1] · q_raw[t]^*  (per transition)
  - q_fwd_lat[t] similarly
  - Q_raw_cycle = ∏_{t=0..T-2} q_fwd_raw[t]   (∼ identity for closed/rigid)
  - Q_lat_cycle = ∏_{t=0..T-2} q_fwd_lat[t]
  - Q_combined_cycle = ∏ q_combined[t] = ∏ (q_fwd_raw · q_fwd_lat)

Output: <stub>_qcycprod.npy float32 shape (12,).
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


def _quat_normalize(q):
    n = max(np.linalg.norm(q), 1e-12)
    return q / n


def _rotation_to_quat(R):
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
    return _quat_normalize(np.array([w, x, y, z], dtype=np.float64))


def _cloud_axes(pts):
    centered = pts - pts.mean(axis=0)
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    R = Vt.T
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
    T = pts_seq.shape[0]
    R_seq = []
    for t in range(T):
        R = _cloud_axes(pts_seq[t])
        if t > 0:
            R = _stabilize(R_seq[-1], R)
        R_seq.append(R)
    return np.stack([_rotation_to_quat(R) for R in R_seq], axis=0)


def process_sample(args):
    npy_id, pts_path = args
    octo_path = pts_path[:-len("_pts.npy")] + "_octo.npy"
    out_path = pts_path[:-len("_pts.npy")] + "_qcycprod.npy"
    if not os.path.exists(pts_path) or not os.path.exists(octo_path):
        return f"miss {pts_path}"
    if os.path.exists(out_path):
        return f"skip {npy_id}"

    raw = np.load(pts_path).astype(np.float32)
    octo = np.load(octo_path).astype(np.float32)
    raw_xyz = raw[..., 4:7]
    lat_xyz = octo[..., 1:4]

    q_raw = _orientations(raw_xyz)   # (T, 4)
    q_lat = _orientations(lat_xyz)
    T = q_raw.shape[0]

    # initialise as identity
    Q_raw = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    Q_lat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    Q_comb = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    for t in range(T - 1):
        q_fwd_r = _quat_normalize(_quat_mul(q_raw[t + 1], _quat_conj(q_raw[t])))
        q_fwd_l = _quat_normalize(_quat_mul(q_lat[t + 1], _quat_conj(q_lat[t])))
        q_c = _quat_normalize(_quat_mul(q_fwd_r, q_fwd_l))
        Q_raw = _quat_normalize(_quat_mul(q_fwd_r, Q_raw))
        Q_lat = _quat_normalize(_quat_mul(q_fwd_l, Q_lat))
        Q_comb = _quat_normalize(_quat_mul(q_c, Q_comb))

    out = np.concatenate([Q_raw, Q_lat, Q_comb]).astype(np.float32)  # (12,)
    np.save(out_path, out)
    return f"ok {npy_id}"


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

"""
Build `<stub>_qkabsch.npy` sidecar — per-sample T-1 Kabsch quaternions from
the corresponded lattice points, plus the composed cycle product.

For each sample (T frames, K=512 corresponded lattice points):
  - Per frame transition t -> t+1, run Kabsch on the K matched lattice points
    to get the rigid rotation R_lat[t -> t+1].
  - Convert to quaternion q_kab[t].
  - Compose cycle: Q_kab_cycle = product over t of q_kab[t].

Output shape per sample: (T, 4) float32. Last frame (t = T-1) zero-padded.
Cycle product is also saved as `<stub>_qkabschprod.npy` shape (4,).
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


def _quat_norm(q):
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
    return _quat_norm(np.array([w, x, y, z], dtype=np.float64))


def _kabsch(P, Q):
    """Kabsch: best rotation R such that R @ P_centered ≈ Q_centered.
       P, Q: (N, 3). Returns R (3, 3)."""
    Pc = P - P.mean(axis=0)
    Qc = Q - Q.mean(axis=0)
    H = Pc.T @ Qc
    U, S, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    D = np.eye(3)
    D[2, 2] = np.sign(d)
    R = Vt.T @ D @ U.T
    return R


def process_sample(args):
    npy_id, octo_path = args
    out_per = octo_path[:-len("_octo.npy")] + "_qkabsch.npy"
    out_prod = octo_path[:-len("_octo.npy")] + "_qkabschprod.npy"
    if not os.path.exists(octo_path):
        return f"miss {octo_path}"
    if os.path.exists(out_per) and os.path.exists(out_prod):
        return f"skip {npy_id}"

    octo = np.load(octo_path).astype(np.float32)  # (T, K, 8)
    lat_xyz = octo[..., 1:4].astype(np.float64)   # (T, K, 3)
    T, K, _ = lat_xyz.shape

    out = np.zeros((T, 4), dtype=np.float32)
    Q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    for t in range(T - 1):
        R = _kabsch(lat_xyz[t], lat_xyz[t + 1])
        q = _rotation_to_quat(R)
        out[t] = q.astype(np.float32)
        Q = _quat_norm(_quat_mul(q, Q))
    # last frame zero pad

    np.save(out_per, out)
    np.save(out_prod, Q.astype(np.float32))
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
    paths = []
    for line in total:
        parts = r.split(line)
        if len(parts) < 2:
            continue
        depth_npy = parts[1]
        paths.append(depth_npy[:-4] + "_octo.npy")

    args_list = list(enumerate(paths))
    print(f"processing {len(args_list)} samples ...")

    n_workers = min(cpu_count(), 16)
    with Pool(processes=n_workers) as pool:
        for i, msg in enumerate(pool.imap_unordered(process_sample, args_list)):
            if i % 200 == 0 or msg.startswith(("bad", "miss")):
                print(f"[{i}/{len(args_list)}] {msg}")
    print("done.")


if __name__ == "__main__":
    main()

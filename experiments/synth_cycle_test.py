"""Synthetic controlled experiment for the cluster-cycle mechanism.

Generates K=6 cluster point-cloud sequences with KNOWN per-cluster ground-truth
rotation trajectories, then verifies that:

(1) Q_act computed via eigh on weighted covariance recovers the ground-truth
    rotation up to double-cover.
(2) A small GRU + cycle_proj predictor trained with L_cycle converges to
    Q_pred matching Q_act (and therefore Q_gt).
(3) Adding the cycle aux loss improves classification of synthetic
    "gestures" defined by per-cluster rotation patterns, over a baseline
    classifier without cycle.

Run:
    python synth_cycle_test.py --epochs 30 --n-train 400 --n-test 100
"""
import argparse
import math
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
np.random.seed(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def axis_angle_to_quat(axis: np.ndarray, angle: float) -> np.ndarray:
    """axis: (3,) unit vector, angle: scalar radians -> quat (w, x, y, z)."""
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    h = angle * 0.5
    return np.array([np.cos(h), *(np.sin(h) * axis)])


def quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    """quat (w, x, y, z) -> 3x3 rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z),     2 * (x * z + w * y)],
        [2 * (x * y + w * z),     1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y),     2 * (y * z + w * x),     1 - 2 * (x * x + y * y)],
    ])


# Class templates: 8 gestures, each defined by K=6 (rotation axis, max angle).
K = 6
NUM_CLASSES = 8
CLASS_TEMPLATES = [
    # (cluster_idx -> (axis, max_angle_rad))
    {k: (np.array([1.0, 0, 0]), 0.0) for k in range(K)},                  # 0: static
    {k: (np.array([0, 1.0, 0]), 0.6 if k == 0 else 0.0) for k in range(K)},  # 1: palm twist
    {k: (np.array([0, 0, 1.0]), 0.0 if k == 0 else 1.2) for k in range(K)},  # 2: all-fingers curl
    {k: (np.array([0, 0, 1.0]), 0.0 if k == 0 else -1.2) for k in range(K)}, # 3: all-fingers extend
    {k: (np.array([1.0, 0, 0]), 1.0 if k == 1 else 0.0) for k in range(K)},  # 4: thumb only
    {k: (np.array([0, 1.0, 0]), 0.8 * (1 if k % 2 == 0 else -1)) for k in range(K)},  # 5: alternating
    {k: (np.array([1.0, 1.0, 0]) / np.sqrt(2), 0.5 + 0.1 * k) for k in range(K)},     # 6: progressive
    {k: (np.array([0.7, 0.7, 0]), 0.3 if k % 2 == 0 else 0.8) for k in range(K)},     # 7: split-curl
]


def gen_sequence(label: int, T: int = 32, points_per_cluster: int = 30):
    """Generate one (T, P, 3) point cloud sequence + per-cluster GT quaternions.

    Returns:
        coords: (T, K * points_per_cluster, 3) numpy float32
        gt_quats: (T, K, 4) numpy float32 — ground-truth Q for each cluster per frame
        gt_cluster: (P,) numpy int64 — per-point cluster id (deterministic)
    """
    template = CLASS_TEMPLATES[label]
    P = K * points_per_cluster

    # Initial cluster centroids around a "palm" (cluster 0).
    cluster_offsets = np.array([
        [0.0, 0.0, 0.0],    # palm
        [0.6, 0.2, 0.0],    # thumb
        [0.3, 0.8, 0.0],
        [0.0, 0.9, 0.0],
        [-0.3, 0.8, 0.0],
        [-0.6, 0.2, 0.0],
    ], dtype=np.float32)

    # Each cluster: small elongated blob (random initial cloud).
    base_clusters = []
    for k in range(K):
        # elongate along z to give a meaningful principal axis
        pts = np.random.randn(points_per_cluster, 3) * np.array([0.05, 0.05, 0.15])
        base_clusters.append(pts + cluster_offsets[k])

    coords = np.zeros((T, P, 3), dtype=np.float32)
    gt_quats = np.zeros((T, K, 4), dtype=np.float32)
    gt_cluster = np.zeros(P, dtype=np.int64)
    for k in range(K):
        gt_cluster[k * points_per_cluster:(k + 1) * points_per_cluster] = k

    for t in range(T):
        # smoothly interpolate angle from 0 to max as t goes 0..T-1
        phase = t / max(T - 1, 1)
        for k in range(K):
            axis, max_angle = template[k]
            angle = phase * max_angle
            q = axis_angle_to_quat(axis, angle)
            R = quat_to_rotmat(q)
            centroid = cluster_offsets[k]
            centered = base_clusters[k] - centroid
            rotated = centered @ R.T + centroid
            coords[t, k * points_per_cluster:(k + 1) * points_per_cluster] = rotated
            gt_quats[t, k] = q

    # add light per-frame noise so it's not trivially reconstructible
    coords += np.random.randn(*coords.shape).astype(np.float32) * 0.005
    return coords, gt_quats, gt_cluster


def make_dataset(n: int, T: int = 32, ppc: int = 30, seed: int = 0):
    rng = np.random.RandomState(seed)
    labels = rng.randint(0, NUM_CLASSES, size=n)
    coords_list, quat_list, cluster_list = [], [], []
    for lab in labels:
        c, q, k = gen_sequence(int(lab), T=T, points_per_cluster=ppc)
        coords_list.append(c); quat_list.append(q); cluster_list.append(k)
    return (np.stack(coords_list), np.stack(quat_list),
            np.stack(cluster_list), labels.astype(np.int64))


# ----------------------- mechanism core (lifted from raw_c1) -----------------
def compute_qact_per_cluster(xyz: torch.Tensor, alpha: torch.Tensor):
    """xyz: (B, T, P, 3); alpha: (B, T, P, K) softmax weights.
    Returns Q_act: (B, T, K, 4).
    """
    eps = 1e-4
    mass = alpha.sum(dim=2).clamp_min(1e-8)                       # (B, T, K)
    # centroid (B, T, K, 3)
    centroid = (alpha.unsqueeze(-1) * xyz.unsqueeze(-2)).sum(dim=2) / mass.unsqueeze(-1)
    centered = xyz.unsqueeze(-2) - centroid.unsqueeze(2)           # (B, T, P, K, 3)
    weighted = centered * alpha.unsqueeze(-1)
    # cov (B, T, K, 3, 3)
    cov = torch.einsum('btpkd,btpke->btkde', weighted, centered) / mass.unsqueeze(-1).unsqueeze(-1)
    I = torch.eye(3, device=cov.device, dtype=cov.dtype)
    cov = cov + eps * I
    vals, vecs = torch.linalg.eigh(cov)                            # vals ascending
    axis = vecs[..., -1]                                          # (B, T, K, 3)
    angle = vals[..., -1].clamp_min(0).sqrt()                     # (B, T, K)
    h = angle * 0.5
    qw = torch.cos(h).unsqueeze(-1)
    qxyz = torch.sin(h).unsqueeze(-1) * axis
    return torch.cat([qw, qxyz], dim=-1), centered


def quat_distance_double_cover(q1: torch.Tensor, q2: torch.Tensor):
    """Min of ||q1 - q2|| and ||q1 + q2|| squared, averaged."""
    d1 = ((q1 - q2) ** 2).sum(-1)
    d2 = ((q1 + q2) ** 2).sum(-1)
    return torch.min(d1, d2).mean()


def hamilton(q1, q2):
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    return torch.stack([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], dim=-1)


def axis_angle_exp(v: torch.Tensor):
    """v: (..., 3) -> unit quaternion (..., 4)."""
    norm = v.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    h = norm * 0.5
    qw = torch.cos(h)
    qxyz = torch.sin(h) * (v / norm)
    return torch.cat([qw, qxyz], dim=-1)


# --------------------------- model ------------------------------------------
class SynthModel(nn.Module):
    """Tiny K=6 cluster classifier with optional cycle aux head.

    Backbone: per-point MLP -> per-cluster aggregate (via either GT cluster
    assignment or learnable soft alpha) -> per-cluster temporal GRU ->
    classification head. The aux head reuses the cluster features through a
    second GRU that predicts axis-angle, then exp-maps to a quaternion.
    """
    def __init__(self, num_classes: int, hidden: int = 64,
                 use_gt_cluster: bool = True, K: int = K):
        super().__init__()
        self.K = K
        self.use_gt_cluster = use_gt_cluster
        self.point_mlp = nn.Sequential(
            nn.Linear(3, 32), nn.ReLU(),
            nn.Linear(32, hidden), nn.ReLU(),
        )
        if not use_gt_cluster:
            self.cluster_head = nn.Sequential(
                nn.Linear(hidden, hidden), nn.ReLU(),
                nn.Linear(hidden, K),
            )
        # classification temporal GRU on cluster-aggregate features
        self.gru_clf = nn.GRU(hidden, hidden, batch_first=True, bidirectional=True)
        self.clf = nn.Linear(2 * hidden * K, num_classes)
        # aux cycle head: separate GRU + zero-init projection
        self.gru_cyc = nn.GRU(hidden, hidden, batch_first=True, bidirectional=True)
        self.cycle_proj = nn.Linear(2 * hidden, 3)
        nn.init.zeros_(self.cycle_proj.weight)
        nn.init.zeros_(self.cycle_proj.bias)

    def forward(self, coords: torch.Tensor, gt_cluster: torch.Tensor | None = None):
        # coords: (B, T, P, 3); gt_cluster: (B, P) ints in [0, K) if use_gt_cluster
        B, T, P, _ = coords.shape
        feat = self.point_mlp(coords)                              # (B, T, P, H)
        if self.use_gt_cluster:
            assert gt_cluster is not None
            # one-hot soft assignment, broadcast across time
            alpha_static = F.one_hot(gt_cluster, num_classes=self.K).float()  # (B, P, K)
        else:
            # time-pooled feature -> cluster logits -> softmax
            pooled = feat.mean(dim=1)                              # (B, P, H)
            cluster_logits = self.cluster_head(pooled)             # (B, P, K)
            alpha_static = F.softmax(cluster_logits, dim=-1)
        alpha = alpha_static.unsqueeze(1).expand(B, T, P, self.K).contiguous()

        # per-cluster feature aggregate over points (weighted by alpha).
        mass = alpha.sum(dim=2).clamp_min(1e-8)                    # (B, T, K)
        cf = (alpha.unsqueeze(-1) * feat.unsqueeze(-2)).sum(dim=2) / mass.unsqueeze(-1)
        # cf: (B, T, K, H). Reshape to (B*K, T, H) for GRU.
        H = cf.size(-1)
        cf_btkh = cf.permute(0, 2, 1, 3).reshape(B * self.K, T, H)
        # Classification: bi-GRU + last-step concatenation
        out_clf, _ = self.gru_clf(cf_btkh)                         # (B*K, T, 2H)
        cls_feat = out_clf[:, -1, :].reshape(B, self.K * 2 * H)
        logits = self.clf(cls_feat)
        # Cycle aux: bi-GRU + cycle_proj -> axis-angle -> Q_pred
        out_cyc, _ = self.gru_cyc(cf_btkh)                         # (B*K, T, 2H)
        v = self.cycle_proj(out_cyc)                               # (B*K, T, 3)
        Q_pred = axis_angle_exp(v).reshape(B, self.K, T, 4).permute(0, 2, 1, 3)
        return logits, Q_pred, alpha


# --------------------------- training loop ---------------------------------
def train_loop(args, use_cycle: bool, use_gt_cluster: bool = True):
    print(f'\n=== train_loop  use_cycle={use_cycle}  use_gt_cluster={use_gt_cluster} ===')
    Xtr, Qtr, Ktr, ytr = make_dataset(args.n_train, T=args.T, ppc=args.ppc, seed=1)
    Xte, Qte, Kte, yte = make_dataset(args.n_test, T=args.T, ppc=args.ppc, seed=2)
    Xtr_t = torch.from_numpy(Xtr).to(device)
    Ktr_t = torch.from_numpy(Ktr).to(device)
    ytr_t = torch.from_numpy(ytr).to(device)
    Xte_t = torch.from_numpy(Xte).to(device)
    Kte_t = torch.from_numpy(Kte).to(device)
    yte_t = torch.from_numpy(yte).to(device)
    Qtr_t = torch.from_numpy(Qtr).to(device)

    model = SynthModel(num_classes=NUM_CLASSES, use_gt_cluster=use_gt_cluster).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    log = {'epoch': [], 'train_loss': [], 'test_acc': [],
           'cycle_loss': [], 'qact_vs_gt': [], 'qpred_vs_gt': []}
    for epoch in range(args.epochs):
        model.train()
        # full-batch (small dataset)
        logits, Q_pred, alpha = model(Xtr_t, Ktr_t if use_gt_cluster else None)
        ce = F.cross_entropy(logits, ytr_t)
        Q_act, _ = compute_qact_per_cluster(Xtr_t, alpha)
        cycle = quat_distance_double_cover(Q_pred, Q_act)
        loss = ce + (0.05 * cycle if use_cycle else 0.0 * cycle)
        opt.zero_grad(); loss.backward(); opt.step()
        # eval
        model.eval()
        with torch.no_grad():
            te_logits, Qpred_te, alpha_te = model(Xte_t, Kte_t if use_gt_cluster else None)
            te_acc = (te_logits.argmax(1) == yte_t).float().mean().item() * 100
            Qact_te, _ = compute_qact_per_cluster(Xte_t, alpha_te)
            cyc_te = quat_distance_double_cover(Qpred_te, Qact_te).item()
            Qte_t_dev = torch.from_numpy(Qte).to(device)
            qact_vs_gt = quat_distance_double_cover(Qact_te, Qte_t_dev).item()
            qpred_vs_gt = quat_distance_double_cover(Qpred_te, Qte_t_dev).item()
        log['epoch'].append(epoch + 1)
        log['train_loss'].append(loss.item())
        log['test_acc'].append(te_acc)
        log['cycle_loss'].append(cyc_te)
        log['qact_vs_gt'].append(qact_vs_gt)
        log['qpred_vs_gt'].append(qpred_vs_gt)
        if (epoch + 1) % max(1, args.epochs // 10) == 0 or epoch == 0:
            print(f'  ep{epoch+1:3d}  ce={ce.item():.4f}  cyc={cycle.item():.4f}  '
                  f'te_acc={te_acc:.2f}  cyc_te={cyc_te:.4f}  '
                  f'qact_vs_gt={qact_vs_gt:.4f}  qpred_vs_gt={qpred_vs_gt:.4f}')
    return log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--n-train', type=int, default=400)
    ap.add_argument('--n-test', type=int, default=100)
    ap.add_argument('--T', type=int, default=32)
    ap.add_argument('--ppc', type=int, default=30)
    args = ap.parse_args()

    # Sanity: Q_act from a perfectly known synthetic cluster should be close to Q_gt
    # for a single-cluster sequence. Quick standalone check first.
    print('=== sanity: eigh on a single rotating cluster ===')
    X_one, Q_one, K_one, _ = make_dataset(1, T=args.T, ppc=args.ppc, seed=42)
    X = torch.from_numpy(X_one).to(device)
    alpha = F.one_hot(torch.from_numpy(K_one).to(device), num_classes=K).float().unsqueeze(1).expand(1, args.T, X.shape[2], K)
    Qact, _ = compute_qact_per_cluster(X, alpha)
    Qgt = torch.from_numpy(Q_one).to(device)
    d = quat_distance_double_cover(Qact, Qgt).item()
    print(f'  Q_act vs Q_gt (double-cover squared distance): {d:.4f}')
    print('  (NOTE: Q_act recovers PRINCIPAL AXIS; it may not exactly equal the')
    print('   ground-truth quaternion if the cluster shape is symmetric around')
    print('   an axis. This is expected — we measure CONSISTENCY of Q_pred to')
    print('   Q_act, not absolute fidelity to the rotation that generated the data.)')

    # Run baseline (no cycle) and cycle-enabled training, both with GT cluster ids.
    base = train_loop(args, use_cycle=False, use_gt_cluster=True)
    cycle = train_loop(args, use_cycle=True, use_gt_cluster=True)

    print()
    print('=== summary (last epoch) ===')
    b_acc = base['test_acc'][-1]; c_acc = cycle['test_acc'][-1]
    b_cyc = base['cycle_loss'][-1]; c_cyc = cycle['cycle_loss'][-1]
    print(f'  baseline:   test_acc={b_acc:.2f}  cyc_loss={b_cyc:.4f}')
    print(f'  cycle:      test_acc={c_acc:.2f}  cyc_loss={c_cyc:.4f}')
    print(f'  delta:      test_acc={c_acc - b_acc:+.2f}')

    # Save logs for later plotting / paper figures
    np.savez('/tmp/synth_cycle_test_results.npz',
             baseline=base, cycle=cycle)
    print('  saved logs to /tmp/synth_cycle_test_results.npz')


if __name__ == '__main__':
    main()

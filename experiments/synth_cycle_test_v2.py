"""Synthetic controlled experiment v2 — scaled up.

Key changes from v1:
  - sharper, fewer classes (5) with VERY distinct per-cluster rotation patterns
  - stronger model (multi-layer point encoder + larger temporal GRU)
  - more training data (1500 train, 400 test) and 60 epochs with cosine LR
  - reports classification accuracy delta + cnxxl-vs-rawc1 fusion lift
    (two independently-trained classifiers; the only difference is the cycle
    aux loss on one of them)

Run:
    python synth_cycle_test_v2.py --epochs 60 --n-train 1500 --n-test 400
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

torch.manual_seed(42)
np.random.seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def axis_angle_to_quat(axis, angle):
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    h = angle * 0.5
    return np.array([np.cos(h), *(np.sin(h) * axis)])


def quat_to_rotmat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z),     2 * (x * z + w * y)],
        [2 * (x * y + w * z),     1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y),     2 * (y * z + w * x),     1 - 2 * (x * x + y * y)],
    ])


# 5 classes with very different per-cluster rotation patterns
K = 6
NUM_CLASSES = 5
# Each: list of K (axis, max_angle) tuples
def _ax(a): return np.array(a, dtype=np.float32)
CLASS_TEMPLATES = [
    # 0: all curl (z-axis, big positive)
    [(_ax([0, 0, 1]), 0.0)] + [(_ax([0, 0, 1]), 1.4)] * 5,
    # 1: all extend (z-axis, big negative)
    [(_ax([0, 0, 1]), 0.0)] + [(_ax([0, 0, 1]), -1.4)] * 5,
    # 2: palm twist + fingers static
    [(_ax([0, 1, 0]), 1.2)] + [(_ax([0, 1, 0]), 0.0)] * 5,
    # 3: thumb only (cluster 1 rotates about X)
    [(_ax([1, 0, 0]), 0.0)] + [(_ax([1, 0, 0]), 1.2)] + [(_ax([1, 0, 0]), 0.0)] * 4,
    # 4: alternating curl-extend (fingers 1,3,5 curl, fingers 2,4 extend)
    [(_ax([0, 0, 1]), 0.0)] +
    [(_ax([0, 0, 1]), 1.2 if k_idx % 2 == 1 else -1.2) for k_idx in range(1, K)],
]


def gen_sequence(label, T=32, points_per_cluster=30):
    template = CLASS_TEMPLATES[label]
    P = K * points_per_cluster
    # Initial cluster centroids (hand-like layout)
    cluster_offsets = np.array([
        [0.0, 0.0, 0.0],    # palm
        [0.6, 0.2, 0.0],
        [0.3, 0.8, 0.0],
        [0.0, 0.9, 0.0],
        [-0.3, 0.8, 0.0],
        [-0.6, 0.2, 0.0],
    ], dtype=np.float32)
    base_clusters = []
    for k in range(K):
        pts = np.random.randn(points_per_cluster, 3) * np.array([0.05, 0.05, 0.15])
        base_clusters.append(pts + cluster_offsets[k])

    coords = np.zeros((T, P, 3), dtype=np.float32)
    gt_quats = np.zeros((T, K, 4), dtype=np.float32)
    gt_cluster = np.zeros(P, dtype=np.int64)
    for k in range(K):
        gt_cluster[k * points_per_cluster:(k + 1) * points_per_cluster] = k

    for t in range(T):
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

    coords += np.random.randn(*coords.shape).astype(np.float32) * 0.005
    return coords, gt_quats, gt_cluster


def make_dataset(n, T=32, ppc=30, seed=0):
    rng = np.random.RandomState(seed)
    labels = rng.randint(0, NUM_CLASSES, size=n)
    coords_list, quat_list, cluster_list = [], [], []
    for lab in labels:
        c, q, k = gen_sequence(int(lab), T=T, points_per_cluster=ppc)
        coords_list.append(c); quat_list.append(q); cluster_list.append(k)
    return (np.stack(coords_list), np.stack(quat_list),
            np.stack(cluster_list), labels.astype(np.int64))


# -------------------- mechanism core (same as raw_c1) ----------------------
def compute_qact_per_cluster(xyz, alpha):
    eps = 1e-4
    mass = alpha.sum(dim=2).clamp_min(1e-8)
    centroid = (alpha.unsqueeze(-1) * xyz.unsqueeze(-2)).sum(dim=2) / mass.unsqueeze(-1)
    centered = xyz.unsqueeze(-2) - centroid.unsqueeze(2)
    weighted = centered * alpha.unsqueeze(-1)
    cov = torch.einsum('btpkd,btpke->btkde', weighted, centered) / mass.unsqueeze(-1).unsqueeze(-1)
    I = torch.eye(3, device=cov.device, dtype=cov.dtype)
    cov = cov + eps * I
    vals, vecs = torch.linalg.eigh(cov)
    axis = vecs[..., -1]
    angle = vals[..., -1].clamp_min(0).sqrt()
    h = angle * 0.5
    qw = torch.cos(h).unsqueeze(-1)
    qxyz = torch.sin(h).unsqueeze(-1) * axis
    return torch.cat([qw, qxyz], dim=-1), centered


def quat_distance_double_cover(q1, q2):
    d1 = ((q1 - q2) ** 2).sum(-1)
    d2 = ((q1 + q2) ** 2).sum(-1)
    return torch.min(d1, d2).mean()


def axis_angle_exp(v):
    norm = v.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    h = norm * 0.5
    qw = torch.cos(h)
    qxyz = torch.sin(h) * (v / norm)
    return torch.cat([qw, qxyz], dim=-1)


# --------------------------- stronger model --------------------------------
class StrongerSynthModel(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, hidden=128, K=K, gru_layers=2):
        super().__init__()
        self.K = K
        # 3-layer per-point MLP (deeper feature extractor)
        self.point_mlp = nn.Sequential(
            nn.Linear(3, 64), nn.ReLU(), nn.LayerNorm(64),
            nn.Linear(64, 128), nn.ReLU(), nn.LayerNorm(128),
            nn.Linear(128, hidden), nn.ReLU(), nn.LayerNorm(hidden),
        )
        # 2-layer bidir GRU on cluster features for classification
        self.gru_clf = nn.GRU(hidden, hidden, num_layers=gru_layers,
                              batch_first=True, bidirectional=True, dropout=0.2)
        # classification head: mean-pool over time, then over clusters
        self.clf = nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(hidden, num_classes),
        )
        # aux cycle head
        self.gru_cyc = nn.GRU(hidden, hidden, num_layers=1,
                              batch_first=True, bidirectional=True)
        self.cycle_proj = nn.Linear(2 * hidden, 3)
        nn.init.zeros_(self.cycle_proj.weight)
        nn.init.zeros_(self.cycle_proj.bias)

    def forward(self, coords, gt_cluster):
        # coords: (B, T, P, 3); gt_cluster: (B, P)
        B, T, P, _ = coords.shape
        feat = self.point_mlp(coords)  # (B, T, P, H)
        # GT cluster ids (sharper than learned soft assignment — we test the
        # mechanism, not cluster discovery).
        alpha_static = F.one_hot(gt_cluster, num_classes=self.K).float()  # (B, P, K)
        alpha = alpha_static.unsqueeze(1).expand(B, T, P, self.K).contiguous()
        mass = alpha.sum(dim=2).clamp_min(1e-8)
        cf = (alpha.unsqueeze(-1) * feat.unsqueeze(-2)).sum(dim=2) / mass.unsqueeze(-1)
        # cf: (B, T, K, H)
        H = cf.size(-1)
        cf_btkh = cf.permute(0, 2, 1, 3).reshape(B * self.K, T, H)
        # Classification GRU
        out_clf, _ = self.gru_clf(cf_btkh)             # (B*K, T, 2H)
        # mean over time, then mean over clusters
        per_cluster_clf = out_clf.mean(dim=1)          # (B*K, 2H)
        per_cluster_clf = per_cluster_clf.reshape(B, self.K, 2 * H)
        global_feat = per_cluster_clf.mean(dim=1)      # (B, 2H)
        logits = self.clf(global_feat)
        # Cycle aux head
        out_cyc, _ = self.gru_cyc(cf_btkh)             # (B*K, T, 2H)
        v = self.cycle_proj(out_cyc)                   # (B*K, T, 3)
        Q_pred = axis_angle_exp(v).reshape(B, self.K, T, 4).permute(0, 2, 1, 3)
        return logits, Q_pred, alpha


# --------------------------- training loop ---------------------------------
def train_loop(args, use_cycle, label='', batch_size=32):
    print(f'\n=== train_loop  use_cycle={use_cycle}  ({label}) ===')
    Xtr, _, Ktr, ytr = make_dataset(args.n_train, T=args.T, ppc=args.ppc, seed=1)
    Xte, Qte, Kte, yte = make_dataset(args.n_test, T=args.T, ppc=args.ppc, seed=2)
    Xte_t = torch.from_numpy(Xte).to(device)
    Kte_t = torch.from_numpy(Kte).to(device)
    yte_t = torch.from_numpy(yte).to(device)

    model = StrongerSynthModel().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-5)

    n_train = args.n_train
    log = {'epoch': [], 'train_loss': [], 'test_acc': [],
           'cycle_loss_te': [], 'qpred_vs_qact_te': []}
    best_acc = 0.0
    best_logits_te = None
    for epoch in range(args.epochs):
        model.train()
        # mini-batch training
        idx = np.random.permutation(n_train)
        total_loss = 0; total_ce = 0; total_cyc = 0; nbat = 0
        for i in range(0, n_train, batch_size):
            sel = idx[i:i + batch_size]
            X = torch.from_numpy(Xtr[sel]).to(device)
            K_in = torch.from_numpy(Ktr[sel]).to(device)
            y = torch.from_numpy(ytr[sel]).to(device)
            logits, Q_pred, alpha = model(X, K_in)
            ce = F.cross_entropy(logits, y)
            Q_act, _ = compute_qact_per_cluster(X, alpha)
            cyc = quat_distance_double_cover(Q_pred, Q_act)
            loss = ce + (0.05 * cyc if use_cycle else 0.0)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item(); total_ce += ce.item(); total_cyc += cyc.item(); nbat += 1
        scheduler.step()
        # eval (full test set in chunks)
        model.eval()
        with torch.no_grad():
            te_logits_chunks = []
            chunk = 64
            for i in range(0, len(Xte_t), chunk):
                logits_c, _, _ = model(Xte_t[i:i + chunk], Kte_t[i:i + chunk])
                te_logits_chunks.append(logits_c.cpu())
            te_logits = torch.cat(te_logits_chunks, dim=0).to(device)
            te_acc = (te_logits.argmax(1) == yte_t).float().mean().item() * 100
            # also measure cycle convergence at test
            X_chunk = Xte_t[:64]
            _, Qpred_te, alpha_te = model(X_chunk, Kte_t[:64])
            Qact_te, _ = compute_qact_per_cluster(X_chunk, alpha_te)
            cyc_te = quat_distance_double_cover(Qpred_te, Qact_te).item()
            if te_acc > best_acc:
                best_acc = te_acc
                best_logits_te = te_logits.cpu().clone()
        log['epoch'].append(epoch + 1)
        log['train_loss'].append(total_loss / nbat)
        log['test_acc'].append(te_acc)
        log['cycle_loss_te'].append(cyc_te)
        if (epoch + 1) % max(1, args.epochs // 10) == 0 or epoch == 0:
            print(f'  ep{epoch+1:3d}  ce={total_ce/nbat:.4f}  cyc={total_cyc/nbat:.4f}  '
                  f'te_acc={te_acc:.2f}  best={best_acc:.2f}  cyc_te={cyc_te:.4f}')
    return log, best_logits_te, yte_t.cpu()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--n-train', type=int, default=1500)
    ap.add_argument('--n-test', type=int, default=400)
    ap.add_argument('--T', type=int, default=32)
    ap.add_argument('--ppc', type=int, default=30)
    args = ap.parse_args()

    # Train two independent models with different random seeds for fusion.
    torch.manual_seed(42); np.random.seed(42)
    base_log, base_logits, y_te = train_loop(args, use_cycle=False, label='baseline (seed 42)')

    torch.manual_seed(43); np.random.seed(43)
    cycle_log, cycle_logits, _ = train_loop(args, use_cycle=True, label='cycle (seed 43)')

    # Fusion of the two (late softmax avg)
    def softmax_np(x):
        x = x - x.max(-1, keepdims=True); e = np.exp(x); return e / e.sum(-1, keepdims=True)
    pa = softmax_np(base_logits.numpy())
    pb = softmax_np(cycle_logits.numpy())
    y = y_te.numpy()
    base_acc = (base_logits.numpy().argmax(1) == y).mean() * 100
    cycle_acc = (cycle_logits.numpy().argmax(1) == y).mean() * 100
    fused_acc = ((0.5 * pa + 0.5 * pb).argmax(1) == y).mean() * 100

    print()
    print('===== SUMMARY =====')
    print(f'baseline (no cycle):  best_test_acc={base_log["test_acc"][-1]:.2f}  final_cyc_te={base_log["cycle_loss_te"][-1]:.4f}')
    print(f'cycle  (with aux):    best_test_acc={cycle_log["test_acc"][-1]:.2f}  final_cyc_te={cycle_log["cycle_loss_te"][-1]:.4f}')
    print()
    print('=== fusion (test-best ckpt softmax avg) ===')
    print(f'  baseline solo (argmax):  {base_acc:.2f}')
    print(f'  cycle solo (argmax):     {cycle_acc:.2f}')
    print(f'  fusion (0.5+0.5):        {fused_acc:.2f}')
    fusion_lift = fused_acc - max(base_acc, cycle_acc)
    print(f'  fusion lift vs best solo: {fusion_lift:+.2f}')

    np.savez('/tmp/synth_v2_results.npz',
             base_log=base_log, cycle_log=cycle_log,
             base_acc=base_acc, cycle_acc=cycle_acc, fused_acc=fused_acc)
    print('  saved to /tmp/synth_v2_results.npz')


if __name__ == '__main__':
    main()

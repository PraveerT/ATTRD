"""Synthetic cycle experiment v3 — NO CORRESPONDENCE.

Cluster identity is NOT provided to the model. The cluster_head must
discover the K=6 cluster decomposition from scratch. Classes are
designed so that global features are insufficient: each class is defined
by WHICH specific clusters move (2 active clusters out of 6), with no
class-level difference in total motion magnitude or direction.

Question: does the cycle aux loss help the model discover the cluster
decomposition (-> better classification or fusion lift) vs a baseline
that uses the same architecture but only classification loss (with
L_balance kept on in both, so clusters don't collapse trivially)?

Run:
    python synth_cycle_test_v3.py --epochs 80 --n-train 2000 --n-test 400
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

torch.manual_seed(7)
np.random.seed(7)
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


# 6 classes — each is defined by WHICH 2 clusters rotate. All 2-active
# patterns rotate the same axis/angle, so the GLOBAL motion signature is
# identical across classes; only the per-cluster pattern differs.
K = 6
NUM_CLASSES = 6
# Set of "active cluster pairs" per class
ACTIVE_CLUSTERS = [
    (0, 1),   # class 0: clusters 0 and 1 rotate
    (2, 3),   # class 1: clusters 2 and 3 rotate
    (4, 5),   # class 2: clusters 4 and 5 rotate
    (0, 3),   # class 3
    (1, 4),   # class 4
    (2, 5),   # class 5
]


def gen_sequence(label, T=32, points_per_cluster=30):
    active = set(ACTIVE_CLUSTERS[label])
    P = K * points_per_cluster
    # Cluster centroids (hand-like layout) — same for every class
    cluster_offsets = np.array([
        [0.0, 0.0, 0.0],    # 0 palm
        [0.6, 0.2, 0.0],    # 1
        [0.3, 0.8, 0.0],    # 2
        [0.0, 0.9, 0.0],    # 3
        [-0.3, 0.8, 0.0],   # 4
        [-0.6, 0.2, 0.0],   # 5
    ], dtype=np.float32)
    # Each cluster: same shape distribution, slightly elongated along z
    base_clusters = []
    for k in range(K):
        pts = np.random.randn(points_per_cluster, 3) * np.array([0.06, 0.06, 0.20])
        base_clusters.append(pts + cluster_offsets[k])

    coords = np.zeros((T, P, 3), dtype=np.float32)
    # Active clusters all rotate around Z, +1.2 rad over the sequence
    rot_axis = np.array([0.0, 0.0, 1.0])
    rot_max = 1.2
    for t in range(T):
        phase = t / max(T - 1, 1)
        for k in range(K):
            if k in active:
                angle = phase * rot_max
            else:
                angle = 0.0
            R = quat_to_rotmat(axis_angle_to_quat(rot_axis, angle))
            centroid = cluster_offsets[k]
            rotated = (base_clusters[k] - centroid) @ R.T + centroid
            coords[t, k * points_per_cluster:(k + 1) * points_per_cluster] = rotated

    # Small per-frame noise
    coords += np.random.randn(*coords.shape).astype(np.float32) * 0.005

    # SHUFFLE points so cluster identity isn't deducible from point order
    perm = np.random.permutation(P)
    coords = coords[:, perm, :]
    # We return GT-cluster-id-after-shuffle for OPTIONAL diagnostic only;
    # the model never receives it.
    gt_cluster_unshuffled = np.repeat(np.arange(K), points_per_cluster)
    gt_cluster = gt_cluster_unshuffled[perm]
    return coords, gt_cluster.astype(np.int64)


def make_dataset(n, T=32, ppc=30, seed=0):
    rng = np.random.RandomState(seed)
    labels = rng.randint(0, NUM_CLASSES, size=n)
    coords_list, cluster_list = [], []
    for lab in labels:
        c, k = gen_sequence(int(lab), T=T, points_per_cluster=ppc)
        coords_list.append(c); cluster_list.append(k)
    return (np.stack(coords_list), np.stack(cluster_list), labels.astype(np.int64))


# --------------- mechanism core (same as raw_c1) ---------------------------
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
    return torch.cat([qw, qxyz], dim=-1), centered, mass


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


def cluster_entropy(alpha):
    """Mean cluster-mass entropy (we want HIGH entropy = balanced clusters)."""
    mean_alpha = alpha.mean(dim=(0, 1, 2)).clamp_min(1e-8)
    mean_alpha = mean_alpha / mean_alpha.sum()
    return -(mean_alpha * mean_alpha.log()).sum()


# --------------------------- model -----------------------------------------
class SynthV3Model(nn.Module):
    """Cluster discovery + temporal classification + optional cycle aux.

    No GT cluster ids. cluster_head produces softmax assignment from
    time-pooled per-point features. Classification uses per-cluster
    pooled features fed to a temporal GRU and a small head.
    """
    def __init__(self, num_classes=NUM_CLASSES, hidden=128, K=K):
        super().__init__()
        self.K = K
        # 3-layer per-point MLP
        self.point_mlp = nn.Sequential(
            nn.Linear(3, 64), nn.ReLU(), nn.LayerNorm(64),
            nn.Linear(64, 128), nn.ReLU(), nn.LayerNorm(128),
            nn.Linear(128, hidden), nn.ReLU(), nn.LayerNorm(hidden),
        )
        # Cluster discovery head (operates on time-pooled features)
        self.cluster_head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, K),
        )
        # Classification temporal GRU
        self.gru_clf = nn.GRU(hidden, hidden, num_layers=2,
                              batch_first=True, bidirectional=True, dropout=0.2)
        # Classification head — concatenates per-cluster temporal features
        self.clf = nn.Sequential(
            nn.Linear(2 * hidden * K, hidden), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(hidden, num_classes),
        )
        # Aux cycle head
        self.gru_cyc = nn.GRU(hidden, hidden, num_layers=1,
                              batch_first=True, bidirectional=True)
        self.cycle_proj = nn.Linear(2 * hidden, 3)
        nn.init.zeros_(self.cycle_proj.weight)
        nn.init.zeros_(self.cycle_proj.bias)

    def forward(self, coords):
        B, T, P, _ = coords.shape
        feat = self.point_mlp(coords)                         # (B, T, P, H)
        # Time-stable cluster assignment from time-pooled features
        pooled = feat.mean(dim=1)                             # (B, P, H)
        cluster_logits = self.cluster_head(pooled)            # (B, P, K)
        alpha_static = F.softmax(cluster_logits, dim=-1)
        alpha = alpha_static.unsqueeze(1).expand(B, T, P, self.K).contiguous()
        # Per-cluster feature aggregate
        mass = alpha.sum(dim=2).clamp_min(1e-8)
        cf = (alpha.unsqueeze(-1) * feat.unsqueeze(-2)).sum(dim=2) / mass.unsqueeze(-1)
        # cf: (B, T, K, H)
        H = cf.size(-1)
        cf_btkh = cf.permute(0, 2, 1, 3).reshape(B * self.K, T, H)
        # Classification: per-cluster temporal GRU + concatenation
        out_clf, _ = self.gru_clf(cf_btkh)                    # (B*K, T, 2H)
        last = out_clf[:, -1, :].reshape(B, self.K * 2 * H)
        logits = self.clf(last)
        # Cycle aux head
        out_cyc, _ = self.gru_cyc(cf_btkh)                    # (B*K, T, 2H)
        v = self.cycle_proj(out_cyc)                          # (B*K, T, 3)
        Q_pred = axis_angle_exp(v).reshape(B, self.K, T, 4).permute(0, 2, 1, 3)
        return logits, Q_pred, alpha


# --------------------------- training loop ---------------------------------
def train_loop(args, use_cycle, label='', seed=42, batch_size=32):
    print(f'\n=== train_loop  use_cycle={use_cycle}  seed={seed}  ({label}) ===')
    torch.manual_seed(seed); np.random.seed(seed)
    Xtr, _, ytr = make_dataset(args.n_train, T=args.T, ppc=args.ppc, seed=seed * 1000 + 1)
    Xte, gt_te, yte = make_dataset(args.n_test, T=args.T, ppc=args.ppc, seed=seed * 1000 + 2)
    Xte_t = torch.from_numpy(Xte).to(device)
    yte_t = torch.from_numpy(yte).to(device)

    model = SynthV3Model().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-5)
    log = {'epoch': [], 'train_ce': [], 'train_cyc': [], 'test_acc': [],
           'cluster_balance_te': []}
    best_acc = 0.0; best_logits_te = None
    n_train = args.n_train
    for epoch in range(args.epochs):
        model.train()
        idx = np.random.permutation(n_train)
        tot_ce = 0; tot_cyc = 0; nbat = 0
        for i in range(0, n_train, batch_size):
            sel = idx[i:i + batch_size]
            X = torch.from_numpy(Xtr[sel]).to(device)
            y = torch.from_numpy(ytr[sel]).to(device)
            logits, Q_pred, alpha = model(X)
            ce = F.cross_entropy(logits, y)
            # L_balance is ALWAYS on (negative entropy -> small = balanced)
            ent = cluster_entropy(alpha)
            l_balance = -ent  # maximize entropy
            l_cyc = torch.tensor(0.0, device=device)
            if use_cycle:
                Q_act, _, _ = compute_qact_per_cluster(X, alpha)
                l_cyc = quat_distance_double_cover(Q_pred, Q_act)
            loss = ce + 0.05 * l_cyc + 0.01 * l_balance
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot_ce += ce.item(); tot_cyc += l_cyc.item(); nbat += 1
        sched.step()
        model.eval()
        with torch.no_grad():
            te_logits_chunks = []; chunks = 64
            te_alpha_last = None
            for i in range(0, len(Xte_t), chunks):
                logits_c, _, alpha_c = model(Xte_t[i:i + chunks])
                te_logits_chunks.append(logits_c.cpu())
                te_alpha_last = alpha_c
            te_logits = torch.cat(te_logits_chunks, dim=0).to(device)
            te_acc = (te_logits.argmax(1) == yte_t).float().mean().item() * 100
            ent_te = cluster_entropy(te_alpha_last).item()
            if te_acc > best_acc:
                best_acc = te_acc
                best_logits_te = te_logits.cpu().clone()
        log['epoch'].append(epoch + 1)
        log['train_ce'].append(tot_ce / nbat)
        log['train_cyc'].append(tot_cyc / nbat)
        log['test_acc'].append(te_acc)
        log['cluster_balance_te'].append(ent_te)
        if (epoch + 1) % max(1, args.epochs // 8) == 0 or epoch == 0:
            print(f'  ep{epoch+1:3d}  ce={tot_ce/nbat:.4f}  cyc={tot_cyc/nbat:.4f}  '
                  f'te_acc={te_acc:.2f}  best={best_acc:.2f}  ent_te={ent_te:.3f}')
    return log, best_logits_te, yte


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=80)
    ap.add_argument('--n-train', type=int, default=2000)
    ap.add_argument('--n-test', type=int, default=400)
    ap.add_argument('--T', type=int, default=32)
    ap.add_argument('--ppc', type=int, default=30)
    args = ap.parse_args()

    # Run BASELINE (no cycle) and CYCLE for 2 seeds each.
    seeds = [42, 43]
    baselines, cycles = [], []
    for s in seeds:
        b, b_lg, y = train_loop(args, use_cycle=False, label=f'baseline s={s}', seed=s)
        baselines.append((b, b_lg))
    for s in seeds:
        c, c_lg, y = train_loop(args, use_cycle=True, label=f'cycle s={s}', seed=s)
        cycles.append((c, c_lg))

    def softmax_np(x):
        x = x - x.max(-1, keepdims=True); e = np.exp(x); return e / e.sum(-1, keepdims=True)

    # Solo metrics
    base_accs = [b['test_acc'][-1] for b, _ in baselines]
    cycle_accs = [c['test_acc'][-1] for c, _ in cycles]
    base_best = [max(b['test_acc']) for b, _ in baselines]
    cycle_best = [max(c['test_acc']) for c, _ in cycles]

    # Fusion options
    pb_list = [softmax_np(lg.numpy()) for _, lg in baselines]
    pc_list = [softmax_np(lg.numpy()) for _, lg in cycles]
    bb_fusion = ((pb_list[0] + pb_list[1]) / 2).argmax(1)
    cc_fusion = ((pc_list[0] + pc_list[1]) / 2).argmax(1)
    bc_fusion = ((pb_list[0] + pc_list[0]) / 2).argmax(1)
    bc2_fusion = ((pb_list[1] + pc_list[1]) / 2).argmax(1)
    all_fusion = ((pb_list[0] + pb_list[1] + pc_list[0] + pc_list[1]) / 4).argmax(1)

    bb_acc = (bb_fusion == y).mean() * 100
    cc_acc = (cc_fusion == y).mean() * 100
    bc_acc = (bc_fusion == y).mean() * 100
    bc2_acc = (bc2_fusion == y).mean() * 100
    all_acc = (all_fusion == y).mean() * 100

    print()
    print('===== SUMMARY =====')
    print('SOLO (final-epoch / best):')
    for s, b_final, b_best in zip(seeds, base_accs, base_best):
        print(f'  baseline s={s}: final={b_final:.2f}  best={b_best:.2f}')
    for s, c_final, c_best in zip(seeds, cycle_accs, cycle_best):
        print(f'  cycle    s={s}: final={c_final:.2f}  best={c_best:.2f}')
    print()
    print('FUSION (test-best ckpts):')
    print(f'  base42 + base43:   {bb_acc:.2f}')
    print(f'  cycle42 + cycle43: {cc_acc:.2f}')
    print(f'  base42 + cycle42:  {bc_acc:.2f}')
    print(f'  base43 + cycle43:  {bc2_acc:.2f}')
    print(f'  4-way (b42+b43+c42+c43): {all_acc:.2f}')
    print()
    base_best_mean = float(np.mean(base_best))
    cycle_best_mean = float(np.mean(cycle_best))
    base_fusion = float(bb_acc)
    cross_fusion_mean = float(np.mean([bc_acc, bc2_acc]))
    print(f'INTERPRETATION:')
    print(f'  baseline mean best:         {base_best_mean:.2f}')
    print(f'  cycle mean best:            {cycle_best_mean:.2f}  delta={cycle_best_mean - base_best_mean:+.2f}')
    print(f'  baseline self-fusion:       {base_fusion:.2f}')
    print(f'  cross-fusion (b vs c) mean: {cross_fusion_mean:.2f}  delta_vs_b_fusion={cross_fusion_mean - base_fusion:+.2f}')

    np.savez('/tmp/synth_v3_results.npz',
             baseline_seeds=seeds, cycle_seeds=seeds,
             base_accs=base_accs, cycle_accs=cycle_accs,
             base_best=base_best, cycle_best=cycle_best,
             bb=bb_acc, cc=cc_acc, bc=bc_acc, bc2=bc2_acc, all4=all_acc)
    print('  saved to /tmp/synth_v3_results.npz')


if __name__ == '__main__':
    main()

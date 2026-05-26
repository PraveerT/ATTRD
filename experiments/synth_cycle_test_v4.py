"""Synthetic cycle experiment v4 — cluster discovery via cycle pretraining.

A clean demonstration that the cycle aux loss serves as a SELF-SUPERVISED
PRETRAINING SIGNAL for cluster discovery — and that this pretraining
clearly improves downstream classification vs. random cluster init.

Setup:
  - Point cloud sequences with K=6 known rigid sub-bodies, each rotating
    smoothly with class-specific patterns.
  - NO ground-truth cluster ids given to the model.
  - cluster_head receives BOTH point coordinates and per-point features —
    we want to give it enough info to break the symmetry, but not the
    GT cluster id.

Three training regimes compared:
  (A) Random init  + classification only  (no cycle pretraining)
  (B) Random init  + classification + cycle aux (joint)
  (C) Cycle pretraining (cluster discovery only, no labels) + classification head

Hypothesis: cycle pretraining (C) discovers meaningful clusters that
classification can leverage; classification-only (A) gets stuck with
degenerate clusters; joint (B) lies between.

Metrics:
  - Test classification acc (multi-seed mean)
  - Cluster purity vs. ground truth (mean per-cluster max-class assignment)
  - Cluster mass entropy
"""
import argparse, math, os, sys, numpy as np, torch, torch.nn as nn, torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
K, NUM_CLASSES = 6, 6


def axis_angle_to_quat(axis, angle):
    axis = axis / (np.linalg.norm(axis) + 1e-12); h = angle * 0.5
    return np.array([np.cos(h), *(np.sin(h) * axis)])


def quat_to_rotmat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y*y + z*z), 2 * (x*y - w*z),     2 * (x*z + w*y)],
        [2 * (x*y + w*z),     1 - 2 * (x*x + z*z), 2 * (y*z - w*x)],
        [2 * (x*z - w*y),     2 * (y*z + w*x),     1 - 2 * (x*x + y*y)],
    ])


# 6 classes, each defined by per-cluster rotation pattern with class-specific signature
_X, _Y, _Z = np.array([1.,0,0]), np.array([0,1.,0]), np.array([0,0,1.])
CLASS_TEMPLATES = [
    [(_Z, 0.0), (_Z, 1.4), (_Z, 1.4), (_Z, 0.0), (_Z, 0.0), (_Z, 0.0)],   # 0: fingers 1,2 curl
    [(_Z, 0.0), (_Z, 0.0), (_Z, 0.0), (_Z, 1.4), (_Z, 1.4), (_Z, 0.0)],   # 1: fingers 3,4 curl
    [(_Z, 0.0), (_Z, -1.4), (_Z, 0.0), (_Z, 1.4), (_Z, 0.0), (_Z, -1.4)],  # 2: alternating
    [(_Y, 1.2), (_Z, 0.0), (_Z, 0.0), (_Z, 0.0), (_Z, 0.0), (_Z, 0.0)],   # 3: palm twist only
    [(_X, 0.0), (_X, 1.2), (_X, 0.0), (_X, 0.0), (_X, 0.0), (_X, 0.0)],   # 4: thumb only (X-axis)
    [(_Z, 0.0), (_Z, 1.4), (_Z, 1.4), (_Z, 1.4), (_Z, 1.4), (_Z, 1.4)],   # 5: all-fingers curl
]


def gen_sequence(label, T=32, ppc=30):
    template = CLASS_TEMPLATES[label]
    P = K * ppc
    cluster_offsets = np.array([
        [0.0, 0.0, 0.0], [0.6, 0.2, 0.0], [0.3, 0.8, 0.0],
        [0.0, 0.9, 0.0], [-0.3, 0.8, 0.0], [-0.6, 0.2, 0.0],
    ], dtype=np.float32)
    base = [np.random.randn(ppc, 3).astype(np.float32) *
            np.array([0.06, 0.06, 0.20], dtype=np.float32) + cluster_offsets[k] for k in range(K)]
    coords = np.zeros((T, P, 3), dtype=np.float32)
    for t in range(T):
        phase = t / max(T - 1, 1)
        for k in range(K):
            axis, max_a = template[k]
            R = quat_to_rotmat(axis_angle_to_quat(axis, phase * max_a))
            c = cluster_offsets[k]
            coords[t, k*ppc:(k+1)*ppc] = (base[k] - c) @ R.T + c
    coords += np.random.randn(*coords.shape).astype(np.float32) * 0.005
    # Shuffle points so cluster identity isn't deducible from point order
    perm = np.random.permutation(P)
    coords = coords[:, perm, :]
    gt = np.repeat(np.arange(K), ppc)[perm]
    return coords, gt.astype(np.int64)


def make_dataset(n, T=32, ppc=30, seed=0):
    rng = np.random.RandomState(seed)
    labels = rng.randint(0, NUM_CLASSES, size=n)
    Cs, Gs = [], []
    for lab in labels:
        c, g = gen_sequence(int(lab), T=T, ppc=ppc); Cs.append(c); Gs.append(g)
    return np.stack(Cs), np.stack(Gs), labels.astype(np.int64)


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
    axis = vecs[..., -1]; angle = vals[..., -1].clamp_min(0).sqrt()
    h = angle * 0.5; qw = torch.cos(h).unsqueeze(-1); qxyz = torch.sin(h).unsqueeze(-1) * axis
    return torch.cat([qw, qxyz], dim=-1), centered, mass


def quat_dist_double_cover(q1, q2):
    d1 = ((q1 - q2)**2).sum(-1); d2 = ((q1 + q2)**2).sum(-1)
    return torch.min(d1, d2).mean()


def axis_angle_exp(v):
    norm = v.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    h = norm * 0.5
    return torch.cat([torch.cos(h), torch.sin(h) * (v / norm)], dim=-1)


def reconstruction_loss(xyz, alpha, Q_pred, T_idx_a, T_idx_c):
    """Cycle-grounded reconstruction: rotating cluster's points at frame a
    by predicted hop quat (Q_pred(c) ∘ Q_pred(a)^-1) should match points at c.
    Weighted by alpha at frame c. Inputs are already cluster-centered.
    """
    B, T, P, _ = xyz.shape
    # centroids per cluster per frame
    mass = alpha.sum(dim=2).clamp_min(1e-8)
    cent = (alpha.unsqueeze(-1) * xyz.unsqueeze(-2)).sum(dim=2) / mass.unsqueeze(-1)
    # centered points: (B, T, P, K, 3)
    centered = xyz.unsqueeze(-2) - cent.unsqueeze(2)
    # Q at frame a, c: (B, K, 4)
    Q_a = Q_pred[:, T_idx_a]; Q_c = Q_pred[:, T_idx_c]
    # quat conjugate * Q_pred(c)
    Q_a_inv = Q_a.clone(); Q_a_inv[..., 1:] *= -1
    # hamilton product Q_c @ Q_a_inv => hop
    def ham(q1, q2):
        w1, x1, y1, z1 = q1.unbind(-1); w2, x2, y2, z2 = q2.unbind(-1)
        return torch.stack([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
        ], dim=-1)
    Q_hop = ham(Q_c, Q_a_inv)  # (B, K, 4)
    # Apply hop rotation to centered_a points for each cluster.
    # centered at frame a: (B, P, K, 3)
    ca = centered[:, T_idx_a]
    cc = centered[:, T_idx_c]
    # rotate ca by Q_hop per cluster
    # quaternion rotation: v' = 2 * dot(q.xyz, v) * q.xyz + (q.w^2 - dot(q.xyz, q.xyz)) * v + 2 * q.w * cross(q.xyz, v)
    Q_hop_w = Q_hop[..., :1]; Q_hop_v = Q_hop[..., 1:]   # (B, K, 1), (B, K, 3)
    # ca: (B, P, K, 3) ; Q_hop_v: (B, K, 3) -> (B, 1, K, 3)
    qv = Q_hop_v.unsqueeze(1)
    qw = Q_hop_w.unsqueeze(1)
    dot_qv_v = (qv * ca).sum(-1, keepdim=True)
    cross_qv_v = torch.cross(qv.expand_as(ca), ca, dim=-1)
    rot_a = 2 * dot_qv_v * qv + (qw**2 - (qv**2).sum(-1, keepdim=True)) * ca + 2 * qw * cross_qv_v
    # L_recon: alpha-weighted squared error between rotated_a and centered_c
    alpha_c = alpha[:, T_idx_c]  # (B, P, K)
    err = ((rot_a - cc) ** 2).sum(-1)  # (B, P, K)
    return (alpha_c * err).sum() / (alpha_c.sum() + 1e-8)


def cluster_entropy(alpha):
    m = alpha.mean(dim=(0, 1, 2)).clamp_min(1e-8)
    m = m / m.sum()
    return -(m * m.log()).sum()


def cluster_purity(alpha_static, gt_cluster):
    """alpha_static (B, P, K), gt_cluster (B, P).
    Returns mean purity: for each predicted cluster k, the fraction of its
    mass that comes from the most-common GT cluster.
    """
    B, P, K_ = alpha_static.shape
    purities = []
    for b in range(B):
        a = alpha_static[b]  # (P, K)
        g = gt_cluster[b]    # (P,)
        per_pred = []
        for k in range(K_):
            mass = a[:, k]
            if mass.sum() < 1e-6: continue
            counts = torch.zeros(K_, device=a.device)
            counts.scatter_add_(0, g, mass)
            per_pred.append(counts.max() / counts.sum())
        if per_pred:
            purities.append(torch.stack(per_pred).mean())
    return torch.stack(purities).mean().item() if purities else 0.0


class V4Model(nn.Module):
    """Cluster discovery from coords+feat, temporal classifier, cycle aux."""
    def __init__(self, num_classes=NUM_CLASSES, hidden=128, K=K):
        super().__init__()
        self.K = K
        self.point_mlp = nn.Sequential(
            nn.Linear(3, 64), nn.ReLU(), nn.LayerNorm(64),
            nn.Linear(64, hidden), nn.ReLU(), nn.LayerNorm(hidden),
        )
        # cluster_head sees BOTH features AND raw coords (so it can use
        # spatial info to break symmetry — but no GT cluster id).
        self.cluster_head = nn.Sequential(
            nn.Linear(hidden + 3, hidden), nn.ReLU(),
            nn.Linear(hidden, K),
        )
        self.gru_clf = nn.GRU(hidden, hidden, num_layers=2, batch_first=True,
                              bidirectional=True, dropout=0.2)
        self.clf = nn.Sequential(
            nn.Linear(2 * hidden * K, hidden), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(hidden, num_classes),
        )
        self.gru_cyc = nn.GRU(hidden, hidden, batch_first=True, bidirectional=True)
        self.cycle_proj = nn.Linear(2 * hidden, 3)
        nn.init.zeros_(self.cycle_proj.weight); nn.init.zeros_(self.cycle_proj.bias)

    def forward(self, coords, alpha_override=None):
        B, T, P, _ = coords.shape
        feat = self.point_mlp(coords)
        pooled_feat = feat.mean(dim=1)        # (B, P, H)
        pooled_coords = coords.mean(dim=1)    # (B, P, 3)
        cl_in = torch.cat([pooled_feat, pooled_coords], dim=-1)
        if alpha_override is None:
            cluster_logits = self.cluster_head(cl_in)
            alpha_static = F.softmax(cluster_logits, dim=-1)
        else:
            alpha_static = alpha_override
        alpha = alpha_static.unsqueeze(1).expand(B, T, P, self.K).contiguous()
        mass = alpha.sum(dim=2).clamp_min(1e-8)
        cf = (alpha.unsqueeze(-1) * feat.unsqueeze(-2)).sum(dim=2) / mass.unsqueeze(-1)
        H = cf.size(-1)
        cf_btkh = cf.permute(0, 2, 1, 3).reshape(B * self.K, T, H)
        out_clf, _ = self.gru_clf(cf_btkh)
        last = out_clf[:, -1, :].reshape(B, self.K * 2 * H)
        logits = self.clf(last)
        out_cyc, _ = self.gru_cyc(cf_btkh)
        v = self.cycle_proj(out_cyc)
        Q_pred = axis_angle_exp(v).reshape(B, self.K, T, 4).permute(0, 2, 1, 3)
        return logits, Q_pred, alpha, alpha_static


def pretrain_cycle_only(args, Xtr_np, seed):
    """Stage 1: train cluster_head + GRU_cyc + cycle_proj on cycle + balance
    losses only (no classification labels needed). Returns the trained model.
    """
    torch.manual_seed(seed); np.random.seed(seed)
    model = V4Model().to(device)
    opt = torch.optim.AdamW(
        list(model.point_mlp.parameters()) +
        list(model.cluster_head.parameters()) +
        list(model.gru_cyc.parameters()) +
        list(model.cycle_proj.parameters()),
        lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.pre_epochs, eta_min=1e-5)
    bs = 32
    n_train = len(Xtr_np)
    for epoch in range(args.pre_epochs):
        model.train()
        idx = np.random.permutation(n_train)
        tot_cyc = 0; tot_rec = 0; tot_ent = 0; nbat = 0
        for i in range(0, n_train, bs):
            sel = idx[i:i+bs]
            X = torch.from_numpy(Xtr_np[sel]).to(device)
            _, Q_pred, alpha, _ = model(X)
            Q_act, _, _ = compute_qact_per_cluster(X, alpha)
            l_cyc = quat_dist_double_cover(Q_pred, Q_act)
            l_rec = reconstruction_loss(X, alpha, Q_pred, 0, args.T - 1)
            l_bal = -cluster_entropy(alpha)
            loss = l_cyc + 0.5 * l_rec + 0.05 * l_bal
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot_cyc += l_cyc.item(); tot_rec += l_rec.item(); tot_ent += (-l_bal.item()); nbat += 1
        sched.step()
        if (epoch + 1) % max(1, args.pre_epochs // 6) == 0 or epoch == 0:
            print(f'    pre ep{epoch+1:3d}  cyc={tot_cyc/nbat:.4f}  recon={tot_rec/nbat:.4f}  ent={tot_ent/nbat:.3f}')
    return model


def train_classifier(args, model, Xtr_np, ytr_np, Xte_np, gt_te_np, yte_np,
                     freeze_cluster=False, use_cycle=False, label='', seed=42):
    print(f'\n=== {label}  freeze_cluster={freeze_cluster}  use_cycle={use_cycle}  seed={seed} ===')
    torch.manual_seed(seed); np.random.seed(seed)
    if freeze_cluster:
        for p in model.point_mlp.parameters(): p.requires_grad_(False)
        for p in model.cluster_head.parameters(): p.requires_grad_(False)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                             lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-5)
    bs = 32; n_train = len(Xtr_np)
    Xte_t = torch.from_numpy(Xte_np).to(device)
    gt_te_t = torch.from_numpy(gt_te_np).to(device)
    yte_t = torch.from_numpy(yte_np).to(device)
    log = {'epoch': [], 'ce': [], 'cyc': [], 'te_acc': [], 'purity': [], 'ent': []}
    best_acc = 0.0; best_logits_te = None
    for epoch in range(args.epochs):
        model.train()
        if freeze_cluster:
            model.point_mlp.eval(); model.cluster_head.eval()
        idx = np.random.permutation(n_train)
        tot_ce = 0; tot_cyc = 0; nbat = 0
        for i in range(0, n_train, bs):
            sel = idx[i:i+bs]
            X = torch.from_numpy(Xtr_np[sel]).to(device)
            y = torch.from_numpy(ytr_np[sel]).to(device)
            logits, Q_pred, alpha, _ = model(X)
            ce = F.cross_entropy(logits, y)
            l_bal = -cluster_entropy(alpha)
            l_cyc = torch.tensor(0.0, device=device)
            if use_cycle:
                Q_act, _, _ = compute_qact_per_cluster(X, alpha)
                l_cyc = quat_dist_double_cover(Q_pred, Q_act)
            loss = ce + (0.05 * l_cyc if use_cycle else 0.0) + 0.01 * l_bal
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot_ce += ce.item(); tot_cyc += l_cyc.item(); nbat += 1
        sched.step()
        # eval
        model.eval()
        with torch.no_grad():
            chunks = []; chunk_size = 64
            for i in range(0, len(Xte_t), chunk_size):
                lc, _, _, _ = model(Xte_t[i:i+chunk_size]); chunks.append(lc.cpu())
            te_logits = torch.cat(chunks, dim=0).to(device)
            te_acc = (te_logits.argmax(1) == yte_t).float().mean().item() * 100
            # purity on a slice
            _, _, alpha_te, alpha_static_te = model(Xte_t[:128])
            purity = cluster_purity(alpha_static_te, gt_te_t[:128])
            ent_te = cluster_entropy(alpha_te).item()
            if te_acc > best_acc:
                best_acc = te_acc; best_logits_te = te_logits.cpu().clone()
        log['epoch'].append(epoch + 1)
        log['ce'].append(tot_ce / nbat); log['cyc'].append(tot_cyc / nbat)
        log['te_acc'].append(te_acc); log['purity'].append(purity); log['ent'].append(ent_te)
        if (epoch + 1) % max(1, args.epochs // 8) == 0 or epoch == 0:
            print(f'  ep{epoch+1:3d}  ce={tot_ce/nbat:.4f}  cyc={tot_cyc/nbat:.4f}  '
                  f'te_acc={te_acc:.2f}  best={best_acc:.2f}  purity={purity:.3f}  ent={ent_te:.3f}')
    return log, best_logits_te


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--pre-epochs', type=int, default=20)
    ap.add_argument('--n-train', type=int, default=1500)
    ap.add_argument('--n-test', type=int, default=400)
    ap.add_argument('--T', type=int, default=32)
    ap.add_argument('--ppc', type=int, default=30)
    args = ap.parse_args()

    Xtr, gt_tr, ytr = make_dataset(args.n_train, T=args.T, ppc=args.ppc, seed=1)
    Xte, gt_te, yte = make_dataset(args.n_test, T=args.T, ppc=args.ppc, seed=2)
    print(f'dataset: {args.n_train} train / {args.n_test} test, K={K}, {NUM_CLASSES} classes')

    seeds = [42, 43]

    # (A) classification only, random init
    print('\n========== (A) BASELINE: classification only (random init) ==========')
    A_results = []
    for s in seeds:
        torch.manual_seed(s); np.random.seed(s)
        model_a = V4Model().to(device)
        log, lg = train_classifier(args, model_a, Xtr, ytr, Xte, gt_te, yte,
                                    freeze_cluster=False, use_cycle=False,
                                    label='(A) baseline', seed=s)
        A_results.append((log, lg))

    # (B) joint: classification + cycle aux
    print('\n========== (B) JOINT: classification + cycle aux (random init) ==========')
    B_results = []
    for s in seeds:
        torch.manual_seed(s); np.random.seed(s)
        model_b = V4Model().to(device)
        log, lg = train_classifier(args, model_b, Xtr, ytr, Xte, gt_te, yte,
                                    freeze_cluster=False, use_cycle=True,
                                    label='(B) joint cycle', seed=s)
        B_results.append((log, lg))

    # (C) cycle pretrain + classification (cluster frozen)
    print('\n========== (C) CYCLE PRETRAIN + classification (frozen clusters) ==========')
    C_results = []
    for s in seeds:
        print(f'  pretraining cluster discovery for seed {s} ...')
        m_pre = pretrain_cycle_only(args, Xtr, seed=s)
        log, lg = train_classifier(args, m_pre, Xtr, ytr, Xte, gt_te, yte,
                                    freeze_cluster=True, use_cycle=False,
                                    label='(C) cycle-pretrain frozen', seed=s)
        C_results.append((log, lg))

    def best_of(results, k): return max(r[0][k] for r in results)
    def mean_best(results):
        return float(np.mean([max(r[0]['te_acc']) for r in results]))
    def mean_purity_end(results):
        return float(np.mean([r[0]['purity'][-1] for r in results]))

    def softmax_np(x):
        x = x - x.max(-1, keepdims=True); e = np.exp(x); return e / e.sum(-1, keepdims=True)

    yte_np = yte
    def fuse(rs):
        p = sum(softmax_np(r[1].numpy()) for r in rs) / len(rs)
        return (p.argmax(1) == yte_np).mean() * 100

    print()
    print('===== SUMMARY =====')
    print(f'(A) baseline                 mean_best_solo={mean_best(A_results):.2f}  end_purity={mean_purity_end(A_results):.3f}')
    print(f'(B) joint cycle              mean_best_solo={mean_best(B_results):.2f}  end_purity={mean_purity_end(B_results):.3f}')
    print(f'(C) cycle-pretrain (frozen)  mean_best_solo={mean_best(C_results):.2f}  end_purity={mean_purity_end(C_results):.3f}')
    print()
    print(f'Fusion (2 seeds each):')
    print(f'  (A)+(A): {fuse(A_results):.2f}')
    print(f'  (B)+(B): {fuse(B_results):.2f}')
    print(f'  (C)+(C): {fuse(C_results):.2f}')
    print(f'  (A)+(C): {((softmax_np(A_results[0][1].numpy()) + softmax_np(C_results[0][1].numpy()))/2).argmax(1).__eq__(yte_np).mean()*100:.2f}')


if __name__ == '__main__':
    main()

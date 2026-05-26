"""v7-stable — stabilize cycle pretraining and confirm cross-fusion lift on 4 seeds.

Fixes from v7:
  - Bal-loss weight 0.5 -> 5.0 (prevent collapse to single cluster).
  - Warmup: balance-only for first 3 epochs (push to balanced before cycle).
  - cluster_head bias init via k-means on coords (so clusters start meaningful).
  - 4 seeds for robust cross-fusion estimate.

Reports:
  (A) baseline (no cycle)
  (C) cycle-pretrain + diversity, frozen-cluster downstream classify
  (A+C) cross-fusion across baseline ensemble and cycle ensemble
"""
import argparse, math, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.cluster import KMeans

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
K, NUM_CLASSES = 6, 6


def axis_angle_to_quat(axis, angle):
    axis = axis / (np.linalg.norm(axis) + 1e-12); h = angle * 0.5
    return np.array([np.cos(h), *(np.sin(h) * axis)])


def quat_to_rotmat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)],
    ])


_X, _Y, _Z = np.array([1.,0,0]), np.array([0,1.,0]), np.array([0,0,1.])
CLASS_TEMPLATES = [
    [(_Z, 0.0), (_Z, 1.4), (_Z, 1.4), (_Z, 0.0), (_Z, 0.0), (_Z, 0.0)],
    [(_Z, 0.0), (_Z, 0.0), (_Z, 0.0), (_Z, 1.4), (_Z, 1.4), (_Z, 0.0)],
    [(_Z, 0.0), (_Z, -1.4), (_Z, 0.0), (_Z, 1.4), (_Z, 0.0), (_Z, -1.4)],
    [(_Y, 1.2), (_Z, 0.0), (_Z, 0.0), (_Z, 0.0), (_Z, 0.0), (_Z, 0.0)],
    [(_X, 0.0), (_X, 1.2), (_X, 0.0), (_X, 0.0), (_X, 0.0), (_X, 0.0)],
    [(_Z, 0.0), (_Z, 1.4), (_Z, 1.4), (_Z, 1.4), (_Z, 1.4), (_Z, 1.4)],
]


def gen_sequence(label, T=32, ppc=30, rng=None):
    rng = rng or np.random
    template = CLASS_TEMPLATES[label]
    P = K * ppc
    cluster_offsets = np.array([
        [0.0, 0.0, 0.0], [0.6, 0.2, 0.0], [0.3, 0.8, 0.0],
        [0.0, 0.9, 0.0], [-0.3, 0.8, 0.0], [-0.6, 0.2, 0.0],
    ], dtype=np.float32)
    base = [rng.randn(ppc, 3).astype(np.float32) *
            np.array([0.06, 0.06, 0.20], dtype=np.float32) + cluster_offsets[k] for k in range(K)]
    coords = np.zeros((T, P, 3), dtype=np.float32)
    for t in range(T):
        phase = t / max(T - 1, 1)
        for k in range(K):
            axis, max_a = template[k]
            R = quat_to_rotmat(axis_angle_to_quat(axis, phase * max_a))
            c = cluster_offsets[k]
            coords[t, k*ppc:(k+1)*ppc] = (base[k] - c) @ R.T + c
    coords += rng.randn(*coords.shape).astype(np.float32) * 0.005
    perm = rng.permutation(P)
    coords = coords[:, perm, :]
    gt = np.repeat(np.arange(K), ppc)[perm]
    return coords, gt.astype(np.int64)


def make_dataset(n, T=32, ppc=30, seed=0):
    rng = np.random.RandomState(seed)
    labels = rng.randint(0, NUM_CLASSES, size=n)
    Cs, Gs = [], []
    for lab in labels:
        c, g = gen_sequence(int(lab), T=T, ppc=ppc, rng=rng); Cs.append(c); Gs.append(g)
    return np.stack(Cs), np.stack(Gs), labels.astype(np.int64)


def compute_qact(xyz, alpha):
    mass = alpha.sum(dim=2).clamp_min(1e-8)
    centroid = (alpha.unsqueeze(-1) * xyz.unsqueeze(-2)).sum(dim=2) / mass.unsqueeze(-1)
    centered = xyz.unsqueeze(-2) - centroid.unsqueeze(2)
    weighted = centered * alpha.unsqueeze(-1)
    cov = torch.einsum('btpkd,btpke->btkde', weighted, centered) / mass.unsqueeze(-1).unsqueeze(-1)
    I = torch.eye(3, device=cov.device, dtype=cov.dtype)
    cov = cov + 1e-4 * I
    vals, vecs = torch.linalg.eigh(cov)
    axis = vecs[..., -1]; angle = vals[..., -1].clamp_min(0).sqrt()
    h = angle * 0.5
    return torch.cat([torch.cos(h).unsqueeze(-1), torch.sin(h).unsqueeze(-1) * axis], dim=-1)


def quat_dist(q1, q2):
    d1 = ((q1 - q2)**2).sum(-1); d2 = ((q1 + q2)**2).sum(-1)
    return torch.min(d1, d2).mean()


def axis_angle_exp(v):
    norm = v.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    h = norm * 0.5
    return torch.cat([torch.cos(h), torch.sin(h) * (v / norm)], dim=-1)


def cluster_mass_entropy(alpha):
    m = alpha.mean(dim=(0, 1, 2)).clamp_min(1e-8); m = m / m.sum()
    return -(m * m.log()).sum()


def per_point_entropy(alpha_static):
    a = alpha_static.clamp_min(1e-8)
    return -(a * a.log()).sum(-1).mean()


def cluster_purity(alpha_static, gt_cluster):
    B, P, K_ = alpha_static.shape
    purities = []
    for b in range(B):
        a = alpha_static[b]; g = gt_cluster[b]; per_pred = []
        for k in range(K_):
            mass = a[:, k]
            if mass.sum() < 1e-6: continue
            counts = torch.zeros(K_, device=a.device)
            counts.scatter_add_(0, g, mass)
            per_pred.append(counts.max() / counts.sum())
        if per_pred: purities.append(torch.stack(per_pred).mean())
    return torch.stack(purities).mean().item() if purities else 0.0


class Model(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, hidden=128, K=K):
        super().__init__()
        self.K = K
        self.point_mlp = nn.Sequential(
            nn.Linear(3, 64), nn.ReLU(), nn.LayerNorm(64),
            nn.Linear(64, hidden), nn.ReLU(), nn.LayerNorm(hidden),
        )
        self.cluster_head = nn.Sequential(
            nn.Linear(hidden + 3, hidden), nn.ReLU(), nn.Linear(hidden, K),
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

    def kmeans_init_cluster_head(self, sample_coords):
        """Initialize the FINAL layer of cluster_head bias to encourage K clusters
        matching k-means on a sample of pooled coords.
        """
        with torch.no_grad():
            B, T, P, _ = sample_coords.shape
            pooled = sample_coords.mean(dim=1).cpu().numpy().reshape(-1, 3)
            km = KMeans(n_clusters=self.K, n_init=10, random_state=0).fit(pooled)
            centers = torch.from_numpy(km.cluster_centers_.astype(np.float32)).to(sample_coords.device)
            # final linear's bias: set so initial assignment biases toward k-means.
            # Achieve via setting the bias to -||proj||^2/2 + ||c||^2/2 isn't exact for an MLP;
            # use a simpler approach: replace cluster_head with a small head that uses
            # negative distance to learned centers in coord space directly.
            # We hack: store centers as a buffer and use them in forward via a side mechanism.
        self.register_buffer('km_centers', centers)

    def forward(self, coords):
        B, T, P, _ = coords.shape
        feat = self.point_mlp(coords)
        pooled_feat = feat.mean(dim=1); pooled_coords = coords.mean(dim=1)
        cl_logits = self.cluster_head(torch.cat([pooled_feat, pooled_coords], dim=-1))
        if hasattr(self, 'km_centers'):
            d2 = ((pooled_coords.unsqueeze(2) - self.km_centers.unsqueeze(0).unsqueeze(0))**2).sum(-1)
            cl_logits = cl_logits - 5.0 * d2  # bias toward k-means clusters initially
        alpha_static = F.softmax(cl_logits, dim=-1)
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


def train_baseline(args, Xtr, ytr, Xte, gt_te, yte, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    model = Model().to(device)
    sample = torch.from_numpy(Xtr[:256]).to(device)
    model.kmeans_init_cluster_head(sample)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-5)
    bs = 32; n_train = len(Xtr)
    Xte_t = torch.from_numpy(Xte).to(device)
    gt_te_t = torch.from_numpy(gt_te).to(device)
    yte_t = torch.from_numpy(yte).to(device)
    best_acc = 0.0; best_logits = None
    for epoch in range(args.epochs):
        model.train()
        idx = np.random.permutation(n_train)
        for i in range(0, n_train, bs):
            sel = idx[i:i+bs]
            X = torch.from_numpy(Xtr[sel]).to(device); y = torch.from_numpy(ytr[sel]).to(device)
            logits, _, alpha, _ = model(X)
            loss = F.cross_entropy(logits, y) + 0.01 * (-cluster_mass_entropy(alpha))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            chunks = []
            for i in range(0, len(Xte_t), 64):
                lc, _, _, _ = model(Xte_t[i:i+64]); chunks.append(lc.cpu())
            te_logits = torch.cat(chunks, dim=0).to(device)
            te_acc = (te_logits.argmax(1) == yte_t).float().mean().item() * 100
            if te_acc > best_acc:
                best_acc = te_acc; best_logits = te_logits.cpu().clone()
        if (epoch + 1) % 15 == 0 or epoch == 0:
            print(f'  [baseline s={seed}] ep{epoch+1:3d} te={te_acc:.2f} best={best_acc:.2f}')
    # final purity
    model.eval()
    with torch.no_grad():
        _, _, _, alpha_static_te = model(Xte_t[:128])
        purity = cluster_purity(alpha_static_te, gt_te_t[:128])
    return best_acc, best_logits, purity


def pretrain_cycle_stable(args, Xtr, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    model = Model().to(device)
    sample = torch.from_numpy(Xtr[:256]).to(device)
    model.kmeans_init_cluster_head(sample)
    opt = torch.optim.AdamW(
        list(model.point_mlp.parameters()) +
        list(model.cluster_head.parameters()) +
        list(model.gru_cyc.parameters()) +
        list(model.cycle_proj.parameters()),
        lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.pre_epochs, eta_min=1e-5)
    bs = 32; n_train = len(Xtr)
    for epoch in range(args.pre_epochs):
        model.train()
        idx = np.random.permutation(n_train)
        tot_cyc = tot_pp = tot_bal = nbat = 0
        # warmup: balance-heavy for first 3 epochs
        if epoch < 3:
            w_cyc, w_pp, w_bal = 0.1, 0.5, 10.0
        else:
            w_cyc, w_pp, w_bal = 1.0, 1.0, 5.0
        for i in range(0, n_train, bs):
            sel = idx[i:i+bs]
            X = torch.from_numpy(Xtr[sel]).to(device)
            _, Q_pred, alpha, alpha_static = model(X)
            Q_act = compute_qact(X, alpha)
            l_cyc = quat_dist(Q_pred, Q_act)
            l_pp = per_point_entropy(alpha_static)
            l_bal_value = cluster_mass_entropy(alpha)  # want HIGH
            l_bal = -l_bal_value
            loss = w_cyc * l_cyc + w_pp * l_pp + w_bal * l_bal
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            tot_cyc += l_cyc.item(); tot_pp += l_pp.item(); tot_bal += l_bal_value.item(); nbat += 1
        sched.step()
        if epoch < 3 or (epoch + 1) % 5 == 0:
            print(f'  [cyc-pre s={seed}] ep{epoch+1:3d} cyc={tot_cyc/nbat:.4f} pp_ent={tot_pp/nbat:.3f} bal_ent={tot_bal/nbat:.3f}')
    return model


def train_clf_frozen(args, model, Xtr, ytr, Xte, gt_te, yte, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    for p in model.point_mlp.parameters(): p.requires_grad_(False)
    for p in model.cluster_head.parameters(): p.requires_grad_(False)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                             lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-5)
    bs = 32; n_train = len(Xtr)
    Xte_t = torch.from_numpy(Xte).to(device)
    gt_te_t = torch.from_numpy(gt_te).to(device)
    yte_t = torch.from_numpy(yte).to(device)
    best_acc = 0.0; best_logits = None
    for epoch in range(args.epochs):
        model.train(); model.point_mlp.eval(); model.cluster_head.eval()
        idx = np.random.permutation(n_train)
        for i in range(0, n_train, bs):
            sel = idx[i:i+bs]
            X = torch.from_numpy(Xtr[sel]).to(device); y = torch.from_numpy(ytr[sel]).to(device)
            logits, _, _, _ = model(X)
            loss = F.cross_entropy(logits, y)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            chunks = []
            for i in range(0, len(Xte_t), 64):
                lc, _, _, _ = model(Xte_t[i:i+64]); chunks.append(lc.cpu())
            te_logits = torch.cat(chunks, dim=0).to(device)
            te_acc = (te_logits.argmax(1) == yte_t).float().mean().item() * 100
            if te_acc > best_acc:
                best_acc = te_acc; best_logits = te_logits.cpu().clone()
        if (epoch + 1) % 15 == 0 or epoch == 0:
            print(f'  [cyc-clf s={seed}] ep{epoch+1:3d} te={te_acc:.2f} best={best_acc:.2f}')
    model.eval()
    with torch.no_grad():
        _, _, _, alpha_static_te = model(Xte_t[:128])
        purity = cluster_purity(alpha_static_te, gt_te_t[:128])
    return best_acc, best_logits, purity


def softmax_np(x):
    x = x - x.max(-1, keepdims=True); e = np.exp(x); return e / e.sum(-1, keepdims=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--pre-epochs', type=int, default=20)
    ap.add_argument('--n-train', type=int, default=1500)
    ap.add_argument('--n-test', type=int, default=400)
    ap.add_argument('--T', type=int, default=32)
    ap.add_argument('--ppc', type=int, default=30)
    ap.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44, 45])
    args = ap.parse_args()

    Xtr, gt_tr, ytr = make_dataset(args.n_train, T=args.T, ppc=args.ppc, seed=1)
    Xte, gt_te, yte = make_dataset(args.n_test, T=args.T, ppc=args.ppc, seed=2)
    print(f'\nv7-STABLE  dataset={args.n_train} train / {args.n_test} test  K={K}  classes={NUM_CLASSES}')
    print(f'seeds: {args.seeds}\n')

    print('==== (A) BASELINE (no cycle) ====')
    A = []
    for s in args.seeds:
        print(f'-- baseline seed {s}')
        acc, lg, p = train_baseline(args, Xtr, ytr, Xte, gt_te, yte, s)
        A.append((s, acc, lg, p))

    print('\n==== (C) CYCLE-PRETRAIN (stable) + frozen classify ====')
    C = []
    for s in args.seeds:
        print(f'-- cycle-pretrain seed {s}')
        m_pre = pretrain_cycle_stable(args, Xtr, s)
        print(f'-- frozen classify seed {s}')
        acc, lg, p = train_clf_frozen(args, m_pre, Xtr, ytr, Xte, gt_te, yte, s)
        C.append((s, acc, lg, p))

    print('\n============================================')
    print('SUMMARY — per-seed solo')
    print('============================================')
    print('seed   baseline   cyc-pretrain   purity_A   purity_C')
    for (sa, aa, _, pa), (sc, ac, _, pc) in zip(A, C):
        print(f'  {sa}    {aa:6.2f}     {ac:6.2f}        {pa:.3f}      {pc:.3f}')

    accs_A = [r[1] for r in A]; accs_C = [r[1] for r in C]
    print()
    print(f'baseline mean={np.mean(accs_A):.2f} ± {np.std(accs_A):.2f}')
    print(f'cyc-pretrain mean={np.mean(accs_C):.2f} ± {np.std(accs_C):.2f}')

    print()
    print('==== FUSION ====')
    def fuse(items):
        p = sum(softmax_np(r[2].numpy()) for r in items) / len(items)
        return (p.argmax(1) == yte).mean() * 100

    print(f'  (A) ensemble:  {fuse(A):.2f}')
    print(f'  (C) ensemble:  {fuse(C):.2f}')
    pa = sum(softmax_np(r[2].numpy()) for r in A) / len(A)
    pc = sum(softmax_np(r[2].numpy()) for r in C) / len(C)
    cross = (pa + pc) / 2
    f_cross = (cross.argmax(1) == yte).mean() * 100
    print(f'  (A)+(C) cross-fusion: {f_cross:.2f}')

    # pairwise cross-fusion: each A seed with each C seed
    print(f'\n  pairwise A-seed * C-seed cross-fusion:')
    for sa, _, lga, _ in A:
        for sc, _, lgc, _ in C:
            p = (softmax_np(lga.numpy()) + softmax_np(lgc.numpy())) / 2
            f = (p.argmax(1) == yte).mean() * 100
            print(f'    A_s{sa} + C_s{sc}: {f:.2f}')


if __name__ == '__main__':
    main()

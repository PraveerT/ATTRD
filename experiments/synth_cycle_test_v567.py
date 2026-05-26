"""Synthetic cycle volley v5/v6/v7 — keep trying until something works.

v5: weak no-temporal backbone. Classifier sees per-cluster mean-pooled
    features (no GRU). With cycle: also concat Q_pred (per-cluster
    quat trajectory, flattened). Cycle becomes the ONLY rotation feature path.

v6: random global SO(3) per sequence. Class defined by per-cluster rotation
    pattern; global pool useless. Forces decomposition by cluster.

v7: cycle pretrain with diversity floor (anti-collapse). Add penalty on
    sample-mean alpha entropy (want low entropy per point but uniform mass
    across clusters). Fix v4 (C) uniform-alpha collapse.
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


def random_so3(rng):
    u1, u2, u3 = rng.uniform(0, 1, 3)
    q = np.array([
        math.sqrt(1 - u1) * math.sin(2*math.pi*u2),
        math.sqrt(1 - u1) * math.cos(2*math.pi*u2),
        math.sqrt(u1)     * math.sin(2*math.pi*u3),
        math.sqrt(u1)     * math.cos(2*math.pi*u3),
    ])
    return quat_to_rotmat(q)


_X, _Y, _Z = np.array([1.,0,0]), np.array([0,1.,0]), np.array([0,0,1.])
CLASS_TEMPLATES = [
    [(_Z, 0.0), (_Z, 1.4), (_Z, 1.4), (_Z, 0.0), (_Z, 0.0), (_Z, 0.0)],
    [(_Z, 0.0), (_Z, 0.0), (_Z, 0.0), (_Z, 1.4), (_Z, 1.4), (_Z, 0.0)],
    [(_Z, 0.0), (_Z, -1.4), (_Z, 0.0), (_Z, 1.4), (_Z, 0.0), (_Z, -1.4)],
    [(_Y, 1.2), (_Z, 0.0), (_Z, 0.0), (_Z, 0.0), (_Z, 0.0), (_Z, 0.0)],
    [(_X, 0.0), (_X, 1.2), (_X, 0.0), (_X, 0.0), (_X, 0.0), (_X, 0.0)],
    [(_Z, 0.0), (_Z, 1.4), (_Z, 1.4), (_Z, 1.4), (_Z, 1.4), (_Z, 1.4)],
]


def gen_sequence(label, T=32, ppc=30, rng=None, apply_global=False):
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
    if apply_global:
        R_g = random_so3(rng).astype(np.float32)
        t_g = rng.randn(3).astype(np.float32) * 0.3
        for t in range(T):
            coords[t] = coords[t] @ R_g.T + t_g
    perm = rng.permutation(P)
    coords = coords[:, perm, :]
    gt = np.repeat(np.arange(K), ppc)[perm]
    return coords, gt.astype(np.int64)


def make_dataset(n, T=32, ppc=30, seed=0, apply_global=False):
    rng = np.random.RandomState(seed)
    labels = rng.randint(0, NUM_CLASSES, size=n)
    Cs, Gs = [], []
    for lab in labels:
        c, g = gen_sequence(int(lab), T=T, ppc=ppc, rng=rng, apply_global=apply_global)
        Cs.append(c); Gs.append(g)
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
    return torch.cat([qw, qxyz], dim=-1)


def quat_dist_double_cover(q1, q2):
    d1 = ((q1 - q2)**2).sum(-1); d2 = ((q1 + q2)**2).sum(-1)
    return torch.min(d1, d2).mean()


def axis_angle_exp(v):
    norm = v.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    h = norm * 0.5
    return torch.cat([torch.cos(h), torch.sin(h) * (v / norm)], dim=-1)


def cluster_entropy(alpha):
    m = alpha.mean(dim=(0, 1, 2)).clamp_min(1e-8)
    m = m / m.sum()
    return -(m * m.log()).sum()


def per_point_entropy(alpha_static):
    """Negative — we want each point to commit (low entropy per point)."""
    a = alpha_static.clamp_min(1e-8)
    return -(a * a.log()).sum(-1).mean()  # mean over batch and points


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


class WeakModel(nn.Module):
    """v5: weak no-temporal classifier. Per-cluster mean-pool features
    + optionally Q_pred (T*4 per cluster) → linear classifier.
    """
    def __init__(self, num_classes=NUM_CLASSES, hidden=64, K=K, T=32, use_qpred=False):
        super().__init__()
        self.K = K; self.T = T; self.use_qpred = use_qpred
        self.point_mlp = nn.Sequential(
            nn.Linear(3, hidden), nn.ReLU(), nn.LayerNorm(hidden),
        )
        self.cluster_head = nn.Sequential(
            nn.Linear(hidden + 3, hidden), nn.ReLU(), nn.Linear(hidden, K),
        )
        self.gru_cyc = nn.GRU(hidden, hidden, batch_first=True, bidirectional=True)
        self.cycle_proj = nn.Linear(2 * hidden, 3)
        nn.init.zeros_(self.cycle_proj.weight); nn.init.zeros_(self.cycle_proj.bias)
        in_dim = hidden * K + (4 * T * K if use_qpred else 0)
        self.clf = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, coords):
        B, T, P, _ = coords.shape
        feat = self.point_mlp(coords)
        pooled_feat = feat.mean(dim=1); pooled_coords = coords.mean(dim=1)
        cl_logits = self.cluster_head(torch.cat([pooled_feat, pooled_coords], dim=-1))
        alpha_static = F.softmax(cl_logits, dim=-1)
        alpha = alpha_static.unsqueeze(1).expand(B, T, P, self.K).contiguous()
        mass = alpha.sum(dim=2).clamp_min(1e-8)
        cf = (alpha.unsqueeze(-1) * feat.unsqueeze(-2)).sum(dim=2) / mass.unsqueeze(-1)
        # cf: (B, T, K, H). Mean-pool over time for weak classifier.
        cf_pool = cf.mean(dim=1)  # (B, K, H)
        H = cf.size(-1)
        cf_btkh = cf.permute(0, 2, 1, 3).reshape(B * self.K, T, H)
        out_cyc, _ = self.gru_cyc(cf_btkh)
        v = self.cycle_proj(out_cyc)
        Q_pred = axis_angle_exp(v).reshape(B, self.K, T, 4)  # (B,K,T,4)
        Q_pred_btk4 = Q_pred.permute(0, 2, 1, 3)  # (B,T,K,4) to match Q_act shape
        if self.use_qpred:
            qf = Q_pred.reshape(B, self.K * T * 4)
            x = torch.cat([cf_pool.reshape(B, self.K * H), qf], dim=-1)
        else:
            x = cf_pool.reshape(B, self.K * H)
        logits = self.clf(x)
        return logits, Q_pred_btk4, alpha, alpha_static


class StrongModel(nn.Module):
    """v6: same as v4 model, temporal GRU classifier. Tested on global-rotation
    distracted data.
    """
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

    def forward(self, coords):
        B, T, P, _ = coords.shape
        feat = self.point_mlp(coords)
        pooled_feat = feat.mean(dim=1); pooled_coords = coords.mean(dim=1)
        cl_logits = self.cluster_head(torch.cat([pooled_feat, pooled_coords], dim=-1))
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


def train(args, model, Xtr, ytr, Xte, gt_te, yte, use_cycle, label, seed):
    print(f'\n--- {label}  seed={seed} ---')
    torch.manual_seed(seed); np.random.seed(seed)
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
        tot_ce = tot_cyc = nbat = 0
        for i in range(0, n_train, bs):
            sel = idx[i:i+bs]
            X = torch.from_numpy(Xtr[sel]).to(device)
            y = torch.from_numpy(ytr[sel]).to(device)
            logits, Q_pred, alpha, _ = model(X)
            ce = F.cross_entropy(logits, y)
            l_bal = -cluster_entropy(alpha)
            l_cyc = torch.tensor(0.0, device=device)
            if use_cycle:
                Q_act = compute_qact_per_cluster(X, alpha)
                l_cyc = quat_dist_double_cover(Q_pred, Q_act)
            loss = ce + (0.05 * l_cyc if use_cycle else 0.0) + 0.01 * l_bal
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot_ce += ce.item(); tot_cyc += l_cyc.item(); nbat += 1
        sched.step()
        model.eval()
        with torch.no_grad():
            chunks = []
            for i in range(0, len(Xte_t), 64):
                lc, _, _, _ = model(Xte_t[i:i+64]); chunks.append(lc.cpu())
            te_logits = torch.cat(chunks, dim=0).to(device)
            te_acc = (te_logits.argmax(1) == yte_t).float().mean().item() * 100
            _, _, alpha_te, alpha_static_te = model(Xte_t[:128])
            purity = cluster_purity(alpha_static_te, gt_te_t[:128])
            if te_acc > best_acc:
                best_acc = te_acc; best_logits = te_logits.cpu().clone()
        if (epoch + 1) % max(1, args.epochs // 6) == 0 or epoch == 0:
            print(f'  ep{epoch+1:3d}  ce={tot_ce/nbat:.4f}  cyc={tot_cyc/nbat:.4f}  '
                  f'te_acc={te_acc:.2f}  best={best_acc:.2f}  purity={purity:.3f}')
    return best_acc, best_logits, purity


def pretrain_with_diversity(args, Xtr, seed):
    """v7: cycle pretrain with diversity floor — prevents uniform-alpha collapse."""
    torch.manual_seed(seed); np.random.seed(seed)
    model = StrongModel().to(device)
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
        tot_cyc = tot_div = tot_bal = nbat = 0
        for i in range(0, n_train, bs):
            sel = idx[i:i+bs]
            X = torch.from_numpy(Xtr[sel]).to(device)
            _, Q_pred, alpha, alpha_static = model(X)
            Q_act = compute_qact_per_cluster(X, alpha)
            l_cyc = quat_dist_double_cover(Q_pred, Q_act)
            # diversity floor: encourage per-point commitment (low per-point entropy)
            # while keeping cluster mass balanced (high mean-mass entropy).
            l_pp_ent = per_point_entropy(alpha_static)  # we want this LOW
            l_bal = -cluster_entropy(alpha)              # we want this HIGH magnitude (LOW value)
            loss = l_cyc + 1.0 * l_pp_ent + 0.5 * l_bal
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot_cyc += l_cyc.item(); tot_div += l_pp_ent.item(); tot_bal += (-l_bal.item()); nbat += 1
        sched.step()
        if (epoch + 1) % max(1, args.pre_epochs // 4) == 0 or epoch == 0:
            print(f'    pre ep{epoch+1:3d}  cyc={tot_cyc/nbat:.4f}  pp_ent={tot_div/nbat:.3f}  bal={tot_bal/nbat:.3f}')
    return model


def train_classifier_only(args, model, Xtr, ytr, Xte, gt_te, yte, label, seed):
    print(f'\n--- {label}  seed={seed} ---')
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
        tot_ce = nbat = 0
        for i in range(0, n_train, bs):
            sel = idx[i:i+bs]
            X = torch.from_numpy(Xtr[sel]).to(device)
            y = torch.from_numpy(ytr[sel]).to(device)
            logits, _, alpha, _ = model(X)
            ce = F.cross_entropy(logits, y)
            l_bal = -cluster_entropy(alpha)
            loss = ce + 0.01 * l_bal
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot_ce += ce.item(); nbat += 1
        sched.step()
        model.eval()
        with torch.no_grad():
            chunks = []
            for i in range(0, len(Xte_t), 64):
                lc, _, _, _ = model(Xte_t[i:i+64]); chunks.append(lc.cpu())
            te_logits = torch.cat(chunks, dim=0).to(device)
            te_acc = (te_logits.argmax(1) == yte_t).float().mean().item() * 100
            _, _, _, alpha_static_te = model(Xte_t[:128])
            purity = cluster_purity(alpha_static_te, gt_te_t[:128])
            if te_acc > best_acc:
                best_acc = te_acc; best_logits = te_logits.cpu().clone()
        if (epoch + 1) % max(1, args.epochs // 5) == 0 or epoch == 0:
            print(f'  ep{epoch+1:3d}  ce={tot_ce/nbat:.4f}  te_acc={te_acc:.2f}  best={best_acc:.2f}  purity={purity:.3f}')
    return best_acc, best_logits, purity


def softmax_np(x):
    x = x - x.max(-1, keepdims=True); e = np.exp(x); return e / e.sum(-1, keepdims=True)


def run_v5(args):
    Xtr, gt_tr, ytr = make_dataset(args.n_train, T=args.T, ppc=args.ppc, seed=1, apply_global=False)
    Xte, gt_te, yte = make_dataset(args.n_test, T=args.T, ppc=args.ppc, seed=2, apply_global=False)
    print(f'\n========== v5  weak no-temporal classifier  ==========')
    print(f'dataset: {args.n_train} train / {args.n_test} test (no global rot)')

    A_res, B_res = [], []
    for s in [42, 43]:
        torch.manual_seed(s); np.random.seed(s)
        m = WeakModel(T=args.T, use_qpred=False).to(device)
        acc, lg, p = train(args, m, Xtr, ytr, Xte, gt_te, yte, use_cycle=False,
                            label='(A) weak no-cycle no-Qpred', seed=s)
        A_res.append((acc, lg, p))
    for s in [42, 43]:
        torch.manual_seed(s); np.random.seed(s)
        m = WeakModel(T=args.T, use_qpred=True).to(device)
        acc, lg, p = train(args, m, Xtr, ytr, Xte, gt_te, yte, use_cycle=True,
                            label='(B) weak cycle + Q_pred input', seed=s)
        B_res.append((acc, lg, p))

    def mean(res, idx): return float(np.mean([r[idx] for r in res]))
    def fuse(res):
        p = sum(softmax_np(r[1].numpy()) for r in res) / len(res)
        return (p.argmax(1) == yte).mean() * 100

    print()
    print('===== v5 SUMMARY =====')
    print(f'(A) weak no-cycle:           best_solo={mean(A_res,0):.2f}  purity={mean(A_res,2):.3f}  fusion={fuse(A_res):.2f}')
    print(f'(B) weak cycle + Q_pred:     best_solo={mean(B_res,0):.2f}  purity={mean(B_res,2):.3f}  fusion={fuse(B_res):.2f}')
    print(f'delta_solo (B - A):  {mean(B_res,0) - mean(A_res,0):+.2f}')
    return mean(A_res,0), mean(B_res,0)


def run_v6(args):
    Xtr, gt_tr, ytr = make_dataset(args.n_train, T=args.T, ppc=args.ppc, seed=1, apply_global=True)
    Xte, gt_te, yte = make_dataset(args.n_test, T=args.T, ppc=args.ppc, seed=2, apply_global=True)
    print(f'\n========== v6  global-rotation distractor + strong classifier  ==========')
    print(f'dataset: {args.n_train} train / {args.n_test} test (random global SO(3))')

    A_res, B_res = [], []
    for s in [42, 43]:
        torch.manual_seed(s); np.random.seed(s)
        m = StrongModel().to(device)
        acc, lg, p = train(args, m, Xtr, ytr, Xte, gt_te, yte, use_cycle=False,
                            label='(A) strong no-cycle (distracted)', seed=s)
        A_res.append((acc, lg, p))
    for s in [42, 43]:
        torch.manual_seed(s); np.random.seed(s)
        m = StrongModel().to(device)
        acc, lg, p = train(args, m, Xtr, ytr, Xte, gt_te, yte, use_cycle=True,
                            label='(B) strong + cycle (distracted)', seed=s)
        B_res.append((acc, lg, p))

    def mean(res, idx): return float(np.mean([r[idx] for r in res]))
    def fuse(res):
        p = sum(softmax_np(r[1].numpy()) for r in res) / len(res)
        return (p.argmax(1) == yte).mean() * 100

    print()
    print('===== v6 SUMMARY =====')
    print(f'(A) strong no-cycle:    best_solo={mean(A_res,0):.2f}  purity={mean(A_res,2):.3f}  fusion={fuse(A_res):.2f}')
    print(f'(B) strong + cycle:     best_solo={mean(B_res,0):.2f}  purity={mean(B_res,2):.3f}  fusion={fuse(B_res):.2f}')
    print(f'delta_solo (B - A):  {mean(B_res,0) - mean(A_res,0):+.2f}')
    return mean(A_res,0), mean(B_res,0)


def run_v7(args):
    Xtr, gt_tr, ytr = make_dataset(args.n_train, T=args.T, ppc=args.ppc, seed=1, apply_global=False)
    Xte, gt_te, yte = make_dataset(args.n_test, T=args.T, ppc=args.ppc, seed=2, apply_global=False)
    print(f'\n========== v7  cycle pretrain with diversity floor  ==========')
    print(f'dataset: {args.n_train} train / {args.n_test} test')

    A_res, C_res = [], []
    for s in [42, 43]:
        torch.manual_seed(s); np.random.seed(s)
        m = StrongModel().to(device)
        acc, lg, p = train(args, m, Xtr, ytr, Xte, gt_te, yte, use_cycle=False,
                            label='(A) strong no-cycle baseline', seed=s)
        A_res.append((acc, lg, p))
    for s in [42, 43]:
        print(f'  pretraining cluster discovery (anti-collapse) seed {s}')
        m_pre = pretrain_with_diversity(args, Xtr, seed=s)
        acc, lg, p = train_classifier_only(args, m_pre, Xtr, ytr, Xte, gt_te, yte,
                                            label='(C) cycle-pretrain w/ diversity frozen', seed=s)
        C_res.append((acc, lg, p))

    def mean(res, idx): return float(np.mean([r[idx] for r in res]))
    def fuse(res):
        p = sum(softmax_np(r[1].numpy()) for r in res) / len(res)
        return (p.argmax(1) == yte).mean() * 100

    print()
    print('===== v7 SUMMARY =====')
    print(f'(A) baseline:            best_solo={mean(A_res,0):.2f}  purity={mean(A_res,2):.3f}  fusion={fuse(A_res):.2f}')
    print(f'(C) cyc-pretrain frozen: best_solo={mean(C_res,0):.2f}  purity={mean(C_res,2):.3f}  fusion={fuse(C_res):.2f}')
    print(f'delta_solo (C - A):  {mean(C_res,0) - mean(A_res,0):+.2f}')
    # cross-fusion
    pa = sum(softmax_np(r[1].numpy()) for r in A_res) / len(A_res)
    pc = sum(softmax_np(r[1].numpy()) for r in C_res) / len(C_res)
    cross = ((pa + pc) / 2).argmax(1)
    f_cross = (cross == yte).mean() * 100
    print(f'(A+C) cross-fusion: {f_cross:.2f}')
    return mean(A_res,0), mean(C_res,0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variant', type=str, default='all', choices=['v5', 'v6', 'v7', 'all'])
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--pre-epochs', type=int, default=20)
    ap.add_argument('--n-train', type=int, default=1200)
    ap.add_argument('--n-test', type=int, default=400)
    ap.add_argument('--T', type=int, default=32)
    ap.add_argument('--ppc', type=int, default=30)
    args = ap.parse_args()

    results = {}
    if args.variant in ('v5', 'all'): results['v5'] = run_v5(args)
    if args.variant in ('v6', 'all'): results['v6'] = run_v6(args)
    if args.variant in ('v7', 'all'): results['v7'] = run_v7(args)

    print()
    print('============================================')
    print('OVERALL VOLLEY SUMMARY')
    print('============================================')
    for v, (a, b) in results.items():
        delta = b - a
        verdict = 'CLEAR HELP' if delta >= 2.0 else ('MARGINAL' if delta >= 0.5 else 'NO HELP' if delta >= -0.5 else 'HURT')
        print(f'  {v}: baseline={a:.2f}  cycle_variant={b:.2f}  delta={delta:+.2f}  -> {verdict}')


if __name__ == '__main__':
    main()

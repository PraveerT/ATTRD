"""v8 — cycle as a DIRECT classifier feature, with Q_act diversity loss.

Key changes from v7:
  - DO NOT freeze cluster_head. Joint train with classification + cycle.
  - Pass per-cluster Q_pred trajectory (T*4 per cluster, flattened) to classifier
    in addition to per-cluster temporal features.
  - Add Q_act diversity loss: maximize pairwise distance among Q_act(k) across
    clusters k. Prevents the degenerate uniform-cluster trivial solution.
  - 3 regimes: (A) baseline, (B) cycle joint, (C) cycle joint + Q_pred-in-clf.

If (C) > (A), cycle is providing direct classification value.
"""
import argparse, math, numpy as np, torch, torch.nn as nn, torch.nn.functional as F

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
    I = torch.eye(3, device=cov.device, dtype=cov.dtype); cov = cov + 1e-4 * I
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


def q_act_diversity(Q_act):
    """Maximize mean pairwise double-cover distance between cluster trajectories."""
    B, T, K_, _ = Q_act.shape
    Qk = Q_act.permute(0, 2, 1, 3).reshape(B, K_, T * 4)
    diff_pos = Qk.unsqueeze(2) - Qk.unsqueeze(1)
    diff_neg = Qk.unsqueeze(2) + Qk.unsqueeze(1)
    d = torch.min((diff_pos ** 2).sum(-1), (diff_neg ** 2).sum(-1))
    mask = 1 - torch.eye(K_, device=Q_act.device)
    return -(d * mask).sum() / (B * K_ * (K_ - 1))


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
    def __init__(self, num_classes=NUM_CLASSES, hidden=128, K=K, T=32, use_q_in_clf=False):
        super().__init__()
        self.K = K; self.T = T; self.use_q_in_clf = use_q_in_clf
        self.point_mlp = nn.Sequential(
            nn.Linear(3, 64), nn.ReLU(), nn.LayerNorm(64),
            nn.Linear(64, hidden), nn.ReLU(), nn.LayerNorm(hidden),
        )
        self.cluster_head = nn.Sequential(
            nn.Linear(hidden + 3, hidden), nn.ReLU(), nn.Linear(hidden, K),
        )
        self.gru_clf = nn.GRU(hidden, hidden, num_layers=2, batch_first=True,
                              bidirectional=True, dropout=0.2)
        clf_in = 2 * hidden * K + (4 * T * K if use_q_in_clf else 0)
        self.clf = nn.Sequential(
            nn.Linear(clf_in, hidden), nn.ReLU(), nn.Dropout(0.3),
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
        out_cyc, _ = self.gru_cyc(cf_btkh)
        v = self.cycle_proj(out_cyc)
        Q_pred = axis_angle_exp(v).reshape(B, self.K, T, 4).permute(0, 2, 1, 3)
        if self.use_q_in_clf:
            qf = Q_pred.permute(0, 2, 1, 3).reshape(B, self.K * T * 4)
            clf_in = torch.cat([last, qf], dim=-1)
        else:
            clf_in = last
        logits = self.clf(clf_in)
        return logits, Q_pred, alpha, alpha_static


def train(args, Xtr, ytr, Xte, gt_te, yte, label, seed,
          use_cycle=False, use_diversity=False, use_q_in_clf=False):
    torch.manual_seed(seed); np.random.seed(seed)
    model = Model(use_q_in_clf=use_q_in_clf).to(device)
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
        tot_ce = tot_cyc = tot_div = nbat = 0
        for i in range(0, n_train, bs):
            sel = idx[i:i+bs]
            X = torch.from_numpy(Xtr[sel]).to(device); y = torch.from_numpy(ytr[sel]).to(device)
            logits, Q_pred, alpha, _ = model(X)
            ce = F.cross_entropy(logits, y)
            l_bal = -cluster_mass_entropy(alpha)
            l_cyc = torch.tensor(0.0, device=device); l_div = torch.tensor(0.0, device=device)
            if use_cycle:
                Q_act = compute_qact(X, alpha)
                l_cyc = quat_dist(Q_pred, Q_act)
                if use_diversity:
                    l_div = q_act_diversity(Q_act)
            loss = ce + 0.01 * l_bal
            if use_cycle: loss = loss + 0.1 * l_cyc + (0.05 * l_div if use_diversity else 0.0)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            tot_ce += ce.item(); tot_cyc += l_cyc.item(); tot_div += l_div.item(); nbat += 1
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
            print(f'  [{label} s={seed}] ep{epoch+1:3d} ce={tot_ce/nbat:.3f} cyc={tot_cyc/nbat:.4f} '
                  f'div={tot_div/nbat:.4f} te={te_acc:.2f} best={best_acc:.2f}')
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
    ap.add_argument('--n-train', type=int, default=1500)
    ap.add_argument('--n-test', type=int, default=400)
    ap.add_argument('--T', type=int, default=32)
    ap.add_argument('--ppc', type=int, default=30)
    ap.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44, 45])
    args = ap.parse_args()

    Xtr, gt_tr, ytr = make_dataset(args.n_train, T=args.T, ppc=args.ppc, seed=1)
    Xte, gt_te, yte = make_dataset(args.n_test, T=args.T, ppc=args.ppc, seed=2)
    print(f'v8 dataset {args.n_train}/{args.n_test} K={K} classes={NUM_CLASSES} seeds={args.seeds}\n')

    print('=== (A) baseline ===')
    A = []
    for s in args.seeds:
        acc, lg, p = train(args, Xtr, ytr, Xte, gt_te, yte, 'A', s,
                            use_cycle=False, use_diversity=False, use_q_in_clf=False)
        A.append((s, acc, lg, p))

    print('\n=== (B) joint cycle (diversity, no Q in clf) ===')
    B = []
    for s in args.seeds:
        acc, lg, p = train(args, Xtr, ytr, Xte, gt_te, yte, 'B', s,
                            use_cycle=True, use_diversity=True, use_q_in_clf=False)
        B.append((s, acc, lg, p))

    print('\n=== (C) joint cycle (diversity) + Q_pred in classifier ===')
    C = []
    for s in args.seeds:
        acc, lg, p = train(args, Xtr, ytr, Xte, gt_te, yte, 'C', s,
                            use_cycle=True, use_diversity=True, use_q_in_clf=True)
        C.append((s, acc, lg, p))

    def stats(R): return np.mean([r[1] for r in R]), np.std([r[1] for r in R]), np.mean([r[3] for r in R])
    def fuse(R):
        p = sum(softmax_np(r[2].numpy()) for r in R) / len(R)
        return (p.argmax(1) == yte).mean() * 100

    print()
    print('===== SUMMARY =====')
    for name, R in [('A baseline', A), ('B joint+div', B), ('C joint+div+Qclf', C)]:
        m, s, p = stats(R); f = fuse(R)
        print(f'  {name:18s}  mean={m:5.2f} +- {s:4.2f}   purity={p:.3f}   ensemble={f:.2f}')

    # cross-fusion all pairs
    print(f'\n  Cross-fusion A_ens + B_ens: {((sum(softmax_np(r[2].numpy()) for r in A)/len(A) + sum(softmax_np(r[2].numpy()) for r in B)/len(B))/2).argmax(1).__eq__(yte).mean()*100:.2f}')
    print(f'  Cross-fusion A_ens + C_ens: {((sum(softmax_np(r[2].numpy()) for r in A)/len(A) + sum(softmax_np(r[2].numpy()) for r in C)/len(C))/2).argmax(1).__eq__(yte).mean()*100:.2f}')
    print(f'  Cross-fusion B_ens + C_ens: {((sum(softmax_np(r[2].numpy()) for r in B)/len(B) + sum(softmax_np(r[2].numpy()) for r in C)/len(C))/2).argmax(1).__eq__(yte).mean()*100:.2f}')


if __name__ == '__main__':
    main()

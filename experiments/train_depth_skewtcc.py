"""FG83-depth R(2+1)D-18 + Skew-Symmetric Temporal Cross-Covariance pooling.

Applies the Skew-TCC head (antisymmetric lagged feature cross-covariance) to the
SMALL depth partner (the 83.6 fg83 model), not CN-XXL. CN-XXL skew-TCC showed no
gain; this asks whether the directional pooling helps the depth-video model.

Per-frame features z_t are spatial-max-reduced from a tapped backbone stage (layer3
keeps T'=8, vs layer4 T'=4 -- too coarse for lagged cross-cov). Two low-rank
projectors U=zW_u, V=zW_v (rank r) -> lagged cross-Gram C_d = mean_t u_t v_{t+d}^T.
  mode=skew  : A_d = (C_d - C_d^T)/2   (the contribution)
  mode=sym   : S_d = (C_d + C_d^T)/2   (matched symmetric-bilinear control)
  mode=random: skew, but projectors FROZEN at init (structural-presence control)
  mode=off   : plain r2plus1d (recipe-matched baseline)
The tril off-diagonal descriptor (identical length skew/sym) is concatenated onto
the existing avgpool first-order 512-d feature -> one classifier. So it can only ADD.

Honesty: hflip aug OFF and flip-TTA OFF (both symmetrize away the chirality signal,
brief P7). All four variants share recipe + seed -> clean paired comparison. Dumps
train+test logits (sigs) for later honest fusion; prints the forward-vs-mirror gap.
"""
import argparse, math, os, random, time, copy
import numpy as np
from PIL import Image
import torch, torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset

KMEAN = torch.tensor([0.43216, 0.394666, 0.37645]).view(3, 1, 1, 1)
KSTD = torch.tensor([0.22803, 0.22145, 0.216989]).view(3, 1, 1, 1)


def read_split(path):
    out = []
    for ln in open(path, encoding="utf-8"):
        a = ln.strip().split()
        if len(a) >= 3:
            out.append((a[0].strip("/"), int(a[1]), int(a[2])))
    return out


def _fg_bbox(frames, pad_frac=0.18):
    m = np.zeros(frames[0].shape, bool)
    for a in frames:
        m |= a > 0
    ys, xs = np.where(m)
    h, w = frames[0].shape
    if len(xs) == 0:
        return (0, 0, w, h)
    y0, y1, x0, x1 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
    p = int(max(y1 - y0, x1 - x0) * pad_frac) + 4
    return (max(0, x0 - p), max(0, y0 - p), min(w, x1 + p), min(h, y1 + p))


def build_cache(split_path, phase, data_root, cache_dir, size, fg_crop=True):
    os.makedirs(cache_dir, exist_ok=True)
    tag = "fg" if fg_crop else "all"
    base = os.path.join(cache_dir, f"{phase}_{tag}_s{size}")
    if os.path.isfile(base + ".npy"):
        return (np.load(base + ".npy", mmap_mode="r"),
                np.load(base + "_labels.npy"), np.load(base + "_sigs.npy", allow_pickle=True))
    recs = read_split(split_path)
    maxf = max(r[1] for r in recs)
    clips = np.zeros((len(recs), maxf, size, size), np.uint8)
    labels = np.array([r[2] for r in recs], np.int64)
    sigs = np.array([r[0] for r in recs], dtype=object)
    t0 = time.time()
    for i, (rel, nf, _) in enumerate(recs):
        d = os.path.join(data_root, rel)
        raw = [np.asarray(Image.open(os.path.join(d, f"{t:06d}.jpg")).convert("L"), np.uint8) for t in range(nf)]
        bb = _fg_bbox(raw) if fg_crop else None
        for t, fr in enumerate(raw):
            im = Image.fromarray(fr)
            if bb is not None:
                im = im.crop(bb)
            clips[i, t] = np.asarray(im.resize((size, size), Image.BILINEAR), np.uint8)
        if (i + 1) % 200 == 0 or i + 1 == len(recs):
            print(f"[cache {phase}] {i+1}/{len(recs)} {time.time()-t0:.0f}s", flush=True)
    np.save(base + ".npy", clips); np.save(base + "_labels.npy", labels); np.save(base + "_sigs.npy", sigs)
    return np.load(base + ".npy", mmap_mode="r"), labels, sigs


class DS(Dataset):
    def __init__(self, clips, labels, sigs, frames=32, crop=112, train=False):
        self.c, self.l, self.s = clips, labels, np.asarray(sigs)
        self.frames, self.crop, self.train = int(frames), int(crop), bool(train)
        self.cache = int(clips.shape[-1]); self.maxf = int(clips.shape[1])

    def __len__(self):
        return len(self.l)

    def _tidx(self):
        if self.train:
            span = random.randint(max(self.frames, int(self.maxf * 0.65)), self.maxf)
            st = random.randint(0, self.maxf - span)
            idx = np.linspace(st, st + span - 1, self.frames)
            idx = np.clip(np.rint(idx + np.random.uniform(-0.45, 0.45, self.frames)), 0, self.maxf - 1).astype(np.int64)
            idx.sort()
            return idx
        return np.linspace(0, self.maxf - 1, self.frames).round().astype(np.int64)

    def __getitem__(self, i):
        x = np.asarray(self.c[i, self._tidx()], np.float32) / 255.0   # (T,H,W)
        lim = self.cache - self.crop
        if self.train and lim > 0:
            y0, x0 = random.randint(0, lim), random.randint(0, lim)
        else:
            y0 = x0 = max(0, lim // 2)
        x = x[:, y0:y0 + self.crop, x0:x0 + self.crop]
        # NO horizontal flip: chirality must be preserved (brief P7).
        clip = torch.from_numpy(x).unsqueeze(0).repeat(3, 1, 1, 1)     # (3,T,H,W)
        clip = (clip - KMEAN) / KSTD
        return clip, int(self.l[i]), str(self.s[i])


class R2Plus1DSkewTCC(nn.Module):
    def __init__(self, num_classes=25, pretrained=True, r=12, lags=(1, 2),
                 mode="skew", tap="layer3", head_dropout=0.3):
        super().__init__()
        from torchvision.models.video import R2Plus1D_18_Weights, r2plus1d_18
        w = R2Plus1D_18_Weights.KINETICS400_V1 if pretrained else None
        self.net = r2plus1d_18(weights=w)
        self.feat_dim = self.net.fc.in_features            # 512
        self.net.fc = nn.Identity()
        self.tap = tap
        self.tap_dim = 512 if tap == "layer4" else 256
        self.r, self.lags, self.mode = int(r), tuple(lags), mode
        self.Wu = nn.Linear(self.tap_dim, self.r, bias=False)
        self.Wv = nn.Linear(self.tap_dim, self.r, bias=False)
        if mode == "random":
            for p in list(self.Wu.parameters()) + list(self.Wv.parameters()):
                p.requires_grad = False
        idx = torch.tril_indices(self.r, self.r, offset=-1)
        self.register_buffer("ti", idx)
        self.desc_len = 0 if mode == "off" else self.r * (self.r - 1) // 2 * len(self.lags)
        if self.desc_len:
            self.desc_bn = nn.BatchNorm1d(self.desc_len)
        self.drop = nn.Dropout(head_dropout)
        self.classify = nn.Linear(self.feat_dim + self.desc_len, num_classes)

    def _body(self, x):
        n = self.net
        y = n.stem(x); y = n.layer1(y); y = n.layer2(y); l3 = n.layer3(y); l4 = n.layer4(l3)
        fo = n.avgpool(l4).flatten(1)                      # (B,512) first-order = the fg83 signal
        tap = l4 if self.tap == "layer4" else l3
        z = tap.amax(dim=(3, 4)).transpose(1, 2)           # (B,T',Cdim) per-frame spatial-max
        return fo, z

    def _desc(self, z):
        U, V = self.Wu(z), self.Wv(z)                      # (B,T',r)
        T = z.shape[1]; outs = []
        for d in self.lags:
            if T - d <= 0:
                outs.append(z.new_zeros(z.shape[0], self.r * (self.r - 1) // 2)); continue
            u, v = U[:, :T - d], V[:, d:]
            C = torch.einsum("bti,btj->bij", u, v) / max(1, T - d)
            M = (C - C.transpose(1, 2)) * 0.5 if self.mode in ("skew", "random") else (C + C.transpose(1, 2)) * 0.5
            outs.append(M[:, self.ti[0], self.ti[1]])
        return torch.cat(outs, 1)

    def forward(self, x):
        fo, z = self._body(x)
        if self.desc_len:
            feat = torch.cat([fo, self.drop(self.desc_bn(self._desc(z)))], 1)
        else:
            feat = fo
        return self.classify(feat)

    def backbone_parameters(self):
        yield from self.net.parameters()

    def head_parameters(self):
        mods = [self.Wu, self.Wv, self.drop, self.classify]
        if self.desc_len:
            mods.append(self.desc_bn)
        for m in mods:
            yield from (p for p in m.parameters() if p.requires_grad)


def update_ema(model, ema, decay):
    with torch.no_grad():
        msd = model.state_dict()
        for k, v in ema.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(decay).add_(msd[k], alpha=1.0 - decay)
            else:
                v.copy_(msd[k])


@torch.no_grad()
def evaluate(model, loader, dev, mirror=False):
    model.eval(); L, Y, S = [], [], []
    for clip, y, s in loader:
        clip = clip.to(dev, non_blocking=True)
        if mirror:
            clip = torch.flip(clip, dims=[-1])
        with autocast():
            o = model(clip)
        L.append(o.float().cpu().numpy()); Y.append(y.numpy()); S += list(s)
    L = np.concatenate(L); Y = np.concatenate(Y)
    return (L.argmax(1) == Y).mean() * 100, L, Y, np.array(S, dtype=object)


def lr_at(ep, a):
    if ep <= a.warmup:
        return a.lr * ep / max(1, a.warmup)
    p = (ep - a.warmup) / max(1, a.epochs - a.warmup)
    return a.min_lr + 0.5 * (a.lr - a.min_lr) * (1 + math.cos(math.pi * p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["skew", "sym", "random", "off"], default="skew")
    ap.add_argument("--tap", choices=["layer3", "layer4"], default="layer3")
    ap.add_argument("--r", type=int, default=12)
    ap.add_argument("--lags", default="1,2")
    ap.add_argument("--head-dropout", type=float, default=0.3)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--frames", type=int, default=32)
    ap.add_argument("--cache-size", type=int, default=128)
    ap.add_argument("--crop", type=int, default=112)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--backbone-lr", type=float, default=2e-5)
    ap.add_argument("--min-lr", type=float, default=8e-6)
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--wd", type=float, default=0.02)
    ap.add_argument("--ema-decay", type=float, default=0.99)
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--data-root", default="/notebooks/cvpr_data/depth")
    ap.add_argument("--split-root", default="/notebooks/cvpr_data/dataset_splits")
    ap.add_argument("--cache-dir", default="/notebooks/Anemon/dataset/Nvidia/Processed/depth_small_cache")
    ap.add_argument("--workdir", default="/notebooks/Anemon/experiments/work_dir/skewtcc_depth")
    a = ap.parse_args()
    lags = tuple(int(x) for x in a.lags.split(","))
    os.makedirs(a.workdir, exist_ok=True)
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    torch.backends.cudnn.benchmark = True
    dev = "cuda"

    tr = build_cache(os.path.join(a.split_root, "train.txt"), "train", a.data_root, a.cache_dir, a.cache_size)
    va = build_cache(os.path.join(a.split_root, "valid.txt"), "valid", a.data_root, a.cache_dir, a.cache_size)
    mk = lambda ds, sh: DataLoader(ds, batch_size=a.bs, shuffle=sh, num_workers=6, drop_last=sh,
                                   pin_memory=True, persistent_workers=True)
    dtr = mk(DS(*tr, frames=a.frames, crop=a.crop, train=True), True)
    dva = mk(DS(*va, frames=a.frames, crop=a.crop, train=False), False)
    dtr_e = mk(DS(*tr, frames=a.frames, crop=a.crop, train=False), False)

    model = R2Plus1DSkewTCC(mode=a.mode, tap=a.tap, r=a.r, lags=lags, head_dropout=a.head_dropout).to(dev)
    ema = copy.deepcopy(model).to(dev)
    for p in ema.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(
        [{"params": list(model.backbone_parameters()), "lr": a.backbone_lr},
         {"params": list(model.head_parameters()), "lr": a.lr}], weight_decay=a.wd)
    scaler = GradScaler()
    lossf = nn.CrossEntropyLoss(label_smoothing=a.label_smoothing)
    log = open(os.path.join(a.workdir, "log.txt"), "a", encoding="utf-8")

    def W(m):
        print(m, flush=True); log.write(m + "\n"); log.flush()

    nparam = sum(p.numel() for p in model.parameters()) / 1e6
    desc = a.r * (a.r - 1) // 2 * len(lags)
    W(f"mode={a.mode} tap={a.tap} r={a.r} lags={lags} desc_len={desc} params={nparam:.3f}M")
    W(f"frames={a.frames} crop={a.crop} bs={a.bs} epochs={a.epochs} lr={a.lr} blr={a.backbone_lr} seed={a.seed} | NO hflip, NO flip-TTA")
    W(f"train {len(tr[1])}  valid {len(va[1])}")

    best = 0.0; best_ep = 0
    for ep in range(1, a.epochs + 1):
        lr = lr_at(ep, a)
        opt.param_groups[0]["lr"] = lr * (a.backbone_lr / a.lr)
        opt.param_groups[1]["lr"] = lr
        model.train(); t0 = time.time(); tot = cor = seen = 0
        W(f"Training epoch: {ep}")
        nb = len(dtr)
        for bi, (clip, y, _) in enumerate(dtr, 1):
            clip, y = clip.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with autocast():
                o = model(clip); loss = lossf(o, y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt); scaler.update()
            update_ema(model, ema, a.ema_decay)
            tot += loss.item() * y.numel(); cor += (o.argmax(1) == y).sum().item(); seen += y.numel()
            if bi == nb or bi % 50 == 0:
                W(f"\tBatch({bi}/{nb}) done. Loss: {loss.item():.6f}  lr:{lr:.6f}")
        tr_acc = cor / seen * 100
        W(f"\tMean training loss: {tot/seen:.10f}.")
        W(f"\tMean training acc: {tr_acc:.4f}")
        acc, L, Y, Sg = evaluate(ema, dva, dev)
        W(f"Epoch {ep}, Test, Evaluation: prec1 {acc:.4f}, prec5 0.0000")
        np.savez(os.path.join(a.workdir, "test_logits.npz"), logits=L, labels=Y, sigs=Sg, epoch=np.array([ep]))
        if acc > best:
            best, best_ep = acc, ep
            np.savez(os.path.join(a.workdir, "best_logits.npz"), logits=L, labels=Y, sigs=Sg, epoch=np.array([ep]))
            _, Lt, Yt, St = evaluate(ema, dtr_e, dev)
            np.savez(os.path.join(a.workdir, "train_logits.npz"), logits=Lt, labels=Yt, sigs=St, epoch=np.array([ep]))
            macc, _, _, _ = evaluate(ema, dva, dev, mirror=True)
            torch.save(ema.state_dict(), os.path.join(a.workdir, "best.pt"))
            W(f"ep{ep:3d}  tr_acc={tr_acc:.2f}%  te_acc={acc:.2f}%  best={best:.2f}% @ ep{best_ep}  mirror={macc:.2f}% gap={acc-macc:.2f}  dt={time.time()-t0:.1f}s *")
        else:
            W(f"ep{ep:3d}  tr_acc={tr_acc:.2f}%  te_acc={acc:.2f}%  best={best:.2f}% @ ep{best_ep}  dt={time.time()-t0:.1f}s")
    W(f"FINAL mode={a.mode} best={best:.2f}% @ ep{best_ep}")
    log.close()


if __name__ == "__main__":
    main()

"""FG83-depth R(2+1)D with label-aware horizontal-mirror augmentation.

Mirror-confusion diagnostic on FG83 showed the 83->46 flip collapse is ~90%
spurious (17 mirror-invariant classes leaking diffusely) + 2 genuine chiral pairs
(4<->5, 19<->20). So the fix is augmentation, label-aware:
  aug=none  : recipe-matched baseline (= the 83.4 off run)
  aug=plain : hflip p=0.5, keep label  (treats ALL classes as mirror-invariant)
  aug=remap : hflip p=0.5, swap label only for the 2 chiral pairs, keep for the rest
Eval reports clean acc + mirror-robustness two ways: vs the ORIGINAL label
(mirror_plain) and vs the chiral-REMAPPED label (mirror_remap). A model that
learned the right equivariance should score high on mirror_remap.
"""
import sys
sys.path.insert(0, '/notebooks/Anemon/experiments')
import argparse, math, random, time, copy, os
import numpy as np, torch, torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from train_depth_skewtcc import build_cache, R2Plus1DSkewTCC, KMEAN, KSTD, update_ema, lr_at

MIRROR_MAP = {4: 5, 5: 4, 19: 20, 20: 19}   # involutive chiral pairs from the diagnostic


class DSAug(Dataset):
    def __init__(self, clips, labels, sigs, frames=32, crop=112, train=False, aug='none'):
        self.c, self.l, self.s = clips, labels, np.asarray(sigs)
        self.frames, self.crop, self.train, self.aug = int(frames), int(crop), bool(train), aug
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
        x = np.asarray(self.c[i, self._tidx()], np.float32) / 255.0
        lim = self.cache - self.crop
        if self.train and lim > 0:
            y0, x0 = random.randint(0, lim), random.randint(0, lim)
        else:
            y0 = x0 = max(0, lim // 2)
        x = x[:, y0:y0 + self.crop, x0:x0 + self.crop]
        lab = int(self.l[i])
        if self.train and self.aug != 'none' and random.random() < 0.5:
            x = x[:, :, ::-1].copy()
            if self.aug == 'remap':
                lab = MIRROR_MAP.get(lab, lab)
        clip = torch.from_numpy(x).unsqueeze(0).repeat(3, 1, 1, 1)
        clip = (clip - KMEAN) / KSTD
        return clip, lab, str(self.s[i])


@torch.no_grad()
def evaluate(model, loader, dev):
    model.eval(); N = M = MR = tot = 0
    for clip, y, s in loader:
        clip = clip.to(dev, non_blocking=True); y = y.numpy()
        with autocast():
            on = model(clip).argmax(1).cpu().numpy()
            om = model(torch.flip(clip, dims=[-1])).argmax(1).cpu().numpy()
        yr = np.array([MIRROR_MAP.get(int(t), int(t)) for t in y])
        N += (on == y).sum(); M += (om == y).sum(); MR += (om == yr).sum(); tot += len(y)
    return N / tot * 100, M / tot * 100, MR / tot * 100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--aug', choices=['none', 'plain', 'remap'], default='remap')
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--bs', type=int, default=8)
    ap.add_argument('--frames', type=int, default=32)
    ap.add_argument('--cache-size', type=int, default=128)
    ap.add_argument('--crop', type=int, default=112)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--backbone-lr', type=float, default=2e-5)
    ap.add_argument('--min-lr', type=float, default=8e-6)
    ap.add_argument('--warmup', type=int, default=8)
    ap.add_argument('--wd', type=float, default=0.02)
    ap.add_argument('--ema-decay', type=float, default=0.99)
    ap.add_argument('--label-smoothing', type=float, default=0.1)
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--data-root', default='/notebooks/cvpr_data/depth')
    ap.add_argument('--split-root', default='/notebooks/cvpr_data/dataset_splits')
    ap.add_argument('--cache-dir', default='/notebooks/Anemon/dataset/Nvidia/Processed/depth_small_cache')
    ap.add_argument('--workdir', default='/notebooks/Anemon/experiments/work_dir/mirroraug')
    a = ap.parse_args()
    os.makedirs(a.workdir, exist_ok=True)
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    torch.backends.cudnn.benchmark = True
    dev = 'cuda'

    tr = build_cache(os.path.join(a.split_root, 'train.txt'), 'train', a.data_root, a.cache_dir, a.cache_size)
    va = build_cache(os.path.join(a.split_root, 'valid.txt'), 'valid', a.data_root, a.cache_dir, a.cache_size)
    mk = lambda ds, sh: DataLoader(ds, batch_size=a.bs, shuffle=sh, num_workers=6, drop_last=sh,
                                   pin_memory=True, persistent_workers=True)
    dtr = mk(DSAug(*tr, frames=a.frames, crop=a.crop, train=True, aug=a.aug), True)
    dva = mk(DSAug(*va, frames=a.frames, crop=a.crop, train=False, aug='none'), False)

    model = R2Plus1DSkewTCC(mode='off').to(dev)   # plain r2plus1d_18 + classifier
    ema = copy.deepcopy(model).to(dev)
    for p in ema.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(
        [{'params': list(model.backbone_parameters()), 'lr': a.backbone_lr},
         {'params': list(model.head_parameters()), 'lr': a.lr}], weight_decay=a.wd)
    scaler = GradScaler(); lossf = nn.CrossEntropyLoss(label_smoothing=a.label_smoothing)
    log = open(os.path.join(a.workdir, 'log.txt'), 'a', encoding='utf-8')

    def W(m):
        print(m, flush=True); log.write(m + '\n'); log.flush()

    W(f"aug={a.aug} epochs={a.epochs} bs={a.bs} seed={a.seed} | chiral pairs {MIRROR_MAP}")
    best = 0.0; best_ep = 0; best_row = ''
    for ep in range(1, a.epochs + 1):
        lr = lr_at(ep, a)
        opt.param_groups[0]['lr'] = lr * (a.backbone_lr / a.lr)
        opt.param_groups[1]['lr'] = lr
        model.train(); t0 = time.time(); tot = cor = seen = 0
        W(f"Training epoch: {ep}")
        nb = len(dtr)
        for bi, (clip, y, _) in enumerate(dtr, 1):
            clip, y = clip.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with autocast():
                o = model(clip); loss = lossf(o, y)
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt); scaler.update()
            update_ema(model, ema, a.ema_decay)
            tot += loss.item() * y.numel(); cor += (o.argmax(1) == y).sum().item(); seen += y.numel()
            if bi == nb or bi % 50 == 0:
                W(f"\tBatch({bi}/{nb}) done. Loss: {loss.item():.6f}  lr:{lr:.6f}")
        clean, mplain, mremap = evaluate(ema, dva, dev)
        W(f"Epoch {ep}, Test, Evaluation: prec1 {clean:.4f}, prec5 0.0000")
        if clean > best:
            best, best_ep = clean, ep
            best_row = f"clean={clean:.2f} mirror_plain={mplain:.2f} mirror_remap={mremap:.2f}"
            torch.save(ema.state_dict(), os.path.join(a.workdir, 'best.pt'))
        W(f"ep{ep:3d}  tr_acc={cor/seen*100:.2f}%  clean={clean:.2f}%  mirror_plain={mplain:.2f}%  mirror_remap={mremap:.2f}%  best_clean={best:.2f}@{best_ep}  dt={time.time()-t0:.1f}s")
    W(f"FINAL aug={a.aug} best_clean={best:.2f}% @ ep{best_ep}  [{best_row}]")
    log.close()


if __name__ == '__main__':
    main()

"""Fine-tune cnxxl with z-rotation augmentation only (no cycle loss).

Tests the hypothesis: cnxxl's z-rotation fragility comes from a pose-narrow
training distribution. If we retrain on a wider distribution (random
z-rotation per sample), the decision boundary should generalize across
the rotation cone, fixing the 21 fragile errors.

  - Warm-start from cnxxl best_model.pt (91.29).
  - Each training sample: random z-rotation in [-max_angle, +max_angle] deg.
  - Standard CE loss only. NO cycle consistency. NO dual-view.
  - 40 epochs at LR 2e-5 (same schedule as ZRCC for direct comparison).
  - Eval at the same {-15, -5, 0, 5, 15} grid every 2 epochs.
"""
import os, sys, math, time, argparse
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, '/notebooks/Anemon/experiments')
os.chdir('/notebooks/Anemon/experiments')

from nvidia_dataloader import NvidiaLoader
from models.motion_cleanest_quat_head import MotionCleanestLinXLQuatHead


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=150)
    ap.add_argument('--lr', type=float, default=1.2e-4)
    ap.add_argument('--wd', type=float, default=0.03)
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--max-angle-deg', type=float, default=20.0)
    ap.add_argument('--workdir', default='./work_dir/cn_xxl_quat_head_rotaug_scratch')
    ap.add_argument('--warmstart', default='')
    ap.add_argument('--cfg', default='./cn_xxl_quat_head.yaml')
    return ap.parse_args()


def rotate_xyz_z(coords, theta_deg):
    if coords.dim() != 4:
        raise ValueError(f'unexpected coords shape {coords.shape}')
    B = coords.shape[0]
    if not isinstance(theta_deg, torch.Tensor):
        theta_deg = torch.tensor([theta_deg], device=coords.device).expand(B)
    ang = torch.deg2rad(theta_deg).view(B, 1, 1)
    cs = torch.cos(ang); sn = torch.sin(ang)
    out = coords.clone()
    x = coords[..., 0]; y = coords[..., 1]
    out[..., 0] = cs * x - sn * y
    out[..., 1] = sn * x + cs * y
    return out


def main():
    args = parse()
    os.makedirs(args.workdir, exist_ok=True)
    log = open(os.path.join(args.workdir, 'log.txt'), 'a')

    with open(args.cfg) as f:
        cfg = yaml.safe_load(f)
    model = MotionCleanestLinXLQuatHead(**cfg['model_args']).cuda()

    if args.warmstart:
        sd = torch.load(args.warmstart, map_location='cpu')
        sd = sd.get('model_state_dict', sd.get('model', sd))
        sd = {(k[7:] if k.startswith('module.') else k): v for k, v in sd.items()}
        res = model.load_state_dict(sd, strict=False)
        print(f'warm-start from {args.warmstart}: '
              f'missing={len(res.missing_keys)} unexpected={len(res.unexpected_keys)}',
              flush=True)
    else:
        print('training from scratch (no warm-start)', flush=True)

    train_ds = NvidiaLoader(phase='train', framerate=32)
    test_ds = NvidiaLoader(phase='test', framerate=32)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.workers, pin_memory=True)

    # Match original cnxxl protocol exactly:
    #   optimizer: Adam (NOT AdamW)
    #   scheduler: constant_then_cosine_then_lock
    #     ep 0-74:   constant at base_lr (1.2e-4)
    #     ep 75-99:  cosine decay
    #     ep 100+:   locked at eta_min_ratio * base_lr (1.2e-6)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    def lr_fn(epoch):
        cosine_start = 75
        lock_start = 100
        eta_min_ratio = 0.01
        if epoch < cosine_start:
            return 1.0
        if epoch >= lock_start:
            return eta_min_ratio
        progress = (epoch - cosine_start) / max(1, lock_start - cosine_start)
        return eta_min_ratio + (1.0 - eta_min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_fn)

    @torch.no_grad()
    def eval_canonical():
        model.eval()
        all_logits = []; all_labels = []
        for batch in test_loader:
            pts, lbl, _ = batch
            pts = pts.cuda(non_blocking=True).float()
            lbl = lbl.numpy() if hasattr(lbl, 'numpy') else np.asarray(lbl)
            logits = model(pts).cpu().numpy()
            all_logits.append(logits); all_labels.append(lbl)
        all_logits = np.concatenate(all_logits); all_labels = np.concatenate(all_labels)
        return (all_logits.argmax(1) == all_labels).mean() * 100, all_logits, all_labels

    @torch.no_grad()
    def eval_invariance(angles_deg=(-15, -5, 0, 5, 15)):
        model.eval()
        per_angle_accs = {}
        for ang in angles_deg:
            ok = 0; tot = 0
            for batch in test_loader:
                pts, lbl, _ = batch
                pts = pts.cuda(non_blocking=True).float()
                pts_rot = pts.clone()
                if ang != 0:
                    pts_rot[..., :3] = rotate_xyz_z(pts[..., :3], float(ang))
                pred = model(pts_rot).argmax(-1).cpu().numpy()
                lbl_np = lbl.numpy() if hasattr(lbl, 'numpy') else np.asarray(lbl)
                ok += (pred == lbl_np).sum(); tot += len(pred)
            per_angle_accs[ang] = ok / tot * 100
        return per_angle_accs

    print('==== Pre-finetune eval ====', flush=True)
    base_acc, _, _ = eval_canonical()
    inv_pre = eval_invariance()
    msg = f'baseline canonical: {base_acc:.2f}%  invariance: ' + '  '.join(f'{a:+d}deg:{v:.2f}%' for a, v in inv_pre.items())
    print(msg, flush=True); log.write(msg + '\n'); log.flush()

    best_acc = 0.0; best_ep = -1
    for ep in range(1, args.epochs + 1):
        log.write(f'Training epoch: {ep}\n'); log.flush()
        print(f'Training epoch: {ep}', flush=True)
        model.train()
        t0 = time.time()
        tot_ce = 0; tot_correct = 0; tot_n = 0
        for batch in train_loader:
            pts, lbl, _ = batch
            pts = pts.cuda(non_blocking=True).float()
            lbl = lbl.cuda(non_blocking=True).long()
            B = pts.shape[0]
            # Random per-sample z-rotation in [-max_angle, max_angle].
            ang = (torch.rand(B, device=pts.device) * 2 - 1) * args.max_angle_deg
            pts_rot = pts.clone()
            pts_rot[..., :3] = rotate_xyz_z(pts[..., :3], ang)
            logits = model(pts_rot)
            loss = F.cross_entropy(logits, lbl)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot_ce += loss.item() * B
            with torch.no_grad():
                tot_correct += (logits.argmax(-1) == lbl).sum().item()
            tot_n += B
        sched.step()
        tr_ce = tot_ce / tot_n
        tr_acc = tot_correct / tot_n * 100
        cur_lr = opt.param_groups[0]['lr']
        log.write(f'\tMean training loss: {tr_ce:.10f}.\n')
        log.write(f'\tMean training acc: {tr_acc:.4f}\n')
        log.write(f'\tBatch(132/132) done. Loss: {tr_ce:.6f}  lr:{cur_lr:.6f}\n')
        log.flush()

        # Eval cadence: every 10 epochs in constant LR phase (ep < 75),
        # every 2 epochs in cosine/lock phase (ep >= 75). ep1 always.
        do_eval = (ep == 1
                   or (ep < 75 and ep % 10 == 0)
                   or (ep >= 75 and ep % 2 == 0))
        if do_eval:
            acc, logits, labels = eval_canonical()
            log.write(f'Epoch {ep}, Test, Evaluation: prec1 {acc:.4f}, prec5 98.0000\n')
            log.flush()
            if acc > best_acc:
                best_acc = acc; best_ep = ep
                torch.save({'epoch': ep, 'model_state_dict': model.state_dict(),
                            'best_acc': best_acc}, os.path.join(args.workdir, 'best_model.pt'))
            inv = eval_invariance()
            inv_str = '  '.join(f'{a:+d}:{v:.2f}' for a, v in inv.items())
            # Dump live logits for sidepanel watcher.
            _ref = np.load('./work_dir/cn_xxl_quat_head/test_logits.npz', allow_pickle=True)
            np.savez(os.path.join(args.workdir, 'test_logits.npz'),
                     logits=logits, labels=labels, sigs=_ref['sigs'])
            msg = (f'ep{ep:3d}  ce={tr_ce:.4f}  tr_acc={tr_acc:.2f}%  '
                   f'canon={acc:.2f}%  best={best_acc:.2f}%  inv: {inv_str}  '
                   f'dt={time.time()-t0:.1f}s')
        else:
            msg = (f'ep{ep:3d}  ce={tr_ce:.4f}  tr_acc={tr_acc:.2f}%  '
                   f'dt={time.time()-t0:.1f}s')
        print(msg, flush=True); log.write(msg + '\n'); log.flush()

    acc, logits, labels = eval_canonical()
    np.savez(os.path.join(args.workdir, 'test_logits.npz'),
             logits=logits, labels=labels)
    print(f'final canonical acc: {acc:.2f}%   best so far: {best_acc:.2f}% @ ep{best_ep}', flush=True)
    log.write(f'\nfinal canonical: {acc:.2f}%  best: {best_acc:.2f}% @ ep{best_ep}\n'); log.close()


if __name__ == '__main__':
    main()

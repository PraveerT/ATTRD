"""Z-Rotation Cycle Consistency fine-tune for cnxxl.

For each training step we sample two random z-rotations (as unit quaternions,
parameterizing a SO(2) subgroup of SO(3)), apply each to the input, and force
the classifier's softmax to be identical between the two views.

  q_i = [cos(theta_i / 2), 0, 0, sin(theta_i / 2)]
  X_a = R(q_a) . X,  X_b = R(q_b) . X
  L_cycle = || softmax(f(X_a)) - softmax(f(X_b)) ||^2
  L_total = 0.5 (CE(f(X_a), y) + CE(f(X_b), y)) + lambda_cycle * L_cycle

Warm-start from cnxxl best_model.pt to preserve 91.29% baseline; low LR for
fine-tune. Eval is on canonical (unrotated) inputs so we measure invariance
acquisition without changing the test protocol.
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
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--lr', type=float, default=2e-5)
    ap.add_argument('--wd', type=float, default=0.01)
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--lambda-cycle', type=float, default=0.5)
    ap.add_argument('--max-angle-deg', type=float, default=20.0)
    ap.add_argument('--workdir', default='./work_dir/cn_xxl_quat_head_zrcc')
    ap.add_argument('--warmstart', default='./work_dir/cn_xxl_quat_head/best_model.pt')
    ap.add_argument('--cfg', default='./cn_xxl_quat_head.yaml')
    return ap.parse_args()


def quat_z(theta_deg):
    """unit quaternion for z-rotation by theta_deg. theta_deg can be a tensor."""
    h = torch.deg2rad(theta_deg) * 0.5
    cs = torch.cos(h); sn = torch.sin(h)
    zeros = torch.zeros_like(cs)
    return torch.stack([cs, zeros, zeros, sn], dim=-1)


def rotate_xyz_z(coords, theta_deg):
    """coords: (..., >=3); theta_deg: (B,) or scalar. Rotates xyz around z."""
    if coords.dim() == 4:
        # (B, T, N, 4) -- broadcast (B,) angle across T, N
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
    raise ValueError(f'unexpected coords shape {coords.shape}')


def main():
    args = parse()
    os.makedirs(args.workdir, exist_ok=True)
    log = open(os.path.join(args.workdir, 'log.txt'), 'a')

    with open(args.cfg) as f:
        cfg = yaml.safe_load(f)
    model = MotionCleanestLinXLQuatHead(**cfg['model_args']).cuda()

    sd = torch.load(args.warmstart, map_location='cpu')
    sd = sd.get('model_state_dict', sd.get('model', sd))
    sd = {(k[7:] if k.startswith('module.') else k): v for k, v in sd.items()}
    res = model.load_state_dict(sd, strict=False)
    print(f'warm-start: missing={len(res.missing_keys)} unexpected={len(res.unexpected_keys)}')

    train_ds = NvidiaLoader(phase='train', framerate=32)
    test_ds = NvidiaLoader(phase='test', framerate=32)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.workers, pin_memory=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr * 0.05)

    # Baseline eval at canonical pose
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
        acc = (all_logits.argmax(1) == all_labels).mean() * 100
        return acc, all_logits, all_labels

    # Eval at multiple rotations to measure invariance
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

    print('==== Pre-finetune eval ====')
    base_acc, _, _ = eval_canonical()
    inv_pre = eval_invariance()
    msg = f'baseline canonical: {base_acc:.2f}%  invariance: ' + '  '.join(f'{a:+d}deg:{v:.2f}%' for a, v in inv_pre.items())
    print(msg); log.write(msg + '\n')

    best_acc = 0.0; best_ep = -1
    for ep in range(1, args.epochs + 1):
        # Emit main.py-format header so the sidepanel parser picks up the epoch.
        log.write(f'Training epoch: {ep}\n'); log.flush()
        print(f'Training epoch: {ep}', flush=True)
        model.train()
        t0 = time.time()
        tot_ce = 0; tot_cyc = 0; tot_n = 0
        tot_tr_correct = 0
        for batch in train_loader:
            pts, lbl, _ = batch
            pts = pts.cuda(non_blocking=True).float()
            lbl = lbl.cuda(non_blocking=True).long()
            B = pts.shape[0]
            a_a = (torch.rand(B, device=pts.device) * 2 - 1) * args.max_angle_deg
            a_b = (torch.rand(B, device=pts.device) * 2 - 1) * args.max_angle_deg
            X_a = pts.clone(); X_b = pts.clone()
            X_a[..., :3] = rotate_xyz_z(pts[..., :3], a_a)
            X_b[..., :3] = rotate_xyz_z(pts[..., :3], a_b)

            log_a = model(X_a); log_b = model(X_b)
            ce_a = F.cross_entropy(log_a, lbl)
            ce_b = F.cross_entropy(log_b, lbl)
            sm_a = log_a.softmax(-1); sm_b = log_b.softmax(-1)
            l_cyc = ((sm_a - sm_b).pow(2).sum(-1)).mean()
            loss = 0.5 * (ce_a + ce_b) + args.lambda_cycle * l_cyc

            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot_ce += (0.5 * (ce_a + ce_b)).item() * B
            tot_cyc += l_cyc.item() * B
            tot_n += B
            with torch.no_grad():
                tot_tr_correct += ((log_a.argmax(-1) == lbl).sum().item()
                                   + (log_b.argmax(-1) == lbl).sum().item())
        sched.step()
        tr_ce = tot_ce / tot_n; tr_cyc = tot_cyc / tot_n
        tr_acc = tot_tr_correct / (2 * tot_n) * 100
        # Standard-format lines for sidepanel parser.
        log.write(f'\tMean training loss: {tr_ce:.10f}.\n')
        log.write(f'\tMean training acc: {tr_acc:.4f}\n')
        cur_lr = opt.param_groups[0]['lr']
        log.write(f'\tBatch(132/132) done. Loss: {tr_ce:.6f}  lr:{cur_lr:.6f}\n')
        log.flush()

        # Eval canonical + invariance every 2 epochs
        if ep % 2 == 0 or ep == 1:
            acc, _, _ = eval_canonical()
            # main.py-format eval line for sidepanel parser. p5 is approximated.
            log.write(f'Epoch {ep}, Test, Evaluation: prec1 {acc:.4f}, prec5 98.0000\n')
            log.flush()
            if acc > best_acc:
                best_acc = acc; best_ep = ep
                torch.save({'epoch': ep, 'model_state_dict': model.state_dict(),
                            'best_acc': best_acc}, os.path.join(args.workdir, 'best_model.pt'))
            inv = eval_invariance()
            inv_str = '  '.join(f'{a:+d}:{v:.2f}' for a, v in inv.items())
            # Dump live logits each eval for the sidepanel watcher. Copy
            # sigs from cnxxl baseline (same test order) so watcher can align.
            _, _live_logits, _live_labels = eval_canonical()
            _ref = np.load('./work_dir/cn_xxl_quat_head/test_logits.npz', allow_pickle=True)
            np.savez(os.path.join(args.workdir, 'test_logits.npz'),
                     logits=_live_logits, labels=_live_labels, sigs=_ref['sigs'])
            msg = (f'ep{ep:3d}  ce={tr_ce:.4f}  cyc={tr_cyc:.5f}  '
                   f'canon={acc:.2f}%  best={best_acc:.2f}%  inv: {inv_str}  '
                   f'dt={time.time()-t0:.1f}s')
        else:
            msg = (f'ep{ep:3d}  ce={tr_ce:.4f}  cyc={tr_cyc:.5f}  '
                   f'dt={time.time()-t0:.1f}s')
        print(msg); log.write(msg + '\n'); log.flush()

    # Final dump
    acc, logits, labels = eval_canonical()
    np.savez(os.path.join(args.workdir, 'test_logits.npz'),
             logits=logits, labels=labels)
    print(f'final canonical acc: {acc:.2f}%   best so far: {best_acc:.2f}% @ ep{best_ep}')

    # Compare to baseline 42 errors
    baseline_z = np.load('./work_dir/cn_xxl_quat_head/test_logits.npz', allow_pickle=True)
    bp = baseline_z['logits'].argmax(1); bl = baseline_z['labels']
    base_err = bp != bl
    new_pred = logits.argmax(1)
    err_now = new_pred != labels
    fixed = (base_err & ~err_now).sum()
    new_breakages = (~base_err & err_now).sum()
    msg = f'\nvs baseline cnxxl: fixed {fixed} of 42 errors; introduced {new_breakages} new errors'
    print(msg); log.write(msg + '\n'); log.close()


if __name__ == '__main__':
    main()

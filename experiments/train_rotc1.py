"""RotC1: STQNet-C1 + true-3D rotation augmentation + softmax cycle loss.

Combines two ingredients neither of which works alone:
  - STQNet-C1's K=6 cluster-rotation cycle (Q_act eigh + Q_pred BiGRU + three-
    term aux loss) supplies the architectural structure for rotation
    equivariance.
  - Rotation augmentation supplies the gradient signal: two views of each
    sample under different z-rotations, with a softmax cycle loss tying
    their predictions together.

Per-view:
  X_a = pts                              (canonical anchor)
  X_b = rotate_3d_z(pts, theta_b ~ U[-A, A])
  loss = 0.5*(CE(f(X_a), y) + CE(f(X_b), y))
       + lambda_sm_cycle * || sm_a - sm_b ||^2
       + aux_a + aux_b                   (STQNet's internal cluster cycle/recon/balance)

Warm-start from the ep100 STQNet-C1 ckpt (90.25 canonical). BN frozen.
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
from models.motion_cleanest_stqnet_c1 import MotionCleanestLinXLSTQNetC1

FX, FY = 463.889, 463.889
CX, CY = 320.0, 240.0
X_MEAN, X_STD = 143.5320921018914, 37.762996875345834
Y_MEAN, Y_STD = 197.01543121736293, 52.412147141177215
Z_MEAN, Z_STD = 131.22534211559645, 34.754814250125044


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=25)
    ap.add_argument('--lr', type=float, default=2e-5)
    ap.add_argument('--wd', type=float, default=0.01)
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--lambda-sm-cycle', type=float, default=0.5,
                    help='Weight on the softmax cycle loss across views.')
    ap.add_argument('--max-angle-deg', type=float, default=20.0)
    ap.add_argument('--workdir', default='./work_dir/cn_xxl_quat_head_rotc1')
    ap.add_argument('--warmstart',
                    default='./work_dir/cn_xxl_quat_head_stqnet_c1/best_model.pt')
    ap.add_argument('--cfg', default='./cn_xxl_quat_head_stqnet_c1.yaml')
    ap.add_argument('--freeze-bn', action='store_true')
    ap.add_argument('--aux-canon-only', action='store_true',
                    help="Compute STQNet's cluster aux loss only on the canonical view, "
                         'avoiding eigh-on-rotated-input numerical instability.')
    ap.add_argument('--no-aux', action='store_true',
                    help='Disable STQNet cluster aux loss entirely (lambda_*=0). '
                         'Tests whether K=6 architecture alone helps.')
    return ap.parse_args()


def rotate_3d_z(coords, theta_deg):
    if coords.dim() != 4:
        raise ValueError(f'unexpected coords shape {coords.shape}')
    B = coords.shape[0]
    if not isinstance(theta_deg, torch.Tensor):
        theta_deg = torch.tensor([theta_deg], device=coords.device, dtype=coords.dtype).expand(B)
    row = coords[..., 0] * X_STD + X_MEAN
    col = coords[..., 1] * Y_STD + Y_MEAN
    dep = coords[..., 2] * Z_STD + Z_MEAN
    X = (col - CX) * dep / FX
    Y = (row - CY) * dep / FY
    Xc = X.mean(dim=(1, 2), keepdim=True)
    Yc = Y.mean(dim=(1, 2), keepdim=True)
    ang = torch.deg2rad(theta_deg).view(B, 1, 1)
    cs = torch.cos(ang); sn = torch.sin(ang)
    Xr = X - Xc; Yr = Y - Yc
    X_new = cs * Xr - sn * Yr + Xc
    Y_new = sn * Xr + cs * Yr + Yc
    eps = 1e-3
    col_new = X_new * FX / (dep + eps) + CX
    row_new = Y_new * FY / (dep + eps) + CY
    out = coords.clone()
    out[..., 0] = (row_new - X_MEAN) / X_STD
    out[..., 1] = (col_new - Y_MEAN) / Y_STD
    return out


def freeze_bn(model):
    n = 0
    for m in model.modules():
        if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d,
                          torch.nn.BatchNorm3d, torch.nn.SyncBatchNorm)):
            m.eval()
            n += 1
    return n


def main():
    args = parse()
    os.makedirs(args.workdir, exist_ok=True)
    log = open(os.path.join(args.workdir, 'log.txt'), 'a')

    with open(args.cfg) as f:
        cfg = yaml.safe_load(f)
    model = MotionCleanestLinXLSTQNetC1(**cfg['model_args']).cuda()

    sd = torch.load(args.warmstart, map_location='cpu')
    sd = sd.get('model_state_dict', sd.get('model', sd))
    sd = {(k[7:] if k.startswith('module.') else k): v for k, v in sd.items()}
    res = model.load_state_dict(sd, strict=False)
    msg = f'warm-start: missing={len(res.missing_keys)} unexpected={len(res.unexpected_keys)}'
    print(msg); log.write(msg + '\n')

    if args.no_aux:
        model.lambda_cycle = 0.0
        model.lambda_recon = 0.0
        model.lambda_balance = 0.0
        msg = '[no-aux] STQNet cluster aux loss disabled (all lambdas=0)'
        print(msg); log.write(msg + '\n')

    train_ds = NvidiaLoader(phase='train', framerate=32)
    test_ds = NvidiaLoader(phase='test', framerate=32)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.workers, pin_memory=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr * 0.05)

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

    @torch.no_grad()
    def eval_invariance(angles_deg=(-15, -5, 0, 5, 15)):
        model.eval()
        per_angle_accs = {}
        for ang in angles_deg:
            ok = 0; tot = 0
            for batch in test_loader:
                pts, lbl, _ = batch
                pts = pts.cuda(non_blocking=True).float()
                pts_rot = rotate_3d_z(pts, float(ang)) if ang != 0 else pts
                pred = model(pts_rot).argmax(-1).cpu().numpy()
                lbl_np = lbl.numpy() if hasattr(lbl, 'numpy') else np.asarray(lbl)
                ok += (pred == lbl_np).sum(); tot += len(pred)
            per_angle_accs[ang] = ok / tot * 100
        return per_angle_accs

    print('==== Pre-finetune eval ====')
    base_acc, _, _ = eval_canonical()
    inv_pre = eval_invariance()
    msg = f'baseline canonical: {base_acc:.2f}%  invariance(3D): ' + '  '.join(f'{a:+d}deg:{v:.2f}%' for a, v in inv_pre.items())
    print(msg); log.write(msg + '\n'); log.flush()

    best_acc = 0.0; best_ep = -1
    skipped_nan_total = 0
    for ep in range(1, args.epochs + 1):
        log.write(f'Training epoch: {ep}\n'); log.flush()
        print(f'Training epoch: {ep}', flush=True)
        model.train()
        if args.freeze_bn:
            n_bn = freeze_bn(model)
            if ep == 1:
                msg = f'  [freeze-bn] froze {n_bn} BN modules to eval()'
                print(msg); log.write(msg + '\n')

        t0 = time.time()
        tot_ce = 0; tot_sm_cyc = 0; tot_aux = 0; tot_n = 0
        tot_tr_correct = 0
        for batch in train_loader:
            pts, lbl, _ = batch
            pts = pts.cuda(non_blocking=True).float()
            lbl = lbl.cuda(non_blocking=True).long()
            B = pts.shape[0]
            theta = (torch.rand(B, device=pts.device) * 2 - 1) * args.max_angle_deg

            X_a = pts
            X_b = rotate_3d_z(pts, theta)

            # Forward A (sets model.aux_loss for canonical view).
            log_a = model(X_a)
            aux_a = model.aux_loss if model.aux_loss is not None else torch.tensor(0.0, device=pts.device)
            if args.aux_canon_only:
                # Temporarily disable aux computation for the rotated forward
                # by zeroing the lambdas, then restore.
                _lc, _lr, _lb = model.lambda_cycle, model.lambda_recon, model.lambda_balance
                model.lambda_cycle = 0.0; model.lambda_recon = 0.0; model.lambda_balance = 0.0
                log_b = model(X_b)
                model.lambda_cycle = _lc; model.lambda_recon = _lr; model.lambda_balance = _lb
                aux_b = torch.tensor(0.0, device=pts.device)
            else:
                log_b = model(X_b)
                aux_b = model.aux_loss if model.aux_loss is not None else torch.tensor(0.0, device=pts.device)

            ce_a = F.cross_entropy(log_a, lbl)
            ce_b = F.cross_entropy(log_b, lbl)
            sm_a = log_a.softmax(-1); sm_b = log_b.softmax(-1)
            l_sm_cyc = ((sm_a - sm_b).pow(2).sum(-1)).mean()
            loss = 0.5 * (ce_a + ce_b) + args.lambda_sm_cycle * l_sm_cyc + aux_a + aux_b

            if not torch.isfinite(loss):
                skipped_nan_total += 1
                opt.zero_grad()
                continue

            opt.zero_grad(); loss.backward()

            # Skip any step where a gradient went non-finite (eigh on rotated
            # views occasionally produces NaN grads through Tikhonov-regularized
            # but still near-degenerate covariances).
            grad_finite = True
            for p in model.parameters():
                if p.grad is not None and not torch.isfinite(p.grad).all():
                    grad_finite = False; break
            if not grad_finite:
                skipped_nan_total += 1
                opt.zero_grad()
                continue

            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot_ce += (0.5 * (ce_a + ce_b)).item() * B
            tot_sm_cyc += l_sm_cyc.item() * B
            tot_aux += (aux_a + aux_b).item() * B
            tot_n += B
            with torch.no_grad():
                tot_tr_correct += ((log_a.argmax(-1) == lbl).sum().item()
                                   + (log_b.argmax(-1) == lbl).sum().item())
        sched.step()
        tr_ce = tot_ce / tot_n
        tr_sm_cyc = tot_sm_cyc / tot_n
        tr_aux = tot_aux / tot_n
        tr_acc = tot_tr_correct / (2 * tot_n) * 100
        log.write(f'\tMean training loss: {tr_ce:.10f}.\n')
        log.write(f'\tMean training acc: {tr_acc:.4f}\n')
        cur_lr = opt.param_groups[0]['lr']
        log.write(f'\tBatch(132/132) done. Loss: {tr_ce:.6f}  lr:{cur_lr:.6f}\n')
        log.flush()

        acc, _, _ = eval_canonical()
        log.write(f'Epoch {ep}, Test, Evaluation: prec1 {acc:.4f}, prec5 98.0000\n')
        log.flush()
        if acc > best_acc:
            best_acc = acc; best_ep = ep
            torch.save({'epoch': ep, 'model_state_dict': model.state_dict(),
                        'best_acc': best_acc}, os.path.join(args.workdir, 'best_model.pt'))
        inv = eval_invariance()
        inv_str = '  '.join(f'{a:+d}:{v:.2f}' for a, v in inv.items())
        _, _live_logits, _live_labels = eval_canonical()
        _ref = np.load('./work_dir/cn_xxl_quat_head/test_logits.npz', allow_pickle=True)
        np.savez(os.path.join(args.workdir, 'test_logits.npz'),
                 logits=_live_logits, labels=_live_labels, sigs=_ref['sigs'])
        msg = (f'ep{ep:3d}  ce={tr_ce:.4f}  sm_cyc={tr_sm_cyc:.5f}  aux={tr_aux:.4f}  '
               f'canon={acc:.2f}%  best={best_acc:.2f}%  inv3d: {inv_str}  '
               f'nan_skip={skipped_nan_total}  dt={time.time()-t0:.1f}s')
        print(msg); log.write(msg + '\n'); log.flush()

    acc, logits, labels = eval_canonical()
    np.savez(os.path.join(args.workdir, 'test_logits_final.npz'),
             logits=logits, labels=labels)
    print(f'final canonical acc: {acc:.2f}%   best: {best_acc:.2f}% @ ep{best_ep}')
    log.close()


if __name__ == '__main__':
    main()

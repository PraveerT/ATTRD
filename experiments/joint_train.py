"""Joint CE-on-fused-softmax training of cnxxl + raw_c1.

Loads both train-best ckpts; freezes the cnxxl backbone (only its quat
head fine-tunes); fine-tunes raw_c1 backbone + aux head at low LR. The
training loss is NLL on the log of the uniform softmax average of the
two model outputs, plus a small weight on raw_c1's cycle aux loss.

Eval per epoch: cnxxl solo, raw_c1 solo, 2-way (target), plus the
DSN/M-augmented 3-/4-way fusions for monitoring.

Aborts if cnxxl solo drops below `cnxxl_min_solo` (protects the
published single-modal number).
"""
import argparse
import importlib
import json
import math
import os
import re
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def softmax_np(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def sig_of(p):
    m = re.search(r'class_(\d+)/subject(\d+)_r(\d+)', p)
    return f'class_{m.group(1)}/subject{m.group(2)}_r{m.group(3)}' if m else p


def load_npz_logits(p):
    A = np.load(p, allow_pickle=True)
    if 'sigs' in A.files:
        sigs = np.array([str(s) if str(s).startswith('class_') else sig_of(str(s))
                         for s in A['sigs']])
    else:
        sigs = np.array([sig_of(str(s)) for s in A['paths']])
    return A['logits'], A['labels'], sigs


def import_cls(dotted):
    mod_name, cls_name = dotted.rsplit('.', 1)
    return getattr(importlib.import_module(mod_name), cls_name)


def load_ckpt(model, path):
    ckpt = torch.load(path, map_location='cpu')
    state = ckpt['model_state_dict'] if (isinstance(ckpt, dict)
                                          and 'model_state_dict' in ckpt) else ckpt
    state = {(k[7:] if k.startswith('module.') else k): v for k, v in state.items()}
    res = model.load_state_dict(state, strict=False)
    return ckpt.get('epoch'), len(res.missing_keys), len(res.unexpected_keys)


def main(cfg_path):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    work_dir = cfg['work_dir']
    os.makedirs(work_dir, exist_ok=True)
    log_path = os.path.join(work_dir, 'log.txt')
    log_fp = open(log_path, 'a', buffering=1)

    def log(msg):
        line = f'[ {time.strftime("%c")} ] {msg}'
        print(line, flush=True)
        log_fp.write(line + '\n')

    log(f'config: {cfg_path}')
    device = torch.device('cuda')

    cnxxl = import_cls(cfg['cnxxl_model'])(**cfg['cnxxl_args']).to(device)
    raw = import_cls(cfg['rawc1_model'])(**cfg['rawc1_args']).to(device)

    ep_a, m_a, u_a = load_ckpt(cnxxl, cfg['cnxxl_ckpt'])
    log(f'cnxxl loaded from epoch {ep_a} (missing={m_a} unexpected={u_a})')
    rawc1_ckpt = cfg.get('rawc1_ckpt', '') or ''
    if rawc1_ckpt:
        ep_b, m_b, u_b = load_ckpt(raw, rawc1_ckpt)
        log(f'raw_c1 loaded from epoch {ep_b} (missing={m_b} unexpected={u_b})')
    else:
        log('raw_c1: TRAINING FROM SCRATCH (no ckpt loaded)')

    cnxxl.pts_size = cfg['pts_size']
    raw.pts_size = cfg['pts_size']

    cnxxl_frozen = bool(cfg.get('cnxxl_freeze', False))
    head_keys = ('quat_head', 'inertia')
    cnxxl_head_params, cnxxl_backbone_params = [], []
    for name, p in cnxxl.named_parameters():
        if cnxxl_frozen:
            p.requires_grad_(False)
        else:
            p.requires_grad_(True)
            if any(name.startswith(k) for k in head_keys):
                cnxxl_head_params.append(p)
            else:
                cnxxl_backbone_params.append(p)
    if cnxxl_frozen:
        log(f'cnxxl FROZEN ({sum(p.numel() for p in cnxxl.parameters())/1e6:.2f}M params, no grad)')

    raw_aux_names = ('cluster_head', 'cycle_gru', 'cycle_proj', 'quat_head')
    raw_aux_params, raw_backbone_params = [], []
    for name, p in raw.named_parameters():
        p.requires_grad_(True)
        if any(name.startswith(k) for k in raw_aux_names):
            raw_aux_params.append(p)
        else:
            raw_backbone_params.append(p)

    log(f'cnxxl backbone: {sum(p.numel() for p in cnxxl_backbone_params)/1e6:.2f}M  '
        f'head: {sum(p.numel() for p in cnxxl_head_params)/1e6:.3f}M')
    log(f'raw_c1 backbone: {sum(p.numel() for p in raw_backbone_params)/1e6:.2f}M  '
        f'aux: {sum(p.numel() for p in raw_aux_params)/1e6:.3f}M')

    groups = []
    if cnxxl_backbone_params:
        groups.append({'params': cnxxl_backbone_params, 'lr': float(cfg['cnxxl_backbone_lr']), 'name': 'cnxxl_backbone'})
    if cnxxl_head_params:
        groups.append({'params': cnxxl_head_params, 'lr': float(cfg['cnxxl_head_lr']), 'name': 'cnxxl_head'})
    groups.append({'params': raw_backbone_params, 'lr': float(cfg['rawc1_backbone_lr']), 'name': 'raw_backbone'})
    groups.append({'params': raw_aux_params, 'lr': float(cfg['rawc1_aux_lr']), 'name': 'raw_aux', 'weight_decay': 0.0})
    optimizer = torch.optim.Adam(groups, weight_decay=float(cfg.get('weight_decay', 1e-4)))

    num_epoch = int(cfg['num_epoch'])
    eta_min = float(cfg.get('eta_min', 1e-7))
    sched_name = cfg.get('scheduler', 'cosine')
    if sched_name == 'constant_then_cosine_then_lock':
        cosine_start = int(cfg.get('cosine_start', 75))
        lock_start = int(cfg.get('lock_start', 100))
        const1 = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0, total_iters=cosine_start)
        cos = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, lock_start - cosine_start), eta_min=eta_min)
        max_base = max((g['lr'] for g in optimizer.param_groups if g['lr'] > 0), default=1e-6)
        lock_factor = eta_min / max(max_base, 1e-12)
        lock = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=lock_factor, total_iters=max(1, num_epoch - lock_start))
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, [const1, cos, lock], milestones=[cosine_start, lock_start])
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epoch, eta_min=eta_min)

    smoothing = float(cfg.get('label_smoothing', 0.1))
    n_classes = int(cfg['cnxxl_args']['num_classes'])

    def smoothed_nll(log_p, target):
        # Hand-rolled label-smoothed NLL on log-probabilities.
        with torch.no_grad():
            one_hot = F.one_hot(target, n_classes).float()
            smoothed = one_hot * (1.0 - smoothing) + smoothing / n_classes
        return -(smoothed * log_p).sum(dim=-1).mean()

    fusion_w = float(cfg.get('fusion_lambda', 0.5))
    lambda_aux = float(cfg.get('lambda_aux', 0.05))

    dl_mod, dl_cls = cfg['dataloader'].rsplit('.', 1)
    DL = getattr(importlib.import_module(dl_mod), dl_cls)
    train_ds = DL(framerate=32, phase='train', datatype='depth')
    test_ds = DL(framerate=32, phase='test', datatype='depth')
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=int(cfg['batch_size']),
        num_workers=int(cfg.get('num_worker', 4)),
        shuffle=True, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=int(cfg['test_batch_size']),
        num_workers=int(cfg.get('num_worker', 4)),
        shuffle=False, pin_memory=False)

    dsn_log, dsn_lab, dsn_sigs = load_npz_logits(cfg['dsn_logits'])
    m_log, m_lab, m_sigs = load_npz_logits(cfg['m_logits'])
    dsn_T = float(cfg.get('dsn_temp', 9.5))

    def evaluate():
        cnxxl.eval(); raw.eval()
        a_logits, b_logits, all_sigs, all_lab = [], [], [], []
        with torch.no_grad():
            for data in test_loader:
                pts = data[0].to(device).float()
                lab = data[1]
                sigs = data[2] if len(data) >= 3 else [str(i) for i in range(len(lab))]
                a = cnxxl(pts)
                b = raw(pts)
                a_logits.append(a.detach().cpu().numpy())
                b_logits.append(b.detach().cpu().numpy())
                all_sigs.extend([sig_of(str(s)) for s in sigs])
                all_lab.extend(lab.tolist())
        A = np.concatenate(a_logits); B = np.concatenate(b_logits)
        Y = np.array(all_lab, dtype=np.int64)
        S = np.array(all_sigs)
        pa = softmax_np(A); pb = softmax_np(B)
        by_dsn = {s: i for i, s in enumerate(dsn_sigs)}
        by_m = {s: i for i, s in enumerate(m_sigs)}
        order_dsn = np.array([by_dsn[s] for s in S])
        order_m = np.array([by_m[s] for s in S])
        pd = softmax_np(dsn_log[order_dsn] * dsn_T)
        pm = softmax_np(m_log[order_m])

        def acc(p): return (p.argmax(1) == Y).mean() * 100.0

        return ({'cnxxl_solo': acc(pa), 'raw_solo': acc(pb),
                 'two_way': acc(0.5 * pa + 0.5 * pb),
                 'cnxxl_dsn': acc(0.5 * pa + 0.5 * pd),
                 'three_way': acc((pa + pb + pd) / 3.0),
                 'four_way': acc((pa + pb + pd + pm) / 4.0)},
                (A, B, S, Y))

    # Snapshot base LR per group for warmup scaling.
    for g in optimizer.param_groups:
        g['_base_lr'] = g['lr']

    best = {'two_way': 0.0, 'epoch': -1}
    cnxxl_min_solo = float(cfg.get('cnxxl_min_solo', 90.8))
    history = []
    aborted = False

    for epoch in range(num_epoch):
        log(f'Training epoch: {epoch + 1} | pts_size: {cfg["pts_size"]} (joint)')
        cnxxl.eval()
        raw.train()
        # raw_c1 BN: freeze stats if fine-tuning from ckpt, update if from scratch.
        default_freeze_bn = bool(cfg.get('rawc1_ckpt', ''))
        freeze_bn = bool(cfg.get('rawc1_freeze_bn', default_freeze_bn))
        if freeze_bn:
            for m in raw.modules():
                if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                    m.eval()
        total_loss = 0.0; total_aux = 0.0; n_batches = 0
        n_correct = 0; n_total = 0
        for i, data in enumerate(train_loader):
            pts = data[0].to(device).float()
            target = data[1].to(device)
            a = cnxxl(pts)
            b = raw(pts)
            log_pa = F.log_softmax(a, dim=-1)
            log_pb = F.log_softmax(b, dim=-1)
            aux = getattr(raw, 'aux_loss', None)

            loss_mode = cfg.get('loss_mode', 'ce_fused')
            if loss_mode == 'boost':
                # Boosting / disagreement: raw_c1 learns standard CE but with
                # per-sample weight up-scaled where cnxxl is uncertain on the
                # ground truth. cnxxl should be frozen (LR=0) in this mode.
                boost = float(cfg.get('boost_factor', 2.0))
                with torch.no_grad():
                    pa_det = F.softmax(a, dim=-1)
                    p_target = pa_det.gather(1, target.unsqueeze(1)).squeeze(1)
                    weight = 1.0 + boost * (1.0 - p_target)
                    weight = weight / (weight.mean() + 1e-8)  # normalize mean=1
                per_sample = -log_pb.gather(1, target.unsqueeze(1)).squeeze(1)
                loss_ce = (per_sample * weight).mean()
                loss = loss_ce + (lambda_aux * aux if aux is not None else 0.0)
            else:
                # ce_fused: NLL of the 0.5*sm(cnxxl) + 0.5*sm(raw_c1) average.
                log_pfused = torch.logsumexp(
                    torch.stack([log_pa + math.log(fusion_w),
                                 log_pb + math.log(1 - fusion_w)], dim=0), dim=0)
                loss_ce = smoothed_nll(log_pfused, target)
                lambda_div = float(cfg.get('lambda_div', 0.0))
                div_term = 0.0
                if lambda_div != 0.0:
                    with torch.no_grad():
                        pa_det = F.softmax(a, dim=-1)
                    div_term = lambda_div * F.kl_div(log_pb, pa_det, reduction='batchmean')
                loss = loss_ce - div_term + (lambda_aux * aux if aux is not None else 0.0)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            # NaN/Inf detection in grads. If any param has bad grad, skip
            # the optimizer step entirely (zero grads and move on).
            bad_grad = False
            gn_sq = 0.0
            for g in optimizer.param_groups:
                for p in g['params']:
                    if p.grad is None: continue
                    g_finite = torch.isfinite(p.grad).all().item()
                    if not g_finite:
                        bad_grad = True
                        break
                    gn_sq += float(p.grad.detach().norm().item() ** 2)
                if bad_grad: break
            gn = gn_sq ** 0.5 if not bad_grad else float('nan')
            if i == 0:
                log(f'  ep{epoch+1:3d} batch 0 grad_norm={gn:.4f}')
            if bad_grad:
                optimizer.zero_grad(set_to_none=True)
                if i % 50 == 0:
                    log(f'  ep{epoch+1:3d} batch {i:3d} SKIP (NaN grad)')
                continue
            grad_clip = float(cfg.get('grad_clip', 0.0) or 0.0)
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for g in optimizer.param_groups for p in g['params']],
                    max_norm=grad_clip)
            # Linear LR warmup over first warmup_steps steps.
            warmup_steps = int(cfg.get('warmup_steps', 0))
            if warmup_steps > 0:
                global_step = epoch * len(train_loader) + i
                scale = min(1.0, (global_step + 1) / warmup_steps)
                for g in optimizer.param_groups:
                    g['lr'] = g.get('_base_lr', g['lr']) * scale
            optimizer.step()
            total_loss += float(loss_ce.item())
            total_aux += float(aux.item()) if aux is not None else 0.0
            n_batches += 1
            preds = (a + b).argmax(1)
            n_correct += (preds == target).sum().item()
            n_total += target.size(0)
            if i % 50 == 0:
                cur_lr = optimizer.param_groups[1]['lr']
                log(f'  ep{epoch+1:3d} batch {i:3d}/{len(train_loader)}  '
                    f'loss={loss_ce.item():.4f}  '
                    f'aux={aux.item() if aux is not None else 0.0:.4f}  '
                    f'lr_raw={cur_lr:.2e}')
        scheduler.step()
        tr_acc = 100.0 * n_correct / max(1, n_total)
        mean_loss = total_loss / max(1, n_batches)
        mean_aux = total_aux / max(1, n_batches)
        log(f'epoch {epoch+1} train: fused_acc={tr_acc:.2f}%  '
            f'mean_loss={mean_loss:.4f}  mean_aux={mean_aux:.4f}')
        # Parser-compatible markers for the sidepanel server.
        log(f'	Mean training acc:  {tr_acc:.4f}%.')
        log(f'	Mean training loss: {mean_loss:.10f}')
        log(f'	Mean auxiliary loss: {mean_aux:.10f}')

        metrics, (A, B, S, Y) = evaluate()
        row = {'epoch': epoch + 1, 'train_fused': round(tr_acc, 2),
               'mean_loss': round(mean_loss, 4),
               'mean_aux': round(mean_aux, 4),
               **{k: round(v, 2) for k, v in metrics.items()}}
        history.append(row)
        log(f'epoch {epoch+1} eval: cnxxl={metrics["cnxxl_solo"]:.2f}  '
            f'raw_c1={metrics["raw_solo"]:.2f}  2way={metrics["two_way"]:.2f}  '
            f'3way={metrics["three_way"]:.2f}  4way={metrics["four_way"]:.2f}')
        # Parser-compatible Test Evaluation line: prec1 = 2-way fusion (target),
        # prec5 = 3-way fusion (so the sidepanel headline shows our actual target).
        log(f'Epoch {epoch + 1}, Test, Evaluation: prec1 {metrics["two_way"]:.4f}, prec5 {metrics["three_way"]:.4f}')

        if metrics['cnxxl_solo'] < cnxxl_min_solo:
            log(f'ABORT: cnxxl solo {metrics["cnxxl_solo"]:.2f} < floor {cnxxl_min_solo}')
            aborted = True
            break

        if metrics['two_way'] > best['two_way']:
            best['two_way'] = metrics['two_way']
            best['epoch'] = epoch + 1
            best['metrics'] = metrics
            torch.save({'epoch': epoch + 1, 'cnxxl': cnxxl.state_dict(),
                        'raw_c1': raw.state_dict(), 'metrics': metrics,
                        'optimizer': optimizer.state_dict()},
                       os.path.join(work_dir, 'best_joint.pt'))
            try:
                pa = softmax_np(A); pb = softmax_np(B)
                fused_logits = np.log(0.5 * pa + 0.5 * pb + 1e-12)
                np.savez(os.path.join(work_dir, 'test_logits'),
                         logits=fused_logits, labels=Y, sigs=S,
                         epoch=np.array([epoch + 1], dtype=np.int64))
            except Exception as e:
                log(f'test_logits dump skipped: {e}')

        save_int = int(cfg.get('save_interval', 5))
        if (epoch + 1) % save_int == 0:
            torch.save({'epoch': epoch + 1, 'cnxxl': cnxxl.state_dict(),
                        'raw_c1': raw.state_dict()},
                       os.path.join(work_dir, f'epoch{epoch+1}_joint.pt'))

        with open(os.path.join(work_dir, 'history.json'), 'w') as f:
            json.dump({'history': history, 'best': best, 'aborted': aborted}, f, indent=2)

    log(f'done. best 2-way={best["two_way"]:.2f} at epoch {best["epoch"]}'
        f'{" (ABORTED)" if aborted else ""}')
    log_fp.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    args = p.parse_args()
    main(args.config)

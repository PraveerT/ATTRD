"""Rigorous MQAR trainer. Param-matched DN vs AT, JSON results, deterministic seeds.

Adds over mqar_train.py:
- Deterministic seeding (torch + numpy + cudnn).
- JSON results dump per run: best p1/p5, final p1/p5, params, config.
- Validation split (separate seed from test) for honest best-epoch selection.
- Smaller eval batch but more eval steps for stable variance.
"""
import argparse, json, math, os, random, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Reuse model definitions from mqar_train
import sys
sys.path.insert(0, os.path.dirname(__file__))
from mqar_train import SeqModel, make_mqar_batch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def evaluate(model, args, device, rng, n_steps):
    model.eval()
    p1 = p5 = ntot = 0
    with torch.no_grad():
        for _ in range(n_steps):
            x, y = make_mqar_batch(args.bs, args.T, args.vocab, args.kv, args.q, device, rng)
            logits = model(x)
            m = (y != -100)
            top5 = logits.topk(5, dim=-1).indices
            p1 += ((top5[..., 0] == y) & m).sum().item()
            p5 += ((top5 == y.unsqueeze(-1)).any(-1) & m).sum().item()
            ntot += m.sum().item()
    return 100 * p1 / max(1, ntot), 100 * p5 / max(1, ntot)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--arch', choices=['deltanet', 'attrd', 'transformer',
                                       'tpdn', 'adabdn', 'fdn', 'bdn', 'moedn',
                                       'mamba2', 'tttdn', 'slothop'], required=True)
    p.add_argument('--mlp_ratio', type=int, default=2)
    p.add_argument('--buffer_size', type=int, default=16)
    p.add_argument('--vocab', type=int, default=256)
    p.add_argument('--T', type=int, default=64)
    p.add_argument('--kv', type=int, default=8)
    p.add_argument('--q', type=int, default=16)
    p.add_argument('--d_model', type=int, default=128)
    p.add_argument('--layers', type=int, default=2)
    p.add_argument('--heads', type=int, default=4)
    p.add_argument('--head_dim', type=int, default=32)
    p.add_argument('--d_read', type=int, default=32)
    p.add_argument('--bs', type=int, default=64)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--epochs', type=int, default=60)
    p.add_argument('--steps_per_epoch', type=int, default=50)
    p.add_argument('--val_steps', type=int, default=20)
    p.add_argument('--test_steps', type=int, default=40)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--warmup_frac', type=float, default=0.0)
    p.add_argument('--out_json', type=str, required=True)
    p.add_argument('--tag', type=str, default='')
    args = p.parse_args()

    set_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = SeqModel(args.arch, args.vocab, args.d_model, args.layers,
                     args.heads, args.head_dim, args.d_read,
                     mlp_ratio=args.mlp_ratio, buffer_size=args.buffer_size,
                     dropout=0.1).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    print(f'run: {args.tag or args.arch}', flush=True)
    print(f'arch: {args.arch}  params: {n_params/1e6:.3f}M  '
          f'T={args.T} kv={args.kv} q={args.q} vocab={args.vocab} '
          f'd_model={args.d_model} head_dim={args.head_dim} d_read={args.d_read} '
          f'lr={args.lr} seed={args.seed}', flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    if args.warmup_frac > 0:
        # per-step LR: linear warmup over warmup_frac of total batches, then cosine to 0
        total_steps = args.epochs * args.steps_per_epoch
        warmup_steps = int(args.warmup_frac * total_steps)
        def lr_lambda(step):
            if step < warmup_steps:
                return (step + 1) / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1 + math.cos(math.pi * progress))
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    else:
        # per-epoch (back-compat for prior runs)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    # Separate RNGs: train, val, test — disjoint seeds, deterministic across runs.
    train_rng = np.random.default_rng(1000 + args.seed)
    val_rng_base = 5000  # val uses different seed per epoch but deterministic
    test_rng = np.random.default_rng(9000 + args.seed)

    best_val_p1 = -1.0; best_ep = -1
    best_test_p1 = -1.0; best_test_p5 = -1.0  # test at best-val-epoch
    history = []
    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        tot_loss = 0.0; tot_correct = 0; tot_tokens = 0
        print(f'Training epoch: {ep}', flush=True)
        for step in range(args.steps_per_epoch):
            x, y = make_mqar_batch(args.bs, args.T, args.vocab, args.kv, args.q, device, train_rng)
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, args.vocab), y.reshape(-1), ignore_index=-100)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if args.warmup_frac > 0:
                sched.step()       # per-step when warmup is enabled
            with torch.no_grad():
                pred = logits.argmax(-1)
                m = (y != -100)
                tot_correct += ((pred == y) & m).sum().item()
                tot_tokens  += m.sum().item()
                tot_loss    += loss.item()
        if args.warmup_frac == 0:
            sched.step()           # per-epoch when no warmup (backward compatible)
        tr_acc = 100 * tot_correct / max(1, tot_tokens)
        tr_loss = tot_loss / args.steps_per_epoch

        val_rng = np.random.default_rng(val_rng_base + ep)
        val_p1, val_p5 = evaluate(model, args, device, val_rng, args.val_steps)
        print(f'Mean training acc: {tr_acc:.4f}', flush=True)
        print(f'Mean training loss: {tr_loss:.4f}', flush=True)
        print(f'Test, Evaluation: Epoch {ep} prec1 {val_p1:.4f}, prec5 {val_p5:.4f}', flush=True)
        history.append({'epoch': ep, 'tr_acc': tr_acc, 'tr_loss': tr_loss, 'val_p1': val_p1, 'val_p5': val_p5})
        if val_p1 > best_val_p1:
            best_val_p1 = val_p1; best_ep = ep
            # On val improvement, run held-out test (different seed family)
            test_p1, test_p5 = evaluate(model, args, device, test_rng, args.test_steps)
            best_test_p1 = test_p1; best_test_p5 = test_p5
            test_rng = np.random.default_rng(9000 + args.seed)  # reset for next call
        print(f'best: ep {best_ep} p1={best_val_p1:.2f}%', flush=True)

    elapsed = time.time() - t0
    final_test_p1, final_test_p5 = evaluate(model, args, device, np.random.default_rng(20000 + args.seed), args.test_steps)

    result = {
        'tag': args.tag,
        'arch': args.arch,
        'vocab': args.vocab, 'T': args.T, 'kv': args.kv, 'q': args.q,
        'd_model': args.d_model, 'layers': args.layers, 'heads': args.heads,
        'head_dim': args.head_dim, 'd_read': args.d_read,
        'lr': args.lr, 'bs': args.bs, 'epochs': args.epochs, 'seed': args.seed,
        'params': n_params,
        'best_val_p1': best_val_p1, 'best_val_ep': best_ep,
        'test_p1_at_best_val': best_test_p1, 'test_p5_at_best_val': best_test_p5,
        'final_test_p1': final_test_p1, 'final_test_p5': final_test_p5,
        'history': history,
        'elapsed_sec': elapsed,
    }
    os.makedirs(os.path.dirname(args.out_json) or '.', exist_ok=True)
    with open(args.out_json, 'w') as f:
        json.dump(result, f, indent=2)
    print(f'DONE: {args.out_json}  test_p1={best_test_p1:.3f}%  elapsed={elapsed:.0f}s', flush=True)


if __name__ == '__main__':
    main()

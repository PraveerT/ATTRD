"""Run the rigorous MQAR grid. Sequential calls to mqar_rigor.py; aggregates JSON.

Stage 1: param-match retest at original config (T=64, vocab=256).
  2 sizes × 2 arches × 2 LRs × 3 seeds = 24 runs
Stage 2: Zoology-config grid at best Stage-1 size + LR.
  3 T values × 2 arches × 3 seeds = 18 runs

Each run writes results/<tag>.json. Final aggregate goes to results/summary.json.
"""
import argparse, itertools, json, os, subprocess, sys, time

ROOT = '/notebooks/PMamba/experiments'
RESULTS = os.path.join(ROOT, 'mqar_results')
LOGS = os.path.join(ROOT, 'work_dir')

# Param-matched triples (vocab independent — embedding doesn't differ between arches)
# Each cell: head_dim, d_read (unused for DN), mlp_ratio (only for TX)
PAIRS = {
    'S': {'deltanet':    {'head_dim': 32, 'd_read': 32, 'mlp_ratio': 1},
          'attrd':       {'head_dim': 24, 'd_read': 16, 'mlp_ratio': 1},
          'transformer': {'head_dim': 16, 'd_read': 32, 'mlp_ratio': 1}},
    'L': {'deltanet':    {'head_dim': 48, 'd_read': 32, 'mlp_ratio': 1},
          'attrd':       {'head_dim': 32, 'd_read': 32, 'mlp_ratio': 1},
          'transformer': {'head_dim': 32, 'd_read': 32, 'mlp_ratio': 1}},
}


def call_one(tag, arch, vocab, T, kv, q, head_dim, d_read, lr, seed, epochs=60, bs=64, mlp_ratio=1):
    out = os.path.join(RESULTS, f'{tag}.json')
    if os.path.exists(out):
        print(f'[skip] {tag} already exists')
        return out
    log = os.path.join(LOGS, f'{tag}.log')
    cmd = [
        sys.executable, os.path.join(ROOT, 'mqar_rigor.py'),
        '--arch', arch, '--vocab', str(vocab), '--T', str(T), '--kv', str(kv), '--q', str(q),
        '--head_dim', str(head_dim), '--d_read', str(d_read), '--lr', str(lr), '--seed', str(seed),
        '--epochs', str(epochs), '--bs', str(bs),
        '--mlp_ratio', str(mlp_ratio),
        '--out_json', out, '--tag', tag,
    ]
    print(f'[run] {tag}', flush=True)
    t0 = time.time()
    with open(log, 'w') as f:
        rc = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT)
    print(f'[done] {tag} rc={rc} elapsed={time.time()-t0:.0f}s', flush=True)
    return out


def stage1():
    """Param-match retest at original config. T=64, vocab=256."""
    runs = []
    for size in ['S', 'L']:
        for arch in ['deltanet', 'attrd']:
            cfg = PAIRS[size][arch]
            for lr in [3e-4, 1e-3]:
                for seed in [0, 1, 2]:
                    tag = f'stage1_T64_v256_{size}_{arch}_lr{lr:.0e}_s{seed}'
                    runs.append(call_one(tag, arch, vocab=256, T=64, kv=8, q=16,
                                          head_dim=cfg['head_dim'], d_read=cfg['d_read'],
                                          lr=lr, seed=seed, epochs=60))
    return runs


def stage2(size, lr_per_arch, T_values=None, arches=None):
    """Zoology-config grid at chosen size + best LR per arch."""
    runs = []
    # Match Zoology: vocab=8192. T sweep. kv = T/8, q = T/8.
    if T_values is None: T_values = [64, 128, 256]
    if arches is None:   arches = ['deltanet', 'attrd', 'transformer']
    for T in T_values:
        kv = T // 8
        q  = T // 8
        for arch in arches:
            cfg = PAIRS[size][arch]
            lr = lr_per_arch[arch]
            for seed in [0, 1, 2]:
                tag = f'stage2_T{T}_v8192_{size}_{arch}_lr{lr:.0e}_s{seed}'
                runs.append(call_one(tag, arch, vocab=8192, T=T, kv=kv, q=q,
                                      head_dim=cfg['head_dim'], d_read=cfg['d_read'],
                                      mlp_ratio=cfg.get('mlp_ratio', 1),
                                      lr=lr, seed=seed, epochs=40))
    return runs


def aggregate():
    rows = []
    for fp in sorted(os.listdir(RESULTS)):
        if not fp.endswith('.json') or fp == 'summary.json':
            continue
        with open(os.path.join(RESULTS, fp)) as f:
            rows.append(json.load(f))
    summary = {
        'n_runs': len(rows),
        'rows': [{k: r.get(k) for k in ['tag','arch','vocab','T','head_dim','d_read','lr','seed',
                                        'params','test_p1_at_best_val','test_p5_at_best_val',
                                        'best_val_p1','elapsed_sec']} for r in rows],
    }
    with open(os.path.join(RESULTS, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    return summary


if __name__ == '__main__':
    os.makedirs(RESULTS, exist_ok=True)
    p = argparse.ArgumentParser()
    p.add_argument('--stage', choices=['1', '2', 'aggregate'], required=True)
    p.add_argument('--size', default='L')           # for stage 2
    p.add_argument('--lr_dn', type=float, default=3e-4)
    p.add_argument('--lr_at', type=float, default=3e-4)
    p.add_argument('--lr_tx', type=float, default=3e-4)
    p.add_argument('--T_values', type=str, default='64')       # comma-sep for stage 2
    p.add_argument('--arches', type=str, default='deltanet,attrd,transformer')
    args = p.parse_args()
    if args.stage == '1':
        stage1()
        aggregate()
    elif args.stage == '2':
        T_values = [int(x) for x in args.T_values.split(',')]
        arches = [x.strip() for x in args.arches.split(',')]
        stage2(args.size,
               {'deltanet': args.lr_dn, 'attrd': args.lr_at, 'transformer': args.lr_tx},
               T_values=T_values, arches=arches)
        aggregate()
    elif args.stage == 'aggregate':
        s = aggregate()
        print(json.dumps(s, indent=2))

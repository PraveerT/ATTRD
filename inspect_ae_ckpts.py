import torch, os
base = '/notebooks/Anemon/experiments/work_dir/ae_pretrain'
for fname in os.listdir(base):
    if not fname.endswith('.pt'):
        continue
    p = os.path.join(base, fname)
    c = torch.load(p, map_location='cpu')
    ep = c.get('best_epoch', '?')
    sc = c.get('best_score', '?')
    cfg = c.get('config', {})
    K = cfg.get('K', '?')
    fd = cfg.get('feature_dim', '?')
    sz = os.path.getsize(p)
    print(f'{fname}: ep={ep} score={sc} K={K} feature_dim={fd} size={sz}')

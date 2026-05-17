import json, glob, numpy as np
from collections import defaultdict

agg = defaultdict(list)
for f in sorted(glob.glob('mqar_results/*.json')):
    try:
        d = json.load(open(f))
    except Exception:
        continue
    name = f.split('/')[-1].replace('.json','')
    parts = name.split('_')
    try:
        T = [p for p in parts if p.startswith('T') and p[1:].isdigit()][0][1:]
        v = [p for p in parts if p.startswith('v') and p[1:].isdigit()][0][1:]
        sz = [p for p in parts if p in ('S','L','M')][0]
        lr_idx = [i for i,p in enumerate(parts) if p.startswith('lr')][0]
        lr = parts[lr_idx][2:]
        sz_idx = parts.index(sz)
        arch = '_'.join(parts[sz_idx+1:lr_idx])
    except Exception:
        continue
    key = (T, v, sz, arch, lr)
    agg[key].append(d.get('test_p1_at_best_val', d.get('best_test_p1', d.get('test_p1', None))))

print('T   v    sz arch                 lr        mean    std    n')
rows = []
for k, vs in agg.items():
    vs = [x for x in vs if x is not None]
    if len(vs) >= 2:
        rows.append((k, float(np.mean(vs)), float(np.std(vs)), len(vs)))
rows.sort()
for k, mu, sd, n in rows:
    T,v,sz,arch,lr = k
    print(f'{T:>3} {v:>4} {sz}  {arch:18s} {lr:8s} {mu:7.2f} {sd:6.2f} {n:3d}')

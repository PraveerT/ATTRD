"""CNXXL point-cloud mirror-confusion diagnostic (the 91.29 anchor).

Adapts dump_cnxxlquat_test.py. Negates each spatial axis (x/y/z) of the input
point cloud; the axis with the biggest accuracy collapse IS the horizontal mirror
(self-validates vs the brief's 440->121). Then per-class mirror-confusion on that
axis: dominant mirror prediction, concentration, involutive chiral pairs. Decides
whether CNXXL's collapse is spurious-scatter (label-aware aug is a real lever to
push 91.29 up) or structured chirality across many pairs (needs the full pair map).
"""
import sys, os, re, numpy as np, torch, yaml, importlib
sys.path.insert(0, '/notebooks/Anemon/experiments')
os.chdir('/notebooks/Anemon/experiments')

CKPT = './work_dir/cn_xxl_quat_head/best_model.pt'
CFG = './work_dir/cn_xxl_quat_head/config.yaml'
cfg = yaml.safe_load(open(CFG))

mod_name, cls_name = cfg['model'].rsplit('.', 1)
Cls = getattr(importlib.import_module(mod_name), cls_name)
model = Cls(**cfg['model_args']).cuda().eval()
sd = torch.load(CKPT, map_location='cpu'); sd = sd.get('model_state_dict', sd)
res = model.load_state_dict(sd, strict=False)
print('load: missing=%d unexpected=%d' % (len(res.missing_keys), len(res.unexpected_keys)))
model.pts_size = cfg['model_args']['pts_size']

mn, cn = cfg['dataloader'].rsplit('.', 1)
DL = getattr(importlib.import_module(mn), cn)
ds = DL(**cfg['test_loader_args'])
loader = torch.utils.data.DataLoader(ds, batch_size=4, num_workers=4, shuffle=False, pin_memory=True)
print('test samples:', len(ds))


def sig_cls(line):
    m = re.search(r'class_(\d+)/subject(\d+)_r(\d+)', line)
    return line if not m else m.group(0)


def mir(pts, ax):
    p = pts.clone(); p[..., ax] = -p[..., ax]; return p


def predict(x):
    o = model(x); o = o[0] if isinstance(o, (list, tuple)) else o
    return o.argmax(1).cpu().numpy()


NORM, MX, MY, MZ, Y = [], [], [], [], []
with torch.no_grad():
    for i, (pts, lab, line) in enumerate(loader):
        pts = pts.cuda().float()
        NORM.append(predict(pts)); MX.append(predict(mir(pts, 0)))
        MY.append(predict(mir(pts, 1))); MZ.append(predict(mir(pts, 2)))
        Y.append(lab.numpy())
NORM = np.concatenate(NORM); MX = np.concatenate(MX); MY = np.concatenate(MY); MZ = np.concatenate(MZ); Y = np.concatenate(Y)
acc = lambda p: (p == Y).mean() * 100
print('\nacc normal=%.2f  mir_x=%.2f  mir_y=%.2f  mir_z=%.2f  (n=%d)' % (acc(NORM), acc(MX), acc(MY), acc(MZ), len(Y)))

cand = {'x': MX, 'y': MY, 'z': MZ}
axn = min(cand, key=lambda k: acc(cand[k])); MIR = cand[axn]
print('mirror axis = %s  (collapse %.2f -> %.2f)\n' % (axn, acc(NORM), acc(MIR)))

C = int(Y.max()) + 1
dom = {}
for c in range(C):
    idx = Y == c
    if idx.sum() == 0:
        continue
    vals, cnts = np.unique(MIR[idx], return_counts=True)
    dom[c] = (int(vals[cnts.argmax()]), cnts.max() / idx.sum())

print('cls  n  mir->  conc  pair?')
pairs, selfs, scat = set(), [], []
for c, (j, frac) in sorted(dom.items()):
    n = int((Y == c).sum())
    if j == c:
        tag = 'SELF'; selfs.append(c)
    elif j in dom and dom[j][0] == c:
        tag = '<->%d pair' % j; pairs.add(frozenset((c, j)))
    else:
        tag = '->%d one-way' % j
    if frac < 0.4:
        tag += ' [SCAT]'; scat.append(c)
    print('%3d %3d ->%3d  %3.0f%%  %s' % (c, n, j, frac * 100, tag))

mc = float(np.mean([f for _, f in dom.values()]))
print('\nmean concentration=%.0f%%  involutive_pairs=%d  self=%d  scattered=%d' % (mc * 100, len(pairs), len(selfs), len(scat)))
fg_pairs = {frozenset((4, 5)), frozenset((19, 20))}
print('FG83 pairs {4<->5, 19<->20} reproduced on CNXXL:', fg_pairs & pairs if (fg_pairs & pairs) else 'NO')
print('VERDICT:', 'STRUCTURED chirality (need full pair map)' if len(pairs) >= 3
      else 'SPURIOUS/scatter -> label-aware aug is a real lever')

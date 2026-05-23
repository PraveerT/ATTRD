"""Dump cn_xxl_quat_head_perpoint_frozen test logits (peak 91.08)."""
import sys, os, re, numpy as np, torch, yaml, importlib
sys.path.insert(0, '/notebooks/Anemon/experiments')
os.chdir('/notebooks/Anemon/experiments')

WORK = './work_dir/cn_xxl_quat_head_perpoint_frozen'
CKPT = f'{WORK}/best_model.pt'
CFG  = f'{WORK}/config.yaml'
OUT  = f'{WORK}/test_logits.npz'

with open(CFG) as f:
    cfg = yaml.safe_load(f)

mod_name, cls_name = cfg['model'].rsplit('.', 1)
mod = importlib.import_module(mod_name)
Cls = getattr(mod, cls_name)
model = Cls(**cfg['model_args']).cuda().eval()
sd = torch.load(CKPT, map_location='cpu')
sd = sd.get('model_state_dict', sd)
res = model.load_state_dict(sd, strict=False)
print(f'load: missing={len(res.missing_keys)} unexpected={len(res.unexpected_keys)}')
if res.missing_keys[:5]: print('  miss5:', res.missing_keys[:5])

mod_n, cls_n = cfg['dataloader'].rsplit('.', 1)
DL = getattr(importlib.import_module(mod_n), cls_n)
ds = DL(framerate=32, phase='test', datatype='depth')
loader = torch.utils.data.DataLoader(ds, batch_size=8, num_workers=4,
                                     shuffle=False, pin_memory=True)
print('test samples:', len(ds))

def sig(line):
    m = re.search(r'class_(\d+)/subject(\d+)_r(\d+)', line)
    return f'class_{m.group(1)}/subject{m.group(2)}_r{m.group(3)}'

model.pts_size = cfg['model_args']['pts_size']
all_log, all_lab, all_sigs = [], [], []
with torch.no_grad():
    for i, (pts, lab, line) in enumerate(loader):
        pts = pts.cuda().float()
        out = model(pts)
        if isinstance(out, (list, tuple)): out = out[0]
        all_log.append(out.cpu().numpy())
        all_lab.append(lab.numpy())
        all_sigs.extend([sig(s) for s in line])
        if i % 20 == 0: print(f'  {i+1}/{len(loader)}', flush=True)

L = np.concatenate(all_log); Y = np.concatenate(all_lab); S = np.array(all_sigs)
acc = (L.argmax(1) == Y).mean() * 100
np.savez(OUT, logits=L, labels=Y, sigs=S)
print(f'shape={L.shape} acc={acc:.2f}% -> {OUT}')

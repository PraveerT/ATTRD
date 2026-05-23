"""Dump cnxxlquat 91.08 logits + softmax on the 482-sample NV test set."""
import sys, os, re, numpy as np, torch, yaml, importlib
sys.path.insert(0, '/notebooks/Anemon/experiments')
os.chdir('/notebooks/Anemon/experiments')

CKPT = './work_dir/cn_xxl_quat_head/epoch100_model.pt'
CFG  = './cn_xxl_quat_head.yaml'
OUT  = './work_dir/cn_xxl_quat_head/test_logits.npz'

with open(CFG) as f:
    cfg = yaml.safe_load(f)

# Build model.
mod_name, cls_name = cfg['model'].rsplit('.', 1)
mod = importlib.import_module(mod_name)
Cls = getattr(mod, cls_name)
model = Cls(**cfg['model_args']).cuda().eval()

sd = torch.load(CKPT, map_location='cpu')
sd = sd.get('model_state_dict', sd)
res = model.load_state_dict(sd, strict=False)
print(f'load: missing={len(res.missing_keys)} unexpected={len(res.unexpected_keys)}')

# Build loader matching test config.
mod_n, cls_n = cfg['dataloader'].rsplit('.', 1)
DL = getattr(importlib.import_module(mod_n), cls_n)
ds = DL(**cfg['test_loader_args'])
loader = torch.utils.data.DataLoader(ds, batch_size=8, num_workers=4,
                                     shuffle=False, pin_memory=True)
print('test samples:', len(ds))

# Sample identifier: class_XX/subjectN_rR from inputs_list.
def sig(line):
    m = re.search(r'class_(\d+)/subject(\d+)_r(\d+)', line)
    return f'class_{m.group(1)}/subject{m.group(2)}_r{m.group(3)}'

all_logits, all_labels, all_sigs = [], [], []
model.pts_size = cfg['model_args']['pts_size']  # 172 from yaml
with torch.no_grad():
    for i, (pts, lab, line) in enumerate(loader):
        pts = pts.cuda().float()  # (B, T, P, C) -- model._sample_points permutes internally
        if i == 0:
            print('input shape:', pts.shape, 'pts_size:', model.pts_size)
        out = model(pts)
        if isinstance(out, (list, tuple)):
            out = out[0]
        all_logits.append(out.cpu().numpy())
        all_labels.append(lab.numpy())
        all_sigs.extend([sig(s) for s in line])
        if i % 20 == 0:
            print(f'  {i+1}/{len(loader)}', flush=True)

L = np.concatenate(all_logits)
Y = np.concatenate(all_labels)
sigs = np.array(all_sigs)
acc = (L.argmax(1) == Y).mean() * 100
np.savez(OUT, logits=L, labels=Y, sigs=sigs)
print(f'shape={L.shape} acc={acc:.2f}% -> {OUT}')

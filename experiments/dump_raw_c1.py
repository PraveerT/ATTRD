"""Dump cn_xxl_quat_head_stqnet_c1 best_model test logits.

Designed to co-run with training: small batch, allocator-friendly env, and
falls back from CUDA -> CPU if the GPU is saturated by training."""
import sys, os, re, numpy as np, torch, yaml, importlib
sys.path.insert(0, '/notebooks/Anemon/experiments')
os.chdir('/notebooks/Anemon/experiments')

os.environ.setdefault(
    'PYTORCH_CUDA_ALLOC_CONF',
    'max_split_size_mb:64,expandable_segments:True',
)

WORK = './work_dir/cn_xxl_quat_head_stqnet_c1'
CKPT = f'{WORK}/best_model.pt'
CFG  = f'{WORK}/config.yaml'
OUT  = f'{WORK}/test_logits.npz'

with open(CFG) as f:
    cfg = yaml.safe_load(f)

mod_name, cls_name = cfg['model'].rsplit('.', 1)
mod = importlib.import_module(mod_name)
Cls = getattr(mod, cls_name)
model = Cls(**cfg['model_args']).eval()
sd = torch.load(CKPT, map_location='cpu')
sd = sd.get('model_state_dict', sd)
res = model.load_state_dict(sd, strict=False)
print(f'load: missing={len(res.missing_keys)} unexpected={len(res.unexpected_keys)}')
ep = torch.load(CKPT, map_location='cpu').get('epoch')
print(f'best_model from log epoch {ep}')


def _dynamic_pts_size(epoch):
    if epoch < 50:
        return int(48 + (172 - 48) * (epoch / 50))
    return 172


use_dynamic = cfg.get('dynamic_pts_size', True)
epoch_idx = int(ep) - 1 if ep is not None else None
pts_size_eval = _dynamic_pts_size(epoch_idx) if (use_dynamic and epoch_idx is not None) else cfg['model_args']['pts_size']
print(f'eval pts_size={pts_size_eval} (log epoch {ep}, dynamic={use_dynamic})')
model.pts_size = pts_size_eval

mod_n, cls_n = cfg['dataloader'].rsplit('.', 1)
DL = getattr(importlib.import_module(mod_n), cls_n)
ds = DL(framerate=32, phase='test', datatype='depth')


def sig(line):
    m = re.search(r'class_(\d+)/subject(\d+)_r(\d+)', line)
    return f'class_{m.group(1)}/subject{m.group(2)}_r{m.group(3)}'


def run(device, batch_size):
    m = model.to(device)
    if device == 'cuda':
        torch.cuda.empty_cache()
    loader = torch.utils.data.DataLoader(
        ds, batch_size=batch_size, num_workers=0,
        shuffle=False, pin_memory=False,
    )
    all_log, all_lab, all_sigs = [], [], []
    with torch.no_grad():
        for i, (pts, lab, line) in enumerate(loader):
            pts = pts.to(device).float()
            out = m(pts)
            if isinstance(out, (list, tuple)): out = out[0]
            all_log.append(out.detach().cpu().numpy())
            all_lab.append(lab.numpy())
            all_sigs.extend([sig(s) for s in line])
            del pts, out
            if device == 'cuda' and (i % 20 == 0):
                torch.cuda.empty_cache()
            if i % max(20, len(loader) // 6) == 0:
                print(f'  [{device}] {i+1}/{len(loader)}', flush=True)
    return (
        np.concatenate(all_log),
        np.concatenate(all_lab),
        np.array(all_sigs),
    )


try:
    L, Y, S = run('cuda', batch_size=2)
    print('eval: cuda OK')
except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
    msg = str(e)
    if 'out of memory' not in msg.lower() and 'CUDA' not in msg:
        raise
    print(f'eval: cuda OOM -> CPU fallback ({msg[:80]}...)', flush=True)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    L, Y, S = run('cpu', batch_size=4)
    print('eval: cpu OK')

acc = (L.argmax(1) == Y).mean() * 100
np.savez(OUT, logits=L, labels=Y, sigs=S)
print(f'shape={L.shape} acc={acc:.2f}% -> {OUT}')

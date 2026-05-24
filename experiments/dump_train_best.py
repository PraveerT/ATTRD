"""Dump test logits using TRAIN-BEST ckpts (honest selection).

cnxxlquat train-best is ep96 (96.67% train) but ep96 not saved (save_interval=5).
Use ep95 (95.05% train, closest saved). canonical train-best ep92 -> use ep90.
canonical_c1 train-best is ep100 (saved). raw_c1 train-best ep149 (saved in
lock-resume).
"""
import sys, os, re, numpy as np, torch, yaml, importlib
sys.path.insert(0, '/notebooks/Anemon/experiments')
os.chdir('/notebooks/Anemon/experiments')

JOBS = [
    # (work_dir, ckpt_file, cfg_file, out_label)
    ('cn_xxl_quat_head',                 'epoch95_model.pt',  'cn_xxl_quat_head.yaml',          'cnxxl_train_best'),
    ('cn_xxl_canonical',                 'epoch90_model.pt',  'cn_xxl_canonical.yaml',          'canonical_train_best'),
    ('cn_xxl_canonical_stqnet_c1_v2',    'epoch100_model.pt', 'cn_xxl_canonical_stqnet_c1.yaml', 'canon_c1_train_best'),
    ('cn_xxl_quat_head_stqnet_c1',       'epoch149_model.pt', 'cn_xxl_quat_head_stqnet_c1.yaml', 'raw_c1_train_best'),
]

with open('../dataset/Nvidia/Processed/test_depth_list.txt') as f:
    test_list_lines = f.readlines()
def sig_from_line(s):
    m = re.search(r'class_(\d+)/subject(\d+)_r(\d+)', s)
    return f'class_{m.group(1)}/subject{m.group(2)}_r{m.group(3)}'
test_sigs_default = [sig_from_line(l) for l in test_list_lines]


def run(workdir, ckpt_file, cfg_file, label):
    cfg = yaml.safe_load(open(cfg_file))
    ckpt_path = f'./work_dir/{workdir}/{ckpt_file}'
    out_path = f'./work_dir/{workdir}/test_logits_train_best.npz'
    print(f'\n=== {label} ===\n  ckpt={ckpt_path}')

    mod_name, cls_name = cfg['model'].rsplit('.', 1)
    Cls = getattr(importlib.import_module(mod_name), cls_name)
    model = Cls(**cfg['model_args']).cuda().eval()
    sd = torch.load(ckpt_path, map_location='cpu')
    sd = sd.get('model_state_dict', sd)
    model.load_state_dict(sd, strict=False)
    model.pts_size = cfg['model_args']['pts_size']

    mod_n, cls_n = cfg['dataloader'].rsplit('.', 1)
    DL = getattr(importlib.import_module(mod_n), cls_n)
    is_canonical = 'canonical_nvidia_loader' in cfg['dataloader']
    ds = DL(framerate=32, phase='test', datatype='depth')
    loader = torch.utils.data.DataLoader(ds, batch_size=8, num_workers=4, shuffle=False, pin_memory=True)

    all_log, all_lab, all_sigs = [], [], []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            pts, lab = batch[0], batch[1]
            line = batch[2] if len(batch) > 2 else None
            pts = pts.cuda().float()
            out = model(pts)
            if isinstance(out, (list, tuple)): out = out[0]
            all_log.append(out.cpu().numpy())
            all_lab.append(lab.numpy())
            if line is not None and not is_canonical:
                all_sigs.extend([sig_from_line(s) for s in line])
            if i % 30 == 0: print(f'    {i+1}/{len(loader)}', flush=True)
    L = np.concatenate(all_log); Y = np.concatenate(all_lab)
    S = np.array(all_sigs) if all_sigs and not is_canonical else np.array(test_sigs_default)
    acc = (L.argmax(1) == Y).mean() * 100
    np.savez(out_path, logits=L, labels=Y, sigs=S)
    print(f'  shape={L.shape} acc={acc:.2f}% -> {out_path}')
    return acc


if __name__ == '__main__':
    results = {}
    for w, c, cf, lab in JOBS:
        results[lab] = run(w, c, cf, lab)
    print('\n=== summary ===')
    for k, v in results.items():
        print(f'  {k}: {v:.2f}%')

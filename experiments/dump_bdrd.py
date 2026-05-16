"""Dump test-set softmax probs for BD-RD (best ep checkpoints) for fusion."""
import sys, os, numpy as np, torch
sys.path.insert(0, '/notebooks/PMamba/experiments')
os.chdir('/notebooks/PMamba/experiments')
from torch.utils.data import DataLoader
import nvidia_dataloader
from models.motion_bdrd import MotionBDRD

# Dump both best_model.pt (test-best) and the epoch checkpoints corresponding
# to BD-RD's high-acc epochs so we can pick a train-best later.
ckpts = [
    ('best_model.pt',     'bdrd_best.npz'),
    ('epoch106_model.pt', 'bdrd_ep106.npz'),
    ('epoch110_model.pt', 'bdrd_ep110.npz'),
    ('epoch115_model.pt', 'bdrd_ep115.npz'),
]

ds = nvidia_dataloader.NvidiaLoader(framerate=32, phase='test')
loader = DataLoader(ds, batch_size=1, num_workers=4, shuffle=False)

for ckpt_name, out_name in ckpts:
    ckpt_path = f'work_dir/pmamba_baseline_bdrd/{ckpt_name}'
    if not os.path.exists(ckpt_path):
        print(f'[skip] {ckpt_path} not found')
        continue
    model = MotionBDRD(num_classes=25, pts_size=256, knn=[32, 24, 48, 24],
                       topk=8, multi_scale_num_scales=5,
                       rd_hidden_dim=128, rd_num_layers=2, rd_num_heads=4,
                       rd_n_q=4, rd_n_v=8, rd_buffer_size=4, rd_dropout=0.3,
                       rd_bidirectional=True).cuda()
    state = torch.load(ckpt_path, map_location='cpu')['model_state_dict']
    res = model.load_state_dict(state, strict=False)
    print(f'{ckpt_name}: missing={len(res.missing_keys)} unexpected={len(res.unexpected_keys)}')
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            x, y = batch[0].cuda().float(), batch[1]
            logits = model(x)
            all_probs.append(torch.softmax(logits, dim=-1).cpu().numpy())
            all_labels.append(y.numpy() if hasattr(y, 'numpy') else np.array(y))
    P = np.concatenate(all_probs); L = np.concatenate(all_labels)
    out = f'dump_probs_runs/{out_name}'
    np.savez(out, probs=P, labels=L)
    print(f'  shape={P.shape} test_acc={(P.argmax(1)==L).mean()*100:.2f}% -> {out}')

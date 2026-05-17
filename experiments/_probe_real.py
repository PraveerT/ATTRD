"""Probe BDN-Q with REAL NVGesture data (correct input shape)."""
import sys, os, torch
sys.path.insert(0, '/notebooks/PMamba/experiments')
os.chdir('/notebooks/PMamba/experiments')

import nvidia_dataloader
from torch.utils.data import DataLoader
from models.motion_bdn_q import MotionBDeltaQ, BDeltaQBlock

_records = []
_orig = BDeltaQBlock.forward
def _patched(self, x):
    B, T, _ = x.shape
    _records.append((B, T, self.W, max(0, T - self.W)))
    return _orig(self, x)
BDeltaQBlock.forward = _patched

ds = nvidia_dataloader.NvidiaLoader(framerate=32, phase='test')
loader = DataLoader(ds, batch_size=1, num_workers=0, shuffle=False)
batch = next(iter(loader))
x = batch[0].cuda().float()
print(f'real input shape from loader: {tuple(x.shape)}')

for axis in ('T', 'N'):
    _records.clear()
    model = MotionBDeltaQ(
        num_classes=25, pts_size=256, knn=[32, 24, 48, 24], topk=8,
        multi_scale_num_scales=5,
        bdnq_hidden_dim=128, bdnq_num_layers=2, bdnq_num_heads=4,
        bdnq_n_q=4, bdnq_n_v=8, bdnq_buffer_size=1, bdnq_dropout=0.3,
        bdnq_bidirectional=True, bdnq_scan_axis=axis,
    ).cuda().eval()
    with torch.no_grad():
        _ = model(x)
    Ts = sorted(set(r[1] for r in _records))
    ejs = [r[3] for r in _records]
    print(f'scan_axis={axis}: {len(_records)} block calls, seq_len={Ts}, ejections/call={sum(ejs)//len(ejs) if ejs else 0}')

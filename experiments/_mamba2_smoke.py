import sys, os, torch
sys.path.insert(0, '/notebooks/PMamba/experiments')
os.chdir('/notebooks/PMamba/experiments')
from torch.utils.data import DataLoader
import nvidia_dataloader
from models.motion_mamba2 import MotionMamba2

ds = nvidia_dataloader.NvidiaLoader(framerate=32, phase='test')
loader = DataLoader(ds, batch_size=1, num_workers=0, shuffle=False)
batch = next(iter(loader))
x = batch[0].cuda().float()
print(f'input: {tuple(x.shape)}')

model = MotionMamba2(
    pts_size=256, num_classes=25, knn=[32, 24, 48, 24], topk=8,
    multi_scale_num_scales=5,
    m2_hidden_dim=256, m2_num_layers=2, m2_d_state=64,
    m2_d_conv=4, m2_expand=2, m2_headdim=64, m2_dropout=0.3,
    m2_bidirectional=True,
).cuda().eval()
print(f'params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M')
with torch.no_grad():
    out = model(x)
print(f'out: {tuple((out[0] if isinstance(out, tuple) else out).shape)}')
print('OK')

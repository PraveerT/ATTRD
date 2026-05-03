import sys, re, os
sys.path.insert(0, '/notebooks/PMamba/experiments')
import numpy as np, torch
from torch.utils.data import DataLoader
from nvidia_dataloader import NvidiaLoader
from models.motion import Motion

# Load PMamba baseline + get logits on test set
m = Motion(num_classes=25, pts_size=256, knn=[32,24,48,24], topk=8, multi_scale_num_scales=5)
m.load_state_dict(torch.load('./work_dir/pmamba_branch/epoch115_model.pt', map_location='cpu')['model_state_dict'], strict=False)
m = m.cuda().eval()

test_l = DataLoader(NvidiaLoader(framerate=32, phase='test', valid_subject=None), batch_size=8, num_workers=4)
pm_logits, labels, paths = [], [], []
r = re.compile('[ \t\n\r:]+')
with torch.no_grad():
    for x, y, p in test_l:
        x = x.cuda().float()
        pm_logits.append(m(x).cpu())
        labels.append(y)
        paths.extend(p)
pm = torch.cat(pm_logits).numpy()
y_arr = torch.cat(labels).numpy()
N = len(y_arr)
print(f'pmamba test N={N}, acc={(pm.argmax(-1)==y_arr).mean():.4f}')

# Load UMDR M logits
umdr_dict = torch.load('/notebooks/PMamba/external_models/M-output.pth', map_location='cpu')
print(f'umdr entries: {len(umdr_dict)}')

# Map our PMamba paths to UMDR sample paths.
# Our path example: ./Nvidia/Processed/test/class_04/subject13_r0/sk_depth.avi/0060_depth_label_03.npy
# UMDR key example:  /notebooks/MotionRGBD-PAMI/nv_data/rgb/test/class_04/subject20_r1/
def pm_path_to_umdr_key(pm_path):
    parts = r.split(pm_path)
    rel = parts[1]  # ./Nvidia/Processed/test/class_04/subject13_r0/sk_depth.avi/...
    seg = rel.split('/')
    # seg: ['.', 'Nvidia', 'Processed', 'test', 'class_04', 'subject13_r0', 'sk_depth.avi', ...]
    cls, subj = seg[4], seg[5]
    return f'/notebooks/MotionRGBD-PAMI/nv_data/rgb/test/{cls}/{subj}/'

umdr_logits = []
missing = 0
for p in paths:
    k = pm_path_to_umdr_key(p)
    if k in umdr_dict:
        umdr_logits.append(umdr_dict[k].numpy())
    else:
        umdr_logits.append(np.zeros(25))
        missing += 1
umdr = np.stack(umdr_logits)
print(f'umdr aligned: {len(umdr)} (missing {missing})')
print(f'umdr alone acc: {(umdr.argmax(-1)==y_arr).mean():.4f}')

# Sweep fusion weights
print()
print(f'{"alpha":>6s} {"acc":>7s}')
for a in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    L = (1-a)*pm + a*umdr
    acc = (L.argmax(-1)==y_arr).mean()
    print(f'  {a:.2f}  {acc:.4f}')

# Also try softmax-space fusion
import scipy.special as sp
P_pm = sp.softmax(pm, axis=-1); P_um = sp.softmax(umdr, axis=-1)
print()
print('softmax-space:')
for a in [0.2, 0.3, 0.4, 0.5]:
    P = (1-a)*P_pm + a*P_um
    acc = (P.argmax(-1)==y_arr).mean()
    print(f'  alpha={a}: acc={acc:.4f}')

# Oracle
or_acc = ((pm.argmax(-1)==y_arr) | (umdr.argmax(-1)==y_arr)).mean()
print(f'\noracle (pm OR umdr): {or_acc:.4f}')

print('\n--- fine-grained softmax sweep ---')
for a in np.linspace(0.05, 0.40, 36):
    P = (1-a)*P_pm + a*P_um
    acc = (P.argmax(-1)==y_arr).mean()
    print(f'  alpha={a:.3f}: acc={acc:.4f}')

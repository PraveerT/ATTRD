"""Mirror-confusion diagnostic on the kept FG83 depth model.

Decides whether NVGesture's mirror collapse is (a) chiral-pair structure (mirror
CHANGES class -> label-aware aug is the move) or (b) spurious cue (scattered ->
plain invariance aug). For each true class, finds the dominant class the model
predicts when the input is horizontally mirrored, the concentration of that
prediction, and whether the map is an involution (c->j and j->c = a clean pair).
"""
import sys
sys.path.insert(0, '/notebooks/Anemon')
sys.path.insert(0, '/notebooks/Anemon/experiments')
import numpy as np, torch
from torch.utils.data import DataLoader
from train_depth_small import MC3Depth
from train_depth_skewtcc import build_cache, DS

dev = 'cuda'
CK = '/notebooks/Anemon/experiments/work_dir/depth_small_r2_fg83_restored_20260528_033028/best_model.pt'
m = MC3Depth(kind='r2plus1d_18', pretrained=False).to(dev).eval()
sd = torch.load(CK, map_location='cpu')
m.load_state_dict(sd['model_state_dict'] if 'model_state_dict' in sd else sd)

va = build_cache('/notebooks/cvpr_data/dataset_splits/valid.txt', 'valid',
                 '/notebooks/cvpr_data/depth',
                 '/notebooks/Anemon/dataset/Nvidia/Processed/depth_small_cache', 128)
dl = DataLoader(DS(*va, frames=32, crop=112, train=False), batch_size=16, num_workers=6)

NORM, MIR, Y = [], [], []
with torch.no_grad(), torch.cuda.amp.autocast():
    for clip, y, s in dl:
        clip = clip.to(dev)
        NORM.append(m(clip).argmax(1).cpu().numpy())
        MIR.append(m(torch.flip(clip, dims=[-1])).argmax(1).cpu().numpy())
        Y.append(y.numpy())
NORM = np.concatenate(NORM); MIR = np.concatenate(MIR); Y = np.concatenate(Y)
print('acc  normal=%.2f%%  mirror=%.2f%%  (n=%d)' % ((NORM == Y).mean()*100, (MIR == Y).mean()*100, len(Y)))

C = int(Y.max()) + 1
dom = {}      # class -> (mirror-target, fraction)
for c in range(C):
    idx = Y == c
    if idx.sum() == 0:
        continue
    vals, cnts = np.unique(MIR[idx], return_counts=True)
    j = int(vals[cnts.argmax()]); frac = cnts.max() / idx.sum()
    dom[c] = (j, frac)

# involution check: c -> j and j -> c (a clean chiral pair)
print('\ncls  n  mirror->  conc   pair?')
pairs, selfs, scattered = set(), [], []
for c, (j, frac) in sorted(dom.items()):
    n = int((Y == c).sum())
    if j == c:
        tag = 'SELF (mirror-invariant class)'; selfs.append(c)
    elif j in dom and dom[j][0] == c:
        tag = f'<-> {j}  (involutive pair)'; pairs.add(frozenset((c, j)))
    else:
        tag = f'-> {j}  (one-way)'
    if frac < 0.4:
        tag += '  [SCATTERED]'; scattered.append(c)
    print(f'{c:3d} {n:3d}  ->{j:3d}    {frac*100:3.0f}%   {tag}')

mean_conc = float(np.mean([f for _, f in dom.values()]))
print(f'\nsummary: mean mirror-prediction concentration = {mean_conc*100:.0f}%')
print(f'  involutive chiral pairs: {len(pairs)}  | self-mirror classes: {len(selfs)}  | scattered(<40%): {len(scattered)}')
print('VERDICT:', 'CHIRAL-PAIR structure (label-aware aug)' if mean_conc > 0.5 and len(pairs) >= 3
      else 'SPURIOUS/scattered (plain invariance aug)')

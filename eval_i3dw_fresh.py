"""Re-evaluate I3DWTrans depth ckpt on 482 valid samples. K-output.pth gives
24% — appears stale/broken. Dump fresh logits aligned to Anemon test order.
"""
import sys, os, numpy as np, torch, re, time
sys.path.insert(0, '/notebooks/MotionRGBD')
sys.path.insert(0, '/notebooks/MotionRGBD/lib')
sys.path.insert(0, '/notebooks/MotionRGBD/utils')
sys.path.insert(0, '/notebooks/MotionRGBD/tools')
os.chdir('/notebooks/MotionRGBD')

from types import SimpleNamespace
from torch.utils.data import DataLoader
from lib.datasets.NvGesture import NvData
from lib.model.DSN import DSNNet

CKPT = '/notebooks/Anemon/dsn_official.pth'

args = SimpleNamespace(
    data='/notebooks/cvpr_data', splits='/notebooks/cvpr_data/dataset_splits',
    dataset='NvGesture', type='K', Network='I3DWTrans', num_classes=25,
    sample_duration=64, sample_size=224, batch_size=4, test_batch_size=2,
    num_workers=4, nprocs=1, local_rank=0, dist=False,
    flip=0.0, rotated=0.0, angle='(0, 0)', Blur=False, resize='(256, 256)',
    crop_size=224, low_frames=16, media_frames=32, high_frames=48,
    w=4, temper=0.4, recoupling=True, knn_attention=0.7, sharpness=False,
    temp=[0.04, 0.07], frp=True, SEHeads=1, N=6, grad_clip=5.0, SYNC_BN=0,
    epoch=0, epochs=100, init_epochs=0, DEBUG=False, MultiLoss=True,
    pretrained=False, phase='valid',
)

ds = NvData(args, ground_truth=f'{args.splits}/valid.txt', modality='depth', phase='valid')
print(f'valid samples: {len(ds)}')

loader = DataLoader(ds, batch_size=args.test_batch_size, num_workers=args.num_workers, shuffle=False, pin_memory=True)

model = DSNNet(args, num_classes=args.num_classes, pretrained=False).cuda()
ckpt = torch.load(CKPT, map_location='cpu')
sd = ckpt['model']
sd = {(k[7:] if k.startswith('module.') else k): v for k, v in sd.items()}
res = model.load_state_dict(sd, strict=False)
print(f'loaded: missing={len(res.missing_keys)} unexpected={len(res.unexpected_keys)}  ep={ckpt.get("epoch")}  bestacc={ckpt.get("bestacc")}')
model.eval()

all_logits, all_labels, all_paths = [], [], []
t0 = time.time()
with torch.no_grad():
    for i, batch in enumerate(loader):
        clip, garr, label, path = batch
        clip = clip.cuda().float(); garr = garr.cuda().float()
        logits_tuple, _, _ = model(clip, garr)
        x, xs, xm, xl = logits_tuple
        all_logits.append(x.cpu().numpy())
        all_labels.append(label.numpy() if hasattr(label,'numpy') else np.array(label))
        all_paths.extend(path if isinstance(path, list) else [path])
        if i % 30 == 0:
            print(f'  {i+1}/{len(loader)} elapsed={time.time()-t0:.0f}s', flush=True)

L = np.concatenate(all_logits)
Y = np.concatenate(all_labels)
acc = (L.argmax(1) == Y).mean() * 100
print(f'\nFRESH eval acc: {acc:.2f}%')

out = '/notebooks/Anemon/dsn_official_valid_logits.npz'
np.savez(out, logits=L, labels=Y, paths=np.array(all_paths))
print(f'-> {out}')

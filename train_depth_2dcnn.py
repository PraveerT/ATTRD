"""Train a small 2D ResNet + GRU on NvGesture depth modality, from scratch.

Goal: complementary fusion partner for cnxxl. Target ~85% solo with error
Jaccard < 0.3 vs cnxxl. Maximally distinct from DSN's heavy I3D-3D pipeline.

Architecture:
  per-frame 2D conv stem -> 4 residual blocks -> AdaptiveAvgPool
  -> GRU(2 layers, bidirectional) -> Linear classifier

Trained from scratch, no Kinetics / ImageNet pretraining. ~3-5M params.
"""
import os, sys, time, math, argparse, json, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from types import SimpleNamespace

sys.path.insert(0, '/notebooks/MotionRGBD')
sys.path.insert(0, '/notebooks/MotionRGBD/lib')
os.chdir('/notebooks/MotionRGBD')

from lib.datasets.NvGesture import NvData


class BasicBlock2D(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)
        if stride != 1 or in_c != out_c:
            self.short = nn.Sequential(
                nn.Conv2d(in_c, out_c, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_c),
            )
        else:
            self.short = nn.Identity()

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.short(x), inplace=True)


class Depth2DCNNGRU(nn.Module):
    def __init__(self, num_classes=25, in_channels=3, hidden=128, gru_hidden=128, dropout=0.3):
        super().__init__()
        # Per-frame 2D CNN.
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        self.layer1 = nn.Sequential(BasicBlock2D(32, 64, 2), BasicBlock2D(64, 64))
        self.layer2 = nn.Sequential(BasicBlock2D(64, 128, 2), BasicBlock2D(128, 128))
        self.layer3 = nn.Sequential(BasicBlock2D(128, 256, 2), BasicBlock2D(256, 256))
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.feat_dim = 256

        # Temporal GRU.
        self.gru = nn.GRU(self.feat_dim, gru_hidden, num_layers=2,
                          batch_first=True, bidirectional=True, dropout=dropout)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(2 * gru_hidden, num_classes)

    def forward(self, x):
        # NvData returns clips as (B, C, T, H, W). Reorder to (B, T, C, H, W).
        if x.dim() == 5 and x.shape[1] == 3 and x.shape[2] != 3:
            x = x.permute(0, 2, 1, 3, 4).contiguous()
        B, T = x.shape[:2]
        x = x.reshape(B * T, *x.shape[2:])
        x = self.stem(x)
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x)
        x = self.gap(x).flatten(1)                  # (B*T, 256)
        x = x.reshape(B, T, self.feat_dim)
        out, _ = self.gru(x)                        # (B, T, 2*H)
        # Mean-pool over time for stability.
        out = out.mean(dim=1)
        out = self.drop(out)
        return self.fc(out)


def build_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='/notebooks/cvpr_data')
    ap.add_argument('--splits', default='/notebooks/cvpr_data/dataset_splits')
    ap.add_argument('--workdir', default='/notebooks/Anemon/experiments/work_dir/depth_2dcnn')
    ap.add_argument('--epochs', type=int, default=120)
    ap.add_argument('--batch-size', type=int, default=16)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--wd', type=float, default=1e-4)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--frames', type=int, default=32)
    ap.add_argument('--crop', type=int, default=112)
    return ap.parse_args()


def make_dataset(args, phase):
    ns = SimpleNamespace(
        data=args.data, splits=args.splits,
        dataset='NvGesture', type='K',
        Network='I3DWTrans', num_classes=25,
        sample_duration=args.frames, sample_size=args.crop,
        batch_size=args.batch_size, test_batch_size=args.batch_size,
        num_workers=args.workers, nprocs=1, local_rank=0, dist=False,
        flip=0.5 if phase == 'train' else 0.0,
        rotated=0.0, angle='(0, 0)', Blur=False, resize='(256, 256)',
        crop_size=args.crop, low_frames=args.frames // 2,
        media_frames=args.frames, high_frames=int(args.frames * 1.5),
        w=4, temper=0.4, recoupling=False, knn_attention=0.7, sharpness=False,
        temp=[0.04, 0.07], frp=False, SEHeads=1, N=6,
        grad_clip=5.0, SYNC_BN=0, epoch=0, epochs=args.epochs,
        init_epochs=0, DEBUG=False, MultiLoss=False, pretrained=False,
        phase=phase,
    )
    split_file = f'{args.splits}/{"train" if phase == "train" else "valid"}.txt'
    return NvData(ns, ground_truth=split_file, modality='depth', phase=phase)


def main():
    args = build_args()
    os.makedirs(args.workdir, exist_ok=True)
    print(f'workdir: {args.workdir}')

    random.seed(0); np.random.seed(0); torch.manual_seed(0)

    train_ds = make_dataset(args, 'train')
    test_ds = make_dataset(args, 'valid')
    print(f'train: {len(train_ds)}, test: {len(test_ds)}')

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.workers, pin_memory=True)

    model = Depth2DCNNGRU(num_classes=25, in_channels=3).cuda()
    nparams = sum(p.numel() for p in model.parameters()) / 1e6
    print(f'params: {nparams:.2f}M')

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr * 0.02)

    best_acc = 0.0; best_logits = None; best_labels = None; best_epoch = -1
    log_path = os.path.join(args.workdir, 'log.txt')
    log = open(log_path, 'a')

    for ep in range(1, args.epochs + 1):
        # Train
        model.train()
        t0 = time.time()
        tot_loss = 0.0; tot_correct = 0; tot_n = 0
        for batch in train_loader:
            clip, _skg, label, _path = batch
            clip = clip.cuda(non_blocking=True).float()
            label = label.cuda(non_blocking=True).long()
            logits = model(clip)
            loss = F.cross_entropy(logits, label)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot_loss += loss.item() * label.size(0)
            tot_correct += (logits.argmax(1) == label).sum().item()
            tot_n += label.size(0)
        sched.step()
        tr_loss = tot_loss / tot_n
        tr_acc = tot_correct / tot_n * 100

        # Eval
        model.eval()
        all_logits, all_labels = [], []
        with torch.no_grad():
            for batch in test_loader:
                clip, _skg, label, _path = batch
                clip = clip.cuda(non_blocking=True).float()
                logits = model(clip)
                all_logits.append(logits.cpu().numpy())
                all_labels.append(label.numpy() if hasattr(label, 'numpy') else np.asarray(label))
        all_logits = np.concatenate(all_logits)
        all_labels = np.concatenate(all_labels)
        te_acc = (all_logits.argmax(1) == all_labels).mean() * 100

        dt = time.time() - t0
        msg = (f'ep{ep:3d}  tr_loss={tr_loss:.4f}  tr_acc={tr_acc:.2f}%  te_acc={te_acc:.2f}%  '
               f'best={best_acc:.2f}%  dt={dt:.1f}s')
        print(msg); log.write(msg + '\n'); log.flush()

        if te_acc > best_acc:
            best_acc = te_acc; best_logits = all_logits; best_labels = all_labels; best_epoch = ep
            torch.save({
                'epoch': ep, 'model_state_dict': model.state_dict(),
                'best_acc': best_acc,
            }, os.path.join(args.workdir, 'best_model.pt'))

    np.savez(os.path.join(args.workdir, 'test_logits.npz'),
             logits=best_logits, labels=best_labels)
    print(f'\nbest: {best_acc:.2f}% @ ep{best_epoch}')
    log.write(f'\nbest: {best_acc:.2f}% @ ep{best_epoch}\n'); log.close()


if __name__ == '__main__':
    main()

"""Apply the AE to every NVGesture sample and save a canonical dataset
matching NvidiaLoader's input distribution.

Per-frame procedure:
  1. Run AE: xyz_in -> canonical (K, 3) via encoder + decoder.
  2. Take first keep_K points.
  3. Normalize canonical xyz with nvidia_dataset_stats (same global z-score
     as NvidiaLoader applies to raw pts) so the classifier sees matching
     statistics between canonical and raw inputs.
  4. Build t channel using GLOBAL t_mean/t_std from the same stats file (not
     per-sample standardization), so the t-channel scale matches raw pts.
  5. Output (keep_K, 4) = (xyz, t).

Output: dataset/Nvidia/Processed/canonical_{phase}.npy  (N, 32, keep_K, 4)
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))

from models.motion_cleanest_ae import FrameEncoder, FrameDecoder
from nvidia_dataloader import NvidiaLoader


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', type=str, required=True)
    p.add_argument('--out-dir', type=str, default='../dataset/Nvidia/Processed')
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--num-worker', type=int, default=8)
    p.add_argument('--keep-K', type=int, default=512)
    p.add_argument('--stats', type=str, default='nvidia_dataset_stats.npy')
    p.add_argument('--no-normalize', action='store_true',
                   help='Skip xyz/t normalization (legacy AE-output-as-is mode).')
    return p.parse_args()


def collect_xyz(phase, model, K, keep_K, batch_size, num_worker):
    """First pass: collect raw AE-output xyz so we can compute its own stats."""
    dataset = NvidiaLoader(framerate=32, phase=phase)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_worker, drop_last=False,
    )
    raw = np.empty((len(dataset), 32, keep_K, 3), dtype=np.float32)
    labels = np.empty(len(dataset), dtype=np.int64)
    cursor = 0
    with torch.no_grad():
        for data in loader:
            inputs = data[0].cuda(non_blocking=True)
            lbl = data[1]
            xyz = inputs[..., :3]
            B, T, N, _ = xyz.shape
            point_feats = model['encoder'](xyz)
            canonical = model['decoder'](point_feats)             # (B,T,K,3)
            top_xyz = canonical[:, :, :keep_K]                     # (B, T, keep_K, 3)
            raw[cursor:cursor + B] = top_xyz.cpu().numpy()
            labels[cursor:cursor + B] = lbl.numpy() if hasattr(lbl, 'numpy') else lbl
            cursor += B
    return raw, labels


def bake_phase(phase, model, K, keep_K, batch_size, num_worker, out_path,
               stats, ae_stats=None):
    """Keep AE output xyz AS-IS (matches original bake -- AE-encoded offsets
    are signal, not noise). Only fix the t channel to use global stats so it
    carries real time info instead of per-sample constant standardization."""
    print(f'[bake] {phase}: collecting raw AE output...')
    raw, labels = collect_xyz(phase, model, K, keep_K, batch_size, num_worker)

    N, T, K_eff, _ = raw.shape
    # Global-stats t channel: same as NvidiaLoader applies to raw t.
    t_idx = np.arange(T, dtype=np.float32)
    t_norm = (t_idx - stats['t_mean']) / stats['t_std']
    t_channel = np.broadcast_to(t_norm.reshape(1, T, 1, 1), (N, T, K_eff, 1)).copy()
    out = np.concatenate([raw, t_channel], axis=-1).astype(np.float32)

    np.save(out_path, out)
    np.save(out_path.replace('.npy', '_labels.npy'), labels)
    print(f'[bake] wrote {out_path} shape={out.shape}  '
          f'xyz mean={out[...,:3].mean():.4f} std={out[...,:3].std():.4f}  '
          f't mean={out[...,3].mean():.4f} std={out[...,3].std():.4f}')
    return None  # no shared ae_stats needed in this mode


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    stats = None
    if not args.no_normalize:
        stats = np.load(args.stats, allow_pickle=True).item()
        print(f'[bake] loaded stats: x_mean={stats["x_mean"]:.4f} x_std={stats["x_std"]:.4f}  '
              f't_mean={stats["t_mean"]:.4f} t_std={stats["t_std"]:.4f}')

    ckpt = torch.load(args.ckpt, map_location='cpu')
    cfg = ckpt['config']
    K = cfg['K']
    print(f'[bake] AE ckpt: {args.ckpt}')
    print(f'[bake] AE config: K={K}, feature_dim={cfg["feature_dim"]}')
    print(f'[bake] AE pretrain chamfer={ckpt["best_score"]:.4f} at ep{ckpt["best_epoch"]}')

    enc = FrameEncoder(feature_dim=cfg['feature_dim']).cuda()
    dec = FrameDecoder(
        feature_dim=cfg['feature_dim'], K=K,
        query_dim=cfg['query_dim'], heads=cfg['heads'],
        num_attn_blocks=cfg['num_attn_blocks'], ffn_mult=cfg['ffn_mult'],
    ).cuda()
    enc.load_state_dict(ckpt['encoder']); enc.eval()
    dec.load_state_dict(ckpt['decoder']); dec.eval()
    model = {'encoder': enc, 'decoder': dec}

    # Bake train first so we can reuse its AE-output stats for test.
    ae_stats = None
    for phase in ('train', 'test'):
        out_path = os.path.join(args.out_dir, f'canonical_{phase}.npy')
        ae_stats = bake_phase(phase, model, K, args.keep_K, args.batch_size,
                              args.num_worker, out_path, stats, ae_stats=ae_stats)
    # Persist AE stats next to the data for later inspection.
    stats_path = os.path.join(args.out_dir, 'canonical_ae_stats.npy')
    np.save(stats_path, ae_stats)
    print(f'[bake] saved AE-output stats -> {stats_path}')


if __name__ == '__main__':
    main()

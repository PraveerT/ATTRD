"""Pair NvidiaLoader point clouds with precomputed rigidity per-point stats.

Returns: ((pmamba_input, rigidity_tensor), label, line). pmamba_input has the
same shape/dtype as NvidiaLoader's output so PMamba accepts it unchanged.
"""
import numpy as np
import torch

from nvidia_dataloader import NvidiaLoader


class NvidiaRigidityLoader(NvidiaLoader):
    def __init__(self, *args, rigidity_per_point=True, rigidity_sort=True,
                 rigidity_norm_scale=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.rigidity_per_point = rigidity_per_point
        self.rigidity_sort = rigidity_sort
        self.rigidity_norm_scale = rigidity_norm_scale

    def __getitem__(self, index):
        pts, label, line = super().__getitem__(index)
        # Resolve rigidity path — pts.npy / depth.npy share the same stem.
        rel = self.r.split(line)[1][1:-4]                    # ./Nvidia/... stem
        suffix = "_rigidity_pp.npy" if self.rigidity_per_point else "_rigidity.npy"
        rig_path = f"../dataset/{rel}{suffix}"
        rig = np.load(rig_path).astype(np.float32)            # (T, P) or (T, K)
        if self.rigidity_per_point and self.rigidity_sort:
            rig = np.sort(rig, axis=-1)[:, ::-1].copy()
        rig = rig * self.rigidity_norm_scale
        return (pts, torch.from_numpy(rig).float()), label, line

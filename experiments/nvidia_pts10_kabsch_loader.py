"""
NvidiaPts10KabschLoader — variant K loader.

Like the F-variant pts10 + cycle-product loader, but the cycle quaternion is
derived from per-frame Kabsch alignment of the corresponded lattice (which
has true correspondence) rather than from PCA-orientation transitions.

Output (T, 512, 18):
  channels [0:10]  = pts10 (raw uvd-t + nearest-lattice)
  channels [10:14] = Q_kabsch_cycle broadcast (input feature)
  channels [14:18] = Q_kabsch_cycle target broadcast (aux MSE)
"""

import os
import re
import numpy as np
import torch
import torch.utils.data as data

from utils.pts_transform import Compose, PointcloudToTensor


class NvidiaPts10KabschLoader(data.Dataset):
    def __init__(self, framerate, valid_subject=None, phase="train",
                 datatype="depth", inputs_type="pts", **kwargs):
        self.phase = phase
        self.datatype = datatype
        self.framerate = framerate
        self.valid_subject = valid_subject
        self.r = re.compile(r"[ \t\n\r:]+")
        self.inputs_list = self.get_inputs_list()

        try:
            self.raw_stats = np.load("nvidia_dataset_stats.npy", allow_pickle=True).item()
        except FileNotFoundError:
            self.raw_stats = None

        try:
            stats = np.load("nvidia_octo_stats.npy", allow_pickle=True).item()
            self.octo_mean = stats["mean"].astype(np.float32)
            self.octo_std = stats["std"].astype(np.float32)
        except FileNotFoundError:
            self.octo_mean = np.zeros(6, dtype=np.float32)
            self.octo_std = np.ones(6, dtype=np.float32)

        self.transform = Compose([PointcloudToTensor()])

    def get_inputs_list(self):
        prefix = "../dataset/Nvidia/Processed"
        if self.phase == "train":
            path = f"{prefix}/train_depth_list.txt"
            lines = open(path).readlines()
            ret = [ln for ln in lines if f"subject{self.valid_subject}_" not in ln]
        elif self.phase == "valid":
            path = f"{prefix}/train_depth_list.txt"
            lines = open(path).readlines()
            ret = [ln for ln in lines if f"subject{self.valid_subject}_" in ln]
        elif self.phase == "test":
            path = f"{prefix}/test_depth_list.txt"
            ret = open(path).readlines()
        else:
            raise AssertionError("phase error")
        return ret

    def __len__(self):
        return len(self.inputs_list)

    def __getitem__(self, index):
        line = self.inputs_list[index]
        label = int(self.r.split(line)[-2])
        depth_npy = self.r.split(line)[1]
        pts10_path = "../dataset/" + depth_npy[2:-4] + "_pts10.npy"
        kab_path = "../dataset/" + depth_npy[2:-4] + "_qkabschprod.npy"

        pts = np.load(pts10_path).astype(np.float32)        # (T, 512, 10)
        qkab = np.load(kab_path).astype(np.float32)         # (4,)
        T, N, _ = pts.shape

        if self.raw_stats is not None:
            pts[..., 0] = (pts[..., 0] - self.raw_stats["x_mean"]) / self.raw_stats["x_std"]
            pts[..., 1] = (pts[..., 1] - self.raw_stats["y_mean"]) / self.raw_stats["y_std"]
            pts[..., 2] = (pts[..., 2] - self.raw_stats["z_mean"]) / self.raw_stats["z_std"]
            pts[..., 3] = (pts[..., 3] - self.raw_stats["t_mean"]) / self.raw_stats["t_std"]

        pts[..., 4:10] = (pts[..., 4:10] - self.octo_mean) / self.octo_std

        kab_per_pt = np.broadcast_to(qkab[None, None, :], (T, N, 4)).copy()
        target_per_pt = np.broadcast_to(qkab[None, None, :], (T, N, 4)).copy()

        out = np.concatenate([pts, kab_per_pt, target_per_pt], axis=-1)  # (T, 512, 18)
        out_t = self.transform(out.reshape(-1, 18))
        if isinstance(out_t, torch.Tensor):
            out_t = out_t.view(T, N, 18)
        else:
            out_t = torch.from_numpy(out_t).view(T, N, 18)
        return out_t.float(), label, line

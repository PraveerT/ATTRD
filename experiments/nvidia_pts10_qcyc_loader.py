"""
NvidiaPts10QcycLoader — pts10 (raw + nearest-lattice) plus per-frame
combined forward quaternion `q_fwd_raw · q_fwd_lat` broadcast as 4 extra
channels per point.

Input shape per sample: (T=32, 512, 14)
  channels [0:4]   = raw uvd-t (z-scored with nvidia_dataset_stats)
  channels [4:10]  = nearest-lattice (x, y, z, vx, vy, vz, z-scored with octo stats)
  channels [10:14] = q_combined per frame (broadcast to all 512 points)
                     q_combined[t] = q_fwd_raw[t] · q_fwd_lat[t]
                     last frame = zeros (no transition)
"""

import os
import re
import numpy as np
import torch
import torch.utils.data as data

from utils.pts_transform import Compose, PointcloudToTensor


class NvidiaPts10QcycLoader(data.Dataset):
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
        qcyc_path = "../dataset/" + depth_npy[2:-4] + "_qcyc.npy"

        pts = np.load(pts10_path).astype(np.float32)         # (T, 512, 10)
        qcyc = np.load(qcyc_path).astype(np.float32)         # (T, 4)
        T, N, _ = pts.shape

        if self.raw_stats is not None:
            pts[..., 0] = (pts[..., 0] - self.raw_stats["x_mean"]) / self.raw_stats["x_std"]
            pts[..., 1] = (pts[..., 1] - self.raw_stats["y_mean"]) / self.raw_stats["y_std"]
            pts[..., 2] = (pts[..., 2] - self.raw_stats["z_mean"]) / self.raw_stats["z_std"]
            pts[..., 3] = (pts[..., 3] - self.raw_stats["t_mean"]) / self.raw_stats["t_std"]

        pts[..., 4:10] = (pts[..., 4:10] - self.octo_mean) / self.octo_std

        # broadcast qcyc (T, 4) to (T, N, 4)
        qcyc_per_pt = np.broadcast_to(qcyc[:, None, :], (T, N, 4)).copy()

        out = np.concatenate([pts, qcyc_per_pt], axis=-1)  # (T, 512, 14)
        out_t = self.transform(out.reshape(-1, 14))
        if isinstance(out_t, torch.Tensor):
            out_t = out_t.view(T, N, 14)
        else:
            out_t = torch.from_numpy(out_t).view(T, N, 14)

        return out_t.float(), label, line

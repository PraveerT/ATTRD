"""
NvidiaPts10Loader — single-stream 10-channel input.

Reads `<stub>_pts10.npy` (T, 512, 10) where:
  channels [0:4] = raw uvd-t (z-scored with nvidia_dataset_stats.npy)
  channels [4:10] = nearest-lattice octo (x, y, z, vx, vy, vz)
                   z-scored with nvidia_octo_stats.npy

Both halves are whitened independently. Augmentations (Compose pipeline) apply
to channels [:4] only — same xyz-style augments the baseline uses on raw —
because channels [4:10] are corresponded lattice features that scaling /
jittering would silently desynchronize.
"""

import os
import re
import numpy as np
import torch
import torch.utils.data as data

from utils.pts_transform import (
    Compose, PointcloudToTensor, PointcloudScale,
    PointcloudRotatePerturbation, PointcloudTranslate,
    TemporalSpeedChange, TemporalTranslate, TemporalCutout, TemporalShuffle,
)


class NvidiaPts10Loader(data.Dataset):
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
            print("[NvidiaPts10Loader] WARN: nvidia_dataset_stats.npy missing")
            self.raw_stats = None

        try:
            stats = np.load("nvidia_octo_stats.npy", allow_pickle=True).item()
            self.octo_mean = stats["mean"].astype(np.float32)
            self.octo_std = stats["std"].astype(np.float32)
        except FileNotFoundError:
            print("[NvidiaPts10Loader] WARN: nvidia_octo_stats.npy missing")
            self.octo_mean = np.zeros(6, dtype=np.float32)
            self.octo_std = np.ones(6, dtype=np.float32)

        if phase == "train":
            self.transform = Compose([
                PointcloudToTensor(),
                PointcloudScale(lo=0.85, hi=1.15),
                PointcloudRotatePerturbation(angle_sigma=0.08, angle_clip=0.18),
                PointcloudTranslate(translate_range=0.1),
                TemporalSpeedChange(speed_range=(0.85, 1.15), prob=0.3),
                TemporalTranslate(max_shift_ratio=0.2, prob=0.4),
                TemporalCutout(max_cutout_ratio=0.2, num_holes=(1, 4), prob=0.3),
                TemporalShuffle(window_size=7, num_shuffles=4, prob=0.4),
            ])
        else:
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

        pts = np.load(pts10_path).astype(np.float32)  # (T, 512, 10)
        T, N, _ = pts.shape

        # raw half: z-score channels 0..3 using uvd-t stats
        if self.raw_stats is not None:
            pts[..., 0] = (pts[..., 0] - self.raw_stats["x_mean"]) / self.raw_stats["x_std"]
            pts[..., 1] = (pts[..., 1] - self.raw_stats["y_mean"]) / self.raw_stats["y_std"]
            pts[..., 2] = (pts[..., 2] - self.raw_stats["z_mean"]) / self.raw_stats["z_std"]
            pts[..., 3] = (pts[..., 3] - self.raw_stats["t_mean"]) / self.raw_stats["t_std"]

        # octo half: z-score channels 4..9
        pts[..., 4:10] = (pts[..., 4:10] - self.octo_mean) / self.octo_std

        # augmentations on raw half only — split, transform, recombine
        raw = pts[..., :4].reshape(-1, 4)
        raw_t = self.transform(raw)
        if isinstance(raw_t, torch.Tensor):
            raw_t = raw_t.view(T, N, 4)
        else:
            raw_t = torch.from_numpy(raw_t).view(T, N, 4)

        octo_t = torch.from_numpy(pts[..., 4:10])

        out = torch.cat([raw_t.float(), octo_t.float()], dim=-1)
        return out, label, line

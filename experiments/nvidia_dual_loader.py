"""
NvidiaDualLoader — loads BOTH raw `<stub>_pts.npy` and lattice `<stub>_octo.npy`
and returns them concatenated along the channel dim:

    output[..., 0:8] = raw  (uvd-t + xyz-t, z-scored using nvidia_dataset_stats)
    output[..., 8:14] = octo (x, y, z, vx, vy, vz, z-scored using nvidia_octo_stats)

Shape: (T=32, P=512, 14). Designed for a two-stream model that splits along
channel and runs separate encoders on each half.

Augmentations apply ONLY to raw channels [0:8] (existing transform pipeline);
octo channels [8:14] pass through unchanged because they have correspondence
that scaling/jittering would break.
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


class NvidiaDualLoader(data.Dataset):
    def __init__(self, framerate, valid_subject=None, phase="train",
                 datatype="depth", inputs_type="pts", **kwargs):
        self.phase = phase
        self.datatype = datatype
        self.framerate = framerate
        self.valid_subject = valid_subject
        self.r = re.compile(r"[ \t\n\r:]+")
        self.inputs_list = self.get_inputs_list()

        # raw uvd-t stats
        try:
            self.raw_stats = np.load("nvidia_dataset_stats.npy", allow_pickle=True).item()
        except FileNotFoundError:
            print("[NvidiaDualLoader] WARN: nvidia_dataset_stats.npy missing")
            self.raw_stats = None

        # octo stats
        try:
            stats = np.load("nvidia_octo_stats.npy", allow_pickle=True).item()
            self.octo_mean = stats["mean"].astype(np.float32)
            self.octo_std = stats["std"].astype(np.float32)
        except FileNotFoundError:
            print("[NvidiaDualLoader] WARN: nvidia_octo_stats.npy missing")
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

    def _normalize_raw(self, pts):
        # pts shape (T, N, 8) — z-score channels 0..3 (uvd-t)
        pts = pts.astype(np.float32)
        if self.raw_stats is not None:
            pts[..., 0] = (pts[..., 0] - self.raw_stats["x_mean"]) / self.raw_stats["x_std"]
            pts[..., 1] = (pts[..., 1] - self.raw_stats["y_mean"]) / self.raw_stats["y_std"]
            pts[..., 2] = (pts[..., 2] - self.raw_stats["z_mean"]) / self.raw_stats["z_std"]
            pts[..., 3] = (pts[..., 3] - self.raw_stats["t_mean"]) / self.raw_stats["t_std"]
        return pts

    def __getitem__(self, index):
        line = self.inputs_list[index]
        label = int(self.r.split(line)[-2])
        depth_npy = self.r.split(line)[1]

        raw_path = "../dataset/" + depth_npy[2:-4] + "_pts.npy"
        octo_path = "../dataset/" + depth_npy[2:-4] + "_octo.npy"

        raw = np.load(raw_path)        # (T, 512, 8) int
        octo = np.load(octo_path)      # (T, 512, 8) float32

        T, N, _ = raw.shape

        # raw → z-scored 8-channel
        raw = self._normalize_raw(raw)

        # raw augmentation: flatten to (T*N, 8), transform, reshape
        flat = raw.reshape(-1, 8)
        flat = self.transform(flat)
        if isinstance(flat, torch.Tensor):
            raw_t = flat.view(T, N, 8)
        else:
            raw_t = torch.from_numpy(flat).view(T, N, 8)

        # octo: drop e0, e4 → 6 channels, z-score, no augmentation (correspondence)
        octo6 = octo[..., [1, 2, 3, 5, 6, 7]].astype(np.float32)
        octo6 = (octo6 - self.octo_mean) / self.octo_std
        octo_t = torch.from_numpy(octo6)

        # concat along channel dim → (T, 512, 14)
        combined = torch.cat([raw_t.float(), octo_t], dim=-1)
        return combined, label, line

"""
NvidiaOctoLoader — loads `<stub>_octo.npy` (T, 512, 8) sidecar files instead
of the raw `<stub>_pts.npy`. Drops the constant scalar parts (e0, e4 = 1)
and returns 6 channels: (x, y, z, vx, vy, vz) per lattice point per frame.

Used for the option-2a drop-in test: corresponded lattice + real velocity as
a 6-channel input to the existing Motion backbone.
"""

import os
import re
import numpy as np
import torch.utils.data as data

from utils.pts_transform import (
    Compose, PointcloudToTensor, PointcloudScale,
    PointcloudRotatePerturbation, PointcloudTranslate, PointcloudJitter,
)


class NvidiaOctoLoader(data.Dataset):
    """Loads octonion lattice sidecar (`_octo.npy`).

    Returns input shape (T, 512, 6): (x, y, z, vx, vy, vz) per point per frame,
    z-scored using `nvidia_octo_stats.npy`.
    """

    def __init__(self, framerate, valid_subject=None, phase="train",
                 datatype="depth", inputs_type="pts", **kwargs):
        self.phase = phase
        self.datatype = datatype
        self.framerate = framerate
        self.valid_subject = valid_subject
        self.r = re.compile(r"[ \t\n\r:]+")
        self.inputs_list = self.get_inputs_list()

        # global octo stats
        try:
            stats = np.load("nvidia_octo_stats.npy", allow_pickle=True).item()
            self.mean = stats["mean"].astype(np.float32)
            self.std = stats["std"].astype(np.float32)
            print(f"[NvidiaOctoLoader] loaded octo stats: "
                  f"mean={self.mean.tolist()}, std={self.std.tolist()}")
        except FileNotFoundError:
            print("[NvidiaOctoLoader] octo stats NOT FOUND; using zero-mean unit-std")
            self.mean = np.zeros(6, dtype=np.float32)
            self.std = np.ones(6, dtype=np.float32)

        if phase == "train":
            self.transform = self._make_train_transform()
        else:
            self.transform = Compose([PointcloudToTensor()])

    @staticmethod
    def _make_train_transform():
        return Compose([
            PointcloudToTensor(),
            PointcloudScale(lo=0.85, hi=1.15),
            PointcloudRotatePerturbation(angle_sigma=0.08, angle_clip=0.18),
            PointcloudTranslate(translate_range=0.1),
            PointcloudJitter(std=0.005, clip=0.02),
        ])

    def get_inputs_list(self):
        prefix = "../dataset/Nvidia/Processed"
        if self.phase == "train":
            path = f"{prefix}/train_depth_list.txt"
            lines = open(path).readlines()
            ret = [ln for ln in lines
                   if f"subject{self.valid_subject}_" not in ln]
        elif self.phase == "valid":
            path = f"{prefix}/train_depth_list.txt"
            lines = open(path).readlines()
            ret = [ln for ln in lines
                   if f"subject{self.valid_subject}_" in ln]
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
        # the dataloader's cwd is experiments/ so paths use '../dataset/'
        octo_path = "../dataset/" + depth_npy[2:-4] + "_octo.npy"
        octo = np.load(octo_path).astype(np.float32)  # (T, 512, 8)

        # take channels (x, y, z, vx, vy, vz) — drop the two constant 1s
        pts6 = octo[..., [1, 2, 3, 5, 6, 7]]  # (T, 512, 6)
        T, N, C = pts6.shape

        # z-score per channel
        pts6 = (pts6 - self.mean) / self.std

        # apply augmentations (Compose works on flattened (T*N, 6))
        flat = pts6.reshape(-1, C)
        flat = self.transform(flat)
        pts6 = flat.reshape(T, N, C) if hasattr(flat, "reshape") else flat.view(T, N, C)

        return pts6, label, line

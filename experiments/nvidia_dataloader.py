import re
import pdb
import sys
import os
import hashlib
import numpy as np
import torch

sys.path.append("..")

from utils import *
import torch.utils.data as data
from utils.pts_transform import *
from dataset import utils as dataset_utils

class NvidiaLoader(data.Dataset):
    # Per-worker in-process RAM cache for decoded npy files (filled lazily).
    _npy_cache = {}

    def __init__(self, framerate, valid_subject=None, phase="train", datatype="depth", inputs_type="pts"):
        self.phase = phase
        self.datatype = datatype
        self.inputs_type = inputs_type
        self.framerate = framerate
        self.valid_subject = valid_subject
        self.inputs_list = self.get_inputs_list()
        self.r = re.compile('[ \t\n\r:]+')

        # Load global dataset statistics
        try:
            self.dataset_stats = np.load('nvidia_dataset_stats.npy', allow_pickle=True).item()
            print("Loaded global dataset statistics for consistent normalization")
        except FileNotFoundError:
            print("Warning: Global dataset statistics not found, using default normalization")
            self.dataset_stats = None

        print(len(self.inputs_list))
        if phase == "train":
            self.transform = self.transform_init("train")
        elif phase in ["test", "valid"]:
            self.transform = self.transform_init("test")

    def __getitem__(self, index):
        line = self.inputs_list[index]
        label = int(self.r.split(line)[-2])
        path = f"../dataset/{self.r.split(line)[1][1:-4]}_pts.npy"
        # RAM cache: first access loads from disk; subsequent epochs hit memory.
        cached = NvidiaLoader._npy_cache.get(path)
        if cached is None:
            cached = np.load(path).astype(np.float32)
            NvidiaLoader._npy_cache[path] = cached
        # Copy because normalize/transform mutate in place.
        input_data = cached.copy()
        input_data = self.normalize(input_data, self.framerate)
        return input_data, label, line

    def get_inputs_list(self):
        prefix = "../dataset/Nvidia/Processed"
        if self.phase == "train":
            if self.datatype == "depth":
                inputs_path = prefix + "/train_depth_list.txt"
            elif self.datatype == "rgb":
                inputs_path = prefix + "/train_color_list.txt"
            inputs_list = open(inputs_path).readlines()
            ret_line = []
            for line in inputs_list:
                if "subject" + str(self.valid_subject) + "_" in line:
                    continue
                ret_line.append(line)
        elif self.phase == "valid":
            if self.datatype == "depth":
                inputs_path = prefix + "/train_depth_list.txt"
            elif self.datatype == "rgb":
                inputs_path = prefix + "/train_color_list.txt"
            inputs_list = open(inputs_path).readlines()
            ret_line = []
            for line in inputs_list:
                if "subject" + str(self.valid_subject) + "_" in line:
                    ret_line.append(line)
        elif self.phase == "test":
            if self.datatype == "depth":
                inputs_path = prefix + "/test_depth_list.txt"
            elif self.datatype == "rgb":
                inputs_path = prefix + "/test_color_list.txt"
            ret_line = open(inputs_path).readlines()
        else:
            AssertionError("Phase error.")
        return ret_line

    def __len__(self):
        return len(self.inputs_list)

    def normalize(self, pts, fs):
        timestep, pts_size, channels = pts.shape
        pts = pts.reshape(-1, channels)
        pts[:, 0] = (pts[:, 0] - self.dataset_stats['x_mean']) / self.dataset_stats['x_std']
        pts[:, 1] = (pts[:, 1] - self.dataset_stats['y_mean']) / self.dataset_stats['y_std']
        pts[:, 2] = (pts[:, 2] - self.dataset_stats['z_mean']) / self.dataset_stats['z_std']
        pts[:, 3] = (pts[:, 3] - self.dataset_stats['t_mean']) / self.dataset_stats['t_std']
        pts = self.transform(pts)
        pts = pts.reshape(timestep, pts_size, channels)
        return pts

    @staticmethod
    def transform_init(phase):
        # All heavy augmentations moved to GpuAugmentor (utils/gpu_augment.py).
        # Loader only does float-tensor conversion; train aug runs on GPU in main.train().
        return Compose([PointcloudToTensor()])

    @staticmethod
    def key_frame_sampling(key_cnt, frame_size):
        factor = frame_size * 1.0 / key_cnt
        index = [int(j / factor) for j in range(frame_size)]
        return index

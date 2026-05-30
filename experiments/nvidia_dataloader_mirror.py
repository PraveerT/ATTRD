"""NvidiaLoader + horizontal (x-axis) mirror augmentation, train-phase only.

Subclass so the shared NvidiaLoader is untouched. Mirror = reflect the x channel
about the per-sample x-centroid (subtract mean-x over all T*P points, negate, add
back) -> a proper reflection of the (already normalized) point cloud, not an
origin flip. Rotations in the existing GpuAugmentor are det +1 and cannot reach
this reflection, so it is genuinely new augmentation. p=0.5, train only.
"""
import torch
from nvidia_dataloader import NvidiaLoader


class NvidiaLoaderMirror(NvidiaLoader):
    def __init__(self, *args, aug_mirror=True, mirror_prob=0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.aug_mirror = bool(aug_mirror)
        self.mirror_prob = float(mirror_prob)

    def __getitem__(self, index):
        sample = self.tensor[index].clone()          # (T, P, 4) -- clone: never mutate the shared preload
        label = int(self.labels_tensor[index])
        if self.phase == 'train' and self.aug_mirror and torch.rand(1).item() < self.mirror_prob:
            xm = sample[:, :, 0].mean()
            sample[:, :, 0] = -(sample[:, :, 0] - xm) + xm   # centroid-centered x reflection
        return sample, label, self.inputs_list[index]

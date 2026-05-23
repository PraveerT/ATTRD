"""CanonicalNvidiaLoader: mirrors NvidiaLoader's code path exactly except
the data source is the baked canonical .npy file (AE output).

Format guarantee: returns (tensor, label, line) with tensor shape (T, P, 4)
and dtype float32, pinned memory, mean~0 std~1 per channel -- bit-for-bit
the same interface as NvidiaLoader so the downstream augmentor/sampler/main
code paths cannot tell them apart.
"""
import os
import numpy as np
import torch
import torch.utils.data as data


class CanonicalNvidiaLoader(data.Dataset):
    # Mirror NvidiaLoader's class-level preload cache.
    _preloaded = {}

    def __init__(self, framerate, valid_subject=None, phase='train',
                 datatype='depth', inputs_type='pts'):
        self.phase = phase
        self.datatype = datatype
        self.inputs_type = inputs_type
        self.framerate = framerate
        self.valid_subject = valid_subject

        prefix = '../dataset/Nvidia/Processed'
        canon_path = os.path.join(prefix, f'canonical_{phase}.npy')
        label_path = os.path.join(prefix, f'canonical_{phase}_labels.npy')

        key = (phase, canon_path)
        if key not in CanonicalNvidiaLoader._preloaded:
            self._preload(key, canon_path, label_path)
        self.tensor, self.labels_tensor = CanonicalNvidiaLoader._preloaded[key]

        # Build a synthetic 'inputs_list' so __getitem__ can return a 'line'
        # string just like NvidiaLoader (the downstream code only uses it for
        # logging; the sig is parseable from the path).
        n = self.tensor.shape[0]
        self.inputs_list = [f'canonical_{phase}_{i}' for i in range(n)]

    def _preload(self, key, canon_path, label_path):
        # NvidiaLoader does: load -> z-score per channel -> stack -> pin.
        # The baked canonical is already z-scored (see bake_canonical.py),
        # so we only need to load, cast, and pin -- matching the storage
        # format that NvidiaLoader leaves in _preloaded[].
        arr = np.load(canon_path).astype(np.float32)        # (N, T, P, 4) z-scored
        lbl = np.load(label_path).astype(np.int64)
        size_mb = arr.nbytes / 1e6
        print(f'[canonical-loader] Preloaded {key[0]}: {arr.shape}, {size_mb:.1f} MB')
        # Cheap sanity: warn if data drifted away from (0, 1) -- normalization
        # broke if so, augmentor will see wrong-scale input.
        xyz_mean = float(arr[..., :3].mean())
        xyz_std  = float(arr[..., :3].std())
        if abs(xyz_mean) > 0.2 or abs(xyz_std - 1.0) > 0.2:
            print(f'[canonical-loader] WARN xyz mean={xyz_mean:.3f} std={xyz_std:.3f} '
                  f'(expected ~0, ~1). Re-bake with bake_canonical.py.')
        tensor = torch.from_numpy(arr).pin_memory()
        labels_t = torch.from_numpy(lbl).pin_memory()
        CanonicalNvidiaLoader._preloaded[key] = (tensor, labels_t)

    def __getitem__(self, index):
        # Exact same contract as NvidiaLoader: (tensor, int label, line str).
        return self.tensor[index], int(self.labels_tensor[index]), self.inputs_list[index]

    def __len__(self):
        return len(self.inputs_list)

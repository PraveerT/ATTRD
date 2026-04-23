"""Yield pmamba pts, depth+tops tensor, and rigidity stats from one sample."""
import torch
from nvidia_dataloader import NvidiaLoader
from depth_branch.dataloader import DepthVideoLoader


class NvidiaMultiLoader(NvidiaLoader):
    """Composes NvidiaLoader (pts) with DepthVideoLoader (depth+tops+rigidity).

    Returns: ((pts, depth_tensor, rigidity_tensor), label, line)
    """

    def __init__(self, *args, img_size=112, use_tops=True, use_rigidity=True,
                 rigidity_per_point=False, rigidity_norm_scale=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        # Share phase / valid_subject / framerate with depth side.
        self._depth_loader = DepthVideoLoader(
            framerate=kwargs.get("framerate", args[0] if args else 32),
            valid_subject=kwargs.get("valid_subject"),
            phase=kwargs.get("phase", "train"),
            img_size=img_size,
            use_tops=use_tops,
            use_rigidity=use_rigidity,
            rigidity_per_point=rigidity_per_point,
            rigidity_norm_scale=rigidity_norm_scale,
            # no train augments here — augmenting pm vs depth independently breaks pairing
            hflip_prob=0.0,
            time_cutout_prob=0.0,
        )

    def __getitem__(self, index):
        pts, label, line = super().__getitem__(index)
        out_d, _, _ = self._depth_loader[index]
        depth_tensor, rigidity_tensor = out_d
        return (pts, depth_tensor, rigidity_tensor), label, line

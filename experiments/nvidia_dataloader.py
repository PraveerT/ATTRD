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
        label = int(self.r.split(self.inputs_list[index])[-2])
        input_data = np.load(f"../dataset/{self.r.split(self.inputs_list[index])[1][1:-4]}_pts.npy").astype(float)
        input_data = self.normalize(input_data, self.framerate)
        return input_data, label, self.inputs_list[index]

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
        
        # if self.dataset_stats is not None:
        # Use global dataset statistics for consistent normalization
        pts[:, 0] = (pts[:, 0] - self.dataset_stats['x_mean']) / self.dataset_stats['x_std']
        pts[:, 1] = (pts[:, 1] - self.dataset_stats['y_mean']) / self.dataset_stats['y_std']
        pts[:, 2] = (pts[:, 2] - self.dataset_stats['z_mean']) / self.dataset_stats['z_std']
        pts[:, 3] = (pts[:, 3] - self.dataset_stats['t_mean']) / self.dataset_stats['t_std']
        # else:
        #     # Fallback to original per-sample normalization if stats not available
        #     pts[:, 0] = (pts[:, 0] - np.mean(pts[:, 0])) / 120
        #     pts[:, 1] = (pts[:, 1] - np.mean(pts[:, 1])) / 160
        #     pts[:, 3] = (pts[:, 3] - fs / 2) / fs * 2
        #     if (pts[:, 2].max() - pts[:, 2].min()) != 0:
        #         pts[:, 2] = (pts[:, 2] - np.mean(pts[:, 2])) / (pts[:, 2].max() - pts[:, 2].min()) * 2
        
        pts = self.transform(pts)
        pts = pts.reshape(timestep, pts_size, channels)
        return pts

    @staticmethod
    def transform_init(phase):
        if phase == 'train':
            transform = Compose([
                PointcloudToTensor(),
                PointcloudScale(lo=0.85, hi=1.15),
                PointcloudRotatePerturbation(angle_sigma=0.08, angle_clip=0.22),
                PointcloudTranslate(translate_range=0.1),
                TemporalSpeedChange(speed_range=(0.85, 1.15), prob=0.3),
                TemporalTranslate(max_shift_ratio=0.2, prob=0.4),
                TemporalCutout(max_cutout_ratio=0.2, num_holes=(1, 4), prob=0.6),
                TemporalShuffle(window_size=7, num_shuffles=4, prob=0.4),
                PointcloudJitter(std=0.015, clip=0.06),
                PointcloudRandomInputDropout(max_dropout_ratio=0.25),
            ])
        else:
            transform = Compose([
                PointcloudToTensor(),
            ])
        return transform

    @staticmethod
    def key_frame_sampling(key_cnt, frame_size):
        factor = frame_size * 1.0 / key_cnt
        index = [int(j / factor) for j in range(frame_size)]
        return index

class NvidiaREQNNLoader(NvidiaLoader):
    """REQNN-specific Nvidia loader that keeps branch-1 data untouched.

    This loader reads the SHREC-style xyz+t representation already stored for the
    Nvidia dataset, and applies light xyz-space augmentations intended for the
    geometry branch only.
    """

    def normalize(self, pts, fs):
        timestep, pts_size, channels = pts.shape

        if channels >= 8:
            xyzt = pts[..., 4:8].astype(np.float32)
        else:
            raw_uvdt = pts[..., :4].astype(np.float32)
            xyzt = np.zeros((timestep, pts_size, 4), dtype=np.float32)
            for frame_idx in range(timestep):
                xyzt[frame_idx] = dataset_utils.uvd2xyz_sherc(raw_uvdt[frame_idx].copy()).astype(np.float32)

        # Keep time as a centered sequence coordinate in [-1, 1].
        time_center = max((fs - 1) / 2.0, 1.0)
        xyzt[..., 3] = (xyzt[..., 3] - time_center) / time_center

        xyzt = self.transform(xyzt.reshape(-1, 4))
        return xyzt.reshape(timestep, pts_size, 4)

    @staticmethod
    def transform_init(phase):
        if phase == 'train':
            transform = Compose([
                PointcloudToTensor(),
                PointcloudScale(lo=0.9, hi=1.1),
                PointcloudRotatePerturbation(angle_sigma=0.06, angle_clip=0.18),
                PointcloudTranslate(translate_range=0.05),
                PointcloudJitter(std=0.01, clip=0.03),
            ])
        else:
            transform = Compose([
                PointcloudToTensor(),
            ])
        return transform

class NvidiaFourierLoader(NvidiaLoader):
    """Net2 dataloader for DS-QCC: returns xyzt (4) + multi-frequency Fourier
    features of XYZ (24) = 28 channels per point. Sin/cos at k in {1,2,4,8}.
    First 4 channels remain raw xyzt so the model's k-NN grouping (distance_dim=[0,1,2])
    works on real spatial coords.
    """

    FOURIER_FREQS = [1.0, 2.0, 4.0, 8.0]  # 4 freqs * (sin+cos) * 3 dims = 24 channels

    def normalize(self, pts, fs):
        timestep, pts_size, channels = pts.shape

        # xyzt from cols 4-8 of the 8-channel _pts.npy
        if channels >= 8:
            xyzt = pts[..., 4:8].astype(np.float32)
        else:
            raw_uvdt = pts[..., :4].astype(np.float32)
            xyzt = np.zeros((timestep, pts_size, 4), dtype=np.float32)
            for f in range(timestep):
                xyzt[f] = dataset_utils.uvd2xyz_sherc(raw_uvdt[f].copy()).astype(np.float32)

        # Time -> [-1, 1]
        time_center = max((fs - 1) / 2.0, 1.0)
        xyzt[..., 3] = (xyzt[..., 3] - time_center) / time_center

        # Normalize XYZ to roughly unit-scale BEFORE Fourier encoding.
        XYZ_SCALE = 50.0
        xyzt[..., :3] = xyzt[..., :3] / XYZ_SCALE

        # Geometric aug + tensorize on normalized 4-d xyzt
        xyzt = self.transform(xyzt.reshape(-1, 4))
        if hasattr(xyzt, 'numpy'):
            xyzt_np = xyzt.numpy()
        else:
            xyzt_np = xyzt
        xyzt_np = xyzt_np.reshape(timestep, pts_size, 4)

        # Fourier-encode XYZ across all freqs
        fourier_chans = []
        for k in self.FOURIER_FREQS:
            for axis in range(3):
                fourier_chans.append(np.sin(2 * np.pi * k * xyzt_np[..., axis:axis+1]))
                fourier_chans.append(np.cos(2 * np.pi * k * xyzt_np[..., axis:axis+1]))
        fourier = np.concatenate(fourier_chans, axis=-1)  # (T, P, 24)

        out = np.concatenate([xyzt_np, fourier.astype(np.float32)], axis=-1)  # (T, P, 28)
        if hasattr(xyzt, 'numpy'):
            import torch
            out = torch.from_numpy(out)
        return out

    @staticmethod
    def transform_init(phase):
        if phase == 'train':
            transform = Compose([
                PointcloudToTensor(),
                PointcloudScale(lo=0.9, hi=1.1),
                PointcloudRotatePerturbation(angle_sigma=0.06, angle_clip=0.18),
                PointcloudTranslate(translate_range=0.05),
                PointcloudJitter(std=0.01, clip=0.03),
            ])
        else:
            transform = Compose([PointcloudToTensor()])
        return transform

class NvidiaTransInvLoader(NvidiaLoader):
    """Translation-invariant Net2 loader: subtract per-clip centroid of (u, v, d)
    so model input has zero absolute position. Preserves inter-frame trajectory
    (all frames shift by the same amount) and intra-frame relative geometry."""

    def normalize(self, pts, fs):
        # Base normalization: dataset mean/std on uvdt + transforms (incl PointcloudToTensor).
        pts = super().normalize(pts, fs)
        # Now subtract per-clip centroid of (u, v, d) -> removes absolute position.
        # pts is a torch.Tensor of shape (T, P, channels) at this point.
        import torch as _torch
        if isinstance(pts, _torch.Tensor):
            centroid = pts[..., :3].mean(dim=(0, 1), keepdim=True)
            pts = pts.clone()
            pts[..., :3] = pts[..., :3] - centroid
        else:
            import numpy as _np
            centroid = pts[..., :3].mean(axis=(0, 1), keepdims=True)
            pts[..., :3] = pts[..., :3] - centroid
        return pts

class NvidiaDTWLoader(NvidiaLoader):
    """DTW time-rescale via cumulative-motion re-sampling.

    Slow gestures expanded, fast gestures compressed. Each output frame represents
    equal cumulative motion budget. The t channel is overwritten with uniform spacing
    so the model sees gesture in motion-normalized time, decoupled from absolute pace.
    """

    def normalize(self, pts, fs):
        import torch as _torch
        pts = super().normalize(pts, fs)
        if not isinstance(pts, _torch.Tensor):
            pts = _torch.from_numpy(pts)
        T = pts.shape[0]
        if T < 2:
            return pts
        centroids = pts[..., :3].mean(dim=1)
        motion = (centroids[1:] - centroids[:-1]).norm(dim=-1)
        cum = _torch.cat([_torch.zeros(1, device=pts.device), motion.cumsum(0)])
        total = cum[-1].item()
        if total < 1e-6:
            return pts
        targets = _torch.linspace(0, total, T, device=pts.device)
        new_idx = _torch.searchsorted(cum, targets).clamp(0, T - 1)
        pts_rescaled = pts[new_idx].clone()
        t_min = pts[..., 3].min().item()
        t_max = pts[..., 3].max().item()
        new_t = _torch.linspace(t_min, t_max, T, device=pts.device)
        pts_rescaled[..., 3] = new_t.unsqueeze(-1).expand(T, pts.shape[1])
        return pts_rescaled

class NvidiaPerPointDTWLoader(NvidiaDTWLoader):
    """Frame-DTW + per-point t-channel modulation by spatial dist from centroid."""

    KAPPA = 0.3  # modulation strength

    def normalize(self, pts, fs):
        import torch as _torch
        pts = super().normalize(pts, fs)  # applies frame-level DTW first
        if not isinstance(pts, _torch.Tensor):
            pts = _torch.from_numpy(pts)
        T, P, _ = pts.shape
        # Per-frame: distance from centroid in (u, v, d) space
        for t in range(T):
            centroid = pts[t, :, :3].mean(dim=0, keepdim=True)  # (1, 3)
            dist = (pts[t, :, :3] - centroid).norm(dim=-1)  # (P,)
            mean_d = dist.mean()
            std_d = dist.std() + 1e-6
            dist_norm = (dist - mean_d) / std_d  # zero-mean unit-variance
            pts[t, :, 3] = pts[t, :, 3] * (1.0 + self.KAPPA * dist_norm)
        return pts



class NvidiaDeformationDTWLoader(NvidiaLoader):
    """DTW with cumulative-deformation budget instead of centroid motion.

    Per frame pair (t, t+1): NN-match + Kabsch -> R. Deformation magnitude =
    mean per-point residual after rigid alignment. Re-sample frames uniformly
    in cum-deformation space: high-deformation segments expanded, low-deformation
    compressed. Net1 sees global motion; this sees frames where shape changes.
    """

    def normalize(self, pts, fs):
        import torch as _torch
        pts = super().normalize(pts, fs)
        if not isinstance(pts, _torch.Tensor):
            pts = _torch.from_numpy(pts)
        T = pts.shape[0]
        if T < 2:
            return pts
        xyz = pts[..., :3]
        defo_mag = _torch.zeros(T - 1, device=xyz.device, dtype=xyz.dtype)
        eye3 = _torch.eye(3, dtype=xyz.dtype, device=xyz.device)
        for t in range(T - 1):
            p_src = xyz[t]
            p_tgt = xyz[t + 1]
            c_s = p_src.mean(dim=0, keepdim=True)
            c_t = p_tgt.mean(dim=0, keepdim=True)
            u = p_src - c_s
            v = p_tgt - c_t
            dist = _torch.cdist(u, v)
            nn = dist.argmin(dim=-1)
            v_m = v[nn]
            H = u.t() @ v_m + 1e-5 * eye3
            try:
                U, S, Vh = _torch.linalg.svd(H)
                V = Vh.t()
                det = _torch.det(V @ U.t())
                D = _torch.diag(_torch.tensor([1.0, 1.0, det.item()], dtype=H.dtype, device=H.device))
                R = V @ D @ U.t()
            except Exception:
                R = eye3
            pred = u @ R.t()
            defo_mag[t] = (v_m - pred).norm(dim=-1).mean()
        cum = _torch.cat([_torch.zeros(1, device=pts.device), defo_mag.cumsum(0)])
        total = cum[-1].item()
        if total < 1e-6:
            return pts
        targets = _torch.linspace(0, total, T, device=pts.device)
        new_idx = _torch.searchsorted(cum, targets).clamp(0, T - 1)
        pts_rescaled = pts[new_idx].clone()
        t_min = pts[..., 3].min().item()
        t_max = pts[..., 3].max().item()
        new_t = _torch.linspace(t_min, t_max, T, device=pts.device)
        pts_rescaled[..., 3] = new_t.unsqueeze(-1).expand(T, pts.shape[1])
        return pts_rescaled




class NvidiaFrenetLoader(NvidiaLoader):
    """Frenet-Serret intrinsic invariants. Replace xyz with (curvature, torsion, speed)."""

    def normalize(self, pts, fs):
        import torch as _torch
        pts = super().normalize(pts, fs)
        if not isinstance(pts, _torch.Tensor):
            pts = _torch.from_numpy(pts)
        T, P, C = pts.shape
        if T < 5:
            return pts
        xyz = pts[..., :3]
        traj = xyz.permute(1, 0, 2)
        gp = _torch.zeros_like(traj)
        gp[:, 1:-1] = (traj[:, 2:] - traj[:, :-2]) / 2
        gpp = _torch.zeros_like(traj)
        gpp[:, 1:-1] = traj[:, 2:] - 2 * traj[:, 1:-1] + traj[:, :-2]
        gppp = _torch.zeros_like(traj)
        gppp[:, 2:-2] = (traj[:, 4:] - 2 * traj[:, 3:-1] + 2 * traj[:, 1:-3] - traj[:, :-4]) / 2
        cross = _torch.cross(gp, gpp, dim=-1)
        norm_gp = gp.norm(dim=-1).clamp(min=1e-6)
        norm_cross = cross.norm(dim=-1).clamp(min=1e-9)
        kappa = norm_cross / (norm_gp ** 3)
        tau = (cross * gppp).sum(dim=-1) / (norm_cross ** 2)
        kappa = kappa.clamp(-100.0, 100.0).permute(1, 0)
        tau = tau.clamp(-100.0, 100.0).permute(1, 0)
        speed = norm_gp.permute(1, 0)
        out = pts.clone()
        for i, val in enumerate([kappa, tau, speed]):
            mu = val.mean()
            std = val.std().clamp(min=1e-3)
            out[..., i] = (val - mu) / std
        t_min = pts[..., 3].min().item()
        t_max = pts[..., 3].max().item()
        new_t = _torch.linspace(t_min, t_max, T, device=pts.device, dtype=pts.dtype)
        out[..., 3] = new_t.unsqueeze(-1).expand(T, P)
        return out




class NvidiaVelocityLoader(NvidiaLoader):
    """Replace per-point xyz with per-point velocity (1st time-derivative).

    Net1 sees positions; this sees momentum/velocity. Conjugate variables in
    mechanics. xyz channels become (vx, vy, vz); time channel preserved.
    """

    def normalize(self, pts, fs):
        import torch as _torch
        pts = super().normalize(pts, fs)
        if not isinstance(pts, _torch.Tensor):
            pts = _torch.from_numpy(pts)
        T, P, C = pts.shape
        if T < 3:
            return pts
        xyz = pts[..., :3]
        vel = _torch.zeros_like(xyz)
        vel[1:-1] = (xyz[2:] - xyz[:-2]) / 2
        vel[0] = vel[1]
        vel[-1] = vel[-2]
        # Standardize per-channel (preserve relative magnitudes)
        for i in range(3):
            mu = vel[..., i].mean()
            std = vel[..., i].std().clamp(min=1e-3)
            vel[..., i] = (vel[..., i] - mu) / std
        out = pts.clone()
        out[..., :3] = vel
        return out




class NvidiaPairwiseDistLoader(NvidiaLoader):
    """Replace xyz with k-nearest-neighbor distances per point (rigid-invariant metric).

    For each point in each frame: distances to k=3 nearest other points (sorted asc).
    Net sees the local metric structure; absolute coordinates discarded. Pure shape.
    """

    def normalize(self, pts, fs):
        import torch as _torch
        pts = super().normalize(pts, fs)
        if not isinstance(pts, _torch.Tensor):
            pts = _torch.from_numpy(pts)
        T, P, C = pts.shape
        xyz = pts[..., :3]
        # Per-frame pairwise dist (T, P, P)
        d = _torch.cdist(xyz, xyz)
        # mask self-distance (diagonal) by adding huge value
        eye = _torch.eye(P, dtype=_torch.bool, device=d.device).unsqueeze(0)
        d = d.masked_fill(eye, float('inf'))
        # top-3 smallest distances per point: (T, P, 3)
        d_knn, _ = d.topk(3, dim=-1, largest=False)
        # Standardize each rank channel
        for i in range(3):
            v = d_knn[..., i]
            mu = v.mean()
            std = v.std().clamp(min=1e-3)
            d_knn[..., i] = (v - mu) / std
        out = pts.clone()
        out[..., :3] = d_knn
        return out




class NvidiaYeoJohnsonDTWLoader(NvidiaLoader):
    """DTW with Yeo-Johnson power-warped motion budget.

    weight[t] = YJ(motion[t], lambda) before cumsum. lambda=1 -> standard DTW (Net2).
    lambda=2 -> quadratic super-DTW (more weight on high-motion frames). lambda=0 ->
    log (close to InvDTW behavior). One scalar parameter LAMBDA controls warp.
    """

    LAMBDA = 2.0

    def normalize(self, pts, fs):
        import torch as _torch
        pts = super().normalize(pts, fs)
        if not isinstance(pts, _torch.Tensor):
            pts = _torch.from_numpy(pts)
        T = pts.shape[0]
        if T < 2:
            return pts
        centroids = pts[..., :3].mean(dim=1)
        motion = (centroids[1:] - centroids[:-1]).norm(dim=-1)
        lam = self.LAMBDA
        if lam == 0.0:
            w = _torch.log1p(motion)
        elif lam == 1.0:
            w = motion
        else:
            w = ((motion + 1.0).pow(lam) - 1.0) / lam
        cum = _torch.cat([_torch.zeros(1, device=pts.device), w.cumsum(0)])
        total = cum[-1].item()
        if total < 1e-6:
            return pts
        targets = _torch.linspace(0, total, T, device=pts.device)
        new_idx = _torch.searchsorted(cum, targets).clamp(0, T - 1)
        pts_rescaled = pts[new_idx].clone()
        t_min = pts[..., 3].min().item()
        t_max = pts[..., 3].max().item()
        new_t = _torch.linspace(t_min, t_max, T, device=pts.device)
        pts_rescaled[..., 3] = new_t.unsqueeze(-1).expand(T, pts.shape[1])
        return pts_rescaled




class NvidiaAntiNet1DTWLoader(NvidiaLoader):
    """DTW resampling weighted by INVERSE Net1 per-frame importance.

    Pre-computed importance via single-frame ablation against Net1
    (work_dir/cn_xxl/epoch115_model.pt). For each frame t in clip:
      importance[t] = ||softmax(full) - softmax(ablate_frame_t)||_2
    DTW weight per frame = 1/(importance + eps). Sample frames Net1 ignores.
    Orthogonal-by-construction to Net1.
    """

    IMPORTANCE_PATH = '../dataset/Nvidia/Processed/net1_importance.npy'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        import numpy as _np
        self._importance = _np.load(self.IMPORTANCE_PATH, allow_pickle=True).item()

    def __getitem__(self, index):
        import torch as _torch
        import numpy as _np
        pts, lbl, name = super().__getitem__(index)
        imp = self._importance.get(name, None)
        if imp is None:
            return pts, lbl, name
        if not isinstance(pts, _torch.Tensor):
            pts = _torch.from_numpy(pts) if isinstance(pts, _np.ndarray) else pts
        T = pts.shape[0]
        if imp.shape[0] != T or T < 2:
            return pts, lbl, name
        weight = 1.0 / (_torch.from_numpy(imp).float() + 1e-3)
        cum = _torch.cat([_torch.zeros(1), weight.cumsum(0)])
        total = cum[-1].item()
        if total < 1e-6:
            return pts, lbl, name
        targets = _torch.linspace(0, total, T)
        new_idx = _torch.searchsorted(cum, targets).clamp(0, T - 1)
        pts_rescaled = pts[new_idx].clone()
        t_min = pts[..., 3].min().item()
        t_max = pts[..., 3].max().item()
        new_t = _torch.linspace(t_min, t_max, T)
        pts_rescaled[..., 3] = new_t.unsqueeze(-1).expand(T, pts.shape[1])
        return pts_rescaled, lbl, name




class NvidiaAntiNet1ExpDTWLoader(NvidiaAntiNet1DTWLoader):
    """Smoother anti-Net1: weight = exp(-importance) instead of 1/importance.

    exp(-imp) is bounded in [exp(-max_imp), 1] so dynamic range is smooth.
    Avoids extreme outliers that may have made Net13 degenerate.
    """

    def __getitem__(self, index):
        import torch as _torch
        import numpy as _np
        # Replicate parent logic but with exp(-imp) weight
        pts, lbl, name = NvidiaLoader.__getitem__(self, index)
        imp = self._importance.get(name, None)
        if imp is None:
            return pts, lbl, name
        if not isinstance(pts, _torch.Tensor):
            pts = _torch.from_numpy(pts) if isinstance(pts, _np.ndarray) else pts
        T = pts.shape[0]
        if imp.shape[0] != T or T < 2:
            return pts, lbl, name
        # Standardize importance to zero mean, unit std before exp for stable scale
        imp_t = _torch.from_numpy(imp).float()
        imp_n = (imp_t - imp_t.mean()) / (imp_t.std().clamp(min=1e-6))
        weight = _torch.exp(-imp_n)
        cum = _torch.cat([_torch.zeros(1), weight.cumsum(0)])
        total = cum[-1].item()
        if total < 1e-6:
            return pts, lbl, name
        targets = _torch.linspace(0, total, T)
        new_idx = _torch.searchsorted(cum, targets).clamp(0, T - 1)
        pts_rescaled = pts[new_idx].clone()
        t_min = pts[..., 3].min().item()
        t_max = pts[..., 3].max().item()
        new_t = _torch.linspace(t_min, t_max, T)
        pts_rescaled[..., 3] = new_t.unsqueeze(-1).expand(T, pts.shape[1])
        return pts_rescaled, lbl, name




class NvidiaAngularMomentumLoader(NvidiaLoader):
    """Replace xyz with cross-product C = r x v (angular-momentum-like).

    A = centered position (r = pos - centroid_per_frame)
    B = velocity (centered finite difference along time)
    C = r x v   per point per frame, an antisymmetric bilinear feature.
    Net1 sees position; this sees the emergent rotation axis perpendicular to
    both position and velocity. Cannot be derived from A or B alone.
    """

    def normalize(self, pts, fs):
        import torch as _torch
        pts = super().normalize(pts, fs)
        if not isinstance(pts, _torch.Tensor):
            pts = _torch.from_numpy(pts)
        T, P, C = pts.shape
        if T < 3:
            return pts
        xyz = pts[..., :3]
        centroid = xyz.mean(dim=1, keepdim=True)
        r = xyz - centroid
        v = _torch.zeros_like(xyz)
        v[1:-1] = (xyz[2:] - xyz[:-2]) / 2
        v[0] = v[1]
        v[-1] = v[-2]
        L = _torch.cross(r, v, dim=-1)
        for i in range(3):
            mu = L[..., i].mean()
            std = L[..., i].std().clamp(min=1e-3)
            L[..., i] = (L[..., i] - mu) / std
        out = pts.clone()
        out[..., :3] = L
        return out




class NvidiaWritheLoader(NvidiaLoader):
    """Add per-point trajectory writhe scalar as 5th spatial channel.

    Writhe Wr(gamma_i) = (1/4pi) sum_{s<t} ((gamma(s)-gamma(t)) . (v(s) x v(t)))
                                            / |gamma(s)-gamma(t)|^3
    Captures self-twist / linking signature of each point's trajectory. Pure
    topological invariant, rigid-motion + reparametrization invariant.
    Broadcast across all frames; keeps xyz intact, replaces channel 4 with writhe.
    """

    def normalize(self, pts, fs):
        import torch as _torch
        pts = super().normalize(pts, fs)
        if not isinstance(pts, _torch.Tensor):
            pts = _torch.from_numpy(pts)
        T, P, C = pts.shape
        if T < 3:
            return pts
        xyz = pts[..., :3]  # (T, P, 3)
        v = _torch.zeros_like(xyz)
        v[1:-1] = (xyz[2:] - xyz[:-2]) / 2
        v[0] = v[1]
        v[-1] = v[-2]
        # Per-point writhe scalar via double sum over time pairs (s,t), s<t
        # Vectorize over P
        traj = xyz.permute(1, 0, 2)  # (P, T, 3)
        vp = v.permute(1, 0, 2)       # (P, T, 3)
        # Broadcasting: pair indices s,t
        # diff[p, s, t] = traj[p,s] - traj[p,t]    (P, T, T, 3)
        diff = traj.unsqueeze(2) - traj.unsqueeze(1)
        cross_v = _torch.cross(
            vp.unsqueeze(2).expand(-1, -1, T, -1),
            vp.unsqueeze(1).expand(-1, T, -1, -1),
            dim=-1,
        )  # (P, T, T, 3)
        denom = diff.norm(dim=-1).clamp(min=1e-3) ** 3
        contrib = (diff * cross_v).sum(dim=-1) / denom  # (P, T, T)
        # Take upper triangle (s<t)
        mask = _torch.triu(_torch.ones(T, T, dtype=_torch.bool, device=pts.device), diagonal=1)
        writhe = contrib.masked_select(mask.unsqueeze(0)).reshape(P, -1).sum(dim=-1) / (4 * 3.14159265)
        # Standardize
        mu = writhe.mean()
        std = writhe.std().clamp(min=1e-3)
        w_n = (writhe - mu) / std  # (P,)
        out = pts.clone()
        # Replace channel 4 (index 4) with writhe broadcast
        out[..., 4] = w_n.unsqueeze(0).expand(T, P)
        return out




class NvidiaVorticityLoader(NvidiaLoader):
    """Replace xyz with per-point vorticity (curl of velocity field)."""

    K_NN = 8

    def normalize(self, pts, fs):
        import torch as _torch
        pts = super().normalize(pts, fs)
        if not isinstance(pts, _torch.Tensor):
            pts = _torch.from_numpy(pts)
        T, P, C = pts.shape
        if T < 3:
            return pts
        xyz = pts[..., :3]
        v = _torch.zeros_like(xyz)
        v[1:-1] = (xyz[2:] - xyz[:-2]) / 2
        v[0] = v[1]; v[-1] = v[-2]
        K = min(self.K_NN, P - 1)
        omega = _torch.zeros(T, P, 3, dtype=pts.dtype, device=pts.device)
        eye3 = _torch.eye(3, device=pts.device, dtype=pts.dtype)
        for t in range(T):
            x_t = xyz[t]; v_t = v[t]
            d = _torch.cdist(x_t, x_t)
            d.fill_diagonal_(float('inf'))
            _, nn_idx = d.topk(K, dim=-1, largest=False)
            x_nb = x_t[nn_idx]; v_nb = v_t[nn_idx]
            dx = x_nb - x_t.unsqueeze(1)
            dv = v_nb - v_t.unsqueeze(1)
            dx_t_dx = _torch.einsum('pki,pkj->pij', dx, dx)
            dx_t_dx = dx_t_dx + 1e-4 * eye3.unsqueeze(0)
            dx_t_dv = _torch.einsum('pki,pkj->pij', dx, dv)
            try:
                inv = _torch.linalg.inv(dx_t_dx)
            except Exception:
                continue
            grad_v_T = _torch.einsum('pij,pjk->pik', inv, dx_t_dv)
            grad_v = grad_v_T.transpose(-1, -2)
            Omega = 0.5 * (grad_v - grad_v.transpose(-1, -2))
            omega[t, :, 0] = Omega[:, 2, 1]
            omega[t, :, 1] = Omega[:, 0, 2]
            omega[t, :, 2] = Omega[:, 1, 0]
        for i in range(3):
            mu = omega[..., i].mean()
            std = omega[..., i].std().clamp(min=1e-3)
            omega[..., i] = (omega[..., i] - mu) / std
        out = pts.clone()
        out[..., :3] = omega
        return out




class NvidiaSpatioTemporalDTWLoader(NvidiaLoader):
    """2D DTW: frame re-sample by cum-motion, then per-frame point re-sample by cum-distance.

    Time axis: standard Net2 DTW (cum centroid-motion budget).
    Space axis: per output frame, sort points by distance to centroid, then
    sample P new points uniformly in cum-distance space. Densely-clustered
    regions get fewer samples, spread regions get more.
    Final output: SSM sees spatially-ordered points within each motion-uniform frame.
    """

    def normalize(self, pts, fs):
        import torch as _torch
        pts = super().normalize(pts, fs)
        if not isinstance(pts, _torch.Tensor):
            pts = _torch.from_numpy(pts)
        T, P, C = pts.shape
        if T < 2:
            return pts
        # --- Time DTW ---
        centroids_t = pts[..., :3].mean(dim=1)
        motion_t = (centroids_t[1:] - centroids_t[:-1]).norm(dim=-1)
        cum_t = _torch.cat([_torch.zeros(1, device=pts.device), motion_t.cumsum(0)])
        total_t = cum_t[-1].item()
        if total_t < 1e-6:
            pts_t = pts
        else:
            targets_t = _torch.linspace(0, total_t, T, device=pts.device)
            new_idx_t = _torch.searchsorted(cum_t, targets_t).clamp(0, T - 1)
            pts_t = pts[new_idx_t].clone()
            t_min = pts[..., 3].min().item()
            t_max = pts[..., 3].max().item()
            new_t = _torch.linspace(t_min, t_max, T, device=pts.device)
            pts_t[..., 3] = new_t.unsqueeze(-1).expand(T, P)
        # --- Spatial DTW per frame ---
        out = pts_t.clone()
        for f in range(T):
            xyz = pts_t[f, :, :3]
            centroid = xyz.mean(dim=0, keepdim=True)
            dist = (xyz - centroid).norm(dim=-1)
            sorted_dist, sort_idx = dist.sort()
            cum_s = _torch.cat([_torch.zeros(1, device=pts.device), sorted_dist.cumsum(0)])
            total_s = cum_s[-1].item()
            if total_s < 1e-6:
                continue
            targets_s = _torch.linspace(0, total_s, P, device=pts.device)
            new_idx_s = _torch.searchsorted(cum_s, targets_s).clamp(0, P - 1)
            out[f] = pts_t[f, sort_idx[new_idx_s]]
        return out




class NvidiaAdaptiveYJDTWLoader(NvidiaLoader):
    """Per-clip adaptive Yeo-Johnson DTW: fit lambda to per-clip motion distribution.

    For each clip, MLE-estimate lambda that makes YJ(motion) closest to Gaussian
    (standard YJ purpose). Use this lambda to power-warp the cum-motion budget
    before frame resampling. Each clip gets its own warp intensity.
    """

    LAMBDA_GRID = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

    def _yj(self, y, lam):
        import torch as _torch
        if lam == 0.0:
            return _torch.log1p(y)
        return ((y + 1).pow(lam) - 1) / lam

    def _fit_lam(self, motion):
        import torch as _torch, math
        best_ll = float('-inf')
        best_lam = 1.0
        for lam in self.LAMBDA_GRID:
            y = self._yj(motion, lam)
            mu = y.mean()
            var = y.var().clamp(min=1e-6)
            n = y.numel()
            ll = -0.5 * (((y - mu) ** 2 / var).sum() + n * (var.log() + math.log(2 * math.pi)))
            log_jac = ((lam - 1) * _torch.log(motion + 1)).sum()
            ll = ll + log_jac
            if ll.item() > best_ll:
                best_ll = ll.item()
                best_lam = lam
        return best_lam

    def normalize(self, pts, fs):
        import torch as _torch
        pts = super().normalize(pts, fs)
        if not isinstance(pts, _torch.Tensor):
            pts = _torch.from_numpy(pts)
        T = pts.shape[0]
        if T < 2:
            return pts
        centroids = pts[..., :3].mean(dim=1)
        motion = (centroids[1:] - centroids[:-1]).norm(dim=-1)
        lam = self._fit_lam(motion)
        w = self._yj(motion, lam)
        cum = _torch.cat([_torch.zeros(1, device=pts.device), w.cumsum(0)])
        total = cum[-1].item()
        if total < 1e-6:
            return pts
        targets = _torch.linspace(0, total, T, device=pts.device)
        new_idx = _torch.searchsorted(cum, targets).clamp(0, T - 1)
        pts_rescaled = pts[new_idx].clone()
        t_min = pts[..., 3].min().item()
        t_max = pts[..., 3].max().item()
        new_t = _torch.linspace(t_min, t_max, T, device=pts.device)
        pts_rescaled[..., 3] = new_t.unsqueeze(-1).expand(T, pts.shape[1])
        return pts_rescaled




class NvidiaCombinedDTWLoader(NvidiaAntiNet1DTWLoader):
    """Combine pace (Net2 motion budget) + anti-Net1 (Net14 exp(-imp)) in one DTW.

    weight[t] = (motion[t]/motion_mean) * exp(-imp_standardized[t])
    Single model sees frames re-sampled by both axes simultaneously.
    """

    def __getitem__(self, index):
        import torch as _torch
        import numpy as _np
        pts, lbl, name = NvidiaLoader.__getitem__(self, index)
        imp = self._importance.get(name, None)
        if imp is None:
            return pts, lbl, name
        if not isinstance(pts, _torch.Tensor):
            pts = _torch.from_numpy(pts) if isinstance(pts, _np.ndarray) else pts
        T = pts.shape[0]
        if imp.shape[0] != T or T < 2:
            return pts, lbl, name
        # Anti-Net1 weight (smooth)
        imp_t = _torch.from_numpy(imp).float()
        imp_n = (imp_t - imp_t.mean()) / imp_t.std().clamp(min=1e-6)
        anti_w = _torch.exp(-imp_n)  # length T
        # Motion weight (Net2 style)
        centroids = pts[..., :3].mean(dim=1)
        motion = (centroids[1:] - centroids[:-1]).norm(dim=-1)  # length T-1
        motion_mean = motion.mean().clamp(min=1e-6)
        motion_n = motion / motion_mean
        # Combine: motion has T-1 entries, anti has T. Truncate anti to T-1 (per-pair) by averaging adjacent
        anti_pair = 0.5 * (anti_w[:-1] + anti_w[1:])  # length T-1
        weight = motion_n * anti_pair
        cum = _torch.cat([_torch.zeros(1), weight.cumsum(0)])
        total = cum[-1].item()
        if total < 1e-6:
            return pts, lbl, name
        targets = _torch.linspace(0, total, T)
        new_idx = _torch.searchsorted(cum, targets).clamp(0, T - 1)
        pts_rescaled = pts[new_idx].clone()
        t_min = pts[..., 3].min().item()
        t_max = pts[..., 3].max().item()
        new_t = _torch.linspace(t_min, t_max, T)
        pts_rescaled[..., 3] = new_t.unsqueeze(-1).expand(T, pts.shape[1])
        return pts_rescaled, lbl, name




class NvidiaMultiScaleDTWLoader(NvidiaLoader):
    """Multi-scale DTW: 2 motion-budget scales packed into 8 channels.

    Channels 0-3: DTW with W=1 motion (consecutive frame deltas) — fine scale.
    Channels 4-7: DTW with W=8 motion (8-frame window) — coarse scale.
    Both with t-channel reset uniformly. Model sees two temporal views jointly.
    """

    W_FINE = 1
    W_COARSE = 8

    def _dtw_idx(self, centroids, T, W):
        import torch as _torch
        if T <= W:
            return _torch.arange(T)
        diff = centroids[W:] - centroids[:-W]
        motion = diff.norm(dim=-1)  # (T-W,)
        # Pad motion to length T-1 by edge repetition
        pad_total = (T - 1) - motion.shape[0]
        left = pad_total // 2
        right = pad_total - left
        if left > 0:
            motion = _torch.cat([motion[:1].repeat(left), motion])
        if right > 0:
            motion = _torch.cat([motion, motion[-1:].repeat(right)])
        cum = _torch.cat([_torch.zeros(1, device=centroids.device), motion.cumsum(0)])
        total = cum[-1].item()
        if total < 1e-6:
            return _torch.arange(T, device=centroids.device)
        targets = _torch.linspace(0, total, T, device=centroids.device)
        return _torch.searchsorted(cum, targets).clamp(0, T - 1)

    def normalize(self, pts, fs):
        import torch as _torch
        pts = super().normalize(pts, fs)
        if not isinstance(pts, _torch.Tensor):
            pts = _torch.from_numpy(pts)
        T, P, C = pts.shape
        if T < 2:
            return pts
        centroids = pts[..., :3].mean(dim=1)
        idx_fine = self._dtw_idx(centroids, T, self.W_FINE)
        idx_coarse = self._dtw_idx(centroids, T, self.W_COARSE)
        out = pts.clone()
        out[..., :4] = pts[idx_fine][..., :4]
        out[..., 4:8] = pts[idx_coarse][..., :4]
        t_min = pts[..., 3].min().item()
        t_max = pts[..., 3].max().item()
        new_t = _torch.linspace(t_min, t_max, T, device=pts.device)
        out[..., 3] = new_t.unsqueeze(-1).expand(T, P)
        out[..., 7] = new_t.unsqueeze(-1).expand(T, P)
        return out


class NvidiaQuaternionQCCLoader(NvidiaLoader):
    """Winner-compatible loader with correspondence supervision from raw depth clips."""

    def __init__(
        self,
        framerate,
        valid_subject=None,
        phase="train",
        datatype="depth",
        inputs_type="pts",
        pts_size=256,
        return_correspondence=True,
        correspondence_radius=2,
        correspondence_depth_weight=0.25,
        correspondence_sample_weight=0.1,
        correspondence_confidence_scale=16.0,
        correspondence_max_dist=64.0,
        correspondence_cache=True,
        correspondence_cache_tag="corr_qcc_v1",
    ):
        self.loader_pts_size = pts_size
        self.return_correspondence = return_correspondence
        self.correspondence_radius = correspondence_radius
        self.correspondence_depth_weight = correspondence_depth_weight
        self.correspondence_sample_weight = correspondence_sample_weight
        self.correspondence_confidence_scale = correspondence_confidence_scale
        self.correspondence_max_dist = correspondence_max_dist
        self.correspondence_cache = correspondence_cache
        self.correspondence_cache_tag = correspondence_cache_tag
        super().__init__(
            framerate=framerate,
            valid_subject=valid_subject,
            phase=phase,
            datatype=datatype,
            inputs_type=inputs_type,
        )

    def __getitem__(self, index):
        label = int(self.r.split(self.inputs_list[index])[-2])
        relative_stub = self.r.split(self.inputs_list[index])[1][1:-4]
        points_path = f"../dataset/{relative_stub}_pts.npy"
        raw_depth_path = f"../dataset/{relative_stub}.npy"

        raw_points = np.load(points_path).astype(np.float32)
        raw_points = self._select_points(raw_points, relative_stub)
        input_data = self.normalize(raw_points.copy(), self.framerate)

        sample = {'points': input_data}
        if self.return_correspondence:
            raw_depth = self._load_aligned_depth(raw_depth_path, raw_points.shape[0])
            correspondence = self._load_or_build_correspondence(raw_depth_path, raw_depth, raw_points)
            sample.update(correspondence)
            sample['corr_frame_count'] = np.int64(raw_points.shape[0])
            sample['corr_points_per_frame'] = np.int64(raw_points.shape[1])

        return sample, label, self.inputs_list[index]

    def _select_points(self, raw_points, clip_token):
        frame_count, point_count, channels = raw_points.shape
        if point_count == self.loader_pts_size:
            return raw_points

        selected_points = np.zeros((frame_count, self.loader_pts_size, channels), dtype=np.float32)
        for frame_idx in range(frame_count):
            indices = self._frame_point_indices(point_count, clip_token, frame_idx)
            selected_points[frame_idx] = raw_points[frame_idx, indices]
        return selected_points

    def _frame_point_indices(self, point_count, clip_token, frame_idx):
        if point_count <= self.loader_pts_size:
            seed = self._frame_seed(clip_token, frame_idx)
            rng = np.random.default_rng(seed)
            return rng.choice(point_count, self.loader_pts_size, replace=True)

        if self.phase == "train":
            seed = self._frame_seed(clip_token, frame_idx)
            rng = np.random.default_rng(seed)
            return np.sort(rng.choice(point_count, self.loader_pts_size, replace=False))

        return np.linspace(0, point_count - 1, self.loader_pts_size).round().astype(np.int64)

    def _frame_seed(self, clip_token, frame_idx):
        seed_source = f"{clip_token}:{frame_idx}:{self.loader_pts_size}".encode("utf-8")
        return int.from_bytes(hashlib.sha1(seed_source).digest()[:8], "little") % (2 ** 32)

    def _load_aligned_depth(self, raw_depth_path, num_frames):
        raw_depth = np.load(raw_depth_path).astype(np.float32)
        if raw_depth.ndim == 4:
            raw_depth = raw_depth[..., 0]
        if raw_depth.shape[0] == num_frames:
            return raw_depth
        frame_indices = dataset_utils.key_frame_sampling(raw_depth.shape[0], num_frames)
        return raw_depth[frame_indices]

    def _load_or_build_correspondence(self, raw_depth_path, raw_depth, raw_points):
        cache_path = self._correspondence_cache_path(raw_depth_path)
        if self.correspondence_cache and os.path.exists(cache_path):
            cached = np.load(cache_path)
            return {
                'corr_src_idx': cached['corr_src_idx'].astype(np.int64),
                'corr_tgt_idx': cached['corr_tgt_idx'].astype(np.int64),
                'corr_weight': cached['corr_weight'].astype(np.float32),
            }

        correspondence = self._build_correspondence(raw_depth, raw_points)
        if self.correspondence_cache:
            tmp_path = "{}.{}.tmp.npz".format(cache_path[:-4], os.getpid())
            np.savez_compressed(
                tmp_path,
                corr_src_idx=correspondence['corr_src_idx'],
                corr_tgt_idx=correspondence['corr_tgt_idx'],
                corr_weight=correspondence['corr_weight'],
            )
            try:
                os.replace(tmp_path, cache_path)
            except OSError:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
        return correspondence

    def _correspondence_cache_path(self, raw_depth_path):
        return "{}_{}_p{}_r{}.npz".format(
            raw_depth_path[:-4],
            self.correspondence_cache_tag,
            self.loader_pts_size,
            self.correspondence_radius,
        )

    def _build_correspondence(self, raw_depth, raw_points):
        frame_count, point_count, _ = raw_points.shape
        pair_count = (frame_count - 1) * point_count
        corr_src_idx = np.zeros(pair_count, dtype=np.int64)
        corr_tgt_idx = np.zeros(pair_count, dtype=np.int64)
        corr_weight = np.zeros(pair_count, dtype=np.float32)

        cursor = 0
        for frame_idx in range(frame_count - 1):
            source_points = raw_points[frame_idx, :, :3]
            target_points = raw_points[frame_idx + 1, :, :3]
            matched_points, valid_mask = self._match_dense_points(raw_depth[frame_idx + 1], source_points)

            pair_src_idx = np.arange(point_count, dtype=np.int64) + frame_idx * point_count
            pair_tgt_idx = np.zeros(point_count, dtype=np.int64) + (frame_idx + 1) * point_count
            pair_weight = np.zeros(point_count, dtype=np.float32)

            if valid_mask.any():
                valid_indices = np.where(valid_mask)[0]
                target_dist = self._sampled_target_distance(matched_points[valid_mask], target_points)
                best_target = np.argmin(target_dist, axis=1)
                best_dist = target_dist[np.arange(best_target.shape[0]), best_target]
                confidence = 1.0 / (1.0 + (best_dist / max(self.correspondence_confidence_scale, 1e-6)))
                confidence = confidence.astype(np.float32)
                confidence[best_dist > self.correspondence_max_dist] = 0.0

                pair_tgt_idx[valid_indices] = best_target.astype(np.int64) + (frame_idx + 1) * point_count
                pair_weight[valid_indices] = confidence

            corr_src_idx[cursor:cursor + point_count] = pair_src_idx
            corr_tgt_idx[cursor:cursor + point_count] = pair_tgt_idx
            corr_weight[cursor:cursor + point_count] = pair_weight
            cursor += point_count

        return {
            'corr_src_idx': corr_src_idx,
            'corr_tgt_idx': corr_tgt_idx,
            'corr_weight': corr_weight,
        }

    def _match_dense_points(self, depth_frame, source_points):
        height, width = depth_frame.shape
        matched_points = np.zeros((source_points.shape[0], 3), dtype=np.float32)
        valid_mask = np.zeros(source_points.shape[0], dtype=np.bool_)
        radius = self.correspondence_radius

        for point_idx, point in enumerate(source_points):
            row = int(np.clip(np.round(point[0]), 0, height - 1))
            col = int(np.clip(np.round(point[1]), 0, width - 1))
            depth = float(point[2])
            best_score = None
            best_match = None

            for row_offset in range(-radius, radius + 1):
                candidate_row = row + row_offset
                if candidate_row < 0 or candidate_row >= height:
                    continue
                for col_offset in range(-radius, radius + 1):
                    candidate_col = col + col_offset
                    if candidate_col < 0 or candidate_col >= width:
                        continue
                    candidate_depth = float(depth_frame[candidate_row, candidate_col])
                    if candidate_depth <= 0.0:
                        continue

                    spatial_score = float(row_offset * row_offset + col_offset * col_offset)
                    depth_score = abs(candidate_depth - depth) * self.correspondence_depth_weight
                    score = spatial_score + depth_score
                    if best_score is None or score < best_score:
                        best_score = score
                        best_match = (candidate_row, candidate_col, candidate_depth)

            if best_match is not None:
                matched_points[point_idx] = best_match
                valid_mask[point_idx] = True

        return matched_points, valid_mask

    def _sampled_target_distance(self, matched_points, target_points):
        diff = target_points[None, :, :3] - matched_points[:, None, :3]
        spatial_dist = np.sum(diff[..., :2] ** 2, axis=-1)
        depth_dist = diff[..., 2] ** 2
        return spatial_dist + self.correspondence_sample_weight * depth_dist

    @staticmethod
    def transform_init(phase):
        if phase == 'train':
            transform = Compose([
                PointcloudToTensor(),
                PointcloudScale(lo=0.9, hi=1.1),
                PointcloudRotatePerturbation(angle_sigma=0.06, angle_clip=0.18),
                PointcloudTranslate(translate_range=0.05),
                PointcloudJitter(std=0.01, clip=0.03),
            ])
        else:
            transform = Compose([
                PointcloudToTensor(),
            ])
        return transform

class NvidiaQuaternionQCCParityLoader(NvidiaQuaternionQCCLoader):
    """Winner-preserving QCC loader.

    This keeps the winner's point tensor path intact and only adds provenance
    metadata plus full-clip correspondences built from the raw depth videos.
    """

    def __init__(
        self,
        framerate,
        valid_subject=None,
        phase="train",
        datatype="depth",
        inputs_type="pts",
        pts_size=None,
        return_correspondence=True,
        correspondence_radius=2,
        correspondence_depth_weight=0.25,
        correspondence_sample_weight=0.1,
        correspondence_confidence_scale=16.0,
        correspondence_max_dist=64.0,
        correspondence_cache=True,
        correspondence_cache_tag="corr_qcc_parity_v2",
        bidirectional_correspondence=True,
        assignment_mode="mutual",
    ):
        self.bidirectional_correspondence = bidirectional_correspondence
        if assignment_mode not in ("mutual", "hungarian"):
            raise ValueError(f"assignment_mode must be 'mutual' or 'hungarian', got {assignment_mode}")
        self.assignment_mode = assignment_mode
        super().__init__(
            framerate=framerate,
            valid_subject=valid_subject,
            phase=phase,
            datatype=datatype,
            inputs_type=inputs_type,
            pts_size=pts_size or 256,
            return_correspondence=return_correspondence,
            correspondence_radius=correspondence_radius,
            correspondence_depth_weight=correspondence_depth_weight,
            correspondence_sample_weight=correspondence_sample_weight,
            correspondence_confidence_scale=correspondence_confidence_scale,
            correspondence_max_dist=correspondence_max_dist,
            correspondence_cache=correspondence_cache,
            correspondence_cache_tag=correspondence_cache_tag,
        )

    def __getitem__(self, index):
        label = int(self.r.split(self.inputs_list[index])[-2])
        relative_stub = self.r.split(self.inputs_list[index])[1][1:-4]
        points_path = f"../dataset/{relative_stub}_pts.npy"
        raw_depth_path = f"../dataset/{relative_stub}.npy"

        raw_points = np.load(points_path).astype(float)
        input_data, orig_flat_idx = self.normalize_with_provenance(raw_points.copy(), self.framerate)

        sample = {
            'points': input_data,
            'orig_flat_idx': orig_flat_idx,
        }
        if self.return_correspondence:
            raw_depth = self._load_aligned_depth(raw_depth_path, raw_points.shape[0])
            full_correspondence = self._load_or_build_full_correspondence(raw_depth_path, raw_depth, raw_points.astype(np.float32, copy=False))
            sample.update(full_correspondence)

        return sample, label, self.inputs_list[index]

    def normalize_with_provenance(self, pts, fs):
        timestep, pts_size, channels = pts.shape
        flat_points = pts.reshape(-1, channels)

        flat_points[:, 0] = (flat_points[:, 0] - self.dataset_stats['x_mean']) / self.dataset_stats['x_std']
        flat_points[:, 1] = (flat_points[:, 1] - self.dataset_stats['y_mean']) / self.dataset_stats['y_std']
        flat_points[:, 2] = (flat_points[:, 2] - self.dataset_stats['z_mean']) / self.dataset_stats['z_std']
        flat_points[:, 3] = (flat_points[:, 3] - self.dataset_stats['t_mean']) / self.dataset_stats['t_std']

        provenance = np.arange(timestep * pts_size, dtype=np.int64)
        points_tensor, provenance_tensor = self._apply_transform_with_provenance(flat_points, provenance)
        points_tensor = points_tensor.reshape(timestep, pts_size, channels)
        provenance_tensor = provenance_tensor.reshape(timestep, pts_size)
        return points_tensor, provenance_tensor

    def _apply_transform_with_provenance(self, points, provenance):
        transformed_points = points
        transformed_provenance = provenance

        for transform in self.transform.transforms:
            if isinstance(transform, PointcloudToTensor):
                transformed_points = torch.from_numpy(transformed_points).float()
                transformed_provenance = torch.from_numpy(transformed_provenance).long()
            elif isinstance(transform, PointcloudScale):
                scaler = torch.FloatTensor(1).uniform_(transform.lo, transform.hi)
                transformed_points[:, :3] *= scaler
            elif isinstance(transform, PointcloudRotatePerturbation):
                angles = transform._get_angles()
                Rx = angle_axis(angles[0], torch.FloatTensor([1.0, 0.0, 0.0]))
                Ry = angle_axis(angles[1], torch.FloatTensor([0.0, 1.0, 0.0]))
                Rz = angle_axis(angles[2], torch.FloatTensor([0.0, 0.0, 1.0]))
                rotation_matrix = torch.matmul(torch.matmul(Rz, Ry), Rx)
                transformed_points[:, :3] = torch.matmul(transformed_points[:, :3], rotation_matrix.t())
            elif isinstance(transform, PointcloudTranslate):
                translation = torch.FloatTensor(3).uniform_(-transform.translate_range, transform.translate_range)
                transformed_points[:, :3] += translation
            elif isinstance(transform, PointcloudJitter):
                transformed_points[:, :3] += torch.clamp(
                    torch.randn(transformed_points[:, :3].shape) * transform.std,
                    -transform.clip,
                    transform.clip,
                )
            elif isinstance(transform, TemporalSpeedChange):
                transformed_points, transformed_provenance = self._temporal_speed_change_with_provenance(
                    transformed_points,
                    transformed_provenance,
                    transform,
                )
            elif isinstance(transform, TemporalTranslate):
                transformed_points, transformed_provenance = self._temporal_translate_with_provenance(
                    transformed_points,
                    transformed_provenance,
                    transform,
                )
            elif isinstance(transform, TemporalCutout):
                transformed_points, transformed_provenance = self._temporal_cutout_with_provenance(
                    transformed_points,
                    transformed_provenance,
                    transform,
                )
            elif isinstance(transform, TemporalShuffle):
                transformed_points, transformed_provenance = self._temporal_shuffle_with_provenance(
                    transformed_points,
                    transformed_provenance,
                    transform,
                )
            else:
                raise TypeError("Unsupported transform for provenance tracking: {}".format(type(transform).__name__))

        return transformed_points, transformed_provenance

    @staticmethod
    def _temporal_speed_change_with_provenance(points, provenance, transform):
        if torch.rand(1) > transform.prob:
            return points, provenance

        total_points = points.shape[0]
        speed = torch.rand(1) * (transform.speed_range[1] - transform.speed_range[0]) + transform.speed_range[0]

        if speed < 1.0:
            num_duplicates = int(total_points * (1 - speed) * 0.1)
            if num_duplicates > 0:
                duplicate_indices = torch.randperm(total_points)[:num_duplicates]
                duplicated_points = points[duplicate_indices]
                duplicated_provenance = provenance[duplicate_indices]
                points = torch.cat([points, duplicated_points], dim=0)[:total_points]
                provenance = torch.cat([provenance, duplicated_provenance], dim=0)[:total_points]
        else:
            skip_ratio = min(0.2, (speed - 1.0))
            num_skip = int(total_points * skip_ratio)
            if num_skip > 0:
                skip_indices = torch.randperm(total_points)[:num_skip]
                keep_mask = torch.ones(total_points, dtype=torch.bool)
                keep_mask[skip_indices] = False
                kept_points = points[keep_mask]
                kept_provenance = provenance[keep_mask]

                while kept_points.shape[0] < total_points:
                    pad_size = min(kept_points.shape[0], total_points - kept_points.shape[0])
                    kept_points = torch.cat([kept_points, kept_points[:pad_size]], dim=0)
                    kept_provenance = torch.cat([kept_provenance, kept_provenance[:pad_size]], dim=0)

                points = kept_points[:total_points]
                provenance = kept_provenance[:total_points]

        return points, provenance

    @staticmethod
    def _temporal_translate_with_provenance(points, provenance, transform):
        if torch.rand(1) > transform.prob:
            return points, provenance

        total_points = points.shape[0]
        max_shift = int(total_points * transform.max_shift_ratio)
        if max_shift == 0:
            return points, provenance

        shift = torch.randint(-max_shift, max_shift + 1, (1,)).item()
        if shift > 0:
            return (
                torch.cat([points[:shift], points[:-shift]], dim=0),
                torch.cat([provenance[:shift], provenance[:-shift]], dim=0),
            )
        if shift < 0:
            shift = abs(shift)
            return (
                torch.cat([points[shift:], points[-shift:]], dim=0),
                torch.cat([provenance[shift:], provenance[-shift:]], dim=0),
            )
        return points, provenance

    @staticmethod
    def _temporal_cutout_with_provenance(points, provenance, transform):
        if torch.rand(1) > transform.prob:
            return points, provenance

        total_points = points.shape[0]
        num_holes = torch.randint(transform.num_holes[0], transform.num_holes[1] + 1, (1,)).item()

        result = points.clone()
        result_provenance = provenance.clone()
        invalid_id = torch.full((1,), -1, dtype=result_provenance.dtype)

        for _ in range(num_holes):
            hole_size = torch.randint(1, int(total_points * transform.max_cutout_ratio) + 1, (1,)).item()
            hole_size = min(hole_size, total_points // 4)
            if hole_size == 0:
                continue

            start_idx = torch.randint(0, total_points - hole_size + 1, (1,)).item()
            end_idx = start_idx + hole_size

            if start_idx > 0 and end_idx < total_points:
                alpha = torch.linspace(0, 1, hole_size).unsqueeze(1)
                before_point = result[start_idx - 1:start_idx]
                after_point = result[end_idx:end_idx + 1]
                result[start_idx:end_idx] = before_point * (1 - alpha) + after_point * alpha
                # Propagate correspondence: nearest boundary provenance
                mid = hole_size // 2
                before_prov = result_provenance[start_idx - 1]
                after_prov = result_provenance[end_idx]
                result_provenance[start_idx:start_idx + mid] = before_prov
                result_provenance[start_idx + mid:end_idx] = after_prov
            elif start_idx == 0:
                result[start_idx:end_idx] = result[end_idx:end_idx + hole_size]
                result_provenance[start_idx:end_idx] = result_provenance[end_idx:end_idx + hole_size]
            else:
                result[start_idx:end_idx] = result[start_idx - hole_size:start_idx]
                result_provenance[start_idx:end_idx] = result_provenance[start_idx - hole_size:start_idx]

        return result, result_provenance

    @staticmethod
    def _temporal_shuffle_with_provenance(points, provenance, transform):
        if torch.rand(1) > transform.prob:
            return points, provenance

        total_points = points.shape[0]
        result = points.clone()
        result_provenance = provenance.clone()

        for _ in range(transform.num_shuffles):
            if total_points < transform.window_size:
                continue

            start_idx = torch.randint(0, total_points - transform.window_size + 1, (1,)).item()
            end_idx = start_idx + transform.window_size
            shuffled_indices = torch.randperm(transform.window_size)
            result[start_idx:end_idx] = result[start_idx:end_idx][shuffled_indices]
            result_provenance[start_idx:end_idx] = result_provenance[start_idx:end_idx][shuffled_indices]

        return result, result_provenance

    def _load_or_build_full_correspondence(self, raw_depth_path, raw_depth, raw_points):
        cache_path = self._full_correspondence_cache_path(raw_depth_path)
        if self.correspondence_cache and os.path.exists(cache_path):
            cached = np.load(cache_path)
            return {
                'corr_full_target_idx': cached['corr_full_target_idx'].astype(np.int64),
                'corr_full_weight': cached['corr_full_weight'].astype(np.float32),
            }

        if self.assignment_mode == "hungarian":
            correspondence = self._build_full_correspondence_hungarian(raw_depth, raw_points)
        else:
            correspondence = self._build_full_correspondence(raw_depth, raw_points)
        if self.correspondence_cache:
            tmp_path = "{}.{}.tmp.npz".format(cache_path[:-4], os.getpid())
            np.savez_compressed(
                tmp_path,
                corr_full_target_idx=correspondence['corr_full_target_idx'],
                corr_full_weight=correspondence['corr_full_weight'],
            )
            try:
                os.replace(tmp_path, cache_path)
            except OSError:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
        return correspondence

    def _full_correspondence_cache_path(self, raw_depth_path):
        direction_tag = "bi" if self.bidirectional_correspondence else "uni"
        if self.assignment_mode == "hungarian":
            return "{}_{}_hu_r{}.npz".format(
                raw_depth_path[:-4],
                self.correspondence_cache_tag,
                self.correspondence_radius,
            )
        return "{}_{}_{}_r{}.npz".format(
            raw_depth_path[:-4],
            self.correspondence_cache_tag,
            direction_tag,
            self.correspondence_radius,
        )

    def _build_full_correspondence(self, raw_depth, raw_points):
        frame_count, point_count, _ = raw_points.shape
        total_points = frame_count * point_count
        corr_full_target_idx = np.full(total_points, -1, dtype=np.int64)
        corr_full_weight = np.zeros(total_points, dtype=np.float32)

        for frame_idx in range(frame_count - 1):
            source_points = raw_points[frame_idx, :, :3]
            target_points = raw_points[frame_idx + 1, :, :3]

            forward_target, forward_confidence = self._frame_best_matches(
                source_points,
                target_points,
                raw_depth[frame_idx + 1],
            )

            if self.bidirectional_correspondence:
                backward_source, backward_confidence = self._frame_best_matches(
                    target_points,
                    source_points,
                    raw_depth[frame_idx],
                )
            else:
                backward_source = None
                backward_confidence = None

            for source_idx, target_idx in enumerate(forward_target):
                if target_idx < 0:
                    continue

                confidence = float(forward_confidence[source_idx])
                if confidence <= 0.0:
                    continue

                if self.bidirectional_correspondence:
                    if backward_source[target_idx] != source_idx:
                        continue
                    confidence = min(confidence, float(backward_confidence[target_idx]))
                    if confidence <= 0.0:
                        continue

                source_flat = frame_idx * point_count + source_idx
                target_flat = (frame_idx + 1) * point_count + target_idx
                corr_full_target_idx[source_flat] = target_flat
                corr_full_weight[source_flat] = confidence

        return {
            'corr_full_target_idx': corr_full_target_idx,
            'corr_full_weight': corr_full_weight,
        }

    def _match_dense_points_vec(self, depth_frame, source_points):
        height, width = depth_frame.shape
        n = source_points.shape[0]
        r = self.correspondence_radius
        sr = np.clip(np.round(source_points[:, 0]).astype(np.int32), 0, height - 1)
        sc = np.clip(np.round(source_points[:, 1]).astype(np.int32), 0, width - 1)
        sd = source_points[:, 2].astype(np.float32)
        drs, dcs = np.meshgrid(np.arange(-r, r + 1), np.arange(-r, r + 1), indexing="ij")
        drs = drs.ravel().astype(np.int32)
        dcs = dcs.ravel().astype(np.int32)
        rr = sr[:, None] + drs[None, :]
        cc = sc[:, None] + dcs[None, :]
        in_bounds = (rr >= 0) & (rr < height) & (cc >= 0) & (cc < width)
        rr_c = np.clip(rr, 0, height - 1)
        cc_c = np.clip(cc, 0, width - 1)
        cand_depth = depth_frame[rr_c, cc_c].astype(np.float32)
        valid_pix = in_bounds & (cand_depth > 0)
        spatial = (drs * drs + dcs * dcs).astype(np.float32)
        score = spatial[None, :] + self.correspondence_depth_weight * np.abs(cand_depth - sd[:, None])
        score = np.where(valid_pix, score, np.inf)
        best = np.argmin(score, axis=1)
        best_score = score[np.arange(n), best]
        ok = np.isfinite(best_score)
        matched = np.stack(
            [rr[np.arange(n), best], cc[np.arange(n), best], cand_depth[np.arange(n), best]],
            axis=1,
        ).astype(np.float32)
        return matched, ok

    def _build_full_correspondence_hungarian(self, raw_depth, raw_points):
        from scipy.optimize import linear_sum_assignment
        frame_count, point_count, _ = raw_points.shape
        total_points = frame_count * point_count
        corr_full_target_idx = np.full(total_points, -1, dtype=np.int64)
        corr_full_weight = np.zeros(total_points, dtype=np.float32)
        big_cost = np.float32(1e8)

        for frame_idx in range(frame_count - 1):
            source_points = raw_points[frame_idx, :, :3]
            target_points = raw_points[frame_idx + 1, :, :3]
            matched, ok = self._match_dense_points_vec(raw_depth[frame_idx + 1], source_points)
            diff = target_points[None, :, :3] - matched[:, None, :3]
            cost = (
                diff[..., 0] ** 2
                + diff[..., 1] ** 2
                + self.correspondence_sample_weight * diff[..., 2] ** 2
            ).astype(np.float32)
            cost_masked = np.where(ok[:, None], cost, big_cost)
            row, col = linear_sum_assignment(cost_masked)
            assigned_cost = cost_masked[row, col]
            keep = (assigned_cost <= self.correspondence_max_dist) & ok[row]
            kept_row = row[keep]
            kept_col = col[keep]
            kept_cost = assigned_cost[keep].astype(np.float32)
            conf = 1.0 / (1.0 + kept_cost / max(self.correspondence_confidence_scale, 1e-6))
            conf = conf.astype(np.float32)
            source_flat = frame_idx * point_count + kept_row
            target_flat = (frame_idx + 1) * point_count + kept_col
            corr_full_target_idx[source_flat] = target_flat
            corr_full_weight[source_flat] = conf

        return {
            "corr_full_target_idx": corr_full_target_idx,
            "corr_full_weight": corr_full_weight,
        }

    def _frame_best_matches(self, source_points, target_points, target_depth_frame):
        matched_points, valid_mask = self._match_dense_points(target_depth_frame, source_points)
        best_target = np.full(source_points.shape[0], -1, dtype=np.int64)
        confidence = np.zeros(source_points.shape[0], dtype=np.float32)

        if valid_mask.any():
            valid_indices = np.where(valid_mask)[0]
            target_dist = self._sampled_target_distance(matched_points[valid_mask], target_points)
            target_choice = np.argmin(target_dist, axis=1)
            best_dist = target_dist[np.arange(target_choice.shape[0]), target_choice]
            target_confidence = 1.0 / (1.0 + (best_dist / max(self.correspondence_confidence_scale, 1e-6)))
            target_confidence = target_confidence.astype(np.float32)
            target_confidence[best_dist > self.correspondence_max_dist] = 0.0

            best_target[valid_indices] = target_choice.astype(np.int64)
            confidence[valid_indices] = target_confidence

        return best_target, confidence

    @staticmethod
    def transform_init(phase):
        return NvidiaLoader.transform_init(phase)

if __name__ == "__main__":
    feeder = BaseFeeder(framerate=80)
    nvidia = torch.utils.data.DataLoader(
        dataset=feeder,
        batch_size=4,
        shuffle=True,
        num_workers=0,
    )
    for batch in nvidia:
        print(batch[0].shape, batch[1])
        pdb.set_trace()

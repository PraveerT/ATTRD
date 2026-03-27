import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.motion import Motion


class SimpleLinearMotion(nn.Module):
    """Reset branch-2 baseline using the same first four channels as branch 1."""

    def __init__(self, num_classes, pts_size, hidden_dims=(64, 128), dropout=0.1):
        super().__init__()
        if len(hidden_dims) != 2:
            raise ValueError("hidden_dims must contain exactly two values.")

        hidden1, hidden2 = hidden_dims
        self.num_classes = num_classes
        self.pts_size = pts_size
        self.feature_dim = hidden2 * 2

        self.encoder = nn.Sequential(
            nn.Linear(4, hidden1),
            nn.GELU(),
            nn.Linear(hidden1, hidden2),
            nn.GELU(),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.feature_dim, num_classes),
        )

    def _sample_points(self, inputs):
        points = inputs[..., :4]
        _, _, point_count, _ = points.shape
        sample_size = min(self.pts_size, point_count)

        if sample_size == point_count:
            return points

        device = points.device
        if self.training:
            indices = torch.randperm(point_count, device=device)[:sample_size]
        else:
            indices = torch.linspace(0, point_count - 1, sample_size, device=device).long()

        return points[:, :, indices, :]

    def extract_features(self, inputs):
        points = self._sample_points(inputs)
        batch_size = points.shape[0]
        encoded = self.encoder(points.reshape(batch_size, -1, 4))
        pooled_max = encoded.max(dim=1).values
        pooled_mean = encoded.mean(dim=1)
        return torch.cat((pooled_max, pooled_mean), dim=1)

    def classify_features(self, features):
        return self.classifier(features)

    def forward(self, inputs):
        features = self.extract_features(inputs)
        return self.classify_features(features)


def _knn_indices(x, k):
    if k <= 0:
        raise ValueError("edgeconv_k must be positive.")
    k = min(k, x.size(-1))
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    return pairwise_distance.topk(k=k, dim=-1)[1]


def _get_graph_feature(x, k, idx=None):
    batch_size, num_dims, num_points = x.shape
    if idx is None:
        idx = _knn_indices(x, k)

    k_eff = idx.size(-1)
    idx_base = torch.arange(batch_size, device=x.device).view(-1, 1, 1) * num_points
    flat_idx = (idx + idx_base).reshape(-1)

    points = x.transpose(2, 1).contiguous()
    feature = points.reshape(batch_size * num_points, num_dims)[flat_idx, :]
    feature = feature.view(batch_size, num_points, k_eff, num_dims)
    center = points.view(batch_size, num_points, 1, num_dims).expand(-1, -1, k_eff, -1)

    return torch.cat((feature - center, center), dim=3).permute(0, 3, 1, 2).contiguous()


def _flatten_framewise_idx(idx):
    if idx.dim() != 4:
        raise ValueError("Framewise neighborhood indices must have shape [B, T, P, K].")
    batch_size, timestep, num_points, _ = idx.shape
    frame_offsets = torch.arange(timestep, device=idx.device).view(1, timestep, 1, 1) * num_points
    return (idx.long() + frame_offsets).reshape(batch_size, timestep * num_points, -1)


class QuaternionPointLinear(nn.Module):
    """Pointwise quaternion linear transform over channel groups of four."""

    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.quat_in = (in_features + 3) // 4
        self.quat_out = (out_features + 3) // 4

        scale = 0.02
        self.weight_r = nn.Parameter(torch.randn(self.quat_out, self.quat_in) * scale)
        self.weight_i = nn.Parameter(torch.randn(self.quat_out, self.quat_in) * scale)
        self.weight_j = nn.Parameter(torch.randn(self.quat_out, self.quat_in) * scale)
        self.weight_k = nn.Parameter(torch.randn(self.quat_out, self.quat_in) * scale)
        self.bias = nn.Parameter(torch.zeros(self.quat_out * 4))

    def forward(self, x):
        batch_size, num_points, channels = x.shape
        if channels % 4 != 0:
            x = F.pad(x, (0, 4 - (channels % 4)))
            channels = x.shape[-1]

        x = x.view(batch_size, num_points, 4, channels // 4)
        x_r, x_i, x_j, x_k = x[:, :, 0], x[:, :, 1], x[:, :, 2], x[:, :, 3]

        out_r = torch.matmul(x_r, self.weight_r.t()) - torch.matmul(x_i, self.weight_i.t()) - \
                torch.matmul(x_j, self.weight_j.t()) - torch.matmul(x_k, self.weight_k.t())
        out_i = torch.matmul(x_r, self.weight_i.t()) + torch.matmul(x_i, self.weight_r.t()) + \
                torch.matmul(x_j, self.weight_k.t()) - torch.matmul(x_k, self.weight_j.t())
        out_j = torch.matmul(x_r, self.weight_j.t()) - torch.matmul(x_i, self.weight_k.t()) + \
                torch.matmul(x_j, self.weight_r.t()) + torch.matmul(x_k, self.weight_i.t())
        out_k = torch.matmul(x_r, self.weight_k.t()) + torch.matmul(x_i, self.weight_j.t()) - \
                torch.matmul(x_j, self.weight_i.t()) + torch.matmul(x_k, self.weight_r.t())

        out = torch.stack((out_r, out_i, out_j, out_k), dim=2).reshape(batch_size, num_points, -1)

        if out.shape[-1] > self.out_features:
            out = out[:, :, :self.out_features]
        elif out.shape[-1] < self.out_features:
            out = F.pad(out, (0, self.out_features - out.shape[-1]))

        return out + self.bias[:self.out_features]


def _make_quaternion_mul(kernel):
    if kernel.size(1) % 4 != 0:
        raise ValueError("Quaternion multiplication kernel expects output divisible by 4.")
    dim = kernel.size(1) // 4
    r, i, j, k = torch.split(kernel, [dim, dim, dim, dim], dim=1)
    r2 = torch.cat([r, -i, -j, -k], dim=0)
    i2 = torch.cat([i, r, -k, j], dim=0)
    j2 = torch.cat([j, k, r, -i], dim=0)
    k2 = torch.cat([k, -j, i, r], dim=0)
    return torch.cat([r2, i2, j2, k2], dim=1)


class DualQuaternionPointLinear(nn.Module):
    """Pointwise dual-quaternion transform adapted from DQGNN's dual-quaternion multiplication."""

    def __init__(self, in_features, out_features):
        super().__init__()
        if in_features % 8 != 0:
            raise ValueError("Dual-quaternion point linear expects input features divisible by 8.")
        if out_features % 8 != 0:
            raise ValueError("Dual-quaternion point linear expects output features divisible by 8.")

        self.in_features = in_features
        self.out_features = out_features
        self.dual_in = in_features // 8
        self.half_out = out_features // 2

        self.A = nn.Parameter(torch.empty(self.dual_in, self.half_out))
        self.B = nn.Parameter(torch.empty(self.dual_in, self.half_out))
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = math.sqrt(6.0 / (self.A.size(0) + self.A.size(1)))
        self.A.data.uniform_(-stdv, stdv)
        self.B.data.uniform_(-stdv, stdv)

    def forward(self, x):
        _, _, channels = x.shape
        if channels < self.in_features:
            x = F.pad(x, (0, self.in_features - channels))
        elif channels > self.in_features:
            x = x[:, :, :self.in_features]

        primary, dual = torch.split(x, self.in_features // 2, dim=-1)
        A_hamilton = _make_quaternion_mul(self.A)
        B_hamilton = _make_quaternion_mul(self.B)

        AC = torch.matmul(primary, A_hamilton)
        AD = torch.matmul(dual, A_hamilton)
        BC = torch.matmul(primary, B_hamilton)
        out = torch.cat((AC, AD + BC), dim=-1)
        return out + self.bias[:self.out_features]


class QuaternionUpdateGatedPointLinear(nn.Module):
    """Single-step quaternion update gate adapted from GatedQGNN without recurrence."""

    def __init__(self, in_features, out_features):
        super().__init__()
        self.message_proj = QuaternionPointLinear(in_features, out_features)
        self.residual_proj = QuaternionPointLinear(in_features, out_features)
        self.update_message = QuaternionPointLinear(out_features, out_features)
        self.update_residual = QuaternionPointLinear(out_features, out_features)

    def forward(self, x):
        residual = self.residual_proj(x)
        message = self.message_proj(x)
        update = torch.sigmoid(
            self.update_message(message) + self.update_residual(residual)
        )
        return message * update + residual * (1.0 - update)


def quaternion_merge(x):
    grouped = _reshape_quaternion_groups(x)
    return torch.sum(grouped * grouped, dim=2)


def quaternion_rms_merge(x, eps=1e-6):
    grouped = _reshape_quaternion_groups(x)
    return torch.sqrt(torch.mean(grouped * grouped, dim=2) + eps)


def quaternion_weighted_rms_merge(x, component_weights, eps=1e-6):
    grouped = _reshape_quaternion_groups(x)
    weight_shape = [1] * grouped.dim()
    weight_shape[2] = 4
    normalized_weights = torch.softmax(component_weights, dim=-1).view(*weight_shape)
    return torch.sqrt(torch.sum(grouped * grouped * normalized_weights, dim=2) + eps)


def _quaternion_similarity_rotate_grouped(grouped, rotation):
    if rotation.dim() == 1 and rotation.numel() == 4:
        unit_rotation = F.normalize(rotation, dim=0)
        rot_r, rot_i, rot_j, rot_k = unit_rotation.unbind()
    elif rotation.dim() == 3 and rotation.size(-1) == 4:
        unit_rotation = F.normalize(rotation, dim=-1)
        rot_r, rot_i, rot_j, rot_k = unit_rotation.unbind(dim=-1)
        rot_r = rot_r.unsqueeze(1)
        rot_i = rot_i.unsqueeze(1)
        rot_j = rot_j.unsqueeze(1)
        rot_k = rot_k.unsqueeze(1)
    else:
        raise ValueError("Quaternion similarity rotation expects shape [4] or [B, N, 4].")

    x_r = grouped[:, :, 0]
    x_i = grouped[:, :, 1]
    x_j = grouped[:, :, 2]
    x_k = grouped[:, :, 3]

    square_i = 2.0 * (rot_i * rot_i)
    square_j = 2.0 * (rot_j * rot_j)
    square_k = 2.0 * (rot_k * rot_k)

    ri = 2.0 * (rot_r * rot_i)
    rj = 2.0 * (rot_r * rot_j)
    rk = 2.0 * (rot_r * rot_k)
    ij = 2.0 * (rot_i * rot_j)
    ik = 2.0 * (rot_i * rot_k)
    jk = 2.0 * (rot_j * rot_k)

    # Proper quaternion similarity rotation q * x * q^*:
    # keep the real component and rotate only the imaginary vector part.
    out_r = x_r
    out_i = (1.0 - (square_j + square_k)) * x_i + (ij - rk) * x_j + (ik + rj) * x_k
    out_j = (ij + rk) * x_i + (1.0 - (square_i + square_k)) * x_j + (jk - ri) * x_k
    out_k = (ik - rj) * x_i + (jk + ri) * x_j + (1.0 - (square_i + square_j)) * x_k
    return torch.stack((out_r, out_i, out_j, out_k), dim=2)


def quaternion_similarity_rotated_weighted_rms_merge(x, component_weights, rotation, eps=1e-6):
    grouped = _reshape_quaternion_groups(x)
    rotated = _quaternion_similarity_rotate_grouped(grouped, rotation)
    weight_shape = [1] * rotated.dim()
    weight_shape[2] = 4
    normalized_weights = torch.softmax(component_weights, dim=-1).view(*weight_shape)
    return torch.sqrt(torch.sum(rotated * rotated * normalized_weights, dim=2) + eps)


def _reshape_quaternion_groups(x):
    if x.size(1) % 4 != 0:
        raise ValueError("Quaternion merge expects channel count divisible by 4.")
    batch_size, channels, num_points = x.shape
    return x.view(batch_size, channels // 4, 4, num_points)


class EdgeConvLinearMotion(SimpleLinearMotion):
    """Stage-1 additive branch: add a single DGCNN-style local-neighborhood block."""

    def __init__(self, num_classes, pts_size, hidden_dims=(64, 128), dropout=0.1, edgeconv_k=20):
        super().__init__(num_classes=num_classes, pts_size=pts_size, hidden_dims=hidden_dims, dropout=dropout)
        hidden1, hidden2 = hidden_dims
        self.edgeconv_k = edgeconv_k
        self.feature_dim = hidden2 * 2

        self.edgeconv = nn.Sequential(
            nn.Conv2d(8, hidden1, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden1),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.encoder = nn.Sequential(
            nn.Conv1d(hidden1, hidden2, kernel_size=1, bias=False),
            nn.BatchNorm1d(hidden2),
            nn.GELU(),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.feature_dim, num_classes),
        )

    def extract_features(self, inputs):
        points = self._sample_points(inputs)
        batch_size = points.shape[0]
        point_features = points.reshape(batch_size, -1, 4).transpose(1, 2).contiguous()

        graph_features = _get_graph_feature(point_features, k=self.edgeconv_k)
        edge_features = self.edgeconv(graph_features).max(dim=-1).values
        encoded = self.encoder(edge_features)

        pooled_max = encoded.max(dim=-1).values
        pooled_mean = encoded.mean(dim=-1)
        return torch.cat((pooled_max, pooled_mean), dim=1)


class EdgeConvQuaternionMergeMotion(EdgeConvLinearMotion):
    """Stage-1 additive branch: quaternion point mixer followed by quaternion-aware merge before pooling."""

    def __init__(self, num_classes, pts_size, hidden_dims=(64, 128), dropout=0.1, edgeconv_k=20):
        super().__init__(num_classes=num_classes, pts_size=pts_size, hidden_dims=hidden_dims, dropout=dropout, edgeconv_k=edgeconv_k)
        hidden1, hidden2 = hidden_dims
        if hidden2 % 4 != 0:
            raise ValueError("hidden_dims[1] must be divisible by 4 for quaternion merge.")

        self.quaternion_encoder = QuaternionPointLinear(hidden1, hidden2)
        self.encoder_norm = nn.BatchNorm1d(hidden2)
        self.encoder_activation = nn.GELU()
        self.merge_proj = nn.Sequential(
            nn.Conv1d(hidden2 // 4, hidden2, kernel_size=1, bias=False),
            nn.BatchNorm1d(hidden2),
            nn.GELU(),
        )

    def merge_quaternions(self, encoded):
        return quaternion_merge(encoded)

    def extract_features(self, inputs):
        points = self._sample_points(inputs)
        batch_size = points.shape[0]
        point_features = points.reshape(batch_size, -1, 4).transpose(1, 2).contiguous()

        graph_features = _get_graph_feature(point_features, k=self.edgeconv_k)
        edge_features = self.edgeconv(graph_features).max(dim=-1).values

        encoded = self.quaternion_encoder(edge_features.transpose(1, 2).contiguous())
        encoded = self.encoder_norm(encoded.transpose(1, 2).contiguous())
        encoded = self.encoder_activation(encoded)
        encoded = self.merge_proj(self.merge_quaternions(encoded))

        pooled_max = encoded.max(dim=-1).values
        pooled_mean = encoded.mean(dim=-1)
        return torch.cat((pooled_max, pooled_mean), dim=1)


class EdgeConvQuaternionRMSMergeMotion(EdgeConvQuaternionMergeMotion):
    """Winner path with RMS quaternion collapse instead of raw squared-energy collapse."""

    def __init__(self, num_classes, pts_size, hidden_dims=(64, 128), dropout=0.1, edgeconv_k=20, merge_eps=1e-6):
        super().__init__(
            num_classes=num_classes,
            pts_size=pts_size,
            hidden_dims=hidden_dims,
            dropout=dropout,
            edgeconv_k=edgeconv_k,
        )
        self.merge_eps = merge_eps

    def merge_quaternions(self, encoded):
        return quaternion_rms_merge(encoded, eps=self.merge_eps)


class EdgeConvQuaternionWeightedRMSMergeMotion(EdgeConvQuaternionRMSMergeMotion):
    """RMS winner with learnable per-component weights in the quaternion collapse."""

    def __init__(self, num_classes, pts_size, hidden_dims=(64, 128), dropout=0.1, edgeconv_k=20, merge_eps=1e-6):
        super().__init__(
            num_classes=num_classes,
            pts_size=pts_size,
            hidden_dims=hidden_dims,
            dropout=dropout,
            edgeconv_k=edgeconv_k,
            merge_eps=merge_eps,
        )
        # Start from the current RMS winner: equal weights over r, i, j, k.
        self.merge_component_logits = nn.Parameter(torch.zeros(4))

    def merge_quaternions(self, encoded):
        return quaternion_weighted_rms_merge(
            encoded,
            component_weights=self.merge_component_logits,
            eps=self.merge_eps,
        )


class EdgeConvQECCacheLocalFrameWeightedRMSMergeMotion(EdgeConvQuaternionWeightedRMSMergeMotion):
    """Weighted RMS winner adapted to cached xyz neighborhoods and local-frame quaternions."""

    def extract_features(self, inputs):
        if not isinstance(inputs, dict):
            raise ValueError("QEC cache branch expects dict inputs with points, lrf_quat, and qec_idx.")

        points = inputs['points']
        lrf_quat = inputs['lrf_quat']
        qec_idx = inputs['qec_idx']

        batch_size = points.shape[0]
        point_features = points.reshape(batch_size, -1, 4).transpose(1, 2).contiguous()
        flat_qec_idx = _flatten_framewise_idx(qec_idx)
        if flat_qec_idx.size(-1) > self.edgeconv_k:
            flat_qec_idx = flat_qec_idx[:, :, :self.edgeconv_k]

        graph_features = _get_graph_feature(point_features, k=self.edgeconv_k, idx=flat_qec_idx)
        edge_features = self.edgeconv(graph_features).max(dim=-1).values

        encoded = self.quaternion_encoder(edge_features.transpose(1, 2).contiguous())
        encoded = self.encoder_norm(encoded.transpose(1, 2).contiguous())
        encoded = self.encoder_activation(encoded)

        flat_lrf_quat = F.normalize(lrf_quat.reshape(batch_size, -1, 4).contiguous(), dim=-1)
        encoded = self.merge_proj(
            quaternion_similarity_rotated_weighted_rms_merge(
                encoded,
                component_weights=self.merge_component_logits,
                rotation=flat_lrf_quat,
                eps=self.merge_eps,
            )
        )

        pooled_max = encoded.max(dim=-1).values
        pooled_mean = encoded.mean(dim=-1)
        return torch.cat((pooled_max, pooled_mean), dim=1)


class EdgeConvQuaternionStackedWeightedRMSMergeMotion(EdgeConvQuaternionWeightedRMSMergeMotion):
    """Weighted RMS winner with one extra quaternion refinement stage before collapse."""

    def __init__(self, num_classes, pts_size, hidden_dims=(64, 128), dropout=0.1, edgeconv_k=20, merge_eps=1e-6):
        super().__init__(
            num_classes=num_classes,
            pts_size=pts_size,
            hidden_dims=hidden_dims,
            dropout=dropout,
            edgeconv_k=edgeconv_k,
            merge_eps=merge_eps,
        )
        _, hidden2 = hidden_dims
        self.quaternion_refine = QuaternionPointLinear(hidden2, hidden2)
        self.refine_norm = nn.BatchNorm1d(hidden2)
        self.refine_activation = nn.GELU()

    def extract_features(self, inputs):
        points = self._sample_points(inputs)
        batch_size = points.shape[0]
        point_features = points.reshape(batch_size, -1, 4).transpose(1, 2).contiguous()

        graph_features = _get_graph_feature(point_features, k=self.edgeconv_k)
        edge_features = self.edgeconv(graph_features).max(dim=-1).values

        encoded = self.quaternion_encoder(edge_features.transpose(1, 2).contiguous())
        encoded = self.encoder_norm(encoded.transpose(1, 2).contiguous())
        encoded = self.encoder_activation(encoded)

        refined = self.quaternion_refine(encoded.transpose(1, 2).contiguous())
        refined = self.refine_norm(refined.transpose(1, 2).contiguous())
        refined = self.refine_activation(refined)
        encoded = encoded + refined

        encoded = self.merge_proj(self.merge_quaternions(encoded))

        pooled_max = encoded.max(dim=-1).values
        pooled_mean = encoded.mean(dim=-1)
        return torch.cat((pooled_max, pooled_mean), dim=1)


class EdgeConvDualQuaternionWeightedRMSMergeMotion(EdgeConvQuaternionWeightedRMSMergeMotion):
    """Weighted RMS winner with a dual-quaternion point mixer from DQGNN-style multiplication."""

    def __init__(self, num_classes, pts_size, hidden_dims=(64, 128), dropout=0.1, edgeconv_k=20, merge_eps=1e-6):
        super().__init__(
            num_classes=num_classes,
            pts_size=pts_size,
            hidden_dims=hidden_dims,
            dropout=dropout,
            edgeconv_k=edgeconv_k,
            merge_eps=merge_eps,
        )
        hidden1, hidden2 = hidden_dims
        self.quaternion_encoder = DualQuaternionPointLinear(hidden1, hidden2)


class EdgeConvQuaternionUpdateGatedWeightedRMSMergeMotion(EdgeConvQuaternionWeightedRMSMergeMotion):
    """Weighted RMS winner with a single quaternion update gate after EdgeConv."""

    def __init__(self, num_classes, pts_size, hidden_dims=(64, 128), dropout=0.1, edgeconv_k=20, merge_eps=1e-6):
        super().__init__(
            num_classes=num_classes,
            pts_size=pts_size,
            hidden_dims=hidden_dims,
            dropout=dropout,
            edgeconv_k=edgeconv_k,
            merge_eps=merge_eps,
        )
        hidden1, hidden2 = hidden_dims
        self.quaternion_encoder = QuaternionUpdateGatedPointLinear(hidden1, hidden2)


class EdgeConvQuaternionSimilarityRotatedWeightedRMSMergeMotion(EdgeConvQuaternionWeightedRMSMergeMotion):
    """Weighted RMS winner with an Orkis-style quaternion similarity rotation before collapse."""

    def __init__(self, num_classes, pts_size, hidden_dims=(64, 128), dropout=0.1, edgeconv_k=20, merge_eps=1e-6):
        super().__init__(
            num_classes=num_classes,
            pts_size=pts_size,
            hidden_dims=hidden_dims,
            dropout=dropout,
            edgeconv_k=edgeconv_k,
            merge_eps=merge_eps,
        )
        # Identity quaternion keeps the initial behavior identical to the current winner.
        self.merge_similarity_rotation = nn.Parameter(torch.tensor([1.0, 0.0, 0.0, 0.0]))

    def merge_quaternions(self, encoded):
        return quaternion_similarity_rotated_weighted_rms_merge(
            encoded,
            component_weights=self.merge_component_logits,
            rotation=self.merge_similarity_rotation,
            eps=self.merge_eps,
        )


class MotionREQNNFusion(nn.Module):
    """Legacy fusion entrypoint, now wired to Motion + the current weighted-RMS additive branch winner."""

    def __init__(
        self,
        num_classes,
        pts_size,
        topk=16,
        downsample=(2, 2, 2),
        knn=(16, 48, 48, 24),
        spatial_hidden_dims=(64, 256),
        spatial_edgeconv_k=20,
        spatial_merge_eps=1e-6,
        spatial_dropout=0.1,
        fusion_hidden=512,
        fusion_dropout=0.3,
        **legacy_kwargs,
    ):
        super().__init__()
        if legacy_kwargs:
            # Accept older fusion configs without relying on the stale REQNN branch implementation.
            if "reqnn_k" in legacy_kwargs:
                spatial_edgeconv_k = legacy_kwargs.pop("reqnn_k")
            if "reqnn_emb_dims" in legacy_kwargs:
                spatial_hidden_dims = (spatial_hidden_dims[0], legacy_kwargs.pop("reqnn_emb_dims"))
            if "spatial_temporal_hidden" in legacy_kwargs:
                hidden2 = legacy_kwargs.pop("spatial_temporal_hidden")
                spatial_hidden_dims = (spatial_hidden_dims[0], hidden2)
            if legacy_kwargs:
                unknown = ", ".join(sorted(legacy_kwargs.keys()))
                raise TypeError(f"Unexpected fusion kwargs: {unknown}")

        self.num_classes = num_classes
        self.temporal_branch = Motion(
            num_classes=num_classes,
            pts_size=pts_size,
            topk=topk,
            downsample=downsample,
            knn=knn,
        )
        self.spatial_branch = EdgeConvQuaternionWeightedRMSMergeMotion(
            num_classes=num_classes,
            pts_size=pts_size,
            hidden_dims=tuple(spatial_hidden_dims),
            dropout=spatial_dropout,
            edgeconv_k=spatial_edgeconv_k,
            merge_eps=spatial_merge_eps,
        )
        self.feature_dim = self.temporal_branch.feature_dim + self.spatial_branch.feature_dim
        self.fusion_weight = nn.Parameter(torch.zeros(1))
        self.fusion_head = nn.Sequential(
            nn.Linear(self.feature_dim, fusion_hidden, bias=False),
            nn.BatchNorm1d(fusion_hidden),
            nn.GELU(),
            nn.Dropout(fusion_dropout),
            nn.Linear(fusion_hidden, num_classes),
        )
        self.temporal_logits = None
        self.spatial_logits = None
        self.alpha_value = 0.5

    @property
    def pts_size(self):
        return self.temporal_branch.pts_size

    @pts_size.setter
    def pts_size(self, value):
        self.temporal_branch.pts_size = value
        self.spatial_branch.pts_size = value

    def forward(self, inputs):
        temporal_features = self.temporal_branch.extract_features(inputs)
        spatial_features = self.spatial_branch.extract_features(inputs)

        self.temporal_logits = self.temporal_branch.classify_features(temporal_features)
        self.spatial_logits = self.spatial_branch.classify_features(spatial_features)

        fused_logits = self.fusion_head(torch.cat((temporal_features, spatial_features), dim=1))
        alpha = torch.sigmoid(self.fusion_weight)
        self.alpha_value = float(alpha.detach().cpu())
        return fused_logits + alpha * self.temporal_logits + (1.0 - alpha) * self.spatial_logits


MotionWeightedRMSFusion = MotionREQNNFusion


# Keep the legacy class name so older configs still resolve.
REQNNMotion = SimpleLinearMotion

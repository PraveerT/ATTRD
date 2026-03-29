import torch
import torch.nn as nn
import torch.nn.functional as F


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

    def _sample_point_indices(self, point_count, device):
        sample_size = min(self.pts_size, point_count)
        if sample_size == point_count:
            return None

        if self.training:
            return torch.randperm(point_count, device=device)[:sample_size]
        return torch.linspace(0, point_count - 1, sample_size, device=device).long()

    def _sample_points_with_aux(self, inputs, aux_input=None):
        points = inputs[..., :4]
        _, _, point_count, _ = points.shape
        indices = self._sample_point_indices(point_count, points.device)
        if indices is None:
            sampled_points = points
        else:
            sampled_points = points[:, :, indices, :]

        sampled_aux = aux_input
        if aux_input is not None:
            sampled_aux = dict(aux_input)
            orig_flat_idx = sampled_aux.get('orig_flat_idx')
            if orig_flat_idx is not None and indices is not None:
                sampled_aux['orig_flat_idx'] = orig_flat_idx[:, :, indices]

        return sampled_points, sampled_aux

    def _sample_points(self, inputs):
        sampled_points, _ = self._sample_points_with_aux(inputs)
        return sampled_points

    def _unpack_inputs(self, inputs):
        if isinstance(inputs, dict):
            return inputs['points'], inputs
        return inputs, None

    def extract_features(self, inputs, aux_input=None):
        points, _ = self._sample_points_with_aux(inputs, aux_input=aux_input)
        batch_size = points.shape[0]
        encoded = self.encoder(points.reshape(batch_size, -1, 4))
        pooled_max = encoded.max(dim=1).values
        pooled_mean = encoded.mean(dim=1)
        return torch.cat((pooled_max, pooled_mean), dim=1)

    def classify_features(self, features):
        return self.classifier(features)

    def forward(self, inputs):
        points, aux_input = self._unpack_inputs(inputs)
        features = self.extract_features(points, aux_input=aux_input)
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


def quaternion_merge(x):
    grouped = _reshape_quaternion_groups(x)
    return torch.sum(grouped * grouped, dim=2)


def quaternion_rms_merge(x, eps=1e-6):
    grouped = _reshape_quaternion_groups(x)
    return torch.sqrt(torch.mean(grouped * grouped, dim=2) + eps)


def quaternion_weighted_rms_merge(x, component_weights, eps=1e-6):
    grouped = _reshape_quaternion_groups(x)
    if component_weights.dim() == 1:
        normalized_weights = torch.softmax(component_weights, dim=-1).view(1, 1, 4, 1)
    elif component_weights.dim() == 2:
        normalized_weights = torch.softmax(component_weights, dim=-1).unsqueeze(1).unsqueeze(-1)
    elif component_weights.dim() == 3:
        normalized_weights = torch.softmax(component_weights, dim=1).unsqueeze(1)
    else:
        raise ValueError("component_weights must have 1, 2, or 3 dimensions.")
    return torch.sqrt(torch.sum(grouped * grouped * normalized_weights, dim=2) + eps)


def quaternion_normalize(x, eps=1e-6):
    grouped = _reshape_quaternion_groups(x)
    norms = torch.sqrt(torch.sum(grouped * grouped, dim=2, keepdim=True) + eps)
    return (grouped / norms).view_as(x)


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

    def extract_features(self, inputs, aux_input=None):
        points, _ = self._sample_points_with_aux(inputs, aux_input=aux_input)
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

    def extract_features(self, inputs, aux_input=None):
        points, _ = self._sample_points_with_aux(inputs, aux_input=aux_input)
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


class EdgeConvQuaternionStackedWeightedRMSMergeMotion(EdgeConvQuaternionWeightedRMSMergeMotion):
    """Weighted RMS winner with one extra quaternion refinement stage before collapse."""

    def __init__(
        self,
        num_classes,
        pts_size,
        hidden_dims=(64, 128),
        dropout=0.1,
        edgeconv_k=20,
        merge_eps=1e-6,
    ):
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

    def extract_features(self, inputs, aux_input=None):
        points, _ = self._sample_points_with_aux(inputs, aux_input=aux_input)
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


class EdgeConvQuaternionStackedWeightedRMSAttentionReadoutMotion(EdgeConvQuaternionStackedWeightedRMSMergeMotion):
    """Stacked winner with an attention-pooled readout instead of plain mean pooling."""

    def __init__(
        self,
        num_classes,
        pts_size,
        hidden_dims=(64, 128),
        dropout=0.1,
        edgeconv_k=20,
        merge_eps=1e-6,
    ):
        super().__init__(
            num_classes=num_classes,
            pts_size=pts_size,
            hidden_dims=hidden_dims,
            dropout=dropout,
            edgeconv_k=edgeconv_k,
            merge_eps=merge_eps,
        )
        _, hidden2 = hidden_dims
        self.readout_attention = nn.Conv1d(hidden2, 1, kernel_size=1, bias=True)
        nn.init.zeros_(self.readout_attention.weight)
        nn.init.zeros_(self.readout_attention.bias)

    def extract_features(self, inputs, aux_input=None):
        points, _ = self._sample_points_with_aux(inputs, aux_input=aux_input)
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
        attention = torch.softmax(self.readout_attention(encoded), dim=-1)
        pooled_attn = torch.sum(encoded * attention, dim=-1)
        return torch.cat((pooled_max, pooled_attn), dim=1)


class EdgeConvQuaternionStackedDualMergeWeightedRMSAttentionReadoutMotion(
    EdgeConvQuaternionStackedWeightedRMSAttentionReadoutMotion
):
    """Winner path with a dual quaternion collapse: weighted RMS plus real-part summary."""

    def __init__(
        self,
        num_classes,
        pts_size,
        hidden_dims=(64, 128),
        dropout=0.1,
        edgeconv_k=20,
        merge_eps=1e-6,
    ):
        super().__init__(
            num_classes=num_classes,
            pts_size=pts_size,
            hidden_dims=hidden_dims,
            dropout=dropout,
            edgeconv_k=edgeconv_k,
            merge_eps=merge_eps,
        )
        _, hidden2 = hidden_dims
        self.merge_proj = nn.Sequential(
            nn.Conv1d(hidden2 // 2, hidden2, kernel_size=1, bias=False),
            nn.BatchNorm1d(hidden2),
            nn.GELU(),
        )

    def merge_quaternions(self, encoded):
        grouped = _reshape_quaternion_groups(encoded)
        merged_rms = quaternion_weighted_rms_merge(
            encoded,
            component_weights=self.merge_component_logits,
            eps=self.merge_eps,
        )
        real_part = grouped[:, :, 0, :]
        return torch.cat((merged_rms, real_part), dim=1)


# ---------------------------------------------------------------------------
# Correspondence-based auxiliary losses (Approaches 1-3)
# All inherit from the 77.18% winner and add a lightweight side loss.
# ---------------------------------------------------------------------------

def _resolve_correspondence_pairs(sampled_orig_idx, corr_full_target_idx, corr_full_weight):
    """Map sampled point indices through the full correspondence table.

    Returns (src_positions, tgt_positions, weights) indexing into the
    *sampled* feature tensor, or None when no valid pairs exist.
    """
    batch_size, num_points = sampled_orig_idx.shape
    device = sampled_orig_idx.device

    full_target_idx = corr_full_target_idx.long()
    full_weight = corr_full_weight.float()
    total_points = full_target_idx.size(1)

    valid_source = sampled_orig_idx >= 0
    safe_source = sampled_orig_idx.clamp(min=0)

    # Look up each sampled point's correspondence target (in original space)
    target_orig = torch.gather(full_target_idx, 1, safe_source)
    target_orig = torch.where(valid_source, target_orig, torch.full_like(target_orig, -1))
    weight = torch.gather(full_weight, 1, safe_source)
    weight = torch.where(valid_source, weight, torch.zeros_like(weight))

    # Build reverse lookup: original flat index -> sampled position
    orig_to_sampled = torch.full(
        (batch_size, total_points), -1, dtype=torch.long, device=device,
    )
    positions = torch.arange(num_points, device=device).unsqueeze(0).expand(batch_size, -1)
    for b in range(batch_size):
        m = valid_source[b]
        if m.any():
            orig_to_sampled[b, sampled_orig_idx[b, m].long()] = positions[b, m]

    # Map target original index -> sampled position
    safe_target_orig = target_orig.clamp(min=0)
    target_pos = torch.full_like(target_orig, -1)
    for b in range(batch_size):
        m = target_orig[b] >= 0
        if m.any():
            target_pos[b, m] = orig_to_sampled[b, safe_target_orig[b, m]]

    # Keep only pairs where both ends landed in the sampled set
    valid = valid_source & (weight > 0) & (target_orig >= 0) & (target_pos >= 0)
    if not valid.any():
        return None

    return positions, target_pos, weight, valid


class NeighborhoodEquivarianceMotion(
    EdgeConvQuaternionStackedDualMergeWeightedRMSAttentionReadoutMotion
):
    """Approach 1: Local neighborhood rotation equivariance.

    For matched point pairs across frames, compute the local rigid rotation
    from the 3D coordinates of their k-NN patches (SVD), then enforce that
    per-point quaternion features are consistent with that rotation.
    """

    def __init__(
        self,
        num_classes,
        pts_size,
        hidden_dims=(64, 128),
        dropout=0.1,
        edgeconv_k=20,
        merge_eps=1e-6,
        aux_weight=0.01,
        aux_k=10,
    ):
        super().__init__(
            num_classes=num_classes,
            pts_size=pts_size,
            hidden_dims=hidden_dims,
            dropout=dropout,
            edgeconv_k=edgeconv_k,
            merge_eps=merge_eps,
        )
        self.aux_weight = aux_weight
        self.aux_k = aux_k
        self.latest_aux_loss = None
        self.latest_aux_metrics = {}

    def get_auxiliary_loss(self):
        return self.latest_aux_loss

    def get_auxiliary_metrics(self):
        return self.latest_aux_metrics

    def _compute_local_equivariance_loss(self, encoded, knn_idx, points_3d,
                                          sampled_aux):
        """Compute local rotation equivariance loss on pre-merge features.

        encoded: (batch, channels, num_points) - quaternion features
        knn_idx: (batch, num_points, k) - k-NN indices
        points_3d: (batch, 3, num_points) - xyz coordinates
        """
        if sampled_aux is None:
            return None

        orig_flat_idx = sampled_aux.get('orig_flat_idx')
        corr_target = sampled_aux.get('corr_full_target_idx')
        corr_weight = sampled_aux.get('corr_full_weight')
        if orig_flat_idx is None or corr_target is None or corr_weight is None:
            return None

        batch_size, channels, num_points = encoded.shape
        orig_flat_idx = orig_flat_idx.reshape(batch_size, -1).long()
        if orig_flat_idx.size(1) != num_points:
            return None

        result = _resolve_correspondence_pairs(
            orig_flat_idx, corr_target, corr_weight,
        )
        if result is None:
            return None
        src_pos, tgt_pos, pair_weight, valid = result

        # Subsample pairs to keep cost bounded
        max_pairs = 128
        k = min(self.aux_k, knn_idx.size(-1))

        total_loss = encoded.new_tensor(0.0)
        total_weight = 0.0

        for b in range(batch_size):
            b_valid = valid[b]
            if not b_valid.any():
                continue
            b_src = src_pos[b, b_valid]
            b_tgt = tgt_pos[b, b_valid].clamp(min=0)
            b_w = pair_weight[b, b_valid]

            if b_src.size(0) > max_pairs:
                perm = torch.randperm(b_src.size(0), device=encoded.device)[:max_pairs]
                b_src, b_tgt, b_w = b_src[perm], b_tgt[perm], b_w[perm]

            num_pairs = b_src.size(0)

            # Gather k-NN neighbourhood 3D coords for src and tgt
            src_knn = knn_idx[b, b_src, :k]  # (pairs, k)
            tgt_knn = knn_idx[b, b_tgt, :k]  # (pairs, k)

            # 3D coordinates: (3, num_points) -> gather
            pts = points_3d[b]  # (3, num_points)
            src_center = pts[:, b_src].t()  # (pairs, 3)
            tgt_center = pts[:, b_tgt].t()

            src_nbr = pts[:, src_knn.reshape(-1)].reshape(3, num_pairs, k).permute(1, 2, 0)  # (pairs, k, 3)
            tgt_nbr = pts[:, tgt_knn.reshape(-1)].reshape(3, num_pairs, k).permute(1, 2, 0)

            # Center the patches
            src_centered = src_nbr - src_center.unsqueeze(1)
            tgt_centered = tgt_nbr - tgt_center.unsqueeze(1)

            # SVD for local rotation: R = V @ U^T from H = src^T @ tgt
            H = torch.bmm(src_centered.transpose(1, 2), tgt_centered)  # (pairs, 3, 3)
            U, S, Vh = torch.linalg.svd(H)
            R = torch.bmm(Vh.transpose(1, 2), U.transpose(1, 2))  # (pairs, 3, 3)

            # Fix reflections
            det = torch.linalg.det(R)
            reflect = det < 0
            if reflect.any():
                Vh_fix = Vh.clone()
                Vh_fix[reflect, 2, :] *= -1
                R[reflect] = torch.bmm(
                    Vh_fix[reflect].transpose(1, 2),
                    U[reflect].transpose(1, 2),
                )

            # Gather quaternion features for src and tgt
            feat = encoded[b]  # (channels, num_points)
            src_feat = feat[:, b_src].t()  # (pairs, channels)
            tgt_feat = feat[:, b_tgt].t()

            # Reshape to quaternion groups: (pairs, groups, 4)
            groups = channels // 4
            src_qg = src_feat.reshape(num_pairs, groups, 4)
            tgt_qg = tgt_feat.reshape(num_pairs, groups, 4)

            # Apply rotation to the imaginary part (i,j,k) of each quaternion group
            # q = (r, i, j, k) -> rotated q = (r, R @ [i,j,k])
            src_real = src_qg[:, :, 0:1]  # (pairs, groups, 1)
            src_imag = src_qg[:, :, 1:]   # (pairs, groups, 3)

            # R: (pairs, 3, 3), src_imag: (pairs, groups, 3)
            # Rotate each group's imaginary part
            rotated_imag = torch.einsum('pij,pgj->pgi', R, src_imag)
            rotated_src = torch.cat([src_real, rotated_imag], dim=-1)

            # Loss: MSE between rotated source and target quaternion features
            diff = F.mse_loss(rotated_src, tgt_qg.detach(), reduction='none').mean(dim=(1, 2))
            pair_loss = (diff * b_w).sum()
            total_loss = total_loss + pair_loss
            total_weight += b_w.sum().item()

        if total_weight < 1e-6:
            return None

        loss = total_loss / max(total_weight, 1e-6)
        self.latest_aux_metrics = {
            'equivar_raw': loss.detach(),
        }
        return self.aux_weight * loss

    def extract_features(self, inputs, aux_input=None):
        points, sampled_aux = self._sample_points_with_aux(inputs, aux_input=aux_input)
        batch_size = points.shape[0]
        point_features = points.reshape(batch_size, -1, 4).transpose(1, 2).contiguous()

        # Keep 3D coords and k-NN indices for the equivariance loss
        points_3d = point_features[:, :3, :]  # (batch, 3, num_points)
        knn_idx = _knn_indices(point_features, self.edgeconv_k)

        graph_features = _get_graph_feature(point_features, k=self.edgeconv_k, idx=knn_idx)
        edge_features = self.edgeconv(graph_features).max(dim=-1).values

        encoded = self.quaternion_encoder(edge_features.transpose(1, 2).contiguous())
        encoded = self.encoder_norm(encoded.transpose(1, 2).contiguous())
        encoded = self.encoder_activation(encoded)

        refined = self.quaternion_refine(encoded.transpose(1, 2).contiguous())
        refined = self.refine_norm(refined.transpose(1, 2).contiguous())
        refined = self.refine_activation(refined)
        encoded = encoded + refined

        # Compute equivariance loss on pre-merge features
        self.latest_aux_loss = None
        self.latest_aux_metrics = {}
        if self.training and self.aux_weight > 0:
            self.latest_aux_loss = self._compute_local_equivariance_loss(
                encoded, knn_idx, points_3d, sampled_aux,
            )

        encoded = self.merge_proj(self.merge_quaternions(encoded))
        pooled_max = encoded.max(dim=-1).values
        attention = torch.softmax(self.readout_attention(encoded), dim=-1)
        pooled_attn = torch.sum(encoded * attention, dim=-1)
        return torch.cat((pooled_max, pooled_attn), dim=1)


class ContrastiveCorrespondenceMotion(
    EdgeConvQuaternionStackedDualMergeWeightedRMSAttentionReadoutMotion
):
    """Approach 2: Contrastive correspondence loss.

    Matched points across frames are positive pairs; random in-batch points
    are hard negatives.  InfoNCE on per-point features pre-merge.
    """

    def __init__(
        self,
        num_classes,
        pts_size,
        hidden_dims=(64, 128),
        dropout=0.1,
        edgeconv_k=20,
        merge_eps=1e-6,
        aux_weight=0.01,
        temperature=0.07,
        num_negatives=64,
        max_pairs=256,
    ):
        super().__init__(
            num_classes=num_classes,
            pts_size=pts_size,
            hidden_dims=hidden_dims,
            dropout=dropout,
            edgeconv_k=edgeconv_k,
            merge_eps=merge_eps,
        )
        self.aux_weight = aux_weight
        self.temperature = temperature
        self.num_negatives = num_negatives
        self.max_pairs = max_pairs
        self.latest_aux_loss = None
        self.latest_aux_metrics = {}

    def get_auxiliary_loss(self):
        return self.latest_aux_loss

    def get_auxiliary_metrics(self):
        return self.latest_aux_metrics

    def _compute_contrastive_loss(self, encoded, sampled_aux):
        if sampled_aux is None:
            return None

        orig_flat_idx = sampled_aux.get('orig_flat_idx')
        corr_target = sampled_aux.get('corr_full_target_idx')
        corr_weight = sampled_aux.get('corr_full_weight')
        if orig_flat_idx is None or corr_target is None or corr_weight is None:
            return None

        batch_size, channels, num_points = encoded.shape
        orig_flat_idx = orig_flat_idx.reshape(batch_size, -1).long()
        if orig_flat_idx.size(1) != num_points:
            return None

        result = _resolve_correspondence_pairs(
            orig_flat_idx, corr_target, corr_weight,
        )
        if result is None:
            return None
        src_pos, tgt_pos, pair_weight, valid = result

        # Normalize features for cosine similarity
        feat_norm = F.normalize(encoded, dim=1)  # (batch, channels, num_points)

        total_loss = encoded.new_tensor(0.0)
        total_pairs = 0

        for b in range(batch_size):
            b_valid = valid[b]
            if not b_valid.any():
                continue
            b_src = src_pos[b, b_valid]
            b_tgt = tgt_pos[b, b_valid].clamp(min=0)
            b_w = pair_weight[b, b_valid]

            if b_src.size(0) > self.max_pairs:
                perm = torch.randperm(b_src.size(0), device=encoded.device)[:self.max_pairs]
                b_src, b_tgt, b_w = b_src[perm], b_tgt[perm], b_w[perm]

            num_pairs = b_src.size(0)
            feat = feat_norm[b]  # (channels, num_points)

            # Anchors and positives
            anchors = feat[:, b_src].t()    # (pairs, channels)
            positives = feat[:, b_tgt].t()  # (pairs, channels)

            # Random negatives from all points in this sample
            neg_idx = torch.randint(0, num_points, (self.num_negatives,), device=encoded.device)
            negatives = feat[:, neg_idx].t()  # (num_neg, channels)

            # Positive similarity
            pos_sim = (anchors * positives).sum(dim=-1) / self.temperature  # (pairs,)

            # Negative similarities
            neg_sim = torch.mm(anchors, negatives.t()) / self.temperature  # (pairs, num_neg)

            # InfoNCE: -log(exp(pos) / (exp(pos) + sum(exp(neg))))
            logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)  # (pairs, 1+num_neg)
            labels = torch.zeros(num_pairs, dtype=torch.long, device=encoded.device)
            pair_loss = F.cross_entropy(logits, labels, reduction='none')

            total_loss = total_loss + (pair_loss * b_w).sum() / b_w.sum().clamp_min(1e-6)
            total_pairs += num_pairs

        if total_pairs == 0:
            return None

        loss = total_loss / batch_size
        self.latest_aux_metrics = {
            'contrastive_raw': loss.detach(),
            'contrastive_pairs': float(total_pairs),
        }
        return self.aux_weight * loss

    def extract_features(self, inputs, aux_input=None):
        points, sampled_aux = self._sample_points_with_aux(inputs, aux_input=aux_input)
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

        self.latest_aux_loss = None
        self.latest_aux_metrics = {}
        if self.training and self.aux_weight > 0:
            self.latest_aux_loss = self._compute_contrastive_loss(
                encoded, sampled_aux,
            )

        encoded = self.merge_proj(self.merge_quaternions(encoded))
        pooled_max = encoded.max(dim=-1).values
        attention = torch.softmax(self.readout_attention(encoded), dim=-1)
        pooled_attn = torch.sum(encoded * attention, dim=-1)
        return torch.cat((pooled_max, pooled_attn), dim=1)


class SO3AugEquivarianceMotion(
    EdgeConvQuaternionStackedDualMergeWeightedRMSAttentionReadoutMotion
):
    """Approach 3: SO(3) augmentation equivariance.

    During training, apply a random 3D rotation to the input point cloud
    and enforce that the post-encoder features are consistent between the
    original and rotated views.  No correspondences needed.
    """

    def __init__(
        self,
        num_classes,
        pts_size,
        hidden_dims=(64, 128),
        dropout=0.1,
        edgeconv_k=20,
        merge_eps=1e-6,
        aux_weight=0.01,
        rotation_sigma=0.3,
    ):
        super().__init__(
            num_classes=num_classes,
            pts_size=pts_size,
            hidden_dims=hidden_dims,
            dropout=dropout,
            edgeconv_k=edgeconv_k,
            merge_eps=merge_eps,
        )
        self.aux_weight = aux_weight
        self.rotation_sigma = rotation_sigma
        self.latest_aux_loss = None
        self.latest_aux_metrics = {}

    def get_auxiliary_loss(self):
        return self.latest_aux_loss

    def get_auxiliary_metrics(self):
        return self.latest_aux_metrics

    @staticmethod
    def _random_rotation_matrix(batch_size, sigma, device):
        """Small random rotation via axis-angle with Gaussian magnitude."""
        axis = torch.randn(batch_size, 3, device=device)
        axis = F.normalize(axis, dim=-1)
        angle = torch.randn(batch_size, 1, device=device) * sigma

        K = torch.zeros(batch_size, 3, 3, device=device)
        K[:, 0, 1] = -axis[:, 2]
        K[:, 0, 2] = axis[:, 1]
        K[:, 1, 0] = axis[:, 2]
        K[:, 1, 2] = -axis[:, 0]
        K[:, 2, 0] = -axis[:, 1]
        K[:, 2, 1] = axis[:, 0]

        I = torch.eye(3, device=device).unsqueeze(0)
        sin_a = torch.sin(angle).unsqueeze(-1)
        cos_a = torch.cos(angle).unsqueeze(-1)
        return I + sin_a * K + (1 - cos_a) * torch.bmm(K, K)

    def _encode_to_pre_merge(self, point_features):
        """Run the encoder pipeline up to (but not including) the merge step."""
        graph_features = _get_graph_feature(point_features, k=self.edgeconv_k)
        edge_features = self.edgeconv(graph_features).max(dim=-1).values

        encoded = self.quaternion_encoder(edge_features.transpose(1, 2).contiguous())
        encoded = self.encoder_norm(encoded.transpose(1, 2).contiguous())
        encoded = self.encoder_activation(encoded)

        refined = self.quaternion_refine(encoded.transpose(1, 2).contiguous())
        refined = self.refine_norm(refined.transpose(1, 2).contiguous())
        refined = self.refine_activation(refined)
        return encoded + refined

    def extract_features(self, inputs, aux_input=None):
        points, sampled_aux = self._sample_points_with_aux(inputs, aux_input=aux_input)
        batch_size = points.shape[0]
        point_features = points.reshape(batch_size, -1, 4).transpose(1, 2).contiguous()

        encoded = self._encode_to_pre_merge(point_features)

        self.latest_aux_loss = None
        self.latest_aux_metrics = {}

        if self.training and self.aux_weight > 0:
            # Apply random rotation to XYZ channels (first 3 of 4)
            R = self._random_rotation_matrix(
                batch_size, self.rotation_sigma, point_features.device,
            )
            rotated_xyz = torch.bmm(R, point_features[:, :3, :])
            rotated_features = torch.cat(
                [rotated_xyz, point_features[:, 3:, :]], dim=1,
            )

            with torch.no_grad():
                encoded_rot = self._encode_to_pre_merge(rotated_features)

            # Feature consistency: normalized MSE between pooled features
            orig_pooled = F.normalize(encoded.mean(dim=-1), dim=-1)
            rot_pooled = F.normalize(encoded_rot.mean(dim=-1), dim=-1)
            loss = F.mse_loss(orig_pooled, rot_pooled.detach())

            self.latest_aux_metrics = {
                'so3_equiv_raw': loss.detach(),
            }
            self.latest_aux_loss = self.aux_weight * loss

        encoded = self.merge_proj(self.merge_quaternions(encoded))
        pooled_max = encoded.max(dim=-1).values
        attention = torch.softmax(self.readout_attention(encoded), dim=-1)
        pooled_attn = torch.sum(encoded * attention, dim=-1)
        return torch.cat((pooled_max, pooled_attn), dim=1)


# Keep the legacy module alias so older imports still resolve.
REQNNMotion = SimpleLinearMotion

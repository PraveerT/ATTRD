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


class SimpleLinearThreeLayerMotion(SimpleLinearMotion):
    """Scratch baseline: pointwise 3-layer MLP with max/mean pooling."""

    def __init__(self, num_classes, pts_size, hidden_dims=(64, 128, 256), dropout=0.1):
        nn.Module.__init__(self)
        if len(hidden_dims) != 3:
            raise ValueError("hidden_dims must contain exactly three values.")

        hidden1, hidden2, hidden3 = hidden_dims
        self.num_classes = num_classes
        self.pts_size = pts_size
        self.feature_dim = hidden3 * 2
        self.point_feature_dim = hidden3

        self.encoder = nn.Sequential(
            nn.Linear(4, hidden1),
            nn.GELU(),
            nn.Linear(hidden1, hidden2),
            nn.GELU(),
            nn.Linear(hidden2, hidden3),
            nn.GELU(),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.feature_dim, num_classes),
        )

    def encode_points(self, inputs, aux_input=None):
        points, sampled_aux = self._sample_points_with_aux(inputs, aux_input=aux_input)
        batch_size = points.shape[0]
        encoded = self.encoder(points.reshape(batch_size, -1, 4))
        return encoded, sampled_aux

    def extract_features(self, inputs, aux_input=None):
        encoded, _ = self.encode_points(inputs, aux_input=aux_input)
        pooled_max = encoded.max(dim=1).values
        pooled_mean = encoded.mean(dim=1)
        return torch.cat((pooled_max, pooled_mean), dim=1)


class SimpleLinearThreeLayerQCCMotion(SimpleLinearThreeLayerMotion):
    """Scratch QCC variant on top of the 3-layer linear baseline."""

    def __init__(
        self,
        num_classes,
        pts_size,
        hidden_dims=(64, 128, 256),
        dropout=0.1,
        qcc_mode='none',
        qcc_weight=0.0,
        qcc_eps=1e-6,
    ):
        super().__init__(
            num_classes=num_classes,
            pts_size=pts_size,
            hidden_dims=hidden_dims,
            dropout=dropout,
        )
        _, _, hidden3 = hidden_dims
        self.qcc_mode = qcc_mode
        self.qcc_weight = qcc_weight
        self.qcc_eps = qcc_eps
        self.latest_aux_loss = None
        self.latest_aux_metrics = {}

        valid_qcc_modes = {'none', 'correspondence_cycle'}
        if self.qcc_mode not in valid_qcc_modes:
            raise ValueError("Unsupported qcc_mode: {}".format(self.qcc_mode))

        if self.qcc_mode == 'correspondence_cycle':
            self.cycle_decoder = nn.Sequential(
                nn.Linear(hidden3, hidden3),
                nn.GELU(),
                nn.Linear(hidden3, hidden3),
            )
            self.recycle_refine = nn.Sequential(
                nn.Linear(hidden3, hidden3),
                nn.GELU(),
            )

    def _reset_auxiliary_state(self):
        self.latest_aux_loss = None
        self.latest_aux_metrics = {}

    def _normalize_point_features(self, point_features):
        return F.normalize(point_features, dim=-1, eps=self.qcc_eps)

    def _weighted_point_loss(self, prediction, target, weights):
        point_loss = F.smooth_l1_loss(prediction, target, reduction='none').mean(dim=-1)
        weight_sum = weights.sum().clamp_min(1e-6)
        return torch.sum(point_loss * weights) / weight_sum

    def _compute_sampled_correspondence_qcc(self, latent_after, aux_input):
        sampled_orig_idx = aux_input.get('orig_flat_idx')
        full_target_idx = aux_input.get('corr_full_target_idx')
        full_weight = aux_input.get('corr_full_weight')
        if sampled_orig_idx is None or full_target_idx is None or full_weight is None:
            return None

        point_latent = latent_after
        batch_size, num_points, _ = point_latent.shape
        sampled_orig_idx = sampled_orig_idx.reshape(batch_size, -1).long()
        if sampled_orig_idx.size(1) != num_points:
            return None

        full_target_idx = full_target_idx.long()
        full_weight = full_weight.float()
        total_points = full_target_idx.size(1)

        valid_source_mask = sampled_orig_idx >= 0
        safe_source_idx = sampled_orig_idx.clamp(min=0)
        sampled_target_orig = torch.gather(full_target_idx, 1, safe_source_idx)
        sampled_target_orig = torch.where(
            valid_source_mask,
            sampled_target_orig,
            torch.full_like(sampled_target_orig, -1),
        )
        sampled_weight = torch.gather(full_weight, 1, safe_source_idx)
        sampled_weight = torch.where(
            valid_source_mask,
            sampled_weight,
            torch.zeros_like(sampled_weight),
        )

        if torch.count_nonzero(sampled_weight).item() == 0:
            return None

        sampled_positions = torch.arange(num_points, device=point_latent.device).view(1, -1).expand(batch_size, -1)
        orig_lookup = torch.full(
            (batch_size, total_points),
            -1,
            dtype=torch.long,
            device=point_latent.device,
        )
        for batch_idx in range(batch_size):
            batch_valid = valid_source_mask[batch_idx]
            if torch.count_nonzero(batch_valid).item() == 0:
                continue
            orig_lookup[batch_idx, sampled_orig_idx[batch_idx, batch_valid]] = sampled_positions[batch_idx, batch_valid]

        safe_target_orig = sampled_target_orig.clamp(min=0)
        target_sampled_idx = torch.full_like(sampled_target_orig, -1)
        for batch_idx in range(batch_size):
            batch_valid = sampled_target_orig[batch_idx] >= 0
            if torch.count_nonzero(batch_valid).item() == 0:
                continue
            target_sampled_idx[batch_idx, batch_valid] = orig_lookup[batch_idx, safe_target_orig[batch_idx, batch_valid]]

        valid_pair_mask = valid_source_mask & (sampled_weight > 0) & (sampled_target_orig >= 0) & (target_sampled_idx >= 0)
        if torch.count_nonzero(valid_pair_mask).item() == 0:
            return None

        gather_target_idx = target_sampled_idx.clamp(min=0).unsqueeze(-1).expand(-1, -1, point_latent.size(-1))
        source_points = point_latent
        target_points = torch.gather(point_latent, 1, gather_target_idx)

        predicted_target = self.cycle_decoder(source_points)
        recycled_source = predicted_target + self.recycle_refine(predicted_target)

        normalized_target = self._normalize_point_features(target_points.detach())
        normalized_source = self._normalize_point_features(source_points.detach())
        normalized_predicted_target = self._normalize_point_features(predicted_target)
        normalized_recycled_source = self._normalize_point_features(recycled_source)

        effective_weight = sampled_weight * valid_pair_mask.float()
        forward_loss = self._weighted_point_loss(normalized_predicted_target, normalized_target, effective_weight)
        backward_loss = self._weighted_point_loss(normalized_recycled_source, normalized_source, effective_weight)
        qcc_raw = 0.5 * (forward_loss + backward_loss)

        self.latest_aux_metrics = {
            'qcc_forward': forward_loss.detach(),
            'qcc_backward': backward_loss.detach(),
            'qcc_raw': qcc_raw.detach(),
            'qcc_valid_ratio': valid_pair_mask.float().mean().detach(),
        }
        return self.qcc_weight * qcc_raw

    def _compute_index_qcc(self, latent_after, aux_input):
        corr_src_idx = aux_input.get('corr_src_idx')
        corr_tgt_idx = aux_input.get('corr_tgt_idx')
        corr_weight = aux_input.get('corr_weight')
        corr_frame_count = aux_input.get('corr_frame_count')
        corr_points_per_frame = aux_input.get('corr_points_per_frame')
        if corr_src_idx is None or corr_tgt_idx is None or corr_weight is None:
            return None

        corr_weight = corr_weight.float()
        if torch.count_nonzero(corr_weight).item() == 0:
            return None

        point_latent = latent_after
        if corr_frame_count is not None and corr_points_per_frame is not None:
            expected_points = (corr_frame_count.long() * corr_points_per_frame.long()).view(-1)
            if torch.any(expected_points != point_latent.new_tensor(point_latent.size(1), dtype=torch.long)):
                return None

        max_index = point_latent.size(1) - 1
        corr_src_idx = corr_src_idx.long().clamp(0, max_index)
        corr_tgt_idx = corr_tgt_idx.long().clamp(0, max_index)

        gather_src_idx = corr_src_idx.unsqueeze(-1).expand(-1, -1, point_latent.size(-1))
        gather_tgt_idx = corr_tgt_idx.unsqueeze(-1).expand(-1, -1, point_latent.size(-1))
        source_points = torch.gather(point_latent, 1, gather_src_idx)
        target_points = torch.gather(point_latent, 1, gather_tgt_idx)

        predicted_target = self.cycle_decoder(source_points)
        recycled_source = predicted_target + self.recycle_refine(predicted_target)

        normalized_target = self._normalize_point_features(target_points.detach())
        normalized_source = self._normalize_point_features(source_points.detach())
        normalized_predicted_target = self._normalize_point_features(predicted_target)
        normalized_recycled_source = self._normalize_point_features(recycled_source)

        forward_loss = self._weighted_point_loss(normalized_predicted_target, normalized_target, corr_weight)
        backward_loss = self._weighted_point_loss(normalized_recycled_source, normalized_source, corr_weight)
        qcc_raw = 0.5 * (forward_loss + backward_loss)

        self.latest_aux_metrics = {
            'qcc_forward': forward_loss.detach(),
            'qcc_backward': backward_loss.detach(),
            'qcc_raw': qcc_raw.detach(),
            'qcc_valid_ratio': (corr_weight > 0).float().mean().detach(),
        }
        return self.qcc_weight * qcc_raw

    def _compute_qcc_loss(self, latent_after, aux_input):
        if (
            not self.training
            or self.qcc_mode == 'none'
            or self.qcc_weight <= 0.0
            or aux_input is None
        ):
            return None

        if aux_input.get('corr_full_target_idx') is not None:
            return self._compute_sampled_correspondence_qcc(latent_after, aux_input)

        return self._compute_index_qcc(latent_after, aux_input)

    def get_auxiliary_loss(self):
        return self.latest_aux_loss

    def get_auxiliary_metrics(self):
        return self.latest_aux_metrics

    def extract_features(self, inputs, aux_input=None):
        encoded, sampled_aux = self.encode_points(inputs, aux_input=aux_input)
        self._reset_auxiliary_state()
        self.latest_aux_loss = self._compute_qcc_loss(encoded, sampled_aux)

        pooled_max = encoded.max(dim=1).values
        pooled_mean = encoded.mean(dim=1)
        return torch.cat((pooled_max, pooled_mean), dim=1)


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


class EdgeConvQuaternionStackedDualMergeQCCWeightedRMSAttentionReadoutMotion(
    EdgeConvQuaternionStackedDualMergeWeightedRMSAttentionReadoutMotion
):
    """Isolated QCC experiment layered on top of the dual-merge winner."""

    def __init__(
        self,
        num_classes,
        pts_size,
        hidden_dims=(64, 128),
        dropout=0.1,
        edgeconv_k=20,
        merge_eps=1e-6,
        qcc_mode='none',
        qcc_weight=0.0,
        qcc_eps=1e-6,
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
        self.qcc_mode = qcc_mode
        self.qcc_weight = qcc_weight
        self.qcc_eps = qcc_eps
        self.latest_aux_loss = None
        self.latest_aux_metrics = {}

        valid_qcc_modes = {'none', 'correspondence_cycle'}
        if self.qcc_mode not in valid_qcc_modes:
            raise ValueError("Unsupported qcc_mode: {}".format(self.qcc_mode))

        if self.qcc_mode == 'correspondence_cycle':
            self.quaternion_cycle_decoder = QuaternionPointLinear(hidden2, hidden2)
            self.cycle_norm = nn.BatchNorm1d(hidden2)
            self.cycle_activation = nn.GELU()

    def _apply_point_block(self, point_features, layer, norm, activation):
        transformed = layer(point_features)
        transformed = norm(transformed.transpose(1, 2).contiguous()).transpose(1, 2).contiguous()
        return activation(transformed)

    def _apply_refine_point_block(self, point_features):
        return self._apply_point_block(
            point_features,
            self.quaternion_refine,
            self.refine_norm,
            self.refine_activation,
        )

    def _apply_refine_block(self, encoded):
        point_features = encoded.transpose(1, 2).contiguous()
        refined = self._apply_refine_point_block(point_features)
        return refined.transpose(1, 2).contiguous()

    def _apply_cycle_decoder(self, point_features):
        return self._apply_point_block(
            point_features,
            self.quaternion_cycle_decoder,
            self.cycle_norm,
            self.cycle_activation,
        )

    def _reset_auxiliary_state(self):
        self.latest_aux_loss = None
        self.latest_aux_metrics = {}

    def _normalize_quaternion_points(self, point_features):
        normalized = quaternion_normalize(
            point_features.transpose(1, 2).contiguous(),
            eps=self.qcc_eps,
        )
        return normalized.transpose(1, 2).contiguous()

    def _weighted_point_loss(self, prediction, target, weights):
        point_loss = F.smooth_l1_loss(prediction, target, reduction='none').mean(dim=-1)
        weight_sum = weights.sum().clamp_min(1e-6)
        return torch.sum(point_loss * weights) / weight_sum

    def _compute_sampled_correspondence_qcc(self, latent_after, aux_input):
        sampled_orig_idx = aux_input.get('orig_flat_idx')
        full_target_idx = aux_input.get('corr_full_target_idx')
        full_weight = aux_input.get('corr_full_weight')
        if sampled_orig_idx is None or full_target_idx is None or full_weight is None:
            return None

        point_latent = latent_after.transpose(1, 2).contiguous()
        batch_size, num_points, _ = point_latent.shape
        sampled_orig_idx = sampled_orig_idx.reshape(batch_size, -1).long()
        if sampled_orig_idx.size(1) != num_points:
            return None

        full_target_idx = full_target_idx.long()
        full_weight = full_weight.float()
        total_points = full_target_idx.size(1)

        valid_source_mask = sampled_orig_idx >= 0
        safe_source_idx = sampled_orig_idx.clamp(min=0)
        sampled_target_orig = torch.gather(full_target_idx, 1, safe_source_idx)
        sampled_target_orig = torch.where(
            valid_source_mask,
            sampled_target_orig,
            torch.full_like(sampled_target_orig, -1),
        )
        sampled_weight = torch.gather(full_weight, 1, safe_source_idx)
        sampled_weight = torch.where(
            valid_source_mask,
            sampled_weight,
            torch.zeros_like(sampled_weight),
        )

        if torch.count_nonzero(sampled_weight).item() == 0:
            return None

        sampled_positions = torch.arange(num_points, device=point_latent.device).view(1, -1).expand(batch_size, -1)
        orig_lookup = torch.full(
            (batch_size, total_points),
            -1,
            dtype=torch.long,
            device=point_latent.device,
        )
        for batch_idx in range(batch_size):
            batch_valid = valid_source_mask[batch_idx]
            if torch.count_nonzero(batch_valid).item() == 0:
                continue
            orig_lookup[batch_idx, sampled_orig_idx[batch_idx, batch_valid]] = sampled_positions[batch_idx, batch_valid]

        safe_target_orig = sampled_target_orig.clamp(min=0)
        target_sampled_idx = torch.full_like(sampled_target_orig, -1)
        for batch_idx in range(batch_size):
            batch_valid = sampled_target_orig[batch_idx] >= 0
            if torch.count_nonzero(batch_valid).item() == 0:
                continue
            target_sampled_idx[batch_idx, batch_valid] = orig_lookup[batch_idx, safe_target_orig[batch_idx, batch_valid]]

        valid_pair_mask = valid_source_mask & (sampled_weight > 0) & (sampled_target_orig >= 0) & (target_sampled_idx >= 0)
        if torch.count_nonzero(valid_pair_mask).item() == 0:
            return None

        gather_target_idx = target_sampled_idx.clamp(min=0).unsqueeze(-1).expand(-1, -1, point_latent.size(-1))
        source_points = point_latent
        target_points = torch.gather(point_latent, 1, gather_target_idx)

        predicted_target = self._apply_cycle_decoder(source_points)
        recycled_source = predicted_target + self._apply_refine_point_block(predicted_target)

        normalized_target = self._normalize_quaternion_points(target_points.detach())
        normalized_source = self._normalize_quaternion_points(source_points.detach())
        normalized_predicted_target = self._normalize_quaternion_points(predicted_target)
        normalized_recycled_source = self._normalize_quaternion_points(recycled_source)

        effective_weight = sampled_weight * valid_pair_mask.float()
        forward_loss = self._weighted_point_loss(normalized_predicted_target, normalized_target, effective_weight)
        backward_loss = self._weighted_point_loss(normalized_recycled_source, normalized_source, effective_weight)
        qcc_raw = 0.5 * (forward_loss + backward_loss)

        self.latest_aux_metrics = {
            'qcc_forward': forward_loss.detach(),
            'qcc_backward': backward_loss.detach(),
            'qcc_raw': qcc_raw.detach(),
            'qcc_valid_ratio': valid_pair_mask.float().mean().detach(),
        }
        return self.qcc_weight * qcc_raw

    def _compute_qcc_loss(self, latent_after, aux_input):
        if (
            not self.training
            or self.qcc_mode == 'none'
            or self.qcc_weight <= 0.0
            or aux_input is None
        ):
            return None

        if aux_input.get('corr_full_target_idx') is not None:
            return self._compute_sampled_correspondence_qcc(latent_after, aux_input)

        corr_src_idx = aux_input.get('corr_src_idx')
        corr_tgt_idx = aux_input.get('corr_tgt_idx')
        corr_weight = aux_input.get('corr_weight')
        corr_frame_count = aux_input.get('corr_frame_count')
        corr_points_per_frame = aux_input.get('corr_points_per_frame')
        if corr_src_idx is None or corr_tgt_idx is None or corr_weight is None:
            return None

        corr_weight = corr_weight.float()
        if torch.count_nonzero(corr_weight).item() == 0:
            return None

        point_latent = latent_after.transpose(1, 2).contiguous()
        if corr_frame_count is not None and corr_points_per_frame is not None:
            expected_points = (corr_frame_count.long() * corr_points_per_frame.long()).view(-1)
            if torch.any(expected_points != point_latent.new_tensor(point_latent.size(1), dtype=torch.long)):
                return None

        max_index = point_latent.size(1) - 1
        corr_src_idx = corr_src_idx.long().clamp(0, max_index)
        corr_tgt_idx = corr_tgt_idx.long().clamp(0, max_index)

        gather_src_idx = corr_src_idx.unsqueeze(-1).expand(-1, -1, point_latent.size(-1))
        gather_tgt_idx = corr_tgt_idx.unsqueeze(-1).expand(-1, -1, point_latent.size(-1))
        source_points = torch.gather(point_latent, 1, gather_src_idx)
        target_points = torch.gather(point_latent, 1, gather_tgt_idx)

        predicted_target = self._apply_cycle_decoder(source_points)
        recycled_source = predicted_target + self._apply_refine_point_block(predicted_target)

        normalized_target = self._normalize_quaternion_points(target_points.detach())
        normalized_source = self._normalize_quaternion_points(source_points.detach())
        normalized_predicted_target = self._normalize_quaternion_points(predicted_target)
        normalized_recycled_source = self._normalize_quaternion_points(recycled_source)

        forward_loss = self._weighted_point_loss(normalized_predicted_target, normalized_target, corr_weight)
        backward_loss = self._weighted_point_loss(normalized_recycled_source, normalized_source, corr_weight)
        qcc_raw = 0.5 * (forward_loss + backward_loss)

        self.latest_aux_metrics = {
            'qcc_forward': forward_loss.detach(),
            'qcc_backward': backward_loss.detach(),
            'qcc_raw': qcc_raw.detach(),
            'qcc_valid_ratio': (corr_weight > 0).float().mean().detach(),
        }
        return self.qcc_weight * qcc_raw

    def get_auxiliary_loss(self):
        return self.latest_aux_loss

    def get_auxiliary_metrics(self):
        return self.latest_aux_metrics

    def _extract_qcc_latent(self, inputs, aux_input=None):
        points, sampled_aux = self._sample_points_with_aux(inputs, aux_input=aux_input)
        batch_size = points.shape[0]
        point_features = points.reshape(batch_size, -1, 4).transpose(1, 2).contiguous()

        graph_features = _get_graph_feature(point_features, k=self.edgeconv_k)
        edge_features = self.edgeconv(graph_features).max(dim=-1).values

        encoded = self.quaternion_encoder(edge_features.transpose(1, 2).contiguous())
        encoded = self.encoder_norm(encoded.transpose(1, 2).contiguous())
        encoded = self.encoder_activation(encoded)

        self._reset_auxiliary_state()
        refined = self._apply_refine_block(encoded)
        encoded = encoded + refined
        self.latest_aux_loss = self._compute_qcc_loss(encoded, sampled_aux)
        return encoded

    def extract_features(self, inputs, aux_input=None):
        encoded = self._extract_qcc_latent(inputs, aux_input=aux_input)
        encoded = self.merge_proj(self.merge_quaternions(encoded))

        pooled_max = encoded.max(dim=-1).values
        attention = torch.softmax(self.readout_attention(encoded), dim=-1)
        pooled_attn = torch.sum(encoded * attention, dim=-1)
        return torch.cat((pooled_max, pooled_attn), dim=1)


# Keep the legacy module alias so older imports still resolve.
REQNNMotion = SimpleLinearMotion

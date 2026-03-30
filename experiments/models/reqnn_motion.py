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
# Bearing-quaternion QCC: geometric rigidity feature
# ---------------------------------------------------------------------------

def _compute_bearing_qcc(points_4d, num_frames, knn_k=10):
    """Compute per-point bearing QCC score from raw XYZ across frames.

    For each point, compute the bearing quaternion (rotation from NORTH to
    the direction from bbox centroid to that point).  Then q_fwd = q_{f+1} *
    conj(q_f) captures per-point angular change between frames.  For rigid
    motion all nearby points share the same q_fwd; pairwise geodesic distance
    among k-NN q_fwd values measures deviation from rigidity.

    Args:
        points_4d: (batch, num_frames, pts_per_frame, 4) raw input
        num_frames: int
        knn_k: number of neighbors for pairwise comparison

    Returns:
        bearing_qcc: (batch, 1, num_frames * pts_per_frame) in [0, 1]
            0 = high inconsistency (deforming), 1 = consistent (rigid)
    """
    batch_size = points_4d.shape[0]
    pts_per_frame = points_4d.shape[2]
    device = points_4d.device

    xyz = points_4d[..., :3]  # (batch, num_frames, pts_per_frame, 3)

    # Centroid per frame: bbox center
    bbox_min = xyz.min(dim=2).values  # (batch, num_frames, 3)
    bbox_max = xyz.max(dim=2).values
    centroids = (bbox_min + bbox_max) / 2  # (batch, num_frames, 3)

    # Direction from centroid to each point
    directions = xyz - centroids.unsqueeze(2)  # (batch, nf, pts, 3)
    dir_norm = directions.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    directions = directions / dir_norm  # unit vectors

    # Convert directions to bearing quaternions
    # q = rotation from NORTH=[0,1,0] to direction d
    # axis = cross(north, d), angle = acos(dot(north, d))
    # w = cos(angle/2), xyz = axis * sin(angle/2)
    dot = directions[..., 1].clamp(-1 + 1e-7, 1 - 1e-7)  # dot with [0,1,0] = y component
    angle = torch.acos(dot)  # (batch, nf, pts)
    half_angle = angle / 2

    # cross([0,1,0], d) = [d_z, 0, -d_x]  (for unit north)
    axis = torch.stack([
        directions[..., 2],
        torch.zeros_like(directions[..., 0]),
        -directions[..., 0],
    ], dim=-1)  # (batch, nf, pts, 3)
    axis_norm = axis.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    axis = axis / axis_norm

    w = torch.cos(half_angle)  # (batch, nf, pts)
    sin_ha = torch.sin(half_angle)
    qx = axis[..., 0] * sin_ha
    qy = axis[..., 1] * sin_ha  # always 0
    qz = axis[..., 2] * sin_ha

    # bearing_q: (batch, nf, pts, 4) as [w, x, y, z]
    bearing_q = torch.stack([w, qx, qy, qz], dim=-1)

    # q_fwd = q_{f+1} * conj(q_f) for each point, each frame transition
    q_curr = bearing_q[:, :-1]  # (batch, nf-1, pts, 4)
    q_next = bearing_q[:, 1:]   # (batch, nf-1, pts, 4)
    # conj(q) = [w, -x, -y, -z]
    q_curr_conj = q_curr * torch.tensor([1, -1, -1, -1], device=device, dtype=q_curr.dtype)

    # Hamilton product: q_next * conj(q_curr)
    aw, ax, ay, az = q_next[..., 0], q_next[..., 1], q_next[..., 2], q_next[..., 3]
    bw, bx, by, bz = q_curr_conj[..., 0], q_curr_conj[..., 1], q_curr_conj[..., 2], q_curr_conj[..., 3]

    q_fwd_w = aw*bw - ax*bx - ay*by - az*bz
    q_fwd_x = aw*bx + ax*bw + ay*bz - az*by
    q_fwd_y = aw*by - ax*bz + ay*bw + az*bx
    q_fwd_z = aw*bz + ax*by - ay*bx + az*bw

    # q_fwd: (batch, nf-1, pts, 4)
    q_fwd = torch.stack([q_fwd_w, q_fwd_x, q_fwd_y, q_fwd_z], dim=-1)
    q_fwd = F.normalize(q_fwd, dim=-1)

    # For each point, compare its q_fwd with k-NN neighbors' q_fwd
    n_transitions = num_frames - 1

    # Per-point inconsistency score (mean geodesic distance to neighbors' q_fwd)
    inconsistency = torch.zeros(batch_size, n_transitions, pts_per_frame, device=device)

    for t in range(n_transitions):
        # Spatial k-NN from frame t
        pts_t = xyz[:, t].transpose(1, 2).contiguous()  # (batch, 3, pts)
        knn_idx = _knn_indices(pts_t, knn_k)  # (batch, pts, k)

        # Gather neighbor q_fwd
        q_fwd_t = q_fwd[:, t]  # (batch, pts, 4)
        idx_exp = knn_idx.unsqueeze(-1).expand(-1, -1, -1, 4)  # (batch, pts, k, 4)
        q_fwd_flat = q_fwd_t.unsqueeze(1).expand(-1, pts_per_frame, -1, -1)
        nbr_q_fwd = torch.gather(q_fwd_flat, 2, idx_exp)
        # nbr_q_fwd: (batch, pts, k, 4)

        # Geodesic distance: 2*arccos(|q1·q2|)
        q_center = q_fwd_t.unsqueeze(2)  # (batch, pts, 1, 4)
        dot_prod = (q_center * nbr_q_fwd).sum(dim=-1).abs()  # (batch, pts, k)
        dot_prod = dot_prod.clamp(0, 1 - 1e-7)
        geo_dist = 2 * torch.acos(dot_prod)  # (batch, pts, k)

        # Mean geodesic distance to neighbors
        inconsistency[:, t] = geo_dist.mean(dim=-1)  # (batch, pts)

    # Average inconsistency across frame transitions per point
    mean_inconsistency = inconsistency.mean(dim=1)  # (batch, pts)

    # Convert to rigidity score: high inconsistency = low rigidity
    if mean_inconsistency.max() > 0:
        scale = mean_inconsistency.median().clamp(min=1e-6)
        rigidity = torch.exp(-mean_inconsistency / scale)
    else:
        rigidity = torch.ones_like(mean_inconsistency)

    # Expand to all frames (each point gets the same score across frames)
    rigidity_expanded = rigidity.unsqueeze(1).expand(-1, num_frames, -1)
    rigidity_flat = rigidity_expanded.reshape(batch_size, 1, -1)

    return rigidity_flat


def _hamilton_product(a, b):
    """Hamilton product of two quaternion tensors (..., 4) in [w,x,y,z] format."""
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return torch.stack([
        aw*bw - ax*bx - ay*by - az*bz,
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw,
    ], dim=-1)


def _quaternion_rotate_vector(q, v):
    """Rotate 3D vectors v by unit quaternions q.

    Args:
        q: (..., 4) unit quaternions [w,x,y,z]
        v: (..., 3) vectors

    Returns:
        rotated: (..., 3) rotated vectors
    """
    # v as pure quaternion [0, vx, vy, vz]
    v_quat = torch.cat([torch.zeros_like(v[..., :1]), v], dim=-1)
    q_conj = q * torch.tensor([1, -1, -1, -1], device=q.device, dtype=q.dtype)
    rotated = _hamilton_product(_hamilton_product(q, v_quat), q_conj)
    return rotated[..., 1:]  # drop w component


class _GroundedCycleConsistency(nn.Module):
    """Grounded quaternion cycle consistency module.

    Splits encoded features into 3 temporal segments, estimates quaternion
    rotations between pairs.  Each quaternion is grounded by a reconstruction
    loss: rotating the source segment's pooled features should match the
    target segment.  The cycle constraint (q_12 * q_23 * q_31 = identity)
    adds mutual consistency on top, which is the novel signal.
    """

    def __init__(self, feat_dim):
        super().__init__()
        # Quaternion estimator: from concatenated segment summaries to [w,x,y,z]
        self.quat_head = nn.Sequential(
            nn.Linear(feat_dim * 2, feat_dim),
            nn.GELU(),
            nn.Linear(feat_dim, 4),
        )

    def _estimate_quaternion(self, src_pooled, tgt_pooled):
        """Estimate unit quaternion from pooled segment features."""
        combined = torch.cat([src_pooled, tgt_pooled], dim=-1)
        q = self.quat_head(combined)  # (batch, 4)
        return F.normalize(q, dim=-1)  # project to unit sphere

    def forward(self, encoded, num_frames, pts_per_frame, points_xyz):
        """Compute grounded cycle consistency loss.

        Args:
            encoded: (batch, feat_dim, num_points) encoder features
            num_frames: int
            pts_per_frame: int
            points_xyz: (batch, num_frames, pts_per_frame, 3) raw XYZ coords

        Returns:
            loss: scalar (reconstruction + cycle)
            metrics: dict
        """
        batch = encoded.shape[0]
        feat_dim = encoded.shape[1]

        seg_size = num_frames // 3

        # Per-point XYZ per segment: (batch, seg_size * pts, 3)
        xyz1 = points_xyz[:, :seg_size].reshape(batch, -1, 3)
        xyz2 = points_xyz[:, seg_size:2*seg_size].reshape(batch, -1, 3)
        xyz3 = points_xyz[:, 2*seg_size:3*seg_size].reshape(batch, -1, 3)

        # Pool encoded features per segment for quaternion estimation
        feat = encoded.permute(0, 2, 1).reshape(
            batch, num_frames, pts_per_frame, feat_dim,
        )
        seg1 = feat[:, :seg_size].reshape(batch, -1, feat_dim).mean(dim=1)
        seg2 = feat[:, seg_size:2*seg_size].reshape(batch, -1, feat_dim).mean(dim=1)
        seg3 = feat[:, 2*seg_size:3*seg_size].reshape(batch, -1, feat_dim).mean(dim=1)

        # Estimate quaternion rotations between segment pairs
        q_12 = self._estimate_quaternion(seg1, seg2.detach())
        q_23 = self._estimate_quaternion(seg2, seg3.detach())
        q_31 = self._estimate_quaternion(seg3, seg1.detach())

        # Reconstruction loss on per-point XYZ (centered per segment)
        # The quaternion must rotate source point cloud to match target
        recon_loss = torch.tensor(0.0, device=encoded.device)
        for src_xyz, tgt_xyz, q in [
            (xyz1, xyz2, q_12), (xyz2, xyz3, q_23), (xyz3, xyz1, q_31),
        ]:
            src_c = src_xyz - src_xyz.mean(dim=1, keepdim=True)
            tgt_c = tgt_xyz - tgt_xyz.mean(dim=1, keepdim=True)
            # q is (batch, 4), broadcast over points
            rotated = _quaternion_rotate_vector(
                q.unsqueeze(1).expand(-1, src_c.shape[1], -1), src_c,
            )
            recon_loss = recon_loss + F.mse_loss(rotated, tgt_c.detach())
        recon_loss = recon_loss / 3.0

        # Cycle loss: q_12 * q_23 * q_31 should compose to identity
        q_cycle = _hamilton_product(_hamilton_product(q_12, q_23), q_31)
        q_id = torch.tensor([1.0, 0.0, 0.0, 0.0], device=encoded.device)
        loss_pos = ((q_cycle - q_id) ** 2).sum(dim=-1)
        loss_neg = ((q_cycle + q_id) ** 2).sum(dim=-1)
        cycle_loss = torch.min(loss_pos, loss_neg).mean()

        total = recon_loss + cycle_loss

        metrics = {
            'cycle_raw': cycle_loss.detach(),
            'recon_raw': recon_loss.detach(),
            'q_cycle_w': q_cycle[:, 0].abs().mean().detach(),
        }
        return total, metrics


class BearingQCCFeatureMotion(
    EdgeConvQuaternionStackedDualMergeWeightedRMSAttentionReadoutMotion
):
    """Bearing-quaternion QCC feature + grounded cycle consistency loss.

    Computes per-point rigidity from geometric bearing quaternion consistency
    and uses it to modulate encoder features.  Additionally, a grounded cycle
    consistency module estimates quaternion rotations between temporal segments
    with pairwise reconstruction grounding and cycle composition constraint.
    """

    def __init__(
        self,
        num_classes,
        pts_size,
        hidden_dims=(64, 128),
        dropout=0.1,
        edgeconv_k=20,
        merge_eps=1e-6,
        so3_weight=0.0,
        rotation_sigma=0.3,
        bearing_knn_k=10,
        qcc_weight=0.1,
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
        self.so3_weight = so3_weight
        self.rotation_sigma = rotation_sigma
        self.bearing_knn_k = bearing_knn_k
        self.qcc_weight = qcc_weight
        self.latest_aux_loss = None
        self.latest_aux_metrics = {}

        # Rigidity modulation: proj maps 1-channel rigidity to hidden2 channels
        # Initialized to zero so model starts identical to baseline
        self.rigidity_proj = nn.Sequential(
            nn.Conv1d(1, hidden2, kernel_size=1, bias=True),
            nn.Tanh(),
        )
        nn.init.zeros_(self.rigidity_proj[0].weight)
        nn.init.zeros_(self.rigidity_proj[0].bias)

        # Grounded cycle consistency module
        self.cycle_module = _GroundedCycleConsistency(feat_dim=hidden2)

    def get_auxiliary_loss(self):
        return self.latest_aux_loss

    def get_auxiliary_metrics(self):
        return self.latest_aux_metrics

    @staticmethod
    def _random_rotation_matrix(batch_size, sigma, device):
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
        batch_size, num_frames, pts_per_frame, _ = points.shape

        # Compute bearing QCC on structured (batch, nf, pts, 4) BEFORE flattening
        rigidity = _compute_bearing_qcc(
            points, num_frames, knn_k=self.bearing_knn_k,
        )  # (batch, 1, num_frames * pts_per_frame)

        # Now flatten for encoder
        point_features = points.reshape(batch_size, -1, 4).transpose(1, 2).contiguous()

        encoded = self._encode_to_pre_merge(point_features)

        # Modulate features with bearing rigidity
        modulation = self.rigidity_proj(rigidity)  # (batch, hidden2, num_points)
        encoded = encoded * (1.0 + modulation)

        # Auxiliary losses
        self.latest_aux_loss = None
        self.latest_aux_metrics = {}
        if self.training:
            total_aux = torch.tensor(0.0, device=encoded.device)
            metrics = {}

            # SO(3) equivariance loss
            if self.so3_weight > 0:
                R = self._random_rotation_matrix(
                    batch_size, self.rotation_sigma, point_features.device,
                )
                rotated_xyz = torch.bmm(R, point_features[:, :3, :])
                rotated_features = torch.cat(
                    [rotated_xyz, point_features[:, 3:, :]], dim=1,
                )
                with torch.no_grad():
                    encoded_rot = self._encode_to_pre_merge(rotated_features)

                orig_pooled = F.normalize(encoded.mean(dim=-1), dim=-1)
                rot_pooled = F.normalize(encoded_rot.mean(dim=-1), dim=-1)
                so3_loss = F.mse_loss(orig_pooled, rot_pooled.detach())
                total_aux = total_aux + self.so3_weight * so3_loss
                metrics['so3_equiv_raw'] = so3_loss.detach()

            # Grounded cycle consistency loss
            if self.qcc_weight > 0:
                qcc_loss, qcc_metrics = self.cycle_module(
                    encoded, num_frames, pts_per_frame,
                    points[..., :3],  # raw XYZ for reconstruction grounding
                )
                total_aux = total_aux + self.qcc_weight * qcc_loss
                metrics.update(qcc_metrics)
                metrics['qcc_raw'] = qcc_loss.detach()
                metrics['qcc_valid_ratio'] = torch.tensor(1.0)

            metrics['rigidity_mean'] = rigidity.mean().detach()
            self.latest_aux_loss = total_aux
            self.latest_aux_metrics = metrics

        encoded = self.merge_proj(self.merge_quaternions(encoded))
        pooled_max = encoded.max(dim=-1).values
        attention = torch.softmax(self.readout_attention(encoded), dim=-1)
        pooled_attn = torch.sum(encoded * attention, dim=-1)
        return torch.cat((pooled_max, pooled_attn), dim=1)


# Keep the legacy module alias so older imports still resolve.
REQNNMotion = SimpleLinearMotion

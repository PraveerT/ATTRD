import torch
import torch.nn as nn
import torch.nn.functional as F

from models.motion import Motion


def _group_quaternion(x):
    if x.shape[0] % 3 != 0:
        raise ValueError("Quaternion tensor batch dimension must be divisible by 3.")
    batch_size = x.shape[0] // 3
    view_shape = (3, batch_size) + tuple(x.shape[1:])
    permute_order = (1, 0) + tuple(range(2, x.dim() + 1))
    return x.view(view_shape).permute(permute_order).contiguous()


def _ungroup_quaternion(x):
    permute_order = (1, 0) + tuple(range(2, x.dim()))
    return x.permute(permute_order).contiguous().view(-1, *x.shape[2:])


def reqnn_knn(x, k):
    if k <= 0:
        raise ValueError("k must be positive for REQNN kNN graph construction.")
    k = min(k, x.size(-1))
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    return pairwise_distance.topk(k=k, dim=-1)[1]


def get_quaternion_graph_feature(x, k, idx=None):
    batch_size = x.size(0) // 3
    ori_dim = x.size(1)
    num_points = x.size(2)

    merged = x.view(3, batch_size, ori_dim, num_points)
    merged = merged.permute(1, 0, 2, 3).contiguous().view(batch_size, -1, num_points)

    if idx is None:
        idx = reqnn_knn(merged, k=k)

    k_eff = idx.size(-1)
    idx_base = torch.arange(batch_size, device=x.device).view(-1, 1, 1) * num_points
    flat_idx = (idx + idx_base).reshape(-1)

    _, num_dims, _ = merged.size()
    merged_points = merged.transpose(2, 1).contiguous()
    feature = merged_points.reshape(batch_size * num_points, num_dims)[flat_idx, :]
    feature = feature.view(batch_size, num_points, k_eff, num_dims)
    center = merged_points.view(batch_size, num_points, 1, num_dims).expand(-1, -1, k_eff, -1)

    feature = feature.view(batch_size, num_points, k_eff, 3, ori_dim)
    feature = feature.permute(3, 0, 1, 2, 4).contiguous().view(-1, num_points, k_eff, ori_dim)
    center = center.view(batch_size, num_points, k_eff, 3, ori_dim)
    center = center.permute(3, 0, 1, 2, 4).contiguous().view(-1, num_points, k_eff, ori_dim)

    feature = torch.cat((feature - center, center), dim=3)
    return feature.permute(0, 3, 1, 2).contiguous()


def quaternion_batch_norm(x, eps=1e-5):
    x_grouped = _group_quaternion(x)
    stats_grouped = _group_quaternion(x.detach())
    modulus_sq = torch.sum(stats_grouped * stats_grouped, dim=1)
    coeff = torch.sqrt(modulus_sq.reshape(modulus_sq.shape[0], -1).mean(dim=1) + eps)
    coeff = coeff.view(modulus_sq.shape[0], 1, *([1] * (x_grouped.dim() - 2)))
    normalized = x_grouped / coeff
    return _ungroup_quaternion(normalized)


def quaternion_activation(x, threshold=1.0, eps=1e-6):
    x_grouped = _group_quaternion(x)
    stats_grouped = _group_quaternion(x.detach())
    modulus = torch.sqrt(torch.sum(stats_grouped * stats_grouped, dim=1) + eps)
    scale = modulus / torch.clamp(modulus, min=threshold)
    activated = x_grouped * scale.unsqueeze(1)
    return _ungroup_quaternion(activated)


def quaternion_merge(x):
    x_grouped = _group_quaternion(x)
    return torch.sum(x_grouped * x_grouped, dim=1)


def quaternion_max_pool(x):
    x_grouped = _group_quaternion(x)
    modulus = torch.sqrt(torch.sum(_group_quaternion(x.detach()) ** 2, dim=1))
    indices = modulus.max(dim=-1, keepdim=True)[1]
    gather_idx = indices.unsqueeze(1).expand(-1, 3, -1, -1, -1)
    pooled = torch.gather(x_grouped, -1, gather_idx).squeeze(-1)
    return _ungroup_quaternion(pooled)


class REQNNFrameEncoder(nn.Module):
    def __init__(self, k=20, emb_dims=256, activation_threshold=1.0):
        super().__init__()
        self.k = k
        self.emb_dims = emb_dims
        self.eps = 1e-5
        self.activation_threshold = activation_threshold

        self.conv1 = nn.Conv2d(2, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False)
        self.conv4 = nn.Conv2d(128 * 2, 256, kernel_size=1, bias=False)
        self.conv5 = nn.Conv1d(512, emb_dims, kernel_size=1, bias=False)
        self.post_merge = nn.Sequential(
            nn.Conv1d(emb_dims, emb_dims, kernel_size=1, bias=False),
            nn.BatchNorm1d(emb_dims),
            nn.GELU(),
        )
        self.feature_dim = emb_dims * 2

    @staticmethod
    def qua_trans(x):
        batch_size, _, num_points = x.shape
        return x.permute(1, 0, 2).contiguous().reshape(batch_size * 3, 1, num_points)

    def _apply_quaternion_block(self, x, conv):
        x = conv(x)
        x = quaternion_batch_norm(x, eps=self.eps)
        x = quaternion_activation(x, threshold=self.activation_threshold)
        return x

    def forward(self, xyz):
        if xyz.shape[1] != 3:
            raise ValueError("REQNNFrameEncoder expects xyz input with shape (B, 3, N).")

        x = self.qua_trans(xyz)
        current_k = min(self.k, x.size(-1))

        x = get_quaternion_graph_feature(x, k=current_k)
        x = self._apply_quaternion_block(x, self.conv1)
        x1 = quaternion_max_pool(x)

        x = get_quaternion_graph_feature(x1, k=current_k)
        x = self._apply_quaternion_block(x, self.conv2)
        x2 = quaternion_max_pool(x)

        x = get_quaternion_graph_feature(x2, k=current_k)
        x = self._apply_quaternion_block(x, self.conv3)
        x3 = quaternion_max_pool(x)

        x = get_quaternion_graph_feature(x3, k=current_k)
        x = self._apply_quaternion_block(x, self.conv4)
        x4 = quaternion_max_pool(x)

        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = self.conv5(x)
        x = quaternion_batch_norm(x, eps=self.eps)
        x = quaternion_activation(x, threshold=self.activation_threshold)
        x = quaternion_merge(x)
        x = self.post_merge(x)

        pooled_max = F.adaptive_max_pool1d(x, 1).flatten(1)
        pooled_avg = F.adaptive_avg_pool1d(x, 1).flatten(1)
        return torch.cat((pooled_max, pooled_avg), dim=1)


class REQNNMotion(nn.Module):
    def __init__(self, num_classes, pts_size, reqnn_k=20, reqnn_emb_dims=256, temporal_hidden=256, dropout=0.3):
        super().__init__()
        self.num_classes = num_classes
        self.pts_size = pts_size
        self.frame_encoder = REQNNFrameEncoder(k=reqnn_k, emb_dims=reqnn_emb_dims)
        self.temporal_model = nn.Sequential(
            nn.Conv1d(self.frame_encoder.feature_dim, temporal_hidden, kernel_size=1, bias=False),
            nn.BatchNorm1d(temporal_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(temporal_hidden, temporal_hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(temporal_hidden),
            nn.GELU(),
        )
        self.feature_dim = temporal_hidden * 2
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

    def _sample_xyz(self, inputs):
        xyz = inputs[..., :3]
        batch_size, timestep, point_count, _ = xyz.shape
        sample_size = min(self.pts_size, point_count)

        if self.training:
            scores = torch.rand(batch_size, timestep, point_count, device=xyz.device)
            indices = scores.topk(sample_size, dim=-1).indices
        else:
            indices = torch.linspace(0, point_count - 1, sample_size, device=xyz.device).long()
            indices = indices.view(1, 1, sample_size).expand(batch_size, timestep, -1)

        gather_idx = indices.unsqueeze(-1).expand(-1, -1, -1, xyz.shape[-1])
        return xyz.gather(2, gather_idx)

    def extract_features(self, inputs):
        xyz = self._sample_xyz(inputs)
        batch_size, timestep, num_points, _ = xyz.shape

        frame_xyz = xyz.reshape(batch_size * timestep, num_points, 3).permute(0, 2, 1).contiguous()
        frame_features = self.frame_encoder(frame_xyz)
        frame_features = frame_features.reshape(batch_size, timestep, -1).transpose(1, 2).contiguous()

        temporal_features = self.temporal_model(frame_features)
        pooled_max = F.adaptive_max_pool1d(temporal_features, 1).flatten(1)
        pooled_avg = F.adaptive_avg_pool1d(temporal_features, 1).flatten(1)
        return torch.cat((pooled_max, pooled_avg), dim=1)

    def classify_features(self, features):
        return self.classifier(features)

    def forward(self, inputs):
        return self.classify_features(self.extract_features(inputs))


class MotionREQNNFusion(nn.Module):
    def __init__(
        self,
        num_classes,
        pts_size,
        topk=16,
        downsample=(2, 2, 2),
        knn=(16, 48, 48, 24),
        reqnn_k=20,
        reqnn_emb_dims=256,
        spatial_temporal_hidden=256,
        spatial_dropout=0.3,
        fusion_hidden=512,
        fusion_dropout=0.3,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.temporal_branch = Motion(
            num_classes=num_classes,
            pts_size=pts_size,
            topk=topk,
            downsample=downsample,
            knn=knn,
        )
        self.spatial_branch = REQNNMotion(
            num_classes=num_classes,
            pts_size=pts_size,
            reqnn_k=reqnn_k,
            reqnn_emb_dims=reqnn_emb_dims,
            temporal_hidden=spatial_temporal_hidden,
            dropout=spatial_dropout,
        )
        self.fusion_weight = nn.Parameter(torch.zeros(1))
        self.fusion_head = nn.Sequential(
            nn.Linear(self.temporal_branch.feature_dim + self.spatial_branch.feature_dim, fusion_hidden, bias=False),
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

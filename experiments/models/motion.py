import torch
import torch.nn as nn
import torch.nn.functional as F
from models.op import MLPBlock, MotionBlock, GroupOperation

# Use installed mamba_ssm for optimal performance
from mamba_ssm.modules.mamba_simple import Mamba


class MultiScaleFeatureProcessor(nn.Module):
    """Multi-scale feature processing layer that creates diverse representations
    at different temporal scales and combines them effectively."""
    
    def __init__(self, in_channels, num_scales=4, feature_dim=32):
        super().__init__()
        self.in_channels = in_channels
        self.num_scales = num_scales
        self.feature_dim = feature_dim
        
        # Multi-scale filters for temporal feature extraction
        self.scale_filters = nn.ModuleList([
            nn.Conv2d(in_channels, feature_dim, kernel_size=(2**i, 1), 
                     stride=(2**i, 1), padding=(2**(i-1), 0))
            for i in range(1, num_scales + 1)
        ])
        
        # Feature interaction network between scales
        self.scale_interaction = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(feature_dim * 2, feature_dim, 1),
                nn.BatchNorm2d(feature_dim),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Conv2d(feature_dim, feature_dim, 1)
            ) for _ in range(num_scales - 1)
        ])
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Conv2d(feature_dim * num_scales + in_channels, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.GELU(),
            nn.Dropout(0.2)
        )
        
    
    def forward(self, x):
        # x shape: B, C, T, N
        B, _, T, N = x.shape
        
        # 1. Multi-scale feature extraction
        scale_features = [scale_filter(x) for scale_filter in self.scale_filters]
        
        # 2. Model feature interaction between scales
        interacted_features = [scale_features[0]]
        for i in range(len(scale_features) - 1):
            # Interpolate features for interaction
            source = F.interpolate(scale_features[i], size=(scale_features[i+1].shape[2], N), 
                                 mode='bilinear', align_corners=False)
            target = scale_features[i + 1]
            
            # Combine source and target
            combined = torch.cat([source, target], dim=1)
            
            # Model feature interaction
            interaction = self.scale_interaction[i](combined)
            interacted_features.append(target + interaction)
        
        # 3. Upsample each scale back to the original resolution
        all_features = [
            F.interpolate(feat, size=(T, N), mode='bilinear', align_corners=False)
            for feat in interacted_features
        ]

        # 4. Combine all feature representations
        combined_features = torch.cat(all_features, dim=1)  # (B, feature_dim * num_scales, T, N)
        
        # 5. Output projection with residual
        combined = torch.cat([x, combined_features], dim=1)
        output = self.output_proj(combined)
        
        return output + x


class QuaternionLinear(nn.Module):
    """Simplified quaternion linear transformation for rotation-equivariant features."""
    
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Quaternion components: real, i, j, k (adjusted for quaternion structure)
        quat_in = in_features // 4 if in_features % 4 == 0 else (in_features + 4 - in_features % 4) // 4
        quat_out = out_features // 4 if out_features % 4 == 0 else (out_features + 4 - out_features % 4) // 4
        
        self.weight_r = nn.Parameter(torch.randn(quat_out, quat_in) * 0.02)
        self.weight_i = nn.Parameter(torch.randn(quat_out, quat_in) * 0.02)
        self.weight_j = nn.Parameter(torch.randn(quat_out, quat_in) * 0.02)
        self.weight_k = nn.Parameter(torch.randn(quat_out, quat_in) * 0.02)
        
        # Adjust bias for quaternion output
        quat_out_total = quat_out * 4
        self.bias = nn.Parameter(torch.zeros(quat_out_total))
        
        # Add dropout for regularization
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # For simplicity, treat input as quaternion by splitting into 4 parts
        B, T, C = x.shape
        
        if C % 4 != 0:
            # Pad to make divisible by 4
            pad_size = 4 - (C % 4)
            x = F.pad(x, (0, pad_size))
            C = x.shape[2]
        
        # Split into quaternion components
        x = x.view(B, T, 4, C // 4)
        x_r, x_i, x_j, x_k = x[:, :, 0], x[:, :, 1], x[:, :, 2], x[:, :, 3]
        
        # Quaternion multiplication (simplified)
        out_r = torch.matmul(x_r, self.weight_r.t()) - torch.matmul(x_i, self.weight_i.t()) - \
                torch.matmul(x_j, self.weight_j.t()) - torch.matmul(x_k, self.weight_k.t())
                
        out_i = torch.matmul(x_r, self.weight_i.t()) + torch.matmul(x_i, self.weight_r.t()) + \
                torch.matmul(x_j, self.weight_k.t()) - torch.matmul(x_k, self.weight_j.t())
                
        out_j = torch.matmul(x_r, self.weight_j.t()) - torch.matmul(x_i, self.weight_k.t()) + \
                torch.matmul(x_j, self.weight_r.t()) + torch.matmul(x_k, self.weight_i.t())
                
        out_k = torch.matmul(x_r, self.weight_k.t()) + torch.matmul(x_i, self.weight_j.t()) - \
                torch.matmul(x_j, self.weight_i.t()) + torch.matmul(x_k, self.weight_r.t())
        
        # Stack and reshape
        out = torch.stack([out_r, out_i, out_j, out_k], dim=2)
        out = out.view(B, T, -1)
        
        # Adjust output size to match expected dimensions and add bias
        if out.shape[2] != self.out_features:
            if out.shape[2] > self.out_features:
                out = out[:, :, :self.out_features]
            else:
                pad_size = self.out_features - out.shape[2]
                out = F.pad(out, (0, pad_size))
        
        out = out + self.bias[:out.shape[2]]
        
        # Apply dropout during training
        out = self.dropout(out)
        
        return out


class MambaTemporalEncoder(nn.Module):
    """Mamba-based temporal encoder for point cloud sequences"""
    def __init__(self, in_channels, hidden_dim, output_dim=None, num_layers=2):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim if output_dim is not None else hidden_dim
        
        # Input projection with quaternion transformation
        self.input_proj = QuaternionLinear(in_channels, hidden_dim)
        
        # Mamba blocks - using direct Mamba layers instead of Block wrapper
        self.mamba_layers = nn.ModuleList([
            Mamba(
                d_model=hidden_dim,
                d_state=16,
                d_conv=4,
                expand=2,
            )
            for _ in range(num_layers)
        ])
        
        # Layer norms for each block
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.3)
        
        # Output projection with quaternion transformation
        self.output_proj = QuaternionLinear(hidden_dim, self.output_dim)
        self.final_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, x):
        # x shape: B, C, T, N
        B, C, T, N = x.shape
        
        # Reshape to B*N, T, C for temporal processing
        x = x.permute(0, 3, 2, 1).reshape(B * N, T, C)
        
        # Project to hidden dimension
        x = self.input_proj(x)
        
        # Apply Mamba layers with residual connections
        for mamba, norm in zip(self.mamba_layers, self.norms):
            residual = x
            x = norm(x)
            x = mamba(x)
            x = self.dropout(x)
            x = x + residual
            
        # Output projection and normalization
        x = self.final_norm(x)
        x = self.output_proj(x)
        
        # Reshape back to B, output_dim, T, N
        x = x.reshape(B, N, T, self.output_dim).permute(0, 3, 2, 1)
        
        return x


class Motion(nn.Module):
    def __init__(self, num_classes, pts_size, topk=16, downsample=(2, 2, 2),
                 knn=(16, 48, 48, 24), coord_channels=4, multi_scale_num_scales=4):
        super(Motion, self).__init__()
        self.coord_channels = coord_channels
        self.stage1 = MLPBlock([coord_channels, 32, 64], 2)
        self.pool1 = nn.AdaptiveMaxPool2d((None, 1))
        self.stage2 = MotionBlock([128, 128, ], 2, coord_channels)
        self.pool2 = nn.AdaptiveMaxPool2d((None, 1))
        self.stage3 = MotionBlock([256, 256, ], 2, coord_channels)
        self.pool3 = nn.AdaptiveMaxPool2d((None, 1))
        # Stage 4 removed to reduce overfitting and improve efficiency
        self.stage5 = MLPBlock([256 + coord_channels, 1024], 2)  # Updated from 512 to 260 (fea3 channels)
        self.pool5 = nn.AdaptiveMaxPool2d((1, 1))
        self.stage6 = MLPBlock([1024, num_classes], 2, with_bn=False)
        self.global_bn = nn.BatchNorm2d(1024)
        self.knn = knn
        self.pts_size = pts_size
        self.downsample = downsample
        self.num_classes = num_classes
        self.group = GroupOperation()
        # Replace LSTM with Mamba temporal encoder
        # Process features from stage3 (256 channels) with temporal modeling
        self.mamba = MambaTemporalEncoder(in_channels=256, hidden_dim=128, output_dim=256, num_layers=2)
        
        # Add Multi-scale Feature Processor layer after stage2
        self.multi_scale = MultiScaleFeatureProcessor(in_channels=(coord_channels + 64) * 2 - coord_channels, num_scales=multi_scale_num_scales, feature_dim=32)
        self.feature_dim = 1024

    def _sample_points(self, inputs):
        points = inputs.permute(0, 3, 1, 2)
        point_count = points.shape[3]
        device = points.device
        sample_size = min(self.pts_size, point_count)

        if self.training:
            # Random sampling during training for augmentation
            indices = torch.randperm(point_count, device=device)[:sample_size]
        else:
            # Deterministic sampling during testing for consistent results
            indices = torch.linspace(0, point_count - 1, sample_size, device=device).long()
        points = points[:, :, :, indices]
        return points[:, :self.coord_channels]

    def _encode_sampled_points(self, coords):
        batchsize, in_dims, timestep, pts_num = coords.shape

        # stage 1: intra-frame
        ret_array1 = self.group.group_points(distance_dim=[0, 1, 2], array1=coords, array2=coords, knn=self.knn[0],
                                             dim=3)
        ret_array1 = ret_array1.reshape(batchsize, in_dims, timestep * pts_num, -1)
        fea1 = self.pool1(self.stage1(ret_array1)).reshape(batchsize, -1, timestep, pts_num)
        fea1 = torch.cat((coords, fea1), dim=1)

        # stage 2: inter-frame, early
        in_dims = fea1.shape[1] * 2 - self.coord_channels
        pts_num //= self.downsample[0]
        ret_group_array2 = self.group.st_group_points(fea1, 3, [0, 1, 2], self.knn[1], 3, coord_dim=self.coord_channels)
        ret_array2, coords = self.select_ind(ret_group_array2, coords, batchsize, in_dims, timestep, pts_num)
        fea2 = self.pool2(self.stage2(ret_array2)).reshape(batchsize, -1, timestep, pts_num)
        fea2 = torch.cat((coords, fea2), dim=1)
        fea2 = self.multi_scale(fea2)

        # stage 3: inter-frame, middle, applying mamba in this stage
        in_dims = fea2.shape[1] * 2 - self.coord_channels
        pts_num //= self.downsample[1]
        ret_group_array3 = self.group.st_group_points(fea2, 3, [0, 1, 2], self.knn[2], 3, coord_dim=self.coord_channels)
        ret_array3, coords = self.select_ind(ret_group_array3, coords, batchsize, in_dims, timestep, pts_num)
        fea3 = self.pool3(self.stage3(ret_array3)).reshape(batchsize, -1, timestep, pts_num)
        fea3_mamba = self.mamba(fea3)
        return torch.cat((coords, fea3_mamba), dim=1)

    def extract_features(self, inputs):
        if isinstance(inputs, dict):
            inputs = inputs['points']
        coords = self._sample_points(inputs)
        fea3 = self._encode_sampled_points(coords)
        output = self.stage5(fea3)
        output = self.pool5(output)
        output = self.global_bn(output)
        return output.flatten(1)

    def classify_features(self, features):
        logits = self.stage6(features.unsqueeze(-1).unsqueeze(-1))
        return logits.view(features.shape[0], self.num_classes)

    def forward(self, inputs):
        features = self.extract_features(inputs)
        return self.classify_features(features)

    def select_ind(self, group_array, inputs, batchsize, in_dim, timestep, pts_num):
        """
        Select indices and apply them to group_array and inputs tensors.
        
        Args:
            group_array: Tensor of shape (B, C, T*P, K) - grouped points
            inputs: Tensor of shape (B, C, T, P) - input points
            batchsize: Batch size
            in_dim: Input dimension
            timestep: Number of timesteps
            pts_num: Number of points to select
            
        Returns:
            ret_group_array: Selected grouped points
            inputs: Selected input points
        """
        # Validate inputs
        if pts_num <= 0:
            raise ValueError("pts_num must be positive")
        
        # Select indices based on point weights
        ind = self.weight_select(group_array, pts_num)
        
        # Apply indices to group_array
        # Optimize tensor operations by precomputing shapes
        ind_expanded = ind.unsqueeze(1).unsqueeze(-1).expand(
            -1, group_array.shape[1], -1, -1, group_array.shape[-1])
        ret_group_array = group_array.gather(-2, ind_expanded)
        ret_group_array = ret_group_array.reshape(batchsize, in_dim, timestep * pts_num, -1)
        
        # Apply indices to inputs
        inputs = inputs.gather(-1, ind.unsqueeze(1).expand(-1, inputs.shape[1], -1, -1))
        
        return ret_group_array, inputs

    @staticmethod
    def _normalize_scores(values):
        values_min = values.min(dim=-1, keepdim=True)[0]
        values_max = values.max(dim=-1, keepdim=True)[0]
        values_range = values_max - values_min
        values_range = torch.where(values_range == 0, torch.ones_like(values_range), values_range)
        return (values - values_min) / values_range

    @staticmethod
    def weight_select(position, topk):
        """
        Select points with larger ranges based on a hybrid metric combining distance and variance.
        
        This function computes a weighted score for each point based on:
        1. Distance from origin (encourages selecting distant points)
        2. Feature variance (encourages selecting points with high variation)
        3. Spatial coverage (encourages selecting spatially diverse points)
        
        Args:
            position: Tensor of shape (B, C, T*P, K) where first 3 channels are x,y,z coordinates
            topk: Number of points to select
            
        Returns:
            idx: Indices of selected points
        """
        # Validate inputs
        if topk <= 0:
            raise ValueError("topk must be positive")
        if position.shape[1] < 3:
            raise ValueError("position tensor must have at least 3 channels for x,y,z coordinates")
            
        # Compute squared Euclidean distances for first 3 dimensions (x,y,z)
        # position[:, :3] selects x,y,z coordinates
        # **2 computes squared distances
        # sum(dim=1) sums across x,y,z dimensions -> (B, T*P, K)
        # max(dim=-1)[0] takes maximum across K neighbors -> (B, T*P)
        distances = torch.max(torch.sum(position[:, :3] ** 2, dim=1), dim=-1)[0]
        
        # Normalize distances to [0, 1] range
        normalized_distances = Motion._normalize_scores(distances)
        
        # Compute feature variance across neighbors if we have more than 3 channels
        if position.shape[1] > 3:
            # Compute variance for feature channels (channels 3 onwards)
            feature_var = torch.var(position[:, 3:], dim=-1).mean(dim=1)  # Mean variance across time
            # Normalize feature variance
            normalized_variance = Motion._normalize_scores(feature_var)
        else:
            # If no feature channels, use zeros
            normalized_variance = torch.zeros_like(normalized_distances)
        
        # Compute spatial coverage metric to encourage diversity
        # Points that are spatially isolated from other selected points are preferred
        # Simplified approach: use distance to centroid of all points as diversity measure
        if position.shape[2] > 1:  # If we have more than one point
            # Extract spatial coordinates of centroids (first neighbor for each point)
            # The shape is (B, 3, T*P) - we need to be careful with dimensions
            coords = position[:, :3, :, 0]  # (B, 3, T*P)
            
            # Compute centroid of all points for each batch
            # We need to compute centroid across the T*P dimension (dim=2)
            centroid = torch.mean(coords, dim=2, keepdim=True)  # (B, 3, 1)
            
            # Compute distance of each point to the centroid
            # Points farther from centroid are more diverse
            # coords: (B, 3, T*P), centroid: (B, 3, 1)
            diversity_measure = torch.sqrt(torch.sum((coords - centroid) ** 2, dim=1))  # (B, T*P)
            
            # Normalize diversity measure
            normalized_diversity = Motion._normalize_scores(diversity_measure)
        else:
            normalized_diversity = torch.zeros_like(normalized_distances)
        
        # Ensure all metrics have the same shape
        # All should be (B, T*P)
        if not (normalized_distances.shape == normalized_variance.shape == normalized_diversity.shape):
            # If there's a shape mismatch, fall back to just distance + variance
            weights = 0.7 * normalized_distances + 0.3 * normalized_variance
        else:
            # Combine metrics with weighted sum
            # Distance: 0.4 (coverage is still important)
            # Variance: 0.3 (as you added)
            # Spatial diversity: 0.3 (diverse point selection)
            weights = 0.4 * normalized_distances + 0.3 * normalized_variance + 0.3 * normalized_diversity
        
        # Select topk points with largest combined weights
        # Using sorted=False for better performance when order doesn't matter
        _, idx = torch.topk(weights, min(topk, weights.shape[-1]), -1, largest=True, sorted=False)
        return idx


if __name__ == '__main__':
    pass


class PMambaTopsMotion(Motion):
    """PMamba with tops field fed only into stage1. Downstream coords stay xyz+time."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Only stage1 changes: accepts 7-channel input.
        self.stage1 = MLPBlock([7, 32, 64], 2)

    def _sample_points(self, inputs):
        points = inputs.permute(0, 3, 1, 2)
        point_count = points.shape[3]
        device = points.device
        sample_size = min(self.pts_size, point_count)
        if self.training:
            indices = torch.randperm(point_count, device=device)[:sample_size]
        else:
            indices = torch.linspace(0, point_count - 1, sample_size, device=device).long()
        points = points[:, :, :, indices]
        sampled = points[:, :4]                                          # (B, 4, T, P)

        xyz = sampled[:, :3]
        centroid = xyz.mean(dim=-1, keepdim=True)
        rel = xyz - centroid
        rel_norm = rel.norm(dim=1, keepdim=True).clamp(min=1e-6)
        tops = (rel / rel_norm).detach()                                 # (B, 3, T, P)
        return torch.cat([sampled, tops], dim=1)                         # (B, 7, T, P)

    def _encode_sampled_points(self, coords7):
        # coords7: (B, 7, T, P). Preserve xyz+time-only for downstream usage.
        batchsize, _, timestep, pts_num = coords7.shape
        coords = coords7[:, :4]                                          # (B, 4, T, P)

        # stage 1: intra-frame, uses full 7-channel coords7.
        ret_array1 = self.group.group_points(
            distance_dim=[0, 1, 2], array1=coords7, array2=coords7,
            knn=self.knn[0], dim=3,
        )
        ret_array1 = ret_array1.reshape(batchsize, 7, timestep * pts_num, -1)
        fea1 = self.pool1(self.stage1(ret_array1)).reshape(batchsize, -1, timestep, pts_num)
        fea1 = torch.cat((coords, fea1), dim=1)                          # (B, 4 + 64, T, P) = (B, 68, T, P)

        # stage 2 onward: unchanged from parent (coords is xyz+time, 4 channels).
        in_dims = fea1.shape[1] * 2 - 4
        pts_num_s2 = pts_num // self.downsample[0]
        ret_group_array2 = self.group.st_group_points(fea1, 3, [0, 1, 2], self.knn[1], 3)
        ret_array2, coords = self.select_ind(
            ret_group_array2, coords, batchsize, in_dims, timestep, pts_num_s2,
        )
        fea2 = self.pool2(self.stage2(ret_array2)).reshape(batchsize, -1, timestep, pts_num_s2)
        fea2 = torch.cat((coords, fea2), dim=1)
        fea2 = self.multi_scale(fea2)

        in_dims = fea2.shape[1] * 2 - 4
        pts_num_s3 = pts_num_s2 // self.downsample[1]
        ret_group_array3 = self.group.st_group_points(fea2, 3, [0, 1, 2], self.knn[2], 3)
        ret_array3, coords = self.select_ind(
            ret_group_array3, coords, batchsize, in_dims, timestep, pts_num_s3,
        )
        fea3 = self.pool3(self.stage3(ret_array3)).reshape(batchsize, -1, timestep, pts_num_s3)
        fea3_mamba = self.mamba(fea3)
        return torch.cat((coords, fea3_mamba), dim=1)


class TemporalFirstMambaMotion(nn.Module):
    """Per-point Mamba over time first, then set pool across points.

    Requires correspondence-aware dataloader (NvidiaQuaternionQCCParityLoader)
    so point i at frame t is the same physical point at frame t+1. Without
    correspondence, per-point temporal signal is noise (stays at chance).
    """

    def __init__(self, num_classes=25, pts_size=96, in_channels=5,
                 hidden=128, mamba_layers=2, dropout=0.1, **kwargs):
        super().__init__()
        self.pts_size = pts_size
        self.in_channels = in_channels
        self.hidden = hidden

        self.input_proj = nn.Sequential(
            nn.Linear(in_channels, hidden),
            nn.GELU(),
        )
        self.temporal = MambaTemporalEncoder(
            in_channels=hidden, hidden_dim=hidden, output_dim=hidden,
            num_layers=mamba_layers,
        )
        self.spatial_mlp = nn.Sequential(
            nn.Linear(hidden * 2, hidden * 2),
            nn.GELU(),
            nn.Linear(hidden * 2, hidden * 2),
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden * 4, num_classes)

    def _correspondence_guided_sample(self, points, aux_input):
        """Sample points following correspondence chains across frames.

        Duplicated from BearingQCCFeatureMotion so this class stays standalone.
        Frame 0 sampled random (train) or uniform (eval); each next frame
        follows the correspondence target of the previous frame's sample.
        """
        batch_size, num_frames, pts_per_frame, channels = points.shape
        sample_size = min(self.pts_size, pts_per_frame)
        device = points.device

        if sample_size == pts_per_frame:
            return points

        orig_flat_idx = aux_input['orig_flat_idx']
        corr_target = aux_input['corr_full_target_idx']
        corr_weight = aux_input['corr_full_weight']
        total_pts = corr_target.shape[-1]
        raw_ppf = total_pts // num_frames

        sampled = torch.zeros(batch_size, num_frames, sample_size, channels,
                              device=device, dtype=points.dtype)

        for b in range(batch_size):
            if self.training:
                idx = torch.randperm(pts_per_frame, device=device)[:sample_size]
            else:
                idx = torch.linspace(0, pts_per_frame - 1, sample_size,
                                     device=device).long()
            sampled[b, 0] = points[b, 0, idx]
            current_prov = orig_flat_idx[b, 0, idx].long()

            for t in range(num_frames - 1):
                next_prov = orig_flat_idx[b, t + 1].long()
                reverse_map = torch.full((total_pts,), -1, dtype=torch.long, device=device)
                reverse_map[next_prov] = torch.arange(pts_per_frame, device=device)

                tgt_flat = corr_target[b, current_prov]
                tgt_w = corr_weight[b, current_prov]
                tgt_flat_safe = tgt_flat.clamp(min=0)
                tgt_frame = tgt_flat // raw_ppf
                tgt_pos = reverse_map[tgt_flat_safe]

                valid = ((tgt_flat >= 0) & (tgt_w > 0)
                         & (tgt_frame == t + 1) & (tgt_pos >= 0))

                next_idx = torch.randint(0, pts_per_frame, (sample_size,), device=device)
                next_idx[valid] = tgt_pos[valid]

                sampled[b, t + 1] = points[b, t + 1, next_idx]
                current_prov = orig_flat_idx[b, t + 1, next_idx].long()

        return sampled

    def _fallback_sample(self, inputs):
        B, T, P, C = inputs.shape
        device = inputs.device
        sample_size = min(self.pts_size, P)
        if self.training:
            indices = torch.randperm(P, device=device)[:sample_size]
        else:
            indices = torch.linspace(0, P - 1, sample_size, device=device).long()
        return inputs[:, :, indices, :]

    def _polar_input(self, inputs):
        xyz = inputs[..., :3]
        time_ch = inputs[..., 3:4]
        centroid = xyz.mean(dim=2, keepdim=True)
        rel = xyz - centroid
        mag = rel.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        direction = (rel / mag).detach()
        return torch.cat([direction, mag, time_ch], dim=-1)

    def forward(self, inputs):
        # Quaternion dataloader returns dict with correspondence aux.
        has_corr = False
        if isinstance(inputs, dict):
            aux = inputs
            points = inputs['points']
            has_corr = ('orig_flat_idx' in aux
                        and 'corr_full_target_idx' in aux
                        and 'corr_full_weight' in aux)
        else:
            points = inputs
            aux = None

        points = points.float()
        if points.dim() == 3:
            B, N, C = points.shape
            T = N // self.pts_size
            points = points.view(B, T, self.pts_size, C)

        if has_corr:
            sampled = self._correspondence_guided_sample(points[..., :4], aux)
        else:
            sampled = self._fallback_sample(points[..., :4])

        polar = self._polar_input(sampled)
        B, T, P, _ = polar.shape

        x = self.input_proj(polar)                               # (B, T, P, hidden)
        x = x.permute(0, 2, 3, 1).contiguous()                   # (B, P, hidden, T)
        x = x.view(B * P, self.hidden, T)
        x_4d = x.unsqueeze(-1)                                   # (B*P, hidden, T, 1)
        x_4d = self.temporal(x_4d)
        x = x_4d.squeeze(-1)                                     # (B*P, hidden, T)

        t_mean = x.mean(dim=-1)
        t_max = x.max(dim=-1).values
        per_point = torch.cat([t_mean, t_max], dim=-1)           # (B*P, 2h)
        per_point = self.spatial_mlp(per_point)
        per_point = per_point.view(B, P, -1)

        s_mean = per_point.mean(dim=1)
        s_max = per_point.max(dim=1).values
        feat = torch.cat([s_mean, s_max], dim=-1)                # (B, 4h)
        feat = self.dropout(feat)
        return self.classifier(feat)

class PMambaRigidityEnsemble(nn.Module):
    """Late-fusion ensemble: PMamba (point-cloud) + RigidityOnlyClassifier
    (per-frame sorted residuals). Logits fused via softmax(alpha) weighted mean
    of softmax probs. alpha starts biased toward PMamba (init logit 2.0
    -> softmax ~0.88 weight on PMamba).
    """

    def __init__(self, num_classes=25, pts_size=256,
                 knn=(32, 24, 48, 24), topk=8,
                 rigidity_dim=256, rigidity_hidden=128, rigidity_lstm_layers=2,
                 rigidity_dropout=0.3,
                 pmamba_weights=None, rigidity_weights=None,
                 freeze_pmamba=False, freeze_rigidity=False,
                 init_alpha_logit=2.0, **kwargs):
        super().__init__()
        self.pmamba = Motion(num_classes=num_classes, pts_size=pts_size,
                             knn=list(knn), topk=topk)
        from depth_branch.model import RigidityOnlyClassifier
        self.rigidity = RigidityOnlyClassifier(
            num_classes=num_classes, rigidity_dim=rigidity_dim,
            hidden=rigidity_hidden, lstm_layers=rigidity_lstm_layers,
            dropout=rigidity_dropout,
        )
        # Learnable fusion scalar (softmax of two logits).
        self.fusion_logits = nn.Parameter(torch.tensor([init_alpha_logit, 0.0]))

        if pmamba_weights:
            sd = torch.load(pmamba_weights, map_location='cpu')
            sd = sd.get('model_state_dict', sd)
            missing, unexpected = self.pmamba.load_state_dict(sd, strict=False)
            print(f"PMamba weights: missing={len(missing)} unexpected={len(unexpected)}")
        if rigidity_weights:
            sd = torch.load(rigidity_weights, map_location='cpu')
            sd = sd.get('model_state_dict', sd)
            missing, unexpected = self.rigidity.load_state_dict(sd, strict=False)
            print(f"Rigidity weights: missing={len(missing)} unexpected={len(unexpected)}")

        if freeze_pmamba:
            for p in self.pmamba.parameters(): p.requires_grad_(False)
        if freeze_rigidity:
            for p in self.rigidity.parameters(): p.requires_grad_(False)

    def forward(self, inputs):
        # Expect a tuple (pmamba_input, rigidity_tensor).
        if isinstance(inputs, (tuple, list)) and len(inputs) == 2:
            pm_in, rig = inputs
        else:
            raise ValueError("PMambaRigidityEnsemble expects (pts, rigidity) tuple")
        pm_logits = self.pmamba(pm_in)
        rig_logits = self.rigidity(rig)
        weights = torch.softmax(self.fusion_logits, dim=0)              # (2,)
        pm_p = torch.softmax(pm_logits, dim=-1)
        rg_p = torch.softmax(rig_logits, dim=-1)
        fused = weights[0] * pm_p + weights[1] * rg_p
        # Return log of fused probs so cross-entropy -> NLL works as usual.
        return torch.log(fused.clamp(min=1e-9))

class PMambaDepthEarlyFusion(nn.Module):
    """Feature-level early-fusion of Motion (PMamba) + DepthCNNLSTM (v9c-style).

    pm_feat:  (B, 1024)  = Motion.extract_features
    dpt_feat: (B, 1024)  = DepthCNNLSTM.extract_features  (lstm_hidden * 4)
    fused = concat -> MLP -> num_classes.
    """

    def __init__(self, num_classes=25, pts_size=256,
                 knn=(32, 24, 48, 24), topk=8,
                 depth_in_channels=4, depth_feat_dim=256, depth_lstm_hidden=256,
                 depth_lstm_layers=2, depth_bidir=True, depth_dropout=0.3,
                 clip_reweight_beta=1.5,
                 pmamba_weights=None, depth_weights=None,
                 freeze_pmamba=False, freeze_depth=False,
                 pmamba_feat_dim=1024, fusion_hidden=512, fusion_dropout=0.3,
                 **kwargs):
        super().__init__()
        from depth_branch.model import DepthCNNLSTM
        self.pmamba = Motion(num_classes=num_classes, pts_size=pts_size,
                             knn=list(knn), topk=topk)
        self.depth = DepthCNNLSTM(
            num_classes=num_classes, in_channels=depth_in_channels,
            feat_dim=depth_feat_dim, lstm_hidden=depth_lstm_hidden,
            lstm_layers=depth_lstm_layers, bidirectional=depth_bidir,
            dropout=depth_dropout,
            rigidity_dim=0, rigidity_aux_dim=0, clip_reweight_beta=clip_reweight_beta,
        )

        if pmamba_weights:
            sd = torch.load(pmamba_weights, map_location='cpu')
            sd = sd.get('model_state_dict', sd)
            m, u = self.pmamba.load_state_dict(sd, strict=False)
            print(f"PMamba weights: missing={len(m)} unexpected={len(u)}")
        if depth_weights:
            sd = torch.load(depth_weights, map_location='cpu')
            sd = sd.get('model_state_dict', sd)
            m, u = self.depth.load_state_dict(sd, strict=False)
            print(f"Depth weights: missing={len(m)} unexpected={len(u)}")

        if freeze_pmamba:
            for p in self.pmamba.parameters(): p.requires_grad_(False)
        if freeze_depth:
            for p in self.depth.parameters(): p.requires_grad_(False)

        mult = 2 if depth_bidir else 1
        depth_feat_out = depth_lstm_hidden * mult * 2
        total = pmamba_feat_dim + depth_feat_out
        self.fusion_head = nn.Sequential(
            nn.Linear(total, fusion_hidden),
            nn.GELU(),
            nn.Dropout(fusion_dropout),
            nn.Linear(fusion_hidden, num_classes),
        )

    def forward(self, inputs):
        # inputs: (pts, depth_tensor, rigidity_tensor) tuple
        if isinstance(inputs, (tuple, list)) and len(inputs) == 3:
            pts, depth, rig = inputs
        else:
            raise ValueError("PMambaDepthEarlyFusion expects (pts, depth, rigidity) tuple")
        pm_feat = self.pmamba.extract_features(pts)
        dp_feat = self.depth.extract_features((depth, rig))
        return self.fusion_head(torch.cat([pm_feat, dp_feat], dim=1))

class PMambaRigidityReweight(Motion):
    """PMamba Motion with v9c-clean per-clip CE reweighting.

    Accepts forward input as either a tensor (no reweighting) or a
    (pts, rigidity_tensor) tuple. Rigidity shape (B, T, K) or (B, T, P).
    Sample weight:
        clip_std_i = std(rigidity_mean_per_frame_i across frames)
        z_i = (clip_std_i - mean(clip_std)) / (std(clip_std) + eps)
        w_i = clip(softplus(beta * z_i), max=2) ; normalized to mean 1
    Stored on self.latest_sample_weights; main.py's weighted-CE path uses it.
    """

    def __init__(self, *args, clip_reweight_beta=2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.clip_reweight_beta = clip_reweight_beta
        self.latest_sample_weights = None

    def forward(self, inputs):
        rig = None
        if isinstance(inputs, (tuple, list)) and len(inputs) == 2:
            pts, rig = inputs
        else:
            pts = inputs
        if self.training and self.clip_reweight_beta != 0 and rig is not None:
            r = rig.float()
            if r.dim() == 3:
                per_frame = r.mean(dim=-1) if r.shape[-1] != 6 else r[..., 0]
            else:
                per_frame = r
            clip_std = per_frame.std(dim=-1)
            mu = clip_std.mean()
            sd = clip_std.std() + 1e-6
            z = (clip_std - mu) / sd
            w = torch.nn.functional.softplus(self.clip_reweight_beta * z)
            w = torch.clamp(w, max=2.0)
            w = w * (w.numel() / (w.sum() + 1e-8))
            self.latest_sample_weights = w
        else:
            self.latest_sample_weights = None
        return super().forward(pts)

class PMambaFlowAux(Motion):
    """Motion with auxiliary per-frame rigidity-summary prediction head.

    Input accepted as either pts tensor or (pts, rigidity_tensor) tuple.
    rigidity_tensor shape (B, T, K=6). During training we minimise
    CE + flow_aux_weight * MSE(head(per_frame_feat), rigidity_tensor).
    """

    def __init__(self, *args, flow_aux_weight=0.1, flow_aux_dim=6,
                 flow_aux_hidden=64, flow_feat_channels=260, **kwargs):
        super().__init__(*args, **kwargs)
        self.flow_aux_weight = flow_aux_weight
        self.flow_aux_dim = flow_aux_dim
        self.flow_head = nn.Sequential(
            nn.Linear(flow_feat_channels, flow_aux_hidden),
            nn.GELU(),
            nn.Linear(flow_aux_hidden, flow_aux_dim),
        )
        self.latest_aux_loss = None
        self.latest_aux_metrics = {}
        self._aux_target = None

    def extract_features(self, inputs):
        # Handle (pts, rig_target) tuple; stash target for aux loss.
        if isinstance(inputs, (tuple, list)) and len(inputs) == 2:
            pts, self._aux_target = inputs
        else:
            pts = inputs
            self._aux_target = None

        coords = self._sample_points(pts)
        fea3 = self._encode_sampled_points(coords)        # (B, 260, T, P)

        # Auxiliary head on per-frame pooled features.
        if self.training and self.flow_aux_weight > 0 and self._aux_target is not None:
            per_frame = fea3.mean(dim=-1)                 # (B, 260, T)
            pred = self.flow_head(per_frame.transpose(1, 2))  # (B, T, K)
            target = self._aux_target.float()
            # Adjust for T mismatch (shouldn't happen with framerate=32 everywhere).
            if target.shape[1] != pred.shape[1]:
                # crop or interp
                target = target[:, :pred.shape[1]]
            aux = torch.nn.functional.mse_loss(pred, target)
            self.latest_aux_loss = self.flow_aux_weight * aux
            self.latest_aux_metrics = {
                'qcc_raw': aux.detach(),
                'qcc_forward': aux.detach(),
                'qcc_backward': aux.detach(),
                'qcc_valid_ratio': torch.ones(1, device=aux.device),
            }
        else:
            self.latest_aux_loss = None
            self.latest_aux_metrics = {}

        output = self.stage5(fea3)
        output = self.pool5(output)
        output = self.global_bn(output)
        return output.flatten(1)

    def get_auxiliary_loss(self):
        return self.latest_aux_loss

    def get_auxiliary_metrics(self):
        return self.latest_aux_metrics

class MotionTops(Motion):
    """PMamba Motion + centroid-radial tops direction as extra stage1 input.

    Only stage1 sees 7-ch [xyz, tops_xyz, t]. fea1 concat uses original
    4-ch coords, so stages 2-3 are identical to vanilla PMamba.
    """

    def __init__(self, num_classes, pts_size, **kwargs):
        super().__init__(num_classes, pts_size, **kwargs)
        # Replace stage1 for 7-channel input; reinits from scratch on load.
        self.stage1 = MLPBlock([7, 32, 64], 2)

    def _encode_sampled_points(self, coords):
        batchsize, in_dims, timestep, pts_num = coords.shape

        # Tops: per-frame centroid-radial unit direction.
        xyz = coords[:, :3]                                           # (B,3,T,P)
        centroid = xyz.mean(dim=-1, keepdim=True)                     # (B,3,T,1)
        rel = xyz - centroid
        tops = rel / rel.norm(dim=1, keepdim=True).clamp(min=1e-6)    # (B,3,T,P)
        coords7 = torch.cat([xyz, tops, coords[:, 3:4]], dim=1)       # (B,7,T,P)

        # stage 1: intra-frame with 7-ch input.
        ret_array1 = self.group.group_points(
            distance_dim=[0, 1, 2], array1=coords7, array2=coords7,
            knn=self.knn[0], dim=3,
        )
        ret_array1 = ret_array1.reshape(batchsize, 7, timestep * pts_num, -1)
        fea1 = self.pool1(self.stage1(ret_array1)).reshape(
            batchsize, -1, timestep, pts_num,
        )
        fea1 = torch.cat((coords, fea1), dim=1)                       # use 4-ch coords

        # stages 2-3 identical to Motion._encode_sampled_points.
        in_dims = fea1.shape[1] * 2 - 4
        pts_num //= self.downsample[0]
        ret_group_array2 = self.group.st_group_points(
            fea1, 3, [0, 1, 2], self.knn[1], 3,
        )
        ret_array2, coords = self.select_ind(
            ret_group_array2, coords, batchsize, in_dims, timestep, pts_num,
        )
        fea2 = self.pool2(self.stage2(ret_array2)).reshape(
            batchsize, -1, timestep, pts_num,
        )
        fea2 = torch.cat((coords, fea2), dim=1)
        fea2 = self.multi_scale(fea2)

        in_dims = fea2.shape[1] * 2 - 4
        pts_num //= self.downsample[1]
        ret_group_array3 = self.group.st_group_points(
            fea2, 3, [0, 1, 2], self.knn[2], 3,
        )
        ret_array3, coords = self.select_ind(
            ret_group_array3, coords, batchsize, in_dims, timestep, pts_num,
        )
        fea3 = self.pool3(self.stage3(ret_array3)).reshape(
            batchsize, -1, timestep, pts_num,
        )
        fea3_mamba = self.mamba(fea3)
        return torch.cat((coords, fea3_mamba), dim=1)

class MotionTopsMag(Motion):
    """PMamba + tops + |rel|. 8-ch [xyz(3), tops(3), |rel|, t]."""

    def __init__(self, num_classes, pts_size, **kwargs):
        super().__init__(num_classes, pts_size, **kwargs)
        self.stage1 = MLPBlock([8, 32, 64], 2)

    def _encode_sampled_points(self, coords):
        batchsize, _, timestep, pts_num = coords.shape
        xyz = coords[:, :3]
        centroid = xyz.mean(dim=-1, keepdim=True)
        rel = xyz - centroid
        mag = rel.norm(dim=1, keepdim=True).clamp(min=1e-6)
        tops = rel / mag
        coords8 = torch.cat([xyz, tops, mag, coords[:, 3:4]], dim=1)

        ret = self.group.group_points(
            distance_dim=[0, 1, 2], array1=coords8, array2=coords8,
            knn=self.knn[0], dim=3,
        )
        ret = ret.reshape(batchsize, 8, timestep * pts_num, -1)
        fea1 = self.pool1(self.stage1(ret)).reshape(
            batchsize, -1, timestep, pts_num)
        fea1 = torch.cat((coords, fea1), dim=1)

        in_dims = fea1.shape[1] * 2 - 4
        pts_num //= self.downsample[0]
        rg2 = self.group.st_group_points(fea1, 3, [0, 1, 2], self.knn[1], 3)
        ret2, coords = self.select_ind(rg2, coords, batchsize, in_dims, timestep, pts_num)
        fea2 = self.pool2(self.stage2(ret2)).reshape(batchsize, -1, timestep, pts_num)
        fea2 = torch.cat((coords, fea2), dim=1)
        fea2 = self.multi_scale(fea2)

        in_dims = fea2.shape[1] * 2 - 4
        pts_num //= self.downsample[1]
        rg3 = self.group.st_group_points(fea2, 3, [0, 1, 2], self.knn[2], 3)
        ret3, coords = self.select_ind(rg3, coords, batchsize, in_dims, timestep, pts_num)
        fea3 = self.pool3(self.stage3(ret3)).reshape(batchsize, -1, timestep, pts_num)
        return torch.cat((coords, self.mamba(fea3)), dim=1)


class MotionTopsClip(Motion):
    """PMamba + tops from CLIP-mean centroid (stable reference)."""

    def __init__(self, num_classes, pts_size, **kwargs):
        super().__init__(num_classes, pts_size, **kwargs)
        self.stage1 = MLPBlock([7, 32, 64], 2)

    def _encode_sampled_points(self, coords):
        batchsize, _, timestep, pts_num = coords.shape
        xyz = coords[:, :3]
        # Mean across points AND frames -> single clip-level centroid per-sample.
        clip_c = xyz.mean(dim=[2, 3], keepdim=True)       # (B,3,1,1)
        rel = xyz - clip_c
        tops = rel / rel.norm(dim=1, keepdim=True).clamp(min=1e-6)
        coords7 = torch.cat([xyz, tops, coords[:, 3:4]], dim=1)

        ret = self.group.group_points(
            distance_dim=[0, 1, 2], array1=coords7, array2=coords7,
            knn=self.knn[0], dim=3,
        )
        ret = ret.reshape(batchsize, 7, timestep * pts_num, -1)
        fea1 = self.pool1(self.stage1(ret)).reshape(
            batchsize, -1, timestep, pts_num)
        fea1 = torch.cat((coords, fea1), dim=1)

        in_dims = fea1.shape[1] * 2 - 4
        pts_num //= self.downsample[0]
        rg2 = self.group.st_group_points(fea1, 3, [0, 1, 2], self.knn[1], 3)
        ret2, coords = self.select_ind(rg2, coords, batchsize, in_dims, timestep, pts_num)
        fea2 = self.pool2(self.stage2(ret2)).reshape(batchsize, -1, timestep, pts_num)
        fea2 = torch.cat((coords, fea2), dim=1)
        fea2 = self.multi_scale(fea2)

        in_dims = fea2.shape[1] * 2 - 4
        pts_num //= self.downsample[1]
        rg3 = self.group.st_group_points(fea2, 3, [0, 1, 2], self.knn[2], 3)
        ret3, coords = self.select_ind(rg3, coords, batchsize, in_dims, timestep, pts_num)
        fea3 = self.pool3(self.stage3(ret3)).reshape(batchsize, -1, timestep, pts_num)
        return torch.cat((coords, self.mamba(fea3)), dim=1)


class MotionDTops(Motion):
    """PMamba + Δtops (frame-to-frame tops delta). Captures angular velocity."""

    def __init__(self, num_classes, pts_size, **kwargs):
        super().__init__(num_classes, pts_size, **kwargs)
        self.stage1 = MLPBlock([7, 32, 64], 2)

    def _encode_sampled_points(self, coords):
        batchsize, _, timestep, pts_num = coords.shape
        xyz = coords[:, :3]
        centroid = xyz.mean(dim=-1, keepdim=True)
        rel = xyz - centroid
        tops = rel / rel.norm(dim=1, keepdim=True).clamp(min=1e-6)  # (B,3,T,P)
        # Δtops: forward diff along T, pad last with zero (no motion info).
        dtops = torch.zeros_like(tops)
        dtops[:, :, :-1] = tops[:, :, 1:] - tops[:, :, :-1]
        coords7 = torch.cat([xyz, dtops, coords[:, 3:4]], dim=1)

        ret = self.group.group_points(
            distance_dim=[0, 1, 2], array1=coords7, array2=coords7,
            knn=self.knn[0], dim=3,
        )
        ret = ret.reshape(batchsize, 7, timestep * pts_num, -1)
        fea1 = self.pool1(self.stage1(ret)).reshape(
            batchsize, -1, timestep, pts_num)
        fea1 = torch.cat((coords, fea1), dim=1)

        in_dims = fea1.shape[1] * 2 - 4
        pts_num //= self.downsample[0]
        rg2 = self.group.st_group_points(fea1, 3, [0, 1, 2], self.knn[1], 3)
        ret2, coords = self.select_ind(rg2, coords, batchsize, in_dims, timestep, pts_num)
        fea2 = self.pool2(self.stage2(ret2)).reshape(batchsize, -1, timestep, pts_num)
        fea2 = torch.cat((coords, fea2), dim=1)
        fea2 = self.multi_scale(fea2)

        in_dims = fea2.shape[1] * 2 - 4
        pts_num //= self.downsample[1]
        rg3 = self.group.st_group_points(fea2, 3, [0, 1, 2], self.knn[2], 3)
        ret3, coords = self.select_ind(rg3, coords, batchsize, in_dims, timestep, pts_num)
        fea3 = self.pool3(self.stage3(ret3)).reshape(batchsize, -1, timestep, pts_num)
        return torch.cat((coords, self.mamba(fea3)), dim=1)


class MotionTopsFull(Motion):
    """PMamba + tops + Δtops. 10-ch [xyz, tops, dtops, t]."""

    def __init__(self, num_classes, pts_size, **kwargs):
        super().__init__(num_classes, pts_size, **kwargs)
        self.stage1 = MLPBlock([10, 32, 64], 2)

    def _encode_sampled_points(self, coords):
        batchsize, _, timestep, pts_num = coords.shape
        xyz = coords[:, :3]
        centroid = xyz.mean(dim=-1, keepdim=True)
        rel = xyz - centroid
        tops = rel / rel.norm(dim=1, keepdim=True).clamp(min=1e-6)
        dtops = torch.zeros_like(tops)
        dtops[:, :, :-1] = tops[:, :, 1:] - tops[:, :, :-1]
        coords10 = torch.cat([xyz, tops, dtops, coords[:, 3:4]], dim=1)

        ret = self.group.group_points(
            distance_dim=[0, 1, 2], array1=coords10, array2=coords10,
            knn=self.knn[0], dim=3,
        )
        ret = ret.reshape(batchsize, 10, timestep * pts_num, -1)
        fea1 = self.pool1(self.stage1(ret)).reshape(
            batchsize, -1, timestep, pts_num)
        fea1 = torch.cat((coords, fea1), dim=1)

        in_dims = fea1.shape[1] * 2 - 4
        pts_num //= self.downsample[0]
        rg2 = self.group.st_group_points(fea1, 3, [0, 1, 2], self.knn[1], 3)
        ret2, coords = self.select_ind(rg2, coords, batchsize, in_dims, timestep, pts_num)
        fea2 = self.pool2(self.stage2(ret2)).reshape(batchsize, -1, timestep, pts_num)
        fea2 = torch.cat((coords, fea2), dim=1)
        fea2 = self.multi_scale(fea2)

        in_dims = fea2.shape[1] * 2 - 4
        pts_num //= self.downsample[1]
        rg3 = self.group.st_group_points(fea2, 3, [0, 1, 2], self.knn[2], 3)
        ret3, coords = self.select_ind(rg3, coords, batchsize, in_dims, timestep, pts_num)
        fea3 = self.pool3(self.stage3(ret3)).reshape(batchsize, -1, timestep, pts_num)
        return torch.cat((coords, self.mamba(fea3)), dim=1)

class MotionRigidityContrastive(Motion):
    """PMamba + InfoNCE contrastive aux on per-point rigidity residuals.

    Residuals computed on-the-fly via NN-matching Kabsch between consecutive
    frames. No preprocess, no correspondence data loader needed.
    """

    def __init__(self, num_classes, pts_size, contrast_weight=0.05,
                 contrast_temp=0.1, contrast_num_anchors=16, **kwargs):
        super().__init__(num_classes, pts_size, **kwargs)
        self.contrast_weight = contrast_weight
        self.contrast_temp = contrast_temp
        self.contrast_num_anchors = contrast_num_anchors
        self.latest_aux_loss = None
        self.latest_aux_metrics = {}

    def get_auxiliary_loss(self):
        return self.latest_aux_loss

    def get_auxiliary_metrics(self):
        return self.latest_aux_metrics

    def extract_features(self, inputs):
        coords = self._sample_points(inputs)
        batchsize, in_dims, timestep, pts_num = coords.shape

        ret_array1 = self.group.group_points(
            distance_dim=[0, 1, 2], array1=coords, array2=coords,
            knn=self.knn[0], dim=3,
        )
        ret_array1 = ret_array1.reshape(batchsize, in_dims, timestep * pts_num, -1)
        fea1_raw = self.pool1(self.stage1(ret_array1)).reshape(
            batchsize, -1, timestep, pts_num,
        )

        self.latest_aux_loss = None
        self.latest_aux_metrics = {}
        if self.training and self.contrast_weight > 0:
            with torch.no_grad():
                rigidity = self._compute_residual(coords[:, :3])  # (B, T, P)
            c_loss, c_metrics = self._contrastive_loss(fea1_raw, rigidity)
            self.latest_aux_loss = self.contrast_weight * c_loss
            c_metrics["qcc_raw"] = c_loss.detach()
            c_metrics["qcc_forward"] = c_loss.detach()
            c_metrics["qcc_backward"] = c_loss.detach()
            c_metrics["qcc_valid_ratio"] = torch.tensor(1.0, device=c_loss.device)
            self.latest_aux_metrics = c_metrics

        fea1 = torch.cat((coords, fea1_raw), dim=1)

        in_dims = fea1.shape[1] * 2 - 4
        pts_num //= self.downsample[0]
        rg2 = self.group.st_group_points(fea1, 3, [0, 1, 2], self.knn[1], 3)
        ret2, coords = self.select_ind(rg2, coords, batchsize, in_dims, timestep, pts_num)
        fea2 = self.pool2(self.stage2(ret2)).reshape(batchsize, -1, timestep, pts_num)
        fea2 = torch.cat((coords, fea2), dim=1)
        fea2 = self.multi_scale(fea2)

        in_dims = fea2.shape[1] * 2 - 4
        pts_num //= self.downsample[1]
        rg3 = self.group.st_group_points(fea2, 3, [0, 1, 2], self.knn[2], 3)
        ret3, coords = self.select_ind(rg3, coords, batchsize, in_dims, timestep, pts_num)
        fea3 = self.pool3(self.stage3(ret3)).reshape(batchsize, -1, timestep, pts_num)
        fea3_mamba = self.mamba(fea3)
        coords_fea3 = torch.cat((coords, fea3_mamba), dim=1)

        output = self.stage5(coords_fea3)
        output = self.pool5(output)
        output = self.global_bn(output)
        return output.flatten(1)

    def _compute_residual(self, xyz):
        """xyz: (B, 3, T, P). Returns per-frame-pair residual magnitude (B, T, P).

        Per frame pair (t, t+1): NN-match points via cdist argmin on centered
        xyz, Kabsch rotation, residual = |v_matched - R*u|. Last frame: copy
        previous (no pair).
        """
        B, _, T, P = xyz.shape
        device = xyz.device
        xyz_p = xyz.permute(0, 2, 3, 1)                          # (B, T, P, 3)

        rig = torch.zeros(B, T, P, device=device, dtype=xyz.dtype)
        eye = torch.eye(3, device=device, dtype=xyz.dtype).unsqueeze(0).expand(B, 3, 3)

        for t in range(T - 1):
            p = xyz_p[:, t]                                       # (B, P, 3)
            q = xyz_p[:, t + 1]
            c_p = p.mean(dim=1, keepdim=True)
            c_q = q.mean(dim=1, keepdim=True)
            u = p - c_p                                           # (B, P, 3)
            v = q - c_q

            # NN match: for each u_i, find nearest v_j in CENTERED space.
            dist = torch.cdist(u, v)                              # (B, P, P)
            nn = dist.argmin(dim=-1)                              # (B, P)
            v_m = torch.gather(v, 1, nn.unsqueeze(-1).expand(-1, -1, 3))

            H = u.transpose(-1, -2) @ v_m                         # (B, 3, 3)
            H = H + 1e-5 * eye
            try:
                U, S, Vh = torch.linalg.svd(H)
            except Exception:
                R = eye
                residual = v_m - u
                rig[:, t] = residual.norm(dim=-1)
                continue

            V = Vh.transpose(-1, -2)
            det = torch.det(V @ U.transpose(-1, -2))
            D_diag = torch.stack(
                [torch.ones_like(det), torch.ones_like(det), det], dim=-1,
            )
            D = torch.diag_embed(D_diag)
            R = V @ D @ U.transpose(-1, -2)

            u_rot = u @ R.transpose(-1, -2)                       # (B, P, 3)
            residual = v_m - u_rot
            rig[:, t] = residual.norm(dim=-1)

        rig[:, -1] = rig[:, -2]                                   # copy last
        return rig

    def _contrastive_loss(self, features, rigidity):
        """features: (B, C, T, P), rigidity: (B, T, P) scalar per-point."""
        B, C, T, P = features.shape
        feat = features.permute(0, 2, 3, 1)                       # (B, T, P, C)
        feat = F.normalize(feat, dim=-1)

        med = rigidity.median(dim=-1, keepdim=True).values
        is_rigid = rigidity <= med

        tau = self.contrast_temp
        total = feat.new_zeros(())
        count = 0
        pos_frac_sum = 0.0

        for b in range(B):
            for t in range(T):
                f = feat[b, t]
                r_mask = is_rigid[b, t]
                pos_idx = r_mask.nonzero(as_tuple=True)[0]
                neg_idx = (~r_mask).nonzero(as_tuple=True)[0]
                if pos_idx.numel() < 2 or neg_idx.numel() < 1:
                    continue

                for anchor_set, other_set in [(pos_idx, neg_idx), (neg_idx, pos_idx)]:
                    N = min(self.contrast_num_anchors, anchor_set.numel() - 1)
                    if N < 1:
                        continue
                    sel = torch.randperm(anchor_set.numel(), device=f.device)[:N]
                    a_idx = anchor_set[sel]
                    anchors = f[a_idx]
                    pos_f = f[anchor_set]
                    neg_f = f[other_set]

                    sim_pos = anchors @ pos_f.T
                    sim_neg = anchors @ neg_f.T

                    self_mask = (anchor_set.unsqueeze(0) == a_idx.unsqueeze(1))
                    sim_pos = sim_pos.masked_fill(self_mask, -1e9)

                    # Hard-negative mining: keep top-K most similar (hardest) negatives per anchor.
                    K_neg = min(self.contrast_num_anchors, sim_neg.shape[-1])
                    sim_neg_hard, _ = sim_neg.topk(K_neg, dim=-1)
                    exp_pos = (sim_pos / tau).exp()
                    exp_neg = (sim_neg_hard / tau).exp()
                    num = exp_pos.sum(dim=-1) + 1e-8
                    den = num + exp_neg.sum(dim=-1)
                    loss = -(num / den).log().mean()
                    total = total + loss
                    count += 1
                pos_frac_sum += pos_idx.numel() / P

        if count == 0:
            return features.new_zeros((), requires_grad=True), {}
        loss = total / count
        metrics = {
            "contrast_raw": loss.detach(),
            "pos_frac": torch.tensor(pos_frac_sum / max(B * T, 1), device=features.device),
        }
        return loss, metrics

class MotionRigidityContrastiveCorr(Motion):
    """PMamba + InfoNCE contrastive aux with clean correspondence-aligned
    per-point residuals."""

    def __init__(self, num_classes, pts_size, contrast_weight=0.05,
                 contrast_temp=0.1, contrast_num_anchors=16, **kwargs):
        super().__init__(num_classes, pts_size, **kwargs)
        self.contrast_weight = contrast_weight
        self.contrast_temp = contrast_temp
        self.contrast_num_anchors = contrast_num_anchors
        self.latest_aux_loss = None
        self.latest_aux_metrics = {}

    def get_auxiliary_loss(self):
        return self.latest_aux_loss

    def get_auxiliary_metrics(self):
        return self.latest_aux_metrics

    def _correspondence_guided_sample(self, points, aux_input):
        """Copied from BearingQCCFeatureMotion. Returns (B, F, S, C) with
        same-index = same-physical-point across frames."""
        batch_size, num_frames, pts_per_frame, channels = points.shape
        sample_size = min(self.pts_size, pts_per_frame)
        device = points.device

        if sample_size == pts_per_frame:
            corr_matched = torch.ones(batch_size, num_frames - 1, pts_per_frame,
                                      dtype=torch.bool, device=device)
            return points, corr_matched

        orig_flat_idx = aux_input['orig_flat_idx']
        corr_target = aux_input['corr_full_target_idx']
        corr_weight = aux_input['corr_full_weight']
        total_pts = corr_target.shape[-1]
        raw_ppf = total_pts // num_frames

        sampled = torch.zeros(batch_size, num_frames, sample_size, channels,
                              device=device, dtype=points.dtype)
        corr_matched = torch.zeros(batch_size, num_frames - 1, sample_size,
                                   dtype=torch.bool, device=device)

        for b in range(batch_size):
            if self.training:
                idx = torch.randperm(pts_per_frame, device=device)[:sample_size]
            else:
                idx = torch.linspace(0, pts_per_frame - 1, sample_size,
                                     device=device).long()
            sampled[b, 0] = points[b, 0, idx]
            current_prov = orig_flat_idx[b, 0, idx].long()

            for t in range(num_frames - 1):
                next_prov = orig_flat_idx[b, t + 1].long()
                reverse_map = torch.full((total_pts,), -1, dtype=torch.long, device=device)
                reverse_map[next_prov] = torch.arange(pts_per_frame, device=device)

                tgt_flat = corr_target[b, current_prov]
                tgt_w = corr_weight[b, current_prov]
                tgt_flat_safe = tgt_flat.clamp(min=0)
                tgt_frame = tgt_flat // raw_ppf
                tgt_pos = reverse_map[tgt_flat_safe]

                valid = ((tgt_flat >= 0) & (tgt_w > 0)
                         & (tgt_frame == t + 1) & (tgt_pos >= 0))

                next_idx = torch.randint(0, pts_per_frame, (sample_size,), device=device)
                next_idx[valid] = tgt_pos[valid]

                sampled[b, t + 1] = points[b, t + 1, next_idx]
                corr_matched[b, t] = valid
                current_prov = orig_flat_idx[b, t + 1, next_idx].long()

        return sampled, corr_matched

    def extract_features(self, inputs):
        # inputs from NvidiaQuaternionQCCParityLoader is a dict.
        if isinstance(inputs, dict):
            points_raw = inputs["points"]                        # (B, T, P_raw, C)
            aux_input = inputs
            has_corr = ("orig_flat_idx" in aux_input
                        and "corr_full_target_idx" in aux_input
                        and "corr_full_weight" in aux_input)
        else:
            points_raw = inputs
            aux_input = None
            has_corr = False

        if has_corr:
            sampled, corr_matched = self._correspondence_guided_sample(
                points_raw[..., :4], aux_input,
            )
            coords = sampled.permute(0, 3, 1, 2).contiguous()   # (B, 4, T, P)
        else:
            coords = self._sample_points(points_raw)
            corr_matched = None

        batchsize, in_dims, timestep, pts_num = coords.shape

        ret_array1 = self.group.group_points(
            distance_dim=[0, 1, 2], array1=coords, array2=coords,
            knn=self.knn[0], dim=3,
        )
        ret_array1 = ret_array1.reshape(batchsize, in_dims, timestep * pts_num, -1)
        fea1_raw = self.pool1(self.stage1(ret_array1)).reshape(
            batchsize, -1, timestep, pts_num,
        )

        self.latest_aux_loss = None
        self.latest_aux_metrics = {}
        if self.training and self.contrast_weight > 0 and has_corr:
            with torch.no_grad():
                rigidity = self._compute_residual_corr(
                    coords[:, :3], corr_matched,
                )
            c_loss, c_metrics = self._contrastive_loss(
                fea1_raw, rigidity, corr_matched,
            )
            self.latest_aux_loss = self.contrast_weight * c_loss
            c_metrics["qcc_raw"] = c_loss.detach()
            c_metrics["qcc_forward"] = c_loss.detach()
            c_metrics["qcc_backward"] = c_loss.detach()
            c_metrics["qcc_valid_ratio"] = corr_matched.float().mean().detach()
            self.latest_aux_metrics = c_metrics

        fea1 = torch.cat((coords, fea1_raw), dim=1)

        in_dims = fea1.shape[1] * 2 - 4
        pts_num //= self.downsample[0]
        rg2 = self.group.st_group_points(fea1, 3, [0, 1, 2], self.knn[1], 3)
        ret2, coords = self.select_ind(rg2, coords, batchsize, in_dims, timestep, pts_num)
        fea2 = self.pool2(self.stage2(ret2)).reshape(batchsize, -1, timestep, pts_num)
        fea2 = torch.cat((coords, fea2), dim=1)
        fea2 = self.multi_scale(fea2)

        in_dims = fea2.shape[1] * 2 - 4
        pts_num //= self.downsample[1]
        rg3 = self.group.st_group_points(fea2, 3, [0, 1, 2], self.knn[2], 3)
        ret3, coords = self.select_ind(rg3, coords, batchsize, in_dims, timestep, pts_num)
        fea3 = self.pool3(self.stage3(ret3)).reshape(batchsize, -1, timestep, pts_num)
        fea3_mamba = self.mamba(fea3)
        coords_fea3 = torch.cat((coords, fea3_mamba), dim=1)

        output = self.stage5(coords_fea3)
        output = self.pool5(output)
        output = self.global_bn(output)
        return output.flatten(1)

    def _compute_residual_corr(self, xyz, corr_matched):
        """xyz: (B, 3, T, P). corr_matched: (B, T-1, P) bool.

        Per frame pair (t, t+1): u/v are correspondence-aligned (same index =
        same physical point). Kabsch using ONLY the corr_matched points for
        robustness, then residual magnitude for ALL points.
        """
        B, _, T, P = xyz.shape
        device = xyz.device
        xyz_p = xyz.permute(0, 2, 3, 1)                          # (B, T, P, 3)
        rig = torch.zeros(B, T, P, device=device, dtype=xyz.dtype)
        eye = torch.eye(3, device=device, dtype=xyz.dtype).unsqueeze(0).expand(B, 3, 3)

        for t in range(T - 1):
            p = xyz_p[:, t]
            q = xyz_p[:, t + 1]
            w = corr_matched[:, t].float()                       # (B, P) mask
            w_sum = w.sum(dim=-1, keepdim=True).clamp(min=1.0).unsqueeze(-1)

            # Centroids on matched subset only.
            c_p = (p * w.unsqueeze(-1)).sum(dim=1, keepdim=True) / w_sum
            c_q = (q * w.unsqueeze(-1)).sum(dim=1, keepdim=True) / w_sum
            u = p - c_p                                           # (B, P, 3)
            v = q - c_q

            # Weighted Kabsch using matched mask.
            u_w = u * w.unsqueeze(-1)
            H = u_w.transpose(-1, -2) @ v                         # (B, 3, 3)
            H = H + 1e-5 * eye
            try:
                U, S, Vh = torch.linalg.svd(H)
            except Exception:
                R = eye
                rig[:, t] = (v - u).norm(dim=-1)
                continue

            V = Vh.transpose(-1, -2)
            det = torch.det(V @ U.transpose(-1, -2))
            D_diag = torch.stack(
                [torch.ones_like(det), torch.ones_like(det), det], dim=-1,
            )
            D = torch.diag_embed(D_diag)
            R = V @ D @ U.transpose(-1, -2)

            u_rot = u @ R.transpose(-1, -2)
            residual = v - u_rot
            rig[:, t] = residual.norm(dim=-1)

        rig[:, -1] = rig[:, -2]
        return rig

    def _contrastive_loss(self, features, rigidity, corr_matched=None):
        """features: (B, C, T, P), rigidity: (B, T, P). Split by per-frame
        median. Only use points that are corr_matched (where reliable)."""
        B, C, T, P = features.shape
        feat = features.permute(0, 2, 3, 1)
        feat = F.normalize(feat, dim=-1)

        med = rigidity.median(dim=-1, keepdim=True).values
        is_rigid = rigidity <= med

        tau = self.contrast_temp
        total = feat.new_zeros(())
        count = 0

        for b in range(B):
            for t in range(T):
                f = feat[b, t]
                r_mask = is_rigid[b, t]
                # Filter to corr-matched points (reliable residuals only).
                if corr_matched is not None and t < T - 1:
                    reliable = corr_matched[b, t]
                elif corr_matched is not None and t == T - 1:
                    reliable = corr_matched[b, t - 1]
                else:
                    reliable = torch.ones_like(r_mask)

                pos_idx = (r_mask & reliable).nonzero(as_tuple=True)[0]
                neg_idx = (~r_mask & reliable).nonzero(as_tuple=True)[0]
                if pos_idx.numel() < 2 or neg_idx.numel() < 1:
                    continue

                for anchor_set, other_set in [(pos_idx, neg_idx), (neg_idx, pos_idx)]:
                    N = min(self.contrast_num_anchors, anchor_set.numel() - 1)
                    if N < 1:
                        continue
                    sel = torch.randperm(anchor_set.numel(), device=f.device)[:N]
                    a_idx = anchor_set[sel]
                    anchors = f[a_idx]
                    pos_f = f[anchor_set]
                    neg_f = f[other_set]

                    sim_pos = anchors @ pos_f.T
                    sim_neg = anchors @ neg_f.T
                    self_mask = (anchor_set.unsqueeze(0) == a_idx.unsqueeze(1))
                    sim_pos = sim_pos.masked_fill(self_mask, -1e9)

                    exp_pos = (sim_pos / tau).exp()
                    exp_neg = (sim_neg / tau).exp()
                    num = exp_pos.sum(dim=-1) + 1e-8
                    den = num + exp_neg.sum(dim=-1)
                    loss = -(num / den).log().mean()
                    total = total + loss
                    count += 1

        if count == 0:
            return features.new_zeros((), requires_grad=True), {}
        loss = total / count
        return loss, {"contrast_raw": loss.detach()}

def _qcc_hamilton(a, b):
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return torch.stack([
        aw*bw - ax*bx - ay*by - az*bz,
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw,
    ], dim=-1)


def _qcc_rot_to_quat(R):
    """Batched rotation matrix -> quaternion [w,x,y,z], Shepperd-style."""
    orig_shape = R.shape[:-2]
    R_flat = R.reshape(-1, 3, 3)
    B = R_flat.shape[0]
    device = R_flat.device
    dtype = R_flat.dtype

    m00, m01, m02 = R_flat[:, 0, 0], R_flat[:, 0, 1], R_flat[:, 0, 2]
    m10, m11, m12 = R_flat[:, 1, 0], R_flat[:, 1, 1], R_flat[:, 1, 2]
    m20, m21, m22 = R_flat[:, 2, 0], R_flat[:, 2, 1], R_flat[:, 2, 2]
    tr = m00 + m11 + m22
    q = torch.zeros(B, 4, device=device, dtype=dtype)

    m1 = tr > 0
    if m1.any():
        s = torch.sqrt(tr[m1].clamp(min=-0.999) + 1.0) * 2.0
        q[m1, 0] = 0.25 * s
        q[m1, 1] = (m21[m1] - m12[m1]) / s
        q[m1, 2] = (m02[m1] - m20[m1]) / s
        q[m1, 3] = (m10[m1] - m01[m1]) / s

    rem = ~m1
    m2a = rem & (m00 > m11) & (m00 > m22)
    if m2a.any():
        s = torch.sqrt(1.0 + m00[m2a] - m11[m2a] - m22[m2a]).clamp(min=1e-8) * 2.0
        q[m2a, 0] = (m21[m2a] - m12[m2a]) / s
        q[m2a, 1] = 0.25 * s
        q[m2a, 2] = (m01[m2a] + m10[m2a]) / s
        q[m2a, 3] = (m02[m2a] + m20[m2a]) / s

    m2b = rem & (~m2a) & (m11 > m22)
    if m2b.any():
        s = torch.sqrt(1.0 + m11[m2b] - m00[m2b] - m22[m2b]).clamp(min=1e-8) * 2.0
        q[m2b, 0] = (m02[m2b] - m20[m2b]) / s
        q[m2b, 1] = (m01[m2b] + m10[m2b]) / s
        q[m2b, 2] = 0.25 * s
        q[m2b, 3] = (m12[m2b] + m21[m2b]) / s

    m2c = rem & (~m2a) & (~m2b)
    if m2c.any():
        s = torch.sqrt(1.0 + m22[m2c] - m00[m2c] - m11[m2c]).clamp(min=1e-8) * 2.0
        q[m2c, 0] = (m10[m2c] - m01[m2c]) / s
        q[m2c, 1] = (m02[m2c] + m20[m2c]) / s
        q[m2c, 2] = (m12[m2c] + m21[m2c]) / s
        q[m2c, 3] = 0.25 * s

    q = F.normalize(q, dim=-1)
    return q.reshape(*orig_shape, 4)


def _qcc_kabsch_quat(src, tgt, weights=None):
    """Batched Kabsch -> unit quaternion. src, tgt: (..., N, 3)."""
    shape = src.shape[:-2]
    N = src.shape[-2]
    src_f = src.reshape(-1, N, 3)
    tgt_f = tgt.reshape(-1, N, 3)
    B = src_f.shape[0]
    device = src_f.device

    if weights is not None:
        w = weights.reshape(-1, N).float()
        w_sum = w.sum(dim=-1, keepdim=True).clamp(min=1.0).unsqueeze(-1)
        src_mean = (src_f * w.unsqueeze(-1)).sum(1, keepdim=True) / w_sum
        tgt_mean = (tgt_f * w.unsqueeze(-1)).sum(1, keepdim=True) / w_sum
        src_c = src_f - src_mean
        tgt_c = tgt_f - tgt_mean
        H = torch.einsum("bn,bni,bnj->bij", w, src_c, tgt_c)
    else:
        src_c = src_f - src_f.mean(1, keepdim=True)
        tgt_c = tgt_f - tgt_f.mean(1, keepdim=True)
        H = torch.einsum("bni,bnj->bij", src_c, tgt_c)

    H = H + 1e-6 * torch.eye(3, device=device, dtype=H.dtype).unsqueeze(0)
    try:
        U, S, Vh = torch.linalg.svd(H)
    except Exception:
        R = torch.eye(3, device=device, dtype=H.dtype).unsqueeze(0).expand(B, 3, 3).contiguous()
        return _qcc_rot_to_quat(R).reshape(*shape, 4)

    V = Vh.transpose(-1, -2)
    det = torch.det(V @ U.transpose(-1, -2))
    D = torch.diag_embed(torch.stack([torch.ones_like(det), torch.ones_like(det), det], dim=-1))
    R = V @ D @ U.transpose(-1, -2)
    bad = ~torch.isfinite(R).all(dim=-1).all(dim=-1)
    if bad.any():
        eye = torch.eye(3, device=device, dtype=R.dtype).unsqueeze(0).expand_as(R)
        R = torch.where(bad.unsqueeze(-1).unsqueeze(-1), eye, R)
    q = _qcc_rot_to_quat(R)
    return q.reshape(*shape, 4)


class MotionQCCAnchored(Motion):
    """PMamba + Mittal-anchored quaternion-pair + transitivity aux.

    Features: stage1 output pooled per-frame. Head predicts q(t, t+1); anchor
    is Kabsch q_obs from correspondence-aligned sampled points.
    """

    def __init__(self, num_classes, pts_size, qcc_weight=0.05,
                 anchor_weight=1.0, trans_weight=0.5, **kwargs):
        super().__init__(num_classes, pts_size, **kwargs)
        self.qcc_weight = qcc_weight
        self.anchor_weight = anchor_weight
        self.trans_weight = trans_weight
        # Stage1 per-point dim is 64 (after pool1 + cat with coords=68, but we
        # use raw stage1 output before cat, which is 64).
        feat_dim = 64
        self.qcc_head = nn.Sequential(
            nn.Linear(feat_dim * 2, feat_dim),
            nn.GELU(),
            nn.Linear(feat_dim, 4),
        )
        self.latest_aux_loss = None
        self.latest_aux_metrics = {}

    def get_auxiliary_loss(self):
        return self.latest_aux_loss

    def get_auxiliary_metrics(self):
        return self.latest_aux_metrics

    def _correspondence_guided_sample(self, points, aux_input):
        batch_size, num_frames, pts_per_frame, channels = points.shape
        sample_size = min(self.pts_size, pts_per_frame)
        device = points.device
        if sample_size == pts_per_frame:
            corr_matched = torch.ones(batch_size, num_frames - 1, pts_per_frame,
                                      dtype=torch.bool, device=device)
            return points, corr_matched

        orig_flat_idx = aux_input["orig_flat_idx"]
        corr_target = aux_input["corr_full_target_idx"]
        corr_weight = aux_input["corr_full_weight"]
        total_pts = corr_target.shape[-1]
        raw_ppf = total_pts // num_frames

        sampled = torch.zeros(batch_size, num_frames, sample_size, channels,
                              device=device, dtype=points.dtype)
        corr_matched = torch.zeros(batch_size, num_frames - 1, sample_size,
                                   dtype=torch.bool, device=device)

        for b in range(batch_size):
            if self.training:
                idx = torch.randperm(pts_per_frame, device=device)[:sample_size]
            else:
                idx = torch.linspace(0, pts_per_frame - 1, sample_size,
                                     device=device).long()
            sampled[b, 0] = points[b, 0, idx]
            current_prov = orig_flat_idx[b, 0, idx].long()
            for t in range(num_frames - 1):
                next_prov = orig_flat_idx[b, t + 1].long()
                reverse_map = torch.full((total_pts,), -1, dtype=torch.long, device=device)
                reverse_map[next_prov] = torch.arange(pts_per_frame, device=device)
                tgt_flat = corr_target[b, current_prov]
                tgt_w = corr_weight[b, current_prov]
                tgt_flat_safe = tgt_flat.clamp(min=0)
                tgt_frame = tgt_flat // raw_ppf
                tgt_pos = reverse_map[tgt_flat_safe]
                valid = ((tgt_flat >= 0) & (tgt_w > 0)
                         & (tgt_frame == t + 1) & (tgt_pos >= 0))
                next_idx = torch.randint(0, pts_per_frame, (sample_size,), device=device)
                next_idx[valid] = tgt_pos[valid]
                sampled[b, t + 1] = points[b, t + 1, next_idx]
                corr_matched[b, t] = valid
                current_prov = orig_flat_idx[b, t + 1, next_idx].long()
        return sampled, corr_matched

    def extract_features(self, inputs):
        if isinstance(inputs, dict):
            points_raw = inputs["points"]
            aux_input = inputs
            has_corr = ("orig_flat_idx" in aux_input
                        and "corr_full_target_idx" in aux_input
                        and "corr_full_weight" in aux_input)
        else:
            points_raw = inputs
            aux_input = None
            has_corr = False

        if has_corr:
            sampled, corr_matched = self._correspondence_guided_sample(
                points_raw[..., :4], aux_input,
            )
            coords = sampled.permute(0, 3, 1, 2).contiguous()
        else:
            coords = self._sample_points(points_raw)
            corr_matched = None

        batchsize, in_dims, timestep, pts_num = coords.shape

        ret_array1 = self.group.group_points(
            distance_dim=[0, 1, 2], array1=coords, array2=coords,
            knn=self.knn[0], dim=3,
        )
        ret_array1 = ret_array1.reshape(batchsize, in_dims, timestep * pts_num, -1)
        fea1_raw = self.pool1(self.stage1(ret_array1)).reshape(
            batchsize, -1, timestep, pts_num,
        )                                                        # (B, 64, T, P)

        self.latest_aux_loss = None
        self.latest_aux_metrics = {}
        if self.training and self.qcc_weight > 0 and has_corr:
            q_loss, q_metrics = self._qcc_aux(fea1_raw, coords[:, :3], corr_matched)
            self.latest_aux_loss = self.qcc_weight * q_loss
            q_metrics["qcc_raw"] = q_loss.detach()
            q_metrics["qcc_forward"] = q_metrics.get("anchor_raw", q_loss.detach())
            q_metrics["qcc_backward"] = q_metrics.get("trans_raw", q_loss.detach())
            q_metrics["qcc_valid_ratio"] = corr_matched.float().mean().detach()
            self.latest_aux_metrics = q_metrics

        fea1 = torch.cat((coords, fea1_raw), dim=1)
        in_dims = fea1.shape[1] * 2 - 4
        pts_num //= self.downsample[0]
        rg2 = self.group.st_group_points(fea1, 3, [0, 1, 2], self.knn[1], 3)
        ret2, coords = self.select_ind(rg2, coords, batchsize, in_dims, timestep, pts_num)
        fea2 = self.pool2(self.stage2(ret2)).reshape(batchsize, -1, timestep, pts_num)
        fea2 = torch.cat((coords, fea2), dim=1)
        fea2 = self.multi_scale(fea2)

        in_dims = fea2.shape[1] * 2 - 4
        pts_num //= self.downsample[1]
        rg3 = self.group.st_group_points(fea2, 3, [0, 1, 2], self.knn[2], 3)
        ret3, coords = self.select_ind(rg3, coords, batchsize, in_dims, timestep, pts_num)
        fea3 = self.pool3(self.stage3(ret3)).reshape(batchsize, -1, timestep, pts_num)
        fea3_mamba = self.mamba(fea3)
        coords_fea3 = torch.cat((coords, fea3_mamba), dim=1)

        output = self.stage5(coords_fea3)
        output = self.pool5(output)
        output = self.global_bn(output)
        return output.flatten(1)

    def _qcc_aux(self, features, xyz, corr_matched):
        """features (B, C, T, P), xyz (B, 3, T, P). Per-frame pooled features
        predict pair quaternions; anchor via Kabsch on corr-aligned xyz."""
        B, C, T, P = features.shape
        device = features.device
        feat = features.mean(dim=-1).transpose(1, 2)             # (B, T, C)
        xyz_p = xyz.permute(0, 2, 3, 1)                           # (B, T, P, 3)

        def pred(f_src, f_tgt):
            q = self.qcc_head(torch.cat([f_src, f_tgt], dim=-1))
            return F.normalize(q, dim=-1)

        q_preds, q_obs = [], []
        for t in range(T - 1):
            q_preds.append(pred(feat[:, t], feat[:, t + 1]))
            with torch.no_grad():
                w = corr_matched[:, t].float()
                q_obs.append(_qcc_kabsch_quat(xyz_p[:, t], xyz_p[:, t + 1], w))
        q_preds_t = torch.stack(q_preds, dim=1)                   # (B, T-1, 4)
        q_obs_t = torch.stack(q_obs, dim=1)

        cos = (q_preds_t * q_obs_t).sum(dim=-1)
        anchor_loss = (1.0 - cos ** 2).mean()

        trans_loss = features.new_zeros(())
        n_trip = 0
        for t in range(T - 2):
            q_skip = pred(feat[:, t], feat[:, t + 2])
            q_comp = _qcc_hamilton(q_preds_t[:, t], q_preds_t[:, t + 1])
            cos_t = (q_skip * q_comp).sum(dim=-1)
            trans_loss = trans_loss + (1.0 - cos_t ** 2).mean()
            n_trip += 1
        if n_trip > 0:
            trans_loss = trans_loss / n_trip

        total = self.anchor_weight * anchor_loss + self.trans_weight * trans_loss
        return total, {
            "anchor_raw": anchor_loss.detach(),
            "trans_raw": trans_loss.detach(),
            "q_cos_mean": cos.abs().mean().detach(),
        }

class MotionRigiditySegmentation(Motion):
    """PMamba + per-point rigid/articulating binary segmentation aux.

    Aux target: 1 if Kabsch residual > per-frame median, else 0. Tiny linear
    head on fea1 per-point features. Binary CE aux averaged over
    correspondence-matched points only.
    """

    def __init__(self, num_classes, pts_size, seg_weight=0.1, **kwargs):
        super().__init__(num_classes, pts_size, **kwargs)
        self.seg_weight = seg_weight
        feat_dim = 64                                            # stage1 out
        self.seg_head = nn.Linear(feat_dim, 2)
        self.latest_aux_loss = None
        self.latest_aux_metrics = {}

    def get_auxiliary_loss(self):
        return self.latest_aux_loss

    def get_auxiliary_metrics(self):
        return self.latest_aux_metrics

    def _correspondence_guided_sample(self, points, aux_input):
        batch_size, num_frames, pts_per_frame, channels = points.shape
        sample_size = min(self.pts_size, pts_per_frame)
        device = points.device
        if sample_size == pts_per_frame:
            corr_matched = torch.ones(batch_size, num_frames - 1, pts_per_frame,
                                      dtype=torch.bool, device=device)
            return points, corr_matched

        orig_flat_idx = aux_input["orig_flat_idx"]
        corr_target = aux_input["corr_full_target_idx"]
        corr_weight = aux_input["corr_full_weight"]
        total_pts = corr_target.shape[-1]
        raw_ppf = total_pts // num_frames

        sampled = torch.zeros(batch_size, num_frames, sample_size, channels,
                              device=device, dtype=points.dtype)
        corr_matched = torch.zeros(batch_size, num_frames - 1, sample_size,
                                   dtype=torch.bool, device=device)

        for b in range(batch_size):
            if self.training:
                idx = torch.randperm(pts_per_frame, device=device)[:sample_size]
            else:
                idx = torch.linspace(0, pts_per_frame - 1, sample_size,
                                     device=device).long()
            sampled[b, 0] = points[b, 0, idx]
            current_prov = orig_flat_idx[b, 0, idx].long()
            for t in range(num_frames - 1):
                next_prov = orig_flat_idx[b, t + 1].long()
                reverse_map = torch.full((total_pts,), -1, dtype=torch.long, device=device)
                reverse_map[next_prov] = torch.arange(pts_per_frame, device=device)
                tgt_flat = corr_target[b, current_prov]
                tgt_w = corr_weight[b, current_prov]
                tgt_flat_safe = tgt_flat.clamp(min=0)
                tgt_frame = tgt_flat // raw_ppf
                tgt_pos = reverse_map[tgt_flat_safe]
                valid = ((tgt_flat >= 0) & (tgt_w > 0)
                         & (tgt_frame == t + 1) & (tgt_pos >= 0))
                next_idx = torch.randint(0, pts_per_frame, (sample_size,), device=device)
                next_idx[valid] = tgt_pos[valid]
                sampled[b, t + 1] = points[b, t + 1, next_idx]
                corr_matched[b, t] = valid
                current_prov = orig_flat_idx[b, t + 1, next_idx].long()
        return sampled, corr_matched

    def extract_features(self, inputs):
        if isinstance(inputs, dict):
            points_raw = inputs["points"]
            aux_input = inputs
            has_corr = ("orig_flat_idx" in aux_input
                        and "corr_full_target_idx" in aux_input
                        and "corr_full_weight" in aux_input)
        else:
            points_raw = inputs
            aux_input = None
            has_corr = False

        if has_corr:
            sampled, corr_matched = self._correspondence_guided_sample(
                points_raw[..., :4], aux_input,
            )
            coords = sampled.permute(0, 3, 1, 2).contiguous()
        else:
            coords = self._sample_points(points_raw)
            corr_matched = None

        batchsize, in_dims, timestep, pts_num = coords.shape

        ret_array1 = self.group.group_points(
            distance_dim=[0, 1, 2], array1=coords, array2=coords,
            knn=self.knn[0], dim=3,
        )
        ret_array1 = ret_array1.reshape(batchsize, in_dims, timestep * pts_num, -1)
        fea1_raw = self.pool1(self.stage1(ret_array1)).reshape(
            batchsize, -1, timestep, pts_num,
        )                                                        # (B, 64, T, P)

        self.latest_aux_loss = None
        self.latest_aux_metrics = {}
        if self.training and self.seg_weight > 0 and has_corr:
            with torch.no_grad():
                rigidity = self._compute_residual_corr(coords[:, :3], corr_matched)
            s_loss, s_metrics = self._seg_loss(fea1_raw, rigidity, corr_matched)
            self.latest_aux_loss = self.seg_weight * s_loss
            s_metrics["qcc_raw"] = s_loss.detach()
            s_metrics["qcc_forward"] = s_loss.detach()
            s_metrics["qcc_backward"] = s_loss.detach()
            s_metrics["qcc_valid_ratio"] = corr_matched.float().mean().detach()
            self.latest_aux_metrics = s_metrics

        fea1 = torch.cat((coords, fea1_raw), dim=1)
        in_dims = fea1.shape[1] * 2 - 4
        pts_num //= self.downsample[0]
        rg2 = self.group.st_group_points(fea1, 3, [0, 1, 2], self.knn[1], 3)
        ret2, coords = self.select_ind(rg2, coords, batchsize, in_dims, timestep, pts_num)
        fea2 = self.pool2(self.stage2(ret2)).reshape(batchsize, -1, timestep, pts_num)
        fea2 = torch.cat((coords, fea2), dim=1)
        fea2 = self.multi_scale(fea2)

        in_dims = fea2.shape[1] * 2 - 4
        pts_num //= self.downsample[1]
        rg3 = self.group.st_group_points(fea2, 3, [0, 1, 2], self.knn[2], 3)
        ret3, coords = self.select_ind(rg3, coords, batchsize, in_dims, timestep, pts_num)
        fea3 = self.pool3(self.stage3(ret3)).reshape(batchsize, -1, timestep, pts_num)
        fea3_mamba = self.mamba(fea3)
        coords_fea3 = torch.cat((coords, fea3_mamba), dim=1)

        output = self.stage5(coords_fea3)
        output = self.pool5(output)
        output = self.global_bn(output)
        return output.flatten(1)

    def _compute_residual_corr(self, xyz, corr_matched):
        B, _, T, P = xyz.shape
        device = xyz.device
        xyz_p = xyz.permute(0, 2, 3, 1)
        rig = torch.zeros(B, T, P, device=device, dtype=xyz.dtype)
        eye = torch.eye(3, device=device, dtype=xyz.dtype).unsqueeze(0).expand(B, 3, 3)
        for t in range(T - 1):
            p = xyz_p[:, t]; q = xyz_p[:, t + 1]
            w = corr_matched[:, t].float()
            w_sum = w.sum(dim=-1, keepdim=True).clamp(min=1.0).unsqueeze(-1)
            c_p = (p * w.unsqueeze(-1)).sum(1, keepdim=True) / w_sum
            c_q = (q * w.unsqueeze(-1)).sum(1, keepdim=True) / w_sum
            u = p - c_p; v = q - c_q
            H = (u * w.unsqueeze(-1)).transpose(-1, -2) @ v
            H = H + 1e-5 * eye
            try:
                U, S, Vh = torch.linalg.svd(H)
            except Exception:
                rig[:, t] = (v - u).norm(dim=-1); continue
            V = Vh.transpose(-1, -2)
            det = torch.det(V @ U.transpose(-1, -2))
            D = torch.diag_embed(torch.stack([torch.ones_like(det), torch.ones_like(det), det], dim=-1))
            R = V @ D @ U.transpose(-1, -2)
            rig[:, t] = (v - u @ R.transpose(-1, -2)).norm(dim=-1)
        rig[:, -1] = rig[:, -2]
        return rig

    def _seg_loss(self, features, rigidity, corr_matched):
        """Binary CE per-point: 1 if residual > per-frame median else 0.
        Mask to corr-matched points only."""
        B, C, T, P = features.shape
        feat = features.permute(0, 2, 3, 1).reshape(-1, C)       # (B*T*P, C)
        logits = self.seg_head(feat)                              # (B*T*P, 2)

        med = rigidity.median(dim=-1, keepdim=True).values
        target = (rigidity > med).long().reshape(-1)              # (B*T*P,)

        # Mask to corr-matched (valid) points. corr_matched is (B, T-1, P).
        # Expand last frame from previous.
        if corr_matched.shape[1] < T:
            last = corr_matched[:, -1:]
            mask_full = torch.cat([corr_matched, last], dim=1)    # (B, T, P)
        else:
            mask_full = corr_matched
        mask = mask_full.reshape(-1)                              # bool

        if mask.sum() < 1:
            return features.new_zeros((), requires_grad=True), {}

        loss = F.cross_entropy(logits[mask], target[mask])
        with torch.no_grad():
            acc = (logits[mask].argmax(-1) == target[mask]).float().mean()
        return loss, {"seg_raw": loss.detach(), "seg_acc": acc}

class MotionCentroidAux(Motion):
    """PMamba + per-frame centroid regression aux."""

    def __init__(self, num_classes, pts_size, centroid_weight=0.05, **kwargs):
        super().__init__(num_classes, pts_size, **kwargs)
        self.centroid_weight = centroid_weight
        feat_dim = 64                                    # stage1 out channels
        self.centroid_head = nn.Sequential(
            nn.Linear(feat_dim, 64),
            nn.GELU(),
            nn.Linear(64, 3),
        )
        self.latest_aux_loss = None
        self.latest_aux_metrics = {}

    def get_auxiliary_loss(self):
        return self.latest_aux_loss

    def get_auxiliary_metrics(self):
        return self.latest_aux_metrics

    def extract_features(self, inputs):
        coords = self._sample_points(inputs)
        batchsize, in_dims, timestep, pts_num = coords.shape

        ret_array1 = self.group.group_points(
            distance_dim=[0, 1, 2], array1=coords, array2=coords,
            knn=self.knn[0], dim=3,
        )
        ret_array1 = ret_array1.reshape(batchsize, in_dims, timestep * pts_num, -1)
        fea1_raw = self.pool1(self.stage1(ret_array1)).reshape(
            batchsize, -1, timestep, pts_num,
        )                                                # (B, 64, T, P)

        self.latest_aux_loss = None
        self.latest_aux_metrics = {}
        if self.training and self.centroid_weight > 0:
            # Per-frame pooled features -> predict centroid.
            with torch.no_grad():
                centroid_target = coords[:, :3].mean(dim=-1).transpose(1, 2)  # (B, T, 3)
            feat_pooled = fea1_raw.mean(dim=-1).transpose(1, 2)                # (B, T, 64)
            pred = self.centroid_head(feat_pooled)                             # (B, T, 3)
            aux_loss = F.mse_loss(pred, centroid_target)
            self.latest_aux_loss = self.centroid_weight * aux_loss
            self.latest_aux_metrics = {
                "qcc_raw": aux_loss.detach(),
                "qcc_forward": aux_loss.detach(),
                "qcc_backward": aux_loss.detach(),
                "qcc_valid_ratio": torch.ones(1, device=aux_loss.device),
            }

        fea1 = torch.cat((coords, fea1_raw), dim=1)
        in_dims = fea1.shape[1] * 2 - 4
        pts_num //= self.downsample[0]
        rg2 = self.group.st_group_points(fea1, 3, [0, 1, 2], self.knn[1], 3)
        ret2, coords = self.select_ind(rg2, coords, batchsize, in_dims, timestep, pts_num)
        fea2 = self.pool2(self.stage2(ret2)).reshape(batchsize, -1, timestep, pts_num)
        fea2 = torch.cat((coords, fea2), dim=1)
        fea2 = self.multi_scale(fea2)

        in_dims = fea2.shape[1] * 2 - 4
        pts_num //= self.downsample[1]
        rg3 = self.group.st_group_points(fea2, 3, [0, 1, 2], self.knn[2], 3)
        ret3, coords = self.select_ind(rg3, coords, batchsize, in_dims, timestep, pts_num)
        fea3 = self.pool3(self.stage3(ret3)).reshape(batchsize, -1, timestep, pts_num)
        fea3_mamba = self.mamba(fea3)
        coords_fea3 = torch.cat((coords, fea3_mamba), dim=1)

        output = self.stage5(coords_fea3)
        output = self.pool5(output)
        output = self.global_bn(output)
        return output.flatten(1)

def _dq_hamilton(a, b):
    aw, ax, ay, az = a[...,0], a[...,1], a[...,2], a[...,3]
    bw, bx, by, bz = b[...,0], b[...,1], b[...,2], b[...,3]
    return torch.stack([
        aw*bw - ax*bx - ay*by - az*bz,
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw,
    ], dim=-1)


def _dq_rot_to_quat(R):
    orig = R.shape[:-2]
    Rf = R.reshape(-1, 3, 3)
    m00,m01,m02 = Rf[:,0,0], Rf[:,0,1], Rf[:,0,2]
    m10,m11,m12 = Rf[:,1,0], Rf[:,1,1], Rf[:,1,2]
    m20,m21,m22 = Rf[:,2,0], Rf[:,2,1], Rf[:,2,2]
    tr = m00 + m11 + m22
    B = Rf.shape[0]; device = Rf.device
    q = torch.zeros(B, 4, device=device, dtype=Rf.dtype)
    m1 = tr > 0
    if m1.any():
        s = torch.sqrt(tr[m1].clamp(min=-0.999) + 1.0) * 2
        q[m1,0]=0.25*s; q[m1,1]=(m21[m1]-m12[m1])/s
        q[m1,2]=(m02[m1]-m20[m1])/s; q[m1,3]=(m10[m1]-m01[m1])/s
    rem = ~m1
    m2a = rem & (m00>m11) & (m00>m22)
    if m2a.any():
        s = torch.sqrt(1+m00[m2a]-m11[m2a]-m22[m2a]).clamp(min=1e-8)*2
        q[m2a,0]=(m21[m2a]-m12[m2a])/s; q[m2a,1]=0.25*s
        q[m2a,2]=(m01[m2a]+m10[m2a])/s; q[m2a,3]=(m02[m2a]+m20[m2a])/s
    m2b = rem & (~m2a) & (m11>m22)
    if m2b.any():
        s = torch.sqrt(1+m11[m2b]-m00[m2b]-m22[m2b]).clamp(min=1e-8)*2
        q[m2b,0]=(m02[m2b]-m20[m2b])/s; q[m2b,1]=(m01[m2b]+m10[m2b])/s
        q[m2b,2]=0.25*s; q[m2b,3]=(m12[m2b]+m21[m2b])/s
    m2c = rem & (~m2a) & (~m2b)
    if m2c.any():
        s = torch.sqrt(1+m22[m2c]-m00[m2c]-m11[m2c]).clamp(min=1e-8)*2
        q[m2c,0]=(m10[m2c]-m01[m2c])/s; q[m2c,1]=(m02[m2c]+m20[m2c])/s
        q[m2c,2]=(m12[m2c]+m21[m2c])/s; q[m2c,3]=0.25*s
    return F.normalize(q, dim=-1).reshape(*orig, 4)


def _dq_kabsch_rt(src, tgt, weights):
    """Kabsch returning (q_r, t). src/tgt (..., N, 3); weights (..., N)."""
    shape = src.shape[:-2]
    N = src.shape[-2]
    sf = src.reshape(-1, N, 3); tf = tgt.reshape(-1, N, 3)
    B = sf.shape[0]; device = sf.device
    w = weights.reshape(-1, N).float().clamp(min=0)
    w_sum = w.sum(-1, keepdim=True).clamp(min=1.0).unsqueeze(-1)
    sm = (sf * w.unsqueeze(-1)).sum(1, keepdim=True) / w_sum
    tm = (tf * w.unsqueeze(-1)).sum(1, keepdim=True) / w_sum
    sc = sf - sm; tc = tf - tm
    H = torch.einsum('bn,bni,bnj->bij', w, sc, tc)
    H = H + 1e-6 * torch.eye(3, device=device).unsqueeze(0)
    U, S, Vh = torch.linalg.svd(H)
    V = Vh.transpose(-1, -2)
    det = torch.det(V @ U.transpose(-1, -2))
    D = torch.diag_embed(torch.stack([torch.ones_like(det), torch.ones_like(det), det], dim=-1))
    R = V @ D @ U.transpose(-1, -2)
    q_r = _dq_rot_to_quat(R)
    t = tm.squeeze(1) - torch.bmm(R, sm.transpose(-1, -2)).squeeze(-1)
    return q_r.reshape(*shape, 4), t.reshape(*shape, 3)


class MotionDQAux(Motion):
    """PMamba + dual-quaternion regression aux."""

    def __init__(self, num_classes, pts_size, dq_weight=0.05, **kwargs):
        super().__init__(num_classes, pts_size, **kwargs)
        self.dq_weight = dq_weight
        feat_dim = 64
        # Predict (q_r, q_d) = 8 dim per pair from concat of t, t+1 features
        self.dq_head = nn.Sequential(
            nn.Linear(feat_dim * 2, feat_dim),
            nn.GELU(),
            nn.Linear(feat_dim, 8),
        )
        self.latest_aux_loss = None
        self.latest_aux_metrics = {}

    def get_auxiliary_loss(self):
        return self.latest_aux_loss

    def get_auxiliary_metrics(self):
        return self.latest_aux_metrics

    def _correspondence_guided_sample(self, points, aux_input):
        batch_size, num_frames, pts_per_frame, channels = points.shape
        sample_size = min(self.pts_size, pts_per_frame)
        device = points.device
        if sample_size == pts_per_frame:
            corr_matched = torch.ones(batch_size, num_frames - 1, pts_per_frame,
                                      dtype=torch.bool, device=device)
            return points, corr_matched
        orig_flat_idx = aux_input["orig_flat_idx"]
        corr_target = aux_input["corr_full_target_idx"]
        corr_weight = aux_input["corr_full_weight"]
        total_pts = corr_target.shape[-1]
        raw_ppf = total_pts // num_frames
        sampled = torch.zeros(batch_size, num_frames, sample_size, channels,
                              device=device, dtype=points.dtype)
        corr_matched = torch.zeros(batch_size, num_frames - 1, sample_size,
                                   dtype=torch.bool, device=device)
        for b in range(batch_size):
            if self.training:
                idx = torch.randperm(pts_per_frame, device=device)[:sample_size]
            else:
                idx = torch.linspace(0, pts_per_frame - 1, sample_size,
                                     device=device).long()
            sampled[b, 0] = points[b, 0, idx]
            current_prov = orig_flat_idx[b, 0, idx].long()
            for t in range(num_frames - 1):
                next_prov = orig_flat_idx[b, t + 1].long()
                reverse_map = torch.full((total_pts,), -1, dtype=torch.long, device=device)
                reverse_map[next_prov] = torch.arange(pts_per_frame, device=device)
                tgt_flat = corr_target[b, current_prov]
                tgt_w = corr_weight[b, current_prov]
                tgt_flat_safe = tgt_flat.clamp(min=0)
                tgt_frame = tgt_flat // raw_ppf
                tgt_pos = reverse_map[tgt_flat_safe]
                valid = ((tgt_flat >= 0) & (tgt_w > 0)
                         & (tgt_frame == t + 1) & (tgt_pos >= 0))
                next_idx = torch.randint(0, pts_per_frame, (sample_size,), device=device)
                next_idx[valid] = tgt_pos[valid]
                sampled[b, t + 1] = points[b, t + 1, next_idx]
                corr_matched[b, t] = valid
                current_prov = orig_flat_idx[b, t + 1, next_idx].long()
        return sampled, corr_matched

    def extract_features(self, inputs):
        if isinstance(inputs, dict):
            points_raw = inputs["points"]
            aux_input = inputs
            has_corr = ("orig_flat_idx" in aux_input
                        and "corr_full_target_idx" in aux_input
                        and "corr_full_weight" in aux_input)
        else:
            points_raw = inputs
            aux_input = None
            has_corr = False

        if has_corr:
            sampled, corr_matched = self._correspondence_guided_sample(
                points_raw[..., :4], aux_input,
            )
            coords = sampled.permute(0, 3, 1, 2).contiguous()
        else:
            coords = self._sample_points(points_raw)
            corr_matched = None

        batchsize, in_dims, timestep, pts_num = coords.shape

        ret_array1 = self.group.group_points(
            distance_dim=[0, 1, 2], array1=coords, array2=coords,
            knn=self.knn[0], dim=3,
        )
        ret_array1 = ret_array1.reshape(batchsize, in_dims, timestep * pts_num, -1)
        fea1_raw = self.pool1(self.stage1(ret_array1)).reshape(
            batchsize, -1, timestep, pts_num,
        )

        self.latest_aux_loss = None
        self.latest_aux_metrics = {}
        if self.training and self.dq_weight > 0 and has_corr:
            # Per-frame pooled features
            feat_frame = fea1_raw.mean(dim=-1).transpose(1, 2)      # (B, T, 64)
            # Predict DQ for each consecutive pair
            T = feat_frame.shape[1]
            pair_in = torch.cat([feat_frame[:, :-1], feat_frame[:, 1:]], dim=-1)  # (B, T-1, 128)
            dq_pred = self.dq_head(pair_in)                         # (B, T-1, 8)
            # Normalize q_r part to unit norm
            q_r_pred = F.normalize(dq_pred[..., :4], dim=-1)
            q_d_pred = dq_pred[..., 4:]
            # Project q_d to be orthogonal to q_r (unit DQ manifold)
            q_d_pred = q_d_pred - (q_d_pred * q_r_pred).sum(-1, keepdim=True) * q_r_pred
            dq_pred_norm = torch.cat([q_r_pred, q_d_pred], dim=-1)

            # Compute observable DQ
            with torch.no_grad():
                xyz_p = coords[:, :3].permute(0, 2, 3, 1)           # (B, T, P, 3)
                q_r_list, t_list = [], []
                for t in range(T - 1):
                    q_r, trans = _dq_kabsch_rt(
                        xyz_p[:, t], xyz_p[:, t + 1],
                        corr_matched[:, t],
                    )
                    q_r_list.append(q_r); t_list.append(trans)
                q_r_obs = torch.stack(q_r_list, dim=1)              # (B, T-1, 4)
                t_obs = torch.stack(t_list, dim=1)                  # (B, T-1, 3)
                # DQ: q_d = 0.5 * [0, t] * q_r
                zero = torch.zeros_like(t_obs[..., :1])
                t_quat = torch.cat([zero, t_obs], dim=-1)
                q_d_obs = 0.5 * _dq_hamilton(t_quat, q_r_obs)
                dq_obs = torch.cat([q_r_obs, q_d_obs], dim=-1)

            # Sign-align q_r (double cover): flip pred if dot w/ obs negative.
            cos_r = (q_r_pred * q_r_obs).sum(-1, keepdim=True)
            sign = torch.where(cos_r >= 0, torch.ones_like(cos_r), -torch.ones_like(cos_r))
            dq_pred_signed = torch.cat([q_r_pred * sign, q_d_pred * sign], dim=-1)

            aux_loss = F.mse_loss(dq_pred_signed, dq_obs)
            self.latest_aux_loss = self.dq_weight * aux_loss
            self.latest_aux_metrics = {
                "qcc_raw": aux_loss.detach(),
                "qcc_forward": aux_loss.detach(),
                "qcc_backward": aux_loss.detach(),
                "qcc_valid_ratio": corr_matched.float().mean().detach(),
            }

        fea1 = torch.cat((coords, fea1_raw), dim=1)
        in_dims = fea1.shape[1] * 2 - 4
        pts_num //= self.downsample[0]
        rg2 = self.group.st_group_points(fea1, 3, [0, 1, 2], self.knn[1], 3)
        ret2, coords = self.select_ind(rg2, coords, batchsize, in_dims, timestep, pts_num)
        fea2 = self.pool2(self.stage2(ret2)).reshape(batchsize, -1, timestep, pts_num)
        fea2 = torch.cat((coords, fea2), dim=1)
        fea2 = self.multi_scale(fea2)

        in_dims = fea2.shape[1] * 2 - 4
        pts_num //= self.downsample[1]
        rg3 = self.group.st_group_points(fea2, 3, [0, 1, 2], self.knn[2], 3)
        ret3, coords = self.select_ind(rg3, coords, batchsize, in_dims, timestep, pts_num)
        fea3 = self.pool3(self.stage3(ret3)).reshape(batchsize, -1, timestep, pts_num)
        fea3_mamba = self.mamba(fea3)
        coords_fea3 = torch.cat((coords, fea3_mamba), dim=1)

        output = self.stage5(coords_fea3)
        output = self.pool5(output)
        output = self.global_bn(output)
        return output.flatten(1)

def _dqcc_hamilton(a, b):
    aw,ax,ay,az = a[...,0],a[...,1],a[...,2],a[...,3]
    bw,bx,by,bz = b[...,0],b[...,1],b[...,2],b[...,3]
    return torch.stack([
        aw*bw - ax*bx - ay*by - az*bz,
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw,
    ], dim=-1)


def _dqcc_dq_mul(p_r, p_d, q_r, q_d):
    return _dqcc_hamilton(p_r, q_r), _dqcc_hamilton(p_r, q_d) + _dqcc_hamilton(p_d, q_r)


def _dqcc_rot_to_quat(R):
    orig = R.shape[:-2]
    Rf = R.reshape(-1, 3, 3)
    m00,m01,m02 = Rf[:,0,0], Rf[:,0,1], Rf[:,0,2]
    m10,m11,m12 = Rf[:,1,0], Rf[:,1,1], Rf[:,1,2]
    m20,m21,m22 = Rf[:,2,0], Rf[:,2,1], Rf[:,2,2]
    tr = m00 + m11 + m22
    B = Rf.shape[0]; device = Rf.device
    q = torch.zeros(B, 4, device=device, dtype=Rf.dtype)
    m1 = tr > 0
    if m1.any():
        s = torch.sqrt(tr[m1].clamp(min=-0.999) + 1.0) * 2
        q[m1,0]=0.25*s; q[m1,1]=(m21[m1]-m12[m1])/s
        q[m1,2]=(m02[m1]-m20[m1])/s; q[m1,3]=(m10[m1]-m01[m1])/s
    rem = ~m1
    m2a = rem & (m00>m11) & (m00>m22)
    if m2a.any():
        s = torch.sqrt(1+m00[m2a]-m11[m2a]-m22[m2a]).clamp(min=1e-8)*2
        q[m2a,0]=(m21[m2a]-m12[m2a])/s; q[m2a,1]=0.25*s
        q[m2a,2]=(m01[m2a]+m10[m2a])/s; q[m2a,3]=(m02[m2a]+m20[m2a])/s
    m2b = rem & (~m2a) & (m11>m22)
    if m2b.any():
        s = torch.sqrt(1+m11[m2b]-m00[m2b]-m22[m2b]).clamp(min=1e-8)*2
        q[m2b,0]=(m02[m2b]-m20[m2b])/s; q[m2b,1]=(m01[m2b]+m10[m2b])/s
        q[m2b,2]=0.25*s; q[m2b,3]=(m12[m2b]+m21[m2b])/s
    m2c = rem & (~m2a) & (~m2b)
    if m2c.any():
        s = torch.sqrt(1+m22[m2c]-m00[m2c]-m11[m2c]).clamp(min=1e-8)*2
        q[m2c,0]=(m10[m2c]-m01[m2c])/s; q[m2c,1]=(m02[m2c]+m20[m2c])/s
        q[m2c,2]=(m12[m2c]+m21[m2c])/s; q[m2c,3]=0.25*s
    return F.normalize(q, dim=-1).reshape(*orig, 4)


def _dqcc_kabsch_rt(src, tgt, weights):
    shape = src.shape[:-2]
    N = src.shape[-2]
    sf = src.reshape(-1, N, 3); tf = tgt.reshape(-1, N, 3)
    B = sf.shape[0]; device = sf.device
    w = weights.reshape(-1, N).float().clamp(min=0)
    w_sum = w.sum(-1, keepdim=True).clamp(min=1.0).unsqueeze(-1)
    sm = (sf * w.unsqueeze(-1)).sum(1, keepdim=True) / w_sum
    tm = (tf * w.unsqueeze(-1)).sum(1, keepdim=True) / w_sum
    sc = sf - sm; tc = tf - tm
    H = torch.einsum('bn,bni,bnj->bij', w, sc, tc)
    H = H + 1e-6 * torch.eye(3, device=device).unsqueeze(0)
    U, S, Vh = torch.linalg.svd(H)
    V = Vh.transpose(-1, -2)
    det = torch.det(V @ U.transpose(-1, -2))
    D = torch.diag_embed(torch.stack([torch.ones_like(det), torch.ones_like(det), det], dim=-1))
    R = V @ D @ U.transpose(-1, -2)
    q_r = _dqcc_rot_to_quat(R)
    t = tm.squeeze(1) - torch.bmm(R, sm.transpose(-1, -2)).squeeze(-1)
    return q_r.reshape(*shape, 4), t.reshape(*shape, 3)


class MotionDQCC(Motion):
    """PMamba + full DQCC aux: anchor + cycle."""

    def __init__(self, num_classes, pts_size, anchor_weight=0.05,
                 cycle_weight=0.02, **kwargs):
        super().__init__(num_classes, pts_size, **kwargs)
        self.anchor_weight = anchor_weight
        self.cycle_weight = cycle_weight
        feat_dim = 64
        self.dq_head = nn.Sequential(
            nn.Linear(feat_dim * 2, feat_dim),
            nn.GELU(),
            nn.Linear(feat_dim, 8),
        )
        self.latest_aux_loss = None
        self.latest_aux_metrics = {}

    def get_auxiliary_loss(self):
        return self.latest_aux_loss

    def get_auxiliary_metrics(self):
        return self.latest_aux_metrics

    def _corr_sample(self, points, aux_input):
        B, F_, P, C = points.shape
        S = min(self.pts_size, P)
        device = points.device
        if S == P:
            return points, torch.ones(B, F_ - 1, P, dtype=torch.bool, device=device)
        orig_flat_idx = aux_input["orig_flat_idx"]
        corr_target = aux_input["corr_full_target_idx"]
        corr_weight = aux_input["corr_full_weight"]
        total_pts = corr_target.shape[-1]
        raw_ppf = total_pts // F_
        sampled = torch.zeros(B, F_, S, C, device=device, dtype=points.dtype)
        matched = torch.zeros(B, F_ - 1, S, dtype=torch.bool, device=device)
        for b in range(B):
            if self.training:
                idx = torch.randperm(P, device=device)[:S]
            else:
                idx = torch.linspace(0, P - 1, S, device=device).long()
            sampled[b, 0] = points[b, 0, idx]
            current_prov = orig_flat_idx[b, 0, idx].long()
            for t in range(F_ - 1):
                next_prov = orig_flat_idx[b, t + 1].long()
                reverse_map = torch.full((total_pts,), -1, dtype=torch.long, device=device)
                reverse_map[next_prov] = torch.arange(P, device=device)
                tgt_flat = corr_target[b, current_prov]
                tgt_w = corr_weight[b, current_prov]
                tgt_flat_safe = tgt_flat.clamp(min=0)
                tgt_frame = tgt_flat // raw_ppf
                tgt_pos = reverse_map[tgt_flat_safe]
                valid = ((tgt_flat >= 0) & (tgt_w > 0)
                         & (tgt_frame == t + 1) & (tgt_pos >= 0))
                next_idx = torch.randint(0, P, (S,), device=device)
                next_idx[valid] = tgt_pos[valid]
                sampled[b, t + 1] = points[b, t + 1, next_idx]
                matched[b, t] = valid
                current_prov = orig_flat_idx[b, t + 1, next_idx].long()
        return sampled, matched

    def extract_features(self, inputs):
        if isinstance(inputs, dict):
            points_raw = inputs["points"]
            aux = inputs
            has_corr = ("orig_flat_idx" in aux and "corr_full_target_idx" in aux
                        and "corr_full_weight" in aux)
        else:
            points_raw = inputs
            aux = None
            has_corr = False

        if has_corr:
            sampled, corr_matched = self._corr_sample(points_raw[..., :4], aux)
            coords = sampled.permute(0, 3, 1, 2).contiguous()
        else:
            coords = self._sample_points(points_raw)
            corr_matched = None

        batchsize, in_dims, timestep, pts_num = coords.shape

        ret_array1 = self.group.group_points(
            distance_dim=[0, 1, 2], array1=coords, array2=coords,
            knn=self.knn[0], dim=3,
        )
        ret_array1 = ret_array1.reshape(batchsize, in_dims, timestep * pts_num, -1)
        fea1_raw = self.pool1(self.stage1(ret_array1)).reshape(
            batchsize, -1, timestep, pts_num,
        )

        self.latest_aux_loss = None
        self.latest_aux_metrics = {}
        active = self.training and has_corr and (self.anchor_weight > 0 or self.cycle_weight > 0)
        if active:
            feat_frame = fea1_raw.mean(dim=-1).transpose(1, 2)      # (B, T, 64)
            T = feat_frame.shape[1]
            # Predict DQ per pair
            pair_in = torch.cat([feat_frame[:, :-1], feat_frame[:, 1:]], dim=-1)
            dq_pred = self.dq_head(pair_in)                         # (B, T-1, 8)
            q_r_pred = F.normalize(dq_pred[..., :4], dim=-1)
            q_d_pred = dq_pred[..., 4:]
            q_d_pred = q_d_pred - (q_d_pred * q_r_pred).sum(-1, keepdim=True) * q_r_pred

            # Observable DQ per pair
            with torch.no_grad():
                xyz_p = coords[:, :3].permute(0, 2, 3, 1)           # (B, T, P, 3)
                q_r_list, t_list = [], []
                for t in range(T - 1):
                    q_r, tr = _dqcc_kabsch_rt(xyz_p[:, t], xyz_p[:, t + 1],
                                              corr_matched[:, t])
                    q_r_list.append(q_r); t_list.append(tr)
                q_r_obs = torch.stack(q_r_list, dim=1)
                t_obs = torch.stack(t_list, dim=1)
                zero = torch.zeros_like(t_obs[..., :1])
                t_quat = torch.cat([zero, t_obs], dim=-1)
                q_d_obs = 0.5 * _dqcc_hamilton(t_quat, q_r_obs)

            # Sign-align q_r double cover
            cos_r = (q_r_pred * q_r_obs).sum(-1, keepdim=True)
            sign = torch.where(cos_r >= 0, torch.ones_like(cos_r), -torch.ones_like(cos_r))
            q_r_signed = q_r_pred * sign
            q_d_signed = q_d_pred * sign

            # Anchor loss
            anchor_loss = (F.mse_loss(q_r_signed, q_r_obs)
                           + F.mse_loss(q_d_signed, q_d_obs))

            # Cycle loss: compose all predicted + compare to cumulative observed
            cum_r_pred, cum_d_pred = q_r_signed[:, 0], q_d_signed[:, 0]
            cum_r_obs, cum_d_obs = q_r_obs[:, 0], q_d_obs[:, 0]
            for t in range(1, T - 1):
                cum_r_pred, cum_d_pred = _dqcc_dq_mul(
                    cum_r_pred, cum_d_pred, q_r_signed[:, t], q_d_signed[:, t])
                cum_r_pred = F.normalize(cum_r_pred, dim=-1)
                with torch.no_grad():
                    cum_r_obs, cum_d_obs = _dqcc_dq_mul(
                        cum_r_obs, cum_d_obs, q_r_obs[:, t], q_d_obs[:, t])
                    cum_r_obs = F.normalize(cum_r_obs, dim=-1)
            cycle_loss = (F.mse_loss(cum_r_pred, cum_r_obs)
                          + F.mse_loss(cum_d_pred, cum_d_obs))

            total = self.anchor_weight * anchor_loss + self.cycle_weight * cycle_loss
            self.latest_aux_loss = total
            self.latest_aux_metrics = {
                "qcc_raw": total.detach(),
                "qcc_forward": anchor_loss.detach(),
                "qcc_backward": cycle_loss.detach(),
                "qcc_valid_ratio": corr_matched.float().mean().detach(),
            }

        fea1 = torch.cat((coords, fea1_raw), dim=1)
        in_dims = fea1.shape[1] * 2 - 4
        pts_num //= self.downsample[0]
        rg2 = self.group.st_group_points(fea1, 3, [0, 1, 2], self.knn[1], 3)
        ret2, coords = self.select_ind(rg2, coords, batchsize, in_dims, timestep, pts_num)
        fea2 = self.pool2(self.stage2(ret2)).reshape(batchsize, -1, timestep, pts_num)
        fea2 = torch.cat((coords, fea2), dim=1)
        fea2 = self.multi_scale(fea2)

        in_dims = fea2.shape[1] * 2 - 4
        pts_num //= self.downsample[1]
        rg3 = self.group.st_group_points(fea2, 3, [0, 1, 2], self.knn[2], 3)
        ret3, coords = self.select_ind(rg3, coords, batchsize, in_dims, timestep, pts_num)
        fea3 = self.pool3(self.stage3(ret3)).reshape(batchsize, -1, timestep, pts_num)
        fea3_mamba = self.mamba(fea3)
        coords_fea3 = torch.cat((coords, fea3_mamba), dim=1)

        output = self.stage5(coords_fea3)
        output = self.pool5(output)
        output = self.global_bn(output)
        return output.flatten(1)

def _rr_kabsch(src, tgt, weights):
    """Kabsch batch. src, tgt: (B, N, 3). weights: (B, N). Returns R (B,3,3), t (B,3)."""
    B = src.shape[0]; device = src.device
    w = weights.float().clamp(min=0)
    w_sum = w.sum(-1, keepdim=True).clamp(min=1.0).unsqueeze(-1)
    sm = (src * w.unsqueeze(-1)).sum(1, keepdim=True) / w_sum
    tm = (tgt * w.unsqueeze(-1)).sum(1, keepdim=True) / w_sum
    sc = src - sm; tc = tgt - tm
    H = torch.einsum('bn,bni,bnj->bij', w, sc, tc)
    H = H + 1e-6 * torch.eye(3, device=device).unsqueeze(0)
    U, S, Vh = torch.linalg.svd(H)
    V = Vh.transpose(-1, -2)
    det = torch.det(V @ U.transpose(-1, -2))
    D = torch.diag_embed(torch.stack(
        [torch.ones_like(det), torch.ones_like(det), det], dim=-1))
    R = V @ D @ U.transpose(-1, -2)
    t = tm.squeeze(1) - torch.bmm(R, sm.transpose(-1, -2)).squeeze(-1)
    return R, t

def _rr_kabsch(src, tgt, weights):
    """Kabsch batch. src, tgt: (B, N, 3). weights: (B, N). Returns R (B,3,3), t (B,3)."""
    B = src.shape[0]; device = src.device
    w = weights.float().clamp(min=0)
    w_sum = w.sum(-1, keepdim=True).clamp(min=1.0).unsqueeze(-1)
    sm = (src * w.unsqueeze(-1)).sum(1, keepdim=True) / w_sum
    tm = (tgt * w.unsqueeze(-1)).sum(1, keepdim=True) / w_sum
    sc = src - sm; tc = tgt - tm
    H = torch.einsum('bn,bni,bnj->bij', w, sc, tc)
    H = H + 1e-6 * torch.eye(3, device=device).unsqueeze(0)
    U, S, Vh = torch.linalg.svd(H)
    V = Vh.transpose(-1, -2)
    det = torch.det(V @ U.transpose(-1, -2))
    D = torch.diag_embed(torch.stack(
        [torch.ones_like(det), torch.ones_like(det), det], dim=-1))
    R = V @ D @ U.transpose(-1, -2)
    t = tm.squeeze(1) - torch.bmm(R, sm.transpose(-1, -2)).squeeze(-1)
    return R, t


class MotionRigidRes(Motion):
    """PMamba + rigid-subtraction residual as extra 3-ch stage1 input."""

    def __init__(self, num_classes, pts_size, **kwargs):
        super().__init__(num_classes, pts_size, **kwargs)
        # 7-ch input for stage1 [xyz, res_xyz, t]
        self.stage1 = MLPBlock([7, 32, 64], 2)

    def _corr_sample(self, points, aux_input):
        B, F_, P, C = points.shape
        S = min(self.pts_size, P)
        device = points.device
        if S == P:
            return points, torch.ones(B, F_ - 1, P, dtype=torch.bool, device=device)
        orig_flat_idx = aux_input["orig_flat_idx"]
        corr_target = aux_input["corr_full_target_idx"]
        corr_weight = aux_input["corr_full_weight"]
        total_pts = corr_target.shape[-1]
        raw_ppf = total_pts // F_
        sampled = torch.zeros(B, F_, S, C, device=device, dtype=points.dtype)
        matched = torch.zeros(B, F_ - 1, S, dtype=torch.bool, device=device)
        for b in range(B):
            if self.training:
                idx = torch.randperm(P, device=device)[:S]
            else:
                idx = torch.linspace(0, P - 1, S, device=device).long()
            sampled[b, 0] = points[b, 0, idx]
            current_prov = orig_flat_idx[b, 0, idx].long()
            for t in range(F_ - 1):
                next_prov = orig_flat_idx[b, t + 1].long()
                reverse_map = torch.full((total_pts,), -1, dtype=torch.long, device=device)
                reverse_map[next_prov] = torch.arange(P, device=device)
                tgt_flat = corr_target[b, current_prov]
                tgt_w = corr_weight[b, current_prov]
                tgt_flat_safe = tgt_flat.clamp(min=0)
                tgt_frame = tgt_flat // raw_ppf
                tgt_pos = reverse_map[tgt_flat_safe]
                valid = ((tgt_flat >= 0) & (tgt_w > 0)
                         & (tgt_frame == t + 1) & (tgt_pos >= 0))
                next_idx = torch.randint(0, P, (S,), device=device)
                next_idx[valid] = tgt_pos[valid]
                sampled[b, t + 1] = points[b, t + 1, next_idx]
                matched[b, t] = valid
                current_prov = orig_flat_idx[b, t + 1, next_idx].long()
        return sampled, matched

    def extract_features(self, inputs):
        if isinstance(inputs, dict):
            points_raw = inputs["points"]
            aux = inputs
            has_corr = ("orig_flat_idx" in aux and "corr_full_target_idx" in aux
                        and "corr_full_weight" in aux)
        else:
            points_raw = inputs
            aux = None
            has_corr = False

        if has_corr:
            sampled, corr_matched = self._corr_sample(points_raw[..., :4], aux)
            coords = sampled.permute(0, 3, 1, 2).contiguous()
        else:
            coords = self._sample_points(points_raw)
            corr_matched = None

        batchsize, in_dims, timestep, pts_num = coords.shape
        xyz = coords[:, :3]                                  # (B, 3, T, P)
        time_ch = coords[:, 3:4]

        # Compute rigid-subtraction residuals per-frame (needs correspondence)
        with torch.no_grad():
            res = torch.zeros_like(xyz)                       # (B, 3, T, P)
            if has_corr and corr_matched is not None:
                xyz_p = xyz.permute(0, 2, 3, 1)               # (B, T, P, 3)
                for t in range(timestep - 1):
                    src = xyz_p[:, t]                         # (B, P, 3)
                    tgt = xyz_p[:, t + 1]
                    w = corr_matched[:, t].float()
                    R, tr = _rr_kabsch(src, tgt, w)           # (B,3,3), (B,3)
                    rigid_pred = torch.bmm(R, src.transpose(-1, -2)).transpose(-1, -2) + tr.unsqueeze(1)
                    r = tgt - rigid_pred                      # (B, P, 3)
                    res[:, :, t + 1] = r.permute(0, 2, 1)     # back to (B,3,P)
                # frame 0 residual stays zero

        coords7 = torch.cat([xyz, res, time_ch], dim=1)       # (B, 7, T, P)

        ret_array1 = self.group.group_points(
            distance_dim=[0, 1, 2], array1=coords7, array2=coords7,
            knn=self.knn[0], dim=3,
        )
        ret_array1 = ret_array1.reshape(batchsize, 7, timestep * pts_num, -1)
        fea1 = self.pool1(self.stage1(ret_array1)).reshape(
            batchsize, -1, timestep, pts_num,
        )
        fea1 = torch.cat((coords, fea1), dim=1)               # use 4-ch coords thereafter

        in_dims = fea1.shape[1] * 2 - 4
        pts_num //= self.downsample[0]
        rg2 = self.group.st_group_points(fea1, 3, [0, 1, 2], self.knn[1], 3)
        ret2, coords = self.select_ind(rg2, coords, batchsize, in_dims, timestep, pts_num)
        fea2 = self.pool2(self.stage2(ret2)).reshape(batchsize, -1, timestep, pts_num)
        fea2 = torch.cat((coords, fea2), dim=1)
        fea2 = self.multi_scale(fea2)

        in_dims = fea2.shape[1] * 2 - 4
        pts_num //= self.downsample[1]
        rg3 = self.group.st_group_points(fea2, 3, [0, 1, 2], self.knn[2], 3)
        ret3, coords = self.select_ind(rg3, coords, batchsize, in_dims, timestep, pts_num)
        fea3 = self.pool3(self.stage3(ret3)).reshape(batchsize, -1, timestep, pts_num)
        fea3_mamba = self.mamba(fea3)
        coords_fea3 = torch.cat((coords, fea3_mamba), dim=1)

        output = self.stage5(coords_fea3)
        output = self.pool5(output)
        output = self.global_bn(output)
        return output.flatten(1)

def _rrfbq_hamilton(a, b):
    aw,ax,ay,az = a[...,0], a[...,1], a[...,2], a[...,3]
    bw,bx,by,bz = b[...,0], b[...,1], b[...,2], b[...,3]
    return torch.stack([
        aw*bw - ax*bx - ay*by - az*bz,
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw,
    ], dim=-1)


def _rrfbq_rot_to_quat(R):
    """Batched R (B,3,3) -> q (B,4) via Shepperd's branchless trick."""
    B = R.shape[0]
    m = R
    tr = m[:,0,0] + m[:,1,1] + m[:,2,2]
    q = torch.zeros(B, 4, device=R.device, dtype=R.dtype)
    # Use tr>0 branch for the majority; handle edge via fallback
    s = torch.sqrt(tr.clamp(min=-0.999) + 1.0) * 2.0
    s = s.clamp(min=1e-6)
    q[:,0] = 0.25 * s
    q[:,1] = (m[:,2,1] - m[:,1,2]) / s
    q[:,2] = (m[:,0,2] - m[:,2,0]) / s
    q[:,3] = (m[:,1,0] - m[:,0,1]) / s
    # Fallback for tr<=0 rows
    bad = tr <= 0
    if bad.any():
        idx = bad.nonzero(as_tuple=True)[0]
        for i in idx.tolist():
            mi = m[i]
            if (mi[0,0] > mi[1,1]) and (mi[0,0] > mi[2,2]):
                s2 = torch.sqrt(1 + mi[0,0] - mi[1,1] - mi[2,2]).clamp(min=1e-6) * 2
                q[i,0] = (mi[2,1] - mi[1,2]) / s2
                q[i,1] = 0.25 * s2
                q[i,2] = (mi[0,1] + mi[1,0]) / s2
                q[i,3] = (mi[0,2] + mi[2,0]) / s2
            elif mi[1,1] > mi[2,2]:
                s2 = torch.sqrt(1 + mi[1,1] - mi[0,0] - mi[2,2]).clamp(min=1e-6) * 2
                q[i,0] = (mi[0,2] - mi[2,0]) / s2
                q[i,1] = (mi[0,1] + mi[1,0]) / s2
                q[i,2] = 0.25 * s2
                q[i,3] = (mi[1,2] + mi[2,1]) / s2
            else:
                s2 = torch.sqrt(1 + mi[2,2] - mi[0,0] - mi[1,1]).clamp(min=1e-6) * 2
                q[i,0] = (mi[1,0] - mi[0,1]) / s2
                q[i,1] = (mi[0,2] + mi[2,0]) / s2
                q[i,2] = (mi[1,2] + mi[2,1]) / s2
                q[i,3] = 0.25 * s2
    q = torch.nn.functional.normalize(q, dim=-1)
    # hemisphere pin
    sign = torch.where(q[:,0:1] < 0, -torch.ones_like(q[:,0:1]), torch.ones_like(q[:,0:1]))
    return q * sign


def _rrfbq_quat_rotate(q, points):
    """q: (B,4) unit, points: (B,N,3) -> (B,N,3) via sandwich product."""
    B, N, _ = points.shape
    q_b = q.unsqueeze(1).expand(B, N, 4)
    pq = torch.cat([torch.zeros(B, N, 1, device=points.device, dtype=points.dtype), points], dim=-1)
    q_conj = torch.cat([q_b[...,0:1], -q_b[...,1:]], dim=-1)
    return _rrfbq_hamilton(_rrfbq_hamilton(q_b, pq), q_conj)[...,1:]


def _rrfbq_kabsch_quat(src, tgt, weights):
    """src,tgt: (B,N,3) weights: (B,N). Returns q (B,4), t (B,3) via quaternion."""
    device = src.device
    w = weights.float().clamp(min=0)
    w_sum = w.sum(-1, keepdim=True).clamp(min=1.0).unsqueeze(-1)
    sm = (src * w.unsqueeze(-1)).sum(1, keepdim=True) / w_sum  # (B,1,3)
    tm = (tgt * w.unsqueeze(-1)).sum(1, keepdim=True) / w_sum
    sc = src - sm; tc = tgt - tm
    H = torch.einsum('bn,bni,bnj->bij', w, sc, tc)
    H = H + 1e-6 * torch.eye(3, device=device).unsqueeze(0)
    U, S, Vh = torch.linalg.svd(H)
    V = Vh.transpose(-1, -2)
    det = torch.det(V @ U.transpose(-1, -2))
    D = torch.diag_embed(torch.stack(
        [torch.ones_like(det), torch.ones_like(det), det], dim=-1))
    R = V @ D @ U.transpose(-1, -2)
    q = _rrfbq_rot_to_quat(R)                                    # (B,4)
    rot_sm = _rrfbq_quat_rotate(q, sm)                           # (B,1,3)
    t = tm.squeeze(1) - rot_sm.squeeze(1)                        # (B,3)
    return q, t


class MotionRigidResFBQ(Motion):
    """PMamba + fwd+bwd quaternion Kabsch residuals as extra 6-ch stage1 input."""

    def __init__(self, num_classes, pts_size, **kwargs):
        super().__init__(num_classes, pts_size, **kwargs)
        # 10-ch input for stage1 [xyz(3), res_fwd(3), res_bwd(3), t(1)]
        self.stage1 = MLPBlock([10, 32, 64], 2)

    def _corr_sample(self, points, aux_input):
        B, F_, P, C = points.shape
        S = min(self.pts_size, P)
        device = points.device
        if S == P:
            return points, torch.ones(B, F_ - 1, P, dtype=torch.bool, device=device)
        orig_flat_idx = aux_input["orig_flat_idx"]
        corr_target = aux_input["corr_full_target_idx"]
        corr_weight = aux_input["corr_full_weight"]
        total_pts = corr_target.shape[-1]
        raw_ppf = total_pts // F_
        sampled = torch.zeros(B, F_, S, C, device=device, dtype=points.dtype)
        matched = torch.zeros(B, F_ - 1, S, dtype=torch.bool, device=device)
        for b in range(B):
            if self.training:
                idx = torch.randperm(P, device=device)[:S]
            else:
                idx = torch.linspace(0, P - 1, S, device=device).long()
            sampled[b, 0] = points[b, 0, idx]
            current_prov = orig_flat_idx[b, 0, idx].long()
            for t in range(F_ - 1):
                next_prov = orig_flat_idx[b, t + 1].long()
                reverse_map = torch.full((total_pts,), -1, dtype=torch.long, device=device)
                reverse_map[next_prov] = torch.arange(P, device=device)
                tgt_flat = corr_target[b, current_prov]
                tgt_w = corr_weight[b, current_prov]
                tgt_flat_safe = tgt_flat.clamp(min=0)
                tgt_frame = tgt_flat // raw_ppf
                tgt_pos = reverse_map[tgt_flat_safe]
                valid = ((tgt_flat >= 0) & (tgt_w > 0)
                         & (tgt_frame == t + 1) & (tgt_pos >= 0))
                next_idx = torch.randint(0, P, (S,), device=device)
                next_idx[valid] = tgt_pos[valid]
                sampled[b, t + 1] = points[b, t + 1, next_idx]
                matched[b, t] = valid
                current_prov = orig_flat_idx[b, t + 1, next_idx].long()
        return sampled, matched

    def extract_features(self, inputs):
        if isinstance(inputs, dict):
            points_raw = inputs["points"]
            aux = inputs
            has_corr = ("orig_flat_idx" in aux and "corr_full_target_idx" in aux
                        and "corr_full_weight" in aux)
        else:
            points_raw = inputs
            aux = None
            has_corr = False

        if has_corr:
            sampled, corr_matched = self._corr_sample(points_raw[..., :4], aux)
            coords = sampled.permute(0, 3, 1, 2).contiguous()
        else:
            coords = self._sample_points(points_raw)
            corr_matched = None

        batchsize, in_dims, timestep, pts_num = coords.shape
        xyz = coords[:, :3]
        time_ch = coords[:, 3:4]

        with torch.no_grad():
            res_fwd = torch.zeros_like(xyz)
            res_bwd = torch.zeros_like(xyz)
            if has_corr and corr_matched is not None:
                xyz_p = xyz.permute(0, 2, 3, 1)                       # (B,T,P,3)
                for t in range(timestep - 1):
                    src = xyz_p[:, t]
                    tgt = xyz_p[:, t + 1]
                    w = corr_matched[:, t].float()
                    # forward: src->tgt, residual stored at t+1
                    q_f, tr_f = _rrfbq_kabsch_quat(src, tgt, w)
                    rigid_f = _rrfbq_quat_rotate(q_f, src) + tr_f.unsqueeze(1)
                    rf = tgt - rigid_f
                    res_fwd[:, :, t + 1] = rf.permute(0, 2, 1)
                    # backward: tgt->src, residual stored at t
                    q_b, tr_b = _rrfbq_kabsch_quat(tgt, src, w)
                    rigid_b = _rrfbq_quat_rotate(q_b, tgt) + tr_b.unsqueeze(1)
                    rb = src - rigid_b
                    res_bwd[:, :, t] = rb.permute(0, 2, 1)
                # res_fwd[0] and res_bwd[T-1] stay zero

        coords10 = torch.cat([xyz, res_fwd, res_bwd, time_ch], dim=1)

        ret_array1 = self.group.group_points(
            distance_dim=[0, 1, 2], array1=coords10, array2=coords10,
            knn=self.knn[0], dim=3,
        )
        ret_array1 = ret_array1.reshape(batchsize, 10, timestep * pts_num, -1)
        fea1 = self.pool1(self.stage1(ret_array1)).reshape(
            batchsize, -1, timestep, pts_num,
        )
        fea1 = torch.cat((coords, fea1), dim=1)

        in_dims = fea1.shape[1] * 2 - 4
        pts_num //= self.downsample[0]
        rg2 = self.group.st_group_points(fea1, 3, [0, 1, 2], self.knn[1], 3)
        ret2, coords = self.select_ind(rg2, coords, batchsize, in_dims, timestep, pts_num)
        fea2 = self.pool2(self.stage2(ret2)).reshape(batchsize, -1, timestep, pts_num)
        fea2 = torch.cat((coords, fea2), dim=1)
        fea2 = self.multi_scale(fea2)

        in_dims = fea2.shape[1] * 2 - 4
        pts_num //= self.downsample[1]
        rg3 = self.group.st_group_points(fea2, 3, [0, 1, 2], self.knn[2], 3)
        ret3, coords = self.select_ind(rg3, coords, batchsize, in_dims, timestep, pts_num)
        fea3 = self.pool3(self.stage3(ret3)).reshape(batchsize, -1, timestep, pts_num)
        fea3_mamba = self.mamba(fea3)
        coords_fea3 = torch.cat((coords, fea3_mamba), dim=1)

        output = self.stage5(coords_fea3)
        output = self.pool5(output)
        output = self.global_bn(output)
        return output.flatten(1)

class MotionRigidStabilize(Motion):
    """PMamba over rigid-motion-removed pointcloud.

    Replaces xyz with frame-0-referenced (stabilized) xyz before PMamba.
    Reuses _rrfbq_* quaternion math from MotionRigidResFBQ.
    Stage1 input is 4-ch [xyz_stable, t] — vanilla PMamba shape.
    """

    def _corr_sample(self, points, aux_input):
        # Same correspondence-aware sampler as MotionRigidResFBQ
        B, F_, P, C = points.shape
        S = min(self.pts_size, P)
        device = points.device
        if S == P:
            return points, torch.ones(B, F_ - 1, P, dtype=torch.bool, device=device)
        orig_flat_idx = aux_input["orig_flat_idx"]
        corr_target = aux_input["corr_full_target_idx"]
        corr_weight = aux_input["corr_full_weight"]
        total_pts = corr_target.shape[-1]
        raw_ppf = total_pts // F_
        sampled = torch.zeros(B, F_, S, C, device=device, dtype=points.dtype)
        matched = torch.zeros(B, F_ - 1, S, dtype=torch.bool, device=device)
        for b in range(B):
            if self.training:
                idx = torch.randperm(P, device=device)[:S]
            else:
                idx = torch.linspace(0, P - 1, S, device=device).long()
            sampled[b, 0] = points[b, 0, idx]
            current_prov = orig_flat_idx[b, 0, idx].long()
            for t in range(F_ - 1):
                next_prov = orig_flat_idx[b, t + 1].long()
                reverse_map = torch.full((total_pts,), -1, dtype=torch.long, device=device)
                reverse_map[next_prov] = torch.arange(P, device=device)
                tgt_flat = corr_target[b, current_prov]
                tgt_w = corr_weight[b, current_prov]
                tgt_flat_safe = tgt_flat.clamp(min=0)
                tgt_frame = tgt_flat // raw_ppf
                tgt_pos = reverse_map[tgt_flat_safe]
                valid = ((tgt_flat >= 0) & (tgt_w > 0)
                         & (tgt_frame == t + 1) & (tgt_pos >= 0))
                next_idx = torch.randint(0, P, (S,), device=device)
                next_idx[valid] = tgt_pos[valid]
                sampled[b, t + 1] = points[b, t + 1, next_idx]
                matched[b, t] = valid
                current_prov = orig_flat_idx[b, t + 1, next_idx].long()
        return sampled, matched

    def _stabilize(self, xyz_p, matched):
        """xyz_p: (B,T,P,3), matched: (B,T-1,P) -> stabilized (B,T,P,3) in frame-0 coords."""
        B, T, P, _ = xyz_p.shape
        device = xyz_p.device
        # accumulated transform (q_t, T_t): xyz[t] = quat_rot(q_t, xyz[0]) + T_t
        q_acc = torch.zeros(B, 4, device=device, dtype=xyz_p.dtype)
        q_acc[:, 0] = 1.0  # identity quat (1,0,0,0)
        T_acc = torch.zeros(B, 3, device=device, dtype=xyz_p.dtype)
        out = torch.zeros_like(xyz_p)
        out[:, 0] = xyz_p[:, 0]  # frame 0 anchor
        for t in range(T - 1):
            src = xyz_p[:, t]
            tgt = xyz_p[:, t + 1]
            w = matched[:, t].float()
            # If a sample has too few matches, kabsch fit is unreliable
            n_match = w.sum(-1)  # (B,)
            ok = n_match >= 4  # need >=4 points for stable fit
            q_pair, tr_pair = _rrfbq_kabsch_quat(src, tgt, w)
            # Compose: q_{t+1} = q_pair * q_acc;  T_{t+1} = rot(q_pair, T_acc) + tr_pair
            q_new = _rrfbq_hamilton(q_pair, q_acc)
            q_new = torch.nn.functional.normalize(q_new, dim=-1)
            T_new = _rrfbq_quat_rotate(q_pair, T_acc.unsqueeze(1)).squeeze(1) + tr_pair
            # Fallback for unreliable rows: keep prev accumulated transform
            ok3 = ok.float().unsqueeze(-1)
            q_acc = ok3 * q_new + (1 - ok3) * q_acc
            q_acc = torch.nn.functional.normalize(q_acc, dim=-1)
            T_acc = ok3 * T_new + (1 - ok3) * T_acc
            # Apply inverse to map xyz[t+1] back into frame-0 coords
            q_conj = torch.cat([q_acc[:, 0:1], -q_acc[:, 1:]], dim=-1)
            shifted = xyz_p[:, t + 1] - T_acc.unsqueeze(1)
            stabilized = _rrfbq_quat_rotate(q_conj, shifted)
            out[:, t + 1] = stabilized
        return out

    def extract_features(self, inputs):
        if isinstance(inputs, dict):
            points_raw = inputs["points"]
            aux = inputs
            has_corr = ("orig_flat_idx" in aux and "corr_full_target_idx" in aux
                        and "corr_full_weight" in aux)
        else:
            points_raw = inputs
            aux = None
            has_corr = False

        if has_corr:
            sampled, corr_matched = self._corr_sample(points_raw[..., :4], aux)
            coords = sampled.permute(0, 3, 1, 2).contiguous()
        else:
            coords = self._sample_points(points_raw)
            corr_matched = None

        batchsize, in_dims, timestep, pts_num = coords.shape
        xyz = coords[:, :3]                                          # (B,3,T,P)
        time_ch = coords[:, 3:4]                                     # (B,1,T,P)

        with torch.no_grad():
            if has_corr and corr_matched is not None:
                xyz_p = xyz.permute(0, 2, 3, 1).contiguous()         # (B,T,P,3)
                xyz_stab = self._stabilize(xyz_p, corr_matched)       # (B,T,P,3)
                xyz = xyz_stab.permute(0, 3, 1, 2).contiguous()       # (B,3,T,P)

        coords4 = torch.cat([xyz, time_ch], dim=1)
        coords = coords4  # downstream stages key xyz/time off coords[:, :4]

        ret_array1 = self.group.group_points(
            distance_dim=[0, 1, 2], array1=coords4, array2=coords4,
            knn=self.knn[0], dim=3,
        )
        ret_array1 = ret_array1.reshape(batchsize, 4, timestep * pts_num, -1)
        fea1 = self.pool1(self.stage1(ret_array1)).reshape(
            batchsize, -1, timestep, pts_num,
        )
        fea1 = torch.cat((coords, fea1), dim=1)

        in_dims = fea1.shape[1] * 2 - 4
        pts_num //= self.downsample[0]
        rg2 = self.group.st_group_points(fea1, 3, [0, 1, 2], self.knn[1], 3)
        ret2, coords = self.select_ind(rg2, coords, batchsize, in_dims, timestep, pts_num)
        fea2 = self.pool2(self.stage2(ret2)).reshape(batchsize, -1, timestep, pts_num)
        fea2 = torch.cat((coords, fea2), dim=1)
        fea2 = self.multi_scale(fea2)

        in_dims = fea2.shape[1] * 2 - 4
        pts_num //= self.downsample[1]
        rg3 = self.group.st_group_points(fea2, 3, [0, 1, 2], self.knn[2], 3)
        ret3, coords = self.select_ind(rg3, coords, batchsize, in_dims, timestep, pts_num)
        fea3 = self.pool3(self.stage3(ret3)).reshape(batchsize, -1, timestep, pts_num)
        fea3_mamba = self.mamba(fea3)
        coords_fea3 = torch.cat((coords, fea3_mamba), dim=1)

        output = self.stage5(coords_fea3)
        output = self.pool5(output)
        output = self.global_bn(output)
        return output.flatten(1)


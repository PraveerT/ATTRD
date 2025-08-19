import pdb
import torch
import torch.nn as nn
import numpy as np
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
        
        
        # Feature weighting gate
        self.feature_gate = nn.Sequential(
            nn.Linear(feature_dim * num_scales, num_scales),
            nn.Softmax(dim=-1)
        )
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Conv2d(feature_dim * num_scales + in_channels, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.GELU(),
            nn.Dropout(0.2)
        )
        
    
    def forward(self, x):
        # x shape: B, C, T, N
        B, C, T, N = x.shape
        
        # 1. Multi-scale feature extraction
        scale_features = []
        for i, filter in enumerate(self.scale_filters):
            # Apply scale-specific filter
            scale_feat = filter(x)
            scale_features.append(scale_feat)
        
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
        
        # 3. Apply feature weighting
        all_features = []
        for i, feat in enumerate(interacted_features):
            # Upsample to original resolution
            upsampled = F.interpolate(feat, size=(T, N), mode='bilinear', align_corners=False)
            all_features.append(upsampled)
        
        feature_stack = torch.stack(all_features, dim=2)  # (B, feature_dim, num_scales, T, N)
        feature_flat = feature_stack.permute(0, 3, 4, 1, 2).reshape(B * T * N, self.feature_dim, self.num_scales)
        
        # Feature weights
        feature_weights = self.feature_gate(feature_flat.reshape(B * T * N, -1))
        feature_weights = feature_weights.unsqueeze(1)  # (B*T*N, 1, num_scales)
        
        # Apply weighting
        weighted_features = (feature_flat * feature_weights).sum(dim=2)  # (B*T*N, feature_dim)
        weighted_features = weighted_features.reshape(B, T, N, self.feature_dim).permute(0, 3, 1, 2)
        
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
    def __init__(self, in_channels, hidden_dim, output_dim=None, num_layers=2, drop_path=0.1):
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
        for i, (mamba, norm) in enumerate(zip(self.mamba_layers, self.norms)):
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
                 knn=(16, 48, 48, 24)):
        super(Motion, self).__init__()
        self.stage1 = MLPBlock([4, 32, 64], 2)
        self.pool1 = nn.AdaptiveMaxPool2d((None, 1))
        self.stage2 = MotionBlock([128, 128, ], 2, 4)
        self.pool2 = nn.AdaptiveMaxPool2d((None, 1))
        self.stage3 = MotionBlock([256, 256, ], 2, 4)
        self.pool3 = nn.AdaptiveMaxPool2d((None, 1))
        # Stage 4 removed to reduce overfitting and improve efficiency
        self.stage5 = MLPBlock([260, 1024], 2)
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
        self.mamba = MambaTemporalEncoder(in_channels=256, hidden_dim=128, output_dim=256, num_layers=2, drop_path=0.1)
        
        # Add Multi-scale Feature Processor layer after stage2
        self.multi_scale = MultiScaleFeatureProcessor(in_channels=132, num_scales=4, feature_dim=32)
        
        # Temporal noise injection for regularization during training
        self.temporal_noise_std = 0.02
        
        # Add new convolution branch
        self.conv_branch = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, num_classes)
        )
        
        # Add gating mechanism for branch fusion
        self.fusion_gate = nn.Sequential(
            nn.Linear(1024 + num_classes, 512),
            nn.GELU(),
            nn.Linear(512, num_classes),
            nn.Sigmoid()
        )

    def forward(self, inputs):
        # B * T * N * D,  e.g. 16 * 32 * 512 * 4
        inputs = inputs.permute(0, 3, 1, 2)
        # Store original inputs for convolution branch
        inputs_raw = inputs.clone()
        if self.training:
            # Random sampling during training for augmentation
            indices = torch.randperm(inputs.shape[3])[:self.pts_size]
        else:
            # Deterministic sampling during testing for consistent results
            indices = torch.linspace(0, inputs.shape[3]-1, self.pts_size, dtype=torch.long)
        inputs = inputs[:, :, :, indices]
        # B * (4 + others) * 32 * 128
        inputs = inputs[:, :4]
        # B * 4 * 32 * 128
        batchsize, in_dims, timestep, pts_num = inputs.shape

        # stage 1: intra-frame
        ret_array1 = self.group.group_points(distance_dim=[0, 1, 2], array1=inputs, array2=inputs, knn=self.knn[0],
                                             dim=3)
        # B * 4 * 32 * 128 * 16
        ret_array1 = ret_array1.contiguous().view(batchsize, in_dims, timestep * pts_num, -1)
        # B * 4 * 4096 * 16
        fea1 = self.pool1(self.stage1(ret_array1)).view(batchsize, -1, timestep, pts_num)
        # B * 64 * 32 * 128
        fea1 = torch.cat((inputs, fea1), dim=1)
        # B * 68 * 32 * 128

        # stage 2: inter-frame, early
        in_dims = fea1.shape[1] * 2 - 4
        pts_num //= self.downsample[0]
        ret_group_array2 = self.group.st_group_points(fea1, 3, [0, 1, 2], self.knn[1], 3)
        ret_array2, inputs, _ = self.select_ind(ret_group_array2, inputs,
                                                batchsize, in_dims, timestep, pts_num)
        fea2 = self.pool2(self.stage2(ret_array2)).view(batchsize, -1, timestep, pts_num)
        fea2 = torch.cat((inputs, fea2), dim=1)
        
        # Apply multi-scale feature processing
        fea2 = self.multi_scale(fea2)
        
        # Apply temporal noise injection during training for regularization
        if self.training:
            noise = torch.randn_like(fea2) * self.temporal_noise_std
            fea2 = fea2 + noise

        # stage 3: inter-frame, middle, applying mamba in this stage
        in_dims = fea2.shape[1] * 2 - 4
        pts_num //= self.downsample[1]
        ret_group_array3 = self.group.st_group_points(fea2, 3, [0, 1, 2], self.knn[2], 3)
        ret_array3, inputs, ind = self.select_ind(ret_group_array3, inputs,
                                                  batchsize, in_dims, timestep, pts_num)
        fea3 = self.pool3(self.stage3(ret_array3)).view(batchsize, -1, timestep, pts_num)
        # Apply Mamba temporal modeling after spatial processing
        fea3_mamba = self.mamba(fea3)
        # Concatenate with inputs for next stage
        fea3 = torch.cat((inputs, fea3_mamba), dim=1)

        # Stage 4 removed - direct connection from Stage 3+Mamba to Stage 5
        # fea3 shape: (batchsize, features, timestep, 32 points)
        
        output = self.stage5(fea3)
        output = self.pool5(output)
        output = self.global_bn(output)
        main_features = output.view(batchsize, -1)  # This should be [batchsize, 1024]
        output = self.stage6(output)
        
        # Get main branch output
        main_output = output.view(batchsize, self.num_classes)
        
        # Process input through convolution branch (using only first 4 channels)
        conv_output = self.conv_branch(inputs_raw[:, :4])
        
        # Gated fusion of both branches
        gate_input = torch.cat([main_features, conv_output], dim=1)
        gate_weight = self.fusion_gate(gate_input)
        combined_output = gate_weight * main_output + (1 - gate_weight) * conv_output
        
        # For debugging/analysis, we can return individual branch outputs
        # In normal operation, we just return the combined output
        if hasattr(self, 'return_branch_outputs') and self.return_branch_outputs:
            return combined_output, main_output, conv_output, gate_weight
        else:
            return combined_output

    def select_ind(self, group_array, inputs, batchsize, in_dim, timestep, pts_num):
        ind = self.weight_select(group_array, pts_num)
        ret_group_array = group_array.gather(-2, ind.unsqueeze(1).unsqueeze(-1).
                                             expand(-1, group_array.shape[1], -1, -1,
                                                    group_array.shape[-1]))
        ret_group_array = ret_group_array.view(batchsize, in_dim, timestep * pts_num, -1)
        inputs = inputs.gather(-1, ind.unsqueeze(1).expand(-1, inputs.shape[1], -1, -1))
        return ret_group_array, inputs, ind

    @staticmethod
    def weight_select(position, topk):
        # select points with larger ranges
        weights = torch.max(torch.sum(position[:, :3] ** 2, dim=1), dim=-1)[0]
        dists, idx = torch.topk(weights, topk, -1, largest=True, sorted=False)
        return idx


if __name__ == '__main__':
    pass

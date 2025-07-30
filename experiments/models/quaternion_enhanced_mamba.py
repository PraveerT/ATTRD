import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba
import math
from typing import Optional, Tuple


class QuaternionLinear(nn.Module):
    """Quaternion linear transformation for rotation-equivariant features."""
    
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features // 4
        self.out_features = out_features // 4
        
        # Quaternion components: real, i, j, k
        self.weight_r = nn.Parameter(torch.Tensor(self.out_features, self.in_features))
        self.weight_i = nn.Parameter(torch.Tensor(self.out_features, self.in_features))
        self.weight_j = nn.Parameter(torch.Tensor(self.out_features, self.in_features))
        self.weight_k = nn.Parameter(torch.Tensor(self.out_features, self.in_features))
        
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight_r)
        nn.init.xavier_uniform_(self.weight_i)
        nn.init.xavier_uniform_(self.weight_j)
        nn.init.xavier_uniform_(self.weight_k)
        nn.init.zeros_(self.bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Split input into quaternion components
        B, T, C = x.shape
        x = x.view(B, T, 4, self.in_features)
        x_r, x_i, x_j, x_k = x[:, :, 0], x[:, :, 1], x[:, :, 2], x[:, :, 3]
        
        # Quaternion multiplication (Hamilton product)
        # (a + bi + cj + dk)(e + fi + gj + hk) = 
        # (ae - bf - cg - dh) + (af + be + ch - dg)i + (ag - bh + ce + df)j + (ah + bg - cf + de)k
        
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
        out = out.view(B, T, -1) + self.bias
        
        return out


class LocalNormPooling(nn.Module):
    """Local Norm Pooling (LNP) block for enhanced local feature extraction."""
    
    def __init__(self, in_channels: int, pool_size: int = 8, norm_type: str = 'batch'):
        super().__init__()
        self.pool_size = pool_size
        self.in_channels = in_channels
        
        # Local feature transformation with same padding
        self.local_conv = nn.Conv1d(in_channels, in_channels, kernel_size=pool_size, 
                                   stride=1, padding='same', groups=in_channels)
        
        # Normalization
        if norm_type == 'batch':
            self.norm = nn.BatchNorm1d(in_channels)
        elif norm_type == 'layer':
            self.norm = nn.LayerNorm(in_channels)
        else:
            self.norm = nn.Identity()
            
        # Feature mixing
        self.mix = nn.Conv1d(in_channels, in_channels, kernel_size=1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C]
        B, T, C = x.shape
        
        # Apply local convolution
        x_local = x.transpose(1, 2)  # [B, C, T]
        x_local = self.local_conv(x_local)
        
        # Ensure output has same temporal dimension
        if x_local.shape[2] != T:
            x_local = F.interpolate(x_local, size=T, mode='linear', align_corners=False)
        
        # Compute local norms
        x_norm = torch.norm(x_local, p=2, dim=1, keepdim=True)  # [B, 1, T]
        x_normalized = x_local / (x_norm + 1e-6)
        
        # Apply normalization and mixing
        x_normalized = self.norm(x_normalized)
        x_mixed = self.mix(x_normalized)
        
        # Residual connection
        out = x_mixed.transpose(1, 2) + x  # [B, T, C]
        
        return out


class BidirectionalQuaternionMamba(nn.Module):
    """Bidirectional SSM with quaternion features for rotation equivariance."""
    
    def __init__(self, d_model: int, d_state: int = 16, expand: int = 2):
        super().__init__()
        self.d_model = d_model
        
        # Forward and backward Mamba blocks
        self.forward_mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=4,
            expand=expand
        )
        
        self.backward_mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=4,
            expand=expand
        )
        
        # Quaternion projection layers
        self.quat_proj_in = QuaternionLinear(d_model, d_model)
        self.quat_proj_out = QuaternionLinear(d_model, d_model)
        
        # Feature channel SSM for backward pass
        self.channel_mamba = Mamba(
            d_model=d_model,
            d_state=d_state//2,
            d_conv=2,
            expand=1
        )
        
        # Fusion layer (adjusted for 2 pathways instead of 3)
        self.fusion = nn.Linear(d_model * 2, d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        
        # Apply quaternion projection
        x_quat = self.quat_proj_in(x)
        
        # Forward pass
        x_forward = self.forward_mamba(x_quat)
        
        # Backward pass
        x_backward = torch.flip(x_quat, dims=[1])
        x_backward = self.backward_mamba(x_backward)
        x_backward = torch.flip(x_backward, dims=[1])
        
        # Skip channel-wise SSM to save memory - just use forward and backward
        # Fuse forward and backward pathways
        x_combined = torch.cat([x_forward, x_backward], dim=-1)
        x_fused = self.fusion(x_combined)
        
        # Final quaternion projection
        out = self.quat_proj_out(x_fused)
        
        return out


class MultiScaleTemporalBlock(nn.Module):
    """Multi-scale temporal processing with different frame rates."""
    
    def __init__(self, d_model: int, scales: list = [1, 2, 4, 8]):
        super().__init__()
        self.scales = scales
        self.d_model = d_model
        
        # Mamba blocks for each scale
        self.scale_mambas = nn.ModuleList([
            BidirectionalQuaternionMamba(d_model) for _ in scales
        ])
        
        # Scale-specific projections
        self.scale_projs = nn.ModuleList([
            nn.Linear(d_model, d_model) for _ in scales
        ])
        
        # Adaptive fusion
        self.fusion_weights = nn.Parameter(torch.ones(len(scales)) / len(scales))
        self.fusion_proj = nn.Linear(d_model * len(scales), d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        
        multi_scale_features = []
        
        for i, (scale, mamba, proj) in enumerate(zip(self.scales, self.scale_mambas, self.scale_projs)):
            if scale > 1:
                # Downsample temporally
                x_scaled = x[:, ::scale, :]
                T_scaled = x_scaled.shape[1]
                
                # Process at this scale
                x_processed = mamba(x_scaled)
                x_processed = proj(x_processed)
                
                # Upsample back to original resolution
                x_upsampled = F.interpolate(
                    x_processed.transpose(1, 2), 
                    size=T, 
                    mode='linear', 
                    align_corners=False
                ).transpose(1, 2)
            else:
                # Process at original scale
                x_processed = mamba(x)
                x_upsampled = proj(x_processed)
            
            # Weight by learnable scale importance
            x_weighted = x_upsampled * self.fusion_weights[i]
            multi_scale_features.append(x_weighted)
        
        # Concatenate and fuse
        x_concat = torch.cat(multi_scale_features, dim=-1)
        out = self.fusion_proj(x_concat)
        
        return out


class QuaternionEnhancedMambaEncoder(nn.Module):
    """Main encoder combining all components."""
    
    def __init__(
        self, 
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 4,
        d_state: int = 16,
        scales: list = [1, 2, 4, 8],
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Ensure hidden_dim is divisible by 4 for quaternion
        hidden_dim = (hidden_dim // 4) * 4
        self.hidden_dim = hidden_dim
        
        # Input projection to quaternion space
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Stack of encoder layers
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            layer = nn.ModuleDict({
                'lnp': LocalNormPooling(hidden_dim, pool_size=8),
                'mamba': BidirectionalQuaternionMamba(hidden_dim, d_state),
                'multiscale': MultiScaleTemporalBlock(hidden_dim, scales),
                'norm1': nn.LayerNorm(hidden_dim),
                'norm2': nn.LayerNorm(hidden_dim),
                'dropout': nn.Dropout(dropout)
            })
            self.layers.append(layer)
        
        # Output projection
        self.output_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Project to hidden dimension
        x = self.input_proj(x)
        
        # Apply encoder layers
        for layer in self.layers:
            # Local Norm Pooling
            x_lnp = layer['lnp'](x)
            
            # Bidirectional Quaternion Mamba
            x_mamba = layer['mamba'](x_lnp)
            x = x + layer['dropout'](x_mamba)
            x = layer['norm1'](x)
            
            # Multi-scale temporal processing
            x_multiscale = layer['multiscale'](x)
            x = x + layer['dropout'](x_multiscale)
            x = layer['norm2'](x)
        
        # Final normalization
        x = self.output_norm(x)
        
        return x


class TemporalMotionHead(nn.Module):
    """Prediction head with motion auxiliary task."""
    
    def __init__(self, hidden_dim: int, num_classes: int, predict_motion: bool = True):
        super().__init__()
        self.predict_motion = predict_motion
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        # Motion prediction head (auxiliary task)
        if predict_motion:
            self.motion_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 3)  # Predict 3D motion vectors
            )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Global pooling over time
        x_pooled = x.mean(dim=1)  # [B, C]
        
        # Classification
        logits = self.classifier(x_pooled)
        
        # Motion prediction
        motion = None
        if self.predict_motion:
            motion = self.motion_head(x[:, :-1])  # Predict next frame motion
        
        return logits, motion
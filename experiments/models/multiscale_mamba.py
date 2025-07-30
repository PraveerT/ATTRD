"""
Multi-Scale Temporal Mamba with Motion Flow Prediction
Novel architecture for breaking 92% accuracy on point cloud action recognition
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from mamba_ssm.modules.mamba_simple import Mamba


class MotionFlowHead(nn.Module):
    """Auxiliary head for predicting motion flow between consecutive frames"""
    
    def __init__(self, feature_dim, hidden_dim=128):
        super().__init__()
        self.flow_predictor = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3)  # Predict 3D motion vector
        )
        
    def forward(self, features_t1, features_t2):
        """
        Predict motion flow from t1 to t2
        features_t1, features_t2: [B, N, D] - features at consecutive time steps
        Returns: [B, N, 3] - predicted motion vectors
        """
        combined = torch.cat([features_t1, features_t2], dim=-1)
        flow = self.flow_predictor(combined)
        return flow
    
    def compute_flow_loss(self, pred_flow, points_t1, points_t2):
        """Compute motion flow prediction loss"""
        actual_flow = points_t2 - points_t1
        loss = F.smooth_l1_loss(pred_flow, actual_flow)
        return loss


class TemporalDownsample(nn.Module):
    """Downsamples temporal dimension by factor of 2 with feature aggregation"""
    
    def __init__(self, feature_dim):
        super().__init__()
        self.conv1d = nn.Conv1d(feature_dim, feature_dim, kernel_size=2, stride=2)
        self.norm = nn.LayerNorm(feature_dim)
        
    def forward(self, x):
        """
        x: [B, T, N, D]
        Returns: [B, T//2, N, D]
        """
        B, T, N, D = x.shape
        # Reshape for 1D conv: [B*N, D, T]
        x = x.permute(0, 2, 3, 1).reshape(B * N, D, T)
        x = self.conv1d(x)  # [B*N, D, T//2]
        # Reshape back
        x = x.reshape(B, N, D, -1).permute(0, 3, 1, 2)  # [B, T//2, N, D]
        return x


class MultiScaleTemporalBlock(nn.Module):
    """Process temporal features at multiple scales"""
    
    def __init__(self, feature_dim, hidden_dim, num_scales=3):
        super().__init__()
        self.num_scales = num_scales
        self.feature_dim = feature_dim
        
        # Downsamplers for each scale
        self.downsamplers = nn.ModuleList([
            TemporalDownsample(feature_dim) for _ in range(num_scales - 1)
        ])
        
        # Mamba blocks for each scale
        self.mamba_blocks = nn.ModuleList([
            Mamba(
                d_model=hidden_dim,
                d_state=16,
                d_conv=4,
                expand=2,
            ) for _ in range(num_scales)
        ])
        
        # Feature projections
        self.input_projs = nn.ModuleList([
            nn.Linear(feature_dim, hidden_dim) for _ in range(num_scales)
        ])
        
        self.output_projs = nn.ModuleList([
            nn.Linear(hidden_dim, feature_dim) for _ in range(num_scales)
        ])
        
        # Scale fusion
        self.scale_fusion = nn.Linear(feature_dim * num_scales, feature_dim)
        self.fusion_norm = nn.LayerNorm(feature_dim)
        
    def forward(self, x):
        """
        x: [B, T, N, D] - input features
        Returns: [B, T, N, D] - multi-scale processed features
        """
        B, T, N, D = x.shape
        
        scale_outputs = []
        
        # Process at original scale (32 fps)
        x_scale = x.reshape(B * N, T, D)
        x_proj = self.input_projs[0](x_scale)
        x_mamba = self.mamba_blocks[0](x_proj)
        x_out = self.output_projs[0](x_mamba)
        x_out = x_out.reshape(B, N, T, D).permute(0, 2, 1, 3)
        scale_outputs.append(x_out)
        
        # Process at downsampled scales (16 fps, 8 fps, etc.)
        x_down = x
        for i in range(1, self.num_scales):
            # Downsample temporal dimension
            x_down = self.downsamplers[i-1](x_down)
            B_s, T_s, N_s, D_s = x_down.shape
            
            # Process with Mamba
            x_scale = x_down.reshape(B_s * N_s, T_s, D_s)
            x_proj = self.input_projs[i](x_scale)
            x_mamba = self.mamba_blocks[i](x_proj)
            x_out = self.output_projs[i](x_mamba)
            x_out = x_out.reshape(B_s, N_s, T_s, D_s).permute(0, 2, 1, 3)
            
            # Upsample back to original temporal resolution
            for j in range(i):
                # Reshape to 3D for interpolation: [B*N*D, 1, T]
                B_up, T_up, N_up, D_up = x_out.shape
                x_out_3d = x_out.permute(0, 2, 3, 1).reshape(B_up * N_up * D_up, 1, T_up)
                x_out_3d = F.interpolate(
                    x_out_3d,
                    scale_factor=2,
                    mode='linear',
                    align_corners=False
                )
                # Reshape back to 4D: [B, T*2, N, D]
                x_out = x_out_3d.reshape(B_up, N_up, D_up, -1).permute(0, 3, 1, 2)
                
            scale_outputs.append(x_out)
        
        # Fuse multi-scale features
        fused = torch.cat(scale_outputs, dim=-1)  # [B, T, N, D*num_scales]
        fused = self.scale_fusion(fused)
        fused = self.fusion_norm(fused)
        
        return fused


class MultiScaleMambaEncoder(nn.Module):
    """Multi-Scale Temporal Mamba with Motion Flow Prediction"""
    
    def __init__(self, in_channels, hidden_dim, output_dim=None, 
                 num_layers=2, num_scales=3, use_motion_flow=True, drop_path=0.1):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim if output_dim is not None else hidden_dim
        self.use_motion_flow = use_motion_flow
        
        # Multi-scale temporal blocks
        self.ms_blocks = nn.ModuleList([
            MultiScaleTemporalBlock(
                self.hidden_dim if i > 0 else in_channels,
                self.hidden_dim,
                num_scales=num_scales
            ) for i in range(num_layers)
        ])
        
        # Layer norms
        self.norms = nn.ModuleList([
            nn.LayerNorm(self.hidden_dim) for _ in range(num_layers)
        ])
        
        # Dropout
        self.dropout = nn.Dropout(drop_path)
        
        # Output projection
        self.output_proj = nn.Linear(self.hidden_dim, self.output_dim)
        self.final_norm = nn.LayerNorm(self.hidden_dim)
        
        # Motion flow prediction head
        if self.use_motion_flow:
            self.flow_head = MotionFlowHead(self.hidden_dim)
            self.flow_weight = 0.01  # Reduced weight for auxiliary loss (was 0.1)
            
    def forward(self, x, point_coords=None, return_flow_loss=False):
        """
        x: [B, C, T, N] - input features
        point_coords: [B, N, T, 3] - optional point coordinates for flow loss
        return_flow_loss: whether to compute and return flow prediction loss
        """
        B, C, T, N = x.shape
        
        # Reshape to [B, T, N, C]
        x = x.permute(0, 2, 3, 1)
        
        # Store features for motion flow prediction
        flow_features = []
        flow_loss = 0.0
        
        # Apply multi-scale blocks
        for i, (block, norm) in enumerate(zip(self.ms_blocks, self.norms)):
            residual = x if i > 0 else None
            x = block(x)
            x = norm(x)
            x = self.dropout(x)
            
            if residual is not None and residual.shape == x.shape:
                x = x + residual
                
            # Store intermediate features for flow prediction
            if self.use_motion_flow and i == len(self.ms_blocks) // 2:
                flow_features = x
        
        # Compute motion flow loss if requested
        if self.use_motion_flow and return_flow_loss and point_coords is not None:
            for t in range(T - 1):
                # Get features at consecutive time steps
                feat_t1 = flow_features[:, t, :, :]  # [B, N, D]
                feat_t2 = flow_features[:, t + 1, :, :]
                
                # Predict flow
                pred_flow = self.flow_head(feat_t1, feat_t2)
                
                # Get actual point positions
                points_t1 = point_coords[:, :, t, :]  # [B, N, 3]
                points_t2 = point_coords[:, :, t + 1, :]
                
                # Compute flow loss
                flow_loss += self.flow_head.compute_flow_loss(
                    pred_flow, points_t1, points_t2
                )
                
            flow_loss = flow_loss / (T - 1) * self.flow_weight
        
        # Final processing
        x = self.final_norm(x)
        x = self.output_proj(x)
        
        # Reshape back to [B, output_dim, T, N]
        x = x.permute(0, 3, 1, 2)
        
        if return_flow_loss:
            return x, flow_loss
        return x


class MultiScaleMambaTemporalEncoder(nn.Module):
    """Wrapper to match the interface of MambaTemporalEncoder"""
    
    def __init__(self, in_channels, hidden_dim, output_dim=None, num_layers=2, drop_path=0.1):
        super().__init__()
        self.encoder = MultiScaleMambaEncoder(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            num_scales=3,  # Process at 32fps, 16fps, 8fps
            use_motion_flow=True,
            drop_path=drop_path
        )
        
    def forward(self, x):
        # Extract coordinates if available (first 3 channels)
        point_coords = None
        if x.size(1) >= 3:
            # Assume first 3 channels are xyz coordinates
            point_coords = x[:, :3].permute(0, 3, 2, 1)  # [B, N, T, 3]
            
        # Forward through multi-scale encoder
        output, flow_loss = self.encoder(x, point_coords, return_flow_loss=True)
        
        # Store flow loss for potential use in training
        self.flow_loss = flow_loss
        
        return output
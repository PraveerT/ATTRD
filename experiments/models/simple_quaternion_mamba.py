import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba
import math
from typing import Optional, Tuple


class QuaternionLinear(nn.Module):
    """Simplified quaternion linear transformation for rotation-equivariant features."""
    
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Use regular linear layer but process quaternion-like features
        self.linear = nn.Linear(in_features, out_features)
        self.quaternion_norm = nn.LayerNorm(out_features)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply linear transformation
        x = self.linear(x)
        # Apply normalization for stability
        x = self.quaternion_norm(x)
        return x


class SimpleBidirectionalMamba(nn.Module):
    """Simplified bidirectional Mamba for memory efficiency."""
    
    def __init__(self, d_model: int, d_state: int = 8):
        super().__init__()
        self.d_model = d_model
        
        # Forward and backward Mamba blocks
        self.forward_mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=2,  # Reduced conv size
            expand=1   # Reduced expansion
        )
        
        self.backward_mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=2,
            expand=1
        )
        
        # Quaternion projections
        self.quat_proj_in = QuaternionLinear(d_model, d_model)
        self.quat_proj_out = QuaternionLinear(d_model, d_model)
        
        # Simple fusion
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
        
        # Fuse pathways
        x_combined = torch.cat([x_forward, x_backward], dim=-1)
        x_fused = self.fusion(x_combined)
        
        # Final quaternion projection
        out = self.quat_proj_out(x_fused)
        
        return out


class SimpleQuaternionMambaEncoder(nn.Module):
    """Simplified Quaternion-Enhanced Mamba encoder for memory efficiency."""
    
    def __init__(
        self, 
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        d_state: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Stack of simplified encoder layers
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            layer = nn.ModuleDict({
                'mamba': SimpleBidirectionalMamba(hidden_dim, d_state),
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
            # Bidirectional Quaternion Mamba
            x_mamba = layer['mamba'](x)
            x = x + layer['dropout'](x_mamba)
            x = layer['norm1'](x)
        
        # Final normalization
        x = self.output_norm(x)
        
        return x


class SimpleTemporalMotionHead(nn.Module):
    """Simplified prediction head."""
    
    def __init__(self, hidden_dim: int, num_classes: int):
        super().__init__()
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Global pooling over time
        x_pooled = x.mean(dim=1)  # [B, C]
        
        # Classification
        logits = self.classifier(x_pooled)
        
        return logits
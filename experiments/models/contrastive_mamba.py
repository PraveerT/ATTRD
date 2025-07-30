"""
Contrastive Learning + Temporal Consistency for Point Cloud Action Recognition
Novel approach to learn discriminative features and enforce temporal consistency
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from mamba_ssm.modules.mamba_simple import Mamba


class TemporalConsistencyLoss(nn.Module):
    """Enforces temporal consistency in feature representations"""
    
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, features):
        """
        features: [B, T, D] - temporal features for each sample
        Returns: consistency loss encouraging smooth temporal transitions
        """
        B, T, D = features.shape
        
        # Compute pairwise similarities between consecutive frames
        similarities = []
        for t in range(T - 1):
            feat_t = features[:, t, :]  # [B, D]
            feat_t1 = features[:, t + 1, :]  # [B, D]
            
            # Cosine similarity
            sim = F.cosine_similarity(feat_t, feat_t1, dim=1)  # [B]
            similarities.append(sim)
            
        similarities = torch.stack(similarities, dim=1)  # [B, T-1]
        
        # Encourage high similarity between consecutive frames
        consistency_loss = -similarities.mean()
        
        return consistency_loss


class ContrastiveLoss(nn.Module):
    """Contrastive loss for learning discriminative action representations"""
    
    def __init__(self, temperature=0.07, margin=0.5):
        super().__init__()
        self.temperature = temperature
        self.margin = margin
        
    def forward(self, features, labels):
        """
        features: [B, D] - global features for each sample
        labels: [B] - action class labels
        Returns: contrastive loss
        """
        B, D = features.shape
        
        # Normalize features
        features = F.normalize(features, dim=1)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(features, features.T) / self.temperature  # [B, B]
        
        # Create label mask (same class = positive, different class = negative)
        labels = labels.view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(features.device)  # [B, B]
        
        # Remove diagonal (self-similarity)
        mask.fill_diagonal_(0)
        
        # Positive pairs (same class)
        pos_mask = mask
        pos_sim = similarity_matrix * pos_mask
        
        # Negative pairs (different class)
        neg_mask = 1 - mask
        neg_mask.fill_diagonal_(0)
        neg_sim = similarity_matrix * neg_mask
        
        # Compute contrastive loss
        # For each anchor, pull positives closer and push negatives away
        pos_loss = -torch.log(torch.exp(pos_sim).sum(dim=1) + 1e-8)
        neg_loss = torch.log(torch.exp(neg_sim).sum(dim=1) + 1e-8)
        
        contrastive_loss = (pos_loss + neg_loss).mean()
        
        return contrastive_loss


class FeatureProjector(nn.Module):
    """Projects features to contrastive learning space"""
    
    def __init__(self, input_dim, hidden_dim=512, output_dim=256):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x):
        return F.normalize(self.projector(x), dim=1)


class ContrastiveMambaEncoder(nn.Module):
    """Mamba encoder with contrastive learning and temporal consistency"""
    
    def __init__(self, in_channels, hidden_dim, output_dim=None, num_layers=2, drop_path=0.1):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim if output_dim is not None else hidden_dim
        
        # Input projection
        self.input_proj = nn.Linear(in_channels, hidden_dim)
        
        # Mamba layers
        self.mamba_layers = nn.ModuleList([
            Mamba(
                d_model=hidden_dim,
                d_state=16,
                d_conv=4,
                expand=2,
            )
            for _ in range(num_layers)
        ])
        
        # Layer norms
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        
        # Dropout
        self.dropout = nn.Dropout(drop_path)
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, self.output_dim)
        self.final_norm = nn.LayerNorm(hidden_dim)
        
        # Contrastive learning components
        self.feature_projector = FeatureProjector(hidden_dim, output_dim=256)
        self.temporal_consistency = TemporalConsistencyLoss(temperature=0.1)
        self.contrastive_loss_fn = ContrastiveLoss(temperature=0.07)
        
        # Loss weights
        self.consistency_weight = 0.1
        self.contrastive_weight = 0.2
        
    def forward(self, x, labels=None, return_losses=False):
        """
        x: [B, C, T, N] - input features
        labels: [B] - action labels for contrastive learning
        return_losses: whether to compute and return auxiliary losses
        """
        B, C, T, N = x.shape
        
        # Reshape to B*N, T, C for temporal processing
        x = x.permute(0, 3, 2, 1).reshape(B * N, T, C)
        
        # Project to hidden dimension
        x = self.input_proj(x)
        
        # Store temporal features for consistency loss
        temporal_features = []
        
        # Apply Mamba layers with residual connections
        for i, (mamba, norm) in enumerate(zip(self.mamba_layers, self.norms)):
            residual = x
            x = norm(x)
            x = mamba(x)
            x = self.dropout(x)
            x = x + residual
            
            # Store features at different layers for temporal consistency
            if i == len(self.mamba_layers) // 2:  # Middle layer
                # Global average pooling over points for temporal consistency
                temp_feat = x.reshape(B, N, T, self.hidden_dim).mean(dim=1)  # [B, T, D]
                temporal_features = temp_feat
                
        # Final processing
        x = self.final_norm(x)
        x = self.output_proj(x)
        
        # Reshape back to [B, output_dim, T, N]
        x = x.reshape(B, N, T, self.output_dim).permute(0, 3, 2, 1)
        
        # Compute auxiliary losses if requested
        auxiliary_losses = {}
        if return_losses and labels is not None:
            # Global feature for contrastive learning (average over time and points)
            global_feat = x.mean(dim=(2, 3))  # [B, output_dim]
            
            # Project to contrastive space
            contrastive_feat = self.feature_projector(global_feat)
            
            # Contrastive loss
            contrastive_loss = self.contrastive_loss_fn(contrastive_feat, labels)
            auxiliary_losses['contrastive'] = contrastive_loss * self.contrastive_weight
            
            # Temporal consistency loss
            if len(temporal_features) > 0:
                consistency_loss = self.temporal_consistency(temporal_features)
                auxiliary_losses['consistency'] = consistency_loss * self.consistency_weight
                
        if return_losses:
            return x, auxiliary_losses
        return x


class TemporalAugmentation(nn.Module):
    """Data augmentation for temporal sequences"""
    
    def __init__(self, drop_rate=0.1, noise_std=0.01):
        super().__init__()
        self.drop_rate = drop_rate
        self.noise_std = noise_std
        
    def forward(self, x, training=True):
        if not training:
            return x
            
        B, C, T, N = x.shape
        
        # Temporal dropout (randomly zero out some time steps)
        if self.drop_rate > 0:
            temporal_mask = torch.rand(B, 1, T, 1, device=x.device) > self.drop_rate
            x = x * temporal_mask
            
        # Add temporal noise
        if self.noise_std > 0:
            noise = torch.randn_like(x) * self.noise_std
            x = x + noise
            
        return x


class ContrastiveMambaTemporalEncoder(nn.Module):
    """Main wrapper for contrastive Mamba temporal encoder"""
    
    def __init__(self, in_channels, hidden_dim, output_dim=None, num_layers=2, drop_path=0.1):
        super().__init__()
        self.encoder = ContrastiveMambaEncoder(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            drop_path=drop_path
        )
        
        # Temporal augmentation
        self.temporal_aug = TemporalAugmentation(drop_rate=0.05, noise_std=0.01)
        
        # Store auxiliary losses
        self.auxiliary_losses = {}
        
    def forward(self, x, labels=None):
        # Apply temporal augmentation during training
        x = self.temporal_aug(x, training=self.training)
        
        # Forward through contrastive encoder
        if self.training and labels is not None:
            output, aux_losses = self.encoder(x, labels, return_losses=True)
            self.auxiliary_losses = aux_losses
        else:
            output = self.encoder(x, labels, return_losses=False)
            self.auxiliary_losses = {}
            
        return output
    
    def get_auxiliary_losses(self):
        """Get auxiliary losses for training"""
        return self.auxiliary_losses
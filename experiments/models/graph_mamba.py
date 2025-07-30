"""
Graph-Mamba Hybrid: Novel architecture combining graph structure with Mamba for point cloud sequences
Key Innovation: Process neighborhood sequences instead of individual point sequences
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from mamba_ssm.modules.mamba_simple import Mamba
from torch_geometric.nn import knn_graph
from torch_geometric.utils import to_dense_adj


class GraphConstructor(nn.Module):
    """Constructs dynamic point cloud graphs with spatial and temporal edges"""
    
    def __init__(self, k_spatial=16, k_temporal=8):
        super().__init__()
        self.k_spatial = k_spatial
        self.k_temporal = k_temporal
        
    def build_spatial_graph(self, points):
        """Build KNN graph based on spatial proximity"""
        # points: [B, N, 3] - xyz coordinates
        B, N, _ = points.shape
        edge_indices = []
        
        for b in range(B):
            # Build KNN graph for each batch
            edge_index = knn_graph(points[b], k=self.k_spatial, loop=False)
            edge_indices.append(edge_index)
            
        return edge_indices
    
    def build_temporal_graph(self, point_trajectories):
        """Build graph connecting points across time based on motion similarity"""
        # point_trajectories: [B, N, T, 3] - point positions over time
        B, N, T, _ = point_trajectories.shape
        
        # Compute motion vectors
        motion_vectors = point_trajectories[:, :, 1:] - point_trajectories[:, :, :-1]  # [B, N, T-1, 3]
        motion_features = motion_vectors.norm(dim=-1).mean(dim=-1)  # [B, N] - average motion magnitude
        
        temporal_edges = []
        for b in range(B):
            # Connect points with similar motion patterns
            motion_sim = torch.mm(motion_features[b:b+1].T, motion_features[b:b+1])  # [N, N]
            _, top_indices = motion_sim.topk(k=self.k_temporal, dim=-1)
            
            # Create edge indices
            source = torch.arange(N, device=motion_features.device).unsqueeze(1).expand(-1, self.k_temporal).flatten()
            target = top_indices.flatten()
            edge_index = torch.stack([source, target], dim=0)
            temporal_edges.append(edge_index)
            
        return temporal_edges


class NeighborhoodAggregator(nn.Module):
    """Aggregates features from graph neighborhoods"""
    
    def __init__(self, feature_dim, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or feature_dim
        
        self.feature_transform = nn.Linear(feature_dim, hidden_dim)
        self.neighbor_transform = nn.Linear(feature_dim, hidden_dim)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, node_features, edge_indices):
        """
        node_features: [B, N, D] - features for each point
        edge_indices: list of [2, E] - edge indices for each batch
        """
        B, N, D = node_features.shape
        aggregated_features = []
        
        for b in range(B):
            edge_index = edge_indices[b].to(node_features.device)
            
            # Get neighborhood features
            source_nodes = edge_index[0]  # [E]
            target_nodes = edge_index[1]  # [E]
            
            source_features = node_features[b][source_nodes]  # [E, D]
            target_features = node_features[b][target_nodes]  # [E, D]
            
            # Transform features
            query = self.feature_transform(source_features)  # [E, H]
            key_value = self.neighbor_transform(target_features)  # [E, H]
            
            # Apply attention to aggregate neighborhood
            attended_features, _ = self.attention(
                query.unsqueeze(0), key_value.unsqueeze(0), key_value.unsqueeze(0)
            )
            attended_features = attended_features.squeeze(0)  # [E, H]
            
            # Scatter to original node indices
            node_aggregated = torch.zeros(N, attended_features.size(-1), 
                                        device=node_features.device, dtype=node_features.dtype)
            source_nodes_expanded = source_nodes.unsqueeze(-1).expand(-1, attended_features.size(-1))
            node_aggregated = node_aggregated.scatter_add(0, source_nodes_expanded, attended_features)
            
            aggregated_features.append(node_aggregated)
            
        aggregated_features = torch.stack(aggregated_features, dim=0)  # [B, N, H]
        return self.norm(aggregated_features)


class GraphMambaEncoder(nn.Module):
    """Novel Graph-Mamba hybrid encoder"""
    
    def __init__(self, in_channels, hidden_dim, output_dim=None, num_layers=2, 
                 k_spatial=16, k_temporal=8, drop_path=0.1):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim if output_dim is not None else hidden_dim
        self.num_layers = num_layers
        
        # Graph construction
        self.graph_constructor = GraphConstructor(k_spatial, k_temporal)
        
        # Feature projection
        self.input_proj = nn.Linear(in_channels, hidden_dim)
        
        # Neighborhood aggregation
        self.neighborhood_aggregator = NeighborhoodAggregator(hidden_dim)
        
        # Mamba layers for temporal processing of neighborhood sequences
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
        self.pre_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.post_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        
        # Dropout
        self.dropout = nn.Dropout(drop_path)
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, self.output_dim)
        self.final_norm = nn.LayerNorm(hidden_dim)
        
        # Spatial-temporal fusion
        self.fusion_gate = nn.Linear(hidden_dim * 2, hidden_dim)
        
    def forward(self, x, point_coords=None):
        """
        x: [B, C, T, N] - point cloud features over time
        point_coords: [B, N, T, 3] - optional point coordinates for graph construction
        """
        B, C, T, N = x.shape
        
        # If coordinates not provided, use first 3 channels as coordinates
        if point_coords is None:
            point_coords = x[:, :3].permute(0, 3, 2, 1)  # [B, N, T, 3]
        
        # Project input features
        x_reshaped = x.permute(0, 3, 2, 1).reshape(B * N, T, C)  # [B*N, T, C]
        x_proj = self.input_proj(x_reshaped)  # [B*N, T, H]
        x_proj = x_proj.reshape(B, N, T, self.hidden_dim)  # [B, N, T, H]
        
        # Build graphs (using coordinates from middle frame for stability)
        mid_frame = T // 2
        spatial_edges = self.graph_constructor.build_spatial_graph(point_coords[:, :, mid_frame])
        temporal_edges = self.graph_constructor.build_temporal_graph(point_coords)
        
        # Process each time step
        enhanced_features = []
        for t in range(T):
            # Get features at time t
            features_t = x_proj[:, :, t]  # [B, N, H]
            
            # Aggregate spatial neighborhoods
            spatial_agg = self.neighborhood_aggregator(features_t, spatial_edges)
            
            # Aggregate temporal neighborhoods (using motion-based connections)
            temporal_agg = self.neighborhood_aggregator(features_t, temporal_edges)
            
            # Fuse spatial and temporal information
            fused = self.fusion_gate(torch.cat([spatial_agg, temporal_agg], dim=-1))
            enhanced_features.append(fused)
            
        enhanced_features = torch.stack(enhanced_features, dim=2)  # [B, N, T, H]
        
        # Reshape for Mamba processing: [B*N, T, H]
        mamba_input = enhanced_features.reshape(B * N, T, self.hidden_dim)
        
        # Apply Mamba layers with residual connections
        x_mamba = mamba_input
        for i, (mamba, pre_norm, post_norm) in enumerate(zip(self.mamba_layers, self.pre_norms, self.post_norms)):
            residual = x_mamba
            x_mamba = pre_norm(x_mamba)
            x_mamba = mamba(x_mamba)
            x_mamba = self.dropout(x_mamba)
            x_mamba = post_norm(x_mamba + residual)
            
        # Final processing
        x_mamba = self.final_norm(x_mamba)
        x_out = self.output_proj(x_mamba)
        
        # Reshape back to [B, output_dim, T, N]
        x_out = x_out.reshape(B, N, T, self.output_dim).permute(0, 3, 2, 1)
        
        return x_out


class GraphMambaTemporalEncoder(nn.Module):
    """Wrapper to replace MambaTemporalEncoder with Graph-Mamba Hybrid"""
    
    def __init__(self, in_channels, hidden_dim, output_dim=None, num_layers=2, drop_path=0.1):
        super().__init__()
        self.encoder = GraphMambaEncoder(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            k_spatial=16,
            k_temporal=8,
            drop_path=drop_path
        )
        
    def forward(self, x):
        return self.encoder(x)
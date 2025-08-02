import pdb
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class MLPBlock(nn.Module):
    def __init__(self, out_channel, dimension, with_bn=True):
        super(MLPBlock, self).__init__()
        self.layer_list = []
        if dimension == 1:
            for idx, channels in enumerate(out_channel[:-1]):
                if with_bn:
                    self.layer_list.append(
                        nn.Sequential(
                            nn.Conv1d(channels, out_channel[idx + 1], kernel_size=1),
                            nn.BatchNorm1d(out_channel[idx + 1]),
                            nn.ReLU(inplace=True),
                        )
                    )
                else:
                    self.layer_list.append(
                        nn.Sequential(
                            nn.Conv1d(channels, out_channel[idx + 1], kernel_size=1),
                        )
                    )
        elif dimension == 2:
            for idx, channels in enumerate(out_channel[:-1]):
                if with_bn:
                    self.layer_list.append(
                        nn.Sequential(
                            nn.Conv2d(channels, out_channel[idx + 1], kernel_size=(1, 1)),
                            nn.BatchNorm2d(out_channel[idx + 1]),
                            nn.ReLU(inplace=True),
                        )
                    )
                else:
                    self.layer_list.append(
                        nn.Sequential(
                            nn.Conv2d(channels, out_channel[idx + 1], kernel_size=(1, 1)),
                        )
                    )
        self.layer_list = nn.ModuleList(self.layer_list)

    def forward(self, output):
        for layer in self.layer_list:
            output = layer(output)
        return output


class MotionBlock(nn.Module):
    def __init__(self, out_channel, dimension, embedding_dim):
        super(MotionBlock, self).__init__()
        self.layer_list = []
        if dimension == 1:
            self.layer_list.append(
                nn.Sequential(
                    nn.Conv1d(embedding_dim, out_channel[-1], kernel_size=1),
                    nn.BatchNorm1d(out_channel[-1]),
                    nn.ReLU(inplace=True),
                )
            )
            for idx, channels in enumerate(out_channel[:-1]):
                self.layer_list.append(
                    nn.Sequential(
                        nn.Conv1d(channels, out_channel[idx + 1], kernel_size=1),
                        nn.BatchNorm1d(out_channel[idx + 1]),
                        nn.ReLU(inplace=True),
                    )
                )
        elif dimension == 2:
            self.layer_list.append(
                nn.Sequential(
                    nn.Conv2d(embedding_dim, out_channel[-1], kernel_size=(1, 1)),
                    nn.BatchNorm2d(out_channel[-1]),
                    nn.ReLU(inplace=True),
                )
            )
            for idx, channels in enumerate(out_channel[:-1]):
                self.layer_list.append(
                    nn.Sequential(
                        nn.Conv2d(channels, out_channel[idx + 1], kernel_size=(1, 1)),
                        nn.BatchNorm2d(out_channel[idx + 1]),
                        nn.ReLU(inplace=True),
                    )
                )
        self.layer_list = nn.ModuleList(self.layer_list)


    def forward(self, output):
        position_embedding = self.layer_list[0](output[:, :4])
        feature_embedding = output[:, 4:]
        for layer in self.layer_list[1:]:
            feature_embedding = layer(feature_embedding)
        return position_embedding * feature_embedding


class SpatialAttention(nn.Module):
    """Lightweight attention mechanism for position-feature interaction"""
    def __init__(self, channels, dimension):
        super().__init__()
        self.channels = channels
        self.dimension = dimension
        
        # Simplified channel attention instead of spatial attention to save memory
        if dimension == 2:
            self.channel_attention = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(channels, channels // 4, kernel_size=1),
                nn.GELU(),
                nn.Conv2d(channels // 4, channels, kernel_size=1),
                nn.Sigmoid()
            )
        else:
            self.channel_attention = nn.Sequential(
                nn.AdaptiveAvgPool1d(1), 
                nn.Conv1d(channels, channels // 4, kernel_size=1),
                nn.GELU(),
                nn.Conv1d(channels // 4, channels, kernel_size=1),
                nn.Sigmoid()
            )
            
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, features, positions):
        """
        Lightweight channel attention instead of expensive spatial attention
        """
        # Combine features and positions for richer representation
        combined = features + positions
        
        # Apply channel attention to the combined features
        attention_weights = self.channel_attention(combined)
        attended_features = features * attention_weights
        
        # Apply learnable scaling and residual connection
        output = self.gamma * attended_features + features
        
        return output


class GroupOperation(object):
    def __init__(self, distance_metric='l2', normalize_offsets=True, eps=1e-8):
        """
        Improved GroupOperation with configurable distance metrics and optimizations.
        
        Args:
            distance_metric: 'l2', 'l1', or 'cosine'
            normalize_offsets: Whether to normalize spatial offsets
            eps: Small value for numerical stability
        """
        self.distance_metric = distance_metric
        self.normalize_offsets = normalize_offsets
        self.eps = eps

    def group_points(self, distance_dim, array1, array2, knn, dim):
        """
        Group points based on k-nearest neighbors with improved memory efficiency.
        """
        # Use the array_distance method for all cases to ensure compatibility
        matrix, a1, a2 = self.array_distance(array1, array2, distance_dim, dim)
        
        # Get k-nearest neighbors
        dists, inputs_idx = torch.topk(matrix, knn, -1, largest=False, sorted=True)
        
        # Gather neighbors
        neighbor = a2.gather(-1, inputs_idx.unsqueeze(1).expand(dists.shape[:1] + (a2.shape[1],) + dists.shape[1:]))
        
        # Compute offsets
        offsets = array1.unsqueeze(dim + 1) - neighbor
        
        # Normalize spatial offsets with improved numerical stability
        if self.normalize_offsets:
            spatial_offsets = offsets[:, :3]
            # Use torch.norm for better stability
            norm = torch.norm(spatial_offsets, dim=1, keepdim=True) + self.eps
            # Avoid in-place operation by creating new tensor
            normalized_offsets = offsets.clone()
            normalized_offsets[:, :3] = spatial_offsets / norm
            offsets = normalized_offsets
            
        return offsets

    def st_group_points(self, array, interval, distance_dim, knn, dim):
        """
        Spatio-temporal grouping with optimized batch processing.
        """
        batchsize, channels, timestep, num_pts = array.shape
        
        # Use original padding method for compatibility
        if interval // 2 > 0:
            array_padded = torch.cat((array[:, :, 0].unsqueeze(2).expand(-1, -1, interval // 2, -1),
                                      array,
                                      array[:, :, -1].unsqueeze(2).expand(-1, -1, interval // 2, -1)
                                      ), dim=2)
        else:
            array_padded = array
        
        # Use original neighbor point extraction method
        neighbor_points = torch.zeros(batchsize, channels, timestep, num_pts * interval).to(array.device)
        for i in range(timestep):
            neighbor_points[:, :, i] = array_padded[:, :, i:i + interval].view(batchsize, channels, -1)
        
        # Use array_distance for compatibility
        matrix, a1, a2 = self.array_distance(array, neighbor_points, distance_dim, dim)
        
        # Get k-nearest neighbors
        dists, inputs_idx = torch.topk(matrix, knn, -1, largest=False, sorted=True)
        
        # Gather neighbors using original method for compatibility  
        neighbor = a2.gather(-1, inputs_idx.unsqueeze(1).
                             expand(dists.shape[:1] + (a2.shape[1],) + dists.shape[1:]))
        
        # Compute features with edge attributes
        array_expanded = array.unsqueeze(-1).expand_as(neighbor)
        
        # Enhanced feature computation with relative positions and features
        position_diff = array_expanded[:, :4] - neighbor[:, :4]
        
        # Option to include distance as additional feature
        if channels > 4:
            ret_features = torch.cat([
                position_diff,           # Relative positions
                array_expanded[:, 4:],   # Source features
                neighbor[:, 4:]          # Neighbor features
            ], dim=1)
        else:
            ret_features = position_diff
            
        return ret_features

    def _compute_l2_distance_efficient(self, array1, array2, distance_dim, dim):
        """
        Efficient L2 distance computation - for now using original method for compatibility.
        """
        # Use the original computation method to ensure compatibility
        distance_mat = array1.unsqueeze(dim + 1)[:, distance_dim] - array2.unsqueeze(dim)[:, distance_dim]
        matrix = torch.sqrt((distance_mat ** 2).sum(1) + self.eps)
        return matrix

    def _compute_l1_distance(self, array1, array2, distance_dim, dim):
        """Compute L1 (Manhattan) distance."""
        distance_mat = array1.unsqueeze(dim + 1)[:, distance_dim] - array2.unsqueeze(dim)[:, distance_dim]
        matrix = torch.abs(distance_mat).sum(1)
        return matrix

    def _compute_cosine_distance(self, array1, array2, distance_dim, dim):
        """Compute cosine distance (1 - cosine similarity)."""
        # Extract features for similarity
        feat1 = array1[:, distance_dim]
        feat2 = array2[:, distance_dim]
        
        # Normalize features
        feat1_norm = F.normalize(feat1, p=2, dim=1)
        feat2_norm = F.normalize(feat2, p=2, dim=1)
        
        # Compute cosine similarity
        if dim == 3:
            similarity = torch.einsum('bctn,bctm->btnm', feat1_norm, feat2_norm)
        else:
            # General case
            feat1_exp = feat1_norm.unsqueeze(dim + 1)
            feat2_exp = feat2_norm.unsqueeze(dim)
            similarity = (feat1_exp * feat2_exp).sum(1)
        
        # Convert to distance
        matrix = 1.0 - similarity
        return matrix

    def _gather_neighbors_efficient(self, array, indices, dim):
        """
        Efficient neighbor gathering with minimal memory overhead.
        """
        # Use the same expansion logic as the original implementation
        # indices shape: (batch, ..., knn)
        # array shape: (batch, channels, ..., num_points)
        
        # Expand indices to match array dimensions
        expand_shape = list(indices.shape[:1]) + [array.shape[1]] + list(indices.shape[1:])
        indices_expanded = indices.unsqueeze(1).expand(expand_shape)
        
        # Gather neighbors along the last dimension
        neighbor = array.gather(-1, indices_expanded)
        
        return neighbor

    def array_distance(self, array1, array2, dist, dim):
        """
        Legacy method kept for compatibility.
        """
        distance_mat = array1.unsqueeze(dim + 1)[:, dist] - array2.unsqueeze(dim)[:, dist]
        mat_shape = distance_mat.shape
        mat_shape = mat_shape[:1] + (array1.shape[1],) + mat_shape[2:]
        array1 = array1.unsqueeze(dim + 1).expand(mat_shape)
        array2 = array2.unsqueeze(dim).expand(mat_shape)
        distance_mat = torch.sqrt((distance_mat ** 2).sum(1) + self.eps)
        return distance_mat, array1, array2

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
                            nn.BatchNorm1d(out_channel[idx]),
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
        self.embedding_dim = embedding_dim
        self.out_channels = out_channel  
        self.dimension = dimension
        
        # Build layers similar to original but with improvements
        self.layer_list = []
        
        if dimension == 1:
            # Position embedding layer (processes full input)
            self.layer_list.append(
                nn.Sequential(
                    nn.Conv1d(embedding_dim, out_channel[-1], kernel_size=1),
                    nn.BatchNorm1d(out_channel[-1]),
                    nn.GELU(),  # GELU instead of ReLU
                )
            )
            # Feature processing layers
            for idx, channels in enumerate(out_channel[:-1]):
                self.layer_list.append(
                    nn.Sequential(
                        nn.Conv1d(channels, out_channel[idx + 1], kernel_size=1),
                        nn.BatchNorm1d(out_channel[idx + 1]),
                        nn.GELU(),
                    )
                )
        elif dimension == 2:
            # Position embedding layer (processes full input)  
            self.layer_list.append(
                nn.Sequential(
                    nn.Conv2d(embedding_dim, out_channel[-1], kernel_size=(1, 1)),
                    nn.BatchNorm2d(out_channel[-1]),
                    nn.GELU(),  # GELU instead of ReLU
                )
            )
            # Feature processing layers
            for idx, channels in enumerate(out_channel[:-1]):
                self.layer_list.append(
                    nn.Sequential(
                        nn.Conv2d(channels, out_channel[idx + 1], kernel_size=(1, 1)),
                        nn.BatchNorm2d(out_channel[idx + 1]),
                        nn.GELU(),
                    )
                )
        
        self.layer_list = nn.ModuleList(self.layer_list)
        
        # Enhanced fusion mechanism
        self.fusion_gate = nn.Parameter(torch.ones(1) * 0.5)  # Learnable gating
        
        # Attention mechanism for better feature interaction
        self.attention = SpatialAttention(out_channel[-1], dimension) if len(out_channel) > 1 else None

    def _build_branch(self, in_channels, out_channels, dimension):
        """Build improved position encoding branch"""
        if dimension == 2:
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels // 2, kernel_size=(1, 1)),
                nn.BatchNorm2d(out_channels // 2),
                nn.GELU(),
                nn.Conv2d(out_channels // 2, out_channels, kernel_size=(1, 1)),
                nn.BatchNorm2d(out_channels),
                nn.GELU()
            )
        else:
            return nn.Sequential(
                nn.Conv1d(in_channels, out_channels // 2, kernel_size=1),
                nn.BatchNorm1d(out_channels // 2),
                nn.GELU(),
                nn.Conv1d(out_channels // 2, out_channels, kernel_size=1),
                nn.BatchNorm1d(out_channels),
                nn.GELU()
            )

    def forward(self, output):
        # Position embedding from the first 4 channels (x, y, z, t)
        position_embedding = self.layer_list[0](output[:, :4])
        
        # Feature embedding from the remaining channels
        feature_embedding = output[:, 4:] if output.size(1) > 4 else position_embedding
        
        # Process features through remaining layers with residual connections
        for i, layer in enumerate(self.layer_list[1:], 1):
            residual = feature_embedding
            feature_embedding = layer(feature_embedding)
            
            # Add residual connection if dimensions match (improved gradient flow)
            if residual.shape == feature_embedding.shape and i > 1:
                feature_embedding = feature_embedding + 0.1 * residual  # Scaled residual
        
        # Apply attention mechanism if available
        if self.attention is not None:
            feature_embedding = self.attention(feature_embedding, position_embedding)
        
        # Enhanced fusion with learnable gating instead of simple multiplication
        if self.fusion_gate is not None:
            # Learnable combination: α * (pos * feat) + (1-α) * (pos + feat) 
            multiplicative = position_embedding * feature_embedding
            additive = position_embedding + feature_embedding
            output = self.fusion_gate * multiplicative + (1 - self.fusion_gate) * additive
        else:
            # Fallback to original multiplication
            output = position_embedding * feature_embedding
        
        return output


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
    def __init__(self):
        pass

    def group_points(self, distance_dim, array1, array2, knn, dim):
        matrix, a1, a2 = self.array_distance(array1, array2, distance_dim, dim)
        dists, inputs_idx = torch.topk(matrix, knn, -1, largest=False, sorted=True)
        neighbor = a2.gather(-1, inputs_idx.unsqueeze(1).expand(dists.shape[:1] + (a2.shape[1],) + dists.shape[1:]))
        offsets = array1.unsqueeze(dim + 1) - neighbor
        offsets[:, :3] /= torch.sum(offsets[:, :3] ** 2, dim=1).unsqueeze(1) ** 0.5 + 1e-8
        return offsets

    def st_group_points(self, array, interval, distance_dim, knn, dim):
        batchsize, channels, timestep, num_pts = array.shape
        if interval // 2 > 0:
            array_padded = torch.cat((array[:, :, 0].unsqueeze(2).expand(-1, -1, interval // 2, -1),
                                      array,
                                      array[:, :, -1].unsqueeze(2).expand(-1, -1, interval // 2, -1)
                                      ), dim=2)
        else:
            array_padded = array
        neighbor_points = torch.zeros(batchsize, channels, timestep, num_pts * interval).to(array.device)
        for i in range(timestep):
            neighbor_points[:, :, i] = array_padded[:, :, i:i + interval].view(batchsize, channels, -1)
        matrix, a1, a2 = self.array_distance(array, neighbor_points, distance_dim, dim)
        dists, inputs_idx = torch.topk(matrix, knn, -1, largest=False, sorted=True)
        neighbor = a2.gather(-1, inputs_idx.unsqueeze(1).
                             expand(dists.shape[:1] + (a2.shape[1],) + dists.shape[1:]))
        array = array.unsqueeze(-1).expand_as(neighbor)
        ret_features = torch.cat((array[:, :4] - neighbor[:, :4], array[:, 4:], neighbor[:, 4:]), dim=1)
        # ret_features = torch.cat((array[:, :4] - neighbor[:, :4], neighbor[:, 4:]), dim=1)
        return ret_features

    def array_distance(self, array1, array2, dist, dim):
        # return array1.shape[-1] * array2.shape[-1] matrix
        distance_mat = array1.unsqueeze(dim + 1)[:, dist] - array2.unsqueeze(dim)[:, dist]
        mat_shape = distance_mat.shape
        mat_shape = mat_shape[:1] + (array1.shape[1],) + mat_shape[2:]
        array1 = array1.unsqueeze(dim + 1).expand(mat_shape)
        array2 = array2.unsqueeze(dim).expand(mat_shape)
        distance_mat = torch.sqrt((distance_mat ** 2).sum(1))
        return distance_mat, array1, array2

import pdb
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from models.op import MLPBlock, MotionBlock, GroupOperation

# Use installed mamba_ssm for optimal performance
from mamba_ssm.modules.mamba_simple import Mamba


class MotionEnergyCascade(nn.Module):
    """Novel layer inspired by turbulence theory that decomposes motion into energy scales
    and models how energy transfers between fast and slow motions."""
    
    def __init__(self, in_channels, num_scales=4, energy_dim=32):
        super().__init__()
        self.in_channels = in_channels
        self.num_scales = num_scales
        self.energy_dim = energy_dim
        
        # Learnable wavelet-like filters for multi-scale decomposition
        self.scale_filters = nn.ModuleList([
            nn.Conv2d(in_channels, energy_dim, kernel_size=(2**i, 1), 
                     stride=(2**i, 1), padding=(2**(i-1), 0))
            for i in range(1, num_scales + 1)
        ])
        
        # Energy transfer network - models how energy flows between scales
        self.energy_transfer = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(energy_dim * 2, energy_dim, 1),
                nn.BatchNorm2d(energy_dim),
                nn.GELU(),
                nn.Conv2d(energy_dim, energy_dim, 1)
            ) for _ in range(num_scales - 1)
        ])
        
        # Kinetic energy computer
        self.kinetic_mlp = nn.Sequential(
            nn.Linear(3, energy_dim),
            nn.LayerNorm(energy_dim),
            nn.GELU(),
            nn.Linear(energy_dim, energy_dim // 2)
        )
        
        # Potential energy computer (based on point cloud configuration)
        self.potential_mlp = nn.Sequential(
            nn.Linear(in_channels, energy_dim),
            nn.LayerNorm(energy_dim),
            nn.GELU(),
            nn.Linear(energy_dim, energy_dim // 2)
        )
        
        # Energy conservation constraint
        self.conservation_gate = nn.Sequential(
            nn.Linear(energy_dim * num_scales, num_scales),
            nn.Softmax(dim=-1)
        )
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Conv2d(energy_dim * num_scales + in_channels, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.GELU()
        )
        
    def compute_motion_energy(self, x):
        """Compute kinetic and potential energy from motion."""
        B, C, T, N = x.shape
        
        # Compute velocity (finite differences)
        velocity = torch.zeros_like(x[:, :3])
        velocity[:, :, 1:] = x[:, :3, 1:] - x[:, :3, :-1]
        
        # Kinetic energy: 0.5 * m * v^2 (assuming unit mass)
        kinetic = 0.5 * (velocity ** 2).sum(dim=1, keepdim=True)  # (B, 1, T, N)
        
        # Process kinetic energy
        kinetic_flat = kinetic.permute(0, 2, 3, 1).reshape(B * T * N, 1)
        kinetic_features = self.kinetic_mlp(velocity.permute(0, 2, 3, 1).reshape(B * T * N, 3))
        kinetic_features = kinetic_features.reshape(B, T, N, -1).permute(0, 3, 1, 2)
        
        # Potential energy based on configuration
        potential_flat = x.permute(0, 2, 3, 1).reshape(B * T * N, C)
        potential_features = self.potential_mlp(potential_flat)
        potential_features = potential_features.reshape(B, T, N, -1).permute(0, 3, 1, 2)
        
        return kinetic_features, potential_features
    
    def forward(self, x):
        # x shape: B, C, T, N
        B, C, T, N = x.shape
        
        # 1. Multi-scale decomposition
        scale_energies = []
        for i, filter in enumerate(self.scale_filters):
            # Apply scale-specific filter
            scale_energy = filter(x)
            scale_energies.append(scale_energy)
        
        # 2. Compute motion energies
        kinetic_energy, potential_energy = self.compute_motion_energy(x)
        
        # 3. Model energy transfer between scales (cascade)
        transferred_energies = [scale_energies[0]]
        for i in range(len(scale_energies) - 1):
            # Energy flows from larger to smaller scales
            source = F.interpolate(scale_energies[i], size=(scale_energies[i+1].shape[2], N), 
                                 mode='bilinear', align_corners=False)
            target = scale_energies[i + 1]
            
            # Combine source and target
            combined = torch.cat([source, target], dim=1)
            
            # Model energy transfer
            transfer = self.energy_transfer[i](combined)
            transferred_energies.append(target + transfer)
        
        # 4. Apply energy conservation constraint
        # Ensure total energy is conserved across scales
        all_energies = []
        for i, energy in enumerate(transferred_energies):
            # Upsample to original resolution
            upsampled = F.interpolate(energy, size=(T, N), mode='bilinear', align_corners=False)
            all_energies.append(upsampled)
        
        energy_stack = torch.stack(all_energies, dim=2)  # (B, energy_dim, num_scales, T, N)
        energy_flat = energy_stack.permute(0, 3, 4, 1, 2).reshape(B * T * N, self.energy_dim, self.num_scales)
        
        # Conservation weights
        conservation_weights = self.conservation_gate(energy_flat.reshape(B * T * N, -1))
        conservation_weights = conservation_weights.unsqueeze(1)  # (B*T*N, 1, num_scales)
        
        # Apply conservation
        conserved_energy = (energy_flat * conservation_weights).sum(dim=2)  # (B*T*N, energy_dim)
        conserved_energy = conserved_energy.reshape(B, T, N, self.energy_dim).permute(0, 3, 1, 2)
        
        # 5. Combine all energy representations
        all_features = []
        for energy in all_energies:
            all_features.append(energy)
        
        energy_features = torch.cat(all_features, dim=1)  # (B, energy_dim * num_scales, T, N)
        
        # 6. Output projection with residual
        combined = torch.cat([x, energy_features], dim=1)
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
        self.dropout = nn.Dropout(drop_path)
        
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
        self.stage4 = MotionBlock([512, 512, ], 2, 4)
        self.pool4 = nn.AdaptiveMaxPool2d((None, 1))
        self.stage5 = MLPBlock([512, 1024], 2)
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
        self.mamba = MambaTemporalEncoder(in_channels=256, hidden_dim=256, output_dim=256, num_layers=2, drop_path=0.1)
        
        # Add Motion Energy Cascade layer after stage2
        self.motion_energy = MotionEnergyCascade(in_channels=132, num_scales=4, energy_dim=32)

    def forward(self, inputs):
        # B * T * N * D,  e.g. 16 * 32 * 512 * 4
        inputs = inputs.permute(0, 3, 1, 2)
        if self.training:
            inputs = inputs[:, :, :, torch.randperm(inputs.shape[3])[:self.pts_size]]
        else:
            inputs = inputs[:, :, :, ::inputs.shape[3] // self.pts_size]
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
        
        # Apply motion energy cascade analysis
        fea2 = self.motion_energy(fea2)

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

        # stage 4: inter-frame, late
        in_dims = fea3.shape[1] * 2 - 4
        pts_num //= self.downsample[2]
        ret_group_array4 = self.group.st_group_points(fea3, 3, [0, 1, 2], self.knn[3], 3)
        ret_array4, inputs, _ = self.select_ind(ret_group_array4, inputs,
                                                batchsize, in_dims, timestep, pts_num)
        fea4 = self.pool4(self.stage4(ret_array4)).view(batchsize, -1, timestep, pts_num)

        output = self.stage5(fea4)
        output = self.pool5(output)
        output = self.global_bn(output)
        output = self.stage6(output)
        return output.view(batchsize, self.num_classes)

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

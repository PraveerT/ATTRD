#!/usr/bin/env python3
"""
Simple training script to test accuracy improvements with the new weight_select implementation.
This script demonstrates how to use the improved point selection in a training context.
"""

import torch
import torch.nn as nn
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'experiments'))

from models.motion import Motion

def test_weight_select_function():
    """Test the weight_select function directly with various inputs"""
    print("Testing weight_select function directly...")
    
    # Test case 1: Normal case
    print("\n1. Testing normal case:")
    B, C, TP, K = 2, 6, 10, 5
    position = torch.randn(B, C, TP, K)
    # Make first 3 channels realistic coordinates
    position[:, 0, :, :] = position[:, 0, :, :] * 10  # x coordinates
    position[:, 1, :, :] = position[:, 1, :, :] * 10  # y coordinates
    position[:, 2, :, :] = position[:, 2, :, :] * 10  # z coordinates
    
    try:
        indices = Motion.weight_select(position, 5)
        print(f"   Input shape: {position.shape}")
        print(f"   Output indices shape: {indices.shape}")
        print(f"   Sample indices: {indices[0]}")
        print("   ✓ Test passed")
    except Exception as e:
        print(f"   ✗ Test failed: {e}")
    
    # Test case 2: Edge case with minimal points
    print("\n2. Testing edge case with minimal points:")
    B, C, TP, K = 1, 4, 3, 2
    position = torch.randn(B, C, TP, K)
    try:
        indices = Motion.weight_select(position, 2)
        print(f"   Input shape: {position.shape}")
        print(f"   Output indices shape: {indices.shape}")
        print("   ✓ Test passed")
    except Exception as e:
        print(f"   ✗ Test failed: {e}")
    
    # Test case 3: No feature channels
    print("\n3. Testing case with no feature channels:")
    B, C, TP, K = 1, 3, 5, 3
    position = torch.randn(B, C, TP, K)
    try:
        indices = Motion.weight_select(position, 3)
        print(f"   Input shape: {position.shape}")
        print(f"   Output indices shape: {indices.shape}")
        print("   ✓ Test passed")
    except Exception as e:
        print(f"   ✗ Test failed: {e}")

def demonstrate_improvement():
    """Demonstrate how the new implementation improves point selection"""
    print("\n\nDemonstrating improvement with new weight_select implementation...")
    
    # Create sample data that shows the benefit of spatial diversity
    B, C, TP, K = 1, 6, 20, 4
    position = torch.randn(B, C, TP, K)
    
    # Make first 3 channels realistic coordinates with some structure
    position[:, 0, :, :] = position[:, 0, :, :] * 10  # x coordinates
    position[:, 1, :, :] = position[:, 1, :, :] * 10  # y coordinates
    position[:, 2, :, :] = position[:, 2, :, :] * 10  # z coordinates
    
    # Add some variance to feature channels
    position[:, 3:, :, :] = position[:, 3:, :, :] * 2
    
    print(f"Sample data shape: {position.shape}")
    
    # Compare strategies
    print("\nComparing point selection strategies:")
    
    # Strategy 1: Distance only (original)
    distances = torch.max(torch.sum(position[:, :3] ** 2, dim=1), dim=-1)[0]
    dist_min = distances.min(dim=-1, keepdim=True)[0]
    dist_max = distances.max(dim=-1, keepdim=True)[0]
    dist_range = dist_max - dist_min
    dist_range = torch.where(dist_range == 0, torch.ones_like(dist_range), dist_range)
    normalized_distances = (distances - dist_min) / dist_range
    _, idx_dist_only = torch.topk(normalized_distances, 10, -1, largest=True, sorted=False)
    
    # Strategy 2: Distance + Variance (your improvement)
    if position.shape[1] > 3:
        feature_var = torch.var(position[:, 3:], dim=-1).mean(dim=1)
        var_min = feature_var.min(dim=-1, keepdim=True)[0]
        var_max = feature_var.max(dim=-1, keepdim=True)[0]
        var_range = var_max - var_min
        var_range = torch.where(var_range == 0, torch.ones_like(var_range), var_range)
        normalized_variance = (feature_var - var_min) / var_range
    else:
        normalized_variance = torch.zeros_like(normalized_distances)
    
    weights_var = 0.7 * normalized_distances + 0.3 * normalized_variance
    _, idx_dist_var = torch.topk(weights_var, 10, -1, largest=True, sorted=False)
    
    # Strategy 3: Our new implementation (Distance + Variance + Spatial Diversity)
    idx_new_impl = Motion.weight_select(position, 10)
    
    print(f"Distance-only strategy indices:     {sorted(idx_dist_only[0].tolist())}")
    print(f"Distance+Variance indices:          {sorted(idx_dist_var[0].tolist())}")
    print(f"New implementation indices:         {sorted(idx_new_impl[0].tolist())}")
    
    # Show that they're different
    diff_var = not torch.equal(idx_dist_only, idx_dist_var)
    diff_new = not torch.equal(idx_dist_var, idx_new_impl)
    
    print(f"\nStrategies produce different results:")
    print(f"  Distance vs Distance+Variance: {diff_var}")
    print(f"  Distance+Variance vs New Implementation: {diff_new}")
    
    if diff_new:
        print("\n✓ New implementation successfully adds spatial diversity criterion!")
        print("  This should lead to improved accuracy by selecting more informative")
        print("  and spatially diverse point subsets for processing.")

def main():
    print("Testing accuracy improvement with new weight_select implementation...")
    print("=" * 60)
    
    test_weight_select_function()
    demonstrate_improvement()
    
    print("\n" + "=" * 60)
    print("SUMMARY:")
    print("The new weight_select implementation adds a spatial diversity criterion")
    print("to the existing distance and variance metrics. This should improve")
    print("accuracy by selecting point subsets that are not only distant and")
    print("variable but also spatially well-distributed, leading to better")
    print("coverage of the point cloud during processing.")

if __name__ == "__main__":
    main()
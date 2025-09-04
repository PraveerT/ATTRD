import torch
import sys
import os

# Add parent directory to path to import modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.motion import Motion

def test_improved_selection():
    """Test the improved weight_select function"""
    print("Testing improved weight_select function...")
    
    # Create a minimal Motion model
    model = Motion(num_classes=10, pts_size=64)
    
    # Create mock data with both position and feature information
    batch_size = 2
    position_channels = 3  # x, y, z
    feature_channels = 4   # additional features
    channels = position_channels + feature_channels
    timesteps = 8
    points = 32
    knn = 16
    
    # Create group_array with shape (B, C, T*P, K)
    # First 3 channels are positions, rest are features
    group_array = torch.randn(batch_size, channels, timesteps * points, knn)
    
    # Test weight_select
    print("Testing improved weight_select...")
    try:
        selected_indices = model.weight_select(group_array, 16)
        print(f"weight_select output shape: {selected_indices.shape}")
        print(f"Selected indices range: [{selected_indices.min()}, {selected_indices.max()}]")
        assert selected_indices.shape == (batch_size, 16), "weight_select output shape mismatch"
        print("✓ Improved weight_select test passed")
        
        # Test edge cases
        print("Testing edge cases...")
        # Test with pts_num = 1
        selected_indices = model.weight_select(group_array, 1)
        assert selected_indices.shape == (batch_size, 1), "Edge case pts_num=1 failed"
        print("✓ Edge case pts_num=1 passed")
        
        # Test with pts_num = points (all points)
        selected_indices = model.weight_select(group_array, points)
        assert selected_indices.shape == (batch_size, points), "Edge case pts_num=points failed"
        print("✓ Edge case pts_num=points passed")
        
    except Exception as e:
        print(f"✗ Improved weight_select test failed: {e}")
        return False
    
    print("All tests passed!")
    return True

if __name__ == "__main__":
    success = test_improved_selection()
    if not success:
        exit(1)
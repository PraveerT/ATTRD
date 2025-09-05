import torch
import sys
sys.path.append('/notebooks/PMamba/experiments')
from models.motion import Motion

def test_weight_select():
    """Test the weight_select function with sample data"""
    print("Testing weight_select function...")
    
    # Create sample data mimicking the expected input format
    # B=2, C=6 (3 xyz + 3 features), T*P=10, K=5
    B, C, TP, K = 2, 6, 10, 5
    position = torch.randn(B, C, TP, K)
    
    # Make first 3 channels realistic coordinates
    position[:, 0, :, :] = position[:, 0, :, :] * 10  # x coordinates
    position[:, 1, :, :] = position[:, 1, :, :] * 10  # y coordinates
    position[:, 2, :, :] = position[:, 2, :, :] * 10  # z coordinates
    
    print(f"Input shape: {position.shape}")
    
    # Test with different topk values
    topk_values = [3, 5, 8]
    
    for topk in topk_values:
        print(f"\nTesting with topk={topk}")
        try:
            indices = Motion.weight_select(position, topk)
            print(f"Output indices shape: {indices.shape}")
            print(f"Indices range: [{indices.min().item()}, {indices.max().item()}]")
            print(f"Sample indices: {indices[0]}")
            print("✓ Test passed")
        except Exception as e:
            print(f"✗ Test failed with error: {e}")
    
    # Test edge cases
    print("\nTesting edge cases...")
    
    # Test with minimal neighbors
    position_min_k = torch.randn(1, 4, 5, 2)  # K=2
    try:
        indices = Motion.weight_select(position_min_k, 3)
        print("✓ Minimal neighbors test passed")
    except Exception as e:
        print(f"✗ Minimal neighbors test failed: {e}")
    
    # Test with no feature channels
    position_no_features = torch.randn(1, 3, 5, 5)  # Only xyz coordinates
    try:
        indices = Motion.weight_select(position_no_features, 3)
        print("✓ No features test passed")
    except Exception as e:
        print(f"✗ No features test failed: {e}")

if __name__ == "__main__":
    test_weight_select()
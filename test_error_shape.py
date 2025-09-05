import torch
import sys
sys.path.append('/notebooks/PMamba/experiments')
from models.motion import Motion

def test_problematic_shape():
    """Test with the shape that caused the error"""
    print("Testing with shape that caused the error...")
    
    # Recreate the problematic shape from the error
    # B=8, C=6, T*P=32*48=1536 (but we'll use 32*24=768 to match the error), K=24
    # But let's use the actual dimensions from the error:
    # distances: torch.Size([8, 32, 48])
    # variance: torch.Size([8, 32, 48]) 
    # diversity: torch.Size([8, 32, 24])  <-- This is the problem
    
    B, C, TP, K = 8, 6, 32*24, 24
    position = torch.randn(B, C, TP, K)
    
    # Make first 3 channels realistic coordinates
    position[:, 0, :, :] = position[:, 0, :, :] * 10  # x coordinates
    position[:, 1, :, :] = position[:, 1, :, :] * 10  # y coordinates
    position[:, 2, :, :] = position[:, 2, :, :] * 10  # z coordinates
    
    print(f"Input shape: {position.shape}")
    
    try:
        indices = Motion.weight_select(position, 24)
        print(f"Output indices shape: {indices.shape}")
        print(f"Indices range: [{indices.min().item()}, {indices.max().item()}]")
        print("✓ Test passed")
    except Exception as e:
        print(f"✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_problematic_shape()
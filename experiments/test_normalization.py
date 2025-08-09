import numpy as np
import torch
from nvidia_dataloader import NvidiaLoader

def test_normalization():
    """Test the new global normalization"""
    
    print("Testing new global normalization...")
    
    # Create data loaders
    train_loader = NvidiaLoader(framerate=32, phase="train")
    test_loader = NvidiaLoader(framerate=32, phase="test")
    
    # Test a few samples
    train_sample, train_label, _ = train_loader[0]
    test_sample, test_label, _ = test_loader[0]
    
    print(f"\nTrain sample shape: {train_sample.shape}")
    print(f"Test sample shape: {test_sample.shape}")
    
    # Check statistics
    print(f"\nTrain sample statistics:")
    print(f"X: mean={train_sample[:,:,0].mean():.4f}, std={train_sample[:,:,0].std():.4f}")
    print(f"Y: mean={train_sample[:,:,1].mean():.4f}, std={train_sample[:,:,1].std():.4f}")
    print(f"Z: mean={train_sample[:,:,2].mean():.4f}, std={train_sample[:,:,2].std():.4f}")
    print(f"T: mean={train_sample[:,:,3].mean():.4f}, std={train_sample[:,:,3].std():.4f}")
    
    print(f"\nTest sample statistics:")
    print(f"X: mean={test_sample[:,:,0].mean():.4f}, std={test_sample[:,:,0].std():.4f}")
    print(f"Y: mean={test_sample[:,:,1].mean():.4f}, std={test_sample[:,:,1].std():.4f}")
    print(f"Z: mean={test_sample[:,:,2].mean():.4f}, std={test_sample[:,:,2].std():.4f}")
    print(f"T: mean={test_sample[:,:,3].mean():.4f}, std={test_sample[:,:,3].std():.4f}")
    
    # Check data ranges
    print(f"\nTrain sample ranges:")
    print(f"X: [{train_sample[:,:,0].min():.4f}, {train_sample[:,:,0].max():.4f}]")
    print(f"Y: [{train_sample[:,:,1].min():.4f}, {train_sample[:,:,1].max():.4f}]")
    print(f"Z: [{train_sample[:,:,2].min():.4f}, {train_sample[:,:,2].max():.4f}]")
    print(f"T: [{train_sample[:,:,3].min():.4f}, {train_sample[:,:,3].max():.4f}]")
    
    print(f"\nTest sample ranges:")
    print(f"X: [{test_sample[:,:,0].min():.4f}, {test_sample[:,:,0].max():.4f}]")
    print(f"Y: [{test_sample[:,:,1].min():.4f}, {test_sample[:,:,1].max():.4f}]")
    print(f"Z: [{test_sample[:,:,2].min():.4f}, {test_sample[:,:,2].max():.4f}]")
    print(f"T: [{test_sample[:,:,3].min():.4f}, {test_sample[:,:,3].max():.4f}]")
    
    print("\nNormalization test completed successfully!")
    
    return True

if __name__ == "__main__":
    test_normalization()
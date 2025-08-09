import numpy as np
import torch
from ntu_dataloader import NTULoader
import os

def test_ntu_normalization():
    """Test the new global normalization for NTU dataset"""
    
    print("Testing NTU global normalization...")
    
    # Check if NTU dataset path exists
    ntu_path = "/notebooks/NTU/nturgb+d_depth_masked"
    if not os.path.exists(ntu_path):
        print(f"⚠️ NTU dataset path not found: {ntu_path}")
        print("Skipping NTU normalization test")
        return True
    
    try:
        # Create data loaders with small sample for testing
        train_loader = NTULoader(
            framerate=16,  # Smaller for faster testing
            phase="train",
            pts_size=64,   # Smaller for faster testing
            data_path=ntu_path,
            use_cache=False  # Don't use cache for testing
        )
        
        if len(train_loader) == 0:
            print("No training samples found, skipping test")
            return True
            
        # Test a sample
        train_sample, train_label, train_path = train_loader[0]
        
        print(f"\nNTU train sample shape: {train_sample.shape}")
        print(f"Label: {train_label}")
        
        # Check statistics
        print(f"\nTrain sample statistics:")
        print(f"U: mean={train_sample[:,:,0].mean():.4f}, std={train_sample[:,:,0].std():.4f}")
        print(f"V: mean={train_sample[:,:,1].mean():.4f}, std={train_sample[:,:,1].std():.4f}")
        print(f"D: mean={train_sample[:,:,2].mean():.4f}, std={train_sample[:,:,2].std():.4f}")
        print(f"T: mean={train_sample[:,:,3].mean():.4f}, std={train_sample[:,:,3].std():.4f}")
        
        # Check data ranges
        print(f"\nTrain sample ranges:")
        print(f"U: [{train_sample[:,:,0].min():.4f}, {train_sample[:,:,0].max():.4f}]")
        print(f"V: [{train_sample[:,:,1].min():.4f}, {train_sample[:,:,1].max():.4f}]")
        print(f"D: [{train_sample[:,:,2].min():.4f}, {train_sample[:,:,2].max():.4f}]")
        print(f"T: [{train_sample[:,:,3].min():.4f}, {train_sample[:,:,3].max():.4f}]")
        
        # Test another sample to ensure consistency
        if len(train_loader) > 1:
            train_sample2, _, _ = train_loader[1]
            print(f"\nSecond sample statistics (should have similar distribution):")
            print(f"U: mean={train_sample2[:,:,0].mean():.4f}, std={train_sample2[:,:,0].std():.4f}")
            print(f"V: mean={train_sample2[:,:,1].mean():.4f}, std={train_sample2[:,:,1].std():.4f}")
            print(f"D: mean={train_sample2[:,:,2].mean():.4f}, std={train_sample2[:,:,2].std():.4f}")
            print(f"T: mean={train_sample2[:,:,3].mean():.4f}, std={train_sample2[:,:,3].std():.4f}")
        
        print("\nNTU normalization test completed successfully!")
        return True
        
    except Exception as e:
        print(f"⚠️ Error testing NTU normalization: {e}")
        print("This might be due to missing NTU dataset files")
        return False

if __name__ == "__main__":
    test_ntu_normalization()
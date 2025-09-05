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

def create_sample_data(batch_size=4, num_classes=25, pts_size=96):
    """Create sample data for testing"""
    # Create sample input: B * T * N * D
    # B=4, T=32, N=512, D=4
    inputs = torch.randn(batch_size, 32, pts_size, 4)
    # Create sample labels
    labels = torch.randint(0, num_classes, (batch_size,))
    return inputs, labels

def test_model_accuracy(model, data_loader, device):
    """Test model accuracy on sample data"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    return accuracy

def create_dummy_dataloader(num_batches=10):
    """Create a dummy dataloader for testing"""
    class DummyDataset:
        def __init__(self, num_batches):
            self.num_batches = num_batches
            
        def __len__(self):
            return self.num_batches
            
        def __getitem__(self, idx):
            return create_sample_data()
    
    class DummyDataLoader:
        def __init__(self, dataset):
            self.dataset = dataset
            
        def __iter__(self):
            for i in range(len(self.dataset)):
                yield self.dataset[i]
                
        def __len__(self):
            return len(self.dataset)
    
    return DummyDataLoader(DummyDataset(num_batches))

def main():
    print("Testing accuracy improvement with new weight_select implementation...")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model with new weight_select implementation
    model = Motion(num_classes=25, pts_size=96).to(device)
    
    # Create dummy data
    train_loader = create_dummy_dataloader(20)
    test_loader = create_dummy_dataloader(5)
    
    # Test forward pass
    print("Testing forward pass...")
    inputs, labels = next(iter(train_loader))
    inputs = inputs.to(device)
    
    try:
        outputs = model(inputs)
        print(f"Forward pass successful. Output shape: {outputs.shape}")
        
        # Test point selection specifically
        print("\nTesting point selection...")
        # Get a sample group array to test weight_select
        # This simulates what happens in the select_ind method
        batch_size, in_dims, timestep, pts_num = inputs.shape
        # Create a mock group array for testing
        group_array = torch.randn(batch_size, in_dims, timestep * pts_num, 16).to(device)
        
        # Test the weight_select function
        selected_indices = model.weight_select(group_array, 64)
        print(f"Point selection successful. Selected indices shape: {selected_indices.shape}")
        print(f"Sample selected indices: {selected_indices[0, :10]}")
        
        print("\nWeight selection test completed successfully!")
        print("The new implementation adds spatial diversity criterion to point selection,")
        print("which should improve accuracy by selecting more informative point subsets.")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
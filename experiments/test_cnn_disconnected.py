import torch
import sys
sys.path.append(".")
from models.motion_cnn_disconnected import MotionWithDisconnectedCNN

def test_disconnected_model():
    # Create a sample input tensor with smaller size to avoid memory issues
    # B * T * N * D,  e.g. 2 * 16 * 128 * 4
    batch_size = 2
    time_steps = 16
    num_points = 128
    features = 4
    num_classes = 25
    
    # Create random input data and labels
    inputs = torch.randn(batch_size, time_steps, num_points, features)
    labels = torch.randint(0, num_classes, (batch_size,))
    
    # Move to CUDA if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = inputs.to(device)
    labels = labels.to(device)
    
    # Initialize the model
    model = MotionWithDisconnectedCNN(
        num_classes=num_classes,
        pts_size=64,
        topk=8,
        downsample=(2, 2, 2),
        knn=(16, 24, 24, 12)
    )
    model = model.to(device)
    
    # Set model to training mode
    model.train()
    
    # Define loss function
    criterion = torch.nn.CrossEntropyLoss()
    
    # Forward pass
    output = model(inputs)
    
    print(f"Input shape: {inputs.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected output shape: ({batch_size}, {num_classes})")
    
    # Check if output has the correct shape
    assert output.shape == (batch_size, num_classes), f"Output shape mismatch: {output.shape} vs ({batch_size}, {num_classes})"
    
    # Check if we have the separate branch outputs
    assert hasattr(model, 'main_logits'), "Model should have main_logits attribute"
    assert hasattr(model, 'cnn_logits'), "Model should have cnn_logits attribute"
    
    print(f"Main branch logits shape: {model.main_logits.shape}")
    print(f"CNN branch logits shape: {model.cnn_logits.shape}")
    
    # Calculate losses for both branches
    main_loss = criterion(model.main_logits, labels)
    cnn_loss = criterion(model.cnn_logits, labels)
    combined_loss = criterion(output, labels)
    
    print(f"Main branch loss: {main_loss.item():.4f}")
    print(f"CNN branch loss: {cnn_loss.item():.4f}")
    print(f"Combined loss: {combined_loss.item():.4f}")
    
    # Calculate accuracies for both branches
    with torch.no_grad():
        main_pred = model.main_logits.argmax(dim=1, keepdim=True)
        cnn_pred = model.cnn_logits.argmax(dim=1, keepdim=True)
        combined_pred = output.argmax(dim=1, keepdim=True)
        
        main_correct = main_pred.eq(labels.view_as(main_pred)).sum().item()
        cnn_correct = cnn_pred.eq(labels.view_as(cnn_pred)).sum().item()
        combined_correct = combined_pred.eq(labels.view_as(combined_pred)).sum().item()
        
        main_acc = 100. * main_correct / batch_size
        cnn_acc = 100. * cnn_correct / batch_size
        combined_acc = 100. * combined_correct / batch_size
        
        print(f"Main branch accuracy: {main_acc:.2f}%")
        print(f"CNN branch accuracy: {cnn_acc:.2f}%")
        print(f"Combined accuracy: {combined_acc:.2f}%")
    
    # Test independent training of CNN branch
    # Zero gradients for all parameters
    model.zero_grad()
    
    # Compute loss for CNN branch only
    cnn_loss.backward(retain_graph=True)
    
    # Check if gradients exist for CNN branch parameters
    cnn_params_with_grad = [name for name, param in model.named_parameters() 
                           if param.grad is not None and 'cnn_branch' in name]
    
    # Check if gradients exist for main branch parameters
    main_params_with_grad = [name for name, param in model.named_parameters() 
                            if param.grad is not None and 'cnn_branch' not in name]
    
    print(f"CNN branch parameters with gradients: {len(cnn_params_with_grad)}")
    print(f"Main branch parameters with gradients: {len(main_params_with_grad)}")
    
    # Verify that main branch parameters don't have gradients (since we only backpropagated CNN loss)
    assert len(main_params_with_grad) == 0, "Main branch should not have gradients when only CNN loss is backpropagated"
    
    print("Independent training test passed!")
    
    # Test evaluation mode as well
    model.eval()
    output_eval = model(inputs)
    assert output_eval.shape == (batch_size, num_classes), "Evaluation mode output shape mismatch"
    
    print("Evaluation mode test passed!")
    
    print("All tests passed!")

if __name__ == "__main__":
    test_disconnected_model()
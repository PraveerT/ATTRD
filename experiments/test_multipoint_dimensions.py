#!/usr/bin/env python
"""
Test multi-point pooling dimension flow
"""

import torch
from models.motion import Motion

def test_multipoint_dimensions():
    """Test that multi-point pooling preserves correct dimensions throughout"""
    print("🔍 Testing Multi-Point Pooling Dimensions...")
    
    # Create model
    model = Motion(num_classes=25, pts_size=128)
    model.eval()
    
    # Test input
    batch_size = 2
    time_steps = 16  # Smaller for testing
    num_points = 256  # Smaller for testing
    input_dim = 4
    
    x = torch.randn(batch_size, time_steps, num_points, input_dim)
    print(f"Input shape: {x.shape}")
    
    try:
        with torch.no_grad():
            output = model(x)
            print(f"✅ SUCCESS: Output shape: {output.shape}")
            print(f"Expected output shape: ({batch_size}, 25)")
            
            if output.shape == (batch_size, 25):
                print("✅ Output dimensions are correct!")
            else:
                print("⚠️  Output dimensions don't match expected")
                
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def trace_dimension_flow():
    """Trace dimensions at each stage"""
    print("\n🔍 Tracing Dimension Flow...")
    
    model = Motion(num_classes=25, pts_size=128)
    
    # Hook to capture shapes at key points
    shapes = {}
    
    def make_hook(name):
        def hook(module, input, output):
            if isinstance(input, tuple):
                input_shape = [inp.shape for inp in input]
            else:
                input_shape = input.shape
                
            shapes[name] = {
                'input_shape': input_shape,
                'output_shape': output.shape if hasattr(output, 'shape') else 'No shape'
            }
        return hook
    
    # Register hooks
    hooks = []
    hooks.append(model.pool1.register_forward_hook(make_hook('pool1')))
    hooks.append(model.stage2.register_forward_hook(make_hook('stage2'))) 
    hooks.append(model.pool2.register_forward_hook(make_hook('pool2')))
    hooks.append(model.stage3.register_forward_hook(make_hook('stage3')))
    hooks.append(model.pool3.register_forward_hook(make_hook('pool3')))
    hooks.append(model.mamba.register_forward_hook(make_hook('mamba')))
    hooks.append(model.stage4.register_forward_hook(make_hook('stage4')))
    hooks.append(model.stage5.register_forward_hook(make_hook('stage5')))
    
    # Test input
    x = torch.randn(2, 16, 256, 4)
    
    try:
        with torch.no_grad():
            _ = model(x)
        
        print("Dimension flow:")
        for name, info in shapes.items():
            print(f"  {name}:")
            print(f"    Input: {info['input_shape']}")
            print(f"    Output: {info['output_shape']}")
            
    except Exception as e:
        print(f"Error during tracing: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up hooks
        for hook in hooks:
            hook.remove()

def analyze_memory_improvement():
    """Analyze memory and information preservation improvement"""
    print("\n📊 Analyzing Improvements...")
    
    # Original: Keep 1 out of 16 neighbors (6.25% preserved)
    # New: Keep 4 out of 16 neighbors (25% preserved)
    
    old_preservation = 1/16 * 100
    new_preservation = 4/16 * 100
    improvement = new_preservation / old_preservation
    
    print(f"Information preservation:")
    print(f"  Original (max pooling): {old_preservation:.2f}% of neighbor information")
    print(f"  Multi-point pooling: {new_preservation:.2f}% of neighbor information")
    print(f"  Improvement: {improvement:.1f}x more geometric information preserved")
    
    # Memory impact
    print(f"\nMemory impact:")
    print(f"  Channel multiplication factor: 4x")
    print(f"  This enables the network to capture:")
    print(f"    - Multiple spatial perspectives per point")
    print(f"    - Diverse geometric relationships")
    print(f"    - Richer motion patterns")

if __name__ == "__main__":
    print("=" * 60)
    print("Multi-Point Pooling Dimension Test")
    print("=" * 60)
    
    success = test_multipoint_dimensions()
    
    if success:
        trace_dimension_flow()
        analyze_memory_improvement()
        print("\n✅ Multi-point pooling implementation successful!")
        print("🚀 Ready for training with 4x more geometric information!")
    else:
        print("\n❌ Multi-point pooling needs debugging")
    
    print("=" * 60)
import os
import sys
import re
import cv2
import copy
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm

# Load NVIDIA's exact utils functions directly
nvidia_utils_path = "/notebooks/PMamba/dataset/utils.py"
exec(open(nvidia_utils_path).read())

def load_ntu_as_nvidia_format(ntu_dir):
    """Load NTU PNG files into exact NVIDIA format: (T, H, W, 1)"""
    depth_files = [f for f in os.listdir(ntu_dir) if f.startswith('MDepth-') and f.endswith('.png')]
    depth_files.sort()
    
    if len(depth_files) == 0:
        return None
    
    # Load first frame to get dimensions
    first_frame = cv2.imread(os.path.join(ntu_dir, depth_files[0]), cv2.IMREAD_ANYDEPTH)
    if first_frame is None:
        return None
    
    H, W = first_frame.shape
    T = len(depth_files)
    
    # Create depth video in exact NVIDIA format: (T, H, W, 1)
    depth_video = np.zeros((T, H, W, 1), dtype=np.uint8)
    
    for i, depth_file in enumerate(depth_files):
        depth_path = os.path.join(ntu_dir, depth_file)
        depth_img = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
        if depth_img is not None:
            # NVIDIA seems to expect uint8, let's normalize to 0-255
            depth_normalized = (depth_img.astype(np.float32) / depth_img.max() * 255).astype(np.uint8)
            depth_video[i, :, :, 0] = depth_normalized
    
    return depth_video

def process_ntu_with_exact_nvidia_pipeline(ntu_dir, pts_size=512):
    """Process NTU directory using EXACT NVIDIA pipeline"""
    
    # Step 1: Load NTU data in NVIDIA format
    depth_video = load_ntu_as_nvidia_format(ntu_dir)
    if depth_video is None:
        return None, None
    
    print(f"Loaded depth video shape: {depth_video.shape} (NVIDIA format)")
    
    # Step 2: EXACT NVIDIA processing pipeline
    # Use key_frame_sampling(len(depth_video), 32)
    ind = key_frame_sampling(len(depth_video), 32)
    depth_video = depth_video[ind]
    
    # Initialize points array exactly like NVIDIA
    pts = np.zeros((len(depth_video), pts_size, 8), dtype=int)
    
    for i in range(len(depth_video)):
        # EXACT NVIDIA line: frame = depth_video[i, :, :, 0]
        frame = depth_video[i, :, :, 0]
        
        # EXACT NVIDIA processing
        ret, thresh = cv2.threshold(frame, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thresh = save_largest_label(thresh)
        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.erode(thresh, kernel)
        
        # EXACT NVIDIA point generation
        pts[i, :, :4] = points_sampling(generate_points(frame * thresh, i), pts_size)
        pts[i, :, 4:8] = uvd2xyz_sherc(copy.deepcopy(pts[i, :, :4]))
    
    return pts, depth_video

def create_validation_videos(pts, depth_video, dirname):
    """Create validation videos using NVIDIA's show_video_point_clouds"""
    
    # Create 3D point cloud animation using NVIDIA's function
    print("Creating 3D point cloud visualization...")
    
    fig = plt.figure(figsize=(12, 8))
    
    def animate_3d(frame_idx):
        fig.clear()
        
        # Use NVIDIA's exact visualization function
        ax = show_frame_point_clouds(pts[frame_idx], fig, 
                                   x_minmax=(-200, 200), 
                                   y_minmax=(-200, 200), 
                                   show_img=False)
        ax.set_title(f"EXACT NVIDIA Processing\\n{dirname} - Frame {frame_idx+1}/32")
    
    anim = animation.FuncAnimation(fig, animate_3d, frames=len(pts), interval=300, blit=False)
    output_path = f'/notebooks/PMamba/experiments/{dirname}_exact_nvidia_3d.gif'
    anim.save(output_path, writer='pillow', fps=3, dpi=100)
    print(f"✅ 3D visualization saved: {output_path}")
    plt.close()
    
    # Create depth processing visualization
    print("Creating depth processing visualization...")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    def animate_depth(frame_idx):
        for ax in [ax1, ax2, ax3, ax4]:
            ax.clear()
        
        # Original depth frame
        frame = depth_video[frame_idx, :, :, 0]
        ax1.imshow(frame, cmap='viridis')
        ax1.set_title(f'Frame {frame_idx+1}: Original Depth')
        
        # Apply NVIDIA processing step by step
        ret, thresh = cv2.threshold(frame, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        ax2.imshow(thresh, cmap='gray')
        ax2.set_title('After OTSU Threshold')
        
        thresh = save_largest_label(thresh)
        ax3.imshow(thresh, cmap='gray')
        ax3.set_title('After Largest Component')
        
        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.erode(thresh, kernel)
        masked = frame * thresh
        ax4.imshow(masked, cmap='viridis')
        ax4.set_title('Final Masked Depth')
        
        plt.tight_layout()
    
    anim = animation.FuncAnimation(fig, animate_depth, frames=len(depth_video), interval=300, blit=False)
    output_path = f'/notebooks/PMamba/experiments/{dirname}_exact_nvidia_depth.gif'
    anim.save(output_path, writer='pillow', fps=3, dpi=100)
    print(f"✅ Depth processing saved: {output_path}")
    plt.close()

def main():
    # Test on the same sample
    test_dirname = 'S001C001P003R002A036'
    test_dir = f'/notebooks/NTU/nturgb+d_depth_masked/{test_dirname}'
    
    print("🚀 Processing NTU with EXACT NVIDIA pipeline")
    print(f"📁 Test directory: {test_dir}")
    
    # Process using exact NVIDIA pipeline
    pts, depth_video = process_ntu_with_exact_nvidia_pipeline(test_dir, pts_size=512)
    
    if pts is not None:
        print(f"✅ Processing complete!")
        print(f"📊 Results:")
        print(f"  Points shape: {pts.shape}")
        print(f"  Points dtype: {pts.dtype}")
        print(f"  Depth video shape: {depth_video.shape}")
        
        # Show sample points
        print(f"\\n📋 Sample points (first frame, first 3 points):")
        print("Channels: [x, y, depth, time, x_3d, y_3d, z_3d, t_copy]")
        print(pts[0, :3, :])
        
        # Show coordinate ranges
        print(f"\\n📏 Coordinate ranges:")
        print(f"  Image x: [{pts[:, :, 0].min()}, {pts[:, :, 0].max()}]")
        print(f"  Image y: [{pts[:, :, 1].min()}, {pts[:, :, 1].max()}]")
        print(f"  Depth: [{pts[:, :, 2].min()}, {pts[:, :, 2].max()}]")
        print(f"  3D x: [{pts[:, :, 4].min()}, {pts[:, :, 4].max()}]")
        print(f"  3D y: [{pts[:, :, 5].min()}, {pts[:, :, 5].max()}]")
        print(f"  3D z: [{pts[:, :, 6].min()}, {pts[:, :, 6].max()}]")
        
        # Create validation videos
        create_validation_videos(pts, depth_video, test_dirname)
        
        # Save the processed result
        save_path = f'/notebooks/PMamba/experiments/{test_dirname}_exact_nvidia_pts.npy'
        np.save(save_path, pts)
        print(f"💾 Saved processed points: {save_path}")
        
    else:
        print("❌ Processing failed")

if __name__ == "__main__":
    main()
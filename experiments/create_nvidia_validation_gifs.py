import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import os

def create_nvidia_validation_gif(pts_file, output_name):
    """Create validation GIF from NVIDIA processed point cloud"""
    
    # Load the point cloud data
    pts = np.load(pts_file)
    dirname = os.path.basename(pts_file).replace('_pts.npy', '')
    
    print(f"Processing {dirname}:")
    print(f"  Shape: {pts.shape}")
    print(f"  Data type: {pts.dtype}")
    print(f"  Coordinate ranges:")
    print(f"    X: [{pts[:, :, 4].min()}, {pts[:, :, 4].max()}]")
    print(f"    Y: [{pts[:, :, 5].min()}, {pts[:, :, 5].max()}]")
    print(f"    Z: [{pts[:, :, 6].min()}, {pts[:, :, 6].max()}]")
    
    # Extract 3D coordinates (channels 4, 5, 6)
    frames_3d = pts[:, :, 4:7].astype(np.float32)
    
    # Create the animation
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    def animate(frame_idx):
        ax.clear()
        
        # Get points for this frame
        points = frames_3d[frame_idx]
        
        # Color points by Z coordinate (depth)
        colors = points[:, 2]
        
        # Plot the 3D point cloud
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                  c=colors, cmap='plasma', s=15, alpha=0.8, edgecolors='none')
        
        # Set labels and title
        ax.set_xlabel('X (NVIDIA Camera Units)', fontsize=10)
        ax.set_ylabel('Y (NVIDIA Camera Units)', fontsize=10)
        ax.set_zlabel('Z (Normalized Depth)', fontsize=10)
        ax.set_title(f'NVIDIA Processing: {dirname}\\nFrame {frame_idx+1}/32 | 512 points',
                    fontsize=12, pad=20)
        
        # Set consistent axis limits
        padding = 10
        ax.set_xlim([pts[:, :, 4].min()-padding, pts[:, :, 4].max()+padding])
        ax.set_ylim([pts[:, :, 5].min()-padding, pts[:, :, 5].max()+padding])
        ax.set_zlim([pts[:, :, 6].min()-padding, pts[:, :, 6].max()+padding])
        
        # Set viewing angle (rotating)
        ax.view_init(elev=20, azim=45 + frame_idx * 3)
        ax.grid(True, alpha=0.3)
    
    print(f"  Creating animation...")
    anim = animation.FuncAnimation(fig, animate, frames=32, interval=200, blit=False)
    
    # Save as GIF
    output_path = f'/notebooks/PMamba/experiments/{output_name}.gif'
    anim.save(output_path, writer='pillow', fps=5, dpi=100)
    print(f"  ✅ Saved: {output_path}")
    
    plt.close()
    return output_path

def create_comparison_gif(pts_files):
    """Create a comparison GIF showing multiple actions side by side"""
    
    if len(pts_files) < 2:
        print("Need at least 2 files for comparison")
        return
    
    # Load first 3 files for comparison
    pts_data = []
    names = []
    
    for i, pts_file in enumerate(pts_files[:3]):
        pts = np.load(pts_file)
        dirname = os.path.basename(pts_file).replace('_pts.npy', '')
        pts_data.append(pts[:, :, 4:7].astype(np.float32))  # 3D coords only
        names.append(dirname)
        
        if i >= 2:  # Limit to 3 for comparison
            break
    
    print(f"Creating comparison GIF with {len(pts_data)} actions...")
    
    # Create the comparison plot
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(16, 6))
    
    axes = []
    for i in range(len(pts_data)):
        ax = fig.add_subplot(1, len(pts_data), i+1, projection='3d')
        axes.append(ax)
    
    def animate_comparison(frame_idx):
        for i, (ax, points_3d, name) in enumerate(zip(axes, pts_data, names)):
            ax.clear()
            
            # Get points for this frame
            points = points_3d[frame_idx]
            colors = points[:, 2]  # Color by Z
            
            # Plot
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                      c=colors, cmap='plasma', s=8, alpha=0.8, edgecolors='none')
            
            # Labels and title
            ax.set_title(f'{name}\\nFrame {frame_idx+1}/32', fontsize=10)
            ax.set_xlabel('X', fontsize=8)
            ax.set_ylabel('Y', fontsize=8)
            ax.set_zlabel('Z', fontsize=8)
            
            # Consistent limits for comparison
            all_pts = np.concatenate(pts_data, axis=1)  # Combine all points
            padding = 5
            ax.set_xlim([all_pts[:, :, 0].min()-padding, all_pts[:, :, 0].max()+padding])
            ax.set_ylim([all_pts[:, :, 1].min()-padding, all_pts[:, :, 1].max()+padding])
            ax.set_zlim([all_pts[:, :, 2].min()-padding, all_pts[:, :, 2].max()+padding])
            
            ax.view_init(elev=15, azim=45)
            ax.grid(True, alpha=0.2)
    
    anim = animation.FuncAnimation(fig, animate_comparison, frames=32, interval=250, blit=False)
    
    # Save comparison GIF
    output_path = '/notebooks/PMamba/experiments/nvidia_actions_comparison.gif'
    anim.save(output_path, writer='pillow', fps=4, dpi=100)
    print(f"✅ Comparison saved: {output_path}")
    
    plt.close()
    return output_path

def main():
    # Find newly generated point cloud files
    print("🔍 Finding newly generated NVIDIA point clouds...")
    
    import glob
    ntu_path = "/notebooks/NTU/nturgb+d_depth_masked"
    pts_files = glob.glob(os.path.join(ntu_path, "**", "*_pts.npy"), recursive=True)
    
    print(f"Found {len(pts_files)} processed point cloud files")
    
    if len(pts_files) == 0:
        print("❌ No point cloud files found!")
        return
    
    # Take first 5 for individual GIFs
    selected_files = pts_files[:5]
    
    print(f"\\n🎬 Creating individual validation GIFs for {len(selected_files)} actions...")
    
    for i, pts_file in enumerate(selected_files):
        dirname = os.path.basename(pts_file).replace('_pts.npy', '')
        output_name = f"nvidia_action_{i+1}_{dirname}"
        
        try:
            create_nvidia_validation_gif(pts_file, output_name)
        except Exception as e:
            print(f"❌ Error processing {dirname}: {e}")
    
    # Create comparison GIF
    print(f"\\n🎬 Creating comparison GIF...")
    try:
        create_comparison_gif(selected_files)
    except Exception as e:
        print(f"❌ Error creating comparison: {e}")
    
    print(f"\\n✅ All GIFs created in /notebooks/PMamba/experiments/")
    print(f"📊 Summary:")
    print(f"  - {len(selected_files)} individual action GIFs")
    print(f"  - 1 comparison GIF")
    print(f"  - All using NVIDIA 512-point processing")
    print(f"  - Camera coordinate system (expected 'off-center')")

if __name__ == "__main__":
    main()
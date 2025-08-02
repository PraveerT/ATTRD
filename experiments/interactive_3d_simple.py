import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from matplotlib.widgets import Slider, Button

class Interactive3DViewer:
    def __init__(self, pts_file):
        # Load the point cloud data
        self.pts = np.load(pts_file)
        self.current_frame = 0
        self.max_frames = len(self.pts)
        
        print(f"Loaded point cloud: {self.pts.shape}")
        print(f"Point cloud data format: [x_img, y_img, depth, t, x_3d, y_3d, z_3d, t_copy]")
        print(f"3D coordinate ranges:")
        print(f"  X: [{self.pts[:, :, 4].min()}, {self.pts[:, :, 4].max()}]")
        print(f"  Y: [{self.pts[:, :, 5].min()}, {self.pts[:, :, 5].max()}]")
        print(f"  Z: [{self.pts[:, :, 6].min()}, {self.pts[:, :, 6].max()}]")
        
        # Set up the plot
        plt.style.use('dark_background')
        self.fig = plt.figure(figsize=(14, 10))
        self.fig.suptitle('Interactive NVIDIA Point Cloud Viewer', fontsize=16, color='white')
        
        # Main 3D plot
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # Initial plot
        self.scatter = None
        self.update_plot()
        
        # Add frame slider at bottom
        ax_frame = plt.axes([0.15, 0.02, 0.4, 0.03])
        self.frame_slider = Slider(ax_frame, 'Frame', 0, self.max_frames-1, 
                                  valinit=0, valstep=1, valfmt='%d')
        self.frame_slider.on_changed(self.update_frame)
        
        # Add animation controls
        ax_play = plt.axes([0.6, 0.02, 0.08, 0.04])
        self.play_button = Button(ax_play, 'Play')
        self.play_button.on_clicked(self.toggle_animation)
        
        ax_reset = plt.axes([0.7, 0.02, 0.08, 0.04])
        self.reset_button = Button(ax_reset, 'Reset')
        self.reset_button.on_clicked(self.reset_view)
        
        ax_center = plt.axes([0.8, 0.02, 0.08, 0.04])
        self.center_button = Button(ax_center, 'Center')
        self.center_button.on_clicked(self.center_view)
        
        # Animation state
        self.is_playing = False
        self.anim = None
        
        plt.subplots_adjust(bottom=0.12)
        
    def update_plot(self):
        """Update the 3D plot with current frame"""
        frame_points = self.pts[self.current_frame, :, 4:7]  # x, y, z coordinates (3D)
        
        if self.scatter is not None:
            self.scatter.remove()
        
        # Color points by Z coordinate (depth) for better visualization
        colors = frame_points[:, 2]  # Z coordinate for coloring
        
        self.scatter = self.ax.scatter(frame_points[:, 0], frame_points[:, 1], frame_points[:, 2],
                                     c=colors, cmap='plasma', s=25, alpha=0.8, edgecolors='none')
        
        # Set labels and title
        self.ax.set_xlabel('X (NVIDIA Camera Units)', fontsize=12)
        self.ax.set_ylabel('Y (NVIDIA Camera Units)', fontsize=12) 
        self.ax.set_zlabel('Z (Normalized Depth)', fontsize=12)
        
        # Dynamic title with coordinate info
        x_range = f"[{frame_points[:, 0].min():.1f}, {frame_points[:, 0].max():.1f}]"
        y_range = f"[{frame_points[:, 1].min():.1f}, {frame_points[:, 1].max():.1f}]"
        z_range = f"[{frame_points[:, 2].min():.1f}, {frame_points[:, 2].max():.1f}]"
        
        self.ax.set_title(f'Frame {self.current_frame+1}/{self.max_frames} | Points: {len(frame_points)}\\n'
                         f'X: {x_range} | Y: {y_range} | Z: {z_range}',
                         fontsize=11, pad=20)
        
        # Set axis limits for consistent view (with some padding)
        all_x, all_y, all_z = self.pts[:, :, 4], self.pts[:, :, 5], self.pts[:, :, 6]
        padding = 15
        self.ax.set_xlim([all_x.min()-padding, all_x.max()+padding])
        self.ax.set_ylim([all_y.min()-padding, all_y.max()+padding])
        self.ax.set_zlim([all_z.min()-padding, all_z.max()+padding])
        
        # Add grid for better depth perception
        self.ax.grid(True, alpha=0.3)
        
        plt.draw()
    
    def update_frame(self, val):
        """Update frame from slider"""
        self.current_frame = int(self.frame_slider.val)
        self.update_plot()
    
    def toggle_animation(self, event):
        """Start/stop animation"""
        if self.is_playing:
            if self.anim:
                self.anim.event_source.stop()
            self.play_button.label.set_text('Play')
            self.is_playing = False
        else:
            self.anim = animation.FuncAnimation(self.fig, self.animate, 
                                              frames=self.max_frames, 
                                              interval=300, repeat=True)
            self.play_button.label.set_text('Stop')
            self.is_playing = True
    
    def animate(self, frame):
        """Animation function"""
        self.current_frame = frame
        self.frame_slider.set_val(frame)
        self.update_plot()
        return self.scatter,
    
    def reset_view(self, event):
        """Reset the 3D view to default"""
        self.ax.view_init(elev=20, azim=45)
        plt.draw()
    
    def center_view(self, event):
        """Center the view on the point cloud"""
        # Calculate centroid
        all_points = self.pts[:, :, 4:7].reshape(-1, 3)
        centroid = np.mean(all_points, axis=0)
        
        # Find good viewing distance
        distances = np.sqrt(np.sum((all_points - centroid)**2, axis=1))
        max_dist = np.max(distances)
        
        # Set view to look at centroid
        self.ax.view_init(elev=15, azim=135)
        
        # Adjust limits around centroid
        padding = max_dist * 0.3
        self.ax.set_xlim([centroid[0]-padding, centroid[0]+padding])
        self.ax.set_ylim([centroid[1]-padding, centroid[1]+padding])
        self.ax.set_zlim([centroid[2]-padding, centroid[2]+padding])
        
        plt.draw()
    
    def show(self):
        """Show the interactive plot"""
        print("\\n🎮 Interactive Controls:")
        print("🖱️  Mouse: Drag to rotate, scroll to zoom")
        print("🎚️  Slider: Change frame (1-32)")
        print("▶️  Play: Animate through all frames")
        print("🔄 Reset: Return to default viewing angle")
        print("🎯 Center: Center view on point cloud")
        print("❌ Close window to exit")
        print("\\n🔍 Point Cloud Analysis:")
        print(f"📊 Total frames: {self.max_frames}")
        print(f"📦 Points per frame: {self.pts.shape[1]}")
        print(f"📐 Coordinate system: NVIDIA camera space")
        print(f"⚖️  This 'off-center' position is normal for camera coordinates!")
        
        plt.show()

def main():
    # Load the exact NVIDIA processed data
    pts_file = '/notebooks/PMamba/experiments/S001C001P003R002A036_exact_nvidia_pts.npy'
    
    print("🚀 Loading Interactive 3D Point Cloud Viewer...")
    print(f"📁 File: {pts_file}")
    
    try:
        viewer = Interactive3DViewer(pts_file)
        viewer.show()
    except FileNotFoundError:
        print(f"❌ File not found: {pts_file}")
        print("Please run the exact NVIDIA processing first!")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from matplotlib.widgets import Slider, Button
import tkinter as tk
from tkinter import ttk

class Interactive3DViewer:
    def __init__(self, pts_file):
        # Load the point cloud data
        self.pts = np.load(pts_file)
        self.current_frame = 0
        self.max_frames = len(self.pts)
        
        print(f"Loaded point cloud: {self.pts.shape}")
        print(f"Coordinate ranges:")
        print(f"  X: [{self.pts[:, :, 4].min()}, {self.pts[:, :, 4].max()}]")
        print(f"  Y: [{self.pts[:, :, 5].min()}, {self.pts[:, :, 5].max()}]")
        print(f"  Z: [{self.pts[:, :, 6].min()}, {self.pts[:, :, 6].max()}]")
        
        # Set up the plot
        self.fig = plt.figure(figsize=(12, 10))
        
        # Main 3D plot
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # Initial plot
        self.scatter = None
        self.update_plot()
        
        # Add frame slider
        ax_frame = plt.axes([0.1, 0.02, 0.5, 0.03])
        self.frame_slider = Slider(ax_frame, 'Frame', 0, self.max_frames-1, 
                                  valinit=0, valstep=1, valfmt='%d')
        self.frame_slider.on_changed(self.update_frame)
        
        # Add animation controls
        ax_play = plt.axes([0.65, 0.02, 0.1, 0.04])
        self.play_button = Button(ax_play, 'Play')
        self.play_button.on_clicked(self.toggle_animation)
        
        ax_reset = plt.axes([0.8, 0.02, 0.1, 0.04])
        self.reset_button = Button(ax_reset, 'Reset View')
        self.reset_button.on_clicked(self.reset_view)
        
        # Animation state
        self.is_playing = False
        self.anim = None
        
        plt.subplots_adjust(bottom=0.15)
        
    def update_plot(self):
        """Update the 3D plot with current frame"""
        frame_points = self.pts[self.current_frame, :, 4:7]  # x, y, z coordinates
        
        if self.scatter is not None:
            self.scatter.remove()
        
        # Color points by Z coordinate (depth)
        colors = frame_points[:, 2]  # Z coordinate for coloring
        
        self.scatter = self.ax.scatter(frame_points[:, 0], frame_points[:, 1], frame_points[:, 2],
                                     c=colors, cmap='viridis', s=30, alpha=0.7)
        
        # Set labels and title
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y') 
        self.ax.set_zlabel('Z')
        self.ax.set_title(f'NVIDIA Point Cloud - Frame {self.current_frame+1}/{self.max_frames}\\n'
                         f'Interactive 3D Viewer (drag to rotate, scroll to zoom)')
        
        # Set axis limits for consistent view
        self.ax.set_xlim([self.pts[:, :, 4].min()-10, self.pts[:, :, 4].max()+10])
        self.ax.set_ylim([self.pts[:, :, 5].min()-10, self.pts[:, :, 5].max()+10])
        self.ax.set_zlim([self.pts[:, :, 6].min()-10, self.pts[:, :, 6].max()+10])
        
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
                                              interval=200, repeat=True)
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
    
    def show(self):
        """Show the interactive plot"""
        print("\\n🎮 Interactive Controls:")
        print("📱 Mouse: Drag to rotate, scroll to zoom")
        print("🎚️ Slider: Change frame")
        print("▶️ Play: Animate through frames")
        print("🔄 Reset View: Return to default angle")
        print("❌ Close window to exit")
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
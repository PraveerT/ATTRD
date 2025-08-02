import os
import re
import cv2
import copy
import numpy as np
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
import pickle


def save_largest_label(thresh):
    """Keep only the largest connected component"""
    if np.sum(thresh) == 0:
        return thresh
    
    # Find connected components
    num_labels, labels = cv2.connectedComponents(thresh.astype(np.uint8))
    if num_labels <= 1:
        return thresh
    
    # Find largest component (excluding background)
    largest_label = 1
    largest_size = 0
    for label in range(1, num_labels):
        size = np.sum(labels == label)
        if size > largest_size:
            largest_size = size
            largest_label = label
    
    # Create mask with only largest component
    result = (labels == largest_label).astype(np.uint8)
    return result


def generate_points(masked_depth, frame_idx):
    """Generate point cloud from masked depth (NVIDIA style)"""
    # NVIDIA uses x, y = np.where(img != 0) - note the order!
    x_coords, y_coords = np.where(masked_depth > 0)
    if len(x_coords) == 0:
        return np.zeros((0, 4))
    
    depths = masked_depth[x_coords, y_coords]
    # NVIDIA format: [x, y, depth, time] where x,y are image coordinates
    points = np.column_stack([x_coords, y_coords, depths, np.full(len(x_coords), frame_idx)])
    return points.astype(np.float32)


def points_sampling(points, target_size):
    """Sample points to target size"""
    if len(points) == 0:
        return np.zeros((target_size, 4), dtype=np.float32)
    elif len(points) >= target_size:
        # Random sampling without replacement
        indices = np.random.choice(len(points), target_size, replace=False)
        return points[indices]
    else:
        # Repeat points to reach target size
        repeat_factor = target_size // len(points)
        remainder = target_size % len(points)
        
        repeated = np.tile(points, (repeat_factor, 1))
        if remainder > 0:
            extra = points[:remainder]
            repeated = np.concatenate([repeated, extra], axis=0)
        
        return repeated.astype(np.float32)


def uvd2xyz_ntu(points_uvdt):
    """Convert UV-Depth to XYZ coordinates (NVIDIA style with NTU parameters)"""
    # NTU RGB+D camera intrinsics  
    fx = 365.481
    fy = 365.481
    cx = 257.346
    cy = 210.347
    
    # Following NVIDIA format: points are [x, y, d, t] where x,y are image coords
    x_img, y_img, d, t = points_uvdt[:, 0], points_uvdt[:, 1], points_uvdt[:, 2], points_uvdt[:, 3]
    
    # Convert to 3D coordinates (note: x_img corresponds to rows, y_img to columns)
    # In camera coordinates: x = (col - cx) * depth / fx, y = (row - cy) * depth / fy
    x_3d = (y_img - cx) * d / fx / 1000.0  # y_img is column, convert mm to meters
    y_3d = (x_img - cy) * d / fy / 1000.0  # x_img is row, convert mm to meters  
    z_3d = d / 1000.0  # Convert mm to meters
    
    # Return [x, y, z, t] in 3D coordinates
    return np.column_stack([x_3d, y_3d, z_3d, t]).astype(np.float32)


def key_frame_sampling(key_cnt, frame_size):
    """Sample key frames from video sequence"""
    if key_cnt <= frame_size:
        return list(range(key_cnt))
    
    factor = key_cnt * 1.0 / frame_size
    indices = [int(j * factor) for j in range(frame_size)]
    return indices


def process_single_directory(args):
    """Process a single NTU directory"""
    dir_path, dir_name, pts_size, framerate = args
    
    try:
        # Check if already processed
        save_path = os.path.join(dir_path, f"{dir_name}_pts.npy")
        if os.path.exists(save_path):
            return f"✅ {dir_name} (cached)"
        
        # Get all depth files
        depth_files = [f for f in os.listdir(dir_path) 
                      if f.startswith('MDepth-') and f.endswith('.png')]
        
        if len(depth_files) == 0:
            return f"❌ {dir_name} (no depth files)"
        
        depth_files.sort()
        
        # Load depth sequence
        depth_video = []
        for depth_file in depth_files:
            depth_path = os.path.join(dir_path, depth_file)
            depth_img = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
            if depth_img is None:
                continue
            depth_video.append(depth_img.astype(np.float32))
        
        if len(depth_video) == 0:
            return f"❌ {dir_name} (could not load depths)"
        
        depth_video = np.array(depth_video)
        
        # Sample key frames
        ind = key_frame_sampling(len(depth_video), framerate)
        depth_video = depth_video[ind]
        
        # Process frames to extract point clouds
        pts = np.zeros((len(depth_video), pts_size, 8), dtype=np.float32)
        
        for i in range(len(depth_video)):
            frame = depth_video[i]
            
            # NVIDIA-style processing: OTSU thresholding + morphological operations
            # Normalize frame for thresholding
            frame_norm = (frame / frame.max() * 255).astype(np.uint8) if frame.max() > 0 else frame.astype(np.uint8)
            ret, thresh = cv2.threshold(frame_norm, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            thresh = save_largest_label(thresh)
            kernel = np.ones((3, 3), np.uint8)
            thresh = cv2.erode(thresh, kernel)
            
            # Generate and sample points
            points_uvdt = points_sampling(generate_points(frame * thresh, i), pts_size)
            pts[i, :, :4] = points_uvdt
            
            # Convert to 3D coordinates
            pts[i, :, 4:8] = uvd2xyz_ntu(copy.deepcopy(pts[i, :, :4]))
        
        # Save processed point clouds
        np.save(save_path, pts)
        return f"✅ {dir_name}"
        
    except Exception as e:
        return f"❌ {dir_name} (error: {str(e)})"


def load_existing_cache():
    """Load existing directory cache from ntu_dataloader runs"""
    cache_path = "/notebooks/NTU/ntu_dirs_cache_train.pkl"
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'rb') as f:
                cached_data = pickle.load(f)
            return cached_data['all_dirs']
        except Exception as e:
            print(f"⚠️ Could not load existing cache: {e}")
    return None


def main():
    # Parameters
    pts_size = 128
    framerate = 32
    ntu_path = "/notebooks/NTU/nturgb+d_depth_masked"
    num_processes = mp.cpu_count() - 1  # Leave one CPU free
    
    print(f"🚀 NTU Point Cloud Preprocessing")
    print(f"📁 Source: {ntu_path}")
    print(f"🎯 Points per frame: {pts_size}")
    print(f"📽️ Framerate: {framerate}")
    print(f"⚡ Processes: {num_processes}")
    
    # Try to use existing directory cache first
    print("📦 Checking for existing directory cache...")
    cached_dirs = load_existing_cache()
    
    if cached_dirs:
        print(f"✅ Loaded {len(cached_dirs)} directories from existing cache")
        all_dirs = [(d['path'], d['dirname'], pts_size, framerate) for d in cached_dirs]
    else:
        # Get all NTU directories by scanning
        print("📊 Scanning NTU directories...")
        all_dirs = []
        
        for dirname in os.listdir(ntu_path):
            dir_path = os.path.join(ntu_path, dirname)
            if not os.path.isdir(dir_path) or not dirname.startswith('S'):
                continue
            
            # Check if it has depth files
            depth_files = [f for f in os.listdir(dir_path) 
                          if f.startswith('MDepth-') and f.endswith('.png')]
            if len(depth_files) > 0:
                all_dirs.append((dir_path, dirname, pts_size, framerate))
    
    print(f"✅ Found {len(all_dirs)} directories to process")
    
    # Process directories with multiprocessing
    print(f"🔄 Processing with {num_processes} workers...")
    
    with mp.Pool(processes=num_processes) as pool:
        results = list(tqdm(
            pool.imap(process_single_directory, all_dirs),
            total=len(all_dirs),
            desc="Processing NTU directories"
        ))
    
    # Summary
    successful = len([r for r in results if r.startswith('✅')])
    cached = len([r for r in results if '(cached)' in r])
    failed = len([r for r in results if r.startswith('❌')])
    
    print(f"\n📈 Processing Summary:")
    print(f"✅ Successful: {successful}")
    print(f"📦 Cached: {cached}")
    print(f"❌ Failed: {failed}")
    print(f"📊 Total: {len(results)}")
    
    # Save processing log
    log_file = os.path.join(ntu_path, "processing_log.txt")
    with open(log_file, 'w') as f:
        f.write("NTU Point Cloud Processing Log\n")
        f.write(f"Parameters: pts_size={pts_size}, framerate={framerate}\n")
        f.write(f"Processes: {num_processes}\n\n")
        for result in results:
            f.write(f"{result}\n")
    
    print(f"📝 Log saved to: {log_file}")
    print("🎉 Processing complete!")


if __name__ == "__main__":
    main()
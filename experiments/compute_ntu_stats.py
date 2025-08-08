import numpy as np
import os
import sys
from tqdm import tqdm
import cv2
from PIL import Image

def compute_ntu_dataset_stats(data_path="/notebooks/NTU/nturgb+d_depth_masked", max_samples=500):
    """Compute global statistics for the NTU dataset"""
    
    # NTU RGB+D camera intrinsics
    fx = 365.481
    fy = 365.481
    cx = 257.346
    cy = 210.347
    
    print(f"Computing statistics for NTU dataset at: {data_path}")
    print(f"Limiting to {max_samples} samples for efficiency")
    
    if not os.path.exists(data_path):
        print(f"Error: Path does not exist: {data_path}")
        return None
    
    # Get all subdirectories
    subdirs = [d for d in os.listdir(data_path) 
              if os.path.isdir(os.path.join(data_path, d)) and d.startswith('S')]
    
    # Limit samples for efficiency
    if len(subdirs) > max_samples:
        subdirs = subdirs[:max_samples]
    
    print(f"Processing {len(subdirs)} directories...")
    
    # Accumulate statistics
    all_u, all_v, all_d, all_t = [], [], [], []
    processed_count = 0
    
    for dirname in tqdm(subdirs, desc="Loading samples"):
        try:
            dir_path = os.path.join(data_path, dirname)
            
            # Get depth files
            depth_files = [f for f in os.listdir(dir_path) 
                          if f.startswith('MDepth-') and f.endswith('.png')]
            
            if len(depth_files) == 0:
                continue
                
            # Sort depth files
            depth_files.sort()
            
            # Sample a few frames from each sequence for efficiency
            sample_frames = min(8, len(depth_files))
            step = max(1, len(depth_files) // sample_frames)
            sampled_files = depth_files[::step][:sample_frames]
            
            for frame_idx, depth_file in enumerate(sampled_files):
                depth_path = os.path.join(dir_path, depth_file)
                
                # Load depth image
                depth_img = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
                if depth_img is None:
                    depth_img = np.array(Image.open(depth_path))
                
                if depth_img.dtype == np.uint16:
                    depth_img = depth_img.astype(np.float32)
                elif depth_img.dtype not in [np.float32, np.float64]:
                    depth_img = depth_img.astype(np.float32)
                
                # Extract valid points (similar to dataloader logic)
                valid_mask = (depth_img > 0) & (depth_img < 8000)
                y_coords, x_coords = np.where(valid_mask)
                
                if len(x_coords) == 0:
                    continue
                
                # Sample points to avoid memory issues
                max_points_per_frame = 1000
                if len(x_coords) > max_points_per_frame:
                    indices = np.random.choice(len(x_coords), max_points_per_frame, replace=False)
                    x_coords = x_coords[indices]
                    y_coords = y_coords[indices]
                
                depths = depth_img[y_coords, x_coords]
                
                # Store coordinates
                all_u.extend(x_coords.astype(float).tolist())
                all_v.extend(y_coords.astype(float).tolist()) 
                all_d.extend(depths.tolist())
                all_t.extend([frame_idx] * len(x_coords))
            
            processed_count += 1
            
        except Exception as e:
            print(f"Error processing {dirname}: {e}")
            continue
    
    if len(all_u) == 0:
        print("No valid data found!")
        return None
    
    # Convert to numpy arrays
    all_u = np.array(all_u)
    all_v = np.array(all_v)
    all_d = np.array(all_d)
    all_t = np.array(all_t)
    
    print(f"Processed {processed_count} directories with {len(all_u)} total points")
    
    # Compute statistics
    stats = {
        'u_mean': np.mean(all_u),
        'u_std': np.std(all_u),
        'v_mean': np.mean(all_v),
        'v_std': np.std(all_v),
        'd_mean': np.mean(all_d),
        'd_std': np.std(all_d),
        'd_min': np.min(all_d),
        'd_max': np.max(all_d),
        't_mean': np.mean(all_t),
        't_std': np.std(all_t),
        'cx': cx,
        'cy': cy,
        'fx': fx,
        'fy': fy
    }
    
    print("NTU Dataset Statistics:")
    for key, value in stats.items():
        print(f"{key}: {value:.6f}")
    
    # Save statistics
    np.save('ntu_dataset_stats.npy', stats)
    print("\nStatistics saved to ntu_dataset_stats.npy")
    
    return stats

if __name__ == "__main__":
    compute_ntu_dataset_stats()
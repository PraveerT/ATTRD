import numpy as np
import os
import re
from tqdm import tqdm

def compute_nvidia_dataset_stats():
    """Compute global statistics for the NVIDIA dataset"""
    r = re.compile('[ \t\n\r:]+')
    
    # Get all data files
    prefix = "../dataset/Nvidia/Processed"
    train_files = open(prefix + "/train_depth_list.txt").readlines()
    test_files = open(prefix + "/test_depth_list.txt").readlines()
    all_files = train_files + test_files
    
    print(f"Computing statistics for {len(all_files)} files...")
    
    # Accumulate statistics
    all_x, all_y, all_z, all_t = [], [], [], []
    
    for file_path in tqdm(all_files, desc="Loading files"):
        try:
            data_path = f"../dataset/{r.split(file_path)[1][1:-4]}_pts.npy"
            if os.path.exists(data_path):
                pts = np.load(data_path).astype(float)
                timestep, pts_size, channels = pts.shape
                pts = pts.reshape(-1, channels)
                
                all_x.extend(pts[:, 0].tolist())
                all_y.extend(pts[:, 1].tolist())
                all_z.extend(pts[:, 2].tolist())
                all_t.extend(pts[:, 3].tolist())
        except Exception as e:
            print(f"Error loading {data_path}: {e}")
            continue
    
    # Convert to numpy arrays
    all_x = np.array(all_x)
    all_y = np.array(all_y)
    all_z = np.array(all_z)
    all_t = np.array(all_t)
    
    # Compute statistics
    stats = {
        'x_mean': np.mean(all_x),
        'x_std': np.std(all_x),
        'y_mean': np.mean(all_y), 
        'y_std': np.std(all_y),
        'z_mean': np.mean(all_z),
        'z_std': np.std(all_z),
        'z_min': np.min(all_z),
        'z_max': np.max(all_z),
        't_mean': np.mean(all_t),
        't_std': np.std(all_t)
    }
    
    print("Dataset Statistics:")
    for key, value in stats.items():
        print(f"{key}: {value:.6f}")
    
    # Save statistics
    np.save('nvidia_dataset_stats.npy', stats)
    print("\nStatistics saved to nvidia_dataset_stats.npy")
    
    return stats

if __name__ == "__main__":
    compute_nvidia_dataset_stats()
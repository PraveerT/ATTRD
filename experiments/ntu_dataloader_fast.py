import os
import re
import sys
import numpy as np
import torch.utils.data as data
from utils.pts_transform import *
import pickle
from tqdm import tqdm

sys.path.append("..")


class NTULoaderFast(data.Dataset):
    """
    Fast NTU RGB+D dataset loader using pre-processed point clouds
    """
    def __init__(self, framerate=32, valid_subject=None, phase="train", datatype="depth", 
                 inputs_type="pts", pts_size=128, data_path="/notebooks/NTU/nturgb+d_depth_masked"):
        self.phase = phase
        self.datatype = datatype
        self.inputs_type = inputs_type
        self.framerate = framerate
        self.valid_subject = valid_subject
        self.pts_size = pts_size
        self.data_path = data_path
        
        # Directory cache file
        self.dir_cache_file = os.path.join(os.path.dirname(data_path), f"ntu_dirs_cache_{phase}.pkl")
        
        print("📁 Loading NTU dataset (fast mode)...")
        self.inputs_list = self.get_inputs_list()
        
        print(f"✅ NTU Dataset: {len(self.inputs_list)} samples")
        if phase == "train":
            self.transform = self.transform_init("train")
        elif phase in ["test", "valid"]:
            self.transform = self.transform_init("test")

    def __getitem__(self, index):
        """Load pre-processed point clouds"""
        dir_info = self.inputs_list[index]
        
        # Parse action class from directory info
        action_class = dir_info['action'] - 1  # Convert to 0-based indexing
        
        try:
            # Load pre-processed point clouds
            pts_file = os.path.join(dir_info['path'], f"{dir_info['dirname']}_pts.npy")
            if not os.path.exists(pts_file):
                print(f"⚠️ Point cloud file not found: {pts_file}")
                # Return dummy data
                dummy_points = np.zeros((self.framerate, self.pts_size, 8))
                return dummy_points, action_class, dir_info['dirname']
            
            # Load point clouds
            point_clouds = np.load(pts_file).astype(np.float32)
            
            # Adjust framerate if needed
            if len(point_clouds) != self.framerate:
                # Resample to target framerate
                indices = self.key_frame_sampling(len(point_clouds), self.framerate)
                point_clouds = point_clouds[indices]
            
            # Adjust point size if needed
            if point_clouds.shape[1] != self.pts_size:
                # Resample points
                point_clouds = self.resample_points(point_clouds, self.pts_size)
            
            # Normalize point clouds (only use first 4 channels: u, v, d, t)
            point_clouds_normalized = self.normalize(point_clouds[:, :, :4], self.framerate)
            
            return point_clouds_normalized, action_class, dir_info['dirname']
            
        except Exception as e:
            print(f"⚠️ Error processing {dir_info['dirname']}: {e}")
            # Return dummy data
            dummy_points = np.zeros((self.framerate, self.pts_size, 4))
            return dummy_points, action_class, dir_info['dirname']

    def resample_points(self, point_clouds, target_pts_size):
        """Resample point clouds to target point size"""
        T, current_pts, channels = point_clouds.shape
        if current_pts == target_pts_size:
            return point_clouds
        
        resampled = np.zeros((T, target_pts_size, channels), dtype=np.float32)
        
        for t in range(T):
            pts = point_clouds[t]
            if current_pts >= target_pts_size:
                # Random sampling
                indices = np.random.choice(current_pts, target_pts_size, replace=False)
                resampled[t] = pts[indices]
            else:
                # Repeat points
                indices = np.random.choice(current_pts, target_pts_size, replace=True)
                resampled[t] = pts[indices]
        
        return resampled

    def get_inputs_list(self):
        """Get list of NTU dataset directories with caching"""
        
        # Check if cache file exists
        if os.path.exists(self.dir_cache_file):
            try:
                print(f"📦 Loading cached directory list from {self.dir_cache_file}")
                with open(self.dir_cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                
                # Verify cache is for the same data path and subject
                if (cached_data['data_path'] == self.data_path and 
                    cached_data['valid_subject'] == self.valid_subject):
                    
                    all_dirs = cached_data['all_dirs']
                    print(f"✅ Loaded {len(all_dirs)} directories from cache")
                    
                    # Apply phase filtering
                    filtered_dirs = self.filter_dirs_by_phase(all_dirs)
                    
                    return filtered_dirs
                else:
                    print("⚠️ Cache is for different data path or subject, rescanning...")
            except Exception as e:
                print(f"⚠️ Error loading cache: {e}, rescanning...")
        
        # If no cache or cache invalid, scan directories
        all_dirs = self.scan_directories()
        
        # Save to cache
        self.save_directory_cache(all_dirs)
        
        # Apply phase filtering
        filtered_dirs = self.filter_dirs_by_phase(all_dirs)
        
        return filtered_dirs
    
    def scan_directories(self):
        """Scan all directories in the dataset"""
        all_dirs = []
        
        print(f"🔍 Scanning directories in: {self.data_path}")
        
        # Check if path exists
        if not os.path.exists(self.data_path):
            print(f"❌ Path does not exist: {self.data_path}")
            return []
        
        # Get all subdirectories
        subdirs = [d for d in os.listdir(self.data_path) 
                  if os.path.isdir(os.path.join(self.data_path, d)) and d.startswith('S')]
        
        print(f"📊 Found {len(subdirs)} action directories, parsing...")
        
        for dirname in tqdm(subdirs, desc="Parsing directories"):
            dir_path = os.path.join(self.data_path, dirname)
            
            # Check if directory has processed point clouds
            pts_file = os.path.join(dir_path, f"{dirname}_pts.npy")
            if not os.path.exists(pts_file):
                continue
            
            # Parse NTU directory name: SsssCcccPpppRrrrAaaa
            if len(dirname) >= 20 and dirname.startswith('S'):
                try:
                    setup = int(dirname[1:4])
                    camera = int(dirname[5:8])
                    subject = int(dirname[9:12])
                    replication = int(dirname[13:16])
                    action = int(dirname[17:20])
                except:
                    continue
            else:
                continue
            
            # Get number of frames from processed file
            try:
                pts_data = np.load(pts_file)
                num_frames = len(pts_data)
            except:
                continue
            
            all_dirs.append({
                'path': dir_path,
                'dirname': dirname,
                'setup': setup,
                'camera': camera,
                'subject': subject,
                'replication': replication,
                'action': action,
                'num_frames': num_frames
            })
        
        print(f"✅ Scanned {len(all_dirs)} valid directories")
        return all_dirs
    
    def save_directory_cache(self, all_dirs):
        """Save directory list to cache file"""
        try:
            cache_data = {
                'data_path': self.data_path,
                'valid_subject': self.valid_subject,
                'all_dirs': all_dirs,
                'timestamp': os.path.getctime(self.data_path) if os.path.exists(self.data_path) else 0
            }
            
            with open(self.dir_cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            print(f"💾 Saved directory cache to {self.dir_cache_file}")
        except Exception as e:
            print(f"⚠️ Could not save cache: {e}")
    
    def filter_dirs_by_phase(self, all_dirs):
        """Filter directories by phase and subject"""
        # Filter by subject if specified
        if self.valid_subject is not None:
            if self.phase == "train":
                filtered_dirs = [d for d in all_dirs if d['subject'] != self.valid_subject]
            elif self.phase == "valid":
                filtered_dirs = [d for d in all_dirs if d['subject'] == self.valid_subject]
            else:  # test
                filtered_dirs = all_dirs  # Use all for test
        else:
            filtered_dirs = all_dirs
        
        # Split train/test based on NTU recommended protocol
        if self.phase == "train":
            # Training samples (setups 2, 4, 6, 8, 10, 12, 14, 16, 18, 20)
            final_dirs = [d for d in filtered_dirs if d['setup'] % 2 == 0]
        elif self.phase == "test":
            # Testing samples (setups 1, 3, 5, 7, 9, 11, 13, 15, 17, 19)
            final_dirs = [d for d in filtered_dirs if d['setup'] % 2 == 1]
        else:  # valid
            # Use training setup for validation
            final_dirs = [d for d in filtered_dirs if d['setup'] % 2 == 0]
        
        print(f"🎯 {self.phase} set: {len(final_dirs)} samples")
        
        # Show some example directories
        if len(final_dirs) > 0:
            print(f"📋 Sample directories:")
            for i, d in enumerate(final_dirs[:3]):
                print(f"  {i+1}. {d['dirname']} ({d['num_frames']} frames)")
        
        return final_dirs

    def __len__(self):
        return len(self.inputs_list)

    def normalize(self, pts, fs):
        """Normalize point cloud coordinates (same as original)"""
        # NTU camera intrinsics
        cx = 257.346
        cy = 210.347
        
        timestep, pts_size, channels = pts.shape
        pts = pts.reshape(-1, channels)
        
        # Normalize u, v coordinates to [-1, 1]
        pts[:, 0] = (pts[:, 0] - cx) / cx  # u coordinate
        pts[:, 1] = (pts[:, 1] - cy) / cy  # v coordinate
        
        # Normalize depth
        if (pts[:, 2].max() - pts[:, 2].min()) != 0:
            pts[:, 2] = (pts[:, 2] - np.mean(pts[:, 2])) / (pts[:, 2].max() - pts[:, 2].min()) * 2
        
        # Normalize temporal coordinate
        pts[:, 3] = (pts[:, 3] - fs / 2) / fs * 2
        
        # Apply transformations
        pts = self.transform(pts)
        pts = pts.reshape(timestep, pts_size, channels)
        
        return pts

    @staticmethod
    def transform_init(phase):
        """Initialize data augmentation transforms"""
        if phase == 'train':
            transform = Compose([
                PointcloudToTensor(),
                PointcloudScale(lo=0.9, hi=1.1),
                PointcloudRotatePerturbation(angle_sigma=0.06, angle_clip=0.18),
                PointcloudRandomInputDropout(max_dropout_ratio=0.2),
            ])
        else:
            transform = Compose([
                PointcloudToTensor(),
            ])
        return transform

    @staticmethod
    def key_frame_sampling(key_cnt, frame_size):
        """Sample key frames from video sequence"""
        if key_cnt <= frame_size:
            return list(range(key_cnt))
        
        factor = key_cnt * 1.0 / frame_size
        indices = [int(j * factor) for j in range(frame_size)]
        return indices


if __name__ == "__main__":
    # Test the fast dataloader
    dataset = NTULoaderFast(
        framerate=32,
        phase="train",
        pts_size=128,
        data_path="/notebooks/NTU/nturgb+d_depth_masked"
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"Sample shape: {sample[0].shape}, Label: {sample[1]}, ID: {sample[2]}")
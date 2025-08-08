import os
import re
import sys
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as data
from utils.pts_transform import *
import cv2
from tqdm import tqdm
from PIL import Image
import pickle

sys.path.append("..")


class NTULoader(data.Dataset):
    """
    NTU RGB+D dataset loader with on-the-fly GPU-based point cloud extraction from depth data
    """
    def __init__(self, framerate=32, valid_subject=None, phase="train", datatype="depth", 
                 inputs_type="pts", pts_size=512, data_path="/notebooks/NTU/nturgb+d_depth_masked",
                 chunk_size=8, use_cache=True):
        self.phase = phase
        self.datatype = datatype
        self.inputs_type = inputs_type
        self.framerate = framerate
        self.valid_subject = valid_subject
        self.pts_size = pts_size
        self.data_path = data_path
        self.chunk_size = chunk_size
        self.use_cache = use_cache
        
        # NTU RGB+D camera intrinsics (from NTU dataset paper)
        self.fx = 365.481
        self.fy = 365.481
        self.cx = 257.346
        self.cy = 210.347
        
        # Cache for processed point clouds
        self.point_cloud_cache = {} if use_cache else None
        
        # Directory cache file
        self.dir_cache_file = os.path.join(os.path.dirname(data_path), f"ntu_dirs_cache_{phase}.pkl")
        
        # Load global dataset statistics
        try:
            self.dataset_stats = np.load('ntu_dataset_stats.npy', allow_pickle=True).item()
            print("Loaded global NTU dataset statistics for consistent normalization")
        except FileNotFoundError:
            print("Warning: Global NTU dataset statistics not found, using default normalization")
            self.dataset_stats = None
        
        print("📁 Loading NTU dataset files...")
        self.inputs_list = self.get_inputs_list()
        self.r = re.compile('[ \t\n\r:]+')

        print(f"✅ NTU Dataset: {len(self.inputs_list)} samples")
        if phase == "train":
            self.transform = self.transform_init("train")
        elif phase in ["test", "valid"]:
            self.transform = self.transform_init("test")

    def __getitem__(self, index):
        """Load depth sequence and extract point clouds on-the-fly"""
        dir_info = self.inputs_list[index]
        
        # Check cache first
        cache_key = f"{dir_info['dirname']}_{self.framerate}_{self.pts_size}"
        if self.use_cache and cache_key in self.point_cloud_cache:
            point_clouds, action_class = self.point_cloud_cache[cache_key]
            # Make a copy of the cached numpy array
            if isinstance(point_clouds, np.ndarray):
                return point_clouds.copy(), action_class, dir_info['dirname']
            else:
                return np.array(point_clouds), action_class, dir_info['dirname']
        
        # Parse action class from directory info
        action_class = dir_info['action'] - 1  # Convert to 0-based indexing
        
        try:
            # Load depth sequence from PNG files
            depth_sequence = self.load_depth_sequence(dir_info)
            
            # Sample key frames
            frame_indices = self.key_frame_sampling(len(depth_sequence), self.framerate)
            depth_sequence = depth_sequence[frame_indices]
            
            # Process point clouds (try GPU first, fallback to fast CPU)
            try:
                point_clouds = self.extract_point_clouds_chunked(depth_sequence)
            except Exception as gpu_error:
                print(f"⚠️ GPU extraction failed, using fast CPU method: {gpu_error}")
                point_clouds = self.extract_point_clouds_cpu_fast(depth_sequence)
            
            # Normalize point clouds
            point_clouds = self.normalize(point_clouds, self.framerate)
            
            # Cache the result (ensure it's a numpy array)
            if self.use_cache:
                if isinstance(point_clouds, np.ndarray):
                    self.point_cloud_cache[cache_key] = (point_clouds.copy(), action_class)
                else:
                    self.point_cloud_cache[cache_key] = (np.array(point_clouds), action_class)
            
            return point_clouds, action_class, dir_info['dirname']
            
        except Exception as e:
            print(f"⚠️ Error processing {dir_info['dirname']}: {e}")
            # Return dummy data
            dummy_points = np.zeros((self.framerate, self.pts_size, 4))
            return dummy_points, action_class, dir_info['dirname']

    def load_depth_sequence(self, dir_info):
        """Load depth sequence from PNG files"""
        dir_path = dir_info['path']
        
        # Get all depth files sorted by frame number
        depth_files = [f for f in os.listdir(dir_path) 
                      if f.startswith('MDepth-') and f.endswith('.png')]
        depth_files.sort()
        
        # Load depth images
        depth_sequence = []
        for depth_file in depth_files:
            depth_path = os.path.join(dir_path, depth_file)
            
            # Load depth image (16-bit PNG)
            depth_img = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
            if depth_img is None:
                # Try with PIL if opencv fails
                depth_img = np.array(Image.open(depth_path))
            
            # Convert uint16 to float32 for PyTorch compatibility
            if depth_img.dtype == np.uint16:
                depth_img = depth_img.astype(np.float32)
            elif depth_img.dtype not in [np.float32, np.float64]:
                depth_img = depth_img.astype(np.float32)
            
            depth_sequence.append(depth_img)
        
        return np.array(depth_sequence, dtype=np.float32)

    def extract_point_clouds_chunked(self, depth_sequence):
        """Extract point clouds in chunks with progress bar"""
        T, H, W = depth_sequence.shape
        all_point_clouds = []
        
        # Process in chunks to manage GPU memory
        chunk_size = min(self.chunk_size, T)
        num_chunks = (T + chunk_size - 1) // chunk_size
        
        for i in range(0, T, chunk_size):
            end_idx = min(i + chunk_size, T)
            chunk = depth_sequence[i:end_idx]
            
            # Convert to torch tensor and move to GPU
            depth_tensor = torch.from_numpy(chunk).float()
            use_gpu = False
            if torch.cuda.is_available():
                try:
                    depth_tensor = depth_tensor.cuda()
                    use_gpu = True
                except RuntimeError:
                    # Fall back to CPU silently
                    pass
            
            # Extract point clouds for this chunk
            chunk_points = self.extract_point_clouds_gpu(depth_tensor)
            
            # Move back to CPU and convert to numpy
            chunk_points = chunk_points.cpu().numpy()
            all_point_clouds.append(chunk_points)
            
            # Clear GPU cache only if using GPU
            if use_gpu and torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Concatenate all chunks
        return np.concatenate(all_point_clouds, axis=0)

    def extract_point_clouds_cpu_fast(self, depth_sequence):
        """NVIDIA-style point cloud extraction"""
        T, H, W = depth_sequence.shape
        pts = np.zeros((T, self.pts_size, 8), dtype=np.float32)
        
        for i in range(T):
            frame = depth_sequence[i]
            
            # NVIDIA-style processing: OTSU thresholding + morphological operations
            ret, thresh = cv2.threshold(frame.astype(np.uint8), 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            thresh = self.save_largest_label(thresh)
            kernel = np.ones((3, 3), np.uint8)
            thresh = cv2.erode(thresh, kernel)
            
            # Generate and sample points (NVIDIA style)
            points_uvdt = self.points_sampling(self.generate_points(frame * thresh, i), self.pts_size)
            pts[i, :, :4] = points_uvdt
            
            # Convert to 3D coordinates (simplified version of uvd2xyz_sherc)
            pts[i, :, 4:8] = self.uvd2xyz_ntu(points_uvdt.copy())
        
        return pts
    
    def save_largest_label(self, binary_img):
        """Keep only the largest connected component"""
        if np.sum(binary_img) == 0:
            return binary_img
        
        # Find connected components
        num_labels, labels = cv2.connectedComponents(binary_img.astype(np.uint8))
        if num_labels <= 1:
            return binary_img
        
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
    
    def generate_points(self, masked_depth, frame_idx):
        """Generate point cloud from masked depth"""
        y_coords, x_coords = np.where(masked_depth > 0)
        if len(x_coords) == 0:
            return np.zeros((0, 4))
        
        depths = masked_depth[y_coords, x_coords]
        points = np.column_stack([x_coords, y_coords, depths, np.full(len(x_coords), frame_idx)])
        return points.astype(np.float32)
    
    def points_sampling(self, points, target_size):
        """Sample points to target size (NVIDIA style)"""
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
    
    def uvd2xyz_ntu(self, points_uvdt):
        """Convert UV-Depth to XYZ coordinates for NTU dataset"""
        # points_uvdt: [u, v, d, t]
        u, v, d, t = points_uvdt[:, 0], points_uvdt[:, 1], points_uvdt[:, 2], points_uvdt[:, 3]
        
        # Convert to meters (NTU depth is in mm)
        d_meters = d / 1000.0
        
        # Convert to 3D coordinates using NTU camera intrinsics
        x = (u - self.cx) * d_meters / self.fx
        y = (v - self.cy) * d_meters / self.fy
        z = d_meters
        
        # Return [x, y, z, t]
        return np.column_stack([x, y, z, t]).astype(np.float32)

    def extract_point_clouds_gpu(self, depth_tensor):
        """
        Extract point clouds from depth tensor on GPU
        Args:
            depth_tensor: (T, H, W) depth sequence
        Returns:
            point_clouds: (T, pts_size, 4) point clouds with [u, v, d, t]
        """
        T, H, W = depth_tensor.shape
        device = depth_tensor.device
        
        # Create coordinate grids
        u_coords, v_coords = torch.meshgrid(
            torch.arange(W, device=device, dtype=torch.float32),
            torch.arange(H, device=device, dtype=torch.float32),
            indexing='xy'
        )
        
        point_clouds = []
        
        for t in range(T):
            depth_frame = depth_tensor[t]
            
            # Threshold depth (remove background and invalid depths)
            valid_mask = (depth_frame > 0) & (depth_frame < 8000)  # Valid depth range in mm
            
            # Apply morphological operations to clean up the mask
            valid_mask = self.morphological_ops_gpu(valid_mask)
            
            # Get valid points
            valid_indices = torch.where(valid_mask)
            if len(valid_indices[0]) == 0:
                # If no valid points, create dummy points
                points = torch.zeros((self.pts_size, 4), device=device)
                point_clouds.append(points)
                continue
            
            u_valid = u_coords[valid_indices]
            v_valid = v_coords[valid_indices]
            d_valid = depth_frame[valid_indices]
            
            # Create point cloud [u, v, d, t]
            points = torch.stack([u_valid, v_valid, d_valid, 
                                torch.full_like(u_valid, t, dtype=torch.float32)], dim=1)
            
            # Sample points to fixed size
            points = self.sample_points_gpu(points, self.pts_size)
            point_clouds.append(points)
        
        return torch.stack(point_clouds)  # (T, pts_size, 4)

    def morphological_ops_gpu(self, mask):
        """Apply morphological operations on GPU to clean up the mask"""
        # Convert to float for morphological operations
        mask_float = mask.float().unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        
        # Erosion followed by dilation (opening operation)
        kernel_size = 3
        padding = kernel_size // 2
        
        # Erosion (minimum pooling)
        mask_eroded = -F.max_pool2d(-mask_float, kernel_size, stride=1, padding=padding)
        
        # Dilation (maximum pooling)
        mask_dilated = F.max_pool2d(mask_eroded, kernel_size, stride=1, padding=padding)
        
        return mask_dilated.squeeze().bool()

    def sample_points_gpu(self, points, target_size):
        """Sample points to target size on GPU"""
        num_points = points.shape[0]
        
        if num_points == 0:
            return torch.zeros((target_size, 4), device=points.device)
        elif num_points >= target_size:
            # Random sampling
            indices = torch.randperm(num_points, device=points.device)[:target_size]
            return points[indices]
        else:
            # Repeat points to reach target size
            repeat_factor = target_size // num_points
            remainder = target_size % num_points
            
            repeated_points = points.repeat(repeat_factor, 1)
            if remainder > 0:
                extra_points = points[:remainder]
                repeated_points = torch.cat([repeated_points, extra_points], dim=0)
            
            return repeated_points

    def get_inputs_list(self):
        """Get list of NTU dataset directories with caching"""
        
        # Check if cache file exists and is recent
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
                    
                    # Show some example directories
                    if len(filtered_dirs) > 0:
                        print(f"📋 Sample directories:")
                        for i, d in enumerate(filtered_dirs[:3]):
                            print(f"  {i+1}. {d['dirname']} ({d['num_frames']} frames)")
                    
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
        
        # Get all subdirectories (each contains one action sequence)
        subdirs = [d for d in os.listdir(self.data_path) 
                  if os.path.isdir(os.path.join(self.data_path, d)) and d.startswith('S')]
        
        print(f"📊 Found {len(subdirs)} action directories, parsing...")
        
        for dirname in tqdm(subdirs, desc="Parsing directories"):
            dir_path = os.path.join(self.data_path, dirname)
            
            # Check if directory has depth images
            depth_files = [f for f in os.listdir(dir_path) if f.startswith('MDepth-') and f.endswith('.png')]
            if len(depth_files) == 0:
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
            
            all_dirs.append({
                'path': dir_path,
                'dirname': dirname,
                'setup': setup,
                'camera': camera,
                'subject': subject,
                'replication': replication,
                'action': action,
                'num_frames': len(depth_files)
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
        return final_dirs

    def __len__(self):
        return len(self.inputs_list)

    def normalize(self, pts, fs):
        """Normalize point cloud coordinates using global dataset statistics"""
        timestep, pts_size, channels = pts.shape
        pts = pts.reshape(-1, channels)
        
        if self.dataset_stats is not None:
            # Use global dataset statistics for consistent normalization
            pts[:, 0] = (pts[:, 0] - self.dataset_stats['u_mean']) / self.dataset_stats['u_std']  # u coordinate
            pts[:, 1] = (pts[:, 1] - self.dataset_stats['v_mean']) / self.dataset_stats['v_std']  # v coordinate  
            pts[:, 2] = (pts[:, 2] - self.dataset_stats['d_mean']) / self.dataset_stats['d_std']  # depth
            pts[:, 3] = (pts[:, 3] - self.dataset_stats['t_mean']) / self.dataset_stats['t_std']  # temporal
        else:
            # Fallback to original per-sample normalization if stats not available
            pts[:, 0] = (pts[:, 0] - self.cx) / self.cx  # u coordinate
            pts[:, 1] = (pts[:, 1] - self.cy) / self.cy  # v coordinate
            
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
    # Test the dataloader
    dataset = NTULoader(
        framerate=32,
        phase="train",
        pts_size=128,
        data_path="/notebooks/NTU/nturgb+d_depth_masked"
    )
    
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=2,
        shuffle=True,
        num_workers=0,
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    for batch_idx, (data, labels, paths) in enumerate(dataloader):
        print(f"Batch {batch_idx}:")
        print(f"  Data shape: {data.shape}")
        print(f"  Labels: {labels}")
        print(f"  Paths: {paths}")
        
        if batch_idx >= 2:  # Test first few batches
            break
import os
import sys
import re
import cv2
import copy
import numpy as np
import pickle
import multiprocessing as mp
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

def process_single_ntu_directory(args):
    """Process a single NTU directory using EXACT NVIDIA pipeline - FORCE OVERWRITE"""
    dir_path, dirname, pts_size = args
    
    try:
        # FORCE PROCESSING - NO CACHE CHECK (will overwrite existing files)
        save_path = os.path.join(dir_path, f"{dirname}_pts.npy")
        
        # Step 1: Load NTU data in NVIDIA format
        depth_video = load_ntu_as_nvidia_format(dir_path)
        if depth_video is None:
            return f"❌ {dirname} (could not load)"
        
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
        
        # Save processed point clouds (OVERWRITES existing files)
        np.save(save_path, pts)
        return f"✅ {dirname} (processed)"
        
    except Exception as e:
        return f"❌ {dirname} (error: {str(e)})"

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
    # Parameters - EXACT NVIDIA settings
    pts_size = 512  # NVIDIA uses 512 points per frame
    ntu_path = "/notebooks/NTU/nturgb+d_depth_masked"
    num_processes = mp.cpu_count() - 1  # Leave one CPU free
    
    print(f"🚀 NTU Point Cloud Processing - EXACT NVIDIA Method (OVERWRITE MODE)")
    print(f"📁 Source: {ntu_path}")
    print(f"🎯 Points per frame: {pts_size} (NVIDIA standard)")
    print(f"📽️ Framerate: 32")
    print(f"⚡ Processes: {num_processes}")
    print(f"🔧 Method: Exact NVIDIA pipeline with SHERC camera params")
    print(f"⚠️  OVERWRITE: Will replace ALL existing _pts.npy files")
    
    # Try to use existing directory cache first
    print("📦 Checking for existing directory cache...")
    cached_dirs = load_existing_cache()
    
    if cached_dirs:
        print(f"✅ Loaded {len(cached_dirs)} directories from existing cache")
        all_dirs = [(d['path'], d['dirname'], pts_size) for d in cached_dirs]
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
                all_dirs.append((dir_path, dirname, pts_size))
    
    print(f"✅ Found {len(all_dirs)} directories to process")
    print(f"🔄 Starting processing with {num_processes} workers (OVERWRITE MODE)...")
    
    # Process directories with multiprocessing
    with mp.Pool(processes=num_processes) as pool:
        results = list(tqdm(
            pool.imap(process_single_ntu_directory, all_dirs),
            total=len(all_dirs),
            desc="Processing NTU directories (NVIDIA overwrite)"
        ))
    
    # Summary
    successful = len([r for r in results if r.startswith('✅')])
    failed = len([r for r in results if r.startswith('❌')])
    
    print(f"\n📈 Processing Summary:")
    print(f"✅ Successfully processed: {successful}")
    print(f"❌ Failed: {failed}")
    print(f"📊 Total: {len(results)}")
    
    # Save processing log
    log_file = os.path.join(ntu_path, "nvidia_processing_overwrite_log.txt")
    with open(log_file, 'w') as f:
        f.write("NTU Point Cloud Processing Log - EXACT NVIDIA Method (OVERWRITE)\n")
        f.write(f"Parameters: pts_size={pts_size}, framerate=32\n")
        f.write(f"Method: NVIDIA pipeline with SHERC camera parameters\n")
        f.write(f"Mode: OVERWRITE - replaced all existing _pts.npy files\n")
        f.write(f"Processes: {num_processes}\n\n")
        for result in results:
            f.write(f"{result}\n")
    
    print(f"📝 Log saved to: {log_file}")
    print("🎉 NVIDIA processing complete (OVERWRITE MODE)!")
    
    # Show some statistics
    if successful > 0:
        print(f"\n📊 Results:")
        print(f"  Format: (32 frames, 512 points, 8 channels)")
        print(f"  Channels: [x_img, y_img, depth, t, x_3d, y_3d, z_3d, t_copy]")
        print(f"  Coordinate system: NVIDIA camera space with SHERC params")
        print(f"  Data type: int64")
        print(f"  ALL files have been regenerated with exact NVIDIA method!")

if __name__ == "__main__":
    main()
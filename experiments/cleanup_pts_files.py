import os
import glob
from tqdm import tqdm

def cleanup_pts_files():
    """Remove all existing _pts.npy files from NTU dataset"""
    
    ntu_path = "/notebooks/NTU/nturgb+d_depth_masked"
    
    print(f"🧹 Cleaning up existing _pts.npy files from {ntu_path}")
    
    # Find all _pts.npy files
    pattern = os.path.join(ntu_path, "**", "*_pts.npy")
    pts_files = glob.glob(pattern, recursive=True)
    
    print(f"📊 Found {len(pts_files)} existing point cloud files to remove")
    
    if len(pts_files) == 0:
        print("✅ No existing _pts.npy files found - already clean!")
        return
    
    # Show first few files as examples
    print(f"📋 Example files to be removed:")
    for i, f in enumerate(pts_files[:5]):
        print(f"  {i+1}. {os.path.basename(f)}")
    if len(pts_files) > 5:
        print(f"  ... and {len(pts_files)-5} more")
    
    # Confirm before deletion
    response = input(f"\n⚠️  Delete all {len(pts_files)} files? (y/N): ")
    if response.lower() != 'y':
        print("❌ Deletion cancelled")
        return
    
    # Remove files with progress bar
    print(f"🗑️  Removing {len(pts_files)} files...")
    removed_count = 0
    
    for pts_file in tqdm(pts_files, desc="Removing files"):
        try:
            os.remove(pts_file)
            removed_count += 1
        except Exception as e:
            print(f"⚠️ Error removing {pts_file}: {e}")
    
    print(f"✅ Successfully removed {removed_count}/{len(pts_files)} point cloud files")
    
    # Verify cleanup
    remaining_files = glob.glob(pattern, recursive=True)
    if len(remaining_files) == 0:
        print("🎉 Cleanup complete - ready for NVIDIA processing regeneration!")
    else:
        print(f"⚠️ {len(remaining_files)} files still remain")

if __name__ == "__main__":
    cleanup_pts_files()
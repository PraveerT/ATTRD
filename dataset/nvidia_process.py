import re
import cv2
import copy
import utils
import numpy as np
from multiprocessing import Pool, cpu_count

def process_video(args):
    npy_id, npy_path, pts_size, r = args
    
    parts = r.split(npy_path)
    npy_path = parts[1]
    label = parts[2]
    print(npy_id, npy_path)
    
    depth_video = np.load(npy_path)
    ind = utils.key_frame_sampling(len(depth_video), 32)
    depth_video = depth_video[ind]
    pts = np.zeros((len(depth_video), pts_size, 8), dtype=int)
    
    for i in range(len(depth_video)):
        frame = depth_video[i, :, :, 0]

        # 🔹 Singular improvement: smooth depth before thresholding
        frame_blur = cv2.GaussianBlur(frame, (5, 5), 0)

        ret, thresh = cv2.threshold(frame_blur, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thresh = utils.save_largest_label(thresh)

        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.erode(thresh, kernel)

        pts[i, :, :4] = utils.points_sampling(utils.generate_points(frame * thresh, i), pts_size)
        pts[i, :, 4:8] = utils.uvd2xyz_sherc(copy.deepcopy(pts[i, :, :4]))
    # utils.show_video_point_clouds(pts)
    save_path = npy_path[:-4] + "_pts"
    np.save(save_path, pts)

if __name__ == "__main__":
    pts_size = 512
    prefix = "./Nvidia"
    r = re.compile('[ \t\n\r:]+')
    train_list_path = f"{prefix}/Processed/train_depth_list.txt"
    test_list_path = f"{prefix}/Processed/test_depth_list.txt"
    total_list = open(test_list_path).readlines() + open(train_list_path).readlines()
    data_pts_size = np.zeros((len(total_list), 80))
    
    # Prepare arguments for multiprocessing
    args_list = [(npy_id, npy_path, pts_size, r) for npy_id, npy_path in enumerate(total_list)]
    
    # Use multiprocessing
    with Pool(processes=cpu_count()) as pool:
        pool.map(process_video, args_list)

import concurrent.futures
import os
from tqdm import tqdm
import cv2
import numpy as np
import tempfile
from matplotlib import pyplot as plt
import open3d as o3d
import pickle
import torchvision
torchvision.disable_beta_transforms_warning()

root_dir = '/mnt/home/rawalk/drive/seg/data/iphone/depth_compose_dump'
output_dir = '/mnt/home/rawalk/drive/seg/data/iphone/point_clouds'

gt_dir = os.path.join(output_dir, 'gt')
pred_dir = os.path.join(output_dir, 'pred')
images_dir = os.path.join(output_dir, 'images')

if not os.path.exists(gt_dir):
    os.makedirs(gt_dir)

if not os.path.exists(pred_dir):
    os.makedirs(pred_dir)

if not os.path.exists(images_dir):
    os.makedirs(images_dir)

###---------------------------------------------
def depth_to_point_cloud(depth_map, mask, image):
    focal_length = 435.886571
    center_x = 177.934254
    center_y = 318.363095

    ##----------------------------------------------------
    depth_foreground = depth_map[mask] ## value in range [0, 1]
    depth_unnormalized = np.full((mask.shape[0], mask.shape[1]), np.inf)
    depth_unnormalized[mask > 0] = depth_foreground ## 0 is near camera, 1 is far from camera

    rows, cols = depth_unnormalized.shape

    # skip_factor = 10  # Adjust this value as needed
    skip_factor = 1

    # Create a grid of pixel coordinates
    cols, rows = np.meshgrid(np.arange(cols), np.arange(rows))

    # Mask to select foreground pixels
    foreground_mask = (mask > 0) & (depth_unnormalized != np.inf)

    # Apply skip factor
    row_indices = np.arange(0, rows.shape[0], skip_factor)
    col_indices = np.arange(0, rows.shape[1], skip_factor)
    rows = rows[row_indices, :][:, col_indices]
    cols = cols[row_indices, :][:, col_indices]
    foreground_mask = foreground_mask[row_indices, :][:, col_indices]
    depth_unnormalized = depth_unnormalized[row_indices, :][:, col_indices]
    image = image[row_indices, :][:, col_indices, :]

    # Convert depth to 3D points
    z = depth_unnormalized[foreground_mask]
    x = (cols[foreground_mask] - center_x) * z / focal_length
    y = (rows[foreground_mask] - center_y) * z / focal_length

    # Stack the coordinates to create the point cloud
    points = np.stack([x, y, z], axis=1)
    colors = image[foreground_mask] / 255.0
    colors = colors[:, [2, 1, 0]]  # Convert BGR to RGB

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.colors = o3d.utility.Vector3dVector(colors)

    return point_cloud

def process_data_sample(data_sample_name):
    data_sample_path = os.path.join(root_dir, data_sample_name)
    with open(data_sample_path, 'rb') as f:
        data_sample = pickle.load(f)

    depth_captured = data_sample['depth_cap']
    depth_predicted = data_sample['depth_sap']
    mask = data_sample['mask']
    image = data_sample['img']
    output_file = os.path.join(images_dir, data_sample_name.replace('.pkl', '.png'))
    cv2.imwrite(output_file, image)

    ## image is not the same size as mask and depth. Downsample
    image = cv2.resize(image, (mask.shape[1], mask.shape[0]))

    ##---------process-----------
    point_cloud = depth_to_point_cloud(depth_captured, mask, image)
    output_file = os.path.join(gt_dir, data_sample_name.replace('.pkl', '.ply'))
    o3d.io.write_point_cloud(output_file, point_cloud)

    point_cloud = depth_to_point_cloud(depth_predicted, mask, image)
    output_file = os.path.join(pred_dir, data_sample_name.replace('.pkl', '.ply'))
    o3d.io.write_point_cloud(output_file, point_cloud)

###---------------------------------------------
data_sample_names = sorted([x for x in os.listdir(root_dir) if x.endswith('.pkl')])

# start_index = 0; end_index = 1000;
# start_index = 1000; end_index = 2000;
# start_index = 2000; end_index = 3000;
# start_index = 3000; end_index = 4000;
# start_index = 4000; end_index = 5000;
# start_index = 5000; end_index = 6000;
# start_index = 6000; end_index = -1
start_index = 0; end_index = -1

if end_index == -1:
    end_index = len(data_sample_names) - 1

data_sample_names = data_sample_names[start_index:end_index + 1]

# Use ThreadPoolExecutor for parallel processing
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
    list(tqdm(executor.map(process_data_sample, data_sample_names), total=len(data_sample_names)))

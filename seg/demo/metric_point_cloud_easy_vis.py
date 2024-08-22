from argparse import ArgumentParser
from mmengine.model import revert_sync_batchnorm
from mmseg.apis import inference_model, init_model, show_result_pyplot
import os
from tqdm import tqdm
import cv2
import numpy as np
import tempfile
from matplotlib import pyplot as plt
import open3d as o3d
import torchvision
torchvision.disable_beta_transforms_warning()

USE_PRED_CAMERA = False
# USE_PRED_CAMERA = True

def sigmoid(x, alpha=10):
    return 1 / (1 + np.exp(-alpha * (x - 0.5)))

def main():
    parser = ArgumentParser()
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('--input', help='Input image dir')
    parser.add_argument('--output_root', '--output-root', default=None, help='Path to output dir')
    parser.add_argument('--seg_dir', '--seg-dir', default=None, help='Path to segmentation dir')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_model(args.config, args.checkpoint, device=args.device)
    if args.device == 'cpu':
        model = revert_sync_batchnorm(model)

    input = args.input
    image_names = []

    # Check if the input is a directory or a text file
    if os.path.isdir(input):
        input_dir = input  # Set input_dir to the directory specified in input
        image_names = [image_name for image_name in sorted(os.listdir(input_dir))
                    if image_name.endswith('.jpg') or image_name.endswith('.png')]
    elif os.path.isfile(input) and input.endswith('.txt'):
        # If the input is a text file, read the paths from it and set input_dir to the directory of the first image
        with open(input, 'r') as file:
            image_paths = [line.strip() for line in file if line.strip()]
        image_names = [os.path.basename(path) for path in image_paths]  # Extract base names for image processing
        input_dir = os.path.dirname(image_paths[0]) if image_paths else ''  # Use the directory of the first image path

    if not os.path.exists(args.output_root):
        os.makedirs(args.output_root)

    seg_dir = args.seg_dir

    for i, image_name in tqdm(enumerate(image_names), total=len(image_names)):
        image_path = os.path.join(input_dir, image_name)
        image = cv2.imread(image_path)

        result = inference_model(model, image_path) ## 0 is nearby and 1 is far from camera.
        result = result.pred_depth_map.data.cpu().numpy()
        depth_map = result[0]

        mask_path = os.path.join(seg_dir, image_name.replace('.png', '.npy').replace('.jpg', '.npy').replace('.jpeg', '.npy'))
        mask = np.load(mask_path)

        ##----------------------------------------
        depth_foreground = depth_map[mask] ## value in range [0, 1]
        min_val, max_val = np.min(depth_foreground), np.max(depth_foreground)
        print('image:{}, min_depth:{}, max_depth:{}'.format(image_name, min_val, max_val))
        depth_unnormalized = np.full((mask.shape[0], mask.shape[1]), np.inf)

        ## in metres. 50 mm lens is common ## suits most images
        # focal_length = 85 * 1e-3
        focal_length = 5000

        depth_unnormalized[mask > 0] = depth_foreground ## 0 is near camera, 1 is far from camera

        ##----------------------------------------------------
        # Create a point cloud from the depth map for the foreground and color it using the pixel colors
        rows, cols = depth_unnormalized.shape
        # Assuming a simple pinhole camera model
        center_x, center_y = cols / 2, rows / 2

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

        # Save the point cloud
        output_file = os.path.join(args.output_root, os.path.basename(image_path).replace('.png', '.ply').replace('.jpg', '.ply').replace('.jpeg', '.ply'))
        o3d.io.write_point_cloud(output_file, point_cloud)


if __name__ == '__main__':
    main()

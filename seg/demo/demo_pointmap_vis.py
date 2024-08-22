from argparse import ArgumentParser
from mmengine.model import revert_sync_batchnorm
from mmseg.apis import inference_model, init_model, show_result_pyplot
import os
from tqdm import tqdm
import cv2
import numpy as np
import tempfile
from matplotlib import pyplot as plt
import torchvision
import open3d as o3d

torchvision.disable_beta_transforms_warning()

def expand_bbox(x, y, w, h, scale, max_width, max_height):
    # Calculate the center of the original bounding box
    center_x, center_y = x + w / 2, y + h / 2

    # Calculate new width and height based on the scale
    new_w = w * scale
    new_h = h * scale

    # Calculate new top-left corner while keeping the center the same
    new_x = max(center_x - new_w / 2, 0)
    new_y = max(center_y - new_h / 2, 0)

    # Adjust the bottom-right corner to not exceed image dimensions
    new_x2 = min(new_x + new_w, max_width)
    new_y2 = min(new_y + new_h, max_height)

    # Return new bounding box as integer coordinates
    return int(new_x), int(new_y), int(new_x2 - new_x), int(new_y2 - new_y)

def crop_image_to_mask(image, mask, scale=1.25):
    # Find contours in the mask
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Get the bounding box of the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)

        # Expand the bounding box
        new_x, new_y, new_w, new_h = expand_bbox(x, y, w, h, scale, image.shape[1], image.shape[0])

        return new_x, new_y, new_x + new_w, new_y + new_h
    else:
        return 0, 0, image.shape[1], image.shape[0]

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
    min_depth = 1e-2; max_depth = 10

    # crop_inference = False;
    crop_inference = True; expand_scale = 1.2

    for i, image_name in tqdm(enumerate(image_names), total=len(image_names)):
        image_path = os.path.join(input_dir, image_name)
        image = cv2.imread(image_path)

        mask_path = os.path.join(seg_dir, image_name.replace('.png', '.npy').replace('.jpg', '.npy').replace('.jpeg', '.npy'))
        mask = np.load(mask_path) ## H x W, boolean

        if crop_inference == True:
            x1, y1, x2, y2 = crop_image_to_mask(image.copy(), mask, scale=expand_scale)
            original_image = image.copy()
            original_mask = mask.copy()
            image = image[y1: y2, x1: x2]
            mask = mask[y1: y2, x1: x2]

        result = inference_model(model, image)
        result = result.pred_depth_map.data.cpu().numpy()
        point_map = result.transpose(1, 2, 0) ### (H, W, C). per pixel 3d coordinate prediction in camera coordinates

        if crop_inference == True:
            original_point_map = np.zeros((original_image.shape[0], original_image.shape[1], 3), dtype=point_map.dtype)
            original_point_map[y1: y2, x1: x2, :] = point_map[:, :, :]

            ## it is possible that original_mask as other contours. so create a new mask. as a boolean
            new_original_mask = np.zeros((original_image.shape[0], original_image.shape[1]), dtype=np.uint8) > 0
            new_original_mask[y1: y2, x1: x2] = mask[:, :] > 0

            point_map = original_point_map
            image = original_image
            mask = new_original_mask

        ## clamp the depth Z to [min_depth, max_depth] for valid pixels
        point_map[:, :, 2] = np.clip(point_map[:, :, 2], min_depth, max_depth)

        ##------------------save pointmap as ply---------------
        save_path = os.path.join(args.output_root, image_name.replace('.png', '.ply').replace('.jpg', '.ply').replace('.jpeg', '.ply'))

        ## flatten the points
        points = point_map[mask > 0].reshape(-1, 3) ## N x 3
        pc = o3d.geometry.PointCloud()
        colors = image[mask > 0] / 255.0
        colors = colors[:, [2, 1, 0]]  # Convert BGR to RGB

        pc.points = o3d.utility.Vector3dVector(points)
        pc.colors = o3d.utility.Vector3dVector(colors)

        ## Create a sphere at the origin to represent the camera
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.05, resolution=20)  # Adjust radius and resolution as needed
        sphere.translate([0, 0, 0])  # Center the sphere at the origin
        sphere.paint_uniform_color([0, 0, 1])  # Color the sphere blue

        ## Convert sphere mesh to point cloud
        sphere_pc = sphere.sample_points_poisson_disk(number_of_points=500)  # Adjust the number of points as needed
        sphere_pc.colors = o3d.utility.Vector3dVector([[0, 0, 1] for _ in range(len(sphere_pc.points))])  # Apply blue color to all points

        ## Combine the original point cloud with the sphere point cloud
        pc = pc + sphere_pc
        o3d.io.write_point_cloud(save_path, pc)

        ##----------------------------------------
        depth_map = point_map[:, :, 2] ## Z axis
        depth_foreground = depth_map[mask > 0] ## not normalized, absolute depth
        processed_depth = np.full((mask.shape[0], mask.shape[1], 3), 100, dtype=np.uint8)

        if len(depth_foreground) > 0:
            min_val, max_val = np.min(depth_foreground), np.max(depth_foreground)
            depth_normalized_foreground = 1 - ((depth_foreground - min_val) / (max_val - min_val)) ## for visualization, foreground is 1 (white), background is 0 (black)
            depth_normalized_foreground = (depth_normalized_foreground * 255.0).astype(np.uint8)

            print('{}, min_depth:{}, max_depth:{}'.format(image_name, min_val, max_val))

            depth_colored_foreground = cv2.applyColorMap(depth_normalized_foreground, cv2.COLORMAP_INFERNO)
            depth_colored_foreground = depth_colored_foreground.reshape(-1, 3)
            processed_depth[mask] = depth_colored_foreground

            # Add text to the image
            text = 'Min depth: {:.2f}, Max depth: {:.2f}'.format(min_val, max_val)
            scale = min(image.shape[0], image.shape[1]) / 600.0  # Dynamically scale the font size

            # Dynamically set the position and thickness based on image dimensions
            text_x = int(image.shape[1] * 0.02)  # Horizontal position is 2% of the image width from the left
            text_y = int(image.shape[0] * 0.05)  # Vertical position is 5% of the image height from the top
            thickness = max(1, int(min(image.shape[0], image.shape[1]) / 300))  # Set thickness based on image size

            cv2.putText(processed_depth, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thickness, cv2.LINE_AA)

        ##---------get surface normal from depth map---------------
        depth_normalized = np.full((mask.shape[0], mask.shape[1]), np.inf)
        depth_normalized[mask > 0] = 1 - ((depth_foreground - min_val) / (max_val - min_val))

        kernel_size = 7 # this can be adjusted
        grad_x = cv2.Sobel(depth_normalized.astype(np.float32), cv2.CV_32F, 1, 0, ksize=kernel_size)
        grad_y = cv2.Sobel(depth_normalized.astype(np.float32), cv2.CV_32F, 0, 1, ksize=kernel_size)
        z = np.full(grad_x.shape, -1)
        normals = np.dstack((-grad_x, -grad_y, z))

        # Normalize the normals
        normals_mag = np.linalg.norm(normals, axis=2, keepdims=True)
        normals_normalized = normals / (normals_mag + 1e-5)  # Add a small epsilon to avoid division by zero

        # Convert normals to a 0-255 scale for visualization
        normal_from_depth = ((normals_normalized + 1) / 2 * 255).astype(np.uint8)

        ## RGB to BGR for cv2
        normal_from_depth = normal_from_depth[:, :, ::-1]

        ##----------------------------------------------------
        output_file = os.path.join(args.output_root, os.path.basename(image_path))

        vis_image = np.concatenate([image, processed_depth, normal_from_depth], axis=1)
        cv2.imwrite(output_file, vis_image)

if __name__ == '__main__':
    main()

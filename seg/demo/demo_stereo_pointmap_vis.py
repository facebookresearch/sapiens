from argparse import ArgumentParser
from mmengine.model import revert_sync_batchnorm
from mmseg.apis import stereo_pointmap_inference_model, init_model, show_result_pyplot
import os
from tqdm import tqdm
import cv2
import numpy as np
import tempfile
from matplotlib import pyplot as plt
import torchvision
import open3d as o3d

torchvision.disable_beta_transforms_warning()

def vis_depth_normal(point_map, mask):
    depth_map = point_map[:, :, 2] ## Z axis
    depth_foreground = depth_map[mask] ## not normalized, absolute depth
    processed_depth = np.full((mask.shape[0], mask.shape[1], 3), 100, dtype=np.uint8)

    if len(depth_foreground) > 0:
        min_val, max_val = np.min(depth_foreground), np.max(depth_foreground)
        depth_normalized_foreground = 1 - ((depth_foreground - min_val) / (max_val - min_val)) ## for visualization, foreground is 1 (white), background is 0 (black)
        depth_normalized_foreground = (depth_normalized_foreground * 255.0).astype(np.uint8)

        depth_colored_foreground = cv2.applyColorMap(depth_normalized_foreground, cv2.COLORMAP_INFERNO)
        depth_colored_foreground = depth_colored_foreground.reshape(-1, 3)
        processed_depth[mask] = depth_colored_foreground

        # Add text to the image
        text = 'Min depth: {:.2f}, Max depth: {:.2f}'.format(min_val, max_val)
        scale = min(mask.shape[0], mask.shape[1]) / 600.0  # Dynamically scale the font size

        # Dynamically set the position and thickness based on image dimensions
        text_x = int(mask.shape[1] * 0.02)  # Horizontal position is 2% of the image width from the left
        text_y = int(mask.shape[0] * 0.05)  # Vertical position is 5% of the image height from the top
        thickness = max(1, int(min(mask.shape[0], mask.shape[1]) / 300))  # Set thickness based on image size

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

    return processed_depth, normal_from_depth

def main():
    parser = ArgumentParser()
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('--input', help='Input image dir')
    parser.add_argument('--output_root', '--output-root', default=None, help='Path to output dir')
    parser.add_argument('--seg_dir', '--seg-dir', default=None, help='Path to segmentation dir')
    parser.add_argument('--time_delta', default=100,)
    parser.add_argument('--device', default='cuda:0', help='Device used for inference')
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
    time_delta = int(args.time_delta)

    extension = image_names[0].split('.')[-1]
    global_image_names = sorted([x for x in os.listdir(input_dir) if x.endswith('.' + extension)])

    for i, image_name in tqdm(enumerate(image_names), total=len(image_names)):
        time1 = global_image_names.index(image_name)
        time2 = (time1 + time_delta) % len(global_image_names)

        image_name1 = image_name
        image_name2 = global_image_names[time2]

        image_path1 = os.path.join(input_dir, image_name1)
        image_path2 = os.path.join(input_dir, image_name2)

        image1 = cv2.imread(image_path1)
        image2 = cv2.imread(image_path2)

        result = stereo_pointmap_inference_model(model, image_path1, image_path2)
        result1 = result.pred_depth_map1.data.cpu().numpy()
        result2 = result.pred_depth_map2.data.cpu().numpy()

        point_map1 = result1.transpose(1, 2, 0) ### (H, W, C). per pixel 3d coordinate prediction in camera coordinates
        point_map2 = result2.transpose(1, 2, 0) ### (H, W, C). per pixel 3d coordinate prediction in camera coordinates

        ## clamp the depth Z to [min_depth, max_depth] for valid pixels
        point_map1[:, :, 2] = np.clip(point_map1[:, :, 2], min_depth, max_depth)
        point_map2[:, :, 2] = np.clip(point_map2[:, :, 2], min_depth, max_depth)

        mask_path1 = os.path.join(seg_dir, image_name1.replace('.png', '.npy').replace('.jpg', '.npy').replace('.jpeg', '.npy'))
        mask_path2 = os.path.join(seg_dir, image_name2.replace('.png', '.npy').replace('.jpg', '.npy').replace('.jpeg', '.npy'))

        mask1 = np.load(mask_path1)
        mask2 = np.load(mask_path2)

        ##------------------save pointmap as ply---------------
        extension = image_name1.split('.')[-1]
        extension = '.' + extension
        save_image_name = '{}_{}'.format(image_name1.replace(extension, ''), image_name2.replace(extension, '')) + '.ply'
        save_path = os.path.join(args.output_root, save_image_name)

        ## flatten the points
        points1 = point_map1[mask1 > 0].reshape(-1, 3) ## N x 3
        points2 = point_map2[mask2 > 0].reshape(-1, 3) ## N x 3

        pc1 = o3d.geometry.PointCloud()
        colors1 = image1[mask1 > 0] / 255.0
        colors1 = colors1[:, [2, 1, 0]]  # Convert BGR to RGB
        pc1.points = o3d.utility.Vector3dVector(points1)
        pc1.colors = o3d.utility.Vector3dVector(colors1)

        pc2 = o3d.geometry.PointCloud()
        colors2 = image2[mask2 > 0] / 255.0
        colors2 = colors2[:, [2, 1, 0]]  # Convert BGR to RGB
        pc2.points = o3d.utility.Vector3dVector(points2)
        pc2.colors = o3d.utility.Vector3dVector(colors2)

        pc = pc1 + pc2

        ## Create a sphere at the origin to represent the camera
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.05, resolution=20)  # Adjust radius and resolution as needed
        sphere.translate([0, 0, 0])  # Center the sphere at the origin
        sphere.paint_uniform_color([0, 0, 1])  # Color the sphere blue

        ## Convert sphere mesh to point cloud
        sphere_pc = sphere.sample_points_poisson_disk(number_of_points=500)  # Adjust the number of points as needed
        sphere_pc.colors = o3d.utility.Vector3dVector([[0, 0, 1] for _ in range(len(sphere_pc.points))])  # Apply blue color to all points

        ## Combine the original point cloud with the sphere point cloud
        # pc = pc + sphere_pc
        # o3d.io.write_point_cloud(save_path, pc)

        pc1 = pc1 + sphere_pc
        pc2 = pc2 + sphere_pc
        o3d.io.write_point_cloud(save_path.replace('.ply', '_1.ply'), pc1)
        o3d.io.write_point_cloud(save_path.replace('.ply', '_2.ply'), pc2)

        ##----------------------------------------
        vis_depth1, vis_normal_from_depth1 = vis_depth_normal(point_map1, mask1)
        vis_depth2, vis_normal_from_depth2 = vis_depth_normal(point_map2, mask2)

        ##----------------------------------------------------
        extension = image_name1.split('.')[-1]
        extension = '.' + extension
        save_image_name = '{}_{}{}'.format(image_name1.replace(extension, ''), image_name2.replace(extension, ''), extension)
        output_file = os.path.join(args.output_root, save_image_name)

        vis_image = np.concatenate([image1, image2, vis_depth1, vis_depth2, vis_normal_from_depth1, vis_normal_from_depth2], axis=1)
        cv2.imwrite(output_file, vis_image)

if __name__ == '__main__':
    main()

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

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
torchvision.disable_beta_transforms_warning()

def main():
    parser = ArgumentParser()
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('--input', help='Input image dir')
    parser.add_argument('--output_root', '--output-root', default=None, help='Path to output dir')
    parser.add_argument('--seg_dir', '--seg-dir', default=None, help='Path to segmentation dir')
    parser.add_argument('--device', default='cuda:0', help='Device used for inference')
    parser.add_argument('--flip', action='store_true', help='Flag to indicate if left right flipping')
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
        os.makedirs(args.output_root, exist_ok=True)

    seg_dir = args.seg_dir
    flip = args.flip

    for i, image_name in tqdm(enumerate(image_names), total=len(image_names)):
        image_path = os.path.join(input_dir, image_name)
        image = cv2.imread(image_path) ## has to be bgr image

        result = inference_model(model, image)
        result = result.pred_depth_map.data.cpu().numpy()
        depth_map = result[0] ## H x W

        if flip == True:
            image_flipped = cv2.flip(image, 1)
            result_flipped = inference_model(model, image_flipped)
            result_flipped = result_flipped.pred_depth_map.data.cpu().numpy()
            depth_map_flipped = result_flipped[0]
            depth_map_flipped = cv2.flip(depth_map_flipped, 1) ## H x W, flip back
            depth_map = (depth_map + depth_map_flipped) / 2 ## H x W, average

        mask_path = os.path.join(seg_dir, image_name.replace('.png', '.npy').replace('.jpg', '.npy').replace('.jpeg', '.npy'))
        mask = np.load(mask_path)

        ##-----------save depth_map to disk---------------------
        save_path = os.path.join(args.output_root, image_name.replace('.png', '.npy').replace('.jpg', '.npy').replace('.jpeg', '.npy'))
        np.save(save_path, depth_map)
        depth_map[~mask] = np.nan

        ##----------------------------------------
        depth_foreground = depth_map[mask] ## value in range [0, 1]
        processed_depth = np.full((mask.shape[0], mask.shape[1], 3), 100, dtype=np.uint8)

        if len(depth_foreground) > 0:
            min_val, max_val = np.min(depth_foreground), np.max(depth_foreground)
            depth_normalized_foreground = 1 - ((depth_foreground - min_val) / (max_val - min_val)) ## for visualization, foreground is 1 (white), background is 0 (black)
            depth_normalized_foreground = (depth_normalized_foreground * 255.0).astype(np.uint8)

            print('{}, min_depth:{}, max_depth:{}'.format(image_name, min_val, max_val))

            depth_colored_foreground = cv2.applyColorMap(depth_normalized_foreground, cv2.COLORMAP_INFERNO)
            depth_colored_foreground = depth_colored_foreground.reshape(-1, 3)
            processed_depth[mask] = depth_colored_foreground

        ##---------get surface normal from depth map---------------
        depth_normalized = np.full((mask.shape[0], mask.shape[1]), np.inf)
        depth_normalized[mask > 0] = 1 - ((depth_foreground - min_val) / (max_val - min_val))

        kernel_size = 7 # ffhq
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

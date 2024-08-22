# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from mmpretrain import FeatureExtractor
import os
import time
from argparse import ArgumentParser

import cv2
import mmcv
import mmengine
import numpy as np
from tqdm import tqdm
import warnings

def main():
    parser = ArgumentParser()
    parser.add_argument('config', help='Config file for pose')
    parser.add_argument('checkpoint', help='Checkpoint file for pose')
    parser.add_argument(
        '--input', type=str, default='', help='Image/Video file')
    parser.add_argument(
        '--output-root',
        type=str,
        default='',
        help='root of the output img file. '
        'Default not saving the visualization images.')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')

    args = parser.parse_args()

    assert args.output_root != ''
    assert args.input != ''

    output_file = None
    if args.output_root:
        mmengine.mkdir_or_exist(args.output_root)
        output_file = os.path.join(args.output_root,
                                   os.path.basename(args.input))

    inferencer = FeatureExtractor(model=args.config, pretrained=args.checkpoint, device=args.device)
    inferencer.model.backbone.out_type = 'featmap' ## removes cls_token and returns spatial feature maps.

    input = args.input
    image_names = []

    # Check if the input is a directory or a text file
    if os.path.isdir(input):
        input_dir = input  # Set input_dir to the directory specified in input
        image_names = [image_name for image_name in sorted(os.listdir(input_dir))
                    if image_name.endswith('.jpg') or image_name.endswith('.jpeg') or image_name.endswith('.png')]
    elif os.path.isfile(input) and input.endswith('.txt'):
        # If the input is a text file, read the paths from it and set input_dir to the directory of the first image
        with open(input, 'r') as file:
            image_paths = [line.strip() for line in file if line.strip()]
        image_names = [os.path.basename(path) for path in image_paths]  # Extract base names for image processing
        input_dir = os.path.dirname(image_paths[0]) if image_paths else ''  # Use the directory of the first image path

    for i, image_name in tqdm(enumerate(image_names), total=len(image_names)):
        image_path = os.path.join(input_dir, image_name)  # Join the directory path with the image file name 
        feature = inferencer(image_path)[0][0] ## embed_dim x H x W. For sapien_1b: 1536 x 64 x 64
        feature = feature.cpu().numpy()

        output_file = os.path.join(args.output_root, os.path.basename(image_path))
        pred_save_path = os.path.join(output_file.replace('.jpg', '.npy').replace('.jpeg', '.npy').replace('.png', '.npy'))
        np.save(pred_save_path, feature)


if __name__ == '__main__':
    main()

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from mmpretrain import MAEInferencer
import os
import time
from argparse import ArgumentParser

import torch
import cv2
import mmcv
import mmengine
import numpy as np
from tqdm import tqdm
import warnings

seed = 42
# seed = 50
# seed = 90
# seed = 128
# seed = 256

np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # for multi-GPU setups
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

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

    # mask_ratio = 0.5
    # mask_ratio = 0.65
    # mask_ratio = 0.7
    mask_ratio = 0.75
    # mask_ratio = 0.8
    # mask_ratio = 0.85
    # mask_ratio = 0.9
    # mask_ratio = 0.95
    # mask_ratio = 0.99+

    copy_inputs = True
    # copy_inputs = False

    args.output_root = '{}_mask{}'.format(args.output_root, 100*mask_ratio)

    output_file = None
    if args.output_root:
        mmengine.mkdir_or_exist(args.output_root)
        output_file = os.path.join(args.output_root,
                                   os.path.basename(args.input))

    inferencer = MAEInferencer(model=args.config, pretrained=args.checkpoint, device=args.device)

    inferencer.model.backbone.mask_ratio = mask_ratio ## update the model mask ratio

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

        vis_image = inferencer(image_path, copy_inputs=copy_inputs)[0]
        output_file = os.path.join(args.output_root, os.path.basename(image_path))
        cv2.imwrite(output_file, vis_image)


if __name__ == '__main__':
    main()

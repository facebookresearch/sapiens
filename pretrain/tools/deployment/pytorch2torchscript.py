# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse

import numpy as np
import torch
import torch._C
import torch.serialization
from mmengine import Config
from mmengine.runner import load_checkpoint
from torch import nn
import os

from mmpretrain import FeatureExtractor
torch.manual_seed(3)

def _demo_mm_inputs(input_shape):
    """Create a superset of inputs needed to run test or train batches."""
    (N, C, H, W) = input_shape

    rng = np.random.RandomState(0)
    imgs = rng.rand(*input_shape)

    img_metas = [{
        'img_shape': (H, W, C),
        'ori_shape': (H, W, C),
        'pad_shape': (H, W, C),
        'filename': '<demo>.png',
        'scale_factor': 1.0,
        'flip': False,
    } for _ in range(N)]

    mm_inputs = {
        'imgs': torch.FloatTensor(imgs).requires_grad_(True),
        'img_metas': img_metas
    }
    return mm_inputs

def pytorch2libtorch(model,
                     input_shape,
                     show=False,
                     output_file='tmp.pt',
                     verify=False):
    
    mm_inputs = _demo_mm_inputs(input_shape)
    imgs = mm_inputs.pop('imgs')
    model.eval()
    traced_model = torch.jit.trace(
        model,
        example_inputs=imgs,
        check_trace=verify,
    )

    if show:
        print(traced_model.graph)

    traced_model.save(output_file)
    print(f'Successfully exported TorchScript model: {output_file}')

def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert MMSeg to TorchScript')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('--checkpoint', help='checkpoint file', default=None)
    parser.add_argument(
        '--show', action='store_true', help='show TorchScript graph')
    parser.add_argument(
        '--verify', action='store_true', help='verify the TorchScript model')
    parser.add_argument('--output-file', type=str, default='tmp.pt')
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        default=[1024, 1024],
        help='input image size (height, width)')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    if len(args.shape) == 1:
        input_shape = (1, 3, args.shape[0], args.shape[0])
    elif len(args.shape) == 2:
        input_shape = (
            1,
            3,
        ) + tuple(args.shape)
    else:
        raise ValueError('invalid input shape')

    model = FeatureExtractor(model=args.config, pretrained=args.checkpoint).model
    model.backbone.out_type = ("featmap")  ## removes cls_token and returns spatial feature maps.

    ## base directory for the args.output_file
    output_dir = os.path.dirname(args.output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # convert the PyTorch model to LibTorch model
    pytorch2libtorch(
        model,
        input_shape,
        show=args.show,
        output_file=args.output_file,
        verify=args.verify)

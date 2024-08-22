# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os

import numpy as np
import torch
import torch._C
import torch.serialization
from mmengine import Config
from mmengine.runner import load_checkpoint
from torch import nn

from mmpose.apis import init_model
from mmpose.registry import MODELS

def digit_version(version_str):
    digit_version = []
    for x in version_str.split('.'):
        if x.isdigit():
            digit_version.append(int(x))
        elif x.find('rc') != -1:
            patch_version = x.split('rc')
            digit_version.append(int(patch_version[0]) - 1)
            digit_version.append(int(patch_version[1]))
    return digit_version

def check_torch_version():
    torch_minimum_version = '1.8.0'
    torch_version = digit_version(torch.__version__)

    assert (torch_version >= digit_version(torch_minimum_version)), \
        f'Torch=={torch.__version__} is not support for converting to ' \
        f'torchscript. Please install pytorch>={torch_minimum_version}.'

def _convert_batchnorm(module):
    module_output = module
    if isinstance(module, torch.nn.SyncBatchNorm):
        module_output = torch.nn.BatchNorm2d(module.num_features, module.eps,
                                             module.momentum, module.affine,
                                             module.track_running_stats)
        if module.affine:
            module_output.weight.data = module.weight.data.clone().detach()
            module_output.bias.data = module.bias.data.clone().detach()
            module_output.weight.requires_grad = module.weight.requires_grad
            module_output.bias.requires_grad = module.bias.requires_grad
        module_output.running_mean = module.running_mean
        module_output.running_var = module.running_var
        module_output.num_batches_tracked = module.num_batches_tracked
    for name, child in module.named_children():
        module_output.add_module(name, _convert_batchnorm(child))
    del module
    return module_output

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
    """Export Pytorch model to TorchScript model and verify the outputs are
    same between Pytorch and TorchScript."""
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
        description='Convert MMPose to TorchScript')
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
        default=[1024, 768],
        help='input image size (height, width)')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    check_torch_version()

    if len(args.shape) == 1:
        input_shape = (1, 3, args.shape[0], args.shape[0])
    elif len(args.shape) == 2:
        input_shape = (1, 3) + tuple(args.shape)
    else:
        raise ValueError('invalid input shape')

    cfg = Config.fromfile(args.config)

    # build the model and load checkpoint
    model = init_model(cfg, args.checkpoint, device='cpu')
    # convert SyncBN to BN
    model = _convert_batchnorm(model)

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

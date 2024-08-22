# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import argparse
import torch
import torch.profiler
from tqdm import tqdm
from mmengine import Config
from mmengine.fileio import dump
from mmengine.model.utils import revert_sync_batchnorm
from mmengine.registry import init_default_scope
from mmengine.runner import Runner, load_checkpoint
from mmseg.registry import MODELS


def parse_args():
    parser = argparse.ArgumentParser(description='MMSeg benchmark a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    parser.add_argument(
        '--tb_log_dir',
        '--tb-log-dir',
        type=str,
        help='input image directory')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    tb_log_dir = args.tb_log_dir
    os.makedirs(tb_log_dir, exist_ok=True)

    cfg = Config.fromfile(args.config)

    init_default_scope(cfg.get('default_scope', 'mmseg'))

    # set cudnn_benchmark
    torch.backends.cudnn.benchmark = False
    cfg.model.pretrained = None

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = MODELS.build(cfg.model)

    if 'checkpoint' in args and os.path.exists(args.checkpoint):
        load_checkpoint(model, args.checkpoint, map_location='cpu')

    cfg.test_dataloader.batch_size = args.batch_size
    data_loader = Runner.build_dataloader(cfg.test_dataloader)

    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()

    with torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(tb_log_dir),
            record_shapes=True,
            profile_memory=True,
            use_cuda=True,
            with_stack=True
    ) as prof:
        for i, data in tqdm(enumerate(data_loader)):
            prof.step()
            data = model.data_preprocessor(data, True)
            inputs = data['inputs']
            data_samples = data['data_samples']

            with torch.no_grad():
                model(inputs, data_samples, mode='predict')
            if i == 6:
                break



if __name__ == '__main__':
    main()

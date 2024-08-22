# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import os.path as osp
import cv2
import numpy as np

from mmengine.config import Config, DictAction
from mmengine.runner import Runner
from mmengine.visualization import Visualizer
from mmpose.registry import VISUALIZERS
from mmengine.structures import InstanceData

def parse_args():
    parser = argparse.ArgumentParser(description='Train a pose model')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume',
        nargs='?',
        type=str,
        const='auto',
        help='If specify checkpint path, resume from it, while if not '
        'specify, try to auto resume from the latest checkpoint '
        'in the work directory.')
    parser.add_argument(
        '--amp',
        action='store_true',
        default=False,
        help='enable automatic-mixed-precision training')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    parser.add_argument(
        '--auto-scale-lr',
        action='store_true',
        help='whether to auto scale the learning rate according to the '
        'actual batch size and the original batch size.')
    parser.add_argument(
        '--show-dir',
        help='directory where the visualization images will be saved.')
    parser.add_argument(
        '--show',
        action='store_true',
        help='whether to display the prediction results in a window.')
    parser.add_argument(
        '--interval',
        type=int,
        default=1,
        help='visualize per interval samples.')
    parser.add_argument(
        '--wait-time',
        type=float,
        default=1,
        help='display time of every window. (second)')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')

    parser.add_argument('--seed', type=int, default=None, help='random seed')

    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def merge_args(cfg, args):
    """Merge CLI arguments to config."""
    if args.no_validate:
        cfg.val_cfg = None
        cfg.val_dataloader = None
        cfg.val_evaluator = None

    cfg.launcher = args.launcher

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    # enable automatic-mixed-precision training
    if args.amp is True:
        from mmengine.optim import AmpOptimWrapper, OptimWrapper
        optim_wrapper = cfg.optim_wrapper.get('type', OptimWrapper)
        assert optim_wrapper in (OptimWrapper, AmpOptimWrapper), \
            '`--amp` is not supported custom optimizer wrapper type ' \
            f'`{optim_wrapper}.'
        cfg.optim_wrapper.type = 'AmpOptimWrapper'
        cfg.optim_wrapper.setdefault('loss_scale', 'dynamic')

    # resume training
    if args.resume == 'auto':
        cfg.resume = True
        cfg.load_from = None
    elif args.resume is not None:
        cfg.resume = True
        cfg.load_from = args.resume

    # enable auto scale learning rate
    if args.auto_scale_lr:
        cfg.auto_scale_lr.enable = True

    # visualization
    if args.show or (args.show_dir is not None):
        assert 'visualization' in cfg.default_hooks, \
            'PoseVisualizationHook is not set in the ' \
            '`default_hooks` field of config. Please set ' \
            '`visualization=dict(type="PoseVisualizationHook")`'

        cfg.default_hooks.visualization.enable = True
        cfg.default_hooks.visualization.show = args.show
        if args.show:
            cfg.default_hooks.visualization.wait_time = args.wait_time
        cfg.default_hooks.visualization.out_dir = args.show_dir
        cfg.default_hooks.visualization.interval = args.interval

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    return cfg


def main():
    args = parse_args()

    # load config
    cfg = Config.fromfile(args.config)

    # merge CLI arguments to config
    cfg = merge_args(cfg, args)

    # set preprocess configs to model
    if 'preprocess_cfg' in cfg:
        cfg.model.setdefault('data_preprocessor',
                             cfg.get('preprocess_cfg', {}))

    # build the runner from config
    runner = Runner.from_cfg(cfg)

    ##--------------------------------------
    num_samples = len(runner.train_dataloader.dataset)

    random_ids = np.arange(num_samples)
    random_ids = np.random.permutation(random_ids)

    dataset = runner.train_dataloader.dataset
    visualizer: Visualizer = Visualizer.get_current_instance()
    visualizer.line_width = 20
    visualizer.radius = 10

    visualizer.set_dataset_meta(runner.train_dataloader.dataset.metainfo)

    for i, idx in enumerate(random_ids):
        print(f'Processing {i} / {len(random_ids)}')
        sample = dataset[idx]
        image = sample['img'] ## 4096 x 2668 x 3, bgr image
        keypoints = sample['keypoints'] ## 1 x 308 x 2
        keypoints_visible = sample['keypoints_visible'] ## 1 x 308

        sample_id = sample['id']
        session_id = sample['session_id']
        camera_id = sample['camera_id']
        frame_id = sample['frame_id']

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) ## convert bgr to rgb image

        instances = InstanceData(metainfo=dict(keypoints=keypoints, keypoints_visible=keypoints_visible, keypoint_scores=keypoints_visible))
        kp_vis_image = visualizer._draw_instances_kpts(image_rgb, instances=instances) ## H, W, C, rgb image
        kp_vis_image = cv2.cvtColor(kp_vis_image, cv2.COLOR_RGB2BGR) ## convert rgb to bgr image

        save_name = f'{session_id}_{camera_id}_{frame_id}'

        save_path = os.path.join(cfg.work_dir, f'{save_name}.jpg')
        cv2.imwrite(save_path, kp_vis_image)


if __name__ == '__main__':
    main()

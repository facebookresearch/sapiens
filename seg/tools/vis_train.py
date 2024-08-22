import argparse
import logging
import os
import os.path as osp
import cv2
import numpy as np
import torch
from mmcv.transforms import to_tensor
from mmengine.config import Config, DictAction
from mmengine.logging import print_log
from mmengine.runner import Runner

from mmseg.registry import RUNNERS
from mmseg.structures import SegDataSample
from mmseg.visualization import SegLocalVisualizer
from mmengine.structures import PixelData
import torchvision

# Disable the beta transforms warning
torchvision.disable_beta_transforms_warning()

def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentor')
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
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    args = parse_args()

    # load config
    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

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
        optim_wrapper = cfg.optim_wrapper.type
        if optim_wrapper == 'AmpOptimWrapper':
            print_log(
                'AMP training is already enabled in your config.',
                logger='current',
                level=logging.WARNING)
        else:
            assert optim_wrapper == 'OptimWrapper', (
                '`--amp` is only supported when the optimizer wrapper type is '
                f'`OptimWrapper` but got {optim_wrapper}.')
            cfg.optim_wrapper.type = 'AmpOptimWrapper'
            cfg.optim_wrapper.loss_scale = 'dynamic'

    # resume training
    if args.resume == 'auto':
        cfg.resume = True
        cfg.load_from = None
    elif args.resume is not None:
        cfg.resume = True
        cfg.load_from = args.resume

    # build the runner from config
    if 'runner_type' not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)

    ##--------------------------------------
    num_samples = len(runner.train_dataloader.dataset)

    random_ids = np.arange(num_samples)
    random_ids = np.random.permutation(random_ids)

    dataset = runner.train_dataloader.dataset

    visualizer: SegLocalVisualizer = SegLocalVisualizer.get_current_instance()
    # visualizer.set_dataset_meta(runner.train_dataloader.dataset.metainfo)

    for i, idx in enumerate(random_ids):
        print(f'Processing {i} / {len(random_ids)}')
        sample = dataset[idx]
        image = sample['img'] ## 4096 x 2668 x 3, bgr image
        sample_id = sample['id']
        session_id = sample['session_id']
        camera_id = sample['camera_id']
        frame_id = sample['frame_id']
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) ## convert bgr to rgb image
        data_sample = SegDataSample()
        data = to_tensor(sample['gt_seg_map'][None, ...].astype(np.int64))
        gt_sem_seg_data = dict(data=data)
        data_sample.gt_sem_seg = PixelData(**gt_sem_seg_data)

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        vis_image = visualizer.custom_add_datasample(name='temp', image=image_rgb, data_sample=data_sample, draw_gt=True, draw_pred=False) ## returns bgr image
        vis_image = np.concatenate([image, vis_image], axis=1)
        vis_image = cv2.resize(vis_image, (2*1334, 2048), interpolation=cv2.INTER_AREA)

        save_name = f'{session_id}_{camera_id}_{frame_id}'
        save_path = os.path.join(cfg.work_dir, f'{save_name}.jpg')
        cv2.imwrite(save_path, vis_image)

if __name__ == '__main__':
    main()

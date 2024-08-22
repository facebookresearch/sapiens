# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os.path as osp
from pathlib import Path

import mmcv
import mmengine
import torch
from mmengine import DictAction

from mmpretrain.datasets import build_dataset
from mmpretrain.structures import DataSample
from mmpretrain.visualization import UniversalVisualizer


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMPreTrain evaluate prediction success/fail')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('result', help='test result json/pkl file')
    parser.add_argument(
        '--out-dir', required=True, help='dir to store output files')
    parser.add_argument(
        '--topk',
        default=20,
        type=int,
        help='Number of images to select for success/fail')
    parser.add_argument(
        '--rescale-factor',
        '-r',
        type=float,
        help='image rescale factor, which is useful if the output is too '
        'large or too small.')
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
    args = parser.parse_args()

    return args


def save_imgs(result_dir, folder_name, results, dataset, rescale_factor=None):
    full_dir = osp.join(result_dir, folder_name)
    vis = UniversalVisualizer()
    vis.dataset_meta = {'classes': dataset.CLASSES}

    # save imgs
    dump_infos = []
    for data_sample in results:
        data_info = dataset.get_data_info(data_sample.sample_idx)
        if 'img' in data_info:
            img = data_info['img']
            name = str(data_sample.sample_idx)
        elif 'img_path' in data_info:
            img = mmcv.imread(data_info['img_path'], channel_order='rgb')
            name = Path(data_info['img_path']).name
        else:
            raise ValueError('Cannot load images from the dataset infos.')
        if rescale_factor is not None:
            img = mmcv.imrescale(img, rescale_factor)
        vis.visualize_cls(
            img, data_sample, out_file=osp.join(full_dir, name + '.png'))

        dump = dict()
        for k, v in data_sample.items():
            if isinstance(v, torch.Tensor):
                dump[k] = v.tolist()
            else:
                dump[k] = v
            dump_infos.append(dump)

    mmengine.dump(dump_infos, osp.join(full_dir, folder_name + '.json'))


def main():
    args = parse_args()

    cfg = mmengine.Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # build the dataloader
    cfg.test_dataloader.dataset.pipeline = []
    dataset = build_dataset(cfg.test_dataloader.dataset)

    results = list()
    for result in mmengine.load(args.result):
        data_sample = DataSample()
        data_sample.set_metainfo({'sample_idx': result['sample_idx']})
        data_sample.set_gt_label(result['gt_label'])
        data_sample.set_pred_label(result['pred_label'])
        data_sample.set_pred_score(result['pred_score'])
        results.append(data_sample)

    # sort result
    results = sorted(results, key=lambda x: torch.max(x.pred_score))

    success = list()
    fail = list()
    for data_sample in results:
        if (data_sample.pred_label == data_sample.gt_label).all():
            success.append(data_sample)
        else:
            fail.append(data_sample)

    success = success[:args.topk]
    fail = fail[:args.topk]

    save_imgs(args.out_dir, 'success', success, dataset, args.rescale_factor)
    save_imgs(args.out_dir, 'fail', fail, dataset, args.rescale_factor)


if __name__ == '__main__':
    main()

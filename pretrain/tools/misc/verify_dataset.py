# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import multiprocessing
import os
from pathlib import Path

from mmengine import (Config, DictAction, track_parallel_progress,
                      track_progress)

from mmpretrain.datasets import build_dataset
from mmpretrain.registry import TRANSFORMS

file_lock = multiprocessing.Lock()


def parse_args():
    parser = argparse.ArgumentParser(description='Verify Dataset')
    parser.add_argument('config', help='config file path')
    parser.add_argument(
        '--out-path',
        type=str,
        default='brokenfiles.log',
        help='output path of all the broken files. If the specified path '
        'already exists, delete the previous file ')
    parser.add_argument(
        '--phase',
        default='train',
        type=str,
        choices=['train', 'test', 'val'],
        help='phase of dataset to visualize, accept "train" "test" and "val".')
    parser.add_argument(
        '--num-process', type=int, default=1, help='number of process to use')
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
    assert args.out_path is not None
    assert args.num_process > 0
    return args


class DatasetValidator():
    """the dataset tool class to check if all file are broken."""

    def __init__(self, dataset_cfg, log_file_path):
        super(DatasetValidator, self).__init__()
        # keep only LoadImageFromFile pipeline
        from mmpretrain.datasets import get_transform_idx

        load_idx = get_transform_idx(dataset_cfg.pipeline, 'LoadImageFromFile')
        assert load_idx >= 0, \
            'This tool is only for datasets needs to load image from files.'
        self.pipeline = TRANSFORMS.build(dataset_cfg.pipeline[load_idx])
        dataset_cfg.pipeline = []
        dataset = build_dataset(dataset_cfg)

        self.dataset = dataset
        self.log_file_path = log_file_path

    def valid_idx(self, idx):
        item = self.dataset[idx]
        try:
            item = self.pipeline(item)
        except Exception:
            with open(self.log_file_path, 'a') as f:
                # add file lock to prevent multi-process writing errors
                filepath = str(Path(item['img_path']))
                file_lock.acquire()
                f.write(filepath + '\n')
                file_lock.release()
                print(f'{filepath} cannot be read correctly, please check it.')

    def __len__(self):
        return len(self.dataset)


def print_info(log_file_path):
    """print some information and do extra action."""
    print()
    with open(log_file_path, 'r') as f:
        content = f.read().strip()
        if content == '':
            print('There is no broken file found.')
            os.remove(log_file_path)
        else:
            num_file = len(content.split('\n'))
            print(f'{num_file} broken files found, name list save in file:'
                  f'{log_file_path}')
    print()


def main():
    # parse cfg and args
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # touch output file to save broken files list.
    output_path = Path(args.out_path)
    if not output_path.parent.exists():
        raise Exception("Path '--out-path' parent directory not found.")
    if output_path.exists():
        os.remove(output_path)
    output_path.touch()

    if args.phase == 'train':
        dataset_cfg = cfg.train_dataloader.dataset
    elif args.phase == 'val':
        dataset_cfg = cfg.val_dataloader.dataset
    elif args.phase == 'test':
        dataset_cfg = cfg.test_dataloader.dataset
    else:
        raise ValueError("'--phase' only support 'train', 'val' and 'test'.")

    # do validate
    validator = DatasetValidator(dataset_cfg, output_path)

    if args.num_process > 1:
        # The default chunksize calcuation method of Pool.map
        chunksize, extra = divmod(len(validator), args.num_process * 8)
        if extra:
            chunksize += 1

        track_parallel_progress(
            validator.valid_idx,
            list(range(len(validator))),
            args.num_process,
            chunksize=chunksize,
            keep_order=False)
    else:
        track_progress(validator.valid_idx, list(range(len(validator))))

    print_info(output_path)


if __name__ == '__main__':
    main()

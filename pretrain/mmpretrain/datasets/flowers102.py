# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import List

import mat4py
from mmengine import get_file_backend

from mmpretrain.registry import DATASETS
from .base_dataset import BaseDataset


@DATASETS.register_module()
class Flowers102(BaseDataset):
    """The Oxford 102 Flower Dataset.

    Support the `Oxford 102 Flowers Dataset <https://www.robots.ox.ac.uk/~vgg/data/flowers/102/>`_ Dataset.
    After downloading and decompression, the dataset directory structure is as follows.

    Flowers102 dataset directory: ::

        Flowers102
        ├── jpg
        │   ├── image_00001.jpg
        │   ├── image_00002.jpg
        │   └── ...
        ├── imagelabels.mat
        ├── setid.mat
        └── ...

    Args:
        data_root (str): The root directory for Oxford 102 Flowers dataset.
        split (str, optional): The dataset split, supports "train",
            "val", "trainval", and "test". Default to "trainval".

    Examples:
        >>> from mmpretrain.datasets import Flowers102
        >>> train_dataset = Flowers102(data_root='data/Flowers102', split='trainval')
        >>> train_dataset
        Dataset Flowers102
            Number of samples:  2040
            Root of dataset:    data/Flowers102
        >>> test_dataset = Flowers102(data_root='data/Flowers102', split='test')
        >>> test_dataset
        Dataset Flowers102
            Number of samples:  6149
            Root of dataset:    data/Flowers102
    """  # noqa: E501

    def __init__(self, data_root: str, split: str = 'trainval', **kwargs):
        splits = ['train', 'val', 'trainval', 'test']
        assert split in splits, \
            f"The split must be one of {splits}, but get '{split}'"
        self.split = split

        ann_file = 'imagelabels.mat'
        data_prefix = 'jpg'
        train_test_split_file = 'setid.mat'
        test_mode = split == 'test'

        self.backend = get_file_backend(data_root, enable_singleton=True)

        self.train_test_split_file = self.backend.join_path(
            data_root, train_test_split_file)

        super(Flowers102, self).__init__(
            ann_file=ann_file,
            data_root=data_root,
            data_prefix=data_prefix,
            test_mode=test_mode,
            **kwargs)

    def load_data_list(self):
        """Load images and ground truth labels."""

        label_dict = mat4py.loadmat(self.ann_file)['labels']
        split_list = mat4py.loadmat(self.train_test_split_file)

        if self.split == 'train':
            split_list = split_list['trnid']
        elif self.split == 'val':
            split_list = split_list['valid']
        elif self.split == 'test':
            split_list = split_list['tstid']
        else:
            train_ids = split_list['trnid']
            val_ids = split_list['valid']
            train_ids.extend(val_ids)
            split_list = train_ids

        data_list = []
        for sample_id in split_list:
            img_name = 'image_%05d.jpg' % (sample_id)
            img_path = self.backend.join_path(self.img_prefix, img_name)
            gt_label = int(label_dict[sample_id - 1]) - 1
            info = dict(img_path=img_path, gt_label=gt_label)
            data_list.append(info)

        return data_list

    def extra_repr(self) -> List[str]:
        """The extra repr information of the dataset."""
        body = [
            f'Root of dataset: \t{self.data_root}',
        ]
        return body

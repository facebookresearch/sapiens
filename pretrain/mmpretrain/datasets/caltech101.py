# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import List

from mmengine import get_file_backend, list_from_file

from mmpretrain.registry import DATASETS
from .base_dataset import BaseDataset
from .categories import CALTECH101_CATEGORIES


@DATASETS.register_module()
class Caltech101(BaseDataset):
    """The Caltech101 Dataset.

    Support the `Caltech101 <https://data.caltech.edu/records/mzrjq-6wc02>`_ Dataset.
    After downloading and decompression, the dataset directory structure is as follows.

    Caltech101 dataset directory: ::

        caltech-101
        ├── 101_ObjectCategories
        │   ├── class_x
        │   │   ├── xx1.jpg
        │   │   ├── xx2.jpg
        │   │   └── ...
        │   ├── class_y
        │   │   ├── yy1.jpg
        │   │   ├── yy2.jpg
        │   │   └── ...
        │   └── ...
        ├── Annotations
        │   ├── class_x
        │   │   ├── xx1.mat
        │   │   └── ...
        │   └── ...
        ├── meta
        │   ├── train.txt
        │   └── test.txt
        └── ....

    Please note that since there is no official splitting for training and
    test set, you can use the train.txt and text.txt provided by us or
    create your own annotation files. Here is the download
    `link <https://download.openmmlab.com/mmpretrain/datasets/caltech_meta.zip>`_
    for the annotations.

    Args:
        data_root (str): The root directory for the Caltech101 dataset.
        split (str, optional): The dataset split, supports "train" and "test".
            Default to "train".

    Examples:
        >>> from mmpretrain.datasets import Caltech101
        >>> train_dataset = Caltech101(data_root='data/caltech-101', split='train')
        >>> train_dataset
        Dataset Caltech101
            Number of samples:  3060
            Number of categories:       102
            Root of dataset:    data/caltech-101
        >>> test_dataset = Caltech101(data_root='data/caltech-101', split='test')
        >>> test_dataset
        Dataset Caltech101
            Number of samples:  6728
            Number of categories:       102
            Root of dataset:    data/caltech-101
    """  # noqa: E501

    METAINFO = {'classes': CALTECH101_CATEGORIES}

    def __init__(self, data_root: str, split: str = 'train', **kwargs):

        splits = ['train', 'test']
        assert split in splits, \
            f"The split must be one of {splits}, but get '{split}'"
        self.split = split

        self.backend = get_file_backend(data_root, enable_singleton=True)

        if split == 'train':
            ann_file = self.backend.join_path('meta', 'train.txt')
        else:
            ann_file = self.backend.join_path('meta', 'test.txt')

        data_prefix = '101_ObjectCategories'
        test_mode = split == 'test'

        super(Caltech101, self).__init__(
            ann_file=ann_file,
            data_root=data_root,
            data_prefix=data_prefix,
            test_mode=test_mode,
            **kwargs)

    def load_data_list(self):
        """Load images and ground truth labels."""

        pairs = list_from_file(self.ann_file)
        data_list = []

        for pair in pairs:
            path, gt_label = pair.split()
            img_path = self.backend.join_path(self.img_prefix, path)
            info = dict(img_path=img_path, gt_label=int(gt_label))
            data_list.append(info)

        return data_list

    def extra_repr(self) -> List[str]:
        """The extra repr information of the dataset."""
        body = [
            f'Root of dataset: \t{self.data_root}',
        ]
        return body

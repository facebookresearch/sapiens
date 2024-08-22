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
from .categories import DTD_CATEGORIES


@DATASETS.register_module()
class DTD(BaseDataset):
    """The Describable Texture Dataset (DTD).

    Support the `Describable Texture Dataset <https://www.robots.ox.ac.uk/~vgg/data/dtd/>`_ Dataset.
    After downloading and decompression, the dataset directory structure is as follows.

    DTD dataset directory: ::

        dtd
        ├── images
        │   ├── banded
        |   |   ├──banded_0002.jpg
        |   |   ├──banded_0004.jpg
        |   |   └── ...
        │   └── ...
        ├── imdb
        │   └── imdb.mat
        ├── labels
        |   |   ├──labels_joint_anno.txt
        |   |   ├──test1.txt
        |   |   ├──test2.txt
        |   |   └── ...
        │   └── ...
        └── ....

    Args:
        data_root (str): The root directory for Describable Texture dataset.
        split (str, optional): The dataset split, supports "train",
            "val", "trainval", and "test". Default to "trainval".

    Examples:
        >>> from mmpretrain.datasets import DTD
        >>> train_dataset = DTD(data_root='data/dtd', split='trainval')
        >>> train_dataset
        Dataset DTD
            Number of samples:  3760
            Number of categories:       47
            Root of dataset:    data/dtd
        >>> test_dataset = DTD(data_root='data/dtd', split='test')
        >>> test_dataset
        Dataset DTD
            Number of samples:  1880
            Number of categories:       47
            Root of dataset:    data/dtd
    """  # noqa: E501

    METAINFO = {'classes': DTD_CATEGORIES}

    def __init__(self, data_root: str, split: str = 'trainval', **kwargs):

        splits = ['train', 'val', 'trainval', 'test']
        assert split in splits, \
            f"The split must be one of {splits}, but get '{split}'"
        self.split = split

        data_prefix = 'images'
        test_mode = split == 'test'

        self.backend = get_file_backend(data_root, enable_singleton=True)
        ann_file = self.backend.join_path('imdb', 'imdb.mat')

        super(DTD, self).__init__(
            ann_file=ann_file,
            data_root=data_root,
            data_prefix=data_prefix,
            test_mode=test_mode,
            **kwargs)

    def load_data_list(self):
        """Load images and ground truth labels."""

        data = mat4py.loadmat(self.ann_file)['images']
        names = data['name']
        labels = data['class']
        parts = data['set']
        num = len(names)
        assert num == len(labels) == len(parts), 'get error ann file'

        if self.split == 'train':
            target_set = {1}
        elif self.split == 'val':
            target_set = {2}
        elif self.split == 'test':
            target_set = {3}
        else:
            target_set = {1, 2}

        data_list = []
        for i in range(num):
            if parts[i] in target_set:
                img_name = names[i]
                img_path = self.backend.join_path(self.img_prefix, img_name)
                gt_label = labels[i] - 1
                info = dict(img_path=img_path, gt_label=gt_label)
                data_list.append(info)

        return data_list

    def extra_repr(self) -> List[str]:
        """The extra repr information of the dataset."""
        body = [
            f'Root of dataset: \t{self.data_root}',
        ]
        return body

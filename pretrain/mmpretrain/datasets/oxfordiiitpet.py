# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import List

from mmengine import get_file_backend, list_from_file

from mmpretrain.registry import DATASETS
from .base_dataset import BaseDataset
from .categories import OxfordIIITPet_CATEGORIES


@DATASETS.register_module()
class OxfordIIITPet(BaseDataset):
    """The Oxford-IIIT Pets Dataset.

    Support the `Oxford-IIIT Pets Dataset <https://www.robots.ox.ac.uk/~vgg/data/pets/>`_ Dataset.
    After downloading and decompression, the dataset directory structure is as follows.

    Oxford-IIIT_Pets dataset directory: ::

        Oxford-IIIT_Pets
        ├── images
        │   ├── Abyssinian_1.jpg
        │   ├── Abyssinian_2.jpg
        │   └── ...
        ├── annotations
        │   ├── trainval.txt
        │   ├── test.txt
        │   ├── list.txt
        │   └── ...
        └── ....

    Args:
        data_root (str): The root directory for Oxford-IIIT Pets dataset.
        split (str, optional): The dataset split, supports "trainval" and "test".
            Default to "trainval".

    Examples:
        >>> from mmpretrain.datasets import OxfordIIITPet
        >>> train_dataset = OxfordIIITPet(data_root='data/Oxford-IIIT_Pets', split='trainval')
        >>> train_dataset
        Dataset OxfordIIITPet
            Number of samples:  3680
            Number of categories:       37
            Root of dataset:    data/Oxford-IIIT_Pets
        >>> test_dataset = OxfordIIITPet(data_root='data/Oxford-IIIT_Pets', split='test')
        >>> test_dataset
        Dataset OxfordIIITPet
            Number of samples:  3669
            Number of categories:       37
            Root of dataset:    data/Oxford-IIIT_Pets
    """  # noqa: E501

    METAINFO = {'classes': OxfordIIITPet_CATEGORIES}

    def __init__(self, data_root: str, split: str = 'trainval', **kwargs):

        splits = ['trainval', 'test']
        assert split in splits, \
            f"The split must be one of {splits}, but get '{split}'"
        self.split = split

        self.backend = get_file_backend(data_root, enable_singleton=True)
        if split == 'trainval':
            ann_file = self.backend.join_path('annotations', 'trainval.txt')
        else:
            ann_file = self.backend.join_path('annotations', 'test.txt')

        data_prefix = 'images'
        test_mode = split == 'test'

        super(OxfordIIITPet, self).__init__(
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
            img_name, class_id, _, _ = pair.split()
            img_name = f'{img_name}.jpg'
            img_path = self.backend.join_path(self.img_prefix, img_name)
            gt_label = int(class_id) - 1
            info = dict(img_path=img_path, gt_label=gt_label)
            data_list.append(info)
        return data_list

    def extra_repr(self) -> List[str]:
        """The extra repr information of the dataset."""
        body = [
            f'Root of dataset: \t{self.data_root}',
        ]
        return body

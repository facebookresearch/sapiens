# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import List

from mmengine import get_file_backend, list_from_file

from mmpretrain.registry import DATASETS
from .base_dataset import BaseDataset
from .categories import FGVCAIRCRAFT_CATEGORIES


@DATASETS.register_module()
class FGVCAircraft(BaseDataset):
    """The FGVC_Aircraft Dataset.

    Support the `FGVC_Aircraft Dataset <https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/>`_ Dataset.
    After downloading and decompression, the dataset directory structure is as follows.

    FGVC_Aircraft dataset directory: ::

        fgvc-aircraft-2013b
        └── data
            ├── images
            │   ├── 1.jpg
            │   ├── 2.jpg
            │   └── ...
            ├── images_variant_train.txt
            ├── images_variant_test.txt
            ├── images_variant_trainval.txt
            ├── images_variant_val.txt
            ├── variants.txt
            └── ....

    Args:
        data_root (str): The root directory for FGVC_Aircraft dataset.
        split (str, optional): The dataset split, supports "train",
            "val", "trainval", and "test". Default to "trainval".

    Examples:
        >>> from mmpretrain.datasets import FGVCAircraft
        >>> train_dataset = FGVCAircraft(data_root='data/fgvc-aircraft-2013b', split='trainval')
        >>> train_dataset
        Dataset FGVCAircraft
            Number of samples:  6667
            Number of categories:       100
            Root of dataset:    data/fgvc-aircraft-2013b
        >>> test_dataset = FGVCAircraft(data_root='data/fgvc-aircraft-2013b', split='test')
        >>> test_dataset
        Dataset FGVCAircraft
            Number of samples:  3333
            Number of categories:       100
            Root of dataset:    data/fgvc-aircraft-2013b
    """  # noqa: E501

    METAINFO = {'classes': FGVCAIRCRAFT_CATEGORIES}

    def __init__(self, data_root: str, split: str = 'trainval', **kwargs):

        splits = ['train', 'val', 'trainval', 'test']
        assert split in splits, \
            f"The split must be one of {splits}, but get '{split}'"
        self.split = split

        self.backend = get_file_backend(data_root, enable_singleton=True)
        ann_file = self.backend.join_path('data',
                                          f'images_variant_{split}.txt')
        data_prefix = self.backend.join_path('data', 'images')
        test_mode = split == 'test'

        super(FGVCAircraft, self).__init__(
            ann_file=ann_file,
            data_root=data_root,
            test_mode=test_mode,
            data_prefix=data_prefix,
            **kwargs)

    def load_data_list(self):
        """Load images and ground truth labels."""

        pairs = list_from_file(self.ann_file)
        data_list = []
        for pair in pairs:
            pair = pair.split()
            img_name = pair[0]
            class_name = ' '.join(pair[1:])
            img_name = f'{img_name}.jpg'
            img_path = self.backend.join_path(self.img_prefix, img_name)
            gt_label = self.METAINFO['classes'].index(class_name)
            info = dict(img_path=img_path, gt_label=gt_label)
            data_list.append(info)

        return data_list

    def extra_repr(self) -> List[str]:
        """The extra repr information of the dataset."""
        body = [
            f'Root of dataset: \t{self.data_root}',
        ]
        return body

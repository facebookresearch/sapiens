# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import List

from mmengine import get_file_backend, list_from_file

from mmpretrain.registry import DATASETS
from .base_dataset import BaseDataset
from .categories import SUN397_CATEGORIES


@DATASETS.register_module()
class SUN397(BaseDataset):
    """The SUN397 Dataset.

    Support the `SUN397 Dataset <https://vision.princeton.edu/projects/2010/SUN/>`_ Dataset.
    After downloading and decompression, the dataset directory structure is as follows.

    SUN397 dataset directory: ::

        SUN397
        ├── SUN397
        │   ├── a
        │   │   ├── abbey
        │   |   |   ├── sun_aaalbzqrimafwbiv.jpg
        │   |   |   └── ...
        │   │   ├── airplane_cabin
        │   |   |   ├── sun_aadqdkqaslqqoblu.jpg
        │   |   |   └── ...
        │   |   └── ...
        │   ├── b
        │   │   └── ...
        │   ├── c
        │   │   └── ...
        │   └── ...
        └── Partitions
            ├── ClassName.txt
            ├── Training_01.txt
            ├── Testing_01.txt
            └── ...

    Args:
        data_root (str): The root directory for Stanford Cars dataset.
        split (str, optional): The dataset split, supports "train" and "test".
            Default to "train".

    Examples:
        >>> from mmpretrain.datasets import SUN397
        >>> train_dataset = SUN397(data_root='data/SUN397', split='train')
        >>> train_dataset
        Dataset SUN397
            Number of samples:  19850
            Number of categories:       397
            Root of dataset:    data/SUN397
        >>> test_dataset = SUN397(data_root='data/SUN397', split='test')
        >>> test_dataset
        Dataset SUN397
            Number of samples:  19850
            Number of categories:       397
            Root of dataset:    data/SUN397

    **Note that some images are not a jpg file although the name ends with ".jpg".
    The backend of SUN397 should be "pillow" as below to read these images properly,**

    .. code-block:: python

        pipeline = [
            dict(type='LoadImageFromFile', imdecode_backend='pillow'),
            dict(type='RandomResizedCrop', scale=224),
            dict(type='PackInputs')
            ]
    """  # noqa: E501

    METAINFO = {'classes': SUN397_CATEGORIES}

    def __init__(self, data_root: str, split: str = 'train', **kwargs):

        splits = ['train', 'test']
        assert split in splits, \
            f"The split must be one of {splits}, but get '{split}'"
        self.split = split

        self.backend = get_file_backend(data_root, enable_singleton=True)
        if split == 'train':
            ann_file = self.backend.join_path('Partitions', 'Training_01.txt')
        else:
            ann_file = self.backend.join_path('Partitions', 'Testing_01.txt')

        data_prefix = 'SUN397'
        test_mode = split == 'test'

        super(SUN397, self).__init__(
            ann_file=ann_file,
            data_root=data_root,
            test_mode=test_mode,
            data_prefix=data_prefix,
            **kwargs)

    def load_data_list(self):
        pairs = list_from_file(self.ann_file)
        data_list = []
        for pair in pairs:
            img_path = self.backend.join_path(self.img_prefix, pair[1:])
            items = pair.split('/')
            class_name = '_'.join(items[2:-1])
            gt_label = self.METAINFO['classes'].index(class_name)
            info = dict(img_path=img_path, gt_label=gt_label)
            data_list.append(info)

        return data_list

    def __getitem__(self, idx: int) -> dict:
        try:
            return super().__getitem__(idx)
        except AttributeError:
            raise RuntimeError(
                'Some images in the SUN397 dataset are not a jpg file '
                'although the name ends with ".jpg". The backend of SUN397 '
                'should be "pillow" to read these images properly.')

    def extra_repr(self) -> List[str]:
        """The extra repr information of the dataset."""
        body = [
            f'Root of dataset: \t{self.data_root}',
        ]
        return body

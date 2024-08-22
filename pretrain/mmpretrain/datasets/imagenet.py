# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, Union

from mmengine import fileio
from mmengine.logging import MMLogger

from mmpretrain.registry import DATASETS
from .categories import IMAGENET_CATEGORIES
from .custom import CustomDataset
import os

@DATASETS.register_module()
class ImageNet(CustomDataset):
    """`ImageNet <http://www.image-net.org>`_ Dataset.

    The dataset supports two kinds of directory format,

    ::

        imagenet
        ├── train
        │   ├──class_x
        |   |   ├── x1.jpg
        |   |   ├── x2.jpg
        |   |   └── ...
        │   ├── class_y
        |   |   ├── y1.jpg
        |   |   ├── y2.jpg
        |   |   └── ...
        |   └── ...
        ├── val
        │   ├──class_x
        |   |   └── ...
        │   ├── class_y
        |   |   └── ...
        |   └── ...
        └── test
            ├── test1.jpg
            ├── test2.jpg
            └── ...

    or ::

        imagenet
        ├── train
        │   ├── x1.jpg
        │   ├── y1.jpg
        │   └── ...
        ├── val
        │   ├── x3.jpg
        │   ├── y3.jpg
        │   └── ...
        ├── test
        │   ├── test1.jpg
        │   ├── test2.jpg
        │   └── ...
        └── meta
            ├── train.txt
            └── val.txt


    Args:
        data_root (str): The root directory for ``data_prefix`` and
            ``ann_file``. Defaults to ''.
        split (str): The dataset split, supports "train", "val" and "test".
            Default to ''.
        data_prefix (str | dict): Prefix for training data. Defaults to ''.
        ann_file (str): Annotation file path. Defaults to ''.
        metainfo (dict, optional): Meta information for dataset, such as class
            information. Defaults to None.
        **kwargs: Other keyword arguments in :class:`CustomDataset` and
            :class:`BaseDataset`.


    Examples:
        >>> from mmpretrain.datasets import ImageNet
        >>> train_dataset = ImageNet(data_root='data/imagenet', split='train')
        >>> train_dataset
        Dataset ImageNet
            Number of samples:  1281167
            Number of categories:       1000
            Root of dataset:    data/imagenet
        >>> test_dataset = ImageNet(data_root='data/imagenet', split='val')
        >>> test_dataset
        Dataset ImageNet
            Number of samples:  50000
            Number of categories:       1000
            Root of dataset:    data/imagenet
    """  # noqa: E501

    IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif')
    METAINFO = {'classes': IMAGENET_CATEGORIES}

    def __init__(self,
                 data_root: str = '',
                 split: str = '',
                 data_prefix: Union[str, dict] = '',
                 ann_file: str = '',
                 metainfo: Optional[dict] = None,
                 **kwargs):
        kwargs = {'extensions': self.IMG_EXTENSIONS, **kwargs}

        if split:
            splits = ['train', 'val', 'test']
            assert split in splits, \
                f"The split must be one of {splits}, but get '{split}'"

            if split == 'test':
                logger = MMLogger.get_current_instance()
                logger.info(
                    'Since the ImageNet1k test set does not provide label'
                    'annotations, `with_label` is set to False')
                kwargs['with_label'] = False

            data_prefix = split if data_prefix == '' else data_prefix

            if ann_file == '':
                _ann_path = fileio.join_path(data_root, 'meta', f'{split}.txt')
                if fileio.exists(_ann_path):
                    ann_file = fileio.join_path('meta', f'{split}.txt')

        super().__init__(
            data_root=data_root,
            data_prefix=data_prefix,
            ann_file=ann_file,
            metainfo=metainfo,
            **kwargs)
        
        self.global_rank = int(os.environ.get("RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.batch_size = int(os.environ.get("TRAIN_BATCH_SIZE_PER_GPU", 512))
        
        print('\033[92m' + 'Rank:{}, World Size:{}, Batch Size:{}, Loaded {} samples'.format(\
            self.global_rank, self.world_size, self.batch_size, self.__len__()) + '\033[0m')

        return

    def extra_repr(self) -> List[str]:
        """The extra repr information of the dataset."""
        body = [
            f'Root of dataset: \t{self.data_root}',
        ]
        return body


@DATASETS.register_module()
class ImageNet21k(CustomDataset):
    """ImageNet21k Dataset.

    Since the dataset ImageNet21k is extremely big, contains 21k+ classes
    and 1.4B files. We won't provide the default categories list. Please
    specify it from the ``classes`` argument.
    The dataset directory structure is as follows,

    ImageNet21k dataset directory ::

        imagenet21k
        ├── train
        │   ├──class_x
        |   |   ├── x1.jpg
        |   |   ├── x2.jpg
        |   |   └── ...
        │   ├── class_y
        |   |   ├── y1.jpg
        |   |   ├── y2.jpg
        |   |   └── ...
        |   └── ...
        └── meta
            └── train.txt


    Args:
        data_root (str): The root directory for ``data_prefix`` and
            ``ann_file``. Defaults to ''.
        data_prefix (str | dict): Prefix for training data. Defaults to ''.
        ann_file (str): Annotation file path. Defaults to ''.
        metainfo (dict, optional): Meta information for dataset, such as class
            information. Defaults to None.
        multi_label (bool): Not implement by now. Use multi label or not.
            Defaults to False.
        **kwargs: Other keyword arguments in :class:`CustomDataset` and
            :class:`BaseDataset`.

    Examples:
        >>> from mmpretrain.datasets import ImageNet21k
        >>> train_dataset = ImageNet21k(data_root='data/imagenet21k', split='train')
        >>> train_dataset
        Dataset ImageNet21k
            Number of samples:  14197088
            Annotation file:    data/imagenet21k/meta/train.txt
            Prefix of images:   data/imagenet21k/train
    """  # noqa: E501

    IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif')

    def __init__(self,
                 data_root: str = '',
                 split: str = '',
                 data_prefix: Union[str, dict] = '',
                 ann_file: str = '',
                 metainfo: Optional[dict] = None,
                 multi_label: bool = False,
                 **kwargs):
        if multi_label:
            raise NotImplementedError(
                'The `multi_label` option is not supported by now.')
        self.multi_label = multi_label

        if split:
            splits = ['train']
            assert split in splits, \
                f"The split must be one of {splits}, but get '{split}'.\
                If you want to specify your own validation set or test set,\
                please set split to None."

            self.split = split
            data_prefix = split if data_prefix == '' else data_prefix

            if not ann_file:
                _ann_path = fileio.join_path(data_root, 'meta', f'{split}.txt')
                if fileio.exists(_ann_path):
                    ann_file = fileio.join_path('meta', f'{split}.txt')

        logger = MMLogger.get_current_instance()

        if not ann_file:
            logger.warning(
                'The ImageNet21k dataset is large, and scanning directory may '
                'consume long time. Considering to specify the `ann_file` to '
                'accelerate the initialization.')

        kwargs = {'extensions': self.IMG_EXTENSIONS, **kwargs}
        super().__init__(
            data_root=data_root,
            data_prefix=data_prefix,
            ann_file=ann_file,
            metainfo=metainfo,
            **kwargs)

        if self.CLASSES is None:
            logger.warning(
                'The CLASSES is not stored in the `ImageNet21k` class. '
                'Considering to specify the `classes` argument if you need '
                'do inference on the ImageNet-21k dataset')

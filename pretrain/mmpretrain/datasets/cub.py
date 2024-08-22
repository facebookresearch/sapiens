# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import List

from mmengine import get_file_backend, list_from_file
from mmengine.logging import MMLogger

from mmpretrain.registry import DATASETS
from .base_dataset import BaseDataset
from .categories import CUB_CATEGORIES


@DATASETS.register_module()
class CUB(BaseDataset):
    """The CUB-200-2011 Dataset.

    Support the `CUB-200-2011 <http://www.vision.caltech.edu/visipedia/CUB-200-2011.html>`_ Dataset.
    Comparing with the `CUB-200 <http://www.vision.caltech.edu/visipedia/CUB-200.html>`_ Dataset,
    there are much more pictures in `CUB-200-2011`. After downloading and decompression, the dataset
    directory structure is as follows.

    CUB dataset directory: ::

        CUB_200_2011
        ├── images
        │   ├── class_x
        │   │   ├── xx1.jpg
        │   │   ├── xx2.jpg
        │   │   └── ...
        │   ├── class_y
        │   │   ├── yy1.jpg
        │   │   ├── yy2.jpg
        │   │   └── ...
        │   └── ...
        ├── images.txt
        ├── image_class_labels.txt
        ├── train_test_split.txt
        └── ....

    Args:
        data_root (str): The root directory for CUB-200-2011 dataset.
        split (str, optional): The dataset split, supports "train" and "test".
            Default to "train".

    Examples:
        >>> from mmpretrain.datasets import CUB
        >>> train_dataset = CUB(data_root='data/CUB_200_2011', split='train')
        >>> train_dataset
        Dataset CUB
            Number of samples:  5994
            Number of categories:       200
            Root of dataset:    data/CUB_200_2011
        >>> test_dataset = CUB(data_root='data/CUB_200_2011', split='test')
        >>> test_dataset
        Dataset CUB
            Number of samples:  5794
            Number of categories:       200
            Root of dataset:    data/CUB_200_2011
    """  # noqa: E501

    METAINFO = {'classes': CUB_CATEGORIES}

    def __init__(self,
                 data_root: str,
                 split: str = 'train',
                 test_mode: bool = False,
                 **kwargs):

        splits = ['train', 'test']
        assert split in splits, \
            f"The split must be one of {splits}, but get '{split}'"
        self.split = split

        # To handle the BC-breaking
        if split == 'train' and test_mode:
            logger = MMLogger.get_current_instance()
            logger.warning('split="train" but test_mode=True. '
                           'The training set will be used.')

        ann_file = 'images.txt'
        data_prefix = 'images'
        image_class_labels_file = 'image_class_labels.txt'
        train_test_split_file = 'train_test_split.txt'

        self.backend = get_file_backend(data_root, enable_singleton=True)
        self.image_class_labels_file = self.backend.join_path(
            data_root, image_class_labels_file)
        self.train_test_split_file = self.backend.join_path(
            data_root, train_test_split_file)
        super(CUB, self).__init__(
            ann_file=ann_file,
            data_root=data_root,
            data_prefix=data_prefix,
            test_mode=test_mode,
            **kwargs)

    def _load_data_from_txt(self, filepath):
        """load data from CUB txt file, the every line of the file is idx and a
        data item."""
        pairs = list_from_file(filepath)
        data_dict = dict()
        for pair in pairs:
            idx, data_item = pair.split()
            # all the index starts from 1 in CUB files,
            # here we need to '- 1' to let them start from 0.
            data_dict[int(idx) - 1] = data_item
        return data_dict

    def load_data_list(self):
        """Load images and ground truth labels."""
        sample_dict = self._load_data_from_txt(self.ann_file)

        label_dict = self._load_data_from_txt(self.image_class_labels_file)

        split_dict = self._load_data_from_txt(self.train_test_split_file)

        assert sample_dict.keys() == label_dict.keys() == split_dict.keys(),\
            f'sample_ids should be same in files {self.ann_file}, ' \
            f'{self.image_class_labels_file} and {self.train_test_split_file}'

        data_list = []
        for sample_id in sample_dict.keys():
            if split_dict[sample_id] == '1' and self.split == 'test':
                # skip train samples when split='test'
                continue
            elif split_dict[sample_id] == '0' and self.split == 'train':
                # skip test samples when split='train'
                continue

            img_path = self.backend.join_path(self.img_prefix,
                                              sample_dict[sample_id])
            gt_label = int(label_dict[sample_id]) - 1
            info = dict(img_path=img_path, gt_label=gt_label)
            data_list.append(info)

        return data_list

    def extra_repr(self) -> List[str]:
        """The extra repr information of the dataset."""
        body = [
            f'Root of dataset: \t{self.data_root}',
        ]
        return body

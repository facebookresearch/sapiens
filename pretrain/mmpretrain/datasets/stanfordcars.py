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
from .categories import STANFORDCARS_CATEGORIES


@DATASETS.register_module()
class StanfordCars(BaseDataset):
    """The Stanford Cars Dataset.

    Support the `Stanford Cars Dataset <https://ai.stanford.edu/~jkrause/cars/car_dataset.html>`_ Dataset.
    The official website provides two ways to organize the dataset.
    Therefore, after downloading and decompression, the dataset directory structure is as follows.

    Stanford Cars dataset directory: ::

        Stanford_Cars
        ├── car_ims
        │   ├── 00001.jpg
        │   ├── 00002.jpg
        │   └── ...
        └── cars_annos.mat

    or ::

        Stanford_Cars
        ├── cars_train
        │   ├── 00001.jpg
        │   ├── 00002.jpg
        │   └── ...
        ├── cars_test
        │   ├── 00001.jpg
        │   ├── 00002.jpg
        │   └── ...
        └── devkit
            ├── cars_meta.mat
            ├── cars_train_annos.mat
            ├── cars_test_annos.mat
            ├── cars_test_annoswithlabels.mat
            ├── eval_train.m
            └── train_perfect_preds.txt

    Args:
        data_root (str): The root directory for Stanford Cars dataset.
        split (str, optional): The dataset split, supports "train"
            and "test". Default to "train".

    Examples:
        >>> from mmpretrain.datasets import StanfordCars
        >>> train_dataset = StanfordCars(data_root='data/Stanford_Cars', split='train')
        >>> train_dataset
        Dataset StanfordCars
            Number of samples:  8144
            Number of categories:       196
            Root of dataset:    data/Stanford_Cars
        >>> test_dataset = StanfordCars(data_root='data/Stanford_Cars', split='test')
        >>> test_dataset
        Dataset StanfordCars
            Number of samples:  8041
            Number of categories:       196
            Root of dataset:    data/Stanford_Cars
    """  # noqa: E501

    METAINFO = {'classes': STANFORDCARS_CATEGORIES}

    def __init__(self, data_root: str, split: str = 'train', **kwargs):

        splits = ['train', 'test']
        assert split in splits, \
            f"The split must be one of {splits}, but get '{split}'"
        self.split = split

        test_mode = split == 'test'
        self.backend = get_file_backend(data_root, enable_singleton=True)

        anno_file_path = self.backend.join_path(data_root, 'cars_annos.mat')
        if self.backend.exists(anno_file_path):
            ann_file = 'cars_annos.mat'
            data_prefix = ''
        else:
            if test_mode:
                ann_file = self.backend.join_path(
                    'devkit', 'cars_test_annos_withlabels.mat')
                data_prefix = 'cars_test'
            else:
                ann_file = self.backend.join_path('devkit',
                                                  'cars_train_annos.mat')
                data_prefix = 'cars_train'

            if not self.backend.exists(
                    self.backend.join_path(data_root, ann_file)):
                doc_url = 'https://mmpretrain.readthedocs.io/en/latest/api/datasets.html#stanfordcars'  # noqa: E501
                raise RuntimeError(
                    f'The dataset is incorrectly organized, please \
                    refer to {doc_url} and reorganize your folders.')

        super(StanfordCars, self).__init__(
            ann_file=ann_file,
            data_root=data_root,
            data_prefix=data_prefix,
            test_mode=test_mode,
            **kwargs)

    def load_data_list(self):
        data = mat4py.loadmat(self.ann_file)['annotations']

        data_list = []
        if 'test' in data.keys():
            # first way
            img_paths, labels, test = data['relative_im_path'], data[
                'class'], data['test']
            num = len(img_paths)
            assert num == len(labels) == len(test), 'get error ann file'
            for i in range(num):
                if not self.test_mode and test[i] == 1:
                    continue
                if self.test_mode and test[i] == 0:
                    continue
                img_path = self.backend.join_path(self.img_prefix,
                                                  img_paths[i])
                gt_label = labels[i] - 1
                info = dict(img_path=img_path, gt_label=gt_label)
                data_list.append(info)
        else:
            # second way
            img_names, labels = data['fname'], data['class']
            num = len(img_names)
            assert num == len(labels), 'get error ann file'
            for i in range(num):
                img_path = self.backend.join_path(self.img_prefix,
                                                  img_names[i])
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

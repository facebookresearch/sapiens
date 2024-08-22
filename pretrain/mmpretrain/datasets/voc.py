# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import xml.etree.ElementTree as ET
from typing import List, Optional, Union

from mmengine import get_file_backend, list_from_file
from mmengine.logging import MMLogger

from mmpretrain.registry import DATASETS
from .base_dataset import expanduser
from .categories import VOC2007_CATEGORIES
from .multi_label import MultiLabelDataset


@DATASETS.register_module()
class VOC(MultiLabelDataset):
    """`Pascal VOC <http://host.robots.ox.ac.uk/pascal/VOC/>`_ Dataset.

    After decompression, the dataset directory structure is as follows:

    VOC dataset directory: ::

        VOC2007
        ├── JPEGImages
        │   ├── xxx.jpg
        │   ├── xxy.jpg
        │   └── ...
        ├── Annotations
        │   ├── xxx.xml
        │   ├── xxy.xml
        │   └── ...
        └── ImageSets
            └── Main
                ├── train.txt
                ├── val.txt
                ├── trainval.txt
                ├── test.txt
                └── ...

    Extra difficult label is in VOC annotations, we will use
    `gt_label_difficult` to record the difficult labels in each sample
    and corresponding evaluation should take care of this field
    to calculate metrics. Usually, difficult labels are reckoned as
    negative in defaults.

    Args:
        data_root (str): The root directory for VOC dataset.
        split (str, optional): The dataset split, supports "train",
            "val", "trainval", and "test". Default to "trainval".
        image_set_path (str, optional): The path of image set, The file which
            lists image ids of the sub dataset, and this path is relative
            to ``data_root``. Default to ''.
        data_prefix (dict): Prefix for data and annotation, keyword
            'img_path' and 'ann_path' can be set. Defaults to be
            ``dict(img_path='JPEGImages', ann_path='Annotations')``.
        metainfo (dict, optional): Meta information for dataset, such as
            categories information. Defaults to None.
        **kwargs: Other keyword arguments in :class:`BaseDataset`.

    Examples:
        >>> from mmpretrain.datasets import VOC
        >>> train_dataset = VOC(data_root='data/VOC2007', split='trainval')
        >>> train_dataset
        Dataset VOC
            Number of samples:  5011
            Number of categories:       20
            Prefix of dataset:  data/VOC2007
            Path of image set:  data/VOC2007/ImageSets/Main/trainval.txt
            Prefix of images:   data/VOC2007/JPEGImages
            Prefix of annotations:      data/VOC2007/Annotations
        >>> test_dataset = VOC(data_root='data/VOC2007', split='test')
        >>> test_dataset
        Dataset VOC
            Number of samples:  4952
            Number of categories:       20
            Prefix of dataset:  data/VOC2007
            Path of image set:  data/VOC2007/ImageSets/Main/test.txt
            Prefix of images:   data/VOC2007/JPEGImages
            Prefix of annotations:      data/VOC2007/Annotations
    """  # noqa: E501

    METAINFO = {'classes': VOC2007_CATEGORIES}

    def __init__(self,
                 data_root: str,
                 split: str = 'trainval',
                 image_set_path: str = '',
                 data_prefix: Union[str, dict] = dict(
                     img_path='JPEGImages', ann_path='Annotations'),
                 test_mode: bool = False,
                 metainfo: Optional[dict] = None,
                 **kwargs):

        self.backend = get_file_backend(data_root, enable_singleton=True)

        if split:
            splits = ['train', 'val', 'trainval', 'test']
            assert split in splits, \
                f"The split must be one of {splits}, but get '{split}'"
            self.split = split

            if not data_prefix:
                data_prefix = dict(
                    img_path='JPEGImages', ann_path='Annotations')
            if not image_set_path:
                image_set_path = self.backend.join_path(
                    'ImageSets', 'Main', f'{split}.txt')

        # To handle the BC-breaking
        if (split == 'train' or split == 'trainval') and test_mode:
            logger = MMLogger.get_current_instance()
            logger.warning(f'split="{split}" but test_mode=True. '
                           f'The {split} set will be used.')

        if isinstance(data_prefix, str):
            data_prefix = dict(img_path=expanduser(data_prefix))
        assert isinstance(data_prefix, dict) and 'img_path' in data_prefix, \
            '`data_prefix` must be a dict with key img_path'

        if (split and split not in ['val', 'test']) or not test_mode:
            assert 'ann_path' in data_prefix and data_prefix[
                'ann_path'] is not None, \
                '"ann_path" must be set in `data_prefix`' \
                'when validation or test set is used.'

        self.data_root = data_root
        self.image_set_path = self.backend.join_path(data_root, image_set_path)

        super().__init__(
            ann_file='',
            metainfo=metainfo,
            data_root=data_root,
            data_prefix=data_prefix,
            test_mode=test_mode,
            **kwargs)

    @property
    def ann_prefix(self):
        """The prefix of images."""
        if 'ann_path' in self.data_prefix:
            return self.data_prefix['ann_path']
        else:
            return None

    def _get_labels_from_xml(self, img_id):
        """Get gt_labels and labels_difficult from xml file."""
        xml_path = self.backend.join_path(self.ann_prefix, f'{img_id}.xml')
        content = self.backend.get(xml_path)
        root = ET.fromstring(content)

        labels, labels_difficult = set(), set()
        for obj in root.findall('object'):
            label_name = obj.find('name').text
            # in case customized dataset has wrong labels
            # or CLASSES has been override.
            if label_name not in self.CLASSES:
                continue
            label = self.class_to_idx[label_name]
            difficult = int(obj.find('difficult').text)
            if difficult:
                labels_difficult.add(label)
            else:
                labels.add(label)

        return list(labels), list(labels_difficult)

    def load_data_list(self):
        """Load images and ground truth labels."""
        data_list = []
        img_ids = list_from_file(self.image_set_path)

        for img_id in img_ids:
            img_path = self.backend.join_path(self.img_prefix, f'{img_id}.jpg')

            labels, labels_difficult = None, None
            if self.ann_prefix is not None:
                labels, labels_difficult = self._get_labels_from_xml(img_id)

            info = dict(
                img_path=img_path,
                gt_label=labels,
                gt_label_difficult=labels_difficult)
            data_list.append(info)

        return data_list

    def extra_repr(self) -> List[str]:
        """The extra repr information of the dataset."""
        body = [
            f'Prefix of dataset: \t{self.data_root}',
            f'Path of image set: \t{self.image_set_path}',
            f'Prefix of images: \t{self.img_prefix}',
            f'Prefix of annotations: \t{self.ann_prefix}'
        ]

        return body

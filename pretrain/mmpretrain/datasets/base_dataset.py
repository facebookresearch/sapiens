# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os.path as osp
from os import PathLike
from typing import List, Optional, Sequence, Union

import mmengine
import numpy as np
from mmengine.dataset import BaseDataset as _BaseDataset

from mmpretrain.registry import DATASETS, TRANSFORMS


def expanduser(path):
    """Expand ~ and ~user constructions.

    If user or $HOME is unknown, do nothing.
    """
    if isinstance(path, (str, PathLike)):
        return osp.expanduser(path)
    else:
        return path


@DATASETS.register_module()
class BaseDataset(_BaseDataset):
    """Base dataset for image classification task.

    This dataset support annotation file in `OpenMMLab 2.0 style annotation
    format`.

    .. _OpenMMLab 2.0 style annotation format:
        https://github.com/open-mmlab/mmengine/blob/main/docs/zh_cn/tutorials/basedataset.md

    Comparing with the :class:`mmengine.BaseDataset`, this class implemented
    several useful methods.

    Args:
        ann_file (str): Annotation file path.
        metainfo (dict, optional): Meta information for dataset, such as class
            information. Defaults to None.
        data_root (str): The root directory for ``data_prefix`` and
            ``ann_file``. Defaults to ''.
        data_prefix (str | dict): Prefix for training data. Defaults to ''.
        filter_cfg (dict, optional): Config for filter data. Defaults to None.
        indices (int or Sequence[int], optional): Support using first few
            data in annotation file to facilitate training/testing on a smaller
            dataset. Defaults to None, which means using all ``data_infos``.
        serialize_data (bool): Whether to hold memory using serialized objects,
            when enabled, data loader workers can use shared RAM from master
            process instead of making a copy. Defaults to True.
        pipeline (Sequence): Processing pipeline. Defaults to an empty tuple.
        test_mode (bool, optional): ``test_mode=True`` means in test phase,
            an error will be raised when getting an item fails, ``test_mode=False``
            means in training phase, another item will be returned randomly.
            Defaults to False.
        lazy_init (bool): Whether to load annotation during instantiation.
            In some cases, such as visualization, only the meta information of
            the dataset is needed, which is not necessary to load annotation
            file. ``Basedataset`` can skip load annotations to save time by set
            ``lazy_init=False``. Defaults to False.
        max_refetch (int): If ``Basedataset.prepare_data`` get a None img.
            The maximum extra number of cycles to get a valid image.
            Defaults to 1000.
        classes (str | Sequence[str], optional): Specify names of classes.

            - If is string, it should be a file path, and the every line of
              the file is a name of a class.
            - If is a sequence of string, every item is a name of class.
            - If is None, use categories information in ``metainfo`` argument,
              annotation file or the class attribute ``METAINFO``.

            Defaults to None.
    """  # noqa: E501

    def __init__(self,
                 ann_file: str,
                 metainfo: Optional[dict] = None,
                 data_root: str = '',
                 data_prefix: Union[str, dict] = '',
                 filter_cfg: Optional[dict] = None,
                 indices: Optional[Union[int, Sequence[int]]] = None,
                 serialize_data: bool = True,
                 pipeline: Sequence = (),
                 test_mode: bool = False,
                 lazy_init: bool = False,
                 max_refetch: int = 1000,
                 classes: Union[str, Sequence[str], None] = None):
        if isinstance(data_prefix, str):
            data_prefix = dict(img_path=expanduser(data_prefix))

        ann_file = expanduser(ann_file)
        metainfo = self._compat_classes(metainfo, classes)

        transforms = []
        for transform in pipeline:
            if isinstance(transform, dict):
                transforms.append(TRANSFORMS.build(transform))
            else:
                transforms.append(transform)

        super().__init__(
            ann_file=ann_file,
            metainfo=metainfo,
            data_root=data_root,
            data_prefix=data_prefix,
            filter_cfg=filter_cfg,
            indices=indices,
            serialize_data=serialize_data,
            pipeline=transforms,
            test_mode=test_mode,
            lazy_init=lazy_init,
            max_refetch=max_refetch)

    @property
    def img_prefix(self):
        """The prefix of images."""
        return self.data_prefix['img_path']

    @property
    def CLASSES(self):
        """Return all categories names."""
        return self._metainfo.get('classes', None)

    @property
    def class_to_idx(self):
        """Map mapping class name to class index.

        Returns:
            dict: mapping from class name to class index.
        """

        return {cat: i for i, cat in enumerate(self.CLASSES)}

    def get_gt_labels(self):
        """Get all ground-truth labels (categories).

        Returns:
            np.ndarray: categories for all images.
        """

        gt_labels = np.array(
            [self.get_data_info(i)['gt_label'] for i in range(len(self))])
        return gt_labels

    def get_cat_ids(self, idx: int) -> List[int]:
        """Get category id by index.

        Args:
            idx (int): Index of data.

        Returns:
            cat_ids (List[int]): Image category of specified index.
        """

        return [int(self.get_data_info(idx)['gt_label'])]

    def _compat_classes(self, metainfo, classes):
        """Merge the old style ``classes`` arguments to ``metainfo``."""
        if isinstance(classes, str):
            # take it as a file path
            class_names = mmengine.list_from_file(expanduser(classes))
        elif isinstance(classes, (tuple, list)):
            class_names = classes
        elif classes is not None:
            raise ValueError(f'Unsupported type {type(classes)} of classes.')

        if metainfo is None:
            metainfo = {}

        if classes is not None:
            metainfo = {'classes': tuple(class_names), **metainfo}

        return metainfo

    def full_init(self):
        """Load annotation file and set ``BaseDataset._fully_initialized`` to
        True."""
        super().full_init()

        #  To support the standard OpenMMLab 2.0 annotation format. Generate
        #  metainfo in internal format from standard metainfo format.
        if 'categories' in self._metainfo and 'classes' not in self._metainfo:
            categories = sorted(
                self._metainfo['categories'], key=lambda x: x['id'])
            self._metainfo['classes'] = tuple(
                [cat['category_name'] for cat in categories])

    def __repr__(self):
        """Print the basic information of the dataset.

        Returns:
            str: Formatted string.
        """
        head = 'Dataset ' + self.__class__.__name__
        body = []
        if self._fully_initialized:
            body.append(f'Number of samples: \t{self.__len__()}')
        else:
            body.append("Haven't been initialized")

        if self.CLASSES is not None:
            body.append(f'Number of categories: \t{len(self.CLASSES)}')

        body.extend(self.extra_repr())

        if len(self.pipeline.transforms) > 0:
            body.append('With transforms:')
            for t in self.pipeline.transforms:
                body.append(f'    {t}')

        lines = [head] + [' ' * 4 + line for line in body]
        return '\n'.join(lines)

    def extra_repr(self) -> List[str]:
        """The extra repr information of the dataset."""
        body = []
        body.append(f'Annotation file: \t{self.ann_file}')
        body.append(f'Prefix of images: \t{self.img_prefix}')
        return body

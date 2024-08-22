# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import copy
import os.path as osp
from os import PathLike
from typing import Optional, Sequence

import mmengine
from mmcv.transforms import Compose
from mmengine.fileio import get_file_backend

from .builder import DATASETS


def expanduser(path):
    if isinstance(path, (str, PathLike)):
        return osp.expanduser(path)
    else:
        return path


def isabs(uri):
    return osp.isabs(uri) or ('://' in uri)


@DATASETS.register_module()
class MultiTaskDataset:
    """Custom dataset for multi-task dataset.

    To use the dataset, please generate and provide an annotation file in the
    below format:

    .. code-block:: json

        {
          "metainfo": {
            "tasks":
              [
              'gender'
              'wear'
              ]
          },
          "data_list": [
            {
              "img_path": "a.jpg",
              gt_label:{
                  "gender": 0,
                  "wear": [1, 0, 1, 0]
                }
            },
            {
              "img_path": "b.jpg",
              gt_label:{
                  "gender": 1,
                  "wear": [1, 0, 1, 0]
                }
            }
          ]
        }

    Assume we put our dataset in the ``data/mydataset`` folder in the
    repository and organize it as the below format: ::

        mmpretrain/
        └── data
            └── mydataset
                ├── annotation
                │   ├── train.json
                │   ├── test.json
                │   └── val.json
                ├── train
                │   ├── a.jpg
                │   └── ...
                ├── test
                │   ├── b.jpg
                │   └── ...
                └── val
                    ├── c.jpg
                    └── ...

    We can use the below config to build datasets:

    .. code:: python

        >>> from mmpretrain.datasets import build_dataset
        >>> train_cfg = dict(
        ...     type="MultiTaskDataset",
        ...     ann_file="annotation/train.json",
        ...     data_root="data/mydataset",
        ...     # The `img_path` field in the train annotation file is relative
        ...     # to the `train` folder.
        ...     data_prefix='train',
        ... )
        >>> train_dataset = build_dataset(train_cfg)

    Or we can put all files in the same folder: ::

        mmpretrain/
        └── data
            └── mydataset
                 ├── train.json
                 ├── test.json
                 ├── val.json
                 ├── a.jpg
                 ├── b.jpg
                 ├── c.jpg
                 └── ...

    And we can use the below config to build datasets:

    .. code:: python

        >>> from mmpretrain.datasets import build_dataset
        >>> train_cfg = dict(
        ...     type="MultiTaskDataset",
        ...     ann_file="train.json",
        ...     data_root="data/mydataset",
        ...     # the `data_prefix` is not required since all paths are
        ...     # relative to the `data_root`.
        ... )
        >>> train_dataset = build_dataset(train_cfg)


    Args:
        ann_file (str): The annotation file path. It can be either absolute
            path or relative path to the ``data_root``.
        metainfo (dict, optional): The extra meta information. It should be
            a dict with the same format as the ``"metainfo"`` field in the
            annotation file. Defaults to None.
        data_root (str, optional): The root path of the data directory. It's
            the prefix of the ``data_prefix`` and the ``ann_file``. And it can
            be a remote path like "s3://openmmlab/xxx/". Defaults to None.
        data_prefix (str, optional): The base folder relative to the
            ``data_root`` for the ``"img_path"`` field in the annotation file.
            Defaults to None.
        pipeline (Sequence[dict]): A list of dict, where each element
            represents a operation defined in
            :mod:`mmpretrain.datasets.pipelines`. Defaults to an empty tuple.
        test_mode (bool): in train mode or test mode. Defaults to False.
    """
    METAINFO = dict()

    def __init__(self,
                 ann_file: str,
                 metainfo: Optional[dict] = None,
                 data_root: Optional[str] = None,
                 data_prefix: Optional[str] = None,
                 pipeline: Sequence = (),
                 test_mode: bool = False):

        self.data_root = expanduser(data_root)

        # Inference the file client
        if self.data_root is not None:
            self.file_backend = get_file_backend(uri=self.data_root)
        else:
            self.file_backend = None

        self.ann_file = self._join_root(expanduser(ann_file))
        self.data_prefix = self._join_root(data_prefix)

        self.test_mode = test_mode
        self.pipeline = Compose(pipeline)
        self.data_list = self.load_data_list(self.ann_file, metainfo)

    def _join_root(self, path):
        """Join ``self.data_root`` with the specified path.

        If the path is an absolute path, just return the path. And if the
        path is None, return ``self.data_root``.

        Examples:
            >>> self.data_root = 'a/b/c'
            >>> self._join_root('d/e/')
            'a/b/c/d/e'
            >>> self._join_root('https://openmmlab.com')
            'https://openmmlab.com'
            >>> self._join_root(None)
            'a/b/c'
        """
        if path is None:
            return self.data_root
        if isabs(path):
            return path

        joined_path = self.file_backend.join_path(self.data_root, path)
        return joined_path

    @classmethod
    def _get_meta_info(cls, in_metainfo: dict = None) -> dict:
        """Collect meta information from the dictionary of meta.

        Args:
            in_metainfo (dict): Meta information dict.

        Returns:
            dict: Parsed meta information.
        """
        # `cls.METAINFO` will be overwritten by in_meta
        metainfo = copy.deepcopy(cls.METAINFO)
        if in_metainfo is None:
            return metainfo

        metainfo.update(in_metainfo)

        return metainfo

    def load_data_list(self, ann_file, metainfo_override=None):
        """Load annotations from an annotation file.

        Args:
            ann_file (str): Absolute annotation file path if ``self.root=None``
                or relative path if ``self.root=/path/to/data/``.

        Returns:
            list[dict]: A list of annotation.
        """
        annotations = mmengine.load(ann_file)
        if not isinstance(annotations, dict):
            raise TypeError(f'The annotations loaded from annotation file '
                            f'should be a dict, but got {type(annotations)}!')
        if 'data_list' not in annotations:
            raise ValueError('The annotation file must have the `data_list` '
                             'field.')
        metainfo = annotations.get('metainfo', {})
        raw_data_list = annotations['data_list']

        # Set meta information.
        assert isinstance(metainfo, dict), 'The `metainfo` field in the '\
            f'annotation file should be a dict, but got {type(metainfo)}'
        if metainfo_override is not None:
            assert isinstance(metainfo_override, dict), 'The `metainfo` ' \
                f'argument should be a dict, but got {type(metainfo_override)}'
            metainfo.update(metainfo_override)
        self._metainfo = self._get_meta_info(metainfo)

        data_list = []
        for i, raw_data in enumerate(raw_data_list):
            try:
                data_list.append(self.parse_data_info(raw_data))
            except AssertionError as e:
                raise RuntimeError(
                    f'The format check fails during parse the item {i} of '
                    f'the annotation file with error: {e}')
        return data_list

    def parse_data_info(self, raw_data):
        """Parse raw annotation to target format.

        This method will return a dict which contains the data information of a
        sample.

        Args:
            raw_data (dict): Raw data information load from ``ann_file``

        Returns:
            dict: Parsed annotation.
        """
        assert isinstance(raw_data, dict), \
            f'The item should be a dict, but got {type(raw_data)}'
        assert 'img_path' in raw_data, \
            "The item doesn't have `img_path` field."
        data = dict(
            img_path=self._join_root(raw_data['img_path']),
            gt_label=raw_data['gt_label'],
        )
        return data

    @property
    def metainfo(self) -> dict:
        """Get meta information of dataset.

        Returns:
            dict: meta information collected from ``cls.METAINFO``,
            annotation file and metainfo argument during instantiation.
        """
        return copy.deepcopy(self._metainfo)

    def prepare_data(self, idx):
        """Get data processed by ``self.pipeline``.

        Args:
            idx (int): The index of ``data_info``.

        Returns:
            Any: Depends on ``self.pipeline``.
        """
        results = copy.deepcopy(self.data_list[idx])
        return self.pipeline(results)

    def __len__(self):
        """Get the length of the whole dataset.

        Returns:
            int: The length of filtered dataset.
        """
        return len(self.data_list)

    def __getitem__(self, idx):
        """Get the idx-th image and data information of dataset after
        ``self.pipeline``.

        Args:
            idx (int): The index of of the data.

        Returns:
            dict: The idx-th image and data information after
            ``self.pipeline``.
        """
        return self.prepare_data(idx)

    def __repr__(self):
        """Print the basic information of the dataset.

        Returns:
            str: Formatted string.
        """
        head = 'Dataset ' + self.__class__.__name__
        body = [f'Number of samples: \t{self.__len__()}']
        if self.data_root is not None:
            body.append(f'Root location: \t{self.data_root}')
        body.append(f'Annotation file: \t{self.ann_file}')
        if self.data_prefix is not None:
            body.append(f'Prefix of images: \t{self.data_prefix}')
        # -------------------- extra repr --------------------
        tasks = self.metainfo['tasks']
        body.append(f'For {len(tasks)} tasks')
        for task in tasks:
            body.append(f' {task} ')
        # ----------------------------------------------------

        if len(self.pipeline.transforms) > 0:
            body.append('With transforms:')
            for t in self.pipeline.transforms:
                body.append(f'    {t}')

        lines = [head] + [' ' * 4 + line for line in body]
        return '\n'.join(lines)

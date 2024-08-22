# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import Callable, List, Sequence

import mmengine
from mmengine.dataset import BaseDataset
from mmengine.fileio import get_file_backend

from mmpretrain.registry import DATASETS


@DATASETS.register_module()
class ScienceQA(BaseDataset):
    """ScienceQA dataset.

    This dataset is used to load the multimodal data of ScienceQA dataset.

    Args:
        data_root (str): The root directory for ``data_prefix`` and
            ``ann_file``.
        split (str): The split of dataset. Options: ``train``, ``val``,
            ``test``, ``trainval``, ``minival``, and ``minitest``.
        split_file (str): The split file of dataset, which contains the
            ids of data samples in the split.
        ann_file (str): Annotation file path.
        image_only (bool): Whether only to load data with image. Defaults to
            False.
        data_prefix (dict): Prefix for data field. Defaults to
            ``dict(img_path='')``.
        pipeline (Sequence): Processing pipeline. Defaults to an empty tuple.
        **kwargs: Other keyword arguments in :class:`BaseDataset`.
    """

    def __init__(self,
                 data_root: str,
                 split: str,
                 split_file: str,
                 ann_file: str,
                 image_only: bool = False,
                 data_prefix: dict = dict(img_path=''),
                 pipeline: Sequence[Callable] = (),
                 **kwargs):
        assert split in [
            'train', 'val', 'test', 'trainval', 'minival', 'minitest'
        ], f'Invalid split {split}'
        self.split = split
        self.split_file = os.path.join(data_root, split_file)
        self.image_only = image_only

        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            data_prefix=data_prefix,
            pipeline=pipeline,
            **kwargs)

    def load_data_list(self) -> List[dict]:
        """Load data list."""
        img_prefix = self.data_prefix['img_path']
        annotations = mmengine.load(self.ann_file)
        current_data_split = mmengine.load(self.split_file)[self.split]  # noqa

        file_backend = get_file_backend(img_prefix)

        data_list = []
        for data_id in current_data_split:
            ann = annotations[data_id]
            if self.image_only and ann['image'] is None:
                continue
            data_info = {
                'image_id':
                data_id,
                'question':
                ann['question'],
                'choices':
                ann['choices'],
                'gt_answer':
                ann['answer'],
                'hint':
                ann['hint'],
                'image_name':
                ann['image'],
                'task':
                ann['task'],
                'grade':
                ann['grade'],
                'subject':
                ann['subject'],
                'topic':
                ann['topic'],
                'category':
                ann['category'],
                'skill':
                ann['skill'],
                'lecture':
                ann['lecture'],
                'solution':
                ann['solution'],
                'split':
                ann['split'],
                'img_path':
                file_backend.join_path(img_prefix, data_id, ann['image'])
                if ann['image'] is not None else None,
                'has_image':
                True if ann['image'] is not None else False,
            }
            data_list.append(data_info)

        return data_list

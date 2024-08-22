# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import List

import mmengine
from mmengine.dataset import BaseDataset

from mmpretrain.registry import DATASETS


@DATASETS.register_module()
class VSR(BaseDataset):
    """VSR: Visual Spatial Reasoning dataset.

    Args:
        data_root (str): The root directory for ``data_prefix``, ``ann_file``
            and ``question_file``.
        data_prefix (str): The directory of images.
        ann_file (str, optional): Annotation file path for training and
            validation. Defaults to an empty string.
        **kwargs: Other keyword arguments in :class:`BaseDataset`.
    """

    def __init__(self,
                 data_root: str,
                 data_prefix: str,
                 ann_file: str = '',
                 **kwarg):
        super().__init__(
            data_root=data_root,
            data_prefix=dict(img_path=data_prefix),
            ann_file=ann_file,
            **kwarg,
        )

    def load_data_list(self) -> List[dict]:
        """Load data list."""
        annotations = mmengine.load(self.ann_file)

        data_list = []
        for ann in annotations:
            # ann example
            # {
            #     "image": "train2017/000000372029.jpg",
            #     "question": "The dog is on the surfboard.",
            #     "answer": true
            # }
            data_info = dict()
            data_info['img_path'] = mmengine.join_path(
                self.data_prefix['img_path'], ann['image'])
            data_info['question'] = ann['question']
            data_info['gt_answer'] = 'yes' if ann['answer'] else 'no'

            data_list.append(data_info)

        return data_list

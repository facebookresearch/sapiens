# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import re
from itertools import chain
from typing import List

import mmengine
from mmengine.dataset import BaseDataset

from mmpretrain.registry import DATASETS


@DATASETS.register_module()
class VisualGenomeQA(BaseDataset):
    """Visual Genome Question Answering dataset.

    dataset structure: ::

        data_root
        ├── image
        │   ├── 1.jpg
        │   ├── 2.jpg
        │   └── ...
        └── question_answers.json

    Args:
        data_root (str): The root directory for ``data_prefix``, ``ann_file``
            and ``question_file``.
        data_prefix (str): The directory of images. Defaults to ``"image"``.
        ann_file (str, optional): Annotation file path for training and
            validation. Defaults to ``"question_answers.json"``.
        **kwargs: Other keyword arguments in :class:`BaseDataset`.
    """

    def __init__(self,
                 data_root: str,
                 data_prefix: str = 'image',
                 ann_file: str = 'question_answers.json',
                 **kwarg):
        super().__init__(
            data_root=data_root,
            data_prefix=dict(img_path=data_prefix),
            ann_file=ann_file,
            **kwarg,
        )

    def _create_image_index(self):
        img_prefix = self.data_prefix['img_path']

        files = mmengine.list_dir_or_file(img_prefix, list_dir=False)
        image_index = {}
        for file in files:
            image_id = re.findall(r'\d+', file)
            if len(image_id) > 0:
                image_id = int(image_id[-1])
                image_index[image_id] = mmengine.join_path(img_prefix, file)

        return image_index

    def load_data_list(self) -> List[dict]:
        """Load data list."""
        annotations = mmengine.load(self.ann_file)

        # The original Visual Genome annotation file and question file includes
        # only image id but no image file paths.
        self.image_index = self._create_image_index()

        data_list = []
        for qas in chain.from_iterable(ann['qas'] for ann in annotations):
            # ann example
            # {
            #     'id': 1,
            #     'qas': [
            #         {
            #             'a_objects': [],
            #             'question': 'What color is the clock?',
            #             'image_id': 1,
            #             'qa_id': 986768,
            #             'answer': 'Two.',
            #             'q_objects': [],
            #         }
            #         ...
            #     ]
            # }

            data_info = {
                'img_path': self.image_index[qas['image_id']],
                'quesiton': qas['quesiton'],
                'question_id': qas['question_id'],
                'image_id': qas['image_id'],
                'gt_answer': [qas['answer']],
            }

            data_list.append(data_info)

        return data_list

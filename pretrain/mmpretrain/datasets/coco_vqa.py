# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os.path as osp
import re
from collections import Counter
from typing import List

import mmengine
from mmengine.dataset import BaseDataset

from mmpretrain.registry import DATASETS


@DATASETS.register_module()
class COCOVQA(BaseDataset):
    """VQAv2 dataset.

    Args:
        data_root (str): The root directory for ``data_prefix``, ``ann_file``
            and ``question_file``.
        data_prefix (str): The directory of images.
        question_file (str): Question file path.
        ann_file (str, optional): Annotation file path for training and
            validation. Defaults to an empty string.
        **kwargs: Other keyword arguments in :class:`BaseDataset`.
    """

    def __init__(self,
                 data_root: str,
                 data_prefix: str,
                 question_file: str,
                 ann_file: str = '',
                 **kwarg):
        self.question_file = question_file
        super().__init__(
            data_root=data_root,
            data_prefix=dict(img_path=data_prefix),
            ann_file=ann_file,
            **kwarg,
        )

    def _join_prefix(self):
        if not mmengine.is_abs(self.question_file) and self.question_file:
            self.question_file = osp.join(self.data_root, self.question_file)

        return super()._join_prefix()

    def _create_image_index(self):
        img_prefix = self.data_prefix['img_path']

        files = mmengine.list_dir_or_file(img_prefix, list_dir=False)
        image_index = {}
        for file in files:
            image_id = re.findall(r'\d{12}', file)
            if len(image_id) > 0:
                image_id = int(image_id[-1])
                image_index[image_id] = mmengine.join_path(img_prefix, file)

        return image_index

    def load_data_list(self) -> List[dict]:
        """Load data list."""
        questions = mmengine.load(self.question_file)['questions']
        if self.ann_file:
            annotations = mmengine.load(self.ann_file)['annotations']
            assert len(questions) == len(annotations)
        else:
            annotations = [None] * len(questions)

        # The original VQAv2 annotation file and question file includes
        # only image id but no image file paths.
        self.image_index = self._create_image_index()

        data_list = []
        for question, ann in zip(questions, annotations):
            # question example
            # {
            #     'image_id': 262144,
            #     'question': "Is the ball flying towards the batter?",
            #     'question_id': 262144000
            # }
            #
            # ann example
            # {
            #     'question_type': "what are the",
            #     'answer_type': "other",
            #     'answers': [
            #         {'answer': 'watching',
            #          'answer_id': 1,
            #          'answer_confidence': 'yes'},
            #         ...
            #     ],
            #     'image_id': 262148,
            #     'question_id': 262148000,
            #     'multiple_choice_answer': 'watching',
            #     'answer_type': 'other',
            # }

            data_info = question
            data_info['img_path'] = self.image_index[question['image_id']]

            if ann is not None:
                assert ann['question_id'] == question['question_id']

                # add answer_weight & answer_count, delete duplicate answer
                answers = [item['answer'] for item in ann.pop('answers')]
                count = Counter(answers)
                answer_weight = [i / len(answers) for i in count.values()]
                data_info['gt_answer'] = list(count.keys())
                data_info['gt_answer_weight'] = answer_weight
                data_info.update(ann)

            data_list.append(data_info)

        return data_list

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections import Counter
from typing import List

import mmengine
from mmengine.dataset import BaseDataset

from mmpretrain.registry import DATASETS


@DATASETS.register_module()
class VizWiz(BaseDataset):
    """VizWiz dataset.

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
            # {
            #     "image": "VizWiz_val_00000001.jpg",
            #     "question": "Can you tell me what this medicine is please?",
            #     "answers": [
            #     {
            #         "answer": "no",
            #         "answer_confidence": "yes"
            #     },
            #     {
            #         "answer": "unanswerable",
            #         "answer_confidence": "yes"
            #     },
            #     {
            #         "answer": "night time",
            #         "answer_confidence": "maybe"
            #     },
            #     {
            #         "answer": "unanswerable",
            #         "answer_confidence": "yes"
            #     },
            #     {
            #         "answer": "night time",
            #         "answer_confidence": "maybe"
            #     },
            #     {
            #         "answer": "night time cold medicine",
            #         "answer_confidence": "maybe"
            #     },
            #     {
            #         "answer": "night time",
            #         "answer_confidence": "maybe"
            #     },
            #     {
            #         "answer": "night time",
            #         "answer_confidence": "maybe"
            #     },
            #     {
            #         "answer": "night time",
            #         "answer_confidence": "maybe"
            #     },
            #     {
            #         "answer": "night time medicine",
            #         "answer_confidence": "yes"
            #     }
            #     ],
            #     "answer_type": "other",
            #     "answerable": 1
            # },
            data_info = dict()
            data_info['question'] = ann['question']
            data_info['img_path'] = mmengine.join_path(
                self.data_prefix['img_path'], ann['image'])

            if 'answerable' not in ann:
                data_list.append(data_info)
            else:
                if ann['answerable'] == 1:
                    # add answer_weight & answer_count, delete duplicate answer
                    answers = []
                    for item in ann.pop('answers'):
                        if item['answer_confidence'] == 'yes' and item[
                                'answer'] != 'unanswerable':
                            answers.append(item['answer'])
                    count = Counter(answers)
                    answer_weight = [i / len(answers) for i in count.values()]
                    data_info['gt_answer'] = list(count.keys())
                    data_info['gt_answer_weight'] = answer_weight
                    # data_info.update(ann)
                    data_list.append(data_info)

        return data_list

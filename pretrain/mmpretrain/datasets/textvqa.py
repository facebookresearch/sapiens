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
class TextVQA(BaseDataset):
    """TextVQA dataset.

    val image:
        https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip
    test image:
        https://dl.fbaipublicfiles.com/textvqa/images/test_images.zip
    val json:
        https://dl.fbaipublicfiles.com/textvqa/data/TextVQA_0.5.1_val.json
    test json:
        https://dl.fbaipublicfiles.com/textvqa/data/TextVQA_0.5.1_test.json

    folder structure:
    data/textvqa
        ├── annotations
        │   ├── TextVQA_0.5.1_test.json
        │   └── TextVQA_0.5.1_val.json
        └── images
            ├── test_images
            └── train_images

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
        annotations = mmengine.load(self.ann_file)['data']

        data_list = []

        for ann in annotations:

            # ann example
            # {
            #     'question': 'what is the brand of...is camera?',
            #     'image_id': '003a8ae2ef43b901',
            #     'image_classes': [
            #         'Cassette deck', 'Printer', ...
            #         ],
            #     'flickr_original_url': 'https://farm2.static...04a6_o.jpg',
            #     'flickr_300k_url': 'https://farm2.static...04a6_o.jpg',
            #     'image_width': 1024,
            #     'image_height': 664,
            #     'answers': [
            #         'nous les gosses',
            #         'dakota',
            #         'clos culombu',
            #         'dakota digital' ...
            #        ],
            #     'question_tokens':
            #         ['what', 'is', 'the', 'brand', 'of', 'this', 'camera'],
            #     'question_id': 34602,
            #     'set_name': 'val'
            # }

            data_info = dict(question=ann['question'])
            data_info['question_id'] = ann['question_id']
            data_info['image_id'] = ann['image_id']

            img_path = mmengine.join_path(self.data_prefix['img_path'],
                                          ann['image_id'] + '.jpg')
            data_info['img_path'] = img_path

            data_info['question_id'] = ann['question_id']

            if 'answers' in ann:
                answers = [item for item in ann.pop('answers')]
                count = Counter(answers)
                answer_weight = [i / len(answers) for i in count.values()]
                data_info['gt_answer'] = list(count.keys())
                data_info['gt_answer_weight'] = answer_weight

            data_list.append(data_info)

        return data_list

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os.path as osp
from typing import List

import mmengine
from mmengine.dataset import BaseDataset

from mmpretrain.registry import DATASETS


@DATASETS.register_module()
class OCRVQA(BaseDataset):
    """OCR-VQA dataset.

    Args:
        data_root (str): The root directory for ``data_prefix``, ``ann_file``
            and ``question_file``.
        data_prefix (str): The directory of images.
        ann_file (str): Annotation file path for training and validation.
        split (str): 'train', 'val' or 'test'.
        **kwargs: Other keyword arguments in :class:`BaseDataset`.
    """

    def __init__(self, data_root: str, data_prefix: str, ann_file: str,
                 split: str, **kwarg):

        assert split in ['train', 'val', 'test'], \
            '`split` must be train, val or test'
        self.split = split
        super().__init__(
            data_root=data_root,
            data_prefix=dict(img_path=data_prefix),
            ann_file=ann_file,
            **kwarg,
        )

    def load_data_list(self) -> List[dict]:
        """Load data list."""

        split_dict = {1: 'train', 2: 'val', 3: 'test'}

        annotations = mmengine.load(self.ann_file)

        # ann example
        # "761183272": {
        #     "imageURL": \
        #         "http://ecx.images-amazon.com/images/I/61Y5cOdHJbL.jpg",
        #     "questions": [
        #         "Who wrote this book?",
        #         "What is the title of this book?",
        #         "What is the genre of this book?",
        #         "Is this a games related book?",
        #         "What is the year printed on this calendar?"],
        #     "answers": [
        #         "Sandra Boynton",
        #         "Mom's Family Wall Calendar 2016",
        #         "Calendars",
        #         "No",
        #         "2016"],
        #     "title": "Mom's Family Wall Calendar 2016",
        #     "authorName": "Sandra Boynton",
        #     "genre": "Calendars",
        #     "split": 1
        # },

        data_list = []

        for key, ann in annotations.items():
            if self.split != split_dict[ann['split']]:
                continue

            extension = osp.splitext(ann['imageURL'])[1]
            if extension not in ['.jpg', '.png']:
                continue
            img_path = mmengine.join_path(self.data_prefix['img_path'],
                                          key + extension)
            for question, answer in zip(ann['questions'], ann['answers']):
                data_info = {}
                data_info['img_path'] = img_path
                data_info['question'] = question
                data_info['gt_answer'] = answer
                data_info['gt_answer_weight'] = [1.0]

                data_info['imageURL'] = ann['imageURL']
                data_info['title'] = ann['title']
                data_info['authorName'] = ann['authorName']
                data_info['genre'] = ann['genre']

                data_list.append(data_info)

        return data_list

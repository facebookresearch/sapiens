# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections import OrderedDict
from typing import List

import mmengine
from mmengine import get_file_backend

from mmpretrain.registry import DATASETS
from .base_dataset import BaseDataset


@DATASETS.register_module()
class Flickr30kRetrieval(BaseDataset):
    """Flickr30k Retrieval dataset.

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
        # get file backend
        img_prefix = self.data_prefix['img_path']
        file_backend = get_file_backend(img_prefix)

        annotations = mmengine.load(self.ann_file)

        # mapping img_id to img filename
        img_dict = OrderedDict()
        img_idx = 0
        sentence_idx = 0
        train_list = []
        for img in annotations['images']:

            # img_example={
            #     "sentids": [0, 1, 2],
            #     "imgid": 0,
            #     "sentences": [
            #         {"raw": "Two men in green shirts standing in a yard.",
            #          "imgid": 0, "sentid": 0},
            #         {"raw": "A man in a blue shirt standing in a garden.",
            #          "imgid": 0, "sentid": 1},
            #         {"raw": "Two friends enjoy time spent together.",
            #          "imgid": 0, "sentid": 2}
            #     ],
            #     "split": "train",
            #     "filename": "1000092795.jpg"
            # },

            if img['split'] != self.split:
                continue

            # create new idx for image
            train_image = dict(
                ori_id=img['imgid'],
                image_id=img_idx,  # used for evaluation
                img_path=file_backend.join_path(img_prefix, img['filename']),
                text=[],
                gt_text_id=[],
                gt_image_id=[],
            )

            for sentence in img['sentences']:
                ann = {}
                ann['text'] = sentence['raw']
                ann['ori_id'] = sentence['sentid']
                ann['text_id'] = sentence_idx  # used for evaluation

                ann['image_ori_id'] = train_image['ori_id']
                ann['image_id'] = train_image['image_id']
                ann['img_path'] = train_image['img_path']
                ann['is_matched'] = True

                # 1. prepare train data list item
                train_list.append(ann)
                # 2. prepare eval data list item based on img dict
                train_image['text'].append(ann['text'])
                train_image['gt_text_id'].append(ann['text_id'])
                train_image['gt_image_id'].append(ann['image_id'])

                sentence_idx += 1

            img_dict[img['imgid']] = train_image
            img_idx += 1

        self.img_size = len(img_dict)
        self.text_size = len(train_list)

        # return needed format data list
        if self.test_mode:
            return list(img_dict.values())
        return train_list

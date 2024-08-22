# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
from collections import OrderedDict
from typing import List

from mmengine import get_file_backend

from mmpretrain.registry import DATASETS
from .base_dataset import BaseDataset


@DATASETS.register_module()
class COCORetrieval(BaseDataset):
    """COCO Retrieval dataset.

    Args:
        ann_file (str): Annotation file path.
        test_mode (bool): Whether dataset is used for evaluation. This will
            decide the annotation format in data list annotations.
            Defaults to False.
        data_root (str): The root directory for ``data_prefix`` and
            ``ann_file``. Defaults to ''.
        data_prefix (str | dict): Prefix for training data. Defaults to ''.
        pipeline (Sequence): Processing pipeline. Defaults to an empty tuple.
        **kwargs: Other keyword arguments in :class:`BaseDataset`.
    """

    def load_data_list(self) -> List[dict]:
        """Load data list."""
        # get file backend
        img_prefix = self.data_prefix['img_path']
        file_backend = get_file_backend(img_prefix)

        anno_info = json.load(open(self.ann_file, 'r'))
        # mapping img_id to img filename
        img_dict = OrderedDict()
        for idx, img in enumerate(anno_info['images']):
            if img['id'] not in img_dict:
                img_rel_path = img['coco_url'].rsplit('/', 2)[-2:]
                img_path = file_backend.join_path(img_prefix, *img_rel_path)

                # create new idx for image
                img_dict[img['id']] = dict(
                    ori_id=img['id'],
                    image_id=idx,  # will be used for evaluation
                    img_path=img_path,
                    text=[],
                    gt_text_id=[],
                    gt_image_id=[],
                )

        train_list = []
        for idx, anno in enumerate(anno_info['annotations']):
            anno['text'] = anno.pop('caption')
            anno['ori_id'] = anno.pop('id')
            anno['text_id'] = idx  # will be used for evaluation
            # 1. prepare train data list item
            train_data = anno.copy()
            train_image = img_dict[train_data['image_id']]
            train_data['img_path'] = train_image['img_path']
            train_data['image_ori_id'] = train_image['ori_id']
            train_data['image_id'] = train_image['image_id']
            train_data['is_matched'] = True
            train_list.append(train_data)
            # 2. prepare eval data list item based on img dict
            img_dict[anno['image_id']]['gt_text_id'].append(anno['text_id'])
            img_dict[anno['image_id']]['text'].append(anno['text'])
            img_dict[anno['image_id']]['gt_image_id'].append(
                train_image['image_id'])

        self.img_size = len(img_dict)
        self.text_size = len(anno_info['annotations'])

        # return needed format data list
        if self.test_mode:
            return list(img_dict.values())
        return train_list

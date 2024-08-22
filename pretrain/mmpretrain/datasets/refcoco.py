# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os.path as osp
from typing import List

import mmengine
import numpy as np
from mmengine.dataset import BaseDataset
from pycocotools.coco import COCO

from mmpretrain.registry import DATASETS


@DATASETS.register_module()
class RefCOCO(BaseDataset):
    """RefCOCO dataset.

    RefCOCO is a popular dataset used for the task of visual grounding.
    Here are the steps for accessing and utilizing the
    RefCOCO dataset.

    You can access the RefCOCO dataset from the official source:
    https://github.com/lichengunc/refer

    The RefCOCO dataset is organized in a structured format: ::

        FeaturesDict({
            'coco_annotations': Sequence({
                'area': int64,
                'bbox': BBoxFeature(shape=(4,), dtype=float32),
                'id': int64,
                'label': int64,
            }),
            'image': Image(shape=(None, None, 3), dtype=uint8),
            'image/id': int64,
            'objects': Sequence({
                'area': int64,
                'bbox': BBoxFeature(shape=(4,), dtype=float32),
                'gt_box_index': int64,
                'id': int64,
                'label': int64,
                'refexp': Sequence({
                    'raw': Text(shape=(), dtype=string),
                    'refexp_id': int64,
                }),
            }),
        })

    Args:
        ann_file (str): Annotation file path.
        data_root (str): The root directory for ``data_prefix`` and
            ``ann_file``. Defaults to ''.
        data_prefix (str): Prefix for training data.
        pipeline (Sequence): Processing pipeline. Defaults to an empty tuple.
        **kwargs: Other keyword arguments in :class:`BaseDataset`.
    """

    def __init__(self,
                 data_root,
                 ann_file,
                 data_prefix,
                 split_file,
                 split='train',
                 **kwargs):
        self.split_file = split_file
        self.split = split

        super().__init__(
            data_root=data_root,
            data_prefix=dict(img_path=data_prefix),
            ann_file=ann_file,
            **kwargs,
        )

    def _join_prefix(self):
        if not mmengine.is_abs(self.split_file) and self.split_file:
            self.split_file = osp.join(self.data_root, self.split_file)

        return super()._join_prefix()

    def load_data_list(self) -> List[dict]:
        """Load data list."""
        with mmengine.get_local_path(self.ann_file) as ann_file:
            coco = COCO(ann_file)
        splits = mmengine.load(self.split_file, file_format='pkl')
        img_prefix = self.data_prefix['img_path']

        data_list = []
        join_path = mmengine.fileio.get_file_backend(img_prefix).join_path
        for refer in splits:
            if refer['split'] != self.split:
                continue

            ann = coco.anns[refer['ann_id']]
            img = coco.imgs[ann['image_id']]
            sentences = refer['sentences']
            bbox = np.array(ann['bbox'], dtype=np.float32)
            bbox[2:4] = bbox[0:2] + bbox[2:4]  # XYWH -> XYXY

            for sent in sentences:
                data_info = {
                    'img_path': join_path(img_prefix, img['file_name']),
                    'image_id': ann['image_id'],
                    'ann_id': ann['id'],
                    'text': sent['sent'],
                    'gt_bboxes': bbox[None, :],
                }
                data_list.append(data_info)

        if len(data_list) == 0:
            raise ValueError(f'No sample in split "{self.split}".')

        return data_list

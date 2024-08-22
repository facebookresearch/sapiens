# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import List

import mmengine
from mmengine.dataset import BaseDataset
from mmengine.fileio import get_file_backend
from pycocotools.coco import COCO

from mmpretrain.registry import DATASETS


@DATASETS.register_module()
class NoCaps(BaseDataset):
    """NoCaps dataset.

    Args:
        data_root (str): The root directory for ``data_prefix`` and
            ``ann_file``..
        ann_file (str): Annotation file path.
        data_prefix (dict): Prefix for data field. Defaults to
            ``dict(img_path='')``.
        pipeline (Sequence): Processing pipeline. Defaults to an empty tuple.
        **kwargs: Other keyword arguments in :class:`BaseDataset`.
    """

    def load_data_list(self) -> List[dict]:
        """Load data list."""
        img_prefix = self.data_prefix['img_path']
        with mmengine.get_local_path(self.ann_file) as ann_file:
            coco = COCO(ann_file)

        file_backend = get_file_backend(img_prefix)
        data_list = []
        for ann in coco.anns.values():
            image_id = ann['image_id']
            image_path = file_backend.join_path(
                img_prefix, coco.imgs[image_id]['file_name'])
            data_info = {
                'image_id': image_id,
                'img_path': image_path,
                'gt_caption': None
            }

            data_list.append(data_info)

        return data_list

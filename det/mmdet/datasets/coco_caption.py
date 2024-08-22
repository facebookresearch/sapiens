# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path
from typing import List

import mmengine
from mmengine.dataset import BaseDataset
from mmengine.fileio import get_file_backend

from mmdet.registry import DATASETS


@DATASETS.register_module()
class CocoCaptionDataset(BaseDataset):
    """COCO2014 Caption dataset."""

    def load_data_list(self) -> List[dict]:
        """Load data list."""
        img_prefix = self.data_prefix['img_path']
        annotations = mmengine.load(self.ann_file)
        file_backend = get_file_backend(img_prefix)

        data_list = []
        for ann in annotations:
            data_info = {
                'img_id': Path(ann['image']).stem.split('_')[-1],
                'img_path': file_backend.join_path(img_prefix, ann['image']),
                'gt_caption': ann['caption'],
            }

            data_list.append(data_info)

        return data_list

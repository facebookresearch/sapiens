# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
from typing import List

from mmengine.fileio import get_file_backend, list_from_file

from mmpretrain.registry import DATASETS
from .base_dataset import BaseDataset


@DATASETS.register_module()
class NLVR2(BaseDataset):
    """COCO Caption dataset."""

    def load_data_list(self) -> List[dict]:
        """Load data list."""

        data_list = []
        img_prefix = self.data_prefix['img_path']
        file_backend = get_file_backend(img_prefix)
        examples = list_from_file(self.ann_file)

        for example in examples:
            example = json.loads(example)
            prefix = example['identifier'].rsplit('-', 1)[0]
            train_data = {}
            train_data['text'] = example['sentence']
            train_data['gt_label'] = {'True': 1, 'False': 0}[example['label']]
            train_data['img_path'] = [
                file_backend.join_path(img_prefix, prefix + f'-img{i}.png')
                for i in range(2)
            ]

            data_list.append(train_data)

        return data_list

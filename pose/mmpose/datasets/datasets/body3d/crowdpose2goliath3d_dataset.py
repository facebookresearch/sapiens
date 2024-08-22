# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from mmpose.registry import DATASETS
from ..base import BaseCocoStyleDataset
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import copy
import os
import sys

##-----------------------------------------------------------------------------------------------------------
try:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../..')) \
        if os.path.join(os.path.dirname(__file__), '../../../..') not in sys.path else None

    from configs._base_.datasets.crowdpose import dataset_info as source_dataset_info
    from configs._base_.datasets.goliath3d import dataset_info as target_dataset_info

    SOURCE_KEYPOINT_ORDER = [v['name'] for k, v in sorted(source_dataset_info['keypoint_info'].items())]
    TARGET_KEYPOINT_ORDER = [v['name'] for k, v in sorted(target_dataset_info['keypoint_info'].items())]

    SOURCE_TO_TARGET_INDICES = {joint_index: None for joint_index in range(len(SOURCE_KEYPOINT_ORDER))}
    for joint_index in range(len(SOURCE_KEYPOINT_ORDER)):
        if SOURCE_KEYPOINT_ORDER[joint_index] in TARGET_KEYPOINT_ORDER:
            SOURCE_TO_TARGET_INDICES[joint_index] = TARGET_KEYPOINT_ORDER.index(SOURCE_KEYPOINT_ORDER[joint_index])
except Exception as e:
    pass

##-----------------------------------------------------------------------------------------------------------
@DATASETS.register_module()
class Crowdpose2Goliath3dDataset(BaseCocoStyleDataset):
    """
    """

    METAINFO: dict = dict(from_file='configs/_base_/datasets/crowdpose2goliath3d.py')

    def parse_data_info(self, raw_data_info: dict) -> Optional[dict]:
        """Parse raw COCO annotation of an instance.

        Args:
            raw_data_info (dict): Raw data information loaded from
                ``ann_file``. It should have following contents:

                - ``'raw_ann_info'``: Raw annotation of an instance
                - ``'raw_img_info'``: Raw information of the image that
                    contains the instance

        Returns:
            dict | None: Parsed instance annotation
        """

        ann = raw_data_info['raw_ann_info']
        img = raw_data_info['raw_img_info']

        # filter invalid instance
        if 'bbox' not in ann or 'keypoints' not in ann:
            return None

        img_w, img_h = img['width'], img['height']

        # get bbox in shape [1, 4], formatted as xywh
        x, y, w, h = ann['bbox']
        x1 = np.clip(x, 0, img_w - 1)
        y1 = np.clip(y, 0, img_h - 1)
        x2 = np.clip(x + w, 0, img_w - 1)
        y2 = np.clip(y + h, 0, img_h - 1)

        bbox = np.array([x1, y1, x2, y2], dtype=np.float32).reshape(1, 4)

        # keypoints in shape [1, K, 2] and keypoints_visible in [1, K]
        _keypoints = np.array(
            ann['keypoints'], dtype=np.float32).reshape(1, -1, 3)
        keypoints = _keypoints[..., :2]
        keypoints_visible = np.minimum(1, _keypoints[..., 2])

        raw_keypoints = keypoints.copy()
        raw_keypoints_visible = keypoints_visible.copy()

        keypoints = np.zeros((1, len(TARGET_KEYPOINT_ORDER), 2), dtype=np.float32)
        keypoints_visible = np.zeros((1, len(TARGET_KEYPOINT_ORDER)), dtype=np.float32)

        for source_joint_index in range(len(SOURCE_KEYPOINT_ORDER)):
            target_joint_index = SOURCE_TO_TARGET_INDICES[source_joint_index]

            if target_joint_index is not None:
                keypoints[0, target_joint_index, :2] = raw_keypoints[0, source_joint_index, :2]
                keypoints_visible[0, target_joint_index] = raw_keypoints_visible[0, source_joint_index]

        num_keypoints = np.count_nonzero(keypoints.max(axis=2))

        data_info = {
            'img_id': ann['image_id'],
            'img_path': img['img_path'],
            'bbox': bbox,
            'bbox_score': np.ones(1, dtype=np.float32),
            'num_keypoints': num_keypoints,
            'keypoints': keypoints,
            'keypoints_visible': keypoints_visible,
            'iscrowd': ann.get('iscrowd', 0),
            'segmentation': ann.get('segmentation', None),
            'id': ann['id'],
            'category_id': ann['category_id'],
            # store the raw annotation of the instance
            # it is useful for evaluation without providing ann_file
            'raw_ann_info': copy.deepcopy(ann),
        }

        if 'crowdIndex' in img:
            data_info['crowd_index'] = img['crowdIndex']

        return data_info

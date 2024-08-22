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

CROWDPOSE_KEYPOINT_ORDER = ['left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
                            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
                            'left_knee', 'right_knee', 'left_ankle', 'right_ankle',
                            'head', 'neck']

COCO_KEYPOINT_ORDER = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
                       'left_shoulder', 'right_shoulder', 'left_elbow',
                       'right_elbow', 'left_wrist', 'right_wrist', 'left_hip',
                       'right_hip', 'left_knee', 'right_knee', 'left_ankle',
                       'right_ankle']


CROWDPOSE_TO_COCO_INDICES = {joint_index: None for joint_index in range(len(CROWDPOSE_KEYPOINT_ORDER))}

for crowdpose_joint_index in range(len(CROWDPOSE_KEYPOINT_ORDER)):
    if CROWDPOSE_KEYPOINT_ORDER[crowdpose_joint_index] in COCO_KEYPOINT_ORDER:
        CROWDPOSE_TO_COCO_INDICES[crowdpose_joint_index] = COCO_KEYPOINT_ORDER.index(CROWDPOSE_KEYPOINT_ORDER[crowdpose_joint_index])


@DATASETS.register_module()
class Crowdpose2CocoDataset(BaseCocoStyleDataset):
    """
    """

    METAINFO: dict = dict(from_file='configs/_base_/datasets/crowdpose2coco.py')

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

        keypoints = np.zeros((1, len(COCO_KEYPOINT_ORDER), 2), dtype=np.float32)
        keypoints_visible = np.zeros((1, len(COCO_KEYPOINT_ORDER)), dtype=np.float32)

        for crowdpose_joint_index in range(len(CROWDPOSE_KEYPOINT_ORDER)):
            coco_joint_index = CROWDPOSE_TO_COCO_INDICES[crowdpose_joint_index]
            
            if coco_joint_index is not None:
                keypoints[0, coco_joint_index, :2] = raw_keypoints[0, crowdpose_joint_index, :2]
                keypoints_visible[0, coco_joint_index] = raw_keypoints_visible[0, crowdpose_joint_index]

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

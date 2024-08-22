# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import os.path as osp
from typing import Callable, List, Optional, Sequence, Tuple, Union

import copy
import numpy as np
from mmengine.fileio import exists, get_local_path
from scipy.io import loadmat

from mmpose.registry import DATASETS
from mmpose.structures.bbox import bbox_cs2xyxy
from ..base import BaseCocoStyleDataset
import os
import sys

##-----------------------------------------------------------------------------------------------------------
try:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../..')) \
        if os.path.join(os.path.dirname(__file__), '../../../..') not in sys.path else None

    from configs._base_.datasets.mpii import dataset_info as source_dataset_info
    from configs._base_.datasets.goliath import dataset_info as target_dataset_info

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
class Mpii2GoliathDataset(BaseCocoStyleDataset):
    """
    """

    METAINFO: dict = dict(from_file='configs/_base_/datasets/mpii2goliath.py')

    def _load_annotations(self) -> Tuple[List[dict], List[dict]]:
        """Load data from annotations in MPII format."""

        assert exists(self.ann_file), 'Annotation file does not exist'
        with get_local_path(self.ann_file) as local_path:
            with open(local_path) as anno_file:
                self.anns = json.load(anno_file)

        instance_list = []
        image_list = []
        used_img_ids = set()
        ann_id = 0

        # mpii bbox scales are normalized with factor 200.
        pixel_std = 200.

        for idx, ann in enumerate(self.anns):
            center = np.array(ann['center'], dtype=np.float32)
            scale = np.array([ann['scale'], ann['scale']],
                             dtype=np.float32) * pixel_std

            # Adjust center/scale slightly to avoid cropping limbs
            if center[0] != -1:
                center[1] = center[1] + 15. / pixel_std * scale[1]

            # MPII uses matlab format, index is 1-based,
            # we should first convert to 0-based index
            center = center - 1

            # unify shape with coco datasets
            center = center.reshape(1, -1)
            scale = scale.reshape(1, -1)
            bbox = bbox_cs2xyxy(center, scale)

            # load keypoints in shape [1, K, 2] and keypoints_visible in [1, K]
            keypoints = np.array(ann['joints']).reshape(1, -1, 2)
            keypoints_visible = np.array(ann['joints_vis']).reshape(1, -1)

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

            instance_info = {
                'id': ann_id,
                'img_id': int(ann['image'].split('.')[0]),
                'img_path': osp.join(self.data_prefix['img'], ann['image']),
                'bbox_center': center,
                'bbox_scale': scale,
                'bbox': bbox,
                'bbox_score': np.ones(1, dtype=np.float32),
                'num_keypoints': num_keypoints,
                'keypoints': keypoints,
                'keypoints_visible': keypoints_visible,
            }

            if instance_info['img_id'] not in used_img_ids:
                used_img_ids.add(instance_info['img_id'])
                image_list.append({
                    'img_id': instance_info['img_id'],
                    'img_path': instance_info['img_path'],
                })

            instance_list.append(instance_info)
            ann_id = ann_id + 1

        return instance_list, image_list

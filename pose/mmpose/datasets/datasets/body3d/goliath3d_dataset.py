# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import warnings
from mmpose.registry import DATASETS
from ..base import BaseCocoStyleDataset
import torch
import torch.utils.data
import torch.multiprocessing as mp
import numpy as np
import os
import cv2
import pickle
from PIL import ImageDraw
from tqdm import tqdm
import io
import json
import copy
import pandas as pd
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
import random
from matplotlib import pyplot as plt

import copy
import os.path as osp
from copy import deepcopy
from itertools import filterfalse, groupby
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from mmengine.dataset import BaseDataset, force_full_init
from mmengine.fileio import exists, get_local_path, load
from mmengine.utils import is_list_of
from xtcocotools.coco import COCO

from mmpose.registry import DATASETS
from mmpose.structures.bbox import bbox_xywh2xyxy
from ..utils import parse_pose_metainfo

from contextlib import redirect_stderr

with open(os.devnull, 'w') as f, redirect_stderr(f):
    try:
        from care.data.io import typed
        from care.strict.data.io.typed.file.calib import load_krt
    except Exception:
        # If the import fails, you can handle it here without printing any errors.
        pass

@DATASETS.register_module()
class Goliath3dDataset(BaseCocoStyleDataset):
    METAINFO: dict = dict(from_file='configs/_base_/datasets/goliath3d.py')

    def __init__(self,
                 ann_file: str = '',
                 bbox_file: Optional[str] = None,
                 data_mode: str = 'topdown',
                 metainfo: Optional[dict] = None,
                 data_root: Optional[str] = None,
                 data_root_3d: Optional[str] = None,
                 data_prefix: dict = dict(img=''),
                 filter_cfg: Optional[dict] = None,
                 indices: Optional[Union[int, Sequence[int]]] = None,
                 serialize_data: bool = True,
                 pipeline: List[Union[dict, Callable]] = [],
                 test_mode: bool = False,
                 lazy_init: bool = False,
                 max_refetch: int = 1000):
        
        self.pose3d_data_root = os.path.join(data_root_3d, '3D')
        self.camera_data_root = os.path.join(data_root_3d, 'KRTs')

        super().__init__(
            ann_file=ann_file,
            bbox_file=bbox_file,
            data_mode=data_mode,
            metainfo=metainfo,
            data_root=data_root,
            data_prefix=data_prefix,
            filter_cfg=filter_cfg,
            indices=indices,
            serialize_data=serialize_data,
            pipeline=pipeline,
            test_mode=test_mode,
            lazy_init=lazy_init,
            max_refetch=max_refetch)

        self.original_idxs = np.array(self.metainfo['idx_to_original_idx_mapping'].values()) ## indices into the original 344 keypoints
        self.max_num_keypoints = 344

        return

    def load_data_list(self) -> List[dict]:
        """Load data list from 344 body points."""

        self._register_airstore_handler()

        with open(self.ann_file, "rb") as f:
            raw = f.read()
        raw_data = json.loads(raw)  # samples=5,267,269

        data_list = []
        for sample in raw_data:
            dp = { "airstore_id": sample["sample_id"],
                "session_id": str(sample["session_id"]),
                "camera_id": str(sample["camera_id"]),
                "frame_id": str(sample["frame_number"]),
                }
            if sample.get("box-default") is not None:
                dp["box"] = sample["box-default"]
            data_list.append(dp)

        print('\033[92msorting by session, camera and frame numbers\033[0m')

        # Sort the data_list by session_id, then by camera_id, and finally by frame_number
        data_list = sorted(data_list, key=lambda y: (y['session_id'], y['camera_id'], y['frame_id']))

        print('\033[92mDone! Loaded total samples: {}\033[0m'.format(len(data_list)))

        return data_list

    def _register_airstore_handler(self) -> None:
        from care.strict.data.io.file_system.airstore_client import register_airstore_in_fsspec
        register_airstore_in_fsspec()
        self.path_template = "airstoreds://rlr_detection_services_ml_datasets_no_user_data"
        self.airstore = True

    def _read_from_airstore(self, asset: str, sid: str) -> io.BytesIO:
        with typed.open(self.path_template + f"/{asset}?sampleId={sid}").open() as f:
            data = io.BytesIO(f.read())
        return data

    def get_data_info(self, idx):
        if self.serialize_data:
            start_addr = 0 if idx == 0 else self.data_address[idx - 1].item()
            end_addr = self.data_address[idx].item()
            bytes = memoryview(
                self.data_bytes[start_addr:end_addr])  # type: ignore
            data_info = pickle.loads(bytes)  # type: ignore
        else:
            data_info = copy.deepcopy(self.data_list[idx])

        ##---------read 3d pose---------------
        pose3d_file_path = os.path.join(self.pose3d_data_root, '{}_{}.txt'.format(data_info['session_id'], data_info['frame_id']))
        session_camera_file = os.path.join(self.camera_data_root, '{}.krt'.format(data_info['session_id']))

        if not exists(pose3d_file_path) or not exists(session_camera_file):
            return None

        pose3d = np.zeros((self.max_num_keypoints, 4), dtype=np.float32)
        df = pd.read_csv(pose3d_file_path, delimiter=' ', header=None, names=['point_id', 'x', 'y', 'z', 'sum_conf', 'num_inlier'])
        df = df[(df['point_id'] >= 0) & (df['point_id'] < self.max_num_keypoints)]
        pose3d[df['point_id'].values.astype(int), :3] = df[['x', 'y', 'z']].values
        pose3d[df['point_id'].values.astype(int), 3] = 1  # Set visibility to 1. N x 4

        try:
            session_krt = load_krt(session_camera_file)
            camera_krt = session_krt[data_info['camera_id']]
            extrinsics = camera_krt['extrin'] ## 3 x 4
            K = camera_krt['intrin'] ## 3 x 3
        except:
            return None
        
        extrinsics_4x4 = np.concatenate((extrinsics, np.array([[0, 0, 0, 1]])), axis=0) # 4 x 4, extrinsics
        pose3d_homogeneous = np.hstack((pose3d[:, :3], np.ones((pose3d.shape[0], 1), dtype=np.float32)))
        pose3d_cam_homogeneous = pose3d_homogeneous @ extrinsics_4x4.T # N x 4
        pose3d_cam = np.hstack((pose3d_cam_homogeneous[:, :3], pose3d[:, 3:4])) # N x 4
        pose3d_cam[pose3d[:, 3] == 0, :] = 0

        ## project 3d pose to image plane
        pose2d_homogeneous = pose3d_cam[:, :3] @ K.T
        valid_rows = pose2d_homogeneous[:, 2] != 0
        pose2d = np.zeros((pose3d_cam.shape[0], 4), dtype=np.float32)
        pose2d[valid_rows, :2] = pose2d_homogeneous[valid_rows, :2] / pose2d_homogeneous[valid_rows, 2:3]
        pose2d[valid_rows, 2] = pose3d_cam[valid_rows, 2] ## depth
        pose2d[valid_rows, 3] = 1 ## 344 x 4: x, y, z, visibility
        pose2d = pose2d.reshape(1, -1, 4) # 1 x N x 4

        keypoints_depth = pose2d[:, :, 2:3] / 1000.0 # 1 x N x 1, millimetre to metre

        ##------------read image-------------------
        try:
            img = Image.open(self._read_from_airstore("image", data_info['airstore_id'])) ## pillow image
        except:
            return None
        
        img = np.array(img) ## RGB image
        img = img[:, :, ::-1]  # Convert RGB to BGR, the model preprocessor will convert this to rgb again

        img_w, img_h = img.shape[1], img.shape[0]

        # process keypoints
        try:
            keypoints_np = np.load(self._read_from_airstore("keypoint", data_info['airstore_id']))  # shape 3 x 344
        except:
            return None
        
        keypoints = keypoints_np[:2].T.reshape(1, -1, 2)  # shape 1 x 344 x 2
        keypoints_visible = np.where(keypoints_np[2].T > 0, 1, 0).reshape(1, -1)  # shape 1 x 344

        # Identify keypoints that are out of bounds for x (width) and y (height)
        out_of_bounds_w = np.logical_or(keypoints[0, :, 0] <= 0, keypoints[0, :, 0] >= img_w)
        out_of_bounds_h = np.logical_or(keypoints[0, :, 1] <= 0, keypoints[0, :, 1] >= img_h)

        # Update keypoints_visible based on the out-of-bounds keypoints
        keypoints_visible[0, out_of_bounds_w | out_of_bounds_h] = 0

        ## remove keypoints from the original set to reduced set
        if self.original_idxs is not None:
            keypoints = keypoints[:, self.original_idxs, :] # 1 x N x 2
            keypoints_visible = keypoints_visible[:, self.original_idxs] # 1 x N
            pose2d = pose2d[:, self.original_idxs, :] # 1 x N x 4
            keypoints_depth = keypoints_depth[:, self.original_idxs, :] # 1 x N
            pose3d_cam = pose3d_cam[self.original_idxs, :] # N x 4

        # ## draw the pose2d on the image
        # import ipdb; ipdb.set_trace()
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # convert bgr to rgb image
        # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) # convert rgb to bgr image

        # img2 = img.copy()

        # for i in range(pose2d.shape[1]):
        #     if keypoints_visible[0, i] == 1:
        #         print('kps_id:{}, u:{}, v:{}, X:{}, Y:{}, Z:{}'.format(i, pose2d[0][i][0], pose2d[0][i][1], pose3d_cam[i][0], pose3d_cam[i][1], pose3d_cam[i][2])) 
        #         cv2.circle(img, (int(pose2d[0][i][0]), int(pose2d[0][i][1])), 8, color=(0, 255, 0), thickness=-1)
        # cv2.imwrite('temp.jpg', img)

        # Z = keypoints_depth[0, :, 0] ## 308 
        # fx, fy = K[0, 0], K[1, 1]
        # cx, cy = K[0, 2], K[1, 2]

        # X = (keypoints[0, :, 0] - cx) * Z / fx
        # Y = (keypoints[0, :, 1] - cy) * Z / fy

        # # Stack X, Y, Z to create pose3d
        # pose3d_new = np.stack([X, Y, Z], axis=-1) # N x 3
        # pose2d_new = pose3d_new @ K.T  ## project 3d keypoints to 2D
        # pose2d_new = pose2d_new[:, :2] / (pose2d_new[:, 2:] + 1e-8) 

        # print()
        # for i in range(pose2d_new.shape[0]):
        #     if keypoints_visible[0, i] == 1:
        #         print('kps_id:{}, u:{}, v:{}, X:{}, Y:{}, Z:{}'.format(i, pose2d_new[i][0], pose2d_new[i][1], pose3d_new[i, 0], pose3d_new[i, 1], pose3d_new[i, 2]))
        #         cv2.circle(img2, (int(pose2d_new[i][0]), int(pose2d_new[i][1])), 8, color=(0, 255, 0), thickness=-1)

        # cv2.imwrite('temp_new.jpg', img2)
        # import ipdb; ipdb.set_trace()

        # Default bounding box to the full image size
        bbox = np.array([0, 0, img_w, img_h], dtype=np.float32).reshape(1, 4)

        if np.any(keypoints_visible):  # If any keypoints are visible
            visible_keypoints = keypoints[0][keypoints_visible[0] == 1]  # Filter out the invisible keypoints

            # Get the bounding box encompassing the keypoints
            x_min, y_min = np.clip(np.min(visible_keypoints, axis=0), [0, 0], [img_w, img_h])
            x_max, y_max = np.clip(np.max(visible_keypoints, axis=0), [0, 0], [img_w, img_h])

            bbox = np.array([x_min, y_min, x_max, y_max], dtype=np.float32).reshape(1, 4)

        num_keypoints = np.count_nonzero(keypoints_visible)

        ## atleast 8 vis keypoints
        if num_keypoints < self.metainfo['min_visible_keypoints']:
            random_idx = np.random.randint(0, len(self.data_list))
            return self.get_data_info(random_idx)

        ## ignore greyscale images for training
        B, G, R = cv2.split(img)
        if np.array_equal(B, G) and np.array_equal(B, R):
            random_idx = np.random.randint(0, len(self.data_list))
            return self.get_data_info(random_idx)

        data_info = {
            'img': img,
            'img_id': '',
            'img_path': '',
            'session_id': data_info['session_id'],
            'camera_id': data_info['camera_id'],
            'frame_id': data_info['frame_id'],
            'airstore_id': data_info['airstore_id'],
            'bbox': bbox,
            'bbox_score': np.ones(1, dtype=np.float32),
            'num_keypoints': num_keypoints,
            'keypoints': keypoints,
            'keypoints_visible': keypoints_visible,
            'keypoints_depth': keypoints_depth,
            'K': K, # 3 x 3
            'iscrowd': 0,
            'segmentation': None,
            'id': idx,
            'category_id': 1,
        }

        # Some codebase needs `sample_idx` of data information. Here we convert
        # the idx to a positive number and save it in data information.
        if idx >= 0:
            data_info['sample_idx'] = idx
        else:
            data_info['sample_idx'] = len(self) + idx

        # Add metainfo items that are required in the pipeline and the model
        metainfo_keys = [
            'upper_body_ids', 'lower_body_ids', 'flip_pairs',
            'dataset_keypoint_weights', 'flip_indices', 'skeleton_links'
        ]

        for key in metainfo_keys:
            assert key not in data_info, (
                f'"{key}" is a reserved key for `metainfo`, but already '
                'exists in the `data_info`.')

            data_info[key] = deepcopy(self._metainfo[key])

        return data_info

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset

import numpy as np
import os
import cv2
import pickle
from PIL import ImageDraw
from tqdm import tqdm
import io
import json
import copy
import os.path as osp
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
import random
from matplotlib import pyplot as plt
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import mmengine.fileio as fileio

##-----------------------------------------------------------------------
@DATASETS.register_module()
class StereoCorrespondencesRenderPeopleDataset(BaseSegDataset):
    """
    """

    def __init__(self,
                 **kwargs) -> None:

        super().__init__(**kwargs)
        return

    def load_data_list(self) -> List[dict]:
        """Load annotation from directory or annotation file.

        Returns:
            list[dict]: All data info of dataset.
        """
        data_list = []

        self.pointmap_dir = os.path.join(self.data_root, 'pointmap')
        self.rgb_dir = os.path.join(self.data_root, 'rgb')
        self.mask_dir = os.path.join(self.data_root, 'mask')
        self.depth_dir = os.path.join(self.data_root, 'depth')
        self.K_dir = os.path.join(self.data_root, 'camera_intrinsics')
        self.M_dir = os.path.join(self.data_root, 'camera_extrinsics') ## cv camera extrinsics

        print('\033[92mLoading StereoCorrespondencesRenderPeople!\033[0m')

        mesh_names = [x for x in os.listdir(self.pointmap_dir) if x.endswith('.txt')]

        self.mesh_names_to_samples = {}

        sample_count = 0
        for mesh_name in mesh_names:
            mesh_path = os.path.join(self.pointmap_dir, mesh_name)
            with open(mesh_path, 'r') as f:
                sample_names = f.readlines()
                sample_names = [x.strip() for x in sample_names] ## these are the sample_names with the current mesh_name

            for sample_name in sample_names:
                if mesh_name not in self.mesh_names_to_samples:
                    self.mesh_names_to_samples[mesh_name] = []

                self.mesh_names_to_samples[mesh_name].append(sample_count) ## at index sample_count in the data_list

                sample_info = {
                    'rgb_path': os.path.join(self.rgb_dir, sample_name + '.png'),
                    'mask_path': os.path.join(self.mask_dir, sample_name + '.png'),
                    'depth_path': os.path.join(self.depth_dir, sample_name + '.npy'),
                    'K_path': os.path.join(self.K_dir, sample_name + '.txt'),
                    'M_path': os.path.join(self.M_dir, sample_name + '.txt'),
                    'mesh_name': mesh_name,
                }

                data_list.append(sample_info)
                sample_count += 1

        print('\033[92mDone! StereoCorrespondencesRenderPeople. Loaded total samples: {}\033[0m'.format(len(data_list)))
        return data_list

    def get_data_info(self, idx):
        data_idx_info = copy.deepcopy(self.data_list[idx])

        ## sample other mesh
        mesh_name = data_idx_info['mesh_name']
        other_idx = random.choice([i for i in self.mesh_names_to_samples[mesh_name] if i != idx])
        other_data_idx_info = copy.deepcopy(self.data_list[other_idx])

        data_info = self.get_data_info_helper(data_idx_info)
        other_data_info = self.get_data_info_helper(other_data_idx_info)

        data_info['idx'] = idx
        other_data_info['idx'] = other_idx

        return data_info, other_data_info


    def get_data_info_helper(self, data_info):
        img = cv2.imread(data_info['rgb_path']) ## bgr image is default
        mask = cv2.imread(data_info['mask_path'])
        mask = mask[:, :, 0] ##

        depth = np.load(data_info['depth_path']) ## H x W, ## is not in 0 to 1
        K = np.loadtxt(data_info['K_path']) ## intrinsics, 3 x 3
        M = np.loadtxt(data_info['M_path']) ## extrinsics, 4 x 4

        if img is None or mask is None or depth is None:
            return None

        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)

        # Find the bounding box's bounds
        y1, y2 = np.where(rows)[0][[0, -1]]
        x1, x2 = np.where(cols)[0][[0, -1]]

        bbox = np.array([x1, y1, x2, y2], dtype=np.float32).reshape(1, 4)

        data_info = {
            'img': img,
            'img_id': os.path.basename(data_info['rgb_path']),
            'img_path': data_info['rgb_path'],
            'gt_depth': depth,
            'K': K,
            'M': M,
            'mask': mask,
            'bbox': bbox,
            'bbox_score': np.ones(1, dtype=np.float32),
        }

        return data_info

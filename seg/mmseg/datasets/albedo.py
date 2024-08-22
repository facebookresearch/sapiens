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
from multiprocessing import Pool
import gzip

##-----------------------------------------------------------------------
def process_camera_dir(args):
    subject_dir, camera_name = args
    camera_dir = os.path.join(subject_dir, camera_name)
    frame_names = sorted([x.replace('_gt.png', '') for x in os.listdir(camera_dir) if x.endswith('_gt.png')])

    data_list = []

    for frame_name in frame_names:
        mask_path = os.path.join(camera_dir, frame_name + '_alpha.png')
        real_image_path = os.path.join(camera_dir, frame_name + '_gt.png')
        synthetic_image_path = os.path.join(camera_dir, frame_name + '_render.png')
        albedo_path = os.path.join(camera_dir, frame_name + '_image_albedo.png')

        if not os.path.exists(mask_path) or \
           not os.path.exists(real_image_path) or \
           not os.path.exists(synthetic_image_path) or \
           not os.path.exists(albedo_path):
            continue

        data_list.append({
            'camera_dir': camera_dir,
            'frame_name': frame_name
        })

    return data_list


@DATASETS.register_module()
class AlbedoDataset(BaseSegDataset):
    def __init__(self,
                 **kwargs) -> None:

        super().__init__(**kwargs)
        return

    def load_data_list(self, use_cache=True) -> List[dict]:
        """Load annotation from directory or annotation file.

        Returns:
            list[dict]: All data info of dataset.
        """
        data_list = []

        ## real_data_list and synthetic_data_list cache jsons
        cache_data_list_path = os.path.join(self.data_root, 'data_list.json')

        print('\033[92mLoading AlbedoDataset! {}\033[0m'.format(self.data_root))

        if use_cache == True:
            if os.path.exists(cache_data_list_path):
                print('\033[92mReading from cache!\033[0m')
                with gzip.open(cache_data_list_path, 'rt') as f:
                    data_list = json.load(f)
                cache_exists = True
                print('\033[92mCache read done!\033[0m')
            else:
                cache_exists = False

        if use_cache == False or cache_exists == False:
            print('\033[92mCreating indexes!\033[0m')
            subjects = [(os.path.join(self.data_root, subject_name, 'rendered_images', 'envspin'), camera_name)
                    for subject_name in sorted(os.listdir(self.data_root))
                    if os.path.isdir(os.path.join(self.data_root, subject_name))
                    for camera_name in sorted([x for x in os.listdir(os.path.join(self.data_root, subject_name, 'rendered_images', 'envspin'))
                            if x.startswith('cam') and os.path.isdir(os.path.join(self.data_root, subject_name, 'rendered_images', 'envspin', x))])
                    if os.path.exists(os.path.join(self.data_root, subject_name, 'rendered_images', 'envspin'))]

            with Pool() as pool:
                results = list(tqdm(pool.imap(process_camera_dir, subjects), total=len(subjects)))

            for data_info in results:
                data_list.extend(data_info)

        if use_cache == True and cache_exists == False:
            print('\033[92mCaching AlbedoDataset!\033[0m')
            with gzip.open(cache_data_list_path, 'wt', compresslevel=5) as f:
                json.dump(data_list, f)
            print('\033[92mCache write done!\033[0m')

        ##---------inflate the data_list to real and synthetic data_list--------
        real_data_list = [
            {
                'rgb_path': os.path.join(data_info['camera_dir'], data_info['frame_name'] + '_gt.png'),
                'mask_path': os.path.join(data_info['camera_dir'], data_info['frame_name'] + '_alpha.png'),
                'albedo_path': os.path.join(data_info['camera_dir'], data_info['frame_name'] + '_image_albedo.png'),
            }
            for data_info in data_list
        ]

        synthetic_data_list = [
            {
                'rgb_path': os.path.join(data_info['camera_dir'], data_info['frame_name'] + '_render.png'),
                'mask_path': os.path.join(data_info['camera_dir'], data_info['frame_name'] + '_alpha.png'),
                'albedo_path': os.path.join(data_info['camera_dir'], data_info['frame_name'] + '_image_albedo.png'),
            }
            for data_info in data_list
        ]

        data_list = real_data_list + synthetic_data_list

        print('\033[92mDone! AlbedoDataset. Loaded total samples: {}. Real:{} Syn:{}\033[0m'.format(\
            len(data_list), len(real_data_list), len(synthetic_data_list)))
        return data_list

    def get_data_info(self, idx):
        if self.serialize_data:
            start_addr = 0 if idx == 0 else self.data_address[idx - 1].item()
            end_addr = self.data_address[idx].item()
            bytes = memoryview(
                self.data_bytes[start_addr:end_addr])  # type: ignore
            data_info = pickle.loads(bytes)  # type: ignore
        else:
            data_info = copy.deepcopy(self.data_list[idx])

        img = cv2.imread(data_info['rgb_path']) ## bgr image is default. (2048, 1334, 3)
        albedo = cv2.imread(data_info['albedo_path']) # (2048, 1334, 3)
        mask = cv2.imread(data_info['mask_path']) # (2048, 1334, 3)
        mask = mask[:, :, 0] ## 2048 x 1334. in 0 to 255

        opacity = mask / 255.0
        mask = (opacity > 0.95) * 255

        ## convert albedo, bgr to rgb
        albedo = cv2.cvtColor(albedo, cv2.COLOR_BGR2RGB)

        # Normalize albedo to the range 0-1
        albedo = albedo.astype(float) / 255.0

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
            'gt_albedo': albedo, ## rgb format
            'mask': mask,
            'id': idx,
            'bbox': bbox,
            'bbox_score': np.ones(1, dtype=np.float32),
        }

        return data_info

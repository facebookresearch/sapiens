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
class HDRIDataset(BaseSegDataset):
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

        ## data_list cache jsons
        cache_data_list_path = os.path.join(self.data_root, 'data_list_hdri.json')

        print('\033[92mLoading HDRIDataset! {}\033[0m'.format(self.data_root))

        if use_cache == True:
            if os.path.exists(cache_data_list_path):
                with open(cache_data_list_path, 'rb') as f:
                    data_list = json.load(f)
                cache_exists = True
            else:
                cache_exists = False

        if use_cache == False or cache_exists == False:
            print('\033[92mCreating indexes!\033[0m')
            for subject_name in sorted(os.listdir(self.data_root)):
                subject_dir = os.path.join(self.data_root, subject_name, 'rendered_images', 'envspin')

                ## check if the directory exists
                if not os.path.exists(subject_dir):
                    continue

                camera_names = sorted([x for x in os.listdir(subject_dir) if x.startswith('cam')])

                for camera_name in camera_names:
                    camera_dir = os.path.join(subject_dir, camera_name)
                    frame_names = sorted([x.replace('_gt.png', '') for x in os.listdir(camera_dir) if x.endswith('_gt.png')])

                    for frame_name in frame_names:
                        image_path = os.path.join(camera_dir, frame_name + '_render.png')
                        hdri_path = os.path.join(camera_dir, frame_name + '_headrel_light_intensity.npy')

                        if not os.path.exists(image_path) or \
                            not os.path.exists(hdri_path):
                            continue

                        data_list.append({
                            'rgb_path': image_path,
                            'hdri_path': hdri_path,
                            'camera_name': camera_name,
                            'frame_name': frame_name
                        })

        if use_cache == True and cache_exists == False:
            print('\033[92mCaching HDRIDataset!\033[0m')
            with open(cache_data_list_path, 'w') as f:
                json.dump(data_list, f)

        print('\033[92mDone! HDRIDataset. Loaded total samples: {}.\033[0m'.format(len(data_list)))
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
        hdri = np.load(data_info['hdri_path']) # 16 x 32 x 3. rgb

        data_info = {
            'img': img,
            'img_id': os.path.basename(data_info['rgb_path']),
            'img_path': data_info['rgb_path'],
            'gt_hdri': hdri, ## rgb format
            'id': idx,
        }

        return data_info

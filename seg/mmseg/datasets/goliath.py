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

from contextlib import redirect_stderr

with open(os.devnull, 'w') as f, redirect_stderr(f):
    try:
        from care.data.io import typed
    except Exception:
        # If the import fails, you can handle it here without printing any errors.
        pass

try:
    from care.strict.data.io.file_system.airstore_client import register_airstore_in_fsspec
    register_airstore_in_fsspec()
except:
    print('Warning: cannot import airstore!')

## 34 classes in total
ORIGINAL_GOLIATH_CLASSES=(
            "Background",
            "Apparel", "Chair", "Eyeglass_Frame", "Eyeglass_Lenses",
            "Face_Neck", "Hair", "Headset", "Left_Foot", "Left_Hand",
            "Left_Lower_Arm", "Left_Lower_Leg", "Left_Shoe", "Left_Sock",
            "Left_Upper_Arm", "Left_Upper_Leg", "Lower_Clothing",
            "Lower_Spandex", "Right_Foot", "Right_Hand", "Right_Lower_Arm",
            "Right_Lower_Leg", "Right_Shoe", "Right_Sock", "Right_Upper_Arm",
            "Right_Upper_Leg", "Torso", "Upper_Clothing", "Visible_Badge",
            "Lower_Lip", "Upper_Lip", "Lower_Teeth", "Upper_Teeth", "Tongue"
        )

ORIGINAL_GOLIATH_PALETTE=[
            [50, 50, 50],
            [255, 218, 0], [102, 204, 0], [14, 0, 204], [0, 204, 160],
            [128, 200, 255], [255, 0, 109], [0, 255, 36], [189, 0, 204],
            [255, 0, 218], [0, 160, 204], [0, 255, 145], [204, 0, 131],
            [182, 0, 255], [255, 109, 0], [0, 255, 255], [72, 0, 255],
            [204, 43, 0], [204, 131, 0], [255, 0, 0], [72, 255, 0],
            [189, 204, 0], [182, 255, 0], [102, 0, 204], [32, 72, 204],
            [0, 145, 255], [14, 204, 0], [0, 128, 72], [204, 0, 43],
            [235, 205, 119], [115, 227, 112], [157, 113, 143], [132, 93, 50],
            [82, 21, 114]
        ]

## 6 classes to remove
REMOVE_CLASSES=("Eyeglass_Frame", "Eyeglass_Lenses", "Visible_Badge", "Chair", "Lower_Spandex", "Headset")

# REMOVE_CLASSES=()

## 34 - 6 = 28 classes left
GOLIATH_CLASSES = tuple([x for x in ORIGINAL_GOLIATH_CLASSES if x not in REMOVE_CLASSES])
GOLIATH_PALETTE = [ORIGINAL_GOLIATH_PALETTE[idx] for idx in range(len(ORIGINAL_GOLIATH_CLASSES)) \
                        if ORIGINAL_GOLIATH_CLASSES[idx] not in REMOVE_CLASSES]

## source to target mapping
SOURCE_CLASSES = ORIGINAL_GOLIATH_CLASSES
TARGET_CLASSES = GOLIATH_CLASSES

SOURCE_TO_TARGET_MAPPING = {src_class: TARGET_CLASSES.index(src_class) if src_class in TARGET_CLASSES else 255 for src_class in SOURCE_CLASSES}
SOURCE_TO_TARGET_INDEX_MAPPING = {i: TARGET_CLASSES.index(SOURCE_CLASSES[i]) if SOURCE_CLASSES[i] in TARGET_CLASSES else 255 for i in range(len(SOURCE_CLASSES))}

##-----------------------------------------------------------------------
@DATASETS.register_module()
class GoliathDataset(BaseSegDataset):
    """
    """
    METAINFO = dict(
        classes=GOLIATH_CLASSES,
        palette=GOLIATH_PALETTE
    )

    def __init__(self,
                 **kwargs) -> None:
        self.path_template = "airstoreds://rlr_detection_services_ml_datasets_no_user_data"
        self.airstore = True

        super().__init__(**kwargs)
        return

    def _read_from_airstore(self, asset: str, sid: str) -> io.BytesIO:
        with typed.open(self.path_template + f"/{asset}?sampleId={sid}").open() as f:
            data = io.BytesIO(f.read())
        return data

    def load_data_list(self) -> List[dict]:
        """Load annotation from directory or annotation file.

        Returns:
            list[dict]: All data info of dataset.
        """
        data_list = []

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

        print('\033[92mDone! Goliath. Loaded total samples: {}\033[0m'.format(len(data_list))) ## 98424 images train
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

        try:
            img = Image.open(self._read_from_airstore("image", data_info['airstore_id'])) ## pillow image
            segmentation = Image.open(self._read_from_airstore("segmentation", data_info['airstore_id']))

        except Exception as e:
            print(f"Error loading image/seg {data_info['airstore_id']}. Retrying!")
            return None

        img = np.array(img) ## rgb image
        img = img[:, :, ::-1] # Important: Convert RGB to BGR, the pretrained model preprocessor will convert this to rgb again

        segmentation = np.array(segmentation)

        ##------remove the extra classes---
        segmentation = np.vectorize(lambda x: SOURCE_TO_TARGET_INDEX_MAPPING.get(x, 255))(segmentation)

        ## get bbox
        mask = (segmentation > 0).astype('uint8') ## 2D binary mask
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)

        # Find the bounding box's bounds
        y1, y2 = np.where(rows)[0][[0, -1]]
        x1, x2 = np.where(cols)[0][[0, -1]]

        bbox = np.array([x1, y1, x2, y2], dtype=np.float32).reshape(1, 4)

        data_info = {
            'img': img,
            'img_id': '',
            'img_path': '',
            'session_id': data_info['session_id'],
            'camera_id': data_info['camera_id'],
            'frame_id': data_info['frame_id'],
            'airstore_id': data_info['airstore_id'],
            'gt_seg_map': segmentation,
            'id': idx,
            'bbox': bbox,
            'bbox_score': np.ones(1, dtype=np.float32),
            'label_map': self.label_map,
            'reduce_zero_label': self.reduce_zero_label,
            'seg_fields': []
        }

        return data_info

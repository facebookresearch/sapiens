# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import os
import cv2
import pickle
from PIL import Image
import numpy as np
import copy

CLASSES = [
    "Background",
    "Skin",
    "Nose",
    "Right_eye",
    "Left_eye",
    "Right_brow",
    "Left_brow",
    "Right_ear",
    "Left_ear",
    "Mouth_interior",
    "Top_lip",
    "Bottom_lip",
    "Neck",
    "Hair",
    "Beard",
    "Clothing",
    "Glasses",
    "Headwear",
    "Facewear",
]

PALETTE = [
    [0, 0, 0],          # Background - Black
    [128, 200, 255],    # Skin - Light Blue
    [255, 200, 150],    # Nose - Light Orange
    [0, 255, 127],      # Right Eye - Spring Green
    [255, 99, 71],      # Left Eye - Tomato Red
    [30, 144, 255],     # Right Brow - Dodger Blue
    [255, 140, 0],      # Left Brow - Dark Orange
    [238, 130, 238],    # Right Ear - Violet
    [255, 215, 0],      # Left Ear - Gold
    [82, 21, 114],      # Mouth Interior - Dark Purple
    [115, 227, 112],    # Top Lip - Light Green
    [235, 205, 119],    # Bottom Lip - Beige
    [255, 182, 193],    # Neck - Light Pink
    [255, 0, 109],      # Hair - Bright Pink
    [169, 169, 169],    # Beard - Dark Gray
    [0, 128, 72],       # Clothing - Forest Green
    [0, 70, 130],       # Glasses - Deep Blue
    [255, 215, 0],      # Headwear - Gold (repeated color for consistency)
    [75, 0, 130],       # Facewear - Indigo
]


##-------------------------------------------------------------------------
@DATASETS.register_module()
class FaceDataset(BaseSegDataset):
    """LIP dataset.

    The ``img_suffix`` is fixed to '.jpg' and ``seg_map_suffix`` is fixed to
    '.png'.
    """
    METAINFO = dict(
        classes=CLASSES,
        palette=PALETTE,
        )

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
        image_names = [name for name in sorted(os.listdir(self.data_root)) if name.endswith('.png') and not name.endswith('_seg.png')]

        for image_name in image_names:
            sample = {
                    'image_path': os.path.join(self.data_root, image_name),
                    'segmentation_path': os.path.join(self.data_root, image_name.replace('.png', '_seg.png')),
            }

            data_list.append(sample)

        print('\033[92mDone! SegFace. Loaded total samples: {}\033[0m'.format(len(data_list)))
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

        img = Image.open(data_info['image_path']) ## pillow image
        img = np.array(img) ## rgb image

        ## if image is grayscale, convert it to RGB
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        img = img[:, :, ::-1] # Important: Convert RGB to BGR, the pretrained model preprocessor will convert this to rgb again

        segmentation = Image.open(data_info['segmentation_path'])
        segmentation = np.array(segmentation)

        ##----------------------------------
        data_info = {
            'img': img,
            'img_id': '',
            'img_path': data_info['image_path'],
            'gt_seg_map': segmentation,
            'id': idx,
            'label_map': self.label_map,
            'reduce_zero_label': self.reduce_zero_label,
            'seg_fields': []
        }

        return data_info

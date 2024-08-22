# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
from .goliath import GoliathDataset
import os
import cv2
import pickle
from PIL import Image
import numpy as np
import copy
import scipy

# PASCAL dataset original
ORIGINAL_SOURCE_CLASSES=('Background', 'Head', 'Left_eye', 'Right_eye',
                 'Left_ear', 'Right_ear', 'Left_eye_brow', 'Right_eye_brow',
                 'Nose', 'Mouth', 'Hair', 'Torso', 'Neck',
                 'Left_lower_arm', 'Left_upper_arm', 'Left_hand',
                 'Right_lower_arm', 'Right_upper_arm', 'Right_hand',
                 'Left_lower_leg', 'Left_upper_leg', 'Left_foot',
                 'Right_lower_leg', 'Right_upper_leg', 'Right_foot',
                 )

#-----------------------------------------------------------------------------------------------
## the feets are approximated as shoes
SOURCE_CLASSES=('Background', 'Face_Neck', 'Face_Neck', 'Face_Neck',
                 'Face_Neck', 'Face_Neck', 'Face_Neck', 'Face_Neck',
                 'Face_Neck', 'Mouth0', 'Hair', 'Upper_Clothing', 'Face_Neck',
                 'Left_Lower_Arm0', 'Left_Upper_Arm0', 'Left_Hand',
                 'Right_Lower_Arm0', 'Right_Upper_Arm0', 'Right_Hand',
                 'Left_Lower_Leg0', 'Left_Upper_Leg0', 'Left_Shoe',
                 'Right_Lower_Leg0', 'Right_Upper_Leg0', 'Right_Shoe',
                 )

TARGET_CLASSES=GoliathDataset.METAINFO['classes']
TARGET_PALETTE=GoliathDataset.METAINFO['palette']

TARGET_PALETTE = TARGET_PALETTE[:len(TARGET_CLASSES)]

#-----------------------------------------------------------------------------------------------
SOURCE_TO_TARGET_MAPPING = {src_class: TARGET_CLASSES.index(src_class) if src_class in TARGET_CLASSES else 255 for src_class in SOURCE_CLASSES}
SOURCE_TO_TARGET_INDEX_MAPPING = {i: TARGET_CLASSES.index(SOURCE_CLASSES[i]) if SOURCE_CLASSES[i] in TARGET_CLASSES else 255 for i in range(len(SOURCE_CLASSES))}

#-----------------------------------------------------------------------------------------------
COLORS = {
    1: [255, 0, 0],     # head - red
    2: [0, 255, 0],     # left eye - green
    3: [0, 0, 255],     # right eye - blue
    4: [255, 255, 0],   # left ear - yellow
    5: [255, 0, 255],   # right ear - magenta
    6: [0, 255, 255],   # left eyebrow - cyan
    7: [128, 0, 0],     # right eyebrow - dark red
    8: [0, 128, 0],     # nose - dark green
    9: [0, 0, 128],     # mouth - dark blue
    10: [128, 128, 0],  # hair - olive
    11: [128, 0, 128],  # torso - purple
    12: [0, 128, 128],  # neck - teal
    13: [255, 128, 0],  # left lower arm - orange
    14: [255, 0, 128],  # left upper arm - pink
    15: [128, 255, 0],  # left hand - lime
    16: [0, 128, 255],  # right lower arm - sky blue
    17: [128, 255, 255],# right upper arm - light cyan
    18: [255, 128, 128],# right hand - light red
    19: [128, 128, 255],# left lower leg - light blue
    20: [128, 255, 128],# left upper leg - light green
    21: [255, 255, 128],# left foot - light yellow
    22: [70, 130, 180], # right lower leg - steel blue
    23: [218, 165, 32], # right upper leg - golden rod
    24: [255, 69, 0]    # right foot - orange red
}

def apply_color_palette(segmentation):
    """Apply the color palette to the segmentation."""
    colored_segmentation = np.zeros((segmentation.shape[0], segmentation.shape[1], 3), dtype=np.uint8)
    for label, color in COLORS.items():
        colored_segmentation[segmentation == label] = color
    return colored_segmentation

##-------------------------------------------------------------------------
@DATASETS.register_module()
class Pascal2GoliathDataset(BaseSegDataset):
    """LIP dataset.

    The ``img_suffix`` is fixed to '.jpg' and ``seg_map_suffix`` is fixed to
    '.png'.
    """
    METAINFO = dict(
        classes=TARGET_CLASSES,
        palette=TARGET_PALETTE,
        )

    def __init__(self,
                 img_suffix='.jpg',
                 seg_map_suffix='.png',
                 images_dir='',
                 segmentations_dir='',
                 **kwargs) -> None:

        self.images_dir = images_dir
        self.segmentations_dir = segmentations_dir

        super().__init__(
            img_suffix=img_suffix, seg_map_suffix=seg_map_suffix, **kwargs)
        return

    def load_data_list(self) -> List[dict]:
        """Load annotation from directory or annotation file.

        Returns:
            list[dict]: All data info of dataset.
        """
        data_list = []
        segmentation_names = [name for name in sorted(os.listdir(self.segmentations_dir)) if name.endswith(self.seg_map_suffix) \
                            and not name.endswith(f'_vis{self.seg_map_suffix}')] ## 3584 files

        for segmentation_name in segmentation_names:
            image_path = os.path.join(self.images_dir, segmentation_name.replace(self.seg_map_suffix, self.img_suffix))
            segmentation_path = os.path.join(self.segmentations_dir, segmentation_name)

            sample = {
                    'image_path': image_path,
                    'segmentation_path': segmentation_path,
                }

            data_list.append(sample)

        print('\033[92mDone! Pascal2Goliath. Loaded total samples: {}\033[0m'.format(len(data_list)))
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

        ##------convert to goliath format---
        segmentation = np.vectorize(lambda x: SOURCE_TO_TARGET_INDEX_MAPPING.get(x, 255))(segmentation)

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

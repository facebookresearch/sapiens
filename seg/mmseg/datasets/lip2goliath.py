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

# ## LIP dataset original
ORIGINAL_SOURCE_CLASSES=('Background', 'Hat', 'Hair', 'Glove', 'Sunglasses',
                 'UpperClothes', 'Dress', 'Coat', 'Socks', 'Pants',
                 'Jumpsuits', 'Scarf', 'Skirt', 'Face', 'Left-arm',
                 'Right-arm', 'Left-leg', 'Right-leg', 'Left-shoe',
                 'Right-shoe')

#-----------------------------------------------------------------------------------------------
## LIP dataset renamed. Add at 0 at the end of the class name to be marked as don't care
SOURCE_CLASSES=('Background', 'Hat0', 'Hair', 'Glove0', 'Sunglasses0',
                 'Upper_Clothing', 'Dress0', 'Upper_Clothing', 'Socks0', 'Lower_Clothing',
                 'Jumpsuits0', 'Scarf0', 'Upper_Clothing', 'Face_Neck', 'Left-arm0',
                 'Right-arm0', 'Left-leg0', 'Right-leg0', 'Left_Shoe',
                 'Right_Shoe')

TARGET_CLASSES=GoliathDataset.METAINFO['classes']
TARGET_PALETTE=GoliathDataset.METAINFO['palette']

#-----------------------------------------------------------------------------------------------
SOURCE_TO_TARGET_MAPPING = {src_class: TARGET_CLASSES.index(src_class) if src_class in TARGET_CLASSES else 255 for src_class in SOURCE_CLASSES}
SOURCE_TO_TARGET_INDEX_MAPPING = {i: TARGET_CLASSES.index(SOURCE_CLASSES[i]) if SOURCE_CLASSES[i] in TARGET_CLASSES else 255 for i in range(len(SOURCE_CLASSES))}

##-------------------------------------------------------------------------
@DATASETS.register_module()
class LIP2GoliathDataset(BaseSegDataset):
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
        image_names = [name for name in sorted(os.listdir(self.images_dir)) if name.endswith(self.img_suffix)]

        for image_name in image_names:
            sample = {
                    'image_path': os.path.join(self.images_dir, image_name),
                    'segmentation_path': os.path.join(self.segmentations_dir, image_name.replace(self.img_suffix, self.seg_map_suffix)),
            }

            data_list.append(sample)

        print('\033[92mDone! LIP2Goliath. Loaded total samples: {}\033[0m'.format(len(data_list)))
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

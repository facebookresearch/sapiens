from mmseg.registry import DATASETS
from ..basesegdataset import BaseSegDataset

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
        print('Warning: cannot import typed from care!')
        pass

try:
    from care.strict.data.io.file_system.airstore_client import register_airstore_in_fsspec
    register_airstore_in_fsspec()
except:
    print('Warning: cannot import airstore!')

##-----------------------------------------------------------------------
@DATASETS.register_module()
class HiCaVertexMapDataset(BaseSegDataset):
    def __init__(self,
                 **kwargs) -> None:
        self.path_template = "airstoreds://codec_avatar_mgr_12k_frames_no_user_data"
        self.airstore = True
        super().__init__(**kwargs)
        return

    def _read_from_airstore(self, asset: str, id_code: str, seg: str, frame: str) -> io.BytesIO:
        with typed.open(self.path_template + f"/image?subject_id={id_code}&segment={seg}&frame_id={frame}").open() as f:
            data = io.BytesIO(f.read())
        return data

    def load_data_list(self) -> List[dict]:
        """Load annotation from directory or annotation file.

        Returns:
            list[dict]: All data info of dataset.
        """
        data_list = []
        with open(self.ann_file, 'rb') as f:
            raw = f.readlines()
            raw = [x.strip().split() for x in raw]
            raw = [[element.decode('utf-8') for element in sublist] for sublist in raw]
        
        for row in raw:
            dp = {
                "airstore_id": row[0],
                "segment": row[1],
                "frame_id": row[2],
            }
            data_list.append(dp)
        
        print('\033[92mDone! HiCaVertexMap. Loaded total samples: {}\033[0m'.format(len(data_list))) ## 98424 images train

        return data_list

    def get_data_info(self, idx):
        data_info = copy.deepcopy(self.data_list[idx])

        img = Image.open(self._read_from_airstore("image", data_info['airstore_id'], data_info['segment'], data_info['frame_id'])) ## pillow image
        img = np.array(img) ## rgb image
        img = img[:, :, ::-1] # Important: Convert RGB to BGR, the pretrained model preprocessor will convert this to rgb again

        container_path = os.path.join(self.data_root, data_info['airstore_id'], "v_img.zip")
        mask_path = os.path.join(
                            self.data_root,
                            data_info['airstore_id'],
                            data_info['segment'],
                            'mask',
                            '{:06d}.png'.format(int(data_info['frame_id'])))

        if not os.path.exists(mask_path) or not os.path.exists(container_path):
            return None
        
        vertex_map_container = typed.open_container(container_path, mode="r")

        frame_id = int(data_info['frame_id'])
        vertex_map = vertex_map_container.load(f"{data_info['segment']}/{frame_id:06d}.npy") # (h, w, 3) & float32

        mask = cv2.imread(mask_path).astype(np.uint8)
        mask = mask[:, :, 0] ## single channel. 0 to 255

        if mask.sum() < 10:
            return None

        ## get bbox
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
            'gt_val': vertex_map,
            'mask': mask,
            'id': idx,
            'bbox': bbox,
            'bbox_score': np.ones(1, dtype=np.float32),
        }

        return data_info

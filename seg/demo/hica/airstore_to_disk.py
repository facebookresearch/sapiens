from mmseg.registry import DATASETS
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
class HiCaDataset:
    def __init__(self) -> None:
        self.path_template = "airstoreds://codec_avatar_mgr_12k_frames_no_user_data"
        self.data_root = '/uca/rongyu/data/mgr/'
        self.ann_file = '/uca/zhengyuzyy/Datastore/frame_list_hica_sprint/mgr_frame_list_train.txt'
        self.airstore = True
        return

    def _read_from_airstore(self, asset: str, id_code: str, seg: str, frame: str) -> io.BytesIO:
        data_url = self.path_template + f"/image?subject_id={id_code}&segment={seg}&frame_id={frame}"
        import ipdb; ipdb.set_trace()
        with typed.open(data_url).open() as f:
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
        
        print('\033[92mDone! HiCa. Loaded total samples: {}\033[0m'.format(len(data_list)))

        return data_list

    def get_data_info(self, idx):
        data_info = copy.deepcopy(self.data_list[idx])

        import ipdb; ipdb.set_trace()

        img = Image.open(self._read_from_airstore("image", data_info['airstore_id'], data_info['segment'], data_info['frame_id'])) ## pillow image
        img = np.array(img) ## rgb image
        img = img[:, :, ::-1] # Important: Convert RGB to BGR, the pretrained model preprocessor will convert this to rgb again

        mask_path = os.path.join(
                            self.data_root,
                            data_info['airstore_id'],
                            data_info['segment'],
                            'mask',
                            '{:06d}.png'.format(int(data_info['frame_id'])))
        
        if not os.path.exists(mask_path):
            return None

        mask = cv2.imread(mask_path).astype(np.uint8)
        mask = mask[:, :, 0] ## single channel. 0 to 255

        if mask.sum() < 10:
            return None

        return img, mask, data_info
    
def main() -> None:
    dataset = HiCaDataset()
    dataset.data_list = dataset.load_data_list()

    save_dir = os.path.join(dataset.data_root, 'mgr_val')

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    image_dir = os.path.join(save_dir, 'images')
    mask_dir = os.path.join(save_dir, 'masks')
    data_info_dir = os.path.join(save_dir, 'data_info')

    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    
    if not os.path.exists(mask_dir):
        os.makedirs(mask_dir)
    
    if not os.path.exists(data_info_dir):
        os.makedirs(data_info_dir)

    for idx in tqdm(range(len(dataset.data_list))):
        img, mask, data_info = dataset.get_data_info(idx)

        save_image_path = os.path.join(image_dir, '{:06d}.png'.format(idx))
        cv2.imwrite(save_image_path, img)

        save_mask_path = os.path.join(mask_dir, '{:06d}.npy'.format(idx))
        np.save(save_mask_path, mask)

        save_data_info_path = os.path.join(data_info_dir, '{:06d}.pkl'.format(idx))
        with open(save_data_info_path, 'wb') as f:
            pickle.dump(data_info, f)



if __name__ == '__main__':
    main()

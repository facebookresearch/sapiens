import torch
import torch.utils.data
import torch.multiprocessing as mp
import numpy as np
import os
import cv2
from PIL import ImageDraw
from tqdm import tqdm
import io
import json
import copy
from PIL import Image
from care.data.io import typed
from concurrent.futures import ThreadPoolExecutor
import random
from matplotlib import pyplot as plt

class AirstoreDataLoader(torch.utils.data.Dataset):
    def __init__(self, ann_file, mode, num_pts, num_seg):
        super(AirstoreDataLoader, self).__init__()
        self.num_pts = num_pts
        self.num_seg = num_seg
        self.mode = mode
        self.ann_file = ann_file

        self._register_airstore_handler()

        with open(ann_file, "rb") as f:
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

        print('\033[92msorting by session, frame and camera id\033[0m')
        self.data_list = sorted(data_list, key=lambda y: (y['session_id'], y['frame_id'], y['camera_id']))
        self.remove_teeth = False

        if self.remove_teeth:
            self.teeth_ids = list(range(220, 256))

        return

    def _register_airstore_handler(self) -> None:
        from care.strict.data.io.file_system.airstore_client import register_airstore_in_fsspec
        register_airstore_in_fsspec()
        self.path_template = "airstoreds://rlr_detection_services_ml_datasets_no_user_data"
        self.airstore = True

    def _read_from_airstore(self, asset: str, sid: str) -> io.BytesIO:
        with typed.open(self.path_template + f"/{asset}?sample_id={sid}").open() as f:
            data = io.BytesIO(f.read())
        return data

    def __getitem__(self, idx: int):  # noqa
        data_info = copy.deepcopy(self.data_list[idx])

        img = Image.open(self._read_from_airstore("image", data_info['airstore_id'])) ## pillow image
        img = np.array(img)
        img = img[:, :, ::-1]  # Convert RGB to BGR, the model preprocessor will convert this to rgb again

        img_w, img_h = img.shape[1], img.shape[0]

        # process keypoints
        keypoints_np = np.load(self._read_from_airstore("keypoint", data_info['airstore_id']))  # shape 3 x 344
        keypoints = keypoints_np[:2].T.reshape(1, -1, 2)  # shape 1 x 344 x 2
        keypoints_visible = np.where(keypoints_np[2].T > 0, 1, 0).reshape(1, -1)  # shape 1 x 344

        # Identify keypoints that are out of bounds for x (width) and y (height)
        out_of_bounds_w = np.logical_or(keypoints[0, :, 0] <= 0, keypoints[0, :, 0] >= img_w)
        out_of_bounds_h = np.logical_or(keypoints[0, :, 1] <= 0, keypoints[0, :, 1] >= img_h)

        # Update keypoints_visible based on the out-of-bounds keypoints
        keypoints_visible[0, out_of_bounds_w | out_of_bounds_h] = 0

        ## remove teeth keypoints
        if self.remove_teeth:
            # Use numpy's boolean indexing to remove keypoints
            mask = np.ones(keypoints.shape[1], dtype=bool)
            mask[self.teeth_ids] = False
            keypoints = keypoints[:, mask, :]
            keypoints_visible = keypoints_visible[:, mask]

        # Default bounding box to the full image size
        bbox = np.array([0, 0, img_w, img_h], dtype=np.float32).reshape(1, 4)

        if np.any(keypoints_visible):  # If any keypoints are visible
            visible_keypoints = keypoints[0][keypoints_visible[0] == 1]  # Filter out the invisible keypoints

            # Get the bounding box encompassing the keypoints
            x_min, y_min = np.clip(np.min(visible_keypoints, axis=0), [0, 0], [img_w, img_h])
            x_max, y_max = np.clip(np.max(visible_keypoints, axis=0), [0, 0], [img_w, img_h])

            bbox = np.array([x_min, y_min, x_max, y_max], dtype=np.float32).reshape(1, 4)

        num_keypoints = np.count_nonzero(keypoints_visible)

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

        return data_info

    def __len__(self):
        return len(self.data_list)

##---------------------------------------------------------------------------------
if __name__ == "__main__":

    mode = 'test'
    # mode = 'train'

    if mode == 'test':
        ann_file = '/home/rawalk/Desktop/foundational/mmpose/data/goliath/goliath_keypoint_344_test:2023082400.json'
    elif mode == 'train':
        ann_file = '/home/rawalk/Desktop/foundational/mmpose/data/goliath/goliath_keypoint_344_train:2023082601.json'

    images_output_dir = f'/home/rawalk/drive/mmpose/data/goliath_mesh3d/{mode}/images'
    keypoints_output_dir = f'/home/rawalk/drive/mmpose/data/goliath_mesh3d/{mode}/keypoints'

    dataset = AirstoreDataLoader(ann_file, mode, 344, 0)
    print("It will take a few minutes to initialize airstore, please wait patiently...")

    step = 1 ##  every nth sample

    print('start processing...')

    for i in range(len(dataset)):
        if i % step != 0:
            continue

        print(f'{i}/{len(dataset)}')
        sample = dataset[i]

        session_id = sample['session_id']
        camera_id = sample['camera_id']
        frame_id = "{:08}".format(int(sample['frame_id']))

        print(f'session:{session_id}, cam:{camera_id}, frame:{frame_id}')

        keypoints = sample['keypoints']
        keypoints_visible = sample['keypoints_visible']

        keypoints = keypoints[0] ## 344 x 2
        keypoints_visible = keypoints_visible[0].reshape(-1, 1) ## 344 x 1

        ## concatenate keypoints and keypoints_visible
        keypoints = np.concatenate((keypoints, keypoints_visible), axis=-1) # 344 x 3

        this_images_output_dir = os.path.join(images_output_dir, session_id, frame_id)
        this_keypoints_output_dir = os.path.join(keypoints_output_dir, session_id, frame_id)

        if not os.path.exists(this_images_output_dir):
            os.makedirs(this_images_output_dir)

        if not os.path.exists(this_keypoints_output_dir):
            os.makedirs(this_keypoints_output_dir)

        save_image_name = f'{camera_id}.jpg'
        save_image_path = os.path.join(this_images_output_dir, save_image_name)
        cv2.imwrite(save_image_path, sample['img'])

        save_keypoints_name = f'{camera_id}.npy'
        save_keypoints_path = os.path.join(this_keypoints_output_dir, save_keypoints_name)
        np.save(save_keypoints_path, keypoints)

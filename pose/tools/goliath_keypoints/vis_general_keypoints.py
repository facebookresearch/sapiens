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
from mmpose.datasets import GoliathDataset
import sys

OUTPUT_DIR='/home/rawalk/Desktop/sapiens/pose/Outputs/vis/misc/goliath'

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

###-------------------------------------------------------------------------------------------------
class AirstoreDataLoader(torch.utils.data.Dataset):
    def __init__(self, path, mode, num_pts, num_seg):
        super(AirstoreDataLoader, self).__init__()
        self.num_pts = num_pts
        self.num_seg = num_seg
        self.mode = mode

        raw_data = []
        for fp in path:
            with open(fp, "rb") as f:
                raw = f.read()
            raw = json.loads(raw)
            raw_data.append(raw)

        manager = mp.Manager()
        self.data = manager.list()
        for dataset in raw_data:
            data = []
            for x in dataset:
                dp = {}
                dp["sample_id"] = x["sample_id"]
                if x.get("box-default") is not None:
                    dp["box"] = x["box-default"]
                data.append(dp)
            self.data.extend(data)

    def _register_airstore_handler(self) -> None:
        from care.strict.data.io.file_system.airstore_client import register_airstore_in_fsspec
        register_airstore_in_fsspec()
        self.path_template = "airstoreds://rlr_detection_services_ml_datasets_no_user_data"
        self.airstore = True

    def _read_from_airstore(self, asset: str, sid: str) -> io.BytesIO:
        with typed.open(self.path_template + f"/{asset}?sampleId={sid}").open() as f:
            data = io.BytesIO(f.read())
        return data

    def __getitem__(self, idx: int):  # noqa
        dp = copy.deepcopy(self.data[idx])
        with_keypoint = self.num_pts > 0 and self.mode != "test"
        with_segmentation = self.num_seg > 0 and self.mode != "test"

        # read data from RSC airstore
        if not hasattr(self, 'airstore'):
            self._register_airstore_handler()
        sid = dp["sample_id"]
        img = Image.open(self._read_from_airstore("image", sid))
        points = segmentations = None
        if with_keypoint:
            points = np.load(self._read_from_airstore("keypoint", sid))

        if with_segmentation:
            segmentations = Image.open(self._read_from_airstore("segmentation", sid))

        print(segmentations)

        return img, points, segmentations


def save_image(index, dl, output_dir):
    image = dl[index][0]
    image_path = os.path.join(output_dir, '{:06d}.jpg'.format(index))
    image.save(image_path)

###-------------------------------------------------------------------------------------------------
def main():
    # test_set = '/uca/hewen/datasets/sociopticon_segmentation_33_test:2023082200.json'
    # train_set = '/uca/hewen/datasets/sociopticon_segmentation_33_train:2023082200.json'

    train_set = '/home/rawalk/Desktop/sapiens/pose/data/goliath/goliath_keypoint_344_train:2023082601.json'
    test_set = '/home/rawalk/Desktop/sapiens/pose/data/goliath/goliath_keypoint_344_test:2023082400.json'

    path = train_set
    dl = AirstoreDataLoader([path], 'train', 344, 1)
    print("It will take a few minutes to initialize airstore, please wait patiently...")

    num_samples = 1000
    sampled_indices = random.sample(range(len(dl.data)), num_samples)  # randomly sample 1000 indices from the length of dl

    for index in sampled_indices:
        image = dl[index][0]  # Pillow image
        keypoints = dl[index][1]  # 3 x 344, x, y, visibility

        image_np = np.array(image) ## rgb image

        # # Use goliath_vis_keypoints to visualize
        vis_image = goliath_vis_keypoints(image_np, keypoints.T)
        vis_image = cv2.resize(vis_image, (vis_image.shape[1]//2, vis_image.shape[0]//2), interpolation=cv2.INTER_LINEAR)

        # Save the image to output_dir with 7 digit padded name
        filename = os.path.join(output_dir, f"{index:07}.jpg")
        cv2.imwrite(filename, cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))  # Convert from RGB to BGR for cv2

        print(f"Saved visualized keypoints to {filename}")


if __name__ == "__main__":
    main()

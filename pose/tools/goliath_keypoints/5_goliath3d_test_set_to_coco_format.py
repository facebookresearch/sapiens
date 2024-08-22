import torch
import torch.utils.data
import torch.multiprocessing as mp
import numpy as np
import os
from typing import Dict, Optional, Sequence
import cv2
from PIL import ImageDraw
from tqdm import tqdm
import io
import json
import copy
import random
import datetime
import sys
from pathlib import Path
from mmengine.fileio import dump

##---------------------------goliath info--------------------------------------------
script_location = Path(__file__).resolve()
root_dir = script_location.parent.parent.parent
sys.path.append(str(os.path.join(root_dir, 'configs', '_base_', 'datasets')))
from configs._base_.datasets.goliath3d import dataset_info as GOLIATH_INFO

##-----------------------------------------------------------------------------
GOLIATH_INFO['name2id'] = {}
for keypoint_id, keypoint_info in GOLIATH_INFO['keypoint_info'].items():
    GOLIATH_INFO['name2id'][keypoint_info['name']] = keypoint_id

GOLIATH_INFO['body_keypoint_ids'] = [GOLIATH_INFO['name2id'][name] \
                for name in GOLIATH_INFO['body_keypoint_names']]

GOLIATH_INFO['foot_keypoint_ids'] = [GOLIATH_INFO['name2id'][name] \
                for name in GOLIATH_INFO['foot_keypoint_names']]

GOLIATH_INFO['face_keypoint_ids'] = [GOLIATH_INFO['name2id'][name] \
                for name in GOLIATH_INFO['face_keypoint_names']]

GOLIATH_INFO['left_hand_keypoint_ids'] = [GOLIATH_INFO['name2id'][name] \
                for name in GOLIATH_INFO['left_hand_keypoint_names']]

GOLIATH_INFO['right_hand_keypoint_ids'] = [GOLIATH_INFO['name2id'][name] \
                for name in GOLIATH_INFO['right_hand_keypoint_names']]

##-----------------------------------------------------------------------------
CATEGORIES = [
    {
        "supercategory": "person",
        "id": 1,
        "name": "person",
        "keypoints": [],
        "skeleton": []
    }
]
def gt_to_coco_json(gt_dicts: Sequence[dict],
                        outfile_path: str) -> str:
        """Convert ground truth to coco format json file.

        Args:
            gt_dicts (Sequence[dict]): Ground truth of the dataset. Each dict
                contains the ground truth information about the data sample.
                Required keys of the each `gt_dict` in `gt_dicts`:
                    - `img_id`: image id of the data sample
                    - `width`: original image width
                    - `height`: original image height
                    - `raw_ann_info`: the raw annotation information
                Optional keys:
                    - `crowd_index`: measure the crowding level of an image,
                        defined in CrowdPose dataset
                It is worth mentioning that, in order to compute `CocoMetric`,
                there are some required keys in the `raw_ann_info`:
                    - `id`: the id to distinguish different annotations
                    - `image_id`: the image id of this annotation
                    - `category_id`: the category of the instance.
                    - `bbox`: the object bounding box
                    - `keypoints`: the keypoints cooridinates along with their
                        visibilities. Note that it need to be aligned
                        with the official COCO format, e.g., a list with length
                        N * 3, in which N is the number of keypoints. And each
                        triplet represent the [x, y, visible] of the keypoint.
                    - 'keypoints'
                    - `iscrowd`: indicating whether the annotation is a crowd.
                        It is useful when matching the detection results to
                        the ground truth.
                There are some optional keys as well:
                    - `area`: it is necessary when `self.use_area` is `True`
                    - `num_keypoints`: it is necessary when `self.iou_type`
                        is set as `keypoints_crowd`.
            outfile_prefix (str): The filename prefix of the json files. If the
                prefix is "somepath/xxx", the json file will be named
                "somepath/xxx.gt.json".
        Returns:
            str: The filename of the json file.
        """
        image_infos = []
        annotations = []
        img_ids = []
        ann_ids = []

        for gt_dict in gt_dicts:
            # filter duplicate image_info
            if gt_dict['img_id'] not in img_ids:
                image_info = dict(
                    id=gt_dict['img_id'],
                    width=gt_dict['width'],
                    height=gt_dict['height'],
                    subject_id=gt_dict['subject_id'],
                    camera_id=gt_dict['camera_id'],
                    frame_index=gt_dict['frame_index'],
                    file_name=gt_dict['file_name'],
                )

                image_infos.append(image_info)
                img_ids.append(gt_dict['img_id'])

            # filter duplicate annotations
            for ann in gt_dict['raw_ann_info']:
                annotation = dict(
                    id=ann['id'],
                    image_id=ann['image_id'],
                    category_id=ann['category_id'],
                    bbox=ann['bbox'],
                    goliath_wholebody_kpts=ann['goliath_wholebody_kpts'],
                    keypoints=ann['keypoints'],
                    foot_kpts=ann['foot_kpts'],
                    face_kpts=ann['face_kpts'],
                    lefthand_kpts=ann['lefthand_kpts'],
                    righthand_kpts=ann['righthand_kpts'],
                    iscrowd=ann['iscrowd'],
                    num_keypoints=ann['num_keypoints'],
                )
                use_area = True
                if use_area:
                    assert 'area' in ann, \
                        '`area` is required when `use_area` is `True`'
                    annotation['area'] = ann['area']

                annotations.append(annotation)
                ann_ids.append(ann['id'])

        info = dict(
            date_created=str(datetime.datetime.now()),
            goliath_info=GOLIATH_INFO,
            description='goliath json file in Coco format.')
        coco_json: dict = dict(
            info=info,
            images=image_infos,
            categories=CATEGORIES,
            licenses=None,
            annotations=annotations,
        )
        outfile_dir = os.path.dirname(outfile_path)
        if not os.path.exists(outfile_dir):
            os.makedirs(outfile_dir)
        converted_json_path = outfile_path
        dump(coco_json, converted_json_path, sort_keys=True, indent=4)
        return converted_json_path

def pose_to_bbox(pose, image_width, image_height, keypoint_thres=0.5, padding=1.4, min_keypoints=5):
    is_valid = (pose[:, 2] > keypoint_thres)
    is_valid = is_valid * (pose[:, 0] > 0) * (pose[:, 0] <= image_width)
    is_valid = is_valid * (pose[:, 1] > 0) * (pose[:, 1] <= image_height)

    if is_valid.sum() < min_keypoints:
        return None, None, None, None, None

    x1 = pose[is_valid, 0].min(); x2 = pose[is_valid, 0].max()
    y1 = pose[is_valid, 1].min(); y2 = pose[is_valid, 1].max()

    center_x = (x1 + x2)/2
    center_y = (y1 + y2)/2

    scale_x = (x2 - x1)*padding
    scale_y = (y2 - y1)*padding

    bbx = max(1, center_x - scale_x/2)
    bby = max(1, center_y - scale_y/2)

    bbw = scale_x
    bbh = scale_y

    return bbx, bby, bbw, bbh, is_valid

##-----------------------------------------------------------------------------
# ROOT_DIR='/home/rawalk/Desktop/sapiens/pose/data/goliath/test_10000'
ROOT_DIR='/home/rawalk/Desktop/sapiens/pose/data/goliath/test_5000'

images_dir = os.path.join(ROOT_DIR, 'images')
gt_keypoints_dir = os.path.join(ROOT_DIR, 'keypoints')
pred_keypoints_dir = os.path.join(ROOT_DIR, 'pred_keypoints')

gt_dicts = []

image_height = GOLIATH_INFO['image_height']
image_width = GOLIATH_INFO['image_width']
original_num_keypoints = len(GOLIATH_INFO['original_keypoint_info']) # 344 originally

image_id = 0

##--------------------------------------------------------------------------
keypoint_original_idxs = None
if GOLIATH_INFO['idx_to_original_idx_mapping'] is not None:
    keypoint_original_idxs = np.array(list(GOLIATH_INFO['idx_to_original_idx_mapping'].values()))

##--------------------------------------------------------------------------
for subject_name in sorted(os.listdir(images_dir)):
    print('subject:', subject_name)
    subject_dir = os.path.join(images_dir, subject_name)

    for camera_name in sorted(os.listdir(subject_dir)):
        print('\tcamera:', camera_name)
        camera_dir = os.path.join(subject_dir, camera_name)
        image_names = sorted([f for f in os.listdir(camera_dir) if f.endswith('.jpg')])

        for image_name in image_names:
            print('\t\timage:', image_name, ' image_id', image_id)
            image_path = os.path.join(camera_dir, image_name)
            gt_keypoints_path = os.path.join(gt_keypoints_dir, subject_name, camera_name, image_name.replace('.jpg', '.npy'))

            with open(gt_keypoints_path, 'rb') as f:
                gt_keypoints = np.load(f) # shape 344 x 3

                if keypoint_original_idxs is not None:
                    gt_keypoints = gt_keypoints[keypoint_original_idxs, :] # N x 3

            ## compute the bbox as min and max visible keypoints
            bbx, bby, bbw, bbh, is_valid = pose_to_bbox(gt_keypoints, image_width, image_height, min_keypoints=5)
            gt_keypoints[:, 2] = is_valid
            if bbx is None or bby is None or bbw is None or bbh is None:
                continue

            # if num_keypoints is less than 8 then skip
            if gt_keypoints[:, 2].sum() < 8:
                continue

            def keypoints_array2list(keypoints_arr):
                kps = [0]*len(keypoints_arr)*3
                kps_v = np.zeros(len(keypoints_arr))
                kps_v[keypoints_arr[:, 2] ==  1] = 2

                kps[0::3] = keypoints_arr[:, 0].round().astype(int)
                kps[1::3] = keypoints_arr[:, 1].round().astype(int)
                kps[2::3] = kps_v.tolist()
                return kps

            body_gt_keypoints = gt_keypoints[GOLIATH_INFO['body_keypoint_ids'], :].copy()
            foot_gt_keypoints = gt_keypoints[GOLIATH_INFO['foot_keypoint_ids'], :].copy()
            face_gt_keypoints = gt_keypoints[GOLIATH_INFO['face_keypoint_ids'], :].copy()
            left_hand_gt_keypoints = gt_keypoints[GOLIATH_INFO['left_hand_keypoint_ids'], :].copy()
            right_hand_gt_keypoints = gt_keypoints[GOLIATH_INFO['right_hand_keypoint_ids'], :].copy()

            num_keypoints = gt_keypoints[:, 2].sum()

            body_kps = keypoints_array2list(body_gt_keypoints) ## this is body keypoints only
            foot_kps = keypoints_array2list(foot_gt_keypoints)
            face_kps = keypoints_array2list(face_gt_keypoints)
            left_hand_kps = keypoints_array2list(left_hand_gt_keypoints)
            right_hand_kps = keypoints_array2list(right_hand_gt_keypoints)

            wholebody_kpts = keypoints_array2list(gt_keypoints)

            ann_info = {
                'id': image_id,
                'image_id': image_id,
                'category_id': 1,
                'area': int(bbw*bbh),
                'bbox': [int(bbx), int(bby), int(bbw), int(bbh)],
                'iscrowd': 0,
                'goliath_wholebody_kpts': wholebody_kpts, ## all keypoints
                'keypoints': body_kps, ## just body keypoints, 17 default
                'foot_kpts': foot_kps, ## 6 foot keypoints
                'face_kpts': face_kps,
                'lefthand_kpts': left_hand_kps,
                'righthand_kpts': right_hand_kps,
                'num_keypoints': num_keypoints,
            }

            gt_dict = {
                'img_id': image_id,
                'width': image_width,
                'height': image_height,
                'subject_id': subject_name,
                'camera_id': camera_name,
                'frame_index': image_name.replace('.jpg', ''),
                'file_name': os.path.join(subject_name, camera_name, image_name),
                'raw_ann_info': [ann_info],
            }

            image_id += 1

            gt_dicts.append(gt_dict)

##---------------------------------------------------------------------
gt_to_coco_json(gt_dicts, outfile_path=os.path.join(ROOT_DIR, 'annotations_3d', 'person_keypoints_test2023.json'))
print('saved annotations to: ', os.path.join(ROOT_DIR, 'annotations_3d', 'person_keypoints_test2023.json'))

##---------------------------------------------------------------------
print('num body keypoints: ', len(GOLIATH_INFO['body_keypoint_ids']))
print('num foot keypoints: ', len(GOLIATH_INFO['foot_keypoint_ids']))
print('num face keypoints: ', len(GOLIATH_INFO['face_keypoint_ids']))
print('num left hand keypoints: ', len(GOLIATH_INFO['left_hand_keypoint_ids']))
print('num right hand keypoints: ', len(GOLIATH_INFO['right_hand_keypoint_ids']))

total_num_keypoints = len(GOLIATH_INFO['body_keypoint_ids']) + len(GOLIATH_INFO['foot_keypoint_ids']) + len(GOLIATH_INFO['face_keypoint_ids']) + len(GOLIATH_INFO['left_hand_keypoint_ids']) + len(GOLIATH_INFO['right_hand_keypoint_ids'])
print('total num keypoints: ', total_num_keypoints)

extra_keypoint_names = [info['name'] for id, info in GOLIATH_INFO['keypoint_info'].items() \
                        if id not in GOLIATH_INFO['body_keypoint_ids'] + \
                        GOLIATH_INFO['foot_keypoint_ids'] + \
                        GOLIATH_INFO['face_keypoint_ids'] + \
                        GOLIATH_INFO['left_hand_keypoint_ids'] + \
                        GOLIATH_INFO['right_hand_keypoint_ids']]

print('remaining extra keypoints: ', extra_keypoint_names)

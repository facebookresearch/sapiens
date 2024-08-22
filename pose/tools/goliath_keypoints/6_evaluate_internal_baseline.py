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
from collections import OrderedDict, defaultdict

from mmpose.evaluation.metrics.goliath_metric import GoliathMetric

PREDICTION_DIR='/home/rawalk/Desktop/foundational/mmpose/data/goliath/test_5000/pred_keypoints'

goliath_metric = GoliathMetric(
                        ann_file='/home/rawalk/drive/vitpose/data/goliath/test_5000/annotations/person_keypoints_test2023.json',
                        use_area=True,
                        iou_type='keypoints',
                        score_mode='bbox_keypoint',
                        keypoint_score_thr=0.2,
                        nms_mode='oks_nms',
                        nms_thr=0.9,
                    )

## get image_name to image_id mapping
image_ids = goliath_metric.coco.getImgIds()
image_infos = goliath_metric.coco.loadImgs(image_ids)
GOLIATH_INFO = goliath_metric.coco.__dict__['dataset']['info']['goliath_info']
original_num_keypoints = len(GOLIATH_INFO['original_keypoint_info']) # 344 originally

if GOLIATH_INFO['remove_teeth'] == True:
    teeth_ids = GOLIATH_INFO['teeth_keypoint_ids'] ## to remove teeths
    teeth_mask = np.ones(original_num_keypoints, dtype=bool)
    teeth_mask[teeth_ids] = False

num_keypoints = original_num_keypoints - len(teeth_ids) if GOLIATH_INFO['remove_teeth'] else original_num_keypoints
goliath_metric.dataset_meta = {'num_keypoints': num_keypoints, 'sigmas': np.array(GOLIATH_INFO['sigmas'])}

results = []
for image_info in image_infos:
    image_id = image_info['id']
    subject_id = image_info['subject_id']
    camera_id = image_info['camera_id']
    frame_index = image_info['frame_index']

    prediction_path = os.path.join(PREDICTION_DIR, subject_id, camera_id, f'{frame_index}.npy')
    prediction = np.load(prediction_path) ## 344 x 3

    # remove the teeth predictions. 344 -> 308
    if GOLIATH_INFO['remove_teeth'] == True:
        prediction = prediction[teeth_mask, :]
        assert len(prediction) == num_keypoints

    keypoints = prediction[:, :2].reshape(1, -1, 2) ## extra dimension for number of detections. 1 for top down, 1 x 308 x 2
    keypoint_scores = prediction[:, 2].reshape(1, -1) ### 1 x 308

    pred = dict()
    pred['id'] = image_id
    pred['img_id'] = image_id
    pred['keypoints'] = keypoints
    pred['keypoint_scores'] = keypoint_scores
    pred['category_id'] = 1
    pred['bbox_scores'] = np.ones(len(keypoints))

    gt = dict()
    results.append((pred, gt))

goliath_metric.compute_metrics(results)

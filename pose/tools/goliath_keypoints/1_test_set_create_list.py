#!/usr/bin/env python3
import os
import sys
import copy
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor as Pool
from multiprocessing import set_start_method
import torch
import cv2

from detcore.utils.slurm import submit_array_job
from detcore.utils import logging, platform
from detcore.utils.str_generator import is_string_generator
from train import main as train, get_argparser as get_train_parser

from typing import Optional, Any, List, Tuple
import json

ROOT_DIR = '/mnt/home/rawalk/drive/mmpose/data/goliath/test'
IMAGES_DIR = os.path.join(ROOT_DIR, 'images')
KEYPOINTS_DIR = os.path.join(ROOT_DIR, 'keypoints')

##------------------------------------------------------------
def get_image_keypoint_lists(IMAGES_DIR, KEYPOINTS_DIR):
    image_list = []
    keypoint_list = []

    subject_names = sorted(os.listdir(IMAGES_DIR))

    for subject_idx, subject_name in enumerate(subject_names):
        camera_names = sorted(os.listdir(os.path.join(IMAGES_DIR, subject_name)))

        for camera_idx, camera_name in enumerate(camera_names):
            image_names = sorted(os.listdir(os.path.join(IMAGES_DIR, subject_name, camera_name)))

            for image_name in image_names:
                image_path = os.path.join(IMAGES_DIR, subject_name, camera_name, image_name)
                keypoint_path = os.path.join(KEYPOINTS_DIR, subject_name, camera_name, image_name.replace('.jpg', '.npy'))
                
                image_list.append(image_path)
                keypoint_list.append(keypoint_path)

    return image_list, keypoint_list


##------------------------------------------------------------
images_file = os.path.join(ROOT_DIR, 'images.txt')
keypoints_file = os.path.join(ROOT_DIR, 'keypoints.txt')
image_list, keypoint_list = get_image_keypoint_lists(IMAGES_DIR, KEYPOINTS_DIR)

##------------------------------------------------------------
with open(images_file, 'w') as f:
    for item in image_list:
        f.write("%s\n" % item)

with open(keypoints_file, 'w') as f:
    for item in keypoint_list:
        f.write("%s\n" % item)

print('test images are {}'.format(len(image_list)))
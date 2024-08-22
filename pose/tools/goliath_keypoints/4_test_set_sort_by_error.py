#!/usr/bin/env python3
import os
import sys
import copy
import argparse
import torch
import cv2
import shutil
import json
import numpy as np

# Assuming gt_keypoint, pred_keypoint, and gt_keypoint_vis are numpy arrays
def weighted_l2_distance(gt_keypoint, pred_keypoint, gt_keypoint_vis):
    # Calculate the squared difference
    diff = gt_keypoint - pred_keypoint
    squared_diff = np.square(diff)

    # Apply weighting by visibility
    weighted_squared_diff = squared_diff * gt_keypoint_vis

    # Count the number of visible keypoints
    num_visible_keypoints = np.sum(gt_keypoint_vis)

    # Calculate the mean error, avoiding division by zero
    mean_error = np.sqrt(np.sum(weighted_squared_diff)) / num_visible_keypoints if num_visible_keypoints > 0 else 0
    return mean_error

##--------------------------------------------------------------------------------
ROOT_DIR = '/mnt/home/rawalk/drive/mmpose/data/goliath/test'
IMAGES_DIR = os.path.join(ROOT_DIR, 'images')
KEYPOINTS_DIR = os.path.join(ROOT_DIR, 'keypoints')

OUTPUT_DIR = '/mnt/home/rawalk/Desktop/foundational/mmpose/Outputs/goliath/test_chunks'

chunk_names = sorted(os.listdir(OUTPUT_DIR))
print('number of chunks:{}'.format(len(chunk_names)))

##--------------------------------------------------------------------------------
image_path_to_error = {}

for chunk_id, chunk_name in enumerate(chunk_names):
    predictions_path = os.path.join(OUTPUT_DIR, chunk_name, 'predictions.pth')
    predictions = torch.load(predictions_path)
    num_samples = len(predictions['path'])

    print('chunk:{}, num_samples:{}'.format(chunk_id, num_samples))
    for path, pred_keypoint, pred_keypoint_score in zip(predictions['path'], predictions['keypoint_location'], predictions['keypoint_score']):
        gt_keypoint_path = path.replace('/images', '/keypoints').replace('.jpg', '.npy')
        gt_keypoint = np.load(gt_keypoint_path) ## 344 x 3

        gt_keypoint_vis = gt_keypoint[:, 2].reshape(-1, 1) ## 344 x 1
        gt_keypoint = gt_keypoint[:, :2] ## 344 x 2

        ## read image
        image = cv2.imread(path)
        B, G, R = cv2.split(image)

        # If all channels are the same, it's grayscale
        if np.array_equal(B, G) and np.array_equal(B, R):
            continue

        # if num_keypoints is less than 8 then skip
        if gt_keypoint_vis.sum() < 8:
            continue

        # Compute the weighted L2 distance
        error = weighted_l2_distance(gt_keypoint, pred_keypoint, gt_keypoint_vis)

        image_path_to_error[path] = error

##-----------------------------------------------------------------------------
num_hard_images = 10000
SAVE_ROOT_DIR = '/mnt/home/rawalk/drive/mmpose/data/goliath/test_{}'.format(num_hard_images)

if not os.path.exists(SAVE_ROOT_DIR):
    os.makedirs(SAVE_ROOT_DIR)

images_save_dir = os.path.join(SAVE_ROOT_DIR, 'images')
if not os.path.exists(images_save_dir):
    os.makedirs(images_save_dir)

# Sort the image paths by error, in descending order
sorted_images = sorted(image_path_to_error.items(), key=lambda x: x[1], reverse=True)

# Select the top 5000 images
top_hard_images = sorted_images[:num_hard_images]

# Extract only the image paths
top_hard_image_paths = [image[0] for image in top_hard_images]

# Store these paths in a file and print them with their errors
hard_images_file = os.path.join(SAVE_ROOT_DIR, 'images.txt')
with open(hard_images_file, 'w') as file:
    print("Top {} hardest images with errors:".format(num_hard_images))
    for path, error in top_hard_images:
        file.write(path + '\n')
        print(f"{path}: {error}")

gt_keypoints_dir = os.path.join(SAVE_ROOT_DIR, 'keypoints')
pred_keypoints_dir = os.path.join(SAVE_ROOT_DIR, 'pred_keypoints')
if not os.path.exists(gt_keypoints_dir):
    os.makedirs(gt_keypoints_dir)
if not os.path.exists(pred_keypoints_dir):
    os.makedirs(pred_keypoints_dir)

print(f"Top {num_hard_images} hardest images are stored in {hard_images_file}")

count = 0
for chunk_id, chunk_name in enumerate(chunk_names):
    predictions_path = os.path.join(OUTPUT_DIR, chunk_name, 'predictions.pth')
    predictions = torch.load(predictions_path)

    for path, pred_keypoint, pred_keypoint_score in zip(predictions['path'], predictions['keypoint_location'], predictions['keypoint_score']):
        
        if path not in top_hard_image_paths:
            continue

        gt_keypoint_path = path.replace('/images', '/keypoints').replace('.jpg', '.npy')
        gt_keypoint = np.load(gt_keypoint_path) ## 344 x 3
        
        # Save the image to the new directory
        save_image_path = path.replace(IMAGES_DIR, images_save_dir)
        save_gt_keypoint_path = save_image_path.replace('/images', '/keypoints').replace('.jpg', '.npy')
        save_pred_keypoint_path = save_image_path.replace('/images', '/pred_keypoints').replace('.jpg', '.npy')

        this_save_dir = os.path.dirname(save_image_path)
        if not os.path.exists(this_save_dir):
            os.makedirs(this_save_dir)
        
        this_save_dir = os.path.dirname(save_gt_keypoint_path)
        if not os.path.exists(this_save_dir):
            os.makedirs(this_save_dir)
        
        this_save_dir = os.path.dirname(save_pred_keypoint_path)
        if not os.path.exists(this_save_dir):
            os.makedirs(this_save_dir)

        shutil.copy(path, save_image_path)
        count += 1

        print('saved {}'.format(count))

        pred_keypoint = np.concatenate((pred_keypoint, pred_keypoint_score.reshape(-1, 1)), axis=1)
        np.save(save_gt_keypoint_path, gt_keypoint)
        np.save(save_pred_keypoint_path, pred_keypoint)


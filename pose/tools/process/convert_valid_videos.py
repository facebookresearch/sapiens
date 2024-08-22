# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import numpy as np
import json
import cv2
from argparse import ArgumentParser
from tqdm import tqdm
import concurrent.futures

def get_video_pose(read_output_video_folder):
    video_pose = []

    frame_jsons = sorted([x for x in os.listdir(read_output_video_folder) if x.endswith('.json')])
    for frame_json in frame_jsons:

        try:
            with open(os.path.join(read_output_video_folder, frame_json), 'r') as f:
                frame_pose = json.load(f)
        except Exception as e:
            print(e)
            print('Error reading file: {}'.format(os.path.join(read_output_video_folder, frame_json)))
            frame_pose = None

        if frame_pose is not None:
            instances = frame_pose['instance_info']
            max_keypoint_score = -1
            max_keypoint_instance = None
            for instance in instances:
                keypoint_score = np.array(instance['keypoint_scores']).sum()
                if keypoint_score > max_keypoint_score:
                    max_keypoint_score = keypoint_score
                    max_keypoint_instance = instance

            frame_pose = np.zeros((133, 3))  # x, y, confidence
            if max_keypoint_instance:
                keypoints = np.array(max_keypoint_instance['keypoints'])
                keypoint_scores = np.array(max_keypoint_instance['keypoint_scores'])
                frame_pose[:, 0] = keypoints[:, 0]
                frame_pose[:, 1] = keypoints[:, 1]
                frame_pose[:, 2] = keypoint_scores

        else:
            frame_pose = np.zeros((133, 3))

        video_pose.append(frame_pose)

    print('Done: {}: Total Frames:{}'.format(read_output_video_folder, len(video_pose)))

    return np.array(video_pose)

def process_video_file(data_dir, read_output_dir, output_dir, video_file):
    read_output_video_folder = video_file.replace(data_dir, read_output_dir).replace('.MP4', '').replace('.mp4', '').replace('_fixed', '')

    video_pose = None

    video_pose = get_video_pose(read_output_video_folder)

    save_video_path = video_file.replace(data_dir, output_dir).replace('.MP4', '.npy').replace('.mp4', '.npy').replace('.MTS', '.npy').replace('.MOV', '.npy').replace('_fixed', '')
    save_video_dir = os.path.dirname(save_video_path)

    if video_pose is not None:
        os.makedirs(save_video_dir, exist_ok=True)
        np.save(save_video_path, video_pose)

    return save_video_path

def main():
    parser = ArgumentParser()
    parser.add_argument('--start_valid_index', type=int, default=0, help='start video index')
    parser.add_argument('--end_valid_index', type=int, default=-1, help='end video index')
    args = parser.parse_args()

    data_dir = '/uca/richardalex/data/casual_conversations/raw_videos'
    read_output_dir = '/home/rawalk/Desktop/sapiens/pose/Outputs/pose_process/keypoints133'
    output_dir = '/uca/richardalex/data/casual_conversations/keypoints133'

    with open(os.path.join(read_output_dir, 'valid_files.txt'), 'r') as f:
        video_files = [x.strip() for x in f.readlines()]

    video_files = sorted(video_files)

    ##----------
    unprocessed_video_files = []

    for video_file in video_files:
        save_video_path = video_file.replace(data_dir, output_dir).replace('.MP4', '.npy').replace('.mp4', '.npy').replace('.MTS', '.npy').replace('.MOV', '.npy')
        part1_save_video_path = save_video_path.replace('keypoints133', 'keypoints133_part1')
        if not os.path.exists(save_video_path) and not os.path.exists(part1_save_video_path):
            unprocessed_video_files.append(video_file)

    print(f'Total unprocessed videos: {len(unprocessed_video_files)}')
    print(f'Total processed videos: {len(video_files) - len(unprocessed_video_files)}')

    video_files = unprocessed_video_files

    ##----------
    print('Total valid videos: {}'.format(len(video_files)))

    if args.end_valid_index == -1 or args.end_valid_index > len(video_files):
        args.end_valid_index = len(video_files) - 1
    video_files = video_files[args.start_valid_index:args.end_valid_index + 1]

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    print(f'Processing {len(video_files)} videos from {args.start_valid_index} to {args.end_valid_index}')

    # Using ThreadPoolExecutor to process files in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        results = list(tqdm(executor.map(lambda vf: process_video_file(data_dir, read_output_dir, output_dir, vf), video_files), total=len(video_files)))

    print(f'Processed files saved to: {output_dir}')

if __name__ == '__main__':
    main()

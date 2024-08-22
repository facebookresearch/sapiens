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

def main():
    parser = ArgumentParser()
    parser.add_argument('--valid_video_index', type=int, default=0, help='start video index')
    args = parser.parse_args()

    data_dir = '/uca/richardalex/data/casual_conversations'

    video_dir = os.path.join(data_dir, 'raw_videos')
    pose_dir = os.path.join(data_dir, 'keypoints133')
    output_dir = os.path.join(data_dir, 'vis_keypoints133')

    with open(os.path.join(data_dir, 'valid_files.txt'), 'r') as f:
        video_files = [x.strip() for x in f.readlines()]

    video_files = sorted(video_files)

    video_file = video_files[args.valid_video_index]
    video_pose_path = video_file.replace(video_dir, pose_dir).replace('.MP4', '.npy')
    vis_video_dir = video_file.replace(video_dir, output_dir).replace('.MP4', '')

    print('Total valid videos: {}'.format(len(video_files)))
    print('Visualing video: {}'.format(video_file))

    if not os.path.exists(vis_video_dir):
        os.makedirs(vis_video_dir)

    video_pose = np.load(video_pose_path)

    cap = cv2.VideoCapture(video_file)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    assert len(video_pose) == total_frames

    frame_idx = 0
    radius = 4

    with tqdm(total=total_frames) as pbar:
        while cap.isOpened():
            success, frame = cap.read()

            if not success:
                break

            pose = video_pose[frame_idx] ## (133, 3)

            # Draw the pose
            for i in range(133):
                if pose[i][2] > 0:  # Check if confidence is greater than zero
                    cv2.circle(frame, (int(pose[i][0]), int(pose[i][1])), radius, (255, 255, 255), -1)  # White color, filled circle

            save_path = os.path.join(vis_video_dir, f'{frame_idx:06d}.jpg')
            cv2.imwrite(save_path, frame)

            frame_idx += 1
            pbar.update(1)

    cap.release()


if __name__ == '__main__':
    main()

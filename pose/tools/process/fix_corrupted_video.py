# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import cv2
import subprocess
from tqdm import tqdm
from argparse import ArgumentParser
import concurrent.futures

def main():
    parser = ArgumentParser()
    parser.add_argument('--start_valid_index', type=int, default=0, help='start video index')
    parser.add_argument('--end_valid_index', type=int, default=-1, help='end video index')
    args = parser.parse_args()

    data_dir = '/uca/richardalex/data/casual_conversations/raw_videos'
    encoded_dir = '/uca/richardalex/data/casual_conversations/raw_videos_fixed'
    output_dir = '/home/rawalk/Desktop/sapiens/pose/Outputs/pose_process/keypoints133'

    if not os.path.exists(encoded_dir):
        os.makedirs(encoded_dir)

    unprocessed_video_files = []
    invalid_file_path = os.path.join(output_dir, 'invalid_files.txt')

    with open(invalid_file_path, 'r') as f:
        unprocessed_video_files = [x.strip() for x in f.readlines()]

    video_files = sorted(unprocessed_video_files)

    if args.end_valid_index == -1 or args.end_valid_index > len(video_files):
        args.end_valid_index = len(video_files) - 1
    video_files = video_files[args.start_valid_index:args.end_valid_index + 1]

    for video_file in tqdm(video_files):
        print('Fixing video: {}'.format(video_file))
        video_name = video_file.split('/')[-1]

        ## change the video file using ffmpeg
        input_path = video_file

        # Output path with _encoded prefix added before the file extension
        output_path = video_file.replace(data_dir, encoded_dir)

        this_output_dir = os.path.dirname(output_path)
        if not os.path.exists(this_output_dir):
            os.makedirs(this_output_dir)

        # FFmpeg command to re-encode the video
        command = ['ffmpeg', '-i', input_path, '-c:v', 'libx264', '-preset', 'fast', '-crf', '22', output_path]

        # Run the FFmpeg command
        subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

if __name__ == '__main__':
    main()

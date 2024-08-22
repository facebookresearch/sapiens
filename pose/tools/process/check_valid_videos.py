# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import cv2
from tqdm import tqdm
import concurrent.futures

def main():
    data_dir = '/uca/richardalex/data/casual_conversations/raw_videos'
    encoded_dir = '/uca/richardalex/data/casual_conversations/raw_videos_fixed'
    output_dir = '/home/rawalk/Desktop/sapiens/pose/Outputs/pose_process/keypoints133'

    # Collect all video file paths
    video_files = collect_video_files(data_dir)

    # Process each video file to check its validity in parallel
    valid_files, invalid_files, corrupted_files = process_video_files_parallel(video_files, data_dir, encoded_dir, output_dir)

    # Print the results
    print('Total files:', len(video_files))
    print('Valid files:', len(valid_files))
    print('Invalid files:', len(invalid_files))
    print('Corrupted files:', len(corrupted_files))

    # Save valid and invalid file lists to disk
    save_file_list(os.path.join(output_dir, 'valid_files.txt'), valid_files)
    save_file_list(os.path.join(output_dir, 'invalid_files.txt'), invalid_files)
    save_file_list(os.path.join(output_dir, 'corrupted_files.txt'), corrupted_files)

def collect_video_files(data_dir):
    video_files = []
    for casual_conversations_name in sorted(os.listdir(data_dir)):
        casual_conversations_dir = os.path.join(data_dir, casual_conversations_name)
        for sample_id in sorted(os.listdir(casual_conversations_dir)):
            sample_dir = os.path.join(casual_conversations_dir, sample_id)
            for video_name in sorted(os.listdir(sample_dir)):
                video_files.append(os.path.join(sample_dir, video_name))
    return video_files

def process_video_file(video_file, data_dir, encoded_dir, output_dir):
    output_video_folder = video_file.replace(data_dir, output_dir).replace('.MP4', '').replace('.mp4', '')

    if os.path.exists(output_video_folder):
        num_processed_frames = len([x for x in os.listdir(output_video_folder) if x.endswith('.json')])
    else:
        num_processed_frames = -1

    encoded_video_file = video_file.replace(data_dir, encoded_dir)

    if os.path.exists(encoded_video_file):
        video_file = encoded_video_file

    cap = cv2.VideoCapture(video_file)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    ## corrupted video
    if total_frames == 0:
        return video_file, -1

    if num_processed_frames == total_frames:
        return video_file, 1
    else:
        return video_file, 0

def process_video_files_parallel(video_files, data_dir, encoded_dir, output_dir):
    valid_files = []
    invalid_files = []
    corrupted_files = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
        futures = [executor.submit(process_video_file, video_file, data_dir, encoded_dir, output_dir) for video_file in video_files]
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(video_files)):
            video_file, is_valid = future.result()
            if is_valid == 1:
                valid_files.append(video_file)
            elif is_valid == 0:
                invalid_files.append(video_file)
            elif is_valid == -1:
                corrupted_files.append(video_file)

    valid_files = sorted(valid_files)
    invalid_files = sorted(invalid_files)
    corrupted_files = sorted(corrupted_files)

    return valid_files, invalid_files, corrupted_files

def save_file_list(filepath, file_list):
    with open(filepath, 'w') as file:
        for video_file in file_list:
            file.write(video_file + '\n')

if __name__ == '__main__':
    main()

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import mimetypes
import os
import time
from argparse import ArgumentParser

import cv2
import json_tricks as json
import mmcv
import mmengine
import numpy as np
import subprocess

from tqdm import tqdm
import warnings
from mmpose.apis import inference_topdown
from mmpose.apis import init_model as init_pose_estimator
from mmpose.evaluation.functional import nms
from mmpose.registry import VISUALIZERS
from mmpose.structures import merge_data_samples, split_instances
from mmpose.utils import adapt_mmdet_pipeline

try:
    from mmdet.apis import inference_detector, init_detector
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False

warnings.filterwarnings("ignore", category=UserWarning, module='torchvision')
warnings.filterwarnings("ignore", category=UserWarning, module='mmengine')
warnings.filterwarnings("ignore", category=UserWarning, module='torch.functional')
warnings.filterwarnings("ignore", category=UserWarning, module='json_tricks.encoders')

def process_one_image(args,
                      img,
                      detector,
                      pose_estimator,
                      visualizer=None,
                      show_interval=0):
    """Visualize predicted keypoints (and heatmaps) of one image."""

    # predict bbox
    det_result = inference_detector(detector, img)
    pred_instance = det_result.pred_instances.cpu().numpy()
    bboxes = np.concatenate(
        (pred_instance.bboxes, pred_instance.scores[:, None]), axis=1)
    bboxes = bboxes[np.logical_and(pred_instance.labels == args.det_cat_id,
                                   pred_instance.scores > args.bbox_thr)]
    bboxes = bboxes[nms(bboxes, args.nms_thr), :4]

    # predict keypoints
    pose_results = inference_topdown(pose_estimator, img, bboxes)
    data_samples = merge_data_samples(pose_results)

    if visualizer is not None:
        # show the results
        if isinstance(img, str):
            img = mmcv.imread(img, channel_order='rgb')
        elif isinstance(img, np.ndarray):
            img = mmcv.bgr2rgb(img)

        visualizer.add_datasample(
            'result',
            img,
            data_sample=data_samples,
            draw_gt=False,
            draw_heatmap=args.draw_heatmap,
            draw_bbox=args.draw_bbox,
            show_kpt_idx=args.show_kpt_idx,
            skeleton_style=args.skeleton_style,
            show=False,
            wait_time=show_interval,
            kpt_thr=args.kpt_thr)

    # if there is no instance detected, return None
    return data_samples.get('pred_instances', None)


def main():
    """Visualize the demo images.

    Using mmdet to detect the human.
    """
    parser = ArgumentParser()
    parser.add_argument('det_config', help='Config file for detection')
    parser.add_argument('det_checkpoint', help='Checkpoint file for detection')
    parser.add_argument('pose_config', help='Config file for pose')
    parser.add_argument('pose_checkpoint', help='Checkpoint file for pose')
    parser.add_argument(
        '--start_index', type=int, default='', help='start video index')
    parser.add_argument(
        '--end_index', type=int, default='', help='end video index')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--det-cat-id',
        type=int,
        default=0,
        help='Category id for bounding box detection model')
    parser.add_argument(
        '--bbox-thr',
        type=float,
        default=0.3,
        help='Bounding box score threshold')
    parser.add_argument(
        '--nms-thr',
        type=float,
        default=0.3,
        help='IoU threshold for bounding box NMS')
    parser.add_argument(
        '--kpt-thr',
        type=float,
        default=0.3,
        help='Visualizing keypoint thresholds')
    parser.add_argument(
        '--draw-heatmap',
        action='store_true',
        default=False,
        help='Draw heatmap predicted by the model')
    parser.add_argument(
        '--show-kpt-idx',
        action='store_true',
        default=False,
        help='Whether to show the index of keypoints')
    parser.add_argument(
        '--skeleton-style',
        default='mmpose',
        type=str,
        choices=['mmpose', 'openpose'],
        help='Skeleton style selection')
    parser.add_argument(
        '--radius',
        type=int,
        default=3,
        help='Keypoint radius for visualization')
    parser.add_argument(
        '--thickness',
        type=int,
        default=1,
        help='Link thickness for visualization')
    parser.add_argument(
        '--show-interval', type=int, default=0, help='Sleep seconds per frame')
    parser.add_argument(
        '--alpha', type=float, default=0.8, help='The transparency of bboxes')
    parser.add_argument(
        '--draw-bbox', action='store_true', help='Draw bboxes of instances')
    parser.add_argument(
        '--visualize', action='store_true', help='Visualize poses')

    assert has_mmdet, 'Please install mmdet to run the demo.'

    args = parser.parse_args()

    assert args.det_config is not None
    assert args.det_checkpoint is not None

    video_start_index = args.start_index
    video_end_index = args.end_index

    visualize = args.visualize

    video_files = []
    data_dir = '/uca/richardalex/data/casual_conversations/raw_videos'
    encoded_data_dir = '/uca/richardalex/data/casual_conversations/raw_videos_fixed'
    output_dir = '/home/rawalk/Desktop/sapiens/pose/Outputs/pose_process/keypoints133'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for casual_conversations_name in sorted(os.listdir(data_dir)):
        casual_conversations_dir = os.path.join(data_dir, casual_conversations_name)

        for sample_id in sorted(os.listdir(casual_conversations_dir)):
            sample_dir = os.path.join(casual_conversations_dir, sample_id)

            for video_name in sorted(os.listdir(sample_dir)):
                video_files.append(os.path.join(sample_dir, video_name))

    ##---------------------------------------------------
    unprocessed_video_files = []
    invalid_file_path = os.path.join(output_dir, 'invalid_files.txt')
    corrupted_file_path = os.path.join(output_dir, 'corrupted_files.txt')

    with open(invalid_file_path, 'r') as f:
        invalid_video_files = [x.strip() for x in f.readlines()]

    with open(corrupted_file_path, 'r') as f:
        corrupted_video_files = [x.strip() for x in f.readlines()]

    # unprocessed_video_files = invalid_video_files + corrupted_video_files
    unprocessed_video_files = corrupted_video_files

    print('Total unprocessed videos: {}'.format(len(unprocessed_video_files)))
    video_files = sorted(unprocessed_video_files)

    ##---------------------------------------------------
    print('Loaded {} videos'.format(len(video_files)))

    if video_end_index == -1:
        video_end_index = len(video_files) - 1

    video_files = video_files[video_start_index:video_end_index + 1]

    print('Total valid videos: {}'.format(len(video_files)))

    # build detector
    detector = init_detector(
        args.det_config, args.det_checkpoint, device=args.device)
    detector.cfg = adapt_mmdet_pipeline(detector.cfg)

    # build pose estimator
    pose_estimator = init_pose_estimator(
        args.pose_config,
        args.pose_checkpoint,
        override_ckpt_meta=True, # dont load the checkpoint meta data, load from config file
        device=args.device,
        cfg_options=dict(
            model=dict(test_cfg=dict(output_heatmaps=args.draw_heatmap))))

    if visualize:
        # build visualizer
        pose_estimator.cfg.visualizer.radius = args.radius
        pose_estimator.cfg.visualizer.alpha = args.alpha
        pose_estimator.cfg.visualizer.line_width = args.thickness
        visualizer = VISUALIZERS.build(pose_estimator.cfg.visualizer)
        # the dataset_meta is loaded from the checkpoint and
        # then pass to the model in init_pose_estimator
        visualizer.set_dataset_meta(pose_estimator.dataset_meta, skeleton_style=args.skeleton_style)

    for video_file in video_files:
        casual_conversations_name = video_file.split('/')[-3]
        sample_id = video_file.split('/')[-2]
        video_name = os.path.basename(video_file).replace('.mp4', '').replace('.MP4', '')

        encoded_video_file = video_file.replace(data_dir, encoded_data_dir)

        if os.path.exists(encoded_video_file):
            print('Using encoded video file: <{}'.format(encoded_video_file))
            video_file = encoded_video_file

        else:
            print('Encoding video file: <{}'.format(video_file))
            ## encode the video
            this_output_dir = os.path.dirname(encoded_video_file)
            if not os.path.exists(this_output_dir):
                os.makedirs(this_output_dir)

            # FFmpeg command to re-encode the video
            command = ['ffmpeg', '-i', video_file, '-c:v', 'libx264', '-preset', 'fast', '-crf', '22', encoded_video_file]

            # Run the FFmpeg command
            subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            video_file = encoded_video_file

        frame_idx = 0
        cap = cv2.VideoCapture(video_file)
        frame_rate = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print('Processing video {}: frame_rate: {} fps, total frames: {}'.format(video_file, frame_rate, total_frames))

        is_corrupted = False

        with tqdm(total=total_frames) as pbar:
            while cap.isOpened():
                success, frame = cap.read()

                if not success:
                    if frame_idx != total_frames or total_frames == 0:
                        print('Video file {} is corrupted'.format(video_file))
                        is_corrupted = True
                    break

                pred_save_dir = os.path.join(output_dir, casual_conversations_name, sample_id, video_name)
                os.makedirs(pred_save_dir, exist_ok=True)
                pred_save_path = os.path.join(pred_save_dir, '{:08d}'.format(frame_idx) + '.json')

                ## skip if already processed
                if os.path.exists(pred_save_path):
                    frame_idx += 1
                    pbar.update(1)
                    continue

                ## process one image
                if visualize == False:
                    pred_instances = process_one_image(args, frame, detector, pose_estimator)
                elif visualize == True:
                    pred_instances = process_one_image(args, frame, detector, pose_estimator, visualizer) ## visualize

                pred_instances_list = split_instances(pred_instances)

                with open(pred_save_path, 'w') as f:
                    json.dump(
                        dict(
                            meta_info=pose_estimator.dataset_meta,
                            instance_info=pred_instances_list),
                        f,
                        indent='\t')

                if visualize == True:
                    output_file = os.path.join(pred_save_dir, '{:08d}'.format(frame_idx) + '.jpg')
                    img_vis = visualizer.get_image()
                    mmcv.imwrite(mmcv.rgb2bgr(img_vis), output_file)

                frame_idx += 1
                pbar.update(1)

        cap.release()

        print('Done! Processed video {}: {} frames, {} total_frames: is_corrupted: {}'.format(video_file, frame_idx, total_frames, is_corrupted))


if __name__ == '__main__':
    main()

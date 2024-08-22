# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import mimetypes
import os
import time
import warnings
from argparse import ArgumentParser
from collections import defaultdict
from typing import List, Optional, Sequence, Union
import copy
import subprocess

import cv2
import json_tricks as json
import mmcv
import mmengine
import numpy as np
import torch
import torch.nn as nn

from timer import Timer
from mmcv.ops import RoIPool
from mmengine.dataset import Compose, pseudo_collate
from mmengine.registry import init_default_scope
from mmpose.apis import inference_topdown, init_model as init_pose_estimator
from mmpose.evaluation.functional import nms
from mmpose.registry import VISUALIZERS
from mmpose.structures import merge_data_samples, PoseDataSample, split_instances
from mmpose.structures.bbox import bbox_xywh2xyxy
from mmpose.utils import adapt_mmdet_pipeline
from PIL import Image

from tqdm import tqdm

try:
    from mmdet.apis import inference_detector, init_detector
    from mmdet.structures import DetDataSample, SampleList
    from mmdet.utils import get_test_pipeline_cfg

    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False


ImagesType = Union[str, np.ndarray, Sequence[str], Sequence[np.ndarray]]

warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")
warnings.filterwarnings("ignore", category=UserWarning, module="mmengine")
warnings.filterwarnings("ignore", category=UserWarning, module="torch.functional")
warnings.filterwarnings("ignore", category=UserWarning, module="json_tricks.encoders")

if torch.cuda.is_available():
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)


def inference_detector(
    model: nn.Module,
    imgs: ImagesType,
    test_pipeline: Optional[Compose] = None,
    text_prompt: Optional[str] = None,
    custom_entities: bool = False,
):
    """Inference image(s) with the detector.

    Args:
        model (nn.Module): The loaded detector.
        imgs (str, ndarray, Sequence[str/ndarray]):
           Either image files or loaded images.
        test_pipeline (:obj:`Compose`): Test pipeline.

    Returns:
        :obj:`DetDataSample` or list[:obj:`DetDataSample`]:
        If imgs is a list or tuple, the same length list type results
        will be returned, otherwise return the detection results directly.
    """

    if isinstance(imgs, (list, tuple)):
        is_batch = True
    else:
        imgs = [imgs]
        is_batch = False

    cfg = model.cfg

    if test_pipeline is None:
        cfg = cfg.copy()
        test_pipeline = get_test_pipeline_cfg(cfg)
        if isinstance(imgs[0], np.ndarray):
            # Calling this method across libraries will result
            # in module unregistered error if not prefixed with mmdet.
            test_pipeline[0].type = "mmdet.LoadImageFromNDArray"

        test_pipeline = Compose(test_pipeline)

    if model.data_preprocessor.device.type == "cpu":
        for m in model.modules():
            assert not isinstance(
                m, RoIPool
            ), "CPU inference with RoIPool is not supported currently."

    result_list = []
    for i, img in enumerate(imgs):
        # prepare data
        if isinstance(img, np.ndarray):
            # TODO: remove img_id.
            data_ = dict(img=img, img_id=0)
        else:
            # TODO: remove img_id.
            data_ = dict(img_path=img, img_id=0)

        if text_prompt:
            data_["text"] = text_prompt
            data_["custom_entities"] = custom_entities

        # build the data pipeline
        data_ = test_pipeline(data_)

        data_["inputs"] = [data_["inputs"]]
        data_["data_samples"] = [data_["data_samples"]]

        # forward the model
        with torch.no_grad():
            results = model.test_step(data_)[0]

        result_list.append(results)

    if not is_batch:
        return result_list[0]
    else:
        return result_list


def batch_inference_topdown(
    model: nn.Module,
    imgs: List[Union[np.ndarray, str]],
    bboxes_list: Optional[Union[List, np.ndarray]] = None,
    bbox_format: str = "xyxy",
) -> List[List[PoseDataSample]]:
    """Inference image with a top-down pose estimator.

    Args:
        model (nn.Module): The top-down pose estimator
        imgs (List[np.ndarray | str]): The loaded image or image file to inference
        bboxes_list (np.ndarray, optional): The bboxes in shape (N, 4), each row
            represents a bbox. If not given, the entire image will be regarded
            as a single bbox area. Defaults to ``None``
        bbox_format (str): The bbox format indicator. Options are ``'xywh'``
            and ``'xyxy'``. Defaults to ``'xyxy'``

    Returns:
        List[List[:obj:`PoseDataSample`]]: The inference results. Specifically, the
        predicted keypoints and scores are saved at
        ``data_sample.pred_instances.keypoints`` and
        ``data_sample.pred_instances.keypoint_scores``.
    """
    assert bbox_format in {"xyxy", "xywh"}, f'Invalid bbox_format "{bbox_format}".'

    scope = model.cfg.get("default_scope", "mmpose")

    if scope is not None:
        init_default_scope(scope)
    pipeline = Compose(model.cfg.test_dataloader.dataset.pipeline)

    # construct batch data samples
    data_list = []
    img_bbox_map = defaultdict(int)
    for i, img in enumerate(imgs):

        if bboxes_list[i] is None or len(bboxes_list[i]) == 0:
            # get bbox from the image size
            if isinstance(img, str):
                w, h = Image.open(img).size
            else:
                h, w = img.shape[:2]

            bboxes = np.array([[0, 0, w, h]], dtype=np.float32)
        else:
            bboxes = bboxes_list[i]
            if isinstance(bboxes, list):
                bboxes = np.array(bboxes)

            if bbox_format == "xywh":
                bboxes = bbox_xywh2xyxy(bboxes)

        img_bbox_map[i] = len(bboxes)
        for bbox in bboxes:
            if isinstance(img, str):
                data_info = dict(img_path=img)
            else:
                data_info = dict(img=img)
            data_info["bbox"] = bbox[None]  # shape (1, 4)
            data_info["bbox_score"] = np.ones(1, dtype=np.float32)  # shape (1,)
            data_info.update(model.dataset_meta)
            data_list.append(pipeline(data_info))

    if data_list:
        # collate data list into a batch, which is a dict with following keys:
        # batch['inputs']: a list of input images
        # batch['data_samples']: a list of :obj:`PoseDataSample`
        batch = pseudo_collate(data_list)
    elapsed_time = 0
    if data_list:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            start_event.record()
        with torch.no_grad():
            results = model.test_step(batch)
        
        if torch.cuda.is_available():
            end_event.record()
            torch.cuda.synchronize()
            elapsed_time = start_event.elapsed_time(end_event)
    else:
        results = []

    batched_results = []
    for _, bbox_len in img_bbox_map.items():
        batched_results.append(results[:bbox_len].copy())
        del results[:bbox_len]
    return batched_results, elapsed_time


def process_one_image_bbox(pred_instance, det_cat_id, bbox_thr, nms_thr):
    bboxes = np.concatenate(
        (pred_instance.bboxes, pred_instance.scores[:, None]), axis=1
    )
    bboxes = bboxes[
        np.logical_and(
            pred_instance.labels == det_cat_id,
            pred_instance.scores > bbox_thr,
        )
    ]
    bboxes = bboxes[nms(bboxes, nms_thr), :4]


def process_images(
    args, imgs, detector, pose_estimator, visualizer=None, compiled=False,
):
    """Visualize predicted keypoints (and heatmaps) of one image."""
    if compiled:
        fake_batch = [np.zeros((imgs[0].shape))] * (32 - len(imgs))# since model is compiled statically with 32 batch size, if changed recompilation occurs
        frame_batch = imgs + fake_batch
    else:
        frame_batch = imgs
    # predict bbox
    det_results = inference_detector(detector, frame_batch)
    pred_instances = list(
        map(lambda det_result: det_result.pred_instances.cpu().numpy(), det_results)
    )
    bboxes_batch = list(
        map(
            lambda pred_instance: process_one_image_bbox(
                pred_instance, args.det_cat_id, args.bbox_thr, args.nms_thr
            ),
            pred_instances,
        )
    )

    # predict keypoints
    pose_results, elapsed_time = batch_inference_topdown(pose_estimator, frame_batch, bboxes_batch)
    data_samples_list = list(map(merge_data_samples, pose_results))

    assert len(frame_batch) == len(data_samples_list), f"{len(frame_batch)} != {len(data_samples_list)}"
    vis_images = []
    for img, data_samples in zip(imgs, data_samples_list):

        # show the results
        if isinstance(img, str):
            img = mmcv.imread(img, channel_order="rgb")
        elif isinstance(img, np.ndarray):
            img = mmcv.bgr2rgb(img)

        if visualizer is not None:
            visualizer.add_datasample(
                "result",
                img,
                data_sample=data_samples,
                draw_gt=False,
                draw_heatmap=args.draw_heatmap,
                draw_bbox=args.draw_bbox,
                show_kpt_idx=args.show_kpt_idx,
                skeleton_style=args.skeleton_style,
                kpt_thr=args.kpt_thr,
            )
            vis_images.append(visualizer.get_image())

    # if there is no instance detected, return None
    if visualizer is not None:
       return vis_images, elapsed_time/len(frame_batch)
    return list(
        map(
            lambda data_samples: data_samples.get("pred_instances", None),
            data_samples_list,
        )
    ), elapsed_time


def main():
    """Visualize the demo images.

    Using mmdet to detect the human.
    """
    parser = ArgumentParser()
    parser.add_argument("det_config", help="Config file for detection")
    parser.add_argument("det_checkpoint", help="Checkpoint file for detection")
    parser.add_argument("pose_config", help="Config file for pose")
    parser.add_argument("pose_checkpoint", help="Checkpoint file for pose")
    parser.add_argument("--video", type=str, default="", help="Video file")
    parser.add_argument(
        "--output-file",
        type=str,
        help="root of the output vid file. ",
    )
    parser.add_argument(
        '--output-fourcc',
        default='MJPG',
        type=str,
        help='Fourcc of the output video')
    parser.add_argument(
        '--output-fps', default=-1, type=int, help='FPS of the output video')
    parser.add_argument(
        '--output-height',
        default=-1,
        type=int,
        help='Frame height of the output video')
    parser.add_argument(
        '--output-width',
        default=-1,
        type=int,
        help='Frame width of the output video')
    parser.add_argument("--device", default="cuda:0", help="Device used for inference")
    parser.add_argument(
        "--det-cat-id",
        type=int,
        default=0,
        help="Category id for bounding box detection model",
    )
    parser.add_argument(
        "--bbox-thr", type=float, default=0.3, help="Bounding box score threshold"
    )
    parser.add_argument(
        "--nms-thr", type=float, default=0.3, help="IoU threshold for bounding box NMS"
    )
    parser.add_argument(
        "--kpt-thr", type=float, default=0.3, help="Visualizing keypoint thresholds"
    )
    parser.add_argument(
        "--draw-heatmap",
        action="store_true",
        default=False,
        help="Draw heatmap predicted by the model",
    )
    parser.add_argument(
        "--show-kpt-idx",
        action="store_true",
        default=False,
        help="Whether to show the index of keypoints",
    )
    parser.add_argument(
        "--skeleton-style",
        default="mmpose",
        type=str,
        choices=["mmpose", "openpose"],
        help="Skeleton style selection",
    )
    parser.add_argument(
        "--radius", type=int, default=3, help="Keypoint radius for visualization"
    )
    parser.add_argument(
        "--thickness", type=int, default=1, help="Link thickness for visualization"
    )
    parser.add_argument(
        "--show-interval", type=int, default=0, help="Sleep seconds per frame"
    )
    parser.add_argument(
        "--alpha", type=float, default=0.8, help="The transparency of bboxes"
    )
    parser.add_argument(
        "--draw-bbox", action="store_true", help="Draw bboxes of instances"
    )

    assert has_mmdet, "Please install mmdet to run the demo."

    args = parser.parse_args()

    assert args.det_config is not None
    assert args.det_checkpoint is not None

    if args.video.isdigit():
        args.video = int(args.video)
    cap = cv2.VideoCapture(args.video)
    assert (cap.isOpened())
    input_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    input_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    input_fps = cap.get(cv2.CAP_PROP_FPS)

    # init output video
    writer = None
    output_height = None
    output_width = None
    if args.output_file is not None:
        os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*args.output_fourcc)
        output_fps = args.output_fps if args.output_fps > 0 else input_fps
        output_height = args.output_height if args.output_height > 0 else int(
            input_height)
        output_width = args.output_width if args.output_width > 0 else int(
            input_width)
        writer = cv2.VideoWriter(args.output_file, fourcc, output_fps,
                                 (output_width*3, output_height), True)

    detector = init_detector(
        args.det_config, args.det_checkpoint, device=args.device
    )
    detector.cfg = adapt_mmdet_pipeline(detector.cfg)

    pose_estimator = init_pose_estimator(
        args.pose_config,
        args.pose_checkpoint,
        override_ckpt_meta=True,  # dont load the checkpoint meta data, load from config file
        device=args.device,
        cfg_options=dict(
            model=dict(test_cfg=dict(output_heatmaps=args.draw_heatmap))
        ),
    )
    pose_compiled = copy.deepcopy(pose_estimator)
    pose_compiled.to(torch.bfloat16)
    torch.compile(pose_compiled, mode="max-autotune", fullgraph=True)

    # build visualizer
    pose_estimator.cfg.visualizer.radius = args.radius
    pose_estimator.cfg.visualizer.alpha = args.alpha
    pose_estimator.cfg.visualizer.line_width = args.thickness
    visualizer = VISUALIZERS.build(pose_estimator.cfg.visualizer)
    # the dataset_meta is loaded from the checkpoint and
    # then pass to the model in init_pose_estimator
    visualizer.set_dataset_meta(
        pose_estimator.dataset_meta, skeleton_style=args.skeleton_style
    )

    pose_compiled.cfg.visualizer.radius = args.radius
    pose_compiled.cfg.visualizer.alpha = args.alpha
    pose_compiled.cfg.visualizer.line_width = args.thickness
    visualizer_compiled = VISUALIZERS.build(pose_compiled.cfg.visualizer)
    # the dataset_meta is loaded from the checkpoint and
    # then pass to the model in init_pose_estimator
    visualizer_compiled.set_dataset_meta(
        pose_compiled.dataset_meta, skeleton_style=args.skeleton_style
    )

    i = 0

    # start looping
    try:
        while True:
            flag, frame = cap.read()
            if not flag:
                break
            
            if i%50 == 0:
                print(f"Processing Frame {i}")
            
            shape = frame.shape
            # frame = frame.transpose(2, 0, 1)
            frame = [frame]
            
            # test a single image
            draw_imgs, original_elapsed_time = process_images(
            args, frame, detector, pose_estimator, visualizer)
            
            # test a single image
            with torch.autocast("cuda"):
                draw_imgs_compile, compile_elapsed_time = process_images(
            args, frame, detector, pose_compiled, visualizer_compiled, compiled=True)

            frame = frame[0] #.transpose(1, 2, 0)
            # result = result[0]#.transpose(1, 2, 0)
            # optimized_result = optimized_result[0]#.transpose(1, 2, 0)

            # blend raw image and prediction
            draw_img = cv2.cvtColor(draw_imgs[0], cv2.COLOR_BGR2RGB)

            if original_elapsed_time:
                org_offset = int(draw_img.shape[0] * 0.1), int(draw_img.shape[1] * 0.85)
                draw_img = cv2.putText(draw_img, f"Inference Time taken by original model: {original_elapsed_time:.2f}ms", (draw_img.shape[1] - org_offset[1], org_offset[0]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
            draw_img_compile = cv2.cvtColor(draw_imgs_compile[0], cv2.COLOR_BGR2RGB)
            if compile_elapsed_time:
                org_offset = int(draw_img_compile.shape[0] * 0.1), int(draw_img_compile.shape[1] * 0.85)
                draw_img_compile = cv2.putText(draw_img_compile, f"Inference Time taken by optimized model: {compile_elapsed_time:.2f}ms", (draw_img_compile.shape[1] - org_offset[1], org_offset[0]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            # if args.show:
            #     cv2.imshow('video_demo', draw_img)
            #     cv2.waitKey(args.show_wait_time)
            if writer:
                if draw_img.shape[0] != output_height or draw_img.shape[
                        1] != output_width or draw_img_compile.shape[0] != output_height or draw_img_compile.shape[1] != output_width :
                    draw_img = cv2.resize(draw_img,
                                          (output_width, output_height))
                    draw_img_compile = cv2.resize(draw_img_compile,
                                          (output_width, output_height))
                vis_image = np.concatenate([frame, draw_img, draw_img_compile], axis=1)
                writer.write(vis_image)
            i +=1
    except BaseException as e:
        raise e
    finally:
        if writer:
            writer.release()
            ffmpeg_cmd = ["ffmpeg", "-y", "-i", args.output_file, "-vcodec", "libx265", "-crf", "28", os.path.join(os.path.dirname(args.output_file), os.path.basename(args.output_file).split(".")[0] + "_compressed.mp4")]
            subprocess.run(ffmpeg_cmd)
        cap.release()

if __name__ == "__main__":
    main()

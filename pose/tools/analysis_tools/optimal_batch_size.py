# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import os.path as osp
import itertools
import time
import json
from collections import defaultdict

from memory_error_utils import is_oom_error, garbage_collection_cuda

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from tqdm import tqdm
from mmcv.ops import RoIPool
from mmengine import Config
from mmengine.dataset import Compose, pseudo_collate
from mmengine.fileio import dump
from mmengine.model.utils import revert_sync_batchnorm
from mmengine.registry import init_default_scope
from mmengine.runner import Runner, load_checkpoint
from mmengine.utils import mkdir_or_exist
from mmdet.apis import init_detector
from mmdet.structures import DetDataSample, SampleList
from mmdet.utils import get_test_pipeline_cfg
from mmpose.evaluation.functional import nms
from mmpose.structures.bbox import bbox_xywh2xyxy
from mmpose.apis import init_model as init_pose_estimator
from mmpose.utils import adapt_mmdet_pipeline
from PIL import Image

from mmseg.registry import MODELS


def _demo_mm_inputs(input_shape):
    """Create a superset of inputs needed to run test or train batches.

    Args:
        input_shape (tuple):
            input batch dimensions
        num_classes (int):
            number of semantic classes
    """
    (N, C, H, W) = input_shape
    rng = np.random.RandomState(0)
    imgs = rng.rand(N, C, H, W)
    if torch.cuda.is_available():
        imgs = torch.Tensor(imgs).cuda()
    return imgs

def warmup_model(model, img_shape):
    print("Warming up ...")
    for _ in range(5):
        inputs = _demo_mm_inputs(img_shape)
        # Try fit
        with torch.no_grad():
            model(inputs)
    del inputs


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


def prep_batch_inference_topdown(
    model,
    imgs,
    bboxes_list,
    bbox_format="xyxy",
):
    """Prep data for Inference image with a top-down pose estimator.

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
        return batch

def inference_detector(
    model,
    imgs,
    test_pipeline=None,
    text_prompt=None,
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


def read_images_in_batch_for_pose(detector, pose_estimator, img_dir, n_imgs, det_cat_id, bbox_thr, nms_thr):
    """Read images from a directory as an infinite generator.
    
    Args:
        img_dir (str): Path to the image directory.
        n_imgs (int): Number of images to read.
        
    Returns:
        list[Tensor]: A list of loaded images.
    """
    assert isinstance(img_dir, str) and osp.isdir(img_dir), \
            f'Expect "img_dir" must be a path to a directory, but got {img_dir}'
    img_paths = [osp.join(img_dir, file) for file in os.listdir(img_dir) if file.endswith(".png") or file.endswith(".jpg") or file.endswith(".jpeg")]
    resize = transforms.Resize((1024, 768))

    while True:
        imgs = []
        sampled_img_paths = list(np.random.choice(img_paths, n_imgs, replace=True))
        # for img_path in sampled_img_paths: 
        #     img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        #     img = np.transpose(img, (2, 0, 1))
        #     imgs.append(resize(torch.Tensor(img)))

        det_results = inference_detector(detector, sampled_img_paths)
        pred_instances = list(
            map(lambda det_result: det_result.pred_instances.cpu().numpy(), det_results)
        )
        bboxes_batch = list(
            map(
                lambda pred_instance: process_one_image_bbox(
                    pred_instance, det_cat_id, bbox_thr, nms_thr
                ),
                pred_instances,
            )
        )
        yield prep_batch_inference_topdown(pose_estimator, sampled_img_paths, bboxes_batch)
        del sampled_img_paths, pred_instances, bboxes_batch, det_results
    

def run_power_scaling_pose(det_model, pose_model, batch_size, img_dir, det_cat_id, bbox_thr, nms_thr, max_trials=100, warmup=5, device=None):
    """ Batch scaling mode where the size is doubled at each iteration until an
        OOM error is encountered. """
    high = None
    best_spi = float('inf')
    best_batch_size = None
    low=0
    det_model.eval()
    pose_model.eval()
    optim_batch_dict = {"warmup": warmup, "max_trials": max_trials}
    while True:
        try:
            img_gen = read_images_in_batch_for_pose(det_model, pose_model, img_dir, batch_size, det_cat_id, bbox_thr, nms_thr)
            spi_list = []
            fps_list = []
            print(f"Running batch size {batch_size}")
            # warmup_model(model, (batch_size, 3, 1024, 768))
            garbage_collection_cuda()
            for i in tqdm(range(warmup + max_trials)):
                inputs = next(img_gen)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                start = time.perf_counter()
                with torch.no_grad():
                    pose_model.test_step(inputs)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                time_taken = time.perf_counter() - start
                spi_list.append(time_taken/batch_size)
                fps_list.append(batch_size/time_taken)
                # inputs = inputs.detach().cpu()
                del inputs
            img_gen.close()
            avg_spi = str(np.mean(spi_list[warmup:])).format("{:.2e}")
            std_spi = str(np.std(fps_list[warmup:])).format("{:.2e}")
            avg_fps = str(np.mean(fps_list[warmup:])).format("{:.2e}")
            std_fps = str(np.std(fps_list[warmup:])).format("{:.2e}")
            optim_batch_dict[f"batch_{batch_size}"] = {
                "batch_size": batch_size,
                "avg_spi": avg_spi,
                "std_spi": std_spi,
                "avg_fps": avg_fps,
                "std_fps": std_fps,
                "spi_list": spi_list,
                "fps_list": fps_list,
            }
            print(f"Successfully ran batch size {batch_size} with {avg_spi} secs per image and std of {std_spi}")
            print(f"Successfully ran batch size {batch_size} with {avg_fps} fps and std of {std_fps}")

            # if memory_optimized:
            #     best_batch_size = batch_size
            #     best_spi = avg_spi
            # elif best_spi > avg_spi:
            #         best_spi = avg_spi
            #         best_batch_size = batch_size
            # else:
            #     # fake OOM error
            #     print("Reducing batch size to improve speed")
            #     raise RuntimeError("For speed optimizaton")

            # Double in size
            # batch_size *= 2
        except RuntimeError as exception:
            # Only these errors should trigger an adjustment
            if is_oom_error(exception):
                # If we fail in power mode, return trying with a middle batch size
                
                garbage_collection_cuda()
                high = batch_size
                batch_size = (low+high) // 2
                print(f"Trying batch size {batch_size}")
                if high - low <=1:
                    break
            else:
                raise  # some other error not memory related
        else:
            low = batch_size
            if high:
                if high - low <=1:
                    break
                batch_size = (low + high) // 2
            else:
                batch_size *= 2
    return optim_batch_dict


def error_plt_batches(optim_batch_dict, output_dir):
    """Plot batch size vs avg spi and std of spi"""
    batch_size = []
    avg_spi = []
    std_spi = []
    
    for key in optim_batch_dict:
        if "batch" not in key:
            continue
        
        batch_size.append(optim_batch_dict[key]["batch_size"])
        avg_spi.append(float(optim_batch_dict[key]["avg_spi"]))
        std_spi.append(float(optim_batch_dict[key]["std_spi"]))
    
    plt.figure(figsize=(10, 6), dpi=80)
    plt.errorbar(batch_size, avg_spi, std_spi, label="avg spi", linestyle='None', marker='^', markersize=15)
    plt.xlabel("Batch size", fontsize=15)
    plt.ylabel("secs/img", fontsize=15)
    plt.xticks(batch_size)
    plt.savefig(os.path.join(output_dir, "optim_batch_size_spi.png"), dpi=80)
    batch_size = []
    avg_spi = []
    std_spi = []
    
    for key in optim_batch_dict:
        if "batch" not in key:
            continue
        
        batch_size.append(optim_batch_dict[key]["batch_size"])
        avg_spi.append(float(optim_batch_dict[key]["avg_fps"]))
        std_spi.append(float(optim_batch_dict[key]["std_fps"]))
    
    plt.figure(figsize=(10, 6), dpi=80)
    plt.errorbar(batch_size, avg_spi, std_spi, label="avg fps", linestyle='None', marker='^', markersize=15)
    plt.xlabel("Batch size", fontsize=15)
    plt.ylabel("FPS", fontsize=15)
    plt.xticks(batch_size)
    plt.savefig(os.path.join(output_dir, "optim_batch_size_fps.png"), dpi=80)

def parse_args():
    parser = argparse.ArgumentParser(description='MMPose benchmark a model')
    parser.add_argument("det_config", help="Config file for detection")
    parser.add_argument("det_checkpoint", help="Checkpoint file for detection")
    parser.add_argument("pose_config", help="Config file for pose")
    parser.add_argument("pose_checkpoint", help="Checkpoint file for pose")
    parser.add_argument(
        '--img_dir',
        '--img-dir',
        type=str,
        help='input image directory')
    parser.add_argument(
        '--output_dir',
        '--output-dir',
        type=str,
        help='input image directory')
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
    args = parser.parse_args()
    return args



def main():
    args = parse_args()

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    device = "cpu"
    if torch.cuda.is_available():
        device="cuda"

    torch.backends.cudnn.benchmark = True

    # build the model and load checkpoint
    # build detector
    detector = init_detector(
        args.det_config, args.det_checkpoint, device=device
    )
    detector.cfg = adapt_mmdet_pipeline(detector.cfg)

    # build pose estimator
    pose_estimator = init_pose_estimator(
        args.pose_config,
        args.pose_checkpoint,
        override_ckpt_meta=True,  # dont load the checkpoint meta data, load from config file
        device=device,
        cfg_options=dict(
            model=dict(test_cfg=dict(output_heatmaps=False))
        ),
    )

    detector.eval()
    pose_estimator.eval()

    print("Running batch size finder ...")

    optimal_batch_size_dict = run_power_scaling_pose(detector, pose_estimator, 1, args.img_dir, args.det_cat_id, args.bbox_thr, args.nms_thr, device="cuda" if torch.cuda.is_available() else "cpu")
    # print(f"Optimal batch size: {optimal_batch_size_dict}")
    json.dump(optimal_batch_size_dict, open(osp.join(output_dir, "optim_batch_dict.json"), 'w'), indent=4)
    error_plt_batches(optimal_batch_size_dict, output_dir)
    print(f"Successfully saved results to {output_dir}")


if __name__ == '__main__':
    main()

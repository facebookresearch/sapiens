# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import itertools
import json
import os
import os.path as osp
import time
from pathlib import Path
from typing import Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

import torch.nn as nn
import torch.nn.utils.parametrize as parametrize
import torch.utils.benchmark as benchmark
from mmdet.apis import inference_detector, init_detector
from mmengine import Config
from mmengine.dataset import Compose, pseudo_collate
from mmengine.fileio import dump
from mmengine.model.utils import revert_sync_batchnorm
from mmengine.registry import init_default_scope
from mmengine.runner import get_state_dict, load_checkpoint, Runner, save_checkpoint
from mmengine.utils import mkdir_or_exist

# from mmseg.models import build_segmentor
from mmpose.apis import init_model as init_pose_estimator
from mmpose.utils import adapt_mmdet_pipeline

from torch._dynamo import is_compiling as dynamo_is_compiling
from torch._higher_order_ops.out_dtype import out_dtype
from torch.profiler import ProfilerActivity
from tqdm import tqdm


def _benchmark(model, inputs, model_name=""):
    # imgs = input["imgs"][0, ...].unsqueeze(0) if model_name.lower() == "original" else input["imgs"]
    if torch.cuda.is_available():
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        if isinstance(inputs, torch.Tensor):
            inputs = inputs.cuda()
        elif isinstance(inputs, dict):
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    inputs[k] = v.cuda()

    time_ = []

    # g = torch.cuda.CUDAGraph()
    # device = imgs.device
    # imgs = imgs.cpu()
    # rand = torch.randn(*imgs.shape, dtype=imgs.dtype, device=device)
    # with torch.cuda.graph(g):
    #     with torch.no_grad():
    #         model(rand)
    # rand.copy_(imgs)
    # g.replay()
    with torch.no_grad():
        for _ in range(5):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                start_event.record()
            if isinstance(inputs, torch.Tensor):
                model(inputs)
            elif isinstance(inputs, dict):
                model(**inputs)
            end_event.record()
            if torch.cuda.is_available():
                end_event.record()
                torch.cuda.synchronize()
                time_.append(start_event.elapsed_time(end_event))
    if isinstance(inputs, torch.Tensor):
        inputs = inputs.cpu()
    elif isinstance(inputs, dict):
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.cpu()
    if isinstance(inputs, torch.Tensor):
        mean_time = np.mean(time_[1:]) / (len(inputs))
    elif isinstance(inputs, dict):
        mean_time = np.mean(time_[1:]) / (len(inputs["inputs"]))
    print(f"For {model_name} model, ")
    print(f"avg time is {mean_time} ms")
    print(f"Total time is {sum(time_)} ms")
    print(f"Each trial time: {time_}")
    return mean_time


def _convert_batchnorm(module):
    module_output = module
    if isinstance(module, torch.nn.SyncBatchNorm):
        module_output = torch.nn.BatchNorm2d(
            module.num_features,
            module.eps,
            module.momentum,
            module.affine,
            module.track_running_stats,
        )
        if module.affine:
            module_output.weight.data = module.weight.data.clone().detach()
            module_output.bias.data = module.bias.data.clone().detach()
            # keep requires_grad unchanged
            module_output.weight.requires_grad = module.weight.requires_grad
            module_output.bias.requires_grad = module.bias.requires_grad
        module_output.running_mean = module.running_mean
        module_output.running_var = module.running_var
        module_output.num_batches_tracked = module.num_batches_tracked
    for name, child in module.named_children():
        module_output.add_module(name, _convert_batchnorm(child))
    del module
    return module_output


def _demo_mm_inputs_det(input_shape):
    (N, C, H, W) = input_shape
    rng = np.random.RandomState(0)
    imgs = rng.rand(*(N, H, W, C))
    return torch.Tensor(imgs).to(torch.float)


def _demo_mm_inputs_pose(input_shape, dataset_meta, pipeline):
    (N, C, H, W) = input_shape
    rng = np.random.RandomState(0)
    imgs = rng.rand(*(N, H, W, C))
    # if num_classes > 1:
    #     segs = rng.randint(low=0, high=num_classes - 1, size=(N, 1, H, W)).astype(np.uint8)
    # else:
    #     segs = rng.uniform(0, 1, size=(N, 1, H, W)).astype(np.uint8)
    img_metas = [
        {
            "img_shape": (H, W, C),
            "ori_shape": (H, W, C),
            "pad_shape": (H, W, C),
            "filename": "<demo>.png",
            "scale_factor": 1.0,
            "flip": False,
        }
        for _ in range(N)
    ]
    data_list = []
    for img in imgs:
        bbox = np.array([0, 0, W, H], dtype=np.float32)
        data_info = dict(img=img)
        data_info["bbox"] = bbox[None]  # shape (1, 4)
        data_info["bbox_score"] = np.ones(1)  # shape (1,)
        data_info.update(dataset_meta)
        data_list.append(pipeline(data_info))
    mm_inputs = pseudo_collate(data_list)
    for key in mm_inputs.keys():
        if isinstance(mm_inputs[key], tuple):
            # convert to mutable
            mm_inputs[key] = list(mm_inputs[key])
        if isinstance(mm_inputs[key], list) and all(
            isinstance(data, torch.Tensor) for data in mm_inputs[key]
        ):
            mm_inputs[key] = torch.stack(mm_inputs[key], dim=0).to(torch.float)

    # print(mm_inputs['data_samples'].keys())
    return mm_inputs


def explain_model(model, inputs):
    # imgs = inputs["imgs"]
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            inputs[k] = v.cuda()
    with torch.no_grad(), torch.autocast("cuda"):
        explanation = torch._dynamo.explain(model)(**inputs)
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            inputs[k] = v.cpu()
    return explanation.graphs, explanation.graph_count, explanation.break_reasons


def compile_model(
    model,
    inputs,
    output_file="compiled_model.pt",
    max_batch_size=48,
    dtype=torch.bfloat16,
):
    # imgs = inputs["imgs"]
    modes = {"Default": "default", "RO": "reduce-overhead", "MA": "max-autotune"}
    min_mean = float("inf")
    best_mode = None
    kwargs = {}
    args = (inputs["inputs"],)
    dynamic_batch = torch.export.Dim("batch", min=1, max=max_batch_size)
    dynamic_shapes = {"inputs": {0: dynamic_batch}}
    exported_model = torch.export.export(
        model, args=args, kwargs=kwargs, dynamic_shapes=dynamic_shapes
    )
    for mode_str, mode in modes.items():
        print(f"Compiling model with {mode_str} mode")
        model = torch.compile(exported_model.module(), mode=mode)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s), torch.no_grad():
            for i in range(3):
                model(inputs["inputs"])
        torch.cuda.current_stream().wait_stream(s)
        mean = _benchmark(
            exported_model.module(), inputs["inputs"], model_name=mode_str
        )
        if mean < min_mean:
            min_mean = mean
            best_mode = mode_str
    print(f"Best compilation mode: {best_mode}")
    torch.export.save(exported_model, output_file)
    return model


def parse_args():
    parser = argparse.ArgumentParser(description="MMSeg sparsify a model")
    parser.add_argument("pose_config", help="Config file for pose")
    parser.add_argument("pose_checkpoint", help="Checkpoint file for pose")
    parser.add_argument(
        "--det-shape",
        type=int,
        nargs="+",
        default=[640, 640],
        help="input image size (height, width)",
    )
    parser.add_argument(
        "--pose-shape",
        type=int,
        nargs="+",
        default=[1024, 768],
        help="input image size (height, width)",
    )
    parser.add_argument(
        "--output_dir", "--output-dir", type=str, help="input image directory"
    )
    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=48,
        help="Maximum batch size for dynamic compile",
    )
    parser.add_argument(
        "--explain-verbose", action="store_true", help="Explains the model compilation"
    )
    parser.add_argument(
        "--force-compile",
        action="store_true",
        help="Force compile the model even if more than one cuda graphs are present",
    )
    parser.add_argument(
        "--fp16", action="store_true", help="To enable fp16. Default is bf16"
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    if len(args.pose_shape) == 1:
        pose_input_shape = (32, 3, args.pose_shape[0], args.pose_shape[0])
    elif len(args.pose_shape) == 2:
        pose_input_shape = (
            32,
            3,
        ) + tuple(args.pose_shape)
    else:
        raise ValueError("invalid pose input shape")

    os.makedirs(args.output_dir, exist_ok=True)
    pose_checkpoint_basename = Path(args.pose_checkpoint).stem

    max_batch_size = args.max_batch_size
    pose_input_shape = (
        max(1, min(pose_input_shape[0], max_batch_size)),
        *pose_input_shape[1:],
    )

    pose_model = init_pose_estimator(
        args.pose_config,
        args.pose_checkpoint,
        override_ckpt_meta=True,  # dont load the checkpoint meta data, load from config file
        device="cuda",
    )
    pose_model.forward = pose_model._forward

    pipeline = Compose(pose_model.cfg.test_dataloader.dataset.pipeline)
    dataset_meta = pose_model.dataset_meta

    mm_inputs_pose = _demo_mm_inputs_pose(pose_input_shape, dataset_meta, pipeline)

    if torch.cuda.is_available():
        pose_model.cuda()

    dtype = torch.bfloat16 if not args.fp16 else torch.half

    _benchmark(pose_model, mm_inputs_pose, "Original")

    graphs, graph_counts, break_reasons = explain_model(pose_model, mm_inputs_pose)

    if args.explain_verbose:
        print(f"Graphs: {graphs}")
        print(f"Graph Counts: {graph_counts}")
        print(f"Reasons: {break_reasons}")

    if not args.force_compile and graph_counts > 1:
        print(f"Graphs are not fusable. Expected 1 graph. Found {graph_counts}")
        return

    pose_model.to(dtype=dtype)
    mm_inputs_pose["inputs"] = mm_inputs_pose["inputs"].to(dtype=dtype).cuda()
    save_path = os.path.join(
        args.output_dir,
        f"{pose_checkpoint_basename}_{'float16' if dtype==torch.float16 else 'bfloat16'}.pt2",
    )
    compile_model(
        pose_model,
        mm_inputs_pose,
        save_path,
        max_batch_size=max_batch_size,
        dtype=dtype,
    )


if __name__ == "__main__":
    main()

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import copy
import itertools
import json
import os
import os.path as osp
import time
from functools import partial
from pathlib import Path
from typing import Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

import torch.nn as nn
import torch.nn.utils.parametrize as parametrize
import torch.utils.benchmark as benchmark
from mmengine.utils import mkdir_or_exist

# import torch_tensorrt
from mmpretrain import FeatureExtractor

from torch._dynamo import is_compiling as dynamo_is_compiling
from torch._higher_order_ops.out_dtype import out_dtype
from torch.profiler import ProfilerActivity
from tqdm import tqdm


def _benchmark(model, input, model_name=""):
    imgs = (
        input["imgs"][0, ...].unsqueeze(0)
        if model_name.lower() == "original"
        else input["imgs"]
    )
    if torch.cuda.is_available():
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

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
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s), torch.no_grad():
        for _ in range(10):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                start_event.record()
            model(imgs)
            end_event.record()
            if torch.cuda.is_available():
                end_event.record()
                torch.cuda.synchronize()
                time_.append(start_event.elapsed_time(end_event))
    torch.cuda.current_stream().wait_stream(s)
    mean_time = np.mean(time_[1:]) / (imgs.shape[0])
    print(f"For {model_name} model, ", flush=True)
    print(f"avg time is {mean_time} ms", flush=True)
    print(f"Total time is {sum(time_)} ms", flush=True)
    print(f"Each trial time: {time_}", flush=True)
    return mean_time


def _convert_batchnorm(module):
    module_output = module
    if isinstance(module, (torch.nn.SyncBatchNorm)):
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
    if isinstance(module, (torch.nn.SiLU)):
        module_output = torch.nn.ReLU(inplace=True)
    for name, child in module.named_children():
        module_output.add_module(name, _convert_batchnorm(child))
    del module
    return module_output


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
    imgs = rng.rand(*input_shape)

    mm_inputs = {
        "imgs": torch.FloatTensor(imgs),
    }
    return mm_inputs


def explain_model(model, inputs):
    imgs = inputs["imgs"]
    with torch.no_grad():
        explanation = torch._dynamo.explain(model, imgs)
    return explanation.graphs, explanation.graph_count, explanation.break_reasons


def compile_model(
    model,
    inputs,
    output_file="compiled_model.pt",
    max_batch_size=64,
    dtype=torch.bfloat16,
):

    imgs = inputs["imgs"]
    modes = {"Deafult": "default", "RO": "reduce-overhead", "MA": "max-autotune"}
    # modes = { "MA": "max-autotune"}
    # modes = {"int8_dq": change_linear_weights_to_int8_dqtensors,}#{"int8_dq": change_linear_weights_to_int8_dqtensors,} #"int8_wo": change_linear_weights_to_int8_woqtensors,}# "int4": change_linear_weights_to_int4_woqtensors}
    # modes = {"int8_int4": Int8DynActInt4WeightQuantizer(groupsize=128).quantize}
    min_mean = float("inf")
    best_mode = None

    inputs["imgs"] = inputs["imgs"].to(dtype).cuda()
    imgs = inputs["imgs"]
    args = (imgs,)
    kwargs = {}
    dynamic_batch = torch.export.Dim("batch", min=1, max=max_batch_size)
    dynamic_shapes = {"inputs": {0: dynamic_batch}}
    with torch.no_grad():
        # model.forward = model._forward
        exported_model = torch.export.export(
            model, args=args, kwargs=kwargs, dynamic_shapes=dynamic_shapes
        )
        for mode_str, mode in modes.items():
            print(f"Compiling model with {mode_str} mode")
            compiled_model = torch.compile(exported_model.module(), mode=mode)
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s), torch.no_grad():
                for i in range(3):
                    compiled_model(imgs)
                    torch.cuda.synchronize()
            torch.cuda.current_stream().wait_stream(s)
            mean = _benchmark(compiled_model, inputs, model_name=mode_str)
            if mean < min_mean:
                min_mean = mean
                best_mode = mode_str
            # inputs["imgs"] = inputs["imgs"].to(torch.bfloat16)
            # model = m
    print(f"Best compilation mode: {best_mode}")
    torch.export.save(exported_model, output_file)
    print(output_file)


def parse_args():
    parser = argparse.ArgumentParser(description="MMSeg sparsify a model")
    parser.add_argument("config", help="test config file path")
    parser.add_argument("checkpoint", help="checkpoint file")
    parser.add_argument(
        "--shape",
        type=int,
        nargs="+",
        default=[1024, 1024],
        help="input image size (height, width)",
    )
    parser.add_argument(
        "--output_dir", "--output-dir", type=str, help="input image directory"
    )
    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=64,
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


def collate_wrapper(calib_dataloader_collate, *args, **kwargs):
    # inputs = calib_dataloader_collate(*args, **kwargs)["inputs"]
    return torch.stack(calib_dataloader_collate(*args, **kwargs)["inputs"], dim=0).to(
        dtype=torch.float
    )


def main():
    args = parse_args()

    if len(args.shape) == 1:
        input_shape = (64, 3, args.shape[0], args.shape[0])
    elif len(args.shape) == 2:
        input_shape = (
            64,
            3,
        ) + tuple(args.shape)
    else:
        raise ValueError("invalid input shape")

    os.makedirs(args.output_dir, exist_ok=True)
    checkpoint_basename = Path(args.checkpoint).stem

    model = FeatureExtractor(model=args.config, pretrained=args.checkpoint).model
    model.backbone.out_type = (
        "featmap"  ## removes cls_token and returns spatial feature maps.
    )
    model.eval()
    max_batch_size = args.max_batch_size
    input_shape = (max(1, min(input_shape[0], max_batch_size)), *input_shape[1:])

    mm_inputs = _demo_mm_inputs(input_shape)

    if torch.cuda.is_available():
        model.cuda()
        mm_inputs["imgs"] = mm_inputs["imgs"].cuda()

    _benchmark(model, mm_inputs, "Original")

    graphs, graph_counts, break_reasons = explain_model(model, mm_inputs)

    if args.explain_verbose:
        print(f"Graphs: {graphs}")
        print(f"Graph Counts: {graph_counts}")
        print(f"Reasons: {break_reasons}")

    if not args.force_compile and graph_counts > 1:
        print(f"Graphs are not fusable. Expected 1 graph. Found {graph_counts}")
        return

    dtype = torch.bfloat16 if not args.fp16 else torch.half

    model.to(dtype)
    mm_inputs["imgs"] = mm_inputs["imgs"].to(dtype)
    save_path = os.path.join(
        args.output_dir,
        f"{checkpoint_basename}_{'float16' if dtype==torch.float16 else 'bfloat16'}.pt2",
    )
    compile_model(
        model, mm_inputs, save_path, max_batch_size=max_batch_size, dtype=dtype
    )


if __name__ == "__main__":
    main()

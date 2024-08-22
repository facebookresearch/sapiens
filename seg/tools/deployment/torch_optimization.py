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
from mmengine import Config
from mmengine.fileio import dump
from mmengine.model.utils import revert_sync_batchnorm
from mmengine.registry import init_default_scope
from mmengine.runner import get_state_dict, load_checkpoint, Runner, save_checkpoint
from mmengine.utils import mkdir_or_exist

# import torch_tensorrt
# import torch_tensorrt.ts.ptq as ptq
# from pytorch_quantization.tensor_quant import QuantDescriptor
# from pytorch_quantization import quant_modules
# # quant_modules.initialize()
# from pytorch_quantization import nn as quant_nn
# from pytorch_quantization import calib
# import modelopt.torch.quantization as mtq
# from modelopt.torch.quantization.utils import export_torch_mode
# from modelopt.torch.quantization.nn import TensorQuantizer
from mmseg.apis import init_model

# from mmseg.models import build_segmentor
from mmseg.registry import MODELS

from pytorch2torchscript import pytorch2libtorch

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


def _demo_mm_inputs(input_shape, num_classes):
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
    if num_classes > 1:
        segs = rng.randint(low=0, high=num_classes - 1, size=(N, 1, H, W)).astype(
            np.uint8
        )
    else:
        segs = rng.uniform(0, 1, size=(N, 1, H, W)).astype(np.uint8)
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
    mm_inputs = {
        "imgs": torch.FloatTensor(imgs),
        "img_metas": img_metas,
        "gt_semantic_seg": torch.LongTensor(segs),
    }
    return mm_inputs


def explain_model(model, inputs):
    imgs = inputs["imgs"]
    with torch.no_grad():
        explanation = torch._dynamo.explain(model, imgs)
    return explanation.graphs, explanation.graph_count, explanation.break_reasons


def fuse_model(model):
    fuse_modules = torch.ao.quantization.fuse_modules
    decode_convs = model.decode_head.conv_layers
    for idx in range(len(decode_convs)):
        if isinstance(decode_convs[idx], torch.nn.Conv2d):
            fuse_modules(
                decode_convs, [str(idx), str(idx + 1), str(idx + 2)], inplace=True
            )
    model.decode_head.conv_layers = decode_convs
    decode_deconvs = model.decode_head.deconv_layers
    for idx in range(len(decode_deconvs)):
        if isinstance(decode_deconvs[idx], torch.nn.ConvTranspose2d):
            fuse_modules(decode_deconvs, [str(idx + 1), str(idx + 2)], inplace=True)
    model.decode_head.deconv_layers = decode_deconvs
    return model


def compile_model(
    model,
    inputs,
    calib_dataloader,
    output_file="compiled_model.pt",
    max_batch_size=32,
    dtype=torch.bfloat16,
):

    imgs = inputs["imgs"]
    modes = {"Default": "default", "RO": "reduce-overhead", "MA": "max-autotune"}
    # modes = { "MA": "max-autotune"}
    # modes = {"int8_dq": change_linear_weights_to_int8_dqtensors,}#{"int8_dq": change_linear_weights_to_int8_dqtensors,} #"int8_wo": change_linear_weights_to_int8_woqtensors,}# "int4": change_linear_weights_to_int4_woqtensors}
    # modes = {"int8_int4": Int8DynActInt4WeightQuantizer(groupsize=128).quantize}
    min_mean = float("inf")
    best_mode = None

    if calib_dataloader:
        import torchao
        from torch.ao.quantization import quantize
        from torchao.quantization.quant_api import (
            change_linear_weights_to_int4_woqtensors,
            change_linear_weights_to_int8_dqtensors,
            change_linear_weights_to_int8_woqtensors,
            Int8DynActInt4WeightQuantizer,
        )
        from torchao.quantization.smoothquant import (
            smooth_fq_linear_to_inference,
            swap_linear_with_smooth_fq_linear,
        )
        from torchao.utils import unwrap_tensor_subclass

        torch._dynamo.config.automatic_dynamic_shapes = False
        torch._inductor.config.force_fuse_int_mm_with_mul = True
        torch._inductor.config.use_mixed_mm = True
        swap_linear_with_smooth_fq_linear(model)
        print("Calibrating ...")
        model.eval()
        with torch.no_grad():
            for batch in calib_dataloader:
                # model.zero_grad()
                model(batch.to(dtype).cuda())
        print("Calibration done")
        smooth_fq_linear_to_inference(model)

    inputs["imgs"] = inputs["imgs"].to(dtype).cuda()
    imgs = inputs["imgs"]
    model.eval()
    args = (imgs,)
    kwargs = {}
    dynamic_batch = torch.export.Dim("batch", min=1, max=max_batch_size)
    dynamic_shapes = {"inputs": {0: dynamic_batch}}
    with torch.no_grad():
        # model.forward = model._forward

        for mode_str, mode in modes.items():
            print(f"Compiling model with {mode_str} mode")
            exported_model = torch.export.export(
                model, args=args, kwargs=kwargs, dynamic_shapes=dynamic_shapes
            )
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


def run(test_loop) -> dict:
    """Launch test."""
    test_loop.runner.call_hook("before_test")
    test_loop.runner.call_hook("before_test_epoch")
    test_loop.runner.model.eval()
    for idx, data_batch in enumerate(test_loop.dataloader):
        run_iter(test_loop, idx, data_batch)

    # compute metrics
    metrics = test_loop.evaluator.evaluate(len(test_loop.dataloader.dataset))
    test_loop.runner.call_hook("after_test_epoch", metrics=metrics)
    test_loop.runner.call_hook("after_test")
    return metrics


@torch.no_grad()
def run_iter(test_loop, idx, data_batch) -> None:
    """Iterate one mini-batch.

    Args:
        data_batch (Sequence[dict]): Batch of data from dataloader.
    """
    test_loop.runner.call_hook("before_test_iter", batch_idx=idx, data_batch=data_batch)
    # predictions should be sequence of BaseDataElement
    # with autocast(enabled=test_loop.fp16):
    #     with torch.autocast(device_type=get_device(), dtype=torch.bfloat16):
    outputs = test_loop.runner.model.test_step(data_batch)
    test_loop.evaluator.process(data_samples=outputs, data_batch=data_batch)
    test_loop.runner.call_hook(
        "after_test_iter", batch_idx=idx, data_batch=data_batch, outputs=outputs
    )


def calib_loop(runner, model):
    """
    Tensorrt quantization loop
    """
    runner.model = runner.wrap_model(runner.cfg.get("model_wrapper_cfg"), model)
    runner.test()


def collect_stats(model, data_loader, num_batches=100):
    """Feed data to the network and collect statistic"""

    # Enable calibrators
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                module.disable_quant()
                module.enable_calib()
            else:
                module.disable()

    for i, (image) in tqdm(enumerate(data_loader), total=num_batches):
        model(image.to(torch.float).cuda())
        if i >= num_batches:
            break

    # Disable calibrators
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                module.enable_quant()
                module.disable_calib()
            else:
                module.enable()


def compute_amax(model, **kwargs):
    # Load calib result
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                if isinstance(module._calibrator, calib.MaxCalibrator):
                    module.load_calib_amax()
                else:
                    module.load_calib_amax(**kwargs)
            print(f"{name:40}: {module}")
    model.cuda()


def quantize_pytorch(model, calib_dataloader):
    quant_desc_input = QuantDescriptor(calib_method="histogram")
    quant_nn.QuantConv2d.set_default_quant_desc_input(quant_desc_input)
    quant_nn.QuantConvTranspose2d.set_default_quant_desc_input(quant_desc_input)
    quant_nn.QuantLinear.set_default_quant_desc_input(quant_desc_input)
    with torch.no_grad():
        collect_stats(model, calib_dataloader)
        compute_amax(model, method="percentile", percentile=99.99)
    return model


def convert_to_tensorrt(
    model, shape, num_classes, calibrate_loop=None, output_file="tensorrt_model.pt"
):
    # imgs = inputs["imgs"]
    shape = shape[1:]
    input_dynamic_batch_shape = (20,)

    # input_sig = (torch_tensorrt.Input(
    #     min_shape=(input_dynamic_batch_shape[0], *shape),
    #     opt_shape=(input_dynamic_batch_shape[1], *shape),
    #     max_shape=(input_dynamic_batch_shape[2], *shape),
    #     dtype=torch.bfloat16))
    model.eval()
    min_mean = float("inf")
    best_mode = None
    best_model = None
    for bs in input_dynamic_batch_shape:
        print(f"Tensorrt model with batch size {bs}")
        inputs = _demo_mm_inputs((bs, *shape), num_classes)
        # inputs["imgs"] = inputs["imgs"].cuda()
        # imgs = inputs["imgs"]
        if calibrate_loop:
            # Other experimented quantization methods
            # quant_cfg = mtq.INT8_SMOOTHQUANT_CFG.copy()
            # print(quant_cfg)
            # quant_cfg["quant_cfg"]["*InstanceNorm*"] = {"enable": False}
            # quant_cfg["quant_cfg"]["*InstanceNorm*weight_quantizer"] = {"enable": False}
            # print(help(mtq.register))
            # mtq.register(original_cls=torch.nn.InstanceNorm2d, quantized_cls=QuantizedInstanceNorm)
            # PTQ with in-place replacement to quantized modules
            # mtq.quantize(model, quant_cfg, forward_loop=calibrate_loop)
            # model = quantize_pytorch(model, calibrate_loop)
            # quant_nn.TensorQuantizer.use_fb_fake_quant = True

            calibrator = ptq.DataLoaderCalibrator(
                calibrate_loop,
                use_cache=False,
                algo_type=ptq.CalibrationAlgo.ENTROPY_CALIBRATION_2,
                device=torch.device("cuda:0"),
            )
            quantized = True
            print("Quantized model")
        else:
            quantized = False

        compile_spec = {
            # "inputs": [torch_tensorrt.Input((bs, *shape))], # , dtype=(torch.int8 if quantized else torch.half)
            "inputs": [
                torch_tensorrt.Input(
                    (bs, *shape), dtype=torch.int8 if quantized else torch.half
                )
            ],
            "enabled_precisions": (torch.int8,) if quantized else (torch.half,),
            # "optimization_level": 5,
            "truncate_long_and_double": True,
            "require_full_compilation": True,
            "allow_shape_tensors": True,
            # "debug": True,
        }
        if quantized:
            # model.to(torch.bfloat16)
            # model.half()
            compile_spec["calibrator"] = calibrator
            model.eval()
            inputs["imgs"] = inputs["imgs"].cuda().to(torch.bfloat16)
            imgs = inputs["imgs"]
        else:
            model.half()
            inputs["imgs"] = inputs["imgs"].half().cuda()
            imgs = inputs["imgs"]

        with torch.no_grad():
            # with export_torch_mode():
            model = torch.jit.trace(model, imgs)
            trt_model = torch_tensorrt.compile(model, ir="torchscript", **compile_spec)
        if quantized:
            inputs["imgs"] = inputs["imgs"].to(torch.int8)
        imgs = inputs["imgs"]
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s), torch.no_grad():
            for i in range(3):
                trt_model(imgs)
        torch.cuda.current_stream().wait_stream(s)
        mean = _benchmark(
            trt_model, inputs, model_name=f"TensorRT with batch size {bs}"
        )
        if mean < min_mean:
            min_mean = mean
            best_mode = bs
            best_model = copy.deepcopy(trt_model).cpu()
            del trt_model
    # print(model)
    print(f"Best batch size: {best_mode}, with avg time {min_mean} ms")
    # torch_tensorrt.save(best_model.cuda(), output_file, inputs=[imgs])


def parse_args():
    parser = argparse.ArgumentParser(description="Sparsify a model")
    parser.add_argument("config", help="test config file path")
    parser.add_argument("checkpoint", help="checkpoint file")
    parser.add_argument(
        "--shape",
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
        default=32,
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
    parser.add_argument("--quant", action="store_true", help="To enable quantization")
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
        input_shape = (16, 3, args.shape[0], args.shape[0])
    elif len(args.shape) == 2:
        input_shape = (
            16,
            3,
        ) + tuple(args.shape)
    else:
        raise ValueError("invalid input shape")

    os.makedirs(args.output_dir, exist_ok=True)
    checkpoint_basename = Path(args.checkpoint).stem

    cfg = Config.fromfile(args.config)
    init_default_scope(cfg.get("default_scope", "mmseg"))
    cfg.model.pretrained = None

    # build the model and load checkpoint
    cfg.model.train_cfg = None

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.output_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.output_dir
    elif cfg.get("work_dir", None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join(
            "./work_dirs", osp.splitext(osp.basename(args.config))[0]
        )

    max_batch_size = args.max_batch_size
    input_shape = (max(1, min(input_shape[0], max_batch_size)), *input_shape[1:])
    cfg.load_from = args.checkpoint
    # model = build_segmentor(cfg.model, train_cfg=None, test_cfg=cfg.get("test_cfg"))
    calib_dataloader = None
    if args.quant:
        runner = Runner.from_cfg(cfg)
        diff_rank_seed = runner._randomness_cfg.get("diff_rank_seed", False)
        calib_dataloader = runner.build_dataloader(
            cfg.get("test_dataloader"), seed=runner.seed, diff_rank_seed=diff_rank_seed
        )

        calib_dataloader.collate_fn = partial(
            collate_wrapper, calib_dataloader.collate_fn
        )
    model = init_model(args.config, args.checkpoint, device="cpu")
    model.eval()
    # convert SyncBN to BN
    model = revert_sync_batchnorm(model)

    # if args.checkpoint:
    #     load_checkpoint(model, args.checkpoint, map_location="cpu")

    if isinstance(model.decode_head, torch.nn.ModuleList):
        num_classes = model.decode_head[-1].num_classes
    else:
        num_classes = model.decode_head.num_classes

    mm_inputs = _demo_mm_inputs(input_shape, num_classes)

    if torch.cuda.is_available():
        model.cuda()
        mm_inputs["imgs"] = mm_inputs["imgs"].cuda()

    dtype = torch.bfloat16 if not args.fp16 else torch.half

    _benchmark(model, mm_inputs, "Original")

    graphs, graph_counts, break_reasons = explain_model(model, mm_inputs)

    if args.explain_verbose:
        print(f"Graphs: {graphs}")
        print(f"Graph Counts: {graph_counts}")
        print(f"Reasons: {break_reasons}")

    if not args.force_compile and graph_counts > 1:
        print(f"Graphs are not fusable. Expected 1 graph. Found {graph_counts}")
        return

    model.to(dtype)
    mm_inputs["imgs"] = mm_inputs["imgs"].to(dtype)
    save_path = os.path.join(
        args.output_dir,
        f"{checkpoint_basename}_{'float16' if dtype==torch.float16 else 'bfloat16'}.pt2",
    )
    compile_model(
        model, mm_inputs, None, save_path, max_batch_size=max_batch_size, dtype=dtype
    )

    # Tensorrt disabled for now
    # save_path = os.path.join(args.output_dir, f"{checkpoint_basename}_trt.ep")
    # convert_to_tensorrt(model, input_shape, num_classes, partial(calib_loop, runner), save_path)
    # print(model)


if __name__ == "__main__":
    main()

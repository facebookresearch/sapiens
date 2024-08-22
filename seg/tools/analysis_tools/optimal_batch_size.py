# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from memory_error_utils import is_oom_error, garbage_collection_cuda
import argparse
import os
import os.path as osp
import itertools
import time
import json

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from tqdm import tqdm
from mmengine import Config
from mmengine.fileio import dump
from mmengine.model.utils import revert_sync_batchnorm
from mmengine.registry import init_default_scope
from mmengine.runner import Runner, load_checkpoint
from mmengine.utils import mkdir_or_exist

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


def read_images_in_batch(img_dir, n_imgs):
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
        sampled_img_paths = np.random.choice(img_paths, n_imgs, replace=True)
        for img_path in sampled_img_paths: 
            img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
            img = np.transpose(img, (2, 0, 1))
            imgs.append(resize(torch.Tensor(img)))
        yield torch.stack(imgs, dim=0)
        del imgs
        imgs = []
    

def run_power_scaling(model, batch_size, img_dir, max_trials=100, warmup=5, device=None):
    """ Batch scaling mode where the size is doubled at each iteration until an
        OOM error is encountered. """
    high = None
    best_spi = float('inf')
    best_batch_size = None
    low=0
    model.eval()
    optim_batch_dict = {"warmup": warmup, "max_trials": max_trials}
    while True:
        try:
            img_gen = read_images_in_batch(img_dir, batch_size)
            spi_list = []
            fps_list = []
            print(f"Running batch size {batch_size}")
            # warmup_model(model, (batch_size, 3, 1024, 768))
            garbage_collection_cuda()
            for i in tqdm(range(warmup + max_trials)):
                inputs = next(img_gen)
                inputs = inputs.to(device=device)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                start = time.perf_counter()
                with torch.no_grad():
                    model(inputs, mode="predict")
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                time_taken = time.perf_counter() - start
                spi_list.append(time_taken/batch_size)
                fps_list.append(batch_size/time_taken)
                inputs = inputs.detach().cpu()
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
    plt.ylabel("Secs/Img", fontsize=15)
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
    parser = argparse.ArgumentParser(description='MMSeg benchmark a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
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
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    cfg = Config.fromfile(args.config)

    init_default_scope(cfg.get('default_scope', 'mmseg'))

    # set cudnn_benchmark
    torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = MODELS.build(cfg.model)

    if 'checkpoint' in args and osp.exists(args.checkpoint):
        load_checkpoint(model, args.checkpoint, map_location='cpu')

    if torch.cuda.is_available():
        model = model.cuda()

    model = revert_sync_batchnorm(model)

    model.eval()

    print("Running batch size finder ...")

    optimal_batch_size_dict = run_power_scaling(model, 1, args.img_dir, device="cuda" if torch.cuda.is_available() else "cpu")
    # print(f"Optimal batch size: {optimal_batch_size_dict}")
    json.dump(optimal_batch_size_dict, open(osp.join(output_dir, "optim_batch_dict.json"), 'w'), indent=4)
    error_plt_batches(optimal_batch_size_dict, output_dir)
    print(f"Successfully saved results to {output_dir}")


if __name__ == '__main__':
    main()

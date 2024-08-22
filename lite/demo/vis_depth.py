# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import multiprocessing as mp
import os
import tempfile
import time
from argparse import ArgumentParser
from functools import partial
from multiprocessing import cpu_count, Pool, Process

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from adhoc_image_dataset import AdhocImageDataset
from tqdm import tqdm

from worker_pool import WorkerPool

torchvision.disable_beta_transforms_warning()

timings = {}
BATCH_SIZE = 18


def _demo_mm_inputs(batch_size, input_shape):
    (C, H, W) = input_shape
    N = batch_size
    rng = np.random.RandomState(0)
    imgs = rng.rand(batch_size, C, H, W)
    if torch.cuda.is_available():
        imgs = torch.Tensor(imgs).cuda()
    return imgs


def warmup_model(model, batch_size):
    imgs = torch.randn(batch_size, 3, 1024, 768).to(dtype=torch.bfloat16).cuda()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s), torch.no_grad(), torch.autocast(
        device_type="cuda", dtype=torch.bfloat16
    ):
        for i in range(3):
            model(imgs)
    torch.cuda.current_stream().wait_stream(s)
    imgs = imgs.detach().cpu().float().numpy()
    del imgs, s


def inference_model(model, imgs, dtype=torch.bfloat16):
    with torch.no_grad():
        results = model(imgs.to(dtype).cuda())
        imgs.cpu()

    results = [r.cpu() for r in results]

    return results


def fake_pad_images_to_batchsize(imgs):
    # if len(imgs) < BATCH_SIZE:
    #     imgs = imgs + [torch.zeros((imgs[0].shape[0], imgs[0].shape[1], imgs[0].shape[2]))] * (BATCH_SIZE - len(imgs))
    # return torch.stack(imgs, dim=0)
    return F.pad(imgs, (0, 0, 0, 0, 0, 0, 0, BATCH_SIZE - imgs.shape[0]), value=0)


def load_and_preprocess(img, shape):
    orig_img = cv2.imread(img)
    img = cv2.resize(orig_img, (768, 1024), interpolation=cv2.INTER_LINEAR).transpose(
        2, 0, 1
    )
    img = torch.from_numpy(img)
    img = img[[2, 1, 0], ...].float()
    mean = torch.tensor([123.5, 116.5, 103.5]).view(-1, 1, 1)
    std = torch.tensor([58.5, 57.0, 57.5]).view(-1, 1, 1)
    img = (img - mean) / std
    return orig_img, img


def img_save_and_viz(image, result, output_path, seg_dir):
    seg_logits = F.interpolate(
        result.unsqueeze(0), size=image.shape[:2], mode="bilinear"
    ).squeeze(0)

    depth_map = seg_logits.data.float().numpy()[0]  ## H x W
    image_name = os.path.basename(output_path)

    mask_path = os.path.join(
        seg_dir,
        image_name.replace(".png", ".npy")
        .replace(".jpg", ".npy")
        .replace(".jpeg", ".npy"),
    )
    mask = np.load(mask_path)

    ##-----------save depth_map to disk---------------------
    save_path = (
        output_path.replace(".png", ".npy")
        .replace(".jpg", ".npy")
        .replace(".jpeg", ".npy")
    )
    np.save(save_path, depth_map)

    depth_map[~mask] = np.nan
    depth_foreground = depth_map[mask]  ## value in range [0, 1]
    processed_depth = np.full((mask.shape[0], mask.shape[1], 3), 100, dtype=np.uint8)

    if len(depth_foreground) > 0:
        min_val, max_val = np.min(depth_foreground), np.max(depth_foreground)
        depth_normalized_foreground = 1 - (
            (depth_foreground - min_val) / (max_val - min_val)
        )  ## for visualization, foreground is 1 (white), background is 0 (black)
        depth_normalized_foreground = (depth_normalized_foreground * 255.0).astype(
            np.uint8
        )

        depth_colored_foreground = cv2.applyColorMap(
            depth_normalized_foreground, cv2.COLORMAP_INFERNO
        )
        depth_colored_foreground = depth_colored_foreground.reshape(-1, 3)
        processed_depth[mask] = depth_colored_foreground

    ##---------get surface normal from depth map---------------
    depth_normalized = np.full((mask.shape[0], mask.shape[1]), np.inf)
    depth_normalized[mask > 0] = 1 - (
        (depth_foreground - min_val) / (max_val - min_val)
    )

    kernel_size = 7
    grad_x = cv2.Sobel(
        depth_normalized.astype(np.float32),
        cv2.CV_32F,
        1,
        0,
        ksize=kernel_size,
    )
    grad_y = cv2.Sobel(
        depth_normalized.astype(np.float32),
        cv2.CV_32F,
        0,
        1,
        ksize=kernel_size,
    )
    z = np.full(grad_x.shape, -1)
    normals = np.dstack((-grad_x, -grad_y, z))

    # Normalize the normals
    normals_mag = np.linalg.norm(normals, axis=2, keepdims=True)

    ## background pixels are nan.
    with np.errstate(divide="ignore", invalid="ignore"):
        normals_normalized = normals / (
            normals_mag + 1e-5
        )  # Add a small epsilon to avoid division by zero

    # Convert normals to a 0-255 scale for visualization
    normals_normalized = np.nan_to_num(
        normals_normalized, nan=-1, posinf=-1, neginf=-1
    )  ## visualize background (nan) as black
    normal_from_depth = ((normals_normalized + 1) / 2 * 255).astype(np.uint8)

    ## RGB to BGR for cv2
    normal_from_depth = normal_from_depth[:, :, ::-1]

    vis_image = np.concatenate([image, processed_depth, normal_from_depth], axis=1)
    cv2.imwrite(output_path, vis_image)

def load_model(checkpoint, use_torchscript=False):
    if use_torchscript:
        return torch.jit.load(checkpoint)
    else:
        return torch.export.load(checkpoint).module()

def main():
    parser = ArgumentParser()
    parser.add_argument("checkpoint", help="Checkpoint file")
    parser.add_argument("--input", help="Input image dir")
    parser.add_argument(
        "--output_root", "--output-root", default=None, help="Path to output dir"
    )
    parser.add_argument("--seg_dir", default=None, help="Path to seg dir")
    parser.add_argument("--device", default="cuda:0", help="Device used for inference")
    parser.add_argument(
        "--batch_size",
        "--batch-size",
        type=int,
        default=18,
        help="Set batch size to do batch inference. ",
    )
    parser.add_argument(
        "--shape",
        type=int,
        nargs="+",
        default=[1024, 768],
        help="input image size (height, width)",
    )
    parser.add_argument(
        "--fp16", action="store_true", default=False, help="Model inference dtype"
    )
    args = parser.parse_args()

    if len(args.shape) == 1:
        input_shape = (3, args.shape[0], args.shape[0])
    elif len(args.shape) == 2:
        input_shape = (3,) + tuple(args.shape)
    else:
        raise ValueError("invalid input shape")

    mp.log_to_stderr()
    torch._inductor.config.force_fuse_int_mm_with_mul = True
    torch._inductor.config.use_mixed_mm = True

    start = time.time()

    USE_TORCHSCRIPT = '_torchscript' in args.checkpoint

    # build the model from a checkpoint file
    exp_model = load_model(args.checkpoint, USE_TORCHSCRIPT)

    ## no precision conversion needed for torchscript. run at fp32
    if not USE_TORCHSCRIPT:
        dtype = torch.half if args.fp16 else torch.bfloat16
        exp_model.to(dtype)
        exp_model = torch.compile(exp_model, mode="max-autotune", fullgraph=True)
    else:
        dtype = torch.float32  # TorchScript models use float32
        exp_model = exp_model.to(args.device)

    input = args.input
    image_names = []

    # Check if the input is a directory or a text file
    if os.path.isdir(input):
        input_dir = input  # Set input_dir to the directory specified in input
        image_names = [
            image_name
            for image_name in sorted(os.listdir(input_dir))
            if image_name.endswith(".jpg")
            or image_name.endswith(".png")
            or image_name.endswith(".jpeg")
        ]
    elif os.path.isfile(input) and input.endswith(".txt"):
        # If the input is a text file, read the paths from it and set input_dir to the directory of the first image
        with open(input, "r") as file:
            image_paths = [line.strip() for line in file if line.strip()]
        image_names = [
            os.path.basename(path) for path in image_paths
        ]  # Extract base names for image processing
        input_dir = (
            os.path.dirname(image_paths[0]) if image_paths else ""
        )  # Use the directory of the first image path

    if not os.path.exists(args.output_root):
        os.makedirs(args.output_root)

    global BATCH_SIZE
    BATCH_SIZE = args.batch_size

    n_batches = (len(image_names) + args.batch_size - 1) // args.batch_size
    inference_dataset = AdhocImageDataset(
        [os.path.join(input_dir, img_name) for img_name in image_names],
        (input_shape[1], input_shape[2]),
        mean=[123.5, 116.5, 103.5],
        std=[58.5, 57.0, 57.5],
    )
    inference_dataloader = torch.utils.data.DataLoader(
        inference_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=max(min(args.batch_size, cpu_count()), 1),
    )
    total_results = []
    image_paths = []
    img_save_pool = WorkerPool(
        img_save_and_viz, processes=max(min(args.batch_size, cpu_count()), 1)
    )
    for batch_idx, (batch_image_name, batch_orig_imgs, batch_imgs) in tqdm(
        enumerate(inference_dataloader), total=len(inference_dataloader)
    ):
        valid_images_len = len(batch_imgs)
        batch_imgs = fake_pad_images_to_batchsize(batch_imgs)
        result = inference_model(exp_model, batch_imgs, dtype=dtype)
        args_list = [
            (
                i,
                r,
                os.path.join(args.output_root, os.path.basename(img_name)),
                args.seg_dir,
            )
            for i, r, img_name in zip(
                batch_orig_imgs[:valid_images_len],
                result[:valid_images_len],
                batch_image_name,
            )
        ]
        img_save_pool.run_async(args_list)

    img_save_pool.finish()

    total_time = time.time() - start
    fps = 1 / ((time.time() - start) / len(image_names))
    print(
        f"\033[92mTotal inference time: {total_time:.2f} seconds. FPS: {fps:.2f}\033[0m"
    )


if __name__ == "__main__":
    main()

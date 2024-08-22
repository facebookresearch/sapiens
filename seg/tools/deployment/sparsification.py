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

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from mmengine import Config
from mmengine.fileio import dump
from mmengine.model.utils import revert_sync_batchnorm
from mmengine.registry import init_default_scope
from mmengine.runner import get_state_dict, load_checkpoint, Runner, save_checkpoint
from mmengine.utils import mkdir_or_exist
from mmseg.registry import MODELS
from torch.sparse import SparseSemiStructuredTensor, to_sparse_semi_structured
from torchvision import transforms
from tqdm import tqdm

# SparseSemiStructuredTensor._FORCE_CUTLASS = True


# Sparsity helper functions
def apply_fake_sparsity(model):
    """
    This function simulates 2:4 sparsity on all linear layers in a model.
    It uses the torch.ao.pruning flow.
    """
    # torch.ao.pruning flow
    from torch.ao.pruning import WeightNormSparsifier

    sparse_config = []
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear):
            sparse_config.append({"tensor_fqn": f"{name}.weight"})

    sparsifier = WeightNormSparsifier(
        sparsity_level=1.0, sparse_block_shape=(1, 4), zeros_per_block=2
    )
    sparsifier.prepare(model, sparse_config)
    sparsifier.step()
    sparsifier.squash_mask()


def apply_sparse(model):
    apply_fake_sparsity(model)
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear):
            mod.weight = torch.nn.Parameter(
                to_sparse_semi_structured(mod.weight).to_dense()
            )


def parse_args():
    parser = argparse.ArgumentParser(description="MMSeg sparsify a model")
    parser.add_argument("config", help="test config file path")
    parser.add_argument("checkpoint", help="checkpoint file")
    parser.add_argument(
        "--output_dir", "--output-dir", type=str, help="input image directory"
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    cfg = Config.fromfile(args.config)

    init_default_scope(cfg.get("default_scope", "mmseg"))

    # set cudnn_benchmark
    torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = MODELS.build(cfg.model)

    checkpoint_basename = Path(args.checkpoint).stem

    if "checkpoint" in args and osp.exists(args.checkpoint):
        checkpoint = load_checkpoint(model, args.checkpoint, map_location="cpu")

    model = model.cuda().to(torch.bfloat16)  # sparsification REQUIRES cuda

    model = revert_sync_batchnorm(model)

    model.eval()

    save_path = os.path.join(args.output_dir, f"{checkpoint_basename}_sparsified.pth")
    input_tensor = torch.randn(8, 3, 1024, 768, dtype=torch.bfloat16, device="cuda")
    with torch.no_grad():
        model(input_tensor)

    print(
        f"Model checkpoint before sparsification: {os.stat(args.checkpoint).st_size / 1024 **2} MB"
    )
    apply_sparse(model)
    torch.save(model, save_path)
    # checkpoint["state_dict"] = get_state_dict(model)
    # save_checkpoint(checkpoint, save_path)
    # load_checkpoint(
    #     model, args.checkpoint, map_location="cpu"
    # )  # check checkpoint loading again
    print(f"Sparse model saved to {save_path}")
    print(
        f"Model checkpoint after sparsification: {os.stat(save_path).st_size / 1024 **2} MB"
    )
    # with torch.no_grad():
    #     model(input_tensor)


if __name__ == "__main__":
    main()

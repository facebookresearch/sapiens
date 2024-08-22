# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import torch
from mmengine.runner.checkpoint import _load_checkpoint_with_prefix

def parse_args():
    parser = argparse.ArgumentParser(description='Train a pose model')
    parser.add_argument('checkpoint_path', help='path to the checkpoint')

    args = parser.parse_args()

    # Check if the checkpoint path exists
    if not os.path.exists(args.checkpoint_path):
        print("\033[91m" + "Error: Checkpoint path does not exist." + "\033[0m")
        exit(1)

    return args

def main():
    args = parse_args()

    checkpoint_path = args.checkpoint_path
    checkpoint_dir = os.path.dirname(checkpoint_path)
    checkpoint_name = os.path.basename(checkpoint_path)

    checkpoint = _load_checkpoint_with_prefix('backbone.', checkpoint_path)

    new_checkpoint_name = checkpoint_name.replace('.pth', '_clean.pth')
    new_checkpoint_path = os.path.join(checkpoint_dir, new_checkpoint_name)

    print('\033[93m' + 'Saving checkpoint after removing "backbone." prefix to {}'.format(new_checkpoint_path) + '\033[0m')
    # Save the modified checkpoint
    torch.save(checkpoint, new_checkpoint_path)

if __name__ == "__main__":
    main()

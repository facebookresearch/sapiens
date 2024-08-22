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
    parser = argparse.ArgumentParser(description='Clean checkpoints by removing unnecessary data')
    parser.add_argument('--checkpoint_dir', help='Directory containing checkpoints')
    parser.add_argument('--checkpoint_path', help='Path to a single checkpoint')

    args = parser.parse_args()

    if not args.checkpoint_dir and not args.checkpoint_path:
        print("\033[91m" + "Error: Either checkpoint_dir or checkpoint_path must be provided." + "\033[0m")
        exit(1)
    
    if args.checkpoint_path and not os.path.exists(args.checkpoint_path):
        print("\033[91m" + "Error: Checkpoint path does not exist." + "\033[0m")
        exit(1)

    if args.checkpoint_dir and not os.path.exists(args.checkpoint_dir):
        print("\033[91m" + "Error: Checkpoint directory does not exist." + "\033[0m")
        exit(1)

    return args

def clean_checkpoint(checkpoint_path):
    print("\033[96m" + f"Cleaning pretrained checkpoint: {checkpoint_path}" + "\033[0m")

    checkpoint = _load_checkpoint_with_prefix('backbone.', checkpoint_path)
    new_checkpoint_path = os.path.join(os.path.dirname(checkpoint_path), os.path.basename(checkpoint_path).replace('.pth', '_clean.pth'))

    print('\033[93m' + f'Saving after removing "backbone." prefix checkpoint to {new_checkpoint_path}' + '\033[0m')
    print()
    torch.save(checkpoint, new_checkpoint_path)
    return

def main():
    args = parse_args()

    if args.checkpoint_dir:
        checkpoint_files = []
        for root, _, files in os.walk(args.checkpoint_dir):
            for file in files:
                if file.endswith('.pth'):
                    checkpoint_files.append(os.path.join(root, file))
        
        checkpoint_files.sort()

        for checkpoint_path in checkpoint_files:
            clean_checkpoint(checkpoint_path)
    
    elif args.checkpoint_path:
        clean_checkpoint(args.checkpoint_path)

if __name__ == "__main__":
    main()

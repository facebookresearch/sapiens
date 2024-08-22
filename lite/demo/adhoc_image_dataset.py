# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import cv2

class AdhocImageDataset(torch.utils.data.Dataset):
    def __init__(self, image_list, shape=None, mean=None, std=None):
        self.image_list = image_list
        if shape:
            assert len(shape) == 2
        if mean or std:
            assert len(mean) == 3
            assert len(std) == 3
        self.shape = shape
        self.mean = torch.tensor(mean) if mean else None
        self.std = torch.tensor(std) if std else None

    def __len__(self):
        return len(self.image_list)
    
    def _preprocess(self, img):
        if self.shape:
            img = cv2.resize(img, (self.shape[1], self.shape[0]), interpolation=cv2.INTER_LINEAR)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img)
        img = img[[2, 1, 0], ...].float()
        if self.mean is not None and self.std is not None:
            mean=self.mean.view(-1, 1, 1)
            std=self.std.view(-1, 1, 1)
            img = (img - mean) / std
        return img
    
    def __getitem__(self, idx):
        orig_img_dir = self.image_list[idx]
        orig_img = cv2.imread(orig_img_dir)
        # orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
        img = self._preprocess(orig_img)
        return orig_img_dir, orig_img, img
        
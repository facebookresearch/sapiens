# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os.path as osp
from typing import Optional, Sequence, Tuple, Union

import cv2
import numpy as np

from mmengine.hooks import Hook
from mmengine.registry import HOOKS
from mmengine.utils.dl_utils import tensor2imgs

DATA_BATCH = Optional[Union[dict, tuple, list]]


# TODO: Due to interface changes, the current class
#  functions incorrectly
@HOOKS.register_module()
class NaiveVisualizationHook(Hook):
    """Show or Write the predicted results during the process of testing.

    Args:
        interval (int): Visualization interval. Defaults to 1.
        draw_gt (bool): Whether to draw the ground truth. Defaults to True.
        draw_pred (bool): Whether to draw the predicted result.
            Defaults to True.
    """
    priority = 'NORMAL'

    def __init__(self,
                 interval: int = 1,
                 draw_gt: bool = True,
                 draw_pred: bool = True):
        self.draw_gt = draw_gt
        self.draw_pred = draw_pred
        self._interval = interval

    def _unpad(self, input: np.ndarray, unpad_shape: Tuple[int,
                                                           int]) -> np.ndarray:
        """Unpad the input image.

        Args:
            input (np.ndarray): The image to unpad.
            unpad_shape (tuple): The shape of image before padding.

        Returns:
            np.ndarray: The image before padding.
        """
        unpad_width, unpad_height = unpad_shape
        unpad_image = input[:unpad_height, :unpad_width]
        return unpad_image

    def before_train(self, runner) -> None:
        """Call add_graph method of visualizer.

        Args:
            runner (Runner): The runner of the training process.
        """
        runner.visualizer.add_graph(runner.model, None)

    def after_test_iter(self,
                        runner,
                        batch_idx: int,
                        data_batch: DATA_BATCH = None,
                        outputs: Optional[Sequence] = None) -> None:
        """Show or Write the predicted results.

        Args:
            runner (Runner): The runner of the training process.
            batch_idx (int): The index of the current batch in the test loop.
            data_batch (dict or tuple or list, optional): Data from dataloader.
            outputs (Sequence, optional): Outputs from model.
        """
        if self.every_n_inner_iters(batch_idx, self._interval):
            for data, output in zip(data_batch, outputs):  # type: ignore
                input = data['inputs']
                data_sample = data['data_sample']
                input = tensor2imgs(input,
                                    **data_sample.get('img_norm_cfg',
                                                      dict()))[0]
                # TODO We will implement a function to revert the augmentation
                # in the future.
                ori_shape = (data_sample.ori_width, data_sample.ori_height)
                if 'pad_shape' in data_sample:
                    input = self._unpad(input,
                                        data_sample.get('scale', ori_shape))
                origin_image = cv2.resize(input, ori_shape)
                name = osp.basename(data_sample.img_path)
                runner.visualizer.add_datasample(name, origin_image,
                                                 data_sample, output,
                                                 self.draw_gt, self.draw_pred)

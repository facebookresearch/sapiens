# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from itertools import zip_longest
from collections import OrderedDict
from typing import Dict, Optional, Union, Tuple
import torch
from torch import Tensor
from mmengine.optim import OptimWrapper

import torch
import os
import numpy as np
import cv2

from mmpose.registry import MODELS
from mmpose.utils.typing import (ConfigType, InstanceList, OptConfigType,
                                 OptMultiConfig, PixelDataList, SampleList)
from .base import BasePoseEstimator

@MODELS.register_module()
class Pose3dTopdownEstimator(BasePoseEstimator):
    def __init__(self,
                 backbone: ConfigType,
                 neck: OptConfigType = None,
                 head: OptConfigType = None,
                 pose3d_head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None,
                 metainfo: Optional[dict] = None):
        super().__init__(
            backbone=backbone,
            neck=neck,
            head=head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg,
            metainfo=metainfo)
        
        if pose3d_head is not None:
            self.pose3d_head = MODELS.build(pose3d_head)
        
        ## custom logic to deal with loss spikes on RSC
        self.rank = int(os.environ.get("RANK", 0))
        return

    def loss(self, inputs: Tensor, data_samples: SampleList) -> dict:
        feats = self.extract_feat(inputs) ## tuple of size 1, for vit-H, B x 1280 x 16 x 12
        losses = dict()

        if self.with_head:
            loss, pose2d_preds = self.head.loss(feats, data_samples, train_cfg=self.train_cfg) ## 2d pose. pose2d_preds is B x K x H x W
            pose3d_loss, preds_dict = self.pose3d_head.loss(feats, pose2d_preds, data_samples, train_cfg=self.train_cfg) ## 3d pose
            losses.update(loss)
            losses.update(pose3d_loss)
            preds_dict['pose2d'] = pose2d_preds

        return losses, preds_dict

    def train_step(self, data: Union[dict, tuple, list],
                   optim_wrapper: OptimWrapper) -> Dict[str, torch.Tensor]:
        # Enable automatic mixed precision training context.
        with optim_wrapper.optim_context(self):
            data = self.data_preprocessor(data, True)
            losses, preds = self._run_forward(data, mode='loss')  # type: ignore
        parsed_losses, log_vars = self.parse_losses(losses)  # type: ignore

        optim_wrapper.update_params(parsed_losses)

        log_vars['vis_preds'] = preds
        return log_vars

    ## from mmengine. The loss spike handling is done here.
    ## As this function is called in both single node and slurm mode
    def parse_losses(
        self, losses: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        log_vars = []
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars.append([loss_name, loss_value.mean()])
            elif is_list_of(loss_value, torch.Tensor):
                log_vars.append(
                    [loss_name,
                     sum(_loss.mean() for _loss in loss_value)])
            else:
                raise TypeError(
                    f'{loss_name} is not a tensor or list of tensors')

        loss = sum(value for key, value in log_vars if 'loss' in key)
        log_vars.insert(0, ['loss', loss])
        log_vars = OrderedDict(log_vars)  # type: ignore

        ##------------loss spike and nan issues handled here---------------
        if loss.isnan().item():
            print("\033[91mWarning: Train loss is nan!\033[0m")
            loss = 0.0*loss
            return loss, log_vars

        return loss, log_vars  # type: ignore

    def predict(self, inputs: Tensor, data_samples: SampleList) -> SampleList:
        assert self.with_head, ('The model must have head to perform prediction.')

        feats = self.extract_feat(inputs)
        batch_pred_instances = self.head.predict(feats, data_samples, test_cfg=self.test_cfg) ## pred_isntances
        batch_pred_fields = None
        batch_pred_instances = self.pose3d_head.predict(feats, batch_pred_instances, data_samples, test_cfg=self.test_cfg) ## pred

        ## ----debug---
        # import ipdb; ipdb.set_trace()
        # image = inputs[0].cpu().numpy() ## 3 x 1024 x 768
        # image = image.transpose(1, 2, 0) ## 1024 x 768 x 3

        # ## renormalize to 0 to 255
        # image = np.clip(image*255, 0, 255)
        # image = image.astype(np.uint8)
        # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # ## draw keypoints
        # keypoints = batch_pred_instances[0].keypoints[0] ## 308 x 2
        # for i in range(keypoints.shape[0]):
        #     x, y = keypoints[i]
        #     if x > 0 and y > 0:
        #         cv2.circle(image, (int(x), int(y)), 3, (0, 255, 0), -1)
        
        # cv2.imwrite('keypoints.png', image)
        # import ipdb; ipdb.set_trace()
        ##------debug--------
        results = self.add_pred_to_datasample(batch_pred_instances,
                                              batch_pred_fields,
                                              data_samples)
        
        return results

    def add_pred_to_datasample(self, batch_pred_instances: InstanceList,
                               batch_pred_fields: Optional[PixelDataList],
                               batch_data_samples: SampleList) -> SampleList:
        """Add predictions into data samples.

        Args:
            batch_pred_instances (List[InstanceData]): The predicted instances
                of the input data batch
            batch_pred_fields (List[PixelData], optional): The predicted
                fields (e.g. heatmaps) of the input batch
            batch_data_samples (List[PoseDataSample]): The input data batch

        Returns:
            List[PoseDataSample]: A list of data samples where the predictions
            are stored in the ``pred_instances`` field of each data sample.
        """
        assert len(batch_pred_instances) == len(batch_data_samples)

        if batch_pred_fields is None:
            batch_pred_fields = []
        output_keypoint_indices = self.test_cfg.get('output_keypoint_indices', None)

        for pred_instances, pred_fields, data_sample in zip_longest(
                batch_pred_instances, batch_pred_fields, batch_data_samples):

            gt_instances = data_sample.gt_instances

            # convert keypoint coordinates from input space to image space
            bbox_centers = gt_instances.bbox_centers
            bbox_scales = gt_instances.bbox_scales
            input_size = data_sample.metainfo['input_size']

            ## convert keypoints to original image size
            pred_instances.keypoints = pred_instances.keypoints / input_size \
                * bbox_scales + bbox_centers - 0.5 * bbox_scales

            # ##----------debug-----------
            # import ipdb; ipdb.set_trace()
            # image = cv2.imread(data_sample.img_path)
            # keypoints = pred_instances.keypoints[0] ## 308 x 2

            # for i in range(keypoints.shape[0]):
            #     x, y = keypoints[i]
            #     if x > 0 and y > 0:
            #         cv2.circle(image, (int(x), int(y)), 3, (0, 255, 0), -1)
            
            # cv2.imwrite('keypoints.png', image)
            # import ipdb; ipdb.set_trace()
            # ##----------debug-----------
            if output_keypoint_indices is not None:
                # select output keypoints with given indices
                num_keypoints = pred_instances.keypoints.shape[1]
                for key, value in pred_instances.all_items():
                    if key.startswith('keypoint'):
                        pred_instances.set_field(
                            value[:, output_keypoint_indices], key)

            # add bbox information into pred_instances
            pred_instances.bboxes = gt_instances.bboxes
            pred_instances.bbox_scores = gt_instances.bbox_scores

            data_sample.pred_instances = pred_instances

            if pred_fields is not None:
                if output_keypoint_indices is not None:
                    # select output heatmap channels with keypoint indices
                    # when the number of heatmap channel matches num_keypoints
                    for key, value in pred_fields.all_items():
                        if value.shape[0] != num_keypoints:
                            continue
                        pred_fields.set_field(value[output_keypoint_indices],
                                              key)
                data_sample.pred_fields = pred_fields

        return batch_data_samples

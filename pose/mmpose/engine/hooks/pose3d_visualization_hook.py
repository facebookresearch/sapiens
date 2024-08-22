# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import warnings
from typing import Optional, Sequence

import torch
import numpy as np
import cv2
import mmcv
import torchvision
import torchvision.transforms as transforms
import mmengine
import mmengine.fileio as fileio
from mmengine.hooks import Hook
from mmengine.runner import Runner
from mmengine.visualization import Visualizer
from matplotlib import pyplot as plt
from mmpose.registry import HOOKS
from mmpose.structures import PoseDataSample, merge_data_samples
from mmpose.registry import VISUALIZERS
from mmengine.structures import InstanceData

@HOOKS.register_module()
class Pose3dVisualizationHook(Hook):
    def __init__(
        self,
        enable: bool = False,
        interval: int = 50,
        kpt_thr: float = 0.3,
        show: bool = False,
        wait_time: float = 0.,
        max_vis_samples: int = 16,
        scale: int = 4,
        line_width: int = 4,
        radius: int = 4,
        out_dir: Optional[str] = None,
        backend_args: Optional[dict] = None,
    ):
        self._visualizer: Visualizer = Visualizer.get_current_instance()
        self.interval = interval
        self.kpt_thr = kpt_thr
        self.show = show
        if self.show:
            # No need to think about vis backends.
            self._visualizer._vis_backends = {}
            warnings.warn('The show is True, it means that only '
                          'the prediction results are visualized '
                          'without storing data, so vis_backends '
                          'needs to be excluded.')

        self.wait_time = wait_time
        self.enable = enable
        self.out_dir = out_dir
        self._test_index = 0
        self.backend_args = backend_args
        self.max_vis_samples = max_vis_samples
        self.scale = scale
        self.init_visualizer = False
        self._visualizer.line_width = line_width
        self._visualizer.radius = radius
        return

    def after_train_iter(self, runner: Runner, batch_idx: int, data_batch: dict,
                       outputs: Sequence[PoseDataSample]) -> None:
        """Run after every ``self.interval`` validation iterations.

        Args:
            runner (:obj:`Runner`): The runner of the validation process.
            batch_idx (int): The index of the current batch in the val loop.
            data_batch (dict): Data from dataloader.
            outputs (Sequence[:obj:`PoseDataSample`]): Outputs from model.
        """
        if self.enable is False:
            return

        # ## check if the rank is 0
        if not runner.rank == 0:
            return

        # There is no guarantee that the same batch of images
        # is visualized for each evaluation.
        total_curr_iter = runner.iter

        if total_curr_iter % self.interval != 0:
            return
        
        ## we divide by 255 to be compatible with the visualization functions
        image = torch.cat([input.unsqueeze(dim=0)/255 for input in data_batch['inputs']], dim=0) ## B x 3 x H x W, not normalized in BGR format
        output = outputs['vis_preds']['pose2d'].detach() ## B x 308 x 2
        output_pose3d = outputs['vis_preds']['pose3d'].detach() ## B x 308 x 3

        batch_size = min(self.max_vis_samples, len(image))

        if self.init_visualizer == False:
            self._visualizer.set_dataset_meta(runner.train_dataloader.dataset.metainfo) ## this sets the skeleton and skeleton links colors
            self.init_visualizer = True

        image = image[:batch_size]
        output = output[:batch_size]
        output_pose3d = output_pose3d[:batch_size].detach().cpu() ## B x 308 x 3

        # target = []
        # for i in range(batch_size):
        #     target.append(torch.tensor(data_batch['data_samples'][i].get('gt_instances').get('transformed_keypoints')))
        # target = torch.cat(target, dim=0) ## B x 308 x 2

        target = []
        for i in range(batch_size):
            target.append(data_batch['data_samples'][i].get('gt_fields').get('heatmaps').unsqueeze(dim=0))
        target = torch.cat(target, dim=0)

        target_weight = []
        for i in range(batch_size):
            target_weight.append(torch.tensor(data_batch['data_samples'][i].get('gt_instances').get('keypoints_visible')))
        target_weight = torch.cat(target_weight, dim=0) ## B x 308

        gt_K = []
        for i in range(batch_size):
            gt_K.append(torch.from_numpy(data_batch['data_samples'][i].K.astype(np.float32)).unsqueeze(dim=0)) ## 3 x 3
        gt_K = torch.cat(gt_K, dim=0) ## B x 3 x 3

        gt_pose3d = []
        for i in range(batch_size):
            gt_pose3d.append(torch.from_numpy(data_batch['data_samples'][i].gt_instances.pose3d[0].astype(np.float32)))
        gt_pose3d = torch.stack(gt_pose3d) ## B x 308 x 4

        ## compute pose2d_from_pose3d using gt_K
        pose2d_homogeneous = torch.bmm(output_pose3d, gt_K.transpose(1, 2))  # [B, 308, 3]
        pose2d = pose2d_homogeneous[:, :, :2] / (pose2d_homogeneous[:, :, 2:3] + 1e-5)

        gt_pose2d_homogeneous = torch.bmm(gt_pose3d, gt_K.transpose(1, 2)) # [B, 308, 3]
        gt_pose2d = gt_pose2d_homogeneous[:, :, :2] / (gt_pose2d_homogeneous[:, :, 2:3] + 1e-5)
        ##------------------------------------
        pose2d_vis_dir = os.path.join(runner.work_dir, 'vis_data', '2d')
        pose3d_vis_dir = os.path.join(runner.work_dir, 'vis_data', '3d')

        if not os.path.exists(pose2d_vis_dir):
            os.makedirs(pose2d_vis_dir, exist_ok=True)
        
        if not os.path.exists(pose3d_vis_dir):
            os.makedirs(pose3d_vis_dir, exist_ok=True)

        pose2d_prefix = os.path.join(pose2d_vis_dir, 'train')
        pose3d_prefix = os.path.join(pose3d_vis_dir, 'train')

        suffix = str(total_curr_iter).zfill(6)

        original_image = image

        self.save_batch_image_with_joints(255*original_image, target, target_weight, '{}_{}_gt.jpg'.format(pose2d_prefix, suffix), scale=self.scale, is_rgb=False)
        self.save_batch_image_with_joints(255*original_image, output, torch.ones_like(target_weight), '{}_{}_pred.jpg'.format(pose2d_prefix, suffix), scale=self.scale, is_rgb=False)
        # self.save_batch_image_with_pose3d(255*original_image, target, target_weight, '{}_{}_gt.jpg'.format(pose2d_prefix, suffix), is_rgb=False)
        # self.save_batch_image_with_pose3d(255*original_image, output, torch.ones_like(target_weight), '{}_{}_pred.jpg'.format(pose2d_prefix, suffix), is_rgb=False)

        self.save_batch_image_with_pose3d(255*original_image, gt_pose2d, torch.ones_like(target_weight), '{}_{}_pose3d_gt.jpg'.format(pose3d_prefix, suffix), is_rgb=False)
        self.save_batch_image_with_pose3d(255*original_image, pose2d, torch.ones_like(target_weight), '{}_{}_pose3d_pred.jpg'.format(pose3d_prefix, suffix), is_rgb=False)

        return
    
    def save_batch_image_with_pose3d(self, batch_image, batch_joints, batch_target_weight, file_name, dataset_info=None, is_rgb=True, nrow=8, padding=2):
        B, C, H, W = batch_image.size()
        num_joints = batch_joints.size(1)

        if isinstance(batch_joints, torch.Tensor):
            batch_joints = batch_joints.detach().cpu().numpy()

        if isinstance(batch_target_weight, torch.Tensor):
            batch_target_weight = batch_target_weight.cpu().numpy()
            batch_target_weight = batch_target_weight.reshape(B, num_joints) ## B x 17

        grid = []

        for i in range(B):
            image = batch_image[i].permute(1, 2, 0).cpu().numpy() #image_size x image_size x BGR. if is_rgb is False.
            image = image.copy()
            kps = batch_joints[i] ## N x 2
            kps_vis = batch_target_weight[i]
            kps_score = batch_target_weight[i]

            ## set min val kps to 0.0
            kps = np.maximum(kps, 0.0)
            kps[:, 0] = np.minimum(kps[:, 0], image.shape[1]) ## x
            kps[:, 1] = np.minimum(kps[:, 1], image.shape[0]) ## y

            if is_rgb == False:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # convert bgr to rgb image
            
            image = image.astype(np.uint8)

            instances = InstanceData(metainfo=dict(keypoints=[kps], keypoints_visible=[kps_vis], keypoint_scores=[kps_score]))
            kp_vis_image = self._visualizer._draw_instances_kpts(image, instances=instances) ## H, W, C, rgb image
            kp_vis_image = cv2.cvtColor(kp_vis_image, cv2.COLOR_RGB2BGR) ## convert rgb to bgr image

            kp_vis_image = kp_vis_image.transpose((2, 0, 1)).astype(np.float32)
            kp_vis_image = torch.from_numpy(kp_vis_image.copy())
            grid.append(kp_vis_image)

        grid = torchvision.utils.make_grid(grid, nrow, padding)
        ndarr = grid.byte().permute(1, 2, 0).cpu().numpy()
        cv2.imwrite(file_name, ndarr)
        return


    def save_batch_image_with_joints(self, batch_image, batch_heatmaps, batch_target_weight, file_name, dataset_info=None, is_rgb=True, scale=4, nrow=8, padding=2):
        '''
        batch_image: [batch_size, channel, height, width]
        batch_joints: [batch_size, num_joints, 3],
        batch_joints_vis: [batch_size, num_joints, 1],
        }
        '''

        B, C, H, W = batch_image.size()
        num_joints = batch_heatmaps.size(1)

        ## check if type of batch_heatmaps is numpy.ndarray
        if isinstance(batch_heatmaps, np.ndarray):
            batch_joints, batch_scores = get_max_preds(batch_heatmaps)
        else:
            batch_joints, batch_scores = get_max_preds(batch_heatmaps.detach().cpu().numpy())

        batch_joints = batch_joints*scale ## 4 is the ratio of output heatmap and input image

        if isinstance(batch_joints, torch.Tensor):
            batch_joints = batch_joints.cpu().numpy()

        if isinstance(batch_target_weight, torch.Tensor):
            batch_target_weight = batch_target_weight.cpu().numpy()
            batch_target_weight = batch_target_weight.reshape(B, num_joints) ## B x 17

        grid = []

        for i in range(B):
            image = batch_image[i].permute(1, 2, 0).cpu().numpy() #image_size x image_size x BGR. if is_rgb is False.
            image = image.copy()
            kps = batch_joints[i]
            kps_vis = batch_target_weight[i]
            kps_score = batch_scores[i].reshape(-1)

            if is_rgb == False:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # convert bgr to rgb image

            instances = InstanceData(metainfo=dict(keypoints=[kps], keypoints_visible=[kps_vis], keypoint_scores=[kps_score]))
            kp_vis_image = self._visualizer._draw_instances_kpts(image, instances=instances) ## H, W, C, rgb image
            kp_vis_image = cv2.cvtColor(kp_vis_image, cv2.COLOR_RGB2BGR) ## convert rgb to bgr image

            kp_vis_image = kp_vis_image.transpose((2, 0, 1)).astype(np.float32)
            kp_vis_image = torch.from_numpy(kp_vis_image.copy())
            grid.append(kp_vis_image)

        grid = torchvision.utils.make_grid(grid, nrow, padding)
        ndarr = grid.byte().permute(1, 2, 0).cpu().numpy()
        cv2.imwrite(file_name, ndarr)
        return
    
###------------------helpers-----------------------
###------------------------------------------------------
def batch_unnormalize_image(images, mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]):
    normalize = transforms.Normalize(mean=mean, std=std)
    images[:, 0, :, :] = (images[:, 0, :, :]*normalize.std[0]) + normalize.mean[0]
    images[:, 1, :, :] = (images[:, 1, :, :]*normalize.std[1]) + normalize.mean[1]
    images[:, 2, :, :] = (images[:, 2, :, :]*normalize.std[2]) + normalize.mean[2]
    return images

def get_max_preds(batch_heatmaps):
    '''
    get predictions from score maps
    heatmaps: numpy.ndarray([batch_size, num_joints, height, width])
    '''
    assert isinstance(batch_heatmaps, np.ndarray), \
        'batch_heatmaps should be numpy.ndarray'
    assert batch_heatmaps.ndim == 4, 'batch_images should be 4-ndim'

    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    width = batch_heatmaps.shape[3]
    heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
    idx = np.argmax(heatmaps_reshaped, 2) ## B x 17
    maxvals = np.amax(heatmaps_reshaped, 2) ## B x 17

    maxvals = maxvals.reshape((batch_size, num_joints, 1)) ## B x 17 x 1
    idx = idx.reshape((batch_size, num_joints, 1)) ## B x 17 x 1

    preds = np.tile(idx, (1, 1, 2)).astype(np.float32) ## B x 17 x 2, like repeat in pytorch

    preds[:, :, 0] = (preds[:, :, 0]) % width
    preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)

    pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
    pred_mask = pred_mask.astype(np.float32)

    preds *= pred_mask
    return preds, maxvals

# ------------------------------------------------------------------------------------

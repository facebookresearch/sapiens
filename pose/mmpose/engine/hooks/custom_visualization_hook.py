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


@HOOKS.register_module()
class CustomPoseVisualizationHook(Hook):
    """Pose Estimation Visualization Hook. Used to visualize validation and
    testing process prediction results.

    In the testing phase:

    1. If ``show`` is True, it means that only the prediction results are
        visualized without storing data, so ``vis_backends`` needs to
        be excluded.
    2. If ``out_dir`` is specified, it means that the prediction results
        need to be saved to ``out_dir``. In order to avoid vis_backends
        also storing data, so ``vis_backends`` needs to be excluded.
    3. ``vis_backends`` takes effect if the user does not specify ``show``
        and `out_dir``. You can set ``vis_backends`` to WandbVisBackend or
        TensorboardVisBackend to store the prediction result in Wandb or
        Tensorboard.

    Args:
        enable (bool): whether to draw prediction results. If it is False,
            it means that no drawing will be done. Defaults to False.
        interval (int): The interval of visualization. Defaults to 50.
        score_thr (float): The threshold to visualize the bboxes
            and masks. Defaults to 0.3.
        show (bool): Whether to display the drawn image. Default to False.
        wait_time (float): The interval of show (s). Defaults to 0.
        out_dir (str, optional): directory where painted images
            will be saved in testing process.
        backend_args (dict, optional): Arguments to instantiate the preifx of
            uri corresponding backend. Defaults to None.
    """

    def __init__(
        self,
        enable: bool = False,
        interval: int = 50,
        kpt_thr: float = 0.3,
        show: bool = False,
        wait_time: float = 0.,
        max_vis_samples: int = 16,
        scale: int = 4,
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
        output = outputs['vis_preds'].detach() ## B x 17 x H x W

        batch_size = min(self.max_vis_samples, len(image))

        image = image[:batch_size]
        output = output[:batch_size]

        target = []
        for i in range(batch_size):
            target.append(data_batch['data_samples'][i].get('gt_fields').get('heatmaps').unsqueeze(dim=0))

        target = torch.cat(target, dim=0)

        target_weight = []
        for i in range(batch_size):
            target_weight.append(data_batch['data_samples'][i].get('gt_instance_labels').get('keypoints_visible').unsqueeze(dim=0))
        target_weight = torch.cat(target_weight, dim=0)

        ##------------------------------------
        vis_dir = os.path.join(runner.work_dir, 'vis_data')
        if not os.path.exists(vis_dir):
            os.makedirs(vis_dir, exist_ok=True)

        prefix = os.path.join(vis_dir, 'train')
        suffix = str(total_curr_iter).zfill(6)

        original_image = image

        save_batch_heatmaps(original_image, target, '{}_{}_hm_gt.jpg'.format(prefix, suffix), normalize=False, scale=self.scale, is_rgb=False)
        save_batch_heatmaps(original_image, output, '{}_{}_hm_pred.jpg'.format(prefix, suffix), normalize=False, scale=self.scale, is_rgb=False)
        save_batch_image_with_joints(255*original_image, target, target_weight, \
                                     '{}_{}_gt.jpg'.format(prefix, suffix), scale=self.scale, is_rgb=False)
        save_batch_image_with_joints(255*original_image, output, torch.ones_like(target_weight), \
                                     '{}_{}_pred.jpg'.format(prefix, suffix), scale=self.scale, is_rgb=False)

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


def save_batch_heatmaps(batch_image, batch_heatmaps, file_name, normalize=True, scale=4, is_rgb=True):
    '''
    batch_image: [batch_size, channel, height, width]
    batch_heatmaps: ['batch_size, num_joints, height, width]
    file_name: saved file name
    '''
    ## normalize image
    if normalize:
        batch_image = batch_image.clone()
        min = float(batch_image.min())
        max = float(batch_image.max())

        batch_image.add_(-min).div_(max - min + 1e-5)

    ## check if type of batch_heatmaps is numpy.ndarray
    if isinstance(batch_heatmaps, np.ndarray):
        preds, maxvals = get_max_preds(batch_heatmaps)
        batch_heatmaps = torch.from_numpy(batch_heatmaps)
    else:
        preds, maxvals = get_max_preds(batch_heatmaps.detach().cpu().numpy())
    
    preds = preds*scale ## scale to original image size

    batch_size = batch_heatmaps.size(0)
    num_joints = batch_heatmaps.size(1)
    heatmap_height = int(batch_heatmaps.size(2)*scale)
    heatmap_width = int(batch_heatmaps.size(3)*scale)

    grid_image = np.zeros((batch_size*heatmap_height,
                           (num_joints+1)*heatmap_width,
                           3),
                          dtype=np.uint8)

    for i in range(batch_size):
        image = batch_image[i].mul(255)\
                              .clamp(0, 255)\
                              .byte()\
                              .permute(1, 2, 0)\
                              .cpu().numpy()
        heatmaps = batch_heatmaps[i].mul(255)\
                                    .clamp(0, 255)\
                                    .byte()\
                                    .cpu().numpy()

        if is_rgb == True:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        resized_image = cv2.resize(image, (int(heatmap_width), int(heatmap_height)))

        height_begin = heatmap_height * i
        height_end = heatmap_height * (i + 1)
        for j in range(num_joints):
            cv2.circle(resized_image,
                       (int(preds[i][j][0]), int(preds[i][j][1])),
                       1, [0, 0, 255], 1)
            heatmap = heatmaps[j, :, :]
            colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            colored_heatmap = cv2.resize(colored_heatmap, (int(heatmap_width), int(heatmap_height)))
            masked_image = colored_heatmap*0.7 + resized_image*0.3
            cv2.circle(masked_image,
                       (int(preds[i][j][0]), int(preds[i][j][1])),
                       1, [0, 0, 255], 1)

            width_begin = heatmap_width * (j+1)
            width_end = heatmap_width * (j+2)
            grid_image[height_begin:height_end, width_begin:width_end, :] = \
                masked_image

        grid_image[height_begin:height_end, 0:heatmap_width, :] = resized_image

    cv2.imwrite(file_name, grid_image)


def save_batch_image_with_joints(batch_image, batch_heatmaps, batch_target_weight, file_name, is_rgb=True, scale=4, nrow=8, padding=2):
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
        batch_joints, _ = get_max_preds(batch_heatmaps)
    else:
        batch_joints, _ = get_max_preds(batch_heatmaps.detach().cpu().numpy())

    batch_joints = batch_joints*scale ## 4 is the ratio of output heatmap and input image

    if isinstance(batch_joints, torch.Tensor):
        batch_joints = batch_joints.cpu().numpy()

    if isinstance(batch_target_weight, torch.Tensor):
        batch_target_weight = batch_target_weight.cpu().numpy()
        batch_target_weight = batch_target_weight.reshape(B, num_joints) ## B x 17

    grid = []

    for i in range(B):
        image = batch_image[i].permute(1, 2, 0).cpu().numpy() #image_size x image_size x RGB
        image = image.copy()
        kps = batch_joints[i]

        kps_vis = batch_target_weight[i].reshape(num_joints, 1)
        kps = np.concatenate((kps, kps_vis), axis=1)

        ## we need rgb images. if BGR convert to RGB
        if is_rgb == False:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        kp_vis_image = coco_vis_keypoints(image, kps, vis_thres=0.3, alpha=0.7) ## H, W, C
        kp_vis_image = kp_vis_image.transpose((2, 0, 1)).astype(np.float32)
        kp_vis_image = torch.from_numpy(kp_vis_image.copy())
        grid.append(kp_vis_image)

    grid = torchvision.utils.make_grid(grid, nrow, padding)
    ndarr = grid.byte().permute(1, 2, 0).cpu().numpy()
    ndarr = cv2.cvtColor(ndarr, cv2.COLOR_RGB2BGR)
    cv2.imwrite(file_name, ndarr)
    return


###------------------------vis-------------------------------
# standard COCO format, 17 joints
COCO_KP_ORDER = [
        'nose',
        'left_eye',
        'right_eye',
        'left_ear',
        'right_ear',
        'left_shoulder',
        'right_shoulder',
        'left_elbow',
        'right_elbow',
        'left_wrist',
        'right_wrist',
        'left_hip',
        'right_hip',
        'left_knee',
        'right_knee',
        'left_ankle',
        'right_ankle'
    ]


def kp_connections(keypoints):
    kp_lines = [
        [keypoints.index('left_eye'), keypoints.index('right_eye')],
        [keypoints.index('left_eye'), keypoints.index('nose')],
        [keypoints.index('right_eye'), keypoints.index('nose')],
        [keypoints.index('right_eye'), keypoints.index('right_ear')],
        [keypoints.index('left_eye'), keypoints.index('left_ear')],
        [keypoints.index('right_shoulder'), keypoints.index('right_elbow')],
        [keypoints.index('right_elbow'), keypoints.index('right_wrist')],
        [keypoints.index('left_shoulder'), keypoints.index('left_elbow')],
        [keypoints.index('left_elbow'), keypoints.index('left_wrist')],
        [keypoints.index('right_hip'), keypoints.index('right_knee')],
        [keypoints.index('right_knee'), keypoints.index('right_ankle')],
        [keypoints.index('left_hip'), keypoints.index('left_knee')],
        [keypoints.index('left_knee'), keypoints.index('left_ankle')],
        [keypoints.index('right_shoulder'), keypoints.index('left_shoulder')],
        [keypoints.index('right_hip'), keypoints.index('left_hip')],
    ]
    return kp_lines


COCO_KP_CONNECTIONS = kp_connections(COCO_KP_ORDER)

# ------------------------------------------------------------------------------------
def coco_vis_keypoints(image, kps, vis_thres=0.3, alpha=0.7):
    # image is [image_size, image_size, RGB] #numpy array
    # kps is [17, 3] #numpy array
    kps = kps.astype(np.int16)
    bgr_image = image[:, :, ::-1] ##if this is directly in function call, this produces weird opecv cv2 Umat errors
    kp_image = vis_keypoints(bgr_image, kps.T, vis_thres, alpha) #convert to bgr
    kp_image = kp_image[:, :, ::-1] #bgr to rgb

    return kp_image

# ------------------------------------------------------------------------------------
def vis_keypoints(img, kps, kp_thresh=-1, alpha=0.7):
    """Visualizes keypoints (adapted from vis_one_image).
    kps has shape (3, #keypoints) where 3 rows are (x, y, depth z).
    needs a BGR image as it only uses opencv functions, returns a bgr image
    """
    dataset_keypoints = COCO_KP_ORDER
    kp_lines = COCO_KP_CONNECTIONS

    # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(kp_lines) + 2)]
    colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]

    # Perform the drawing on a copy of the image, to allow for blending.
    kp_mask = np.copy(img)

    # Draw mid shoulder / mid hip first for better visualization.
    mid_shoulder = (
        kps[:2, dataset_keypoints.index('right_shoulder')] +
        kps[:2, dataset_keypoints.index('left_shoulder')]) // 2
    sc_mid_shoulder = np.minimum(
        kps[2, dataset_keypoints.index('right_shoulder')],
        kps[2, dataset_keypoints.index('left_shoulder')])
    mid_hip = (
        kps[:2, dataset_keypoints.index('right_hip')] +
        kps[:2, dataset_keypoints.index('left_hip')]) // 2
    sc_mid_hip = np.minimum(
        kps[2, dataset_keypoints.index('right_hip')],
        kps[2, dataset_keypoints.index('left_hip')])
    nose_idx = dataset_keypoints.index('nose')

    if sc_mid_shoulder > kp_thresh and kps[2, nose_idx] > kp_thresh:
        kp_mask = cv2.line(
            kp_mask, tuple(mid_shoulder), tuple(kps[:2, nose_idx]),
            color=colors[len(kp_lines)], thickness=2, lineType=cv2.LINE_AA)
    if sc_mid_shoulder > kp_thresh and sc_mid_hip > kp_thresh:
        kp_mask = cv2.line(
            kp_mask, tuple(mid_shoulder), tuple(mid_hip),
            color=colors[len(kp_lines) + 1], thickness=2, lineType=cv2.LINE_AA)

    # Draw the keypoints.
    for l in range(len(kp_lines)):
        i1 = kp_lines[l][0]
        i2 = kp_lines[l][1]
        p1 = kps[0, i1], kps[1, i1]
        p2 = kps[0, i2], kps[1, i2]
        if kps[2, i1] > kp_thresh and kps[2, i2] > kp_thresh:
            kp_mask = cv2.line(
                kp_mask, p1, p2,
                color=colors[l], thickness=2, lineType=cv2.LINE_AA)
        if kps[2, i1] > kp_thresh:
            kp_mask = cv2.circle(
                kp_mask, p1,
                radius=3, color=colors[l], thickness=-1, lineType=cv2.LINE_AA)
        if kps[2, i2] > kp_thresh:
            kp_mask = cv2.circle(
                kp_mask, p2,
                radius=3, color=colors[l], thickness=-1, lineType=cv2.LINE_AA)

    ## weird opencv bug on cv2UMat vs numpy
    if type(kp_mask) != type(img):
        kp_mask = kp_mask.get()

    # Blend the keypoints.
    result = cv2.addWeighted(img, 1.0 - alpha, kp_mask, alpha, 0)
    return result
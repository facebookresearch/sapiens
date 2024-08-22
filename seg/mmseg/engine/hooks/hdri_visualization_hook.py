import os.path as osp
import warnings
from typing import Optional, Sequence

import mmcv
import mmengine.fileio as fileio
from mmengine.hooks import Hook
from mmengine.runner import Runner

from mmseg.registry import HOOKS
from mmseg.structures import SegDataSample
from mmseg.visualization import SegLocalVisualizer
from mmengine.model import MMDistributedDataParallel

import matplotlib.pyplot as plt
import matplotlib.cm as cm

import numpy as np
import os
import cv2
import tempfile

##-----------------------------------------------------------------------------------------------------------
@HOOKS.register_module()
class HDRIVisualizationHook(Hook):
    def __init__(self,
                 draw: bool = False,
                 interval: int = 50,
                 show: bool = False,
                 wait_time: float = 0.,
                 max_samples = 4,
                 vis_image_width = 512,
                 vis_image_height = 1024,
                 backend_args: Optional[dict] = None):
        self._visualizer: SegLocalVisualizer = \
            SegLocalVisualizer.get_current_instance()
        self.interval = interval
        self.show = show

        self.wait_time = wait_time
        self.backend_args = backend_args.copy() if backend_args else None
        self.draw = draw

        self.max_samples = max_samples
        self.vis_image_width = vis_image_width
        self.vis_image_height = vis_image_height

    def after_train_iter(self,
                    runner: Runner,
                    batch_idx: int,
                    data_batch: dict,
                    outputs: Sequence[SegDataSample],
                    mode: str = 'val') -> None:
        """Run after every ``self.interval`` validation iterations.

        Args:
            runner (:obj:`Runner`): The runner of the validation process.
            batch_idx (int): The index of the current batch in the val loop.
            data_batch (dict): Data from dataloader.
            outputs (Sequence[:obj:`SegDataSample`]): Outputs from model.
            mode (str): mode (str): Current mode of runner. Defaults to 'val'.
        """

        # ## check if the rank is 0
        if not runner.rank == 0:
            return

        total_curr_iter = runner.iter

        if total_curr_iter % self.interval != 0:
            return

        inputs = data_batch['inputs'] ## list of images as tensor. BGR images. they are made RGB by model preprocessor
        data_samples = data_batch['data_samples']

        ##------------------------------------
        vis_dir = os.path.join(runner.work_dir, 'vis_data')
        if not os.path.exists(vis_dir):
            os.makedirs(vis_dir, exist_ok=True)

        prefix = os.path.join(vis_dir, 'train')
        suffix = str(total_curr_iter).zfill(6)

        # suffix += '_' + data_samples[0].img_path.split('/')[-1].replace('.png', '')
        frame_name = data_samples[0].img_path.split('/')[-1].replace('.png', '')
        camera_name = data_samples[0].img_path.split('/')[-2]
        subject_name = data_samples[0].img_path.split('/')[-5]
        suffix += '_' + subject_name + '_' + camera_name + '_' + frame_name

        batch_size = min(self.max_samples, len(inputs))
        inputs = inputs[:batch_size]
        data_samples = data_samples[:batch_size]
        seg_logits = outputs['vis_preds'][:batch_size] ## B x 3 x H x W

        # Check if the model is wrapped with MMDistributedDataParallel
        model = runner.model.module if isinstance(runner.model, MMDistributedDataParallel) else runner.model
        data_samples_with_pred = model.postprocess_train_result(seg_logits, data_samples) ##

        seg_logits = seg_logits.cpu().detach().numpy() ## B x 3 x H x W
        seg_logits = seg_logits.transpose((0, 2, 3, 1)) ## B x H x W x 3 ## RGB

        vis_images = []

        for i, (input, data_sample_with_pred, seg_logit) in enumerate(zip(inputs, data_samples_with_pred, seg_logits)):
            image = input.permute(1, 2, 0).cpu().numpy() ## bgr image
            image = np.ascontiguousarray(image.copy())

            gt_hdri = data_sample_with_pred.gt_depth_map.data.numpy() ## 3 x H x W
            gt_hdri = gt_hdri.transpose((1, 2, 0)) ## H x W x 3, RGB
            gt_hdri = gt_hdri[:, :, ::-1] ## convert rgb to bgr

            vis_gt_hdri = (gt_hdri * 10 * 255).astype(np.uint8) ## extra amplification of 10 for better vis. the number 10 is arbitrary
            vis_gt_hdri = np.clip(vis_gt_hdri, 0, 255)

            ### resize pred to the size of image
            pred_hdri = seg_logit

            ## clip the pred to [0, 1]
            pred_hdri = np.clip(pred_hdri, 0, 1)
            pred_hdri = pred_hdri[:, :, ::-1] ## convert rgb to bgr

            vis_pred_hdri = (pred_hdri * 10 * 255).astype(np.uint8)
            vis_pred_hdri = np.clip(vis_pred_hdri, 0, 255)

            ## resize vis_gt_hdri and vis_pred_hdri to the size of image
            vis_gt_hdri = cv2.resize(vis_gt_hdri, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_AREA)
            vis_pred_hdri = cv2.resize(vis_pred_hdri, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_AREA)

            vis_image = np.concatenate([image, vis_gt_hdri, vis_pred_hdri], axis=1)
            vis_image = cv2.resize(vis_image, (3*self.vis_image_width, self.vis_image_height), interpolation=cv2.INTER_AREA)
            vis_images.append(vis_image)

        grid_image = np.concatenate(vis_images, axis=0)

        # Save the grid image to a file
        grid_out_file = '{}_{}.jpg'.format(prefix, suffix)
        cv2.imwrite(grid_out_file, grid_image)

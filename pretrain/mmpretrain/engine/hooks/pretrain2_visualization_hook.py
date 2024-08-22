# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
import os
import os.path as osp
import cv2
from typing import Optional, Sequence

from mmengine.fileio import join_path
from mmengine.hooks import Hook
from mmengine.runner import EpochBasedTrainLoop, Runner
from mmengine.visualization import Visualizer

from torch.nn import functional as F
from mmpretrain.registry import HOOKS
from mmpretrain.structures import DataSample
import numpy as np
from mmengine.model import MMDistributedDataParallel

@HOOKS.register_module()
class Pretrain2VisualizationHook(Hook):
    """Classification Visualization Hook. Used to visualize validation and
    testing prediction results.

    - If ``out_dir`` is specified, all storage backends are ignored
      and save the image to the ``out_dir``.
    - If ``show`` is True, plot the result image in a window, please
      confirm you are able to access the graphical interface.

    Args:
        enable (bool): Whether to enable this hook. Defaults to False.
        interval (int): The interval of samples to visualize. Defaults to 5000.
        show (bool): Whether to display the drawn image. Defaults to False.
        out_dir (str, optional): directory where painted images will be saved
            in the testing process. If None, handle with the backends of the
            visualizer. Defaults to None.
        **kwargs: other keyword arguments of
            :meth:`mmpretrain.visualization.UniversalVisualizer.visualize_cls`.
    """

    def __init__(self,
                 enable=False,
                 vis_max_samples = 16,
                 vis_every_iters = 500,
                 out_dir: Optional[str] = None,
                 **kwargs):
        self._visualizer: Visualizer = Visualizer.get_current_instance()

        self.enable = enable
        self.vis_every_iters = vis_every_iters
        self.out_dir = out_dir
        self.vis_max_samples = vis_max_samples

    def _draw_samples(self,
                      runner: Runner,
                      batch_idx: int,
                      data_batch: dict,
                      data_samples: Sequence[DataSample]) -> None:
        """Visualize every ``self.interval`` samples from a data batch.

        Args:
            batch_idx (int): The index of the current batch in the val loop.
            data_batch (dict): Data from dataloader.
            outputs (Sequence[:obj:`DataSample`]): Outputs from model.
            step (int): Global step value to record. Defaults to 0.
        """

        if self.enable is False:
            return
        
        # Check if the model is wrapped with MMDistributedDataParallel
        model = runner.model.module if isinstance(runner.model, MMDistributedDataParallel) else runner.model

        B, C, H, W = data_batch['inputs'].shape

        # prepare images
        images = data_batch['inputs'].float()  ## B x 3 x 4096 x 4096, not normalized in BGR format
        images = F.interpolate(input=images, size=(H//4, W//4), mode='bicubic', align_corners=False) ## 4k -> 1K

        patch_images = model.head.patchify(images) ## B x (num_tokens = 4096) x (16 x 16 x 3). 

        ## get the norm_pix constants from the images
        processed_data_batch = model.data_preprocessor(data_batch, True)
        processed_images = processed_data_batch['inputs'] ## B x 3 x 4096 x 4096, normalized in RGB format. at 4K
        processed_images = F.interpolate(input=processed_images, size=(H//4, W//4), mode='bicubic', align_corners=False) ## 4k -> 1K

        # prepare preds
        patch_preds = data_samples['vis_preds'].detach().cpu() ## this is RGB format, B x (H*W) x (P*P*3)

        if model.head.norm_pix == True:
            mean, var = model.head.get_norm_pix_mean_var(processed_images)
            mean = mean.cpu(); var = var.cpu()
            patch_preds = (patch_preds * (var + 1.e-6)**.5) + mean

        preds_normalized = model.head.unpatchify(patch_preds) ## B x 3 x 1024 x 1024
        preds = model.unnormalize_image(preds_normalized, use_cpu=True)
        preds = preds.clamp(0, 255)

        ## prepare masks
        masks = data_samples['vis_masks'].cpu() ## B x num_tokens. num_tokens = 4096
        masked_patch_images = patch_images*(1-masks).unsqueeze(2) + 100*masks.unsqueeze(2) ## the binary mask: 0 is keep, 1 is remove
        masked_images = model.head.unpatchify(masked_patch_images) ## B x 3 x 4096 x 4096

        batch_size = min(len(images), self.vis_max_samples)

        save_images = []

        for sample_id in range(batch_size):
            image = images[sample_id]
            image = image.permute(1, 2, 0).cpu().numpy().astype('uint8')

            masked_image = masked_images[sample_id]
            masked_image = masked_image.permute(1, 2, 0).cpu().numpy().astype('uint8')

            pred = preds[sample_id]
            pred = pred.permute(1, 2, 0).cpu().numpy().astype('uint8')
            pred = cv2.cvtColor(pred, cv2.COLOR_RGB2BGR)

            save_image = np.concatenate((image, masked_image, pred), axis=1)
            save_images.append(save_image)

        if self.out_dir is not None:
            out_file = join_path(self.out_dir, f'epoch{runner.epoch:04}_iter{runner.iter:06}.png')
        else:
            out_dir = os.path.join(runner.work_dir, 'vis_data')

            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            
            out_file = join_path(out_dir, f'epoch{runner.epoch:04}_iter{runner.iter:06}.png')

        img_height, img_width = save_images[0].shape[:2]

        cols = int(math.ceil(math.sqrt(batch_size)))
        rows = int(math.ceil(batch_size / cols))

        canvas_height = rows * img_height
        canvas_width = cols * img_width

        canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

        for idx, image in enumerate(save_images):
            row = idx // cols
            col = idx % cols
            canvas[row * img_height: (row + 1) * img_height, col * img_width: (col + 1) * img_width] = image
        
        ## downsample canvas by 2x
        canvas = cv2.resize(canvas, (canvas_width//2, canvas_height//2), interpolation=cv2.INTER_AREA)
        cv2.imwrite(out_file, canvas)

        return

    def after_train_iter(self, runner: Runner, batch_idx: int, data_batch: dict,
                       outputs: Sequence[DataSample]) -> None:
        """Visualize every ``self.interval`` samples during validation.

        Args:
            runner (:obj:`Runner`): The runner of the validation process.
            batch_idx (int): The index of the current batch in the val loop.
            data_batch (dict): Data from dataloader.
            outputs (Sequence[:obj:`DataSample`]): Outputs from model.
        """
        if runner.iter % self.vis_every_iters != 0:
            return
        
        # ## check if the rank is 0
        if not runner.rank == 0:
            return

        runner.logger.info(f'\033[96mVisualizing for this iteration: {runner.iter}!\033[0m')
        self._draw_samples(runner, batch_idx, data_batch, outputs)
        runner.logger.info(f'\033[96mDone visualizing for this iteration!\033[0m')

        return

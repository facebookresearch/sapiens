# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, List, Optional, Union

import torch
from mmcv.image import imread
from mmengine.config import Config
from mmengine.dataset import Compose, default_collate

from mmpretrain.registry import TRANSFORMS
from .base import BaseInferencer, InputType
from .model import list_models
import cv2
import numpy as np

class MAEInferencer(BaseInferencer):
    def __call__(self,
                 inputs: InputType,
                 batch_size: int = 1,
                 **kwargs) -> dict:
        """Call the inferencer.

        Args:
            inputs (str | array | list): The image path or array, or a list of
                images.
            batch_size (int): Batch size. Defaults to 1.
            **kwargs: Other keyword arguments accepted by the `extract_feat`
                method of the model.

        Returns:
            tensor | Tuple[tensor]: The extracted features.
        """
        ori_inputs = self._inputs_to_list(inputs)
        inputs = self.preprocess(ori_inputs, batch_size=batch_size)
        preds = []
        for data in inputs:
            vis_image = self.forward(data, **kwargs)
            preds.append(vis_image)
        return preds

    ## single image inference only!
    @torch.no_grad()
    def forward(self, inputs: Union[dict, tuple], copy_inputs=False,  **kwargs):
        raw_inputs = inputs['inputs'].clone()
        inputs = self.model.data_preprocessor(inputs, False)['inputs']

        losses, pred, mask = self.model.inference(inputs, data_samples=None, **kwargs)

        ## ----visualize------
        images = raw_inputs  ## B x 3 x H x W, not normalized in BGR format
        patch_images = self.model.head.patchify(images.float()) ## B x (H*W) x (P*P*3)
        patch_preds = pred.detach().cpu() ## this is RGB format, B x (H*W) x (P*P*3)
        masks = mask.cpu()

        if self.model.head.norm_pix == True:
            mean, var = self.model.head.get_norm_pix_mean_var(inputs) ## B x 3 x H x W, normalized in RGB format.
            mean = mean.cpu(); var = var.cpu()
            patch_preds = (patch_preds * (var + 1.e-6)**.5) + mean
        
        preds_normalized = self.model.head.unpatchify(patch_preds) ## B x 3 x H x W
        preds = self.model.unnormalize_image(preds_normalized, use_cpu=True)
        preds = preds.clamp(0, 255) # B x 3 x H x W, rgb.
        preds = preds[:, [2, 1, 0], :, :] ## B x 3 x H x W. BGR

        masked_patch_images = patch_images*(1-masks).unsqueeze(2) + 100*masks.unsqueeze(2) ## the binary mask: 0 is keep, 1 is remove
        masked_images = self.model.head.unpatchify(masked_patch_images)

        ## copy paste the input pixels to output
        if copy_inputs == True:
            temp_masks = masks.unsqueeze(2).repeat(1, 1, patch_preds.shape[-1]) ## B x (H*W) x (P*P*3)
            unpatch_masks = self.model.head.unpatchify(temp_masks) ## B x 3 x H x W. boolean
            preds = preds * unpatch_masks + images * (1-unpatch_masks)

        image = images[0]
        image = image.permute(1, 2, 0).cpu().numpy().astype('uint8')

        masked_image = masked_images[0]
        masked_image = masked_image.permute(1, 2, 0).cpu().numpy().astype('uint8')

        pred = preds[0]
        pred = pred.permute(1, 2, 0).cpu().numpy().astype('uint8')

        save_image = np.concatenate((image, masked_image, pred), axis=1)

        return save_image

    def _init_pipeline(self, cfg: Config) -> Callable:
        test_pipeline_cfg = cfg.test_dataloader.dataset.pipeline
        from mmpretrain.datasets import remove_transform

        # Image loading is finished in `self.preprocess`.
        test_pipeline_cfg = remove_transform(test_pipeline_cfg,
                                             'LoadImageFromFile')
        test_pipeline = Compose(
            [TRANSFORMS.build(t) for t in test_pipeline_cfg])
        return test_pipeline

    def preprocess(self, inputs: List[InputType], batch_size: int = 1):

        def load_image(input_):
            img = imread(input_)
            if img is None:
                raise ValueError(f'Failed to read image {input_}.')
            return dict(
                img=img,
                img_shape=img.shape[:2],
                ori_shape=img.shape[:2],
            )

        pipeline = Compose([load_image, self.pipeline])

        chunked_data = self._get_chunk_data(map(pipeline, inputs), batch_size)
        yield from map(default_collate, chunked_data)

    def visualize(self):
        raise NotImplementedError(
            "The FeatureExtractor doesn't support visualization.")

    def postprocess(self):
        raise NotImplementedError(
            "The FeatureExtractor doesn't need postprocessing.")

    @staticmethod
    def list_models(pattern: Optional[str] = None):
        """List all available model names.

        Args:
            pattern (str | None): A wildcard pattern to match model names.

        Returns:
            List[str]: a list of model names.
        """
        return list_models(pattern=pattern)

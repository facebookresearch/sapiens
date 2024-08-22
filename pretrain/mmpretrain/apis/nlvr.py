# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from copy import deepcopy
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import torch
from mmcv.image import imread
from mmengine.config import Config
from mmengine.dataset import Compose, default_collate

from mmpretrain.registry import TRANSFORMS
from mmpretrain.structures import DataSample
from .base import BaseInferencer
from .model import list_models

InputType = Tuple[Union[str, np.ndarray], Union[str, np.ndarray], str]
InputsType = Union[List[InputType], InputType]


class NLVRInferencer(BaseInferencer):
    """The inferencer for Natural Language for Visual Reasoning.

    Args:
        model (BaseModel | str | Config): A model name or a path to the config
            file, or a :obj:`BaseModel` object. The model name can be found
            by ``NLVRInferencer.list_models()`` and you can also
            query it in :doc:`/modelzoo_statistics`.
        pretrained (str, optional): Path to the checkpoint. If None, it will
            try to find a pre-defined weight from the model you specified
            (only work if the ``model`` is a model name). Defaults to None.
        device (str, optional): Device to run inference. If None, the available
            device will be automatically used. Defaults to None.
        **kwargs: Other keyword arguments to initialize the model (only work if
            the ``model`` is a model name).
    """

    visualize_kwargs: set = {
        'resize', 'draw_score', 'show', 'show_dir', 'wait_time'
    }

    def __call__(self,
                 inputs: InputsType,
                 return_datasamples: bool = False,
                 batch_size: int = 1,
                 **kwargs) -> dict:
        """Call the inferencer.

        Args:
            inputs (tuple, List[tuple]): The input data tuples, every tuple
                should include three items (left image, right image, text).
                The image can be a path or numpy array.
            return_datasamples (bool): Whether to return results as
                :obj:`DataSample`. Defaults to False.
            batch_size (int): Batch size. Defaults to 1.
            resize (int, optional): Resize the short edge of the image to the
                specified length before visualization. Defaults to None.
            draw_score (bool): Whether to draw the prediction scores
                of prediction categories. Defaults to True.
            show (bool): Whether to display the visualization result in a
                window. Defaults to False.
            wait_time (float): The display time (s). Defaults to 0, which means
                "forever".
            show_dir (str, optional): If not None, save the visualization
                results in the specified directory. Defaults to None.

        Returns:
            list: The inference results.
        """
        assert isinstance(inputs, (tuple, list))
        if isinstance(inputs, tuple):
            inputs = [inputs]
        for input_ in inputs:
            assert isinstance(input_, tuple)
            assert len(input_) == 3

        return super().__call__(
            inputs,
            return_datasamples=return_datasamples,
            batch_size=batch_size,
            **kwargs)

    def _init_pipeline(self, cfg: Config) -> Callable:
        test_pipeline_cfg = cfg.test_dataloader.dataset.pipeline
        assert test_pipeline_cfg[0]['type'] == 'ApplyToList'

        list_pipeline = deepcopy(test_pipeline_cfg[0])
        if list_pipeline.scatter_key == 'img_path':
            # Remove `LoadImageFromFile`
            list_pipeline.transforms.pop(0)
            list_pipeline.scatter_key = 'img'

        test_pipeline = Compose(
            [TRANSFORMS.build(list_pipeline)] +
            [TRANSFORMS.build(t) for t in test_pipeline_cfg[1:]])
        return test_pipeline

    def preprocess(self, inputs: InputsType, batch_size: int = 1):

        def load_image(input_):
            img1 = imread(input_[0])
            img2 = imread(input_[1])
            text = input_[2]
            if img1 is None:
                raise ValueError(f'Failed to read image {input_[0]}.')
            if img2 is None:
                raise ValueError(f'Failed to read image {input_[1]}.')
            return dict(
                img=[img1, img2],
                img_shape=[img1.shape[:2], img2.shape[:2]],
                ori_shape=[img1.shape[:2], img2.shape[:2]],
                text=text,
            )

        pipeline = Compose([load_image, self.pipeline])

        chunked_data = self._get_chunk_data(map(pipeline, inputs), batch_size)
        yield from map(default_collate, chunked_data)

    def postprocess(self,
                    preds: List[DataSample],
                    visualization: List[np.ndarray],
                    return_datasamples=False) -> dict:
        if return_datasamples:
            return preds

        results = []
        for data_sample in preds:
            pred_scores = data_sample.pred_score
            pred_score = float(torch.max(pred_scores).item())
            pred_label = torch.argmax(pred_scores).item()
            result = {
                'pred_scores': pred_scores.detach().cpu().numpy(),
                'pred_label': pred_label,
                'pred_score': pred_score,
            }
            results.append(result)

        return results

    @staticmethod
    def list_models(pattern: Optional[str] = None):
        """List all available model names.

        Args:
            pattern (str | None): A wildcard pattern to match model names.

        Returns:
            List[str]: a list of model names.
        """
        return list_models(pattern=pattern, task='NLVR')

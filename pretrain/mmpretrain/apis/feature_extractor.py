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


class FeatureExtractor(BaseInferencer):
    """The inferencer for extract features.

    Args:
        model (BaseModel | str | Config): A model name or a path to the config
            file, or a :obj:`BaseModel` object. The model name can be found
            by ``FeatureExtractor.list_models()`` and you can also query it in
            :doc:`/modelzoo_statistics`.
        pretrained (str, optional): Path to the checkpoint. If None, it will
            try to find a pre-defined weight from the model you specified
            (only work if the ``model`` is a model name). Defaults to None.
        device (str, optional): Device to run inference. If None, the available
            device will be automatically used. Defaults to None.
        **kwargs: Other keyword arguments to initialize the model (only work if
            the ``model`` is a model name).

    Example:
        >>> from mmpretrain import FeatureExtractor
        >>> inferencer = FeatureExtractor('resnet50_8xb32_in1k', backbone=dict(out_indices=(0, 1, 2, 3)))
        >>> feats = inferencer('demo/demo.JPEG', stage='backbone')[0]
        >>> for feat in feats:
        >>>     print(feat.shape)
        torch.Size([256, 56, 56])
        torch.Size([512, 28, 28])
        torch.Size([1024, 14, 14])
        torch.Size([2048, 7, 7])
    """  # noqa: E501

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
            preds.extend(self.forward(data, **kwargs))

        return preds

    @torch.no_grad()
    def forward(self, inputs: Union[dict, tuple], **kwargs):
        inputs = self.model.data_preprocessor(inputs, False)['inputs']
        outputs = self.model.extract_feat(inputs, **kwargs)

        def scatter(feats, index):
            if isinstance(feats, torch.Tensor):
                return feats[index]
            else:
                # Sequence of tensor
                return type(feats)([scatter(item, index) for item in feats])

        results = []
        for i in range(inputs.shape[0]):
            results.append(scatter(outputs, i))

        return results

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

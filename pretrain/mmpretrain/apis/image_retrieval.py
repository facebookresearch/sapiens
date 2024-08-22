# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path
from typing import Callable, List, Optional, Union

import numpy as np
import torch
from mmcv.image import imread
from mmengine.config import Config
from mmengine.dataset import BaseDataset, Compose, default_collate

from mmpretrain.registry import TRANSFORMS
from mmpretrain.structures import DataSample
from .base import BaseInferencer, InputType, ModelType
from .model import list_models


class ImageRetrievalInferencer(BaseInferencer):
    """The inferencer for image to image retrieval.

    Args:
        model (BaseModel | str | Config): A model name or a path to the config
            file, or a :obj:`BaseModel` object. The model name can be found
            by ``ImageRetrievalInferencer.list_models()`` and you can also
            query it in :doc:`/modelzoo_statistics`.
        prototype (str | list | dict | DataLoader, BaseDataset): The images to
            be retrieved. It can be the following types:

            - str: The directory of the the images.
            - list: A list of path of the images.
            - dict: A config dict of the a prototype dataset.
            - BaseDataset: A prototype dataset.
            - DataLoader: A data loader to load the prototype data.

        prototype_cache (str, optional): The path of the generated prototype
            features. If exists, directly load the cache instead of re-generate
            the prototype features. If not exists, save the generated features
            to the path. Defaults to None.
        pretrained (str, optional): Path to the checkpoint. If None, it will
            try to find a pre-defined weight from the model you specified
            (only work if the ``model`` is a model name). Defaults to None.
        device (str, optional): Device to run inference. If None, the available
            device will be automatically used. Defaults to None.
        **kwargs: Other keyword arguments to initialize the model (only work if
            the ``model`` is a model name).

    Example:
        >>> from mmpretrain import ImageRetrievalInferencer
        >>> inferencer = ImageRetrievalInferencer(
        ...     'resnet50-arcface_inshop',
        ...     prototype='./demo/',
        ...     prototype_cache='img_retri.pth')
        >>> inferencer('demo/cat-dog.png', topk=2)[0][1]
        {'match_score': tensor(0.4088, device='cuda:0'),
         'sample_idx': 3,
         'sample': {'img_path': './demo/dog.jpg'}}
    """  # noqa: E501

    visualize_kwargs: set = {
        'draw_score', 'resize', 'show_dir', 'show', 'wait_time', 'topk'
    }
    postprocess_kwargs: set = {'topk'}

    def __init__(
        self,
        model: ModelType,
        prototype,
        prototype_cache=None,
        prepare_batch_size=8,
        pretrained: Union[bool, str] = True,
        device: Union[str, torch.device, None] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            model=model, pretrained=pretrained, device=device, **kwargs)

        self.prototype_dataset = self._prepare_prototype(
            prototype, prototype_cache, prepare_batch_size)

    def _prepare_prototype(self, prototype, cache=None, batch_size=8):
        from mmengine.dataset import DefaultSampler
        from torch.utils.data import DataLoader

        def build_dataloader(dataset):
            return DataLoader(
                dataset,
                batch_size=batch_size,
                collate_fn=default_collate,
                sampler=DefaultSampler(dataset, shuffle=False),
                persistent_workers=False,
            )

        if isinstance(prototype, str):
            # A directory path of images
            prototype = dict(
                type='CustomDataset', with_label=False, data_root=prototype)

        if isinstance(prototype, list):
            test_pipeline = [dict(type='LoadImageFromFile'), self.pipeline]
            dataset = BaseDataset(
                lazy_init=True, serialize_data=False, pipeline=test_pipeline)
            dataset.data_list = [{
                'sample_idx': i,
                'img_path': file
            } for i, file in enumerate(prototype)]
            dataset._fully_initialized = True
            dataloader = build_dataloader(dataset)
        elif isinstance(prototype, dict):
            # A config of dataset
            from mmpretrain.registry import DATASETS
            test_pipeline = [dict(type='LoadImageFromFile'), self.pipeline]
            dataset = DATASETS.build(prototype)
            dataloader = build_dataloader(dataset)
        elif isinstance(prototype, DataLoader):
            dataset = prototype.dataset
            dataloader = prototype
        elif isinstance(prototype, BaseDataset):
            dataset = prototype
            dataloader = build_dataloader(dataset)
        else:
            raise TypeError(f'Unsupported prototype type {type(prototype)}.')

        if cache is not None and Path(cache).exists():
            self.model.prototype = cache
        else:
            self.model.prototype = dataloader
        self.model.prepare_prototype()

        from mmengine.logging import MMLogger
        logger = MMLogger.get_current_instance()
        if cache is None:
            logger.info('The prototype has been prepared, you can use '
                        '`save_prototype` to dump it into a pickle '
                        'file for the future usage.')
        elif not Path(cache).exists():
            self.save_prototype(cache)
            logger.info(f'The prototype has been saved at {cache}.')

        return dataset

    def save_prototype(self, path):
        self.model.dump_prototype(path)

    def __call__(self,
                 inputs: InputType,
                 return_datasamples: bool = False,
                 batch_size: int = 1,
                 **kwargs) -> dict:
        """Call the inferencer.

        Args:
            inputs (str | array | list): The image path or array, or a list of
                images.
            return_datasamples (bool): Whether to return results as
                :obj:`DataSample`. Defaults to False.
            batch_size (int): Batch size. Defaults to 1.
            resize (int, optional): Resize the long edge of the image to the
                specified length before visualization. Defaults to None.
            draw_score (bool): Whether to draw the match scores.
                Defaults to True.
            show (bool): Whether to display the visualization result in a
                window. Defaults to False.
            wait_time (float): The display time (s). Defaults to 0, which means
                "forever".
            show_dir (str, optional): If not None, save the visualization
                results in the specified directory. Defaults to None.

        Returns:
            list: The inference results.
        """
        return super().__call__(inputs, return_datasamples, batch_size,
                                **kwargs)

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

    def visualize(self,
                  ori_inputs: List[InputType],
                  preds: List[DataSample],
                  topk: int = 3,
                  resize: Optional[int] = 224,
                  show: bool = False,
                  wait_time: int = 0,
                  draw_score=True,
                  show_dir=None):
        if not show and show_dir is None:
            return None

        if self.visualizer is None:
            from mmpretrain.visualization import UniversalVisualizer
            self.visualizer = UniversalVisualizer()

        visualization = []
        for i, (input_, data_sample) in enumerate(zip(ori_inputs, preds)):
            image = imread(input_)
            if isinstance(input_, str):
                # The image loaded from path is BGR format.
                image = image[..., ::-1]
                name = Path(input_).stem
            else:
                name = str(i)

            if show_dir is not None:
                show_dir = Path(show_dir)
                show_dir.mkdir(exist_ok=True)
                out_file = str((show_dir / name).with_suffix('.png'))
            else:
                out_file = None

            self.visualizer.visualize_image_retrieval(
                image,
                data_sample,
                self.prototype_dataset,
                topk=topk,
                resize=resize,
                draw_score=draw_score,
                show=show,
                wait_time=wait_time,
                name=name,
                out_file=out_file)
            visualization.append(self.visualizer.get_image())
        if show:
            self.visualizer.close()
        return visualization

    def postprocess(
        self,
        preds: List[DataSample],
        visualization: List[np.ndarray],
        return_datasamples=False,
        topk=1,
    ) -> dict:
        if return_datasamples:
            return preds

        results = []
        for data_sample in preds:
            match_scores, indices = torch.topk(data_sample.pred_score, k=topk)
            matches = []
            for match_score, sample_idx in zip(match_scores, indices):
                sample = self.prototype_dataset.get_data_info(
                    sample_idx.item())
                sample_idx = sample.pop('sample_idx')
                matches.append({
                    'match_score': match_score,
                    'sample_idx': sample_idx,
                    'sample': sample
                })
            results.append(matches)

        return results

    @staticmethod
    def list_models(pattern: Optional[str] = None):
        """List all available model names.

        Args:
            pattern (str | None): A wildcard pattern to match model names.

        Returns:
            List[str]: a list of model names.
        """
        return list_models(pattern=pattern, task='Image Retrieval')

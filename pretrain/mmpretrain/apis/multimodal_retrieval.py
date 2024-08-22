# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from copy import deepcopy
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

import mmengine
import numpy as np
import torch
from mmcv.image import imread
from mmengine.config import Config
from mmengine.dataset import BaseDataset, Compose, default_collate

from mmpretrain.registry import TRANSFORMS
from mmpretrain.structures import DataSample
from mmpretrain.utils import track
from .base import BaseInferencer
from .base import InputType as ImageType
from .base import ModelType
from .model import list_models


def filter_transforms(transforms: list, data_info: dict):
    """Filter pipeline to avoid KeyError with partial data info."""
    data_info = deepcopy(data_info)
    filtered_transforms = []
    for t in transforms:
        try:
            data_info = t(data_info)
            filtered_transforms.append(t)
        except KeyError:
            pass
    return filtered_transforms


class TextToImageRetrievalInferencer(BaseInferencer):
    """The inferencer for text to image retrieval.

    Args:
        model (BaseModel | str | Config): A model name or a path to the config
            file, or a :obj:`BaseModel` object. The model name can be found
            by ``TextToImageRetrievalInferencer.list_models()`` and you can also
            query it in :doc:`/modelzoo_statistics`.
        prototype (str | list | dict | DataLoader | BaseDataset): The images to
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
        fast_match (bool): Some algorithms will record extra image features for
            further matching, which may consume large memory, set True to avoid
            this behavior. Defaults to True.
        pretrained (str, optional): Path to the checkpoint. If None, it will
            try to find a pre-defined weight from the model you specified
            (only work if the ``model`` is a model name). Defaults to None.
        device (str, optional): Device to run inference. If None, the available
            device will be automatically used. Defaults to None.
        **kwargs: Other keyword arguments to initialize the model (only work if
            the ``model`` is a model name).

    Example:
        >>> from mmpretrain import TextToImageRetrievalInferencer
        >>> inferencer = TextToImageRetrievalInferencer(
        ...     'blip-base_3rdparty_retrieval',
        ...     prototype='./demo/',
        ...     prototype_cache='t2i_retri.pth')
        >>> inferencer('A cat and a dog.')[0]
        {'match_score': tensor(0.3855, device='cuda:0'),
         'sample_idx': 1,
         'sample': {'img_path': './demo/cat-dog.png'}}
    """  # noqa: E501

    visualize_kwargs: set = {
        'draw_score', 'show_dir', 'show', 'wait_time', 'figsize', 'topk'
    }
    postprocess_kwargs: set = {'topk'}

    def __init__(self,
                 model: ModelType,
                 prototype,
                 prototype_cache=None,
                 fast_match=True,
                 prepare_batch_size=8,
                 pretrained: Union[bool, str] = True,
                 device: Union[str, torch.device, None] = None,
                 **kwargs) -> None:
        super().__init__(
            model=model, pretrained=pretrained, device=device, **kwargs)

        self.img_pipeline, self.text_pipeline = self.pipeline

        if hasattr(self.model, 'fast_match'):
            self.model.fast_match = fast_match

        self.prototype_dataset = self._prepare_prototype(
            prototype, prototype_cache, batch_size=prepare_batch_size)

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
            test_pipeline = [dict(type='LoadImageFromFile'), self.img_pipeline]
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
            test_pipeline = [dict(type='LoadImageFromFile'), self.img_pipeline]
            prototype.setdefault('pipeline', test_pipeline)
            dataset = DATASETS.build(prototype)
            dataloader = build_dataloader(dataset)
        elif isinstance(prototype, list):
            test_pipeline = [dict(type='LoadImageFromFile'), self.img_pipeline]
            dataset = BaseDataset(
                lazy_init=True, serialize_data=False, pipeline=test_pipeline)
            dataset.data_list = [{
                'sample_idx': i,
                'img_path': file
            } for i, file in enumerate(prototype)]
            dataset._fully_initialized = True
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
            self.prototype = torch.load(cache)
        else:
            prototype = []
            for data_batch in track(dataloader, 'Prepare prototype...'):
                with torch.no_grad():
                    data_batch = self.model.data_preprocessor(
                        data_batch, False)
                    feats = self.model._run_forward(data_batch, mode='tensor')
                    prototype.append(feats)
            prototype = {
                k: torch.cat([d[k] for d in prototype])
                for k in prototype[0]
            }
            self.prototype = prototype

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
        torch.save(self.prototype, path)

    def __call__(self,
                 inputs: ImageType,
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

    @torch.no_grad()
    def forward(self, data: dict, **kwargs):
        """Feed the inputs to the model."""
        data = self.model.data_preprocessor(data, False)
        data_samples = data['data_samples']
        feats = self.prototype.copy()
        feats.update(self.model.extract_feat(data_samples=data_samples))
        return self.model.predict_all(feats, data_samples, cal_i2t=False)[0]

    def _init_pipeline(self, cfg: Config) -> Callable:
        test_pipeline_cfg = cfg.test_dataloader.dataset.pipeline
        test_transfroms = [TRANSFORMS.build(t) for t in test_pipeline_cfg]
        img_info = {'img': np.zeros((224, 224, 3), dtype=np.uint8)}
        text_info = {'text': 'example'}
        img_pipeline = Compose(filter_transforms(test_transfroms, img_info))
        text_pipeline = Compose(filter_transforms(test_transfroms, text_info))
        return img_pipeline, text_pipeline

    def preprocess(self, inputs: List[str], batch_size: int = 1):

        def process_text(input_: str):
            return self.text_pipeline({'text': input_})

        chunked_data = self._get_chunk_data(
            map(process_text, inputs), batch_size)
        yield from map(default_collate, chunked_data)

    def visualize(self,
                  ori_inputs: List[str],
                  preds: List[DataSample],
                  topk: int = 3,
                  figsize: Tuple[int, int] = (16, 9),
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
        for i, (text, data_sample) in enumerate(zip(ori_inputs, preds)):
            name = str(i)

            if show_dir is not None:
                show_dir = Path(show_dir)
                show_dir.mkdir(exist_ok=True)
                out_file = str((show_dir / name).with_suffix('.png'))
            else:
                out_file = None

            self.visualizer.visualize_t2i_retrieval(
                text,
                data_sample,
                self.prototype_dataset,
                topk=topk,
                fig_cfg=dict(figsize=figsize),
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
        return list_models(pattern=pattern, task='Text-To-Image Retrieval')


class ImageToTextRetrievalInferencer(BaseInferencer):
    """The inferencer for image to text retrieval.

    Args:
        model (BaseModel | str | Config): A model name or a path to the config
            file, or a :obj:`BaseModel` object. The model name can be found
            by ``ImageToTextRetrievalInferencer.list_models()`` and you can
            also query it in :doc:`/modelzoo_statistics`.
        prototype (str | list | dict | DataLoader, BaseDataset): The images to
            be retrieved. It can be the following types:

            - str: The file path to load the string list.
            - list: A list of string.

        prototype_cache (str, optional): The path of the generated prototype
            features. If exists, directly load the cache instead of re-generate
            the prototype features. If not exists, save the generated features
            to the path. Defaults to None.
        fast_match (bool): Some algorithms will record extra image features for
            further matching, which may consume large memory, set True to avoid
            this behavior. Defaults to True.
        pretrained (str, optional): Path to the checkpoint. If None, it will
            try to find a pre-defined weight from the model you specified
            (only work if the ``model`` is a model name). Defaults to None.
        device (str, optional): Device to run inference. If None, the available
            device will be automatically used. Defaults to None.
        **kwargs: Other keyword arguments to initialize the model (only work if
            the ``model`` is a model name).

    Example:
        >>> from mmpretrain import ImageToTextRetrievalInferencer
        >>> inferencer = ImageToTextRetrievalInferencer(
        ...     'blip-base_3rdparty_retrieval',
        ...     prototype=['cat', 'dog', 'snake', 'bird'],
        ...     prototype_cache='i2t_retri.pth')
        >>> inferencer('demo/bird.JPEG')[0]
        {'match_score': tensor(0.3855, device='cuda:0'),
         'sample_idx': 1,
         'sample': {'img_path': './demo/cat-dog.png'}}
    """  # noqa: E501

    visualize_kwargs: set = {
        'draw_score', 'resize', 'show_dir', 'show', 'wait_time', 'topk'
    }
    postprocess_kwargs: set = {'topk'}

    def __init__(self,
                 model: ModelType,
                 prototype,
                 prototype_cache=None,
                 fast_match=True,
                 prepare_batch_size=8,
                 pretrained: Union[bool, str] = True,
                 device: Union[str, torch.device, None] = None,
                 **kwargs) -> None:
        super().__init__(
            model=model, pretrained=pretrained, device=device, **kwargs)

        self.img_pipeline, self.text_pipeline = self.pipeline

        if hasattr(self.model, 'fast_match'):
            self.model.fast_match = fast_match

        self.prototype_dataset = self._prepare_prototype(
            prototype, cache=prototype_cache, batch_size=prepare_batch_size)

    def _prepare_prototype(self, prototype, cache=None, batch_size=8):
        from mmengine.dataset import DefaultSampler
        from torch.utils.data import DataLoader

        def build_dataloader(dataset):
            return DataLoader(
                [
                    self.text_pipeline({
                        'sample_idx': i,
                        'text': text
                    }) for i, text in enumerate(dataset)
                ],
                batch_size=batch_size,
                collate_fn=default_collate,
                sampler=DefaultSampler(dataset, shuffle=False),
                persistent_workers=False,
            )

        if isinstance(prototype, str):
            # A file path of a list of string
            dataset = mmengine.list_from_file(prototype)
        elif mmengine.utils.is_seq_of(prototype, str):
            dataset = prototype
        else:
            raise TypeError(f'Unsupported prototype type {type(prototype)}.')

        dataloader = build_dataloader(dataset)

        if cache is not None and Path(cache).exists():
            self.prototype = torch.load(cache)
        else:
            prototype = []
            for data_batch in track(dataloader, 'Prepare prototype...'):
                with torch.no_grad():
                    data_batch = self.model.data_preprocessor(
                        data_batch, False)
                    feats = self.model._run_forward(data_batch, mode='tensor')
                    prototype.append(feats)
            prototype = {
                k: torch.cat([d[k] for d in prototype])
                for k in prototype[0]
            }
            self.prototype = prototype

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
        torch.save(self.prototype, path)

    def __call__(self,
                 inputs: ImageType,
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

    @torch.no_grad()
    def forward(self, data: dict, **kwargs):
        """Feed the inputs to the model."""
        data = self.model.data_preprocessor(data, False)
        feats = self.prototype.copy()
        feats.update(self.model.extract_feat(images=data['images']))
        return self.model.predict_all(
            feats, data['data_samples'], cal_t2i=False)[0]

    def _init_pipeline(self, cfg: Config) -> Callable:
        test_pipeline_cfg = cfg.test_dataloader.dataset.pipeline
        test_transfroms = [TRANSFORMS.build(t) for t in test_pipeline_cfg]
        img_info = {'img': np.zeros((224, 224, 3), dtype=np.uint8)}
        text_info = {'text': 'example'}
        img_pipeline = Compose(filter_transforms(test_transfroms, img_info))
        text_pipeline = Compose(filter_transforms(test_transfroms, text_info))
        return img_pipeline, text_pipeline

    def preprocess(self, inputs: List[ImageType], batch_size: int = 1):

        def load_image(input_):
            img = imread(input_)
            if img is None:
                raise ValueError(f'Failed to read image {input_}.')
            return dict(
                img=img,
                img_shape=img.shape[:2],
                ori_shape=img.shape[:2],
            )

        pipeline = Compose([load_image, self.img_pipeline])

        chunked_data = self._get_chunk_data(map(pipeline, inputs), batch_size)
        yield from map(default_collate, chunked_data)

    def visualize(self,
                  ori_inputs: List[ImageType],
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

            self.visualizer.visualize_i2t_retrieval(
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
                text = self.prototype_dataset[sample_idx.item()]
                matches.append({
                    'match_score': match_score,
                    'sample_idx': sample_idx,
                    'text': text
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
        return list_models(pattern=pattern, task='Image-To-Text Retrieval')

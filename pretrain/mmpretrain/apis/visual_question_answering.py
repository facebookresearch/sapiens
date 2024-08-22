# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path
from typing import Callable, List, Optional, Union

import numpy as np
from mmcv.image import imread
from mmengine.config import Config
from mmengine.dataset import Compose, default_collate

from mmpretrain.registry import TRANSFORMS
from mmpretrain.structures import DataSample
from .base import BaseInferencer
from .model import list_models


class VisualQuestionAnsweringInferencer(BaseInferencer):
    """The inferencer for visual question answering.

    Args:
        model (BaseModel | str | Config): A model name or a path to the config
            file, or a :obj:`BaseModel` object. The model name can be found
            by ``VisualQuestionAnsweringInferencer.list_models()`` and you can
            also query it in :doc:`/modelzoo_statistics`.
        pretrained (str, optional): Path to the checkpoint. If None, it will
            try to find a pre-defined weight from the model you specified
            (only work if the ``model`` is a model name). Defaults to None.
        device (str, optional): Device to run inference. If None, the available
            device will be automatically used. Defaults to None.
        **kwargs: Other keyword arguments to initialize the model (only work if
            the ``model`` is a model name).

    Example:
        >>> from mmpretrain import VisualQuestionAnsweringInferencer
        >>> inferencer = VisualQuestionAnsweringInferencer('ofa-base_3rdparty-zeroshot_vqa')
        >>> inferencer('demo/cat-dog.png', "What's the animal next to the dog?")[0]
        {'question': "What's the animal next to the dog?", 'pred_answer': 'cat'}
    """  # noqa: E501

    visualize_kwargs: set = {'resize', 'show', 'show_dir', 'wait_time'}

    def __call__(self,
                 images: Union[str, np.ndarray, list],
                 questions: Union[str, list],
                 return_datasamples: bool = False,
                 batch_size: int = 1,
                 objects: Optional[List[str]] = None,
                 **kwargs) -> dict:
        """Call the inferencer.

        Args:
            images (str | array | list): The image path or array, or a list of
                images.
            questions (str | list): The question to the correspondding image.
            return_datasamples (bool): Whether to return results as
                :obj:`DataSample`. Defaults to False.
            batch_size (int): Batch size. Defaults to 1.
            objects (List[List[str]], optional): Some algorithms like OFA
                fine-tuned VQA models requires extra object description list
                for every image. Defaults to None.
            resize (int, optional): Resize the short edge of the image to the
                specified length before visualization. Defaults to None.
            show (bool): Whether to display the visualization result in a
                window. Defaults to False.
            wait_time (float): The display time (s). Defaults to 0, which means
                "forever".
            show_dir (str, optional): If not None, save the visualization
                results in the specified directory. Defaults to None.

        Returns:
            list: The inference results.
        """
        if not isinstance(images, (list, tuple)):
            assert isinstance(questions, str)
            inputs = [{'img': images, 'question': questions}]
            if objects is not None:
                assert isinstance(objects[0], str)
                inputs[0]['objects'] = objects
        else:
            inputs = []
            for i in range(len(images)):
                input_ = {'img': images[i], 'question': questions[i]}
                if objects is not None:
                    input_['objects'] = objects[i]
                inputs.append(input_)

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

    def preprocess(self, inputs: List[dict], batch_size: int = 1):

        def load_image(input_: dict):
            img = imread(input_['img'])
            if img is None:
                raise ValueError(f'Failed to read image {input_}.')
            return {**input_, 'img': img}

        pipeline = Compose([load_image, self.pipeline])

        chunked_data = self._get_chunk_data(map(pipeline, inputs), batch_size)
        yield from map(default_collate, chunked_data)

    def visualize(self,
                  ori_inputs: List[dict],
                  preds: List[DataSample],
                  show: bool = False,
                  wait_time: int = 0,
                  resize: Optional[int] = None,
                  show_dir=None):
        if not show and show_dir is None:
            return None

        if self.visualizer is None:
            from mmpretrain.visualization import UniversalVisualizer
            self.visualizer = UniversalVisualizer()

        visualization = []
        for i, (input_, data_sample) in enumerate(zip(ori_inputs, preds)):
            image = imread(input_['img'])
            if isinstance(input_['img'], str):
                # The image loaded from path is BGR format.
                image = image[..., ::-1]
                name = Path(input_['img']).stem
            else:
                name = str(i)

            if show_dir is not None:
                show_dir = Path(show_dir)
                show_dir.mkdir(exist_ok=True)
                out_file = str((show_dir / name).with_suffix('.png'))
            else:
                out_file = None

            self.visualizer.visualize_vqa(
                image,
                data_sample,
                resize=resize,
                show=show,
                wait_time=wait_time,
                name=name,
                out_file=out_file)
            visualization.append(self.visualizer.get_image())
        if show:
            self.visualizer.close()
        return visualization

    def postprocess(self,
                    preds: List[DataSample],
                    visualization: List[np.ndarray],
                    return_datasamples=False) -> dict:
        if return_datasamples:
            return preds

        results = []
        for data_sample in preds:
            results.append({
                'question': data_sample.get('question'),
                'pred_answer': data_sample.get('pred_answer'),
            })

        return results

    @staticmethod
    def list_models(pattern: Optional[str] = None):
        """List all available model names.

        Args:
            pattern (str | None): A wildcard pattern to match model names.

        Returns:
            List[str]: a list of model names.
        """
        return list_models(pattern=pattern, task='Visual Question Answering')

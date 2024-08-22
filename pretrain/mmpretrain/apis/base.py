# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from abc import abstractmethod
from math import ceil
from typing import Callable, Iterable, List, Optional, Tuple, Union

import numpy as np
import torch
from mmengine.config import Config
from mmengine.dataset import default_collate
from mmengine.fileio import get_file_backend
from mmengine.model import BaseModel
from mmengine.runner import load_checkpoint

from mmpretrain.structures import DataSample
from mmpretrain.utils import track
from .model import get_model, list_models

ModelType = Union[BaseModel, str, Config]
InputType = Union[str, np.ndarray, list]


class BaseInferencer:
    """Base inferencer for various tasks.

    The BaseInferencer provides the standard workflow for inference as follows:

    1. Preprocess the input data by :meth:`preprocess`.
    2. Forward the data to the model by :meth:`forward`. ``BaseInferencer``
       assumes the model inherits from :class:`mmengine.models.BaseModel` and
       will call `model.test_step` in :meth:`forward` by default.
    3. Visualize the results by :meth:`visualize`.
    4. Postprocess and return the results by :meth:`postprocess`.

    When we call the subclasses inherited from BaseInferencer (not overriding
    ``__call__``), the workflow will be executed in order.

    All subclasses of BaseInferencer could define the following class
    attributes for customization:

    - ``preprocess_kwargs``: The keys of the kwargs that will be passed to
      :meth:`preprocess`.
    - ``forward_kwargs``: The keys of the kwargs that will be passed to
      :meth:`forward`
    - ``visualize_kwargs``: The keys of the kwargs that will be passed to
      :meth:`visualize`
    - ``postprocess_kwargs``: The keys of the kwargs that will be passed to
      :meth:`postprocess`

    All attributes mentioned above should be a ``set`` of keys (strings),
    and each key should not be duplicated. Actually, :meth:`__call__` will
    dispatch all the arguments to the corresponding methods according to the
    ``xxx_kwargs`` mentioned above.

    Subclasses inherited from ``BaseInferencer`` should implement
    :meth:`_init_pipeline`, :meth:`visualize` and :meth:`postprocess`:

    - _init_pipeline: Return a callable object to preprocess the input data.
    - visualize: Visualize the results returned by :meth:`forward`.
    - postprocess: Postprocess the results returned by :meth:`forward` and
      :meth:`visualize`.

    Args:
        model (BaseModel | str | Config): A model name or a path to the config
            file, or a :obj:`BaseModel` object. The model name can be found
            by ``cls.list_models()`` and you can also query it in
            :doc:`/modelzoo_statistics`.
        pretrained (str, optional): Path to the checkpoint. If None, it will
            try to find a pre-defined weight from the model you specified
            (only work if the ``model`` is a model name). Defaults to None.
        device (str | torch.device | None): Transfer the model to the target
            device. Defaults to None.
        device_map (str | dict | None): A map that specifies where each
            submodule should go. It doesn't need to be refined to each
            parameter/buffer name, once a given module name is inside, every
            submodule of it will be sent to the same device. You can use
            `device_map="auto"` to automatically generate the device map.
            Defaults to None.
        offload_folder (str | None): If the `device_map` contains any value
            `"disk"`, the folder where we will offload weights.
        **kwargs: Other keyword arguments to initialize the model (only work if
            the ``model`` is a model name).
    """

    preprocess_kwargs: set = set()
    forward_kwargs: set = set()
    visualize_kwargs: set = set()
    postprocess_kwargs: set = set()

    def __init__(self,
                 model: ModelType,
                 pretrained: Union[bool, str] = True,
                 device: Union[str, torch.device, None] = None,
                 device_map=None,
                 offload_folder=None,
                 **kwargs) -> None:

        if isinstance(model, BaseModel):
            if isinstance(pretrained, str):
                load_checkpoint(model, pretrained, map_location='cpu')
            if device_map is not None:
                from .utils import dispatch_model
                model = dispatch_model(
                    model,
                    device_map=device_map,
                    offload_folder=offload_folder)
            elif device is not None:
                model.to(device)
        else:
            model = get_model(
                model,
                pretrained,
                device=device,
                device_map=device_map,
                offload_folder=offload_folder,
                **kwargs)

        model.eval()

        self.config = model._config
        self.model = model
        self.pipeline = self._init_pipeline(self.config)
        self.visualizer = None

    def __call__(
        self,
        inputs,
        return_datasamples: bool = False,
        batch_size: int = 1,
        **kwargs,
    ) -> dict:
        """Call the inferencer.

        Args:
            inputs (InputsType): Inputs for the inferencer.
            return_datasamples (bool): Whether to return results as
                :obj:`BaseDataElement`. Defaults to False.
            batch_size (int): Batch size. Defaults to 1.
            **kwargs: Key words arguments passed to :meth:`preprocess`,
                :meth:`forward`, :meth:`visualize` and :meth:`postprocess`.
                Each key in kwargs should be in the corresponding set of
                ``preprocess_kwargs``, ``forward_kwargs``, ``visualize_kwargs``
                and ``postprocess_kwargs``.

        Returns:
            dict: Inference and visualization results.
        """
        (
            preprocess_kwargs,
            forward_kwargs,
            visualize_kwargs,
            postprocess_kwargs,
        ) = self._dispatch_kwargs(**kwargs)

        ori_inputs = self._inputs_to_list(inputs)
        inputs = self.preprocess(
            ori_inputs, batch_size=batch_size, **preprocess_kwargs)
        preds = []
        for data in track(
                inputs, 'Inference', total=ceil(len(ori_inputs) / batch_size)):
            preds.extend(self.forward(data, **forward_kwargs))
        visualization = self.visualize(ori_inputs, preds, **visualize_kwargs)
        results = self.postprocess(preds, visualization, return_datasamples,
                                   **postprocess_kwargs)
        return results

    def _inputs_to_list(self, inputs: InputType) -> list:
        """Preprocess the inputs to a list.

        Cast the input data to a list of data.

        - list or tuple: return inputs
        - str:
            - Directory path: return all files in the directory
            - other cases: return a list containing the string. The string
              could be a path to file, a url or other types of string according
              to the task.
        - other: return a list with one item.

        Args:
            inputs (str | array | list): Inputs for the inferencer.

        Returns:
            list: List of input for the :meth:`preprocess`.
        """
        if isinstance(inputs, str):
            backend = get_file_backend(inputs)
            if hasattr(backend, 'isdir') and backend.isdir(inputs):
                # Backends like HttpsBackend do not implement `isdir`, so only
                # those backends that implement `isdir` could accept the inputs
                # as a directory
                file_list = backend.list_dir_or_file(inputs, list_dir=False)
                inputs = [
                    backend.join_path(inputs, file) for file in file_list
                ]

        if not isinstance(inputs, (list, tuple)):
            inputs = [inputs]

        return list(inputs)

    def preprocess(self, inputs: InputType, batch_size: int = 1, **kwargs):
        """Process the inputs into a model-feedable format.

        Customize your preprocess by overriding this method. Preprocess should
        return an iterable object, of which each item will be used as the
        input of ``model.test_step``.

        ``BaseInferencer.preprocess`` will return an iterable chunked data,
        which will be used in __call__ like this:

        .. code-block:: python

            def __call__(self, inputs, batch_size=1, **kwargs):
                chunked_data = self.preprocess(inputs, batch_size, **kwargs)
                for batch in chunked_data:
                    preds = self.forward(batch, **kwargs)

        Args:
            inputs (InputsType): Inputs given by user.
            batch_size (int): batch size. Defaults to 1.

        Yields:
            Any: Data processed by the ``pipeline`` and ``default_collate``.
        """
        chunked_data = self._get_chunk_data(
            map(self.pipeline, inputs), batch_size)
        yield from map(default_collate, chunked_data)

    @torch.no_grad()
    def forward(self, inputs: Union[dict, tuple], **kwargs):
        """Feed the inputs to the model."""
        import ipdb; ipdb.set_trace()
        return self.model.test_step(inputs)

    def visualize(self,
                  inputs: list,
                  preds: List[DataSample],
                  show: bool = False,
                  **kwargs) -> List[np.ndarray]:
        """Visualize predictions.

        Customize your visualization by overriding this method. visualize
        should return visualization results, which could be np.ndarray or any
        other objects.

        Args:
            inputs (list): Inputs preprocessed by :meth:`_inputs_to_list`.
            preds (Any): Predictions of the model.
            show (bool): Whether to display the image in a popup window.
                Defaults to False.

        Returns:
            List[np.ndarray]: Visualization results.
        """
        if show:
            raise NotImplementedError(
                f'The `visualize` method of {self.__class__.__name__} '
                'is not implemented.')

    @abstractmethod
    def postprocess(
        self,
        preds: List[DataSample],
        visualization: List[np.ndarray],
        return_datasample=False,
        **kwargs,
    ) -> dict:
        """Process the predictions and visualization results from ``forward``
        and ``visualize``.

        This method should be responsible for the following tasks:

        1. Convert datasamples into a json-serializable dict if needed.
        2. Pack the predictions and visualization results and return them.
        3. Dump or log the predictions.

        Customize your postprocess by overriding this method. Make sure
        ``postprocess`` will return a dict with visualization results and
        inference results.

        Args:
            preds (List[Dict]): Predictions of the model.
            visualization (np.ndarray): Visualized predictions.
            return_datasample (bool): Whether to return results as datasamples.
                Defaults to False.

        Returns:
            dict: Inference and visualization results with key ``predictions``
            and ``visualization``

            - ``visualization (Any)``: Returned by :meth:`visualize`
            - ``predictions`` (dict or DataSample): Returned by
              :meth:`forward` and processed in :meth:`postprocess`.
              If ``return_datasample=False``, it usually should be a
              json-serializable dict containing only basic data elements such
              as strings and numbers.
        """

    @abstractmethod
    def _init_pipeline(self, cfg: Config) -> Callable:
        """Initialize the test pipeline.

        Return a pipeline to handle various input data, such as ``str``,
        ``np.ndarray``. It is an abstract method in BaseInferencer, and should
        be implemented in subclasses.

        The returned pipeline will be used to process a single data.
        It will be used in :meth:`preprocess` like this:

        .. code-block:: python
            def preprocess(self, inputs, batch_size, **kwargs):
                ...
                dataset = map(self.pipeline, dataset)
                ...
        """

    def _get_chunk_data(self, inputs: Iterable, chunk_size: int):
        """Get batch data from dataset.

        Args:
            inputs (Iterable): An iterable dataset.
            chunk_size (int): Equivalent to batch size.

        Yields:
            list: batch data.
        """
        inputs_iter = iter(inputs)
        while True:
            try:
                chunk_data = []
                for _ in range(chunk_size):
                    processed_data = next(inputs_iter)
                    chunk_data.append(processed_data)
                yield chunk_data
            except StopIteration:
                if chunk_data:
                    yield chunk_data
                break

    def _dispatch_kwargs(self, **kwargs) -> Tuple[dict, dict, dict, dict]:
        """Dispatch kwargs to preprocess(), forward(), visualize() and
        postprocess() according to the actual demands.

        Returns:
            Tuple[Dict, Dict, Dict, Dict]: kwargs passed to preprocess,
            forward, visualize and postprocess respectively.
        """
        # Ensure each argument only matches one function
        method_kwargs = self.preprocess_kwargs | self.forward_kwargs | \
            self.visualize_kwargs | self.postprocess_kwargs

        union_kwargs = method_kwargs | set(kwargs.keys())
        if union_kwargs != method_kwargs:
            unknown_kwargs = union_kwargs - method_kwargs
            raise ValueError(
                f'unknown argument {unknown_kwargs} for `preprocess`, '
                '`forward`, `visualize` and `postprocess`')

        preprocess_kwargs = {}
        forward_kwargs = {}
        visualize_kwargs = {}
        postprocess_kwargs = {}

        for key, value in kwargs.items():
            if key in self.preprocess_kwargs:
                preprocess_kwargs[key] = value
            if key in self.forward_kwargs:
                forward_kwargs[key] = value
            if key in self.visualize_kwargs:
                visualize_kwargs[key] = value
            if key in self.postprocess_kwargs:
                postprocess_kwargs[key] = value

        return (
            preprocess_kwargs,
            forward_kwargs,
            visualize_kwargs,
            postprocess_kwargs,
        )

    @staticmethod
    def list_models(pattern: Optional[str] = None):
        """List models defined in metafile of corresponding packages.

        Args:
            pattern (str | None): A wildcard pattern to match model names.

        Returns:
            List[str]: a list of model names.
        """
        return list_models(pattern=pattern)

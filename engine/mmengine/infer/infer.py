# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import copy
import importlib
import os.path as osp
import re
import warnings
from abc import ABCMeta, abstractmethod
from datetime import datetime
from typing import (Any, Callable, Dict, Iterable, List, Optional, Sequence,
                    Tuple, Union)

import numpy as np
import torch
import torch.nn as nn
from rich.progress import track

from mmengine.config import Config, ConfigDict
from mmengine.config.utils import MODULE2PACKAGE
from mmengine.dataset import pseudo_collate
from mmengine.device import get_device
from mmengine.fileio import (get_file_backend, isdir, join_path,
                             list_dir_or_file, load)
from mmengine.logging import print_log
from mmengine.registry import FUNCTIONS, MODELS, VISUALIZERS, DefaultScope
from mmengine.runner.checkpoint import (_load_checkpoint,
                                        _load_checkpoint_to_model)
from mmengine.structures import InstanceData
from mmengine.visualization import Visualizer

InstanceList = List[InstanceData]
InputType = Union[str, np.ndarray, torch.Tensor]
InputsType = Union[InputType, Sequence[InputType]]
ImgType = Union[np.ndarray, Sequence[np.ndarray]]
ResType = Union[Dict, List[Dict]]
ConfigType = Union[Config, ConfigDict]
ModelType = Union[dict, ConfigType, str]


class InferencerMeta(ABCMeta):
    """Check the legality of the inferencer.

    All Inferencers should not define duplicated keys for
    ``preprocess_kwargs``, ``forward_kwargs``, ``visualize_kwargs`` and
    ``postprocess_kwargs``.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert isinstance(self.preprocess_kwargs, set)
        assert isinstance(self.forward_kwargs, set)
        assert isinstance(self.visualize_kwargs, set)
        assert isinstance(self.postprocess_kwargs, set)

        all_kwargs = (
            self.preprocess_kwargs | self.forward_kwargs
            | self.visualize_kwargs | self.postprocess_kwargs)

        assert len(all_kwargs) == (
            len(self.preprocess_kwargs) + len(self.forward_kwargs) +
            len(self.visualize_kwargs) + len(self.postprocess_kwargs)), (
                f'Class define error! {self.__name__} should not '
                'define duplicated keys for `preprocess_kwargs`, '
                '`forward_kwargs`, `visualize_kwargs` and '
                '`postprocess_kwargs` are not allowed.')


class BaseInferencer(metaclass=InferencerMeta):
    """Base inferencer for downstream tasks.

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
    ``xxx_kwargs`` mentioned above, therefore, the key in sets should
    be unique to avoid ambiguous dispatching.

    Warning:
        If subclasses defined the class attributes mentioned above with
        duplicated keys, an ``AssertionError`` will be raised during import
        process.

    Subclasses inherited from ``BaseInferencer`` should implement
    :meth:`_init_pipeline`, :meth:`visualize` and :meth:`postprocess`:

    - _init_pipeline: Return a callable object to preprocess the input data.
    - visualize: Visualize the results returned by :meth:`forward`.
    - postprocess: Postprocess the results returned by :meth:`forward` and
      :meth:`visualize`.

    Args:
        model (str, optional): Path to the config file or the model name
            defined in metafile. Take the `mmdet metafile <https://github.com/open-mmlab/mmdetection/blob/master/configs/retinanet/metafile.yml>`_
            as an example, the `model` could be `retinanet_r18_fpn_1x_coco` or
            its alias. If model is not specified, user must provide the
            `weights` saved by MMEngine which contains the config string.
            Defaults to None.
        weights (str, optional): Path to the checkpoint. If it is not specified
            and model is a model name of metafile, the weights will be loaded
            from metafile. Defaults to None.
        device (str, optional): Device to run inference. If None, the available
            device will be automatically used. Defaults to None.
        scope (str, optional): The scope of the model. Defaults to None.
        show_progress (bool): Control whether to display the progress bar during
            the inference process. Defaults to True.
            `New in version 0.7.4.`

    Note:
        Since ``Inferencer`` could be used to infer batch data,
        `collate_fn` should be defined. If `collate_fn` is not defined in config
        file, the `collate_fn` will be `pseudo_collate` by default.
    """  # noqa: E501

    preprocess_kwargs: set = set()
    forward_kwargs: set = set()
    visualize_kwargs: set = set()
    postprocess_kwargs: set = set()

    def __init__(self,
                 model: Union[ModelType, str, None] = None,
                 weights: Optional[str] = None,
                 device: Optional[str] = None,
                 scope: Optional[str] = None,
                 show_progress: bool = True) -> None:
        if scope is None:
            default_scope = DefaultScope.get_current_instance()
            if default_scope is not None:
                scope = default_scope.scope_name
        self.scope = scope
        # Load config to cfg
        cfg: ConfigType
        if isinstance(model, str):
            if osp.isfile(model):
                cfg = Config.fromfile(model)
            else:
                # Load config and weights from metafile. If `weights` is
                # assigned, the weights defined in metafile will be ignored.
                cfg, _weights = self._load_model_from_metafile(model)
                if weights is None:
                    weights = _weights
        elif isinstance(model, (Config, ConfigDict)):
            cfg = copy.deepcopy(model)
        elif isinstance(model, dict):
            cfg = copy.deepcopy(ConfigDict(model))
        elif model is None:
            if weights is None:
                raise ValueError(
                    'If model is None, the weights must be specified since '
                    'the config needs to be loaded from the weights')
            cfg = ConfigDict()
        else:
            raise TypeError('model must be a filepath or any ConfigType'
                            f'object, but got {type(model)}')

        if device is None:
            device = get_device()

        self.model = self._init_model(cfg, weights, device)  # type: ignore
        self.pipeline = self._init_pipeline(cfg)
        self.collate_fn = self._init_collate(cfg)
        self.visualizer = self._init_visualizer(cfg)
        self.cfg = cfg
        self.show_progress = show_progress

    def __call__(
        self,
        inputs: InputsType,
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
        for data in (track(inputs, description='Inference')
                     if self.show_progress else inputs):
            preds.extend(self.forward(data, **forward_kwargs))
        visualization = self.visualize(
            ori_inputs, preds,
            **visualize_kwargs)  # type: ignore  # noqa: E501
        results = self.postprocess(preds, visualization, return_datasamples,
                                   **postprocess_kwargs)
        return results

    def _inputs_to_list(self, inputs: InputsType) -> list:
        """Preprocess the inputs to a list.

        Preprocess inputs to a list according to its type:

        - list or tuple: return inputs
        - str:
            - Directory path: return all files in the directory
            - other cases: return a list containing the string. The string
              could be a path to file, a url or other types of string according
              to the task.

        Args:
            inputs (InputsType): Inputs for the inferencer.

        Returns:
            list: List of input for the :meth:`preprocess`.
        """
        if isinstance(inputs, str):
            backend = get_file_backend(inputs)
            if hasattr(backend, 'isdir') and isdir(inputs):
                # Backends like HttpsBackend do not implement `isdir`, so only
                # those backends that implement `isdir` could accept the inputs
                # as a directory
                filename_list = list_dir_or_file(inputs, list_dir=False)
                inputs = [
                    join_path(inputs, filename) for filename in filename_list
                ]

        if not isinstance(inputs, (list, tuple)):
            inputs = [inputs]

        return list(inputs)

    def preprocess(self, inputs: InputsType, batch_size: int = 1, **kwargs):
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
            Any: Data processed by the ``pipeline`` and ``collate_fn``.
        """
        chunked_data = self._get_chunk_data(
            map(self.pipeline, inputs), batch_size)
        yield from map(self.collate_fn, chunked_data)

    @torch.no_grad()
    def forward(self, inputs: Union[dict, tuple], **kwargs) -> Any:
        """Feed the inputs to the model."""
        return self.model.test_step(inputs)

    @abstractmethod
    def visualize(self,
                  inputs: list,
                  preds: Any,
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

    @abstractmethod
    def postprocess(
        self,
        preds: Any,
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

    def _load_model_from_metafile(self, model: str) -> Tuple[Config, str]:
        """Load config and weights from metafile.

        Args:
            model (str): model name defined in metafile.

        Returns:
            Tuple[Config, str]: Loaded Config and weights path defined in
            metafile.
        """
        model = model.lower()

        assert self.scope is not None, (
            'scope should be initialized if you want '
            'to load config from metafile.')
        assert self.scope in MODULE2PACKAGE, (
            f'{self.scope} not in {MODULE2PACKAGE}!,'
            'please pass a valid scope.')

        repo_or_mim_dir = BaseInferencer._get_repo_or_mim_dir(self.scope)
        for model_cfg in BaseInferencer._get_models_from_metafile(
                repo_or_mim_dir):
            model_name = model_cfg['Name'].lower()
            model_aliases = model_cfg.get('Alias', [])
            if isinstance(model_aliases, str):
                model_aliases = [model_aliases.lower()]
            else:
                model_aliases = [alias.lower() for alias in model_aliases]
            if (model_name == model or model in model_aliases):
                cfg = Config.fromfile(
                    osp.join(repo_or_mim_dir, model_cfg['Config']))
                weights = model_cfg['Weights']
                weights = weights[0] if isinstance(weights, list) else weights
                return cfg, weights
        raise ValueError(f'Cannot find model: {model} in {self.scope}')

    @staticmethod
    def _get_repo_or_mim_dir(scope):
        """Get the directory where the ``Configs`` located when the package is
        installed or ``PYTHONPATH`` is set.

        Args:
            scope (str): The scope of repository.

        Returns:
            str: The directory where the ``Configs`` is located.
        """
        try:
            module = importlib.import_module(scope)
        except ImportError:
            if scope not in MODULE2PACKAGE:
                raise KeyError(
                    f'{scope} is not a valid scope. The available scopes '
                    f'are {MODULE2PACKAGE.keys()}')
            else:
                project = MODULE2PACKAGE[scope]
                raise ImportError(
                    f'Cannot import {scope} correctly, please try to install '
                    f'the {project} by "pip install {project}"')
        # Since none of OpenMMLab series packages are namespace packages
        # (https://docs.python.org/3/glossary.html#term-namespace-package),
        # The first element of module.__path__ means package installation path.
        package_path = module.__path__[0]

        if osp.exists(osp.join(osp.dirname(package_path), 'configs')):
            repo_dir = osp.dirname(package_path)
            return repo_dir
        else:
            mim_dir = osp.join(package_path, '.mim')
            if not osp.exists(osp.join(mim_dir, 'configs')):
                raise FileNotFoundError(
                    f'Cannot find `configs` directory in {package_path}!, '
                    f'please check the completeness of the {scope}.')
            return mim_dir

    def _init_model(
        self,
        cfg: ConfigType,
        weights: Optional[str],
        device: str = 'cpu',
    ) -> nn.Module:
        """Initialize the model with the given config and checkpoint on the
        specific device.

        Args:
            cfg (ConfigType): Config containing the model information.
            weights (str, optional): Path to the checkpoint.
            device (str, optional): Device to run inference. Defaults to 'cpu'.

        Returns:
            nn.Module: Model loaded with checkpoint.
        """
        checkpoint: Optional[dict] = None
        if weights is not None:
            checkpoint = _load_checkpoint(weights, map_location='cpu')

        if not cfg:
            assert checkpoint is not None
            try:
                # Prefer to get config from `message_hub` since `message_hub`
                # is a more stable module to store all runtime information.
                # However, the early version of MMEngine will not save config
                # in `message_hub`, so we will try to load config from `meta`.
                cfg_string = checkpoint['message_hub']['runtime_info']['cfg']
            except KeyError:
                assert 'meta' in checkpoint, (
                    'If model(config) is not provided, the checkpoint must'
                    'contain the config string in `meta` or `message_hub`, '
                    'but both `meta` and `message_hub` are not found in the '
                    'checkpoint.')
                meta = checkpoint['meta']
                if 'cfg' in meta:
                    cfg_string = meta['cfg']
                else:
                    raise ValueError(
                        'Cannot find the config in the checkpoint.')
            cfg.update(
                Config.fromstring(cfg_string, file_format='.py')._cfg_dict)

        # Delete the `pretrained` field to prevent model from loading the
        # the pretrained weights unnecessarily.
        if cfg.model.get('pretrained') is not None:
            del cfg.model.pretrained

        model = MODELS.build(cfg.model)
        model.cfg = cfg
        self._load_weights_to_model(model, checkpoint, cfg)
        model.to(device)
        model.eval()
        return model

    def _load_weights_to_model(self, model: nn.Module,
                               checkpoint: Optional[dict],
                               cfg: Optional[ConfigType]) -> None:
        """Loading model weights and meta information from cfg and checkpoint.

        Subclasses could override this method to load extra meta information
        from ``checkpoint`` and ``cfg`` to model.

        Args:
            model (nn.Module): Model to load weights and meta information.
            checkpoint (dict, optional): The loaded checkpoint.
            cfg (Config or ConfigDict, optional): The loaded config.
        """
        if checkpoint is not None:
            _load_checkpoint_to_model(model, checkpoint)
        else:
            warnings.warn('Checkpoint is not loaded, and the inference '
                          'result is calculated by the randomly initialized '
                          'model!')

    def _init_collate(self, cfg: ConfigType) -> Callable:
        """Initialize the ``collate_fn`` with the given config.

        The returned ``collate_fn`` will be used to collate the batch data.
        If will be used in :meth:`preprocess` like this

        .. code-block:: python
            def preprocess(self, inputs, batch_size, **kwargs):
                ...
                dataloader = map(self.collate_fn, dataloader)
                yield from dataloader

        Args:
            cfg (ConfigType): Config which could contained the `collate_fn`
                information. If `collate_fn` is not defined in config, it will
                be :func:`pseudo_collate`.

        Returns:
            Callable: Collate function.
        """
        try:
            with FUNCTIONS.switch_scope_and_registry(self.scope) as registry:
                collate_fn = registry.get(cfg.test_dataloader.collate_fn)
        except AttributeError:
            collate_fn = pseudo_collate
        return collate_fn  # type: ignore

    @abstractmethod
    def _init_pipeline(self, cfg: ConfigType) -> Callable:
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

    def _init_visualizer(self, cfg: ConfigType) -> Optional[Visualizer]:
        """Initialize visualizers.

        Args:
            cfg (ConfigType): Config containing the visualizer information.

        Returns:
            Visualizer or None: Visualizer initialized with config.
        """
        if 'visualizer' not in cfg:
            return None
        timestamp = str(datetime.timestamp(datetime.now()))
        name = cfg.visualizer.get('name', timestamp)
        if Visualizer.check_instance_created(name):
            name = f'{name}-{timestamp}'
        cfg.visualizer.name = name
        return VISUALIZERS.build(cfg.visualizer)

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

    def _dispatch_kwargs(self, **kwargs) -> Tuple[Dict, Dict, Dict, Dict]:
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
            elif key in self.forward_kwargs:
                forward_kwargs[key] = value
            elif key in self.visualize_kwargs:
                visualize_kwargs[key] = value
            else:
                postprocess_kwargs[key] = value

        return (
            preprocess_kwargs,
            forward_kwargs,
            visualize_kwargs,
            postprocess_kwargs,
        )

    @staticmethod
    def _get_models_from_metafile(dir: str):
        """Load model config defined in metafile from package path.

        Args:
            dir (str): Path to the directory of Config. It requires the
                directory ``Config``, file ``model-index.yml`` exists in the
                ``dir``.

        Yields:
            dict: Model config defined in metafile.
        """
        meta_indexes = load(osp.join(dir, 'model-index.yml'))
        for meta_path in meta_indexes['Import']:
            # meta_path example: mmcls/.mim/configs/conformer/metafile.yml
            meta_path = osp.join(dir, meta_path)
            metainfo = load(meta_path)
            yield from metainfo['Models']

    @staticmethod
    def list_models(scope: Optional[str] = None, patterns: str = r'.*'):
        """List models defined in metafile of corresponding packages.

        Args:
            scope (str, optional): The scope to which the model belongs.
                Defaults to None.
            patterns (str, optional): Regular expressions for the searched
                models. Once matched with ``Alias`` or ``Name`` filed in
                metafile, corresponding model will be added to the return list.
                Defaults to '.*'.

        Returns:
            dict: Model dict with model name and its alias.
        """
        matched_models = []
        if scope is None:
            default_scope = DefaultScope.get_current_instance()
            assert default_scope is not None, (
                'scope should be initialized if you want '
                'to load config from metafile.')
        assert scope in MODULE2PACKAGE, (
            f'{scope} not in {MODULE2PACKAGE}!, please make pass a valid '
            'scope.')
        root_or_mim_dir = BaseInferencer._get_repo_or_mim_dir(scope)
        for model_cfg in BaseInferencer._get_models_from_metafile(
                root_or_mim_dir):
            model_name = [model_cfg['Name']]
            model_name.extend(model_cfg.get('Alias', []))
            for name in model_name:
                if re.match(patterns, name) is not None:
                    matched_models.append(name)
        output_str = ''
        for name in matched_models:
            output_str += f'model_name: {name}\n'
        print_log(output_str, logger='current')
        return matched_models

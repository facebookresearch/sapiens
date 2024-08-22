# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import copy
import functools
import logging
import os
import os.path as osp
import platform
import warnings
from abc import ABCMeta, abstractmethod
from collections.abc import MutableMapping
from typing import Any, Callable, List, Optional, Sequence, Union

import cv2
import numpy as np
import torch

from mmengine.config import Config, ConfigDict
from mmengine.fileio import dump
from mmengine.hooks.logger_hook import SUFFIX_TYPE
from mmengine.logging import MMLogger, print_log
from mmengine.registry import VISBACKENDS
from mmengine.utils import digit_version, scandir
from mmengine.utils.dl_utils import TORCH_VERSION


def force_init_env(old_func: Callable) -> Any:
    """Those methods decorated by ``force_init_env`` will be forced to call
    ``_init_env`` if the instance has not been fully initiated. This function
    will decorated all the `add_xxx` method and `experiment` method, because
    `VisBackend` is initialized only when used its API.

    Args:
        old_func (Callable): Decorated function, make sure the first arg is an
            instance with ``_init_env`` method.

    Returns:
        Any: Depends on old_func.
    """

    @functools.wraps(old_func)
    def wrapper(obj: object, *args, **kwargs):
        # The instance must have `_init_env` method.
        if not hasattr(obj, '_init_env'):
            raise AttributeError(f'{type(obj)} does not have _init_env '
                                 'method.')
        # If instance does not have `_env_initialized` attribute or
        # `_env_initialized` is False, call `_init_env` and set
        # `_env_initialized` to True
        if not getattr(obj, '_env_initialized', False):
            print_log(
                'Attribute `_env_initialized` is not defined in '
                f'{type(obj)} or `{type(obj)}._env_initialized is '
                'False, `_init_env` will be called and '
                f'{type(obj)}._env_initialized will be set to True',
                logger='current',
                level=logging.DEBUG)
            obj._init_env()  # type: ignore
            obj._env_initialized = True  # type: ignore

        return old_func(obj, *args, **kwargs)

    return wrapper


class BaseVisBackend(metaclass=ABCMeta):
    """Base class for visualization backend.

    All backends must inherit ``BaseVisBackend`` and implement
    the required functions.

    Args:
        save_dir (str, optional): The root directory to save
            the files produced by the backend.
    """

    def __init__(self, save_dir: str):
        self._save_dir = save_dir
        self._env_initialized = False

    @property
    @abstractmethod
    def experiment(self) -> Any:
        """Return the experiment object associated with this visualization
        backend.

        The experiment attribute can get the visualization backend, such as
        wandb, tensorboard. If you want to write other data, such as writing a
        table, you can directly get the visualization backend through
        experiment.
        """
        pass

    @abstractmethod
    def _init_env(self) -> Any:
        """Setup env for VisBackend."""
        pass

    def add_config(self, config: Config, **kwargs) -> None:
        """Record the config.

        Args:
            config (Config): The Config object
        """
        pass

    def add_graph(self, model: torch.nn.Module, data_batch: Sequence[dict],
                  **kwargs) -> None:
        """Record the model graph.

        Args:
            model (torch.nn.Module): Model to draw.
            data_batch (Sequence[dict]): Batch of data from dataloader.
        """
        pass

    def add_image(self,
                  name: str,
                  image: np.ndarray,
                  step: int = 0,
                  **kwargs) -> None:
        """Record the image.

        Args:
            name (str): The image identifier.
            image (np.ndarray): The image to be saved. The format
                should be RGB. Defaults to None.
            step (int): Global step value to record. Defaults to 0.
        """
        pass

    def add_scalar(self,
                   name: str,
                   value: Union[int, float],
                   step: int = 0,
                   **kwargs) -> None:
        """Record the scalar.

        Args:
            name (str): The scalar identifier.
            value (int, float): Value to save.
            step (int): Global step value to record. Defaults to 0.
        """
        pass

    def add_scalars(self,
                    scalar_dict: dict,
                    step: int = 0,
                    file_path: Optional[str] = None,
                    **kwargs) -> None:
        """Record the scalars' data.

        Args:
            scalar_dict (dict): Key-value pair storing the tag and
                corresponding values.
            step (int): Global step value to record. Defaults to 0.
            file_path (str, optional): The scalar's data will be
                saved to the `file_path` file at the same time
                if the `file_path` parameter is specified.
                Defaults to None.
        """
        pass

    def close(self) -> None:
        """close an opened object."""
        pass


@VISBACKENDS.register_module()
class LocalVisBackend(BaseVisBackend):
    """Local visualization backend class.

    It can write image, config, scalars, etc.
    to the local hard disk. You can get the drawing backend
    through the experiment property for custom drawing.

    Examples:
        >>> from mmengine.visualization import LocalVisBackend
        >>> import numpy as np
        >>> local_vis_backend = LocalVisBackend(save_dir='temp_dir')
        >>> img = np.random.randint(0, 256, size=(10, 10, 3))
        >>> local_vis_backend.add_image('img', img)
        >>> local_vis_backend.add_scalar('mAP', 0.6)
        >>> local_vis_backend.add_scalars({'loss': [1, 2, 3], 'acc': 0.8})
        >>> cfg = Config(dict(a=1, b=dict(b1=[0, 1])))
        >>> local_vis_backend.add_config(cfg)

    Args:
        save_dir (str, optional): The root directory to save the files
            produced by the visualizer. If it is none, it means no data
            is stored.
        img_save_dir (str): The directory to save images.
            Defaults to 'vis_image'.
        config_save_file (str): The file name to save config.
            Defaults to 'config.py'.
        scalar_save_file (str):  The file name to save scalar values.
            Defaults to 'scalars.json'.
    """

    def __init__(self,
                 save_dir: str,
                 img_save_dir: str = 'vis_image',
                 config_save_file: str = 'config.py',
                 scalar_save_file: str = 'scalars.json'):
        assert config_save_file.split('.')[-1] == 'py'
        assert scalar_save_file.split('.')[-1] == 'json'
        super().__init__(save_dir)
        self._img_save_dir = img_save_dir
        self._config_save_file = config_save_file
        self._scalar_save_file = scalar_save_file

    def _init_env(self):
        """Init save dir."""
        if not os.path.exists(self._save_dir):
            os.makedirs(self._save_dir, exist_ok=True)
        self._img_save_dir = osp.join(
            self._save_dir,  # type: ignore
            self._img_save_dir)
        self._config_save_file = osp.join(
            self._save_dir,  # type: ignore
            self._config_save_file)
        self._scalar_save_file = osp.join(
            self._save_dir,  # type: ignore
            self._scalar_save_file)

    @property  # type: ignore
    @force_init_env
    def experiment(self) -> 'LocalVisBackend':
        """Return the experiment object associated with this visualization
        backend."""
        return self

    @force_init_env
    def add_config(self, config: Config, **kwargs) -> None:
        """Record the config to disk.

        Args:
            config (Config): The Config object
        """
        assert isinstance(config, Config)
        config.dump(self._config_save_file)

    @force_init_env
    def add_image(self,
                  name: str,
                  image: np.array,
                  step: int = 0,
                  **kwargs) -> None:
        """Record the image to disk.

        Args:
            name (str): The image identifier.
            image (np.ndarray): The image to be saved. The format
                should be RGB. Defaults to None.
            step (int): Global step value to record. Defaults to 0.
        """
        assert image.dtype == np.uint8
        drawn_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        os.makedirs(self._img_save_dir, exist_ok=True)
        save_file_name = f'{name}_{step}.png'
        cv2.imwrite(osp.join(self._img_save_dir, save_file_name), drawn_image)

    @force_init_env
    def add_scalar(self,
                   name: str,
                   value: Union[int, float, torch.Tensor, np.ndarray],
                   step: int = 0,
                   **kwargs) -> None:
        """Record the scalar data to disk.

        Args:
            name (str): The scalar identifier.
            value (int, float, torch.Tensor, np.ndarray): Value to save.
            step (int): Global step value to record. Defaults to 0.
        """
        if isinstance(value, torch.Tensor):
            value = value.item()
        self._dump({name: value, 'step': step}, self._scalar_save_file, 'json')

    @force_init_env
    def add_scalars(self,
                    scalar_dict: dict,
                    step: int = 0,
                    file_path: Optional[str] = None,
                    **kwargs) -> None:
        """Record the scalars to disk.

        The scalar dict will be written to the default and
        specified files if ``file_path`` is specified.

        Args:
            scalar_dict (dict): Key-value pair storing the tag and
                corresponding values. The value must be dumped
                into json format.
            step (int): Global step value to record. Defaults to 0.
            file_path (str, optional): The scalar's data will be
                saved to the ``file_path`` file at the same time
                if the ``file_path`` parameter is specified.
                Defaults to None.
        """
        assert isinstance(scalar_dict, dict)
        scalar_dict = copy.deepcopy(scalar_dict)
        scalar_dict.setdefault('step', step)

        if file_path is not None:
            assert file_path.split('.')[-1] == 'json'
            new_save_file_path = osp.join(
                self._save_dir,  # type: ignore
                file_path)
            assert new_save_file_path != self._scalar_save_file, \
                '``file_path`` and ``scalar_save_file`` have the ' \
                'same name, please set ``file_path`` to another value'
            self._dump(scalar_dict, new_save_file_path, 'json')
        self._dump(scalar_dict, self._scalar_save_file, 'json')

    def _dump(self, value_dict: dict, file_path: str,
              file_format: str) -> None:
        """dump dict to file.

        Args:
           value_dict (dict) : The dict data to saved.
           file_path (str): The file path to save data.
           file_format (str): The file format to save data.
        """
        with open(file_path, 'a+') as f:
            dump(value_dict, f, file_format=file_format)
            f.write('\n')


@VISBACKENDS.register_module()
class WandbVisBackend(BaseVisBackend):
    """Wandb visualization backend class.

    Examples:
        >>> from mmengine.visualization import WandbVisBackend
        >>> import numpy as np
        >>> wandb_vis_backend = WandbVisBackend()
        >>> img=np.random.randint(0, 256, size=(10, 10, 3))
        >>> wandb_vis_backend.add_image('img', img)
        >>> wandb_vis_backend.add_scaler('mAP', 0.6)
        >>> wandb_vis_backend.add_scalars({'loss': [1, 2, 3],'acc': 0.8})
        >>> cfg = Config(dict(a=1, b=dict(b1=[0, 1])))
        >>> wandb_vis_backend.add_config(cfg)

    Args:
        save_dir (str, optional): The root directory to save the files
            produced by the visualizer.
        init_kwargs (dict, optional): wandb initialization
            input parameters.
            See `wandb.init <https://docs.wandb.ai/ref/python/init>`_ for
            details. Defaults to None.
        define_metric_cfg (dict or list[dict], optional):
            When a dict is set, it is a dict of metrics and summary for
            ``wandb.define_metric``.
            The key is metric and the value is summary.
            When a list is set, each dict should be a valid argument of
            the ``define_metric``.
            For example, ``define_metric_cfg={'coco/bbox_mAP': 'max'}``,
            means the maximum value of ``coco/bbox_mAP`` is logged on wandb UI.
            When ``define_metric_cfg=[dict(name='loss',
            step_metric='epoch')]``,
            the "loss" will be plotted against the epoch.
            See `wandb define_metric <https://docs.wandb.ai/ref/python/
            run#define_metric>`_ for details.
            Defaults to None.
        commit (bool, optional) Save the metrics dict to the wandb server
            and increment the step.  If false `wandb.log` just updates the
            current metrics dict with the row argument and metrics won't be
            saved until `wandb.log` is called with `commit=True`.
            Defaults to True.
        log_code_name (str, optional) The name of code artifact.
            By default, the artifact will be named
            source-$PROJECT_ID-$ENTRYPOINT_RELPATH. See
            `wandb log_code <https://docs.wandb.ai/ref/python/run#log_code>`_
            for details. Defaults to None.
            `New in version 0.3.0.`
        watch_kwargs (optional, dict): Agurments for ``wandb.watch``.
            `New in version 0.4.0.`
    """

    def __init__(self,
                 save_dir: str,
                 init_kwargs: Optional[dict] = None,
                 define_metric_cfg: Union[dict, list, None] = None,
                 commit: Optional[bool] = True,
                 log_code_name: Optional[str] = None,
                 watch_kwargs: Optional[dict] = None):
        super().__init__(save_dir)
        self._init_kwargs = init_kwargs
        self._define_metric_cfg = define_metric_cfg
        self._commit = commit
        self._log_code_name = log_code_name
        self._watch_kwargs = watch_kwargs if watch_kwargs is not None else {}

    def _init_env(self):
        """Setup env for wandb."""
        if not os.path.exists(self._save_dir):
            os.makedirs(self._save_dir, exist_ok=True)  # type: ignore
        if self._init_kwargs is None:
            self._init_kwargs = {'dir': self._save_dir}
        else:
            self._init_kwargs.setdefault('dir', self._save_dir)
        try:
            import wandb
        except ImportError:
            raise ImportError(
                'Please run "pip install wandb" to install wandb')

        wandb.init(**self._init_kwargs)
        if self._define_metric_cfg is not None:
            if isinstance(self._define_metric_cfg, dict):
                for metric, summary in self._define_metric_cfg.items():
                    wandb.define_metric(metric, summary=summary)
            elif isinstance(self._define_metric_cfg, list):
                for metric_cfg in self._define_metric_cfg:
                    wandb.define_metric(**metric_cfg)
            else:
                raise ValueError('define_metric_cfg should be dict or list')
        self._wandb = wandb

    @property  # type: ignore
    @force_init_env
    def experiment(self):
        """Return wandb object.

        The experiment attribute can get the wandb backend, If you want to
        write other data, such as writing a table, you can directly get the
        wandb backend through experiment.
        """
        return self._wandb

    @force_init_env
    def add_config(self, config: Config, **kwargs) -> None:
        """Record the config to wandb.

        Args:
            config (Config): The Config object
        """
        assert isinstance(self._init_kwargs, dict)
        allow_val_change = self._init_kwargs.get('allow_val_change', False)
        self._wandb.config.update(
            dict(config), allow_val_change=allow_val_change)
        self._wandb.run.log_code(name=self._log_code_name)

    @force_init_env
    def add_graph(self, model: torch.nn.Module, data_batch: Sequence[dict],
                  **kwargs) -> None:
        """Record the model graph.

        Args:
            model (torch.nn.Module): Model to draw.
            data_batch (Sequence[dict]): Batch of data from dataloader.
        """
        self._wandb.watch(model, **self._watch_kwargs)

    @force_init_env
    def add_image(self,
                  name: str,
                  image: np.ndarray,
                  step: int = 0,
                  **kwargs) -> None:
        """Record the image to wandb.

        Args:
            name (str): The image identifier.
            image (np.ndarray): The image to be saved. The format
                should be RGB.
            step (int): Useless parameter. Wandb does not
                need this parameter. Defaults to 0.
        """
        image = self._wandb.Image(image)
        self._wandb.log({name: image}, commit=self._commit)

    @force_init_env
    def add_scalar(self,
                   name: str,
                   value: Union[int, float, torch.Tensor, np.ndarray],
                   step: int = 0,
                   **kwargs) -> None:
        """Record the scalar data to wandb.

        Args:
            name (str): The scalar identifier.
            value (int, float, torch.Tensor, np.ndarray): Value to save.
            step (int): Useless parameter. Wandb does not
                need this parameter. Defaults to 0.
        """
        self._wandb.log({name: value}, commit=self._commit)

    @force_init_env
    def add_scalars(self,
                    scalar_dict: dict,
                    step: int = 0,
                    file_path: Optional[str] = None,
                    **kwargs) -> None:
        """Record the scalar's data to wandb.

        Args:
            scalar_dict (dict): Key-value pair storing the tag and
                corresponding values.
            step (int): Useless parameter. Wandb does not
                need this parameter. Defaults to 0.
            file_path (str, optional): Useless parameter. Just for
                interface unification. Defaults to None.
        """
        self._wandb.log(scalar_dict, commit=self._commit)

    def close(self) -> None:
        """close an opened wandb object."""
        if hasattr(self, '_wandb'):
            self._wandb.join()


@VISBACKENDS.register_module()
class TensorboardVisBackend(BaseVisBackend):
    """Tensorboard visualization backend class.

    It can write images, config, scalars, etc. to a
    tensorboard file.

    Examples:
        >>> from mmengine.visualization import TensorboardVisBackend
        >>> import numpy as np
        >>> vis_backend = TensorboardVisBackend(save_dir='temp_dir')
        >>> img = np.random.randint(0, 256, size=(10, 10, 3))
        >>> vis_backend.add_image('img', img)
        >>> vis_backend.add_scaler('mAP', 0.6)
        >>> vis_backend.add_scalars({'loss': 0.1,'acc':0.8})
        >>> cfg = Config(dict(a=1, b=dict(b1=[0, 1])))
        >>> vis_backend.add_config(cfg)

    Args:
        save_dir (str): The root directory to save the files
            produced by the backend.
    """

    def __init__(self, save_dir: str):
        super().__init__(save_dir)

    def _init_env(self):
        """Setup env for Tensorboard."""
        if not os.path.exists(self._save_dir):
            os.makedirs(self._save_dir, exist_ok=True)  # type: ignore
        if TORCH_VERSION == 'parrots':
            try:
                from tensorboardX import SummaryWriter
            except ImportError:
                raise ImportError('Please install tensorboardX to use '
                                  'TensorboardLoggerHook.')
        else:
            try:
                from torch.utils.tensorboard import SummaryWriter
            except ImportError:
                raise ImportError(
                    'Please run "pip install future tensorboard" to install '
                    'the dependencies to use torch.utils.tensorboard '
                    '(applicable to PyTorch 1.1 or higher)')
        self._tensorboard = SummaryWriter(self._save_dir)

    @property  # type: ignore
    @force_init_env
    def experiment(self):
        """Return Tensorboard object."""
        return self._tensorboard

    @force_init_env
    def add_config(self, config: Config, **kwargs) -> None:
        """Record the config to tensorboard.

        Args:
            config (Config): The Config object
        """
        self._tensorboard.add_text('config', config.pretty_text)

    @force_init_env
    def add_image(self,
                  name: str,
                  image: np.ndarray,
                  step: int = 0,
                  **kwargs) -> None:
        """Record the image to tensorboard.

        Args:
            name (str): The image identifier.
            image (np.ndarray): The image to be saved. The format
                should be RGB.
            step (int): Global step value to record. Defaults to 0.
        """
        self._tensorboard.add_image(name, image, step, dataformats='HWC')

    @force_init_env
    def add_scalar(self,
                   name: str,
                   value: Union[int, float, torch.Tensor, np.ndarray],
                   step: int = 0,
                   **kwargs) -> None:
        """Record the scalar data to tensorboard.

        Args:
            name (str): The scalar identifier.
            value (int, float, torch.Tensor, np.ndarray): Value to save.
            step (int): Global step value to record. Defaults to 0.
        """
        if isinstance(value,
                      (int, float, torch.Tensor, np.ndarray, np.number)):
            self._tensorboard.add_scalar(name, value, step)
        else:
            warnings.warn(f'Got {type(value)}, but numpy array, torch tensor, '
                          f'int or float are expected. skip it!')

    @force_init_env
    def add_scalars(self,
                    scalar_dict: dict,
                    step: int = 0,
                    file_path: Optional[str] = None,
                    **kwargs) -> None:
        """Record the scalar's data to tensorboard.

        Args:
            scalar_dict (dict): Key-value pair storing the tag and
                corresponding values.
            step (int): Global step value to record. Defaults to 0.
            file_path (str, optional): Useless parameter. Just for
                interface unification. Defaults to None.
        """
        assert isinstance(scalar_dict, dict)
        assert 'step' not in scalar_dict, 'Please set it directly ' \
                                          'through the step parameter'
        for key, value in scalar_dict.items():
            self.add_scalar(key, value, step)

    def close(self):
        """close an opened tensorboard object."""
        if hasattr(self, '_tensorboard'):
            self._tensorboard.close()


@VISBACKENDS.register_module()
class MLflowVisBackend(BaseVisBackend):
    """MLflow visualization backend class.

    It can write images, config, scalars, etc. to a
    mlflow file.

    Examples:
        >>> from mmengine.visualization import MLflowVisBackend
        >>> from mmengine import Config
        >>> import numpy as np
        >>> vis_backend = MLflowVisBackend(save_dir='temp_dir')
        >>> img = np.random.randint(0, 256, size=(10, 10, 3))
        >>> vis_backend.add_image('img.png', img)
        >>> vis_backend.add_scalar('mAP', 0.6)
        >>> vis_backend.add_scalars({'loss': 0.1,'acc':0.8})
        >>> cfg = Config(dict(a=1, b=dict(b1=[0, 1])))
        >>> vis_backend.add_config(cfg)

    Args:
        save_dir (str): The root directory to save the files
            produced by the backend.
        exp_name (str, optional): The experiment name. Defaults to None.
        run_name (str, optional): The run name. Defaults to None.
        tags (dict, optional): The tags to be added to the experiment.
            Defaults to None.
        params (dict, optional): The params to be added to the experiment.
            Defaults to None.
        tracking_uri (str, optional): The tracking uri. Defaults to None.
        artifact_suffix (Tuple[str] or str, optional): The artifact suffix.
            Defaults to ('.json', '.log', '.py', 'yaml').
        tracked_config_keys (dict, optional): The top level keys of config that
            will be added to the experiment. If it is None, which means all
            the config will be added. Defaults to None.
            `New in version 0.7.4.`
    """

    def __init__(self,
                 save_dir: str,
                 exp_name: Optional[str] = None,
                 run_name: Optional[str] = None,
                 tags: Optional[dict] = None,
                 params: Optional[dict] = None,
                 tracking_uri: Optional[str] = None,
                 artifact_suffix: SUFFIX_TYPE = ('.json', '.log', '.py',
                                                 'yaml'),
                 tracked_config_keys: Optional[dict] = None):
        super().__init__(save_dir)
        self._exp_name = exp_name
        self._run_name = run_name
        self._tags = tags
        self._params = params
        self._tracking_uri = tracking_uri
        self._artifact_suffix = artifact_suffix
        self._tracked_config_keys = tracked_config_keys

    def _init_env(self):
        """Setup env for MLflow."""
        if not os.path.exists(self._save_dir):
            os.makedirs(self._save_dir, exist_ok=True)  # type: ignore

        try:
            import mlflow
        except ImportError:
            raise ImportError(
                'Please run "pip install mlflow" to install mlflow'
            )  # type: ignore
        self._mlflow = mlflow

        # when mlflow is imported, a default logger is created.
        # at this time, the default logger's stream is None
        # so the stream is reopened only when the stream is None
        # or the stream is closed
        logger = MMLogger.get_current_instance()
        for handler in logger.handlers:
            if handler.stream is None or handler.stream.closed:
                handler.stream = open(handler.baseFilename, 'a')

        if self._tracking_uri is not None:
            logger.warning(
                'Please make sure that the mlflow server is running.')
            self._mlflow.set_tracking_uri(self._tracking_uri)
        else:
            if os.name == 'nt':
                file_url = f'file:\\{os.path.abspath(self._save_dir)}'
            else:
                file_url = f'file://{os.path.abspath(self._save_dir)}'
            self._mlflow.set_tracking_uri(file_url)

        self._exp_name = self._exp_name or 'Default'

        if self._mlflow.get_experiment_by_name(self._exp_name) is None:
            self._mlflow.create_experiment(self._exp_name)

        self._mlflow.set_experiment(self._exp_name)

        if self._run_name is not None:
            self._mlflow.set_tag('mlflow.runName', self._run_name)
        if self._tags is not None:
            self._mlflow.set_tags(self._tags)
        if self._params is not None:
            self._mlflow.log_params(self._params)

    @property  # type: ignore
    @force_init_env
    def experiment(self):
        """Return MLflow object."""
        return self._mlflow

    @force_init_env
    def add_config(self, config: Config, **kwargs) -> None:
        """Record the config to mlflow.

        Args:
            config (Config): The Config object
        """
        self.cfg = config
        if self._tracked_config_keys is None:
            self._mlflow.log_params(self._flatten(self.cfg))
        else:
            tracked_cfg = dict()
            for k in self._tracked_config_keys:
                tracked_cfg[k] = self.cfg[k]
            self._mlflow.log_params(self._flatten(tracked_cfg))
        self._mlflow.log_text(self.cfg.pretty_text, 'config.py')

    @force_init_env
    def add_image(self,
                  name: str,
                  image: np.ndarray,
                  step: int = 0,
                  **kwargs) -> None:
        """Record the image to mlflow.

        Args:
            name (str): The image identifier.
            image (np.ndarray): The image to be saved. The format
                should be RGB.
            step (int): Global step value to record. Default to 0.
        """
        self._mlflow.log_image(image, name)

    @force_init_env
    def add_scalar(self,
                   name: str,
                   value: Union[int, float, torch.Tensor, np.ndarray],
                   step: int = 0,
                   **kwargs) -> None:
        """Record the scalar data to mlflow.

        Args:
            name (str): The scalar identifier.
            value (int, float, torch.Tensor, np.ndarray): Value to save.
            step (int): Global step value to record. Default to 0.
        """
        self._mlflow.log_metric(name, value, step)

    @force_init_env
    def add_scalars(self,
                    scalar_dict: dict,
                    step: int = 0,
                    file_path: Optional[str] = None,
                    **kwargs) -> None:
        """Record the scalar's data to mlflow.

        Args:
            scalar_dict (dict): Key-value pair storing the tag and
                corresponding values.
            step (int): Global step value to record. Default to 0.
            file_path (str, optional): Useless parameter. Just for
                interface unification. Defaults to None.
        """
        assert isinstance(scalar_dict, dict)
        assert 'step' not in scalar_dict, 'Please set it directly ' \
                                          'through the step parameter'
        self._mlflow.log_metrics(scalar_dict, step)

    def close(self) -> None:
        """Close the mlflow."""
        if not hasattr(self, '_mlflow'):
            return

        file_paths = dict()
        for filename in scandir(self.cfg.work_dir, self._artifact_suffix,
                                True):
            file_path = osp.join(self.cfg.work_dir, filename)
            relative_path = os.path.relpath(file_path, self.cfg.work_dir)
            dir_path = os.path.dirname(relative_path)
            file_paths[file_path] = dir_path

        for file_path, dir_path in file_paths.items():
            self._mlflow.log_artifact(file_path, dir_path)

        self._mlflow.end_run()

    def _flatten(self, d, parent_key='', sep='.') -> dict:
        """Flatten the dict."""
        items = dict()
        for k, v in d.items():
            new_key = parent_key + sep + k if parent_key else k
            if isinstance(v, MutableMapping):
                items.update(self._flatten(v, new_key, sep=sep))
            elif isinstance(v, list):
                if any(isinstance(x, dict) for x in v):
                    for i, x in enumerate(v):
                        items.update(
                            self._flatten(x, new_key + sep + str(i), sep=sep))
                else:
                    items[new_key] = v
            else:
                items[new_key] = v
        return items


@VISBACKENDS.register_module()
class ClearMLVisBackend(BaseVisBackend):
    """Clearml visualization backend class. It requires `clearml`_ to be
    installed.

    Examples:
        >>> from mmengine.visualization import ClearMLVisBackend
        >>> from mmengine import Config
        >>> import numpy as np
        >>> vis_backend = ClearMLVisBackend(save_dir='temp_dir')
        >>> img = np.random.randint(0, 256, size=(10, 10, 3))
        >>> vis_backend.add_image('img.png', img)
        >>> vis_backend.add_scalar('mAP', 0.6)
        >>> vis_backend.add_scalars({'loss': 0.1,'acc':0.8})
        >>> cfg = Config(dict(a=1, b=dict(b1=[0, 1])))
        >>> vis_backend.add_config(cfg)

    Args:
        save_dir (str, optional): Useless parameter. Just for
            interface unification. Defaults to None.
        init_kwargs (dict, optional): A dict contains the arguments of
            ``clearml.Task.init`` . See `taskinit`_  for more details.
            Defaults to None
        artifact_suffix (Tuple[str] or str): The artifact suffix.
            Defaults to ('.py', 'pth').

    .. _clearml:
        https://clear.ml/docs/latest/docs/

    .. _taskinit:
        https://clear.ml/docs/latest/docs/references/sdk/task/#taskinit
    """

    def __init__(self,
                 save_dir: Optional[str] = None,
                 init_kwargs: Optional[dict] = None,
                 artifact_suffix: SUFFIX_TYPE = ('.py', '.pth')):
        super().__init__(save_dir)  # type: ignore
        self._init_kwargs = init_kwargs
        self._artifact_suffix = artifact_suffix

    def _init_env(self) -> None:
        try:
            import clearml
        except ImportError:
            raise ImportError(
                'Please run "pip install clearml" to install clearml')

        task_kwargs = self._init_kwargs or {}
        self._clearml = clearml
        self._task = self._clearml.Task.init(**task_kwargs)
        self._logger = self._task.get_logger()

    @property  # type: ignore
    @force_init_env
    def experiment(self):
        """Return clearml object."""
        return self._clearml

    @force_init_env
    def add_config(self, config: Config, **kwargs) -> None:
        """Record the config to clearml.

        Args:
            config (Config): The Config object
        """
        self.cfg = config
        self._task.connect_configuration(vars(config))

    @force_init_env
    def add_image(self,
                  name: str,
                  image: np.ndarray,
                  step: int = 0,
                  **kwargs) -> None:
        """Record the image to clearml.

        Args:
            name (str): The image identifier.
            image (np.ndarray): The image to be saved. The format
                should be RGB.
            step (int): Global step value to record. Defaults to 0.
        """
        self._logger.report_image(
            title=name, series=name, iteration=step, image=image)

    @force_init_env
    def add_scalar(self,
                   name: str,
                   value: Union[int, float, torch.Tensor, np.ndarray],
                   step: int = 0,
                   **kwargs) -> None:
        """Record the scalar data to clearml.

        Args:
            name (str): The scalar identifier.
            value (int, float, torch.Tensor, np.ndarray): Value to save.
            step (int): Global step value to record. Defaults to 0.
        """
        self._logger.report_scalar(
            title=name, series=name, value=value, iteration=step)

    @force_init_env
    def add_scalars(self,
                    scalar_dict: dict,
                    step: int = 0,
                    file_path: Optional[str] = None,
                    **kwargs) -> None:
        """Record the scalar's data to clearml.

        Args:
            scalar_dict (dict): Key-value pair storing the tag and
                corresponding values.
            step (int): Global step value to record. Defaults to 0.
            file_path (str, optional): Useless parameter. Just for
                interface unification. Defaults to None.
        """
        assert 'step' not in scalar_dict, 'Please set it directly ' \
                                          'through the step parameter'
        for key, value in scalar_dict.items():
            self._logger.report_scalar(
                title=key, series=key, value=value, iteration=step)

    def close(self) -> None:
        """Close the clearml."""
        if not hasattr(self, '_clearml'):
            return

        file_paths: List[str] = list()
        if (hasattr(self, 'cfg')
                and osp.isdir(getattr(self.cfg, 'work_dir', ''))):
            for filename in scandir(self.cfg.work_dir, self._artifact_suffix,
                                    False):
                file_path = osp.join(self.cfg.work_dir, filename)
                file_paths.append(file_path)

        for file_path in file_paths:
            self._task.upload_artifact(os.path.basename(file_path), file_path)
        self._task.close()


@VISBACKENDS.register_module()
class NeptuneVisBackend(BaseVisBackend):
    """Neptune visualization backend class.

    Examples:
        >>> from mmengine.visualization import NeptuneVisBackend
        >>> from mmengine import Config
        >>> import numpy as np
        >>> init_kwargs = {'project': 'your_project_name'}
        >>> neptune_vis_backend = NeptuneVisBackend(init_kwargs=init_kwargs)
        >>> img = np.random.randint(0, 256, size=(10, 10, 3))
        >>> neptune_vis_backend.add_image('img', img)
        >>> neptune_vis_backend.add_scalar('mAP', 0.6)
        >>> neptune_vis_backend.add_scalars({'loss': 0.1, 'acc': 0.8})
        >>> cfg = Config(dict(a=1, b=dict(b1=[0, 1])))
        >>> neptune_vis_backend.add_config(cfg)

    Note:
        `New in version 0.8.5.`

    Args:
        save_dir (str, optional): The root directory to save the files
            produced by the visualizer. NeptuneVisBackend does
            not require this argument. Defaults to None.
        init_kwargs (dict, optional): Neptune initialization parameters.
            Defaults to None.

            - project (str): Name of a project in a form of
              `namespace/project_name`. If `project` is not specified,
              the value of `NEPTUNE_PROJECT` environment variable
              will be taken.
            - api_token (str): User's API token. If api_token is not api_token,
              the value of `NEPTUNE_API_TOKEN` environment variable will
              be taken. Note: It is strongly recommended to use
              `NEPTUNE_API_TOKEN` environment variable rather than
              placing your API token here.

            If 'project' and 'api_token are not specified in `init_kwargs`,
            the 'mode' will be set to 'offline'.
            See `neptune.init_run
            <https://docs.neptune.ai/api/neptune/#init_run>`_ for
            details.
    """

    def __init__(self,
                 save_dir: Optional[str] = None,
                 init_kwargs: Optional[dict] = None):
        super().__init__(save_dir)  # type:ignore
        self._init_kwargs = init_kwargs

    def _init_env(self):
        """Setup env for neptune."""
        try:
            import neptune
        except ImportError:
            raise ImportError(
                'Please run "pip install -U neptune" to install neptune')
        if self._init_kwargs is None:
            self._init_kwargs = {'mode': 'offline'}

        self._neptune = neptune.init_run(**self._init_kwargs)

    @property  # type: ignore
    @force_init_env
    def experiment(self):
        """Return Neptune object."""
        return self._neptune

    @force_init_env
    def add_config(self, config: Config, **kwargs) -> None:
        """Record the config to neptune.

        Args:
            config (Config): The Config object
        """
        from neptune.types import File
        self._neptune['config'].upload(File.from_content(config.pretty_text))

    @force_init_env
    def add_image(self,
                  name: str,
                  image: np.ndarray,
                  step: int = 0,
                  **kwargs) -> None:
        """Record the image.

        Args:
            name (str): The image identifier.
            image (np.ndarray): The image to be saved. The format
                should be RGB. Defaults to None.
            step (int): Global step value to record. Defaults to 0.
        """
        from neptune.types import File

        # values in the array need to be in the [0, 1] range
        img = image.astype(np.float32) / 255.0
        self._neptune['images'].append(
            File.as_image(img), name=name, step=step)

    @force_init_env
    def add_scalar(self,
                   name: str,
                   value: Union[int, float],
                   step: int = 0,
                   **kwargs) -> None:
        """Record the scalar.

        Args:
            name (str): The scalar identifier.
            value (int, float): Value to save.
            step (int): Global step value to record. Defaults to 0.
        """
        self._neptune[name].append(value, step=step)

    @force_init_env
    def add_scalars(self,
                    scalar_dict: dict,
                    step: int = 0,
                    file_path: Optional[str] = None,
                    **kwargs) -> None:
        """Record the scalars' data.

        Args:
            scalar_dict (dict): Key-value pair storing the tag and
                corresponding values.
            step (int): Global step value to record. Defaults to 0.
            file_path (str, optional): The scalar's data will be
                saved to the `file_path` file at the same time
                if the `file_path` parameter is specified.
                Defaults to None.
        """
        assert isinstance(scalar_dict, dict)
        assert 'step' not in scalar_dict, 'Please set it directly ' \
                                          'through the step parameter'

        for k, v in scalar_dict.items():
            self._neptune[k].append(v, step=step)

    def close(self) -> None:
        """close an opened object."""
        if hasattr(self, '_neptune'):
            self._neptune.stop()


@VISBACKENDS.register_module()
class DVCLiveVisBackend(BaseVisBackend):
    """DVCLive visualization backend class.

    Examples:
        >>> from mmengine.visualization import DVCLiveVisBackend
        >>> import numpy as np
        >>> dvclive_vis_backend = DVCLiveVisBackend(save_dir='temp_dir')
        >>> img=np.random.randint(0, 256, size=(10, 10, 3))
        >>> dvclive_vis_backend.add_image('img', img)
        >>> dvclive_vis_backend.add_scalar('mAP', 0.6)
        >>> dvclive_vis_backend.add_scalars({'loss': 0.1, 'acc': 0.8})
        >>> cfg = Config(dict(a=1, b=dict(b1=[0, 1])))
        >>> dvclive_vis_backend.add_config(cfg)

    Note:
        `New in version 0.8.5.`

    Args:
        save_dir (str, optional): The root directory to save the files
            produced by the visualizer.
        artifact_suffix (Tuple[str] or str, optional): The artifact suffix.
            Defaults to ('.json', '.py', 'yaml').
        init_kwargs (dict, optional): DVCLive initialization parameters.
            See `DVCLive <https://dvc.org/doc/dvclive/live>`_ for details.
            Defaults to None.
    """

    def __init__(self,
                 save_dir: str,
                 artifact_suffix: SUFFIX_TYPE = ('.json', '.py', 'yaml'),
                 init_kwargs: Optional[dict] = None):
        super().__init__(save_dir)
        self._artifact_suffix = artifact_suffix
        self._init_kwargs = init_kwargs

    def _init_env(self):
        """Setup env for dvclive."""
        if digit_version(platform.python_version()) < digit_version('3.8'):
            raise RuntimeError('Please use Python 3.8 or higher version '
                               'to use DVCLiveVisBackend.')

        try:
            import pygit2
            from dvclive import Live
        except ImportError:
            raise ImportError(
                'Please run "pip install dvclive" to install dvclive')
        # if no git info, init dvc without git to avoid SCMError
        try:
            path = pygit2.discover_repository(os.fspath(os.curdir), True, '')
            pygit2.Repository(path).default_signature
        except KeyError:
            os.system('dvc init -f --no-scm')

        if self._init_kwargs is None:
            self._init_kwargs = {}
        self._init_kwargs.setdefault('dir', self._save_dir)
        self._init_kwargs.setdefault('save_dvc_exp', True)
        self._init_kwargs.setdefault('cache_images', True)

        self._dvclive = Live(**self._init_kwargs)

    @property  # type: ignore
    @force_init_env
    def experiment(self):
        """Return dvclive object.

        The experiment attribute can get the dvclive backend, If you want to
        write other data, such as writing a table, you can directly get the
        dvclive backend through experiment.
        """
        return self._dvclive

    @force_init_env
    def add_config(self, config: Config, **kwargs) -> None:
        """Record the config to dvclive.

        Args:
            config (Config): The Config object
        """
        assert isinstance(config, Config)
        self.cfg = config
        self._dvclive.log_params(self._to_dvc_paramlike(self.cfg))

    @force_init_env
    def add_image(self,
                  name: str,
                  image: np.ndarray,
                  step: int = 0,
                  **kwargs) -> None:
        """Record the image to dvclive.

        Args:
            name (str): The image identifier.
            image (np.ndarray): The image to be saved. The format
                should be RGB.
            step (int): Useless parameter. Dvclive does not
                need this parameter. Defaults to 0.
        """
        assert image.dtype == np.uint8
        save_file_name = f'{name}.png'

        self._dvclive.log_image(save_file_name, image)

    @force_init_env
    def add_scalar(self,
                   name: str,
                   value: Union[int, float, torch.Tensor, np.ndarray],
                   step: int = 0,
                   **kwargs) -> None:
        """Record the scalar data to dvclive.

        Args:
            name (str): The scalar identifier.
            value (int, float, torch.Tensor, np.ndarray): Value to save.
            step (int): Global step value to record. Defaults to 0.
        """
        if isinstance(value, torch.Tensor):
            value = value.numpy()
        self._dvclive.step = step
        self._dvclive.log_metric(name, value)

    @force_init_env
    def add_scalars(self,
                    scalar_dict: dict,
                    step: int = 0,
                    file_path: Optional[str] = None,
                    **kwargs) -> None:
        """Record the scalar's data to dvclive.

        Args:
            scalar_dict (dict): Key-value pair storing the tag and
                corresponding values.
            step (int): Global step value to record. Defaults to 0.
            file_path (str, optional): Useless parameter. Just for
                interface unification. Defaults to None.
        """
        for key, value in scalar_dict.items():
            self.add_scalar(key, value, step, **kwargs)

    def close(self) -> None:
        """close an opened dvclive object."""
        if not hasattr(self, '_dvclive'):
            return

        file_paths = dict()
        for filename in scandir(self._save_dir, self._artifact_suffix, True):
            file_path = osp.join(self._save_dir, filename)
            relative_path = os.path.relpath(file_path, self._save_dir)
            dir_path = os.path.dirname(relative_path)
            file_paths[file_path] = dir_path

        for file_path, dir_path in file_paths.items():
            self._dvclive.log_artifact(file_path, dir_path)

        self._dvclive.end()

    def _to_dvc_paramlike(self,
                          value: Union[int, float, dict, list, tuple, Config,
                                       ConfigDict, torch.Tensor, np.ndarray]):
        """Convert the input value to a DVC `ParamLike` recursively.

        Or the `log_params` method of dvclive will raise an error.
        """

        if isinstance(value, (dict, Config, ConfigDict)):
            return {k: self._to_dvc_paramlike(v) for k, v in value.items()}
        elif isinstance(value, (tuple, list)):
            return [self._to_dvc_paramlike(item) for item in value]
        elif isinstance(value, (torch.Tensor, np.ndarray)):
            return value.tolist()
        elif isinstance(value, np.generic):
            return value.item()
        else:
            return value


@VISBACKENDS.register_module()
class AimVisBackend(BaseVisBackend):
    """Aim visualization backend class.

    Examples:
        >>> from mmengine.visualization import AimVisBackend
        >>> import numpy as np
        >>> aim_vis_backend = AimVisBackend()
        >>> img=np.random.randint(0, 256, size=(10, 10, 3))
        >>> aim_vis_backend.add_image('img', img)
        >>> aim_vis_backend.add_scalar('mAP', 0.6)
        >>> aim_vis_backend.add_scalars({'loss': 0.1, 'acc': 0.8})
        >>> cfg = Config(dict(a=1, b=dict(b1=[0, 1])))
        >>> aim_vis_backend.add_config(cfg)

    Note:
        1. `New in version 0.8.5.`
        2. Refer to
           `Github issue <https://github.com/aimhubio/aim/issues/2064>`_ ,
           Aim is not unable to be install on Windows for now.

    Args:
        save_dir (str, optional): The root directory to save the files
            produced by the visualizer.
        init_kwargs (dict, optional): Aim initialization parameters. See
            `Aim <https://aimstack.readthedocs.io/en/latest/refs/sdk.html>`_
            for details. Defaults to None.
    """

    def __init__(self,
                 save_dir: Optional[str] = None,
                 init_kwargs: Optional[dict] = None):
        super().__init__(save_dir)  # type:ignore
        self._init_kwargs = init_kwargs

    def _init_env(self):
        """Setup env for Aim."""
        try:
            from aim import Run
        except ImportError:
            raise ImportError('Please run "pip install aim" to install aim')

        from datetime import datetime

        if self._save_dir is not None:
            path_list = os.path.normpath(self._save_dir).split(os.sep)
            exp_name = f'{path_list[-2]}_{path_list[-1]}'
        else:
            exp_name = datetime.now().strftime('%Y%m%d_%H%M%S')

        if self._init_kwargs is None:
            self._init_kwargs = {}
        self._init_kwargs.setdefault('experiment', exp_name)
        self._aim_run = Run(**self._init_kwargs)

    @property  # type: ignore
    @force_init_env
    def experiment(self):
        """Return Aim object."""
        return self._aim_run

    @force_init_env
    def add_config(self, config, **kwargs) -> None:
        """Record the config to Aim.

        Args:
            config (Config): The Config object
        """
        if isinstance(config, Config):
            config = config.to_dict()
        self._aim_run['hparams'] = config

    @force_init_env
    def add_image(self,
                  name: str,
                  image: np.ndarray,
                  step: int = 0,
                  **kwargs) -> None:
        """Record the image.

        Args:
            name (str): The image identifier.
            image (np.ndarray): The image to be saved. The format
                should be RGB. Defaults to None.
            step (int): Global step value to record. Defaults to 0.
        """
        from aim import Image
        self._aim_run.track(name=name, value=Image(image), step=step)

    @force_init_env
    def add_scalar(self,
                   name: str,
                   value: Union[int, float, torch.Tensor, np.ndarray],
                   step: int = 0,
                   **kwargs) -> None:
        """Record the scalar data to Aim.

        Args:
            name (str): The scalar identifier.
            value (int, float, torch.Tensor, np.ndarray): Value to save.
            step (int): Global step value to record. Default to 0.
        """
        self._aim_run.track(name=name, value=value, step=step)

    @force_init_env
    def add_scalars(self,
                    scalar_dict: dict,
                    step: int = 0,
                    file_path: Optional[str] = None,
                    **kwargs) -> None:
        """Record the scalar's data to wandb.

        Args:
            scalar_dict (dict): Key-value pair storing the tag and
                corresponding values.
            step (int): Global step value to record. Default to 0.
            file_path (str, optional): Useless parameter. Just for
                interface unification. Defaults to None.
        """
        for key, value in scalar_dict.items():
            self._aim_run.track(name=key, value=value, step=step)

    def close(self) -> None:
        """Close the Aim."""
        if not hasattr(self, '_aim_run'):
            return

        self._aim_run.close()

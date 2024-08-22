# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import importlib
import os.path as osp

from mmengine.config import Config
from mmengine.config.utils import (_get_cfg_metainfo,
                                   _get_external_cfg_base_path,
                                   _get_package_and_cfg_path)
from mmengine.registry import MODELS, DefaultScope
from mmengine.runner import load_checkpoint
from mmengine.utils import get_installed_path, install_package


def get_config(cfg_path: str, pretrained: bool = False) -> Config:
    """Get config from external package.

    Args:
        cfg_path (str): External relative config path.
        pretrained (bool): Whether to save pretrained model path. If
            ``pretrained==True``, the url of pretrained model can be accessed
            by ``cfg.model_path``. Defaults to False.

    Examples:
        >>> cfg = get_config('mmdet::faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py', pretrained=True)
        >>> # Equivalent to
        >>> # cfg = Config.fromfile('/path/to/faster-rcnn_r50_fpn_1x_coco.py')
        >>> cfg.model_path
        https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth

    Returns:
        Config: A `Config` parsed from external package.
    """  # noqa E301
    # Get package name and relative config path.
    package, cfg_path = _get_package_and_cfg_path(cfg_path)
    # Install package if it's not installed.
    install_package(package)
    package_path = get_installed_path(package)
    try:
        # Use `cfg_path` to search target config file.
        cfg_meta = _get_cfg_metainfo(package_path, cfg_path)
        cfg_path = osp.join(package_path, '.mim', cfg_meta['Config'])
        cfg = Config.fromfile(cfg_path)
        if pretrained:
            assert 'Weights' in cfg_meta, ('Cannot find `Weights` in cfg_file'
                                           '.metafile.yml, please check the'
                                           'metafile')
            cfg.model_path = cfg_meta['Weights']
    except ValueError:
        # Since the base config does not contain a metafile, the absolute
        # config is `osp.join(package_path, cfg_path_prefix, cfg_name)`
        cfg_path = _get_external_cfg_base_path(package_path, cfg_path)
        cfg = Config.fromfile(cfg_path)
    except Exception as e:
        raise e
    return cfg


def get_model(cfg_path: str, pretrained: bool = False, **kwargs):
    """Get built model from external package.

    Args:
        cfg_path (str): External relative config path with prefix
            'package::' and without suffix.
        pretrained (bool): Whether to load pretrained model. Defaults to False.
        kwargs (dict): Default arguments to build model.

    Examples:
        >>> model = get_model('mmdet::faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py', pretrained=True)
        >>> type(model)
        <class 'mmdet.models.detectors.faster_rcnn.FasterRCNN'>

    Returns:
        nn.Module: Built model.
    """  # noqa E301
    package = cfg_path.split('::')[0]
    with DefaultScope.overwrite_default_scope(package):  # type: ignore
        cfg = get_config(cfg_path, pretrained)
        if 'data_preprocessor' in cfg:
            cfg.model.data_preprocessor = cfg.data_preprocessor
        models_module = importlib.import_module(f'{package}.utils')
        models_module.register_all_modules()  # type: ignore
        model = MODELS.build(cfg.model, default_args=kwargs)
        if pretrained:
            load_checkpoint(model, cfg.model_path)
            # Hack to use pretrained weights.
            # If we do not set _is_init here, Runner will call
            # `model.init_weights()` to overwrite the pretrained model.
            model._is_init = True
        return model

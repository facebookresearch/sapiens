# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from mmpose.utils.typing import ConfigDict


def adapt_mmdet_pipeline(cfg: ConfigDict) -> ConfigDict:
    """Converts pipeline types in MMDetection's test dataloader to use the
    'mmdet' namespace.

    Args:
        cfg (ConfigDict): Configuration dictionary for MMDetection.

    Returns:
        ConfigDict: Configuration dictionary with updated pipeline types.
    """
    # use lazy import to avoid hard dependence on mmdet
    from mmdet.datasets import transforms

    if 'test_dataloader' not in cfg:
        return cfg

    pipeline = cfg.test_dataloader.dataset.pipeline
    for trans in pipeline:
        if trans['type'] in dir(transforms):
            trans['type'] = 'mmdet.' + trans['type']

    return cfg

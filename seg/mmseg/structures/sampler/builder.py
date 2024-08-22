# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import warnings

from mmseg.registry import TASK_UTILS

PIXEL_SAMPLERS = TASK_UTILS


def build_pixel_sampler(cfg, **default_args):
    """Build pixel sampler for segmentation map."""
    warnings.warn(
        '``build_pixel_sampler`` would be deprecated soon, please use '
        '``mmseg.registry.TASK_UTILS.build()`` ')
    return TASK_UTILS.build(cfg, default_args=default_args)

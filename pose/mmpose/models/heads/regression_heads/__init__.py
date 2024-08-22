# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from .dsnt_head import DSNTHead
from .integral_regression_head import IntegralRegressionHead
from .regression_head import RegressionHead
from .rle_head import RLEHead
from .temporal_regression_head import TemporalRegressionHead
from .trajectory_regression_head import TrajectoryRegressionHead

__all__ = [
    'RegressionHead',
    'IntegralRegressionHead',
    'DSNTHead',
    'RLEHead',
    'TemporalRegressionHead',
    'TrajectoryRegressionHead',
]

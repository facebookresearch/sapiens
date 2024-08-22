# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Collecting some commonly used type hint in mmflow."""
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
from mmengine.config import ConfigDict

from mmseg.structures import SegDataSample

# Type hint of config data
ConfigType = Union[ConfigDict, dict]
OptConfigType = Optional[ConfigType]
# Type hint of one or more config data
MultiConfig = Union[ConfigType, Sequence[ConfigType]]
OptMultiConfig = Optional[MultiConfig]

SampleList = Sequence[SegDataSample]
OptSampleList = Optional[SampleList]

# Type hint of Tensor
TensorDict = Dict[str, torch.Tensor]
TensorList = Sequence[torch.Tensor]

ForwardResults = Union[Dict[str, torch.Tensor], List[SegDataSample],
                       Tuple[torch.Tensor], torch.Tensor]

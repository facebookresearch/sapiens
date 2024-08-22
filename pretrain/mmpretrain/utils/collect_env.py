# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import mmcv
from mmengine.utils import get_git_hash
from mmengine.utils.dl_utils import collect_env as collect_base_env

import mmpretrain


def collect_env(with_torch_comiling_info=False):
    """Collect the information of the running environments."""
    env_info = collect_base_env()
    env_info['MMCV'] = mmcv.__version__
    if not with_torch_comiling_info:
        env_info.pop('PyTorch compiling details')
    env_info['MMPreTrain'] = mmpretrain.__version__ + '+' + get_git_hash()[:7]
    return env_info

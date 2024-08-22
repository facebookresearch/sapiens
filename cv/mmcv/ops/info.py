# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch

if torch.__version__ == 'parrots':
    import parrots

    def get_compiler_version():
        return 'GCC ' + parrots.version.compiler

    def get_compiling_cuda_version():
        return parrots.version.cuda
else:
    from ..utils import ext_loader
    ext_module = ext_loader.load_ext(
        '_ext', ['get_compiler_version', 'get_compiling_cuda_version'])

    def get_compiler_version():
        return ext_module.get_compiler_version()

    def get_compiling_cuda_version():
        return ext_module.get_compiling_cuda_version()

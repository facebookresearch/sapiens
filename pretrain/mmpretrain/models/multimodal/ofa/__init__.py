# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from .ofa import OFA
from .ofa_modules import OFADecoder, OFAEncoder, OFAEncoderDecoder

__all__ = ['OFAEncoderDecoder', 'OFA', 'OFAEncoder', 'OFADecoder']

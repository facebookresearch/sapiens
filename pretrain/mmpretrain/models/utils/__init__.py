# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from mmpretrain.utils.dependency import WITH_MULTIMODAL
from .attention import (BEiTAttention, ChannelMultiheadAttention,
                        CrossMultiheadAttention, LeAttention,
                        MultiheadAttention, PromptMultiheadAttention,
                        ShiftWindowMSA, WindowMSA, WindowMSAV2)
from .batch_augments import CutMix, Mixup, RandomBatchAugment, ResizeMix
from .batch_shuffle import batch_shuffle_ddp, batch_unshuffle_ddp
from .channel_shuffle import channel_shuffle
from .clip_generator_helper import QuickGELU, build_clip_model
from .data_preprocessor import (ClsDataPreprocessor,
                                MultiModalDataPreprocessor,
                                SelfSupDataPreprocessor,
                                TwoNormDataPreprocessor, VideoDataPreprocessor)
from .ema import CosineEMA
from .embed import (HybridEmbed, PatchEmbed, PatchMerging, resize_pos_embed,
                    resize_relative_position_bias_table)
from .helpers import is_tracing, to_2tuple, to_3tuple, to_4tuple, to_ntuple
from .inverted_residual import InvertedResidual
from .layer_scale import LayerScale
from .make_divisible import make_divisible
from .norm import GRN, LayerNorm2d, build_norm_layer
from .position_encoding import (ConditionalPositionEncoding,
                                PositionEncodingFourier, RotaryEmbeddingFast,
                                build_2d_sincos_position_embedding)
from .res_layer_extra_norm import ResLayerExtraNorm
from .se_layer import SELayer
from .sparse_modules import (SparseAvgPooling, SparseBatchNorm2d, SparseConv2d,
                             SparseHelper, SparseLayerNorm2D, SparseMaxPooling,
                             SparseSyncBatchNorm2d)
from .swiglu_ffn import SwiGLUFFN, SwiGLUFFNFused
from .vector_quantizer import NormEMAVectorQuantizer

__all__ = [
    'channel_shuffle',
    'make_divisible',
    'InvertedResidual',
    'SELayer',
    'to_ntuple',
    'to_2tuple',
    'to_3tuple',
    'to_4tuple',
    'PatchEmbed',
    'PatchMerging',
    'HybridEmbed',
    'RandomBatchAugment',
    'ShiftWindowMSA',
    'is_tracing',
    'MultiheadAttention',
    'ConditionalPositionEncoding',
    'resize_pos_embed',
    'resize_relative_position_bias_table',
    'ClsDataPreprocessor',
    'Mixup',
    'CutMix',
    'ResizeMix',
    'BEiTAttention',
    'LayerScale',
    'WindowMSA',
    'WindowMSAV2',
    'ChannelMultiheadAttention',
    'PositionEncodingFourier',
    'LeAttention',
    'GRN',
    'LayerNorm2d',
    'build_norm_layer',
    'CrossMultiheadAttention',
    'build_2d_sincos_position_embedding',
    'PromptMultiheadAttention',
    'NormEMAVectorQuantizer',
    'build_clip_model',
    'batch_shuffle_ddp',
    'batch_unshuffle_ddp',
    'SelfSupDataPreprocessor',
    'TwoNormDataPreprocessor',
    'VideoDataPreprocessor',
    'CosineEMA',
    'ResLayerExtraNorm',
    'MultiModalDataPreprocessor',
    'QuickGELU',
    'SwiGLUFFN',
    'SwiGLUFFNFused',
    'RotaryEmbeddingFast',
    'SparseAvgPooling',
    'SparseConv2d',
    'SparseHelper',
    'SparseMaxPooling',
    'SparseBatchNorm2d',
    'SparseLayerNorm2D',
    'SparseSyncBatchNorm2d',
]

if WITH_MULTIMODAL:
    from .huggingface import (no_load_hf_pretrained_model, register_hf_model,
                              register_hf_tokenizer)
    from .tokenizer import (Blip2Tokenizer, BlipTokenizer, FullTokenizer,
                            OFATokenizer)

    __all__.extend([
        'BlipTokenizer', 'OFATokenizer', 'Blip2Tokenizer', 'register_hf_model',
        'register_hf_tokenizer', 'no_load_hf_pretrained_model', 'FullTokenizer'
    ])

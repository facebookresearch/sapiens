# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# This is a BETA new format config file, and the usage may change recently.
from mmpretrain.models import (MAE, MAEPretrainDecoder, MAEPretrainHead,
                               MAEViT, PixelReconstructionLoss)

# model settings
model = dict(
    type=MAE,
    backbone=dict(type=MAEViT, arch='b', patch_size=16, mask_ratio=0.75),
    neck=dict(
        type=MAEPretrainDecoder,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4.,
    ),
    head=dict(
        type=MAEPretrainHead,
        norm_pix=True,
        patch_size=16,
        loss=dict(type=PixelReconstructionLoss, criterion='L2')),
    init_cfg=[
        dict(type='Xavier', layer='Linear', distribution='uniform'),
        dict(type='Constant', layer='LayerNorm', val=1.0, bias=0.0)
    ])

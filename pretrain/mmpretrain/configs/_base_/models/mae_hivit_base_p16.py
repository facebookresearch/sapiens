# This is a BETA new format config file, and the usage may change recently.
from mmpretrain.models import (MAE, MAEHiViT, MAEPretrainDecoder,
                               MAEPretrainHead, PixelReconstructionLoss)

# model settings
model = dict(
    type=MAE,
    backbone=dict(type=MAEHiViT, patch_size=16, arch='base', mask_ratio=0.75),
    neck=dict(
        type=MAEPretrainDecoder,
        patch_size=16,
        in_chans=3,
        embed_dim=512,
        decoder_embed_dim=512,
        decoder_depth=6,
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
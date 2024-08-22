# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
from functools import partial
from typing import Optional, Sequence, Union

import torch
import torch.nn as nn
from mmengine.model import BaseModule
from mmengine.utils import digit_version

from ..utils import to_2tuple

# After pytorch v1.10.0, use torch.meshgrid without indexing
# will raise extra warning. For more details,
# refers to https://github.com/pytorch/pytorch/issues/50276
if digit_version(torch.__version__) >= digit_version('1.10.0'):
    torch_meshgrid = partial(torch.meshgrid, indexing='ij')
else:
    torch_meshgrid = torch.meshgrid


class ConditionalPositionEncoding(BaseModule):
    """The Conditional Position Encoding (CPE) module.

    The CPE is the implementation of 'Conditional Positional Encodings
    for Vision Transformers <https://arxiv.org/abs/2102.10882>'_.

    Args:
       in_channels (int): Number of input channels.
       embed_dims (int): The feature dimension. Default: 768.
       stride (int): Stride of conv layer. Default: 1.
    """

    def __init__(self, in_channels, embed_dims=768, stride=1, init_cfg=None):
        super(ConditionalPositionEncoding, self).__init__(init_cfg=init_cfg)
        self.proj = nn.Conv2d(
            in_channels,
            embed_dims,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=True,
            groups=embed_dims)
        self.stride = stride

    def forward(self, x, hw_shape):
        B, N, C = x.shape
        H, W = hw_shape
        feat_token = x
        # convert (B, N, C) to (B, C, H, W)
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W).contiguous()
        if self.stride == 1:
            x = self.proj(cnn_feat) + cnn_feat
        else:
            x = self.proj(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        return x


class PositionEncodingFourier(BaseModule):
    """The Position Encoding Fourier (PEF) module.

    The PEF is adopted from EdgeNeXt <https://arxiv.org/abs/2206.10589>'_.
    Args:
        in_channels (int): Number of input channels.
            Default: 32
        embed_dims (int): The feature dimension.
            Default: 768.
        temperature (int): Temperature.
            Default: 10000.
        dtype (torch.dtype): The data type.
            Default: torch.float32.
        init_cfg (dict): The config dict for initializing the module.
            Default: None.
    """

    def __init__(self,
                 in_channels=32,
                 embed_dims=768,
                 temperature=10000,
                 dtype=torch.float32,
                 init_cfg=None):
        super(PositionEncodingFourier, self).__init__(init_cfg=init_cfg)
        self.proj = nn.Conv2d(in_channels * 2, embed_dims, kernel_size=1)
        self.scale = 2 * math.pi
        self.in_channels = in_channels
        self.embed_dims = embed_dims
        self.dtype = dtype

        if digit_version(torch.__version__) < digit_version('1.8.0'):
            floor_div = torch.floor_divide
        else:
            floor_div = partial(torch.div, rounding_mode='floor')
        dim_t = torch.arange(in_channels, dtype=self.dtype)
        self.dim_t = temperature**(2 * floor_div(dim_t, 2) / in_channels)

    def forward(self, bhw_shape):
        B, H, W = bhw_shape
        mask = torch.zeros(B, H, W).bool().to(self.proj.weight.device)
        not_mask = ~mask
        eps = 1e-6
        y_embed = not_mask.cumsum(1, dtype=self.dtype)
        x_embed = not_mask.cumsum(2, dtype=self.dtype)
        y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
        x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = self.dim_t.to(mask.device)
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()),
            dim=4).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()),
            dim=4).flatten(3)

        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        pos = self.proj(pos)

        return pos


def build_2d_sincos_position_embedding(
        patches_resolution: Union[int, Sequence[int]],
        embed_dims: int,
        temperature: Optional[int] = 10000.,
        cls_token: Optional[bool] = False) -> torch.Tensor:
    """The function is to build position embedding for model to obtain the
    position information of the image patches.

    Args:
        patches_resolution (Union[int, Sequence[int]]): The resolution of each
            patch.
        embed_dims (int): The dimension of the embedding vector.
        temperature (int, optional): The temperature parameter. Defaults to
            10000.
        cls_token (bool, optional): Whether to concatenate class token.
            Defaults to False.

    Returns:
        torch.Tensor: The position embedding vector.
    """

    if isinstance(patches_resolution, int):
        patches_resolution = (patches_resolution, patches_resolution)

    h, w = patches_resolution
    grid_w = torch.arange(w, dtype=torch.float32)
    grid_h = torch.arange(h, dtype=torch.float32)
    grid_w, grid_h = torch_meshgrid(grid_w, grid_h)
    assert embed_dims % 4 == 0, \
        'Embed dimension must be divisible by 4.'
    pos_dim = embed_dims // 4

    omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
    omega = 1. / (temperature**omega)
    out_w = torch.einsum('m,d->md', [grid_w.flatten(), omega])
    out_h = torch.einsum('m,d->md', [grid_h.flatten(), omega])

    pos_emb = torch.cat(
        [
            torch.sin(out_w),
            torch.cos(out_w),
            torch.sin(out_h),
            torch.cos(out_h)
        ],
        dim=1,
    )[None, :, :]

    if cls_token:
        cls_token_pe = torch.zeros([1, 1, embed_dims], dtype=torch.float32)
        pos_emb = torch.cat([cls_token_pe, pos_emb], dim=1)

    return pos_emb


class RotaryEmbeddingFast(BaseModule):
    """Implements 2D rotary embedding (RoPE) for image tokens. Position
    encoding is implemented with sin and cos functions,

        .. math::
            Pos_{cos} = cos(\frac{t}{\theta^{\frac{2i}{d}}} \\
            Pos_{sin} = sin(\frac{t}{\theta^{\frac{2i}{d}}}
    Args:
        embed_dims (int): The feature dimension for each head.
        patch_resolution (int | tuple): The resolution of the
            image, in format (H, W).
        theta (float): The hyperparameter for position coding.
            Defaults to 10000.
        init_cfg (dict, optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 embed_dims,
                 patch_resolution,
                 theta=10000.,
                 init_cfg=None):
        super(RotaryEmbeddingFast, self).__init__(init_cfg=init_cfg)

        self.half_dim = embed_dims // 2
        self.patch_resolution = to_2tuple(patch_resolution)
        self.theta = theta

        freqs_cos, freqs_sin = self.compute_position_embedding()
        self.register_buffer('freqs_cos', freqs_cos)
        self.register_buffer('freqs_sin', freqs_sin)

    def compute_position_embedding(self):
        frequency = self.theta**(
            torch.arange(0, self.half_dim, 2).float() / self.half_dim)
        frequency = 1. / frequency

        h, w = self.patch_resolution
        th = torch.arange(h) / h * self.half_dim
        tw = torch.arange(w) / w * self.half_dim

        position_h = (th[:, None] @ frequency[None, :]).repeat(1, 2)
        position_w = (tw[:, None] @ frequency[None, :]).repeat(1, 2)

        height = position_h[:, None, :].expand(h, w, self.half_dim)
        width = position_w[None, :, :].expand(h, w, self.half_dim)
        position = torch.cat((height, width), dim=-1)

        freqs_cos = position.cos().view(-1, position.shape[-1])
        freqs_sin = position.sin().view(-1, position.shape[-1])
            
        return freqs_cos, freqs_sin

    def forward(self, x, patch_resolution, mask=None, ids_restore=None, ids_keep=None):
        # Check whether the patch resolution is the predefined size
        patch_resolution = to_2tuple(patch_resolution)
        
        if patch_resolution != self.patch_resolution:
            self.patch_resolution = patch_resolution
            freqs_cos, freqs_sin = self.compute_position_embedding()
            self.register_buffer('freqs_cos', freqs_cos.to(x.device))
            self.register_buffer('freqs_sin', freqs_sin.to(x.device))

        batch, num_heads, num_patches, dim = x.shape ## 8 x 24 x 576 x 64

        ##-------------dynamic----------------
        if ids_keep is None:
            freqs_cos = self.freqs_cos
            freqs_sin = self.freqs_sin
        else:
            ## add extra batch
            freqs_cos = self.freqs_cos.unsqueeze(dim=0).repeat(batch, 1, 1) ## B x 2304 x 64
            freqs_sin = self.freqs_sin.unsqueeze(dim=0).repeat(batch, 1, 1) ## B x 2304 x 64

            freqs_cos = torch.gather(freqs_cos, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, dim)) ## B x 576 x 64
            freqs_sin = torch.gather(freqs_sin, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, dim)) ## B x 576 x 64

            ## add extra heads
            freqs_cos = freqs_cos.unsqueeze(dim=1).repeat(1, num_heads, 1, 1) ## B x 24 x 576 x 64
            freqs_sin = freqs_sin.unsqueeze(dim=1).repeat(1, num_heads, 1, 1) ## B x 24 x 576 x 64
            
        ##----------------------------------
        inputs = x
        x = x.reshape(batch, num_heads, num_patches, -1, 2)
        x1, x2 = x.unbind(dim=-1)
        x = torch.stack((-x2, x1), dim=-1)
        x = x.reshape(batch, num_heads, num_patches, dim)

        return inputs * freqs_cos + x * freqs_sin


        

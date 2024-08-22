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
import torch.nn.functional as F
from mmcv.cnn.bricks import ConvModule, DropPath
from mmcv.cnn.bricks.transformer import FFN
from mmengine.model import BaseModule, Sequential
from mmengine.model.weight_init import trunc_normal_
from mmengine.utils import digit_version

from mmpretrain.registry import MODELS
from ..utils import build_norm_layer, to_2tuple
from .base_backbone import BaseBackbone

if digit_version(torch.__version__) < digit_version('1.8.0'):
    floor_div = torch.floor_divide
else:
    floor_div = partial(torch.div, rounding_mode='floor')


class ClassAttntion(BaseModule):
    """Class Attention Module.

    A PyTorch implementation of Class Attention Module introduced by:
    `Going deeper with Image Transformers <https://arxiv.org/abs/2103.17239>`_

    taken from
    https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    with slight modifications to do CA

    Args:
        dim (int): The feature dimension.
        num_heads (int): Parallel attention heads. Defaults to 8.
        qkv_bias (bool): enable bias for qkv if True. Defaults to False.
        attn_drop (float): The drop out rate for attention output weights.
            Defaults to 0.
        proj_drop (float): The drop out rate for linear output weights.
            Defaults to 0.
        init_cfg (dict | list[dict], optional): Initialization config dict.
            Defaults to None.
    """  # noqa: E501

    def __init__(self,
                 dim: int,
                 num_heads: int = 8,
                 qkv_bias: bool = False,
                 attn_drop: float = 0.,
                 proj_drop: float = 0.,
                 init_cfg=None):

        super(ClassAttntion, self).__init__(init_cfg=init_cfg)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        # We only need to calculate query of cls token.
        q = self.q(x[:, 0]).unsqueeze(1).reshape(B, 1, self.num_heads,
                                                 C // self.num_heads).permute(
                                                     0, 2, 1, 3)
        k = self.k(x).reshape(B, N, self.num_heads,
                              C // self.num_heads).permute(0, 2, 1, 3)

        q = q * self.scale
        v = self.v(x).reshape(B, N, self.num_heads,
                              C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x_cls = (attn @ v).transpose(1, 2).reshape(B, 1, C)
        x_cls = self.proj(x_cls)
        x_cls = self.proj_drop(x_cls)

        return x_cls


class PositionalEncodingFourier(BaseModule):
    """Positional Encoding using a fourier kernel.

    A PyTorch implementation of Positional Encoding relying on
    a fourier kernel introduced by:
    `Attention is all you Need <https://arxiv.org/abs/1706.03762>`_

    Based on the `official XCiT code
    <https://github.com/facebookresearch/xcit/blob/master/xcit.py>`_

    Args:
        hidden_dim (int): The hidden feature dimension. Defaults to 32.
        dim (int): The output feature dimension. Defaults to 768.
        temperature (int): A control variable for position encoding.
            Defaults to 10000.
        init_cfg (dict | list[dict], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 hidden_dim: int = 32,
                 dim: int = 768,
                 temperature: int = 10000,
                 init_cfg=None):
        super(PositionalEncodingFourier, self).__init__(init_cfg=init_cfg)

        self.token_projection = ConvModule(
            in_channels=hidden_dim * 2,
            out_channels=dim,
            kernel_size=1,
            conv_cfg=None,
            norm_cfg=None,
            act_cfg=None)
        self.scale = 2 * math.pi
        self.temperature = temperature
        self.hidden_dim = hidden_dim
        self.dim = dim
        self.eps = 1e-6

    def forward(self, B: int, H: int, W: int):
        device = self.token_projection.conv.weight.device
        y_embed = torch.arange(
            1, H + 1, device=device).unsqueeze(1).repeat(1, 1, W).float()
        x_embed = torch.arange(1, W + 1, device=device).repeat(1, H, 1).float()
        y_embed = y_embed / (y_embed[:, -1:, :] + self.eps) * self.scale
        x_embed = x_embed / (x_embed[:, :, -1:] + self.eps) * self.scale

        dim_t = torch.arange(self.hidden_dim, device=device).float()
        dim_t = floor_div(dim_t, 2)
        dim_t = self.temperature**(2 * dim_t / self.hidden_dim)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            [pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()],
            dim=4).flatten(3)
        pos_y = torch.stack(
            [pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()],
            dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        pos = self.token_projection(pos)
        return pos.repeat(B, 1, 1, 1)  # (B, C, H, W)


class ConvPatchEmbed(BaseModule):
    """Patch Embedding using multiple convolution layers.

    Args:
        img_size (int, tuple): input image size.
            Defaults to 224, means the size is 224*224.
        patch_size (int): The patch size in conv patch embedding.
            Defaults to 16.
        in_channels (int): The input channels of this module.
            Defaults to 3.
        embed_dims (int): The feature dimension
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to ``dict(type='BN')``.
        act_cfg (dict): Config dict for activation layer.
            Defaults to ``dict(type='GELU')``.
        init_cfg (dict | list[dict], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 img_size: Union[int, tuple] = 224,
                 patch_size: int = 16,
                 in_channels: int = 3,
                 embed_dims: int = 768,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='GELU'),
                 init_cfg=None):
        super(ConvPatchEmbed, self).__init__(init_cfg=init_cfg)
        img_size = to_2tuple(img_size)
        num_patches = (img_size[1] // patch_size) * (img_size[0] // patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        conv = partial(
            ConvModule,
            kernel_size=3,
            stride=2,
            padding=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )

        layer = []
        if patch_size == 16:
            layer.append(
                conv(in_channels=in_channels, out_channels=embed_dims // 8))
            layer.append(
                conv(
                    in_channels=embed_dims // 8, out_channels=embed_dims // 4))
        elif patch_size == 8:
            layer.append(
                conv(in_channels=in_channels, out_channels=embed_dims // 4))
        else:
            raise ValueError('For patch embedding, the patch size must be 16 '
                             f'or 8, but get patch size {self.patch_size}.')

        layer.append(
            conv(in_channels=embed_dims // 4, out_channels=embed_dims // 2))
        layer.append(
            conv(
                in_channels=embed_dims // 2,
                out_channels=embed_dims,
                act_cfg=None,
            ))

        self.proj = Sequential(*layer)

    def forward(self, x: torch.Tensor):
        x = self.proj(x)
        Hp, Wp = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)  # (B, N, C)
        return x, (Hp, Wp)


class ClassAttentionBlock(BaseModule):
    """Transformer block using Class Attention.

    Args:
        dim (int): The feature dimension.
        num_heads (int): Parallel attention heads.
        mlp_ratio (float): The hidden dimension ratio for FFN.
            Defaults to 4.
        qkv_bias (bool): enable bias for qkv if True. Defaults to False.
        drop (float): Probability of an element to be zeroed
            after the feed forward layer. Defaults to 0.
        attn_drop (float): The drop out rate for attention output weights.
            Defaults to 0.
        drop_path (float): Stochastic depth rate. Defaults to 0.
        layer_scale_init_value (float): The initial value for layer scale.
            Defaults to 1.
        tokens_norm (bool): Whether to normalize all tokens or just the
            cls_token in the CA. Defaults to False.
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to ``dict(type='LN', eps=1e-6)``.
        act_cfg (dict): Config dict for activation layer.
            Defaults to ``dict(type='GELU')``.
        init_cfg (dict | list[dict], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 dim: int,
                 num_heads: int,
                 mlp_ratio: float = 4.,
                 qkv_bias: bool = False,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 layer_scale_init_value=1.,
                 tokens_norm=False,
                 norm_cfg=dict(type='LN', eps=1e-6),
                 act_cfg=dict(type='GELU'),
                 init_cfg=None):

        super(ClassAttentionBlock, self).__init__(init_cfg=init_cfg)

        self.norm1 = build_norm_layer(norm_cfg, dim)

        self.attn = ClassAttntion(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = build_norm_layer(norm_cfg, dim)

        self.ffn = FFN(
            embed_dims=dim,
            feedforward_channels=int(dim * mlp_ratio),
            act_cfg=act_cfg,
            ffn_drop=drop,
        )

        if layer_scale_init_value > 0:
            self.gamma1 = nn.Parameter(layer_scale_init_value *
                                       torch.ones(dim))
            self.gamma2 = nn.Parameter(layer_scale_init_value *
                                       torch.ones(dim))
        else:
            self.gamma1, self.gamma2 = 1.0, 1.0

        # See https://github.com/rwightman/pytorch-image-models/pull/747#issuecomment-877795721  # noqa: E501
        self.tokens_norm = tokens_norm

    def forward(self, x):
        x_norm1 = self.norm1(x)
        x_attn = torch.cat([self.attn(x_norm1), x_norm1[:, 1:]], dim=1)
        x = x + self.drop_path(self.gamma1 * x_attn)
        if self.tokens_norm:
            x = self.norm2(x)
        else:
            x = torch.cat([self.norm2(x[:, 0:1]), x[:, 1:]], dim=1)
        x_res = x
        cls_token = x[:, 0:1]
        cls_token = self.gamma2 * self.ffn(cls_token, identity=0)
        x = torch.cat([cls_token, x[:, 1:]], dim=1)
        x = x_res + self.drop_path(x)
        return x


class LPI(BaseModule):
    """Local Patch Interaction module.

    A PyTorch implementation of Local Patch Interaction module
    as in XCiT introduced by `XCiT: Cross-Covariance Image Transformers
    <https://arxiv.org/abs/2106.096819>`_

    Local Patch Interaction module that allows explicit communication between
    tokens in 3x3 windows to augment the implicit communication performed by
    the block diagonal scatter attention. Implemented using 2 layers of
    separable 3x3 convolutions with GeLU and BatchNorm2d

    Args:
        in_features (int): The input channels.
        out_features (int, optional): The output channels. Defaults to None.
        kernel_size (int): The kernel_size in ConvModule. Defaults to 3.
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to ``dict(type='BN')``.
        act_cfg (dict): Config dict for activation layer.
            Defaults to ``dict(type='GELU')``.
        init_cfg (dict | list[dict], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 in_features: int,
                 out_features: Optional[int] = None,
                 kernel_size: int = 3,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='GELU'),
                 init_cfg=None):
        super(LPI, self).__init__(init_cfg=init_cfg)

        out_features = out_features or in_features
        padding = kernel_size // 2

        self.conv1 = ConvModule(
            in_channels=in_features,
            out_channels=in_features,
            kernel_size=kernel_size,
            padding=padding,
            groups=in_features,
            bias=True,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            order=('conv', 'act', 'norm'))

        self.conv2 = ConvModule(
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=kernel_size,
            padding=padding,
            groups=out_features,
            norm_cfg=None,
            act_cfg=None)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, N, C = x.shape
        x = x.permute(0, 2, 1).reshape(B, C, H, W)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.reshape(B, C, N).permute(0, 2, 1)
        return x


class XCA(BaseModule):
    r"""Cross-Covariance Attention module.

    A PyTorch implementation of Cross-Covariance Attention module
    as in XCiT introduced by `XCiT: Cross-Covariance Image Transformers
    <https://arxiv.org/abs/2106.096819>`_

    In Cross-Covariance Attention (XCA), the channels are updated using a
    weighted sum. The weights are obtained from the (softmax normalized)
    Cross-covariance matrix :math:`(Q^T \cdot K \in d_h \times d_h)`

    Args:
        dim (int): The feature dimension.
        num_heads (int): Parallel attention heads. Defaults to 8.
        qkv_bias (bool): enable bias for qkv if True. Defaults to False.
        attn_drop (float): The drop out rate for attention output weights.
            Defaults to 0.
        proj_drop (float): The drop out rate for linear output weights.
            Defaults to 0.
        init_cfg (dict | list[dict], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 dim: int,
                 num_heads: int = 8,
                 qkv_bias: bool = False,
                 attn_drop: float = 0.,
                 proj_drop: float = 0.,
                 init_cfg=None):
        super(XCA, self).__init__(init_cfg=init_cfg)
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        # (qkv, B, num_heads, channels per head, N)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads,
                                  C // self.num_heads).permute(2, 0, 3, 4, 1)
        q, k, v = qkv.unbind(0)

        # Paper section 3.2 l2-Normalization and temperature scaling
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # (B, num_heads, C', N) -> (B, N, num_heads, C') -> (B, N C)
        x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class XCABlock(BaseModule):
    """Transformer block using XCA.

    Args:
        dim (int): The feature dimension.
        num_heads (int): Parallel attention heads.
        mlp_ratio (float): The hidden dimension ratio for FFNs.
            Defaults to 4.
        qkv_bias (bool): enable bias for qkv if True. Defaults to False.
        drop (float): Probability of an element to be zeroed
            after the feed forward layer. Defaults to 0.
        attn_drop (float): The drop out rate for attention output weights.
            Defaults to 0.
        drop_path (float): Stochastic depth rate. Defaults to 0.
        layer_scale_init_value (float): The initial value for layer scale.
            Defaults to 1.
        bn_norm_cfg (dict): Config dict for batchnorm in LPI and
            ConvPatchEmbed. Defaults to ``dict(type='BN')``.
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to ``dict(type='LN', eps=1e-6)``.
        act_cfg (dict): Config dict for activation layer.
            Defaults to ``dict(type='GELU')``.
        init_cfg (dict | list[dict], optional): Initialization config dict.
    """

    def __init__(self,
                 dim: int,
                 num_heads: int,
                 mlp_ratio: float = 4.,
                 qkv_bias: bool = False,
                 drop: float = 0.,
                 attn_drop: float = 0.,
                 drop_path: float = 0.,
                 layer_scale_init_value: float = 1.,
                 bn_norm_cfg=dict(type='BN'),
                 norm_cfg=dict(type='LN', eps=1e-6),
                 act_cfg=dict(type='GELU'),
                 init_cfg=None):
        super(XCABlock, self).__init__(init_cfg=init_cfg)

        self.norm1 = build_norm_layer(norm_cfg, dim)
        self.attn = XCA(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()

        self.norm3 = build_norm_layer(norm_cfg, dim)
        self.local_mp = LPI(
            in_features=dim,
            norm_cfg=bn_norm_cfg,
            act_cfg=act_cfg,
        )

        self.norm2 = build_norm_layer(norm_cfg, dim)
        self.ffn = FFN(
            embed_dims=dim,
            feedforward_channels=int(dim * mlp_ratio),
            act_cfg=act_cfg,
            ffn_drop=drop,
        )

        self.gamma1 = nn.Parameter(layer_scale_init_value * torch.ones(dim))
        self.gamma3 = nn.Parameter(layer_scale_init_value * torch.ones(dim))
        self.gamma2 = nn.Parameter(layer_scale_init_value * torch.ones(dim))

    def forward(self, x, H: int, W: int):
        x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
        # NOTE official code has 3 then 2, so keeping it the same to be
        # consistent with loaded weights See
        # https://github.com/rwightman/pytorch-image-models/pull/747#issuecomment-877795721  # noqa: E501
        x = x + self.drop_path(
            self.gamma3 * self.local_mp(self.norm3(x), H, W))
        x = x + self.drop_path(
            self.gamma2 * self.ffn(self.norm2(x), identity=0))
        return x


@MODELS.register_module()
class XCiT(BaseBackbone):
    """XCiT backbone.

    A PyTorch implementation of XCiT backbone introduced by:
    `XCiT: Cross-Covariance Image Transformers
    <https://arxiv.org/abs/2106.096819>`_

    Args:
        img_size (int, tuple): Input image size. Defaults to 224.
        patch_size (int): Patch size. Defaults to 16.
        in_channels (int): Number of input channels. Defaults to 3.
        embed_dims (int): Embedding dimension. Defaults to 768.
        depth (int): depth of vision transformer. Defaults to 12.
        cls_attn_layers (int): Depth of Class attention layers.
            Defaults to 2.
        num_heads (int): Number of attention heads. Defaults to 12.
        mlp_ratio (int): Ratio of mlp hidden dim to embedding dim.
            Defaults to 4.
        qkv_bias (bool): enable bias for qkv if True. Defaults to True.
        drop_rate (float): Probability of an element to be zeroed
            after the feed forward layer. Defaults to 0.
        attn_drop_rate (float): The drop out rate for attention output weights.
            Defaults to 0.
        drop_path_rate (float): Stochastic depth rate. Defaults to 0.
        use_pos_embed (bool): Whether to use positional encoding.
            Defaults to True.
        layer_scale_init_value (float): The initial value for layer scale.
            Defaults to 1.
        tokens_norm (bool): Whether to normalize all tokens or just the
            cls_token in the CA. Defaults to False.
        out_indices (Sequence[int]): Output from which layers.
            Defaults to (-1, ).
        frozen_stages (int): Layers to be frozen (all param fixed), and 0
            means to freeze the stem stage. Defaults to -1, which means
            not freeze any parameters.
        bn_norm_cfg (dict): Config dict for the batch norm layers in LPI and
            ConvPatchEmbed. Defaults to ``dict(type='BN')``.
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to ``dict(type='LN', eps=1e-6)``.
        act_cfg (dict): Config dict for activation layer.
            Defaults to ``dict(type='GELU')``.
        init_cfg (dict | list[dict], optional): Initialization config dict.
    """

    def __init__(self,
                 img_size: Union[int, tuple] = 224,
                 patch_size: int = 16,
                 in_channels: int = 3,
                 embed_dims: int = 768,
                 depth: int = 12,
                 cls_attn_layers: int = 2,
                 num_heads: int = 12,
                 mlp_ratio: float = 4.,
                 qkv_bias: bool = True,
                 drop_rate: float = 0.,
                 attn_drop_rate: float = 0.,
                 drop_path_rate: float = 0.,
                 use_pos_embed: bool = True,
                 layer_scale_init_value: float = 1.,
                 tokens_norm: bool = False,
                 out_type: str = 'cls_token',
                 out_indices: Sequence[int] = (-1, ),
                 final_norm: bool = True,
                 frozen_stages: int = -1,
                 bn_norm_cfg=dict(type='BN'),
                 norm_cfg=dict(type='LN', eps=1e-6),
                 act_cfg=dict(type='GELU'),
                 init_cfg=dict(type='TruncNormal', layer='Linear')):
        super(XCiT, self).__init__(init_cfg=init_cfg)

        img_size = to_2tuple(img_size)
        if (img_size[0] % patch_size != 0) or (img_size[1] % patch_size != 0):
            raise ValueError(f'`patch_size` ({patch_size}) should divide '
                             f'the image shape ({img_size}) evenly.')

        self.embed_dims = embed_dims

        assert out_type in ('raw', 'featmap', 'avg_featmap', 'cls_token')
        self.out_type = out_type

        self.patch_embed = ConvPatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dims=embed_dims,
            norm_cfg=bn_norm_cfg,
            act_cfg=act_cfg,
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dims))
        self.use_pos_embed = use_pos_embed
        if use_pos_embed:
            self.pos_embed = PositionalEncodingFourier(dim=embed_dims)
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.xca_layers = nn.ModuleList()
        self.ca_layers = nn.ModuleList()
        self.num_layers = depth + cls_attn_layers

        for _ in range(depth):
            self.xca_layers.append(
                XCABlock(
                    dim=embed_dims,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=drop_path_rate,
                    bn_norm_cfg=bn_norm_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    layer_scale_init_value=layer_scale_init_value,
                ))

        for _ in range(cls_attn_layers):
            self.ca_layers.append(
                ClassAttentionBlock(
                    dim=embed_dims,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    act_cfg=act_cfg,
                    norm_cfg=norm_cfg,
                    layer_scale_init_value=layer_scale_init_value,
                    tokens_norm=tokens_norm,
                ))

        if final_norm:
            self.norm = build_norm_layer(norm_cfg, embed_dims)

        # Transform out_indices
        if isinstance(out_indices, int):
            out_indices = [out_indices]
        assert isinstance(out_indices, Sequence), \
            f'"out_indices" must by a sequence or int, ' \
            f'get {type(out_indices)} instead.'
        out_indices = list(out_indices)
        for i, index in enumerate(out_indices):
            if index < 0:
                out_indices[i] = self.num_layers + index
            assert 0 <= out_indices[i] <= self.num_layers, \
                f'Invalid out_indices {index}.'
        self.out_indices = out_indices

        if frozen_stages > self.num_layers + 1:
            raise ValueError('frozen_stages must be less than '
                             f'{self.num_layers} but get {frozen_stages}')
        self.frozen_stages = frozen_stages

    def init_weights(self):
        super().init_weights()

        if self.init_cfg is not None and self.init_cfg['type'] == 'Pretrained':
            return

        trunc_normal_(self.cls_token, std=.02)

    def _freeze_stages(self):
        if self.frozen_stages < 0:
            return

        # freeze position embedding
        if self.use_pos_embed:
            self.pos_embed.eval()
            for param in self.pos_embed.parameters():
                param.requires_grad = False
        # freeze patch embedding
        self.patch_embed.eval()
        for param in self.patch_embed.parameters():
            param.requires_grad = False
        # set dropout to eval model
        self.pos_drop.eval()
        # freeze cls_token, only use in self.Clslayers
        if self.frozen_stages > len(self.xca_layers):
            self.cls_token.requires_grad = False
        # freeze layers
        for i in range(1, self.frozen_stages):
            if i <= len(self.xca_layers):
                m = self.xca_layers[i - 1]
            else:
                m = self.ca_layers[i - len(self.xca_layers) - 1]
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

        # freeze the last layer norm if all_stages are frozen
        if self.frozen_stages == len(self.xca_layers) + len(self.ca_layers):
            self.norm.eval()
            for param in self.norm.parameters():
                param.requires_grad = False

    def forward(self, x):
        outs = []
        B = x.shape[0]
        # x is (B, N, C). (Hp, Hw) is the patch resolution
        x, (Hp, Wp) = self.patch_embed(x)

        if self.use_pos_embed:
            # (B, C, Hp, Wp) -> (B, C, N) -> (B, N, C)
            pos_encoding = self.pos_embed(B, Hp, Wp)
            x = x + pos_encoding.reshape(B, -1, x.size(1)).permute(0, 2, 1)
        x = self.pos_drop(x)

        for i, layer in enumerate(self.xca_layers):
            x = layer(x, Hp, Wp)
            if i in self.out_indices:
                outs.append(self._format_output(x, (Hp, Wp), False))

        x = torch.cat((self.cls_token.expand(B, -1, -1), x), dim=1)

        for i, layer in enumerate(self.ca_layers):
            x = layer(x)
            if i == len(self.ca_layers) - 1:
                x = self.norm(x)
            if i + len(self.xca_layers) in self.out_indices:
                outs.append(self._format_output(x, (Hp, Wp), True))

        return tuple(outs)

    def _format_output(self, x, hw, with_cls_token: bool):
        if self.out_type == 'raw':
            return x
        if self.out_type == 'cls_token':
            if not with_cls_token:
                raise ValueError(
                    'Cannot output cls_token since there is no cls_token.')
            return x[:, 0]

        patch_token = x[:, 1:] if with_cls_token else x
        if self.out_type == 'featmap':
            B = x.size(0)
            # (B, N, C) -> (B, H, W, C) -> (B, C, H, W)
            return patch_token.reshape(B, *hw, -1).permute(0, 3, 1, 2)
        if self.out_type == 'avg_featmap':
            return patch_token.mean(dim=1)

    def train(self, mode=True):
        super().train(mode)
        self._freeze_stages()

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from mmcv.cnn import Conv2d, build_activation_layer, build_norm_layer
from mmcv.cnn.bricks import DropPath
from mmcv.cnn.bricks.transformer import PatchEmbed
from mmengine.model import BaseModule, ModuleList
from mmengine.utils.dl_utils.parrots_wrapper import _BatchNorm

from mmpretrain.registry import MODELS
from .base_backbone import BaseBackbone


class MixFFN(BaseModule):
    """An implementation of MixFFN of VAN. Refer to
    mmdetection/mmdet/models/backbones/pvt.py.

    The differences between MixFFN & FFN:
        1. Use 1X1 Conv to replace Linear layer.
        2. Introduce 3X3 Depth-wise Conv to encode positional information.

    Args:
        embed_dims (int): The feature dimension. Same as
            `MultiheadAttention`.
        feedforward_channels (int): The hidden dimension of FFNs.
        act_cfg (dict, optional): The activation config for FFNs.
            Default: dict(type='GELU').
        ffn_drop (float, optional): Probability of an element to be
            zeroed in FFN. Default 0.0.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 embed_dims,
                 feedforward_channels,
                 act_cfg=dict(type='GELU'),
                 ffn_drop=0.,
                 init_cfg=None):
        super(MixFFN, self).__init__(init_cfg=init_cfg)

        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.act_cfg = act_cfg

        self.fc1 = Conv2d(
            in_channels=embed_dims,
            out_channels=feedforward_channels,
            kernel_size=1)
        self.dwconv = Conv2d(
            in_channels=feedforward_channels,
            out_channels=feedforward_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
            groups=feedforward_channels)
        self.act = build_activation_layer(act_cfg)
        self.fc2 = Conv2d(
            in_channels=feedforward_channels,
            out_channels=embed_dims,
            kernel_size=1)
        self.drop = nn.Dropout(ffn_drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class LKA(BaseModule):
    """Large Kernel Attention(LKA) of VAN.

    .. code:: text
            DW_conv (depth-wise convolution)
                            |
                            |
        DW_D_conv (depth-wise dilation convolution)
                            |
                            |
        Transition Convolution (1×1 convolution)

    Args:
        embed_dims (int): Number of input channels.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    """

    def __init__(self, embed_dims, init_cfg=None):
        super(LKA, self).__init__(init_cfg=init_cfg)

        # a spatial local convolution (depth-wise convolution)
        self.DW_conv = Conv2d(
            in_channels=embed_dims,
            out_channels=embed_dims,
            kernel_size=5,
            padding=2,
            groups=embed_dims)

        # a spatial long-range convolution (depth-wise dilation convolution)
        self.DW_D_conv = Conv2d(
            in_channels=embed_dims,
            out_channels=embed_dims,
            kernel_size=7,
            stride=1,
            padding=9,
            groups=embed_dims,
            dilation=3)

        self.conv1 = Conv2d(
            in_channels=embed_dims, out_channels=embed_dims, kernel_size=1)

    def forward(self, x):
        u = x.clone()
        attn = self.DW_conv(x)
        attn = self.DW_D_conv(attn)
        attn = self.conv1(attn)

        return u * attn


class SpatialAttention(BaseModule):
    """Basic attention module in VANBloack.

    Args:
        embed_dims (int): Number of input channels.
        act_cfg (dict, optional): The activation config for FFNs.
            Default: dict(type='GELU').
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    """

    def __init__(self, embed_dims, act_cfg=dict(type='GELU'), init_cfg=None):
        super(SpatialAttention, self).__init__(init_cfg=init_cfg)

        self.proj_1 = Conv2d(
            in_channels=embed_dims, out_channels=embed_dims, kernel_size=1)
        self.activation = build_activation_layer(act_cfg)
        self.spatial_gating_unit = LKA(embed_dims)
        self.proj_2 = Conv2d(
            in_channels=embed_dims, out_channels=embed_dims, kernel_size=1)

    def forward(self, x):
        shorcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shorcut
        return x


class VANBlock(BaseModule):
    """A block of VAN.

    Args:
        embed_dims (int): Number of input channels.
        ffn_ratio (float): The expansion ratio of feedforward network hidden
            layer channels. Defaults to 4.
        drop_rate (float): Dropout rate after embedding. Defaults to 0.
        drop_path_rate (float): Stochastic depth rate. Defaults to 0.1.
        act_cfg (dict, optional): The activation config for FFNs.
            Default: dict(type='GELU').
        layer_scale_init_value (float): Init value for Layer Scale.
            Defaults to 1e-2.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 embed_dims,
                 ffn_ratio=4.,
                 drop_rate=0.,
                 drop_path_rate=0.,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='BN', eps=1e-5),
                 layer_scale_init_value=1e-2,
                 init_cfg=None):
        super(VANBlock, self).__init__(init_cfg=init_cfg)
        self.out_channels = embed_dims

        self.norm1 = build_norm_layer(norm_cfg, embed_dims)[1]
        self.attn = SpatialAttention(embed_dims, act_cfg=act_cfg)
        self.drop_path = DropPath(
            drop_path_rate) if drop_path_rate > 0. else nn.Identity()

        self.norm2 = build_norm_layer(norm_cfg, embed_dims)[1]
        mlp_hidden_dim = int(embed_dims * ffn_ratio)
        self.mlp = MixFFN(
            embed_dims=embed_dims,
            feedforward_channels=mlp_hidden_dim,
            act_cfg=act_cfg,
            ffn_drop=drop_rate)
        self.layer_scale_1 = nn.Parameter(
            layer_scale_init_value * torch.ones((embed_dims)),
            requires_grad=True) if layer_scale_init_value > 0 else None
        self.layer_scale_2 = nn.Parameter(
            layer_scale_init_value * torch.ones((embed_dims)),
            requires_grad=True) if layer_scale_init_value > 0 else None

    def forward(self, x):
        identity = x
        x = self.norm1(x)
        x = self.attn(x)
        if self.layer_scale_1 is not None:
            x = self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * x
        x = identity + self.drop_path(x)

        identity = x
        x = self.norm2(x)
        x = self.mlp(x)
        if self.layer_scale_2 is not None:
            x = self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * x
        x = identity + self.drop_path(x)

        return x


class VANPatchEmbed(PatchEmbed):
    """Image to Patch Embedding of VAN.

    The differences between VANPatchEmbed & PatchEmbed:
        1. Use BN.
        2. Do not use 'flatten' and 'transpose'.
    """

    def __init__(self, *args, norm_cfg=dict(type='BN'), **kwargs):
        super(VANPatchEmbed, self).__init__(*args, norm_cfg=norm_cfg, **kwargs)

    def forward(self, x):
        """
        Args:
            x (Tensor): Has shape (B, C, H, W). In most case, C is 3.
        Returns:
            tuple: Contains merged results and its spatial shape.
            - x (Tensor): Has shape (B, out_h * out_w, embed_dims)
            - out_size (tuple[int]): Spatial shape of x, arrange as
              (out_h, out_w).
        """

        if self.adaptive_padding:
            x = self.adaptive_padding(x)

        x = self.projection(x)
        out_size = (x.shape[2], x.shape[3])
        if self.norm is not None:
            x = self.norm(x)
        return x, out_size


@MODELS.register_module()
class VAN(BaseBackbone):
    """Visual Attention Network.

    A PyTorch implement of : `Visual Attention Network
    <https://arxiv.org/pdf/2202.09741v2.pdf>`_

    Inspiration from
    https://github.com/Visual-Attention-Network/VAN-Classification

    Args:
        arch (str | dict): Visual Attention Network architecture.
            If use string, choose from 'tiny', 'small', 'base' and 'large'.
            If use dict, it should have below keys:

            - **embed_dims** (List[int]): The dimensions of embedding.
            - **depths** (List[int]): The number of blocks in each stage.
            - **ffn_ratios** (List[int]): The number of expansion ratio of
              feedforward network hidden layer channels.

            Defaults to 'tiny'.
        patch_sizes (List[int | tuple]): The patch size in patch embeddings.
            Defaults to [7, 3, 3, 3].
        in_channels (int): The num of input channels. Defaults to 3.
        drop_rate (float): Dropout rate after embedding. Defaults to 0.
        drop_path_rate (float): Stochastic depth rate. Defaults to 0.1.
        out_indices (Sequence[int]): Output from which stages.
            Default: ``(3, )``.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters. Defaults to -1.
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only. Defaults to False.
        norm_cfg (dict): Config dict for normalization layer for all output
            features. Defaults to ``dict(type='LN')``
        block_cfgs (Sequence[dict] | dict): The extra config of each block.
            Defaults to empty dicts.
        init_cfg (dict, optional): The Config for initialization.
            Defaults to None.

    Examples:
        >>> from mmpretrain.models import VAN
        >>> import torch
        >>> cfg = dict(arch='tiny')
        >>> model = VAN(**cfg)
        >>> inputs = torch.rand(1, 3, 224, 224)
        >>> outputs = model(inputs)
        >>> for out in outputs:
        >>>     print(out.size())
        (1, 256, 7, 7)
    """
    arch_zoo = {
        **dict.fromkeys(['t', 'tiny'],
                        {'embed_dims': [32, 64, 160, 256],
                         'depths': [3, 3, 5, 2],
                         'ffn_ratios': [8, 8, 4, 4]}),
        **dict.fromkeys(['s', 'small'],
                        {'embed_dims': [64, 128, 320, 512],
                         'depths': [2, 2, 4, 2],
                         'ffn_ratios': [8, 8, 4, 4]}),
        **dict.fromkeys(['b', 'base'],
                        {'embed_dims': [64, 128, 320, 512],
                         'depths': [3, 3, 12, 3],
                         'ffn_ratios': [8, 8, 4, 4]}),
        **dict.fromkeys(['l', 'large'],
                        {'embed_dims': [64, 128, 320, 512],
                         'depths': [3, 5, 27, 3],
                         'ffn_ratios': [8, 8, 4, 4]}),
    }  # yapf: disable

    def __init__(self,
                 arch='tiny',
                 patch_sizes=[7, 3, 3, 3],
                 in_channels=3,
                 drop_rate=0.,
                 drop_path_rate=0.,
                 out_indices=(3, ),
                 frozen_stages=-1,
                 norm_eval=False,
                 norm_cfg=dict(type='LN'),
                 block_cfgs=dict(),
                 init_cfg=None):
        super(VAN, self).__init__(init_cfg=init_cfg)

        if isinstance(arch, str):
            arch = arch.lower()
            assert arch in set(self.arch_zoo), \
                f'Arch {arch} is not in default archs {set(self.arch_zoo)}'
            self.arch_settings = self.arch_zoo[arch]
        else:
            essential_keys = {'embed_dims', 'depths', 'ffn_ratios'}
            assert isinstance(arch, dict) and set(arch) == essential_keys, \
                f'Custom arch needs a dict with keys {essential_keys}'
            self.arch_settings = arch

        self.embed_dims = self.arch_settings['embed_dims']
        self.depths = self.arch_settings['depths']
        self.ffn_ratios = self.arch_settings['ffn_ratios']
        self.num_stages = len(self.depths)
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.norm_eval = norm_eval

        total_depth = sum(self.depths)
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, total_depth)
        ]  # stochastic depth decay rule

        cur_block_idx = 0
        for i, depth in enumerate(self.depths):
            patch_embed = VANPatchEmbed(
                in_channels=in_channels if i == 0 else self.embed_dims[i - 1],
                input_size=None,
                embed_dims=self.embed_dims[i],
                kernel_size=patch_sizes[i],
                stride=patch_sizes[i] // 2 + 1,
                padding=(patch_sizes[i] // 2, patch_sizes[i] // 2),
                norm_cfg=dict(type='BN'))

            blocks = ModuleList([
                VANBlock(
                    embed_dims=self.embed_dims[i],
                    ffn_ratio=self.ffn_ratios[i],
                    drop_rate=drop_rate,
                    drop_path_rate=dpr[cur_block_idx + j],
                    **block_cfgs) for j in range(depth)
            ])
            cur_block_idx += depth
            norm = build_norm_layer(norm_cfg, self.embed_dims[i])[1]

            self.add_module(f'patch_embed{i + 1}', patch_embed)
            self.add_module(f'blocks{i + 1}', blocks)
            self.add_module(f'norm{i + 1}', norm)

    def train(self, mode=True):
        super(VAN, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()

    def _freeze_stages(self):
        for i in range(0, self.frozen_stages + 1):
            # freeze patch embed
            m = getattr(self, f'patch_embed{i + 1}')
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

            # freeze blocks
            m = getattr(self, f'blocks{i + 1}')
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

            # freeze norm
            m = getattr(self, f'norm{i + 1}')
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def forward(self, x):
        outs = []
        for i in range(self.num_stages):
            patch_embed = getattr(self, f'patch_embed{i + 1}')
            blocks = getattr(self, f'blocks{i + 1}')
            norm = getattr(self, f'norm{i + 1}')
            x, hw_shape = patch_embed(x)
            for block in blocks:
                x = block(x)
            x = x.flatten(2).transpose(1, 2)
            x = norm(x)
            x = x.reshape(-1, *hw_shape,
                          block.out_channels).permute(0, 3, 1, 2).contiguous()
            if i in self.out_indices:
                outs.append(x)

        return tuple(outs)

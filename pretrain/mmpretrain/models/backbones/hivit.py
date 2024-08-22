# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
import torch.nn as nn
from mmcv.cnn.bricks import DropPath
from mmengine.model.weight_init import trunc_normal_

from mmpretrain.registry import MODELS
from ..utils import build_norm_layer, to_2tuple
from .base_backbone import BaseBackbone


class Mlp(nn.Module):
    """MLP block.

    Args:
        in_features (int): Number of input dims.
        hidden_features (int): Number of hidden dims.
        out_feature (int): Number of out dims.
        act_layer: MLP activation layer.
        drop (float): MLP dropout rate.
    """

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    """Attention.

    Args:
        input size (int): Input size.
        dim (int): Number of input dims.
        num_heads (int): Number of attention heads.
        qkv_bias (bool): Enable bias for qkv projections if True.
        qk_scale (float): The number of divider after q@k. Default to None.
        attn_drop (float): The drop out rate for attention output weights.
            Defaults to 0.
        proj_drop (float): Probability of an element to be zeroed
            after the feed forward layer. Defaults to 0.
        rpe (bool): If True, add relative position embedding to
            the patch embedding.
    """

    def __init__(self,
                 input_size,
                 dim,
                 num_heads,
                 qkv_bias=True,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 rpe=True):
        super().__init__()
        self.input_size = input_size
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * input_size - 1) *
                        (2 * input_size - 1), num_heads)) if rpe else None
        if rpe:
            coords_h = torch.arange(input_size)
            coords_w = torch.arange(input_size)
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
            coords_flatten = torch.flatten(coords, 1)
            relative_coords = coords_flatten[:, :,
                                             None] - coords_flatten[:, None, :]
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()
            relative_coords[:, :, 0] += input_size - 1
            relative_coords[:, :, 1] += input_size - 1
            relative_coords[:, :, 0] *= 2 * input_size - 1
            relative_position_index = relative_coords.sum(-1)
            self.register_buffer('relative_position_index',
                                 relative_position_index)

            trunc_normal_(self.relative_position_bias_table, std=.02)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, rpe_index=None, mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads,
                                  C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[
            2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        if rpe_index is not None:
            rpe_index = self.relative_position_index.view(-1)
            S = int(math.sqrt(rpe_index.size(-1)))
            relative_position_bias = self.relative_position_bias_table[
                rpe_index].view(-1, S, S, self.num_heads)
            relative_position_bias = relative_position_bias.permute(
                0, 3, 1, 2).contiguous()
            attn = attn + relative_position_bias
        if mask is not None:
            mask = mask.bool()
            attn = attn.masked_fill(~mask[:, None, None, :], float('-inf'))
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class BlockWithRPE(nn.Module):
    """HiViT block.

    Args:
        input_size (int): Input size.
        dim (int): Number of input dims.
        num_heads (int): Number of attention heads.
        mlp_ratio (int): Ratio of MLP hidden dim to embedding dim.
        qkv_bias (bool): Enable bias for qkv projections if True.
        qk_scale (float): The number of divider after q@k. Default to None.
        drop (float): Probability of an element to be zeroed
            after the feed forward layer. Defaults to 0.
        attn_drop (float): The drop out rate for attention output weights.
            Defaults to 0.
        drop_path (float): Stochastic depth rate. Defaults to 0.
        rpe (bool): If True, add relative position embedding to
            the patch embedding.
        layer_scale_init_value (float): Layer-scale init values. Defaults to 0.
        act_layer: MLP activation layer.
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to ``dict(type='LN')``.
    """

    def __init__(self,
                 input_size,
                 dim,
                 num_heads=0.,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 rpe=True,
                 layer_scale_init_value=0.0,
                 act_layer=nn.GELU,
                 norm_cfg=dict(type='LN')):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        with_attn = num_heads > 0.

        self.norm1 = build_norm_layer(norm_cfg, dim) if with_attn else None
        self.attn = Attention(
            input_size,
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            rpe=rpe,
        ) if with_attn else None

        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = build_norm_layer(norm_cfg, dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop)

        if layer_scale_init_value > 0:
            self.gamma_1 = nn.Parameter(
                layer_scale_init_value * torch.ones(
                    (dim)), requires_grad=True) if with_attn else None
            self.gamma_2 = nn.Parameter(
                layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x, rpe_index=None, mask=None):
        if self.attn is not None:
            if self.gamma_1 is not None:
                x = x + self.drop_path(
                    self.gamma_1 * self.attn(self.norm1(x), rpe_index, mask))
            else:
                x = x + self.drop_path(
                    self.attn(self.norm1(x), rpe_index, mask))
        if self.gamma_2 is not None:
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """PatchEmbed for HiViT.

    Args:
        img_size (int): Input image size.
        patch_size (int): Patch size. Defaults to 16.
        inner_patches (int): Inner patch. Defaults to 4.
        in_chans (int): Number of image input channels.
        embed_dim (int): Transformer embedding dimension.
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to ``dict(type='LN')``.
        kernel_size (int): Kernel size.
        pad_size (int): Pad size.
    """

    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 inner_patches=4,
                 in_chans=3,
                 embed_dim=128,
                 norm_cfg=None,
                 kernel_size=None,
                 pad_size=None):
        super().__init__()
        img_size = to_2tuple(img_size) if not isinstance(img_size,
                                                         tuple) else img_size
        patch_size = to_2tuple(patch_size)
        patches_resolution = [
            img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        ]
        self.img_size = img_size
        self.patch_size = patch_size
        self.inner_patches = inner_patches
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        conv_size = [size // inner_patches for size in patch_size]
        kernel_size = kernel_size or conv_size
        pad_size = pad_size or 0
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=kernel_size,
            stride=conv_size,
            padding=pad_size)
        if norm_cfg is not None:
            self.norm = build_norm_layer(norm_cfg, embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        patches_resolution = (H // self.patch_size[0], W // self.patch_size[1])
        num_patches = patches_resolution[0] * patches_resolution[1]
        x = self.proj(x).view(
            B,
            -1,
            patches_resolution[0],
            self.inner_patches,
            patches_resolution[1],
            self.inner_patches,
        ).permute(0, 2, 4, 3, 5, 1).reshape(B, num_patches, self.inner_patches,
                                            self.inner_patches, -1)
        if self.norm is not None:
            x = self.norm(x)
        return x


class PatchMerge(nn.Module):
    """PatchMerge for HiViT.

    Args:
        dim (int): Number of input channels.
        norm_cfg (dict): Config dict for normalization layer.
    """

    def __init__(self, dim, norm_cfg):
        super().__init__()
        self.norm = build_norm_layer(norm_cfg, dim * 4)
        self.reduction = nn.Linear(dim * 4, dim * 2, bias=False)

    def forward(self, x, *args, **kwargs):
        is_main_stage = len(x.shape) == 3
        if is_main_stage:
            B, N, C = x.shape
            S = int(math.sqrt(N))
            x = x.reshape(B, S // 2, 2, S // 2, 2, C) \
                .permute(0, 1, 3, 2, 4, 5) \
                .reshape(B, -1, 2, 2, C)
        x0 = x[..., 0::2, 0::2, :]
        x1 = x[..., 1::2, 0::2, :]
        x2 = x[..., 0::2, 1::2, :]
        x3 = x[..., 1::2, 1::2, :]

        x = torch.cat([x0, x1, x2, x3], dim=-1)
        x = self.norm(x)
        x = self.reduction(x)

        if is_main_stage:
            x = x[:, :, 0, 0, :]
        return x


@MODELS.register_module()
class HiViT(BaseBackbone):
    """HiViT.

    A PyTorch implement of: `HiViT: A Simple and More Efficient Design
    of Hierarchical Vision Transformer <https://arxiv.org/abs/2205.14949>`_.

    Args:
        arch (str | dict): Swin Transformer architecture. If use string, choose
            from 'tiny', 'small', and'base'. If use dict, it should
            have below keys:

            - **embed_dims** (int): The dimensions of embedding.
            - **depths** (List[int]): The number of blocks in each stage.
            - **num_heads** (int): The number of heads in attention
              modules of each stage.

        Defaults to 'tiny'.
        img_size (int): Input image size.
        patch_size (int): Patch size. Defaults to 16.
        inner_patches (int): Inner patch. Defaults to 4.
        in_chans (int): Number of image input channels.
        embed_dim (int): Transformer embedding dimension.
        depths (list[int]): Number of successive HiViT blocks.
        num_heads (int): Number of attention heads.
        stem_mlp_ratio (int): Ratio of MLP hidden dim to embedding dim
            in the first two stages.
        mlp_ratio (int): Ratio of MLP hidden dim to embedding dim in
            the last stage.
        qkv_bias (bool): Enable bias for qkv projections if True.
        qk_scale (float): The number of divider after q@k. Default to None.
        drop_rate (float): Probability of an element to be zeroed
            after the feed forward layer. Defaults to 0.
        attn_drop_rate (float): The drop out rate for attention output weights.
            Defaults to 0.
        drop_path_rate (float): Stochastic depth rate. Defaults to 0.
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to ``dict(type='LN')``.
        ape (bool): If True, add absolute position embedding to
            the patch embedding.
        rpe (bool): If True, add relative position embedding to
            the patch embedding.
        patch_norm (bool): If True, use norm_cfg for normalization layer.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters. Defaults to -1.
        kernel_size (int): Kernel size.
        pad_size (int): Pad size.
        layer_scale_init_value (float): Layer-scale init values. Defaults to 0.
        init_cfg (dict, optional): The extra config for initialization.
             Defaults to None.
    """
    arch_zoo = {
        **dict.fromkeys(['t', 'tiny'],
                        {'embed_dims': 384,
                         'depths': [1, 1, 10],
                         'num_heads': 6}),
        **dict.fromkeys(['s', 'small'],
                        {'embed_dims': 384,
                         'depths': [2, 2, 20],
                         'num_heads': 6}),
        **dict.fromkeys(['b', 'base'],
                        {'embed_dims': 512,
                         'depths': [2, 2, 24],
                         'num_heads': 8}),
        **dict.fromkeys(['l', 'large'],
                        {'embed_dims': 768,
                         'depths': [2, 2, 40],
                         'num_heads': 12}),
    }  # yapf: disable

    num_extra_tokens = 0

    def __init__(self,
                 arch='base',
                 img_size=224,
                 patch_size=16,
                 inner_patches=4,
                 in_chans=3,
                 stem_mlp_ratio=3.,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.0,
                 norm_cfg=dict(type='LN'),
                 out_indices=[23],
                 ape=True,
                 rpe=False,
                 patch_norm=True,
                 frozen_stages=-1,
                 kernel_size=None,
                 pad_size=None,
                 layer_scale_init_value=0.0,
                 init_cfg=None):
        super(HiViT, self).__init__(init_cfg=init_cfg)

        if isinstance(arch, str):
            arch = arch.lower()
            assert arch in set(self.arch_zoo), \
                f'Arch {arch} is not in default archs {set(self.arch_zoo)}'
            self.arch_settings = self.arch_zoo[arch]
        else:
            essential_keys = {'embed_dims', 'depths', 'num_heads'}
            assert isinstance(arch, dict) and set(arch) == essential_keys, \
                f'Custom arch needs a dict with keys {essential_keys}'
            self.arch_settings = arch
        self.embed_dims = self.arch_settings['embed_dims']
        self.depths = self.arch_settings['depths']
        self.num_heads = self.arch_settings['num_heads']

        self.num_stages = len(self.depths)
        self.ape = ape
        self.rpe = rpe
        self.patch_size = patch_size
        self.num_features = self.embed_dims
        self.mlp_ratio = mlp_ratio
        self.num_main_blocks = self.depths[-1]
        self.out_indices = out_indices
        self.out_indices[-1] = self.depths[-1] - 1

        img_size = to_2tuple(img_size) if not isinstance(img_size,
                                                         tuple) else img_size

        embed_dim = self.embed_dims // 2**(self.num_stages - 1)
        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            inner_patches=inner_patches,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_cfg=norm_cfg if patch_norm else None,
            kernel_size=kernel_size,
            pad_size=pad_size)
        num_patches = self.patch_embed.num_patches
        Hp, Wp = self.patch_embed.patches_resolution

        if rpe:
            assert Hp == Wp, 'If you use relative position, make sure H == W '
            'of input size'

        # absolute position embedding
        if ape:
            self.pos_embed = nn.Parameter(
                torch.zeros(1, num_patches, self.num_features))
            trunc_normal_(self.pos_embed, std=.02)
        if rpe:
            # get pair-wise relative position index for each token inside the
            # window
            coords_h = torch.arange(Hp)
            coords_w = torch.arange(Wp)
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
            coords_flatten = torch.flatten(coords, 1)
            relative_coords = coords_flatten[:, :,
                                             None] - coords_flatten[:, None, :]
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()
            relative_coords[:, :, 0] += Hp - 1
            relative_coords[:, :, 1] += Wp - 1
            relative_coords[:, :, 0] *= 2 * Wp - 1
            relative_position_index = relative_coords.sum(-1)
            self.register_buffer('relative_position_index',
                                 relative_position_index)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = iter(
            x.item()
            for x in torch.linspace(0, drop_path_rate,
                                    sum(self.depths) + sum(self.depths[:-1])))

        # build blocks
        self.blocks = nn.ModuleList()
        for stage_i, stage_depth in enumerate(self.depths):
            is_main_stage = embed_dim == self.num_features
            nhead = self.num_heads if is_main_stage else 0
            ratio = mlp_ratio if is_main_stage else stem_mlp_ratio
            # every block not in main stage includes two mlp blocks
            stage_depth = stage_depth if is_main_stage else stage_depth * 2
            for _ in range(stage_depth):
                self.blocks.append(
                    BlockWithRPE(
                        Hp,
                        embed_dim,
                        nhead,
                        ratio,
                        qkv_bias,
                        qk_scale,
                        drop=drop_rate,
                        attn_drop=attn_drop_rate,
                        drop_path=next(dpr),
                        rpe=rpe,
                        norm_cfg=norm_cfg,
                        layer_scale_init_value=layer_scale_init_value,
                    ))
            if stage_i + 1 < self.num_stages:
                self.blocks.append(PatchMerge(embed_dim, norm_cfg))
                embed_dim *= 2

        self.frozen_stages = frozen_stages
        if self.frozen_stages > 0:
            self._freeze_stages()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def interpolate_pos_encoding(self, x, h, w):
        npatch = x.shape[1]
        N = self.pos_embed.shape[1]
        if npatch == N and w == h:
            return self.pos_embed
        patch_pos_embed = self.pos_embed
        dim = x.shape[-1]
        w0 = w // self.patch_size
        h0 = h // self.patch_size
        # we add a small number to avoid floating point error in interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)),
                                    dim).permute(0, 3, 1, 2),
            scale_factor=(h0 / math.sqrt(N), w0 / math.sqrt(N)),
            mode='bicubic',
        )
        assert int(h0) == patch_pos_embed.shape[-2] and int(
            w0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return patch_pos_embed

    def forward(self, x):
        B, C, H, W = x.shape
        Hp, Wp = H // self.patch_size, W // self.patch_size

        x = self.patch_embed(x)

        outs = []
        for i, blk in enumerate(self.blocks[:-self.num_main_blocks]):
            x = blk(x)
            if i in self.out_indices:
                x = x.reshape(B, Hp, Wp, *x.shape[-3:]).permute(
                    0, 5, 1, 3, 2, 4).reshape(B, -1, Hp * x.shape[-3],
                                              Wp * x.shape[-2]).contiguous()
                outs.append(x)

        x = x[..., 0, 0, :]
        if self.ape:
            x = x + self.interpolate_pos_encoding(x, H, W)
        x = self.pos_drop(x)

        rpe_index = True if self.rpe else None

        for i, blk in enumerate(self.blocks[-self.num_main_blocks:]):
            x = blk(x, rpe_index)
            if i in self.out_indices:
                x = x.transpose(1, 2).view(B, -1, Hp, Wp).contiguous()
                outs.append(x)

        return tuple(outs)

    def _freeze_stages(self):
        # freeze position embedding
        if self.pos_embed is not None:
            self.pos_embed.requires_grad = False
        # set dropout to eval model
        self.pos_drop.eval()
        # freeze patch embedding
        self.patch_embed.eval()
        for param in self.patch_embed.parameters():
            param.requires_grad = False
        # freeze layers
        for i in range(1, self.frozen_stages + 1):
            m = self.blocks[i - 1]
            m.eval()
            for param in m.parameters():
                param.requires_grad = False
        # freeze the last layer norm
        for param in self.fc_norm.parameters():
            param.requires_grad = False

    def get_layer_depth(self, param_name: str, prefix: str = ''):
        """Get the layer-wise depth of a parameter.

        Args:
            param_name (str): The name of the parameter.
            prefix (str): The prefix for the parameter.
                Defaults to an empty string.

        Returns:
            Tuple[int, int]: The layer-wise depth and the num of layers.

        Note:
            The first depth is the stem module (``layer_depth=0``), and the
            last depth is the subsequent module (``layer_depth=num_layers-1``)
        """
        self.num_layers = len(self.blocks)
        num_layers = self.num_layers + 2

        if not param_name.startswith(prefix):
            # For subsequent module like head
            return num_layers - 1, num_layers

        param_name = param_name[len(prefix):]

        if param_name in 'pos_embed':
            layer_depth = 0
        elif param_name.startswith('patch_embed'):
            layer_depth = 0
        elif param_name.startswith('layers'):
            layer_id = int(param_name.split('.')[1])
            layer_depth = layer_id + 1
        else:
            layer_depth = num_layers - 1

        return layer_depth, num_layers

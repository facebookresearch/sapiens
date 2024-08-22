# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, Union

import torch
from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks.drop import DropPath
from mmcv.cnn.bricks.transformer import PatchEmbed, PatchMerging
from mmengine.model import BaseModule
from torch import nn
from torch.utils.checkpoint import checkpoint

from mmpretrain.registry import MODELS
from ..utils import WindowMSA, to_2tuple
from .base_backbone import BaseBackbone
from .vision_transformer import TransformerEncoderLayer


class MixMIMWindowAttention(WindowMSA):
    """MixMIM Window Attention.

    Compared with WindowMSA, we add some modifications
    in ``forward`` to meet the requirement of MixMIM during
    pretraining.

    Implements one windown attention in MixMIM.
    Args:
        embed_dims (int): The feature dimension.
        window_size (list): The height and width of the window.
        num_heads (int): The number of head in attention.
        qkv_bias (bool): Whether to add bias for qkv in attention modules.
            Defaults to True.
        qk_scale (float, optional): Override default qk scale of
            ``head_dim ** -0.5`` if set. Defaults to None.
        attn_drop_rate (float): attention drop rate.
            Defaults to 0.
        proj_drop_rate (float): Probability of an element to be zeroed.
            Defaults to 0.
        init_cfg (dict, optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 embed_dims,
                 window_size,
                 num_heads,
                 qkv_bias=True,
                 qk_scale=None,
                 attn_drop_rate=0.,
                 proj_drop_rate=0.,
                 init_cfg=None):

        super().__init__(
            embed_dims=embed_dims,
            window_size=window_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop_rate,
            proj_drop=proj_drop_rate,
            init_cfg=init_cfg)

    def forward(self, x, mask=None):

        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads,
                                  C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[
            2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)].view(
                self.window_size[0] * self.window_size[1],
                self.window_size[0] * self.window_size[1],
                -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(
            2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            mask = mask.reshape(B_, 1, 1, N)
            mask_new = mask * mask.transpose(
                2, 3) + (1 - mask) * (1 - mask).transpose(2, 3)
            mask_new = 1 - mask_new

            if mask_new.dtype == torch.float16:
                attn = attn - 65500 * mask_new
            else:
                attn = attn - 1e30 * mask_new

            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MixMIMBlock(TransformerEncoderLayer):
    """MixMIM Block. Implements one block in MixMIM.

    Args:
        embed_dims (int): The feature dimension.
        input_resolution (tuple): Input resolution of this layer.
        num_heads (int): The number of head in attention,
        window_size (list): The height and width of the window.
        mlp_ratio (int): The MLP ration in FFN.
        num_fcs (int): The number of linear layers in a block.
        qkv_bias (bool): Whether to add bias for qkv in attention modules.
            Defaults to True.
        proj_drop_rate (float): Probability of an element to be zeroed.
            Defaults to 0.
        attn_drop_rate (float): attention drop rate.
            Defaults to 0.
        drop_path_rate (float): stochastic depth rate.
            Defaults to 0.
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to ``dict(type='LN')``.
        init_cfg (dict, optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 embed_dims,
                 input_resolution,
                 num_heads,
                 window_size=7,
                 mlp_ratio=4.,
                 num_fcs=2,
                 qkv_bias=True,
                 proj_drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 init_cfg: Optional[Union[List[dict], dict]] = None) -> None:

        super().__init__(
            embed_dims=embed_dims,
            num_heads=num_heads,
            feedforward_channels=int(mlp_ratio * embed_dims),
            drop_rate=proj_drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            num_fcs=num_fcs,
            qkv_bias=qkv_bias,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            init_cfg=init_cfg)

        self.embed_dims = embed_dims
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio

        if min(self.input_resolution) <= self.window_size:
            self.window_size = min(self.input_resolution)

        self.attn = MixMIMWindowAttention(
            embed_dims=embed_dims,
            window_size=to_2tuple(self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop_rate=attn_drop_rate,
            proj_drop_rate=proj_drop_rate)

        self.drop_path = DropPath(
            drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    @staticmethod
    def window_reverse(windows, H, W, window_size):
        B = int(windows.shape[0] / (H * W / window_size / window_size))
        x = windows.view(B, H // window_size, W // window_size, window_size,
                         window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
        return x

    @staticmethod
    def window_partition(x, window_size):
        B, H, W, C = x.shape
        x = x.view(B, H // window_size, window_size, W // window_size,
                   window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        windows = windows.view(-1, window_size, window_size, C)
        return windows

    def forward(self, x, attn_mask=None):
        H, W = self.input_resolution
        B, L, C = x.shape

        shortcut = x
        x = self.ln1(x)
        x = x.view(B, H, W, C)

        # partition windows
        x_windows = self.window_partition(
            x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size,
                                   C)  # nW*B, window_size*window_size, C
        if attn_mask is not None:
            attn_mask = attn_mask.repeat(B, 1, 1)  # B, N, 1
            attn_mask = attn_mask.view(B, H, W, 1)
            attn_mask = self.window_partition(attn_mask, self.window_size)
            attn_mask = attn_mask.view(-1, self.window_size * self.window_size,
                                       1)

        # W-MSA/SW-MSA
        attn_windows = self.attn(
            x_windows, mask=attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size,
                                         self.window_size, C)
        x = self.window_reverse(attn_windows, H, W,
                                self.window_size)  # B H' W' C

        x = x.view(B, H * W, C)

        x = shortcut + self.drop_path(x)

        x = self.ffn(self.norm2(x), identity=x)  # ffn contains DropPath

        return x


class MixMIMLayer(BaseModule):
    """Implements one MixMIM layer, which may contains several MixMIM blocks.

    Args:
        embed_dims (int): The feature dimension.
        input_resolution (tuple): Input resolution of this layer.
        depth (int): The number of blocks in this layer.
        num_heads (int): The number of head in attention,
        window_size (list): The height and width of the window.
        mlp_ratio (int): The MLP ration in FFN.
        qkv_bias (bool): Whether to add bias for qkv in attention modules.
            Defaults to True.
        proj_drop_rate (float): Probability of an element to be zeroed.
            Defaults to 0.
        attn_drop_rate (float): attention drop rate.
            Defaults to 0.
        drop_path_rate (float): stochastic depth rate.
            Defaults to 0.
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to ``dict(type='LN')``.
        downsample (class, optional): Downsample the output of blocks b
            y patch merging.Defaults to None.
        use_checkpoint (bool): Whether use the checkpoint to
        reduce GPU memory cost.
        init_cfg (dict, optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 embed_dims: int,
                 input_resolution: int,
                 depth: int,
                 num_heads: int,
                 window_size: int,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 proj_drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=[0.],
                 norm_cfg=dict(type='LN'),
                 downsample=None,
                 use_checkpoint=False,
                 init_cfg: Optional[Union[List[dict], dict]] = None) -> None:
        super().__init__(init_cfg=init_cfg)
        self.embed_dims = embed_dims
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList()
        for i in range(depth):
            self.blocks.append(
                MixMIMBlock(
                    embed_dims=embed_dims,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    window_size=window_size,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_drop_rate=proj_drop_rate,
                    attn_drop_rate=attn_drop_rate,
                    drop_path_rate=drop_path_rate[i],
                    norm_cfg=norm_cfg))
        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(
                in_channels=embed_dims,
                out_channels=2 * embed_dims,
                norm_cfg=norm_cfg)
        else:
            self.downsample = None

    def forward(self, x, attn_mask=None):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint(blk, x, attn_mask)
            else:
                x = blk(x, attn_mask=attn_mask)
        if self.downsample is not None:
            x, _ = self.downsample(x, self.input_resolution)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.embed_dims}, \
    input_resolution={self.input_resolution}, depth={self.depth}'


@MODELS.register_module()
class MixMIMTransformer(BaseBackbone):
    """MixMIM backbone.

    A PyTorch implement of : ` MixMIM: Mixed and Masked Image
    Modeling for Efficient Visual Representation Learning
    <https://arxiv.org/abs/2205.13137>`_

    Args:
        arch (str | dict): MixMIM architecture. If use string,
            choose from 'base','large' and 'huge'.
            If use dict, it should have below keys:

            - **embed_dims** (int): The dimensions of embedding.
            - **depths** (int): The number of transformer encoder layers.
            - **num_heads** (int): The number of heads in attention modules.

            Defaults to 'base'.
        mlp_ratio (int): The mlp ratio in FFN.  Defaults to 4.
        img_size (int | tuple): The expected input image shape. Because we
            support dynamic input shape, just set the argument to mlp_ratio
            the most common input image shape. Defaults to 224.
        patch_size (int | tuple): The patch size in patch embedding.
            Defaults to 16.
        in_channels (int): The num of input channels. Defaults to 3.
        window_size (list): The height and width of the window.
        qkv_bias (bool): Whether to add bias for qkv in attention modules.
            Defaults to True.
        patch_cfg (dict): Extra config dict for patch embedding.
            Defaults to an empty dict.
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to ``dict(type='LN')``.
        drop_rate (float): Probability of an element to be zeroed.
            Defaults to 0.
        drop_path_rate (float): stochastic depth rate. Defaults to 0.
        attn_drop_rate (float): attention drop rate. Defaults to 0.
        use_checkpoint (bool): Whether use the checkpoint to
        reduce GPU memory cost.
        init_cfg (dict, optional): Initialization config dict.
            Defaults to None.
    """
    arch_zoo = {
        **dict.fromkeys(
            ['b', 'base'], {
                'embed_dims': 128,
                'depths': [2, 2, 18, 2],
                'num_heads': [4, 8, 16, 32]
            }),
        **dict.fromkeys(
            ['l', 'large'], {
                'embed_dims': 192,
                'depths': [2, 2, 18, 2],
                'num_heads': [6, 12, 24, 48]
            }),
        **dict.fromkeys(
            ['h', 'huge'], {
                'embed_dims': 352,
                'depths': [2, 2, 18, 2],
                'num_heads': [11, 22, 44, 88]
            }),
    }

    def __init__(
        self,
        arch='base',
        mlp_ratio=4,
        img_size=224,
        patch_size=4,
        in_channels=3,
        window_size=[14, 14, 14, 7],
        qkv_bias=True,
        patch_cfg=dict(),
        norm_cfg=dict(type='LN'),
        drop_rate=0.0,
        drop_path_rate=0.0,
        attn_drop_rate=0.0,
        use_checkpoint=False,
        init_cfg: Optional[dict] = None,
    ) -> None:
        super(MixMIMTransformer, self).__init__(init_cfg=init_cfg)

        if isinstance(arch, str):
            arch = arch.lower()
            assert arch in set(self.arch_zoo), \
                f'Arch {arch} is not in default archs {set(self.arch_zoo)}'
            self.arch_settings = self.arch_zoo[arch]
        else:
            essential_keys = {'embed_dims', 'depths', 'num_heads'}
            assert isinstance(arch, dict) and essential_keys <= set(arch), \
                f'Custom arch needs a dict with keys {essential_keys}'
            self.arch_settings = arch

        self.embed_dims = self.arch_settings['embed_dims']
        self.depths = self.arch_settings['depths']
        self.num_heads = self.arch_settings['num_heads']

        self.encoder_stride = 32

        self.num_layers = len(self.depths)
        self.qkv_bias = qkv_bias
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.use_checkpoint = use_checkpoint
        self.mlp_ratio = mlp_ratio
        self.window_size = window_size

        _patch_cfg = dict(
            in_channels=in_channels,
            input_size=img_size,
            embed_dims=self.embed_dims,
            conv_type='Conv2d',
            kernel_size=patch_size,
            stride=patch_size,
            norm_cfg=dict(type='LN'),
        )
        _patch_cfg.update(patch_cfg)
        self.patch_embed = PatchEmbed(**_patch_cfg)
        self.patch_resolution = self.patch_embed.init_out_size

        self.dpr = [
            x.item()
            for x in torch.linspace(0, drop_path_rate, sum(self.depths))
        ]
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            self.layers.append(
                MixMIMLayer(
                    embed_dims=int(self.embed_dims * 2**i_layer),
                    input_resolution=(self.patch_resolution[0] // (2**i_layer),
                                      self.patch_resolution[1] //
                                      (2**i_layer)),
                    depth=self.depths[i_layer],
                    num_heads=self.num_heads[i_layer],
                    window_size=self.window_size[i_layer],
                    mlp_ratio=self.mlp_ratio,
                    qkv_bias=self.qkv_bias,
                    proj_drop_rate=self.drop_rate,
                    attn_drop_rate=self.attn_drop_rate,
                    drop_path_rate=self.dpr[sum(self.depths[:i_layer]
                                                ):sum(self.depths[:i_layer +
                                                                  1])],
                    norm_cfg=norm_cfg,
                    downsample=PatchMerging if
                    (i_layer < self.num_layers - 1) else None,
                    use_checkpoint=self.use_checkpoint))

        self.num_features = int(self.embed_dims * 2**(self.num_layers - 1))
        self.drop_after_pos = nn.Dropout(p=self.drop_rate)

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.num_patches = self.patch_resolution[0] * self.patch_resolution[1]
        self.absolute_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, self.embed_dims),
            requires_grad=False)

        _, self.norm = build_norm_layer(norm_cfg, self.num_features)

    def forward(self, x: torch.Tensor):
        x, _ = self.patch_embed(x)

        x = x + self.absolute_pos_embed
        x = self.drop_after_pos(x)

        for layer in self.layers:
            x = layer(x, attn_mask=None)

        x = self.norm(x)
        x = self.avgpool(x.transpose(1, 2))  # B C 1
        x = torch.flatten(x, 1)

        return (x, )

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
        num_layers = sum(self.depths) + 2

        if not param_name.startswith(prefix):
            # For subsequent module like neck and head
            if param_name.startswith('neck'):
                return num_layers - 2, num_layers
            else:
                return num_layers - 1, num_layers

        param_name = param_name[len(prefix):]

        stem_layers = ('patch_embed', 'absolute_pos_embed', 'pos_embed')
        if any(stem in param_name for stem in stem_layers):
            layer_depth = 0
        elif param_name.startswith('layers'):
            layer_id = int(param_name.split('.')[1])
            block_id = param_name.split('.')[3]

            if block_id in ('downsample', 'reduction', 'norm'):
                layer_depth = sum(self.depths[:layer_id + 1])
            else:
                layer_depth = sum(self.depths[:layer_id]) + int(block_id) + 1
        else:
            layer_depth = num_layers - 2

        return layer_depth, num_layers

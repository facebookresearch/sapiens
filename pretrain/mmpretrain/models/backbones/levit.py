# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import itertools

import torch
import torch.nn as nn
from mmcv.cnn import build_activation_layer, fuse_conv_bn
from mmcv.cnn.bricks import DropPath
from mmengine.model import BaseModule, ModuleList, Sequential

from mmpretrain.models.backbones.base_backbone import BaseBackbone
from mmpretrain.registry import MODELS
from ..utils import build_norm_layer


class HybridBackbone(BaseModule):

    def __init__(
            self,
            embed_dim,
            kernel_size=3,
            stride=2,
            pad=1,
            dilation=1,
            groups=1,
            act_cfg=dict(type='HSwish'),
            conv_cfg=None,
            norm_cfg=dict(type='BN'),
            init_cfg=None,
    ):
        super(HybridBackbone, self).__init__(init_cfg=init_cfg)

        self.input_channels = [
            3, embed_dim // 8, embed_dim // 4, embed_dim // 2
        ]
        self.output_channels = [
            embed_dim // 8, embed_dim // 4, embed_dim // 2, embed_dim
        ]
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        self.patch_embed = Sequential()

        for i in range(len(self.input_channels)):
            conv_bn = ConvolutionBatchNorm(
                self.input_channels[i],
                self.output_channels[i],
                kernel_size=kernel_size,
                stride=stride,
                pad=pad,
                dilation=dilation,
                groups=groups,
                norm_cfg=norm_cfg,
            )
            self.patch_embed.add_module('%d' % (2 * i), conv_bn)
            if i < len(self.input_channels) - 1:
                self.patch_embed.add_module('%d' % (i * 2 + 1),
                                            build_activation_layer(act_cfg))

    def forward(self, x):
        x = self.patch_embed(x)
        return x


class ConvolutionBatchNorm(BaseModule):

    def __init__(
            self,
            in_channel,
            out_channel,
            kernel_size=3,
            stride=2,
            pad=1,
            dilation=1,
            groups=1,
            norm_cfg=dict(type='BN'),
    ):
        super(ConvolutionBatchNorm, self).__init__()
        self.conv = nn.Conv2d(
            in_channel,
            out_channel,
            kernel_size=kernel_size,
            stride=stride,
            padding=pad,
            dilation=dilation,
            groups=groups,
            bias=False)
        self.bn = build_norm_layer(norm_cfg, out_channel)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

    @torch.no_grad()
    def fuse(self):
        return fuse_conv_bn(self).conv


class LinearBatchNorm(BaseModule):

    def __init__(self, in_feature, out_feature, norm_cfg=dict(type='BN1d')):
        super(LinearBatchNorm, self).__init__()
        self.linear = nn.Linear(in_feature, out_feature, bias=False)
        self.bn = build_norm_layer(norm_cfg, out_feature)

    def forward(self, x):
        x = self.linear(x)
        x = self.bn(x.flatten(0, 1)).reshape_as(x)
        return x

    @torch.no_grad()
    def fuse(self):
        w = self.bn.weight / (self.bn.running_var + self.bn.eps)**0.5
        w = self.linear.weight * w[:, None]
        b = self.bn.bias - self.bn.running_mean * self.bn.weight / \
            (self.bn.running_var + self.bn.eps) ** 0.5

        factory_kwargs = {
            'device': self.linear.weight.device,
            'dtype': self.linear.weight.dtype
        }
        bias = nn.Parameter(
            torch.empty(self.linear.out_features, **factory_kwargs))
        self.linear.register_parameter('bias', bias)
        self.linear.weight.data.copy_(w)
        self.linear.bias.data.copy_(b)
        return self.linear


class Residual(BaseModule):

    def __init__(self, block, drop_path_rate=0.):
        super(Residual, self).__init__()
        self.block = block
        if drop_path_rate > 0:
            self.drop_path = DropPath(drop_path_rate)
        else:
            self.drop_path = nn.Identity()

    def forward(self, x):
        x = x + self.drop_path(self.block(x))
        return x


class Attention(BaseModule):

    def __init__(
            self,
            dim,
            key_dim,
            num_heads=8,
            attn_ratio=4,
            act_cfg=dict(type='HSwish'),
            resolution=14,
    ):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.scale = key_dim**-0.5
        self.key_dim = key_dim
        self.nh_kd = nh_kd = key_dim * num_heads
        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * num_heads
        self.attn_ratio = attn_ratio
        h = self.dh + nh_kd * 2
        self.qkv = LinearBatchNorm(dim, h)
        self.proj = nn.Sequential(
            build_activation_layer(act_cfg), LinearBatchNorm(self.dh, dim))

        points = list(itertools.product(range(resolution), range(resolution)))
        N = len(points)
        attention_offsets = {}
        idxs = []
        for p1 in points:
            for p2 in points:
                offset = (abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                idxs.append(attention_offsets[offset])
        self.attention_biases = torch.nn.Parameter(
            torch.zeros(num_heads, len(attention_offsets)))
        self.register_buffer('attention_bias_idxs',
                             torch.LongTensor(idxs).view(N, N))

    @torch.no_grad()
    def train(self, mode=True):
        """change the mode of model."""
        super(Attention, self).train(mode)
        if mode and hasattr(self, 'ab'):
            del self.ab
        else:
            self.ab = self.attention_biases[:, self.attention_bias_idxs]

    def forward(self, x):  # x (B,N,C)
        B, N, C = x.shape  # 2 196 128
        qkv = self.qkv(x)  # 2 196 128
        q, k, v = qkv.view(B, N, self.num_heads, -1).split(
            [self.key_dim, self.key_dim, self.d],
            dim=3)  # q 2 196 4 16 ; k 2 196 4 16; v 2 196 4 32
        q = q.permute(0, 2, 1, 3)  # 2 4 196 16
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        attn = ((q @ k.transpose(-2, -1)) *
                self.scale  # 2 4 196 16 * 2 4 16 196 -> 2 4 196 196
                + (self.attention_biases[:, self.attention_bias_idxs]
                   if self.training else self.ab))
        attn = attn.softmax(dim=-1)  # 2 4 196 196 -> 2 4 196 196
        x = (attn @ v).transpose(1, 2).reshape(
            B, N,
            self.dh)  # 2 4 196 196 * 2 4 196 32 -> 2 4 196 32 -> 2 196 128
        x = self.proj(x)
        return x


class MLP(nn.Sequential):

    def __init__(self, embed_dim, mlp_ratio, act_cfg=dict(type='HSwish')):
        super(MLP, self).__init__()
        h = embed_dim * mlp_ratio
        self.linear1 = LinearBatchNorm(embed_dim, h)
        self.activation = build_activation_layer(act_cfg)
        self.linear2 = LinearBatchNorm(h, embed_dim)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        return x


class Subsample(BaseModule):

    def __init__(self, stride, resolution):
        super(Subsample, self).__init__()
        self.stride = stride
        self.resolution = resolution

    def forward(self, x):
        B, _, C = x.shape
        # B, N, C -> B, H, W, C
        x = x.view(B, self.resolution, self.resolution, C)
        x = x[:, ::self.stride, ::self.stride]
        x = x.reshape(B, -1, C)  # B, H', W', C -> B, N', C
        return x


class AttentionSubsample(nn.Sequential):

    def __init__(self,
                 in_dim,
                 out_dim,
                 key_dim,
                 num_heads=8,
                 attn_ratio=2,
                 act_cfg=dict(type='HSwish'),
                 stride=2,
                 resolution=14):
        super(AttentionSubsample, self).__init__()
        self.num_heads = num_heads
        self.scale = key_dim**-0.5
        self.key_dim = key_dim
        self.nh_kd = nh_kd = key_dim * num_heads
        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * self.num_heads
        self.attn_ratio = attn_ratio
        self.sub_resolution = (resolution - 1) // stride + 1
        h = self.dh + nh_kd
        self.kv = LinearBatchNorm(in_dim, h)

        self.q = nn.Sequential(
            Subsample(stride, resolution), LinearBatchNorm(in_dim, nh_kd))
        self.proj = nn.Sequential(
            build_activation_layer(act_cfg), LinearBatchNorm(self.dh, out_dim))

        self.stride = stride
        self.resolution = resolution
        points = list(itertools.product(range(resolution), range(resolution)))
        sub_points = list(
            itertools.product(
                range(self.sub_resolution), range(self.sub_resolution)))
        N = len(points)
        N_sub = len(sub_points)
        attention_offsets = {}
        idxs = []
        for p1 in sub_points:
            for p2 in points:
                size = 1
                offset = (abs(p1[0] * stride - p2[0] + (size - 1) / 2),
                          abs(p1[1] * stride - p2[1] + (size - 1) / 2))
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                idxs.append(attention_offsets[offset])
        self.attention_biases = torch.nn.Parameter(
            torch.zeros(num_heads, len(attention_offsets)))
        self.register_buffer('attention_bias_idxs',
                             torch.LongTensor(idxs).view(N_sub, N))

    @torch.no_grad()
    def train(self, mode=True):
        super(AttentionSubsample, self).train(mode)
        if mode and hasattr(self, 'ab'):
            del self.ab
        else:
            self.ab = self.attention_biases[:, self.attention_bias_idxs]

    def forward(self, x):
        B, N, C = x.shape
        k, v = self.kv(x).view(B, N, self.num_heads,
                               -1).split([self.key_dim, self.d], dim=3)
        k = k.permute(0, 2, 1, 3)  # BHNC
        v = v.permute(0, 2, 1, 3)  # BHNC
        q = self.q(x).view(B, self.sub_resolution**2, self.num_heads,
                           self.key_dim).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale + \
               (self.attention_biases[:, self.attention_bias_idxs]
                if self.training else self.ab)
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, -1, self.dh)
        x = self.proj(x)
        return x


@MODELS.register_module()
class LeViT(BaseBackbone):
    """LeViT backbone.

    A PyTorch implementation of `LeViT: A Vision Transformer in ConvNet's
    Clothing for Faster Inference <https://arxiv.org/abs/2104.01136>`_

    Modified from the official implementation:
    https://github.com/facebookresearch/LeViT

    Args:
        arch (str | dict): LeViT architecture.

            If use string, choose from '128s', '128', '192', '256' and '384'.
            If use dict, it should have below keys:

            - **embed_dims** (List[int]): The embed dimensions of each stage.
            - **key_dims** (List[int]): The embed dimensions of the key in the
              attention layers of each stage.
            - **num_heads** (List[int]): The number of heads in each stage.
            - **depths** (List[int]): The number of blocks in each stage.

        img_size (int): Input image size
        patch_size (int | tuple): The patch size. Deault to 16
        attn_ratio (int): Ratio of hidden dimensions of the value in attention
            layers. Defaults to 2.
        mlp_ratio (int): Ratio of hidden dimensions in MLP layers.
            Defaults to 2.
        act_cfg (dict): The config of activation functions.
            Defaults to ``dict(type='HSwish')``.
        hybrid_backbone (callable): A callable object to build the patch embed
            module. Defaults to use :class:`HybridBackbone`.
        out_indices (Sequence | int): Output from which stages.
            Defaults to -1, means the last stage.
        deploy (bool): Whether to switch the model structure to
            deployment mode. Defaults to False.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Defaults to None.
    """
    arch_zoo = {
        '128s': {
            'embed_dims': [128, 256, 384],
            'num_heads': [4, 6, 8],
            'depths': [2, 3, 4],
            'key_dims': [16, 16, 16],
        },
        '128': {
            'embed_dims': [128, 256, 384],
            'num_heads': [4, 8, 12],
            'depths': [4, 4, 4],
            'key_dims': [16, 16, 16],
        },
        '192': {
            'embed_dims': [192, 288, 384],
            'num_heads': [3, 5, 6],
            'depths': [4, 4, 4],
            'key_dims': [32, 32, 32],
        },
        '256': {
            'embed_dims': [256, 384, 512],
            'num_heads': [4, 6, 8],
            'depths': [4, 4, 4],
            'key_dims': [32, 32, 32],
        },
        '384': {
            'embed_dims': [384, 512, 768],
            'num_heads': [6, 9, 12],
            'depths': [4, 4, 4],
            'key_dims': [32, 32, 32],
        },
    }

    def __init__(self,
                 arch,
                 img_size=224,
                 patch_size=16,
                 attn_ratio=2,
                 mlp_ratio=2,
                 act_cfg=dict(type='HSwish'),
                 hybrid_backbone=HybridBackbone,
                 out_indices=-1,
                 deploy=False,
                 drop_path_rate=0,
                 init_cfg=None):
        super(LeViT, self).__init__(init_cfg=init_cfg)

        if isinstance(arch, str):
            arch = arch.lower()
            assert arch in set(self.arch_zoo), \
                f'Arch {arch} is not in default archs {set(self.arch_zoo)}'
            self.arch = self.arch_zoo[arch]
        elif isinstance(arch, dict):
            essential_keys = {'embed_dim', 'num_heads', 'depth', 'key_dim'}
            assert isinstance(arch, dict) and set(arch) == essential_keys, \
                f'Custom arch needs a dict with keys {essential_keys}'
            self.arch = arch
        else:
            raise TypeError('Expect "arch" to be either a string '
                            f'or a dict, got {type(arch)}')

        self.embed_dims = self.arch['embed_dims']
        self.num_heads = self.arch['num_heads']
        self.key_dims = self.arch['key_dims']
        self.depths = self.arch['depths']
        self.num_stages = len(self.embed_dims)
        self.drop_path_rate = drop_path_rate

        self.patch_embed = hybrid_backbone(self.embed_dims[0])

        self.resolutions = []
        resolution = img_size // patch_size
        self.stages = ModuleList()
        for i, (embed_dims, key_dims, depth, num_heads) in enumerate(
                zip(self.embed_dims, self.key_dims, self.depths,
                    self.num_heads)):
            blocks = []
            if i > 0:
                downsample = AttentionSubsample(
                    in_dim=self.embed_dims[i - 1],
                    out_dim=embed_dims,
                    key_dim=key_dims,
                    num_heads=self.embed_dims[i - 1] // key_dims,
                    attn_ratio=4,
                    act_cfg=act_cfg,
                    stride=2,
                    resolution=resolution)
                blocks.append(downsample)
                resolution = downsample.sub_resolution
                if mlp_ratio > 0:  # mlp_ratio
                    blocks.append(
                        Residual(
                            MLP(embed_dims, mlp_ratio, act_cfg=act_cfg),
                            self.drop_path_rate))
            self.resolutions.append(resolution)
            for _ in range(depth):
                blocks.append(
                    Residual(
                        Attention(
                            embed_dims,
                            key_dims,
                            num_heads,
                            attn_ratio=attn_ratio,
                            act_cfg=act_cfg,
                            resolution=resolution,
                        ), self.drop_path_rate))
                if mlp_ratio > 0:
                    blocks.append(
                        Residual(
                            MLP(embed_dims, mlp_ratio, act_cfg=act_cfg),
                            self.drop_path_rate))

            self.stages.append(Sequential(*blocks))

        if isinstance(out_indices, int):
            out_indices = [out_indices]
        elif isinstance(out_indices, tuple):
            out_indices = list(out_indices)
        elif not isinstance(out_indices, list):
            raise TypeError('"out_indices" must by a list, tuple or int, '
                            f'get {type(out_indices)} instead.')
        for i, index in enumerate(out_indices):
            if index < 0:
                out_indices[i] = self.num_stages + index
            assert 0 <= out_indices[i] < self.num_stages, \
                f'Invalid out_indices {index}.'
        self.out_indices = out_indices

        self.deploy = False
        if deploy:
            self.switch_to_deploy()

    def switch_to_deploy(self):
        if self.deploy:
            return
        fuse_parameters(self)
        self.deploy = True

    def forward(self, x):
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)  # B, C, H, W -> B, L, C
        outs = []
        for i, stage in enumerate(self.stages):
            x = stage(x)
            B, _, C = x.shape
            if i in self.out_indices:
                out = x.reshape(B, self.resolutions[i], self.resolutions[i], C)
                out = out.permute(0, 3, 1, 2).contiguous()
                outs.append(out)

        return tuple(outs)


def fuse_parameters(module):
    for child_name, child in module.named_children():
        if hasattr(child, 'fuse'):
            setattr(module, child_name, child.fuse())
        else:
            fuse_parameters(child)

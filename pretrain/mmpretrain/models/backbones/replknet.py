# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from mmcv.cnn import build_activation_layer, build_norm_layer
from mmcv.cnn.bricks import DropPath
from mmengine.model import BaseModule
from mmengine.utils.dl_utils.parrots_wrapper import _BatchNorm

from mmpretrain.registry import MODELS
from .base_backbone import BaseBackbone


def conv_bn(in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            groups,
            dilation=1,
            norm_cfg=dict(type='BN')):
    """Construct a sequential conv and bn.

    Args:
        in_channels (int): Dimension of input features.
        out_channels (int): Dimension of output features.
        kernel_size (int): kernel_size of the convolution.
        stride (int): stride of the convolution.
        padding (int): stride of the convolution.
        groups (int): groups of the convolution.
        dilation (int): dilation of the convolution. Default to 1.
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default to  ``dict(type='BN', requires_grad=True)``.

    Returns:
        nn.Sequential(): A conv layer and a batch norm layer.
    """
    if padding is None:
        padding = kernel_size // 2
    result = nn.Sequential()
    result.add_module(
        'conv',
        nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=False))
    result.add_module('bn', build_norm_layer(norm_cfg, out_channels)[1])
    return result


def conv_bn_relu(in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 groups,
                 dilation=1):
    """Construct a sequential conv, bn and relu.

    Args:
        in_channels (int): Dimension of input features.
        out_channels (int): Dimension of output features.
        kernel_size (int): kernel_size of the convolution.
        stride (int): stride of the convolution.
        padding (int): stride of the convolution.
        groups (int): groups of the convolution.
        dilation (int): dilation of the convolution. Default to 1.

    Returns:
        nn.Sequential(): A conv layer, batch norm layer and a relu function.
    """

    if padding is None:
        padding = kernel_size // 2
    result = conv_bn(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        groups=groups,
        dilation=dilation)
    result.add_module('nonlinear', nn.ReLU())
    return result


def fuse_bn(conv, bn):
    """Fuse the parameters in a branch with a conv and bn.

    Args:
        conv (nn.Conv2d): The convolution module to fuse.
        bn (nn.BatchNorm2d): The batch normalization to fuse.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: The parameters obtained after
        fusing the parameters of conv and bn in one branch.
        The first element is the weight and the second is the bias.
    """
    kernel = conv.weight
    running_mean = bn.running_mean
    running_var = bn.running_var
    gamma = bn.weight
    beta = bn.bias
    eps = bn.eps
    std = (running_var + eps).sqrt()
    t = (gamma / std).reshape(-1, 1, 1, 1)
    return kernel * t, beta - running_mean * gamma / std


class ReparamLargeKernelConv(BaseModule):
    """Super large kernel implemented by with large convolutions.

    Input: Tensor with shape [B, C, H, W].
    Output: Tensor with shape [B, C, H, W].

    Args:
        in_channels (int): Dimension of input features.
        out_channels (int): Dimension of output features.
        kernel_size (int): kernel_size of the large convolution.
        stride (int): stride of the large convolution.
        groups (int): groups of the large convolution.
        small_kernel (int): kernel_size of the small convolution.
        small_kernel_merged (bool): Whether to switch the model structure to
            deployment mode (merge the small kernel to the large kernel).
            Default to  False.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Defaults to None
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 groups,
                 small_kernel,
                 small_kernel_merged=False,
                 init_cfg=None):
        super(ReparamLargeKernelConv, self).__init__(init_cfg)
        self.kernel_size = kernel_size
        self.small_kernel = small_kernel
        self.small_kernel_merged = small_kernel_merged
        # We assume the conv does not change the feature map size,
        # so padding = k//2.
        # Otherwise, you may configure padding as you wish,
        # and change the padding of small_conv accordingly.
        padding = kernel_size // 2
        if small_kernel_merged:
            self.lkb_reparam = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=1,
                groups=groups,
                bias=True)
        else:
            self.lkb_origin = conv_bn(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=1,
                groups=groups)
            if small_kernel is not None:
                assert small_kernel <= kernel_size
                self.small_conv = conv_bn(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=small_kernel,
                    stride=stride,
                    padding=small_kernel // 2,
                    groups=groups,
                    dilation=1)

    def forward(self, inputs):
        if hasattr(self, 'lkb_reparam'):
            out = self.lkb_reparam(inputs)
        else:
            out = self.lkb_origin(inputs)
            if hasattr(self, 'small_conv'):
                out += self.small_conv(inputs)
        return out

    def get_equivalent_kernel_bias(self):
        eq_k, eq_b = fuse_bn(self.lkb_origin.conv, self.lkb_origin.bn)
        if hasattr(self, 'small_conv'):
            small_k, small_b = fuse_bn(self.small_conv.conv,
                                       self.small_conv.bn)
            eq_b += small_b
            #   add to the central part
            eq_k += nn.functional.pad(
                small_k, [(self.kernel_size - self.small_kernel) // 2] * 4)
        return eq_k, eq_b

    def merge_kernel(self):
        """Switch the model structure from training mode to deployment mode."""
        if self.small_kernel_merged:
            return
        eq_k, eq_b = self.get_equivalent_kernel_bias()
        self.lkb_reparam = nn.Conv2d(
            in_channels=self.lkb_origin.conv.in_channels,
            out_channels=self.lkb_origin.conv.out_channels,
            kernel_size=self.lkb_origin.conv.kernel_size,
            stride=self.lkb_origin.conv.stride,
            padding=self.lkb_origin.conv.padding,
            dilation=self.lkb_origin.conv.dilation,
            groups=self.lkb_origin.conv.groups,
            bias=True)

        self.lkb_reparam.weight.data = eq_k
        self.lkb_reparam.bias.data = eq_b
        self.__delattr__('lkb_origin')
        if hasattr(self, 'small_conv'):
            self.__delattr__('small_conv')

        self.small_kernel_merged = True


class ConvFFN(BaseModule):
    """Mlp implemented by with 1*1 convolutions.

    Input: Tensor with shape [B, C, H, W].
    Output: Tensor with shape [B, C, H, W].

    Args:
        in_channels (int): Dimension of input features.
        internal_channels (int): Dimension of hidden features.
        out_channels (int): Dimension of output features.
        drop_path (float): Stochastic depth rate. Defaults to 0.
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default to  ``dict(type='BN', requires_grad=True)``.
        act_cfg (dict): The config dict for activation between pointwise
            convolution. Defaults to ``dict(type='GELU')``.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 in_channels,
                 internal_channels,
                 out_channels,
                 drop_path,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='GELU'),
                 init_cfg=None):
        super(ConvFFN, self).__init__(init_cfg)
        self.drop_path = DropPath(
            drop_prob=drop_path) if drop_path > 0. else nn.Identity()
        self.preffn_bn = build_norm_layer(norm_cfg, in_channels)[1]
        self.pw1 = conv_bn(
            in_channels=in_channels,
            out_channels=internal_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1)
        self.pw2 = conv_bn(
            in_channels=internal_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1)
        self.nonlinear = build_activation_layer(act_cfg)

    def forward(self, x):
        out = self.preffn_bn(x)
        out = self.pw1(out)
        out = self.nonlinear(out)
        out = self.pw2(out)
        return x + self.drop_path(out)


class RepLKBlock(BaseModule):
    """RepLKBlock for RepLKNet backbone.

    Args:
        in_channels (int): The input channels of the block.
        dw_channels (int): The intermediate channels of the block,
            i.e., input channels of the large kernel convolution.
        block_lk_size (int): size of the super large kernel. Defaults: 31.
        small_kernel (int): size of the parallel small kernel. Defaults: 5.
        drop_path (float): Stochastic depth rate. Defaults: 0.
        small_kernel_merged (bool): Whether to switch the model structure to
            deployment mode (merge the small kernel to the large kernel).
            Default to  False.
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default to  ``dict(type='BN', requires_grad=True)``.
        act_cfg (dict): Config dict for activation layer.
            Default to  ``dict(type='ReLU')``.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default to  None
    """

    def __init__(self,
                 in_channels,
                 dw_channels,
                 block_lk_size,
                 small_kernel,
                 drop_path,
                 small_kernel_merged=False,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 init_cfg=None):
        super(RepLKBlock, self).__init__(init_cfg)
        self.pw1 = conv_bn_relu(in_channels, dw_channels, 1, 1, 0, groups=1)
        self.pw2 = conv_bn(dw_channels, in_channels, 1, 1, 0, groups=1)
        self.large_kernel = ReparamLargeKernelConv(
            in_channels=dw_channels,
            out_channels=dw_channels,
            kernel_size=block_lk_size,
            stride=1,
            groups=dw_channels,
            small_kernel=small_kernel,
            small_kernel_merged=small_kernel_merged)
        self.lk_nonlinear = build_activation_layer(act_cfg)
        self.prelkb_bn = build_norm_layer(norm_cfg, in_channels)[1]
        self.drop_path = DropPath(
            drop_prob=drop_path) if drop_path > 0. else nn.Identity()
        # print('drop path:', self.drop_path)

    def forward(self, x):
        out = self.prelkb_bn(x)
        out = self.pw1(out)
        out = self.large_kernel(out)
        out = self.lk_nonlinear(out)
        out = self.pw2(out)
        return x + self.drop_path(out)


class RepLKNetStage(BaseModule):
    """
    generate RepLKNet blocks for a stage
    return: RepLKNet blocks

    Args:
        channels (int): The input channels of the stage.
        num_blocks (int): The number of blocks of the stage.
        stage_lk_size (int): size of the super large kernel. Defaults: 31.
        drop_path (float): Stochastic depth rate. Defaults: 0.
        small_kernel (int): size of the parallel small kernel. Defaults: 5.
        dw_ratio (float): The intermediate channels
            expansion ratio of the block. Defaults: 1.
        ffn_ratio (float): Mlp expansion ratio. Defaults to 4.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default to  False.
        small_kernel_merged (bool): Whether to switch the model structure to
            deployment mode (merge the small kernel to the large kernel).
            Default to  False.
        norm_intermediate_features (bool): Construct and config norm layer
            or not.
            Using True will normalize the intermediate features for
            downstream dense prediction tasks.
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default to  ``dict(type='BN', requires_grad=True)``.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default to  None
    """

    def __init__(
            self,
            channels,
            num_blocks,
            stage_lk_size,
            drop_path,
            small_kernel,
            dw_ratio=1,
            ffn_ratio=4,
            with_cp=False,  # train with torch.utils.checkpoint to save memory
            small_kernel_merged=False,
            norm_intermediate_features=False,
            norm_cfg=dict(type='BN'),
            init_cfg=None):
        super(RepLKNetStage, self).__init__(init_cfg)
        self.with_cp = with_cp
        blks = []
        for i in range(num_blocks):
            block_drop_path = drop_path[i] if isinstance(drop_path,
                                                         list) else drop_path
            #   Assume all RepLK Blocks within a stage share the same lk_size.
            #   You may tune it on your own model.
            replk_block = RepLKBlock(
                in_channels=channels,
                dw_channels=int(channels * dw_ratio),
                block_lk_size=stage_lk_size,
                small_kernel=small_kernel,
                drop_path=block_drop_path,
                small_kernel_merged=small_kernel_merged)
            convffn_block = ConvFFN(
                in_channels=channels,
                internal_channels=int(channels * ffn_ratio),
                out_channels=channels,
                drop_path=block_drop_path)
            blks.append(replk_block)
            blks.append(convffn_block)
        self.blocks = nn.ModuleList(blks)
        if norm_intermediate_features:
            self.norm = build_norm_layer(norm_cfg, channels)[1]
        else:
            self.norm = nn.Identity()

    def forward(self, x):
        for blk in self.blocks:
            if self.with_cp:
                x = checkpoint.checkpoint(blk, x)  # Save training memory
            else:
                x = blk(x)
        return x


@MODELS.register_module()
class RepLKNet(BaseBackbone):
    """RepLKNet backbone.

    A PyTorch impl of :
    `Scaling Up Your Kernels to 31x31: Revisiting Large Kernel Design in CNNs
    <https://arxiv.org/abs/2203.06717>`_

    Args:
        arch (str | dict): The parameter of RepLKNet.
            If it's a dict, it should contain the following keys:

            - large_kernel_sizes (Sequence[int]):
                Large kernel size in each stage.
            - layers (Sequence[int]): Number of blocks in each stage.
            - channels (Sequence[int]): Number of channels in each stage.
            - small_kernel (int): size of the parallel small kernel.
            - dw_ratio (float): The intermediate channels
                expansion ratio of the block.
        in_channels (int): Number of input image channels. Default to  3.
        ffn_ratio (float): Mlp expansion ratio. Defaults to 4.
        out_indices (Sequence[int]): Output from which stages.
            Default to  (3, ).
        strides (Sequence[int]): Strides of the first block of each stage.
            Default to  (2, 2, 2, 2).
        dilations (Sequence[int]): Dilation of each stage.
            Default to  (1, 1, 1, 1).
        frozen_stages (int): Stages to be frozen
            (all param fixed). -1 means not freezing any parameters.
            Default to  -1.
        conv_cfg (dict | None): The config dict for conv layers.
            Default to None.
        norm_cfg (dict): The config dict for norm layers.
            Default to  ``dict(type='BN')``.
        act_cfg (dict): Config dict for activation layer.
            Default to  ``dict(type='ReLU')``.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default to False.
        deploy (bool): Whether to switch the model structure to deployment
            mode. Default to False.
        norm_intermediate_features (bool): Construct and
            config norm layer or not.
            Using True will normalize the intermediate features
            for downstream dense prediction tasks.
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only. Default to False.
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    arch_settings = {
        '31B':
        dict(
            large_kernel_sizes=[31, 29, 27, 13],
            layers=[2, 2, 18, 2],
            channels=[128, 256, 512, 1024],
            small_kernel=5,
            dw_ratio=1),
        '31L':
        dict(
            large_kernel_sizes=[31, 29, 27, 13],
            layers=[2, 2, 18, 2],
            channels=[192, 384, 768, 1536],
            small_kernel=5,
            dw_ratio=1),
        'XL':
        dict(
            large_kernel_sizes=[27, 27, 27, 13],
            layers=[2, 2, 18, 2],
            channels=[256, 512, 1024, 2048],
            small_kernel=None,
            dw_ratio=1.5),
    }

    def __init__(self,
                 arch,
                 in_channels=3,
                 ffn_ratio=4,
                 out_indices=(3, ),
                 strides=(2, 2, 2, 2),
                 dilations=(1, 1, 1, 1),
                 frozen_stages=-1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 with_cp=False,
                 drop_path_rate=0.3,
                 small_kernel_merged=False,
                 norm_intermediate_features=False,
                 norm_eval=False,
                 init_cfg=[
                     dict(type='Kaiming', layer=['Conv2d']),
                     dict(
                         type='Constant',
                         val=1,
                         layer=['_BatchNorm', 'GroupNorm'])
                 ]):
        super(RepLKNet, self).__init__(init_cfg)

        if isinstance(arch, str):
            assert arch in self.arch_settings, \
                f'"arch": "{arch}" is not one of the arch_settings'
            arch = self.arch_settings[arch]
        elif not isinstance(arch, dict):
            raise TypeError('Expect "arch" to be either a string '
                            f'or a dict, got {type(arch)}')

        assert len(arch['layers']) == len(
            arch['channels']) == len(strides) == len(dilations)
        assert max(out_indices) < len(arch['layers'])

        self.arch = arch
        self.in_channels = in_channels
        self.out_indices = out_indices
        self.strides = strides
        self.dilations = dilations
        self.frozen_stages = frozen_stages
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.with_cp = with_cp
        self.drop_path_rate = drop_path_rate
        self.small_kernel_merged = small_kernel_merged
        self.norm_eval = norm_eval
        self.norm_intermediate_features = norm_intermediate_features

        self.out_indices = out_indices

        base_width = self.arch['channels'][0]
        self.norm_intermediate_features = norm_intermediate_features
        self.num_stages = len(self.arch['layers'])
        self.stem = nn.ModuleList([
            conv_bn_relu(
                in_channels=in_channels,
                out_channels=base_width,
                kernel_size=3,
                stride=2,
                padding=1,
                groups=1),
            conv_bn_relu(
                in_channels=base_width,
                out_channels=base_width,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=base_width),
            conv_bn_relu(
                in_channels=base_width,
                out_channels=base_width,
                kernel_size=1,
                stride=1,
                padding=0,
                groups=1),
            conv_bn_relu(
                in_channels=base_width,
                out_channels=base_width,
                kernel_size=3,
                stride=2,
                padding=1,
                groups=base_width)
        ])
        # stochastic depth. We set block-wise drop-path rate.
        # The higher level blocks are more likely to be dropped.
        # This implementation follows Swin.
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate,
                                             sum(self.arch['layers']))
        ]
        self.stages = nn.ModuleList()
        self.transitions = nn.ModuleList()
        for stage_idx in range(self.num_stages):
            layer = RepLKNetStage(
                channels=self.arch['channels'][stage_idx],
                num_blocks=self.arch['layers'][stage_idx],
                stage_lk_size=self.arch['large_kernel_sizes'][stage_idx],
                drop_path=dpr[sum(self.arch['layers'][:stage_idx]
                                  ):sum(self.arch['layers'][:stage_idx + 1])],
                small_kernel=self.arch['small_kernel'],
                dw_ratio=self.arch['dw_ratio'],
                ffn_ratio=ffn_ratio,
                with_cp=with_cp,
                small_kernel_merged=small_kernel_merged,
                norm_intermediate_features=(stage_idx in out_indices))
            self.stages.append(layer)
            if stage_idx < len(self.arch['layers']) - 1:
                transition = nn.Sequential(
                    conv_bn_relu(
                        self.arch['channels'][stage_idx],
                        self.arch['channels'][stage_idx + 1],
                        1,
                        1,
                        0,
                        groups=1),
                    conv_bn_relu(
                        self.arch['channels'][stage_idx + 1],
                        self.arch['channels'][stage_idx + 1],
                        3,
                        stride=2,
                        padding=1,
                        groups=self.arch['channels'][stage_idx + 1]))
                self.transitions.append(transition)

    def forward_features(self, x):
        x = self.stem[0](x)
        for stem_layer in self.stem[1:]:
            if self.with_cp:
                x = checkpoint.checkpoint(stem_layer, x)  # save memory
            else:
                x = stem_layer(x)

        #   Need the intermediate feature maps
        outs = []
        for stage_idx in range(self.num_stages):
            x = self.stages[stage_idx](x)
            if stage_idx in self.out_indices:
                outs.append(self.stages[stage_idx].norm(x))
                # For RepLKNet-XL normalize the features
                # before feeding them into the heads
            if stage_idx < self.num_stages - 1:
                x = self.transitions[stage_idx](x)
        return outs

    def forward(self, x):
        x = self.forward_features(x)
        return tuple(x)

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.stem.eval()
            for param in self.stem.parameters():
                param.requires_grad = False
        for i in range(self.frozen_stages):
            stage = self.stages[i]
            stage.eval()
            for param in stage.parameters():
                param.requires_grad = False

    def train(self, mode=True):
        super(RepLKNet, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()

    def switch_to_deploy(self):
        for m in self.modules():
            if hasattr(m, 'merge_kernel'):
                m.merge_kernel()
        self.small_kernel_merged = True

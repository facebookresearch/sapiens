# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from mmcv.cnn import (ConvModule, build_activation_layer, build_conv_layer,
                      build_norm_layer)
from mmengine.model import BaseModule, Sequential
from mmengine.utils.dl_utils.parrots_wrapper import _BatchNorm
from torch import nn

from mmpretrain.registry import MODELS
from ..utils.se_layer import SELayer
from .base_backbone import BaseBackbone


class RepVGGBlock(BaseModule):
    """RepVGG block for RepVGG backbone.

    Args:
        in_channels (int): The input channels of the block.
        out_channels (int): The output channels of the block.
        stride (int): Stride of the 3x3 and 1x1 convolution layer. Default: 1.
        padding (int): Padding of the 3x3 convolution layer.
        dilation (int): Dilation of the 3x3 convolution layer.
        groups (int): Groups of the 3x3 and 1x1 convolution layer. Default: 1.
        padding_mode (str): Padding mode of the 3x3 convolution layer.
            Default: 'zeros'.
        se_cfg (None or dict): The configuration of the se module.
            Default: None.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.
        conv_cfg (dict, optional): Config dict for convolution layer.
            Default: None, which means using conv2d.
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: dict(type='BN', requires_grad=True).
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='ReLU').
        deploy (bool): Whether to switch the model structure to
            deployment mode. Default: False.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1,
                 padding=1,
                 dilation=1,
                 groups=1,
                 padding_mode='zeros',
                 se_cfg=None,
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 deploy=False,
                 init_cfg=None):
        super(RepVGGBlock, self).__init__(init_cfg)

        assert se_cfg is None or isinstance(se_cfg, dict)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.se_cfg = se_cfg
        self.with_cp = with_cp
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.deploy = deploy

        if deploy:
            self.branch_reparam = build_conv_layer(
                conv_cfg,
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=True,
                padding_mode=padding_mode)
        else:
            # judge if input shape and output shape are the same.
            # If true, add a normalized identity shortcut.
            if out_channels == in_channels and stride == 1 and \
                    padding == dilation:
                self.branch_norm = build_norm_layer(norm_cfg, in_channels)[1]
            else:
                self.branch_norm = None

            self.branch_3x3 = self.create_conv_bn(
                kernel_size=3,
                dilation=dilation,
                padding=padding,
            )
            self.branch_1x1 = self.create_conv_bn(kernel_size=1)

        if se_cfg is not None:
            self.se_layer = SELayer(channels=out_channels, **se_cfg)
        else:
            self.se_layer = None

        self.act = build_activation_layer(act_cfg)

    def create_conv_bn(self, kernel_size, dilation=1, padding=0):
        conv_bn = Sequential()
        conv_bn.add_module(
            'conv',
            build_conv_layer(
                self.conv_cfg,
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=kernel_size,
                stride=self.stride,
                dilation=dilation,
                padding=padding,
                groups=self.groups,
                bias=False))
        conv_bn.add_module(
            'norm',
            build_norm_layer(self.norm_cfg, num_features=self.out_channels)[1])

        return conv_bn

    def forward(self, x):

        def _inner_forward(inputs):
            if self.deploy:
                return self.branch_reparam(inputs)

            if self.branch_norm is None:
                branch_norm_out = 0
            else:
                branch_norm_out = self.branch_norm(inputs)

            inner_out = self.branch_3x3(inputs) + self.branch_1x1(
                inputs) + branch_norm_out

            if self.se_cfg is not None:
                inner_out = self.se_layer(inner_out)

            return inner_out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        out = self.act(out)

        return out

    def switch_to_deploy(self):
        """Switch the model structure from training mode to deployment mode."""
        if self.deploy:
            return
        assert self.norm_cfg['type'] == 'BN', \
            "Switch is not allowed when norm_cfg['type'] != 'BN'."

        reparam_weight, reparam_bias = self.reparameterize()
        self.branch_reparam = build_conv_layer(
            self.conv_cfg,
            self.in_channels,
            self.out_channels,
            kernel_size=3,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
            bias=True)
        self.branch_reparam.weight.data = reparam_weight
        self.branch_reparam.bias.data = reparam_bias

        for param in self.parameters():
            param.detach_()
        delattr(self, 'branch_3x3')
        delattr(self, 'branch_1x1')
        delattr(self, 'branch_norm')

        self.deploy = True

    def reparameterize(self):
        """Fuse all the parameters of all branches.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Parameters after fusion of all
                branches. the first element is the weights and the second is
                the bias.
        """
        weight_3x3, bias_3x3 = self._fuse_conv_bn(self.branch_3x3)
        weight_1x1, bias_1x1 = self._fuse_conv_bn(self.branch_1x1)
        # pad a conv1x1 weight to a conv3x3 weight
        weight_1x1 = F.pad(weight_1x1, [1, 1, 1, 1], value=0)

        weight_norm, bias_norm = 0, 0
        if self.branch_norm:
            tmp_conv_bn = self._norm_to_conv3x3(self.branch_norm)
            weight_norm, bias_norm = self._fuse_conv_bn(tmp_conv_bn)

        return (weight_3x3 + weight_1x1 + weight_norm,
                bias_3x3 + bias_1x1 + bias_norm)

    def _fuse_conv_bn(self, branch):
        """Fuse the parameters in a branch with a conv and bn.

        Args:
            branch (mmcv.runner.Sequential): A branch with conv and bn.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: The parameters obtained after
                fusing the parameters of conv and bn in one branch.
                The first element is the weight and the second is the bias.
        """
        if branch is None:
            return 0, 0
        conv_weight = branch.conv.weight
        running_mean = branch.norm.running_mean
        running_var = branch.norm.running_var
        gamma = branch.norm.weight
        beta = branch.norm.bias
        eps = branch.norm.eps

        std = (running_var + eps).sqrt()
        fused_weight = (gamma / std).reshape(-1, 1, 1, 1) * conv_weight
        fused_bias = -running_mean * gamma / std + beta

        return fused_weight, fused_bias

    def _norm_to_conv3x3(self, branch_nrom):
        """Convert a norm layer to a conv3x3-bn sequence.

        Args:
            branch (nn.BatchNorm2d): A branch only with bn in the block.

        Returns:
            tmp_conv3x3 (mmcv.runner.Sequential): a sequential with conv3x3 and
                bn.
        """
        input_dim = self.in_channels // self.groups
        conv_weight = torch.zeros((self.in_channels, input_dim, 3, 3),
                                  dtype=branch_nrom.weight.dtype)

        for i in range(self.in_channels):
            conv_weight[i, i % input_dim, 1, 1] = 1
        conv_weight = conv_weight.to(branch_nrom.weight.device)

        tmp_conv3x3 = self.create_conv_bn(kernel_size=3)
        tmp_conv3x3.conv.weight.data = conv_weight
        tmp_conv3x3.norm = branch_nrom
        return tmp_conv3x3


class MTSPPF(BaseModule):
    """MTSPPF block for YOLOX-PAI RepVGG backbone.

    Args:
        in_channels (int): The input channels of the block.
        out_channels (int): The output channels of the block.
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: dict(type='BN').
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='ReLU').
        kernel_size (int): Kernel size of pooling. Default: 5.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 kernel_size=5):
        super().__init__()
        hidden_features = in_channels // 2  # hidden channels
        self.conv1 = ConvModule(
            in_channels,
            hidden_features,
            1,
            stride=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.conv2 = ConvModule(
            hidden_features * 4,
            out_channels,
            1,
            stride=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.maxpool = nn.MaxPool2d(
            kernel_size=kernel_size, stride=1, padding=kernel_size // 2)

    def forward(self, x):
        x = self.conv1(x)
        y1 = self.maxpool(x)
        y2 = self.maxpool(y1)
        return self.conv2(torch.cat([x, y1, y2, self.maxpool(y2)], 1))


@MODELS.register_module()
class RepVGG(BaseBackbone):
    """RepVGG backbone.

    A PyTorch impl of : `RepVGG: Making VGG-style ConvNets Great Again
    <https://arxiv.org/abs/2101.03697>`_

    Args:
        arch (str | dict): RepVGG architecture. If use string, choose from
            'A0', 'A1`', 'A2', 'B0', 'B1', 'B1g2', 'B1g4', 'B2', 'B2g2',
            'B2g4', 'B3', 'B3g2', 'B3g4'  or 'D2se'. If use dict, it should
            have below keys:

            - **num_blocks** (Sequence[int]): Number of blocks in each stage.
            - **width_factor** (Sequence[float]): Width deflator in each stage.
            - **group_layer_map** (dict | None): RepVGG Block that declares
              the need to apply group convolution.
            - **se_cfg** (dict | None): SE Layer config.
            - **stem_channels** (int, optional): The stem channels, the final
              stem channels will be
              ``min(stem_channels, base_channels*width_factor[0])``.
              If not set here, 64 is used by default in the code.

        in_channels (int): Number of input image channels. Defaults to 3.
        base_channels (int): Base channels of RepVGG backbone, work with
            width_factor together. Defaults to 64.
        out_indices (Sequence[int]): Output from which stages.
            Defaults to ``(3, )``.
        strides (Sequence[int]): Strides of the first block of each stage.
            Defaults to ``(2, 2, 2, 2)``.
        dilations (Sequence[int]): Dilation of each stage.
            Defaults to ``(1, 1, 1, 1)``.
        frozen_stages (int): Stages to be frozen (all param fixed). -1 means
            not freezing any parameters. Defaults to -1.
        conv_cfg (dict | None): The config dict for conv layers.
            Defaults to None.
        norm_cfg (dict): The config dict for norm layers.
            Defaults to ``dict(type='BN')``.
        act_cfg (dict): Config dict for activation layer.
            Defaults to ``dict(type='ReLU')``.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Defaults to False.
        deploy (bool): Whether to switch the model structure to deployment
            mode. Defaults to False.
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only. Defaults to False.
        add_ppf (bool): Whether to use the MTSPPF block. Defaults to False.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Defaults to None.
    """

    groupwise_layers = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26]
    g2_layer_map = {layer: 2 for layer in groupwise_layers}
    g4_layer_map = {layer: 4 for layer in groupwise_layers}

    arch_settings = {
        'A0':
        dict(
            num_blocks=[2, 4, 14, 1],
            width_factor=[0.75, 0.75, 0.75, 2.5],
            group_layer_map=None,
            se_cfg=None),
        'A1':
        dict(
            num_blocks=[2, 4, 14, 1],
            width_factor=[1, 1, 1, 2.5],
            group_layer_map=None,
            se_cfg=None),
        'A2':
        dict(
            num_blocks=[2, 4, 14, 1],
            width_factor=[1.5, 1.5, 1.5, 2.75],
            group_layer_map=None,
            se_cfg=None),
        'B0':
        dict(
            num_blocks=[4, 6, 16, 1],
            width_factor=[1, 1, 1, 2.5],
            group_layer_map=None,
            se_cfg=None,
            stem_channels=64),
        'B1':
        dict(
            num_blocks=[4, 6, 16, 1],
            width_factor=[2, 2, 2, 4],
            group_layer_map=None,
            se_cfg=None),
        'B1g2':
        dict(
            num_blocks=[4, 6, 16, 1],
            width_factor=[2, 2, 2, 4],
            group_layer_map=g2_layer_map,
            se_cfg=None),
        'B1g4':
        dict(
            num_blocks=[4, 6, 16, 1],
            width_factor=[2, 2, 2, 4],
            group_layer_map=g4_layer_map,
            se_cfg=None),
        'B2':
        dict(
            num_blocks=[4, 6, 16, 1],
            width_factor=[2.5, 2.5, 2.5, 5],
            group_layer_map=None,
            se_cfg=None),
        'B2g2':
        dict(
            num_blocks=[4, 6, 16, 1],
            width_factor=[2.5, 2.5, 2.5, 5],
            group_layer_map=g2_layer_map,
            se_cfg=None),
        'B2g4':
        dict(
            num_blocks=[4, 6, 16, 1],
            width_factor=[2.5, 2.5, 2.5, 5],
            group_layer_map=g4_layer_map,
            se_cfg=None),
        'B3':
        dict(
            num_blocks=[4, 6, 16, 1],
            width_factor=[3, 3, 3, 5],
            group_layer_map=None,
            se_cfg=None),
        'B3g2':
        dict(
            num_blocks=[4, 6, 16, 1],
            width_factor=[3, 3, 3, 5],
            group_layer_map=g2_layer_map,
            se_cfg=None),
        'B3g4':
        dict(
            num_blocks=[4, 6, 16, 1],
            width_factor=[3, 3, 3, 5],
            group_layer_map=g4_layer_map,
            se_cfg=None),
        'D2se':
        dict(
            num_blocks=[8, 14, 24, 1],
            width_factor=[2.5, 2.5, 2.5, 5],
            group_layer_map=None,
            se_cfg=dict(ratio=16, divisor=1)),
        'yolox-pai-small':
        dict(
            num_blocks=[3, 5, 7, 3],
            width_factor=[1, 1, 1, 1],
            group_layer_map=None,
            se_cfg=None,
            stem_channels=32),
    }

    def __init__(self,
                 arch,
                 in_channels=3,
                 base_channels=64,
                 out_indices=(3, ),
                 strides=(2, 2, 2, 2),
                 dilations=(1, 1, 1, 1),
                 frozen_stages=-1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 with_cp=False,
                 deploy=False,
                 norm_eval=False,
                 add_ppf=False,
                 init_cfg=[
                     dict(type='Kaiming', layer=['Conv2d']),
                     dict(
                         type='Constant',
                         val=1,
                         layer=['_BatchNorm', 'GroupNorm'])
                 ]):
        super(RepVGG, self).__init__(init_cfg)

        if isinstance(arch, str):
            assert arch in self.arch_settings, \
                f'"arch": "{arch}" is not one of the arch_settings'
            arch = self.arch_settings[arch]
        elif not isinstance(arch, dict):
            raise TypeError('Expect "arch" to be either a string '
                            f'or a dict, got {type(arch)}')

        assert len(arch['num_blocks']) == len(
            arch['width_factor']) == len(strides) == len(dilations)
        assert max(out_indices) < len(arch['num_blocks'])
        if arch['group_layer_map'] is not None:
            assert max(arch['group_layer_map'].keys()) <= sum(
                arch['num_blocks'])

        if arch['se_cfg'] is not None:
            assert isinstance(arch['se_cfg'], dict)

        self.base_channels = base_channels
        self.arch = arch
        self.in_channels = in_channels
        self.out_indices = out_indices
        self.strides = strides
        self.dilations = dilations
        self.deploy = deploy
        self.frozen_stages = frozen_stages
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.with_cp = with_cp
        self.norm_eval = norm_eval

        # defaults to 64 to prevert BC-breaking if stem_channels
        # not in arch dict;
        # the stem channels should not be larger than that of stage1.
        channels = min(
            arch.get('stem_channels', 64),
            int(self.base_channels * self.arch['width_factor'][0]))
        self.stem = RepVGGBlock(
            self.in_channels,
            channels,
            stride=2,
            se_cfg=arch['se_cfg'],
            with_cp=with_cp,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            deploy=deploy)

        next_create_block_idx = 1
        self.stages = []
        for i in range(len(arch['num_blocks'])):
            num_blocks = self.arch['num_blocks'][i]
            stride = self.strides[i]
            dilation = self.dilations[i]
            out_channels = int(self.base_channels * 2**i *
                               self.arch['width_factor'][i])

            stage, next_create_block_idx = self._make_stage(
                channels, out_channels, num_blocks, stride, dilation,
                next_create_block_idx, init_cfg)
            stage_name = f'stage_{i + 1}'
            self.add_module(stage_name, stage)
            self.stages.append(stage_name)

            channels = out_channels

        if add_ppf:
            self.ppf = MTSPPF(
                out_channels,
                out_channels,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                kernel_size=5)
        else:
            self.ppf = nn.Identity()

    def _make_stage(self, in_channels, out_channels, num_blocks, stride,
                    dilation, next_create_block_idx, init_cfg):
        strides = [stride] + [1] * (num_blocks - 1)
        dilations = [dilation] * num_blocks

        blocks = []
        for i in range(num_blocks):
            groups = self.arch['group_layer_map'].get(
                next_create_block_idx,
                1) if self.arch['group_layer_map'] is not None else 1
            blocks.append(
                RepVGGBlock(
                    in_channels,
                    out_channels,
                    stride=strides[i],
                    padding=dilations[i],
                    dilation=dilations[i],
                    groups=groups,
                    se_cfg=self.arch['se_cfg'],
                    with_cp=self.with_cp,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg,
                    deploy=self.deploy,
                    init_cfg=init_cfg))
            in_channels = out_channels
            next_create_block_idx += 1

        return Sequential(*blocks), next_create_block_idx

    def forward(self, x):
        x = self.stem(x)
        outs = []
        for i, stage_name in enumerate(self.stages):
            stage = getattr(self, stage_name)
            x = stage(x)
            if i + 1 == len(self.stages):
                x = self.ppf(x)
            if i in self.out_indices:
                outs.append(x)

        return tuple(outs)

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.stem.eval()
            for param in self.stem.parameters():
                param.requires_grad = False
        for i in range(self.frozen_stages):
            stage = getattr(self, f'stage_{i+1}')
            stage.eval()
            for param in stage.parameters():
                param.requires_grad = False

    def train(self, mode=True):
        super(RepVGG, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()

    def switch_to_deploy(self):
        for m in self.modules():
            if isinstance(m, RepVGGBlock):
                m.switch_to_deploy()
        self.deploy = True

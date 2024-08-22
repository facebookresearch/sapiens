# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
from mmengine.model import ModuleList, Sequential

from mmpretrain.registry import MODELS
from ..utils import (SparseAvgPooling, SparseConv2d, SparseHelper,
                     SparseMaxPooling, build_norm_layer)
from .convnext import ConvNeXt, ConvNeXtBlock


class SparseConvNeXtBlock(ConvNeXtBlock):
    """Sparse ConvNeXt Block.

    Note:
        There are two equivalent implementations:
        1. DwConv -> SparseLayerNorm -> 1x1 Conv -> GELU -> 1x1 Conv;
           all outputs are in (N, C, H, W).
        2. DwConv -> SparseLayerNorm -> Permute to (N, H, W, C) -> Linear ->
           GELU -> Linear; Permute back
        As default, we use the second to align with the official repository.
        And it may be slightly faster.
    """

    def forward(self, x):

        def _inner_forward(x):
            shortcut = x
            x = self.depthwise_conv(x)

            if self.linear_pw_conv:
                x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
                x = self.norm(x, data_format='channel_last')
                x = self.pointwise_conv1(x)
                x = self.act(x)
                if self.grn is not None:
                    x = self.grn(x, data_format='channel_last')
                x = self.pointwise_conv2(x)
                x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
            else:
                x = self.norm(x, data_format='channel_first')
                x = self.pointwise_conv1(x)
                x = self.act(x)

                if self.grn is not None:
                    x = self.grn(x, data_format='channel_first')
                x = self.pointwise_conv2(x)

            if self.gamma is not None:
                x = x.mul(self.gamma.view(1, -1, 1, 1))

            x *= SparseHelper._get_active_map_or_index(
                H=x.shape[2], returning_active_map=True)

            x = shortcut + self.drop_path(x)
            return x

        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)
        return x


@MODELS.register_module()
class SparseConvNeXt(ConvNeXt):
    """ConvNeXt with sparse module conversion function.

    Modified from
    https://github.com/keyu-tian/SparK/blob/main/models/convnext.py
    and
    https://github.com/keyu-tian/SparK/blob/main/encoder.py
    To use ConvNeXt v2, please set ``use_grn=True`` and ``layer_scale_init_value=0.``.

    Args:
        arch (str | dict): The model's architecture. If string, it should be
            one of architecture in ``ConvNeXt.arch_settings``. And if dict, it
            should include the following two keys:
            - depths (list[int]): Number of blocks at each stage.
            - channels (list[int]): The number of channels at each stage.
            Defaults to 'tiny'.
        in_channels (int): Number of input image channels. Defaults to 3.
        stem_patch_size (int): The size of one patch in the stem layer.
            Defaults to 4.
        norm_cfg (dict): The config dict for norm layers.
            Defaults to ``dict(type='SparseLN2d', eps=1e-6)``.
        act_cfg (dict): The config dict for activation between pointwise
            convolution. Defaults to ``dict(type='GELU')``.
        linear_pw_conv (bool): Whether to use linear layer to do pointwise
            convolution. Defaults to True.
        use_grn (bool): Whether to add Global Response Normalization in the
            blocks. Defaults to False.
        drop_path_rate (float): Stochastic depth rate. Defaults to 0.
        layer_scale_init_value (float): Init value for Layer Scale.
            Defaults to 1e-6.
        out_indices (Sequence | int): Output from which stages.
            Defaults to -1, means the last stage.
        frozen_stages (int): Stages to be frozen (all param fixed).
            Defaults to 0, which means not freezing any parameters.
        gap_before_output (bool): Whether to globally average the feature
            map before the final norm layer. In the official repo, it's only
            used in classification task. Defaults to True.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Defaults to False.
        init_cfg (dict, optional): Initialization config dict.
    """  # noqa: E501

    def __init__(self,
                 arch: str = 'small',
                 in_channels: int = 3,
                 stem_patch_size: int = 4,
                 norm_cfg: dict = dict(type='SparseLN2d', eps=1e-6),
                 act_cfg: dict = dict(type='GELU'),
                 linear_pw_conv: bool = True,
                 use_grn: bool = False,
                 drop_path_rate: float = 0,
                 layer_scale_init_value: float = 1e-6,
                 out_indices: int = -1,
                 frozen_stages: int = 0,
                 gap_before_output: bool = True,
                 with_cp: bool = False,
                 init_cfg: Optional[Union[dict, List[dict]]] = [
                     dict(
                         type='TruncNormal',
                         layer=['Conv2d', 'Linear'],
                         std=.02,
                         bias=0.),
                     dict(
                         type='Constant', layer=['LayerNorm'], val=1.,
                         bias=0.),
                 ]):
        super(ConvNeXt, self).__init__(init_cfg=init_cfg)

        if isinstance(arch, str):
            assert arch in self.arch_settings, \
                f'Unavailable arch, please choose from ' \
                f'({set(self.arch_settings)}) or pass a dict.'
            arch = self.arch_settings[arch]
        elif isinstance(arch, dict):
            assert 'depths' in arch and 'channels' in arch, \
                f'The arch dict must have "depths" and "channels", ' \
                f'but got {list(arch.keys())}.'

        self.depths = arch['depths']
        self.channels = arch['channels']
        assert (isinstance(self.depths, Sequence)
                and isinstance(self.channels, Sequence)
                and len(self.depths) == len(self.channels)), \
            f'The "depths" ({self.depths}) and "channels" ({self.channels}) ' \
            'should be both sequence with the same length.'

        self.num_stages = len(self.depths)

        if isinstance(out_indices, int):
            out_indices = [out_indices]
        assert isinstance(out_indices, Sequence), \
            f'"out_indices" must by a sequence or int, ' \
            f'get {type(out_indices)} instead.'
        for i, index in enumerate(out_indices):
            if index < 0:
                out_indices[i] = 4 + index
                assert out_indices[i] >= 0, f'Invalid out_indices {index}'
        self.out_indices = out_indices

        self.frozen_stages = frozen_stages
        self.gap_before_output = gap_before_output

        # 4 downsample layers between stages, including the stem layer.
        self.downsample_layers = ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(
                in_channels,
                self.channels[0],
                kernel_size=stem_patch_size,
                stride=stem_patch_size),
            build_norm_layer(norm_cfg, self.channels[0]),
        )
        self.downsample_layers.append(stem)

        # stochastic depth decay rule
        dpr = [
            x.item()
            for x in torch.linspace(0, drop_path_rate, sum(self.depths))
        ]
        block_idx = 0

        # 4 feature resolution stages, each consisting of multiple residual
        # blocks
        self.stages = nn.ModuleList()
        for i in range(self.num_stages):
            depth = self.depths[i]
            channels = self.channels[i]

            if i >= 1:
                downsample_layer = nn.Sequential(
                    build_norm_layer(norm_cfg, self.channels[i - 1]),
                    nn.Conv2d(
                        self.channels[i - 1],
                        channels,
                        kernel_size=2,
                        stride=2),
                )
                self.downsample_layers.append(downsample_layer)

            stage = Sequential(*[
                SparseConvNeXtBlock(
                    in_channels=channels,
                    drop_path_rate=dpr[block_idx + j],
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    linear_pw_conv=linear_pw_conv,
                    layer_scale_init_value=layer_scale_init_value,
                    use_grn=use_grn,
                    with_cp=with_cp) for j in range(depth)
            ])
            block_idx += depth

            self.stages.append(stage)

        self.dense_model_to_sparse(m=self)

    def forward(self, x):
        outs = []
        for i, stage in enumerate(self.stages):
            x = self.downsample_layers[i](x)
            x = stage(x)
            if i in self.out_indices:
                if self.gap_before_output:
                    gap = x.mean([-2, -1], keepdim=True)
                    outs.append(gap.flatten(1))
                else:
                    outs.append(x)

        return tuple(outs)

    def dense_model_to_sparse(self, m: nn.Module) -> nn.Module:
        """Convert regular dense modules to sparse modules."""
        output = m
        if isinstance(m, nn.Conv2d):
            m: nn.Conv2d
            bias = m.bias is not None
            output = SparseConv2d(
                m.in_channels,
                m.out_channels,
                kernel_size=m.kernel_size,
                stride=m.stride,
                padding=m.padding,
                dilation=m.dilation,
                groups=m.groups,
                bias=bias,
                padding_mode=m.padding_mode,
            )
            output.weight.data.copy_(m.weight.data)
            if bias:
                output.bias.data.copy_(m.bias.data)

        elif isinstance(m, nn.MaxPool2d):
            m: nn.MaxPool2d
            output = SparseMaxPooling(
                m.kernel_size,
                stride=m.stride,
                padding=m.padding,
                dilation=m.dilation,
                return_indices=m.return_indices,
                ceil_mode=m.ceil_mode)

        elif isinstance(m, nn.AvgPool2d):
            m: nn.AvgPool2d
            output = SparseAvgPooling(
                m.kernel_size,
                m.stride,
                m.padding,
                ceil_mode=m.ceil_mode,
                count_include_pad=m.count_include_pad,
                divisor_override=m.divisor_override)

        # elif isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
        #     m: nn.BatchNorm2d
        #     output = (SparseSyncBatchNorm2d
        #               if enable_sync_bn else SparseBatchNorm2d)(
        #                   m.weight.shape[0],
        #                   eps=m.eps,
        #                   momentum=m.momentum,
        #                   affine=m.affine,
        #                   track_running_stats=m.track_running_stats)
        #     output.weight.data.copy_(m.weight.data)
        #     output.bias.data.copy_(m.bias.data)
        #     output.running_mean.data.copy_(m.running_mean.data)
        #     output.running_var.data.copy_(m.running_var.data)
        #     output.num_batches_tracked.data.copy_(m.num_batches_tracked.data)

        for name, child in m.named_children():
            output.add_module(name, self.dense_model_to_sparse(child))
        del m
        return output

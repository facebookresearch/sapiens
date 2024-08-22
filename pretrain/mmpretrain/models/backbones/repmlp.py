# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Adapted from official impl at https://github.com/DingXiaoH/RepMLP.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import (ConvModule, build_activation_layer, build_conv_layer,
                      build_norm_layer)
from mmcv.cnn.bricks.transformer import PatchEmbed as _PatchEmbed
from mmengine.model import BaseModule, ModuleList, Sequential

from mmpretrain.models.utils import SELayer, to_2tuple
from mmpretrain.registry import MODELS


def fuse_bn(conv_or_fc, bn):
    """fuse conv and bn."""
    std = (bn.running_var + bn.eps).sqrt()
    tmp_weight = bn.weight / std
    tmp_weight = tmp_weight.reshape(-1, 1, 1, 1)

    if len(tmp_weight) == conv_or_fc.weight.size(0):
        return (conv_or_fc.weight * tmp_weight,
                bn.bias - bn.running_mean * bn.weight / std)
    else:
        # in RepMLPBlock, dim0 of fc3 weights and fc3_bn weights
        # are different.
        repeat_times = conv_or_fc.weight.size(0) // len(tmp_weight)
        repeated = tmp_weight.repeat_interleave(repeat_times, 0)
        fused_weight = conv_or_fc.weight * repeated
        bias = bn.bias - bn.running_mean * bn.weight / std
        fused_bias = (bias).repeat_interleave(repeat_times, 0)
        return (fused_weight, fused_bias)


class PatchEmbed(_PatchEmbed):
    """Image to Patch Embedding.

    Compared with default Patch Embedding(in ViT), Patch Embedding of RepMLP
     have ReLu and do not convert output tensor into shape (N, L, C).

    Args:
        in_channels (int): The num of input channels. Default: 3
        embed_dims (int): The dimensions of embedding. Default: 768
        conv_type (str): The type of convolution
            to generate patch embedding. Default: "Conv2d".
        kernel_size (int): The kernel_size of embedding conv. Default: 16.
        stride (int): The slide stride of embedding conv.
            Default: 16.
        padding (int | tuple | string): The padding length of
            embedding conv. When it is a string, it means the mode
            of adaptive padding, support "same" and "corner" now.
            Default: "corner".
        dilation (int): The dilation rate of embedding conv. Default: 1.
        bias (bool): Bias of embed conv. Default: True.
        norm_cfg (dict, optional): Config dict for normalization layer.
            Default: None.
        input_size (int | tuple | None): The size of input, which will be
            used to calculate the out size. Only works when `dynamic_size`
            is False. Default: None.
        init_cfg (`mmcv.ConfigDict`, optional): The Config for initialization.
            Default: None.
    """

    def __init__(self, *args, **kwargs):
        super(PatchEmbed, self).__init__(*args, **kwargs)
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Args:
            x (Tensor): Has shape (B, C, H, W). In most case, C is 3.
        Returns:
            tuple: Contains merged results and its spatial shape.
            - x (Tensor): The output tensor.
            - out_size (tuple[int]): Spatial shape of x, arrange as
              (out_h, out_w).
        """

        if self.adaptive_padding:
            x = self.adaptive_padding(x)

        x = self.projection(x)
        if self.norm is not None:
            x = self.norm(x)
        x = self.relu(x)
        out_size = (x.shape[2], x.shape[3])
        return x, out_size


class GlobalPerceptron(SELayer):
    """GlobalPerceptron implemented by using ``mmpretrain.modes.SELayer``.

    Args:
        input_channels (int): The number of input (and output) channels
            in the GlobalPerceptron.
        ratio (int): Squeeze ratio in GlobalPerceptron, the intermediate
            channel will be ``make_divisible(channels // ratio, divisor)``.
    """

    def __init__(self, input_channels: int, ratio: int, **kwargs) -> None:
        super(GlobalPerceptron, self).__init__(
            channels=input_channels,
            ratio=ratio,
            return_weight=True,
            act_cfg=(dict(type='ReLU'), dict(type='Sigmoid')),
            **kwargs)


class RepMLPBlock(BaseModule):
    """Basic RepMLPNet, consists of PartitionPerceptron and GlobalPerceptron.

    Args:
        channels (int): The number of input and the output channels of the
            block.
        path_h (int): The height of patches.
        path_w (int): The weidth of patches.
        reparam_conv_kernels (Squeue(int) | None): The conv kernels in the
            GlobalPerceptron. Default: None.
        globalperceptron_ratio (int): The reducation ratio in the
            GlobalPerceptron. Default: 4.
        num_sharesets (int): The number of sharesets in the
            PartitionPerceptron. Default 1.
        conv_cfg (dict, optional): Config dict for convolution layer.
            Default: None, which means using conv2d.
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: dict(type='BN', requires_grad=True).
        deploy (bool): Whether to switch the model structure to
            deployment mode. Default: False.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """

    def __init__(self,
                 channels,
                 path_h,
                 path_w,
                 reparam_conv_kernels=None,
                 globalperceptron_ratio=4,
                 num_sharesets=1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 deploy=False,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)

        self.deploy = deploy
        self.channels = channels
        self.num_sharesets = num_sharesets
        self.path_h, self.path_w = path_h, path_w
        # the input channel of fc3
        self._path_vec_channles = path_h * path_w * num_sharesets

        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        self.gp = GlobalPerceptron(
            input_channels=channels, ratio=globalperceptron_ratio)

        # using a conv layer to implement a fc layer
        self.fc3 = build_conv_layer(
            conv_cfg,
            in_channels=self._path_vec_channles,
            out_channels=self._path_vec_channles,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=deploy,
            groups=num_sharesets)
        if deploy:
            self.fc3_bn = nn.Identity()
        else:
            norm_layer = build_norm_layer(norm_cfg, num_sharesets)[1]
            self.add_module('fc3_bn', norm_layer)

        self.reparam_conv_kernels = reparam_conv_kernels
        if not deploy and reparam_conv_kernels is not None:
            for k in reparam_conv_kernels:
                conv_branch = ConvModule(
                    in_channels=num_sharesets,
                    out_channels=num_sharesets,
                    kernel_size=k,
                    stride=1,
                    padding=k // 2,
                    norm_cfg=dict(type='BN', requires_grad=True),
                    groups=num_sharesets,
                    act_cfg=None)
                self.__setattr__('repconv{}'.format(k), conv_branch)

    def partition(self, x, h_parts, w_parts):
        # convert (N, C, H, W) to (N, h_parts, w_parts, C, path_h, path_w)
        x = x.reshape(-1, self.channels, h_parts, self.path_h, w_parts,
                      self.path_w)
        x = x.permute(0, 2, 4, 1, 3, 5)
        return x

    def partition_affine(self, x, h_parts, w_parts):
        """perform Partition Perceptron."""
        fc_inputs = x.reshape(-1, self._path_vec_channles, 1, 1)
        out = self.fc3(fc_inputs)
        out = out.reshape(-1, self.num_sharesets, self.path_h, self.path_w)
        out = self.fc3_bn(out)
        out = out.reshape(-1, h_parts, w_parts, self.num_sharesets,
                          self.path_h, self.path_w)
        return out

    def forward(self, inputs):
        # Global Perceptron
        global_vec = self.gp(inputs)

        origin_shape = inputs.size()
        h_parts = origin_shape[2] // self.path_h
        w_parts = origin_shape[3] // self.path_w

        partitions = self.partition(inputs, h_parts, w_parts)

        # Channel Perceptron
        fc3_out = self.partition_affine(partitions, h_parts, w_parts)

        # perform Local Perceptron
        if self.reparam_conv_kernels is not None and not self.deploy:
            conv_inputs = partitions.reshape(-1, self.num_sharesets,
                                             self.path_h, self.path_w)
            conv_out = 0
            for k in self.reparam_conv_kernels:
                conv_branch = self.__getattr__('repconv{}'.format(k))
                conv_out += conv_branch(conv_inputs)
            conv_out = conv_out.reshape(-1, h_parts, w_parts,
                                        self.num_sharesets, self.path_h,
                                        self.path_w)
            fc3_out += conv_out

        # N, h_parts, w_parts, num_sharesets, out_h, out_w
        fc3_out = fc3_out.permute(0, 3, 1, 4, 2, 5)
        out = fc3_out.reshape(*origin_shape)
        out = out * global_vec
        return out

    def get_equivalent_fc3(self):
        """get the equivalent fc3 weight and bias."""
        fc_weight, fc_bias = fuse_bn(self.fc3, self.fc3_bn)
        if self.reparam_conv_kernels is not None:
            largest_k = max(self.reparam_conv_kernels)
            largest_branch = self.__getattr__('repconv{}'.format(largest_k))
            total_kernel, total_bias = fuse_bn(largest_branch.conv,
                                               largest_branch.bn)
            for k in self.reparam_conv_kernels:
                if k != largest_k:
                    k_branch = self.__getattr__('repconv{}'.format(k))
                    kernel, bias = fuse_bn(k_branch.conv, k_branch.bn)
                    total_kernel += F.pad(kernel, [(largest_k - k) // 2] * 4)
                    total_bias += bias
            rep_weight, rep_bias = self._convert_conv_to_fc(
                total_kernel, total_bias)
            final_fc3_weight = rep_weight.reshape_as(fc_weight) + fc_weight
            final_fc3_bias = rep_bias + fc_bias
        else:
            final_fc3_weight = fc_weight
            final_fc3_bias = fc_bias
        return final_fc3_weight, final_fc3_bias

    def local_inject(self):
        """inject the Local Perceptron into Partition Perceptron."""
        self.deploy = True
        #  Locality Injection
        fc3_weight, fc3_bias = self.get_equivalent_fc3()
        #  Remove Local Perceptron
        if self.reparam_conv_kernels is not None:
            for k in self.reparam_conv_kernels:
                self.__delattr__('repconv{}'.format(k))
        self.__delattr__('fc3')
        self.__delattr__('fc3_bn')
        self.fc3 = build_conv_layer(
            self.conv_cfg,
            self._path_vec_channles,
            self._path_vec_channles,
            1,
            1,
            0,
            bias=True,
            groups=self.num_sharesets)
        self.fc3_bn = nn.Identity()
        self.fc3.weight.data = fc3_weight
        self.fc3.bias.data = fc3_bias

    def _convert_conv_to_fc(self, conv_kernel, conv_bias):
        """convert conv_k1 to fc, which is still a conv_k2, and the k2 > k1."""
        in_channels = torch.eye(self.path_h * self.path_w).repeat(
            1, self.num_sharesets).reshape(self.path_h * self.path_w,
                                           self.num_sharesets, self.path_h,
                                           self.path_w).to(conv_kernel.device)
        fc_k = F.conv2d(
            in_channels,
            conv_kernel,
            padding=(conv_kernel.size(2) // 2, conv_kernel.size(3) // 2),
            groups=self.num_sharesets)
        fc_k = fc_k.reshape(self.path_w * self.path_w, self.num_sharesets *
                            self.path_h * self.path_w).t()
        fc_bias = conv_bias.repeat_interleave(self.path_h * self.path_w)
        return fc_k, fc_bias


class RepMLPNetUnit(BaseModule):
    """A basic unit in RepMLPNet : [REPMLPBlock + BN + ConvFFN + BN].

    Args:
        channels (int): The number of input and the output channels of the
            unit.
        path_h (int): The height of patches.
        path_w (int): The weidth of patches.
        reparam_conv_kernels (Squeue(int) | None): The conv kernels in the
            GlobalPerceptron. Default: None.
        globalperceptron_ratio (int): The reducation ratio in the
            GlobalPerceptron. Default: 4.
        num_sharesets (int): The number of sharesets in the
            PartitionPerceptron. Default 1.
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
                 channels,
                 path_h,
                 path_w,
                 reparam_conv_kernels,
                 globalperceptron_ratio,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 ffn_expand=4,
                 num_sharesets=1,
                 deploy=False,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.repmlp_block = RepMLPBlock(
            channels=channels,
            path_h=path_h,
            path_w=path_w,
            reparam_conv_kernels=reparam_conv_kernels,
            globalperceptron_ratio=globalperceptron_ratio,
            num_sharesets=num_sharesets,
            deploy=deploy)
        self.ffn_block = ConvFFN(channels, channels * ffn_expand)
        norm1 = build_norm_layer(norm_cfg, channels)[1]
        self.add_module('norm1', norm1)
        norm2 = build_norm_layer(norm_cfg, channels)[1]
        self.add_module('norm2', norm2)

    def forward(self, x):
        y = x + self.repmlp_block(self.norm1(x))
        out = y + self.ffn_block(self.norm2(y))
        return out


class ConvFFN(nn.Module):
    """ConvFFN implemented by using point-wise convs."""

    def __init__(self,
                 in_channels,
                 hidden_channels=None,
                 out_channels=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='GELU')):
        super().__init__()
        out_features = out_channels or in_channels
        hidden_features = hidden_channels or in_channels
        self.ffn_fc1 = ConvModule(
            in_channels=in_channels,
            out_channels=hidden_features,
            kernel_size=1,
            stride=1,
            padding=0,
            norm_cfg=norm_cfg,
            act_cfg=None)
        self.ffn_fc2 = ConvModule(
            in_channels=hidden_features,
            out_channels=out_features,
            kernel_size=1,
            stride=1,
            padding=0,
            norm_cfg=norm_cfg,
            act_cfg=None)
        self.act = build_activation_layer(act_cfg)

    def forward(self, x):
        x = self.ffn_fc1(x)
        x = self.act(x)
        x = self.ffn_fc2(x)
        return x


@MODELS.register_module()
class RepMLPNet(BaseModule):
    """RepMLPNet backbone.

    A PyTorch impl of : `RepMLP: Re-parameterizing Convolutions into
    Fully-connected Layers for Image Recognition
    <https://arxiv.org/abs/2105.01883>`_

    Args:
        arch (str | dict): RepMLP architecture. If use string, choose
            from 'base' and 'b'. If use dict, it should have below keys:

            - channels (List[int]): Number of blocks in each stage.
            - depths (List[int]): The number of blocks in each branch.
            - sharesets_nums (List[int]): RepVGG Block that declares
              the need to apply group convolution.

        img_size (int | tuple): The size of input image. Defaults: 224.
        in_channels (int): Number of input image channels. Default: 3.
        patch_size (int | tuple): The patch size in patch embedding.
            Defaults to 4.
        out_indices (Sequence[int]): Output from which stages.
            Default: ``(3, )``.
        reparam_conv_kernels (Squeue(int) | None): The conv kernels in the
            GlobalPerceptron. Default: None.
        globalperceptron_ratio (int): The reducation ratio in the
            GlobalPerceptron. Default: 4.
        num_sharesets (int): The number of sharesets in the
            PartitionPerceptron. Default 1.
        conv_cfg (dict | None): The config dict for conv layers. Default: None.
        norm_cfg (dict): The config dict for norm layers.
            Default: dict(type='BN', requires_grad=True).
        patch_cfg (dict): Extra config dict for patch embedding.
            Defaults to an empty dict.
        final_norm (bool): Whether to add a additional layer to normalize
            final feature map. Defaults to True.
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='ReLU').
        deploy (bool): Whether to switch the model structure to deployment
            mode. Default: False.
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """
    arch_zoo = {
        **dict.fromkeys(['b', 'base'],
                        {'channels':       [96, 192, 384, 768],
                         'depths':         [2, 2, 12, 2],
                         'sharesets_nums': [1, 4, 32, 128]}),
    }  # yapf: disable

    num_extra_tokens = 0  # there is no cls-token in RepMLP

    def __init__(self,
                 arch,
                 img_size=224,
                 in_channels=3,
                 patch_size=4,
                 out_indices=(3, ),
                 reparam_conv_kernels=(3, ),
                 globalperceptron_ratio=4,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 patch_cfg=dict(),
                 final_norm=True,
                 deploy=False,
                 init_cfg=None):
        super(RepMLPNet, self).__init__(init_cfg=init_cfg)
        if isinstance(arch, str):
            arch = arch.lower()
            assert arch in set(self.arch_zoo), \
                f'Arch {arch} is not in default archs {set(self.arch_zoo)}'
            self.arch_settings = self.arch_zoo[arch]
        else:
            essential_keys = {'channels', 'depths', 'sharesets_nums'}
            assert isinstance(arch, dict) and set(arch) == essential_keys, \
                f'Custom arch needs a dict with keys {essential_keys}.'
            self.arch_settings = arch

        self.img_size = to_2tuple(img_size)
        self.patch_size = to_2tuple(patch_size)
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        self.num_stage = len(self.arch_settings['channels'])
        for value in self.arch_settings.values():
            assert isinstance(value, list) and len(value) == self.num_stage, (
                'Length of setting item in arch dict must be type of list and'
                ' have the same length.')

        self.channels = self.arch_settings['channels']
        self.depths = self.arch_settings['depths']
        self.sharesets_nums = self.arch_settings['sharesets_nums']

        _patch_cfg = dict(
            in_channels=in_channels,
            input_size=self.img_size,
            embed_dims=self.channels[0],
            conv_type='Conv2d',
            kernel_size=self.patch_size,
            stride=self.patch_size,
            norm_cfg=self.norm_cfg,
            bias=False)
        _patch_cfg.update(patch_cfg)
        self.patch_embed = PatchEmbed(**_patch_cfg)
        self.patch_resolution = self.patch_embed.init_out_size

        self.patch_hs = [
            self.patch_resolution[0] // 2**i for i in range(self.num_stage)
        ]
        self.patch_ws = [
            self.patch_resolution[1] // 2**i for i in range(self.num_stage)
        ]

        self.stages = ModuleList()
        self.downsample_layers = ModuleList()
        for stage_idx in range(self.num_stage):
            # make stage layers
            _stage_cfg = dict(
                channels=self.channels[stage_idx],
                path_h=self.patch_hs[stage_idx],
                path_w=self.patch_ws[stage_idx],
                reparam_conv_kernels=reparam_conv_kernels,
                globalperceptron_ratio=globalperceptron_ratio,
                norm_cfg=self.norm_cfg,
                ffn_expand=4,
                num_sharesets=self.sharesets_nums[stage_idx],
                deploy=deploy)
            stage_blocks = [
                RepMLPNetUnit(**_stage_cfg)
                for _ in range(self.depths[stage_idx])
            ]
            self.stages.append(Sequential(*stage_blocks))

            # make downsample layers
            if stage_idx < self.num_stage - 1:
                self.downsample_layers.append(
                    ConvModule(
                        in_channels=self.channels[stage_idx],
                        out_channels=self.channels[stage_idx + 1],
                        kernel_size=2,
                        stride=2,
                        padding=0,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        inplace=True))

        self.out_indice = out_indices

        if final_norm:
            norm_layer = build_norm_layer(norm_cfg, self.channels[-1])[1]
        else:
            norm_layer = nn.Identity()
        self.add_module('final_norm', norm_layer)

    def forward(self, x):
        assert x.shape[2:] == self.img_size, \
            "The Rep-MLP doesn't support dynamic input shape. " \
            f'Please input images with shape {self.img_size}'

        outs = []

        x, _ = self.patch_embed(x)
        for i, stage in enumerate(self.stages):
            x = stage(x)

            # downsample after each stage except last stage
            if i < len(self.stages) - 1:
                downsample = self.downsample_layers[i]
                x = downsample(x)

            if i in self.out_indice:
                if self.final_norm and i == len(self.stages) - 1:
                    out = self.final_norm(x)
                else:
                    out = x
                outs.append(out)

        return tuple(outs)

    def switch_to_deploy(self):
        for m in self.modules():
            if hasattr(m, 'local_inject'):
                m.local_inject()

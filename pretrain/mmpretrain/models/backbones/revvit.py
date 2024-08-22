# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import sys

import numpy as np
import torch
from mmcv.cnn.bricks.drop import build_dropout
from mmcv.cnn.bricks.transformer import FFN, PatchEmbed
from mmengine.model import BaseModule, ModuleList
from mmengine.model.weight_init import trunc_normal_
from torch import nn
from torch.autograd import Function as Function

from mmpretrain.models.backbones.base_backbone import BaseBackbone
from mmpretrain.registry import MODELS
from ..utils import (MultiheadAttention, build_norm_layer, resize_pos_embed,
                     to_2tuple)


class RevBackProp(Function):
    """Custom Backpropagation function to allow (A) flushing memory in forward
    and (B) activation recomputation reversibly in backward for gradient
    calculation.

    Inspired by
    https://github.com/RobinBruegger/RevTorch/blob/master/revtorch/revtorch.py
    """

    @staticmethod
    def forward(
            ctx,
            x,
            layers,
            buffer_layers,  # List of layer ids for int activation to buffer
    ):
        """Reversible Forward pass.

        Any intermediate activations from `buffer_layers` are cached in ctx for
        forward pass. This is not necessary for standard usecases. Each
        reversible layer implements its own forward pass logic.
        """
        buffer_layers.sort()
        x1, x2 = torch.chunk(x, 2, dim=-1)
        intermediate = []

        for layer in layers:
            x1, x2 = layer(x1, x2)
            if layer.layer_id in buffer_layers:
                intermediate.extend([x1.detach(), x2.detach()])

        if len(buffer_layers) == 0:
            all_tensors = [x1.detach(), x2.detach()]
        else:
            intermediate = [torch.LongTensor(buffer_layers), *intermediate]
            all_tensors = [x1.detach(), x2.detach(), *intermediate]

        ctx.save_for_backward(*all_tensors)
        ctx.layers = layers

        return torch.cat([x1, x2], dim=-1)

    @staticmethod
    def backward(ctx, dx):
        """Reversible Backward pass.

        Any intermediate activations from `buffer_layers` are recovered from
        ctx. Each layer implements its own loic for backward pass (both
        activation recomputation and grad calculation).
        """
        d_x1, d_x2 = torch.chunk(dx, 2, dim=-1)
        # retrieve params from ctx for backward
        x1, x2, *int_tensors = ctx.saved_tensors
        # no buffering
        if len(int_tensors) != 0:
            buffer_layers = int_tensors[0].tolist()
        else:
            buffer_layers = []

        layers = ctx.layers

        for _, layer in enumerate(layers[::-1]):
            if layer.layer_id in buffer_layers:
                x1, x2, d_x1, d_x2 = layer.backward_pass(
                    y1=int_tensors[buffer_layers.index(layer.layer_id) * 2 +
                                   1],
                    y2=int_tensors[buffer_layers.index(layer.layer_id) * 2 +
                                   2],
                    d_y1=d_x1,
                    d_y2=d_x2,
                )
            else:
                x1, x2, d_x1, d_x2 = layer.backward_pass(
                    y1=x1,
                    y2=x2,
                    d_y1=d_x1,
                    d_y2=d_x2,
                )

        dx = torch.cat([d_x1, d_x2], dim=-1)

        del int_tensors
        del d_x1, d_x2, x1, x2

        return dx, None, None


class RevTransformerEncoderLayer(BaseModule):
    """Reversible Transformer Encoder Layer.

    This module is a building block of Reversible Transformer Encoder,
    which support backpropagation without storing activations.
    The residual connection is not applied to the FFN layer.

    Args:
        embed_dims (int): The feature dimension.
        num_heads (int): Parallel attention heads.
        feedforward_channels (int): The hidden dimension for FFNs.
        drop_rate (float): Probability of an element to be zeroed.
            Default: 0.0
        attn_drop_rate (float): The drop out rate for attention layer.
            Default: 0.0
        drop_path_rate (float): stochastic depth rate.
            Default 0.0
        num_fcs (int): The number of linear in FFN
            Default: 2
        qkv_bias (bool): enable bias for qkv if True.
            Default: True
        act_cfg (dict): The activation config for FFNs.
            Default: dict(type='GELU')
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='LN')
        layer_id (int): The layer id of current layer. Used in RevBackProp.
            Default: 0
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    def __init__(self,
                 embed_dims: int,
                 num_heads: int,
                 feedforward_channels: int,
                 drop_rate: float = 0.,
                 attn_drop_rate: float = 0.,
                 drop_path_rate: float = 0.,
                 num_fcs: int = 2,
                 qkv_bias: bool = True,
                 act_cfg: dict = dict(type='GELU'),
                 norm_cfg: dict = dict(type='LN'),
                 layer_id: int = 0,
                 init_cfg=None):
        super(RevTransformerEncoderLayer, self).__init__(init_cfg=init_cfg)

        self.drop_path_cfg = dict(type='DropPath', drop_prob=drop_path_rate)
        self.embed_dims = embed_dims

        self.ln1 = build_norm_layer(norm_cfg, self.embed_dims)

        self.attn = MultiheadAttention(
            embed_dims=embed_dims,
            num_heads=num_heads,
            attn_drop=attn_drop_rate,
            proj_drop=drop_rate,
            qkv_bias=qkv_bias)

        self.ln2 = build_norm_layer(norm_cfg, self.embed_dims)

        self.ffn = FFN(
            embed_dims=embed_dims,
            feedforward_channels=feedforward_channels,
            num_fcs=num_fcs,
            ffn_drop=drop_rate,
            act_cfg=act_cfg,
            add_identity=False)

        self.layer_id = layer_id
        self.seeds = {}

    def init_weights(self):
        super(RevTransformerEncoderLayer, self).init_weights()
        for m in self.ffn.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.normal_(m.bias, std=1e-6)

    def seed_cuda(self, key):
        """Fix seeds to allow for stochastic elements such as dropout to be
        reproduced exactly in activation recomputation in the backward pass."""
        # randomize seeds
        # use cuda generator if available
        if (hasattr(torch.cuda, 'default_generators')
                and len(torch.cuda.default_generators) > 0):
            # GPU
            device_idx = torch.cuda.current_device()
            seed = torch.cuda.default_generators[device_idx].seed()
        else:
            # CPU
            seed = int(torch.seed() % sys.maxsize)

        self.seeds[key] = seed
        torch.manual_seed(self.seeds[key])

    def forward(self, x1, x2):
        """
        Implementation of Reversible TransformerEncoderLayer

        `
        x = x + self.attn(self.ln1(x))
        x = self.ffn(self.ln2(x), identity=x)
        `
        """
        self.seed_cuda('attn')
        # attention output
        f_x2 = self.attn(self.ln1(x2))
        # apply droppath on attention output
        self.seed_cuda('droppath')
        f_x2_dropped = build_dropout(self.drop_path_cfg)(f_x2)
        y1 = x1 + f_x2_dropped

        # free memory
        if self.training:
            del x1

        # ffn output
        self.seed_cuda('ffn')
        g_y1 = self.ffn(self.ln2(y1))
        # apply droppath on ffn output
        torch.manual_seed(self.seeds['droppath'])
        g_y1_dropped = build_dropout(self.drop_path_cfg)(g_y1)
        # final output
        y2 = x2 + g_y1_dropped

        # free memory
        if self.training:
            del x2

        return y1, y2

    def backward_pass(self, y1, y2, d_y1, d_y2):
        """Activation re-compute with the following equation.

        x2 = y2 - g(y1), g = FFN
        x1 = y1 - f(x2), f = MSHA
        """

        # temporarily record intermediate activation for G
        # and use them for gradient calculation of G
        with torch.enable_grad():
            y1.requires_grad = True

            torch.manual_seed(self.seeds['ffn'])
            g_y1 = self.ffn(self.ln2(y1))

            torch.manual_seed(self.seeds['droppath'])
            g_y1 = build_dropout(self.drop_path_cfg)(g_y1)

            g_y1.backward(d_y2, retain_graph=True)

        # activate recomputation is by design and not part of
        # the computation graph in forward pass
        with torch.no_grad():
            x2 = y2 - g_y1
            del g_y1

            d_y1 = d_y1 + y1.grad
            y1.grad = None

        # record F activation and calculate gradients on F
        with torch.enable_grad():
            x2.requires_grad = True

            torch.manual_seed(self.seeds['attn'])
            f_x2 = self.attn(self.ln1(x2))

            torch.manual_seed(self.seeds['droppath'])
            f_x2 = build_dropout(self.drop_path_cfg)(f_x2)

            f_x2.backward(d_y1, retain_graph=True)

        # propagate reverse computed activations at the
        # start of the previous block
        with torch.no_grad():
            x1 = y1 - f_x2
            del f_x2, y1

            d_y2 = d_y2 + x2.grad

            x2.grad = None
            x2 = x2.detach()

        return x1, x2, d_y1, d_y2


class TwoStreamFusion(nn.Module):
    """A general constructor for neural modules fusing two equal sized tensors
    in forward.

    Args:
        mode (str): The mode of fusion. Options are 'add', 'max', 'min',
            'avg', 'concat'.
    """

    def __init__(self, mode: str):
        super().__init__()
        self.mode = mode

        if mode == 'add':
            self.fuse_fn = lambda x: torch.stack(x).sum(dim=0)
        elif mode == 'max':
            self.fuse_fn = lambda x: torch.stack(x).max(dim=0).values
        elif mode == 'min':
            self.fuse_fn = lambda x: torch.stack(x).min(dim=0).values
        elif mode == 'avg':
            self.fuse_fn = lambda x: torch.stack(x).mean(dim=0)
        elif mode == 'concat':
            self.fuse_fn = lambda x: torch.cat(x, dim=-1)
        else:
            raise NotImplementedError

    def forward(self, x):
        # split the tensor into two halves in the channel dimension
        x = torch.chunk(x, 2, dim=2)
        return self.fuse_fn(x)


@MODELS.register_module()
class RevVisionTransformer(BaseBackbone):
    """Reversible Vision Transformer.

    A PyTorch implementation of : `Reversible Vision Transformers
    <https://openaccess.thecvf.com/content/CVPR2022/html/Mangalam_Reversible_Vision_Transformers_CVPR_2022_paper.html>`_ # noqa: E501

    Args:
        arch (str | dict): Vision Transformer architecture. If use string,
            choose from 'small', 'base', 'large', 'deit-tiny', 'deit-small'
            and 'deit-base'. If use dict, it should have below keys:

            - **embed_dims** (int): The dimensions of embedding.
            - **num_layers** (int): The number of transformer encoder layers.
            - **num_heads** (int): The number of heads in attention modules.
            - **feedforward_channels** (int): The hidden dimensions in
              feedforward modules.

            Defaults to 'base'.
        img_size (int | tuple): The expected input image shape. Because we
            support dynamic input shape, just set the argument to the most
            common input image shape. Defaults to 224.
        patch_size (int | tuple): The patch size in patch embedding.
            Defaults to 16.
        in_channels (int): The num of input channels. Defaults to 3.
        drop_rate (float): Probability of an element to be zeroed.
            Defaults to 0.
        drop_path_rate (float): stochastic depth rate. Defaults to 0.
        qkv_bias (bool): Whether to add bias for qkv in attention modules.
            Defaults to True.
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to ``dict(type='LN')``.
        final_norm (bool): Whether to add a additional layer to normalize
            final feature map. Defaults to True.
        out_type (str): The type of output features. Please choose from

            - ``"cls_token"``: The class token tensor with shape (B, C).
            - ``"featmap"``: The feature map tensor from the patch tokens
              with shape (B, C, H, W).
            - ``"avg_featmap"``: The global averaged feature map tensor
              with shape (B, C).
            - ``"raw"``: The raw feature tensor includes patch tokens and
              class tokens with shape (B, L, C).

            Defaults to ``"avg_featmap"``.
        with_cls_token (bool): Whether concatenating class token into image
            tokens as transformer input. Defaults to False.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters. Defaults to -1.
        interpolate_mode (str): Select the interpolate mode for position
            embeding vector resize. Defaults to "bicubic".
        patch_cfg (dict): Configs of patch embeding. Defaults to an empty dict.
        layer_cfgs (Sequence | dict): Configs of each transformer layer in
            encoder. Defaults to an empty dict.
        fusion_mode (str): The fusion mode of transformer layers.
            Defaults to 'concat'.
        no_custom_backward (bool): Whether to use custom backward.
            Defaults to False.
        init_cfg (dict, optional): Initialization config dict.
            Defaults to None.
    """
    arch_zoo = {
        **dict.fromkeys(
            ['s', 'small'], {
                'embed_dims': 768,
                'num_layers': 8,
                'num_heads': 8,
                'feedforward_channels': 768 * 3,
            }),
        **dict.fromkeys(
            ['b', 'base'], {
                'embed_dims': 768,
                'num_layers': 12,
                'num_heads': 12,
                'feedforward_channels': 3072
            }),
        **dict.fromkeys(
            ['l', 'large'], {
                'embed_dims': 1024,
                'num_layers': 24,
                'num_heads': 16,
                'feedforward_channels': 4096
            }),
        **dict.fromkeys(
            ['h', 'huge'],
            {
                # The same as the implementation in MAE
                # <https://arxiv.org/abs/2111.06377>
                'embed_dims': 1280,
                'num_layers': 32,
                'num_heads': 16,
                'feedforward_channels': 5120
            }),
        **dict.fromkeys(
            ['deit-t', 'deit-tiny'], {
                'embed_dims': 192,
                'num_layers': 12,
                'num_heads': 3,
                'feedforward_channels': 192 * 4
            }),
        **dict.fromkeys(
            ['deit-s', 'deit-small'], {
                'embed_dims': 384,
                'num_layers': 12,
                'num_heads': 6,
                'feedforward_channels': 384 * 4
            }),
        **dict.fromkeys(
            ['deit-b', 'deit-base'], {
                'embed_dims': 768,
                'num_layers': 12,
                'num_heads': 12,
                'feedforward_channels': 768 * 4
            }),
    }
    num_extra_tokens = 0  # The official RevViT doesn't have class token
    OUT_TYPES = {'raw', 'cls_token', 'featmap', 'avg_featmap'}

    def __init__(self,
                 arch='base',
                 img_size=224,
                 patch_size=16,
                 in_channels=3,
                 drop_rate=0.,
                 drop_path_rate=0.,
                 qkv_bias=True,
                 norm_cfg=dict(type='LN', eps=1e-6),
                 final_norm=True,
                 out_type='avg_featmap',
                 with_cls_token=False,
                 frozen_stages=-1,
                 interpolate_mode='bicubic',
                 patch_cfg=dict(),
                 layer_cfgs=dict(),
                 fusion_mode='concat',
                 no_custom_backward=False,
                 init_cfg=None):
        super(RevVisionTransformer, self).__init__(init_cfg)

        if isinstance(arch, str):
            arch = arch.lower()
            assert arch in set(self.arch_zoo), \
                f'Arch {arch} is not in default archs {set(self.arch_zoo)}'
            self.arch_settings = self.arch_zoo[arch]
        else:
            essential_keys = {
                'embed_dims', 'num_layers', 'num_heads', 'feedforward_channels'
            }
            assert isinstance(arch, dict) and essential_keys <= set(arch), \
                f'Custom arch needs a dict with keys {essential_keys}'
            self.arch_settings = arch

        self.embed_dims = self.arch_settings['embed_dims']
        self.num_layers = self.arch_settings['num_layers']
        self.img_size = to_2tuple(img_size)
        self.no_custom_backward = no_custom_backward

        # Set patch embedding
        _patch_cfg = dict(
            in_channels=in_channels,
            input_size=img_size,
            embed_dims=self.embed_dims,
            conv_type='Conv2d',
            kernel_size=patch_size,
            stride=patch_size,
        )
        _patch_cfg.update(patch_cfg)
        self.patch_embed = PatchEmbed(**_patch_cfg)
        self.patch_resolution = self.patch_embed.init_out_size
        num_patches = self.patch_resolution[0] * self.patch_resolution[1]

        # Set out type
        if out_type not in self.OUT_TYPES:
            raise ValueError(f'Unsupported `out_type` {out_type}, please '
                             f'choose from {self.OUT_TYPES}')
        self.out_type = out_type

        # Set cls token
        if with_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dims))
            self.num_extra_tokens = 1
        elif out_type != 'cls_token':
            self.cls_token = None
            self.num_extra_tokens = 0
        else:
            raise ValueError(
                'with_cls_token must be True when `out_type="cls_token"`.')

        # Set position embedding
        self.interpolate_mode = interpolate_mode
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + self.num_extra_tokens,
                        self.embed_dims))
        self._register_load_state_dict_pre_hook(self._prepare_pos_embed)

        self.drop_after_pos = nn.Dropout(p=drop_rate)

        # stochastic depth decay rule
        dpr = np.linspace(0, drop_path_rate, self.num_layers)

        self.layers = ModuleList()
        if isinstance(layer_cfgs, dict):
            layer_cfgs = [layer_cfgs] * self.num_layers
        for i in range(self.num_layers):
            _layer_cfg = dict(
                embed_dims=self.embed_dims,
                num_heads=self.arch_settings['num_heads'],
                feedforward_channels=self.
                arch_settings['feedforward_channels'],
                drop_rate=drop_rate,
                drop_path_rate=dpr[i],
                qkv_bias=qkv_bias,
                layer_id=i,
                norm_cfg=norm_cfg)
            _layer_cfg.update(layer_cfgs[i])
            self.layers.append(RevTransformerEncoderLayer(**_layer_cfg))

        # fusion operation for the final output
        self.fusion_layer = TwoStreamFusion(mode=fusion_mode)

        self.frozen_stages = frozen_stages
        self.final_norm = final_norm
        if final_norm:
            self.ln1 = build_norm_layer(norm_cfg, self.embed_dims * 2)

        # freeze stages only when self.frozen_stages > 0
        if self.frozen_stages > 0:
            self._freeze_stages()

    def init_weights(self):
        super(RevVisionTransformer, self).init_weights()
        if not (isinstance(self.init_cfg, dict)
                and self.init_cfg['type'] == 'Pretrained'):
            trunc_normal_(self.pos_embed, std=0.02)

    def _prepare_pos_embed(self, state_dict, prefix, *args, **kwargs):
        name = prefix + 'pos_embed'
        if name not in state_dict.keys():
            return

        ckpt_pos_embed_shape = state_dict[name].shape
        if self.pos_embed.shape != ckpt_pos_embed_shape:
            from mmengine.logging import MMLogger
            logger = MMLogger.get_current_instance()
            logger.info(
                f'Resize the pos_embed shape from {ckpt_pos_embed_shape} '
                f'to {self.pos_embed.shape}.')

            ckpt_pos_embed_shape = to_2tuple(
                int(np.sqrt(ckpt_pos_embed_shape[1] - self.num_extra_tokens)))
            pos_embed_shape = self.patch_embed.init_out_size

            state_dict[name] = resize_pos_embed(state_dict[name],
                                                ckpt_pos_embed_shape,
                                                pos_embed_shape,
                                                self.interpolate_mode,
                                                self.num_extra_tokens)

    @staticmethod
    def resize_pos_embed(*args, **kwargs):
        """Interface for backward-compatibility."""
        return resize_pos_embed(*args, **kwargs)

    def _freeze_stages(self):
        # freeze position embedding
        self.pos_embed.requires_grad = False
        # set dropout to eval model
        self.drop_after_pos.eval()
        # freeze patch embedding
        self.patch_embed.eval()
        for param in self.patch_embed.parameters():
            param.requires_grad = False
        # freeze cls_token
        if self.cls_token is not None:
            self.cls_token.requires_grad = False
        # freeze layers
        for i in range(1, self.frozen_stages + 1):
            m = self.layers[i - 1]
            m.eval()
            for param in m.parameters():
                param.requires_grad = False
        # freeze the last layer norm
        if self.frozen_stages == len(self.layers) and self.final_norm:
            self.ln1.eval()
            for param in self.ln1.parameters():
                param.requires_grad = False

    def forward(self, x):
        B = x.shape[0]
        x, patch_resolution = self.patch_embed(x)

        if self.cls_token is not None:
            cls_token = self.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_token, x), dim=1)

        x = x + resize_pos_embed(
            self.pos_embed,
            self.patch_resolution,
            patch_resolution,
            mode=self.interpolate_mode,
            num_extra_tokens=self.num_extra_tokens)
        x = self.drop_after_pos(x)

        x = torch.cat([x, x], dim=-1)

        # forward with different conditions
        if not self.training or self.no_custom_backward:
            # in eval/inference model
            executing_fn = RevVisionTransformer._forward_vanilla_bp
        else:
            # use custom backward when self.training=True.
            executing_fn = RevBackProp.apply

        x = executing_fn(x, self.layers, [])

        if self.final_norm:
            x = self.ln1(x)
        x = self.fusion_layer(x)

        return (self._format_output(x, patch_resolution), )

    @staticmethod
    def _forward_vanilla_bp(hidden_state, layers, buffer=[]):
        """Using reversible layers without reversible backpropagation.

        Debugging purpose only. Activated with self.no_custom_backward
        """
        # split into ffn state(ffn_out) and attention output(attn_out)
        ffn_out, attn_out = torch.chunk(hidden_state, 2, dim=-1)
        del hidden_state

        for _, layer in enumerate(layers):
            attn_out, ffn_out = layer(attn_out, ffn_out)

        return torch.cat([attn_out, ffn_out], dim=-1)

    def _format_output(self, x, hw):
        if self.out_type == 'raw':
            return x
        if self.out_type == 'cls_token':
            return x[:, 0]

        patch_token = x[:, self.num_extra_tokens:]
        if self.out_type == 'featmap':
            B = x.size(0)
            # (B, N, C) -> (B, H, W, C) -> (B, C, H, W)
            return patch_token.reshape(B, *hw, -1).permute(0, 3, 1, 2)
        if self.out_type == 'avg_featmap':
            return patch_token.mean(dim=1)

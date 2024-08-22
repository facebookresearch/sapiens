# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass
from functools import partial
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn.bricks import DropPath
from mmengine.model import BaseModule
from mmengine.utils import digit_version
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions, ModelOutput, Seq2SeqLMOutput)
from transformers.modeling_utils import (GenerationConfig, GenerationMixin,
                                         PretrainedConfig)

from mmpretrain.registry import MODELS
from ...backbones.resnet import Bottleneck, ResNet

if digit_version(torch.__version__) >= digit_version('1.10.0'):
    torch_meshgrid = partial(torch.meshgrid, indexing='ij')
else:
    torch_meshgrid = torch.meshgrid


def make_token_bucket_position(bucket_size, max_position=1024):
    context_pos = torch.arange(max_position, dtype=torch.long)[:, None]
    memory_pos = torch.arange(max_position, dtype=torch.long)[None, :]
    relative_pos = context_pos - memory_pos
    sign = torch.sign(relative_pos)
    mid = bucket_size // 2
    abs_pos = torch.where((relative_pos < mid) & (relative_pos > -mid),
                          mid - 1, torch.abs(relative_pos))
    log_pos = torch.ceil(
        torch.log(abs_pos / mid) / math.log(
            (max_position - 1) / mid) * (mid - 1)) + mid
    log_pos = log_pos.int()
    bucket_pos = torch.where(abs_pos.le(mid), relative_pos,
                             log_pos * sign).long()
    return bucket_pos + bucket_size - 1


def make_image_bucket_position(bucket_size, num_relative_distance):
    coords_h = torch.arange(bucket_size)
    coords_w = torch.arange(bucket_size)
    # (2, h, w)
    coords = torch.stack(torch_meshgrid([coords_h, coords_w]))
    # (2, h*w)
    coords_flatten = torch.flatten(coords, 1)
    # (2, h*w, h*w)
    relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
    # (h*w, h*w, 2)
    relative_coords = relative_coords.permute(1, 2, 0).contiguous()
    relative_coords[:, :, 0] += bucket_size - 1  # shift to start from 0
    relative_coords[:, :, 1] += bucket_size - 1
    relative_coords[:, :, 0] *= 2 * bucket_size - 1
    relative_position_index = torch.zeros(
        size=(bucket_size * bucket_size + 1, ) * 2,
        dtype=relative_coords.dtype)
    # (h*w, h*w)
    relative_position_index[1:, 1:] = relative_coords.sum(-1)
    relative_position_index[0, 0:] = num_relative_distance - 3
    relative_position_index[0:, 0] = num_relative_distance - 2
    relative_position_index[0, 0] = num_relative_distance - 1
    return relative_position_index


def _make_causal_mask(input_ids_shape: torch.Size,
                      dtype: torch.dtype,
                      past_key_values_length: int = 0):
    """Make causal mask used for uni-directional self-attention."""
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), float('-inf'))
    mask_cond = torch.arange(mask.size(-1))
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat(
            [torch.zeros(tgt_len, past_key_values_length, dtype=dtype), mask],
            dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len,
                                         tgt_len + past_key_values_length)


def _expand_mask(mask: torch.Tensor,
                 dtype: torch.dtype,
                 tgt_len: Optional[int] = None):
    """Expands attention_mask from ``[B, L_s]`` to ``[B, 1, L_t, L_s]``.

    Where ``B`` is batch_size, `L_s`` is the source sequence length, and
    ``L_t`` is the target sequence length.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len,
                                                  src_len).to(dtype)
    return expanded_mask.masked_fill(expanded_mask.bool(),
                                     torch.finfo(dtype).min)


class MultiheadAttention(BaseModule):
    """Multi-head Attention Module for OFA.

    Args:
        embedding_dim (int): The embedding dimension of query.
        num_heads (int): Parallel attention heads.
        kdim (int, optional): The embedding dimension of key.
            Defaults to None, which means the same as the `embedding_dim`.
        vdim (int, optional): The embedding dimension of value.
            Defaults to None, which means the same as the `embedding_dim`.
        attn_drop (float): Dropout rate of the dropout layer after the
            attention calculation of query and key. Defaults to 0.
        qkv_bias (bool): If True, add a learnable bias to q, k, v.
            Defaults to True.
        scale_factor (float): The scale of qk will be
            ``(head_dim * scale_factor) ** -0.5``. Defaults to 1.
        proj_bias (bool) If True, add a learnable bias to output projection.
            Defaults to True.
        init_cfg (dict, optional): The Config for initialization.
            Defaults to None.
    """

    def __init__(self,
                 embedding_dim,
                 num_heads,
                 kdim=None,
                 vdim=None,
                 attn_drop=0.,
                 scale_factor=1.,
                 qkv_bias=True,
                 proj_bias=True,
                 scale_heads=False,
                 init_cfg=None):
        super(MultiheadAttention, self).__init__(init_cfg=init_cfg)

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.kdim = kdim or embedding_dim
        self.vdim = vdim or embedding_dim

        self.head_dim = embedding_dim // num_heads
        self.scale = (self.head_dim * scale_factor)**-0.5

        self.q_proj = nn.Linear(embedding_dim, embedding_dim, bias=qkv_bias)
        self.k_proj = nn.Linear(self.kdim, embedding_dim, bias=qkv_bias)
        self.v_proj = nn.Linear(self.vdim, embedding_dim, bias=qkv_bias)
        self.out_proj = nn.Linear(embedding_dim, embedding_dim, bias=proj_bias)

        self.attn_drop = nn.Dropout(p=attn_drop)

        if scale_heads:
            self.c_attn = nn.Parameter(torch.ones(num_heads))
        else:
            self.c_attn = None

    def forward(
        self,
        query,
        key_value=None,
        attn_mask=None,
        attn_bias=None,
        past_key_value=None,
        output_attentions=False,
    ):
        B, _, C = query.shape
        assert C == self.head_dim * self.num_heads

        is_cross_attention = key_value is not None
        if key_value is None:
            key_value = query

        # (B, L, C) -> (B, num_heads, L, head_dims)
        q = self.q_proj(query).reshape(B, -1, self.num_heads,
                                       self.head_dim).transpose(1, 2)

        if is_cross_attention and past_key_value is not None:
            # Reuse key and value in cross_attentions
            k, v = past_key_value
        else:
            k = self.k_proj(key_value).reshape(B, -1, self.num_heads,
                                               self.head_dim).transpose(1, 2)
            v = self.v_proj(key_value).reshape(B, -1, self.num_heads,
                                               self.head_dim).transpose(1, 2)
            if past_key_value is not None:
                past_key, past_value = past_key_value
                k = torch.cat([past_key, k], dim=2)
                v = torch.cat([past_value, v], dim=2)

        past_key_value = (k, v)

        attn_weights = q @ k.transpose(-2, -1) * self.scale

        if attn_bias is not None:
            src_len = k.size(2)
            attn_weights[:, :, -src_len:] += attn_bias[:, :, -src_len:]

        if attn_mask is not None:
            attn_weights += attn_mask
        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn = self.attn_drop(attn_weights) @ v

        if self.c_attn is not None:
            attn = torch.einsum('bhlc,h->bhlc', attn, self.c_attn)

        # (B, num_heads, L, head_dims) -> (B, L, C)
        attn = attn.transpose(1, 2).reshape(B, -1, self.embedding_dim)
        attn = self.out_proj(attn)

        if output_attentions:
            return attn, attn_weights, past_key_value
        else:
            return attn, None, past_key_value


@MODELS.register_module(force=True)
class OFAResNet(ResNet):
    """ResNet module for OFA.

    The ResNet in OFA has only three stages.
    """
    arch_settings = {
        50: (Bottleneck, (3, 4, 6)),
        101: (Bottleneck, (3, 4, 23)),
        152: (Bottleneck, (3, 8, 36)),
    }

    def __init__(self, depth, *args, **kwargs):
        super().__init__(
            depth=depth,
            *args,
            num_stages=3,
            out_indices=(2, ),
            dilations=(1, 1, 1),
            strides=(1, 2, 2),
            **kwargs)


@dataclass
class OFAEncoderOutput(ModelOutput):
    """OFA encoder outputs.

    Args:
        last_hidden_state (torch.tensor): The hidden-states of the output at
            the last layer of the model. The shape is (B, L, C).
        hidden_states (Tuple[torch.tensor]): The initial embedding and the
            output of each layer. The shape of every item is (B, L, C).
        attentions (Tuple[torch.tensor]): The attention weights after the
            attention softmax, used to compute the weighted average in the
            self-attention heads. The shape of every item is
            (B, num_heads, L, L).
        position_embedding (torch.tensor): The positional embeddings of the
            inputs. The shape is (B, L, C).
    """

    last_hidden_state: torch.FloatTensor = None
    padding_mask: torch.Tensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    position_embedding: Optional[torch.FloatTensor] = None


class OFAEncoderLayer(nn.Module):
    """OFAEncoder layer block."""

    def __init__(self,
                 embedding_dim,
                 num_heads,
                 dropout_rate=0.,
                 drop_path_rate=0.,
                 attn_drop=0.,
                 act_drop=0.,
                 scale_factor=2.,
                 mlp_ratio=4.,
                 scale_heads=True,
                 normformer=True,
                 pre_norm=True,
                 act_cfg=dict(type='GELU')):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.pre_norm = pre_norm

        self.attn = MultiheadAttention(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            attn_drop=attn_drop,
            scale_factor=scale_factor,
            scale_heads=scale_heads,
        )

        mid_channels = int(embedding_dim * mlp_ratio)
        self.fc1 = nn.Linear(embedding_dim, mid_channels)
        self.fc2 = nn.Linear(mid_channels, embedding_dim)
        self.act = MODELS.build(act_cfg)
        self.act_drop = nn.Dropout(
            act_drop) if act_drop > 0. else nn.Identity()

        # LayerNorm between attention block and ffn block.
        self.attn_ln = nn.LayerNorm(embedding_dim)
        self.ffn_ln = nn.LayerNorm(embedding_dim)

        # Extra LayerNorm
        self.normformer = normformer
        if self.normformer:
            self.attn_mid_ln = nn.LayerNorm(embedding_dim)
            self.ffn_mid_ln = nn.LayerNorm(mid_channels)

        self.dropout = nn.Dropout(dropout_rate)
        self.drop_path = DropPath(
            drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()

    def forward(self,
                x,
                attention_mask=None,
                attn_bias=None,
                output_attentions=False):
        """Forward the encoder layer.

        Args:
            x (torch.tensor): The input to the layer of shape ``(B, L, C)``.
            attention_mask (torch.Tensor, optional): The attention mask of size
                ``(B, 1, L, L)``, where padding elements are indicated by very
                large negative values. Defaults to None.
            attn_bias (torch.tensor, optional): The bias for positional
                information. Defaults to None.
            output_attentions (bool): Whether to return the attentions tensors
                of the attention layer.

        Returns:
            List[torch.tensor]: The first element is the encoded output of
            shape ``(B, L, C)``. And the second element is the output
            attentions if ``output_attentions=True``.
        """
        residual = x

        # Attention block
        if self.pre_norm:
            x = self.attn_ln(x)
        x, attn_weights, _ = self.attn(
            query=x,
            attn_mask=attention_mask,
            attn_bias=attn_bias,
            output_attentions=output_attentions)
        if self.normformer:
            x = self.attn_mid_ln(x)
        x = self.dropout(x)
        x = residual + self.drop_path(x)
        if not self.pre_norm:
            x = self.attn_ln(x)

        residual = x

        # FFN block
        if self.pre_norm:
            x = self.ffn_ln(x)
        x = self.act(self.fc1(x))
        x = self.act_drop(x)
        if self.normformer:
            x = self.ffn_mid_ln(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = residual + self.drop_path(x)
        if not self.pre_norm:
            x = self.ffn_ln(x)

        if output_attentions:
            return [x, attn_weights]
        else:
            return [x]


class OFADecoderLayer(nn.Module):
    """OFADecoder layer block."""

    def __init__(self,
                 embedding_dim,
                 num_heads,
                 dropout_rate=0.,
                 drop_path_rate=0.,
                 attn_drop=0.,
                 act_drop=0.,
                 scale_factor=2.,
                 mlp_ratio=4.,
                 encoder_embed_dim=None,
                 scale_heads=True,
                 normformer=True,
                 pre_norm=True,
                 act_cfg=dict(type='GELU')):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.pre_norm = pre_norm

        self.self_attn = MultiheadAttention(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            attn_drop=attn_drop,
            scale_factor=scale_factor,
            scale_heads=scale_heads,
        )

        self.cross_attn = MultiheadAttention(
            embedding_dim=embedding_dim,
            kdim=encoder_embed_dim,
            vdim=encoder_embed_dim,
            num_heads=num_heads,
            attn_drop=attn_drop,
            scale_factor=scale_factor,
            scale_heads=scale_heads,
        )

        mid_channels = int(embedding_dim * mlp_ratio)
        self.fc1 = nn.Linear(embedding_dim, mid_channels)
        self.fc2 = nn.Linear(mid_channels, embedding_dim)
        self.act = MODELS.build(act_cfg)
        self.act_drop = nn.Dropout(
            act_drop) if act_drop > 0. else nn.Identity()

        # LayerNorm between attention block and ffn block.
        self.self_attn_ln = nn.LayerNorm(embedding_dim)
        self.cross_attn_ln = nn.LayerNorm(embedding_dim)
        self.ffn_ln = nn.LayerNorm(embedding_dim)

        # Extra LayerNorm
        self.normformer = normformer
        if self.normformer:
            self.self_attn_mid_ln = nn.LayerNorm(embedding_dim)
            self.cross_attn_mid_ln = nn.LayerNorm(embedding_dim)
            self.ffn_mid_ln = nn.LayerNorm(mid_channels)

        self.dropout = nn.Dropout(dropout_rate)
        self.drop_path = DropPath(
            drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()

    def forward(
        self,
        x,
        attention_mask=None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[List[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        self_attn_bias: Optional[torch.Tensor] = None,
        cross_attn_bias: Optional[torch.Tensor] = None,
    ):
        """Forward the decoder layer.

        Args:
            x (torch.tensor): The input to the layer of shape ``(B, L, C)``.
            attention_mask (torch.Tensor, optional): The attention mask of size
                ``(B, 1, L, L)``, where padding elements are indicated by very
                large negative values. Defaults to None.
            encoder_hidden_states (torch.Tensor, optional): The cross attention
                input to the layer of size ``(B, L, C)``. Defaults to None.
            encoder_attention_mask (torch.Tensor, optional): The cross
                attention mask where padding elements are indicated by very
                large negative values. Defaults to None.
            past_key_value (Tuple[torch.tensor], optional): The cached past key
                and value projection states. Defaults to none.
            output_attentions (bool): whether to return the attentions tensors
                of all attention layers. Defaults to False.
            use_cache (bool, optional): Whether to use cache.
                Defaults to False.
            self_attn_bias (torch.Tensor, optional): The self attention bias
                for positional information. Defaults to None.
            cross_attn_bias (torch.Tensor, optional): The cross attention bias
                for positional information. Defaults to None.

        Returns:
            List[torch.tensor]: The first element is the encoded output of
            shape ``(B, L, C)``. The following two elements can be the output
            self-attentions and cross-attentions if ``output_attentions=True``.
            The following one element can be the cached past key and value
            projection states.
        """
        residual = x

        if past_key_value is not None:
            self_past_key_value = past_key_value[:2]
            cross_past_key_value = past_key_value[2:]
        else:
            self_past_key_value, cross_past_key_value = None, None

        # Self-Attention block
        if self.pre_norm:
            x = self.self_attn_ln(x)
        x, self_attn_weights, present_key_value = self.self_attn(
            query=x,
            past_key_value=self_past_key_value,
            attn_mask=attention_mask,
            output_attentions=output_attentions,
            attn_bias=self_attn_bias,
        )
        if self.normformer:
            x = self.self_attn_mid_ln(x)
        x = self.dropout(x)
        x = residual + self.drop_path(x)
        if not self.pre_norm:
            x = self.self_attn_ln(x)

        # Cross-Attention block
        if encoder_hidden_states is not None:
            residual = x
            if self.pre_norm:
                x = self.cross_attn_ln(x)
            x, cross_attn_weights, cross_key_value = self.cross_attn.forward(
                query=x,
                key_value=encoder_hidden_states,
                attn_mask=encoder_attention_mask,
                past_key_value=cross_past_key_value,
                output_attentions=output_attentions,
                attn_bias=cross_attn_bias)
            if self.normformer:
                x = self.cross_attn_mid_ln(x)
            x = self.dropout(x)
            x = residual + self.drop_path(x)
            if not self.pre_norm:
                x = self.cross_attn_ln(x)

            present_key_value = present_key_value + cross_key_value

        residual = x

        # FFN block
        if self.pre_norm:
            x = self.ffn_ln(x)
        x = self.act(self.fc1(x))
        x = self.act_drop(x)
        if self.normformer:
            x = self.ffn_mid_ln(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = residual + self.drop_path(x)
        if not self.pre_norm:
            x = self.ffn_ln(x)

        outputs = [x]

        if output_attentions:
            outputs.extend([self_attn_weights, cross_attn_weights])

        if use_cache:
            outputs.append(present_key_value)

        return outputs


class OFAEncoder(BaseModule):
    """The encoder module of OFA.

    Args:
        embed_tokens (nn.Embedding): The embedding module to embed the
            input tokens.
        embed_images (dict | nn.Module): The module to embed the input
            images into features. The output number of channels should
            be 1024.
        num_layers (int): The number of encoder layers. Defaults to 6.
        num_heads (int): The number of heads of attention. Defaults to 12.
        dropout_rate (float): The prob of dropout for embedding and
            transformer layers. Defaults to 0.
        drop_path_rate (float): The prob of droppath for transformer layers.
            Defaults to 0.
        max_source_positions (int): The maximum length of the input tokens.
            Defaults to 1024.
        token_bucket_size (int): The token bucket size, it's used as the
            maximum relative position index in relative position embedding
            of input tokens. Defaults to 256.
        image_bucket_size (int): The image bucket size, it's used to generate
            the image relative position embedding table. It should be larger
            than the shape of image feature map. Defaults to 42.
        attn_scale_factor (float): The scale factor to calculate qk scale in
            attentions. Defaults to 2.
        scale_embedding (bool): Whether to scale the embeddings by the square
            root of the dimension. Defaults to False.
        add_embedding_ln (bool): Whether to add an extra layer norm for token
            embeddings. Defaults to True.
        add_image_embedding_ln (bool): Whether to add an extra layer norm for
            image embeddings. Defaults to True.
        pre_norm (bool): Whether to do layer norm before attention and ffn
            blocks in transformer layers. Defaults to True.
        entangle_position_embedding (bool): Whether to add the position
            embedding on the embeddings directly. Defaults to False.
        init_cfg (dict, optional): The initialization config. Defaults to None.
    """

    def __init__(
        self,
        embed_tokens,
        embed_images: dict,
        num_layers=6,
        num_heads=12,
        dropout_rate=0.,
        drop_path_rate=0.,
        max_source_positions=1024,
        token_bucket_size=256,
        image_bucket_size=42,
        attn_scale_factor=2.,
        scale_embedding=False,
        add_embedding_ln=True,
        add_type_embed=True,
        add_image_embedding_ln=True,
        pre_norm=True,
        entangle_position_embedding=False,
        init_cfg=None,
    ):
        super().__init__(init_cfg=init_cfg)

        self.num_layers = num_layers
        embedding_dim = embed_tokens.embedding_dim
        self.embedding_dim = embedding_dim
        self.padding_idx = embed_tokens.padding_idx
        self.max_source_positions = max_source_positions
        self.num_heads = num_heads

        # Build embedding process components
        self.embed_tokens = embed_tokens
        self.embedding_scale = math.sqrt(
            embedding_dim) if scale_embedding else 1.0

        if not isinstance(embed_images, nn.Module):
            self.embed_images = MODELS.build(embed_images)
        else:
            self.embed_images = embed_images
        self.image_proj = nn.Linear(1024, embedding_dim)

        if add_embedding_ln:
            self.embedding_ln = nn.LayerNorm(embedding_dim)
        else:
            self.embedding_ln = None

        if add_type_embed:
            self.embed_type = nn.Embedding(2, embedding_dim)
        else:
            self.embed_type = None

        if add_image_embedding_ln:
            self.image_embedding_ln = nn.LayerNorm(embedding_dim)
        else:
            self.image_embedding_ln = None

        self.entangle_position_embedding = entangle_position_embedding

        # Build position embedding
        self.embed_positions = nn.Embedding(self.max_source_positions + 2,
                                            embedding_dim)
        self.pos_ln = nn.LayerNorm(embedding_dim)
        self.embed_image_positions = nn.Embedding(image_bucket_size**2 + 1,
                                                  embedding_dim)
        self.image_pos_ln = nn.LayerNorm(embedding_dim)

        self.pos_scaling = float(embedding_dim / num_heads *
                                 attn_scale_factor)**-0.5
        self.pos_q_linear = nn.Linear(embedding_dim, embedding_dim)
        self.pos_k_linear = nn.Linear(embedding_dim, embedding_dim)

        self.dropout = nn.Dropout(
            dropout_rate) if dropout_rate > 0. else nn.Identity()

        # Register token relative position embedding table
        self.token_bucket_size = token_bucket_size
        token_num_rel_dis = 2 * token_bucket_size - 1
        token_rp_bucket = make_token_bucket_position(token_bucket_size,
                                                     self.max_source_positions)
        self.register_buffer('token_rp_bucket', token_rp_bucket)
        self.token_rel_pos_table_list = nn.ModuleList()

        # Register image relative position embedding table
        self.image_bucket_size = image_bucket_size
        image_num_rel_dis = (2 * image_bucket_size -
                             1) * (2 * image_bucket_size - 1) + 3
        image_rp_bucket = make_image_bucket_position(image_bucket_size,
                                                     image_num_rel_dis)
        self.register_buffer('image_rp_bucket', image_rp_bucket)
        self.image_rel_pos_table_list = nn.ModuleList()

        # Build encoder layers
        self.layers = nn.ModuleList()
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_layers)]
        for index in range(self.num_layers):
            layer = OFAEncoderLayer(
                embedding_dim=embedding_dim,
                num_heads=num_heads,
                dropout_rate=dropout_rate,
                drop_path_rate=dpr[index],
                scale_factor=attn_scale_factor,
                pre_norm=pre_norm,
            )
            self.layers.append(layer)
            token_pos_table = nn.Embedding(token_num_rel_dis, self.num_heads)
            image_pos_table = nn.Embedding(image_num_rel_dis, self.num_heads)
            nn.init.constant_(token_pos_table.weight, 0.)
            nn.init.constant_(image_pos_table.weight, 0.)
            self.token_rel_pos_table_list.append(token_pos_table)
            self.image_rel_pos_table_list.append(image_pos_table)

        if pre_norm:
            self.final_ln = nn.LayerNorm(embedding_dim)
        else:
            self.final_ln = None

    main_input_name = 'input_ids'

    def forward(self,
                input_ids,
                images,
                images_mask,
                output_attentions=False,
                output_hidden_states=False,
                sample_patch_num=None):
        padding_mask = input_ids.eq(self.padding_idx)
        has_pads = padding_mask.any()
        token_embedding = self.embed_tokens(input_ids)
        token_embedding = self.embedding_scale * token_embedding

        # Embed the token position
        src_pos_idx = torch.arange(input_ids.size(-1), device=input_ids.device)
        src_pos_idx = src_pos_idx.expand(*input_ids.shape).contiguous()
        pos_embedding = self.embed_positions(src_pos_idx)

        # Embed the input tokens
        x = self.process_embedding(
            embedding=token_embedding,
            type_tokens=input_ids.new_zeros(token_embedding.shape[:2]),
            pos_embedding=pos_embedding,
            embedding_ln=self.embedding_ln,
        )
        pos_embedding = self.pos_ln(pos_embedding)

        # Embed the input images
        if images is not None:
            (image_tokens, image_padding_mask, image_position_ids,
             image_pos_embedding) = self.get_image_tokens(
                 images,
                 sample_patch_num,
                 images_mask,
             )
            image_embedding = self.image_proj(image_tokens)

            image_x = self.process_embedding(
                embedding=image_embedding,
                type_tokens=input_ids.new_ones(image_embedding.shape[:2]),
                pos_embedding=image_pos_embedding,
                embedding_ln=self.image_embedding_ln,
            )
            image_pos_embedding = self.image_pos_ln(image_pos_embedding)

            x = torch.cat([image_x, x], dim=1)
            padding_mask = torch.cat([image_padding_mask, padding_mask], dim=1)
            pos_embedding = torch.cat([image_pos_embedding, pos_embedding],
                                      dim=1)

        # account for padding while computing the representation
        if has_pads:
            x = x * (1 - padding_mask.unsqueeze(-1).type_as(x))

        # Decoupled position embedding
        B, L = pos_embedding.shape[:2]
        pos_q = self.pos_q_linear(pos_embedding).view(
            B, L, self.num_heads, -1).transpose(1, 2) * self.pos_scaling
        pos_k = self.pos_k_linear(pos_embedding).view(B, L, self.num_heads,
                                                      -1).transpose(1, 2)
        abs_pos_bias = torch.matmul(pos_q, pos_k.transpose(2, 3))

        all_hidden_states = [] if output_hidden_states else None
        all_attentions = [] if output_attentions else None

        for idx, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states.append(x)

            self_attn_bias = abs_pos_bias.clone()
            # Add decoupled position embedding for input tokens.
            token_len = input_ids.size(1)
            rel_pos_bias = self.get_rel_pos_bias(input_ids, idx)
            self_attn_bias[:, :, -token_len:, -token_len:] += rel_pos_bias

            # Add decoupled position embedding for images
            if images is not None:
                token_len = image_tokens.size(1)
                rel_pos_bias = self.get_image_rel_pos_bias(
                    image_position_ids, idx)
                self_attn_bias[:, :, :token_len, :token_len] += rel_pos_bias

            if has_pads:
                attention_mask = _expand_mask(padding_mask, dtype=x.dtype)
            else:
                attention_mask = None

            out = layer(
                x,
                attention_mask=attention_mask,
                attn_bias=self_attn_bias,
                output_attentions=output_attentions)
            x = out[0]

            if output_attentions:
                all_attentions.append(out[1])

        if output_hidden_states:
            all_hidden_states.append(x)

        if self.final_ln is not None:
            x = self.final_ln(x)

        return OFAEncoderOutput(
            last_hidden_state=x,  # (B, L, C)
            padding_mask=padding_mask,  # (B, L)
            position_embedding=pos_embedding,  # (B, L, C)
            hidden_states=all_hidden_states,  # list of (B, L, C)
            attentions=all_attentions,  # list of (B, num_heads, L, head_dims)
        )

    def get_image_tokens(self, images, sample_patch_num, images_mask):
        image_embedding = self.embed_images(images)[-1]
        B, C, H, W = image_embedding.shape
        num_patches = H * W

        padding_mask = images.new_zeros((B, num_patches)).bool()
        position_col = torch.arange(W).unsqueeze(0)
        position_row = torch.arange(H).unsqueeze(1) * self.image_bucket_size
        position_idx = (position_col + position_row + 1).view(-1)
        position_idx = position_idx.to(images.device).expand(B, num_patches)

        # (B, C, H, W) -> (B, C, H*W) -> (B, H*W, C)
        image_embedding = image_embedding.flatten(2).transpose(1, 2)
        if sample_patch_num is not None:
            patch_orders = torch.stack([
                torch.randperm(num_patches)[:sample_patch_num]
                for _ in range(B)
            ])
            num_patches = sample_patch_num
            image_embedding = image_embedding.gather(
                dim=1, index=patch_orders.unsqueeze(2).expand(-1, -1, C))
            padding_mask = padding_mask.gather(1, patch_orders)
            position_idx = position_idx.gather(1, patch_orders)

        pos_embedding = self.embed_image_positions(position_idx)
        padding_mask[~images_mask] = True
        return image_embedding, padding_mask, position_idx, pos_embedding

    def process_embedding(self,
                          embedding,
                          pos_embedding=None,
                          type_tokens=None,
                          embedding_ln=None):
        if self.entangle_position_embedding and pos_embedding is not None:
            embedding += pos_embedding
        if self.embed_type is not None:
            embedding += self.embed_type(type_tokens)
        if embedding_ln is not None:
            embedding = embedding_ln(embedding)
        embedding = self.dropout(embedding)

        return embedding

    def get_rel_pos_bias(self, x, idx):
        seq_len = x.size(1)
        rp_bucket = self.token_rp_bucket[:seq_len, :seq_len]
        values = F.embedding(rp_bucket,
                             self.token_rel_pos_table_list[idx].weight)
        values = values.unsqueeze(0).expand(x.size(0), -1, -1, -1)
        values = values.permute([0, 3, 1, 2])
        return values.contiguous()

    def get_image_rel_pos_bias(self, image_position_ids, idx):
        bsz, seq_len = image_position_ids.shape
        rp_bucket_size = self.image_rp_bucket.size(1)

        rp_bucket = self.image_rp_bucket.unsqueeze(0).expand(
            bsz, rp_bucket_size, rp_bucket_size).gather(
                1, image_position_ids[:, :, None].expand(
                    bsz, seq_len, rp_bucket_size)).gather(
                        2, image_position_ids[:, None, :].expand(
                            bsz, seq_len, seq_len))
        values = F.embedding(rp_bucket,
                             self.image_rel_pos_table_list[idx].weight)
        values = values.permute(0, 3, 1, 2)
        return values


class OFADecoder(BaseModule):
    """The decoder module of OFA.

    Args:
        embed_tokens (nn.Embedding): The embedding module to embed the
            input tokens.
        num_layers (int): The number of decoder layers. Defaults to 6.
        num_heads (int): The number of heads of attention. Defaults to 12.
        dropout_rate (float): The prob of dropout for embedding and
            transformer layers. Defaults to 0.
        drop_path_rate (float): The prob of droppath for transformer layers.
            Defaults to 0.
        max_target_positions (int): The maximum length of the input tokens.
            Defaults to 1024.
        code_image_size (int): The resolution of the generated image in the
            image infilling task. Defaults to 128.
        token_bucket_size (int): The token bucket size, it's used as the
            maximum relative position index in relative position embedding
            of input tokens. Defaults to 256.
        image_bucket_size (int): The image bucket size, it's used to generate
            the image relative position embedding table. It should be larger
            than the shape of image feature map. Defaults to 42.
        attn_scale_factor (float): The scale factor to calculate qk scale in
            attentions. Defaults to 2.
        scale_embedding (bool): Whether to scale the embeddings by the square
            root of the dimension. Defaults to False.
        add_embedding_ln (bool): Whether to add an extra layer norm for token
            embeddings. Defaults to True.
        add_code_embedding_ln (bool): Whether to add an extra layer norm for
            code embeddings. Defaults to True.
        pre_norm (bool): Whether to do layer norm before attention and ffn
            blocks in transformer layers. Defaults to True.
        entangle_position_embedding (bool): Whether to add the position
            embedding on the embeddings directly. Defaults to False.
        share_input_output_embed (bool): Share the weights of the input token
            embedding module and the output projection module.
            Defaults to True.
        init_cfg (dict, optional): The initialization config. Defaults to None.
    """

    def __init__(
        self,
        embed_tokens,
        num_layers=6,
        num_heads=12,
        dropout_rate=0.,
        drop_layer_rate=0.,
        drop_path_rate=0.,
        max_target_positions=1024,
        code_image_size=128,
        token_bucket_size=256,
        image_bucket_size=42,
        attn_scale_factor=2.,
        scale_embedding=False,
        add_embedding_ln=True,
        add_code_embedding_ln=True,
        pre_norm=True,
        entangle_position_embedding=False,
        share_input_output_embed=True,
        init_cfg=None,
    ):
        super().__init__(init_cfg=init_cfg)
        self._future_mask = torch.empty(0)

        self.num_layers = num_layers
        embedding_dim = embed_tokens.embedding_dim
        self.embedding_dim = embedding_dim
        self.padding_idx = embed_tokens.padding_idx
        self.max_target_positions = max_target_positions
        self.num_heads = num_heads

        # Build embedding process components
        self.embed_tokens = embed_tokens
        self.embedding_scale = math.sqrt(
            embedding_dim) if scale_embedding else 1.0

        if add_embedding_ln:
            self.embedding_ln = nn.LayerNorm(embedding_dim)
        else:
            self.embedding_ln = None

        if add_code_embedding_ln:
            self.code_embedding_ln = nn.LayerNorm(embedding_dim)
        else:
            self.code_embedding_ln = None

        # Build position embedding
        self.embed_positions = nn.Embedding(self.max_target_positions + 2,
                                            embedding_dim)
        self.pos_ln = nn.LayerNorm(embedding_dim)
        self.embed_image_positions = nn.Embedding(image_bucket_size**2 + 1,
                                                  embedding_dim)
        self.image_pos_ln = nn.LayerNorm(embedding_dim)

        self.pos_scaling = float(embedding_dim / num_heads *
                                 attn_scale_factor)**-0.5
        self.self_pos_q_linear = nn.Linear(embedding_dim, embedding_dim)
        self.self_pos_k_linear = nn.Linear(embedding_dim, embedding_dim)
        self.cross_pos_q_linear = nn.Linear(embedding_dim, embedding_dim)
        self.cross_pos_k_linear = nn.Linear(embedding_dim, embedding_dim)

        self.entangle_position_embedding = entangle_position_embedding

        self.dropout = nn.Dropout(
            dropout_rate) if dropout_rate > 0. else nn.Identity()
        if drop_layer_rate > 0.:
            raise NotImplementedError

        # Register token relative position embedding table
        self.token_bucket_size = token_bucket_size
        token_num_rel_dis = 2 * token_bucket_size - 1
        token_rp_bucket = make_token_bucket_position(token_bucket_size)
        self.register_buffer('token_rp_bucket', token_rp_bucket)
        self.token_rel_pos_table_list = nn.ModuleList()

        # Register image relative position embedding table
        self.image_bucket_size = image_bucket_size
        image_num_rel_dis = (2 * image_bucket_size -
                             1) * (2 * image_bucket_size - 1) + 3
        image_rp_bucket = make_image_bucket_position(image_bucket_size,
                                                     image_num_rel_dis)
        self.register_buffer('image_rp_bucket', image_rp_bucket)
        self.image_rel_pos_table_list = nn.ModuleList()

        self.window_size = code_image_size // 8

        position_col = torch.arange(self.window_size).unsqueeze(0)
        position_row = torch.arange(
            self.window_size).unsqueeze(1) * self.image_bucket_size
        image_position_idx = (position_col + position_row + 1)
        image_position_idx = torch.cat(
            [torch.tensor([0]), image_position_idx.view(-1)])
        image_position_idx = torch.cat(
            [image_position_idx,
             torch.tensor([1024] * 768)])
        self.register_buffer('image_position_idx', image_position_idx)

        # Build decoder layers
        self.layers = nn.ModuleList()
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_layers)]
        for index in range(self.num_layers):
            layer = OFADecoderLayer(
                embedding_dim=embedding_dim,
                num_heads=num_heads,
                dropout_rate=dropout_rate,
                drop_path_rate=dpr[index],
                scale_factor=attn_scale_factor,
                pre_norm=pre_norm,
            )
            self.layers.append(layer)
            token_pos_table = nn.Embedding(token_num_rel_dis, self.num_heads)
            image_pos_table = nn.Embedding(image_num_rel_dis, self.num_heads)
            nn.init.constant_(token_pos_table.weight, 0.)
            nn.init.constant_(image_pos_table.weight, 0.)
            self.token_rel_pos_table_list.append(token_pos_table)
            self.image_rel_pos_table_list.append(image_pos_table)

        if pre_norm:
            self.final_ln = nn.LayerNorm(embedding_dim)
        else:
            self.final_ln = None

        # Build output projection
        if share_input_output_embed:
            self.output_projection = nn.Linear(
                self.embed_tokens.weight.shape[1],
                self.embed_tokens.weight.shape[0],
                bias=False,
            )
            self.output_projection.weight = self.embed_tokens.weight
        else:
            vocab_size = self.embed_tokens.num_embeddings
            self.output_projection = nn.Linear(
                embedding_dim, vocab_size, bias=False)
            nn.init.normal_(
                self.output_projection.weight,
                mean=0,
                std=embedding_dim**-0.5,
            )

    main_input_name = 'input_ids'

    def forward(
        self,
        input_ids: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        encoder_hidden_states: torch.Tensor = None,
        encoder_attention_mask: torch.Tensor = None,
        code_masks: Optional[torch.Tensor] = None,
        encoder_pos_embedding: Optional[torch.Tensor] = None,
        past_key_values: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ):

        if past_key_values is not None and len(past_key_values) > 0:
            B, _, L_past, _ = past_key_values[0][0].shape
            L = L_past + 1
        else:
            B, L = input_ids.shape
            L_past = 0

        # Embed the token position
        target_pos_idx = torch.arange(
            L, device=input_ids.device).expand([B, L]).contiguous()
        pos_embedding = self.embed_positions(target_pos_idx)

        # Embed the code positions
        if code_masks is not None and torch.any(code_masks):
            image_position_idx = self.image_position_idx[:input_ids.size(1)]
            image_position_idx = image_position_idx.unsqueeze(0).expand(B, L)
            pos_embedding[code_masks] = self.embed_image_positions(
                image_position_idx)[code_masks]

        # Self-attention position bias (B, num_heads, L_t, L_t)
        self_abs_pos_bias = self.get_pos_info(self.pos_ln(pos_embedding))
        if code_masks is not None and torch.any(code_masks):
            self_image_abs_pos_bias = self.get_pos_info(
                self.image_pos_ln(pos_embedding))
            self_abs_pos_bias[code_masks] = self_image_abs_pos_bias[code_masks]

        # Cross-attention position bias (B, num_heads, L_t, L_s)
        cross_abs_pos_bias = self.get_pos_info(
            self.pos_ln(pos_embedding), encoder_pos_embedding)
        if code_masks is not None and torch.any(code_masks):
            cross_image_abs_pos_bias = self.get_pos_info(
                self.image_pos_ln(pos_embedding), encoder_pos_embedding)
            cross_abs_pos_bias[code_masks] = cross_image_abs_pos_bias[
                code_masks]

        all_prev_output_tokens = input_ids.clone()
        if past_key_values is not None and len(past_key_values) > 0:
            input_ids = input_ids[:, -1:]
            cross_abs_pos_bias = cross_abs_pos_bias[:, :, -1:, :]
            pos_embedding = pos_embedding[:, -1:, :]

        # Embed the input tokens
        x = self.embed_tokens(input_ids) * self.embedding_scale

        if self.entangle_position_embedding:
            x += pos_embedding

        if self.embedding_ln is not None:
            if (code_masks is None or not code_masks.any()
                    or self.code_embedding_ln is None):
                x = self.embedding_ln(x)
            elif code_masks is not None and code_masks.all():
                x = self.code_embedding_ln(x)
            else:
                x[~code_masks] = self.embedding_ln(x[~code_masks])
                x[code_masks] = self.code_embedding_ln(x[code_masks])

        x = self.dropout(x)

        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, input_ids.shape, x.dtype, L_past)
        attention_mask = attention_mask.to(x.device)

        # decoder layers
        all_hidden_states = [] if output_hidden_states else None
        all_self_attns = [] if output_attentions else None
        all_cross_attentions = [] if (
            output_attentions and encoder_hidden_states is not None) else None
        next_decoder_cache = [] if use_cache else None

        for idx, layer in enumerate(self.layers):
            # add hidden states from the last decoder layer
            if output_hidden_states:
                all_hidden_states.append(x)

            if past_key_values is not None and len(past_key_values) > 0:
                past_key_value = past_key_values[idx]
            else:
                past_key_value = None

            self_attn_bias = self_abs_pos_bias.clone()
            if code_masks is None or not code_masks.any():
                self_attn_bias += self.get_rel_pos_bias(
                    all_prev_output_tokens, idx)
            elif code_masks is not None and code_masks.all():
                self_attn_bias += self.get_image_rel_pos_bias(
                    all_prev_output_tokens, idx)
            else:
                self_attn_bias[~code_masks] += self.get_rel_pos_bias(
                    all_prev_output_tokens, idx)
                self_attn_bias[code_masks] += self.get_image_rel_pos_bias(
                    all_prev_output_tokens, idx)

            if past_key_value is not None:
                self_attn_bias = self_attn_bias[:, :, -1:, :]

            out = layer(
                x,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                self_attn_bias=self_attn_bias,
                cross_attn_bias=cross_abs_pos_bias,
            )
            x = out.pop(0)

            if output_attentions:
                all_self_attns.append(out.pop(0))
                if encoder_hidden_states is not None:
                    all_cross_attentions.append(out.pop(0))

            if use_cache:
                next_decoder_cache.append(out.pop(0))

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (x, )

        if self.final_ln is not None:
            x = self.final_ln(x)

        x = self.output_projection(x)

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=x,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=all_cross_attentions,
        )

    def _prepare_decoder_attention_mask(
        self,
        attention_mask,
        input_shape,
        dtype,
        past_key_values_length,
    ):
        r"""
        Create causal mask for unidirectional decoding.
        [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        """
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape,
                dtype,
                past_key_values_length=past_key_values_length).to(
                    attention_mask.device)

        if attention_mask is not None:
            # (B, L_s) -> (B, 1, L_t, L_s)
            expanded_attention_mask = _expand_mask(
                attention_mask, dtype, tgt_len=input_shape[-1])
            combined_attention_mask = (
                expanded_attention_mask if combined_attention_mask is None else
                expanded_attention_mask + combined_attention_mask)

        return combined_attention_mask

    def get_pos_info(self, pos_embedding, src_pos_embedding=None):
        B, tgt_len = pos_embedding.shape[:2]
        if src_pos_embedding is not None:
            src_len = src_pos_embedding.size(1)
            pos_q = self.cross_pos_q_linear(pos_embedding).view(
                B, tgt_len, self.num_heads, -1).transpose(1, 2)
            pos_q = pos_q * self.pos_scaling
            pos_k = self.cross_pos_k_linear(src_pos_embedding).view(
                B, src_len, self.num_heads, -1).transpose(1, 2)
        else:
            pos_q = self.self_pos_q_linear(pos_embedding).view(
                B, tgt_len, self.num_heads, -1).transpose(1, 2)
            pos_q = pos_q * self.pos_scaling
            pos_k = self.self_pos_k_linear(pos_embedding).view(
                B, tgt_len, self.num_heads, -1).transpose(1, 2)

        abs_pos_bias = torch.matmul(pos_q, pos_k.transpose(2, 3))

        return abs_pos_bias

    def get_rel_pos_bias(self, x, idx):
        seq_len = x.size(1)
        rp_bucket = self.token_rp_bucket[:seq_len, :seq_len]
        values = F.embedding(rp_bucket,
                             self.token_rel_pos_table_list[idx].weight)
        values = values.unsqueeze(0).expand(x.size(0), -1, -1, -1)
        values = values.permute([0, 3, 1, 2])
        return values.contiguous()

    def get_image_rel_pos_bias(self, image_position_ids, idx):
        bsz, seq_len = image_position_ids.shape
        rp_bucket_size = self.image_rp_bucket.size(1)

        rp_bucket = self.image_rp_bucket.unsqueeze(0).expand(
            bsz, rp_bucket_size, rp_bucket_size).gather(
                1, image_position_ids[:, :, None].expand(
                    bsz, seq_len, rp_bucket_size)).gather(
                        2, image_position_ids[:, None, :].expand(
                            bsz, seq_len, seq_len))
        values = F.embedding(rp_bucket,
                             self.image_rel_pos_table_list[idx].weight)
        values = values.permute(0, 3, 1, 2)
        return values


class OFAEncoderDecoder(BaseModule, GenerationMixin):
    """The OFA main architecture with an encoder and a decoder.

    Args:
        encoder_cfg (dict): The config of the encoder, accept the keyword
            arguments of :class:`OFAEncoder`.
        decoder_cfg (dict): The config of the decoder, accept the keyword
            arguments of :class:`OFADecoder`.
        padding_idx (int): The index of the padding token.
        vocab_size (int): The size of the vocabulary.
        embedding_dim (int): The embedding dimensions of both the encoder
            and the decoder.
        generation_cfg (dict): The extra generation config, accept the keyword
            arguments of :class:`~transformers.GenerationConfig`.
            Defaults to an empty dict.
        init_cfg (dict, optional): The initialization config. Defaults to None.
    """

    def __init__(
            self,
            encoder_cfg,
            decoder_cfg,
            padding_idx,
            vocab_size,
            embedding_dim,
            generation_cfg=dict(),
            init_cfg=None,
    ):
        super().__init__(init_cfg=init_cfg)

        self.padding_idx = padding_idx
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        embed_tokens = nn.Embedding(vocab_size, embedding_dim, padding_idx)

        self.encoder = OFAEncoder(embed_tokens, **encoder_cfg)
        self.decoder = OFADecoder(embed_tokens, **decoder_cfg)

        self.config = PretrainedConfig(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
            bos_token_id=0,
            decoder_start_token_id=0,
            pad_token_id=1,
            eos_token_id=2,
            forced_eos_token_id=2,
            use_cache=False,
            is_encoder_decoder=True,
        )
        self.config.update(generation_cfg)

        self.generation_config = GenerationConfig.from_model_config(
            self.config)

    @property
    def device(self):
        return next(self.parameters()).device

    def can_generate(self):
        return True

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def max_decoder_positions(self):
        """Maximum length supported by the decoder."""
        return self.decoder.max_positions()

    def get_normalized_probs(self, net_output, log_probs: bool, sample=None):
        """Get normalized probabilities (or log probs) from a net's output."""
        return self.get_normalized_probs_scriptable(net_output, log_probs,
                                                    sample)

    def get_normalized_probs_scriptable(
        self,
        net_output,
        log_probs: bool,
        sample=None,
    ):
        """Scriptable helper function for get_normalized_probs in.

        ~BaseFairseqModel.
        """
        if hasattr(self, 'decoder'):
            return self.decoder.get_normalized_probs(net_output, log_probs,
                                                     sample)
        elif torch.is_tensor(net_output):
            # syntactic sugar for simple models which don't have a decoder
            # (e.g., the classification tutorial)
            logits = net_output.float()
            if log_probs:
                return F.log_softmax(logits, dim=-1)
            else:
                return F.softmax(logits, dim=-1)
        raise NotImplementedError

    main_input_name = 'input_ids'

    def forward(self,
                input_ids=None,
                images=None,
                images_mask=None,
                sample_patch_num=None,
                decoder_input_ids=None,
                code_masks=None,
                attention_mask=None,
                encoder_outputs=None,
                past_key_values=None,
                use_cache=False,
                output_attentions=False,
                output_hidden_states=False,
                constrain_fn=None,
                return_dict=False):
        """Forword the module.

        Args:
            input_ids (torch.Tensor): The indices of the input tokens in the
                vocabulary, and padding will be ignored by default. The indices
                can be obtained using :class:`OFATokenizer`.
                The shape is (B, L).
            images (torch.Tensor): The input images. The shape is (B, 3, H, W).
            images_mask (torch.Tensor): The mask of all available images. The
                shape is (B, ).
            sample_patch_num (int): The number of patches to sample for the
                images. Defaults to None, which means to use all patches.
            decoder_input_ids (torch.Tensor): The indices of the input tokens
                for the decoder.
            code_masks (torch.Tensor): The mask of all samples for image
                generation. The shape is (B, ).
            attention_mask (torch.Tensor): The attention mask for decoding.
                The shape is (B, L).
            encoder_outputs (OFAEncoderOutput): The encoder outputs with hidden
                states, positional embeddings, and padding masks.
            past_key_values (Tuple[Tuple[torch.Tensor]]): If use cache, the
                parameter is a tuple of length ``num_layers``. Every item is
                also a tuple with four tensors, two for the key and value of
                self-attention, two for the key and value of cross-attention.
            use_cache (bool): Whether to use cache for faster inference.
                Defaults to False.
            output_attentions (bool): Whether to output attention weights.
                Defaults to False.
            output_hidden_states (bool): Whether to output hidden states.
                Defaults to False.
            constrain_fn (Callable, optional): The function to constrain the
                output logits. Defaults to None.
            return_dict (bool): Not used, it's only for compat with the
                interface of the ``generate`` of ``transformers``.

        Returns:
            Seq2SeqLMOutput:

            - logits (``torch.Tensor``): The last decoder hidden states.
              The shape is (B, L, C).
            - past_key_values (``Tuple[Tuple[torch.Tensor]]``): The past keys
              and values for faster inference.
            - decoder_hidden_states (``Tuple[torch.Tensor]``): the decoder
              hidden states of all layers.
            - decoder_attentions (``Tuple[torch.Tensor]``): The self-attention
              weights of all layers in the decoder.
            - cross_attentions (``Tuple[torch.Tensor]``): The cross-attention
              weights of all layers in the decoder.
            - encoder_last_hidden_state (``torch.Tensor``): The last encoder
              hidden states.
            - encoder_hidden_states (``Tuple[torch.Tensor]``): The encoder
              hidden states of all layers, including the embeddings.
            - encoder_attentions (``Tuple[torch.Tensor]``): The self-attention
              weights of all layers in the encoder.
        """

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                images=images,
                images_mask=images_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                sample_patch_num=sample_patch_num,
            )

        if decoder_input_ids.eq(self.padding_idx).any():
            attention_mask = decoder_input_ids.eq(self.padding_idx)

        encoder_hidden_states = encoder_outputs.last_hidden_state
        encoder_attention_mask = _expand_mask(encoder_outputs.padding_mask,
                                              encoder_hidden_states.dtype,
                                              decoder_input_ids.shape[-1])
        src_pos_embed = encoder_outputs.position_embedding

        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            code_masks=code_masks,
            encoder_pos_embedding=src_pos_embed,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        # The constrain operation for fine-tuned model in OFA is applied
        # before log_softmax, therefore we cannot use
        # `prefix_allowed_tokens_fn` to implement it.
        if constrain_fn is not None:
            logits = constrain_fn(decoder_input_ids,
                                  decoder_outputs.last_hidden_state)
        else:
            logits = decoder_outputs.last_hidden_state

        return Seq2SeqLMOutput(
            logits=logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

    def prepare_inputs_for_generation(self,
                                      decoder_input_ids=None,
                                      past=None,
                                      attention_mask=None,
                                      code_masks=None,
                                      use_cache=False,
                                      encoder_outputs=None,
                                      constrain_fn=None,
                                      **kwargs):
        # if attention_mask is None:
        attention_mask = decoder_input_ids.new_zeros(decoder_input_ids.shape)

        # cut decoder_input_ids if past is used
        if past is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        return {
            'input_ids': None,
            'images': None,
            'images_mask': None,
            'sample_patch_num': None,
            'attention_mask': attention_mask,
            'encoder_outputs': encoder_outputs,
            'past_key_values': past,
            'decoder_input_ids': decoder_input_ids,
            'code_masks': code_masks,
            'use_cache': use_cache,
            'constrain_fn': constrain_fn,
        }

    def _prepare_encoder_decoder_kwargs_for_generation(
            self,
            inputs_tensor: torch.Tensor,
            model_kwargs,
            model_input_name: Optional[str] = None):
        # 1. get encoder
        encoder = self.get_encoder()

        # 2. prepare encoder args and encoder kwargs from model kwargs
        irrelevant_prefix = [
            'decoder_', 'cross_attn', 'use_cache', 'attention_mask',
            'constrain_fn'
        ]
        encoder_kwargs = {
            argument: value
            for argument, value in model_kwargs.items()
            if not any(argument.startswith(p) for p in irrelevant_prefix)
        }

        if encoder_kwargs.get('images_mask') is None:
            encoder_kwargs['images_mask'] = torch.tensor([True] *
                                                         inputs_tensor.size(0))

        # 3. make sure that encoder returns `ModelOutput`
        model_input_name = model_input_name or self.main_input_name
        encoder_kwargs[model_input_name] = inputs_tensor
        model_kwargs['encoder_outputs']: ModelOutput = encoder(
            **encoder_kwargs)
        model_kwargs['attention_mask'] = None

        return model_kwargs

    @staticmethod
    def _reorder_cache(past, beam_idx):
        reordered_past = ()
        for layer_past in past:
            reordered_past += (tuple(
                past_state.index_select(0, beam_idx)
                for past_state in layer_past), )
        return reordered_past

    @staticmethod
    def _expand_inputs_for_generation(
        input_ids: torch.LongTensor,
        expand_size: int = 1,
        is_encoder_decoder: bool = False,
        attention_mask: Optional[torch.LongTensor] = None,
        encoder_outputs: Optional[ModelOutput] = None,
        **model_kwargs,
    ):
        expanded_return_idx = (
            torch.arange(input_ids.shape[0]).view(-1, 1).repeat(
                1, expand_size).view(-1).to(input_ids.device))
        input_ids = input_ids.index_select(0, expanded_return_idx)

        if attention_mask is not None:
            model_kwargs['attention_mask'] = attention_mask.index_select(
                0, expanded_return_idx)

        if is_encoder_decoder:
            if encoder_outputs is None:
                raise ValueError('If `is_encoder_decoder` is True, make '
                                 'sure that `encoder_outputs` is defined.')
            encoder_outputs['last_hidden_state'] = encoder_outputs.\
                last_hidden_state.index_select(0, expanded_return_idx)
            encoder_outputs['position_embedding'] = encoder_outputs.\
                position_embedding.index_select(0, expanded_return_idx)
            encoder_outputs['padding_mask'] = encoder_outputs.\
                padding_mask.index_select(0, expanded_return_idx)
            model_kwargs['encoder_outputs'] = encoder_outputs
        return input_ids, model_kwargs

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections import OrderedDict
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from mmengine.model import BaseModel, BaseModule
from torch import nn

from mmpretrain.datasets.categories import CIFAR100_CATEGORIES_CN
from mmpretrain.registry import MODELS, TOKENIZER
from mmpretrain.structures import DataSample
from mmpretrain.utils import track_on_main_process
from .utils import OPENAI_PROMPT

PROTOTYPE_MAP = {'cifar100': CIFAR100_CATEGORIES_CN}
PROMPT_MAP = {'openai': OPENAI_PROMPT}


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            self.downsample = nn.Sequential(
                OrderedDict([('-1', nn.AvgPool2d(stride)),
                             ('0',
                              nn.Conv2d(
                                  inplanes,
                                  planes * self.expansion,
                                  1,
                                  stride=1,
                                  bias=False)),
                             ('1', nn.BatchNorm2d(planes * self.expansion))]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class AttentionPool2d(nn.Module):

    def __init__(self,
                 spacial_dim: int,
                 embed_dim: int,
                 num_heads: int,
                 output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(
            torch.randn(spacial_dim**2 + 1, embed_dim) / embed_dim**0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[1],
                      x.shape[2] * x.shape[3]).permute(2, 0,
                                                       1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x,
            key=x,
            value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat(
                [self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False)

        return x[0]


@MODELS.register_module()
class ModifiedResNet(BaseModule):
    """A modified ResNet contains the following changes:

    - Apply deep stem with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is
      prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """ # noqa

    arch_settings = {
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3))
    }

    def __init__(self,
                 depth: int = 50,
                 base_channels: int = 64,
                 input_size: int = 224,
                 num_attn_heads: int = 32,
                 output_dim: int = 1024,
                 init_cfg: Optional[dict] = None):
        super().__init__(init_cfg=init_cfg)
        self.input_size = input_size
        self.block, stage_blocks = self.arch_settings[depth]

        # the 3-layer stem
        self.conv1 = nn.Conv2d(
            3,
            base_channels // 2,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False)
        self.bn1 = nn.BatchNorm2d(base_channels // 2)
        self.conv2 = nn.Conv2d(
            base_channels // 2,
            base_channels // 2,
            kernel_size=3,
            padding=1,
            bias=False)
        self.bn2 = nn.BatchNorm2d(base_channels // 2)
        self.conv3 = nn.Conv2d(
            base_channels // 2,
            base_channels,
            kernel_size=3,
            padding=1,
            bias=False)
        self.bn3 = nn.BatchNorm2d(base_channels)
        self.avgpool = nn.AvgPool2d(2)
        self.relu = nn.ReLU(inplace=True)

        # residual layers
        # this is a *mutable* variable used during construction
        self._inplanes = base_channels
        self.layer1 = self._make_layer(base_channels, stage_blocks[0])
        self.layer2 = self._make_layer(
            base_channels * 2, stage_blocks[1], stride=2)
        self.layer3 = self._make_layer(
            base_channels * 4, stage_blocks[2], stride=2)
        self.layer4 = self._make_layer(
            base_channels * 8, stage_blocks[3], stride=2)

        embed_dim = base_channels * 32
        self.attnpool = AttentionPool2d(input_size // 32, embed_dim,
                                        num_attn_heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):

        def stem(x):
            for conv, bn in [(self.conv1, self.bn1), (self.conv2, self.bn2),
                             (self.conv3, self.bn3)]:
                x = self.relu(bn(conv(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)

        return x


@MODELS.register_module()
class ChineseCLIP(BaseModel):
    """The implementation of `ChineseCLIP <https://arxiv.org/abs/2211.01335>`_.

    Args:
        vision_backbone (dict): Config dict for vision backbone.
        text_backbone (dict): Config dict for text backbone.
        tokenizer (dict): Config dict for text tokenizer.
        proj_dim (int): Projection dimension for similarity computation.
        text_prototype (str): Text prototype, which can be a key in
            `PROTOTYPE_MAP` or list of text.
        text_prompt (str): The prompt for text prototype. Defaults to 'openai'.
        context_length (int): The context length to use. Defaults to 52.
        data_preprocessor (Union[dict, nn.Module], optional): The config for
            preprocessing input data. If None or no specified type, it will use
            "MultiModalDataPreprocessor" as type.
            See :class:`MultiModalDataPreprocessor` for more details.
            Defaults to None.
        init_cfg (dict, optional): The config to control the initialization.
            Defaults to None.
    """

    def __init__(self,
                 vision_backbone: dict,
                 text_backbone: dict,
                 tokenizer: dict,
                 proj_dim: int,
                 text_prototype: Union[str, List[str]],
                 text_prompt: str = 'openai',
                 context_length: int = 52,
                 data_preprocessor: Optional[dict] = None,
                 init_cfg: Optional[dict] = None):
        if data_preprocessor is None:
            data_preprocessor = {}
        data_preprocessor.setdefault('type', 'MultiModalDataPreprocessor')
        data_preprocessor = MODELS.build(data_preprocessor)

        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)

        self.vision_backbone = MODELS.build(vision_backbone)
        self.text_backbone = MODELS.build(text_backbone)

        if not isinstance(self.vision_backbone, ModifiedResNet):
            self.vision_projection = nn.Parameter(
                torch.empty(self.vision_backbone.embed_dims, proj_dim))
        text_hidden_size = text_backbone['config']['hidden_size']
        self.text_projection = nn.Parameter(
            torch.empty(text_hidden_size, proj_dim))

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.tokenizer = TOKENIZER.build(tokenizer)
        self.context_length = context_length

        # for zero-shot classification
        if isinstance(text_prototype,
                      str) and text_prototype in PROTOTYPE_MAP.keys():
            self.prototype = PROTOTYPE_MAP[text_prototype]
        else:
            self.prototype = text_prototype
        self.text_prototype_embeds = None

        self.prompt = PROMPT_MAP[text_prompt]

    def forward(
        self,
        images: torch.Tensor,
        data_samples: Optional[list] = None,
        mode: str = 'predict',
        **kwargs,
    ):
        """The unified entry for a forward process in both training and test.
        The method accepts the following modes:

        - "predict": Forward and return a list of data samples contain the
          predict results.

        Args:
            images (torch.Tensor): the preprocessed image tensor of shape
                ``(N, C, H, W)``.
            data_samples (List[DataSample], optional): The annotation data
                of every samples. Defaults to None.
            mode (str): Return what kind of value. Defaults to 'predict'.
        """
        if mode == 'predict':
            return self.predict(images, data_samples, **kwargs)
        else:
            raise RuntimeError(f'Invalid mode "{mode}".')

    def extract_image_feat(self, images: torch.Tensor) -> torch.Tensor:
        """The function to extract image latent features."""
        if isinstance(self.vision_backbone, ModifiedResNet):
            return self.vision_backbone(images)
        return self.vision_backbone(images)[-1] @ self.vision_projection

    def extract_text_feat(self, texts: torch.Tensor) -> torch.Tensor:
        """The function to extract text latent features."""
        pad_index = self.tokenizer.vocab['[PAD]']
        attn_mask = texts.ne(pad_index)
        # [batch_size, seq_length, hidden_size]
        x = self.text_backbone(texts, attention_mask=attn_mask)[0]
        return x[:, 0, :] @ self.text_projection

    def extract_feat(
            self, images: torch.Tensor,
            texts: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor]]:
        """The function to extract image and text latent features, the input
        image or text can not both be None."""

        assert images is not None or texts is not None, \
            'text and image cannot both be None!'
        if images is None:
            return self.extract_text_feat(texts)
        elif texts is None:
            return self.extract_image_feat(images)

        image_features = self.extract_image_feat(images)
        text_features = self.extract_text_feat(texts)

        image_features = image_features / image_features.norm(
            dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(
            dim=-1, keepdim=True)

        return image_features, text_features

    def compute_similarity(self, images, texts):
        """Extract images and texts features and compute cosine similarity."""
        image_features, text_features = self.extract_feat(
            images=images, texts=texts)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        # shape (N, N)
        return logits_per_image, logits_per_text

    def predict(self,
                images: torch.Tensor,
                data_samples: DataSample = None) -> DataSample:
        """Predict the classes of the input images.

        The prediction is for zero-shot classification and the text prototypes
        will be prepared in thisfunction.

        Args:
            images (torch.Tensor): The input images.
            data_samples (DataSample): The data samples with information from
                dataset.

        Returns:
            DataSample: The results of prediction.
        """

        if self.text_prototype_embeds is None:
            self.prepare_text_prototype(device=images.device)

        image_features = self.extract_image_feat(images=images)
        image_features /= image_features.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logits_per_image = image_features @ self.text_prototype_embeds.to(
            image_features.device) * self.logit_scale.exp()

        pred_scores = F.softmax(logits_per_image, dim=1)
        pred_labels = pred_scores.argmax(dim=1, keepdim=True).detach()

        out_data_samples = []
        if data_samples is None:
            data_samples = [None for _ in range(pred_scores.size(0))]

        for data_sample, score, label in zip(data_samples, pred_scores,
                                             pred_labels):
            if data_sample is None:
                data_sample = DataSample()

            data_sample.set_pred_score(score).set_pred_label(label)
            out_data_samples.append(data_sample)
        return out_data_samples

    def prepare_text_prototype(self, device) -> None:
        """The function to prepare text prototypes with prompt."""
        class_embeddings = []
        for classname in track_on_main_process(self.prototype,
                                               'Prepare text prototype...'):
            # format with class
            texts = [prompt(classname) for prompt in self.prompt]
            tokenized_texts = self.tokenize(texts)
            class_features = self.extract_text_feat(tokenized_texts.to(device))
            class_features /= class_features.norm(dim=-1, keepdim=True)
            class_feature = class_features.mean(dim=0)
            class_feature /= class_feature.norm()
            class_embeddings.append(class_feature)
        self.text_prototype_embeds = torch.stack(
            class_embeddings, dim=1).to(device)

    def tokenize(self, texts: Union[str, List[str]]) -> torch.LongTensor:
        """Returns the tokenized representation of given input string(s)

        Args:
            texts (Union[str, List[str]]): An input string or a list of input
                strings to tokenize
            context_length (int): The context length to use. Defaults to 52.

        Returns:
            torch.Tensor: Resulting tokens.
        """
        if isinstance(texts, str):
            texts = [texts]

        all_tokens = []
        for text in texts:
            # adapt the text to Chinese BERT vocab
            text = text.lower().replace('“', "\"").replace('”', "\"")

            # add special tokens
            all_tokens.append(
                [self.tokenizer.vocab['[CLS]']] +
                self.tokenizer.convert_tokens_to_ids(
                    self.tokenizer.tokenize(text))[:self.context_length - 2] +
                [self.tokenizer.vocab['[SEP]']])

        result = torch.zeros(
            len(all_tokens), self.context_length, dtype=torch.long)

        for i, tokens in enumerate(all_tokens):
            assert len(tokens) <= self.context_length
            result[i, :len(tokens)] = torch.tensor(tokens)

        return result

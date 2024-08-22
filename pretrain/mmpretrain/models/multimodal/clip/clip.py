# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from abc import abstractmethod
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from mmengine.model import BaseModel
from torch import nn

from mmpretrain.datasets.categories import (CIFAR100_CATEGORIES,
                                            IMAGENET_SIMPLE_CATEGORIES)
from mmpretrain.registry import MODELS, TOKENIZER
from mmpretrain.structures import DataSample
from mmpretrain.utils import track_on_main_process
from .utils import (OPENAI_CIFAR100_PROMPT, OPENAI_IMAGENET_PROMPT,
                    OPENAI_IMAGENET_PROMPT_SUB)

CIFAR100_CATEGORIES = [' '.join(c.split('_')) for c in CIFAR100_CATEGORIES]
PROTOTYPE_MAP = {
    'imagenet': IMAGENET_SIMPLE_CATEGORIES,
    'cifar100': CIFAR100_CATEGORIES,
}
PROMPT_MAP = {
    'openai_imagenet': OPENAI_IMAGENET_PROMPT,
    'openai_cifar100': OPENAI_CIFAR100_PROMPT,
    'vanilla': [lambda c: f'a photo of a {c}'],
    'openai_imagenet_sub': OPENAI_IMAGENET_PROMPT_SUB
}


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function."""
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class CLIP(BaseModel):
    """The implementation of `CLIP <https://arxiv.org/abs/2103.00020>`_.

    Args:
        vision_backbone (dict): Config dict for vision backbone.
        text_backbone (dict): Config dict for text backbone.
        tokenizer (dict): Config dict for text tokenizer.
        proj_dim (int): Projection dimension for similarity computation.
        text_prototype (str): Text prototype, which can be a key in
            `PROTOTYPE_MAP` or list of text.
        text_prompt (str): The prompt for text prototype.
            Defaults to 'vanilla',which refers to "a photo of {cls}".
        context_length (int): The context length to use. Defaults to 77.
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
                 projection: dict,
                 text_backbone: dict,
                 tokenizer: dict,
                 vocab_size: int,
                 transformer_width: int,
                 proj_dim: int,
                 context_length: int = 77,
                 data_preprocessor: Optional[dict] = None,
                 init_cfg: Optional[dict] = None):
        if data_preprocessor is None:
            data_preprocessor = {}
        data_preprocessor.setdefault('type', 'MultiModalDataPreprocessor')
        data_preprocessor = MODELS.build(data_preprocessor)

        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)

        self.context_length = context_length

        # build the vision transformer
        self.visual = MODELS.build(vision_backbone)

        # build the visual projection
        self.visual_proj = MODELS.build(projection)

        # build attn_mask for casual-attn
        text_backbone['attn_mask'] = self.build_attention_mask()

        # build the text transformer
        self.transformer = MODELS.build(text_backbone)

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(
            torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(
            torch.empty(transformer_width, proj_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.initialize_parameters()

        self.tokenizer = TOKENIZER.build(tokenizer)

        self.tokenizer.vocab = self.tokenizer.get_vocab(
        )  # CLIPTokenizer has no attribute named 'vocab', so manually

    def initialize_parameters(self) -> None:
        """Initialize the parameters.

        The pretrained weight will override the initialized parameters by this
        function.
        """
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        proj_std = (self.transformer.width**-0.5) * (
            (2 * self.transformer.layers)**-0.5)
        attn_std = self.transformer.width**-0.5
        fc_std = (2 * self.transformer.width)**-0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(
                self.text_projection, std=self.transformer.width**-0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask,
        # with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float('-inf'))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

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
        return self.visual_proj(self.visual(images))[0]

    def extract_text_feat(self, texts: torch.Tensor) -> torch.Tensor:
        """The function to extract text latent features."""
        x = self.token_embedding(texts)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)[0]

        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding
        # (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]),
              texts.argmax(dim=-1)] @ self.text_projection

        return x

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

    @abstractmethod
    def predict(self,
                images: torch.Tensor,
                data_samples: DataSample = None) -> DataSample:
        raise NotImplementedError

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
            # text = text.lower().replace('“', "\"").replace('”', "\"")

            # add special tokens
            all_tokens.append(
                [self.tokenizer.vocab['<|startoftext|>']
                 ] +  # <|startoftext|>代表[CLS] token
                self.tokenizer.convert_tokens_to_ids(
                    self.tokenizer.tokenize(text))[:self.context_length - 2] +
                [self.tokenizer.vocab['<|endoftext|>']])

        result = torch.zeros(
            len(all_tokens), self.context_length, dtype=torch.long)

        for i, tokens in enumerate(all_tokens):
            assert len(tokens) <= self.context_length
            result[i, :len(tokens)] = torch.tensor(tokens)

        return result


@MODELS.register_module()
class CLIPZeroShot(CLIP):

    def __init__(
        self,
        vision_backbone: dict,
        projection: dict,
        text_backbone: dict,
        tokenizer: dict,
        vocab_size: int,
        transformer_width: int,
        proj_dim: int,
        context_length: int = 77,
        data_preprocessor: Optional[dict] = None,
        init_cfg: Optional[dict] = None,
        text_prototype: Union[str, List[str]] = 'imagenet',
        text_prompt: str = 'vanilla',
    ):
        super(CLIPZeroShot,
              self).__init__(vision_backbone, projection, text_backbone,
                             tokenizer, vocab_size, transformer_width,
                             proj_dim, context_length, data_preprocessor,
                             init_cfg)

        # for zero-shot classification
        if isinstance(text_prototype,
                      str) and text_prototype in PROTOTYPE_MAP.keys():
            self.prototype = PROTOTYPE_MAP[text_prototype]
        else:
            self.prototype = text_prototype
        self.text_prototype_embeds = None

        self.prompt = PROMPT_MAP[text_prompt]

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

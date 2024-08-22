# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import random
import re
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from mmengine.logging import MMLogger
from mmengine.model import BaseModel

from mmpretrain.registry import MODELS, TOKENIZER
from mmpretrain.structures import DataSample


@MODELS.register_module()
class MiniGPT4(BaseModel):
    """The multi-modality model of MiniGPT-4.

    The implementation of `MiniGPT-4 <https://arxiv.org/abs/2304.10592>`_.
    Modified from https://github.com/Vision-CAIR/MiniGPT-4/blob/main/minigpt4/models/mini_gpt4.py

    Args:
        vision_encoder (dict): The config for vision encoder.
        q_former_model (dict): The config for Qformer.
        lang_encoder (dict): The config for language model.
        tokenizer (dict): The config for tokenizer.
        task (str): To define the task, which control the processing of text.
            Defaults to 'caption'.
        freeze_vit (bool): Freeze the training of ViT. Defaults to True.
        freeze_q_former (bool): Freeze the training of Qformer. Defaults to
            True.
        num_query_token (int): Number of query tokens of Qformer. Defaults to
            32.
        prompt_template (str): Prompt template of the model. Defaults to
            '###Human: {} ###Assistant: '.
        raw_prompts (list): Prompts for training. Defaults to None.
        max_txt_len (int): Max token length while doing tokenization. Defaults
            to 32.
        end_sym (str): Ended symbol of the sequence. Defaults to '\\n'.
        generation_cfg (dict): The config of text generation. Defaults to
            dict().
        data_preprocessor (:obj:`BaseDataPreprocessor`): Used for
            pre-processing data sampled by dataloader to the format accepted by
            :meth:`forward`. Defaults to None.
        init_cfg (dict): Initialization config dict. Defaults to None.
    """ # noqa

    def __init__(self,
                 vision_encoder: dict,
                 q_former_model: dict,
                 lang_encoder: dict,
                 tokenizer: dict,
                 task: str = 'caption',
                 freeze_vit: bool = True,
                 freeze_q_former: bool = True,
                 num_query_token: int = 32,
                 prompt_template: str = '###Human: {} ###Assistant: ',
                 raw_prompts: Optional[list] = None,
                 max_txt_len: int = 32,
                 end_sym: str = '\n',
                 generation_cfg: dict = dict(),
                 data_preprocessor: Optional[dict] = None,
                 init_cfg: Optional[dict] = None):
        if data_preprocessor is None:
            data_preprocessor = {}
        data_preprocessor.setdefault('type', 'MultiModalDataPreprocessor')
        data_preprocessor = MODELS.build(data_preprocessor)

        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        self.task = task
        logger = MMLogger.get_current_instance()

        # build vision model
        vision_encoder_weight = vision_encoder.pop('pretrained', None)
        self.vision_encoder = MODELS.build(vision_encoder)
        self.ln_vision = nn.LayerNorm(self.vision_encoder.embed_dims)

        if vision_encoder_weight is not None:
            from mmengine.runner.checkpoint import load_checkpoint
            load_checkpoint(self.vision_encoder, vision_encoder_weight)
            self.vision_encoder.is_init = True
        if freeze_vit:
            for name, param in self.ln_vision.named_parameters():
                param.requires_grad = False
            self.ln_vision = self.ln_vision.eval()
        else:
            logger.warning('Please check `frozen_stages` in the dict of'
                           '`vision_encoder`. Also set it to be -1 if do not'
                           'freeze ViT.')

        # build Qformer
        q_former_model_weight = q_former_model.pop('pretrained', None)
        self.q_former = MODELS.build(q_former_model)
        self.q_former.cls = None
        self.q_former.bert.embeddings.word_embeddings = None
        self.q_former.bert.embeddings.position_embeddings = None
        for layer in self.q_former.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None

        self.query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, self.q_former.config.hidden_size))
        self.query_tokens.data.normal_(
            mean=0.0, std=self.q_former.config.initializer_range)

        if q_former_model_weight is not None:
            from mmengine.runner.checkpoint import CheckpointLoader
            state_dict = CheckpointLoader.load_checkpoint(
                q_former_model_weight)['state_dict']
            self.load_state_dict(state_dict, strict=False)
            # The ln_vision weights are also in the q-former checkpoint.
            setattr(self.ln_vision, 'is_init', True)
            setattr(self.q_former, 'is_init', True)

        if freeze_q_former:
            for name, param in self.q_former.named_parameters():
                param.requires_grad = False
            self.q_former.eval()
            self.query_tokens.requires_grad = False

        # build language model
        self.llama_tokenizer = TOKENIZER.build(tokenizer)
        self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token

        self.llama_model = MODELS.build(lang_encoder)
        for name, param in self.llama_model.named_parameters():
            param.requires_grad = False

        # build linear projection layer
        self.llama_proj = nn.Linear(self.q_former.config.hidden_size,
                                    self.llama_model.config.hidden_size)
        self.max_txt_len = max_txt_len
        self.end_sym = end_sym
        self.end_token_id = self.llama_tokenizer.encode(end_sym)[-1]

        # set prompts
        if raw_prompts is not None:
            filted_prompts = [
                raw_prompt for raw_prompt in raw_prompts
                if '<ImageHere>' in raw_prompt
            ]
            self.prompt_list = [
                prompt_template.format(p) for p in filted_prompts
            ]
        else:
            self.prompt_list = []

        # update generation configs
        self.generation_cfg = dict(
            max_new_tokens=300,
            num_beams=1,
            do_sample=True,
            min_length=1,
            top_p=0.9,
            repetition_penalty=1.0,
            length_penalty=1.0,
            temperature=1.0)
        self.generation_cfg.update(**generation_cfg)

        if hasattr(self, 'register_load_state_dict_post_hook'):
            self.register_load_state_dict_post_hook(self._load_llama_proj_hook)

    def encode_img(self,
                   images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """The function to encode the images."""
        device = images.device
        x = self.vision_encoder(images)[0]
        image_embeds = self.ln_vision(x).to(device)
        image_atts = torch.ones(
            image_embeds.size()[:-1], dtype=torch.long).to(device)

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_output = self.q_former.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )

        inputs_llama = self.llama_proj(query_output.last_hidden_state)
        atts_llama = torch.ones(
            inputs_llama.size()[:-1], dtype=torch.long).to(images.device)
        return inputs_llama, atts_llama

    def prompt_wrap(self, img_embeds: torch.Tensor, atts_img: torch.Tensor,
                    prompt: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """The function to wrap the image and prompt.

        Currently, the function only supports applying one prompt to all input
        images in the one batch.

        Args:
            img_embeds (torch.Tensor): The embedding of the input images.
            atts_img (torch.Tensor): Attention map of the image embeddings.
            prompt (str): The prompt of the batch data.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The embedding and attention map.
        """
        if prompt:
            batch_size = img_embeds.shape[0]
            p_before, p_after = prompt.split('<ImageHere>')
            p_before_tokens = self.llama_tokenizer(
                p_before, return_tensors='pt',
                add_special_tokens=False).to(img_embeds.device)
            p_after_tokens = self.llama_tokenizer(
                p_after, return_tensors='pt',
                add_special_tokens=False).to(img_embeds.device)
            p_before_embeds = self.llama_model.model.embed_tokens(
                p_before_tokens.input_ids).expand(batch_size, -1, -1)
            p_after_embeds = self.llama_model.model.embed_tokens(
                p_after_tokens.input_ids).expand(batch_size, -1, -1)
            wrapped_img_embeds = torch.cat(
                [p_before_embeds, img_embeds, p_after_embeds], dim=1)
            wrapped_atts_img = atts_img[:, :1].expand(
                -1, wrapped_img_embeds.shape[1])
            return wrapped_img_embeds, wrapped_atts_img
        else:
            return img_embeds, atts_img

    def loss(self,
             images: torch.Tensor,
             data_samples: Optional[List[DataSample]] = None) -> dict:
        """The forward function in training.

        Args:
            inputs (List[torch.Tensor]): The input images.
            data_samples (List[DataSample]): All elements required
                during the forward function.

        Returns:
            Dict[str, torch.Tensor]: A dictionary of loss components.
        """
        img_embeds, atts_img = self.encode_img(images)

        if self.task == 'caption' and self.prompt_list:
            prompt = random.choice(self.prompt_list)
            img_embeds, atts_img = self.prompt_wrap(img_embeds, atts_img,
                                                    prompt)

        self.llama_tokenizer.padding_side = 'right'

        text = [t + self.end_sym for t in data_samples['text_input']]

        to_regress_tokens = self.llama_tokenizer(
            text,
            return_tensors='pt',
            padding='longest',
            truncation=True,
            max_length=self.max_txt_len,
            add_special_tokens=False).to(images.device)

        targets = to_regress_tokens.input_ids.masked_fill(
            to_regress_tokens.input_ids == self.llama_tokenizer.pad_token_id,
            -100)

        empty_targets = (
            torch.ones([atts_img.shape[0], atts_img.shape[1] + 1],
                       dtype=torch.long).to(images.device).fill_(
                           -100)  # plus one for bos
        )
        targets = torch.cat([empty_targets, targets], dim=1)

        batch_size = img_embeds.shape[0]
        bos = torch.ones([batch_size, 1],
                         dtype=to_regress_tokens.input_ids.dtype,
                         device=to_regress_tokens.input_ids.device
                         ) * self.llama_tokenizer.bos_token_id
        bos_embeds = self.llama_model.model.embed_tokens(bos)
        atts_bos = atts_img[:, :1]

        to_regress_embeds = self.llama_model.model.embed_tokens(
            to_regress_tokens.input_ids)
        inputs_embeds = torch.cat([bos_embeds, img_embeds, to_regress_embeds],
                                  dim=1)
        attention_mask = torch.cat(
            [atts_bos, atts_img, to_regress_tokens.attention_mask], dim=1)

        outputs = self.llama_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
            labels=targets,
        )
        loss = outputs.loss
        return dict(loss=loss)

    def predict(
            self,
            images: torch.Tensor,
            data_samples: Optional[List[DataSample]] = None
    ) -> List[DataSample]:

        with torch.no_grad():
            img_embeds, atts_img = self.encode_img(images)

        if self.task == 'caption' and self.prompt_list:
            prompt = random.choice(self.prompt_list)
            img_embeds, atts_img = self.prompt_wrap(img_embeds, atts_img,
                                                    prompt)

        batch_size = img_embeds.shape[0]
        bos = torch.ones(
            [batch_size, 1], dtype=torch.long,
            device=img_embeds.device) * self.llama_tokenizer.bos_token_id
        bos_embeds = self.llama_model.model.embed_tokens(bos)
        inputs_embeds = torch.cat([bos_embeds, img_embeds], dim=1)

        outputs = self.llama_model.generate(
            inputs_embeds=inputs_embeds,
            eos_token_id=self.end_token_id,
            **self.generation_cfg)

        return self.post_process(outputs, data_samples)

    def post_process(
            self, outputs: torch.Tensor,
            data_samples: Optional[List[DataSample]]) -> List[DataSample]:
        """Perform post process for outputs for different task.

        Args:
            outputs (torch.Tensor): The generated outputs.
            data_samples (List[DataSample], optional): The annotation
                data of every samples.

        Returns:
            List[DataSample]: Return list of data samples.
        """
        outputs = self.llama_tokenizer.batch_decode(
            outputs, skip_special_tokens=True)

        if data_samples is None:
            data_samples = [DataSample() for _ in range(len(outputs))]

        for output, data_sample in zip(outputs, data_samples):
            if self.task == 'caption':
                output = output.split('###')[0]
                output = output.split('Assistant:')[-1].strip()
                data_sample.pred_caption = output
            else:
                # raw output
                data_sample.pred_output = output
        return data_samples

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
        if mode == 'loss':
            return self.loss(images, data_samples)
        elif mode == 'predict':
            return self.predict(images, data_samples, **kwargs)
        else:
            raise RuntimeError(f'Invalid mode "{mode}".')

    @staticmethod
    def _load_llama_proj_hook(module, incompatible_keys):
        """Avoid warning missing keys except LLaMA projection keys."""
        proj_patterns = [
            'vision_encoder.*',
            'ln_vision.*',
            'q_former.*',
            'query_tokens',
            'llama_model.*',
        ]
        for key in list(incompatible_keys.missing_keys):
            if any(re.match(pattern, key) for pattern in proj_patterns):
                incompatible_keys.missing_keys.remove(key)

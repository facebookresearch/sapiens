# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional

import torch

from mmpretrain.registry import MODELS, TOKENIZER
from mmpretrain.structures import DataSample
from ..flamingo.flamingo import ExtendModule, Flamingo, PerceiverResampler


@MODELS.register_module()
class Otter(Flamingo):
    """The Otter model for multiple tasks.

    Args:
        vision_encoder (dict): The config of the vision encoder.
        lang_encoder (dict): The config of the language encoder.
        tokenizer (dict): The tokenizer to encode the text.
        task (int): The task to perform prediction.
        zeroshot_prompt (str): Prompt used for zero-shot inference.
            Defaults to an.
        shot_prompt_tmpl (str): Prompt used for few-shot inference.
            Defaults to ``<image>User:Please describe the image.
            GPT:<answer>{caption}<|endofchunk|>``.
        final_prompt_tmpl (str): Final part of prompt used for inference.
            Defaults to '<image>User:Please describe the image. GPT:<answer>'.
        generation_cfg (dict): The extra generation config, accept the keyword
            arguments of [~`transformers.GenerationConfig`].
            Defaults to an empty dict.
        data_preprocessor (Optional[dict]): The config for preprocessing input
            data. If None or no specified type, it will use
            "MutimodalDataPreprocessor" as type.
            See :class:`MutimodalDataPreprocessor` for more details.
            Defaults to None.
        init_cfg (dict, optional): The initialization config. Defaults to None.
    """

    support_tasks = {'caption', 'vqa'}
    _no_split_modules = [
        'TransformerEncoderLayer', 'PerceiverAttention',
        'GatedCrossAttentionBlock', 'FlamingoLayer'
    ]

    def __init__(
            self,
            vision_encoder: dict,
            lang_encoder: dict,
            tokenizer: dict,
            task: str = 'caption',
            zeroshot_prompt: str = '',
            shot_prompt_tmpl: str = ('<image>User:Please describe the image. '
                                     'GPT:<answer>{caption}<|endofchunk|>'),
            final_prompt_tmpl: str = ('<image>User:Please describe the image. '
                                      'GPT:<answer>'),
            generation_cfg: dict = dict(),
            data_preprocessor: Optional[dict] = None,
            init_cfg: Optional[dict] = None):
        if data_preprocessor is None:
            data_preprocessor = {}
        if isinstance(data_preprocessor, dict):
            data_preprocessor.setdefault('type', 'MultiModalDataPreprocessor')
            data_preprocessor = MODELS.build(data_preprocessor)

        super(Flamingo, self).__init__(
            init_cfg=init_cfg, data_preprocessor=data_preprocessor)

        if task not in self.support_tasks:
            raise ValueError(f'Unsupported task {task}, please select '
                             f'the task from {self.support_tasks}.')
        self.task = task

        # init tokenizer
        self.tokenizer = TOKENIZER.build(tokenizer)
        # add Otter special tokens to the tokenizer
        self.tokenizer.add_special_tokens({
            'additional_special_tokens':
            ['<|endofchunk|>', '<image>', '<answer>']
        })
        self.tokenizer.bos_token_id = 1
        if self.tokenizer.pad_token is None:
            # Issue: GPT models don't have a pad token, which we use to
            # modify labels for the loss.
            self.tokenizer.add_special_tokens({'pad_token': '<PAD>'})

        # Template to format the prompt input
        self.zeroshot_prompt = zeroshot_prompt
        self.shot_prompt_tmpl = shot_prompt_tmpl
        self.final_prompt_tmpl = final_prompt_tmpl

        # init vision encoder related modules
        vision_encoder_weight = vision_encoder.pop('pretrained', None)
        self.vision_encoder = MODELS.build(vision_encoder)
        if vision_encoder_weight is not None:
            from mmengine.runner.checkpoint import load_checkpoint
            load_checkpoint(
                self.vision_encoder,
                vision_encoder_weight,
                map_location='cpu',
                revise_keys=[(r'^backbone\.', '')],
            )
            self.vision_encoder.is_init = True

        self.perceiver = PerceiverResampler(dim=self.vision_encoder.embed_dims)

        # init language encoder related modules
        self.lang_encoder = ExtendModule(**lang_encoder)
        self.lang_encoder.resize_token_embeddings(len(self.tokenizer))
        self.lang_encoder.media_token_id = self.tokenizer.encode('<image>')[-1]

        # other necessary parameters
        self.eoc_token_id = self.tokenizer.encode('<|endofchunk|>')[-1]
        self.generation_cfg = generation_cfg

        if hasattr(self, 'register_load_state_dict_post_hook'):
            self.register_load_state_dict_post_hook(self._load_adapter_hook)

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
        outputs = self.tokenizer.batch_decode(
            outputs, skip_special_tokens=True)

        if data_samples is None:
            data_samples = [DataSample() for _ in range(len(outputs))]

        for output, data_sample in zip(outputs, data_samples):
            # remove text pattern
            if self.task == 'caption':
                data_sample.pred_caption = output
            elif self.task == 'vqa':
                data_sample.pred_answer = output

        return data_samples

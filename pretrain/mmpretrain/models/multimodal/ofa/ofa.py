# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import string
from collections import defaultdict
from functools import partial
from typing import Optional, Union

import mmengine
import torch
from mmengine.model import BaseModel

from mmpretrain.datasets import CleanCaption
from mmpretrain.registry import MODELS, TOKENIZER
from mmpretrain.structures import DataSample
from .ofa_modules import OFAEncoderDecoder


class TreeNode():

    def __init__(self):
        self.child = defaultdict(TreeNode)


class Trie:

    def __init__(self, eos):
        self.root = TreeNode()
        self.eos = eos

    def insert(self, word):
        cur = self.root
        for c in word:
            cur = cur.child[c]

    def get_next_layer(self, word):
        cur = self.root
        for c in word:
            cur = cur.child.get(c)
            if cur is None:
                return [self.eos]
        return list(cur.child.keys())


def apply_constraint(
    input_ids: torch.Tensor,
    logits: torch.Tensor,
    decoder_prompts: Optional[list],
    num_beams: int,
    constraint_trie: Trie = None,
):
    if decoder_prompts is None and constraint_trie is None:
        return logits

    mask = logits.new_zeros(logits[:, -1, :].size(), dtype=torch.bool)
    input_ids = input_ids.view(-1, num_beams, input_ids.shape[-1])
    for batch_id, beam_sent in enumerate(input_ids):
        for beam_id, sent in enumerate(beam_sent):
            if decoder_prompts is None:
                prompt_len = 0
            else:
                prompt_len = len(decoder_prompts[batch_id])

            if sent.size(0) - 1 < prompt_len:
                allowed_tokens = [decoder_prompts[batch_id][sent.size(0) - 1]]
                mask[batch_id * num_beams + beam_id, allowed_tokens] = True
            elif constraint_trie is not None:
                answer_tokens = [0] + sent[prompt_len + 1:].tolist()
                allowed_tokens = constraint_trie.get_next_layer(answer_tokens)
                mask[batch_id * num_beams + beam_id, allowed_tokens] = True
            else:
                mask[batch_id * num_beams + beam_id, :] = True
    logits[:, -1, :].masked_fill_(~mask, float('-inf'))
    return logits


@MODELS.register_module()
class OFA(BaseModel):
    """The OFA model for multiple tasks.

    Args:
        encoder_cfg (dict): The config of the encoder, accept the keyword
            arguments of :class:`OFAEncoder`.
        decoder_cfg (dict): The config of the decoder, accept the keyword
            arguments of :class:`OFADecoder`.
        vocab_size (int): The size of the vocabulary.
        embedding_dim (int): The embedding dimensions of both the encoder
            and the decoder.
        tokenizer (dict | PreTrainedTokenizer): The tokenizer to encode
            the text.
        task (str): The task name, supported tasks are "caption", "vqa" and
            "refcoco".
        prompt (str, optional): The prompt template for the following tasks,
            If None, use default prompt:

            - **caption**: ' what does the image describe?'
            - **refcoco**: ' which region does the text " {} " describe?'

            Defaults to None
        ans2label (str | Sequence | None): The answer to label mapping for
            the vqa task. If a string, it should be a pickle or json file.
            The sequence constrains the output answers. Defaults to None,
            which means no constraint.
        generation_cfg (dict): The extra generation config, accept the keyword
            arguments of :class:`~transformers.GenerationConfig`.
            Defaults to an empty dict.
        data_preprocessor (dict, optional): The config for preprocessing input
            data. If None or no specified type, it will use
            "MultiModalDataPreprocessor" as type. See :class:
            `MultiModalDataPreprocessor` for more details. Defaults to None.
        init_cfg (dict, optional): The initialization config. Defaults to None.
    """
    support_tasks = {'caption', 'vqa', 'refcoco'}

    def __init__(
        self,
        encoder_cfg,
        decoder_cfg,
        vocab_size,
        embedding_dim,
        tokenizer,
        task,
        prompt=None,
        ans2label: Union[dict, str, None] = None,
        generation_cfg=dict(),
        data_preprocessor: Optional[dict] = None,
        init_cfg=None,
    ):
        if data_preprocessor is None:
            data_preprocessor = {}
        if isinstance(data_preprocessor, dict):
            data_preprocessor.setdefault('type', 'MultiModalDataPreprocessor')
            data_preprocessor = MODELS.build(data_preprocessor)

        super().__init__(
            init_cfg=init_cfg, data_preprocessor=data_preprocessor)

        if isinstance(tokenizer, dict):
            self.tokenizer = TOKENIZER.build(tokenizer)
        else:
            self.tokenizer = tokenizer

        if task not in self.support_tasks:
            raise ValueError(f'Unsupported task {task}, please select '
                             f'the task from {self.support_tasks}.')

        self.prompt = prompt
        self.task = task

        if isinstance(ans2label, str):
            self.ans2label = mmengine.load(ans2label)
        else:
            self.ans2label = ans2label

        if self.task == 'vqa' and self.ans2label is not None:
            self.constraint_trie = Trie(eos=self.tokenizer.eos_token_id)
            answers = [f' {answer}' for answer in self.ans2label]
            answer_tokens = self.tokenizer(answers, padding=False)
            for answer_token in answer_tokens['input_ids']:
                self.constraint_trie.insert(answer_token)
        else:
            self.constraint_trie = None

        generation_cfg = {
            'num_beams': 5,
            'max_new_tokens': 20,
            'no_repeat_ngram_size': 3,
            **generation_cfg,
        }
        self.model = OFAEncoderDecoder(
            encoder_cfg=encoder_cfg,
            decoder_cfg=decoder_cfg,
            padding_idx=self.tokenizer.pad_token_id,
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            generation_cfg=generation_cfg,
        )

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

    def predict(
        self,
        images,
        data_samples=None,
        post_process=True,
        **generation_config,
    ):
        text_tokens = self.preprocess_text(data_samples, images.size(0),
                                           images.device)

        if 'images_mask' in data_samples[0]:
            images_mask = torch.tensor([
                sample.get('images_mask') for sample in data_samples
            ]).bool().to(images.device)
        else:
            images_mask = None

        num_beams = generation_config.get(
            'num_beams', getattr(self.model.generation_config, 'num_beams'))
        decoder_prompts = self.get_decoder_prompts(data_samples)
        constrain_fn = partial(
            apply_constraint,
            constraint_trie=self.constraint_trie,
            decoder_prompts=decoder_prompts,
            num_beams=num_beams,
        )

        outputs = self.model.generate(
            input_ids=text_tokens,
            images=images,
            images_mask=images_mask,
            constrain_fn=constrain_fn,
            **generation_config,
        )

        if decoder_prompts is not None:
            # Remove the prefix decoder prompt.
            for prompt_ids, token in zip(decoder_prompts, outputs):
                token[1:len(prompt_ids) + 1] = self.tokenizer.pad_token_id

        if post_process:
            return self.post_process(outputs, data_samples)
        else:
            return outputs

    def get_decoder_prompts(self, data_samples):
        decoder_prompts = []
        if 'decoder_prompt' not in data_samples[0]:
            return None
        for sample in data_samples:
            prompt = ' ' + sample.get('decoder_prompt')
            prompt_ids = self.tokenizer(prompt, add_special_tokens=False)
            prompt_ids = prompt_ids['input_ids']
            decoder_prompts.append(prompt_ids)
        return decoder_prompts

    def preprocess_text(self, data_samples, batch_size, device):
        if self.task == 'caption':
            prompt = self.prompt or ' what does the image describe?'
            prompts = [prompt] * batch_size
            prompts = self.tokenizer(prompts, return_tensors='pt')
            return prompts.input_ids.to(device)
        elif self.task == 'vqa':
            prompts = []
            for sample in data_samples:
                assert 'question' in sample
                prompt = ' ' + sample.get('question')
                prompts.append(prompt)
            prompts = self.tokenizer(
                prompts, return_tensors='pt', padding=True)
            return prompts.input_ids.to(device)
        elif self.task == 'refcoco':
            prompt_template = self.prompt or \
                ' which region does the text " {} " describe?'
            prompts = []
            for sample in data_samples:
                assert 'text' in sample
                prompt = prompt_template.format(sample.get('text'))
                prompts.append(prompt)
            prompts = self.tokenizer(
                prompts, return_tensors='pt', padding=True)
            return prompts.input_ids.to(device)

    def post_process(self, outputs, data_samples):

        out_data_samples = []
        if data_samples is None:
            data_samples = [None] * outputs.size(0)

        for data_sample, token in zip(data_samples, outputs):
            if data_sample is None:
                data_sample = DataSample()

            if self.task == 'caption':
                text = self.tokenizer.decode(token, skip_special_tokens=True)
                text = CleanCaption(
                    lowercase=False,
                    remove_chars=string.punctuation).clean(text)
                data_sample.pred_caption = text
            elif self.task == 'vqa':
                text = self.tokenizer.decode(token, skip_special_tokens=True)
                data_sample.pred_answer = text.strip()
            elif self.task == 'refcoco':
                bbox = token[1:5] - self.tokenizer.bin_offset
                # During training, the bbox is normalized by 512. It's related
                # to the `max_image_size` config in the official repo.
                bbox = bbox / self.tokenizer.num_bins * 512
                scale_factor = data_sample.get('scale_factor', (1, 1))
                bbox[0::2] /= scale_factor[0]
                bbox[1::2] /= scale_factor[1]
                data_sample.pred_bboxes = bbox.unsqueeze(0)
                if 'gt_bboxes' in data_sample:
                    gt_bboxes = bbox.new_tensor(data_sample.gt_bboxes)
                    gt_bboxes[:, 0::2] /= scale_factor[0]
                    gt_bboxes[:, 1::2] /= scale_factor[1]
                    data_sample.gt_bboxes = gt_bboxes
            out_data_samples.append(data_sample)

        return out_data_samples

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, Tuple, Union

import torch
from mmengine.model import BaseModel

from mmpretrain.registry import MODELS, TOKENIZER
from mmpretrain.structures import DataSample


@MODELS.register_module()
class BlipVQA(BaseModel):
    """BLIP VQA.

    Args:
        tokenizer: (dict): The config for tokenizer.
        vision_backbone (dict): Encoder for extracting image features.
        multimodal_backbone (dict): Backbone for extracting
            multi-modal features. We apply this part as VQA fusion module.
        head (dict): The head module to calculate
            loss from processed features.
        data_preprocessor (Optional[dict]): The config for preprocessing input
            data. If None or no specified type, it will use
            `MutimodalDataPreprocessor` as type.
            See :class:`MutimodalDataPreprocessor` for more details.
            Defaults to None.
        init_cfg (Optional[dict]): the config to control the initialization.
            Defaults to None.
    """

    def __init__(self,
                 tokenizer: dict,
                 vision_backbone: dict,
                 multimodal_backbone: dict,
                 head: dict,
                 data_preprocessor: Optional[dict] = None,
                 init_cfg: Optional[dict] = None):

        if data_preprocessor is None:
            data_preprocessor = {}
        data_preprocessor.setdefault('type', 'MultiModalDataPreprocessor')
        data_preprocessor = MODELS.build(data_preprocessor)

        super(BlipVQA, self).__init__(
            init_cfg=init_cfg, data_preprocessor=data_preprocessor)

        self.tokenizer = TOKENIZER.build(tokenizer)
        self.vision_backbone = MODELS.build(vision_backbone)
        self.multimodal_backbone = MODELS.build(multimodal_backbone)
        self.vqa_head = MODELS.build(head)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(
        self,
        images: torch.Tensor,
        data_samples: Optional[List[DataSample]] = None,
        mode: str = 'loss',
    ):
        """The unified entry for a forward process in both training and test.

        - "loss": For training. Forward and return a dict of losses according
          to the given inputs and data samples. Note that this method doesn't
          handle neither back propagation nor optimizer updating, which are
          done in the :meth:`train_step`.
        - "predict": For testing. Forward and return a list of data_sample that
          contains pred_answer for each question.

        Args:
            images (Tensor): A batch of images. The shape of it should be
                (B, C, H, W) for images and (B, T, C, H, W) for videos.
            data_samples (List[DataSample], optional): The annotation data of
                every samples. Required when ``mode="loss"``. Defaults to None.
            mode (str): Return what kind of value. Defaults to 'loss'.

        Returns:
            The return type depends on ``mode``.
            - If ``mode="loss"``, return a dict of tensor.
            - If ``mode="predict"``, return a list of `DataSample`
        """

        if mode == 'loss':
            return self.loss(images, data_samples)
        elif mode == 'predict':
            return self.predict(images, data_samples)
        else:
            raise RuntimeError(f'Invalid mode "{mode}".')

    def extract_feat(self, images: torch.Tensor) -> torch.Tensor:
        """Extract features from the input tensor with shape (N, C, ..).

        Args:
            images (Tensor): A batch of images. The shape of it should be
                (B, C, H, W) for images and (B, T, C, H, W) for videos.

        Returns:
            visual_embeds (Tensor): The output features.
        """
        # extract visual feature
        if images.ndim == 4:
            visual_embeds = self.vision_backbone(images)[0]
        elif images.ndim == 5:
            # [batch, T, C, H, W] -> [batch * T, C, H, W]
            bs = images.size(0)
            images = images.reshape(-1, *images.shape[2:])
            visual_embeds = self.vision_backbone(images)[0]
            # [batch * num_segs, L, dim] -> [batch, num_segs * L, dim]
            visual_embeds = visual_embeds.reshape(bs, -1,
                                                  *visual_embeds.shape[2:])
        else:
            raise ValueError(
                f'Images with {images.ndim} dims is not supported.')
        return visual_embeds

    def loss(
        self,
        images: torch.Tensor,
        data_samples: Optional[List[DataSample]] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor]]:
        """generate train_loss from the input tensor and data_samples.

        Args:
            images (Tensor): A batch of images. The shape of it should be
                (B, C, H, W) for images and (B, T, C, H, W) for videos.
            data_samples (List[DataSample], optional): The annotation
                data of every samples.

        Returns:
            Dict[torch.Tensor]: The losses features.
        """
        visual_embeds = self.extract_feat(images)
        image_atts = torch.ones(
            visual_embeds.size()[:-1], dtype=torch.long).to(self.device)

        questions = []
        for sample in data_samples:
            questions.append(sample.get('question'))
        questions = self.tokenizer(
            questions, padding='longest', return_tensors='pt').to(self.device)

        questions.input_ids[:, 0] = \
            self.tokenizer.additional_special_tokens_ids[0]

        # multimodal fusion
        multimodal_embeds = self.multimodal_backbone(
            questions.input_ids,
            attention_mask=questions.attention_mask,
            encoder_hidden_states=visual_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )

        # put answer from data_samples into tensor form
        answer_raw_text = []
        for sample in data_samples:
            answer_raw_text.extend(sample.gt_answer)
        answer = self.tokenizer(
            answer_raw_text, padding='longest',
            return_tensors='pt').to(self.device)
        answer_targets = answer.input_ids.masked_fill(
            answer.input_ids == self.tokenizer.pad_token_id, -100)
        for sample in data_samples:
            # follow BLIP setting, set answer_weight to 0.2 for VG dataset.
            if not hasattr(sample, 'gt_answer_weight'):
                sample.gt_answer_weight = torch.tensor([0.2])
            else:
                sample.gt_answer_weight = torch.tensor(sample.gt_answer_weight)
        answer_weight = torch.cat(
            [sample.gt_answer_weight for sample in data_samples],
            dim=0).to(self.device)
        answer_count = torch.tensor(
            [len(sample.gt_answer) for sample in data_samples]).to(self.device)

        question_states, question_atts = [], []
        for b, n in enumerate(answer_count):
            question_states += [multimodal_embeds.last_hidden_state[b]] * n
            question_atts += [questions.attention_mask[b]] * n

        question_states = torch.stack(question_states, dim=0).to(self.device)
        question_atts = torch.stack(question_atts, dim=0).to(self.device)

        head_feats = dict(
            answer_input_ids=answer.input_ids,
            answer_attention_mask=answer.attention_mask,
            answer_weight=answer_weight,
            answer_targets=answer_targets,
            question_states=question_states,
            question_atts=question_atts,
            batch_size=len(data_samples),
        )

        losses = self.vqa_head.loss(head_feats)

        return losses

    def predict(
        self,
        images: torch.Tensor,
        data_samples: Optional[List[DataSample]] = None,
    ):
        """update data_samples that contain pred_answer for each question.

        Args:
            images (Tensor): A batch of images. The shape of it should be
                (B, C, H, W) for images and (B, T, C, H, W) for videos.
            data_samples (List[DataSample], optional): The annotation
                data of every samples.

        Returns:
            Dict[torch.Tensor]: The losses features.
        """
        visual_embeds = self.extract_feat(images)
        image_atts = torch.ones(
            visual_embeds.size()[:-1], dtype=torch.long).to(self.device)

        questions = []
        for sample in data_samples:
            questions.append(sample.get('question'))
        questions = self.tokenizer(
            questions, padding='longest', return_tensors='pt').to(self.device)

        questions.input_ids[:, 0] = \
            self.tokenizer.additional_special_tokens_ids[0]

        # multimodal fusion
        multimodal_embeds = self.multimodal_backbone(
            questions.input_ids,
            attention_mask=questions.attention_mask,
            encoder_hidden_states=visual_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )

        if self.vqa_head.inference_method == 'rank':
            answer_candidates = self.tokenizer(
                self.vqa_head.answer_list,
                padding='longest',
                return_tensors='pt').to(self.device)
            answer_candidates.input_ids[:, 0] = self.tokenizer.bos_token_id
        elif self.vqa_head.inference_method == 'generate':
            answer_candidates = None

        head_feats = dict(
            multimodal_embeds=multimodal_embeds.last_hidden_state,
            question_atts=questions.attention_mask,
            answer_candidates=answer_candidates,
            bos_token_id=self.tokenizer.bos_token_id,
            sep_token_id=self.tokenizer.sep_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        if self.vqa_head.inference_method == 'rank':
            answers = self.vqa_head.predict(head_feats)
            for answer, data_sample in zip(answers, data_samples):
                data_sample.pred_answer = answer

        elif self.vqa_head.inference_method == 'generate':
            outputs = self.vqa_head.predict(head_feats)
            for output, data_sample in zip(outputs, data_samples):
                data_sample.pred_answer = self.tokenizer.decode(
                    output, skip_special_tokens=True)

        return data_samples

#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

from typing import List, Optional, Union

import torch
import torch.nn as nn
from transformers import PreTrainedModel

DEFAULT_IMAGE_TOKEN = '<image>'
DEFAULT_IMAGE_PATCH_TOKEN = '<im_patch>'
DEFAULT_IM_START_TOKEN = '<im_start>'
DEFAULT_IM_END_TOKEN = '<im_end>'


class LlavaLlamaForCausalLM(PreTrainedModel):

    def __init__(self,
                 vision_encoder,
                 lang_encoder,
                 mm_hidden_size,
                 use_im_start_end=True,
                 use_mm_proj=True,
                 im_start_token: Optional[int] = None,
                 im_end_token: Optional[int] = None,
                 im_patch_token: Optional[int] = None,
                 mm_vision_select_layer: int = -1):
        super().__init__(lang_encoder.config)
        self.vision_tower = vision_encoder
        self.lang_encoder = lang_encoder

        self.use_im_start_end = use_im_start_end
        self.im_start_token = im_start_token
        self.im_end_token = im_end_token
        self.im_patch_token = im_patch_token
        self.mm_hidden_size = mm_hidden_size
        self.mm_vision_select_layer = mm_vision_select_layer
        self.lang_hidden_size = lang_encoder.config.hidden_size

        if use_mm_proj and not hasattr(lang_encoder.model, 'mm_projector'):
            mm_projector = nn.Linear(self.mm_hidden_size,
                                     self.lang_hidden_size)
            self.lang_encoder.model.add_module('mm_projector', mm_projector)
        elif not use_mm_proj:
            self.lang_encoder.model.add_module('mm_projector', nn.Identity())

        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
    ):
        output_attentions = (
            output_attentions if output_attentions is not None else
            self.config.output_attentions)
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else
            self.config.output_hidden_states)
        return_dict = (
            return_dict
            if return_dict is not None else self.config.use_return_dict)

        # decoder outputs consists of
        # (dec_features, layer_state, dec_hidden, dec_attn)
        if inputs_embeds is None:
            inputs_embeds = self.lang_encoder.model.embed_tokens(input_ids)

        inputs_embeds = self.forward_vision_tower(input_ids, inputs_embeds,
                                                  images)

        return self.lang_encoder(
            input_ids=None,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            labels=labels,
        )

    def prepare_inputs_for_generation(self,
                                      input_ids,
                                      past_key_values=None,
                                      attention_mask=None,
                                      inputs_embeds=None,
                                      **kwargs):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use
        # them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {'inputs_embeds': inputs_embeds}
        else:
            model_inputs = {'input_ids': input_ids}

        model_inputs.update({
            'past_key_values': past_key_values,
            'use_cache': kwargs.get('use_cache'),
            'attention_mask': attention_mask,
            'images': kwargs.get('images', None),
        })
        return model_inputs

    def forward_vision_tower(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: torch.FloatTensor,
        images: Union[torch.FloatTensor, list, None] = None,
    ):
        if self.use_im_start_end:
            assert self.im_start_token is not None
            assert self.im_end_token is not None
        if images is not None:
            assert self.im_patch_token is not None

        if self.vision_tower is None or images is None or (
                input_ids.shape[1] == 1 and not self.training):
            return inputs_embeds

        with torch.no_grad():
            if isinstance(images, (list, tuple)):
                # variable length images
                image_features = []
                for image in images:
                    feats = self.vision_tower(image.unsqueeze(0))
                    image_feature = feats[self.mm_vision_select_layer][:, 1:]
                    image_features.append(image_feature)
            else:
                feats = self.vision_tower(images)
                image_features = feats[self.mm_vision_select_layer][:, 1:]

        mm_projector = self.lang_encoder.model.mm_projector
        if isinstance(images, (list, tuple)):
            image_features = [
                mm_projector(image_feature)[0]
                for image_feature in image_features
            ]
        else:
            image_features = mm_projector(image_features)

        dummy_image_features = torch.zeros(
            256, 1024, device=inputs_embeds.device, dtype=inputs_embeds.dtype)
        dummy_image_features = mm_projector(dummy_image_features)

        new_input_embeds = []
        cur_image_idx = 0
        for cur_input_ids, cur_input_embeds in zip(input_ids, inputs_embeds):
            if (cur_input_ids != self.im_patch_token).all():
                # multimodal LLM, but the current sample is not multimodal
                cur_input_embeds = cur_input_embeds + (
                    0. * dummy_image_features).sum()
                new_input_embeds.append(cur_input_embeds)
                cur_image_idx += 1
                continue
            if self.use_im_start_end:
                cur_image_features = image_features[cur_image_idx]
                num_patches = cur_image_features.shape[0]
                if (cur_input_ids == self.im_start_token).sum() != (
                        cur_input_ids == self.im_end_token).sum():
                    raise ValueError('The number of image start tokens and '
                                     'image end tokens should be the same.')
                image_start_tokens = torch.where(
                    cur_input_ids == self.im_start_token)[0]
                for image_start_token_pos in image_start_tokens:
                    cur_image_features = image_features[cur_image_idx].to(
                        device=cur_input_embeds.device)
                    num_patches = cur_image_features.shape[0]
                    if cur_input_ids[image_start_token_pos + num_patches +
                                     1] != self.im_end_token:
                        raise ValueError('The image end token should follow '
                                         'the image start token.')
                    cur_new_input_embeds = torch.cat(
                        (cur_input_embeds[:image_start_token_pos + 1],
                         cur_image_features,
                         cur_input_embeds[image_start_token_pos + num_patches +
                                          1:]),
                        dim=0)
                    cur_image_idx += 1
                new_input_embeds.append(cur_new_input_embeds)
            else:
                cur_image_features = image_features[cur_image_idx]
                num_patches = cur_image_features.shape[0]
                if (cur_input_ids == self.im_patch_token).sum() != num_patches:
                    print(f'Debug: num_patches: {num_patches}')
                    raise ValueError(
                        'The number of image patch tokens should '
                        'be the same as the number of image patches.')
                masked_indices = torch.where(
                    cur_input_ids == self.im_patch_token)[0]
                mask_index_start = masked_indices[0]
                if (masked_indices != torch.arange(
                        mask_index_start,
                        mask_index_start + num_patches,
                        device=masked_indices.device,
                        dtype=masked_indices.dtype)).any():
                    raise ValueError(
                        'The image patch tokens should be consecutive.')
                cur_new_input_embeds = torch.cat(
                    (cur_input_embeds[:mask_index_start], cur_image_features,
                     cur_input_embeds[mask_index_start + num_patches:]),
                    dim=0)
                new_input_embeds.append(cur_new_input_embeds)
                cur_image_idx += 1
        inputs_embeds = torch.stack(new_input_embeds, dim=0)

        return inputs_embeds

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (tuple(
                past_state.index_select(0, beam_idx)
                for past_state in layer_past), )
        return reordered_past

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Optional, Tuple, Union

import mmengine.dist as dist
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.utils import track_iter_progress

from mmpretrain.registry import MODELS, TOKENIZER
from mmpretrain.structures import DataSample
from ..blip.blip_retrieval import BlipRetrieval, all_gather_concat


@MODELS.register_module()
class Blip2Retrieval(BlipRetrieval):
    """BLIP2 Retriever.

    Args:
        vision_backbone (dict): Backbone for extracting image features.
        text_backbone (dict): Backbone for extracting text features.
        multimodal_backbone (Optional[dict]): Backbone for extracting
            multi-modal features.
        vision_neck (Optional[dict]): The neck module to process image features
            from vision backbone. Defaults to None.
        text_neck (Optional[dict]): The neck module to process text features
            from text backbone. Defaults to None.
        head (Optional[Union[List[dict], dict]]): The head module to calculate
            loss from processed single modality features.
            See :mod:`mmmultimodal.models.heads`.
            Notice that if the head is not set, `loss` method cannot be used.
            Defaults to None.
        multimodal_head (Optional[Union[List[dict], dict]]): The multi-modal
            head module to calculate loss from processed multimodal features.
            See :mod:`mmmultimodal.models.heads`.
            Notice that if the head is not set, `loss` method cannot be used.
            Defaults to None.
        tokenizer (Optional[dict]): The config for tokenizer. Defaults to None.
        temperature (float): Temperature parameter that controls the
            concentration level of the distribution. Defaults to 0.07.
        fast_match (bool): If False, select topk similarity as candidates and
            compute the matching score. If True, return the similarity as the
            matching score directly. Defaults to False.
        topk (int): Select topk similarity as candidates for compute matching
            scores. Notice that this is not the topk in evaluation.
            Defaults to 256.
        data_preprocessor (Optional[dict]): The config for preprocessing input
            data. If None or no specified type, it will use
            "MultiModalDataPreprocessor" as type.
            See :class:`MultiModalDataPreprocessor` for more details.
            Defaults to None.
        init_cfg (Optional[dict]): the config to control the initialization.
            Defaults to None.
    """

    def __init__(self,
                 vision_backbone: dict,
                 text_backbone: Optional[dict] = None,
                 multimodal_backbone: Optional[dict] = None,
                 vision_neck: Optional[dict] = None,
                 text_neck: Optional[dict] = None,
                 head: Optional[Union[List[dict], dict]] = None,
                 multimodal_head: Optional[Union[List[dict], dict]] = None,
                 tokenizer: Optional[dict] = None,
                 temperature: float = 0.07,
                 fast_match: bool = False,
                 topk: int = 256,
                 data_preprocessor: Optional[dict] = None,
                 init_cfg: Optional[dict] = None) -> None:
        if data_preprocessor is None:
            data_preprocessor = {}
        if isinstance(data_preprocessor, dict):
            data_preprocessor.setdefault('type', 'MultiModalDataPreprocessor')
            data_preprocessor = MODELS.build(data_preprocessor)

        # Skip BlipRetrieval init
        super(BlipRetrieval, self).__init__(
            init_cfg=init_cfg, data_preprocessor=data_preprocessor)

        self.vision_backbone = MODELS.build(vision_backbone)
        self.ln_vision_backbone = nn.LayerNorm(self.vision_backbone.embed_dims)
        self.tokenizer = TOKENIZER.build(tokenizer)

        if text_backbone is not None:
            self.text_backbone = MODELS.build(text_backbone)

        if multimodal_backbone is not None:
            self.multimodal_backbone = MODELS.build(multimodal_backbone)
            self.multimodal_backbone.resize_token_embeddings(
                len(self.tokenizer))
        self.query_tokens = nn.Parameter(
            torch.zeros(1, self.multimodal_backbone.bert.config.query_length,
                        self.multimodal_backbone.bert.config.hidden_size))
        self.query_tokens.data.normal_(
            mean=0.0,
            std=self.multimodal_backbone.bert.config.initializer_range)

        if vision_neck is not None:
            self.vision_neck = MODELS.build(vision_neck)

        if text_neck is not None:
            self.text_neck = MODELS.build(text_neck)

        if head is not None:
            self.head = MODELS.build(head)

        if multimodal_head is not None:
            self.multimodal_head = MODELS.build(multimodal_head)

        self.temp = nn.Parameter(temperature * torch.ones([]))

        # Notice that this topk is used for select k candidate to compute
        # image-text score, but not the final metric topk in evaluation.
        self.fast_match = fast_match
        self.topk = topk

    def _extract_feat(self, inputs: Union[torch.Tensor, dict],
                      modality: str) -> Tuple[torch.Tensor]:
        """Extract features from the single modality.
        Args:
            inputs (Union[torch.Tensor, dict]): A batch of inputs.
                For image, a tensor of shape (N, C, ...) in general.
                For text, a dict of tokenized text inputs.
            modality (str): Modality feature to be extracted. Only two
                options are supported.

                - ``images``: Only extract image features, mostly used for
                    inference.
                - ``texts``: Only extract text features, mostly used for
                    inference.
        Returns:
            Tuple[torch.Tensor]: The output features.
        """
        if modality == 'images':
            # extract image features
            # TODO:
            # Add layernorm inside backbone and handle the concat outside
            image_embeds = self.ln_vision_backbone(
                self.vision_backbone(inputs)[0])
            image_atts = torch.ones(
                image_embeds.size()[:-1], dtype=torch.long).to(self.device)

            query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1,
                                                    -1)
            query_output = self.multimodal_backbone.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                use_cache=True,
                return_dict=True,
            )
            image_feat = F.normalize(
                self.vision_neck([query_output.last_hidden_state]), dim=-1)
            return {
                'image_embeds': image_embeds,
                'image_feat': image_feat,
                'query_output': query_output
            }
        elif modality == 'texts':
            # extract text features
            text_output = self.multimodal_backbone.bert(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                return_dict=True,
            )
            text_embeds = text_output.last_hidden_state
            text_feat = F.normalize(
                self.text_neck([text_embeds[:, 0, :]]), dim=-1)
            return {'text_embeds': text_embeds, 'text_feat': text_feat}
        else:
            raise RuntimeError(f'Invalid modality "{modality}".')

    def loss(
        self,
        images: torch.Tensor,
        data_samples: Optional[List[DataSample]] = None,
    ) -> Dict[str, torch.tensor]:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            inputs (dict): A batch of inputs. The input tensor with of
                at least one modality. For image, the value is a tensor
                of shape (N, C, ...) in general.
                For text, the value is a dict of tokenized text inputs.
            data_samples (Optional[List[DataSample]]):
                The annotation data of every samples. Defaults to None.

        Returns:
            Dict[str, torch.tensor]: a dictionary of loss components of
                both head and multimodal head.
        """
        output = self.extract_feat(images, data_samples)

        text_ids = output['text_ids']
        text_attn_mask = output['text_attn_mask']
        image_embeds = output['image_embeds']
        image_feat = output['image_feat']
        text_feat = output['text_feat']
        query_output = output['query_output']

        # ITC Loss
        # B*world_size, num_query, D
        image_feat_all = torch.cat(dist.all_gather(image_feat))
        # B*world_size, D
        text_feat_all = torch.cat(dist.all_gather(text_feat))

        # B, B*world_size, num_query
        sim_q2t = torch.matmul(
            image_feat.unsqueeze(1), text_feat_all.unsqueeze(-1)).squeeze()

        # image to text similarity
        sim_i2t, _ = sim_q2t.max(-1)
        sim_i2t = sim_i2t / self.temp

        # B, B*world_size, num_query
        sim_t2q = torch.matmul(
            text_feat.unsqueeze(1).unsqueeze(1),
            image_feat_all.permute(0, 2, 1)).squeeze()

        # text-image similarity
        sim_t2i, _ = sim_t2q.max(-1)
        sim_t2i = sim_t2i / self.temp

        rank = dist.get_rank()
        bs = images.size(0)
        targets = torch.linspace(
            rank * bs, rank * bs + bs - 1, bs, dtype=int).to(self.device)

        itc_loss = (F.cross_entropy(sim_i2t, targets, label_smoothing=0.1) +
                    F.cross_entropy(sim_t2i, targets, label_smoothing=0.1)) / 2

        # prepare for itm
        text_input_ids_world = torch.cat(dist.all_gather(text_ids))
        text_attention_mask_world = torch.cat(dist.all_gather(text_attn_mask))
        image_embeds_world = torch.cat(dist.all_gather(image_embeds))
        with torch.no_grad():
            weights_t2i = F.softmax(sim_t2i, dim=1) + 1e-4
            weights_t2i[:, rank * bs:rank * bs + bs].fill_diagonal_(0)
            weights_i2t = F.softmax(sim_i2t, dim=1) + 1e-4
            weights_i2t[:, rank * bs:rank * bs + bs].fill_diagonal_(0)

        # select a negative image for each text
        image_embeds_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_t2i[b], 1).item()
            image_embeds_neg.append(image_embeds_world[neg_idx])
        image_embeds_neg = torch.stack(image_embeds_neg, dim=0)

        # select a negative text for each image
        text_ids_neg = []
        text_atts_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_i2t[b], 1).item()
            text_ids_neg.append(text_input_ids_world[neg_idx])
            text_atts_neg.append(text_attention_mask_world[neg_idx])

        text_ids_neg = torch.stack(text_ids_neg, dim=0)
        text_atts_neg = torch.stack(text_atts_neg, dim=0)

        text_ids_all = torch.cat([text_ids, text_ids, text_ids_neg],
                                 dim=0)  # pos, pos, neg
        text_atts_all = torch.cat(
            [text_attn_mask, text_attn_mask, text_atts_neg],
            dim=0,
        )

        query_tokens_itm = self.query_tokens.expand(text_ids_all.shape[0], -1,
                                                    -1)
        query_atts_itm = torch.ones(
            query_tokens_itm.size()[:-1], dtype=torch.long).to(self.device)
        attention_mask_all = torch.cat([query_atts_itm, text_atts_all], dim=1)

        image_embeds_all = torch.cat(
            [image_embeds, image_embeds_neg, image_embeds],
            dim=0)  # pos, neg, pos
        image_atts_all = torch.ones(
            image_embeds_all.size()[:-1], dtype=torch.long).to(self.device)

        output_itm = self.multimodal_backbone.bert(
            text_ids_all,
            query_embeds=query_tokens_itm,
            attention_mask=attention_mask_all,
            encoder_hidden_states=image_embeds_all,
            encoder_attention_mask=image_atts_all,
            return_dict=True,
        )

        vl_embeddings = output_itm.last_hidden_state[:, :query_tokens_itm.
                                                     size(1), :]

        # create false data samples
        data_samples.extend(
            [DataSample(is_matched=False) for _ in range(2 * bs)])
        loss_multimodal = self.multimodal_head.loss((vl_embeddings, ),
                                                    data_samples)

        # LM loss
        decoder_input_ids = text_ids.clone()
        decoder_input_ids[:, 0] = self.tokenizer.bos_token_id
        labels = decoder_input_ids.masked_fill(
            decoder_input_ids == self.tokenizer.pad_token_id, -100)

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_atts = torch.ones(
            query_tokens.size()[:-1], dtype=torch.long).to(self.device)
        attention_mask = torch.cat([query_atts, text_attn_mask], dim=1)
        lm_output = self.multimodal_backbone(
            decoder_input_ids,
            attention_mask=attention_mask,
            past_key_values=query_output.past_key_values,
            return_dict=True,
            labels=labels,
        )

        return dict(
            itc_loss=itc_loss, **loss_multimodal, lm_loss=lm_output.loss)

    def predict_all(self,
                    feats: Dict[str, torch.Tensor],
                    data_samples: List[DataSample],
                    num_images: int = None,
                    num_texts: int = None,
                    cal_i2t: bool = True,
                    cal_t2i: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute similarity matrix between images and texts across all ranks.

        Args:
            feats (Dict[str, torch.Tensor]): Features from the current rank.
            data_samples (List[DataSample]): Data samples from the current
                rank.
            num_images (int, optional): Number of images to use.
                Defaults to None.
            num_texts (int, optional): Number of texts to use.
                Defaults to None.
            cal_i2t (bool, optional): Whether to compute image-to-text
                similarity. Defaults to True.
            cal_t2i (bool, optional): Whether to compute text-to-image
                similarity. Defaults to True.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Image-to-text and text-to-image
            similarity matrices.
        """
        text_ids = feats['text_ids']
        text_attn_mask = feats['text_attn_mask']
        image_embeds = feats.get('image_embeds', None)
        image_feat = feats['image_feat']
        text_feat = feats['text_feat']

        num_images = num_images or image_feat.size(0)
        num_texts = num_texts or text_feat.size(0)

        if not self.fast_match:
            image_embeds_all = all_gather_concat(image_embeds)[:num_images]
        else:
            image_embeds_all = None
        image_feat_all = all_gather_concat(image_feat)[:num_images]
        text_feat_all = all_gather_concat(text_feat)[:num_texts]
        text_ids_all = all_gather_concat(text_ids)[:num_texts]
        text_attn_mask_all = all_gather_concat(text_attn_mask)[:num_texts]

        results = []
        if cal_i2t:
            result_i2t = self.compute_score_matrix_i2t(
                image_feat,
                image_embeds,
                text_feat_all,
                text_ids_all,
                text_attn_mask_all,
            )
            results.append(
                self._get_predictions(result_i2t, data_samples, mode='i2t'))
        if cal_t2i:
            result_t2i = self.compute_score_matrix_t2i(
                image_feat_all,
                image_embeds_all,
                text_feat,
                text_ids,
                text_attn_mask,
            )
            results.append(
                self._get_predictions(result_t2i, data_samples, mode='t2i'))
        return tuple(results)

    def compute_score_matrix_i2t(self, img_feats: torch.Tensor,
                                 img_embeds: List[torch.Tensor],
                                 text_feats: torch.Tensor,
                                 text_ids: torch.Tensor,
                                 text_atts: torch.Tensor) -> torch.Tensor:
        """Compare the score matrix for image-to-text retrieval. Every image
        should compare to all the text features.

        Args:
            img_feats (torch.Tensor): The input tensor with shape (M, C).
                M stands for numbers of samples on a single GPU.
            img_embeds (List[torch.Tensor]): Image features from each layer of
                the vision backbone.
            text_feats (torch.Tensor): The input tensor with shape (N, C).
                N stands for numbers of all samples on all GPUs.
            text_ids (torch.Tensor): The input tensor with shape (N, C).
            text_atts (torch.Tensor): The input tensor with shape (N, C).

        Returns:
            torch.Tensor: Score matrix of image-to-text retrieval.
        """

        # compute i2t sim matrix
        # TODO: check correctness
        sim_matrix_i2t, _ = (img_feats @ text_feats.t()).max(1)
        if self.fast_match:
            return sim_matrix_i2t

        score_matrix_i2t = torch.full((img_feats.size(0), text_feats.size(0)),
                                      -100.0).to(self.device)

        for i in track_iter_progress(range(img_feats.size(0))):
            sims = sim_matrix_i2t[i]
            topk_sim, topk_idx = sims.topk(k=self.topk, dim=0)
            # get repeated image embeddings
            encoder_output = img_embeds[i].repeat(self.topk, 1, 1)
            encoder_att = torch.ones(
                encoder_output.size()[:-1], dtype=torch.long).to(self.device)
            # query embeds and attention masks
            query_tokens = self.query_tokens.expand(encoder_output.shape[0],
                                                    -1, -1)
            query_atts = torch.ones(
                query_tokens.size()[:-1], dtype=torch.long).to(self.device)
            attention_mask = torch.cat([query_atts, text_atts[topk_idx]],
                                       dim=1)
            output = self.multimodal_backbone.bert(
                text_ids[topk_idx],
                query_embeds=query_tokens,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_output,
                encoder_attention_mask=encoder_att,
                return_dict=True,
            )
            score = self.multimodal_head(
                (output.last_hidden_state[:, :query_tokens.size(1), :],
                 ))[:, :, 1].mean(dim=1)
            score_matrix_i2t[i, topk_idx] = score + topk_sim

        return score_matrix_i2t

    def compute_score_matrix_t2i(self, img_feats: torch.Tensor,
                                 img_embeds: List[torch.Tensor],
                                 text_feats: torch.Tensor,
                                 text_ids: torch.Tensor,
                                 text_atts: torch.Tensor) -> torch.Tensor:
        """Compare the score matrix for text-to-image retrieval.

        Every text should compare to all the image features.

        Args:
            img_feats (torch.Tensor): The input tensor with shape (N, C).
                N stands for numbers of all samples on all GPUs.
            img_embeds (List[torch.Tensor]): Image features from each layer of
                the vision backbone.
            text_feats (torch.Tensor): The input tensor with shape (M, C).
                M stands for numbers of samples on a single GPU.
            text_ids (torch.Tensor): The input tensor with shape (M, C).
            text_atts (torch.Tensor): The input tensor with shape (M, C).

        Returns:
            torch.Tensor: Score matrix of text-to-image retrieval.
        """

        # compute t2i sim matrix
        # TODO: check correctness
        sim_matrix_i2t, _ = (img_feats @ text_feats.t()).max(1)
        sim_matrix_t2i = sim_matrix_i2t.t()
        if self.fast_match:
            return sim_matrix_i2t

        score_matrix_t2i = torch.full((text_feats.size(0), img_feats.size(0)),
                                      -100.0).to(self.device)

        for i in track_iter_progress(range(text_feats.size(0))):
            sims = sim_matrix_t2i[i]
            topk_sim, topk_idx = sims.topk(k=self.topk, dim=0)
            # get topk image embeddings
            encoder_output = img_embeds[topk_idx]
            encoder_att = torch.ones(
                encoder_output.size()[:-1], dtype=torch.long).to(self.device)
            # get query embeds and attention masks
            query_tokens = self.query_tokens.expand(encoder_output.shape[0],
                                                    -1, -1)
            query_atts = torch.ones(
                query_tokens.size()[:-1], dtype=torch.long).to(self.device)
            attention_mask = torch.cat(
                [query_atts, text_atts[i].repeat(self.topk, 1)], dim=1)
            output = self.multimodal_backbone.bert(
                text_ids[i].repeat(self.topk, 1),
                query_embeds=query_tokens,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_output,
                encoder_attention_mask=encoder_att,
                return_dict=True,
            )
            score = self.multimodal_head(
                (output.last_hidden_state[:, :query_tokens.size(1), :],
                 ))[:, :, 1].mean(dim=1)
            score_matrix_t2i[i, topk_idx] = score + topk_sim

        return score_matrix_t2i

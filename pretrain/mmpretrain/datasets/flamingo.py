# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import random
from abc import abstractmethod
from collections import Counter
from typing import List

import mmengine
import numpy as np
from mmengine.dataset import BaseDataset
from pycocotools.coco import COCO

from mmpretrain.registry import DATASETS
from .coco_vqa import COCOVQA


class FlamingoFewShotMixin:
    """Flamingo fewshot eval dataset minin.

    Args:
        num_shots (int): Number of shots to perform evaluation.
            Defaults to 0.
            Note: 0 does not mean a strict zero-shot in Flamingo setting.
            It will use 2 only-text prompt without in context images.
        num_support_examples (int): Number of support examples to get the
            few shots from. Defaults to 2048.
        num_query_examples (int): Number of query examples to perform the
            final evaluation. Defaults to 5000.
        incontext_prompt_temp (str): In context prompt template for few shot
            examples. Defaults to ''.
        final_prompt_temp (str): Final query prompt template. Defaults to ''.
        **kwargs: Other keyword arguments in :class:`BaseDataset`.
    """

    def __init__(self,
                 num_shots: int = 0,
                 num_support_examples: int = 2048,
                 num_query_examples: int = 5000,
                 incontext_prompt_temp: str = '',
                 final_prompt_temp: str = '',
                 **kwarg):
        self.num_shots = num_shots
        self.num_support_examples = num_support_examples
        self.num_query_examples = num_query_examples
        self.incontext_prompt_temp = incontext_prompt_temp
        self.final_prompt_temp = final_prompt_temp
        super().__init__(**kwarg)

    def get_subset_idx(self, total_num):
        random_idx = np.random.choice(
            total_num,
            self.num_support_examples + self.num_query_examples,
            replace=False)

        support_idx = random_idx[:self.num_support_examples]
        query_idx = random_idx[self.num_support_examples:]
        return support_idx, query_idx

    @abstractmethod
    def parse_basic_anno(self, anno: dict) -> dict:
        """Parse basic annotation for support and query set."""
        pass

    @abstractmethod
    def parse_fewshot_anno(self, anno: dict, support_list: List) -> dict:
        """Parse fewshot related annotation for query set with support list."""
        pass


@DATASETS.register_module()
class FlamingoEvalCOCOVQA(FlamingoFewShotMixin, COCOVQA):
    """Flamingo few shot VQAv2 dataset.

    Args:
        data_root (str): The root directory for ``data_prefix`` and
            ``ann_file``.
        ann_file (str): Annotation file path.
        question_file (str): Question file path.
        num_shots (int): Number of shots to perform evaluation.
            Defaults to 0.
            Note: 0 does not mean a strict zero-shot in Flamingo setting.
            It will use 2 only-text prompt without in context images.
        num_support_examples (int): Number of support examples to get the
            few shots from. Defaults to 2048.
        num_query_examples (int): Number of query examples to perform the
            final evaluation. Defaults to 5000.
        **kwargs: Other keyword arguments in :class:`BaseDataset`.
    """

    def __init__(self,
                 data_root: str,
                 question_file: str,
                 ann_file: str = '',
                 num_shots: int = 0,
                 num_support_examples: int = 2048,
                 num_query_examples: int = 5000,
                 **kwarg):
        super().__init__(
            data_root=data_root,
            question_file=question_file,
            ann_file=ann_file,
            num_shots=num_shots,
            num_support_examples=num_support_examples,
            num_query_examples=num_query_examples,
            **kwarg)

    def parse_basic_anno(self, ann: dict) -> dict:
        """Parse basic annotation for support and query set.

        Args:
            anno (dict): Annotation for single example.

        Return:
            dict: Parsed annotation for single example.
        """
        if ann is None:
            return {}

        answers = [a['answer'] for a in ann['answers']]
        count = Counter(answers)
        answer_weight = [i / len(answers) for i in count.values()]
        answer_info = {
            'gt_answer': list(count.keys()),
            'gt_answer_weight': answer_weight
        }
        return answer_info

    def parse_fewshot_anno(self, query: dict, support_list: List) -> dict:
        """Parse fewshot related annotation for query set with support list.

        Args:
            anno (dict): Annotation for single example.
            support_list (List): List of support subset to subsample few shots.

        Return:
            dict: Parsed annotation for single example.
        """
        # prepare n shots examples
        shots = random.sample(support_list, self.num_shots)

        # append image path for n shots
        img_path = [shot['img_path'] for shot in shots]
        img_path.append(query['img_path'])
        query['img_path'] = img_path

        query['shots'] = [
            dict(
                question=item['question'],
                answer=item['gt_answer'][0],
            ) for item in shots
        ]
        return query

    def load_data_list(self) -> List[dict]:
        """Load data list."""
        questions = mmengine.load(self.question_file)['questions']
        if self.ann_file:
            annotations = mmengine.load(self.ann_file)['annotations']
            assert len(questions) == len(annotations)
        else:
            annotations = [None] * len(questions)
            if self.num_shots > 0:
                raise ValueError('Unable to construct few-shot examples '
                                 'since no annotation file.')

        # The original VQAv2 annotation file and question file includes
        # only image id but no image file paths.
        self.image_index = self._create_image_index()

        num_data = len(questions)
        support_idx, query_idx = self.get_subset_idx(num_data)

        # prepare support subset
        if self.num_shots > 0:
            support_list = []
            for idx in support_idx:
                question = questions[idx]
                ann = annotations[idx]
                support = {**question, **self.parse_basic_anno(ann)}
                support['img_path'] = self.image_index[question['image_id']]
                support_list.append(support)

        # prepare query subset
        data_list = []
        for idx in query_idx:
            question = questions[idx]
            ann = annotations[idx]
            data_info = {**question, **self.parse_basic_anno(ann)}
            data_info['img_path'] = self.image_index[question['image_id']]
            if self.num_shots > 0:
                data_info = self.parse_fewshot_anno(data_info, support_list)
            data_list.append(data_info)

        return data_list


@DATASETS.register_module()
class FlamingoEvalCOCOCaption(FlamingoFewShotMixin, BaseDataset):
    """Flamingo few shot COCO Caption dataset.

    Args:
        data_root (str): The root directory for ``data_prefix`` and
            ``ann_file``.
        ann_file (str): Annotation file path.
        data_prefix (dict): Prefix for data field. Defaults to
            ``dict(img_path='')``.
        num_shots (int): Number of shots to perform evaluation.
            Defaults to 0.
        num_support_examples (int): Number of support examples to get the
            few shots from. Defaults to 2048.
        num_query_examples (int): Number of query examples to perform the
            final evaluation. Defaults to 5000.
        **kwargs: Other keyword arguments in :class:`BaseDataset`.
    """

    def __init__(self,
                 data_root: str,
                 ann_file: str,
                 num_shots: int = 0,
                 num_support_examples: int = 2048,
                 num_query_examples: int = 5000,
                 **kwarg):
        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            num_shots=num_shots,
            num_support_examples=num_support_examples,
            num_query_examples=num_query_examples,
            **kwarg)

    def parse_basic_anno(self, ann: dict, coco: COCO) -> dict:
        """Parse basic annotation for support and query set.

        Args:
            anno (dict): Annotation for single example.
            coco (COCO): The coco dataset.

        Return:
            dict: Parsed annotation for single example.
        """
        img_prefix = self.data_prefix['img_path']
        img = coco.imgs[ann['image_id']]
        data_info = dict(
            img_path=mmengine.join_path(img_prefix, img['file_name']),
            gt_caption=ann['caption'],
            image_id=ann['image_id'],
        )
        return data_info

    def parse_fewshot_anno(self, query: dict, support_list: List) -> dict:
        """Parse fewshot related annotation for query set with support list.

        Args:
            query (dict): Annotation for single example.
            support_list (List): List of support subset to subsample few shots.
            coco (COCO): The coco dataset.

        Return:
            dict: Parsed annotation for single example.
        """
        # prepare n shots examples
        shots = random.sample(support_list, self.num_shots)

        # append image path for n shots
        img_path = [shot['img_path'] for shot in shots]
        img_path.append(query['img_path'])
        query['img_path'] = img_path

        query['shots'] = [dict(caption=item['gt_caption']) for item in shots]
        return query

    def load_data_list(self) -> List[dict]:
        """Load data list."""
        with mmengine.get_local_path(self.ann_file) as ann_file:
            coco = COCO(ann_file)

        num_data = len(coco.anns)
        support_idx, query_idx = self.get_subset_idx(num_data)
        ann_ids = list(coco.anns)

        # prepare support subset
        if self.num_shots > 0:
            support_list = []
            for idx in support_idx:
                support = self.parse_basic_anno(coco.anns[ann_ids[idx]], coco)
                support_list.append(support)

        # prepare query subset
        query_list = []
        for idx in query_idx:
            data_info = self.parse_basic_anno(coco.anns[ann_ids[idx]], coco)
            if self.num_shots > 0:
                data_info = self.parse_fewshot_anno(data_info, support_list)
            query_list.append(data_info)

        return query_list

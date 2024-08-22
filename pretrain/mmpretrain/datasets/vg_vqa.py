# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import List

from mmengine.fileio import load

from mmpretrain.registry import DATASETS
from .base_dataset import BaseDataset


@DATASETS.register_module()
class VGVQA(BaseDataset):
    """Visual Genome VQA dataset."""

    def load_data_list(self) -> List[dict]:
        """Load data list.

        Compare to BaseDataset, the only difference is that coco_vqa annotation
        file is already a list of data. There is no 'metainfo'.
        """

        raw_data_list = load(self.ann_file)
        if not isinstance(raw_data_list, list):
            raise TypeError(
                f'The VQA annotations loaded from annotation file '
                f'should be a dict, but got {type(raw_data_list)}!')

        # load and parse data_infos.
        data_list = []
        for raw_data_info in raw_data_list:
            # parse raw data information to target format
            data_info = self.parse_data_info(raw_data_info)
            if isinstance(data_info, dict):
                # For VQA tasks, each `data_info` looks like:
                # {
                #   "question_id": 986769,
                #   "question": "How many people are there?",
                #   "answer": "two",
                #   "image": "image/1.jpg",
                #   "dataset": "vg"
                # }

                # change 'image' key to 'img_path'
                # TODO: This process will be removed, after the annotation file
                # is preprocess.
                data_info['img_path'] = data_info['image']
                del data_info['image']

                if 'answer' in data_info:
                    # add answer_weight & answer_count, delete duplicate answer
                    if data_info['dataset'] == 'vqa':
                        answer_weight = {}
                        for answer in data_info['answer']:
                            if answer in answer_weight.keys():
                                answer_weight[answer] += 1 / len(
                                    data_info['answer'])
                            else:
                                answer_weight[answer] = 1 / len(
                                    data_info['answer'])

                        data_info['answer'] = list(answer_weight.keys())
                        data_info['answer_weight'] = list(
                            answer_weight.values())
                        data_info['answer_count'] = len(answer_weight)

                    elif data_info['dataset'] == 'vg':
                        data_info['answers'] = [data_info['answer']]
                        data_info['answer_weight'] = [0.2]
                        data_info['answer_count'] = 1

                data_list.append(data_info)

            else:
                raise TypeError(
                    f'Each VQA data element loaded from annotation file '
                    f'should be a dict, but got {type(data_info)}!')

        return data_list

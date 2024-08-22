# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import List

import mmengine
from mmengine.dataset import BaseDataset
from mmengine.fileio import list_dir_or_file
from mmengine.utils import check_file_exist

from mmpretrain.registry import DATASETS


@DATASETS.register_module()
class IconQA(BaseDataset):
    """IconQA: A benchmark for abstract diagram understanding
        and visual language reasoning.

    Args:
        data_root (str): The root directory for ``data_prefix``, ``ann_file``
            and ``question_file``.
        data_prefix (str): The directory of the specific task and split.
            eg. ``iconqa/val/choose_text/``.
        **kwargs: Other keyword arguments in :class:`BaseDataset`.
    """

    def __init__(self, data_root: str, data_prefix: str, **kwarg):
        super().__init__(
            data_root=data_root,
            data_prefix=dict(img_path=data_prefix),
            **kwarg,
        )

    def load_data_list(self) -> List[dict]:
        """Load data list."""
        sample_list = list(
            list_dir_or_file(self.data_prefix['img_path'], list_file=False))

        data_list = list()
        for sample_id in sample_list:
            # data json
            # {
            # "question": "How likely is it that you will pick a black one?",
            # "choices": [
            #     "certain",
            #     "unlikely",
            #     "impossible",
            #     "probable"
            # ],
            # "answer": 2,
            # "ques_type": "choose_txt",
            # "grade": "grade1",
            # "label": "S2"
            # }
            data_info = mmengine.load(
                mmengine.join_path(self.data_prefix['img_path'], sample_id,
                                   'data.json'))
            data_info['gt_answer'] = data_info['choices'][int(
                data_info['answer'])]
            data_info['img_path'] = mmengine.join_path(
                self.data_prefix['img_path'], sample_id, 'image.png')
            check_file_exist(data_info['img_path'])
            data_list.append(data_info)

        return data_list

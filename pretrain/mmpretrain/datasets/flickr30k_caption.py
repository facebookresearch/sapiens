# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import List

import mmengine
from mmengine.dataset import BaseDataset
from mmengine.fileio import get_file_backend

from mmpretrain.registry import DATASETS


@DATASETS.register_module()
class Flickr30kCaption(BaseDataset):
    """Flickr30k Caption dataset. To generate coco-style GT annotation for
    evaluation, please refer to
    tools/dataset_converters/convert_flickr30k_ann.py.

    Args:
        data_root (str): The root directory for ``data_prefix``, ``ann_file``
            and ``question_file``.
        data_prefix (str): The directory of images.
        ann_file (str): Annotation file path for training and validation.
        split (str): 'train', 'val' or 'test'.
        **kwargs: Other keyword arguments in :class:`BaseDataset`.
    """

    def __init__(self, data_root: str, data_prefix: str, ann_file: str,
                 split: str, **kwarg):

        assert split in ['train', 'val', 'test'], \
            '`split` must be train, val or test'
        self.split = split
        super().__init__(
            data_root=data_root,
            data_prefix=dict(img_path=data_prefix),
            ann_file=ann_file,
            **kwarg,
        )

    def load_data_list(self) -> List[dict]:
        """Load data list."""
        img_prefix = self.data_prefix['img_path']
        annotations = mmengine.load(self.ann_file)
        file_backend = get_file_backend(img_prefix)

        data_list = []

        for img in annotations['images']:

            # img_example={
            #     "sentids": [0, 1, 2],
            #     "imgid": 0,
            #     "sentences": [
            #         {"raw": "Two men in green shirts standing in a yard.",
            #          "imgid": 0, "sentid": 0},
            #         {"raw": "A man in a blue shirt standing in a garden.",
            #          "imgid": 0, "sentid": 1},
            #         {"raw": "Two friends enjoy time spent together.",
            #          "imgid": 0, "sentid": 2}
            #     ],
            #     "split": "train",
            #     "filename": "1000092795.jpg"
            # },

            if img['split'] != self.split:
                continue

            for sentence in img['sentences']:
                data_info = {
                    'image_id': img['imgid'],
                    'img_path': file_backend.join_path(img_prefix,
                                                       img['filename']),
                    'gt_caption': sentence['raw']
                }

                data_list.append(data_info)

        return data_list

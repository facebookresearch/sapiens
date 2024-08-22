# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class CIHPDataset(BaseSegDataset):
    """LIP dataset.

    The ``img_suffix`` is fixed to '.jpg' and ``seg_map_suffix`` is fixed to
    '.png'.
    """
    METAINFO = dict(
        classes=('Background', 'Hat', 'Hair', 'Glove', 'Sunglasses',
                 'UpperClothes', 'Dress', 'Coat', 'Socks', 'Pants',
                 'Jumpsuits', 'Scarf', 'Skirt', 'Face', 'Left-arm',
                 'Right-arm', 'Left-leg', 'Right-leg', 'Left-shoe',
                 'Right-shoe'),
        palette=(
            [0, 0, 0],
            [128, 0, 0],
            [255, 0, 0],
            [0, 85, 0],
            [170, 0, 51],
            [255, 85, 0],
            [0, 0, 85],
            [0, 119, 221],
            [85, 85, 0],
            [0, 85, 85],
            [85, 51, 0],
            [52, 86, 128],
            [0, 128, 0],
            [0, 0, 255],
            [51, 170, 221],
            [0, 255, 255],
            [85, 255, 170],
            [170, 255, 85],
            [255, 255, 0],
            [255, 170, 0],
        ))

    def __init__(self,
                 img_suffix='.jpg',
                 seg_map_suffix='.png',
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix, seg_map_suffix=seg_map_suffix, **kwargs)

        return

    def load_data_list(self):
        ## call parent load_data_list
        data_list = super().load_data_list()

        ## for debug
        # if self.test_mode == False:
        #     data_list = data_list[:16]

        print('\033[92mDone! LIP. Loaded total samples: {}\033[0m'.format(len(data_list))) ## 98424 images train

        return data_list

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import List

from mmpretrain.registry import DATASETS
from .base_dataset import BaseDataset


@DATASETS.register_module()
class MultiLabelDataset(BaseDataset):
    """Multi-label Dataset.

    This dataset support annotation file in `OpenMMLab 2.0 style annotation
    format`.

    The annotation format is shown as follows.

    .. code-block:: none

        {
            "metainfo":
            {
              "classes":['A', 'B', 'C'....]
            },
            "data_list":
            [
              {
                "img_path": "test_img1.jpg",
                'gt_label': [0, 1],
              },
              {
                "img_path": "test_img2.jpg",
                'gt_label': [2],
              },
            ]
            ....
        }


    Args:
        ann_file (str): Annotation file path.
        metainfo (dict, optional): Meta information for dataset, such as class
            information. Defaults to None.
        data_root (str): The root directory for ``data_prefix`` and
            ``ann_file``. Defaults to ''.
        data_prefix (str | dict): Prefix for training data. Defaults to ''.
        filter_cfg (dict, optional): Config for filter data. Defaults to None.
        indices (int or Sequence[int], optional): Support using first few
            data in annotation file to facilitate training/testing on a smaller
            dataset. Defaults to None which means using all ``data_infos``.
        serialize_data (bool, optional): Whether to hold memory using
            serialized objects, when enabled, data loader workers can use
            shared RAM from master process instead of making a copy. Defaults
            to True.
        pipeline (list, optional): Processing pipeline. Defaults to [].
        test_mode (bool, optional): ``test_mode=True`` means in test phase.
            Defaults to False.
        lazy_init (bool, optional): Whether to load annotation during
            instantiation. In some cases, such as visualization, only the meta
            information of the dataset is needed, which is not necessary to
            load annotation file. ``Basedataset`` can skip load annotations to
            save time by set ``lazy_init=False``. Defaults to False.
        max_refetch (int, optional): If ``Basedataset.prepare_data`` get a
            None img. The maximum extra number of cycles to get a valid
            image. Defaults to 1000.
        classes (str | Sequence[str], optional): Specify names of classes.

            - If is string, it should be a file path, and the every line of
              the file is a name of a class.
            - If is a sequence of string, every item is a name of class.
            - If is None, use categories information in ``metainfo`` argument,
              annotation file or the class attribute ``METAINFO``.

            Defaults to None.
    """

    def get_cat_ids(self, idx: int) -> List[int]:
        """Get category ids by index.

        Args:
            idx (int): Index of data.

        Returns:
            cat_ids (List[int]): Image categories of specified index.
        """
        return self.get_data_info(idx)['gt_label']

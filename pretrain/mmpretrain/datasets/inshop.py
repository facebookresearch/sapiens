# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from mmengine import get_file_backend, list_from_file

from mmpretrain.registry import DATASETS
from .base_dataset import BaseDataset


@DATASETS.register_module()
class InShop(BaseDataset):
    """InShop Dataset for Image Retrieval.

    Please download the images from the homepage
    'https://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/InShopRetrieval.html'
    (In-shop Clothes Retrieval Benchmark -> Img -> img.zip,
    Eval/list_eval_partition.txt), and organize them as follows way: ::

        In-shop Clothes Retrieval Benchmark (data_root)/
           ├── Eval /
           │    └── list_eval_partition.txt (ann_file)
           ├── Img (img_prefix)
           │    └── img/
           ├── README.txt
           └── .....

    Args:
        data_root (str): The root directory for dataset.
        split (str): Choose from 'train', 'query' and 'gallery'.
            Defaults to 'train'.
        data_prefix (str | dict): Prefix for training data.
            Defaults to 'Img'.
        ann_file (str): Annotation file path, path relative to
            ``data_root``. Defaults to 'Eval/list_eval_partition.txt'.
        **kwargs: Other keyword arguments in :class:`BaseDataset`.

    Examples:
        >>> from mmpretrain.datasets import InShop
        >>>
        >>> # build train InShop dataset
        >>> inshop_train_cfg = dict(data_root='data/inshop', split='train')
        >>> inshop_train = InShop(**inshop_train_cfg)
        >>> inshop_train
        Dataset InShop
            Number of samples:  25882
            The `CLASSES` meta info is not set.
            Root of dataset:    data/inshop
        >>>
        >>> # build query InShop dataset
        >>> inshop_query_cfg =  dict(data_root='data/inshop', split='query')
        >>> inshop_query = InShop(**inshop_query_cfg)
        >>> inshop_query
        Dataset InShop
            Number of samples:  14218
            The `CLASSES` meta info is not set.
            Root of dataset:    data/inshop
        >>>
        >>> # build gallery InShop dataset
        >>> inshop_gallery_cfg = dict(data_root='data/inshop', split='gallery')
        >>> inshop_gallery = InShop(**inshop_gallery_cfg)
        >>> inshop_gallery
        Dataset InShop
            Number of samples:  12612
            The `CLASSES` meta info is not set.
            Root of dataset:    data/inshop
    """

    def __init__(self,
                 data_root: str,
                 split: str = 'train',
                 data_prefix: str = 'Img',
                 ann_file: str = 'Eval/list_eval_partition.txt',
                 **kwargs):

        assert split in ('train', 'query', 'gallery'), "'split' of `InShop`" \
            f" must be one of ['train', 'query', 'gallery'], bu get '{split}'"
        self.backend = get_file_backend(data_root, enable_singleton=True)
        self.split = split
        super().__init__(
            data_root=data_root,
            data_prefix=data_prefix,
            ann_file=ann_file,
            **kwargs)

    def _process_annotations(self):
        lines = list_from_file(self.ann_file)

        anno_train = dict(metainfo=dict(), data_list=list())
        anno_gallery = dict(metainfo=dict(), data_list=list())

        # item_id to label, each item corresponds to one class label
        class_num = 0
        gt_label_train = {}

        # item_id to label, each label corresponds to several items
        gallery_num = 0
        gt_label_gallery = {}

        # (lines[0], lines[1]) is the image number and the field name;
        # Each line format as 'image_name, item_id, evaluation_status'
        for line in lines[2:]:
            img_name, item_id, status = line.split()
            img_path = self.backend.join_path(self.img_prefix, img_name)
            if status == 'train':
                if item_id not in gt_label_train:
                    gt_label_train[item_id] = class_num
                    class_num += 1
                # item_id to class_id (for the training set)
                anno_train['data_list'].append(
                    dict(img_path=img_path, gt_label=gt_label_train[item_id]))
            elif status == 'gallery':
                if item_id not in gt_label_gallery:
                    gt_label_gallery[item_id] = []
                # Since there are multiple images for each item,
                # record the corresponding item for each image.
                gt_label_gallery[item_id].append(gallery_num)
                anno_gallery['data_list'].append(
                    dict(img_path=img_path, sample_idx=gallery_num))
                gallery_num += 1

        if self.split == 'train':
            anno_train['metainfo']['class_number'] = class_num
            anno_train['metainfo']['sample_number'] = \
                len(anno_train['data_list'])
            return anno_train
        elif self.split == 'gallery':
            anno_gallery['metainfo']['sample_number'] = gallery_num
            return anno_gallery

        # Generate the label for the query(val) set
        anno_query = dict(metainfo=dict(), data_list=list())
        query_num = 0
        for line in lines[2:]:
            img_name, item_id, status = line.split()
            img_path = self.backend.join_path(self.img_prefix, img_name)
            if status == 'query':
                anno_query['data_list'].append(
                    dict(
                        img_path=img_path, gt_label=gt_label_gallery[item_id]))
                query_num += 1

        anno_query['metainfo']['sample_number'] = query_num
        return anno_query

    def load_data_list(self):
        """load data list.

        For the train set, return image and ground truth label. For the query
        set, return image and ids of images in gallery. For the gallery set,
        return image and its id.
        """
        data_info = self._process_annotations()
        data_list = data_info['data_list']
        return data_list

    def extra_repr(self):
        """The extra repr information of the dataset."""
        body = [f'Root of dataset: \t{self.data_root}']
        return body

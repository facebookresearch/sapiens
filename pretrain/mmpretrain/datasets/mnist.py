# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import codecs
from typing import List, Optional
from urllib.parse import urljoin

import mmengine.dist as dist
import numpy as np
import torch
from mmengine.fileio import LocalBackend, exists, get_file_backend, join_path
from mmengine.logging import MMLogger

from mmpretrain.registry import DATASETS
from .base_dataset import BaseDataset
from .categories import FASHIONMNIST_CATEGORITES, MNIST_CATEGORITES
from .utils import (download_and_extract_archive, open_maybe_compressed_file,
                    rm_suffix)


@DATASETS.register_module()
class MNIST(BaseDataset):
    """`MNIST <http://yann.lecun.com/exdb/mnist/>`_ Dataset.

    This implementation is modified from
    https://github.com/pytorch/vision/blob/master/torchvision/datasets/mnist.py

    Args:
        data_root (str): The root directory of the MNIST Dataset.
        split (str, optional): The dataset split, supports "train" and "test".
            Default to "train".
        metainfo (dict, optional): Meta information for dataset, such as
            categories information. Defaults to None.
        download (bool): Whether to download the dataset if not exists.
            Defaults to True.
        **kwargs: Other keyword arguments in :class:`BaseDataset`.
    """  # noqa: E501

    url_prefix = 'http://yann.lecun.com/exdb/mnist/'
    # train images and labels
    train_list = [
        ['train-images-idx3-ubyte.gz', 'f68b3c2dcbeaaa9fbdd348bbdeb94873'],
        ['train-labels-idx1-ubyte.gz', 'd53e105ee54ea40749a09fcbcd1e9432'],
    ]
    # test images and labels
    test_list = [
        ['t10k-images-idx3-ubyte.gz', '9fb629c4189551a2d022fa330f9573f3'],
        ['t10k-labels-idx1-ubyte.gz', 'ec29112dd5afa0611ce80d1b7f02629c'],
    ]
    METAINFO = {'classes': MNIST_CATEGORITES}

    def __init__(self,
                 data_root: str = '',
                 split: str = 'train',
                 metainfo: Optional[dict] = None,
                 download: bool = True,
                 data_prefix: str = '',
                 test_mode: bool = False,
                 **kwargs):

        splits = ['train', 'test']
        assert split in splits, \
            f"The split must be one of {splits}, but get '{split}'"
        self.split = split

        # To handle the BC-breaking
        if split == 'train' and test_mode:
            logger = MMLogger.get_current_instance()
            logger.warning('split="train" but test_mode=True. '
                           'The training set will be used.')

        if not data_root and not data_prefix:
            raise RuntimeError('Please set ``data_root`` to'
                               'specify the dataset path')

        self.download = download
        super().__init__(
            # The MNIST dataset doesn't need specify annotation file
            ann_file='',
            metainfo=metainfo,
            data_root=data_root,
            data_prefix=dict(root=data_prefix),
            test_mode=test_mode,
            **kwargs)

    def load_data_list(self):
        """Load images and ground truth labels."""
        root = self.data_prefix['root']
        backend = get_file_backend(root, enable_singleton=True)

        if dist.is_main_process() and not self._check_exists():
            if not isinstance(backend, LocalBackend):
                raise RuntimeError(f'The dataset on {root} is not integrated, '
                                   f'please manually handle it.')

            if self.download:
                self._download()
            else:
                raise RuntimeError(
                    f'Cannot find {self.__class__.__name__} dataset in '
                    f"{self.data_prefix['root']}, you can specify "
                    '`download=True` to download automatically.')

        dist.barrier()
        assert self._check_exists(), \
            'Download failed or shared storage is unavailable. Please ' \
            f'download the dataset manually through {self.url_prefix}.'

        if not self.test_mode:
            file_list = self.train_list
        else:
            file_list = self.test_list

        # load data from SN3 files
        imgs = read_image_file(join_path(root, rm_suffix(file_list[0][0])))
        gt_labels = read_label_file(
            join_path(root, rm_suffix(file_list[1][0])))

        data_infos = []
        for img, gt_label in zip(imgs, gt_labels):
            gt_label = np.array(gt_label, dtype=np.int64)
            info = {'img': img.numpy(), 'gt_label': gt_label}
            data_infos.append(info)
        return data_infos

    def _check_exists(self):
        """Check the exists of data files."""
        root = self.data_prefix['root']

        for filename, _ in (self.train_list + self.test_list):
            # get extracted filename of data
            extract_filename = rm_suffix(filename)
            fpath = join_path(root, extract_filename)
            if not exists(fpath):
                return False
        return True

    def _download(self):
        """Download and extract data files."""
        root = self.data_prefix['root']

        for filename, md5 in (self.train_list + self.test_list):
            url = urljoin(self.url_prefix, filename)
            download_and_extract_archive(
                url, download_root=root, filename=filename, md5=md5)

    def extra_repr(self) -> List[str]:
        """The extra repr information of the dataset."""
        body = [f"Prefix of data: \t{self.data_prefix['root']}"]
        return body


@DATASETS.register_module()
class FashionMNIST(MNIST):
    """`Fashion-MNIST <https://github.com/zalandoresearch/fashion-mnist>`_
    Dataset.

    Args:
        data_root (str): The root directory of the MNIST Dataset.
        split (str, optional): The dataset split, supports "train" and "test".
            Default to "train".
        metainfo (dict, optional): Meta information for dataset, such as
            categories information. Defaults to None.
        download (bool): Whether to download the dataset if not exists.
            Defaults to True.
        **kwargs: Other keyword arguments in :class:`BaseDataset`.
    """

    url_prefix = 'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/'
    # train images and labels
    train_list = [
        ['train-images-idx3-ubyte.gz', '8d4fb7e6c68d591d4c3dfef9ec88bf0d'],
        ['train-labels-idx1-ubyte.gz', '25c81989df183df01b3e8a0aad5dffbe'],
    ]
    # test images and labels
    test_list = [
        ['t10k-images-idx3-ubyte.gz', 'bef4ecab320f06d8554ea6380940ec79'],
        ['t10k-labels-idx1-ubyte.gz', 'bb300cfdad3c16e7a12a480ee83cd310'],
    ]
    METAINFO = {'classes': FASHIONMNIST_CATEGORITES}


def get_int(b: bytes) -> int:
    """Convert bytes to int."""
    return int(codecs.encode(b, 'hex'), 16)


def read_sn3_pascalvincent_tensor(path: str,
                                  strict: bool = True) -> torch.Tensor:
    """Read a SN3 file in "Pascal Vincent" format (Lush file 'libidx/idx-
    io.lsh').

    Argument may be a filename, compressed filename, or file object.
    """
    # typemap
    if not hasattr(read_sn3_pascalvincent_tensor, 'typemap'):
        read_sn3_pascalvincent_tensor.typemap = {
            8: (torch.uint8, np.uint8, np.uint8),
            9: (torch.int8, np.int8, np.int8),
            11: (torch.int16, np.dtype('>i2'), 'i2'),
            12: (torch.int32, np.dtype('>i4'), 'i4'),
            13: (torch.float32, np.dtype('>f4'), 'f4'),
            14: (torch.float64, np.dtype('>f8'), 'f8')
        }
    # read
    with open_maybe_compressed_file(path) as f:
        data = f.read()
    # parse
    magic = get_int(data[0:4])
    nd = magic % 256
    ty = magic // 256
    assert nd >= 1 and nd <= 3
    assert ty >= 8 and ty <= 14
    m = read_sn3_pascalvincent_tensor.typemap[ty]
    s = [get_int(data[4 * (i + 1):4 * (i + 2)]) for i in range(nd)]
    parsed = np.frombuffer(data, dtype=m[1], offset=(4 * (nd + 1)))
    assert parsed.shape[0] == np.prod(s) or not strict
    return torch.from_numpy(parsed.astype(m[2], copy=False)).view(*s)


def read_label_file(path: str) -> torch.Tensor:
    """Read labels from SN3 label file."""
    with open(path, 'rb') as f:
        x = read_sn3_pascalvincent_tensor(f, strict=False)
    assert (x.dtype == torch.uint8)
    assert (x.ndimension() == 1)
    return x.long()


def read_image_file(path: str) -> torch.Tensor:
    """Read images from SN3 image file."""
    with open(path, 'rb') as f:
        x = read_sn3_pascalvincent_tensor(f, strict=False)
    assert (x.dtype == torch.uint8)
    assert (x.ndimension() == 3)
    return x

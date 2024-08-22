# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from mmpretrain.registry import DATASETS


def build_dataset(cfg):
    """Build dataset.

    Examples:
        >>> from mmpretrain.datasets import build_dataset
        >>> mnist_train = build_dataset(
        ...     dict(type='MNIST', data_prefix='data/mnist/', test_mode=False))
        >>> print(mnist_train)
        Dataset MNIST
            Number of samples:  60000
            Number of categories:       10
            Prefix of data:     data/mnist/
        >>> mnist_test = build_dataset(
        ...     dict(type='MNIST', data_prefix='data/mnist/', test_mode=True))
        >>> print(mnist_test)
        Dataset MNIST
            Number of samples:  10000
            Number of categories:       10
            Prefix of data:     data/mnist/
    """
    return DATASETS.build(cfg)

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from .base import BaseStorageBackend
from .http_backend import HTTPBackend
from .lmdb_backend import LmdbBackend
from .local_backend import LocalBackend
from .memcached_backend import MemcachedBackend
from .petrel_backend import PetrelBackend
from .registry_utils import backends, prefix_to_backends, register_backend

__all__ = [
    'BaseStorageBackend', 'LocalBackend', 'HTTPBackend', 'LmdbBackend',
    'MemcachedBackend', 'PetrelBackend', 'register_backend', 'backends',
    'prefix_to_backends'
]

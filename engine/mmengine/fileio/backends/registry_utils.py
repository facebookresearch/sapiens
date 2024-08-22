# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import inspect
from typing import Optional, Type, Union

from .base import BaseStorageBackend
from .http_backend import HTTPBackend
from .lmdb_backend import LmdbBackend
from .local_backend import LocalBackend
from .memcached_backend import MemcachedBackend
from .petrel_backend import PetrelBackend

backends: dict = {}
prefix_to_backends: dict = {}


def _register_backend(name: str,
                      backend: Type[BaseStorageBackend],
                      force: bool = False,
                      prefixes: Union[str, list, tuple, None] = None):
    """Register a backend.

    Args:
        name (str): The name of the registered backend.
        backend (BaseStorageBackend): The backend class to be registered,
            which must be a subclass of :class:`BaseStorageBackend`.
        force (bool): Whether to override the backend if the name has already
            been registered. Defaults to False.
        prefixes (str or list[str] or tuple[str], optional): The prefix
            of the registered storage backend. Defaults to None.
    """
    global backends, prefix_to_backends

    if not isinstance(name, str):
        raise TypeError('the backend name should be a string, '
                        f'but got {type(name)}')

    if not inspect.isclass(backend):
        raise TypeError(f'backend should be a class, but got {type(backend)}')
    if not issubclass(backend, BaseStorageBackend):
        raise TypeError(
            f'backend {backend} is not a subclass of BaseStorageBackend')

    if name in backends and not force:
        raise ValueError(f'{name} is already registered as a storage backend, '
                         'add "force=True" if you want to override it')
    backends[name] = backend

    if prefixes is not None:
        if isinstance(prefixes, str):
            prefixes = [prefixes]
        else:
            assert isinstance(prefixes, (list, tuple))

        for prefix in prefixes:
            if prefix in prefix_to_backends and not force:
                raise ValueError(
                    f'{prefix} is already registered as a storage backend,'
                    ' add "force=True" if you want to override it')

            prefix_to_backends[prefix] = backend


def register_backend(name: str,
                     backend: Optional[Type[BaseStorageBackend]] = None,
                     force: bool = False,
                     prefixes: Union[str, list, tuple, None] = None):
    """Register a backend.

    Args:
        name (str): The name of the registered backend.
        backend (class, optional): The backend class to be registered,
            which must be a subclass of :class:`BaseStorageBackend`.
            When this method is used as a decorator, backend is None.
            Defaults to None.
        force (bool): Whether to override the backend if the name has already
            been registered. Defaults to False.
        prefixes (str or list[str] or tuple[str], optional): The prefix
            of the registered storage backend. Defaults to None.

    This method can be used as a normal method or a decorator.

    Examples:

        >>> class NewBackend(BaseStorageBackend):
        ...     def get(self, filepath):
        ...         return filepath
        ...
        ...     def get_text(self, filepath):
        ...         return filepath
        >>> register_backend('new', NewBackend)

        >>> @register_backend('new')
        ... class NewBackend(BaseStorageBackend):
        ...     def get(self, filepath):
        ...         return filepath
        ...
        ...     def get_text(self, filepath):
        ...         return filepath
    """
    if backend is not None:
        _register_backend(name, backend, force=force, prefixes=prefixes)
        return

    def _register(backend_cls):
        _register_backend(name, backend_cls, force=force, prefixes=prefixes)
        return backend_cls

    return _register


register_backend('local', LocalBackend, prefixes='')
register_backend('memcached', MemcachedBackend)
register_backend('lmdb', LmdbBackend)
# To avoid breaking backward Compatibility, 's3' is also used as a
# prefix for PetrelBackend
register_backend('petrel', PetrelBackend, prefixes=['petrel', 's3'])
register_backend('http', HTTPBackend, prefixes=['http', 'https'])

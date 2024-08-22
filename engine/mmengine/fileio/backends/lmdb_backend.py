# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path
from typing import Union

from .base import BaseStorageBackend


class LmdbBackend(BaseStorageBackend):
    """Lmdb storage backend.

    Args:
        db_path (str): Lmdb database path.
        readonly (bool): Lmdb environment parameter. If True, disallow any
            write operations. Defaults to True.
        lock (bool): Lmdb environment parameter. If False, when concurrent
            access occurs, do not lock the database. Defaults to False.
        readahead (bool): Lmdb environment parameter. If False, disable the OS
            filesystem readahead mechanism, which may improve random read
            performance when a database is larger than RAM. Defaults to False.
        **kwargs: Keyword arguments passed to `lmdb.open`.

    Attributes:
        db_path (str): Lmdb database path.
    """

    def __init__(self,
                 db_path,
                 readonly=True,
                 lock=False,
                 readahead=False,
                 **kwargs):
        try:
            import lmdb  # noqa: F401
        except ImportError:
            raise ImportError(
                'Please run "pip install lmdb" to enable LmdbBackend.')

        self.db_path = str(db_path)
        self.readonly = readonly
        self.lock = lock
        self.readahead = readahead
        self.kwargs = kwargs
        self._client = None

    def get(self, filepath: Union[str, Path]) -> bytes:
        """Get values according to the filepath.

        Args:
            filepath (str or Path): Here, filepath is the lmdb key.

        Returns:
            bytes: Expected bytes object.

        Examples:
            >>> backend = LmdbBackend('path/to/lmdb')
            >>> backend.get('key')
            b'hello world'
        """
        if self._client is None:
            self._client = self._get_client()

        filepath = str(filepath)
        with self._client.begin(write=False) as txn:
            value_buf = txn.get(filepath.encode('ascii'))
        return value_buf

    def get_text(self, filepath, encoding=None):
        raise NotImplementedError

    def _get_client(self):
        import lmdb

        return lmdb.open(
            self.db_path,
            readonly=self.readonly,
            lock=self.lock,
            readahead=self.readahead,
            **self.kwargs)

    def __del__(self):
        if self._client is not None:
            self._client.close()

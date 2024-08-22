# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, Union
from urllib.request import urlopen

from .base import BaseStorageBackend


class HTTPBackend(BaseStorageBackend):
    """HTTP and HTTPS storage bachend."""

    def get(self, filepath: str) -> bytes:
        """Read bytes from a given ``filepath``.

        Args:
            filepath (str): Path to read data.

        Returns:
            bytes: Expected bytes object.

        Examples:
            >>> backend = HTTPBackend()
            >>> backend.get('http://path/of/file')
            b'hello world'
        """
        return urlopen(filepath).read()

    def get_text(self, filepath, encoding='utf-8') -> str:
        """Read text from a given ``filepath``.

        Args:
            filepath (str): Path to read data.
            encoding (str): The encoding format used to open the ``filepath``.
                Defaults to 'utf-8'.

        Returns:
            str: Expected text reading from ``filepath``.

        Examples:
            >>> backend = HTTPBackend()
            >>> backend.get_text('http://path/of/file')
            'hello world'
        """
        return urlopen(filepath).read().decode(encoding)

    @contextmanager
    def get_local_path(
            self, filepath: str) -> Generator[Union[str, Path], None, None]:
        """Download a file from ``filepath`` to a local temporary directory,
        and return the temporary path.

        ``get_local_path`` is decorated by :meth:`contxtlib.contextmanager`. It
        can be called with ``with`` statement, and when exists from the
        ``with`` statement, the temporary path will be released.

        Args:
            filepath (str): Download a file from ``filepath``.

        Yields:
            Iterable[str]: Only yield one temporary path.

        Examples:
            >>> backend = HTTPBackend()
            >>> # After existing from the ``with`` clause,
            >>> # the path will be removed
            >>> with backend.get_local_path('http://path/of/file') as path:
            ...     # do something here
        """
        try:
            f = tempfile.NamedTemporaryFile(delete=False)
            f.write(self.get(filepath))
            f.close()
            yield f.name
        finally:
            os.remove(f.name)

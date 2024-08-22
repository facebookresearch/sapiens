# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""This module provides unified file I/O related functions, which support
operating I/O with different file backends based on the specified filepath or
backend_args.

MMEngine currently supports five file backends:

- LocalBackend
- PetrelBackend
- HTTPBackend
- LmdbBackend
- MemcacheBackend

Note that this module provide a union of all of the above file backends so
NotImplementedError will be raised if the interface in the file backend is not
implemented.

There are two ways to call a method of a file backend:

- Initialize a file backend with ``get_file_backend`` and call its methods.
- Directory call unified I/O functions, which will call ``get_file_backend``
  first and then call the corresponding backend method.

Examples:
    >>> # Initialize a file backend and call its methods
    >>> import mmengine.fileio as fileio
    >>> backend = fileio.get_file_backend(backend_args={'backend': 'petrel'})
    >>> backend.get('s3://path/of/your/file')

    >>> # Directory call unified I/O functions
    >>> fileio.get('s3://path/of/your/file')
"""
import json
import warnings
from contextlib import contextmanager
from io import BytesIO, StringIO
from pathlib import Path
from typing import Generator, Iterator, Optional, Tuple, Union

from mmengine.utils import is_filepath, is_str
from .backends import backends, prefix_to_backends
from .file_client import FileClient
# file_handlers and register_handler had been moved to
# mmengine/fileio/handlers/registry_utis. Import them
# in this file to keep backward compatibility.
from .handlers import file_handlers, register_handler  # noqa: F401

backend_instances: dict = {}


def _parse_uri_prefix(uri: Union[str, Path]) -> str:
    """Parse the prefix of uri.

    Args:
        uri (str or Path): Uri to be parsed that contains the file prefix.

    Examples:
        >>> _parse_uri_prefix('/home/path/of/your/file')
        ''
        >>> _parse_uri_prefix('s3://path/of/your/file')
        's3'
        >>> _parse_uri_prefix('clusterName:s3://path/of/your/file')
        's3'

    Returns:
        str: Return the prefix of uri if the uri contains '://'. Otherwise,
        return ''.
    """
    assert is_filepath(uri)
    uri = str(uri)
    # if uri does not contains '://', the uri will be handled by
    # LocalBackend by default
    if '://' not in uri:
        return ''
    else:
        prefix, _ = uri.split('://')
        # In the case of PetrelBackend, the prefix may contain the cluster
        # name like clusterName:s3://path/of/your/file
        if ':' in prefix:
            _, prefix = prefix.split(':')
        return prefix


def _get_file_backend(prefix: str, backend_args: dict):
    """Return a file backend based on the prefix or backend_args.

    Args:
        prefix (str): Prefix of uri.
        backend_args (dict): Arguments to instantiate the corresponding
            backend.
    """
    # backend name has a higher priority
    if 'backend' in backend_args:
        # backend_args should not be modified
        backend_args_bak = backend_args.copy()
        backend_name = backend_args_bak.pop('backend')
        backend = backends[backend_name](**backend_args_bak)
    else:
        backend = prefix_to_backends[prefix](**backend_args)
    return backend


def get_file_backend(
    uri: Union[str, Path, None] = None,
    *,
    backend_args: Optional[dict] = None,
    enable_singleton: bool = False,
):
    """Return a file backend based on the prefix of uri or backend_args.

    Args:
        uri (str or Path): Uri to be parsed that contains the file prefix.
        backend_args (dict, optional): Arguments to instantiate the
            corresponding backend. Defaults to None.
        enable_singleton (bool): Whether to enable the singleton pattern.
            If it is True, the backend created will be reused if the
            signature is same with the previous one. Defaults to False.

    Returns:
        BaseStorageBackend: Instantiated Backend object.

    Examples:
        >>> # get file backend based on the prefix of uri
        >>> uri = 's3://path/of/your/file'
        >>> backend = get_file_backend(uri)
        >>> # get file backend based on the backend_args
        >>> backend = get_file_backend(backend_args={'backend': 'petrel'})
        >>> # backend name has a higher priority if 'backend' in backend_args
        >>> backend = get_file_backend(uri, backend_args={'backend': 'petrel'})
    """
    global backend_instances

    if backend_args is None:
        backend_args = {}

    if uri is None and 'backend' not in backend_args:
        raise ValueError(
            'uri should not be None when "backend" does not exist in '
            'backend_args')

    if uri is not None:
        prefix = _parse_uri_prefix(uri)
    else:
        prefix = ''

    if enable_singleton:
        # TODO: whether to pass sort_key to json.dumps
        unique_key = f'{prefix}:{json.dumps(backend_args)}'
        if unique_key in backend_instances:
            return backend_instances[unique_key]

        backend = _get_file_backend(prefix, backend_args)
        backend_instances[unique_key] = backend
        return backend
    else:
        backend = _get_file_backend(prefix, backend_args)
        return backend


def get(
    filepath: Union[str, Path],
    backend_args: Optional[dict] = None,
) -> bytes:
    """Read bytes from a given ``filepath`` with 'rb' mode.

    Args:
        filepath (str or Path): Path to read data.
        backend_args (dict, optional): Arguments to instantiate the
            corresponding backend. Defaults to None.

    Returns:
        bytes: Expected bytes object.

    Examples:
        >>> filepath = '/path/of/file'
        >>> get(filepath)
        b'hello world'
    """
    backend = get_file_backend(
        filepath, backend_args=backend_args, enable_singleton=True)
    return backend.get(filepath)


def get_text(
    filepath: Union[str, Path],
    encoding='utf-8',
    backend_args: Optional[dict] = None,
) -> str:
    """Read text from a given ``filepath`` with 'r' mode.

    Args:
        filepath (str or Path): Path to read data.
        encoding (str): The encoding format used to open the ``filepath``.
            Defaults to 'utf-8'.
        backend_args (dict, optional): Arguments to instantiate the
            corresponding backend. Defaults to None.

    Returns:
        str: Expected text reading from ``filepath``.

    Examples:
        >>> filepath = '/path/of/file'
        >>> get_text(filepath)
        'hello world'
    """
    backend = get_file_backend(
        filepath, backend_args=backend_args, enable_singleton=True)
    return backend.get_text(filepath, encoding)


def put(
    obj: bytes,
    filepath: Union[str, Path],
    backend_args: Optional[dict] = None,
) -> None:
    """Write bytes to a given ``filepath`` with 'wb' mode.

    Note:
        ``put`` should create a directory if the directory of
        ``filepath`` does not exist.

    Args:
        obj (bytes): Data to be written.
        filepath (str or Path): Path to write data.
        backend_args (dict, optional): Arguments to instantiate the
            corresponding backend. Defaults to None.

    Examples:
        >>> filepath = '/path/of/file'
        >>> put(b'hello world', filepath)
    """
    backend = get_file_backend(
        filepath, backend_args=backend_args, enable_singleton=True)
    backend.put(obj, filepath)


def put_text(
    obj: str,
    filepath: Union[str, Path],
    backend_args: Optional[dict] = None,
) -> None:
    """Write text to a given ``filepath`` with 'w' mode.

    Note:
        ``put_text`` should create a directory if the directory of
        ``filepath`` does not exist.

    Args:
        obj (str): Data to be written.
        filepath (str or Path): Path to write data.
        encoding (str, optional): The encoding format used to open the
            ``filepath``. Defaults to 'utf-8'.
        backend_args (dict, optional): Arguments to instantiate the
            corresponding backend. Defaults to None.

    Examples:
        >>> filepath = '/path/of/file'
        >>> put_text('hello world', filepath)
    """
    backend = get_file_backend(
        filepath, backend_args=backend_args, enable_singleton=True)
    backend.put_text(obj, filepath)


def exists(
    filepath: Union[str, Path],
    backend_args: Optional[dict] = None,
) -> bool:
    """Check whether a file path exists.

    Args:
        filepath (str or Path): Path to be checked whether exists.
        backend_args (dict, optional): Arguments to instantiate the
            corresponding backend. Defaults to None.

    Returns:
        bool: Return ``True`` if ``filepath`` exists, ``False`` otherwise.

    Examples:
        >>> filepath = '/path/of/file'
        >>> exists(filepath)
        True
    """
    backend = get_file_backend(
        filepath, backend_args=backend_args, enable_singleton=True)
    return backend.exists(filepath)


def isdir(
    filepath: Union[str, Path],
    backend_args: Optional[dict] = None,
) -> bool:
    """Check whether a file path is a directory.

    Args:
        filepath (str or Path): Path to be checked whether it is a
            directory.
        backend_args (dict, optional): Arguments to instantiate the
            corresponding backend. Defaults to None.

    Returns:
        bool: Return ``True`` if ``filepath`` points to a directory,
        ``False`` otherwise.

    Examples:
        >>> filepath = '/path/of/dir'
        >>> isdir(filepath)
        True
    """
    backend = get_file_backend(
        filepath, backend_args=backend_args, enable_singleton=True)
    return backend.isdir(filepath)


def isfile(
    filepath: Union[str, Path],
    backend_args: Optional[dict] = None,
) -> bool:
    """Check whether a file path is a file.

    Args:
        filepath (str or Path): Path to be checked whether it is a file.
        backend_args (dict, optional): Arguments to instantiate the
            corresponding backend. Defaults to None.

    Returns:
        bool: Return ``True`` if ``filepath`` points to a file, ``False``
        otherwise.

    Examples:
        >>> filepath = '/path/of/file'
        >>> isfile(filepath)
        True
    """
    backend = get_file_backend(
        filepath, backend_args=backend_args, enable_singleton=True)
    return backend.isfile(filepath)


def join_path(
    filepath: Union[str, Path],
    *filepaths: Union[str, Path],
    backend_args: Optional[dict] = None,
) -> Union[str, Path]:
    r"""Concatenate all file paths.

    Join one or more filepath components intelligently. The return value
    is the concatenation of filepath and any members of \*filepaths.

    Args:
        filepath (str or Path): Path to be concatenated.
        *filepaths (str or Path): Other paths to be concatenated.
        backend_args (dict, optional): Arguments to instantiate the
            corresponding backend. Defaults to None.

    Returns:
        str: The result of concatenation.

    Examples:
        >>> filepath1 = '/path/of/dir1'
        >>> filepath2 = 'dir2'
        >>> filepath3 = 'path/of/file'
        >>> join_path(filepath1, filepath2, filepath3)
        '/path/of/dir/dir2/path/of/file'
    """
    backend = get_file_backend(
        filepath, backend_args=backend_args, enable_singleton=True)
    return backend.join_path(filepath, *filepaths)


@contextmanager
def get_local_path(
    filepath: Union[str, Path],
    backend_args: Optional[dict] = None,
) -> Generator[Union[str, Path], None, None]:
    """Download data from ``filepath`` and write the data to local path.

    ``get_local_path`` is decorated by :meth:`contxtlib.contextmanager`. It
    can be called with ``with`` statement, and when exists from the
    ``with`` statement, the temporary path will be released.

    Note:
        If the ``filepath`` is a local path, just return itself and it will
        not be released (removed).

    Args:
        filepath (str or Path): Path to be read data.
        backend_args (dict, optional): Arguments to instantiate the
            corresponding backend. Defaults to None.

    Yields:
        Iterable[str]: Only yield one path.

    Examples:
        >>> with get_local_path('s3://bucket/abc.jpg') as path:
        ...     # do something here
    """
    backend = get_file_backend(
        filepath, backend_args=backend_args, enable_singleton=True)
    with backend.get_local_path(str(filepath)) as local_path:
        yield local_path


def copyfile(
    src: Union[str, Path],
    dst: Union[str, Path],
    backend_args: Optional[dict] = None,
) -> Union[str, Path]:
    """Copy a file src to dst and return the destination file.

    src and dst should have the same prefix. If dst specifies a directory,
    the file will be copied into dst using the base filename from src. If
    dst specifies a file that already exists, it will be replaced.

    Args:
        src (str or Path): A file to be copied.
        dst (str or Path): Copy file to dst.
        backend_args (dict, optional): Arguments to instantiate the
            corresponding backend. Defaults to None.

    Returns:
        str: The destination file.

    Raises:
        SameFileError: If src and dst are the same file, a SameFileError will
            be raised.

    Examples:
        >>> # dst is a file
        >>> src = '/path/of/file'
        >>> dst = '/path1/of/file1'
        >>> # src will be copied to '/path1/of/file1'
        >>> copyfile(src, dst)
        '/path1/of/file1'

        >>> # dst is a directory
        >>> dst = '/path1/of/dir'
        >>> # src will be copied to '/path1/of/dir/file'
        >>> copyfile(src, dst)
        '/path1/of/dir/file'
    """
    backend = get_file_backend(
        src, backend_args=backend_args, enable_singleton=True)
    return backend.copyfile(src, dst)


def copytree(
    src: Union[str, Path],
    dst: Union[str, Path],
    backend_args: Optional[dict] = None,
) -> Union[str, Path]:
    """Recursively copy an entire directory tree rooted at src to a directory
    named dst and return the destination directory.

    src and dst should have the same prefix and dst must not already exist.

    Args:
        src (str or Path): A directory to be copied.
        dst (str or Path): Copy directory to dst.
        backend_args (dict, optional): Arguments to instantiate the
            corresponding backend. Defaults to None.

    Returns:
        str: The destination directory.

    Raises:
        FileExistsError: If dst had already existed, a FileExistsError will be
            raised.

    Examples:
        >>> src = '/path/of/dir1'
        >>> dst = '/path/of/dir2'
        >>> copytree(src, dst)
        '/path/of/dir2'
    """
    backend = get_file_backend(
        src, backend_args=backend_args, enable_singleton=True)
    return backend.copytree(src, dst)


def copyfile_from_local(
    src: Union[str, Path],
    dst: Union[str, Path],
    backend_args: Optional[dict] = None,
) -> Union[str, Path]:
    """Copy a local file src to dst and return the destination file.

    Note:
        If the backend is the instance of LocalBackend, it does the same
        thing with :func:`copyfile`.

    Args:
        src (str or Path): A local file to be copied.
        dst (str or Path): Copy file to dst.
        backend_args (dict, optional): Arguments to instantiate the
            corresponding backend. Defaults to None.

    Returns:
        str: If dst specifies a directory, the file will be copied into dst
        using the base filename from src.

    Examples:
        >>> # dst is a file
        >>> src = '/path/of/file'
        >>> dst = 's3://openmmlab/mmengine/file1'
        >>> # src will be copied to 's3://openmmlab/mmengine/file1'
        >>> copyfile_from_local(src, dst)
        s3://openmmlab/mmengine/file1

        >>> # dst is a directory
        >>> dst = 's3://openmmlab/mmengine'
        >>> # src will be copied to 's3://openmmlab/mmengine/file''
        >>> copyfile_from_local(src, dst)
        's3://openmmlab/mmengine/file'
    """
    backend = get_file_backend(
        dst, backend_args=backend_args, enable_singleton=True)
    return backend.copyfile_from_local(src, dst)


def copytree_from_local(
    src: Union[str, Path],
    dst: Union[str, Path],
    backend_args: Optional[dict] = None,
) -> Union[str, Path]:
    """Recursively copy an entire directory tree rooted at src to a directory
    named dst and return the destination directory.

    Note:
        If the backend is the instance of LocalBackend, it does the same
        thing with :func:`copytree`.

    Args:
        src (str or Path): A local directory to be copied.
        dst (str or Path): Copy directory to dst.
        backend_args (dict, optional): Arguments to instantiate the
            corresponding backend. Defaults to None.

    Returns:
        str: The destination directory.

    Examples:
        >>> src = '/path/of/dir'
        >>> dst = 's3://openmmlab/mmengine/dir'
        >>> copyfile_from_local(src, dst)
        's3://openmmlab/mmengine/dir'
    """
    backend = get_file_backend(
        dst, backend_args=backend_args, enable_singleton=True)
    return backend.copytree_from_local(src, dst)


def copyfile_to_local(
    src: Union[str, Path],
    dst: Union[str, Path],
    backend_args: Optional[dict] = None,
) -> Union[str, Path]:
    """Copy the file src to local dst and return the destination file.

    If dst specifies a directory, the file will be copied into dst using
    the base filename from src. If dst specifies a file that already
    exists, it will be replaced.

    Note:
        If the backend is the instance of LocalBackend, it does the same
        thing with :func:`copyfile`.

    Args:
        src (str or Path): A file to be copied.
        dst (str or Path): Copy file to to local dst.
        backend_args (dict, optional): Arguments to instantiate the
            corresponding backend. Defaults to None.

    Returns:
        str: If dst specifies a directory, the file will be copied into dst
        using the base filename from src.

    Examples:
        >>> # dst is a file
        >>> src = 's3://openmmlab/mmengine/file'
        >>> dst = '/path/of/file'
        >>> # src will be copied to '/path/of/file'
        >>> copyfile_to_local(src, dst)
        '/path/of/file'

        >>> # dst is a directory
        >>> dst = '/path/of/dir'
        >>> # src will be copied to '/path/of/dir/file'
        >>> copyfile_to_local(src, dst)
        '/path/of/dir/file'
    """
    backend = get_file_backend(
        dst, backend_args=backend_args, enable_singleton=True)
    return backend.copyfile_to_local(src, dst)


def copytree_to_local(
    src: Union[str, Path],
    dst: Union[str, Path],
    backend_args: Optional[dict] = None,
) -> Union[str, Path]:
    """Recursively copy an entire directory tree rooted at src to a local
    directory named dst and return the destination directory.

    Note:
        If the backend is the instance of LocalBackend, it does the same
        thing with :func:`copytree`.

    Args:
        src (str or Path): A directory to be copied.
        dst (str or Path): Copy directory to local dst.
        backend_args (dict, optional): Arguments to instantiate the
            corresponding backend. Defaults to None.

    Returns:
        str: The destination directory.

    Examples:
        >>> src = 's3://openmmlab/mmengine/dir'
        >>> dst = '/path/of/dir'
        >>> copytree_to_local(src, dst)
        '/path/of/dir'
    """
    backend = get_file_backend(
        dst, backend_args=backend_args, enable_singleton=True)
    return backend.copytree_to_local(src, dst)


def remove(
    filepath: Union[str, Path],
    backend_args: Optional[dict] = None,
) -> None:
    """Remove a file.

    Args:
        filepath (str, Path): Path to be removed.
        backend_args (dict, optional): Arguments to instantiate the
            corresponding backend. Defaults to None.

    Raises:
        FileNotFoundError: If filepath does not exist, an FileNotFoundError
            will be raised.
        IsADirectoryError: If filepath is a directory, an IsADirectoryError
            will be raised.

    Examples:
        >>> filepath = '/path/of/file'
        >>> remove(filepath)
    """
    backend = get_file_backend(
        filepath, backend_args=backend_args, enable_singleton=True)
    backend.remove(filepath)


def rmtree(
    dir_path: Union[str, Path],
    backend_args: Optional[dict] = None,
) -> None:
    """Recursively delete a directory tree.

    Args:
        dir_path (str or Path): A directory to be removed.
        backend_args (dict, optional): Arguments to instantiate the
            corresponding backend. Defaults to None.

    Examples:
        >>> dir_path = '/path/of/dir'
        >>> rmtree(dir_path)
    """
    backend = get_file_backend(
        dir_path, backend_args=backend_args, enable_singleton=True)
    backend.rmtree(dir_path)


def copy_if_symlink_fails(
    src: Union[str, Path],
    dst: Union[str, Path],
    backend_args: Optional[dict] = None,
) -> bool:
    """Create a symbolic link pointing to src named dst.

    If failed to create a symbolic link pointing to src, directory copy src to
    dst instead.

    Args:
        src (str or Path): Create a symbolic link pointing to src.
        dst (str or Path): Create a symbolic link named dst.
        backend_args (dict, optional): Arguments to instantiate the
            corresponding backend. Defaults to None.

    Returns:
        bool: Return True if successfully create a symbolic link pointing to
        src. Otherwise, return False.

    Examples:
        >>> src = '/path/of/file'
        >>> dst = '/path1/of/file1'
        >>> copy_if_symlink_fails(src, dst)
        True
        >>> src = '/path/of/dir'
        >>> dst = '/path1/of/dir1'
        >>> copy_if_symlink_fails(src, dst)
        True
    """
    backend = get_file_backend(
        src, backend_args=backend_args, enable_singleton=True)
    return backend.copy_if_symlink_fails(src, dst)


def list_dir_or_file(
    dir_path: Union[str, Path],
    list_dir: bool = True,
    list_file: bool = True,
    suffix: Optional[Union[str, Tuple[str]]] = None,
    recursive: bool = False,
    backend_args: Optional[dict] = None,
) -> Iterator[str]:
    """Scan a directory to find the interested directories or files in
    arbitrary order.

    Note:
        :meth:`list_dir_or_file` returns the path relative to ``dir_path``.

    Args:
        dir_path (str or Path): Path of the directory.
        list_dir (bool): List the directories. Defaults to True.
        list_file (bool): List the path of files. Defaults to True.
        suffix (str or tuple[str], optional): File suffix that we are
            interested in. Defaults to None.
        recursive (bool): If set to True, recursively scan the directory.
            Defaults to False.
        backend_args (dict, optional): Arguments to instantiate the
            corresponding backend. Defaults to None.

    Yields:
        Iterable[str]: A relative path to ``dir_path``.

    Examples:
        >>> dir_path = '/path/of/dir'
        >>> for file_path in list_dir_or_file(dir_path):
        ...     print(file_path)
        >>> # list those files and directories in current directory
        >>> for file_path in list_dir_or_file(dir_path):
        ...     print(file_path)
        >>> # only list files
        >>> for file_path in list_dir_or_file(dir_path, list_dir=False):
        ...     print(file_path)
        >>> # only list directories
        >>> for file_path in list_dir_or_file(dir_path, list_file=False):
        ...     print(file_path)
        >>> # only list files ending with specified suffixes
        >>> for file_path in list_dir_or_file(dir_path, suffix='.txt'):
        ...     print(file_path)
        >>> # list all files and directory recursively
        >>> for file_path in list_dir_or_file(dir_path, recursive=True):
        ...     print(file_path)
    """
    backend = get_file_backend(
        dir_path, backend_args=backend_args, enable_singleton=True)
    yield from backend.list_dir_or_file(dir_path, list_dir, list_file, suffix,
                                        recursive)


def generate_presigned_url(
    url: str,
    client_method: str = 'get_object',
    expires_in: int = 3600,
    backend_args: Optional[dict] = None,
) -> str:
    """Generate the presigned url of video stream which can be passed to
    mmcv.VideoReader. Now only work on Petrel backend.

    Note:
        Now only work on Petrel backend.

    Args:
        url (str): Url of video stream.
        client_method (str): Method of client, 'get_object' or
            'put_object'. Defaults to 'get_object'.
        expires_in (int): expires, in seconds. Defaults to 3600.
        backend_args (dict, optional): Arguments to instantiate the
            corresponding backend. Defaults to None.

    Returns:
        str: Generated presigned url.
    """
    backend = get_file_backend(
        url, backend_args=backend_args, enable_singleton=True)
    return backend.generate_presigned_url(url, client_method, expires_in)


def load(file,
         file_format=None,
         file_client_args=None,
         backend_args=None,
         **kwargs):
    """Load data from json/yaml/pickle files.

    This method provides a unified api for loading data from serialized files.

    ``load`` supports loading data from serialized files those can be storaged
    in different backends.

    Args:
        file (str or :obj:`Path` or file-like object): Filename or a file-like
            object.
        file_format (str, optional): If not specified, the file format will be
            inferred from the file extension, otherwise use the specified one.
            Currently supported formats include "json", "yaml/yml" and
            "pickle/pkl".
        file_client_args (dict, optional): Arguments to instantiate a
            FileClient. See :class:`mmengine.fileio.FileClient` for details.
            Defaults to None. It will be deprecated in future. Please use
            ``backend_args`` instead.
        backend_args (dict, optional): Arguments to instantiate the
            prefix of uri corresponding backend. Defaults to None.
            New in v0.2.0.

    Examples:
        >>> load('/path/of/your/file')  # file is storaged in disk
        >>> load('https://path/of/your/file')  # file is storaged in Internet
        >>> load('s3://path/of/your/file')  # file is storaged in petrel

    Returns:
        The content from the file.
    """
    if isinstance(file, Path):
        file = str(file)
    if file_format is None and is_str(file):
        file_format = file.split('.')[-1]
    if file_format not in file_handlers:
        raise TypeError(f'Unsupported format: {file_format}')

    if file_client_args is not None:
        warnings.warn(
            '"file_client_args" will be deprecated in future. '
            'Please use "backend_args" instead', DeprecationWarning)
        if backend_args is not None:
            raise ValueError(
                '"file_client_args and "backend_args" cannot be set at the '
                'same time.')

    handler = file_handlers[file_format]
    if is_str(file):
        if file_client_args is not None:
            file_client = FileClient.infer_client(file_client_args, file)
            file_backend = file_client
        else:
            file_backend = get_file_backend(file, backend_args=backend_args)

        if handler.str_like:
            with StringIO(file_backend.get_text(file)) as f:
                obj = handler.load_from_fileobj(f, **kwargs)
        else:
            with BytesIO(file_backend.get(file)) as f:
                obj = handler.load_from_fileobj(f, **kwargs)
    elif hasattr(file, 'read'):
        obj = handler.load_from_fileobj(file, **kwargs)
    else:
        raise TypeError('"file" must be a filepath str or a file-object')
    return obj


def dump(obj,
         file=None,
         file_format=None,
         file_client_args=None,
         backend_args=None,
         **kwargs):
    """Dump data to json/yaml/pickle strings or files.

    This method provides a unified api for dumping data as strings or to files,
    and also supports custom arguments for each file format.

    ``dump`` supports dumping data as strings or to files which is saved to
    different backends.

    Args:
        obj (any): The python object to be dumped.
        file (str or :obj:`Path` or file-like object, optional): If not
            specified, then the object is dumped to a str, otherwise to a file
            specified by the filename or file-like object.
        file_format (str, optional): Same as :func:`load`.
        file_client_args (dict, optional): Arguments to instantiate a
            FileClient. See :class:`mmengine.fileio.FileClient` for details.
            Defaults to None. It will be deprecated in future. Please use
            ``backend_args`` instead.
        backend_args (dict, optional): Arguments to instantiate the
            prefix of uri corresponding backend. Defaults to None.
            New in v0.2.0.

    Examples:
        >>> dump('hello world', '/path/of/your/file')  # disk
        >>> dump('hello world', 's3://path/of/your/file')  # ceph or petrel

    Returns:
        bool: True for success, False otherwise.
    """
    if isinstance(file, Path):
        file = str(file)
    if file_format is None:
        if is_str(file):
            file_format = file.split('.')[-1]
        elif file is None:
            raise ValueError(
                'file_format must be specified since file is None')
    if file_format not in file_handlers:
        raise TypeError(f'Unsupported format: {file_format}')

    if file_client_args is not None:
        warnings.warn(
            '"file_client_args" will be deprecated in future. '
            'Please use "backend_args" instead', DeprecationWarning)
        if backend_args is not None:
            raise ValueError(
                '"file_client_args" and "backend_args" cannot be set at the '
                'same time.')

    handler = file_handlers[file_format]
    if file is None:
        return handler.dump_to_str(obj, **kwargs)
    elif is_str(file):
        if file_client_args is not None:
            file_client = FileClient.infer_client(file_client_args, file)
            file_backend = file_client
        else:
            file_backend = get_file_backend(file, backend_args=backend_args)

        if handler.str_like:
            with StringIO() as f:
                handler.dump_to_fileobj(obj, f, **kwargs)
                file_backend.put_text(f.getvalue(), file)
        else:
            with BytesIO() as f:
                handler.dump_to_fileobj(obj, f, **kwargs)
                file_backend.put(f.getvalue(), file)
    elif hasattr(file, 'write'):
        handler.dump_to_fileobj(obj, file, **kwargs)
    else:
        raise TypeError('"file" must be a filename str or a file-object')

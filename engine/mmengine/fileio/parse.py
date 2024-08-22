# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import warnings
from io import StringIO

from .file_client import FileClient
from .io import get_text


def list_from_file(filename,
                   prefix='',
                   offset=0,
                   max_num=0,
                   encoding='utf-8',
                   file_client_args=None,
                   backend_args=None):
    """Load a text file and parse the content as a list of strings.

    ``list_from_file`` supports loading a text file which can be storaged in
    different backends and parsing the content as a list for strings.

    Args:
        filename (str): Filename.
        prefix (str): The prefix to be inserted to the beginning of each item.
        offset (int): The offset of lines.
        max_num (int): The maximum number of lines to be read,
            zeros and negatives mean no limitation.
        encoding (str): Encoding used to open the file. Defaults to utf-8.
        file_client_args (dict, optional): Arguments to instantiate a
            FileClient. See :class:`mmengine.fileio.FileClient` for details.
            Defaults to None. It will be deprecated in future. Please use
            ``backend_args`` instead.
        backend_args (dict, optional): Arguments to instantiate the
            prefix of uri corresponding backend. Defaults to None.
            New in v0.2.0.

    Examples:
        >>> list_from_file('/path/of/your/file')  # disk
        ['hello', 'world']
        >>> list_from_file('s3://path/of/your/file')  # ceph or petrel
        ['hello', 'world']

    Returns:
        list[str]: A list of strings.
    """
    if file_client_args is not None:
        warnings.warn(
            '"file_client_args" will be deprecated in future. '
            'Please use "backend_args" instead', DeprecationWarning)
        if backend_args is not None:
            raise ValueError(
                '"file_client_args" and "backend_args" cannot be set at the '
                'same time.')
    cnt = 0
    item_list = []

    if file_client_args is not None:
        file_client = FileClient.infer_client(file_client_args, filename)
        text = file_client.get_text(filename, encoding)
    else:
        text = get_text(filename, encoding, backend_args=backend_args)

    with StringIO(text) as f:
        for _ in range(offset):
            f.readline()
        for line in f:
            if 0 < max_num <= cnt:
                break
            item_list.append(prefix + line.rstrip('\n\r'))
            cnt += 1
    return item_list


def dict_from_file(filename,
                   key_type=str,
                   encoding='utf-8',
                   file_client_args=None,
                   backend_args=None):
    """Load a text file and parse the content as a dict.

    Each line of the text file will be two or more columns split by
    whitespaces or tabs. The first column will be parsed as dict keys, and
    the following columns will be parsed as dict values.

    ``dict_from_file`` supports loading a text file which can be storaged in
    different backends and parsing the content as a dict.

    Args:
        filename(str): Filename.
        key_type(type): Type of the dict keys. str is user by default and
            type conversion will be performed if specified.
        encoding (str): Encoding used to open the file. Defaults to utf-8.
        file_client_args (dict, optional): Arguments to instantiate a
            FileClient. See :class:`mmengine.fileio.FileClient` for details.
            Defaults to None. It will be deprecated in future. Please use
            ``backend_args`` instead.
        backend_args (dict, optional): Arguments to instantiate the
            prefix of uri corresponding backend. Defaults to None.
            New in v0.2.0.

    Examples:
        >>> dict_from_file('/path/of/your/file')  # disk
        {'key1': 'value1', 'key2': 'value2'}
        >>> dict_from_file('s3://path/of/your/file')  # ceph or petrel
        {'key1': 'value1', 'key2': 'value2'}

    Returns:
        dict: The parsed contents.
    """
    if file_client_args is not None:
        warnings.warn(
            '"file_client_args" will be deprecated in future. '
            'Please use "backend_args" instead', DeprecationWarning)
        if backend_args is not None:
            raise ValueError(
                '"file_client_args" and "backend_args" cannot be set at the '
                'same time.')

    mapping = {}

    if file_client_args is not None:
        file_client = FileClient.infer_client(file_client_args, filename)
        text = file_client.get_text(filename, encoding)
    else:
        text = get_text(filename, encoding, backend_args=backend_args)

    with StringIO(text) as f:
        for line in f:
            items = line.rstrip('\n').split()
            assert len(items) >= 2
            key = key_type(items[0])
            val = items[1:] if len(items) > 2 else items[1]
            mapping[key] = val
    return mapping

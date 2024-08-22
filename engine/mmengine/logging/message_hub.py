# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import copy
import logging
from collections import OrderedDict
from typing import TYPE_CHECKING, Any, Optional, Union

import numpy as np

from mmengine.utils import ManagerMixin
from .history_buffer import HistoryBuffer
from .logger import print_log

if TYPE_CHECKING:
    import torch


class MessageHub(ManagerMixin):
    """Message hub for component interaction. MessageHub is created and
    accessed in the same way as ManagerMixin.

    ``MessageHub`` will record log information and runtime information. The
    log information refers to the learning rate, loss, etc. of the model
    during training phase, which will be stored as ``HistoryBuffer``. The
    runtime information refers to the iter times, meta information of
    runner etc., which will be overwritten by next update.

    Args:
        name (str): Name of message hub used to get corresponding instance
            globally.
        log_scalars (dict, optional): Each key-value pair in the
            dictionary is the name of the log information such as "loss", "lr",
            "metric" and their corresponding values. The type of value must be
            HistoryBuffer. Defaults to None.
        runtime_info (dict, optional): Each key-value pair in the
            dictionary is the name of the runtime information and their
            corresponding values. Defaults to None.
        resumed_keys (dict, optional): Each key-value pair in the
            dictionary decides whether the key in :attr:`_log_scalars` and
            :attr:`_runtime_info` will be serialized.

    Note:
        Key in :attr:`_resumed_keys` belongs to :attr:`_log_scalars` or
        :attr:`_runtime_info`. The corresponding value cannot be set
        repeatedly.

    Examples:
        >>> # create empty `MessageHub`.
        >>> message_hub1 = MessageHub('name')
        >>> log_scalars = dict(loss=HistoryBuffer())
        >>> runtime_info = dict(task='task')
        >>> resumed_keys = dict(loss=True)
        >>> # create `MessageHub` from data.
        >>> message_hub2 = MessageHub(
        >>>     name='name',
        >>>     log_scalars=log_scalars,
        >>>     runtime_info=runtime_info,
        >>>     resumed_keys=resumed_keys)
    """

    def __init__(self,
                 name: str,
                 log_scalars: Optional[dict] = None,
                 runtime_info: Optional[dict] = None,
                 resumed_keys: Optional[dict] = None):
        super().__init__(name)
        self._log_scalars = self._parse_input('log_scalars', log_scalars)
        self._runtime_info = self._parse_input('runtime_info', runtime_info)
        self._resumed_keys = self._parse_input('resumed_keys', resumed_keys)

        for value in self._log_scalars.values():
            assert isinstance(value, HistoryBuffer), \
                ("The type of log_scalars'value must be HistoryBuffer, but "
                 f'got {type(value)}')

        for key in self._resumed_keys.keys():
            assert key in self._log_scalars or key in self._runtime_info, \
                ('Key in `resumed_keys` must contained in `log_scalars` or '
                 f'`runtime_info`, but got {key}')

    @classmethod
    def get_current_instance(cls) -> 'MessageHub':
        """Get latest created ``MessageHub`` instance.

        :obj:`MessageHub` can call :meth:`get_current_instance` before any
        instance has been created, and return a message hub with the instance
        name "mmengine".

        Returns:
            MessageHub: Empty ``MessageHub`` instance.
        """
        if not cls._instance_dict:
            cls.get_instance('mmengine')
        return super().get_current_instance()

    def update_scalar(self,
                      key: str,
                      value: Union[int, float, np.ndarray, 'torch.Tensor'],
                      count: int = 1,
                      resumed: bool = True) -> None:
        """Update :attr:_log_scalars.

        Update ``HistoryBuffer`` in :attr:`_log_scalars`. If corresponding key
        ``HistoryBuffer`` has been created, ``value`` and ``count`` is the
        argument of ``HistoryBuffer.update``, Otherwise, ``update_scalar``
        will create an ``HistoryBuffer`` with value and count via the
        constructor of ``HistoryBuffer``.

        Examples:
            >>> message_hub = MessageHub(name='name')
            >>> # create loss `HistoryBuffer` with value=1, count=1
            >>> message_hub.update_scalar('loss', 1)
            >>> # update loss `HistoryBuffer` with value
            >>> message_hub.update_scalar('loss', 3)
            >>> message_hub.update_scalar('loss', 3, resumed=False)
            AssertionError: loss used to be true, but got false now. resumed
            keys cannot be modified repeatedly'

        Note:
            The ``resumed`` argument needs to be consistent for the same
            ``key``.

        Args:
            key (str): Key of ``HistoryBuffer``.
            value (torch.Tensor or np.ndarray or int or float): Value of log.
            count (torch.Tensor or np.ndarray or int or float): Accumulation
                times of log, defaults to 1. `count` will be used in smooth
                statistics.
            resumed (str): Whether the corresponding ``HistoryBuffer``
                could be resumed. Defaults to True.
        """
        self._set_resumed_keys(key, resumed)
        checked_value = self._get_valid_value(value)
        assert isinstance(count, int), (
            f'The type of count must be int. but got {type(count): {count}}')
        if key in self._log_scalars:
            self._log_scalars[key].update(checked_value, count)
        else:
            self._log_scalars[key] = HistoryBuffer([checked_value], [count])

    def update_scalars(self, log_dict: dict, resumed: bool = True) -> None:
        """Update :attr:`_log_scalars` with a dict.

        ``update_scalars`` iterates through each pair of log_dict key-value,
        and calls ``update_scalar``. If type of value is dict, the value should
        be ``dict(value=xxx) or dict(value=xxx, count=xxx)``. Item in
        ``log_dict`` has the same resume option.

        Note:
            The ``resumed`` argument needs to be consistent for the same
            ``log_dict``.

        Args:
            log_dict (str): Used for batch updating :attr:`_log_scalars`.
            resumed (bool): Whether all ``HistoryBuffer`` referred in
                log_dict should be resumed. Defaults to True.

        Examples:
            >>> message_hub = MessageHub.get_instance('mmengine')
            >>> log_dict = dict(a=1, b=2, c=3)
            >>> message_hub.update_scalars(log_dict)
            >>> # The default count of  `a`, `b` and `c` is 1.
            >>> log_dict = dict(a=1, b=2, c=dict(value=1, count=2))
            >>> message_hub.update_scalars(log_dict)
            >>> # The count of `c` is 2.
        """
        assert isinstance(log_dict, dict), ('`log_dict` must be a dict!, '
                                            f'but got {type(log_dict)}')
        for log_name, log_val in log_dict.items():
            if isinstance(log_val, dict):
                assert 'value' in log_val, \
                    f'value must be defined in {log_val}'
                count = self._get_valid_value(log_val.get('count', 1))
                value = log_val['value']
            else:
                count = 1
                value = log_val
            assert isinstance(count,
                              int), ('The type of count must be int. but got '
                                     f'{type(count): {count}}')
            self.update_scalar(log_name, value, count, resumed)

    def update_info(self, key: str, value: Any, resumed: bool = True) -> None:
        """Update runtime information.

        The key corresponding runtime information will be overwritten each
        time calling ``update_info``.

        Note:
            The ``resumed`` argument needs to be consistent for the same
            ``key``.

        Examples:
            >>> message_hub = MessageHub(name='name')
            >>> message_hub.update_info('iter', 100)

        Args:
            key (str): Key of runtime information.
            value (Any): Value of runtime information.
            resumed (bool): Whether the corresponding ``HistoryBuffer``
                could be resumed.
        """
        self._set_resumed_keys(key, resumed)
        self._runtime_info[key] = value

    def pop_info(self, key: str, default: Optional[Any] = None) -> Any:
        """Remove runtime information by key. If the key does not exist, this
        method will return the default value.

        Args:
            key (str): Key of runtime information.
            default (Any, optional): The default returned value for the
                given key.

        Returns:
            Any: The runtime information if the key exists.
        """
        return self._runtime_info.pop(key, default)

    def update_info_dict(self, info_dict: dict, resumed: bool = True) -> None:
        """Update runtime information with dictionary.

        The key corresponding runtime information will be overwritten each
        time calling ``update_info``.

        Note:
            The ``resumed`` argument needs to be consistent for the same
            ``info_dict``.

        Examples:
            >>> message_hub = MessageHub(name='name')
            >>> message_hub.update_info({'iter': 100})

        Args:
            info_dict (str): Runtime information dictionary.
            resumed (bool): Whether the corresponding ``HistoryBuffer``
                could be resumed.
        """
        assert isinstance(info_dict, dict), ('`log_dict` must be a dict!, '
                                             f'but got {type(info_dict)}')
        for key, value in info_dict.items():
            self.update_info(key, value, resumed=resumed)

    def _set_resumed_keys(self, key: str, resumed: bool) -> None:
        """Set corresponding resumed keys.

        This method is called by ``update_scalar``, ``update_scalars`` and
        ``update_info`` to set the corresponding key is true or false in
        :attr:`_resumed_keys`.

        Args:
            key (str): Key of :attr:`_log_scalrs` or :attr:`_runtime_info`.
            resumed (bool): Whether the corresponding ``HistoryBuffer``
                could be resumed.
        """
        if key not in self._resumed_keys:
            self._resumed_keys[key] = resumed
        else:
            assert self._resumed_keys[key] == resumed, \
                f'{key} used to be {self._resumed_keys[key]}, but got ' \
                '{resumed} now. resumed keys cannot be modified repeatedly.'

    @property
    def log_scalars(self) -> OrderedDict:
        """Get all ``HistoryBuffer`` instances.

        Note:
            Considering the large memory footprint of history buffers in the
            post-training, :meth:`get_scalar` will return a reference of
            history buffer rather than a copy.

        Returns:
            OrderedDict: All ``HistoryBuffer`` instances.
        """
        return self._log_scalars

    @property
    def runtime_info(self) -> OrderedDict:
        """Get all runtime information.

        Returns:
            OrderedDict: A copy of all runtime information.
        """
        return self._runtime_info

    def get_scalar(self, key: str) -> HistoryBuffer:
        """Get ``HistoryBuffer`` instance by key.

        Note:
            Considering the large memory footprint of history buffers in the
            post-training, :meth:`get_scalar` will not return a reference of
            history buffer rather than a copy.

        Args:
            key (str): Key of ``HistoryBuffer``.

        Returns:
            HistoryBuffer: Corresponding ``HistoryBuffer`` instance if the
            key exists.
        """
        if key not in self.log_scalars:
            raise KeyError(f'{key} is not found in Messagehub.log_buffers: '
                           f'instance name is: {MessageHub.instance_name}')
        return self.log_scalars[key]

    def get_info(self, key: str, default: Optional[Any] = None) -> Any:
        """Get runtime information by key. If the key does not exist, this
        method will return default information.

        Args:
            key (str): Key of runtime information.
            default (Any, optional): The default returned value for the
                given key.

        Returns:
            Any: A copy of corresponding runtime information if the key exists.
        """
        if key not in self.runtime_info:
            return default
        else:
            # TODO： There are restrictions on objects that can be saved
            # return copy.deepcopy(self._runtime_info[key])
            return self._runtime_info[key]

    def _get_valid_value(
        self,
        value: Union['torch.Tensor', np.ndarray, np.number, int, float],
    ) -> Union[int, float]:
        """Convert value to python built-in type.

        Args:
            value (torch.Tensor or np.ndarray or np.number or int or float):
                value of log.

        Returns:
            float or int: python built-in type value.
        """
        if isinstance(value, (np.ndarray, np.number)):
            assert value.size == 1
            value = value.item()
        elif isinstance(value, (int, float)):
            value = value
        else:
            # check whether value is torch.Tensor but don't want
            # to import torch in this file
            assert hasattr(value, 'numel') and value.numel() == 1
            value = value.item()
        return value  # type: ignore

    def state_dict(self) -> dict:
        """Returns a dictionary containing log scalars, runtime information and
        resumed keys, which should be resumed.

        The returned ``state_dict`` can be loaded by :meth:`load_state_dict`.

        Returns:
            dict: A dictionary contains ``log_scalars``, ``runtime_info`` and
            ``resumed_keys``.
        """
        saved_scalars = OrderedDict()
        saved_info = OrderedDict()

        for key, value in self._log_scalars.items():
            if self._resumed_keys.get(key, False):
                saved_scalars[key] = copy.deepcopy(value)

        for key, value in self._runtime_info.items():
            if self._resumed_keys.get(key, False):
                try:
                    saved_info[key] = copy.deepcopy(value)
                except:  # noqa: E722
                    print_log(
                        f'{key} in message_hub cannot be copied, '
                        f'just return its reference. ',
                        logger='current',
                        level=logging.WARNING)
                    saved_info[key] = value
        return dict(
            log_scalars=saved_scalars,
            runtime_info=saved_info,
            resumed_keys=self._resumed_keys)

    def load_state_dict(self, state_dict: Union['MessageHub', dict]) -> None:
        """Loads log scalars, runtime information and resumed keys from
        ``state_dict`` or ``message_hub``.

        If ``state_dict`` is a dictionary returned by :meth:`state_dict`, it
        will only make copies of data which should be resumed from the source
        ``message_hub``.

        If ``state_dict`` is a ``message_hub`` instance, it will make copies of
        all data from the source message_hub. We suggest to load data from
        ``dict`` rather than a ``MessageHub`` instance.

        Args:
            state_dict (dict or MessageHub): A dictionary contains key
                ``log_scalars`` ``runtime_info`` and ``resumed_keys``, or a
                MessageHub instance.
        """
        if isinstance(state_dict, dict):
            for key in ('log_scalars', 'runtime_info', 'resumed_keys'):
                assert key in state_dict, (
                    'The loaded `state_dict` of `MessageHub` must contain '
                    f'key: `{key}`')
            # The old `MessageHub` could save non-HistoryBuffer `log_scalars`,
            # therefore the loaded `log_scalars` needs to be filtered.
            for key, value in state_dict['log_scalars'].items():
                if not isinstance(value, HistoryBuffer):
                    print_log(
                        f'{key} in message_hub is not HistoryBuffer, '
                        f'just skip resuming it.',
                        logger='current',
                        level=logging.WARNING)
                    continue
                self.log_scalars[key] = value

            for key, value in state_dict['runtime_info'].items():
                try:
                    self._runtime_info[key] = copy.deepcopy(value)
                except:  # noqa: E722
                    print_log(
                        f'{key} in message_hub cannot be copied, '
                        f'just return its reference.',
                        logger='current',
                        level=logging.WARNING)
                    self._runtime_info[key] = value

            for key, value in state_dict['resumed_keys'].items():
                if key not in set(self.log_scalars.keys()) | \
                        set(self._runtime_info.keys()):
                    print_log(
                        f'resumed key: {key} is not defined in message_hub, '
                        f'just skip resuming this key.',
                        logger='current',
                        level=logging.WARNING)
                    continue
                elif not value:
                    print_log(
                        f'Although resumed key: {key} is False, {key} '
                        'will still be loaded this time. This key will '
                        'not be saved by the next calling of '
                        '`MessageHub.state_dict()`',
                        logger='current',
                        level=logging.WARNING)
                self._resumed_keys[key] = value

        # Since some checkpoints saved serialized `message_hub` instance,
        # `load_state_dict` support loading `message_hub` instance for
        # compatibility
        else:
            self._log_scalars = copy.deepcopy(state_dict._log_scalars)
            self._runtime_info = copy.deepcopy(state_dict._runtime_info)
            self._resumed_keys = copy.deepcopy(state_dict._resumed_keys)

    def _parse_input(self, name: str, value: Any) -> OrderedDict:
        """Parse input value.

        Args:
            name (str): name of input value.
            value (Any): Input value.

        Returns:
            dict: Parsed input value.
        """
        if value is None:
            return OrderedDict()
        elif isinstance(value, dict):
            return OrderedDict(value)
        else:
            raise TypeError(f'{name} should be a dict or `None`, but '
                            f'got {type(name)}')

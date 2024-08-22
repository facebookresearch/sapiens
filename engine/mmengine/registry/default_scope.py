# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import copy
import time
from contextlib import contextmanager
from typing import Generator, Optional

from mmengine.utils.manager import ManagerMixin, _accquire_lock, _release_lock


class DefaultScope(ManagerMixin):
    """Scope of current task used to reset the current registry, which can be
    accessed globally.

    Consider the case of resetting the current ``Registry`` by
    ``default_scope`` in the internal module which cannot access runner
    directly, it is difficult to get the ``default_scope`` defined in
    ``Runner``. However, if ``Runner`` created ``DefaultScope`` instance
    by given ``default_scope``, the internal module can get
    ``default_scope`` by ``DefaultScope.get_current_instance`` everywhere.

    Args:
        name (str): Name of default scope for global access.
        scope_name (str): Scope of current task.

    Examples:
        >>> from mmengine.model import MODELS
        >>> # Define default scope in runner.
        >>> DefaultScope.get_instance('task', scope_name='mmdet')
        >>> # Get default scope globally.
        >>> scope_name = DefaultScope.get_instance('task').scope_name
    """

    def __init__(self, name: str, scope_name: str):
        super().__init__(name)
        assert isinstance(
            scope_name,
            str), (f'scope_name should be a string, but got {scope_name}')
        self._scope_name = scope_name

    @property
    def scope_name(self) -> str:
        """
        Returns:
            str: Get current scope.
        """
        return self._scope_name

    @classmethod
    def get_current_instance(cls) -> Optional['DefaultScope']:
        """Get latest created default scope.

        Since default_scope is an optional argument for ``Registry.build``.
        ``get_current_instance`` should return ``None`` if there is no
        ``DefaultScope`` created.

        Examples:
            >>> default_scope = DefaultScope.get_current_instance()
            >>> # There is no `DefaultScope` created yet,
            >>> # `get_current_instance` return `None`.
            >>> default_scope = DefaultScope.get_instance(
            >>>     'instance_name', scope_name='mmengine')
            >>> default_scope.scope_name
            mmengine
            >>> default_scope = DefaultScope.get_current_instance()
            >>> default_scope.scope_name
            mmengine

        Returns:
            Optional[DefaultScope]: Return None If there has not been
            ``DefaultScope`` instance created yet, otherwise return the
            latest created DefaultScope instance.
        """
        _accquire_lock()
        if cls._instance_dict:
            instance = super().get_current_instance()
        else:
            instance = None
        _release_lock()
        return instance

    @classmethod
    @contextmanager
    def overwrite_default_scope(cls, scope_name: Optional[str]) -> Generator:
        """overwrite the current default scope with `scope_name`"""
        if scope_name is None:
            yield
        else:
            tmp = copy.deepcopy(cls._instance_dict)
            # To avoid create an instance with the same name.
            time.sleep(1e-6)
            cls.get_instance(f'overwrite-{time.time()}', scope_name=scope_name)
            try:
                yield
            finally:
                cls._instance_dict = tmp

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import importlib
from typing import Any, Optional, Union

from mmengine.utils import is_seq_of


class LazyObject:
    """LazyObject is used to lazily initialize the imported module during
    parsing the configuration file.

    During parsing process, the syntax like:

    Examples:
        >>> import torch.nn as nn
        >>> from mmdet.models import RetinaNet
        >>> import mmcls.models
        >>> import mmcls.datasets
        >>> import mmcls

    Will be parsed as:

    Examples:
        >>> # import torch.nn as nn
        >>> nn = lazyObject('torch.nn')
        >>> # from mmdet.models import RetinaNet
        >>> RetinaNet = lazyObject('mmdet.models', 'RetinaNet')
        >>> # import mmcls.models; import mmcls.datasets; import mmcls
        >>> mmcls = lazyObject(['mmcls', 'mmcls.datasets', 'mmcls.models'])

    ``LazyObject`` records all module information and will be further
    referenced by the configuration file.

    Args:
        module (str or list or tuple): The module name to be imported.
        imported (str, optional): The imported module name. Defaults to None.
        location (str, optional): The filename and line number of the imported
            module statement happened.
    """

    def __init__(self,
                 module: Union[str, list, tuple],
                 imported: Optional[str] = None,
                 location: Optional[str] = None):
        if not isinstance(module, str) and not is_seq_of(module, str):
            raise TypeError('module should be `str`, `list`, or `tuple`'
                            f'but got {type(module)}, this might be '
                            'a bug of MMEngine, please report it to '
                            'https://github.com/open-mmlab/mmengine/issues')
        self._module: Union[str, list, tuple] = module

        if not isinstance(imported, str) and imported is not None:
            raise TypeError('imported should be `str` or None, but got '
                            f'{type(imported)}, this might be '
                            'a bug of MMEngine, please report it to '
                            'https://github.com/open-mmlab/mmengine/issues')
        self._imported = imported
        self.location = location

    def build(self) -> Any:
        """Return imported object.

        Returns:
            Any: Imported object
        """
        if isinstance(self._module, str):
            try:
                module = importlib.import_module(self._module)
            except Exception as e:
                raise type(e)(f'Failed to import {self._module} '
                              f'in {self.location} for {e}')

            if self._imported is not None:
                if hasattr(module, self._imported):
                    module = getattr(module, self._imported)
                else:
                    raise ImportError(
                        f'Failed to import {self._imported} '
                        f'from {self._module} in {self.location}')

            return module
        else:
            # import xxx.xxx
            # import xxx.yyy
            # import xxx.zzz
            # return imported xxx
            try:
                for module in self._module:
                    importlib.import_module(module)  # type: ignore
                module_name = self._module[0].split('.')[0]
                return importlib.import_module(module_name)
            except Exception as e:
                raise type(e)(f'Failed to import {self.module} '
                              f'in {self.location} for {e}')

    @property
    def module(self):
        if isinstance(self._module, str):
            return self._module
        return self._module[0].split('.')[0]

    def __call__(self, *args, **kwargs):
        raise RuntimeError()

    def __deepcopy__(self, memo):
        return LazyObject(self._module, self._imported, self.location)

    def __getattr__(self, name):
        # Cannot locate the line number of the getting attribute.
        # Therefore only record the filename.
        if self.location is not None:
            location = self.location.split(', line')[0]
        else:
            location = self.location
        return LazyAttr(name, self, location)

    def __str__(self) -> str:
        if self._imported is not None:
            return self._imported
        return self.module

    __repr__ = __str__

    # `pickle.dump` will try to get the `__getstate__` and `__setstate__`
    # methods of the dumped object. If these two methods are not defined,
    # LazyObject will return a `__getstate__` LazyObject` or `__setstate__`
    # LazyObject.
    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__ = state


class LazyAttr:
    """The attribute of the LazyObject.

    When parsing the configuration file, the imported syntax will be
    parsed as the assignment ``LazyObject``. During the subsequent parsing
    process, users may reference the attributes of the LazyObject.
    To ensure that these attributes also contain information needed to
    reconstruct the attribute itself, LazyAttr was introduced.

    Examples:
        >>> models = LazyObject(['mmdet.models'])
        >>> model = dict(type=models.RetinaNet)
        >>> print(type(model['type']))  # <class 'mmengine.config.lazy.LazyAttr'>
        >>> print(model['type'].build())  # <class 'mmdet.models.detectors.retinanet.RetinaNet'>
    """  # noqa: E501

    def __init__(self,
                 name: str,
                 source: Union['LazyObject', 'LazyAttr'],
                 location=None):
        self.name = name
        self.source: Union[LazyAttr, LazyObject] = source

        if isinstance(self.source, LazyObject):
            if isinstance(self.source._module, str):
                if self.source._imported is None:
                    # source code:
                    # from xxx.yyy import zzz
                    # equivalent code:
                    # zzz = LazyObject('xxx.yyy', 'zzz')
                    # The source code of get attribute:
                    # eee = zzz.eee
                    # Then, `eee._module` should be "xxx.yyy.zzz"
                    self._module = self.source._module
                else:
                    # source code:
                    # import xxx.yyy as zzz
                    # equivalent code:
                    # zzz = LazyObject('xxx.yyy')
                    # The source code of get attribute:
                    # eee = zzz.eee
                    # Then, `eee._module` should be "xxx.yyy"
                    self._module = f'{self.source._module}.{self.source}'
            else:
                # The source code of LazyObject should be
                # 1. import xxx.yyy
                # 2. import xxx.zzz
                # Equivalent to
                # xxx = LazyObject(['xxx.yyy', 'xxx.zzz'])

                # The source code of LazyAttr should be
                # eee = xxx.eee
                # Then, eee._module = xxx
                self._module = str(self.source)
        elif isinstance(self.source, LazyAttr):
            # 1. import xxx
            # 2. zzz = xxx.yyy.zzz

            # Equivalent to:
            # xxx = LazyObject('xxx')
            # zzz = xxx.yyy.zzz
            # zzz._module = xxx.yyy._module + zzz.name
            self._module = f'{self.source._module}.{self.source.name}'
        self.location = location

    @property
    def module(self):
        return self._module

    def __call__(self, *args, **kwargs: Any) -> Any:
        raise RuntimeError()

    def __getattr__(self, name: str) -> 'LazyAttr':
        return LazyAttr(name, self)

    def __deepcopy__(self, memo):
        return LazyAttr(self.name, self.source)

    def build(self) -> Any:
        """Return the attribute of the imported object.

        Returns:
            Any: attribute of the imported object.
        """
        obj = self.source.build()
        try:
            return getattr(obj, self.name)
        except AttributeError:
            raise ImportError(f'Failed to import {self.module}.{self.name} in '
                              f'{self.location}')
        except ImportError as e:
            raise e

    def __str__(self) -> str:
        return self.name

    __repr__ = __str__

    # `pickle.dump` will try to get the `__getstate__` and `__setstate__`
    # methods of the dumped object. If these two methods are not defined,
    # LazyAttr will return a `__getstate__` LazyAttr` or `__setstate__`
    # LazyAttr.
    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__ = state

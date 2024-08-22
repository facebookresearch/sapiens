# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import inspect
import logging
import sys
from collections.abc import Callable
from contextlib import contextmanager
from importlib import import_module
from typing import Any, Dict, Generator, List, Optional, Tuple, Type, Union

from rich.console import Console
from rich.table import Table

from mmengine.config.utils import MODULE2PACKAGE
from mmengine.utils import get_object_from_string, is_seq_of
from .default_scope import DefaultScope


class Registry:
    """A registry to map strings to classes or functions.

    Registered object could be built from registry. Meanwhile, registered
    functions could be called from registry.

    Args:
        name (str): Registry name.
        build_func (callable, optional): A function to construct instance
            from Registry. :func:`build_from_cfg` is used if neither ``parent``
            or ``build_func`` is specified. If ``parent`` is specified and
            ``build_func`` is not given,  ``build_func`` will be inherited
            from ``parent``. Defaults to None.
        parent (:obj:`Registry`, optional): Parent registry. The class
            registered in children registry could be built from parent.
            Defaults to None.
        scope (str, optional): The scope of registry. It is the key to search
            for children registry. If not specified, scope will be the name of
            the package where class is defined, e.g. mmdet, mmcls, mmseg.
            Defaults to None.
        locations (list): The locations to import the modules registered
            in this registry. Defaults to [].
            New in version 0.4.0.

    Examples:
        >>> # define a registry
        >>> MODELS = Registry('models')
        >>> # registry the `ResNet` to `MODELS`
        >>> @MODELS.register_module()
        >>> class ResNet:
        >>>     pass
        >>> # build model from `MODELS`
        >>> resnet = MODELS.build(dict(type='ResNet'))
        >>> @MODELS.register_module()
        >>> def resnet50():
        >>>     pass
        >>> resnet = MODELS.build(dict(type='resnet50'))

        >>> # hierarchical registry
        >>> DETECTORS = Registry('detectors', parent=MODELS, scope='det')
        >>> @DETECTORS.register_module()
        >>> class FasterRCNN:
        >>>     pass
        >>> fasterrcnn = DETECTORS.build(dict(type='FasterRCNN'))

        >>> # add locations to enable auto import
        >>> DETECTORS = Registry('detectors', parent=MODELS,
        >>>     scope='det', locations=['det.models.detectors'])
        >>> # define this class in 'det.models.detectors'
        >>> @DETECTORS.register_module()
        >>> class MaskRCNN:
        >>>     pass
        >>> # The registry will auto import det.models.detectors.MaskRCNN
        >>> fasterrcnn = DETECTORS.build(dict(type='det.MaskRCNN'))

    More advanced usages can be found at
    https://mmengine.readthedocs.io/en/latest/advanced_tutorials/registry.html.
    """

    def __init__(self,
                 name: str,
                 build_func: Optional[Callable] = None,
                 parent: Optional['Registry'] = None,
                 scope: Optional[str] = None,
                 locations: List = []):
        from .build_functions import build_from_cfg
        self._name = name
        self._module_dict: Dict[str, Type] = dict()
        self._children: Dict[str, 'Registry'] = dict()
        self._locations = locations
        self._imported = False

        if scope is not None:
            assert isinstance(scope, str)
            self._scope = scope
        else:
            self._scope = self.infer_scope()

        # See https://mypy.readthedocs.io/en/stable/common_issues.html#
        # variables-vs-type-aliases for the use
        self.parent: Optional['Registry']
        if parent is not None:
            assert isinstance(parent, Registry)
            parent._add_child(self)
            self.parent = parent
        else:
            self.parent = None

        # self.build_func will be set with the following priority:
        # 1. build_func
        # 2. parent.build_func
        # 3. build_from_cfg
        self.build_func: Callable
        if build_func is None:
            if self.parent is not None:
                self.build_func = self.parent.build_func
            else:
                self.build_func = build_from_cfg
        else:
            self.build_func = build_func

    def __len__(self):
        return len(self._module_dict)

    def __contains__(self, key):
        return self.get(key) is not None

    def __repr__(self):
        table = Table(title=f'Registry of {self._name}')
        table.add_column('Names', justify='left', style='cyan')
        table.add_column('Objects', justify='left', style='green')

        for name, obj in sorted(self._module_dict.items()):
            table.add_row(name, str(obj))

        console = Console()
        with console.capture() as capture:
            console.print(table, end='')

        return capture.get()

    @staticmethod
    def infer_scope() -> str:
        """Infer the scope of registry.

        The name of the package where registry is defined will be returned.

        Returns:
            str: The inferred scope name.

        Examples:
            >>> # in mmdet/models/backbone/resnet.py
            >>> MODELS = Registry('models')
            >>> @MODELS.register_module()
            >>> class ResNet:
            >>>     pass
            >>> # The scope of ``ResNet`` will be ``mmdet``.
        """
        from ..logging import print_log

        # `sys._getframe` returns the frame object that many calls below the
        # top of the stack. The call stack for `infer_scope` can be listed as
        # follow:
        # frame-0: `infer_scope` itself
        # frame-1: `__init__` of `Registry` which calls the `infer_scope`
        # frame-2: Where the `Registry(...)` is called
        module = inspect.getmodule(sys._getframe(2))
        if module is not None:
            filename = module.__name__
            split_filename = filename.split('.')
            scope = split_filename[0]
        else:
            # use "mmengine" to handle some cases which can not infer the scope
            # like initializing Registry in interactive mode
            scope = 'mmengine'
            print_log(
                'set scope as "mmengine" when scope can not be inferred. You '
                'can silence this warning by passing a "scope" argument to '
                'Registry like `Registry(name, scope="toy")`',
                logger='current',
                level=logging.WARNING)

        return scope

    @staticmethod
    def split_scope_key(key: str) -> Tuple[Optional[str], str]:
        """Split scope and key.

        The first scope will be split from key.

        Return:
            tuple[str | None, str]: The former element is the first scope of
            the key, which can be ``None``. The latter is the remaining key.

        Examples:
            >>> Registry.split_scope_key('mmdet.ResNet')
            'mmdet', 'ResNet'
            >>> Registry.split_scope_key('ResNet')
            None, 'ResNet'
        """
        split_index = key.find('.')
        if split_index != -1:
            return key[:split_index], key[split_index + 1:]
        else:
            return None, key

    @property
    def name(self):
        return self._name

    @property
    def scope(self):
        return self._scope

    @property
    def module_dict(self):
        return self._module_dict

    @property
    def children(self):
        return self._children

    @property
    def root(self):
        return self._get_root_registry()

    @contextmanager
    def switch_scope_and_registry(self, scope: Optional[str]) -> Generator:
        """Temporarily switch default scope to the target scope, and get the
        corresponding registry.

        If the registry of the corresponding scope exists, yield the
        registry, otherwise yield the current itself.

        Args:
            scope (str, optional): The target scope.

        Examples:
            >>> from mmengine.registry import Registry, DefaultScope, MODELS
            >>> import time
            >>> # External Registry
            >>> MMDET_MODELS = Registry('mmdet_model', scope='mmdet',
            >>>     parent=MODELS)
            >>> MMCLS_MODELS = Registry('mmcls_model', scope='mmcls',
            >>>     parent=MODELS)
            >>> # Local Registry
            >>> CUSTOM_MODELS = Registry('custom_model', scope='custom',
            >>>     parent=MODELS)
            >>>
            >>> # Initiate DefaultScope
            >>> DefaultScope.get_instance(f'scope_{time.time()}',
            >>>     scope_name='custom')
            >>> # Check default scope
            >>> DefaultScope.get_current_instance().scope_name
            custom
            >>> # Switch to mmcls scope and get `MMCLS_MODELS` registry.
            >>> with CUSTOM_MODELS.switch_scope_and_registry(scope='mmcls') as registry:
            >>>     DefaultScope.get_current_instance().scope_name
            mmcls
            >>>     registry.scope
            mmcls
            >>> # Nested switch scope
            >>> with CUSTOM_MODELS.switch_scope_and_registry(scope='mmdet') as mmdet_registry:
            >>>     DefaultScope.get_current_instance().scope_name
            mmdet
            >>>     mmdet_registry.scope
            mmdet
            >>>     with CUSTOM_MODELS.switch_scope_and_registry(scope='mmcls') as mmcls_registry:
            >>>         DefaultScope.get_current_instance().scope_name
            mmcls
            >>>         mmcls_registry.scope
            mmcls
            >>>
            >>> # Check switch back to original scope.
            >>> DefaultScope.get_current_instance().scope_name
            custom
        """  # noqa: E501
        from ..logging import print_log

        # Switch to the given scope temporarily. If the corresponding registry
        # can be found in root registry, return the registry under the scope,
        # otherwise return the registry itself.
        with DefaultScope.overwrite_default_scope(scope):
            # Get the global default scope
            default_scope = DefaultScope.get_current_instance()
            # Get registry by scope
            if default_scope is not None:
                scope_name = default_scope.scope_name
                try:
                    import_module(f'{scope_name}.registry')
                except (ImportError, AttributeError, ModuleNotFoundError):
                    if scope in MODULE2PACKAGE:
                        print_log(
                            f'{scope} is not installed and its '
                            'modules will not be registered. If you '
                            'want to use modules defined in '
                            f'{scope}, Please install {scope} by '
                            f'`pip install {MODULE2PACKAGE[scope]}.',
                            logger='current',
                            level=logging.WARNING)
                    else:
                        print_log(
                            f'Failed to import `{scope}.registry` '
                            f'make sure the registry.py exists in `{scope}` '
                            'package.',
                            logger='current',
                            level=logging.WARNING)
                root = self._get_root_registry()
                registry = root._search_child(scope_name)
                if registry is None:
                    # if `default_scope` can not be found, fallback to argument
                    # `registry`
                    print_log(
                        f'Failed to search registry with scope "{scope_name}" '
                        f'in the "{root.name}" registry tree. '
                        f'As a workaround, the current "{self.name}" registry '
                        f'in "{self.scope}" is used to build instance. This '
                        'may cause unexpected failure when running the built '
                        f'modules. Please check whether "{scope_name}" is a '
                        'correct scope, or whether the registry is '
                        'initialized.',
                        logger='current',
                        level=logging.WARNING)
                    registry = self
            # If there is no built default scope, just return current registry.
            else:
                registry = self
            yield registry

    def _get_root_registry(self) -> 'Registry':
        """Return the root registry."""
        root = self
        while root.parent is not None:
            root = root.parent
        return root

    def import_from_location(self) -> None:
        """import modules from the pre-defined locations in self._location."""
        if not self._imported:
            # Avoid circular import
            from ..logging import print_log

            # avoid BC breaking
            if len(self._locations) == 0 and self.scope in MODULE2PACKAGE:
                print_log(
                    f'The "{self.name}" registry in {self.scope} did not '
                    'set import location. Fallback to call '
                    f'`{self.scope}.utils.register_all_modules` '
                    'instead.',
                    logger='current',
                    level=logging.DEBUG)
                try:
                    module = import_module(f'{self.scope}.utils')
                except (ImportError, AttributeError, ModuleNotFoundError):
                    if self.scope in MODULE2PACKAGE:
                        print_log(
                            f'{self.scope} is not installed and its '
                            'modules will not be registered. If you '
                            'want to use modules defined in '
                            f'{self.scope}, Please install {self.scope} by '
                            f'`pip install {MODULE2PACKAGE[self.scope]}.',
                            logger='current',
                            level=logging.WARNING)
                    else:
                        print_log(
                            f'Failed to import {self.scope} and register '
                            'its modules, please make sure you '
                            'have registered the module manually.',
                            logger='current',
                            level=logging.WARNING)
                else:
                    # The import errors triggered during the registration
                    # may be more complex, here just throwing
                    # the error to avoid causing more implicit registry errors
                    # like `xxx`` not found in `yyy` registry.
                    module.register_all_modules(False)  # type: ignore

            for loc in self._locations:
                import_module(loc)
                print_log(
                    f"Modules of {self.scope}'s {self.name} registry have "
                    f'been automatically imported from {loc}',
                    logger='current',
                    level=logging.DEBUG)
            self._imported = True

    def get(self, key: str) -> Optional[Type]:
        """Get the registry record.

        If `key`` represents the whole object name with its module
        information, for example, `mmengine.model.BaseModel`, ``get``
        will directly return the class object :class:`BaseModel`.

        Otherwise, it will first parse ``key`` and check whether it
        contains a scope name. The logic to search for ``key``:

        - ``key`` does not contain a scope name, i.e., it is purely a module
          name like "ResNet": :meth:`get` will search for ``ResNet`` from the
          current registry to its parent or ancestors until finding it.

        - ``key`` contains a scope name and it is equal to the scope of the
          current registry (e.g., "mmcls"), e.g., "mmcls.ResNet": :meth:`get`
          will only search for ``ResNet`` in the current registry.

        - ``key`` contains a scope name and it is not equal to the scope of
          the current registry (e.g., "mmdet"), e.g., "mmcls.FCNet": If the
          scope exists in its children, :meth:`get` will get "FCNet" from
          them. If not, :meth:`get` will first get the root registry and root
          registry call its own :meth:`get` method.

        Args:
            key (str): Name of the registered item, e.g., the class name in
                string format.

        Returns:
            Type or None: Return the corresponding class if ``key`` exists,
            otherwise return None.

        Examples:
            >>> # define a registry
            >>> MODELS = Registry('models')
            >>> # register `ResNet` to `MODELS`
            >>> @MODELS.register_module()
            >>> class ResNet:
            >>>     pass
            >>> resnet_cls = MODELS.get('ResNet')

            >>> # hierarchical registry
            >>> DETECTORS = Registry('detector', parent=MODELS, scope='det')
            >>> # `ResNet` does not exist in `DETECTORS` but `get` method
            >>> # will try to search from its parents or ancestors
            >>> resnet_cls = DETECTORS.get('ResNet')
            >>> CLASSIFIER = Registry('classifier', parent=MODELS, scope='cls')
            >>> @CLASSIFIER.register_module()
            >>> class MobileNet:
            >>>     pass
            >>> # `get` from its sibling registries
            >>> mobilenet_cls = DETECTORS.get('cls.MobileNet')
        """
        # Avoid circular import
        from ..logging import print_log

        if not isinstance(key, str):
            raise TypeError(
                'The key argument of `Registry.get` must be a str, '
                f'got {type(key)}')

        scope, real_key = self.split_scope_key(key)
        obj_cls = None
        registry_name = self.name
        scope_name = self.scope

        # lazy import the modules to register them into the registry
        self.import_from_location()

        if scope is None or scope == self._scope:
            # get from self
            if real_key in self._module_dict:
                obj_cls = self._module_dict[real_key]
            elif scope is None:
                # try to get the target from its parent or ancestors
                parent = self.parent
                while parent is not None:
                    if real_key in parent._module_dict:
                        obj_cls = parent._module_dict[real_key]
                        registry_name = parent.name
                        scope_name = parent.scope
                        break
                    parent = parent.parent
        else:
            # import the registry to add the nodes into the registry tree
            try:
                import_module(f'{scope}.registry')
                print_log(
                    f'Registry node of {scope} has been automatically '
                    'imported.',
                    logger='current',
                    level=logging.DEBUG)
            except (ImportError, AttributeError, ModuleNotFoundError):
                print_log(
                    f'Cannot auto import {scope}.registry, please check '
                    f'whether the package "{scope}" is installed correctly '
                    'or import the registry manually.',
                    logger='current',
                    level=logging.DEBUG)
            # get from self._children
            if scope in self._children:
                obj_cls = self._children[scope].get(real_key)
                registry_name = self._children[scope].name
                scope_name = scope
            else:
                root = self._get_root_registry()

                if scope != root._scope and scope not in root._children:
                    # If not skip directly, `root.get(key)` will recursively
                    # call itself until RecursionError is thrown.
                    pass
                else:
                    obj_cls = root.get(key)

        if obj_cls is None:
            # Actually, it's strange to implement this `try ... except` to
            # get the object by its name in `Registry.get`. However, If we
            # want to build the model using a configuration like
            # `dict(type='mmengine.model.BaseModel')`, which can
            # be dumped by lazy import config, we need this code snippet
            # for `Registry.get` to work.
            try:
                obj_cls = get_object_from_string(key)
            except Exception:
                raise RuntimeError(f'Failed to get {key}')

        if obj_cls is not None:
            # For some rare cases (e.g. obj_cls is a partial function), obj_cls
            # doesn't have `__name__`. Use default value to prevent error
            cls_name = getattr(obj_cls, '__name__', str(obj_cls))
            print_log(
                f'Get class `{cls_name}` from "{registry_name}"'
                f' registry in "{scope_name}"',
                logger='current',
                level=logging.DEBUG)

        return obj_cls

    def _search_child(self, scope: str) -> Optional['Registry']:
        """Depth-first search for the corresponding registry in its children.

        Note that the method only search for the corresponding registry from
        the current registry. Therefore, if we want to search from the root
        registry, :meth:`_get_root_registry` should be called to get the
        root registry first.

        Args:
            scope (str): The scope name used for searching for its
                corresponding registry.

        Returns:
            Registry or None: Return the corresponding registry if ``scope``
            exists, otherwise return None.
        """
        if self._scope == scope:
            return self

        for child in self._children.values():
            registry = child._search_child(scope)
            if registry is not None:
                return registry

        return None

    def build(self, cfg: dict, *args, **kwargs) -> Any:
        """Build an instance.

        Build an instance by calling :attr:`build_func`.

        Args:
            cfg (dict): Config dict needs to be built.

        Returns:
            Any: The constructed object.

        Examples:
            >>> from mmengine import Registry
            >>> MODELS = Registry('models')
            >>> @MODELS.register_module()
            >>> class ResNet:
            >>>     def __init__(self, depth, stages=4):
            >>>         self.depth = depth
            >>>         self.stages = stages
            >>> cfg = dict(type='ResNet', depth=50)
            >>> model = MODELS.build(cfg)
        """
        return self.build_func(cfg, *args, **kwargs, registry=self)

    def _add_child(self, registry: 'Registry') -> None:
        """Add a child for a registry.

        Args:
            registry (:obj:`Registry`): The ``registry`` will be added as a
                child of the ``self``.
        """

        assert isinstance(registry, Registry)
        assert registry.scope is not None
        assert registry.scope not in self.children, \
            f'scope {registry.scope} exists in {self.name} registry'
        self.children[registry.scope] = registry

    def _register_module(self,
                         module: Type,
                         module_name: Optional[Union[str, List[str]]] = None,
                         force: bool = False) -> None:
        """Register a module.

        Args:
            module (type): Module to be registered. Typically a class or a
                function, but generally all ``Callable`` are acceptable.
            module_name (str or list of str, optional): The module name to be
                registered. If not specified, the class name will be used.
                Defaults to None.
            force (bool): Whether to override an existing class with the same
                name. Defaults to False.
        """
        if not callable(module):
            raise TypeError(f'module must be Callable, but got {type(module)}')

        if module_name is None:
            module_name = module.__name__
        if isinstance(module_name, str):
            module_name = [module_name]
        for name in module_name:
            if not force and name in self._module_dict:
                existed_module = self.module_dict[name]
                raise KeyError(f'{name} is already registered in {self.name} '
                               f'at {existed_module.__module__}')
            self._module_dict[name] = module

    def register_module(
            self,
            name: Optional[Union[str, List[str]]] = None,
            force: bool = False,
            module: Optional[Type] = None) -> Union[type, Callable]:
        """Register a module.

        A record will be added to ``self._module_dict``, whose key is the class
        name or the specified name, and value is the class itself.
        It can be used as a decorator or a normal function.

        Args:
            name (str or list of str, optional): The module name to be
                registered. If not specified, the class name will be used.
            force (bool): Whether to override an existing class with the same
                name. Defaults to False.
            module (type, optional): Module class or function to be registered.
                Defaults to None.

        Examples:
            >>> backbones = Registry('backbone')
            >>> # as a decorator
            >>> @backbones.register_module()
            >>> class ResNet:
            >>>     pass
            >>> backbones = Registry('backbone')
            >>> @backbones.register_module(name='mnet')
            >>> class MobileNet:
            >>>     pass

            >>> # as a normal function
            >>> class ResNet:
            >>>     pass
            >>> backbones.register_module(module=ResNet)
        """
        if not isinstance(force, bool):
            raise TypeError(f'force must be a boolean, but got {type(force)}')

        # raise the error ahead of time
        if not (name is None or isinstance(name, str) or is_seq_of(name, str)):
            raise TypeError(
                'name must be None, an instance of str, or a sequence of str, '
                f'but got {type(name)}')

        # use it as a normal method: x.register_module(module=SomeClass)
        if module is not None:
            self._register_module(module=module, module_name=name, force=force)
            return module

        # use it as a decorator: @x.register_module()
        def _register(module):
            self._register_module(module=module, module_name=name, force=force)
            return module

        return _register

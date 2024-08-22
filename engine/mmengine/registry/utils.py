# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import datetime
import logging
import os.path as osp
from typing import Optional

from mmengine.fileio import dump
from mmengine.logging import print_log
from . import root
from .default_scope import DefaultScope
from .registry import Registry


def traverse_registry_tree(registry: Registry, verbose: bool = True) -> list:
    """Traverse the whole registry tree from any given node, and collect
    information of all registered modules in this registry tree.

    Args:
        registry (Registry): a registry node in the registry tree.
        verbose (bool): Whether to print log. Defaults to True

    Returns:
        list: Statistic results of all modules in each node of the registry
        tree.
    """
    root_registry = registry.root
    modules_info = []

    def _dfs_registry(_registry):
        if isinstance(_registry, Registry):
            num_modules = len(_registry.module_dict)
            scope = _registry.scope
            registry_info = dict(num_modules=num_modules, scope=scope)
            for name, registered_class in _registry.module_dict.items():
                folder = '/'.join(registered_class.__module__.split('.')[:-1])
                if folder in registry_info:
                    registry_info[folder].append(name)
                else:
                    registry_info[folder] = [name]
            if verbose:
                print_log(
                    f"Find {num_modules} modules in {scope}'s "
                    f"'{_registry.name}' registry ",
                    logger='current')
            modules_info.append(registry_info)
        else:
            return
        for _, child in _registry.children.items():
            _dfs_registry(child)

    _dfs_registry(root_registry)
    return modules_info


def count_registered_modules(save_path: Optional[str] = None,
                             verbose: bool = True) -> dict:
    """Scan all modules in MMEngine's root and child registries and dump to
    json.

    Args:
        save_path (str, optional): Path to save the json file.
        verbose (bool): Whether to print log. Defaults to True.

    Returns:
        dict: Statistic results of all registered modules.
    """
    # import modules to trigger registering
    import mmengine.dataset
    import mmengine.evaluator
    import mmengine.hooks
    import mmengine.model
    import mmengine.optim
    import mmengine.runner
    import mmengine.visualization  # noqa: F401

    registries_info = {}
    # traverse all registries in MMEngine
    for item in dir(root):
        if not item.startswith('__'):
            registry = getattr(root, item)
            if isinstance(registry, Registry):
                registries_info[item] = traverse_registry_tree(
                    registry, verbose)
    scan_data = dict(
        scan_date=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        registries=registries_info)
    if verbose:
        print_log(
            f'Finish registry analysis, got: {scan_data}', logger='current')
    if save_path is not None:
        json_path = osp.join(save_path, 'modules_statistic_results.json')
        dump(scan_data, json_path, indent=2)
        print_log(f'Result has been saved to {json_path}', logger='current')
    return scan_data


def init_default_scope(scope: str) -> None:
    """Initialize the given default scope.

    Args:
        scope (str): The name of the default scope.
    """
    never_created = DefaultScope.get_current_instance(
    ) is None or not DefaultScope.check_instance_created(scope)
    if never_created:
        DefaultScope.get_instance(scope, scope_name=scope)
        return
    current_scope = DefaultScope.get_current_instance()  # type: ignore
    if current_scope.scope_name != scope:  # type: ignore
        print_log(
            'The current default scope '  # type: ignore
            f'"{current_scope.scope_name}" is not "{scope}", '
            '`init_default_scope` will force set the current'
            f'default scope to "{scope}".',
            logger='current',
            level=logging.WARNING)
        # avoid name conflict
        new_instance_name = f'{scope}-{datetime.datetime.now()}'
        DefaultScope.get_instance(new_instance_name, scope_name=scope)

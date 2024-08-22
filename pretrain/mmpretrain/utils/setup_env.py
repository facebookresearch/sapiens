# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import datetime
import warnings

from mmengine import DefaultScope


def register_all_modules(init_default_scope: bool = True) -> None:
    """Register all modules in mmpretrain into the registries.

    Args:
        init_default_scope (bool): Whether initialize the mmpretrain default
            scope. If True, the global default scope will be set to
            `mmpretrain`, and all registries will build modules from
            mmpretrain's registry node. To understand more about the registry,
            please refer to
            https://github.com/open-mmlab/mmengine/blob/main/docs/en/tutorials/registry.md
            Defaults to True.
    """  # noqa: E501
    import mmpretrain.datasets  # noqa: F401,F403
    import mmpretrain.engine  # noqa: F401,F403
    import mmpretrain.evaluation  # noqa: F401,F403
    import mmpretrain.models  # noqa: F401,F403
    import mmpretrain.structures  # noqa: F401,F403
    import mmpretrain.visualization  # noqa: F401,F403

    if not init_default_scope:
        return

    current_scope = DefaultScope.get_current_instance()
    if current_scope is None:
        DefaultScope.get_instance('mmpretrain', scope_name='mmpretrain')
    elif current_scope.scope_name != 'mmpretrain':
        warnings.warn(
            f'The current default scope "{current_scope.scope_name}" '
            'is not "mmpretrain", `register_all_modules` will force '
            'the current default scope to be "mmpretrain". If this is '
            'not expected, please set `init_default_scope=False`.')
        # avoid name conflict
        new_instance_name = f'mmpretrain-{datetime.datetime.now()}'
        DefaultScope.get_instance(new_instance_name, scope_name='mmpretrain')

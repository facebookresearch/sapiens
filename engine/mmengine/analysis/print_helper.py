# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Modified from
# https://github.com/facebookresearch/fvcore/blob/main/fvcore/nn/print_model_statistics.py

from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union

import torch
from rich import box
from rich.console import Console
from rich.table import Table
from torch import nn

from mmengine.utils import is_tuple_of
from .complexity_analysis import (ActivationAnalyzer, FlopAnalyzer,
                                  parameter_count)


def _format_size(x: int, sig_figs: int = 3, hide_zero: bool = False) -> str:
    """Formats an integer for printing in a table or model representation.

    Expresses the number in terms of 'kilo', 'mega', etc., using
    'K', 'M', etc. as a suffix.

    Args:
        x (int): The integer to format.
        sig_figs (int): The number of significant figures to keep.
            Defaults to 3.
        hide_zero (bool): If True, x=0 is replaced with an empty string
            instead of '0'. Defaults to False.

    Returns:
        str: The formatted string.
    """
    if hide_zero and x == 0:
        return ''

    def fmt(x: float) -> str:
        # use fixed point to avoid scientific notation
        return f'{{:.{sig_figs}f}}'.format(x).rstrip('0').rstrip('.')

    if abs(x) > 1e14:
        return fmt(x / 1e15) + 'P'
    if abs(x) > 1e11:
        return fmt(x / 1e12) + 'T'
    if abs(x) > 1e8:
        return fmt(x / 1e9) + 'G'
    if abs(x) > 1e5:
        return fmt(x / 1e6) + 'M'
    if abs(x) > 1e2:
        return fmt(x / 1e3) + 'K'
    return str(x)


def _pretty_statistics(statistics: Dict[str, Dict[str, int]],
                       sig_figs: int = 3,
                       hide_zero: bool = False) -> Dict[str, Dict[str, str]]:
    """Converts numeric statistics to strings with kilo/mega/giga/etc. labels.

    Args:
        statistics (dict[str, dict[str, int]]) : the statistics to
            format. Organized as a dictionary over modules, which are
            each a dictionary over statistic types.
        sig_figs (int): the number of significant figures for each stat.
            Defaults to 3.
        hide_zero (bool): if True, statistics that are zero will be
            written as an empty string. Defaults to False.

    Returns:
        dict[str, dict[str, str]]: the input statistics as pretty strings
    """
    out_stats = {}
    for mod, stats in statistics.items():
        out_stats[mod] = {
            s: _format_size(val, sig_figs, hide_zero)
            for s, val in stats.items()
        }
    return out_stats


def _group_by_module(
        statistics: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Converts statistics organized first by statistic type and then by module
    to statistics organized first by module and then by statistic type.

    Args:
        statistics (dict[str, dict[str, any]]): the statistics to convert

    Returns:
        dict[str, dict[str, any]]: the reorganized statistics
    """
    out_stats = defaultdict(dict)  # type: Dict[str, Dict[str, Any]]
    for stat_name, stat in statistics.items():
        for mod, val in stat.items():
            out_stats[mod][stat_name] = val
    return dict(out_stats)


def _indicate_uncalled_modules(
    statistics: Dict[str, Dict[str, str]],
    stat_name: str,
    uncalled_modules: Set[str],
    uncalled_indicator: str = 'N/A',
) -> Dict[str, Dict[str, str]]:
    """If a module is in the set of uncalled modules, replace its statistics
    with the specified indicator, instead of using the existing string.

    Assumes the statistic is already formatting in string form.

    Args:
        statistics (dict[str, dict[str, str]]): the statistics to
            format. Organized as a dictionary over modules, which are
            each a dictionary over statistic types. Expects statistics
            have already been converted to strings.
        stat_name (str): the name of the statistic being modified
        uncalled_modules set(str): a set of names of uncalled modules.
        indicator (str): the string that will be used to indicate
            unused modules. Defaults to 'N/A'.

    Returns:
        dict[str, dict[str, str]]: the modified statistics
    """

    stats_out = {mod: stats.copy() for mod, stats in statistics.items()}
    for mod in uncalled_modules:
        if mod not in stats_out:
            stats_out[mod] = {}
        stats_out[mod][stat_name] = uncalled_indicator
    return stats_out


def _remove_zero_statistics(
    statistics: Dict[str, Dict[str, int]],
    force_keep: Optional[Set[str]] = None,
    require_trivial_children: bool = False,
) -> Dict[str, Dict[str, int]]:
    """Any module that has zero for all available statistics is removed from
    the set of statistics.

    This can help declutter the reporting of statistics
    if many submodules have zero statistics. Assumes the statistics have
    a model hierarchy starting with a root that has name ''.

    Args:
        statistics (dict[str, dict[str, int]]): the statistics to
            remove zeros from. Organized as a dictionary over modules,
            which are each a dictionary over statistic types.
        force_keep (set[str] or None): a set of modules to always keep, even
            if they are all zero.
        require_trivial_children (bool): If True, a statistic will only
            be deleted if all its children are also deleted. Defaults to
            False.

    Returns:
        dict[str, dict[str, int]]: the input statistics dictionary,
        with submodules removed if they have zero for all statistics.
    """
    out_stats: Dict[str, Dict[str, int]] = {}
    _force_keep: Set[str] = force_keep if force_keep else set() | {''}

    def keep_stat(name: str) -> None:
        prefix = name + ('.' if name else '')
        trivial_children = True
        for mod in statistics:
            # 'if mod' excludes root = '', which is never a child
            if mod and mod.count('.') == prefix.count('.') and mod.startswith(
                    prefix):
                keep_stat(mod)
                trivial_children &= mod not in out_stats

        if ((not all(val == 0 for val in statistics[name].values()))
                or (name in _force_keep)
                or (require_trivial_children and not trivial_children)):
            out_stats[name] = statistics[name].copy()

    keep_stat('')
    return out_stats


def _fill_missing_statistics(
        model: nn.Module,
        statistics: Dict[str, Dict[str, int]]) -> Dict[str, Dict[str, int]]:
    """If, for a given submodule name in the model, a statistic is missing from
    statistics, fills it in with zero.

    This visually uniformizes the reporting of statistics.

    Args:
        model (nn.Module): the model whose submodule names will be
            used to fill in statistics
        statistics (dict[str, dict[str, int]]) : the statistics to
            fill in missing values for. Organized as a dictionary
            over statistics, which are each a dictionary over submodules'
            names. The statistics are assumed to be formatted already
            to the desired string format for printing.

    Returns:
        dict[str, dict[str, int]]: the input statistics with missing
        values filled with zero.
    """
    out_stats = {name: stat.copy() for name, stat in statistics.items()}
    for mod_name, _ in model.named_modules():
        for stat in out_stats.values():
            if mod_name not in stat:
                stat[mod_name] = 0
    return out_stats


def _model_stats_str(model: nn.Module,
                     statistics: Dict[str, Dict[str, str]]) -> str:
    """This produces a representation of the model much like 'str(model)'
    would, except the provided statistics are written out as additional
    information for each submodule.

    Args:
        model (nn.Module): the model to form a representation of.
        statistics (dict[str, dict[str, str]]): the statistics to
            include in the model representations. Organized as a dictionary
            over module names, which are each a dictionary over statistics.
            The statistics are assumed to be formatted already to the
            desired string format for printing.

    Returns:
        str: the string representation of the model with the statistics
        inserted.
    """

    # Copied from nn.Module._addindent
    def _addindent(s_: str, numSpaces: int) -> str:
        s = s_.split('\n')
        # don't do anything for single-line stuff
        if len(s) == 1:
            return s_
        first = s.pop(0)
        s = [(numSpaces * ' ') + line for line in s]
        s = '\n'.join(s)  # type: ignore
        s = first + '\n' + s  # type: ignore
        return s  # type: ignore

    def print_statistics(name: str) -> str:
        if name not in statistics:
            return ''
        printed_stats = [f'{k}: {v}' for k, v in statistics[name].items()]
        return ', '.join(printed_stats)

    # This comes directly from nn.Module.__repr__ with small changes
    # to include the statistics.
    def repr_with_statistics(module: nn.Module, name: str) -> str:
        # We treat the extra repr like the sub-module, one item per line
        extra_lines = []
        extra_repr = module.extra_repr()
        printed_stats = print_statistics(name)
        # empty string will be split into list ['']
        if extra_repr:
            extra_lines.extend(extra_repr.split('\n'))
        if printed_stats:
            extra_lines.extend(printed_stats.split('\n'))
        child_lines = []
        for key, submod in module._modules.items():
            submod_name = name + ('.' if name else '') + key
            # pyre-fixme[6]: Expected `Module` for 1st param but got
            #  `Optional[nn.modules.module.Module]`.
            submod_str = repr_with_statistics(submod, submod_name)
            submod_str = _addindent(submod_str, 2)
            child_lines.append('(' + key + '): ' + submod_str)
        lines = extra_lines + child_lines

        main_str = module._get_name() + '('
        if lines:
            # simple one-liner info, which most builtin Modules will use
            if len(extra_lines) == 1 and not child_lines:
                main_str += extra_lines[0]
            else:
                main_str += '\n  ' + '\n  '.join(lines) + '\n'

        main_str += ')'
        return main_str

    return repr_with_statistics(model, '')


def _get_input_sizes(iterable: Iterable[Any]) -> List[Any]:  # type: ignore
    """Gets the sizes of all torch tensors in an iterable.

    If an element of the iterable is a non-torch tensor iterable, it recurses
    into that iterable to continue calculating sizes. Any non-iterable is given
    a size of None. The output consists of nested lists with the same nesting
    structure as the input iterables.
    """
    out_list = []
    for i in iterable:
        if isinstance(i, torch.Tensor):
            out_list.append(list(i.size()))
        elif isinstance(i, Iterable):
            sublist_sizes = _get_input_sizes(i)
            if all(j is None for j in sublist_sizes):
                out_list.append(None)  # type: ignore
            else:
                out_list.append(sublist_sizes)
        else:
            out_list.append(None)  # type: ignore
    return out_list


def _get_single_child(name: str,
                      statistics: Dict[str, Dict[str, str]]) -> Optional[str]:
    """If the given module has only a single child in statistics, return it.

    Otherwise, return None.
    """
    prefix = name + ('.' if name else '')
    child = None
    for mod in statistics:
        # 'if mod' excludes root = '', which is never a child
        if mod and mod.count('.') == prefix.count('.') and mod.startswith(
                prefix):
            if child is None:
                child = mod
            else:
                return None  # We found a second child, so return None
    return child


def _try_combine(stats1: Dict[str, str],
                 stats2: Dict[str, str]) -> Optional[Dict[str, str]]:
    """Try combine two statistics dict to display in one row.

    If they conflict, returns None.
    """
    ret = {}
    if set(stats1.keys()) != set(stats2.keys()):
        return None
    for k, v1 in stats1.items():
        v2 = stats2[k]
        if v1 != v2 and len(v1) and len(v2):
            return None
        ret[k] = v1 if len(v1) else v2
    return ret


def _fastforward(
        name: str,
        statistics: Dict[str, Dict[str, str]]) -> Tuple[str, Dict[str, str]]:
    """If the given module has only a single child and matches statistics with
    that child, merge statistics and their names into one row.

    Then repeat until the condition isn't met.

    Returns:
        tuple[str, dict]: the new name and the combined statistics of this row
    """
    single_child = _get_single_child(name, statistics)
    if single_child is None:
        return name, statistics[name]
    combined = _try_combine(statistics[name], statistics[single_child])
    if combined is None:
        return name, statistics[name]
    statistics[single_child] = combined
    return _fastforward(single_child, statistics)


def _stats_table_format(
    statistics: Dict[str, Dict[str, str]],
    max_depth: int = 3,
    stat_columns: Optional[List[str]] = None,
) -> str:
    """Formats the statistics obtained from a model in a nice table.

    Args:
        statistics (dict[str, dict[str, str]]): The statistics to print.
            Organized as a dictionary over modules, then as a dictionary
            over statistics in the model. The statistics are assumed to
            already be formatted for printing.
        max_depth (int): The maximum submodule depth to recurse to.
            Defaults to 3.
        stat_columns (list[str]): Specify the order of the columns to print.
            If None, columns are found automatically from the provided
            statistics. Defaults to None.

    Return:
        str: The formatted table.
    """
    if stat_columns is None:
        stat_columns = set()  # type: ignore
        for stats in statistics.values():
            stat_columns.update(stats.keys())  # type: ignore
        stat_columns = list(stat_columns)  # type: ignore

    headers = ['module'] + stat_columns
    rows: List[List[str]] = []

    def build_row(name: str, stats: Dict[str, str],
                  indent_lvl: int) -> List[str]:
        indent = ' ' * indent_lvl
        row = [indent + name]
        for stat_name in stat_columns:  # type: ignore
            row_str = (indent + stats[stat_name]) if stat_name in stats else ''
            row.append(row_str)
        return row

    def fill(indent_lvl: int, prefix: str) -> None:
        if indent_lvl > max_depth:
            return
        for mod_name in statistics:
            # 'if mod' excludes root = '', which is never a child
            if (mod_name and mod_name.count('.') == prefix.count('.')
                    and mod_name.startswith(prefix)):
                mod_name, curr_stats = _fastforward(mod_name, statistics)
                if root_prefix and mod_name.startswith(root_prefix):
                    # Skip the root_prefix shared by all submodules as it
                    # carries 0 information
                    pretty_mod_name = mod_name[len(root_prefix):]
                else:
                    pretty_mod_name = mod_name
                row = build_row(pretty_mod_name, curr_stats, indent_lvl)
                rows.append(row)
                fill(indent_lvl + 1, mod_name + '.')

    root_name, curr_stats = _fastforward('', statistics)
    row = build_row(root_name or 'model', curr_stats, indent_lvl=0)
    rows.append(row)
    root_prefix = root_name + ('.' if root_name else '')
    fill(indent_lvl=1, prefix=root_prefix)

    table = Table(box=box.ASCII2)
    for header in headers:
        table.add_column(header)

    for row in rows:
        table.add_row(*row)

    console = Console()
    with console.capture() as capture:
        console.print(table, end='')

    return capture.get()


def complexity_stats_str(
        flops: FlopAnalyzer,
        activations: Optional[ActivationAnalyzer] = None) -> str:
    """Calculates the parameters and flops of the model with the given inputs
    and returns a string representation of the model that includes the
    parameters and flops of every submodule. The string is structured to be
    similar that given by str(model), though it is not guaranteed to be
    identical in form if the default string representation of a module has been
    overridden. If a module has zero parameters and flops, statistics will not
    be reported for succinctness. The trace can only register the scope of a
    module if it is called directly, which means flops (and activations)
    arising from explicit calls to .forward() or to other python functions of
    the module will not be attributed to that module. Modules that are never
    called will have 'N/A' listed for their flops; this means they are either
    unused or their statistics are missing for this reason. Any such flops are
    still counted towards the parent.

    Examples:
        >>> import torch
        >>> import torch.nn as nn
        >>> class InnerNet(nn.Module):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.fc1 = nn.Linear(10,10)
        ...         self.fc2 = nn.Linear(10,10)
        ...     def forward(self, x):
        ...         return self.fc1(self.fc2(x))
        >>> class TestNet(nn.Module):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.fc1 = nn.Linear(10,10)
        ...         self.fc2 = nn.Linear(10,10)
        ...         self.inner = InnerNet()
        ...     def forward(self, x):
        ...         return self.fc1(self.fc2(self.inner(x)))
        >>> inputs = torch.randn((1,10))
        >>> print(complexity_stats_str(FlopAnalyzer(model, inputs)))
        TestNet(
          #params: 0.44K, #flops: 0.4K
          (fc1): Linear(
            in_features=10, out_features=10, bias=True
            #params: 0.11K, #flops: 100
          )
          (fc2): Linear(
            in_features=10, out_features=10, bias=True
            #params: 0.11K, #flops: 100
          )
          (inner): InnerNet(
            #params: 0.22K, #flops: 0.2K
            (fc1): Linear(
              in_features=10, out_features=10, bias=True
              #params: 0.11K, #flops: 100
            )
            (fc2): Linear(
              in_features=10, out_features=10, bias=True
              #params: 0.11K, #flops: 100
            )
          )
        )

    Args:
        flops (FlopAnalyzer): the flop counting object
        activations (ActivationAnalyzer or None): If given, the activations of
            each layer will also be calculated and included in the
            representation. Defaults to None.

    Returns:
        str: a string representation of the model with the number of
        parameters and flops included.
    """
    # cast to dict since pyre doesn't like the implicit defaultdict->dict
    model = flops._model
    params = dict(parameter_count(model))

    flops.unsupported_ops_warnings(False)
    flops.uncalled_modules_warnings(False)
    flops.tracer_warnings('none')
    stats = {'#params': params, '#flops': flops.by_module()}

    if activations is not None:
        activations.unsupported_ops_warnings(False)
        activations.uncalled_modules_warnings(False)
        activations.tracer_warnings('none')
        stats['#acts'] = activations.by_module()

    all_uncalled = flops.uncalled_modules() | (
        activations.uncalled_modules() if activations is not None else set())
    stats = _fill_missing_statistics(model, stats)
    stats = _group_by_module(stats)
    stats = _remove_zero_statistics(stats, force_keep=all_uncalled)
    stats = _pretty_statistics(stats, sig_figs=2)  # type: ignore
    stats = _indicate_uncalled_modules(  # type: ignore
        stats,  # type: ignore
        '#flops',  # type: ignore
        flops.uncalled_modules())  # type: ignore
    if activations is not None:
        stats = _indicate_uncalled_modules(  # type: ignore
            stats,  # type: ignore
            '#acts',  # type: ignore
            activations.uncalled_modules())  # type: ignore

    model_string = ''
    if all_uncalled:
        model_string += (
            'N/A indicates a possibly missing statistic due to how '
            'the module was called. Missing values are still included '
            "in the parent's total.\n")
    model_string += _model_stats_str(model, stats)  # type: ignore
    return model_string


def complexity_stats_table(
    flops: FlopAnalyzer,
    max_depth: int = 3,
    activations: Optional[ActivationAnalyzer] = None,
    show_param_shapes: bool = True,
) -> str:
    """
    Format the per-module parameters and flops of a model in a table.
    It looks like this:
    ::
        | model                            | #parameters or shape| #flops    |
        |:---------------------------------|:--------------------|:----------|
        | model                            | 34.6M               | 65.7G     |
        |  s1                              |  15.4K              |  4.32G    |
        |   s1.pathway0_stem               |   9.54K             |   1.23G   |
        |    s1.pathway0_stem.conv         |    9.41K            |    1.23G  |
        |    s1.pathway0_stem.bn           |    0.128K           |           |
        |   s1.pathway1_stem               |   5.9K              |   3.08G   |
        |    s1.pathway1_stem.conv         |    5.88K            |    3.08G  |
        |    s1.pathway1_stem.bn           |    16               |           |
        |  s1_fuse                         |  0.928K             |  29.4M    |
        |   s1_fuse.conv_f2s               |   0.896K            |   29.4M   |
        |    s1_fuse.conv_f2s.weight       |    (16, 8, 7, 1, 1) |           |
        |   s1_fuse.bn                     |   32                |           |
        |    s1_fuse.bn.weight             |    (16,)            |           |
        |    s1_fuse.bn.bias               |    (16,)            |           |
        |  s2                              |  0.226M             |  7.73G    |
        |   s2.pathway0_res0               |   80.1K             |   2.58G   |
        |    s2.pathway0_res0.branch1      |    20.5K            |    0.671G |
        |    s2.pathway0_res0.branch1_bn   |    0.512K           |           |
        |    s2.pathway0_res0.branch2      |    59.1K            |    1.91G  |
        |   s2.pathway0_res1.branch2       |   70.4K             |   2.28G   |
        |    s2.pathway0_res1.branch2.a    |    16.4K            |    0.537G |
        |    s2.pathway0_res1.branch2.a_bn |    0.128K           |           |
        |    s2.pathway0_res1.branch2.b    |    36.9K            |    1.21G  |
        |    s2.pathway0_res1.branch2.b_bn |    0.128K           |           |
        |    s2.pathway0_res1.branch2.c    |    16.4K            |    0.537G |
        |    s2.pathway0_res1.branch2.c_bn |    0.512K           |           |
        |   s2.pathway0_res2.branch2       |   70.4K             |   2.28G   |
        |    s2.pathway0_res2.branch2.a    |    16.4K            |    0.537G |
        |    s2.pathway0_res2.branch2.a_bn |    0.128K           |           |
        |    s2.pathway0_res2.branch2.b    |    36.9K            |    1.21G  |
        |    s2.pathway0_res2.branch2.b_bn |    0.128K           |           |
        |    s2.pathway0_res2.branch2.c    |    16.4K            |    0.537G |
        |    s2.pathway0_res2.branch2.c_bn |    0.512K           |           |
        |    ............................. |    ......           |    ...... |

    Args:
        flops (FlopAnalyzer): the flop counting object
        max_depth (int): The max depth of submodules to include in the
            table. Defaults to 3.
        activations (ActivationAnalyzer or None): If given, include
            activation counts as an additional column in the table.
            Defaults to None.
        show_param_shapes (bool): If true, shapes for parameters will be
            included in the table. Defaults to True.

    Returns:
        str: The formatted table.

    Examples:
        >>> print(complexity_stats_table(FlopAnalyzer(model, inputs)))
    """
    params_header = '#parameters' + (' or shape' if show_param_shapes else '')
    flops_header, acts_header = '#flops', '#activations'

    model = flops._model
    # cast to dict since pyre doesn't like the implicit defaultdict->dict
    params = dict(parameter_count(model))

    flops.unsupported_ops_warnings(False)
    flops.uncalled_modules_warnings(False)
    flops.tracer_warnings('none')

    stats = {params_header: params, flops_header: flops.by_module()}
    stat_columns = [params_header, flops_header]

    if activations is not None:
        activations.unsupported_ops_warnings(False)
        activations.uncalled_modules_warnings(False)
        activations.tracer_warnings('none')
        stats[acts_header] = activations.by_module()
        stat_columns += [acts_header]

    stats = _group_by_module(stats)
    stats = _remove_zero_statistics(
        stats,  # type: ignore
        require_trivial_children=True)  # type: ignore
    stats = _pretty_statistics(stats, hide_zero=False)  # type: ignore
    stats = _indicate_uncalled_modules(  # type: ignore
        stats,  # type: ignore
        flops_header,  # type: ignore
        flops.uncalled_modules() & stats.keys(),  # type: ignore
        uncalled_indicator='',  # type: ignore
    )
    if activations:
        stats = _indicate_uncalled_modules(  # type: ignore
            stats,  # type: ignore
            acts_header,  # type: ignore
            activations.uncalled_modules() & stats.keys(),  # type: ignore
            uncalled_indicator='',  # type: ignore
        )

    # Swap in shapes for parameters or delete shapes from dict
    param_shapes: Dict[str, Tuple[int, ...]] = {
        k: tuple(v.shape)
        for k, v in model.named_parameters()
    }
    to_delete = []
    for mod in stats:
        if mod in param_shapes:
            if show_param_shapes:
                stats[mod][params_header] = str(  # type: ignore
                    param_shapes[mod])  # type: ignore
            else:
                to_delete.append(mod)
    for mod in to_delete:
        del stats[mod]

    return _stats_table_format(
        statistics=stats,  # type: ignore
        max_depth=max_depth,
        stat_columns=stat_columns,
    )


def get_model_complexity_info(
    model: nn.Module,
    input_shape: Union[Tuple[int, ...], Tuple[Tuple[int, ...], ...],
                       None] = None,
    inputs: Union[torch.Tensor, Tuple[torch.Tensor, ...], Tuple[Any, ...],
                  None] = None,
    show_table: bool = True,
    show_arch: bool = True,
):
    """Interface to get the complexity of a model.

    The parameter `inputs` are fed to the forward method of model.
    If `inputs` is not specified, the `input_shape` is required and
    it will be used to construct the dummy input fed to model.
    If the forward of model requires two or more inputs, the `inputs`
    should be a tuple of tensor or the `input_shape` should be a tuple
    of tuple which each element will be constructed into a dumpy input.

    Examples:
        >>> # the forward of model accepts only one input
        >>> input_shape = (3, 224, 224)
        >>> get_model_complexity_info(model, input_shape=input_shape)
        >>> # the forward of model accepts two or more inputs
        >>> input_shape = ((3, 224, 224), (3, 10))
        >>> get_model_complexity_info(model, input_shape=input_shape)

    Args:
        model (nn.Module): The model to analyze.
        input_shape (Union[Tuple[int, ...], Tuple[Tuple[int, ...]], None]):
            The input shape of the model.
            If "inputs" is not specified, the "input_shape" should be set.
            Defaults to None.
        inputs (torch.Tensor, tuple[torch.Tensor, ...] or Tuple[Any, ...],\
            optional]):
            The input tensor(s) of the model. If not given the input tensor
            will be generated automatically with the given input_shape.
            Defaults to None.
        show_table (bool): Whether to show the complexity table.
            Defaults to True.
        show_arch (bool): Whether to show the complexity arch.
            Defaults to True.

    Returns:
        dict: The complexity information of the model.
    """
    if input_shape is None and inputs is None:
        raise ValueError('One of "input_shape" and "inputs" should be set.')
    elif input_shape is not None and inputs is not None:
        raise ValueError('"input_shape" and "inputs" cannot be both set.')

    if inputs is None:
        device = next(model.parameters()).device
        if is_tuple_of(input_shape, int):  # tuple of int, construct one tensor
            inputs = (torch.randn(1, *input_shape).to(device), )
        elif is_tuple_of(input_shape, tuple) and all([
                is_tuple_of(one_input_shape, int)
                for one_input_shape in input_shape  # type: ignore
        ]):  # tuple of tuple of int, construct multiple tensors
            inputs = tuple([
                torch.randn(1, *one_input_shape).to(device)
                for one_input_shape in input_shape  # type: ignore
            ])
        else:
            raise ValueError(
                '"input_shape" should be either a `tuple of int` (to construct'
                'one input tensor) or a `tuple of tuple of int` (to construct'
                'multiple input tensors).')

    flop_handler = FlopAnalyzer(model, inputs)
    activation_handler = ActivationAnalyzer(model, inputs)

    flops = flop_handler.total()
    activations = activation_handler.total()
    params = parameter_count(model)['']

    flops_str = _format_size(flops)
    activations_str = _format_size(activations)
    params_str = _format_size(params)

    if show_table:
        complexity_table = complexity_stats_table(
            flops=flop_handler,
            activations=activation_handler,
            show_param_shapes=True,
        )
        complexity_table = '\n' + complexity_table
    else:
        complexity_table = ''

    if show_arch:
        complexity_arch = complexity_stats_str(
            flops=flop_handler,
            activations=activation_handler,
        )
        complexity_arch = '\n' + complexity_arch
    else:
        complexity_arch = ''

    return {
        'flops': flops,
        'flops_str': flops_str,
        'activations': activations,
        'activations_str': activations_str,
        'params': params,
        'params_str': params_str,
        'out_table': complexity_table,
        'out_arch': complexity_arch
    }

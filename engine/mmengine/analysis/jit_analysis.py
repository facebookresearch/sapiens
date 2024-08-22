# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Modified from
# https://github.com/facebookresearch/fvcore/blob/main/fvcore/nn/jit_analysis.py

import logging
import typing
import warnings
from collections import Counter
from copy import copy
from dataclasses import dataclass
from numbers import Number
from typing import (Any, Dict, Iterable, Iterator, List, Optional, Set, Tuple,
                    TypeVar, Union)

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.jit import TracerWarning, _get_trace_graph

from mmengine.logging import print_log
from .jit_handles import Handle

T = TypeVar('T', bound='JitModelAnalysis')

# Only ignore ops that are technically truly 0 flops:
# shape-manipulation ops, integer ops, memory copy ops
_IGNORED_OPS: Set[str] = {
    'aten::Int',
    'aten::ScalarImplicit',
    'aten::__and__',
    'aten::arange',
    'aten::bitwise_not',
    'aten::cat',
    'aten::chunk',
    'aten::clamp',
    'aten::clamp_',
    'aten::constant_pad_nd',
    'aten::contiguous',
    'aten::copy_',
    'aten::detach',
    'aten::dropout',
    'aten::empty',
    'aten::eq',
    'aten::expand',
    'aten::flatten',
    'aten::floor',
    'aten::floor_divide',
    'aten::full',
    'aten::full_like',
    'aten::gather',
    'aten::ge',
    'aten::gt',
    'aten::index',
    'aten::index_put_',
    'aten::masked_fill',
    'aten::max',
    'aten::narrow',
    'aten::new_empty',
    'aten::new_full',
    'aten::new_zeros',
    'aten::nonzero',
    'aten::ones',
    'aten::permute',
    'aten::relu',
    'aten::relu_',
    'aten::remainder',
    'aten::reshape',
    'aten::roll',
    'aten::select',
    'aten::size',
    'aten::slice',
    'aten::split',
    'aten::split_with_sizes',
    'aten::squeeze',
    'aten::stack',
    'aten::t',
    'aten::to',
    'aten::transpose',
    'aten::type_as',
    'aten::unbind',
    'aten::unsqueeze',
    'aten::unsqueeze_',
    'aten::view',
    'aten::zeros',
    'aten::zeros_like',
}


@dataclass
class Statistics:
    """For keeping track of the various model statistics recorded during
    analysis."""

    counts: Dict[str, typing.Counter[str]]
    unsupported_ops: Dict[str, typing.Counter[str]]
    uncalled_mods: Set[str]


def _named_modules_with_dup(model: nn.Module,
                            prefix: str = ''
                            ) -> Iterable[Tuple[str, nn.Module]]:
    """The same as `model.named_modules()`, except that it includes duplicated
    modules that have more than one name."""
    yield prefix, model
    for name, module in model._modules.items():
        if module is None:
            continue
        submodule_prefix = prefix + ('.' if prefix else '') + name
        yield from _named_modules_with_dup(module, submodule_prefix)


def _named_modules_without_dup(
        model: nn.Module) -> Iterator[Tuple[str, nn.Module]]:
    """Like .named_modules(), but the results are slightly different for some
    wrapped models."""
    seen = set()
    for name, mod in _named_modules_with_dup(model):
        if mod not in seen:
            seen.add(mod)
            yield name, mod


def _get_scoped_trace_graph(
    module: nn.Module,
    inputs: Union[Tensor, Tuple[Tensor, ...]],
    aliases: Dict[Union[str, nn.Module], str],
) -> torch._C.Graph:
    """Traces the provided module using torch.jit._get_trace_graph, but adds
    submodule scope information to each graph node.

    The resulting graph is in-lined and has all model parameters treated as
    inputs. The input model has the scope name '', while its descendants
    have names of the form 'child.grandchild.grandgrandchild...'.

    Args:
        model (nn.Module): The module to trace
        inputs (tuple): Inputs used during the trace of the model
        aliases (dict[str or nn.Module, str]): maps modules and module
            names to the canonical name to be used as the scope for
            that module.

    Returns:
        graph (torch._C.Graph): The pytorch JIT trace of the model
    """

    # torch.jit._get_trace_graph can trace torch function like `aten::linear`,
    # `aten::add` etc. However, the traced node(function) cannot tell it is
    # called by which module. `ScopePushHook` and `ScopePopHook` can
    # help traced node get the module name information by `node.scopeName()`.
    class ScopePushHook:

        def __init__(self, name: str) -> None:
            self.name = name

        def __call__(self, module: nn.Module, inputs: Any) -> Any:
            tracing_state = torch._C._get_tracing_state()
            if tracing_state:
                tracing_state.push_scope(self.name)
            return inputs

    class ScopePopHook:

        def __call__(self, module: nn.Module, inputs: Any,
                     outputs: Any) -> Any:
            tracing_state = torch._C._get_tracing_state()
            if tracing_state:
                tracing_state.pop_scope()
            return outputs

    hook_handles: List[Any] = []

    def register_hooks(mod: nn.Module, name: str) -> None:
        prehook = mod.register_forward_pre_hook(ScopePushHook(name))
        posthook = mod.register_forward_hook(ScopePopHook())
        hook_handles.append(prehook)
        hook_handles.append(posthook)

    # Unwrap DDP, but correct the scope names for the root module.
    module_list = (nn.parallel.distributed.DistributedDataParallel,
                   nn.DataParallel)
    # Since DataParallel just wraps the model, add an extra set of hooks
    # to the model it wraps to account for the wrapper. Then trace it.
    if isinstance(module, module_list):
        root_name = aliases[module]
        module = module.module
        register_hooks(module, root_name)

    for name, mod in _named_modules_without_dup(module):
        name = aliases[mod]
        register_hooks(mod, name)

    graph, _ = _get_trace_graph(module, inputs)

    for handle in hook_handles:
        handle.remove()

    return graph


class JitModelAnalysis:
    """Provides access to per-submodule model statistics obtained by tracing a
    model with pytorch's jit tracing functionality.

    Calculates a statistic on a per-operator basis using the provided set of
    functions that acts on the inputs and outputs to the operator, then
    aggregates this over modules in the model. Can return the aggregate
    statistic for any submodule in the model. Is lazily evaluated, and will
    perform the trace when a statistic is first requested. Changing the
    operator handles will cause the trace to be rerun on the next request.

    Submodules may be referred to using the module's name. The input model has
    name "", while its descendants have names of the form
    "child.grandchild.grandgrandchild...".

    An operator is treated as within the scope of a module if calling that
    module directly resulted in that operator being run. In particular, this
    means that calls to other functions owned by a module or explicit
    calls to module.forward(...) will not register resulting operators as
    contributing statistics to that module.

    We will trace the execution of `model.forward(inputs)`. This means
    inputs have to be tensors or tuple of tensors (see
    https://pytorch.org/docs/stable/generated/torch.jit.trace.html#torch.jit.trace).
    In order to trace other methods or unsupported input types,
    you may need to implement a wrapper module.

    Args:
        model: The model to analyze
        inputs: The inputs to the model for analysis.
    """

    def __init__(
        self,
        model: nn.Module,
        inputs: Union[Tensor, Tuple[Tensor, ...]],
    ) -> None:
        self._model = model
        self._inputs = inputs
        self._op_handles: Dict[str, Handle] = {}
        # Mapping from names to submodules
        self._named_modules: Dict[str, nn.Module] = dict(
            _named_modules_with_dup(model))
        # Mapping from submodules and their aliases to the canonical name
        # of each submodule
        self._aliases: Dict[Union[nn.Module, str],
                            str] = self._get_aliases(model)
        self._stats: Optional[Statistics] = None

        self._ignored_ops: Set[str] = copy(_IGNORED_OPS)
        self.unsupported_ops_warnings(True)
        self.uncalled_modules_warnings(True)
        self.tracer_warnings('no_tracer_warning')
        self.ancestor_mode('owner')

    def total(self, module_name: str = '') -> int:
        """Returns the total aggregated statistic across all operators for the
        requested module.

        Args:
            module_name (str): The submodule to get data for. Defaults to
                the entire model.

        Returns:
            int: The aggregated statistic.
        """
        stats = self._analyze()
        module_name = self.canonical_module_name(module_name)
        total_count = sum(stats.counts[module_name].values())
        return total_count

    def by_operator(self, module_name: str = '') -> typing.Counter[str]:
        """Returns the statistics for a requested module, grouped by operator
        type.

        The operator handle determines the name associated with each
        operator type.

        Args:
            module_name (str): The submodule to get data for. Defaults
                to the entire model.

        Returns:
            Counter(str): The statistics for each operator.
        """
        stats = self._analyze()
        module_name = self.canonical_module_name(module_name)
        return stats.counts[module_name]

    def by_module_and_operator(self) -> Dict[str, typing.Counter[str]]:
        """Returns the statistics for all submodules, separated out by operator
        type for each submodule.

        The operator handle determines the name associated with
        each operator type.

        Returns:
            dict[str, Counter(str)]: The statistics for each submodule
            and each operator. Grouped by submodule names, then
            by operator name.
        """
        stats = self._analyze()
        return stats.counts

    def by_module(self) -> typing.Counter[str]:
        """Returns the statistics for all submodules, aggregated over all
        operators.

        Returns:
            Counter(str): statistics counter grouped by submodule names
        """
        stats = self._analyze()
        summed_counts = Counter()  # type: Counter
        for mod, results in stats.counts.items():
            summed_counts[mod] = sum(results.values())
        return summed_counts

    def unsupported_ops(self, module_name: str = '') -> typing.Counter[str]:
        """Lists the number of operators that were encountered but unsupported
        because no operator handle is available for them.

        Does not include operators that are explicitly ignored.

        Args:
            module_name (str): The submodule to list unsupported ops.
                Defaults to the entire model.

        Returns:
            Counter(str): The number of occurrences each unsupported operator.
        """
        if self._stats is None:
            raise RuntimeError('Analysis results should be computed '
                               'before calling unsupported_ops()')
        module_name = self.canonical_module_name(module_name)
        return self._stats.unsupported_ops[module_name]  # pyre-fixme

    def uncalled_modules(self) -> Set[str]:
        """Returns a set of submodules that were never called during the trace
        of the graph.

        This may be because they were unused, or because they were
        accessed via direct calls .forward() or with other python methods.
        In the latter case, statistics will not be attributed to the submodule,
        though the statistics will be included
        in the parent module.

        Returns:
            set[str]: The set of submodule names that were never called
            during the trace of the model.
        """
        stats = self._analyze()
        return stats.uncalled_mods

    def set_op_handle(self, *args,
                      **kwargs: Optional[Handle]) -> 'JitModelAnalysis':
        """Sets additional operator handles, or replaces existing ones.

        If a handle is ``None``, the op will be explicitly ignored. Otherwise,
        handle should be a function that calculates the desirable statistic
        from an operator. The function must take two arguments, which are the
        inputs and outputs of the operator, in the form of
        ``list(torch._C.Value)``. The function should return a counter object
        with per-operator statistics.

        Args:
            args: (str, Handle) pairs of operator names and handles.
            kwargs: mapping from operator names to handles.

        Examples:
            >>> handlers = {"aten::linear": my_handler}
            >>> counter.set_op_handle("aten::matmul", None,
            ...     "aten::bmm", my_handler2).set_op_handle(**handlers)
        """
        self._stats = None
        if len(args) % 2 != 0:
            raise TypeError(
                'set_op_handle should be called with pairs of names and'
                'handles!')
        for name, handle in zip(args[::2], args[1::2]):
            kwargs[name] = handle
        for name, handle in kwargs.items():
            if handle is None:
                self._ignored_ops.add(name)
            else:
                self._op_handles[name] = handle
        return self

    def clear_op_handles(self) -> 'JitModelAnalysis':
        """Clears all operator handles currently set."""
        self._op_handles = {}
        self._ignored_ops = copy(_IGNORED_OPS)
        self._stats = None
        return self

    def canonical_module_name(self, name: str) -> str:
        """Returns the canonical module name of the given ``name``, which might
        be different from the given ``name`` if the module is shared.

        This is the name that will be used as a key when statistics are
        output using .by_module() and .by_module_and_operator().

        Args:
            name (str): The name of the module to find the canonical name for.

        Returns:
            str: The canonical name of the module.
        """
        # Blocks access by a direct module reference
        assert isinstance(name, str), 'Module name must be a string.'
        if name in self._aliases:
            return self._aliases[name]
        else:
            raise KeyError('Requested module name is not among '
                           'the descendants of the analyzed model.')

    def copy(
        self,
        new_model: Optional[nn.Module] = None,
        new_inputs: Union[None, Tensor, Tuple[Tensor, ...]] = None,
    ) -> 'JitModelAnalysis':
        """Returns a copy of the :class:`JitModelAnalysis` object, keeping all
        settings, but on a new model or new inputs.

        Args:
            new_model (nn.Module or None): a new model for the new
                JitModelAnalysis. If None, uses the original model.
                Defaults to None.
            new_inputs (typing.Tuple[object, ...], optional): new inputs
                for the new JitModelAnalysis. If None, uses the original
                inputs. Defaults to None.

        Returns:
            JitModelAnalysis: the new model analysis object
        """
        model = self._model if new_model is None else new_model
        inputs = self._inputs if new_inputs is None else new_inputs
        return (JitModelAnalysis(model=model, inputs=inputs).set_op_handle(
            **self._op_handles).unsupported_ops_warnings(
                self._enable_warn_unsupported_ops).uncalled_modules_warnings(
                    self._enable_warn_uncalled_mods).tracer_warnings(
                        self._warn_trace))

    def tracer_warnings(self: T, mode: str) -> T:
        """Sets which warnings to print when tracing the graph to calculate
        statistics. There are three modes. Defaults to 'no_tracer_warning'.
        Allowed values are:

        * 'all' : keeps all warnings raised while tracing
        * 'no_tracer_warning' : suppress torch.jit.TracerWarning only
        * 'none' : suppress all warnings raised while tracing

        Args:
            mode (str) : warning mode in one of the above values.
        """
        if mode not in ['all', 'no_tracer_warning', 'none']:
            raise ValueError(f'Unrecognized tracer warning mode {mode}.')
        self._warn_trace = mode
        return self

    def ancestor_mode(self: T, mode: str) -> T:
        """Sets how to determine the ancestor modules of an operator. Must be
        one of "owner" or "caller".

        * "caller": an operator belongs to all modules that are currently
            executing `forward()` at the time the operator is called.
        * "owner": an operator belongs to the last module that's executing
            `forward()` at the time the operator is called, plus this
            module's recursive parents. If an module has multiple parents
            (e.g. a shared module), only one will be picked.

        For most cases, a module only calls submodules it owns, so both
        options would work identically. In certain edge cases, this option
        will affect the hierarchy of results, but won't affect the total
        count.
        """
        if mode not in ['owner', 'caller']:
            raise ValueError(f'Unrecognized ancestor mode: {mode}')
        self._ancestor_mode = mode
        return self

    def unsupported_ops_warnings(self: T, enabled: bool) -> T:
        """Sets if warnings for unsupported operators are shown.

        Defaults to True. Counts of unsupported operators may be
        obtained from :meth:`unsupported_ops` regardless of this setting.

        Args:
            enabled (bool): Set to 'True' to show unsupported operator
                warnings.
        """
        self._enable_warn_unsupported_ops = enabled
        return self

    def uncalled_modules_warnings(self: T, enabled: bool) -> T:
        """Sets if warnings from uncalled submodules are shown.

        Defaults to true. A submodule is considered "uncalled" if it is never
        called during tracing. This may be because it is actually unused, or
        because it is accessed via calls to ``.forward()`` or other methods of
        the module. The set of uncalled modules may be obtained from
        :meth:`uncalled_modules` regardless of this setting.

        Args:
            enabled (bool): Set to 'True' to show warnings.
        """
        self._enable_warn_uncalled_mods = enabled
        return self

    def _warn_unsupported_ops(self, ops: typing.Counter[str]) -> None:
        if not self._enable_warn_unsupported_ops:
            return

        for op, freq in ops.items():
            print_log(
                'Unsupported operator {} encountered {} time(s)'.format(
                    op, freq),
                'current',
                logging.WARNING,
            )

    def _warn_uncalled_mods(self, uncalled_mods: Set[str]) -> None:
        if not self._enable_warn_uncalled_mods:
            return
        uncalled_mods = {x for x in uncalled_mods if self._has_forward(x)}
        if len(uncalled_mods) == 0:
            return

        print_log(
            'The following submodules of the model were never '
            'called during the trace of the graph. They may be '
            'unused, or they were accessed by direct calls to '
            '.forward() or via other python methods. In the latter '
            'case they will have zeros for statistics, though their '
            'statistics will still contribute to their parent calling '
            'module.\n' + ', '.join(sorted(uncalled_mods)), 'current',
            logging.WARNING)

    def _get_aliases(self,
                     model: nn.Module) -> Dict[Union[str, nn.Module], str]:
        aliases = {}
        for name, module in _named_modules_with_dup(model):
            if module not in aliases:
                aliases[module] = name
            aliases[name] = aliases[module]
        return aliases

    def _get_all_ancestors(self, module_name: str) -> Set[str]:
        """Get all ancestors of the given module, defined by ownership.

        If the given module has multiple owners, use its canonical name.
        """
        parts = self.canonical_module_name(module_name).split('.')
        res = {''}
        for k in range(len(parts) + 1):
            res.add('.'.join(parts[:k]))
        return res

    def _analyze(self) -> 'Statistics':
        # Don't calculate if results are already stored.
        stats = self._stats
        if stats is not None:
            return stats

        with warnings.catch_warnings():
            if self._warn_trace == 'none':
                warnings.simplefilter('ignore')
            elif self._warn_trace == 'no_tracer_warning':
                warnings.filterwarnings('ignore', category=TracerWarning)
            graph = _get_scoped_trace_graph(self._model, self._inputs,
                                            self._aliases)

        # Assures even modules not in the trace graph are initialized to
        # zero count
        counts = {}  # type: Dict
        unsupported_ops = {}  # type: Dict
        # We don't need the duplication here, but self._model.named_modules()
        # gives slightly different results for some wrapped models.
        for _, mod in _named_modules_with_dup(self._model):
            name = self._aliases[mod]
            counts[name] = Counter()
            unsupported_ops[name] = Counter()

        all_seen = set()
        for node in graph.nodes():
            kind = node.kind()
            if kind == 'prim::PythonOp':
                # for PythonOp, pyname contains the actual name in Python
                # pyre-fixme[16]: `Node` has no attribute `pyname`.
                kind = kind + '.' + node.pyname()
            scope_names = node.scopeName().split('/')
            all_seen.update(scope_names)
            # The result of node.scopeName() is like: `layer1/layer1.layer`
            # Therefore, if there is not shared module ancestors will have the
            # same value. However, if layer1.layer is used by multiple modules.
            # scopeName() will return
            # `layer1/layer1.layer`
            # `layer2/layer1.layer` respectively
            # If mode is `caller`, the ancestors will be:
            # 'layer1', 'layer2', 'layer1.layer'
            # else, the ancestors will be:
            # 'layer1', 'layer1.layer'
            # which means only the flops will only be counted into `layer1`.
            if self._ancestor_mode == 'caller':
                ancestors = set(scope_names)
            else:
                ancestors = self._get_all_ancestors(scope_names[-1])
                all_seen.update(ancestors)
            if kind not in self._op_handles:
                if self._should_ignore_node(node):
                    continue
                for name in ancestors:
                    unsupported_ops[name][kind] += 1
            else:
                inputs, outputs = list(node.inputs()), list(node.outputs())
                op_counts = self._op_handles[kind](inputs, outputs)
                if isinstance(op_counts, Number):
                    op_counts = Counter(
                        {self._simplify_op_name(kind): op_counts})
                for v in op_counts.values():  # type: ignore
                    if not isinstance(v, (int, float, np.float64, np.int64)):
                        raise ValueError(
                            f'Invalid type {type(v)} for the flop count! '
                            'Please use a wider type to avoid overflow.')

                # Assures an op contributes at most once to a module
                for name in ancestors:
                    counts[name] += op_counts

        uncalled_mods = set(self._aliases.values()) - all_seen
        stats = Statistics(
            counts=counts,
            unsupported_ops=unsupported_ops,
            uncalled_mods=uncalled_mods)
        self._stats = stats
        self._warn_unsupported_ops(unsupported_ops[''])
        self._warn_uncalled_mods(uncalled_mods)
        return stats

    def _simplify_op_name(self, full_op_name: str) -> str:
        """Get simplified name of the op without the preceding namespace, e.g.
        aten::batch_norm -> batch_norm."""
        p = full_op_name.find('::')
        if p != -1:
            return full_op_name[p + 2:]
        else:
            return full_op_name

    def _has_forward(self, mod_name: str) -> bool:
        # Whether the module has a valid forward method.
        # Modules without forward are not expected to get called
        # and therefore should not produce "uncalled" warnings
        module = self._named_modules.get(mod_name)
        if module is None:
            return False
        module_type = type(module)
        # Containers are not meant to be called anyway (they don't have
        # forward)
        # NOTE: We add nn.Identity as well to silence the uncalled warning,
        # but it's different from other containers: Identity has a forward
        # but the forward does not contain ops, so it appears "uncalled" after
        # tracing. A more proper way may be to use forward hooks (instead of
        # the graph) to decide whether a module has been called.
        no_forward_mods = {
            nn.ModuleList, nn.ModuleDict, nn.Module, nn.Identity
        }
        for mod in no_forward_mods:
            if module_type.forward is mod.forward:
                return False
        return True

    def _should_ignore_node(self, node) -> bool:
        kind = node.kind()
        if kind in self._ignored_ops:
            return True
        # Ignore all prim:: operators, with two exceptions:
        # * prim::PythonOp can be a user-implemented `torch.autograd.Function`
        # * prim::CallFunction an be a call to scripted module/function.
        if kind.startswith('prim::PythonOp') or kind.startswith(
                'prim::CallFunction'):
            return False
        if kind.startswith('prim::'):
            return True
        return False

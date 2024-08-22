# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import os.path as osp
import sys
from typing import Callable, Optional, Union

import torch

from mmengine.dist import master_only
from mmengine.hooks import Hook
from mmengine.logging import print_log
from mmengine.registry import HOOKS


def check_kineto() -> bool:  # noqa
    kineto_exist = False
    try:
        if torch.autograd.kineto_available():
            kineto_exist = True
    except AttributeError:
        print_log('NO KINETO', logger='current', level=logging.WARNING)
    return kineto_exist


@HOOKS.register_module()
class ProfilerHook(Hook):
    """A hook to analyze performance during training and inference.

    PyTorch Profiler is a tool that allows the collection of the performance
    metrics during the training. More details on Profiler can be found at
    `official docs <https://pytorch.org/docs/stable/profiler.html
    #torch.profiler.profile>`_

    Args:
        by_epoch (bool): Profile performance by epoch or by iteration.
            Defaults to True.
        profile_times (int): The period (epoch/iter) recorded by the profiler.
            Defaults to 1. For example, profile_iters=10 and by_epoch=False,
            indicate that 0-10 iterations are recorded.
        activity_with_cpu (bool): Activities to be used in the analysis (CPU)
        activity_with_cuda (bool): Activities to be used in the analysis (CUDA)
        schedule (dict, optional): Key-word arguments passed to
            `torch.profile.schedule <https://pytorch.org/docs/stable/
            profiler.html#torch.profiler.schedule>`_.
            Defaults to None, which means profiling without a schedule
        on_trace_ready (callable, dict, optional): Either a handler or a dict
            of generating handler. Defaults to None, which means profiling
            without an on_trace_ready.The Callable type needs to construct its
            own function that can handle 'torch.autograd.profiler.profile'.
            Two officially recommended ways are provided:

            - ``schedule=dict(type='log_trace')``: Print the profiling result
              in the terminal. See more details in the `PyTorch official tutorial`_.
              The configurable arguments are the same as
              ``prof.key_averages().table``
            - ``scheduler=dict(type='tb_trace')``: Profile the performance
              with tensorboard. See more details in the tutorial
              `profile with tensorboard`_.

        record_shapes (bool): Save information about operator's input shapes.
            Defaults to False.
        profile_memory (bool): Track tensor memory allocation/deallocation.
            Defaults to False.
        with_stack (bool): Record source information (file and line number)
            for the ops. Defaults to False.
        with_flops (bool): Use formula to estimate the FLOPS of specific
            operators (matrix multiplication and 2D convolution).
            Defaults to False.
        json_trace_path (str, optional): Exports the collected trace in Chrome
            JSON format. Chrome use 'chrome://tracing' view json file.
            Defaults to None, which means profiling does not store json files.

    Warnings:
        The profiler will be closed after ``profile_times`` iterations
        automatically. Please make sure the configuration of your scheduler
        will not close the profiler before the iteration reach the value of
        ``profile_times``

    Examples:
        >>> # tensorboard trace
        >>> trace_config = dict(type='tb_trace')
        >>> profiler_hook_cfg = dict(on_trace_ready=trace_config)

    .. _PyTorch official tutorial: https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html#using-profiler-to-analyze-execution-time
    .. _profile with tensorboard: https://pytorch.org/tutorials/intermediate/tensorboard_profiler_tutorial.html#pytorch-profiler-with-tensorboard
    """  # noqa: E501
    priority = 'VERY_LOW'

    def __init__(self,
                 *,
                 by_epoch: bool = True,
                 profile_times: int = 1,
                 activity_with_cpu: bool = True,
                 activity_with_cuda: bool = False,
                 schedule: Optional[dict] = None,
                 on_trace_ready: Union[Callable, dict, None] = None,
                 record_shapes: bool = False,
                 profile_memory: bool = False,
                 with_stack: bool = False,
                 with_flops: bool = False,
                 json_trace_path: Optional[str] = None) -> None:

        try:
            from torch import profiler
        except ImportError:
            raise ImportError('please upgrade torch above 1.8.1')
        if not check_kineto():
            raise ImportError('Due to Kineto support issues, please upgrade '
                              'pytorch above 1.8.1(windows users above 1.9.1)')

        assert isinstance(by_epoch, bool), '``by_epoch`` should be a boolean.'
        self.by_epoch = by_epoch

        if profile_times < 1:
            raise ValueError('profile_iters should be greater than 0, '
                             f'but got {profile_times}')
        if by_epoch and profile_times > 1:
            raise ValueError(
                f'Profiler will profile 0-{profile_times} epochs.\n'
                'Since profiler will slow down the training, it is recommended'
                ' to train 1 epoch with ProfilerHook and adjust your setting '
                'according to the profiler summary.\n'
                'During normal training(epoch > 1), '
                'you may disable the ProfilerHook.')
        self.profile_times = profile_times

        assert isinstance(activity_with_cpu, bool), \
            '``activity_with_cpu`` should be a boolean.'
        assert isinstance(activity_with_cuda, bool), \
            '``activity_with_cuda`` should be a boolean.'
        self.activities = []
        if activity_with_cpu:
            self.activities.append(profiler.ProfilerActivity.CPU)
        if activity_with_cuda:
            self.activities.append(profiler.ProfilerActivity.CUDA)

        if schedule is not None:
            assert isinstance(schedule, dict), '``schedule`` should be a dict.'
            self.schedule = profiler.schedule(**schedule)
        else:
            self.schedule = None

        self.on_trace_ready = on_trace_ready
        self.record_shapes = record_shapes
        self.profile_memory = profile_memory
        self.with_stack = with_stack
        self.with_flops = with_flops

        self.json_trace_path = json_trace_path
        self._closed = False

    def before_run(self, runner):
        """Initialize the profiler.

        Through the runner parameter, the validity of the parameter is further
        determined.
        """
        max_times = runner.max_epochs if self.by_epoch else runner.max_iters
        if max_times < self.profile_times:
            raise ValueError(
                f'``profile_times`` should not be greater than {max_times}')

        on_trace_ready = self._parse_trace_config(runner)

        self.profiler = torch.profiler.profile(  # noqa
            activities=self.activities,
            schedule=self.schedule,
            on_trace_ready=on_trace_ready,
            record_shapes=self.record_shapes,
            profile_memory=self.profile_memory,
            with_stack=self.with_stack,
            with_flops=self.with_flops)

        self.profiler.__enter__()
        runner.logger.info('profiler is profiling...')

    def _parse_trace_config(self, runner):
        """Used to parse the parameter 'on_trace_ready'."""
        if self.on_trace_ready is None:
            _on_trace_ready = None
        elif callable(self.on_trace_ready):
            _on_trace_ready = self.on_trace_ready
        elif isinstance(self.on_trace_ready, dict):
            trace_cfg = self.on_trace_ready.copy()
            trace_type = trace_cfg.pop('type')

            # Build a log printing handle
            if trace_type == 'log_trace':

                def _log_handler(_profile):
                    print(_profile.key_averages().table(**trace_cfg))

                _on_trace_ready = _log_handler

            elif trace_type == 'tb_trace':  # tensorboard_trace handler
                try:
                    import torch_tb_profiler  # noqa: F401
                except ImportError:
                    raise ImportError(
                        'please run ``pip install torch-tb-profiler``')

                if 'dir_name' not in trace_cfg:
                    trace_cfg['dir_name'] = osp.join(runner.log_dir,
                                                     'tf_tracing_logs')
                elif not osp.isabs(trace_cfg['dir_name']):
                    trace_cfg['dir_name'] = osp.join(runner.log_dir,
                                                     trace_cfg['dir_name'])
                runner.logger.info('trace_files of ProfilerHook will be '
                                   f'saved to {trace_cfg["dir_name"]}.')

                if self.json_trace_path is not None:
                    runner.logger.warn(
                        'When using tensorboard_trace, it is recommended to '
                        'save json files by setting ``worker_name`` instead of'
                        ' setting ``json_trace_path``')
                _on_trace_ready = torch.profiler.tensorboard_trace_handler(
                    **trace_cfg)
            else:
                raise ValueError('trace_type should be "log_trace" or '
                                 f'"tb_trace", but got {trace_type}')
        else:
            raise ValueError(
                '``on_trace_ready`` should be a handler, or dict, or None, '
                f'but got {self.on_trace_ready}')
        return _on_trace_ready

    def after_train_epoch(self, runner):
        """Determine if the content is exported."""
        # `after_train_epoch` will also be called in IterBasedTrainLoop.
        # Here we check `self._closed` to avoid exiting twice.
        if not self._closed:
            self._export_chrome_trace(runner)

    def after_train_iter(self, runner, batch_idx, data_batch, outputs):
        """profiler will call `step` method if it is not closed."""
        if not self._closed:
            self.profiler.step()
        if runner.iter == self.profile_times - 1 and not self.by_epoch:
            self._export_chrome_trace(runner)

    def _export_chrome_trace(self, runner):
        """Exporting content."""
        self._closed = True
        runner.logger.info('profiler may take a few minutes...')
        self.profiler.__exit__(None, None, None)
        if self.json_trace_path is not None:
            self.profiler.export_chrome_trace(self.json_trace_path)


@HOOKS.register_module()
class NPUProfilerHook(Hook):
    """NPUProfiler to analyze performance during training.

    NPU Profiling is used to count the device execution time of all operators.
    The torch_npu.npu.profile interface is used to complete the profiling data
    collection at each stage of the project, and the data is analyzed by the
    msprof tool and the data can be dumped to further manually analyze the
    key performance bottlenecks. For more details on the torch_npu.npu.profile
    interface, please visit
    https://gitee.com/ascend/pytorch/blob/master/torch_npu/npu/profiler.py#profile

    Args:
        begin (int): Number of start iterations for profiling. Defaults to 0.
        end (int): Number of end iterations for profiling. Defaults to 1.
        result_path (str): The path to save the profiling results file.
            Defaults to 'cann_profiling'.
        exit_after_profiling (bool): Whether to exit the program after
            profiling. Defaults to True.
        use_e2e_profiler (bool): Turn on E2E profiling, E2E profiling combines
            performance data at the Pytorch level and the NPU level to analyze
            the bottlenecks of model performance end-to-end, and cannot show
            detailed content, and only as an auxiliary analysis.
            Defaults to False.
        ge_profiling_to_std_out (bool): Turn on GE profiling, GE uses to
            collect the profiling data of the host side scheduling of the
            Assend device. Defaults to False.

    Examples:
        >>> cfg = ...
        >>> profiler_config = dict(type='NPUProfilerHook', end=2)
        >>> cfg.merge_from_dict({'custom_hooks': custom_hooks})
        >>> runner = Runner.from_cfg(cfg)
        >>> runner.train()
    """
    priority = 'VERY_LOW'

    def __init__(self,
                 *,
                 begin: int = 0,
                 end: int = 1,
                 result_path: str = 'cann_profiling',
                 exit_after_profiling: bool = True,
                 use_e2e_profiler: bool = False,
                 ge_profiling_to_std_out: bool = False):

        try:
            import torch_npu
        except ImportError:
            raise ImportError('Failed to import torch_npu module')

        if begin >= end:
            raise ValueError(
                'The iteration to start profiling should not be greater'
                'than or equal to profile end')

        self.begin = begin
        self.end = end
        self.result_path = result_path
        self.exit_after_profiling = exit_after_profiling

        if ge_profiling_to_std_out:
            os.environ['GE_PROFILING_TO_STD_OUT'] = '1'

        if not osp.exists(self.result_path):
            os.makedirs(self.result_path, exist_ok=True)

        self.profiler = torch_npu.npu.profile(
            self.result_path, use_e2e_profiler=use_e2e_profiler)

    @master_only
    def before_run(self, runner):

        if self.end > runner.max_iters:
            raise ValueError(
                'The profiling end iteration should not be greater'
                'than the max iteration')

    @master_only
    def before_train_iter(self, runner, batch_idx, data_batch=None):

        if runner.iter == self.begin:
            self.profiler.__enter__()
            runner.logger.info('NPUProfiler starts profiling...')

    @master_only
    def after_train_iter(self,
                         runner,
                         batch_idx,
                         data_batch=None,
                         outputs=None):

        if runner.iter == self.end - 1:
            runner.logger.info('profiler may take a few minutes to'
                               ' save the profiling result.')
            self.profiler.__exit__(None, None, None)
            if self.exit_after_profiling:
                sys.exit()

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import copy
import datetime
import re
from collections import OrderedDict
from itertools import chain
from typing import List, Optional, Tuple

import numpy as np
import torch

from mmengine.device import get_max_cuda_memory, is_cuda_available
from mmengine.registry import LOG_PROCESSORS


@LOG_PROCESSORS.register_module()
class LogProcessor:
    """A log processor used to format log information collected from
    ``runner.message_hub.log_scalars``.

    ``LogProcessor`` instance is built by runner and will format
    ``runner.message_hub.log_scalars`` to ``tag`` and ``log_str``, which can
    directly used by ``LoggerHook`` and ``MMLogger``. Besides, the argument
    ``custom_cfg`` of constructor can control the statistics method of logs.

    Args:
        window_size (int): default smooth interval. Defaults to 10.
        by_epoch (bool): Whether to format logs with epoch stype. Defaults to
            True.
        custom_cfg (list[dict], optional): Contains multiple log config dict,
            in which key means the data source name of log and value means the
            statistic method and corresponding arguments used to count the
            data source. Defaults to None.

            - If custom_cfg is None, all logs will be formatted via default
              methods, such as smoothing loss by default window_size. If
              custom_cfg is defined as a list of config dict, for example:
              [dict(data_src='loss', method='mean', log_name='global_loss',
              window_size='global')]. It means the log item ``loss`` will be
              counted as global mean and additionally logged as ``global_loss``
              (defined by ``log_name``). If ``log_name`` is not defined in
              config dict, the original logged key will be overwritten.

            - The original log item cannot be overwritten twice. Here is
              an error example:
              [dict(data_src='loss', method='mean', window_size='global'),
              dict(data_src='loss', method='mean', window_size='epoch')].
              Both log config dict in custom_cfg do not have ``log_name`` key,
              which means the loss item will be overwritten twice.

            - For those statistic methods with the ``window_size`` argument,
              if ``by_epoch`` is set to False, ``windows_size`` should not be
              `epoch` to statistics log value by epoch.
        num_digits (int): The number of significant digit shown in the
            logging message. Defaults to 4.
        log_with_hierarchy (bool): Whether to log with hierarchy. If it is
            True, the information is written to visualizer backend such as
            :obj:`LocalVisBackend` and :obj:`TensorboardBackend`
            with hierarchy. For example, ``loss`` will be saved as
            ``train/loss``, and accuracy will be saved as ``val/accuracy``.
            Defaults to False.
            `New in version 0.7.0.`
        mean_pattern (str): This is a regular expression used to match the log
            that need to be included in the smoothing statistics.
            `New in version 0.7.3.`

    Examples:
        >>> # `log_name` is defined, `loss_large_window` will be an additional
        >>> # record.
        >>> log_processor = dict(
        >>>     window_size=10,
        >>>     by_epoch=True,
        >>>     custom_cfg=[dict(data_src='loss',
        >>>                       log_name='loss_large_window',
        >>>                       method_name='mean',
        >>>                       window_size=100)])
        >>> # `log_name` is not defined. `loss` will be overwritten.
        >>> log_processor = dict(
        >>>     window_size=10,
        >>>     by_epoch=True,
        >>>     custom_cfg=[dict(data_src='loss',
        >>>                       method_name='mean',
        >>>                       window_size=100)])
        >>> # Record loss with different statistics methods.
        >>> log_processor = dict(
        >>>     window_size=10,
        >>>     by_epoch=True,
        >>>     custom_cfg=[dict(data_src='loss',
        >>>                       log_name='loss_large_window',
        >>>                       method_name='mean',
        >>>                       window_size=100),
        >>>                  dict(data_src='loss',
        >>>                       method_name='mean',
        >>>                       window_size=100)])
        >>> # Overwrite loss item twice will raise an error.
        >>> log_processor = dict(
        >>>     window_size=10,
        >>>     by_epoch=True,
        >>>     custom_cfg=[dict(data_src='loss',
        >>>                       method_name='mean',
        >>>                       window_size=100),
        >>>                  dict(data_src='loss',
        >>>                       method_name='max',
        >>>                       window_size=100)])
        AssertionError
    """

    def __init__(self,
                 window_size=10,
                 by_epoch=True,
                 custom_cfg: Optional[List[dict]] = None,
                 num_digits: int = 4,
                 log_with_hierarchy: bool = False,
                 mean_pattern=r'.*(loss|time|data_time|grad_norm).*'):
        self.window_size = window_size
        self.by_epoch = by_epoch
        self.custom_cfg = custom_cfg if custom_cfg else []
        self.num_digits = num_digits
        self.log_with_hierarchy = log_with_hierarchy
        self.mean_pattern = re.compile(mean_pattern)
        self._check_custom_cfg()

    def get_log_after_iter(self, runner, batch_idx: int,
                           mode: str) -> Tuple[dict, str]:
        """Format log string after training, validation or testing iteration.

        Args:
            runner (Runner): The runner of training phase.
            batch_idx (int): The index of the current batch in the current
                loop.
            mode (str): Current mode of runner, train, test or val.

        Return:
            Tuple[dict, str]: Formatted log dict/string which will be
            recorded by :obj:`runner.message_hub` and :obj:`runner.visualizer`.
        """
        assert mode in ['train', 'test', 'val']
        # Overwrite ``window_size`` defined in ``custom_cfg`` to int value.
        parsed_cfg = self._parse_windows_size(runner, batch_idx,
                                              self.custom_cfg)
        # log_tag is used to write log information to terminal
        log_tag = self._collect_scalars(parsed_cfg, runner, mode)

        # If `self.log_with_hierarchy` is False, the tag is the same as
        # log_tag. Otherwise, each key in tag starts with prefix `train`,
        # `test` or `val`
        if not self.log_with_hierarchy:
            tag = copy.deepcopy(log_tag)
        else:
            tag = self._collect_scalars(parsed_cfg, runner, mode, True)

        # Record learning rate.
        lr_str_list = []
        for key, value in tag.items():
            if key.endswith('lr'):
                key = self._remove_prefix(key, f'{mode}/')
                log_tag.pop(key)
                lr_str_list.append(f'{key}: '
                                   f'{value:.{self.num_digits}e}')
        lr_str = ' '.join(lr_str_list)
        # Format log header.
        # by_epoch == True
        #   train/val: Epoch [5][5/10]  ...
        #   test: Epoch [5/10]
        # by_epoch == False
        #  train: Epoch [5/10000] ... (divided by `max_iter`)
        #  val/test: Epoch [5/2000] ... (divided by length of dataloader)
        if self.by_epoch:
            # Align the iteration log:
            # Epoch(train)  [  9][010/270]
            # ...                 ||| |||
            # Epoch(train)  [ 10][100/270]
            dataloader_len = self._get_dataloader_size(runner, mode)
            cur_iter = self._get_iter(runner, batch_idx)
            cur_iter_str = str(cur_iter).rjust(len(str(dataloader_len)))
            if mode in ['train', 'val']:
                cur_epoch = self._get_epoch(runner, mode)
                if not (isinstance(runner._train_loop, dict)
                        or runner._train_loop is None):
                    # Right Align the epoch log:
                    # Epoch(train)   [9][100/270]
                    # ...             ||
                    # Epoch(train) [100][100/270]
                    max_epochs = runner.max_epochs
                    # 3 means the three characters: "[", "]", and " " occupied
                    # in " [{max_epochs}]"
                    cur_epoch_str = f'[{cur_epoch}]'.rjust(
                        len(str(max_epochs)) + 3, ' ')
                else:
                    cur_epoch_str = f'[{cur_epoch}]'
                tag['epoch'] = cur_epoch
                log_str = (f'Epoch({mode}){cur_epoch_str}'
                           f'[{cur_iter_str}/{dataloader_len}]  ')
            else:
                log_str = (f'Epoch({mode}) '
                           f'[{cur_iter_str}/{dataloader_len}]  ')
        else:
            if mode == 'train':
                cur_iter = self._get_iter(runner, batch_idx)
                cur_iter_str = str(cur_iter).rjust(len(str(runner.max_iters)))
                log_str = (f'Iter({mode}) '
                           f'[{cur_iter_str}/{runner.max_iters}]  ')
            else:
                dataloader_len = self._get_dataloader_size(runner, mode)
                cur_iter_str = str(batch_idx + 1).rjust(
                    len(str(dataloader_len)))
                log_str = (f'Iter({mode}) [{cur_iter_str}/{dataloader_len}]  ')
        # Add global iter.
        if isinstance(runner._train_loop, dict) or runner._train_loop is None:
            tag['iter'] = 0
        else:
            tag['iter'] = runner.iter + 1
        # Concatenate lr, momentum string with log header.
        log_str += f'{lr_str}  '
        # If IterTimerHook used in runner, eta, time, and data_time should be
        # recorded.
        if (all(item in log_tag for item in ['time', 'data_time'])
                and 'eta' in runner.message_hub.runtime_info):
            eta = runner.message_hub.get_info('eta')
            eta_str = str(datetime.timedelta(seconds=int(eta)))
            log_str += f'eta: {eta_str}  '
            log_str += (f'time: {log_tag["time"]:.{self.num_digits}f}  '
                        f'data_time: '
                        f'{log_tag["data_time"]:.{self.num_digits}f}  ')
            # Pop recorded keys
            log_tag.pop('time')
            log_tag.pop('data_time')

        # If cuda is available, the max memory occupied should be calculated.
        if is_cuda_available():
            max_memory = self._get_max_memory(runner)
            log_str += f'memory: {max_memory}  '
            tag['memory'] = max_memory
        # Loop left keys to fill `log_str`.
        if mode in ('train', 'val'):
            log_items = []
            for name, val in log_tag.items():
                if mode == 'val' and not name.startswith('val/loss'):
                    continue
                if isinstance(val, float):
                    val = f'{val:.{self.num_digits}f}'
                log_items.append(f'{name}: {val}')
            log_str += '  '.join(log_items)
        return tag, log_str

    def get_log_after_epoch(self,
                            runner,
                            batch_idx: int,
                            mode: str,
                            with_non_scalar: bool = False) -> Tuple[dict, str]:
        """Format log string after validation or testing epoch.

        Args:
            runner (Runner): The runner of validation/testing phase.
            batch_idx (int): The index of the current batch in the current
                loop.
            mode (str): Current mode of runner.
            with_non_scalar (bool): Whether to include non-scalar infos in the
                returned tag. Defaults to False.

        Return:
            Tuple[dict, str]: Formatted log dict/string which will be
            recorded by :obj:`runner.message_hub` and :obj:`runner.visualizer`.
        """
        assert mode in [
            'test', 'val'
        ], ('`_get_metric_log_str` only accept val or test mode, but got '
            f'{mode}')
        dataloader_len = self._get_dataloader_size(runner, mode)

        # By epoch:
        #     Epoch(val) [10][1000/1000]  ...
        #     Epoch(test) [1000/1000] ...
        # By iteration:
        #     Iteration(val) [1000/1000]  ...
        #     Iteration(test) [1000/1000]  ...
        if self.by_epoch:
            if mode == 'val':
                cur_epoch = self._get_epoch(runner, mode)
                log_str = (f'Epoch({mode}) [{cur_epoch}][{dataloader_len}/'
                           f'{dataloader_len}]  ')
            else:
                log_str = (
                    f'Epoch({mode}) [{dataloader_len}/{dataloader_len}]  ')

        else:
            log_str = (f'Iter({mode}) [{dataloader_len}/{dataloader_len}]  ')

        custom_cfg_copy = copy.deepcopy(self.custom_cfg)
        # remove prefix
        custom_keys = [
            self._remove_prefix(cfg['data_src'], f'{mode}/')
            for cfg in custom_cfg_copy
        ]
        # Count the averaged time and data_time by epoch
        if 'time' not in custom_keys:
            custom_cfg_copy.append(
                dict(data_src='time', window_size='epoch', method_name='mean'))
        if 'data_time' not in custom_keys:
            custom_cfg_copy.append(
                dict(
                    data_src='data_time',
                    window_size='epoch',
                    method_name='mean'))
        parsed_cfg = self._parse_windows_size(runner, batch_idx,
                                              custom_cfg_copy)
        # tag is used to write log information to different backends.
        ori_tag = self._collect_scalars(parsed_cfg, runner, mode,
                                        self.log_with_hierarchy)
        non_scalar_tag = self._collect_non_scalars(runner, mode)
        # move `time` or `data_time` to the end of the log
        tag = OrderedDict()
        time_tag = OrderedDict()
        for key, value in ori_tag.items():
            if key in (f'{mode}/time', f'{mode}/data_time', 'time',
                       'data_time'):
                time_tag[key] = value
            else:
                tag[key] = value
        # Log other messages.
        log_items = []
        log_str += '  '
        for name, val in chain(tag.items(), non_scalar_tag.items(),
                               time_tag.items()):
            if isinstance(val, float):
                val = f'{val:.{self.num_digits}f}'
            if isinstance(val, (torch.Tensor, np.ndarray)):
                # newline to display tensor and array.
                val = f'\n{val}\n'
            log_items.append(f'{name}: {val}')
        log_str += '  '.join(log_items)

        if with_non_scalar:
            tag.update(non_scalar_tag)
        tag.update(time_tag)
        return tag, log_str

    def _collect_scalars(self,
                         custom_cfg: List[dict],
                         runner,
                         mode: str,
                         reserve_prefix: bool = False) -> dict:
        """Collect log information to compose a dict according to mode.

        Args:
            custom_cfg (List[dict]): A copy of ``self.custom_cfg`` with int
                ``window_size``.
            runner (Runner): The runner of the training/testing/validation
                process.
            mode (str): Current mode of runner.
            reserve_prefix (bool): Whether to reserve the prefix of the key.

        Returns:
            dict: Statistical values of logs.
        """
        custom_cfg = copy.deepcopy(custom_cfg)
        tag = OrderedDict()
        # history_scalars of train/val/test phase.
        history_scalars = runner.message_hub.log_scalars
        # corresponding mode history_scalars
        mode_history_scalars = OrderedDict()
        # extract log scalars and remove prefix to `mode_history_scalars`
        # according to mode.
        for prefix_key, log_buffer in history_scalars.items():
            if prefix_key.startswith(mode):
                if not reserve_prefix:
                    key = self._remove_prefix(prefix_key, f'{mode}/')
                else:
                    key = prefix_key
                mode_history_scalars[key] = log_buffer
        for key in mode_history_scalars:
            # Update the latest learning rate and smoothed time logs.
            if re.search(self.mean_pattern, key) is not None:
                tag[key] = mode_history_scalars[key].mean(self.window_size)
            else:
                # Default statistic method is current.
                tag[key] = mode_history_scalars[key].current()
        # Update custom keys.
        for log_cfg in custom_cfg:
            data_src = log_cfg.pop('data_src')
            log_name = log_cfg.pop('log_name', data_src)
            if reserve_prefix:
                data_src = f'{mode}/{data_src}'
                log_name = f'{mode}/{log_name}'
            # log item in custom_cfg could only exist in train or val
            # mode.
            if data_src in mode_history_scalars:
                tag[log_name] = mode_history_scalars[data_src].statistics(
                    **log_cfg)
        return tag

    def _collect_non_scalars(self, runner, mode: str) -> dict:
        """Collect log information to compose a dict according to mode.

        Args:
            runner (Runner): The runner of the training/testing/validation
                process.
            mode (str): Current mode of runner.

        Returns:
            dict: non-scalar infos of the specified mode.
        """
        # infos of train/val/test phase.
        infos = runner.message_hub.runtime_info
        # corresponding mode infos
        mode_infos = OrderedDict()
        # extract log info and remove prefix to `mode_infos` according to mode.
        for prefix_key, value in infos.items():
            if prefix_key.startswith(mode):
                if self.log_with_hierarchy:
                    key = prefix_key
                else:
                    key = self._remove_prefix(prefix_key, f'{mode}/')
                mode_infos[key] = value
        return mode_infos

    def _remove_prefix(self, string: str, prefix: str):
        """Remove the prefix ``train``, ``val`` and ``test`` of the key."""
        if string.startswith(prefix):
            return string[len(prefix):]
        else:
            return string

    def _check_custom_cfg(self) -> None:
        """Check the legality of ``self.custom_cfg``."""

        def _check_window_size():
            for log_cfg in self.custom_cfg:
                if not self.by_epoch:
                    assert log_cfg['window_size'] != 'epoch', \
                        'window_size cannot be epoch if LoggerHook.by_epoch' \
                        ' is False.'

        def _check_repeated_log_name():
            # The `log_name` of the same data_src should not be repeated.
            # If `log_name` is not specified, `data_src` will be overwritten.
            # But only allowed to be overwritten once.
            check_set = set()
            for log_cfg in self.custom_cfg:
                assert 'data_src' in log_cfg
                data_src = log_cfg['data_src']
                log_name = log_cfg.get('log_name', data_src)
                assert log_name not in check_set, (
                    f'Found duplicate {log_name} for {data_src}. Please check'
                    'your `custom_cfg` for `log_processor`. You should '
                    f'neither define duplicate `{log_name}` for {data_src} '
                    f'nor do not define any {log_name} for multiple '
                    f'{data_src}, See more information in the docstring of '
                    'LogProcessor')

                check_set.add(log_name)

        _check_repeated_log_name()
        _check_window_size()

    def _parse_windows_size(self,
                            runner,
                            batch_idx: int,
                            custom_cfg: Optional[list] = None) -> list:
        """Parse window_size defined in custom_cfg to int value.

        Args:
            runner (Runner): The runner of the training/testing/validation
                process.
            batch_idx (int): The iteration index of current dataloader.
            custom_cfg (list): A copy of ``self.custom_cfg``. Defaults to None
                to keep backward compatibility.
        """
        if custom_cfg is None:
            custom_cfg = copy.deepcopy(self.custom_cfg)
        else:
            custom_cfg = copy.deepcopy(custom_cfg)
        for log_cfg in custom_cfg:
            window_size = log_cfg.get('window_size', None)
            if window_size is None or isinstance(window_size, int):
                continue
            elif window_size == 'epoch':
                log_cfg['window_size'] = batch_idx + 1
            elif window_size == 'global':
                log_cfg['window_size'] = runner.iter + 1
            else:
                raise TypeError(
                    'window_size should be int, epoch or global, but got '
                    f'invalid {window_size}')
        return custom_cfg

    def _get_max_memory(self, runner) -> int:
        """Returns the maximum GPU memory occupied by tensors in megabytes (MB)
        for a given device.

        Args:
            runner (Runner): The runner of the training/testing/validation
                process.

        Returns:
            The maximum GPU memory occupied by tensors in megabytes for a given
            device.
        """

        device = getattr(runner.model, 'output_device', None)
        return get_max_cuda_memory(device)

    def _get_iter(self, runner, batch_idx: int) -> int:
        """Get current iteration index.

        Args:
            runner (Runner): The runner of the training/testing/validation
                process.
            batch_idx (int): The iteration index of current
                dataloader. Defaults to None.

        Returns:
            int: The current global iter or inner iter.
        """
        if self.by_epoch:
            current_iter = batch_idx + 1
        else:
            current_iter = runner.iter + 1
        return current_iter

    def _get_epoch(self, runner, mode: str) -> int:
        """Get current epoch according to mode.

        Args:
            runner (Runner): The runner of the training/testing/validation
                process.
            mode (str): Current mode of runner.

        Returns:
            int: The current epoch.
        """
        if mode == 'train':
            epoch = runner.epoch + 1
        elif mode == 'val':
            if (isinstance(runner._train_loop, dict)
                    or runner._train_loop is None):
                epoch = 0
            else:
                # normal val mode
                # runner.epoch += 1 has been done before validation
                epoch = runner.epoch
        else:
            raise ValueError(
                f"runner mode should be 'train' or 'val', but got {mode}")
        return epoch

    def _get_cur_loop(self, runner, mode: str):
        """Get current loop according to mode.

        Args:
            runner (Runner): The runner of the training/validation/testing
                process.
            mode (str): Current mode of runner.

        Returns:
            BaseLoop: Current loop of runner.
        """
        # returns type hint will occur circular import
        if mode == 'train':
            return runner.train_loop
        elif mode == 'val':
            return runner.val_loop
        else:
            return runner.test_loop

    def _get_dataloader_size(self, runner, mode) -> int:
        """Get dataloader size of current loop.

        Args:
            runner (Runner): The runner of the training/validation/testing
            mode (str): Current mode of runner.

        Returns:
            int: The dataloader size of current loop.
        """
        return len(self._get_cur_loop(runner=runner, mode=mode).dataloader)

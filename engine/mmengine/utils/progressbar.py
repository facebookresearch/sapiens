# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import sys
from collections.abc import Iterable
from multiprocessing import Pool
from shutil import get_terminal_size
from typing import Callable, Sequence

from .timer import Timer


class ProgressBar:
    """A progress bar which can print the progress.

    Args:
        task_num (int): Number of total steps. Defaults to 0.
        bar_width (int): Width of the progress bar. Defaults to 50.
        start (bool): Whether to start the progress bar in the constructor.
            Defaults to True.
        file (callable): Progress bar output mode. Defaults to "sys.stdout".

    Examples:
        >>> import mmengine
        >>> import time
        >>> bar = mmengine.ProgressBar(10)
        >>> for i in range(10):
        >>>    bar.update()
        >>>    time.sleep(1)
    """

    def __init__(self,
                 task_num: int = 0,
                 bar_width: int = 50,
                 start: bool = True,
                 file=sys.stdout):
        self.task_num = task_num
        self.bar_width = bar_width
        self.completed = 0
        self.file = file
        if start:
            self.start()

    @property
    def terminal_width(self):
        width, _ = get_terminal_size()
        return width

    def start(self):
        if self.task_num > 0:
            self.file.write(f'[{" " * self.bar_width}] 0/{self.task_num}, '
                            'elapsed: 0s, ETA:')
        else:
            self.file.write('completed: 0, elapsed: 0s')
        self.file.flush()
        self.timer = Timer()

    def update(self, num_tasks: int = 1):
        """update progressbar.

        Args:
            num_tasks (int): Update step size.
        """
        assert num_tasks > 0
        self.completed += num_tasks
        elapsed = self.timer.since_start()
        if elapsed > 0:
            fps = self.completed / elapsed
        else:
            fps = float('inf')
        if self.task_num > 0:
            percentage = self.completed / float(self.task_num)
            eta = int(elapsed * (1 - percentage) / percentage + 0.5)
            msg = f'\r[{{}}] {self.completed}/{self.task_num}, ' \
                  f'{fps:.1f} task/s, elapsed: {int(elapsed + 0.5)}s, ' \
                  f'ETA: {eta:5}s'

            bar_width = min(self.bar_width,
                            int(self.terminal_width - len(msg)) + 2,
                            int(self.terminal_width * 0.6))
            bar_width = max(2, bar_width)
            mark_width = int(bar_width * percentage)
            bar_chars = '>' * mark_width + ' ' * (bar_width - mark_width)
            self.file.write(msg.format(bar_chars))
        else:
            self.file.write(
                f'completed: {self.completed}, elapsed: {int(elapsed + 0.5)}s,'
                f' {fps:.1f} tasks/s')
        self.file.flush()


def track_progress(func: Callable,
                   tasks: Sequence,
                   bar_width: int = 50,
                   file=sys.stdout,
                   **kwargs):
    """Track the progress of tasks execution with a progress bar.

    Tasks are done with a simple for-loop.

    Args:
        func (callable): The function to be applied to each task.
        tasks (Sequence): If tasks is a tuple, it must contain two elements,
            the first being the tasks to be completed and the other being the
            number of tasks. If it is not a tuple, it represents the tasks to
            be completed.
        bar_width (int): Width of progress bar.

    Returns:
        list: The task results.
    """
    if isinstance(tasks, tuple):
        assert len(tasks) == 2
        assert isinstance(tasks[0], Iterable)
        assert isinstance(tasks[1], int)
        task_num = tasks[1]
        tasks = tasks[0]  # type: ignore
    elif isinstance(tasks, Sequence):
        task_num = len(tasks)
    else:
        raise TypeError(
            '"tasks" must be a tuple object or a sequence object, but got '
            f'{type(tasks)}')
    prog_bar = ProgressBar(task_num, bar_width, file=file)
    results = []
    for task in tasks:
        results.append(func(task, **kwargs))
        prog_bar.update()
    prog_bar.file.write('\n')
    return results


def init_pool(process_num, initializer=None, initargs=None):
    if initializer is None:
        return Pool(process_num)
    elif initargs is None:
        return Pool(process_num, initializer)
    else:
        if not isinstance(initargs, tuple):
            raise TypeError('"initargs" must be a tuple')
        return Pool(process_num, initializer, initargs)


def track_parallel_progress(func: Callable,
                            tasks: Sequence,
                            nproc: int,
                            initializer: Callable = None,
                            initargs: tuple = None,
                            bar_width: int = 50,
                            chunksize: int = 1,
                            skip_first: bool = False,
                            keep_order: bool = True,
                            file=sys.stdout):
    """Track the progress of parallel task execution with a progress bar.

    The built-in :mod:`multiprocessing` module is used for process pools and
    tasks are done with :func:`Pool.map` or :func:`Pool.imap_unordered`.

    Args:
        func (callable): The function to be applied to each task.
        tasks (Sequence): If tasks is a tuple, it must contain two elements,
            the first being the tasks to be completed and the other being the
            number of tasks. If it is not a tuple, it represents the tasks to
            be completed.
        nproc (int): Process (worker) number.
        initializer (None or callable): Refer to :class:`multiprocessing.Pool`
            for details.
        initargs (None or tuple): Refer to :class:`multiprocessing.Pool` for
            details.
        chunksize (int): Refer to :class:`multiprocessing.Pool` for details.
        bar_width (int): Width of progress bar.
        skip_first (bool): Whether to skip the first sample for each worker
            when estimating fps, since the initialization step may takes
            longer.
        keep_order (bool): If True, :func:`Pool.imap` is used, otherwise
            :func:`Pool.imap_unordered` is used.

    Returns:
        list: The task results.
    """
    if isinstance(tasks, tuple):
        assert len(tasks) == 2
        assert isinstance(tasks[0], Iterable)
        assert isinstance(tasks[1], int)
        task_num = tasks[1]
        tasks = tasks[0]  # type: ignore
    elif isinstance(tasks, Sequence):
        task_num = len(tasks)
    else:
        raise TypeError(
            '"tasks" must be a tuple object or a sequence object, but got '
            f'{type(tasks)}')
    pool = init_pool(nproc, initializer, initargs)
    start = not skip_first
    task_num -= nproc * chunksize * int(skip_first)
    prog_bar = ProgressBar(task_num, bar_width, start, file=file)
    results = []
    if keep_order:
        gen = pool.imap(func, tasks, chunksize)
    else:
        gen = pool.imap_unordered(func, tasks, chunksize)
    for result in gen:
        results.append(result)
        if skip_first:
            if len(results) < nproc * chunksize:
                continue
            elif len(results) == nproc * chunksize:
                prog_bar.start()
                continue
        prog_bar.update()
    prog_bar.file.write('\n')
    pool.close()
    pool.join()
    return results


def track_iter_progress(tasks: Sequence, bar_width: int = 50, file=sys.stdout):
    """Track the progress of tasks iteration or enumeration with a progress
    bar.

    Tasks are yielded with a simple for-loop.

    Args:
        tasks (Sequence): If tasks is a tuple, it must contain two elements,
            the first being the tasks to be completed and the other being the
            number of tasks. If it is not a tuple, it represents the tasks to
            be completed.
        bar_width (int): Width of progress bar.

    Yields:
        list: The task results.
    """
    if isinstance(tasks, tuple):
        assert len(tasks) == 2
        assert isinstance(tasks[0], Iterable)
        assert isinstance(tasks[1], int)
        task_num = tasks[1]
        tasks = tasks[0]  # type: ignore
    elif isinstance(tasks, Sequence):
        task_num = len(tasks)
    else:
        raise TypeError(
            '"tasks" must be a tuple object or a sequence object, but got '
            f'{type(tasks)}')
    prog_bar = ProgressBar(task_num, bar_width, file=file)
    for task in tasks:
        yield task
        prog_bar.update()
    prog_bar.file.write('\n')

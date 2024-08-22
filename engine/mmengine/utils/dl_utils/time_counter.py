# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import time
from typing import Optional, Union

import torch

from mmengine.dist.utils import master_only
from mmengine.logging import MMLogger, print_log


class TimeCounter:
    """A tool that counts the average running time of a function or a method.
    Users can use it as a decorator or context manager to calculate the average
    running time of code blocks.

    Args:
        log_interval (int): The interval of logging. Defaults to 1.
        warmup_interval (int): The interval of warmup. Defaults to 1.
        with_sync (bool): Whether to synchronize cuda. Defaults to True.
        tag (str, optional): Function tag. Used to distinguish between
            different functions or methods being called. Defaults to None.
        logger (MMLogger, optional): Formatted logger used to record messages.
                Defaults to None.

    Examples:
        >>> import time
        >>> from mmengine.utils.dl_utils import TimeCounter
        >>> @TimeCounter()
        ... def fun1():
        ...     time.sleep(0.1)
        ... fun1()
        [fun1]-time per run averaged in the past 1 runs: 100.0 ms

        >>> @@TimeCounter(log_interval=2, tag='fun')
        ... def fun2():
        ...    time.sleep(0.2)
        >>> for _ in range(3):
        ...    fun2()
        [fun]-time per run averaged in the past 2 runs: 200.0 ms

        >>> with TimeCounter(tag='fun3'):
        ...      time.sleep(0.3)
        [fun3]-time per run averaged in the past 1 runs: 300.0 ms
    """

    instance_dict: dict = dict()

    log_interval: int
    warmup_interval: int
    logger: Optional[MMLogger]
    __count: int
    __pure_inf_time: float

    def __new__(cls,
                log_interval: int = 1,
                warmup_interval: int = 1,
                with_sync: bool = True,
                tag: Optional[str] = None,
                logger: Optional[MMLogger] = None):
        assert warmup_interval >= 1
        if tag is not None and tag in cls.instance_dict:
            return cls.instance_dict[tag]

        instance = super().__new__(cls)
        cls.instance_dict[tag] = instance

        instance.log_interval = log_interval
        instance.warmup_interval = warmup_interval
        instance.with_sync = with_sync
        instance.tag = tag
        instance.logger = logger

        instance.__count = 0
        instance.__pure_inf_time = 0.
        instance.__start_time = 0.

        return instance

    @master_only
    def __call__(self, fn):
        if self.tag is None:
            self.tag = fn.__name__

        def wrapper(*args, **kwargs):
            self.__count += 1

            if self.with_sync and torch.cuda.is_available():
                torch.cuda.synchronize()
            start_time = time.perf_counter()

            result = fn(*args, **kwargs)

            if self.with_sync and torch.cuda.is_available():
                torch.cuda.synchronize()

            elapsed = time.perf_counter() - start_time
            self.print_time(elapsed)

            return result

        return wrapper

    @master_only
    def __enter__(self):
        assert self.tag is not None, 'In order to clearly distinguish ' \
                                     'printing information in different ' \
                                     'contexts, please specify the ' \
                                     'tag parameter'

        self.__count += 1

        if self.with_sync and torch.cuda.is_available():
            torch.cuda.synchronize()
        self.__start_time = time.perf_counter()

    @master_only
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.with_sync and torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - self.__start_time
        self.print_time(elapsed)

    def print_time(self, elapsed: Union[int, float]) -> None:
        """print times per count."""
        if self.__count >= self.warmup_interval:
            self.__pure_inf_time += elapsed

            if self.__count % self.log_interval == 0:
                times_per_count = 1000 * self.__pure_inf_time / (
                    self.__count - self.warmup_interval + 1)
                print_log(
                    f'[{self.tag}]-time per run averaged in the past '
                    f'{self.__count} runs: {times_per_count:.1f} ms',
                    self.logger)

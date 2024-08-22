# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import functools
import multiprocessing as mp
import traceback as tb
from multiprocessing.pool import Pool


class AsyncWorkerExceptionsWrapper:
    def __init__(self, callable):
        self.__callable = callable
        self._logger = mp.log_to_stderr()

    def __call__(self, *args, **kwargs):
        try:
            result = self.__callable(*args, **kwargs)

        except Exception as e:
            self._logger.error(tb.format_exc())
            raise

        # It was fine, give a normal answer
        return result


class WorkerPool(Pool):
    """Worker pool that runs a function on each value that is put().
    This pool is designed that if an exception is thrown in a child, the main process should stop
    as well.
    """

    def __init__(self, func, *args, **kwargs):
        self.func = func
        super().__init__(*args, **kwargs)

    def _result_collector(self, result):
        """
        Collects results from the pool and stores them in a list.
        Args:
            result: The result of the function that was run on the pool.
        """
        if isinstance(result, (list, tuple)):
            self.results.extend(result)
        else:
            self.results.append(result)

    def run(self, iterable, chunksize=1):
        """Runs func on each item in iterable by using either map or starmap asynchronously. Also calls shutdown to finish up.
        Args:
            iterable: Iterable of items to run func on.
            chunksize: Number of items to run func on at once.
        Returns:
            results from the map operation.
        """
        if all(isinstance(x, (list, tuple)) for x in iterable):
            results = self.starmap(self.func, iterable, chunksize)
        else:
            results = self.map(self.func, iterable)
        return results

    def run_async(self, iterable, chunksize=1):
        """Runs func on each item in iterable by using either map or starmap asynchronously. Also calls shutdown to finish up.
        Args:
            iterable: Iterable of items to run func on.
            chunksize: Number of items to run func on at once.
        Returns:
            results from the map operation.
        """
        self.results = []
        if all(isinstance(x, (list, tuple)) for x in iterable):
            self.starmap_async(
                AsyncWorkerExceptionsWrapper(self.func),
                iterable,
                chunksize,
                callback=self._result_collector,
            )
        else:
            self.map_async(
                AsyncWorkerExceptionsWrapper(self.func),
                iterable,
                chunksize,
                callback=self._result_collector,
            )
        return self.results

    def finish(self) -> None:
        """Shutdown the pool and clean-up threads."""
        self.close()
        self.join()

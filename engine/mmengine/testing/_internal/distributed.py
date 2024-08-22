# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) https://github.com/pytorch/pytorch
# Modified from https://github.com/pytorch/pytorch/blob/master/torch/testing/_internal/common_distributed.py  # noqa: E501
import faulthandler
import logging
import multiprocessing
import sys
import tempfile
import threading
import time
import traceback
import types
import unittest
from enum import Enum
from functools import wraps
from typing import NamedTuple
from unittest import TestCase

import torch
from torch.multiprocessing import active_children

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestSkip(NamedTuple):
    exit_code: int
    message: str


TEST_SKIPS = {
    'backend_unavailable':
    TestSkip(10, 'Skipped because distributed backend is not available.'),
    'no_cuda':
    TestSkip(11, 'CUDA is not available.'),
    'multi-gpu-2':
    TestSkip(12, 'Need at least 2 CUDA device'),
    'generic':
    TestSkip(
        13, 'Test skipped at subprocess level, look at subprocess log for '
        'skip reason'),
}

# [How does MultiProcessTestCase work?]
# Each MultiProcessTestCase instance uses 1 + `world_size()` processes, by
# default `world_size()` returns 2. Let's take `test_rpc_spawn.py` as an
# example which inherits from this class. Its `Setup()` methods calls into
# `MultiProcessTestCase._spawn_processes()` which spawns `world_size()`
# subprocesses. During the spawn, the main process passes the test name to
# subprocesses, and the name is acquired from self.id(). The subprocesses
# then use the provided test function name to retrieve the function attribute
# from the test instance and run it. The main process simply waits for all
# subprocesses to join.


class MultiProcessTestCase(TestCase):
    MAIN_PROCESS_RANK = -1

    # This exit code is used to indicate that the test code had an error and
    # exited abnormally. There are certain tests that might use sys.exit() to
    # simulate failures and in those cases, we can't have an exit code of 0,
    # but we still want to ensure we didn't run into any other errors.
    TEST_ERROR_EXIT_CODE = 10

    # do not early terminate for distributed tests.
    def _should_stop_test_suite(self) -> bool:
        return False

    def prepare_subprocess(self):
        pass

    @property
    def world_size(self) -> int:
        return 2

    @property
    def timeout(self) -> int:
        return 1000

    def join_or_run(self, fn):

        @wraps(fn)
        def wrapper(self):
            if self.rank == self.MAIN_PROCESS_RANK:
                self._join_processes(fn)
            else:
                fn()

        return types.MethodType(wrapper, self)

    # The main process spawns N subprocesses that run the test.
    # Constructor patches current instance test method to
    # assume the role of the main process and join its subprocesses,
    # or run the underlying test function.
    def __init__(self, method_name: str = 'runTest') -> None:
        super().__init__(method_name)
        fn = getattr(self, method_name)
        setattr(self, method_name, self.join_or_run(fn))

    def setUp(self) -> None:
        super().setUp()
        self.skip_return_code_checks = []  # type: ignore[var-annotated]
        self.processes = []  # type: ignore[var-annotated]
        self.rank = self.MAIN_PROCESS_RANK
        self.file_name = tempfile.NamedTemporaryFile(delete=False).name
        # pid to pipe consisting of error message from process.
        self.pid_to_pipe = {}  # type: ignore[var-annotated]

    def tearDown(self) -> None:
        super().tearDown()
        for p in self.processes:
            p.terminate()
        # Each Process instance holds a few open file descriptors. The unittest
        # runner creates a new TestCase instance for each test method and keeps
        # it alive until the end of the entire suite. We must thus reset the
        # processes to prevent an effective file descriptor leak.
        self.processes = []

    def _current_test_name(self) -> str:
        # self.id()
        # e.g. '__main__.TestDistributed.TestAdditive.test_get_rank'
        return self.id().split('.')[-1]

    def _start_processes(self, proc) -> None:
        self.processes = []
        for rank in range(int(self.world_size)):
            parent_conn, child_conn = torch.multiprocessing.Pipe()
            process = proc(
                target=self.__class__._run,
                name='process ' + str(rank),
                args=(rank, self._current_test_name(), self.file_name,
                      child_conn),
            )
            process.start()
            self.pid_to_pipe[process.pid] = parent_conn
            self.processes.append(process)

    def _spawn_processes(self) -> None:
        proc = torch.multiprocessing.get_context('spawn').Process
        self._start_processes(proc)

    class Event(Enum):
        GET_TRACEBACK = 1

    @staticmethod
    def _event_listener(parent_pipe, signal_pipe, rank: int):
        while True:
            ready_pipes = multiprocessing.connection.wait(
                [parent_pipe, signal_pipe])

            if parent_pipe in ready_pipes:

                if parent_pipe.closed:
                    return

                event = parent_pipe.recv()

                if event == MultiProcessTestCase.Event.GET_TRACEBACK:
                    # Return traceback to the parent process.
                    with tempfile.NamedTemporaryFile(mode='r+') as tmp_file:
                        faulthandler.dump_traceback(tmp_file)
                        # Flush buffers and seek to read from the beginning
                        tmp_file.flush()
                        tmp_file.seek(0)
                        parent_pipe.send(tmp_file.read())

            if signal_pipe in ready_pipes:
                return

    @classmethod
    def _run(cls, rank: int, test_name: str, file_name: str,
             parent_pipe) -> None:
        self = cls(test_name)
        try:
            self.prepare_subprocess()
        except Exception:
            raise sys.exit(MultiProcessTestCase.TEST_ERROR_EXIT_CODE)
        self.rank = rank
        self.file_name = file_name
        self.run_test(test_name, parent_pipe)

    def run_test(self, test_name: str, parent_pipe) -> None:
        # Start event listener thread.
        signal_recv_pipe, signal_send_pipe = torch.multiprocessing.Pipe(
            duplex=False)
        event_listener_thread = threading.Thread(
            target=MultiProcessTestCase._event_listener,
            args=(parent_pipe, signal_recv_pipe, self.rank),
            daemon=True,
        )
        event_listener_thread.start()

        # self.id() == e.g. '__main__.TestDistributed.test_get_rank'
        # We're retrieving a corresponding test and executing it.
        try:
            getattr(self, test_name)()
        except unittest.SkipTest as se:
            logger.info(f'Process {self.rank} skipping test {test_name} for '
                        f'following reason: {str(se)}')
            sys.exit(TEST_SKIPS['generic'].exit_code)
        except Exception:
            logger.error(
                f'Caught exception: \n{traceback.format_exc()} exiting '
                f'process {self.rank} with exit code: '
                f'{MultiProcessTestCase.TEST_ERROR_EXIT_CODE}')
            # Send error to parent process.
            parent_pipe.send(traceback.format_exc())
            sys.exit(MultiProcessTestCase.TEST_ERROR_EXIT_CODE)
        finally:
            if signal_send_pipe is not None:
                signal_send_pipe.send(None)

            assert event_listener_thread is not None
            event_listener_thread.join()
            # Close pipe after done with test.
            parent_pipe.close()

    def _get_timedout_process_traceback(self) -> None:
        pipes = []
        for i, process in enumerate(self.processes):
            if process.exitcode is None:
                pipe = self.pid_to_pipe[process.pid]
                try:
                    pipe.send(MultiProcessTestCase.Event.GET_TRACEBACK)
                    pipes.append((i, pipe))
                except ConnectionError as e:
                    logger.error(
                        'Encountered error while trying to get traceback '
                        f'for process {i}: {e}')

        # Wait for results.
        for rank, pipe in pipes:
            try:
                # Wait for traceback
                if pipe.poll(5):
                    if pipe.closed:
                        logger.info(
                            f'Pipe closed for process {rank}, cannot retrieve '
                            'traceback')
                        continue

                    traceback = pipe.recv()
                    logger.error(f'Process {rank} timed out with traceback: '
                                 f'\n\n{traceback}')
                else:
                    logger.error('Could not retrieve traceback for timed out '
                                 f'process: {rank}')
            except ConnectionError as e:
                logger.error(
                    'Encountered error while trying to get traceback for '
                    f'process {rank}: {e}')

    def _join_processes(self, fn) -> None:
        start_time = time.time()
        subprocess_error = False
        try:
            while True:
                # check to see if any subprocess exited with an error early.
                for (i, p) in enumerate(self.processes):
                    # This is the exit code processes exit with if they
                    # encountered an exception.
                    if p.exitcode == MultiProcessTestCase.TEST_ERROR_EXIT_CODE:
                        print(
                            f'Process {i} terminated with exit code '
                            f'{p.exitcode}, terminating remaining processes.')
                        _active_children = active_children()
                        for ac in _active_children:
                            ac.terminate()
                        subprocess_error = True
                        break
                if subprocess_error:
                    break
                # All processes have joined cleanly if they all a valid
                # exitcode
                if all([p.exitcode is not None for p in self.processes]):
                    break
                # Check if we should time out the test. If so, we terminate
                # each process.
                elapsed = time.time() - start_time
                if elapsed > self.timeout:
                    self._get_timedout_process_traceback()
                    print(f'Timing out after {self.timeout} seconds and '
                          'killing subprocesses.')
                    for p in self.processes:
                        p.terminate()
                    break
                # Sleep to avoid excessive busy polling.
                time.sleep(0.1)

            elapsed_time = time.time() - start_time

            if fn in self.skip_return_code_checks:
                self._check_no_test_errors(elapsed_time)
            else:
                self._check_return_codes(elapsed_time)
        finally:
            # Close all pipes
            for pid, pipe in self.pid_to_pipe.items():
                pipe.close()

    def _check_no_test_errors(self, elapsed_time) -> None:
        """Checks that we didn't have any errors thrown in the child
        processes."""
        for i, p in enumerate(self.processes):
            if p.exitcode is None:
                raise RuntimeError(
                    'Process {} timed out after {} seconds'.format(
                        i, elapsed_time))
            self.assertNotEqual(self.TEST_ERROR_EXIT_CODE, p.exitcode)

    def _check_return_codes(self, elapsed_time) -> None:
        """Checks that the return codes of all spawned processes match, and
        skips tests if they returned a return code indicating a skipping
        condition."""
        first_process = self.processes[0]
        # first, we check if there are errors in actual processes
        # (via TEST_ERROR_EXIT CODE), and raise an exception for those.
        # the reason we do this is to attempt to raise a more helpful error
        # message than "Process x terminated/timed out"
        # TODO: we should pipe the exception of the failed subprocess here.
        # Currently, the actual exception is displayed as a logging output.
        errored_processes = [
            (i, p) for i, p in enumerate(self.processes)
            if p.exitcode == MultiProcessTestCase.TEST_ERROR_EXIT_CODE
        ]
        if errored_processes:
            error = ''
            for i, process in errored_processes:
                # Get error from pipe.
                error_message = self.pid_to_pipe[process.pid].recv()
                error += (
                    'Process {} exited with error code {} and exception:\n{}\n'
                    .format(i, MultiProcessTestCase.TEST_ERROR_EXIT_CODE,
                            error_message))
            raise RuntimeError(error)
        # If no process exited uncleanly, we check for timeouts, and then
        # ensure each process exited cleanly.
        for i, p in enumerate(self.processes):
            if p.exitcode is None:
                raise RuntimeError(
                    f'Process {i} terminated or timed out after '
                    '{elapsed_time} seconds')

        for skip in TEST_SKIPS.values():
            if first_process.exitcode == skip.exit_code:
                raise unittest.SkipTest(skip.message)

        # Skip the unittest since the raised error maybe not caused by
        # the tested function. For example, in CI environment, the tested
        # method could be terminated by system signal for the limited
        # resources.
        self.skipTest(f'Skip test {self._testMethodName} due to '
                      'the program abort')

    @property
    def is_master(self) -> bool:
        return self.rank == 0

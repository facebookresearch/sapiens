# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import inspect
import logging
import os
import os.path as osp
import sys
import warnings
from getpass import getuser
from logging import Logger, LogRecord, handlers
from socket import gethostname
from typing import Dict, Optional, Union

from termcolor import colored

from mmengine.utils import ManagerMixin
from mmengine.utils.manager import _accquire_lock, _release_lock


class FilterDuplicateWarning(logging.Filter):
    """Filter the repeated warning message.

    Args:
        name (str): name of the filter.
    """

    def __init__(self, name: str = 'mmengine'):
        super().__init__(name)
        self.seen: set = set()

    def filter(self, record: LogRecord) -> bool:
        """Filter the repeated warning message.

        Args:
            record (LogRecord): The log record.

        Returns:
            bool: Whether to output the log record.
        """
        if record.levelno != logging.WARNING:
            return True

        if record.msg not in self.seen:
            self.seen.add(record.msg)
            return True
        return False


class MMFormatter(logging.Formatter):
    """Colorful format for MMLogger. If the log level is error, the logger will
    additionally output the location of the code.

    Args:
        color (bool): Whether to use colorful format. filehandler is not
            allowed to use color format, otherwise it will be garbled.
        blink (bool): Whether to blink the ``INFO`` and ``DEBUG`` logging
            level.
        **kwargs: Keyword arguments passed to
            :meth:`logging.Formatter.__init__`.
    """
    _color_mapping: dict = dict(
        ERROR='red', WARNING='yellow', INFO='white', DEBUG='green')

    def __init__(self, color: bool = True, blink: bool = False, **kwargs):
        super().__init__(**kwargs)
        assert not (not color and blink), (
            'blink should only be available when color is True')
        # Get prefix format according to color.
        error_prefix = self._get_prefix('ERROR', color, blink=True)
        warn_prefix = self._get_prefix('WARNING', color, blink=True)
        info_prefix = self._get_prefix('INFO', color, blink)
        debug_prefix = self._get_prefix('DEBUG', color, blink)

        # Config output format.
        self.err_format = (f'%(asctime)s - %(name)s - {error_prefix} - '
                           '%(pathname)s - %(funcName)s - %(lineno)d - '
                           '%(message)s')
        self.warn_format = (f'%(asctime)s - %(name)s - {warn_prefix} - %('
                            'message)s')
        self.info_format = (f'%(asctime)s - %(name)s - {info_prefix} - %('
                            'message)s')
        self.debug_format = (f'%(asctime)s - %(name)s - {debug_prefix} - %('
                             'message)s')

    def _get_prefix(self, level: str, color: bool, blink=False) -> str:
        """Get the prefix of the target log level.

        Args:
            level (str): log level.
            color (bool): Whether to get colorful prefix.
            blink (bool): Whether the prefix will blink.

        Returns:
            str: The plain or colorful prefix.
        """
        if color:
            attrs = ['underline']
            if blink:
                attrs.append('blink')
            prefix = colored(level, self._color_mapping[level], attrs=attrs)
        else:
            prefix = level
        return prefix

    def format(self, record: LogRecord) -> str:
        """Override the `logging.Formatter.format`` method `. Output the
        message according to the specified log level.

        Args:
            record (LogRecord): A LogRecord instance represents an event being
                logged.

        Returns:
            str: Formatted result.
        """
        if record.levelno == logging.ERROR:
            self._style._fmt = self.err_format
        elif record.levelno == logging.WARNING:
            self._style._fmt = self.warn_format
        elif record.levelno == logging.INFO:
            self._style._fmt = self.info_format
        elif record.levelno == logging.DEBUG:
            self._style._fmt = self.debug_format

        result = logging.Formatter.format(self, record)
        return result


class MMLogger(Logger, ManagerMixin):
    """Formatted logger used to record messages.

    ``MMLogger`` can create formatted logger to log message with different
    log levels and get instance in the same way as ``ManagerMixin``.
    ``MMLogger`` has the following features:

    - Distributed log storage, ``MMLogger`` can choose whether to save log of
      different ranks according to `log_file`.
    - Message with different log levels will have different colors and format
      when displayed on terminal.

    Note:
        - The `name` of logger and the ``instance_name`` of ``MMLogger`` could
          be different. We can only get ``MMLogger`` instance by
          ``MMLogger.get_instance`` but not ``logging.getLogger``. This feature
          ensures ``MMLogger`` will not be incluenced by third-party logging
          config.
        - Different from ``logging.Logger``, ``MMLogger`` will not log warning
          or error message without ``Handler``.

    Examples:
        >>> logger = MMLogger.get_instance(name='MMLogger',
        >>>                                logger_name='Logger')
        >>> # Although logger has name attribute just like `logging.Logger`
        >>> # We cannot get logger instance by `logging.getLogger`.
        >>> assert logger.name == 'Logger'
        >>> assert logger.instance_name = 'MMLogger'
        >>> assert id(logger) != id(logging.getLogger('Logger'))
        >>> # Get logger that do not store logs.
        >>> logger1 = MMLogger.get_instance('logger1')
        >>> # Get logger only save rank0 logs.
        >>> logger2 = MMLogger.get_instance('logger2', log_file='out.log')
        >>> # Get logger only save multiple ranks logs.
        >>> logger3 = MMLogger.get_instance('logger3', log_file='out.log',
        >>>                                 distributed=True)

    Args:
        name (str): Global instance name.
        logger_name (str): ``name`` attribute of ``Logging.Logger`` instance.
            If `logger_name` is not defined, defaults to 'mmengine'.
        log_file (str, optional): The log filename. If specified, a
            ``FileHandler`` will be added to the logger. Defaults to None.
        log_level (str): The log level of the handler. Defaults to
            'INFO'. If log level is 'DEBUG', distributed logs will be saved
            during distributed training.
        file_mode (str): The file mode used to open log file. Defaults to 'w'.
        distributed (bool): Whether to save distributed logs, Defaults to
            false.
        file_handler_cfg (dict, optional): Configuration of file handler.
            Defaults to None. If ``file_handler_cfg`` is not specified,
            ``logging.FileHandler`` will be used by default. If it is
            specified, the ``type`` key should be set. It can be
            ``RotatingFileHandler``, ``TimedRotatingFileHandler``,
            ``WatchedFileHandler`` or other file handlers, and the remaining
            fields will be used to build the handler.

            Examples:
                >>> file_handler_cfg = dict(
                >>>    type='TimedRotatingFileHandler',
                >>>    when='MIDNIGHT',
                >>>    interval=1,
                >>>    backupCount=365)

            `New in version 0.8.5.`
    """

    def __init__(self,
                 name: str,
                 logger_name='mmengine',
                 log_file: Optional[str] = None,
                 log_level: Union[int, str] = 'INFO',
                 file_mode: str = 'w',
                 distributed=False,
                 file_handler_cfg: Optional[dict] = None):
        Logger.__init__(self, logger_name)
        ManagerMixin.__init__(self, name)
        # Get rank in DDP mode.
        if isinstance(log_level, str):
            log_level = logging._nameToLevel[log_level]
        global_rank = _get_rank()
        device_id = _get_device_id()

        # Config stream_handler. If `rank != 0`. stream_handler can only
        # export ERROR logs.
        stream_handler = logging.StreamHandler(stream=sys.stdout)
        # `StreamHandler` record month, day, hour, minute, and second
        # timestamp.
        stream_handler.setFormatter(
            MMFormatter(color=True, datefmt='%m/%d %H:%M:%S'))
        # Only rank0 `StreamHandler` will log messages below error level.
        if global_rank == 0:
            stream_handler.setLevel(log_level)
        else:
            stream_handler.setLevel(logging.ERROR)
        stream_handler.addFilter(FilterDuplicateWarning(logger_name))
        self.handlers.append(stream_handler)

        if log_file is not None:
            world_size = _get_world_size()
            is_distributed = (log_level <= logging.DEBUG
                              or distributed) and world_size > 1
            if is_distributed:
                filename, suffix = osp.splitext(osp.basename(log_file))
                hostname = _get_host_info()
                if hostname:
                    filename = (f'{filename}_{hostname}_device{device_id}_'
                                f'rank{global_rank}{suffix}')
                else:
                    # Omit hostname if it is empty
                    filename = (f'{filename}_device{device_id}_'
                                f'rank{global_rank}{suffix}')
                log_file = osp.join(osp.dirname(log_file), filename)
            # Save multi-ranks logs if distributed is True. The logs of rank0
            # will always be saved.
            if global_rank == 0 or is_distributed:
                if file_handler_cfg is not None:
                    assert 'type' in file_handler_cfg
                    file_handler_type = file_handler_cfg.pop('type')
                    file_handlers_map = _get_logging_file_handlers()
                    if file_handler_type in file_handlers_map:
                        file_handler_cls = file_handlers_map[file_handler_type]
                        file_handler_cfg.setdefault('filename', log_file)
                        file_handler = file_handler_cls(**file_handler_cfg)
                    else:
                        raise ValueError('`logging.handlers` does not '
                                         f'contain {file_handler_type}')
                else:
                    # Here, the default behavior of the official
                    # logger is 'a'. Thus, we provide an interface to
                    # change the file mode to the default behavior.
                    # `FileHandler` is not supported to have colors,
                    # otherwise it will appear garbled.
                    file_handler = logging.FileHandler(log_file, file_mode)

                # `StreamHandler` record year, month, day hour, minute,
                # and second timestamp. file_handler will only record logs
                # without color to avoid garbled code saved in files.
                file_handler.setFormatter(
                    MMFormatter(color=False, datefmt='%Y/%m/%d %H:%M:%S'))
                file_handler.setLevel(log_level)
                file_handler.addFilter(FilterDuplicateWarning(logger_name))
                self.handlers.append(file_handler)
        self._log_file = log_file

    @property
    def log_file(self):
        return self._log_file

    @classmethod
    def get_current_instance(cls) -> 'MMLogger':
        """Get latest created ``MMLogger`` instance.

        :obj:`MMLogger` can call :meth:`get_current_instance` before any
        instance has been created, and return a logger with the instance name
        "mmengine".

        Returns:
            MMLogger: Configured logger instance.
        """
        if not cls._instance_dict:
            cls.get_instance('mmengine')
        return super().get_current_instance()

    def callHandlers(self, record: LogRecord) -> None:
        """Pass a record to all relevant handlers.

        Override ``callHandlers`` method in ``logging.Logger`` to avoid
        multiple warning messages in DDP mode. Loop through all handlers of
        the logger instance and its parents in the logger hierarchy. If no
        handler was found, the record will not be output.

        Args:
            record (LogRecord): A ``LogRecord`` instance contains logged
                message.
        """
        for handler in self.handlers:
            if record.levelno >= handler.level:
                handler.handle(record)

    def setLevel(self, level):
        """Set the logging level of this logger.

        If ``logging.Logger.selLevel`` is called, all ``logging.Logger``
        instances managed by ``logging.Manager`` will clear the cache. Since
        ``MMLogger`` is not managed by ``logging.Manager`` anymore,
        ``MMLogger`` should override this method to clear caches of all
        ``MMLogger`` instance which is managed by :obj:`ManagerMixin`.

        level must be an int or a str.
        """
        self.level = logging._checkLevel(level)
        _accquire_lock()
        # The same logic as `logging.Manager._clear_cache`.
        for logger in MMLogger._instance_dict.values():
            logger._cache.clear()
        _release_lock()


def print_log(msg,
              logger: Optional[Union[Logger, str]] = None,
              level=logging.INFO) -> None:
    """Print a log message.

    Args:
        msg (str): The message to be logged.
        logger (Logger or str, optional): If the type of logger is
        ``logging.Logger``, we directly use logger to log messages.
            Some special loggers are:

            - "silent": No message will be printed.
            - "current": Use latest created logger to log message.
            - other str: Instance name of logger. The corresponding logger
              will log message if it has been created, otherwise ``print_log``
              will raise a `ValueError`.
            - None: The `print()` method will be used to print log messages.
        level (int): Logging level. Only available when `logger` is a Logger
            object, "current", or a created logger instance name.
    """
    if logger is None:
        print(msg)
    elif isinstance(logger, logging.Logger):
        logger.log(level, msg)
    elif logger == 'silent':
        pass
    elif logger == 'current':
        logger_instance = MMLogger.get_current_instance()
        logger_instance.log(level, msg)
    elif isinstance(logger, str):
        # If the type of `logger` is `str`, but not with value of `current` or
        # `silent`, we assume it indicates the name of the logger. If the
        # corresponding logger has not been created, `print_log` will raise
        # a `ValueError`.
        if MMLogger.check_instance_created(logger):
            logger_instance = MMLogger.get_instance(logger)
            logger_instance.log(level, msg)
        else:
            raise ValueError(f'MMLogger: {logger} has not been created!')
    else:
        raise TypeError(
            '`logger` should be either a logging.Logger object, str, '
            f'"silent", "current" or None, but got {type(logger)}')


def _get_world_size():
    """Support using logging module without torch."""
    try:
        # requires torch
        from mmengine.dist import get_world_size
    except ImportError:
        return 1
    else:
        return get_world_size()


def _get_rank():
    """Support using logging module without torch."""
    try:
        # requires torch
        from mmengine.dist import get_rank
    except ImportError:
        return 0
    else:
        return get_rank()


def _get_device_id():
    """Get device id of current machine."""
    try:
        import torch
    except ImportError:
        return 0
    else:
        local_rank = int(os.getenv('LOCAL_RANK', '0'))
        # TODO: return device id of npu and mlu.
        if not torch.cuda.is_available():
            return local_rank
        cuda_visible_devices = os.getenv('CUDA_VISIBLE_DEVICES', None)
        if cuda_visible_devices is None:
            num_device = torch.cuda.device_count()
            cuda_visible_devices = list(range(num_device))
        else:
            cuda_visible_devices = cuda_visible_devices.split(',')
        try:
            return int(cuda_visible_devices[local_rank])
        except ValueError:
            # handle case for Multi-Instance GPUs
            # see #1148 for details
            return cuda_visible_devices[local_rank]


def _get_host_info() -> str:
    """Get hostname and username.

    Return empty string if exception raised, e.g. ``getpass.getuser()`` will
    lead to error in docker container
    """
    host = ''
    try:
        host = f'{getuser()}@{gethostname()}'
    except Exception as e:
        warnings.warn(f'Host or user not found: {str(e)}')
    finally:
        return host


def _get_logging_file_handlers() -> Dict:
    """Get additional file_handlers in ``logging.handlers``.

    Returns:
        Dict: A map of file_handlers.
    """
    file_handlers_map = {}
    for module_name in dir(handlers):
        if module_name.startswith('__'):
            continue
        _fh = getattr(handlers, module_name)
        if inspect.isclass(_fh) and issubclass(_fh, logging.FileHandler):
            file_handlers_map[module_name] = _fh
    return file_handlers_map

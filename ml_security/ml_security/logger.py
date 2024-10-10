import functools
import inspect
import sys
from typing import Callable, Dict

import structlog


def trace(func: Callable) -> Callable:
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        filename = inspect.stack()[1].filename
        function = inspect.stack()[1].function

        kwargs["src_"] = filename
        kwargs["func_"] = function
        return func(*args, **kwargs)

    return wrapper


class Logger:
    def __init__(
        self,
        pre_bound: Dict = {},
    ) -> None:
        self.build_logger("debug")

        self.logger = structlog.get_logger(**pre_bound)

    @trace
    def debug(self, message: str, **kwargs) -> None:
        self.logger.debug(message, **kwargs)

    def info(self, message: str, **kwargs) -> None:
        self.logger.info(message, **kwargs)

    @trace
    def warning(self, message: str, **kwargs) -> None:
        self.logger.warning(message, **kwargs)

    @trace
    def error(self, message: str, **kwargs) -> None:
        self.logger.error(message, **kwargs)

    def build_logger(self, min_level) -> None:
        structlog.configure(
            processors=[
                structlog.contextvars.merge_contextvars,
                structlog.processors.add_log_level,
                structlog.processors.UnicodeDecoder(),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S"),
                structlog.dev.ConsoleRenderer(
                    colors=True,
                ),
            ],
            context_class=dict,
            logger_factory=structlog.PrintLoggerFactory(sys.stdout),
            cache_logger_on_first_use=True,
        )


logger = Logger()

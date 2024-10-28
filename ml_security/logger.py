import logging
import sys
from typing import Dict, List, Literal

import structlog

MODES = Literal["dev", "prod"]
LEVELS = Literal["debug", "info", "warning", "error"]


class Logger:
    def __init__(
        self, mode: MODES = "dev", min_level: LEVELS = "info", pre_bound: Dict = {}
    ) -> None:
        self.mode = mode
        self.min_level = min_level

        if self.mode == "dev":
            self.__build_dev_logger(min_level)
        else:
            self._build_prod_logger(min_level)

        self.logger = structlog.get_logger(**pre_bound)

    def bind(self, **kwargs) -> None:
        """
        Description:
            * Binds the given key-value pairs to the logger.

        Args:
            * kwargs: The key-value pairs to bind to the logger.
        """
        self.logger = self.logger.bind(**kwargs)

    def unbind(self, keys: List[str]) -> None:
        """
        Description:
            * Unbinds the given keys from the logger.

        Args:
            * keys: The list of keys to unbind from the logger.
        """
        self.logger = self.logger.unbind(*keys)

    def debug(self, message: str, **kwargs) -> None:
        """
        Description:
            * Logs a debug message. Debug messages are used for development purposes and are not
              shown in production. They should log information that is useful for debugging.

        Args:
            * message: The message to log.
            * kwargs: The key-value pairs to log.
        """
        self.logger.debug(message, **kwargs)

    def info(self, message: str, **kwargs) -> None:
        """
        Description:
            * Logs an informative message. Informative messages are used to log information that is
              useful for understanding the state of the application or to mark an event.

        Args:
            * message: The message to log.
            * kwargs: The key-value pairs to log.
        """
        self.logger.info(message, **kwargs)

    def warning(self, message: str, **kwargs) -> None:
        """
        Description:
            * Logs a warning message. Warning messages are used to log information that could be used
              to prevent an error from occurring. For instance, losing and reconnecting to a database,
              or a non-critical exception that is still meaningful.

        Args:
            * message: The message to log.
            * kwargs: The key-value pairs to log.
        """
        self.logger.warning(message, **kwargs)

    def error(self, message: str, **kwargs) -> None:
        """
        Description:
            * Logs an error message. Error messages are used to log information about an error that
              occurred in the application. This kind of logs should be used to wake people up and
              trigger alerts. An example of an error message is a database connection error.

        Args:
            * message: The message to log.
            * kwargs: The key-value pairs to log.
        """
        self.logger.error(message, **kwargs)

    def __build_dev_logger(self, min_level: LEVELS) -> None:
        """
        Description:
            * Configures the logger for development mode. In development mode, the logger is more verbose
              and logs are colored and well formatted.

        Args:
            * min_level: The minimum log level to log.
        """
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
            wrapper_class=structlog.make_filtering_bound_logger(
                self.__literal_to_level(min_level)
            ),
            cache_logger_on_first_use=True,
        )

    def _build_prod_logger(self, min_level: LEVELS) -> None:
        """
        Description:
            * Configures the logger for production mode. In production mode, the logger is less verbose
              and logs follow the "logfmt" format.

        Args:
            * min_level: The minimum log level to log.
        """
        structlog.configure(
            processors=[
                structlog.contextvars.merge_contextvars,
                structlog.processors.add_log_level,
                structlog.processors.UnicodeDecoder(),
                structlog.processors.EventRenamer("msg"),
                structlog.processors.TimeStamper(
                    fmt="%Y-%m-%d %H:%M:%S",
                ),
                structlog.processors.KeyValueRenderer(
                    sort_keys=True,
                    key_order=["level", "msg", "timestamp"],
                    drop_missing=False,
                ),
            ],
            context_class=dict,
            logger_factory=structlog.PrintLoggerFactory(sys.stdout),
            wrapper_class=structlog.make_filtering_bound_logger(
                self.__literal_to_level(min_level)
            ),
            cache_logger_on_first_use=True,
        )

    def __literal_to_level(self, level: LEVELS) -> int:
        """
        Description:
            * Converts the given log level literal to the corresponding integer value. The logging library
              does not expose this method.

        Args:
            * level: The log level literal to convert.
        """
        return getattr(logging, level.upper())


logger = Logger()

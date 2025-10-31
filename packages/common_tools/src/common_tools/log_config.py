import logging
import sys
from typing import List

# Mapping of log4j levels to Python logging levels
LOG4J_TO_PYTHON_LEVELS = {
    "TRACE": "DEBUG",
    "DEBUG": "DEBUG",
    "INFO": "INFO",
    "WARN": "WARNING",
    "WARNING": "WARNING",
    "ERROR": "ERROR",
    "FATAL": "CRITICAL",
    "CRITICAL": "CRITICAL",
}


def _setup_logging(level_name: str) -> None:
    """Helper to configure root logger to specified level."""
    numeric_level = getattr(logging, level_name, None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level name: {level_name}")

    log = logging.getLogger()
    log.setLevel(numeric_level)

    if log.hasHandlers():
        log.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(numeric_level)
    formatter = logging.Formatter("%(levelname)s: %(message)s")
    handler.setFormatter(formatter)
    log.addHandler(handler)


def configure_logging_from_argv(default_level: str = "WARNING") -> List[str]:
    """
    Configures logging based on '--log LEVEL' in sys.argv and returns remaining args.

    Args:
        default_level (str): Default log level if '--log' not provided.

    Returns:
        List[str]: Command-line arguments excluding '--log' and its level value.

    Raises:
        ValueError: If log level is missing or invalid.
    """

    args = sys.argv[1:]
    cleaned_args = args.copy()
    log_level_str = default_level.upper()

    if "--log" in args:
        log_index = args.index("--log")
        if log_index + 1 >= len(args):
            raise ValueError("Missing log level after '--log'")
        potential_level = args[log_index + 1].upper()
        mapped_level = LOG4J_TO_PYTHON_LEVELS.get(potential_level)
        if not mapped_level:
            raise ValueError(f"Invalid log level '{potential_level}' after '--log'")
        log_level_str = mapped_level
        # Remove '--log' and the level value from args once
        del cleaned_args[log_index : log_index + 2]

    _setup_logging(log_level_str)
    logging.debug(f"Logging configured to level: {log_level_str}")

    return cleaned_args

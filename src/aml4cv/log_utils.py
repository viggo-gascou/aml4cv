"""Utility functions for logging things beautifully."""

import logging

from termcolor import colored

logger = logging.getLogger("aml4cv")


def log(message: str, level: int = logging.INFO, colour: str | None = None) -> None:
    """Log a message.

    Args:
        message:
            The message to log.
        level:
            The logging level. Defaults to logging.INFO.
        colour:
            The colour to use for the message. If None, a default colour will be used
            based on the logging level.

    Raises:
        ValueError:
            If the logging level is invalid.
    """
    match level:
        case logging.DEBUG:
            message = colored(
                text=message,
                color=colour or "light_blue",
            )
            logger.debug(message)
        case logging.INFO:
            if colour is not None:
                message = colored(text=message, color="white")
            logger.info(message)
        case logging.WARNING:
            message = colored(text=message, color=colour or "yellow")
            logger.warning(message)
        case logging.ERROR:
            message = colored(text=message, color=colour or "red")
            logger.error(message)
        case logging.CRITICAL:
            message = colored(text=message, color=colour or "red", attrs=["bold"])
            logger.critical(message)
        case _:
            raise ValueError(f"Invalid logging level: {level}")


def set_logging_level(level: int) -> int:
    """Set the logging level.

    Args:
        level:
            The logging level.

    Returns:
        The logging level that was set.
    """
    logger.setLevel(level)
    return level

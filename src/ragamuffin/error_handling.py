import logging
import sys
from collections.abc import Callable
from functools import wraps
from typing import Any

from ragamuffin.settings import get_settings

logger = logging.getLogger(__name__)


class MuffinError(Exception):
    pass


class ConfigurationError(MuffinError):
    pass


def exit_on_error(func: Callable) -> Callable:
    """Exit Ragamuffin if an error occurs."""

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        debug_mode = get_settings().get("debug_mode")
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if debug_mode:
                raise
            logger.error(e, exc_info=False)
            logger.error("Exiting due to error. Use RAGAMUFFIN_DEBUG=1 for more information.")
            sys.exit(1)

    return wrapper


def ensure_string(value: Any) -> str:
    """Ensure that the value is a string."""
    if not isinstance(value, str):
        raise ConfigurationError(f"Expected a string but got {value}")
    return value


def ensure_int(value: Any) -> int:
    """Ensure that the value is an integer."""
    if not isinstance(value, int):
        raise ConfigurationError(f"Expected an integer but got {value}")
    return value

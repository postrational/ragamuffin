import os
import random
from collections.abc import Callable
from contextlib import contextmanager
from functools import wraps
from typing import Any

import numpy as np
import torch


def seed(seed_value: int):
    """Decorator to set RNG seed values for reproducibility."""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Save current RNG states
            prev_random_state = random.getstate()
            prev_numpy_state = np.random.get_state()
            prev_torch_state = torch.get_rng_state()

            # Set the provided seed
            random.seed(seed_value)
            np.random.seed(seed_value)
            torch.manual_seed(seed_value)

            try:
                # Run the test function
                result = func(*args, **kwargs)
            finally:
                # Reset RNG states
                random.setstate(prev_random_state)
                np.random.set_state(prev_numpy_state)
                torch.set_rng_state(prev_torch_state)

            return result

        return wrapper

    return decorator


@contextmanager
def _env_vars(vars: dict[str, str]):
    """Context manager which temporarily sets environment variables."""
    original_values = {key: os.getenv(key) for key in vars}

    try:
        for key, value in vars.items():
            os.environ[key] = value
        yield
    finally:
        for key, original_value in original_values.items():
            if original_value is None:
                del os.environ[key]
            else:
                os.environ[key] = original_value


def env_vars(func: Callable | None = None, **vars) -> Any:
    """Can be used as both a context manager and a decorator to set environment variables.
    Usage:
      - As a decorator: @env_vars(VAR1="value1", VAR2="value2")
      - As a context manager: with env_vars(VAR1="value1", VAR2="value2"):
    """
    if func is None:
        # Used as a context manager
        return _env_vars(vars)

    # Used as a decorator
    @wraps(func)
    def wrapper(*args, **kwargs):
        with _env_vars(vars):
            return func(*args, **kwargs)

    return wrapper

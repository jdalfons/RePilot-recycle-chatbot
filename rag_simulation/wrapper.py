import functools
from datetime import datetime
from typing import Callable, TypeVar, Any

# Define a generic type for the function being wrapped
F = TypeVar("F", bound=Callable[..., Any])


def track_latency(func: F) -> Callable[..., dict[str, Any]]:
    """
    Wrapper to track the execution latency of a function.

    Args:
        func (Callable): The function to measure latency for.

    Returns:
        Callable: A wrapped function that returns a dictionary containing:
            - "result": The original function's result.
            - "latency_ms": The latency in milliseconds.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> dict[str, Any]:
        """
        The wrapper function that tracks the latency of the original function.

        Args:
            *args: Positional arguments passed to the original function.
            **kwargs: Keyword arguments passed to the original function.

        Returns:
            dict[str, Any]: A dictionary containing the function's result and latency.
        """
        start_time = datetime.now()
        result = func(*args, **kwargs)
        end_time = datetime.now()

        # Calculate latency in milliseconds
        latency_ms = (end_time - start_time).total_seconds() * 1000  # ms

        return {
            "result": result,
            "latency_ms": latency_ms,
        }

    return wrapper

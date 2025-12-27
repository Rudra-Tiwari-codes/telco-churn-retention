"""
Retry utilities with exponential backoff for resilient operations.

This module provides decorators for retrying failed operations with configurable
backoff strategies, useful for handling transient failures in database connections,
API calls, and other external service interactions.
"""

from __future__ import annotations

import functools
import logging
import random
import time
from collections.abc import Callable
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


def retry_with_backoff(
    max_retries: int = 3,
    initial_delay: float = 0.5,
    max_delay: float = 10.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    exceptions: tuple[type[Exception], ...] = (Exception,),
) -> Callable[[F], F]:
    """Decorator for retrying a function with exponential backoff.

    This decorator wraps a function and retries it on failure with configurable
    exponential backoff. It's useful for handling transient failures when
    connecting to external services like Redis, Postgres, Kafka, etc.

    Args:
        max_retries: Maximum number of retry attempts. Default is 3.
        initial_delay: Initial delay between retries in seconds. Default is 0.5.
        max_delay: Maximum delay between retries in seconds. Default is 10.0.
        exponential_base: Base for exponential backoff calculation. Default is 2.0.
        jitter: If True, adds random jitter to delay to prevent thundering herd.
        exceptions: Tuple of exception types to catch and retry on.

    Returns:
        Decorated function with retry logic.

    Example:
        @retry_with_backoff(max_retries=3, initial_delay=1.0, exceptions=(ConnectionError,))
        def connect_to_database():
            # Connection logic that might fail transiently
            pass
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exception: Exception | None = None
            delay = initial_delay

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e

                    if attempt == max_retries:
                        logger.error(
                            f"Function {func.__name__} failed after {max_retries + 1} attempts. "
                            f"Last error: {e}"
                        )
                        raise

                    # Calculate delay with optional jitter
                    if jitter:
                        actual_delay = delay * (0.5 + random.random())
                    else:
                        actual_delay = delay

                    logger.warning(
                        f"Function {func.__name__} failed (attempt {attempt + 1}/{max_retries + 1}): {e}. "
                        f"Retrying in {actual_delay:.2f}s..."
                    )

                    time.sleep(actual_delay)

                    # Increase delay for next attempt (exponential backoff)
                    delay = min(delay * exponential_base, max_delay)

            # This should never be reached, but just in case
            if last_exception:
                raise last_exception

        return wrapper  # type: ignore[return-value]

    return decorator


class RetryError(Exception):
    """Exception raised when all retry attempts have been exhausted."""

    def __init__(self, message: str, last_exception: Exception | None = None) -> None:
        """Initialize RetryError.

        Args:
            message: Error message.
            last_exception: The last exception that caused the retry to fail.
        """
        super().__init__(message)
        self.last_exception = last_exception

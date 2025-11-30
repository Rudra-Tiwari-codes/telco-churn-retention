"""Centralized logging configuration."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path


def setup_logging(
    level: str | int | None = None,
    log_file: Path | str | None = None,
    format_string: str | None = None,
) -> None:
    """Configure application-wide logging.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL) or int.
               Defaults to INFO, or from LOG_LEVEL env var.
        log_file: Optional path to log file. If None, logs only to console.
        format_string: Optional custom format string. Defaults to standard format.
    """
    # Determine log level
    if level is None:
        level_str = os.getenv("LOG_LEVEL", "INFO").upper()
        level = getattr(logging, level_str, logging.INFO)
    elif isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    # Default format
    if format_string is None:
        format_string = (
            "%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s"
        )

    # Configure handlers
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path))

    # Configure root logger
    logging.basicConfig(
        level=level,
        format=format_string,
        handlers=handlers,
        force=True,  # Override any existing configuration
    )

    # Set specific logger levels if needed
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("fastapi").setLevel(logging.INFO)


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the given name.

    Args:
        name: Logger name (typically __name__).

    Returns:
        Logger instance.
    """
    return logging.getLogger(name)

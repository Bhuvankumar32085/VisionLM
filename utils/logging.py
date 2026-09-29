"""Logging configuration utilities."""

import logging
import sys
from typing import Optional


def get_logger(name: str = "custom_vlm", log_file: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    """Configures and returns a logger instance.

    Args:
        name: Name of the logger.
        log_file: Optional file path to write logs to.
        level: Logging level (default INFO).

    Returns:
        logging.Logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers if called multiple times
    if not logger.handlers:
        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        if log_file:
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger

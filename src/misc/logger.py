"""Logging utilities for FP-DETR."""

import logging
import sys


def get_logger(name: str = "fp_detr", level: int = logging.INFO) -> logging.Logger:
    """Return a logger with a consistent format."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        fmt = "[%(asctime)s %(name)s %(levelname)s] %(message)s"
        handler.setFormatter(logging.Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S"))
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger

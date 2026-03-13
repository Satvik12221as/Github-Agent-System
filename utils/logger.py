# utils/logger.py

import logging
import sys
from datetime import datetime

def get_logger(name: str) -> logging.Logger:
    """
    Creates a logger for any module that asks for one.
    Pass in __name__ and it labels logs with the file they came from.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Avoid adding duplicate handlers if logger already exists
    if logger.handlers:
        return logger

    # Create a handler that prints to terminal
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)

    # Format: [2026-03-13 14:32:01] code_reader - INFO - Fetching issue...
    formatter = logging.Formatter(
        '[%(asctime)s] %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger
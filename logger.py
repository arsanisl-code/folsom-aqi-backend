"""
logger.py — Shared structured logging factory for the Folsom AQI Forecast system.

Usage in every module:
    from logger import get_logger
    log = get_logger(__name__)

    log.info("Forecast cached → %s", path)
    log.warning("FIRMS_MAP_KEY missing. Wildfire features will be 0.")
    log.error("Fetch failed: %s", exc, exc_info=True)
"""

import logging
import os
import sys


class _LevelFilter(logging.Filter):
    """
    Route log records by level range so that INFO and below go to stdout
    and WARNING and above go to stderr — matching the original print() behavior
    where informational messages went to stdout and errors to sys.stderr.
    """

    def __init__(self, min_level: int = logging.DEBUG, max_level: int = logging.CRITICAL):
        super().__init__()
        self.min_level = min_level
        self.max_level = max_level

    def filter(self, record: logging.LogRecord) -> bool:
        return self.min_level <= record.levelno <= self.max_level


def get_logger(name: str, level: int | None = None) -> logging.Logger:
    """
    Return a configured Logger for the given module name.

    Routing:
      - DEBUG and INFO  → stdout
      - WARNING, ERROR, CRITICAL → stderr

    Format:
      2025-07-15T14:32:01 INFO     backend.inference — Fetching recent data...

    Level resolution (highest priority first):
      1. `level` parameter if explicitly passed
      2. LOG_LEVEL environment variable (e.g. LOG_LEVEL=DEBUG)
      3. INFO (default)

    Idempotent: calling get_logger("foo") twice returns the same logger
    with handlers attached only once.
    """
    logger = logging.getLogger(name)

    # Idempotency guard — only configure once per logger name
    if logger.handlers:
        return logger

    # Resolve effective level
    if level is None:
        env_level = os.environ.get("LOG_LEVEL", "INFO").upper()
        level = getattr(logging, env_level, logging.INFO)

    logger.setLevel(level)

    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    # stdout handler: DEBUG and INFO only
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)
    stdout_handler.addFilter(_LevelFilter(min_level=logging.DEBUG, max_level=logging.INFO))

    # stderr handler: WARNING, ERROR, CRITICAL
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setFormatter(formatter)
    stderr_handler.addFilter(_LevelFilter(min_level=logging.WARNING, max_level=logging.CRITICAL))

    logger.addHandler(stdout_handler)
    logger.addHandler(stderr_handler)

    # Prevent log records from propagating to the root logger, which would
    # cause duplicate output if the root logger also has handlers configured.
    logger.propagate = False

    return logger

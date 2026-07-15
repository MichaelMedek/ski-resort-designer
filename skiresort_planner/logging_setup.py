"""Logging setup: configure ONLY our package logger, never root.

Root config (`basicConfig`) would cascade our level to every library — at DEBUG.
`propagate=False` keeps our level to our code; libraries keep their default (WARNING+ still reaches stderr).

    SKIRESORT_LOG_LEVEL   default INFO; set DEBUG for per-rerun/click/transition detail.
"""

import logging
import os

ENV_LEVEL = "SKIRESORT_LOG_LEVEL"
PACKAGE_LOGGER = "skiresort_planner"


def configure_logging() -> logging.Logger:
    """Configure and return the package logger: level from SKIRESORT_LOG_LEVEL, one handler (idempotent).

    Returned so the entry script (app.py, whose __name__ is "__main__") can log under our hierarchy
    via `configure_logging().getChild("app")` instead of a hardcoded name.
    """
    logger = logging.getLogger(PACKAGE_LOGGER)
    logger.setLevel(os.environ.get(ENV_LEVEL, "INFO").upper())
    logger.propagate = False  # our level applies to our code alone — root/libraries stay at their default

    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s", datefmt="%H:%M:%S"))
        logger.addHandler(handler)
    return logger

"""Logging setup: configure ONLY our package logger, never root.

Root config (`basicConfig`) would cascade our level to every library — at DEBUG.
`propagate=False` keeps our level to our code; libraries keep their default (WARNING+ still reaches stderr).

    SKIRESORT_LOG_LEVEL   default INFO; set DEBUG for per-rerun/click/transition detail.

Logs go to stderr AND a timestamped file under LOG_DIR (next to data/backups), so packaged-app
users (no visible terminal) still have a log to inspect or send when something breaks.
"""

import logging
import os
from datetime import datetime

from skiresort_planner.constants import LOG_DIR

ENV_LEVEL = "SKIRESORT_LOG_LEVEL"
PACKAGE_LOGGER = "skiresort_planner"

_FORMATTER = logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s", datefmt="%H:%M:%S")


def configure_logging() -> logging.Logger:
    """Configure and return the package logger: level from SKIRESORT_LOG_LEVEL, stderr + file handlers
    (idempotent).

    Returned so the entry script (app.py, whose __name__ is "__main__") can log under our hierarchy
    via `configure_logging().getChild("app")` instead of a hardcoded name.
    """
    logger = logging.getLogger(PACKAGE_LOGGER)
    logger.setLevel(os.environ.get(ENV_LEVEL, "INFO").upper())
    logger.propagate = False  # our level applies to our code alone — root/libraries stay at their default

    if not logger.handlers:
        stream = logging.StreamHandler()
        stream.setFormatter(_FORMATTER)
        logger.addHandler(stream)

        # One timestamped file per run, alongside data/backups (same mkdir contract as those dirs).
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        log_file = LOG_DIR / f"skiresort_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(_FORMATTER)
        logger.addHandler(file_handler)
    return logger

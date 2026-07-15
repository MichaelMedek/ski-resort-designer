"""Central logging configuration for the Ski Resort Planner.

`configure_logging()` is called once at app startup (app.py). It reads one environment variable:

    SKIRESORT_LOG_LEVEL   Root log level, default INFO (case-insensitive: DEBUG/INFO/WARNING/ERROR).
                          Set DEBUG to surface the per-rerun/per-click detail INFO suppresses.

Capture a run to a file by redirecting: `streamlit run skiresort_planner/app.py > run.log 2>&1`.

Level policy: Streamlit re-runs the whole script on every interaction, so render/dispatch/click/
transition events are logged at DEBUG to keep the default INFO console readable. INFO is for
once-per-action milestones (resort loaded, OSM import summary, autosave); WARNING/ERROR mark real
problems with identifying context.
"""

import logging
import os

ENV_LEVEL = "SKIRESORT_LOG_LEVEL"


def configure_logging() -> None:
    """Configure root logging from SKIRESORT_LOG_LEVEL (default INFO).

    Idempotent: logging.basicConfig no-ops if the root logger already has handlers, so repeated
    Streamlit reruns don't stack handlers.
    """
    logging.basicConfig(
        level=os.environ.get(ENV_LEVEL, "INFO").upper(),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

"""Unit tests for central logging configuration (logging_setup.py).

configure_logging() configures ONLY our package logger ("skiresort_planner"), never the root logger,
so a DEBUG level applies to our code alone and never cascades to noisy third-party libraries. These
tests drive the real logger and assert its level, propagate flag, single handler, and — crucially —
that a sibling library logger is left untouched.
"""

import logging

import pytest

from skiresort_planner import logging_setup


@pytest.fixture(autouse=True)
def _reset_package_logger() -> "object":
    """Reset our package logger between tests so handler/level state doesn't leak."""
    logger = logging.getLogger(logging_setup.PACKAGE_LOGGER)
    saved_handlers, saved_level, saved_propagate = logger.handlers[:], logger.level, logger.propagate
    logger.handlers.clear()
    yield
    logger.handlers[:] = saved_handlers
    logger.setLevel(saved_level)
    logger.propagate = saved_propagate


def test_default_level_is_info(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(logging_setup.ENV_LEVEL, raising=False)
    logging_setup.configure_logging()
    assert logging.getLogger(logging_setup.PACKAGE_LOGGER).level == logging.INFO


def test_env_override_is_upper_cased(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(logging_setup.ENV_LEVEL, "debug")
    logging_setup.configure_logging()
    assert logging.getLogger(logging_setup.PACKAGE_LOGGER).level == logging.DEBUG


def test_invalid_level_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    # Raise-fast: a typo'd level goes straight to setLevel, which rejects unknown names with
    # ValueError, rather than us validating/swallowing it.
    monkeypatch.setenv(logging_setup.ENV_LEVEL, "verbose")
    with pytest.raises(ValueError, match="Unknown level"):
        logging_setup.configure_logging()


def test_configures_only_our_package_not_root_or_libraries(monkeypatch: pytest.MonkeyPatch) -> None:
    # The whole point: DEBUG on our logger must NOT reach root or a third-party logger like rasterio.
    monkeypatch.setenv(logging_setup.ENV_LEVEL, "DEBUG")
    root_level_before = logging.getLogger().level
    logging_setup.configure_logging()

    ours = logging.getLogger(logging_setup.PACKAGE_LOGGER)
    assert ours.propagate is False, "must not propagate to root, or libraries' handlers would see our config"
    assert logging.getLogger().level == root_level_before, "root logger left untouched"
    # A library logger has no DEBUG-emitting handler from us — its effective level stays WARNING+.
    assert logging.getLogger("rasterio").getEffectiveLevel() >= logging.WARNING


def test_returns_the_package_logger(monkeypatch: pytest.MonkeyPatch) -> None:
    # Returned so app.py (whose __name__ is "__main__") can derive its child without a hardcoded name.
    monkeypatch.setenv(logging_setup.ENV_LEVEL, "INFO")
    returned = logging_setup.configure_logging()
    assert returned is logging.getLogger(logging_setup.PACKAGE_LOGGER)
    assert returned.getChild("app").name == "skiresort_planner.app"


def test_idempotent_no_duplicate_handlers(monkeypatch: pytest.MonkeyPatch) -> None:
    # Streamlit reruns call configure_logging() repeatedly; our handler must not stack. (We assert
    # the count doesn't GROW rather than == 1, since pytest's caplog attaches its own handlers here.)
    monkeypatch.setenv(logging_setup.ENV_LEVEL, "INFO")
    logging_setup.configure_logging()
    count_after_first = len(logging.getLogger(logging_setup.PACKAGE_LOGGER).handlers)
    logging_setup.configure_logging()
    logging_setup.configure_logging()
    assert len(logging.getLogger(logging_setup.PACKAGE_LOGGER).handlers) == count_after_first

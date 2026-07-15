"""Unit tests for central logging configuration (logging_setup.py).

configure_logging() is a thin wrapper over logging.basicConfig; its only logic is the level string
it computes from SKIRESORT_LOG_LEVEL (default INFO, upper-cased) and forwards. These tests spy on
basicConfig to assert that forwarded value — cleaner and more direct than driving the real root
logger, which pytest's own log-capture handler interferes with.
"""

import logging

import pytest

from skiresort_planner import logging_setup


@pytest.fixture
def basicconfig_spy(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    """Replace logging.basicConfig with a recorder; return the dict of captured kwargs."""
    captured: dict[str, object] = {}
    monkeypatch.setattr(logging, "basicConfig", lambda **kwargs: captured.update(kwargs))
    return captured


def test_default_level_is_info(monkeypatch: pytest.MonkeyPatch, basicconfig_spy: dict[str, object]) -> None:
    monkeypatch.delenv(logging_setup.ENV_LEVEL, raising=False)
    logging_setup.configure_logging()
    assert basicconfig_spy["level"] == "INFO"


def test_env_override_is_upper_cased(monkeypatch: pytest.MonkeyPatch, basicconfig_spy: dict[str, object]) -> None:
    monkeypatch.setenv(logging_setup.ENV_LEVEL, "debug")
    logging_setup.configure_logging()
    assert basicconfig_spy["level"] == "DEBUG"


def test_invalid_level_forwarded_unmodified(
    monkeypatch: pytest.MonkeyPatch, basicconfig_spy: dict[str, object]
) -> None:
    # Raise-fast: we forward a typo'd value straight to basicConfig (which rejects unknown names
    # with ValueError) rather than validating/swallowing it ourselves.
    monkeypatch.setenv(logging_setup.ENV_LEVEL, "verbose")
    logging_setup.configure_logging()
    assert basicconfig_spy["level"] == "VERBOSE"

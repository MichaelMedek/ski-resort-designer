"""Unit tests for the Nominatim place geocoder (generators/geocoder.py).

We monkeypatch requests.get so no network is hit: assert the top match is parsed into a
GeocodeResult, and that empty results / network errors yield None (the UI shows a message
rather than crashing).
"""

import pytest
import requests

from skiresort_planner.generators import geocoder


class _FakeResponse:
    def __init__(self, payload: list[dict[str, str]]) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        pass

    def json(self) -> list[dict[str, str]]:
        return self._payload


def test_returns_top_match(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = [
        {"lat": "47.0125", "lon": "10.2916", "display_name": "Ischgl, Tyrol, Austria"},
        {"lat": "0.0", "lon": "0.0", "display_name": "second, ignored"},
    ]
    monkeypatch.setattr("skiresort_planner.generators.geocoder.requests.get", lambda *a, **k: _FakeResponse(payload))

    result = geocoder.geocode("Ischgl")

    assert result is not None
    assert result.lat == 47.0125
    assert result.lon == 10.2916
    assert result.display_name == "Ischgl, Tyrol, Austria"


def test_no_results_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("skiresort_planner.generators.geocoder.requests.get", lambda *a, **k: _FakeResponse([]))
    assert geocoder.geocode("asdfqwer") is None


def test_network_error_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(*a: object, **k: object) -> _FakeResponse:
        raise requests.RequestException("connection refused")

    monkeypatch.setattr("skiresort_planner.generators.geocoder.requests.get", _boom)
    assert geocoder.geocode("Ischgl") is None


def test_blank_query_skips_request(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(*a: object, **k: object) -> _FakeResponse:
        raise AssertionError("must not hit the network for a blank query")

    monkeypatch.setattr("skiresort_planner.generators.geocoder.requests.get", _boom)
    assert geocoder.geocode("   ") is None


def test_request_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    from skiresort_planner.constants import OSMConfig

    captured: dict[str, object] = {}

    def _capture(*args: object, **kwargs: object) -> _FakeResponse:
        captured["args"] = args
        captured["kwargs"] = kwargs
        return _FakeResponse([{"lat": "1.0", "lon": "2.0", "display_name": "x"}])

    monkeypatch.setattr("skiresort_planner.generators.geocoder.requests.get", _capture)

    geocoder.geocode("Ischgl")

    args = captured["args"]
    kwargs = captured["kwargs"]
    assert isinstance(args, tuple)
    assert isinstance(kwargs, dict)
    assert args[0] == OSMConfig.NOMINATIM_URL
    assert kwargs["params"] == {"q": "Ischgl", "format": "json", "limit": "1"}
    assert kwargs["headers"]["User-Agent"] == OSMConfig.USER_AGENT
    assert kwargs["timeout"] == OSMConfig.NOMINATIM_TIMEOUT_S

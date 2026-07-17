"""Unit tests for the OpenStreetMap importer base + lift-only child (generators/osm_importer.py,
generators/osm_lift_importer.py).

The importer takes GEOMETRY ONLY from OSM; elevation/pylons are recomputed. These tests feed
synthetic Overpass `elements` + a controllable fake DEM (with a nodata hole) and assert: the
aerialway→lift-type map (mapping-only; unmapped values are silently ignored), named-only filter
(unnamed lifts skipped), min-length gate, out-of-box / over-nodata skips, longest-per-name dedup,
the fetch/retry classification, and bbox_around. Slope geometry lives in the connected-graph
builder (test_osm_import_rules.py), not here.
"""

from skiresort_planner.core.dem_service import DEMService
from skiresort_planner.generators.osm_importer import ImportResult, OverpassElement
from skiresort_planner.generators.osm_lift_importer import LiftOnlyImporter

M = 111320.0  # metres per degree near the equator


class _FakeDEM(DEMService):
    """20% south slope; a nodata hole for lon > `hole_lon` (returns None there)."""

    def __new__(cls, hole_lon: float = 999.0) -> "_FakeDEM":
        return object.__new__(cls)

    def __init__(self, hole_lon: float = 999.0) -> None:
        self.hole_lon = hole_lon

    def get_elevation(self, lon: float, lat: float) -> float | None:
        if lon > self.hole_lon:
            return None
        return 2500.0 - lat * M * 0.20


# Import box (min_lon, min_lat, max_lon, max_lat): a wide box around the test geometry so every
# normal test vertex is well inside; the truncation test uses a vertex far outside it.
BBOX = (-0.1, -0.11, 0.1, 0.09)


def _way(tags: dict[str, str], verts: list[tuple[float, float]]) -> OverpassElement:
    """Build a synthetic Overpass way element with inline geometry."""
    return {"type": "way", "tags": tags, "geometry": [{"lon": x, "lat": y} for x, y in verts]}


def _lifts(elements: list[OverpassElement], dem: _FakeDEM | None = None) -> ImportResult:
    """Assemble a lift-only import (no network) from synthetic elements."""
    return LiftOnlyImporter(dem=dem or _FakeDEM(), bbox=BBOX)._assemble(elements)


class TestLiftOnlyResult:
    def test_slope_chains_always_empty(self) -> None:
        result = _lifts([_way({"aerialway": "gondola", "name": "Gipfelbahn"}, [(0.0, 0.0), (0.0, -0.02)])])
        assert result.slope_chains == [], "lift-only import never produces slopes"
        assert result.source == "OSM"


class TestNamedOnly:
    def test_unnamed_lift_skipped(self) -> None:
        result = _lifts([_way({"aerialway": "chair_lift"}, [(0.0, 0.0), (0.0, -0.02)])])
        assert result.lifts == [] and result.skipped == 1


class TestFetch:
    """fetch() runs ONE Overpass query; on a transient 429/504 it waits for a free slot and retries
    once. _is_transient classifies which errors are worth that retry; _seconds_until_free_slot reads
    the wait from /api/status text.
    """

    def test_transient_classification(self) -> None:
        import requests

        from skiresort_planner.generators.osm_importer import _is_transient

        def _err(status: int | None) -> requests.RequestException:
            exc = requests.RequestException()
            if status is not None:
                exc.response = requests.Response()
                exc.response.status_code = status
            return exc

        assert _is_transient(_err(429)), "rate limit is retryable"
        assert _is_transient(_err(504)), "gateway timeout is retryable"
        assert _is_transient(_err(None)), "connection error (no response) is retryable"
        assert not _is_transient(_err(406)), "missing User-Agent (406) is not retryable"

    def test_single_query_no_retry_on_success(self, monkeypatch) -> None:
        import requests

        calls = {"n": 0}

        def fake_post(*_args, **_kwargs):
            calls["n"] += 1
            resp = requests.Response()
            resp.status_code = 200
            resp._content = b'{"elements": [{"id": 1}]}'
            return resp

        monkeypatch.setattr("skiresort_planner.generators.osm_importer.requests.post", fake_post)
        elements = LiftOnlyImporter(dem=_FakeDEM(), bbox=BBOX).fetch()
        assert elements == [{"id": 1}] and calls["n"] == 1, "one query, no retry when it succeeds"

    def test_transient_error_waits_then_retries_once(self, monkeypatch) -> None:
        import requests

        from skiresort_planner.generators import osm_importer

        monkeypatch.setattr(osm_importer, "_seconds_until_free_slot", lambda: 0.0)  # no real wait
        calls = {"n": 0}

        def fake_post(*_args, **_kwargs):
            calls["n"] += 1
            resp = requests.Response()
            if calls["n"] == 1:
                resp.status_code = 504  # first attempt: busy
            else:
                resp.status_code = 200
                resp._content = b'{"elements": [{"id": 2}]}'
            return resp

        monkeypatch.setattr("skiresort_planner.generators.osm_importer.requests.post", fake_post)
        elements = LiftOnlyImporter(dem=_FakeDEM(), bbox=BBOX).fetch()
        assert elements == [{"id": 2}] and calls["n"] == 2, "one retry after a transient failure"

    def test_non_transient_error_not_retried(self, monkeypatch) -> None:
        import pytest
        import requests

        calls = {"n": 0}

        def fake_post(*_args, **_kwargs):
            calls["n"] += 1
            resp = requests.Response()
            resp.status_code = 406  # missing User-Agent — a client bug, not transient
            return resp

        monkeypatch.setattr("skiresort_planner.generators.osm_importer.requests.post", fake_post)
        with pytest.raises(requests.HTTPError):
            LiftOnlyImporter(dem=_FakeDEM(), bbox=BBOX).fetch()
        assert calls["n"] == 1, "a non-transient error is raised at once, not retried"

    def test_seconds_until_free_slot_parsing(self, monkeypatch) -> None:
        from skiresort_planner.generators import osm_importer

        def status(text: str):
            resp = type("R", (), {"text": text})()
            monkeypatch.setattr("skiresort_planner.generators.osm_importer.requests.get", lambda *a, **k: resp)

        status("Rate limit: 2\n2 slots available now.\n")
        assert osm_importer._seconds_until_free_slot() == 0.0, "free slot → no wait"

        status("Rate limit: 2\nSlot available after: ..., in 7 seconds.\nSlot available after: ..., in 3 seconds.\n")
        assert osm_importer._seconds_until_free_slot() == 3.0, "wait the soonest of the busy slots"

        from skiresort_planner.constants import OSMConfig

        status("Rate limit: 2\nSlot available after: ..., in 999 seconds.\n")
        assert osm_importer._seconds_until_free_slot() == OSMConfig.SLOT_WAIT_MAX_S, "clamped to the cap"


class TestLifts:
    def test_aerialway_type_map_and_orientation(self) -> None:
        # bottom vertex given first but at higher lat (=higher elev on south slope); importer
        # must orient bottom = lower station.
        result = _lifts([_way({"aerialway": "gondola", "name": "Gipfelbahn"}, [(0.0, 0.0), (0.0, -0.02)])])
        assert len(result.lifts) == 1
        bottom, top, lift_type, name = result.lifts[0]
        assert lift_type == "gondola"
        assert name == "Gipfelbahn"
        assert bottom.elevation < top.elevation, "bottom is the lower station"

    def test_all_four_lift_types(self) -> None:
        elements = [
            _way({"aerialway": "t-bar", "name": "A"}, [(0.0, 0.0), (0.0, -0.01)]),
            _way({"aerialway": "chair_lift", "name": "B"}, [(0.001, 0.0), (0.001, -0.01)]),
            _way({"aerialway": "mixed_lift", "name": "C"}, [(0.002, 0.0), (0.002, -0.01)]),
            _way({"aerialway": "cable_car", "name": "D"}, [(0.003, 0.0), (0.003, -0.01)]),
        ]
        types = sorted(lift_type for _b, _t, lift_type, _n in _lifts(elements).lifts)
        assert types == ["aerial_tram", "chairlift", "gondola", "surface_lift"]

    def test_unmapped_aerialway_ignored_silently(self) -> None:
        # Only values in AERIALWAY_TO_LIFT_TYPE import. Everything else — infrastructure
        # (station/pylon), non-ski (zip_line), kiddie lifts (magic_carpet/rope_tow), the generic
        # "yes", and any unrecognised value — is silently ignored (not a lift, not counted).
        elements = [
            _way({"aerialway": "station", "building": "yes"}, [(0.0, 0.0), (0.0, -0.02)]),
            _way({"aerialway": "pylon"}, [(0.001, 0.0), (0.001, -0.02)]),
            _way({"aerialway": "zip_line"}, [(0.002, 0.0), (0.002, -0.02)]),
            _way({"aerialway": "magic_carpet"}, [(0.003, 0.0), (0.003, -0.02)]),
            _way({"aerialway": "rope_tow"}, [(0.004, 0.0), (0.004, -0.02)]),
            _way({"aerialway": "yes"}, [(0.005, 0.0), (0.005, -0.02)]),
            _way({"aerialway": "flying_carpet_9000"}, [(0.006, 0.0), (0.006, -0.02)]),
        ]
        result = _lifts(elements)
        assert result.lifts == [] and result.skipped == 0

    def test_short_lift_skipped(self) -> None:
        # A named gondola under MIN_LIFT_LENGTH_M (300 m) is a nursery lift → skipped and counted.
        result = _lifts([_way({"aerialway": "gondola", "name": "Baby"}, [(0.0, 0.0), (0.0, -0.002)])])  # ~223 m
        assert result.lifts == [] and result.skipped == 1

    def test_truncated_lift_skipped_entirely(self) -> None:
        # A lift with a vertex outside the box (far east) is dropped whole, not clipped.
        result = _lifts([_way({"aerialway": "gondola", "name": "Out"}, [(0.0, 0.0), (0.5, 0.0)])])
        assert result.lifts == [] and result.skipped == 1

    def test_nodata_lift_skipped_entirely(self) -> None:
        # Fully inside the box but a station over a DEM nodata hole → not sampleable → skipped.
        dem = _FakeDEM(hole_lon=0.0)  # any lon > 0 is nodata
        result = _lifts([_way({"aerialway": "gondola", "name": "Hole"}, [(0.001, 0.0), (0.001, -0.02)])], dem=dem)
        assert result.lifts == [] and result.skipped == 1

    def test_lift_name_resolution_and_unnamed(self) -> None:
        named = _way({"aerialway": "chair_lift", "name": "Gipfelbahn"}, [(0.0, 0.0), (0.0, -0.02)])
        by_ref = _way({"aerialway": "chair_lift", "ref": "B7"}, [(0.001, 0.0), (0.001, -0.02)])
        unnamed = _way({"aerialway": "chair_lift"}, [(0.002, 0.0), (0.002, -0.02)])
        result = _lifts([named, by_ref, unnamed])
        assert {name for _b, _t, _lt, name in result.lifts} == {"Gipfelbahn", "B7"}, "name→ref resolution"
        assert result.skipped == 1, "unnamed lift is skipped"


class TestDedupeByName:
    """OSM sometimes maps the same lift twice (an outdated way + a re-drawn one, same name), which
    import as two identical entities. We keep the LONGEST per name and count the shorter duplicates.
    """

    def test_shorter_same_name_lift_dropped(self) -> None:
        elements = [
            _way({"aerialway": "gondola", "name": "Gipfelbahn"}, [(0.0, 0.0), (0.0, -0.02)]),
            _way({"aerialway": "gondola", "name": "Gipfelbahn"}, [(0.0, 0.0), (0.0, -0.006)]),
        ]
        result = _lifts(elements)
        assert len(result.lifts) == 1
        bottom, top, _lt, name = result.lifts[0]
        assert name == "Gipfelbahn"
        assert bottom.distance_to(other=top) > 1000.0, "kept the longer lift"
        assert result.skipped == 1

    def test_distinct_names_all_kept(self) -> None:
        elements = [
            _way({"aerialway": "gondola", "name": "Gipfelbahn"}, [(0.0, 0.0), (0.0, -0.02)]),
            _way({"aerialway": "chair_lift", "name": "Talbahn"}, [(0.001, 0.0), (0.001, -0.02)]),
        ]
        result = _lifts(elements)
        assert {name for _b, _t, _lt, name in result.lifts} == {"Gipfelbahn", "Talbahn"}
        assert result.skipped == 0


class TestEndpointsMatch:
    def test_match_is_order_independent(self) -> None:
        from skiresort_planner.model.path_point import PathPoint, endpoints_match

        a = PathPoint(lon=10.5678, lat=47.1234, elevation=2400.0)
        b = PathPoint(lon=10.5700, lat=47.1300, elevation=2600.0)
        assert endpoints_match(pair_a=(a, b), pair_b=(b, a), tol_m=30.0), "reversed endpoints still match"

    def test_match_within_tolerance(self) -> None:
        from skiresort_planner.model.path_point import PathPoint, endpoints_match

        m = 111320.0
        a = PathPoint(lon=10.5678, lat=47.1234, elevation=2400.0)
        b = PathPoint(lon=10.5700, lat=47.1300, elevation=2600.0)
        # Each endpoint nudged ~10 m (< 30 m tol) → still the same run (absorbs import snap).
        a2 = PathPoint(lon=10.5678 + 10 / m, lat=47.1234, elevation=2401.0)
        b2 = PathPoint(lon=10.5700 - 10 / m, lat=47.1300, elevation=2599.0)
        assert endpoints_match(pair_a=(a, b), pair_b=(a2, b2), tol_m=30.0)

    def test_no_match_when_endpoint_moves_far(self) -> None:
        from skiresort_planner.model.path_point import PathPoint, endpoints_match

        a = PathPoint(lon=10.5678, lat=47.1234, elevation=2400.0)
        b = PathPoint(lon=10.5700, lat=47.1300, elevation=2600.0)
        moved = PathPoint(lon=10.6000, lat=47.1300, elevation=2600.0)  # ~2.4 km away
        assert not endpoints_match(pair_a=(a, b), pair_b=(a, moved), tol_m=30.0)


class TestBBoxAround:
    """bbox_around builds a square-in-metres box centered on a point: the lat half-span is
    half_width / metres-per-degree; the lon half-span is scaled up by 1/cos(lat) so the box is
    square on the ground, not in degrees. Tuple order is (min_lon, min_lat, max_lon, max_lat).
    """

    def test_centered_square_box(self) -> None:
        import math

        import pytest

        from skiresort_planner.core.geo_calculator import GeoCalculator
        from skiresort_planner.generators.osm_importer import bbox_around

        min_lon, min_lat, max_lon, max_lat = bbox_around(center_lon=10.0, center_lat=47.0, half_width_m=1000.0)
        # Box is centered on (10, 47) and ordered min<max on both axes.
        assert min_lon < max_lon and min_lat < max_lat, "min corner precedes max corner"
        assert (min_lon + max_lon) / 2 == pytest.approx(10.0), "centered in lon"
        assert (min_lat + max_lat) / 2 == pytest.approx(47.0), "centered in lat"
        # Exact half-spans from the same formula the source uses (haversine 1-deg metres).
        m_per_deg = GeoCalculator.haversine_distance_m(lat1=0.0, lon1=0.0, lat2=1.0, lon2=0.0)
        dlat = 1000.0 / m_per_deg
        dlon = dlat / math.cos(math.radians(47.0))
        assert (max_lat - min_lat) / 2 == pytest.approx(dlat)
        assert (max_lon - min_lon) / 2 == pytest.approx(dlon)
        # Lon span is wider than lat span by 1/cos(lat) to stay square on the ground.
        assert (max_lon - min_lon) / (max_lat - min_lat) == pytest.approx(1.0 / math.cos(math.radians(47.0)))

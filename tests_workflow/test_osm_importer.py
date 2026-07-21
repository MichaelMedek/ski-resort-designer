"""Unit tests for the OpenStreetMap importer base + lift-only child (generators/osm_importer.py,
generators/osm_lift_importer.py).

The importer takes GEOMETRY ONLY from OSM; elevation/pylons are recomputed. These tests feed
synthetic Overpass `elements` + a controllable fake DEM (with a nodata hole) and assert: the
aerialway→lift-type map (mapping-only; unmapped values are silently ignored), named-only filter
(unnamed lifts skipped), min-length gate, out-of-box / over-nodata skips, longest-per-name dedup,
the fetch/retry classification, and bbox_around. Slope geometry lives in the connected-graph
builder (test_osm_import_rules.py), not here.
"""

from skiresort_planner.constants import MapConfig
from skiresort_planner.core.dem_service import DEMService
from skiresort_planner.generators.osm_importer import ImportResult, OverpassElement
from skiresort_planner.generators.osm_lift_importer import LiftOnlyImporter


class _FakeDEM(DEMService):
    """20% south slope; a nodata hole for lon > `hole_lon` (returns None there)."""

    def __new__(cls, hole_lon: float = 999.0) -> "_FakeDEM":
        return object.__new__(cls)

    def __init__(self, hole_lon: float = 999.0) -> None:
        self.hole_lon = hole_lon

    def get_elevation(self, lon: float, lat: float) -> float | None:
        if lon > self.hole_lon:
            return None
        return 2500.0 - lat * MapConfig.METERS_PER_DEGREE_EQUATOR * 0.20


# Import box (min_lon, min_lat, max_lon, max_lat): a wide box around the test geometry so every
# normal test vertex is well inside; the truncation test uses a vertex far outside it.
BBOX = (-0.1, -0.11, 0.1, 0.09)


def _noop_progress(frac: float, text: str) -> None:
    """No-op ProgressFn for importer tests that don't assert on progress."""


def _ok_response(body: bytes):
    """A 200 requests.Response with `body` as its (JSON) content — what response.json() reads."""
    import requests

    resp = requests.Response()
    resp.status_code = 200
    resp._content = body
    return resp


def _way(tags: dict[str, str], verts: list[tuple[float, float]], node_ids: list[int] | None = None) -> OverpassElement:
    """Build a synthetic Overpass way element with inline geometry + a parallel `nodes` id array.

    Real `out geom;` ways always carry `nodes` aligned 1:1 with geometry; default to auto-numbered
    ids so tests mirror that invariant. Pass explicit `node_ids` to reference a station node.
    """
    ids = node_ids if node_ids is not None else list(range(len(verts)))
    return {"type": "way", "tags": tags, "geometry": [{"lon": x, "lat": y} for x, y in verts], "nodes": ids}


def _station(node_id: int, lon: float, lat: float) -> OverpassElement:
    """Build a synthetic Overpass `aerialway=station` node element (lat/lon directly on it)."""
    return {"type": "node", "id": node_id, "lon": lon, "lat": lat, "tags": {"aerialway": "station"}}


def _lifts(elements: list[OverpassElement], dem: _FakeDEM | None = None) -> ImportResult:
    """Assemble a lift-only import (no network) from synthetic elements."""
    return LiftOnlyImporter(dem=dem or _FakeDEM(), bbox=BBOX)._assemble(elements, on_progress=_noop_progress)


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
        calls = {"n": 0}

        def fake_post(*_args, **_kwargs):
            calls["n"] += 1
            return _ok_response(b'{"elements": [{"id": 1}]}')

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
            if calls["n"] == 1:
                resp = requests.Response()
                resp.status_code = 504  # first attempt: busy
                return resp
            return _ok_response(b'{"elements": [{"id": 2}]}')

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
        # A named gondola under MIN_LIFT_LENGTH_M (200 m) is a nursery lift → skipped and counted.
        result = _lifts([_way({"aerialway": "gondola", "name": "Baby"}, [(0.0, 0.0), (0.0, -0.0015)])])  # ~167 m
        assert result.lifts == [] and result.skipped == 1

    def test_truncated_lift_skipped_entirely(self) -> None:
        # A lift with a vertex outside the box (far east) is dropped whole by the shared in-box filter,
        # not clipped (and not counted — out-of-box ways never enter the lift-only skip tally).
        result = _lifts([_way({"aerialway": "gondola", "name": "Out"}, [(0.0, 0.0), (0.5, 0.0)])])
        assert result.lifts == []

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
    """OSM sometimes maps the same lift twice (an outdated way + a re-drawn one, same name). We keep
    the LONGEST and count the shorter ONLY when they are geometrically COINCIDENT (both endpoints
    match) — a mere name clash between distinct lifts is not a duplicate.
    """

    def test_shorter_same_name_lift_dropped(self) -> None:
        # Same name AND (near-)coincident endpoints — a redraw of the SAME lift. Keep the longest.
        elements = [
            _way({"aerialway": "gondola", "name": "Gipfelbahn"}, [(0.0, 0.0), (0.0, -0.02)]),
            _way({"aerialway": "gondola", "name": "Gipfelbahn"}, [(0.0, 0.0), (0.0, -0.0195)]),  # top ~56 m off
        ]
        result = _lifts(elements)
        assert len(result.lifts) == 1
        bottom, top, _lt, name = result.lifts[0]
        assert name == "Gipfelbahn"
        assert bottom.distance_to(other=top) > 2000.0, "kept the longer lift"
        assert result.skipped == 1

    def test_same_name_distinct_geometry_both_kept(self) -> None:
        # Same name but far-apart endpoints — two DIFFERENT lifts that happen to share a name. Keep both.
        elements = [
            _way({"aerialway": "gondola", "name": "Dorfbahn"}, [(0.0, 0.0), (0.0, -0.02)]),
            _way({"aerialway": "gondola", "name": "Dorfbahn"}, [(0.05, 0.0), (0.05, -0.02)]),
        ]
        result = _lifts(elements)
        assert len(result.lifts) == 2 and result.skipped == 0

    def test_distinct_names_all_kept(self) -> None:
        elements = [
            _way({"aerialway": "gondola", "name": "Gipfelbahn"}, [(0.0, 0.0), (0.0, -0.02)]),
            _way({"aerialway": "chair_lift", "name": "Talbahn"}, [(0.001, 0.0), (0.001, -0.02)]),
        ]
        result = _lifts(elements)
        assert {name for _b, _t, _lt, name in result.lifts} == {"Gipfelbahn", "Talbahn"}
        assert result.skipped == 0


class TestMidStationSplit:
    """A lift way whose INTERIOR carries an `aerialway=station` node splits into per-section lifts:
    bottom→mid, mid→top (N interior stations → N+1 sections). Matching is by exact OSM node id.
    """

    def test_one_mid_station_splits_into_two(self) -> None:
        # 3-vertex gondola; the middle vertex (node id 200) is a station → two ~1.1 km sections.
        way = _way(
            {"aerialway": "gondola", "name": "Silvrettabahn"},
            [(0.0, 0.0), (0.0, -0.01), (0.0, -0.02)],
            node_ids=[100, 200, 300],
        )
        result = _lifts([way, _station(200, 0.0, -0.01)])
        assert len(result.lifts) == 2
        lifts = sorted(result.lifts, key=lambda lf: lf[3] or "")
        (b1, t1, _lt1, n1), (b2, t2, _lt2, n2) = lifts
        assert n1 == "Silvrettabahn (1)" and n2 == "Silvrettabahn (2)"
        assert all(b.elevation < t.elevation for b, t in ((b1, t1), (b2, t2))), "each section oriented up"
        # Sections meet at the shared mid-station coordinate (top of one == bottom of the other).
        assert t1.distance_to(other=b2) < 1.0 or t2.distance_to(other=b1) < 1.0

    def test_station_at_endpoint_no_split(self) -> None:
        # Station node id matches an ENDPOINT vertex (not interior) → normal two-station lift, no split.
        way = _way(
            {"aerialway": "gondola", "name": "Endbahn"},
            [(0.0, 0.0), (0.0, -0.02)],
            node_ids=[100, 200],
        )
        result = _lifts([way, _station(200, 0.0, -0.02)])
        assert len(result.lifts) == 1
        assert result.lifts[0][3] == "Endbahn", "single-section lift keeps its bare name"

    def test_two_mid_stations_split_into_three(self) -> None:
        way = _way(
            {"aerialway": "gondola", "name": "Dreibahn"},
            [(0.0, 0.0), (0.0, -0.01), (0.0, -0.02), (0.0, -0.03)],
            node_ids=[100, 200, 300, 400],
        )
        result = _lifts([way, _station(200, 0.0, -0.01), _station(300, 0.0, -0.02)])
        assert len(result.lifts) == 3
        assert {lf[3] for lf in result.lifts} == {"Dreibahn (1)", "Dreibahn (2)", "Dreibahn (3)"}

    def test_short_section_skipped_other_kept(self) -> None:
        # A mid-station near the bottom makes the lower section ~167 m (< 200 m min) → dropped+counted;
        # the long upper section survives.
        way = _way(
            {"aerialway": "gondola", "name": "Kurzbahn"},
            [(0.0, 0.0), (0.0, -0.0015), (0.0, -0.02)],
            node_ids=[100, 200, 300],
        )
        result = _lifts([way, _station(200, 0.0, -0.0015)])
        assert len(result.lifts) == 1 and result.skipped == 1
        assert result.lifts[0][3] == "Kurzbahn (2)", "kept the long upper section"

    def test_station_id_not_on_any_way_no_split(self) -> None:
        way = _way(
            {"aerialway": "gondola", "name": "Solobahn"},
            [(0.0, 0.0), (0.0, -0.01), (0.0, -0.02)],
            node_ids=[100, 200, 300],
        )
        result = _lifts([way, _station(999, 0.05, 0.05)])  # station id present but on no way vertex
        assert len(result.lifts) == 1 and result.lifts[0][3] == "Solobahn"


class TestSplitHelpers:
    """Direct unit tests for the pure split/naming helpers."""

    def test_split_at_interior_station(self) -> None:
        from skiresort_planner.generators.osm_importer import split_lift_way_at_stations

        verts = [(0.0, 0.0), (0.0, 1.0), (0.0, 2.0)]
        sections = split_lift_way_at_stations(vertices=verts, node_ids=[1, 2, 3], station_ids={2})
        assert sections == [[(0.0, 0.0), (0.0, 1.0)], [(0.0, 1.0), (0.0, 2.0)]]

    def test_no_interior_station_returns_whole(self) -> None:
        from skiresort_planner.generators.osm_importer import split_lift_way_at_stations

        verts = [(0.0, 0.0), (0.0, 1.0), (0.0, 2.0)]
        # Endpoints are stations, but never split points.
        assert split_lift_way_at_stations(vertices=verts, node_ids=[1, 2, 3], station_ids={1, 3}) == [verts]

    def test_misaligned_lengths_raise(self) -> None:
        import pytest

        from skiresort_planner.generators.osm_importer import split_lift_way_at_stations

        with pytest.raises(AssertionError):
            split_lift_way_at_stations(vertices=[(0.0, 0.0), (0.0, 1.0)], node_ids=[1], station_ids={9})

    def test_suffixed_name(self) -> None:
        from skiresort_planner.generators.osm_importer import suffixed_name

        assert suffixed_name("Bahn", 0, 1) == "Bahn", "single section keeps bare name"
        assert suffixed_name("Bahn", 0, 2) == "Bahn (1)"
        assert suffixed_name("Bahn", 1, 2) == "Bahn (2)"
        assert suffixed_name(None, 0, 2) is None

    def test_station_node_ids(self) -> None:
        from skiresort_planner.generators.osm_importer import station_node_ids

        elements: list[OverpassElement] = [
            _station(5, 0.0, 0.0),
            {"type": "node", "id": 6, "lon": 0.0, "lat": 0.0, "tags": {"aerialway": "pylon"}},
            _way({"aerialway": "gondola", "name": "X"}, [(0.0, 0.0), (0.0, -0.02)]),
        ]
        assert station_node_ids(elements) == {5}


class TestEndpointsMatch:
    def test_match_is_order_independent(self) -> None:
        from skiresort_planner.model.path_point import PathPoint, endpoints_match

        a = PathPoint(lon=10.5678, lat=47.1234, elevation=2400.0)
        b = PathPoint(lon=10.5700, lat=47.1300, elevation=2600.0)
        assert endpoints_match(pair_a=(a, b), pair_b=(b, a), tol_m=30.0), "reversed endpoints still match"

    def test_match_within_tolerance(self) -> None:
        from skiresort_planner.model.path_point import PathPoint, endpoints_match

        a = PathPoint(lon=10.5678, lat=47.1234, elevation=2400.0)
        b = PathPoint(lon=10.5700, lat=47.1300, elevation=2600.0)
        # Each endpoint nudged ~10 m (< 30 m tol) → still the same run (absorbs import snap).
        a2 = PathPoint(lon=10.5678 + 10 / MapConfig.METERS_PER_DEGREE_EQUATOR, lat=47.1234, elevation=2401.0)
        b2 = PathPoint(lon=10.5700 - 10 / MapConfig.METERS_PER_DEGREE_EQUATOR, lat=47.1300, elevation=2599.0)
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

        from skiresort_planner.generators.osm_importer import bbox_around

        min_lon, min_lat, max_lon, max_lat = bbox_around(center_lon=10.0, center_lat=47.0, half_width_m=1000.0)
        # Box is centered on (10, 47) and ordered min<max on both axes.
        assert min_lon < max_lon and min_lat < max_lat, "min corner precedes max corner"
        assert (min_lon + max_lon) / 2 == pytest.approx(10.0), "centered in lon"
        assert (min_lat + max_lat) / 2 == pytest.approx(47.0), "centered in lat"
        # Exact half-spans from the same formula the source uses (the one central metres-per-degree constant).
        m_per_deg = MapConfig.METERS_PER_DEGREE_EQUATOR
        dlat = 1000.0 / m_per_deg
        dlon = dlat / math.cos(math.radians(47.0))
        assert (max_lat - min_lat) / 2 == pytest.approx(dlat)
        assert (max_lon - min_lon) / 2 == pytest.approx(dlon)
        # Lon span is wider than lat span by 1/cos(lat) to stay square on the ground.
        assert (max_lon - min_lon) / (max_lat - min_lat) == pytest.approx(1.0 / math.cos(math.radians(47.0)))


class TestProgress:
    """Real progress reporting: fetch fills the first half, assemble the second; monotonic to 1.0."""

    def test_sub_progress_maps_child_range_onto_slice(self) -> None:
        from skiresort_planner.generators.osm_importer import sub_progress

        seen: list[float] = []
        child = sub_progress(lambda frac, _t: seen.append(frac), 0.5, 1.0)
        child(0.0, "start")
        child(0.5, "mid")
        child(1.0, "end")
        assert seen == [0.5, 0.75, 1.0], "child 0..1 maps exactly onto [lo, hi]"

    def test_sub_progress_clamps_out_of_range(self) -> None:
        from skiresort_planner.generators.osm_importer import sub_progress

        seen: list[float] = []
        child = sub_progress(lambda frac, _t: seen.append(frac), 0.0, 0.5)
        child(-1.0, "under")
        child(2.0, "over")
        assert seen == [0.0, 0.5], "child fractions clamped to [0, 1] before mapping"

    def test_run_reports_monotonic_progress_across_fetch_and_assemble(self, monkeypatch) -> None:
        # Full lift-only run() with a mocked network: assert progress is non-decreasing, ends at 1.0,
        # and spans BOTH halves (a fetch point in [0, 0.5], an assemble/finish point in (0.5, 1.0]).
        def fake_post(*_args, **_kwargs):
            return _ok_response(
                b'{"elements": [{"type": "way", "tags": {"aerialway": "gondola", '
                b'"name": "G"}, "geometry": [{"lon": 0.0, "lat": 0.0}, {"lon": 0.0, "lat": -0.02}], '
                b'"nodes": [1, 2]}]}'
            )

        monkeypatch.setattr("skiresort_planner.generators.osm_importer.requests.post", fake_post)
        seen: list[float] = []
        LiftOnlyImporter(dem=_FakeDEM(), bbox=BBOX).run(on_progress=lambda frac, _t: seen.append(frac))

        assert seen == sorted(seen), f"progress must be monotonic non-decreasing: {seen}"
        assert seen[-1] == 1.0, "progress ends at 1.0"
        assert any(f <= 0.5 for f in seen), "at least one fetch-half report"
        assert any(f > 0.5 for f in seen), "at least one assemble-half report"


class TestPisteFilter:
    """`_is_importable_piste` is the piste allow-list (counterpart to the lift aerialway map): only
    standard groomed downhill grades import; off-piste variants (freeride/extreme, backcountry) and
    non-downhill types are excluded; connectors are always kept.
    """

    def test_standard_downhill_grades_kept(self) -> None:
        from skiresort_planner.generators.osm_graph_builder import _is_importable_piste

        for diff in ("novice", "easy", "intermediate", "advanced", "expert"):
            assert _is_importable_piste({"piste:type": "downhill", "piste:difficulty": diff}), diff

    def test_offpiste_variants_excluded(self) -> None:
        from skiresort_planner.generators.osm_graph_builder import _is_importable_piste

        assert not _is_importable_piste({"piste:type": "downhill", "piste:difficulty": "freeride"})
        assert not _is_importable_piste({"piste:type": "downhill", "piste:difficulty": "extreme"})
        # A backcountry-groomed freeride run — the tag combo that marks off-piste in real data.
        assert not _is_importable_piste(
            {"piste:type": "downhill", "piste:difficulty": "freeride", "piste:grooming": "backcountry"}
        )

    def test_downhill_without_difficulty_excluded(self) -> None:
        from skiresort_planner.generators.osm_graph_builder import _is_importable_piste

        assert not _is_importable_piste({"piste:type": "downhill"}), "untagged difficulty is not a standard run"

    def test_non_downhill_types_excluded(self) -> None:
        from skiresort_planner.generators.osm_graph_builder import _is_importable_piste

        for ptype in ("skitour", "nordic", "sled", "hike", "snow_park"):
            assert not _is_importable_piste({"piste:type": ptype}), ptype

    def test_connection_always_kept(self) -> None:
        from skiresort_planner.generators.osm_graph_builder import _is_importable_piste

        assert _is_importable_piste({"piste:type": "connection"}), "connectors kept for connectivity"
        assert _is_importable_piste({"piste:type": "connection", "piste:difficulty": "freeride"})

    def test_importable_lift_map(self) -> None:
        """extract_lift_sections keeps only MAPPED aerialway types; station/pylon/unmapped ways yield no section."""
        from skiresort_planner.generators.osm_importer import extract_lift_sections

        line = [(0.0, 0.0), (0.0, -0.02)]
        assert extract_lift_sections([_way({"aerialway": "gondola", "name": "G"}, line)], BBOX), "gondola kept"
        assert not extract_lift_sections([_way({"aerialway": "station", "name": "S"}, line)], BBOX), "station dropped"
        assert not extract_lift_sections([_way({"aerialway": "pylon", "name": "P"}, line)], BBOX), "pylon dropped"
        assert not extract_lift_sections([_way({"highway": "path", "name": "X"}, line)], BBOX), "non-aerialway dropped"

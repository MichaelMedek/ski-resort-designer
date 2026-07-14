"""Unit tests for the OpenStreetMap importer (generators/osm_importer.py).

The importer takes GEOMETRY ONLY from OSM; elevation/difficulty/pylons/WIDTH are recomputed. These
tests feed synthetic Overpass `elements` + a controllable fake DEM (with a nodata hole) and
assert: downhill-only piste filter, named-only filter (unnamed ways skipped), aerialway→lift-type
map (mapping-only; unmapped values are silently ignored), linear resample spacing + DEM Z, min-length
gate, and that ways reaching outside the box / over nodata are skipped ENTIRELY (only full entities
import).
"""

from skiresort_planner.core.dem_service import DEMService
from skiresort_planner.generators.osm_importer import ImportSummary, OSMImporter, OverpassElement

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


def _convert(elements: list[OverpassElement], dem: _FakeDEM | None = None) -> ImportSummary:
    return OSMImporter(dem=dem or _FakeDEM()).convert(bbox=BBOX, elements=elements)


class TestPisteFiltering:
    def test_only_downhill_imported(self) -> None:
        elements = [
            _way({"piste:type": "downhill", "name": "Ried"}, [(0.0, 0.0), (0.0, -0.01), (0.0, -0.02)]),
            _way({"piste:type": "connection"}, [(0.0, 0.0), (0.0, -0.01)]),
            _way({"piste:type": "snow_park"}, [(0.0, 0.0), (0.0, -0.01)]),
            _way({"piste:type": "yes"}, [(0.0, 0.0), (0.0, -0.01)]),
        ]
        summary = _convert(elements)
        assert len(summary.pistes) == 1, "only the downhill run imports"
        assert summary.pistes[0].name == "Ried"

    def test_name_resolution_and_unnamed(self) -> None:
        named = _way({"piste:type": "downhill", "piste:ref": "7"}, [(0.0, 0.0), (0.0, -0.01)])
        unnamed = _way({"piste:type": "downhill"}, [(0.0, 0.0), (0.0, -0.02)])
        summary = _convert([named, unnamed])
        assert len(summary.pistes) == 1, "only the named run imports"
        assert summary.pistes[0].name == "7"  # falls back name→piste:name→piste:ref→ref
        assert summary.skipped == 1, "unnamed run is skipped (potentially outdated/duplicate)"


class TestNamedOnly:
    def test_unnamed_piste_skipped(self) -> None:
        # Unnamed downhill runs are frequently outdated/duplicate → skipped and counted.
        summary = _convert([_way({"piste:type": "downhill"}, [(0.0, 0.0), (0.0, -0.02)])])
        assert summary.pistes == [] and summary.skipped == 1

    def test_unnamed_lift_skipped(self) -> None:
        summary = _convert([_way({"aerialway": "chair_lift"}, [(0.0, 0.0), (0.0, -0.02)])])
        assert summary.lifts == [] and summary.skipped == 1


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
        elements = OSMImporter(dem=_FakeDEM()).fetch(BBOX)
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
        elements = OSMImporter(dem=_FakeDEM()).fetch(BBOX)
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
            OSMImporter(dem=_FakeDEM()).fetch(BBOX)
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


class TestResample:
    def test_spacing_and_dem_z(self) -> None:
        # ~2.2 km straight run, 30 m step → many points, all draped on the DEM.
        summary = _convert([_way({"piste:type": "downhill", "name": "Ried"}, [(0.0, 0.0), (0.0, -0.02)])])
        pts = summary.pistes[0].points
        assert len(pts) > 50, "long run resampled to many points"
        assert all(p.elevation == 2500.0 - p.lat * M * 0.20 for p in pts), "Z is the DEM value"
        # Consecutive spacing ≈ RESAMPLE_STEP_M (30 m), not the raw 2-vertex polyline.
        gaps = [pts[i].distance_to(other=pts[i + 1]) for i in range(len(pts) - 1)]
        assert max(gaps) < 40.0, f"no gap far above the 30 m step, got {max(gaps):.0f}"

    def test_two_vertex_run_keeps_endpoints_oriented_downhill(self) -> None:
        # A straight 2-vertex run resamples to interior points; the DEM here rises toward -lat, so
        # the run is reoriented to descend (top→bottom). Its endpoints are the two way vertices.
        summary = _convert([_way({"piste:type": "downhill", "name": "Ried"}, [(0.0, 0.0), (0.0, -0.01)])])
        pts = summary.pistes[0].points
        assert pts[0].elevation >= pts[-1].elevation, "imported run must descend top→bottom"
        endpoints = {(round(pts[0].lat, 6), pts[0].lon), (round(pts[-1].lat, 6), pts[-1].lon)}
        assert endpoints == {(0.0, 0.0), (-0.01, 0.0)}, "trim keeps the two way endpoints"


class TestFullOnly:
    def test_truncated_way_skipped_entirely(self) -> None:
        # A way with a vertex outside the box (far east) is dropped whole, not clipped.
        summary = _convert([_way({"piste:type": "downhill", "name": "Ried"}, [(0.0, 0.0), (0.5, 0.0)])])
        assert summary.pistes == [] and summary.skipped == 1

    def test_nodata_way_skipped_entirely(self) -> None:
        # Fully inside the box but over a DEM nodata hole → not fully sampleable → skipped.
        dem = _FakeDEM(hole_lon=0.0)  # any lon > 0 is nodata
        summary = _convert([_way({"piste:type": "downhill", "name": "Ried"}, [(0.001, 0.0), (0.001, -0.02)])], dem=dem)
        assert summary.pistes == [] and summary.skipped == 1


class TestLifts:
    def test_aerialway_type_map_and_orientation(self) -> None:
        # bottom vertex given first but at higher lat (=higher elev on south slope); importer
        # must orient bottom = lower station.
        summary = _convert([_way({"aerialway": "gondola", "name": "Gipfelbahn"}, [(0.0, 0.0), (0.0, -0.02)])])
        assert len(summary.lifts) == 1
        lift = summary.lifts[0]
        assert lift.lift_type == "gondola"
        assert lift.bottom.elevation < lift.top.elevation, "bottom is the lower station"

    def test_all_four_lift_types(self) -> None:
        elements = [
            _way({"aerialway": "t-bar", "name": "A"}, [(0.0, 0.0), (0.0, -0.01)]),
            _way({"aerialway": "chair_lift", "name": "B"}, [(0.001, 0.0), (0.001, -0.01)]),
            _way({"aerialway": "mixed_lift", "name": "C"}, [(0.002, 0.0), (0.002, -0.01)]),
            _way({"aerialway": "cable_car", "name": "D"}, [(0.003, 0.0), (0.003, -0.01)]),
        ]
        types = sorted(lift.lift_type for lift in _convert(elements).lifts)
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
        summary = _convert(elements)
        assert summary.lifts == [] and summary.skipped == 0

    def test_short_lift_skipped(self) -> None:
        # A named gondola under MIN_LIFT_LENGTH_M (300 m) is a nursery lift → skipped and counted.
        summary = _convert([_way({"aerialway": "gondola", "name": "Baby"}, [(0.0, 0.0), (0.0, -0.002)])])  # ~223 m
        assert summary.lifts == [] and summary.skipped == 1

    def test_lift_name_resolution_and_unnamed(self) -> None:
        named = _way({"aerialway": "chair_lift", "name": "Gipfelbahn"}, [(0.0, 0.0), (0.0, -0.02)])
        by_ref = _way({"aerialway": "chair_lift", "ref": "B7"}, [(0.001, 0.0), (0.001, -0.02)])
        unnamed = _way({"aerialway": "chair_lift"}, [(0.002, 0.0), (0.002, -0.02)])
        summary = _convert([named, by_ref, unnamed])
        assert {lift.name for lift in summary.lifts} == {"Gipfelbahn", "B7"}, "name→ref resolution"
        assert summary.skipped == 1, "unnamed lift is skipped"


class TestMinLength:
    def test_short_piste_skipped(self) -> None:
        # A named downhill run under MIN_PISTE_LENGTH_M (200 m) is a stub → skipped and counted.
        summary = _convert([_way({"piste:type": "downhill", "name": "Stub"}, [(0.0, 0.0), (0.0, -0.0015)])])  # ~167 m
        assert summary.pistes == [] and summary.skipped == 1

    def test_long_piste_imported(self) -> None:
        # A run comfortably over the minimum imports normally.
        summary = _convert([_way({"piste:type": "downhill", "name": "Ried"}, [(0.0, 0.0), (0.0, -0.004)])])  # ~445 m
        assert len(summary.pistes) == 1 and summary.skipped == 0


class TestDescendingTrim:
    """OSM out-and-back pistes (drawn up AND down) must be trimmed to their descending run.

    The _FakeDEM is a pure south slope (elev = 2500 - lat·M·0.20), so a run going south descends and
    a run going north climbs. A polyline that goes down then back up drapes to an up-and-down profile
    with ~0 net drop — the bug the user hit ('7a', 0% gradient). We keep only the longest descending
    stretch, judged at rolling-window scale so DEM bumps don't fragment a real run.
    """

    def test_out_and_back_trimmed_to_descending_arm(self) -> None:
        # South 600 m (down) then back north 600 m (up): only the descending arm should survive.
        verts = [(0.0, 0.0), (0.0, -0.006), (0.0, 0.0)]
        summary = _convert([_way({"piste:type": "downhill", "name": "7a"}, verts)])
        assert len(summary.pistes) == 1, "the descending arm alone still clears the length min"
        pts = summary.pistes[0].points
        assert pts[0].elevation > pts[-1].elevation, "kept run descends"
        assert pts[-1].lat == 0.0 and pts[0].lat < 0.0, "kept the south (descending) arm"
        assert pts[0].elevation - pts[-1].elevation > 100.0, "real drop, not the ~0 of the out-and-back"

    def test_pure_descent_is_unchanged(self) -> None:
        summary = _convert([_way({"piste:type": "downhill", "name": "Ried"}, [(0.0, -0.006), (0.0, 0.0)])])
        pts = summary.pistes[0].points
        assert pts[0].elevation > pts[-1].elevation
        assert (pts[0].lat, pts[-1].lat) == (-0.006, 0.0)

    def test_out_and_back_where_each_arm_too_short_is_skipped(self) -> None:
        # Down ~122 m then up ~122 m: each arm < 200 m min, so after trimming the run is dropped.
        verts = [(0.0, 0.0), (0.0, -0.0011), (0.0, 0.0)]
        summary = _convert([_way({"piste:type": "downhill", "name": "TinyOut"}, verts)])
        assert summary.pistes == [] and summary.skipped == 1

    def test_longest_descending_run_helper(self) -> None:
        from skiresort_planner.generators.osm_importer import _longest_descending_run
        from skiresort_planner.model.path_point import PathPoint

        def run(elevs):
            # Points spaced 30 m so the series spans well past the rolling window (realistic scale).
            pts = [PathPoint(lon=0.0, lat=-i * 30 / M, elevation=e) for i, e in enumerate(elevs)]
            out = _longest_descending_run(pts)
            return out[0].elevation, out[-1].elevation, len(out)

        # Pure 30-point descent: unchanged and oriented top→bottom.
        pure = [100.0 - 2 * i for i in range(30)]
        assert run(pure) == (100.0, pure[-1], 30)
        assert run(pure)[0] > run(pure)[1]

        # 20 down then 20 up: trimmed to the descending arm only.
        out_and_back = [100.0 - 3 * i for i in range(20)] + [40.0 + 3 * i for i in range(20)]
        first, last, n = run(out_and_back)
        assert first > last and n < len(out_and_back), "out-and-back trimmed to its descending arm"

        # Net descent with periodic small up-bumps: the whole run is kept (bumps tolerated).
        undulating = []
        e = 200.0
        for i in range(30):
            e += 6.0 if i % 5 == 0 else 0.0
            e -= 4.0
            undulating.append(e)
        _first, _last, n = run(undulating)
        assert n == 30, "an undulating-but-descending run is not fragmented by DEM bumps"


class TestDedupeByName:
    """OSM sometimes maps the same run/lift twice (an outdated way + a re-drawn one, same name),
    which import as two identical entities. We keep the LONGEST per name within each kind and count
    the shorter same-name duplicates as skipped. Pistes and lifts dedupe independently.
    """

    def test_shorter_same_name_piste_dropped(self) -> None:
        # Two 'Ried' downhill runs; the longer (0→-0.02) is kept, the shorter (0→-0.006) dropped.
        elements = [
            _way({"piste:type": "downhill", "name": "Ried"}, [(0.0, 0.0), (0.0, -0.006)]),
            _way({"piste:type": "downhill", "name": "Ried"}, [(0.0, 0.0), (0.0, -0.02)]),
        ]
        summary = _convert(elements)
        assert len(summary.pistes) == 1, "only the longer 'Ried' survives"
        kept = summary.pistes[0]
        assert kept.name == "Ried"
        assert abs(kept.points[0].lat - kept.points[-1].lat) > 0.01, "kept the longer geometry"
        assert summary.skipped == 1, "the shorter duplicate is counted as skipped"

    def test_shorter_same_name_lift_dropped(self) -> None:
        elements = [
            _way({"aerialway": "gondola", "name": "Gipfelbahn"}, [(0.0, 0.0), (0.0, -0.02)]),
            _way({"aerialway": "gondola", "name": "Gipfelbahn"}, [(0.0, 0.0), (0.0, -0.006)]),
        ]
        summary = _convert(elements)
        assert len(summary.lifts) == 1 and summary.lifts[0].name == "Gipfelbahn"
        drop = summary.lifts[0].bottom.distance_to(other=summary.lifts[0].top)
        assert drop > 1000.0, "kept the longer lift"
        assert summary.skipped == 1

    def test_distinct_names_all_kept(self) -> None:
        elements = [
            _way({"piste:type": "downhill", "name": "Ried"}, [(0.0, 0.0), (0.0, -0.02)]),
            _way({"piste:type": "downhill", "name": "Kessel"}, [(0.001, 0.0), (0.001, -0.02)]),
        ]
        summary = _convert(elements)
        assert {p.name for p in summary.pistes} == {"Ried", "Kessel"}
        assert summary.skipped == 0

    def test_same_name_different_kind_not_deduped(self) -> None:
        # A piste and a lift sharing a name must NOT collide — kinds dedupe independently.
        elements = [
            _way({"piste:type": "downhill", "name": "Sonnblick"}, [(0.0, 0.0), (0.0, -0.02)]),
            _way({"aerialway": "gondola", "name": "Sonnblick"}, [(0.001, 0.0), (0.001, -0.02)]),
        ]
        summary = _convert(elements)
        assert len(summary.pistes) == 1 and len(summary.lifts) == 1
        assert summary.skipped == 0

    def test_three_same_name_keeps_only_longest(self) -> None:
        elements = [
            _way({"piste:type": "downhill", "name": "Ried"}, [(0.0, 0.0), (0.0, -0.006)]),
            _way({"piste:type": "downhill", "name": "Ried"}, [(0.0, 0.0), (0.0, -0.03)]),
            _way({"piste:type": "downhill", "name": "Ried"}, [(0.0, 0.0), (0.0, -0.01)]),
        ]
        summary = _convert(elements)
        assert len(summary.pistes) == 1, "only the single longest 'Ried' survives"
        assert abs(summary.pistes[0].points[0].lat - summary.pistes[0].points[-1].lat) > 0.02
        assert summary.skipped == 2, "the two shorter duplicates are both skipped"


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

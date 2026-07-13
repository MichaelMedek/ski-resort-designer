"""Unit tests for the OpenStreetMap importer (generators/osm_importer.py).

The importer takes GEOMETRY ONLY from OSM; elevation/difficulty/pylons are recomputed. These
tests feed synthetic Overpass `elements` + a controllable fake DEM (with a nodata hole) and
assert: downhill-only piste filter, aerialway→lift-type map (mapping-only; unmapped values are
silently ignored), linear resample spacing + DEM Z, min-length gate, and that ways truncated by
the circle / over nodata are skipped ENTIRELY (only full entities import).
"""

from skiresort_planner.generators.osm_importer import OSMImporter

M = 111320.0  # metres per degree near the equator


class _FakeDEM:
    """20% south slope; a nodata hole for lon > `hole_lon` (returns None there)."""

    def __init__(self, hole_lon: float = 999.0) -> None:
        self.hole_lon = hole_lon

    def get_elevation(self, lon: float, lat: float) -> float | None:
        if lon > self.hole_lon:
            return None
        return 2500.0 - lat * M * 0.20


# Import circle (center_lon, center_lat, radius_m): centered on the test geometry, 10 km radius so
# every normal test vertex is well inside; the truncation test uses a vertex far outside it.
REGION = (0.0, -0.01, 10_000.0)


def _way(tags: dict, verts: list[tuple[float, float]]) -> dict:
    """Build a synthetic Overpass way element with inline geometry."""
    return {"type": "way", "tags": tags, "geometry": [{"lon": x, "lat": y} for x, y in verts]}


def _convert(elements: list[dict], dem: _FakeDEM | None = None):
    return OSMImporter(dem=dem or _FakeDEM()).convert(region=REGION, elements=elements)


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
        assert summary.pistes[0].name == "7"  # falls back name→piste:name→piste:ref→ref
        assert summary.pistes[1].name is None  # 39% of real runs are unnamed


class TestResample:
    def test_spacing_and_dem_z(self) -> None:
        # ~2.2 km straight run, 30 m step → many points, all draped on the DEM.
        summary = _convert([_way({"piste:type": "downhill"}, [(0.0, 0.0), (0.0, -0.02)])])
        pts = summary.pistes[0].points
        assert len(pts) > 50, "long run resampled to many points"
        assert all(p.elevation == 2500.0 - p.lat * M * 0.20 for p in pts), "Z is the DEM value"
        # Consecutive spacing ≈ RESAMPLE_STEP_M (30 m), not the raw 2-vertex polyline.
        gaps = [pts[i].distance_to(other=pts[i + 1]) for i in range(len(pts) - 1)]
        assert max(gaps) < 40.0, f"no gap far above the 30 m step, got {max(gaps):.0f}"

    def test_two_vertex_run_keeps_endpoints(self) -> None:
        # A 2-vertex run above the min length resamples to its endpoints plus interior points.
        summary = _convert([_way({"piste:type": "downhill"}, [(0.0, 0.0), (0.0, -0.01)])])
        pts = summary.pistes[0].points
        assert (pts[0].lat, pts[0].lon) == (0.0, 0.0)
        assert (pts[-1].lat, pts[-1].lon) == (-0.01, 0.0)


class TestFullOnly:
    def test_truncated_way_skipped_entirely(self) -> None:
        # A way with a vertex outside the circle (far east) is dropped whole, not clipped.
        summary = _convert([_way({"piste:type": "downhill"}, [(0.0, 0.0), (0.5, 0.0)])])
        assert summary.pistes == [] and summary.skipped == 1

    def test_nodata_way_skipped_entirely(self) -> None:
        # Fully inside the circle but over a DEM nodata hole → not fully sampleable → skipped.
        dem = _FakeDEM(hole_lon=0.0)  # any lon > 0 is nodata
        summary = _convert([_way({"piste:type": "downhill"}, [(0.001, 0.0), (0.001, -0.02)])], dem=dem)
        assert summary.pistes == [] and summary.skipped == 1


class TestLifts:
    def test_aerialway_type_map_and_orientation(self) -> None:
        # bottom vertex given first but at higher lat (=higher elev on south slope); importer
        # must orient bottom = lower station.
        summary = _convert([_way({"aerialway": "gondola"}, [(0.0, 0.0), (0.0, -0.02)])])
        assert len(summary.lifts) == 1
        lift = summary.lifts[0]
        assert lift.lift_type == "gondola"
        assert lift.bottom.elevation < lift.top.elevation, "bottom is the lower station"

    def test_all_four_lift_types(self) -> None:
        elements = [
            _way({"aerialway": "t-bar"}, [(0.0, 0.0), (0.0, -0.01)]),
            _way({"aerialway": "chair_lift"}, [(0.001, 0.0), (0.001, -0.01)]),
            _way({"aerialway": "mixed_lift"}, [(0.002, 0.0), (0.002, -0.01)]),
            _way({"aerialway": "cable_car"}, [(0.003, 0.0), (0.003, -0.01)]),
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
        # A gondola under MIN_LIFT_LENGTH_M (500 m) is a nursery lift → skipped and counted.
        summary = _convert([_way({"aerialway": "gondola"}, [(0.0, 0.0), (0.0, -0.002)])])  # ~223 m
        assert summary.lifts == [] and summary.skipped == 1

    def test_lift_name_resolution_and_unnamed(self) -> None:
        named = _way({"aerialway": "chair_lift", "name": "Gipfelbahn"}, [(0.0, 0.0), (0.0, -0.02)])
        by_ref = _way({"aerialway": "chair_lift", "ref": "B7"}, [(0.001, 0.0), (0.001, -0.02)])
        unnamed = _way({"aerialway": "chair_lift"}, [(0.002, 0.0), (0.002, -0.02)])
        names = {lift.name for lift in _convert([named, by_ref, unnamed]).lifts}
        assert names == {"Gipfelbahn", "B7", None}, "name→ref→None resolution"


class TestMinLength:
    def test_short_piste_skipped(self) -> None:
        # A downhill run under MIN_PISTE_LENGTH_M (300 m) is a stub → skipped and counted.
        summary = _convert([_way({"piste:type": "downhill"}, [(0.0, 0.0), (0.0, -0.002)])])  # ~223 m
        assert summary.pistes == [] and summary.skipped == 1

    def test_long_piste_imported(self) -> None:
        # A run comfortably over the minimum imports normally.
        summary = _convert([_way({"piste:type": "downhill"}, [(0.0, 0.0), (0.0, -0.004)])])  # ~445 m
        assert len(summary.pistes) == 1 and summary.skipped == 0


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

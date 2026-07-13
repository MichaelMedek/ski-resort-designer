"""Unit tests for whole-path junction smoothing (model/path_smoothing.py).

Builds an explicit sharp-kink two-segment path and asserts the smoother rounds the shape
between nodes while pinning EVERY node (outer endpoints + the junction) exactly onto the
ribbon, preserves the segment split, and never raises.
"""

import math

from skiresort_planner.core.geo_calculator import GeoCalculator
from skiresort_planner.model.path_point import PathPoint
from skiresort_planner.model.path_smoothing import resample_cubic_spline, smooth_joined_path

M = 111320.0  # metres per degree near the equator


def _leg(start_lon: float, start_lat: float, d_lon: float, d_lat: float, n: int, z0: float, dz: float) -> list:
    """A straight leg of n points stepping (d_lon, d_lat) per point, elevation z0 + i*dz."""
    return [PathPoint(lon=start_lon + d_lon * i, lat=start_lat + d_lat * i, elevation=z0 + dz * i) for i in range(n)]


def _anchors(segs: list[list[PathPoint]]) -> list[PathPoint]:
    """Boundary node coords for a joined path: first start, then each segment's end."""
    return [segs[0][0], *(seg[-1] for seg in segs)]


def _turn_deg(a: PathPoint, b: PathPoint, c: PathPoint) -> float:
    """Absolute heading change (deg) at b for the polyline a->b->c."""
    h1 = GeoCalculator.initial_bearing_deg(lon1=a.lon, lat1=a.lat, lon2=b.lon, lat2=b.lat)
    h2 = GeoCalculator.initial_bearing_deg(lon1=b.lon, lat1=b.lat, lon2=c.lon, lat2=c.lat)
    d = abs(h1 - h2) % 360
    return d if d <= 180 else 360 - d


def _sharp_L_path() -> list[list[PathPoint]]:
    """Two legs meeting at a ~90° corner: east then north, ~10m point spacing, gentle descent."""
    step = 10 / M
    seg1 = _leg(0.0, 0.0, step, 0.0, 30, z0=2100.0, dz=-0.5)  # heads east
    j = seg1[-1]
    seg2 = _leg(j.lon, j.lat, 0.0, step, 30, z0=j.elevation, dz=-0.5)  # heads north
    return [seg1, seg2]


class TestSmoothJoinedPath:
    def test_reduces_junction_turn_angle(self) -> None:
        segs = _sharp_L_path()
        # Raw junction is a hard ~90° corner.
        raw_turn = _turn_deg(segs[0][-2], segs[0][-1], segs[1][1])
        assert raw_turn > 60

        out = smooth_joined_path(segment_point_lists=segs, node_anchors=_anchors(segs))
        # After smoothing, the max turn anywhere along the joined path is a smooth (not kinked) bend.
        joined = out[0] + out[1][1:]
        max_turn = max(_turn_deg(joined[i - 1], joined[i], joined[i + 1]) for i in range(1, len(joined) - 1))
        assert max_turn < raw_turn, f"junction should round (raw {raw_turn:.1f} -> {max_turn:.1f})"
        assert max_turn < 30, f"junction should be a smooth bend <30deg, got {max_turn:.1f}"

    def test_all_nodes_pinned_onto_ribbon(self) -> None:
        # Every node (start, junction, end) must sit EXACTLY on the smoothed ribbon so its
        # marker connects and any node can be a branch point — this is the whole purpose.
        segs = _sharp_L_path()
        anchors = _anchors(segs)
        out = smooth_joined_path(segment_point_lists=segs, node_anchors=anchors)
        assert out[0][0] == anchors[0], "start node pinned"
        assert out[0][-1] == anchors[1], "junction node pinned on segment 1 end"
        assert out[1][0] == anchors[1], "junction node pinned on segment 2 start"
        assert out[-1][-1] == anchors[2], "end node pinned"

    def test_junction_shared_by_value(self) -> None:
        segs = _sharp_L_path()
        out = smooth_joined_path(segment_point_lists=segs, node_anchors=_anchors(segs))
        assert len(out) == 2, "segment count preserved"
        assert out[0][-1] == out[1][0], "adjacent segments must share the junction point by value"
        assert len(out[0]) >= 2 and len(out[1]) >= 2, "each segment keeps >=2 points"

    def test_three_segments_pins_every_junction(self) -> None:
        step = 10 / M
        s1 = _leg(0.0, 0.0, step, 0.0, 20, z0=2100.0, dz=-0.5)
        s2 = _leg(s1[-1].lon, s1[-1].lat, 0.0, step, 20, z0=s1[-1].elevation, dz=-0.5)
        s3 = _leg(s2[-1].lon, s2[-1].lat, step, 0.0, 20, z0=s2[-1].elevation, dz=-0.5)
        segs = [s1, s2, s3]
        anchors = _anchors(segs)
        out = smooth_joined_path(segment_point_lists=segs, node_anchors=anchors)
        assert len(out) == 3
        assert out[0][-1] == anchors[1] and out[1][0] == anchors[1], "junction 1 pinned + shared"
        assert out[1][-1] == anchors[2] and out[2][0] == anchors[2], "junction 2 pinned + shared"

    def test_single_segment_returned_unchanged(self) -> None:
        seg = _leg(0.0, 0.0, 10 / M, 0.0, 20, z0=2000.0, dz=-0.5)
        out = smooth_joined_path(segment_point_lists=[seg], node_anchors=[seg[0], seg[-1]])
        assert out == [seg]

    def test_short_path_returns_inputs_unchanged(self) -> None:
        # Fewer than 4 joined points → spline can't fit; inputs come back untouched.
        a = [PathPoint(lon=0.0, lat=0.0, elevation=2000.0), PathPoint(lon=1 / M, lat=0.0, elevation=1999.0)]
        b = [PathPoint(lon=1 / M, lat=0.0, elevation=1999.0), PathPoint(lon=2 / M, lat=0.0, elevation=1998.0)]
        out = smooth_joined_path(segment_point_lists=[a, b], node_anchors=[a[0], a[-1], b[-1]])
        assert out == [a, b]


class TestResampleCubicSpline:
    def test_too_few_points_unchanged(self) -> None:
        pts = [PathPoint(lon=0.0, lat=0.0, elevation=2000.0), PathPoint(lon=1 / M, lat=0.0, elevation=1999.0)]
        assert resample_cubic_spline(points=pts, smoothing_factor=3.0, step_m=7.0) is pts

    def test_resamples_a_long_path(self) -> None:
        pts = _leg(0.0, 0.0, 10 / M, 0.0, 40, z0=2200.0, dz=-1.0)  # ~390m, jittered-free straight
        out = resample_cubic_spline(points=pts, smoothing_factor=3.0, step_m=7.0)
        assert len(out) > 2
        # Endpoints stay close to the input extremes (spline is not wildly off).
        assert math.isclose(out[0].lon, pts[0].lon, abs_tol=1e-4)

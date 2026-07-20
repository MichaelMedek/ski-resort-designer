"""Unit tests for whole-path junction smoothing (model/path_smoothing.py).

Builds explicit multi-segment paths and asserts the weighted-spline smoother rounds the
shape between nodes (no zero-speed cusp), pins the outer endpoints exactly, keeps internal
junctions near their node and shared by value, preserves the segment split, and never raises.
"""

import math

from scipy.interpolate import splev, splprep

from skiresort_planner.constants import GeometricTuningConfig, MapConfig
from skiresort_planner.core.geo_calculator import GeoCalculator
from skiresort_planner.model.path_point import PathPoint
from skiresort_planner.model.path_smoothing import resample_cubic_spline, smooth_joined_path, smooth_proposal_points


def _min_curvature_radius_m(points: list[PathPoint]) -> float:
    """Smallest turn radius (m) along a cubic through points. A cusp (sharp edge) → ~0."""
    lat0 = points[0].lat
    k = MapConfig.METERS_PER_DEGREE_EQUATOR * math.cos(math.radians(lat0))
    xs = [p.lon * k for p in points]
    ys = [p.lat * MapConfig.METERS_PER_DEGREE_EQUATOR for p in points]
    u = [0.0]
    for i in range(1, len(points)):
        u.append(u[-1] + max(1e-9, math.hypot(xs[i] - xs[i - 1], ys[i] - ys[i - 1])))
    tck, _ = splprep([xs, ys], u=u, s=0, k=3)
    uu = [u[-1] * t / 1500 for t in range(1501)]
    dx, dy = splev(uu, tck, der=1)
    ddx, ddy = splev(uu, tck, der=2)
    radii = [(dx[i] ** 2 + dy[i] ** 2) ** 1.5 / max(abs(dx[i] * ddy[i] - dy[i] * ddx[i]), 1e-9) for i in range(len(uu))]
    return float(min(radii))


def _leg(
    start_lon: float, start_lat: float, d_lon: float, d_lat: float, n: int, z0: float, dz: float
) -> list[PathPoint]:
    """A straight leg of n points stepping (d_lon, d_lat) per point, elevation z0 + i*dz."""
    return [PathPoint(lon=start_lon + d_lon * i, lat=start_lat + d_lat * i, elevation=z0 + dz * i) for i in range(n)]


def _anchors(segs: list[list[PathPoint]]) -> list[PathPoint]:
    """Boundary node coords for a joined path: first start, then each segment's end."""
    return [segs[0][0], *(seg[-1] for seg in segs)]


def _smooth(segs: list[list[PathPoint]], anchors: list[PathPoint] | None = None) -> list[list[PathPoint]]:
    """Call smooth_joined_path with the real GeometricTuningConfig knobs (slope factor).

    smooth_joined_path takes every knob explicitly (no import-time defaults), so tests pass
    them the same way production does.
    """
    return smooth_joined_path(
        segment_point_lists=segs,
        node_anchors=anchors if anchors is not None else _anchors(segs),
        step_m=GeometricTuningConfig.RESAMPLE_STEP_M,
        smoothing_factor=GeometricTuningConfig.SLOPE_SMOOTHING_FACTOR,
        node_weight=GeometricTuningConfig.NODE_WEIGHT,
        corridor_weight=GeometricTuningConfig.CORRIDOR_WEIGHT,
    )


def _turn_deg(a: PathPoint, b: PathPoint, c: PathPoint) -> float:
    """Absolute heading change (deg) at b for the polyline a->b->c."""
    h1 = GeoCalculator.initial_bearing_deg(lon1=a.lon, lat1=a.lat, lon2=b.lon, lat2=b.lat)
    h2 = GeoCalculator.initial_bearing_deg(lon1=b.lon, lat1=b.lat, lon2=c.lon, lat2=c.lat)
    d = abs(h1 - h2) % 360
    return d if d <= 180 else 360 - d


def _sharp_L_path() -> list[list[PathPoint]]:
    """Two legs meeting at a ~90° corner: east then north, ~10m point spacing, gentle descent."""
    step = 10 / MapConfig.METERS_PER_DEGREE_EQUATOR
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

        out = _smooth(segs)
        # After smoothing, the max turn anywhere along the joined path is a smooth (not kinked) bend.
        joined = out[0] + out[1][1:]
        max_turn = max(_turn_deg(joined[i - 1], joined[i], joined[i + 1]) for i in range(1, len(joined) - 1))
        assert max_turn < raw_turn, f"junction should round (raw {raw_turn:.1f} -> {max_turn:.1f})"
        assert max_turn < 30, f"junction should be a smooth bend <30deg, got {max_turn:.1f}"

    def test_no_cusp_at_junction(self) -> None:
        # The core guarantee: the smoothed ribbon rounds the corner with a real radius, not a
        # zero-speed CUSP (sharp edge). Regression for the switchback sharp-edge bug — an
        # over-heavy node weight used to collapse curvature to ~0 here.
        segs = _sharp_L_path()
        out = _smooth(segs)
        joined = out[0] + out[1][1:]
        min_radius = _min_curvature_radius_m(joined)
        assert min_radius > 1.0, f"smoothed curve must have a real turn radius (no cusp), got {min_radius:.2f}m"

    def test_nodes_on_ribbon_endpoints_exact_junctions_near(self) -> None:
        # Outer endpoints are pinned EXACTLY (entity termini, shared with other entities).
        # Internal junctions are shared by value and sit within a small tolerance of the node
        # — they are NOT snapped back exactly, so a switchback stays a smooth radius.
        segs = _sharp_L_path()
        anchors = _anchors(segs)
        out = _smooth(segs, anchors)
        assert out[0][0] == anchors[0], "start endpoint pinned exactly"
        assert out[-1][-1] == anchors[2], "end endpoint pinned exactly"
        assert out[0][-1] == out[1][0], "junction shared by value across the two segments"
        junction = out[0][-1]
        assert junction.distance_to(other=anchors[1]) < 15.0, "junction stays within a few metres of its node"

    def test_junction_shared_by_value(self) -> None:
        segs = _sharp_L_path()
        out = _smooth(segs)
        assert len(out) == 2, "segment count preserved"
        assert out[0][-1] == out[1][0], "adjacent segments must share the junction point by value"
        assert len(out[0]) >= 2 and len(out[1]) >= 2, "each segment keeps >=2 points"

    def test_three_segments_share_every_junction(self) -> None:
        step = 10 / MapConfig.METERS_PER_DEGREE_EQUATOR
        s1 = _leg(0.0, 0.0, step, 0.0, 20, z0=2100.0, dz=-0.5)
        s2 = _leg(s1[-1].lon, s1[-1].lat, 0.0, step, 20, z0=s1[-1].elevation, dz=-0.5)
        s3 = _leg(s2[-1].lon, s2[-1].lat, step, 0.0, 20, z0=s2[-1].elevation, dz=-0.5)
        segs = [s1, s2, s3]
        anchors = _anchors(segs)
        out = _smooth(segs, anchors)
        assert len(out) == 3
        assert out[0][-1] == out[1][0], "junction 1 shared by value"
        assert out[1][-1] == out[2][0], "junction 2 shared by value"
        assert out[0][-1].distance_to(other=anchors[1]) < 15.0, "junction 1 near its node"
        assert out[1][-1].distance_to(other=anchors[2]) < 15.0, "junction 2 near its node"
        # No cusp at EITHER junction — multi-junction paths were where the sharp-edge bug lived.
        joined = out[0] + out[1][1:] + out[2][1:]
        assert _min_curvature_radius_m(joined) > 1.0, "multi-junction ribbon must be cusp-free"

    def test_slope_hugs_terrain_more_than_road(self) -> None:
        # The road/slope smoothing split, as an invariant on the RELATIONSHIP (not a magic
        # number): over one identical path, a SLOPE must stay at least as close to the raw
        # committed corridor as a ROAD (skiers are flexible → less earthwork). This guards the
        # feature against a future edit swapping ROAD/SLOPE_SMOOTHING_FACTOR while leaving the
        # exact values free to be re-tuned.
        step = 10 / MapConfig.METERS_PER_DEGREE_EQUATOR
        s1 = _leg(0.0, 0.0, step, 0.0, 25, z0=2100.0, dz=-0.5)
        j = s1[-1]
        s2 = _leg(j.lon, j.lat, step * 0.7, step * 0.7, 25, z0=j.elevation, dz=-0.5)  # ~45° junction
        segs = [s1, s2]
        anchors = _anchors(segs)
        raw = s1 + s2[1:]

        def mean_terrain_deviation(smoothed: list[list[PathPoint]]) -> float:
            ribbon = smoothed[0] + smoothed[1][1:]
            return sum(min(p.distance_to(other=r) for r in raw) for p in ribbon) / len(ribbon)

        def smooth_with(factor: float) -> list[list[PathPoint]]:
            return smooth_joined_path(
                segment_point_lists=segs,
                node_anchors=anchors,
                step_m=GeometricTuningConfig.RESAMPLE_STEP_M,
                smoothing_factor=factor,
                node_weight=GeometricTuningConfig.NODE_WEIGHT,
                corridor_weight=GeometricTuningConfig.CORRIDOR_WEIGHT,
            )

        slope_dev = mean_terrain_deviation(smooth_with(GeometricTuningConfig.SLOPE_SMOOTHING_FACTOR))
        road_dev = mean_terrain_deviation(smooth_with(GeometricTuningConfig.ROAD_SMOOTHING_FACTOR))
        assert slope_dev <= road_dev, (
            f"slope must hug terrain at least as tightly as road (slope {slope_dev:.2f}m > road {road_dev:.2f}m — "
            "are ROAD/SLOPE_SMOOTHING_FACTOR swapped?)"
        )

    def test_single_segment_is_smoothed(self) -> None:
        # A single segment has no junction but is still smoothed: endpoints pinned exactly,
        # resampled at step spacing, and a jittery corridor rounded into a broad radius.
        step = 10 / MapConfig.METERS_PER_DEGREE_EQUATOR
        pts = [PathPoint(lon=step * i, lat=(step if i % 2 else 0.0), elevation=2000.0 - 0.5 * i) for i in range(20)]
        out = _smooth([pts], [pts[0], pts[-1]])
        assert len(out) == 1, "single segment stays a single segment"
        smoothed = out[0]
        assert smoothed[0] == pts[0], "start pinned exactly"
        assert smoothed[-1] == pts[-1], "end pinned exactly"
        assert _min_curvature_radius_m(smoothed) > _min_curvature_radius_m(pts), "zigzag rounded into a broader radius"

    def test_short_path_returns_inputs_unchanged(self) -> None:
        # Fewer than 4 joined points → spline can't fit; inputs come back untouched.
        a = [
            PathPoint(lon=0.0, lat=0.0, elevation=2000.0),
            PathPoint(lon=1 / MapConfig.METERS_PER_DEGREE_EQUATOR, lat=0.0, elevation=1999.0),
        ]
        b = [
            PathPoint(lon=1 / MapConfig.METERS_PER_DEGREE_EQUATOR, lat=0.0, elevation=1999.0),
            PathPoint(lon=2 / MapConfig.METERS_PER_DEGREE_EQUATOR, lat=0.0, elevation=1998.0),
        ]
        out = _smooth([a, b], [a[0], a[-1], b[-1]])
        assert out == [a, b]

    def test_elevation_smoothed_within_raw_band_and_junction_near_node(self) -> None:
        # The spline's 3rd dimension (elevation) must survive smoothing: a regression that
        # dropped/zeroed Z would push points far outside the raw descent band. On _sharp_L_path
        # both legs descend 0.5m/point, so raw elevation runs 2100.0 -> 2071.0.
        segs = _sharp_L_path()
        anchors = _anchors(segs)
        raw = segs[0] + segs[1][1:]
        raw_min = min(p.elevation for p in raw)
        raw_max = max(p.elevation for p in raw)
        out = _smooth(segs, anchors)
        ribbon = out[0] + out[1][1:]
        for p in ribbon:
            assert raw_min - 1.0 <= p.elevation <= raw_max + 1.0, (
                f"smoothed elevation {p.elevation:.2f} left the raw band "
                f"[{raw_min:.1f}, {raw_max:.1f}] — is the spline's 3rd dimension dropped?"
            )
        # The heavily-weighted junction node keeps its elevation (anchors[1] == seg1 end = 2085.5).
        assert abs(out[0][-1].elevation - anchors[1].elevation) < 2.0, "junction elevation stays near its node"

    def test_elevation_does_not_overshoot_on_steep_descent(self) -> None:
        # Elevation is now a shape-preserving PCHIP over arc length:
        # on a MONOTONE-descending corridor the smoothed elevation must also be monotone (never rise)
        # and never leave the input band.
        seg = _leg(
            8.0,
            46.0,
            8 / MapConfig.METERS_PER_DEGREE_EQUATOR,
            2 / MapConfig.METERS_PER_DEGREE_EQUATOR,
            30,
            z0=3000.0,
            dz=-8.0,
        )  # steep, strictly descending
        out = _smooth([seg], anchors=[seg[0], seg[-1]])[0]
        elevs = [p.elevation for p in out]
        assert max(elevs) <= 3000.0 + 0.5 and min(elevs) >= elevs[-1] - 0.5, "elevation left the input band"
        rises = [(i, elevs[i - 1], elevs[i]) for i in range(1, len(elevs)) if elevs[i] > elevs[i - 1] + 0.1]
        assert not rises, f"monotone descent must not develop rises (overshoot): {rises[:3]}"


class TestResampleCubicSpline:
    def test_too_few_points_unchanged(self) -> None:
        pts = [
            PathPoint(lon=0.0, lat=0.0, elevation=2000.0),
            PathPoint(lon=1 / MapConfig.METERS_PER_DEGREE_EQUATOR, lat=0.0, elevation=1999.0),
        ]
        assert resample_cubic_spline(points=pts, smoothing_factor=3.0, step_m=7.0) is pts

    def test_resamples_a_long_path(self) -> None:
        pts = _leg(
            0.0, 0.0, 10 / MapConfig.METERS_PER_DEGREE_EQUATOR, 0.0, 40, z0=2200.0, dz=-1.0
        )  # ~390m, jittered-free straight
        out = resample_cubic_spline(points=pts, smoothing_factor=3.0, step_m=7.0)
        assert len(out) > 2
        # Endpoints stay close to the input extremes (spline is not wildly off).
        assert math.isclose(out[0].lon, pts[0].lon, abs_tol=1e-4)


class TestSmoothProposalPoints:
    """smooth_proposal_points: spline-round then DEM-requery — shared by the fan and grid planner."""

    def test_requeries_elevation_from_the_callable(self) -> None:
        pts = _leg(0.0, 0.0, 10 / MapConfig.METERS_PER_DEGREE_EQUATOR, 0.0, 40, z0=2200.0, dz=-1.0)
        # A DEM stub returning a fixed elevation: every output point must take that value.
        out = smooth_proposal_points(points=pts, smoothing_factor=3.0, step_m=7.0, elevation_fn=lambda lon, lat: 1234.0)
        assert len(out) > 2
        assert all(p.elevation == 1234.0 for p in out), "elevations come from the DEM callable, not the spline"

    def test_falls_back_to_point_elevation_when_dem_none(self) -> None:
        pts = _leg(0.0, 0.0, 10 / MapConfig.METERS_PER_DEGREE_EQUATOR, 0.0, 40, z0=2200.0, dz=-1.0)
        out = smooth_proposal_points(points=pts, smoothing_factor=3.0, step_m=7.0, elevation_fn=lambda lon, lat: None)
        assert all(p.elevation is not None for p in out), "None DEM lookup falls back to the spline elevation"

    def test_too_short_returns_input_unchanged(self) -> None:
        pts = [
            PathPoint(lon=0.0, lat=0.0, elevation=2000.0),
            PathPoint(lon=1 / MapConfig.METERS_PER_DEGREE_EQUATOR, lat=0.0, elevation=1999.0),
        ]
        assert (
            smooth_proposal_points(points=pts, smoothing_factor=3.0, step_m=7.0, elevation_fn=lambda lon, lat: 0.0)
            is pts
        )

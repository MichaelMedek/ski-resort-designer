"""Tests for classify_segment_profile — bridge/tunnel/ground by deviation from the DEM surface."""

import numpy as np
import pytest

from skiresort_planner.model.path_point import PathPoint
from skiresort_planner.model.path_segment import PathSegment, SegmentKind
from skiresort_planner.model.segment_profile import (
    SegmentProfile,
    classify_segment_profile,
)
from tests_workflow.conftest import MockDEMService

# Flat DEM at 1000m everywhere (no slope), so a point's deviation IS its elevation minus 1000.
_FLAT_DEM = MockDEMService(base_elevation=1000.0, slope_ns_pct=0.0, slope_ew_pct=0.0)
_THRESHOLD_M = 15.0


def _segment(elevations: list[float]) -> PathSegment:
    """Build a straight segment near the origin whose points carry the given elevations."""
    pts = [PathPoint(lon=0.0001 * i, lat=0.0, elevation=e) for i, e in enumerate(elevations)]
    return PathSegment(id="S1", points=pts, kind=SegmentKind.SLOPE)


def test_floats_above_terrain_is_bridge() -> None:
    seg = _segment([1000.0, 1030.0, 1000.0])  # +30m above the 1000m ground
    res = classify_segment_profile(segment=seg, dem=_FLAT_DEM, threshold_m=_THRESHOLD_M)
    assert res.profile == SegmentProfile.BRIDGE
    assert res.max_above_m == pytest.approx(30.0)
    assert res.max_below_m == pytest.approx(0.0)


def test_cuts_below_terrain_is_tunnel() -> None:
    seg = _segment([1000.0, 970.0, 1000.0])  # -30m below ground
    res = classify_segment_profile(segment=seg, dem=_FLAT_DEM, threshold_m=_THRESHOLD_M)
    assert res.profile == SegmentProfile.TUNNEL
    assert res.max_below_m == pytest.approx(30.0)
    assert res.max_above_m == pytest.approx(0.0)


def test_within_threshold_is_ground() -> None:
    seg = _segment([1000.0, 1010.0, 995.0])  # ±10m, both under the 15m bar
    res = classify_segment_profile(segment=seg, dem=_FLAT_DEM, threshold_m=_THRESHOLD_M)
    assert res.profile == SegmentProfile.GROUND


def test_both_directions_larger_magnitude_wins() -> None:
    # +20m above but -40m below → below dominates → TUNNEL.
    seg = _segment([1000.0, 1020.0, 960.0])
    res = classify_segment_profile(segment=seg, dem=_FLAT_DEM, threshold_m=_THRESHOLD_M)
    assert res.profile == SegmentProfile.TUNNEL
    assert res.max_above_m == pytest.approx(20.0)
    assert res.max_below_m == pytest.approx(40.0)


def test_equal_magnitude_ties_to_bridge() -> None:
    # +30m above and -30m below → tie → BRIDGE (max_above_m >= max_below_m).
    seg = _segment([1030.0, 970.0])
    res = classify_segment_profile(segment=seg, dem=_FLAT_DEM, threshold_m=_THRESHOLD_M)
    assert res.profile == SegmentProfile.BRIDGE


def test_out_of_coverage_point_is_skipped_not_crashed() -> None:
    # A LOADED backup may sit partly off the current DEM (other region / updated file). Off-DEM points
    # come back NaN and are SKIPPED; the in-coverage points still classify (here +50m → BRIDGE).
    class _NaNDem(MockDEMService):
        def get_elevations(self, lons, lats):  # noqa: ANN001, ANN201
            out = np.full(len(list(lons)), 1000.0, dtype=float)
            out[0] = np.nan  # first point off-coverage
            return out

    dem = _NaNDem(base_elevation=1000.0, slope_ns_pct=0.0, slope_ew_pct=0.0)
    seg = _segment([1000.0, 1050.0])
    res = classify_segment_profile(segment=seg, dem=dem, threshold_m=_THRESHOLD_M)
    assert res.profile == SegmentProfile.BRIDGE
    assert res.max_above_m == pytest.approx(50.0)


def test_all_points_off_coverage_is_ground() -> None:
    # Every point off the current DEM (backup from a different region) → no deviation to measure → GROUND.
    class _AllNaNDem(MockDEMService):
        def get_elevations(self, lons, lats):  # noqa: ANN001, ANN201
            return np.full(len(list(lons)), np.nan, dtype=float)

    dem = _AllNaNDem(base_elevation=1000.0, slope_ns_pct=0.0, slope_ew_pct=0.0)
    seg = _segment([1000.0, 1050.0, 900.0])
    res = classify_segment_profile(segment=seg, dem=dem, threshold_m=_THRESHOLD_M)
    assert res.profile == SegmentProfile.GROUND
    assert res.max_above_m == pytest.approx(0.0)
    assert res.max_below_m == pytest.approx(0.0)

"""Unit tests for PathPoint (model/path_point.py) — distance + validation."""

import math

import pytest

from skiresort_planner.model.path_point import PathPoint


class TestPathPoint:
    """Tests for PathPoint data structure and distance calculations."""

    def test_distance_to_another_point(self) -> None:
        """PathPoint.distance_to() calculates correct haversine distance."""
        p1 = PathPoint(lon=10.0, lat=46.0, elevation=2000.0)
        p2 = PathPoint(lon=10.001, lat=46.001, elevation=2100.0)

        # Independent oracle: haversine over R=6_371_000 m for a 0.001deg NE step at lat 46 ~= 135.4 m.
        assert abs(p1.distance_to(other=p2) - 135.4) < 0.5
        assert p1.distance_to(other=p1) == 0.0

    def test_nan_elevation_raises(self) -> None:
        """PathPoint with NaN elevation raises ValueError."""
        with pytest.raises(ValueError, match="NaN"):
            PathPoint(lon=10.0, lat=46.0, elevation=math.nan)


class TestCumulativeDistances:
    """PathPoint.cumulative_distances — the single source for polyline arc length."""

    def test_starts_at_zero_and_accumulates(self) -> None:
        pts = [
            PathPoint(lon=10.0, lat=46.0, elevation=2000.0),
            PathPoint(lon=10.001, lat=46.0, elevation=1990.0),
            PathPoint(lon=10.002, lat=46.0, elevation=1980.0),
        ]
        cum = PathPoint.cumulative_distances(pts)
        assert len(cum) == len(pts)
        assert cum[0] == 0.0
        # Monotonic non-decreasing, and each step equals the pairwise distance.
        assert cum[1] == pytest.approx(pts[0].distance_to(other=pts[1]))
        assert cum[2] == pytest.approx(cum[1] + pts[1].distance_to(other=pts[2]))
        assert cum[2] > cum[1] > cum[0]

    def test_single_point_is_zero(self) -> None:
        assert PathPoint.cumulative_distances([PathPoint(lon=10.0, lat=46.0, elevation=2000.0)]) == [0.0]

    def test_empty_is_zero_seed(self) -> None:
        # Seeded with [0.0]; no pairs to accumulate.
        assert PathPoint.cumulative_distances([]) == [0.0]


class TestTotalLength:
    """PathPoint.total_length_m — single source for polyline length (was inline-duplicated)."""

    def test_matches_cumulative_last(self) -> None:
        pts = [
            PathPoint(lon=10.0, lat=46.0, elevation=2000.0),
            PathPoint(lon=10.001, lat=46.0, elevation=1990.0),
            PathPoint(lon=10.002, lat=46.0, elevation=1980.0),
        ]
        assert PathPoint.total_length_m(pts) == pytest.approx(PathPoint.cumulative_distances(pts)[-1])
        assert PathPoint.total_length_m(pts) > 0

    def test_fewer_than_two_points_is_zero(self) -> None:
        assert PathPoint.total_length_m([]) == 0.0
        assert PathPoint.total_length_m([PathPoint(lon=10.0, lat=46.0, elevation=2000.0)]) == 0.0

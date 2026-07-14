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

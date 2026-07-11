"""Unit tests for PathPoint (model/path_point.py) — distance + validation."""

import math

import pytest

from skiresort_planner.core.geo_calculator import GeoCalculator
from skiresort_planner.model.path_point import PathPoint


class TestPathPoint:
    """Tests for PathPoint data structure and distance calculations."""

    def test_distance_to_another_point(self) -> None:
        """PathPoint.distance_to() calculates correct haversine distance."""
        p1 = PathPoint(lon=10.0, lat=46.0, elevation=2000.0)
        p2 = PathPoint(lon=10.001, lat=46.001, elevation=2100.0)

        expected_dist = GeoCalculator.haversine_distance_m(lat1=p1.lat, lon1=p1.lon, lat2=p2.lat, lon2=p2.lon)
        assert abs(p1.distance_to(other=p2) - expected_dist) < 0.1
        assert p1.distance_to(other=p1) == 0.0

    def test_nan_elevation_raises(self) -> None:
        """PathPoint with NaN elevation raises ValueError."""
        with pytest.raises(ValueError, match="NaN"):
            PathPoint(lon=10.0, lat=46.0, elevation=math.nan)

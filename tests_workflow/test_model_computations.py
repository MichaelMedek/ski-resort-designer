"""Unit tests for model computed properties using parametrize.

Tests PathPoint distance calculation, segment properties, and slope/lift ID parsing.
"""

import math

import pytest

from skiresort_planner.constants import SlopeConfig
from skiresort_planner.core.geo_calculator import GeoCalculator
from skiresort_planner.model.lift import Lift
from skiresort_planner.model.node import Node
from skiresort_planner.model.path_point import PathPoint
from skiresort_planner.model.proposed_path import ProposedSlopeSegment
from skiresort_planner.model.slope import Slope


# =============================================================================
# PATHPOINT TESTS
# =============================================================================


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


class TestNodeDistanceCalculation:
    """Tests for Node distance calculation."""

    def test_node_distance_to_point(self) -> None:
        """Node.distance_to() calculates correct distance to coordinates."""
        node = Node(id="N1", location=PathPoint(lon=10.0, lat=46.0, elevation=2000.0))

        assert node.distance_to(lon=10.0, lat=46.0) == 0.0
        dist_nearby = node.distance_to(lon=10.0001, lat=46.0001)
        assert 0 < dist_nearby < 100


# =============================================================================
# ID PARSING TESTS (COMBINED)
# =============================================================================


class TestIdParsing:
    """Parametrized tests for ID number extraction."""

    @pytest.mark.parametrize(
        "model_type,id_str,expected_number",
        [
            pytest.param("slope", "SL1", 1, id="slope_single_digit"),
            pytest.param("slope", "SL5", 5, id="slope_mid_digit"),
            pytest.param("slope", "SL10", 10, id="slope_double_digit"),
            pytest.param("slope", "SL123", 123, id="slope_triple_digit"),
            pytest.param("lift", "L1", 1, id="lift_single_digit"),
            pytest.param("lift", "L7", 7, id="lift_mid_digit"),
            pytest.param("lift", "L99", 99, id="lift_double_digit"),
        ],
    )
    def test_number_from_id(self, model_type: str, id_str: str, expected_number: int) -> None:
        """Slope/Lift.number_from_id() extracts numeric part from ID."""
        if model_type == "slope":
            result = Slope.number_from_id(slope_id=id_str)
        else:
            result = Lift.number_from_id(lift_id=id_str)
        assert result == expected_number


# =============================================================================
# PROPOSED SEGMENT TESTS
# =============================================================================


class TestProposedSegmentComputedProperties:
    """Tests for ProposedSlopeSegment computed metrics."""

    def test_computed_metrics_from_path_points(self, path_points_blue) -> None:
        """ProposedSlopeSegment computes drop, length, slope, difficulty."""
        segment = ProposedSlopeSegment(
            points=path_points_blue,
            target_slope_pct=20.0,
            target_difficulty="blue",
            sector_name="Test",
        )

        assert segment.total_drop_m > 0
        assert 750 < segment.length_m < 850
        assert 15 < segment.avg_slope_pct < 25
        assert segment.difficulty == "blue"


class TestMaxSlopeRollingWindow:
    """Tests for max_slope_pct rolling window algorithm."""

    def test_detects_steep_section_in_variable_terrain(self) -> None:
        """max_slope_pct rolling window detects steep section within gradual terrain."""
        window_m = SlopeConfig.ROLLING_WINDOW_M
        step_m = 100
        steps_per_section = max(3, (window_m // step_m) + 1)

        base_lon = 10.27
        lat_per_step = 0.0009

        sections = [
            (steps_per_section, 10.0),
            (steps_per_section, 45.0),
            (steps_per_section, 10.0),
        ]

        points = []
        lat = 46.97
        elev = 2500.0
        points.append(PathPoint(lon=base_lon, lat=lat, elevation=elev))

        for num_steps, drop in sections:
            for _ in range(num_steps):
                lat -= lat_per_step
                elev -= drop
                points.append(PathPoint(lon=base_lon, lat=lat, elevation=elev))

        seg = ProposedSlopeSegment(points=points)

        total_steps = steps_per_section * 3
        expected_length = total_steps * step_m

        assert seg.length_m > window_m
        assert expected_length * 0.9 < seg.length_m < expected_length * 1.1

        avg = seg.avg_slope_pct
        assert 15 < avg < 30

        assert seg.max_slope_pct > 40
        assert seg.max_slope_pct < 50

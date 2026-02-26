"""Unit tests for model computed properties.

Tests PathPoint distance calculation, segment properties, and slope/lift ID parsing.
"""

import pytest

from skiresort_planner.constants import SlopeConfig
from skiresort_planner.core.geo_calculator import GeoCalculator
from skiresort_planner.model.lift import Lift
from skiresort_planner.model.node import Node
from skiresort_planner.model.path_point import PathPoint
from skiresort_planner.model.proposed_path import ProposedSlopeSegment
from skiresort_planner.model.slope import Slope


class TestPathPoint:
    """Tests for PathPoint data structure."""

    def test_distance_to_another_point(self) -> None:
        """PathPoint.distance_to() calculates correct haversine distance.

        Tests:
        - Distance between two points matches GeoCalculator
        - Distance to same point is 0
        """
        p1 = PathPoint(lon=10.0, lat=46.0, elevation=2000.0)
        p2 = PathPoint(lon=10.001, lat=46.001, elevation=2100.0)

        # Distance between points should match GeoCalculator
        expected_dist = GeoCalculator.haversine_distance_m(lat1=p1.lat, lon1=p1.lon, lat2=p2.lat, lon2=p2.lon)
        assert abs(p1.distance_to(other=p2) - expected_dist) < 0.1, "Distance should match"

        # Distance to self should be 0
        assert p1.distance_to(other=p1) == 0.0, "Distance to self should be 0"

    def test_nan_elevation_raises(self) -> None:
        """PathPoint with NaN elevation raises ValueError."""
        import math

        with pytest.raises(ValueError, match="NaN"):
            PathPoint(lon=10.0, lat=46.0, elevation=math.nan)


class TestNodeDistanceCalculation:
    """Tests for Node distance calculation (delegates to PathPoint)."""

    def test_node_distance_to_point(self) -> None:
        """Node.distance_to() calculates correct distance to coordinates."""
        node = Node(
            id="N1",
            location=PathPoint(lon=10.0, lat=46.0, elevation=2000.0),
        )

        # Distance to same location should be 0
        dist = node.distance_to(lon=10.0, lat=46.0)
        assert dist == 0.0, "Distance to same coords should be 0"

        # Distance to nearby point should be small
        dist_nearby = node.distance_to(lon=10.0001, lat=46.0001)
        assert 0 < dist_nearby < 100, "Nearby point should be < 100m away"


class TestProposedSegmentComputedProperties:
    """Tests for ProposedSlopeSegment computed metrics."""

    def test_computed_metrics_from_path_points(self, path_points_blue) -> None:
        """ProposedSlopeSegment computes drop, length, slope, difficulty.

        Tests:
        - drop_m = start elevation - end elevation
        - length_m = sum of distances between consecutive points
        - avg_slope_pct = drop / length * 100
        - difficulty = classification from avg_slope
        """
        segment = ProposedSlopeSegment(
            points=path_points_blue,
            target_slope_pct=20.0,
            target_difficulty="blue",
            sector_name="Test",
        )

        # Drop should be positive (going downhill)
        assert segment.total_drop_m > 0, "Drop should be positive for downhill path"

        # Length should be approximately 800m (4 segments of 200m each)
        assert 750 < segment.length_m < 850, "Length should be ~800m"

        # Slope percentage should be around 20% (mock DEM slope)
        assert 15 < segment.avg_slope_pct < 25, "Slope should be ~20%"

        # Difficulty should be blue
        assert segment.difficulty == "blue", "20% slope should be blue difficulty"


class TestSlopeIdParsing:
    """Tests for Slope ID number extraction."""

    def test_number_from_id_extracts_correctly(self) -> None:
        """Slope.number_from_id() extracts numeric part from ID.

        Tests:
        - Single digit ID (SL1 → 1)
        - Multi-digit ID (SL123 → 123)
        """
        assert Slope.number_from_id(slope_id="SL1") == 1
        assert Slope.number_from_id(slope_id="SL5") == 5
        assert Slope.number_from_id(slope_id="SL10") == 10
        assert Slope.number_from_id(slope_id="SL123") == 123


class TestLiftIdParsing:
    """Tests for Lift ID number extraction."""

    def test_number_from_id_extracts_correctly(self) -> None:
        """Lift.number_from_id() extracts numeric part from ID.

        Tests:
        - Single digit ID (L1 → 1)
        - Multi-digit ID (L99 → 99)
        """
        assert Lift.number_from_id(lift_id="L1") == 1
        assert Lift.number_from_id(lift_id="L7") == 7
        assert Lift.number_from_id(lift_id="L99") == 99


class TestMaxSlopeRollingWindow:
    """Tests for max_slope_pct rolling window algorithm.

    The max_slope_pct property uses a rolling window to detect steep sections
    within a slope, which is critical for safety grading.
    """

    def test_detects_steep_section_in_variable_terrain(self) -> None:
        """max_slope_pct rolling window detects steep section within gradual terrain.

        Test scenario (total length adapts to ROLLING_WINDOW_M):
        - Section 1: Gradual at 10% slope
        - Section 2: Steep at 45% slope (longer than rolling window)
        - Section 3: Gradual at 10% slope

        The steep section should be detected by the rolling window algorithm,
        even when avg_slope_pct is much lower. This is critical for safety
        grading (black >= 40%).
        """
        window_m = SlopeConfig.ROLLING_WINDOW_M
        step_m = 100  # Distance per point
        steps_per_section = max(3, (window_m // step_m) + 1)  # Ensure section > window

        # Build points going south (lat decreases, 0.0009° ≈ 100m at 46°N)
        base_lon = 10.27
        lat_per_step = 0.0009  # ~100m per step

        # Define sections: (num_steps, drop_per_step)
        sections = [
            (steps_per_section, 10.0),  # Gradual: 10% slope
            (steps_per_section, 45.0),  # Steep: 45% slope (> window size)
            (steps_per_section, 10.0),  # Gradual: 10% slope
        ]

        points = []
        lat = 46.97
        elev = 2500.0

        # First point
        points.append(PathPoint(lon=base_lon, lat=lat, elevation=elev))

        # Add segments
        for num_steps, drop in sections:
            for _ in range(num_steps):
                lat -= lat_per_step
                elev -= drop
                points.append(PathPoint(lon=base_lon, lat=lat, elevation=elev))

        seg = ProposedSlopeSegment(points=points)

        # Verify geometry
        total_steps = steps_per_section * 3
        expected_length = total_steps * step_m

        assert seg.length_m > window_m, f"Path must be longer than window ({window_m}m)"
        assert expected_length * 0.9 < seg.length_m < expected_length * 1.1

        # Average slope: weighted by length, should be between gradual and steep
        avg = seg.avg_slope_pct
        assert 15 < avg < 30, f"Expected avg between gradual and steep, got {avg}%"

        # KEY TEST: max_slope_pct should find the steep 45% section
        assert seg.max_slope_pct > 40, f"Should find steep section >=40%, got {seg.max_slope_pct}"
        assert seg.max_slope_pct < 50, f"Should not exceed 45% steep section, got {seg.max_slope_pct}"

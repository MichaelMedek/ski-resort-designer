"""Tests for skiresort_planner generators module.

Tests: PathFactory, LeastCostPathPlanner (Dijkstra connection planner)
Focus: Path generation, difficulty variants, smoothness, slope matching

Note: Fixtures are defined in conftest.py (DEMService, path points, nodes, graphs).
"""

import pytest
from typing import TYPE_CHECKING
from skiresort_planner.constants import MapConfig, SlopeConfig
from skiresort_planner.core.geo_calculator import GeoCalculator
from skiresort_planner.core.path_tracer import PathTracer
from skiresort_planner.core.terrain_analyzer import TerrainAnalyzer
from skiresort_planner.generators.connection_planners import LeastCostPathPlanner
from skiresort_planner.generators.path_factory import PathFactory, Side

if TYPE_CHECKING:
    from skiresort_planner.core.dem_service import DEMService


# =============================================================================
# TESTS FOR PATH FACTORY
# =============================================================================


class TestPathFactory:
    """PathFactory - fan generation and path tracing."""

    def test_generate_fan_on_blue_terrain(self, mock_dem_blue_slope_south: "DEMService") -> None:
        """Fan generation yields multiple downhill paths on blue terrain."""
        factory = PathFactory(dem_service=mock_dem_blue_slope_south)
        paths = list(factory.generate_fan(lon=0.0, lat=0.0, target_length_m=300))

        # Should generate at least green paths on blue terrain
        assert len(paths) >= 2, "Should generate at least 2 paths (green left/right)"

        # All paths must go downhill
        for path in paths:
            start_elev = path.points[0].elevation
            end_elev = path.points[-1].elevation
            assert end_elev < start_elev, f"Path {path.sector_name} goes uphill: {start_elev} → {end_elev}"

    def test_generate_fan_path_difficulties(self, mock_dem_red_slope_southeast: "DEMService") -> None:
        """Fan on red terrain generates multiple difficulties."""
        factory = PathFactory(dem_service=mock_dem_red_slope_southeast)
        paths = list(factory.generate_fan(lon=0.0, lat=0.0, target_length_m=300))

        difficulties = {p.difficulty for p in paths}
        # Red terrain (25%) should support green and blue at minimum
        assert "green" in difficulties, "Should generate green paths"

    def test_path_factory_with_tracer(self, mock_dem_blue_slope_south: "DEMService") -> None:
        """PathFactory correctly uses injected PathTracer."""
        analyzer = TerrainAnalyzer(dem=mock_dem_blue_slope_south)
        tracer = PathTracer(dem=mock_dem_blue_slope_south, analyzer=analyzer)
        factory = PathFactory(
            dem_service=mock_dem_blue_slope_south,
            path_tracer=tracer,
            terrain_analyzer=analyzer,
        )

        paths = list(factory.generate_fan(lon=0.0, lat=0.0, target_length_m=300))
        assert len(paths) > 0, "Should generate paths with custom tracer"


class TestPathFactoryDifficulties:
    """PathFactory difficulty target values."""

    def test_difficulty_targets_configuration(self) -> None:
        """SlopeConfig has valid targets for all difficulty levels."""
        for diff in ["green", "blue", "red", "black"]:
            assert diff in SlopeConfig.DIFFICULTY_TARGETS
            targets = SlopeConfig.DIFFICULTY_TARGETS[diff]
            assert "gentle" in targets, f"{diff} missing gentle target"
            assert "steep" in targets, f"{diff} missing steep target"

            # Target slopes must be within difficulty thresholds
            low, high = SlopeConfig.DIFFICULTY_THRESHOLDS[diff]
            for grade, target in targets.items():
                assert low <= target <= high, f"{diff} {grade} target {target}% outside range [{low}, {high}]%"


# =============================================================================
# TESTS FOR LEAST COST PATH PLANNER
# =============================================================================


class TestLeastCostPathPlanner:
    """LeastCostPathPlanner - Dijkstra connection path algorithm."""

    def test_plan_valid_downhill_path(self, mock_dem_blue_slope_south: "DEMService") -> None:
        """Planner returns valid connector path for downhill target."""
        terrain_analyzer = TerrainAnalyzer(dem=mock_dem_blue_slope_south)
        planner = LeastCostPathPlanner(dem_service=mock_dem_blue_slope_south, terrain_analyzer=terrain_analyzer)
        M = MapConfig.METERS_PER_DEGREE_EQUATOR

        # Start at origin, target 300m south on 20% slope (60m drop)
        start_lon, start_lat = 0.0, 0.0
        target_lon, target_lat = 0.0, -300 / M

        start_elev = mock_dem_blue_slope_south.get_elevation(lon=start_lon, lat=start_lat)
        target_elev = mock_dem_blue_slope_south.get_elevation(lon=target_lon, lat=target_lat)
        assert start_elev is not None and target_elev is not None

        result = planner.plan(
            start_lon=start_lon,
            start_lat=start_lat,
            start_elevation=start_elev,
            target_lon=target_lon,
            target_lat=target_lat,
            target_elevation=target_elev,
            target_slope_pct=12.0,
            side="left",
        )

        assert result is not None, "Should find a path on suitable terrain"
        assert len(result.points) >= 2, "Path should have at least start and end"
        assert result.is_connector is True, "Connection path should have is_connector=True"

    def test_plan_returns_none_for_uphill_target(self, mock_dem_blue_slope_south: "DEMService") -> None:
        """Planner returns None when target is uphill from start."""
        terrain_analyzer = TerrainAnalyzer(dem=mock_dem_blue_slope_south)
        planner = LeastCostPathPlanner(dem_service=mock_dem_blue_slope_south, terrain_analyzer=terrain_analyzer)
        M = MapConfig.METERS_PER_DEGREE_EQUATOR

        # Start 300m south (lower), target at origin (higher) = uphill
        start_lon, start_lat = 0.0, -300 / M
        target_lon, target_lat = 0.0, 0.0

        start_elev = mock_dem_blue_slope_south.get_elevation(lon=start_lon, lat=start_lat)
        target_elev = mock_dem_blue_slope_south.get_elevation(lon=target_lon, lat=target_lat)
        assert start_elev is not None and target_elev is not None

        result = planner.plan(
            start_lon=start_lon,
            start_lat=start_lat,
            start_elevation=start_elev,
            target_lon=target_lon,
            target_lat=target_lat,
            target_elevation=target_elev,
            target_slope_pct=12.0,
            side="left",
        )

        assert result is None, "Should not find path to uphill target"

    def test_plan_returns_none_for_too_close_target(self, mock_dem_blue_slope_south: "DEMService") -> None:
        """Planner returns None when target is closer than step size."""
        terrain_analyzer = TerrainAnalyzer(dem=mock_dem_blue_slope_south)
        planner = LeastCostPathPlanner(dem_service=mock_dem_blue_slope_south, terrain_analyzer=terrain_analyzer)

        # Target just 5m away (less than STEP_SIZE_M)
        start_lon, start_lat = 0.0, 0.0
        target_lon, target_lat = 0.00001, -0.00001

        result = planner.plan(
            start_lon=start_lon,
            start_lat=start_lat,
            start_elevation=2500.0,
            target_lon=target_lon,
            target_lat=target_lat,
            target_elevation=2490.0,
            target_slope_pct=12.0,
            side="left",
        )

        assert result is None, "Should not find path to target closer than step size"


class TestLeastCostPathPlannerSmoothness:
    """LeastCostPathPlanner path smoothness - bearing change penalty."""

    def test_path_endpoints_match_inputs(self, mock_dem_blue_slope_south: "DEMService") -> None:
        """Path starts and ends near specified coordinates."""
        terrain_analyzer = TerrainAnalyzer(dem=mock_dem_blue_slope_south)
        planner = LeastCostPathPlanner(dem_service=mock_dem_blue_slope_south, terrain_analyzer=terrain_analyzer)
        M = MapConfig.METERS_PER_DEGREE_EQUATOR

        start_lon, start_lat = 0.0, 0.0
        target_lon, target_lat = 0.0, -300 / M

        start_elev = mock_dem_blue_slope_south.get_elevation(lon=start_lon, lat=start_lat)
        target_elev = mock_dem_blue_slope_south.get_elevation(lon=target_lon, lat=target_lat)
        assert start_elev is not None and target_elev is not None

        result = planner.plan(
            start_lon=start_lon,
            start_lat=start_lat,
            start_elevation=start_elev,
            target_lon=target_lon,
            target_lat=target_lat,
            target_elevation=target_elev,
            target_slope_pct=15.0,
            side="right",
        )

        assert result is not None

        # Start point should be within grid resolution of input
        start_dist = GeoCalculator.haversine_distance_m(
            lat1=start_lat,
            lon1=start_lon,
            lat2=result.points[0].lat,
            lon2=result.points[0].lon,
        )
        assert start_dist < 50, f"Start point {start_dist:.0f}m from input"

        # End point should be within grid resolution of target
        end_dist = GeoCalculator.haversine_distance_m(
            lat1=target_lat,
            lon1=target_lon,
            lat2=result.points[-1].lat,
            lon2=result.points[-1].lon,
        )
        assert end_dist < 50, f"End point {end_dist:.0f}m from target"


# =============================================================================
# INTEGRATION TESTS WITH REAL DEM (skipped if unavailable)
# =============================================================================


class TestGeneratorsWithRealDEM:
    """Integration tests using real EuroDEM data."""

    def test_path_factory_fan_on_real_terrain(self, real_dem_eurodem: "DEMService") -> None:
        """Fan generation works on real Alpine terrain."""
        factory = PathFactory(dem_service=real_dem_eurodem)
        # Ischgl area
        paths = list(factory.generate_fan(lon=10.27, lat=46.97, target_length_m=400))

        assert len(paths) > 0, "Should generate paths on real terrain"

        # All paths should go downhill
        for path in paths:
            assert path.total_drop_m > 0, f"Path {path.sector_name} doesn't go downhill"

    def test_connection_planner_on_real_terrain(self, real_dem_eurodem: "DEMService") -> None:
        """Dijkstra planner finds paths on real terrain."""
        terrain_analyzer = TerrainAnalyzer(dem=real_dem_eurodem)
        planner = LeastCostPathPlanner(dem_service=real_dem_eurodem, terrain_analyzer=terrain_analyzer)

        # Points in Ischgl area with known elevation drop
        start_lon, start_lat = 10.295, 46.987
        target_lon, target_lat = 10.298, 46.983

        start_elev = real_dem_eurodem.get_elevation(lon=start_lon, lat=start_lat)
        target_elev = real_dem_eurodem.get_elevation(lon=target_lon, lat=target_lat)
        assert start_elev is not None and target_elev is not None

        assert target_elev < start_elev - 10, "Target should be enough downhill from start for valid path"

        result = planner.plan(
            start_lon=start_lon,
            start_lat=start_lat,
            start_elevation=start_elev,
            target_lon=target_lon,
            target_lat=target_lat,
            target_elevation=target_elev,
            target_slope_pct=12.0,
            side="left",
        )

        assert result is not None, "Should find path on real terrain"
        assert len(result.points) >= 3, "Path should have multiple points"


"""Advanced tests for PathFactory - covering untested branches.

Focus on:
- generate_manual_paths (lines 356-420) - User-critical path generation
- _are_paths_similar / _deduplicate_paths (lines 272-330) - UX-critical deduplication
- _create_straight_line_path (lines 436-450) - Fallback path creation
- Edge cases with Hypothesis property-based testing

Reference: Coverage report showing 47% coverage in path_factory.py
"""

import pytest
from hypothesis import given, settings, strategies as st
from typing import TYPE_CHECKING

from skiresort_planner.constants import MapConfig, SlopeConfig
from skiresort_planner.core.terrain_analyzer import TerrainAnalyzer
from skiresort_planner.generators.path_factory import PathFactory, GradeConfig, Side
from skiresort_planner.model.path_point import PathPoint
from skiresort_planner.model.proposed_path import ProposedSlopeSegment

if TYPE_CHECKING:
    from skiresort_planner.core.dem_service import DEMService


# =============================================================================
# UNIT TESTS FOR _are_paths_similar
# =============================================================================


class TestArePathsSimilar:
    """Tests for path similarity comparison - lines 272-293."""

    @pytest.fixture
    def factory(self, mock_dem_blue_slope_south: "DEMService") -> PathFactory:
        return PathFactory(dem_service=mock_dem_blue_slope_south)

    def _make_path(self, points: list[tuple[float, float, float]]) -> ProposedSlopeSegment:
        """Helper to create ProposedSlopeSegment from (lon, lat, elev) tuples."""
        return ProposedSlopeSegment(
            points=[PathPoint(lon=p[0], lat=p[1], elevation=p[2]) for p in points],
            target_slope_pct=15.0,
            target_difficulty="blue",
            sector_name="Test Path",
            is_connector=False,
        )

    def test_identical_paths_are_similar(self, factory: PathFactory) -> None:
        """Two identical paths should be detected as similar."""
        path = self._make_path(
            [
                (10.0, 47.0, 2500),
                (10.001, 46.999, 2480),
                (10.002, 46.998, 2460),
                (10.003, 46.997, 2440),
            ]
        )
        assert factory._are_paths_similar(path1=path, path2=path) is True

    def test_very_different_paths_not_similar(self, factory: PathFactory) -> None:
        """Paths going in different directions should not be similar."""
        path1 = self._make_path(
            [
                (10.0, 47.0, 2500),
                (10.01, 46.99, 2400),
                (10.02, 46.98, 2300),
                (10.03, 46.97, 2200),
            ]
        )
        path2 = self._make_path(
            [
                (10.0, 47.0, 2500),
                (9.99, 46.99, 2400),
                (9.98, 46.98, 2300),
                (9.97, 46.97, 2200),
            ]
        )
        assert factory._are_paths_similar(path1=path1, path2=path2) is False

    def test_empty_paths_not_similar(self, factory: PathFactory) -> None:
        """Empty paths should not be considered similar."""
        empty_path = self._make_path([])
        normal_path = self._make_path(
            [
                (10.0, 47.0, 2500),
                (10.001, 46.999, 2480),
                (10.002, 46.998, 2460),
            ]
        )
        assert factory._are_paths_similar(path1=empty_path, path2=normal_path) is False
        assert factory._are_paths_similar(path1=empty_path, path2=empty_path) is False

    def test_short_paths_not_similar(self, factory: PathFactory) -> None:
        """Paths with fewer than 3 points should not be considered similar."""
        short_path = self._make_path(
            [
                (10.0, 47.0, 2500),
                (10.001, 46.999, 2480),
            ]
        )
        normal_path = self._make_path(
            [
                (10.0, 47.0, 2500),
                (10.001, 46.999, 2480),
                (10.002, 46.998, 2460),
            ]
        )
        assert factory._are_paths_similar(path1=short_path, path2=normal_path) is False

    def test_slightly_offset_paths_are_similar(self, factory: PathFactory) -> None:
        """Paths with small offset (within tolerance) should be similar."""
        # Offset of ~5m at 47° lat (1 degree lon ≈ 75km at 47° lat)
        tiny_offset = 0.00005  # ~5m

        path1 = self._make_path(
            [
                (10.0, 47.0, 2500),
                (10.001, 46.999, 2480),
                (10.002, 46.998, 2460),
                (10.003, 46.997, 2440),
            ]
        )
        path2 = self._make_path(
            [
                (10.0 + tiny_offset, 47.0, 2500),
                (10.001 + tiny_offset, 46.999, 2480),
                (10.002 + tiny_offset, 46.998, 2460),
                (10.003 + tiny_offset, 46.997, 2440),
            ]
        )
        assert factory._are_paths_similar(path1=path1, path2=path2) is True


# =============================================================================
# UNIT TESTS FOR _deduplicate_paths
# =============================================================================


class TestDeduplicatePaths:
    """Tests for path deduplication - lines 307-330."""

    @pytest.fixture
    def factory(self, mock_dem_blue_slope_south: "DEMService") -> PathFactory:
        return PathFactory(dem_service=mock_dem_blue_slope_south)

    def _make_path(self, points: list[tuple[float, float, float]]) -> ProposedSlopeSegment:
        """Helper to create path from (lon, lat, elev) tuples."""
        return ProposedSlopeSegment(
            points=[PathPoint(lon=p[0], lat=p[1], elevation=p[2]) for p in points],
            target_slope_pct=15.0,
            target_difficulty="blue",
            sector_name="Test Path",
            is_connector=False,
        )

    def test_empty_list_returns_empty(self, factory: PathFactory) -> None:
        """Empty input returns empty output."""
        assert factory._deduplicate_paths(paths=[]) == []

    def test_single_path_unchanged(self, factory: PathFactory) -> None:
        """Single path is returned unchanged."""
        path = self._make_path(
            [
                (10.0, 47.0, 2500),
                (10.001, 46.999, 2480),
                (10.002, 46.998, 2460),
            ]
        )
        result = factory._deduplicate_paths(paths=[path])
        assert len(result) == 1
        assert result[0] is path

    def test_keeps_gentlest_slope_when_duplicates(self, factory: PathFactory) -> None:
        """When paths are similar, keeps the one with gentlest avg_slope_pct.

        avg_slope_pct is computed from (total_drop / length) * 100.
        To get different slopes with similar horizontal paths, we use different
        elevation drops. Both paths have same start (2500m) and end horizontal,
        but different elevation profiles.
        """
        # Steep path: 200m drop over ~330m horizontal = ~60% slope
        steep_points: list[tuple[float, float, float]] = [
            (10.0, 47.0, 2700.0),
            (10.001, 46.999, 2600.0),
            (10.002, 46.998, 2500.0),
        ]

        # Gentle path: 20m drop over ~330m horizontal = ~6% slope
        gentle_points: list[tuple[float, float, float]] = [
            (10.0, 47.0, 2520.0),
            (10.001, 46.999, 2510.0),
            (10.002, 46.998, 2500.0),
        ]

        steep_path = self._make_path(points=steep_points)
        steep_path.sector_name = "Steep Version"

        gentle_path = self._make_path(points=gentle_points)
        gentle_path.sector_name = "Gentle Version"

        # Input order: steep first, gentle second
        result = factory._deduplicate_paths(paths=[steep_path, gentle_path])

        assert len(result) == 1, "Should remove duplicate"
        assert result[0].sector_name == "Gentle Version", "Should keep gentlest slope"

    def test_different_paths_both_kept(self, factory: PathFactory) -> None:
        """Paths going different directions should both be kept."""
        path_left = self._make_path(
            [
                (10.0, 47.0, 2500),
                (9.999, 46.999, 2480),
                (9.998, 46.998, 2460),
                (9.997, 46.997, 2440),
            ]
        )
        path_left.sector_name = "Left Path"

        path_right = self._make_path(
            [
                (10.0, 47.0, 2500),
                (10.001, 46.999, 2480),
                (10.002, 46.998, 2460),
                (10.003, 46.997, 2440),
            ]
        )
        path_right.sector_name = "Right Path"

        result = factory._deduplicate_paths(paths=[path_left, path_right])
        assert len(result) == 2, "Different paths should both be kept"


# =============================================================================
# TESTS FOR generate_manual_paths
# =============================================================================


class TestGenerateManualPaths:
    """Tests for generate_manual_paths - lines 356-420."""

    def test_generates_paths_for_valid_downhill_target(self, mock_dem_blue_slope_south: "DEMService") -> None:
        """Manual path generation works for valid downhill target."""
        factory = PathFactory(dem_service=mock_dem_blue_slope_south)
        M = MapConfig.METERS_PER_DEGREE_EQUATOR

        # Start at origin (2500m), target 300m south (lower elevation)
        start_lon, start_lat = 0.0, 0.0
        target_lon, target_lat = 0.0, -300 / M

        start_elev = mock_dem_blue_slope_south.get_elevation(lon=start_lon, lat=start_lat)
        target_elev = mock_dem_blue_slope_south.get_elevation(lon=target_lon, lat=target_lat)
        assert start_elev is not None and target_elev is not None

        paths = list(
            factory.generate_manual_paths(
                start_lon=start_lon,
                start_lat=start_lat,
                start_elevation=start_elev,
                target_lon=target_lon,
                target_lat=target_lat,
                target_elevation=target_elev,
            )
        )

        assert len(paths) >= 1, "Should generate at least one path"

        # All paths should have is_connector=True (from LeastCostPathPlanner)
        for path in paths:
            assert path.is_connector is True
            assert len(path.points) >= 2

    def test_fallback_straight_line_when_no_path_found(self, mock_dem_blue_slope_south: "DEMService") -> None:
        """When Dijkstra fails, creates straight-line fallback path."""
        factory = PathFactory(dem_service=mock_dem_blue_slope_south)
        M = MapConfig.METERS_PER_DEGREE_EQUATOR

        # Target very close (less than step size) - should trigger fallback
        start_lon, start_lat = 0.0, 0.0
        target_lon, target_lat = 0.00001, -5 / M  # ~5m south

        start_elev = mock_dem_blue_slope_south.get_elevation(lon=start_lon, lat=start_lat)
        target_elev = mock_dem_blue_slope_south.get_elevation(lon=target_lon, lat=target_lat)
        assert start_elev is not None and target_elev is not None

        paths = list(
            factory.generate_manual_paths(
                start_lon=start_lon,
                start_lat=start_lat,
                start_elevation=start_elev,
                target_lon=target_lon,
                target_lat=target_lat,
                target_elevation=target_elev,
            )
        )

        # Should get fallback path
        assert len(paths) >= 1, "Should always return at least fallback path"

    def test_returns_empty_for_no_elevation_at_target(self, mock_dem_blue_slope_south: "DEMService") -> None:
        """Returns empty when target elevation cannot be determined."""
        from unittest.mock import patch

        factory = PathFactory(dem_service=mock_dem_blue_slope_south)

        # Mock get_elevation to return None for target
        with patch.object(factory.dem_service, "get_elevation", return_value=None):
            paths = list(
                factory.generate_manual_paths(
                    start_lon=0.0,
                    start_lat=0.0,
                    start_elevation=2500.0,
                    target_lon=10.0,
                    target_lat=47.0,
                    target_elevation=None,  # Will query DEM, which returns None
                )
            )

        assert len(paths) == 0, "Should return empty when no elevation at target"


# =============================================================================
# TESTS FOR _create_straight_line_path
# =============================================================================


class TestCreateStraightLinePath:
    """Tests for fallback straight-line path - lines 436-450."""

    def test_creates_two_point_path(self, mock_dem_blue_slope_south: "DEMService") -> None:
        """Straight line path has exactly 2 points (start and end)."""
        factory = PathFactory(dem_service=mock_dem_blue_slope_south)

        path = factory._create_straight_line_path(
            start_lon=10.0,
            start_lat=47.0,
            start_elevation=2500.0,
            target_lon=10.001,
            target_lat=46.999,
            target_elevation=2400.0,
        )

        assert len(path.points) == 2
        assert path.points[0].lon == 10.0
        assert path.points[0].lat == 47.0
        assert path.points[0].elevation == 2500.0
        assert path.points[1].lon == 10.001
        assert path.points[1].lat == 46.999
        assert path.points[1].elevation == 2400.0

    def test_is_marked_as_connector(self, mock_dem_blue_slope_south: "DEMService") -> None:
        """Straight line path has is_connector=True."""
        factory = PathFactory(dem_service=mock_dem_blue_slope_south)

        path = factory._create_straight_line_path(
            start_lon=10.0,
            start_lat=47.0,
            start_elevation=2500.0,
            target_lon=10.001,
            target_lat=46.999,
            target_elevation=2400.0,
        )

        assert path.is_connector is True

    def test_has_fallback_sector_name(self, mock_dem_blue_slope_south: "DEMService") -> None:
        """Straight line path has 'fallback' in sector name."""
        factory = PathFactory(dem_service=mock_dem_blue_slope_south)

        path = factory._create_straight_line_path(
            start_lon=10.0,
            start_lat=47.0,
            start_elevation=2500.0,
            target_lon=10.001,
            target_lat=46.999,
            target_elevation=2400.0,
        )

        assert "fallback" in path.sector_name.lower()

    def test_difficulty_computed_from_actual_slope(self, mock_dem_blue_slope_south: "DEMService") -> None:
        """Difficulty is computed from actual slope, not target."""
        factory = PathFactory(dem_service=mock_dem_blue_slope_south)
        M = MapConfig.METERS_PER_DEGREE_EQUATOR

        # 100m drop over 200m distance = 50% slope = black
        path = factory._create_straight_line_path(
            start_lon=10.0,
            start_lat=47.0,
            start_elevation=2600.0,
            target_lon=10.0,
            target_lat=47.0 - (200 / M),
            target_elevation=2500.0,
        )

        # 50% slope should be black difficulty
        assert path.difficulty == "black", f"50% slope should be black, got {path.difficulty}"


# =============================================================================
# HYPOTHESIS PROPERTY-BASED TESTS
# =============================================================================


class _SimpleMockDEM:
    """Simple mock DEM for Hypothesis tests that don't need pytest fixtures."""

    def __init__(self) -> None:
        self._bounds = (-1.0, -1.0, 1.0, 1.0)

    @property
    def is_loaded(self) -> bool:
        return True

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        return self._bounds

    def get_elevation(self, lon: float, lat: float) -> float:
        return 2500.0 + lat * 111320 * 0.2  # 20% north-south slope


class TestPathFactoryHypothesis:
    """Property-based tests using Hypothesis for edge cases.

    Note: These tests don't use fixtures since Hypothesis doesn't work well
    with function-scoped pytest fixtures. Instead, they use _SimpleMockDEM inline.
    """

    @given(
        lon=st.floats(min_value=5.0, max_value=16.0, allow_nan=False),
        lat=st.floats(min_value=43.0, max_value=49.0, allow_nan=False),
        start_elev=st.floats(min_value=1000.0, max_value=4000.0, allow_nan=False),
        drop=st.floats(min_value=10.0, max_value=500.0, allow_nan=False),
    )
    @settings(max_examples=30)
    def test_straight_line_path_never_crashes(
        self,
        lon: float,
        lat: float,
        start_elev: float,
        drop: float,
    ) -> None:
        """Straight line path creation never raises exception for valid inputs.

        Tests _create_straight_line_path with randomly generated Alps coordinates.
        This is a standalone test that doesn't need DEM (straight line doesn't query it).
        """
        mock_dem = _SimpleMockDEM()
        factory = PathFactory(dem_service=mock_dem)  # type: ignore[arg-type]
        M = MapConfig.METERS_PER_DEGREE_EQUATOR

        # Always go downhill
        target_lat = lat - (100 / M)  # 100m south
        target_elev = start_elev - drop

        path = factory._create_straight_line_path(
            start_lon=lon,
            start_lat=lat,
            start_elevation=start_elev,
            target_lon=lon,
            target_lat=target_lat,
            target_elevation=target_elev,
        )

        # Should always produce valid path
        assert path is not None
        assert len(path.points) == 2
        assert path.is_connector is True
        assert path.points[0].elevation > path.points[1].elevation

    @given(
        drop1=st.floats(min_value=10.0, max_value=200.0, allow_nan=False),
        drop2=st.floats(min_value=10.0, max_value=200.0, allow_nan=False),
    )
    @settings(max_examples=30)
    def test_deduplication_always_keeps_gentlest(
        self,
        drop1: float,
        drop2: float,
    ) -> None:
        """Deduplication always keeps path with lowest avg_slope_pct.

        Creates two paths with same horizontal coordinates but different elevation
        drops to test that deduplication keeps the gentlest (lowest slope) one.
        """
        mock_dem = _SimpleMockDEM()
        factory = PathFactory(dem_service=mock_dem)  # type: ignore[arg-type]

        # Create two similar paths with different elevation drops
        # Same horizontal, different vertical = different slope
        base_lat = 47.0
        delta_lat = 0.003  # ~330m at Alps latitude

        path1 = ProposedSlopeSegment(
            points=[
                PathPoint(lon=10.0, lat=base_lat, elevation=2500.0),
                PathPoint(lon=10.001, lat=base_lat - delta_lat / 2, elevation=2500.0 - drop1 / 2),
                PathPoint(lon=10.002, lat=base_lat - delta_lat, elevation=2500.0 - drop1),
            ],
            target_slope_pct=15.0,
            target_difficulty="blue",
            sector_name="Path 1",
            is_connector=False,
        )

        path2 = ProposedSlopeSegment(
            points=[
                PathPoint(lon=10.0, lat=base_lat, elevation=2500.0),
                PathPoint(lon=10.001, lat=base_lat - delta_lat / 2, elevation=2500.0 - drop2 / 2),
                PathPoint(lon=10.002, lat=base_lat - delta_lat, elevation=2500.0 - drop2),
            ],
            target_slope_pct=15.0,
            target_difficulty="blue",
            sector_name="Path 2",
            is_connector=False,
        )

        result = factory._deduplicate_paths(paths=[path1, path2])

        assert len(result) == 1
        # Path with smaller drop has smaller slope (gentlest)
        # When drops are nearly equal (within floating point tolerance), either is acceptable
        epsilon = 1e-9
        if abs(drop1 - drop2) < epsilon:
            # Drops are essentially equal - either path is acceptable
            assert result[0].sector_name in ("Path 1", "Path 2")
        elif drop1 < drop2:
            assert result[0].sector_name == "Path 1"
        else:
            assert result[0].sector_name == "Path 2"


# =============================================================================
# TESTS FOR GradeConfig
# =============================================================================


class TestGradeConfig:
    """Tests for GradeConfig dataclass - line 83 (color property)."""

    def test_color_for_all_difficulties(self) -> None:
        """GradeConfig.color returns correct color for each difficulty."""
        from skiresort_planner.constants import StyleConfig

        for diff in ["green", "blue", "red", "black"]:
            config = GradeConfig(
                difficulty=diff,
                grade="gentle",
                target_slope_pct=10.0,
                side=Side.LEFT,
            )
            assert config.color == StyleConfig.SLOPE_COLORS[diff]

    def test_name_format(self) -> None:
        """GradeConfig.name has correct format."""
        config = GradeConfig(
            difficulty="blue",
            grade="steep",
            target_slope_pct=18.0,
            side=Side.RIGHT,
        )
        assert config.name == "Blue Right (Steep)"

        config_center = GradeConfig(
            difficulty="green",
            grade="gentle",
            target_slope_pct=8.0,
            side=Side.CENTER,
        )
        assert config_center.name == "Green Center (Gentle)"

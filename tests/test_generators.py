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

    def test_path_factory_fan_on_real_terrain(self, real_dem_eurodem) -> None:
        """Fan generation works on real Alpine terrain."""
        factory = PathFactory(dem_service=real_dem_eurodem)
        # Ischgl area
        paths = list(factory.generate_fan(lon=10.27, lat=46.97, target_length_m=400))

        assert len(paths) > 0, "Should generate paths on real terrain"

        # All paths should go downhill
        for path in paths:
            assert path.total_drop_m > 0, f"Path {path.sector_name} doesn't go downhill"

    def test_connection_planner_on_real_terrain(self, real_dem_eurodem) -> None:
        """Dijkstra planner finds paths on real terrain."""
        terrain_analyzer = TerrainAnalyzer(dem=real_dem_eurodem)
        planner = LeastCostPathPlanner(dem_service=real_dem_eurodem, terrain_analyzer=terrain_analyzer)

        # Points in Ischgl area with known elevation drop
        start_lon, start_lat = 10.295, 46.987
        target_lon, target_lat = 10.298, 46.983

        start_elev = real_dem_eurodem.get_elevation(lon=start_lon, lat=start_lat)
        target_elev = real_dem_eurodem.get_elevation(lon=target_lon, lat=target_lat)

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

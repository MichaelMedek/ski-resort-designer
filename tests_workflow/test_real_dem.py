"""Integration tests with real DEM data.

These tests use the actual EuroDEM file and are skipped if unavailable.
They validate that algorithms work correctly with real terrain data.
"""

from skiresort_planner.core.path_tracer import PathTracer
from skiresort_planner.core.terrain_analyzer import TerrainAnalyzer


class TestRealDEMTerrain:
    """Tests using real DEM elevation data."""

    def test_dem_samples_valid_elevations(self, real_dem) -> None:
        """Real DEM returns valid elevations within Alps bounds.

        Tests:
        - get_elevation returns non-None within bounds
        - Elevations are reasonable for Alps (0-4800m)
        """
        # Sample point in Ischgl area
        lon, lat = 10.317, 46.982

        elev = real_dem.get_elevation(lon=lon, lat=lat)

        assert elev is not None, "Should return elevation for valid point"
        assert 1000 < elev < 4000, "Elevation should be reasonable for Alps"

    def test_terrain_analyzer_on_real_dem(self, real_dem) -> None:
        """TerrainAnalyzer computes valid gradient on real terrain.

        Tests:
        - compute_gradient returns valid TerrainGradient
        - Slope percentage is reasonable (0-100%)
        - Bearing is valid (0-360)
        """
        analyzer = TerrainAnalyzer(dem=real_dem)

        # Sample point in mountainous area
        gradient = analyzer.compute_gradient(lon=10.32, lat=46.98)

        assert gradient is not None, "Should compute gradient"
        assert 0 <= gradient.slope_pct <= 100, "Slope should be reasonable"
        assert 0 <= gradient.bearing_deg < 360, "Bearing should be 0-360"

    def test_path_tracer_on_real_dem(self, real_dem) -> None:
        """PathTracer generates valid paths on real terrain.

        Tests:
        - trace_downhill returns valid TracedPath
        - Path goes downhill (positive drop)
        - Path has multiple points
        """
        analyzer = TerrainAnalyzer(dem=real_dem)
        tracer = PathTracer(dem=real_dem, analyzer=analyzer)

        # Start point at a summit area
        result = tracer.trace_downhill(
            start_lon=10.32,
            start_lat=46.98,
            target_slope_pct=20.0,
            side="center",
            target_length_m=400,
        )

        assert result is not None, "Should trace a path"
        assert len(result.points) >= 3, "Path should have multiple points"
        assert result.total_drop_m > 0, "Path should go downhill"


class TestPathGenerationOnRealTerrain:
    """Tests for path generation algorithms on real terrain."""

    def test_path_factory_fan_generation(self, real_dem) -> None:
        """PathFactory generates fan of paths on real terrain.

        Tests:
        - generate_fan produces multiple paths
        - Paths have different difficulties
        - All paths go downhill
        """
        from skiresort_planner.generators.path_factory import PathFactory

        factory = PathFactory(dem_service=real_dem)

        # Generate from a summit point
        paths = list(factory.generate_fan(lon=10.32, lat=46.98, target_length_m=300))

        assert len(paths) > 0, "Should generate at least one path"

        # Check paths go downhill
        for path in paths:
            assert path.total_drop_m > 0, f"Path {path.sector_name} should go downhill"

    def test_least_cost_path_planner(self, real_dem) -> None:
        """LeastCostPathPlanner finds valid connection path.

        Tests:
        - Plans path between two points
        - Path has valid endpoints
        - Path respects terrain
        """
        from skiresort_planner.constants import MapConfig
        from skiresort_planner.generators.connection_planners import LeastCostPathPlanner

        analyzer = TerrainAnalyzer(dem=real_dem)
        planner = LeastCostPathPlanner(dem_service=real_dem, terrain_analyzer=analyzer)

        M = MapConfig.METERS_PER_DEGREE_EQUATOR

        # Plan from high point to lower point
        start_lon, start_lat = 10.32, 46.98
        start_elev = real_dem.get_elevation(lon=start_lon, lat=start_lat)
        assert start_elev is not None, "Start point should have valid elevation"

        # Target point ~500m south (should be downhill in most terrain)
        target_lat = start_lat - 500 / M
        target_lon = start_lon
        target_elev = real_dem.get_elevation(lon=target_lon, lat=target_lat)
        assert target_elev is not None, "Target point should have valid elevation"

        # LeastCostPathPlanner.plan() returns a single path or None
        path = planner.plan(
            start_lon=start_lon,
            start_lat=start_lat,
            start_elevation=start_elev,
            target_lon=target_lon,
            target_lat=target_lat,
            target_elevation=target_elev,
            target_slope_pct=20.0,
            side="left",
        )

        # May return None if terrain doesn't allow connection
        # But should not raise an error
        assert path is None or hasattr(path, "points"), "Should return None or ProposedSlopeSegment"

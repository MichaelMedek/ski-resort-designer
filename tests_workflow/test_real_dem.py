"""Integration tests with real DEM data.

These tests use the actual EuroDEM file and are skipped if unavailable.
They validate that algorithms work correctly with real terrain data.
"""

from skiresort_planner.core.path_tracer import PathTracer
from skiresort_planner.core.terrain_analyzer import TerrainAnalyzer
from skiresort_planner.model.path_segment import SegmentKind


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

        # bearing_deg is the fall line (steepest DESCENT). The [0, 360) range is
        # merely the `% 360` modulo, so instead verify the direction is real:
        # stepping downhill along the bearing must reach lower terrain.
        from skiresort_planner.core.geo_calculator import GeoCalculator

        center_elev = real_dem.get_elevation(lon=10.32, lat=46.98)
        down_lon, down_lat = GeoCalculator.destination(
            lon=10.32, lat=46.98, bearing_deg=gradient.bearing_deg, distance_m=30.0
        )
        down_elev = real_dem.get_elevation(lon=down_lon, lat=down_lat)
        assert center_elev is not None and down_elev is not None, "sample points need valid elevation"
        assert down_elev < center_elev, "fall-line bearing must point downhill"

    def test_path_tracer_on_real_dem(self, real_dem) -> None:
        """PathTracer generates valid paths on real terrain.

        Tests:
        - trace_hill returns valid TracedPath
        - Path goes downhill (positive drop)
        - Path has multiple points
        """
        analyzer = TerrainAnalyzer(dem=real_dem)
        tracer = PathTracer(dem=real_dem, analyzer=analyzer)

        # Start point at a summit area
        result = tracer.trace_hill(
            start_lon=10.32,
            start_lat=46.98,
            target_grade_pct=20.0,
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
        - generate_slope_fan produces multiple paths
        - Paths have different difficulties
        - All paths go downhill
        """
        from skiresort_planner.generators.path_factory import PathFactory

        factory = PathFactory(dem_service=real_dem)

        # Generate from a summit point
        paths = list(factory.generate_fan(kind=SegmentKind.SLOPE, lon=10.32, lat=46.98, target_length_m=300))

        assert len(paths) > 0, "Should generate at least one path"

        # Check paths go downhill
        for path in paths:
            assert path.total_drop_m > 0, f"Path {path.sector_name} should go downhill"

    def test_least_cost_path_planner(self, real_dem) -> None:
        """LeastCostPathPlanner finds a valid downhill connection path.

        Tests:
        - Plans a path between two real points (start guaranteed higher)
        - Returned path genuinely connects start → target with ordered points
        """
        from skiresort_planner.constants import MapConfig
        from skiresort_planner.core.geo_calculator import GeoCalculator
        from skiresort_planner.generators.connection_planners import LeastCostPathPlanner

        analyzer = TerrainAnalyzer(dem=real_dem)
        planner = LeastCostPathPlanner(dem_service=real_dem, terrain_analyzer=analyzer)

        M = MapConfig.METERS_PER_DEGREE_EQUATOR

        # Two points ~500m apart on the N-S line. Slope-mode planning REQUIRES net
        # descent, so orient the pair by elevation: higher point is the start.
        lon = 10.32
        lat_a, lat_b = 46.98, 46.98 - 500 / M
        elev_a = real_dem.get_elevation(lon=lon, lat=lat_a)
        elev_b = real_dem.get_elevation(lon=lon, lat=lat_b)
        assert elev_a is not None and elev_b is not None, "Both sample points need valid elevation"

        (start_lat, start_elev), (target_lat, target_elev) = (
            ((lat_a, elev_a), (lat_b, elev_b)) if elev_a >= elev_b else ((lat_b, elev_b), (lat_a, elev_a))
        )

        path = planner.plan(
            start_lon=lon,
            start_lat=start_lat,
            start_elevation=start_elev,
            target_lon=lon,
            target_lat=target_lat,
            target_elevation=target_elev,
            target_grade_pct=20.0,
        )

        # With guaranteed descent over 500m of real terrain, a path must exist and
        # genuinely connect start → target with ordered points.
        assert path is not None, "downhill connection over 500m must yield a path"
        assert len(path.points) >= 2, "a real path has at least start and end points"
        first, last = path.points[0], path.points[-1]
        assert GeoCalculator.haversine_distance_m(lat1=first.lat, lon1=first.lon, lat2=start_lat, lon2=lon) < 60
        assert GeoCalculator.haversine_distance_m(lat1=last.lat, lon1=last.lon, lat2=target_lat, lon2=lon) < 60

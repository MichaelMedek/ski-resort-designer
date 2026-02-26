"""Unit tests for core mathematical functions.

Tests GeoCalculator geodesic functions and TerrainAnalyzer difficulty classification.
These are pure functions with deterministic outputs - perfect for unit testing.
"""

from skiresort_planner.core.geo_calculator import GeoCalculator
from skiresort_planner.core.path_tracer import PathTracer
from skiresort_planner.core.terrain_analyzer import TerrainAnalyzer


class TestGeoCalculator:
    """Tests for geodesic calculations."""

    def test_haversine_and_bearing_cardinal_directions(self) -> None:
        """Haversine distance and bearing for cardinal directions.

        Tests:
        - 1° latitude = ~111km
        - 1° longitude at 46°N = ~77km
        - Bearing north = 0°, south = 180°, east = 90°
        - destination() roundtrip consistent with haversine()
        """
        # 1 degree latitude should be ~111km
        dist_lat = GeoCalculator.haversine_distance_m(lat1=46.0, lon1=10.0, lat2=47.0, lon2=10.0)
        assert 110_000 < dist_lat < 112_000, "1° lat should be ~111km"

        # 1 degree longitude at 46°N should be ~77km (cos(46°) factor)
        dist_lon = GeoCalculator.haversine_distance_m(lat1=46.0, lon1=10.0, lat2=46.0, lon2=11.0)
        assert 76_000 < dist_lon < 78_000, "1° lon at 46°N should be ~77km"

        # North bearing should be ~0°
        bearing_north = GeoCalculator.initial_bearing_deg(lon1=10.0, lat1=46.0, lon2=10.0, lat2=47.0)
        assert bearing_north < 1 or bearing_north > 359, "North should be ~0°"

        # South bearing should be ~180°
        bearing_south = GeoCalculator.initial_bearing_deg(lon1=10.0, lat1=47.0, lon2=10.0, lat2=46.0)
        assert 179 < bearing_south < 181, "South should be ~180°"

        # East bearing should be ~90°
        bearing_east = GeoCalculator.initial_bearing_deg(lon1=10.0, lat1=46.0, lon2=11.0, lat2=46.0)
        assert 89 < bearing_east < 91, "East should be ~90°"

    def test_destination_roundtrip_consistency(self) -> None:
        """destination() should be consistent with haversine() distance."""
        lon_end, lat_end = GeoCalculator.destination(lon=10.0, lat=46.0, bearing_deg=45.0, distance_m=1000.0)
        dist_check = GeoCalculator.haversine_distance_m(lat1=46.0, lon1=10.0, lat2=lat_end, lon2=lon_end)
        assert abs(dist_check - 1000) < 10, "Roundtrip should be within 10m tolerance"


class TestDifficultyClassification:
    """Tests for slope difficulty classification."""

    def test_classify_difficulty_all_thresholds(self) -> None:
        """Difficulty classification at all threshold boundaries.

        Tests boundary values for green/blue/red/black classification.
        """
        # Green: 0-15%
        assert TerrainAnalyzer.classify_difficulty(slope_pct=0.0) == "green"
        assert TerrainAnalyzer.classify_difficulty(slope_pct=5.0) == "green"
        assert TerrainAnalyzer.classify_difficulty(slope_pct=14.9) == "green"

        # Blue: 15-25%
        assert TerrainAnalyzer.classify_difficulty(slope_pct=15.0) == "blue"
        assert TerrainAnalyzer.classify_difficulty(slope_pct=20.0) == "blue"
        assert TerrainAnalyzer.classify_difficulty(slope_pct=24.9) == "blue"

        # Red: 25-40%
        assert TerrainAnalyzer.classify_difficulty(slope_pct=25.0) == "red"
        assert TerrainAnalyzer.classify_difficulty(slope_pct=30.0) == "red"
        assert TerrainAnalyzer.classify_difficulty(slope_pct=39.9) == "red"

        # Black: 40%+
        assert TerrainAnalyzer.classify_difficulty(slope_pct=40.0) == "black"
        assert TerrainAnalyzer.classify_difficulty(slope_pct=60.0) == "black"
        assert TerrainAnalyzer.classify_difficulty(slope_pct=100.0) == "black"


class TestPathTracerOnMockTerrain:
    """Tests for path tracing algorithm."""

    def test_trace_downhill_produces_valid_diverging_paths(self, mock_dem_blue_slope) -> None:
        """PathTracer generates valid downhill paths with left/right divergence.

        Tests:
        - trace_downhill returns non-None on valid terrain
        - Path goes downhill (end elevation < start)
        - Left/right paths diverge significantly
        - Path length approximates target
        """
        tracer = PathTracer(dem=mock_dem_blue_slope)

        left = tracer.trace_downhill(
            start_lon=0.0,
            start_lat=0.0,
            target_slope_pct=15.0,
            side="left",
            target_length_m=300,
        )
        right = tracer.trace_downhill(
            start_lon=0.0,
            start_lat=0.0,
            target_slope_pct=15.0,
            side="right",
            target_length_m=300,
        )

        # Both paths should exist
        assert left is not None, "Left path should be generated"
        assert right is not None, "Right path should be generated"

        # Both should go downhill
        assert left.points[-1].elevation < left.points[0].elevation, "Left path should go downhill"
        assert right.points[-1].elevation < right.points[0].elevation, "Right path should go downhill"

        # Paths should diverge
        end_dist = GeoCalculator.haversine_distance_m(
            lat1=left.points[-1].lat,
            lon1=left.points[-1].lon,
            lat2=right.points[-1].lat,
            lon2=right.points[-1].lon,
        )
        assert end_dist > 30, "Left/right paths should diverge at endpoints"

        # Approximate target length (within 50%)
        assert 0.5 * 300 < left.length_m < 1.5 * 300, "Path should approximate target length"

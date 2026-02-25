"""Tests for skiresort_planner core functionality.

Tests: GeoCalculator, TerrainAnalyzer, PathTracer
Focus: Integration with mock DEM for deterministic tests, real DEM for integration

Note: Fixtures are defined in conftest.py (MockDEMService, real_dem_eurodem).
"""

from math import radians, cos
from typing import TYPE_CHECKING

import pytest

from skiresort_planner.constants import DEMConfig, MapConfig
from skiresort_planner.core.dem_service import DEMService
from skiresort_planner.core.geo_calculator import GeoCalculator
from skiresort_planner.core.path_tracer import PathTracer
from skiresort_planner.core.terrain_analyzer import TerrainAnalyzer
from skiresort_planner.model.path_point import PathPoint

if TYPE_CHECKING:
    from conftest import MockDEMService


# =============================================================================
# TESTS FOR CORE CLASSES
# =============================================================================


class TestGeoCalculator:
    """GeoCalculator - geodesic calculations on Earth's surface."""

    def test_haversine_distance_one_degree_latitude(self) -> None:
        """1 degree latitude ≈ 111km."""
        dist = GeoCalculator.haversine_distance_m(lat1=46.0, lon1=10.0, lat2=47.0, lon2=10.0)
        assert 110_000 < dist < 112_000

    def test_haversine_distance_one_degree_longitude(self) -> None:
        """1 degree longitude at 46°N ≈ 77km."""
        dist = GeoCalculator.haversine_distance_m(lat1=46.0, lon1=10.0, lat2=46.0, lon2=11.0)
        # At 46°N, longitude degree is ~77km (111km * cos(46°))
        expected = 111_000 * cos(radians(46))
        assert abs(dist - expected) < 2000  # Within 2km

    def test_bearing_north(self) -> None:
        """Bearing from south to north is ~0°."""
        bearing = GeoCalculator.initial_bearing_deg(lon1=10.0, lat1=46.0, lon2=10.0, lat2=47.0)
        assert bearing < 1 or bearing > 359  # Near 0°

    def test_bearing_south(self) -> None:
        """Bearing from north to south is ~180°."""
        bearing = GeoCalculator.initial_bearing_deg(lon1=10.0, lat1=47.0, lon2=10.0, lat2=46.0)
        assert 179 < bearing < 181

    def test_bearing_east(self) -> None:
        """Bearing to east is ~90°."""
        bearing = GeoCalculator.initial_bearing_deg(lon1=10.0, lat1=46.0, lon2=11.0, lat2=46.0)
        assert 89 < bearing < 91

    def test_destination_roundtrip(self) -> None:
        """destination() and haversine_distance_m() are consistent."""
        lon_end, lat_end = GeoCalculator.destination(lon=10.0, lat=46.0, bearing_deg=45.0, distance_m=1000.0)
        actual_dist = GeoCalculator.haversine_distance_m(lat1=46.0, lon1=10.0, lat2=lat_end, lon2=lon_end)
        assert abs(actual_dist - 1000) < 10  # Within 10m error


class TestTerrainAnalyzer:
    """TerrainAnalyzer - terrain gradient and orientation."""

    def test_classify_difficulty_green(self) -> None:
        """Slopes <15% classified as green."""
        assert TerrainAnalyzer.classify_difficulty(slope_pct=5.0) == "green"
        assert TerrainAnalyzer.classify_difficulty(slope_pct=14.9) == "green"

    def test_classify_difficulty_blue(self) -> None:
        """Slopes 15-25% classified as blue."""
        assert TerrainAnalyzer.classify_difficulty(slope_pct=15.0) == "blue"
        assert TerrainAnalyzer.classify_difficulty(slope_pct=24.9) == "blue"

    def test_classify_difficulty_red(self) -> None:
        """Slopes 25-40% classified as red."""
        assert TerrainAnalyzer.classify_difficulty(slope_pct=25.0) == "red"
        assert TerrainAnalyzer.classify_difficulty(slope_pct=39.9) == "red"

    def test_classify_difficulty_black(self) -> None:
        """Slopes >=40% classified as black."""
        assert TerrainAnalyzer.classify_difficulty(slope_pct=40.0) == "black"
        assert TerrainAnalyzer.classify_difficulty(slope_pct=60.0) == "black"

    def test_compute_gradient_on_mock_terrain(self, mock_dem_blue_slope_south: "MockDEMService") -> None:
        """Gradient computation returns reasonable slope and correct direction."""
        analyzer = TerrainAnalyzer(dem=mock_dem_blue_slope_south)
        gradient = analyzer.compute_gradient(lon=0.0, lat=0.0)

        assert gradient is not None
        # Mock DEM is 20% slope south. Due to multi-point weighted averaging,
        # measured gradient is lower than raw slope (roughly half).
        assert 5 < gradient.slope_pct < 25, f"Expected reasonable slope, got {gradient.slope_pct}%"
        # Mock DEM slopes south, so fall line should be ~180°
        assert 170 < gradient.bearing_deg < 190, f"Expected ~180°, got {gradient.bearing_deg}°"

    def test_get_orientation_on_diagonal_slope(self, mock_dem_red_slope_southeast: "MockDEMService") -> None:
        """Orientation returns correct fall line on diagonal terrain."""
        analyzer = TerrainAnalyzer(dem=mock_dem_red_slope_southeast)
        orientation = analyzer.get_orientation(lon=0.0, lat=0.0)

        # Mock DEM: 25% south + 5% east = fall line ~southeast (~169°)
        assert orientation is not None
        # Fall line should be between south (180°) and east (90°)
        assert 140 < orientation.fall_line < 200, f"Expected SE, got {orientation.fall_line}°"

    def test_terrain_on_real_dem(self, real_dem_eurodem: DEMService) -> None:
        """Calculate terrain gradient and orientation using real DEM."""
        analyzer = TerrainAnalyzer(dem=real_dem_eurodem)
        gradient = analyzer.compute_gradient(lon=10.27, lat=46.97)
        orientation = analyzer.get_orientation(lon=10.27, lat=46.97)

        # Gradient should have reasonable slope
        assert gradient is not None
        assert 0 <= gradient.slope_pct <= 100

        # Orientation should have valid fall line bearing
        assert orientation is not None
        assert 0 <= orientation.fall_line < 360


class TestPathTracerWithMockDEM:
    """PathTracer trace_downhill with deterministic mock terrain."""

    def test_trace_downhill_basic(self, mock_dem_blue_slope_south: "MockDEMService") -> None:
        """trace_downhill returns valid downhill path on suitable terrain."""
        tracer = PathTracer(dem=mock_dem_blue_slope_south)
        result = tracer.trace_downhill(
            start_lon=0.0,
            start_lat=0.0,
            target_slope_pct=15.0,
            side="left",
            target_length_m=300,
        )

        assert result is not None, "Should trace path on 20% slope terrain"
        assert len(result.points) >= 3, "Path should have multiple points"

        # Path must go downhill
        start_elev = result.points[0].elevation
        end_elev = result.points[-1].elevation
        assert end_elev < start_elev, f"Path should go downhill: {start_elev} → {end_elev}"

    def test_trace_downhill_respects_target_length(self, mock_dem_blue_slope_south: "MockDEMService") -> None:
        """Traced path length is close to target length."""
        tracer = PathTracer(dem=mock_dem_blue_slope_south)
        target_length = 400

        result = tracer.trace_downhill(
            start_lon=0.0,
            start_lat=0.0,
            target_slope_pct=12.0,
            side="left",
            target_length_m=target_length,
        )

        assert result is not None
        # Path length should be within 20% of target
        assert 0.8 * target_length < result.length_m < 1.2 * target_length, (
            f"Length {result.length_m}m not close to target {target_length}m"
        )

    def test_trace_downhill_side_affects_direction(self, mock_dem_blue_slope_south: "MockDEMService") -> None:
        """Left and right paths diverge from each other."""
        tracer = PathTracer(dem=mock_dem_blue_slope_south)

        left_path = tracer.trace_downhill(
            start_lon=0.0,
            start_lat=0.0,
            target_slope_pct=12.0,
            side="left",
            target_length_m=300,
        )
        right_path = tracer.trace_downhill(
            start_lon=0.0,
            start_lat=0.0,
            target_slope_pct=12.0,
            side="right",
            target_length_m=300,
        )

        assert left_path is not None and right_path is not None
        # Endpoints should be different (different traverse directions)
        left_end = left_path.points[-1]
        right_end = right_path.points[-1]
        end_dist = GeoCalculator.haversine_distance_m(
            lat1=left_end.lat,
            lon1=left_end.lon,
            lat2=right_end.lat,
            lon2=right_end.lon,
        )
        assert end_dist > 50, f"Left/right paths should diverge, but only {end_dist}m apart"


class TestDEMServiceReal:
    """Tests using real DEM data (skipped if file unavailable)."""

    def test_dem_file_exists(self) -> None:
        """Check if EuroDEM file exists."""
        assert DEMConfig.EURODEM_PATH.exists()

    def test_load_real_dem_and_sample(self, real_dem_eurodem: DEMService) -> None:
        """Load real DEM and sample elevation at known point."""
        # Sample near Ischgl, Austria (default map center)
        elev = real_dem_eurodem.get_elevation(lon=10.27, lat=46.97)
        assert elev is not None
        # Alps elevation typically
        assert 1000 < elev < 4000


class TestPathTracerWithRealDEM:
    """Test path tracing with real DEM data."""

    def test_tracer_terrain_analysis(self, real_dem_eurodem: DEMService) -> None:
        """PathTracer analyzer can compute gradient and orientation."""
        tracer = PathTracer(dem=real_dem_eurodem)
        gradient = tracer.analyzer.compute_gradient(lon=10.27, lat=46.97)
        orientation = tracer.analyzer.get_orientation(lon=10.27, lat=46.97)

        assert gradient is not None
        assert 0 <= gradient.slope_pct <= 100

        assert orientation is not None
        assert 0 <= orientation.fall_line < 360

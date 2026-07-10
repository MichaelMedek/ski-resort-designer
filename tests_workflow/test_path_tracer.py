"""Unit tests for the PathTracer downhill tracing algorithm (core/path_tracer.py)."""

from skiresort_planner.core.geo_calculator import GeoCalculator
from skiresort_planner.core.path_tracer import PathTracer


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

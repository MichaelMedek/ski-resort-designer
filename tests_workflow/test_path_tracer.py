"""Unit tests for the PathTracer downhill tracing algorithm (core/path_tracer.py)."""

import pytest

from skiresort_planner.core.geo_calculator import GeoCalculator
from skiresort_planner.core.path_tracer import PathTracer


class TestPathTracerOnMockTerrain:
    """Tests for path tracing algorithm on the deterministic linear DEM."""

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

    @pytest.mark.parametrize("target_pct", [10.0, 20.0, 30.0])
    def test_average_slope_converges_toward_target(self, mock_dem_red_slope_diagonal, target_pct: float) -> None:
        """The cumulative-drop algorithm should converge the path's average slope
        toward the requested target (its whole purpose), not just 'go downhill'.
        On the 30%-south/10%-east DEM there is enough grade to hit each target.
        """
        tracer = PathTracer(dem=mock_dem_red_slope_diagonal)
        path = tracer.trace_downhill(
            start_lon=0.0, start_lat=0.0, target_slope_pct=target_pct, side="left", target_length_m=400
        )
        assert path is not None
        # Converges within a real tolerance (±8pp) — tight enough to catch a
        # wrong-target regression, loose enough for terrain-adaptive tracing.
        assert path.avg_slope_pct == pytest.approx(target_pct, abs=8.0)

    def test_steeper_target_drops_more(self, mock_dem_black_slope) -> None:
        """A steeper target over the same length must lose more elevation."""
        tracer = PathTracer(dem=mock_dem_black_slope)
        gentle = tracer.trace_downhill(
            start_lon=0.0, start_lat=0.0, target_slope_pct=15.0, side="left", target_length_m=400
        )
        steep = tracer.trace_downhill(
            start_lon=0.0, start_lat=0.0, target_slope_pct=40.0, side="left", target_length_m=400
        )
        assert gentle is not None and steep is not None
        assert steep.total_drop_m > gentle.total_drop_m

    def test_start_outside_dem_returns_none(self) -> None:
        """No elevation at the start point → trace returns None (early guard)."""
        from tests_workflow.conftest import MockDEMService

        dem = MockDEMService(base_elevation=2500.0, slope_ns_pct=20.0, slope_ew_pct=0.0)
        tracer = PathTracer(dem=dem)
        # Far outside the mock's (-1,-1,1,1) bounds → start elevation is None.
        result = tracer.trace_downhill(
            start_lon=99.0, start_lat=99.0, target_slope_pct=15.0, side="left", target_length_m=300
        )
        assert result is None

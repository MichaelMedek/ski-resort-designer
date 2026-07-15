"""Unit tests for the PathTracer signed-grade tracing algorithm (core/path_tracer.py).

The tracer holds a SIGNED target grade: positive descends the fall line, negative
climbs against it, zero contours across it. The mock DEMs are planar with exact
slopes, so expected geometry is analytically predictable — assertions check physics
(direction climbed, grade held, contour stays level), not the code's own outputs.
"""

import pytest

from skiresort_planner.constants import SlopeConfig
from skiresort_planner.core.geo_calculator import GeoCalculator
from skiresort_planner.core.path_tracer import PathTracer


class TestPathTracerDownhill:
    """Descent behavior (positive targets) — the slope fan's use of the tracer."""

    def test_trace_hill_produces_valid_diverging_paths(self, mock_dem_blue_slope) -> None:
        """PathTracer generates valid downhill paths with left/right divergence.

        Tests:
        - trace_hill returns non-None on valid terrain
        - Path goes downhill (end elevation < start)
        - Left/right paths diverge significantly
        - Path length approximates target
        """
        tracer = PathTracer(dem=mock_dem_blue_slope)

        left = tracer.trace_hill(
            start_lon=0.0,
            start_lat=0.0,
            target_grade_pct=15.0,
            side="left",
            target_length_m=300,
        )
        right = tracer.trace_hill(
            start_lon=0.0,
            start_lat=0.0,
            target_grade_pct=15.0,
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
        path = tracer.trace_hill(
            start_lon=0.0, start_lat=0.0, target_grade_pct=target_pct, side="left", target_length_m=400
        )
        assert path is not None
        # Converges within a real tolerance (±8pp) — tight enough to catch a
        # wrong-target regression, loose enough for terrain-adaptive tracing.
        assert path.avg_slope_pct == pytest.approx(target_pct, abs=8.0)

    @pytest.mark.parametrize("target_pct", [7.0, 12.0, -7.0])
    def test_signed_target_not_biased_steeper(self, mock_dem_blue_slope, target_pct: float) -> None:
        """The traced average must not be systematically STEEPER than the target.

        Regression for the step-target floor bug: on 20% terrain a 7% descent used to
        drift to ~11% (and a −7% climb to ~−11%) because the clamp floored each step at
        MIN_SKIABLE_PCT, so an overshoot could never self-correct back toward the target.
        ±3pp is tight enough to catch that bias (the ±8pp convergence test cannot). Noise
        off for a clean path.
        """
        import random

        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(random, "gauss", lambda mu, sigma: 0.0)
        try:
            tracer = PathTracer(dem=mock_dem_blue_slope)
            path = tracer.trace_hill(
                start_lon=0.0, start_lat=0.0, target_grade_pct=target_pct, side="center", target_length_m=500
            )
        finally:
            monkeypatch.undo()
        assert path is not None
        assert path.avg_slope_pct == pytest.approx(target_pct, abs=3.0), (
            "traced average must track the target, not ratchet steeper"
        )

    def test_steeper_target_drops_more(self, mock_dem_black_slope) -> None:
        """A steeper target over the same length must lose more elevation."""
        tracer = PathTracer(dem=mock_dem_black_slope)
        gentle = tracer.trace_hill(
            start_lon=0.0, start_lat=0.0, target_grade_pct=15.0, side="left", target_length_m=400
        )
        steep = tracer.trace_hill(start_lon=0.0, start_lat=0.0, target_grade_pct=40.0, side="left", target_length_m=400)
        assert gentle is not None and steep is not None
        assert steep.total_drop_m > gentle.total_drop_m

    @pytest.mark.parametrize("target_pct", [7.0, 12.0, 20.0, -7.0, 0.0])
    def test_drop_equals_avg_slope_times_length_identity(self, mock_dem_blue_slope, target_pct: float) -> None:
        """The reported drop, average slope and length must be mutually CONSISTENT.

        Independent of how well the tracer hits its target, the three summary numbers it
        reports describe one path, so total_drop_m must equal avg_slope_pct/100 × length_m
        exactly. This algebraic identity catches a bookkeeping bug (drop summed over a
        different span than length, a sign error, a units slip) that the ±8pp
        target-convergence checks would sail straight past. Noise off for a clean path.
        """
        import random

        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(random, "gauss", lambda mu, sigma: 0.0)
        try:
            tracer = PathTracer(dem=mock_dem_blue_slope)
            path = tracer.trace_hill(
                start_lon=0.0, start_lat=0.0, target_grade_pct=target_pct, side="center", target_length_m=500
            )
        finally:
            monkeypatch.undo()
        assert path is not None
        implied_drop = path.avg_slope_pct / 100.0 * path.length_m
        assert path.total_drop_m == pytest.approx(implied_drop, rel=1e-6), (
            "drop, average slope and length must describe the same path"
        )

    def test_target_at_or_above_terrain_holds_grade_tightly(self, mock_dem_blue_slope) -> None:
        """When the target grade ≥ terrain steepness, the center path holds it almost exactly.

        The blue DEM reads ~10% steepness (weighted), so a 20% target needs no traverse — the
        tracer descends the fall line and the achieved grade should land within ±0.5pp of 20%,
        far tighter than the ±8pp terrain-adaptive band. Pins the fall-line/no-traverse branch.
        """
        import random

        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(random, "gauss", lambda mu, sigma: 0.0)
        try:
            tracer = PathTracer(dem=mock_dem_blue_slope)
            path = tracer.trace_hill(
                start_lon=0.0, start_lat=0.0, target_grade_pct=20.0, side="center", target_length_m=500
            )
        finally:
            monkeypatch.undo()
        assert path is not None
        assert path.avg_slope_pct == pytest.approx(20.0, abs=0.5), "target ≥ terrain → grade held tightly"

    def test_descent_follows_the_fall_line(self, monkeypatch, mock_dem_red_slope_diagonal) -> None:
        """With noise removed and a steep target, a descent trends DOWN the fall line.

        On the 30%-south/10%-east DEM the fall line points south-and-slightly-east.
        A target at/above the terrain steepness needs no traverse, so the path runs
        straight down the fall line and must end both south of AND east of the start.
        This pins the positive branch to the physical fall-line direction the DEM
        dictates (not a byte snapshot) — a reference-bearing regression would fail here.
        """
        import random

        monkeypatch.setattr(random, "gauss", lambda mu, sigma: 0.0)
        tracer = PathTracer(dem=mock_dem_red_slope_diagonal)
        # 40% target ≥ ~31.6% terrain → traverse angle collapses to the minimum,
        # so the path descends essentially straight down the fall line.
        path = tracer.trace_hill(
            start_lon=0.0, start_lat=0.0, target_grade_pct=40.0, side="center", target_length_m=300
        )
        assert path is not None
        assert path.points[-1].elevation < path.points[0].elevation, "Descent must lose elevation"
        assert path.points[-1].lat < path.points[0].lat, "Fall line points south → end south of start"
        assert path.points[-1].lon > path.points[0].lon, "Fall line has an east component → end east of start"


class TestPathTracerUphillAndContour:
    """Climb and contour behavior (negative and zero targets) — new capabilities."""

    def test_uphill_target_actually_climbs(self, mock_dem_blue_slope) -> None:
        """A negative target climbs against the fall line and holds the climb grade.

        mock_dem_blue_slope drops 20% going south, so NORTH is uphill. A −7% target
        must gain elevation, progress north (against the south fall line), and report
        a signed average grade near −7%. Fails if the reference-bearing 180° flip or
        signed drop tracking is wrong.
        """
        tracer = PathTracer(dem=mock_dem_blue_slope)
        path = tracer.trace_hill(start_lon=0.0, start_lat=0.0, target_grade_pct=-7.0, side="left", target_length_m=300)
        assert path is not None
        assert path.points[-1].elevation > path.points[0].elevation, "Climb must gain elevation"
        assert path.points[-1].lat > path.points[0].lat, "Climb progresses north, against the south fall line"
        assert path.avg_slope_pct == pytest.approx(-7.0, abs=8.0), "Signed average grade near the −7% target"
        assert path.total_drop_m < 0, "A climb has negative (signed) drop"

    def test_contour_target_stays_near_level(self, mock_dem_black_slope) -> None:
        """A zero target contours across the slope, staying near level.

        On the 45%-south DEM the fall line is south, so a contour runs roughly
        east/west and holds elevation. Exercises the acos(0)→~89° traverse and the
        contour clamp branch; fails if a contour drifts downhill.
        """
        tracer = PathTracer(dem=mock_dem_black_slope)
        path = tracer.trace_hill(start_lon=0.0, start_lat=0.0, target_grade_pct=0.0, side="right", target_length_m=300)
        assert path is not None
        assert abs(path.avg_slope_pct) < SlopeConfig.MIN_SKIABLE_PCT, "Contour holds a near-level grade"
        # Moves mostly east/west (across the south fall line), not north/south.
        east_west_m = abs(path.points[-1].lon - path.points[0].lon)
        north_south_m = abs(path.points[-1].lat - path.points[0].lat)
        assert east_west_m > north_south_m, "Contour runs across the fall line, not along it"

    @pytest.mark.parametrize("side", ["left", "right"])
    def test_contour_stays_level_on_curved_terrain(self, cone_dem_steep, side: str) -> None:
        """A 0% contour must stay near level even where the fall line ROTATES under it.

        Regression for the reference-bearing bug: on the radial cone (fall line rotates
        along every contour) a contour used to drift to ~6.5% because the reference stayed
        on the downhill fall line and no step could climb the accumulated drop back. This
        is the "a flat trace perpendicular to the ridge must ALWAYS exist" guarantee, on
        the curved terrain planar mocks cannot express. Noise off for a clean path.
        """
        import random

        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(random, "gauss", lambda mu, sigma: 0.0)
        try:
            tracer = PathTracer(dem=cone_dem_steep)
            path = tracer.trace_hill(
                start_lon=0.005, start_lat=0.0, target_grade_pct=0.0, side=side, target_length_m=400
            )
        finally:
            monkeypatch.undo()
        assert path is not None
        assert abs(path.avg_slope_pct) < SlopeConfig.MIN_SKIABLE_PCT, "Contour must hold near-level on curved terrain"

    def test_climb_diverges_left_right(self, mock_dem_red_slope_diagonal) -> None:
        """Left/right still diverge when climbing (side_sign applies to the flipped bearing)."""
        tracer = PathTracer(dem=mock_dem_red_slope_diagonal)
        left = tracer.trace_hill(start_lon=0.0, start_lat=0.0, target_grade_pct=-12.0, side="left", target_length_m=300)
        right = tracer.trace_hill(
            start_lon=0.0, start_lat=0.0, target_grade_pct=-12.0, side="right", target_length_m=300
        )
        assert left is not None and right is not None
        assert left.points[-1].elevation > left.points[0].elevation, "Left path climbs"
        assert right.points[-1].elevation > right.points[0].elevation, "Right path climbs"
        end_dist = GeoCalculator.haversine_distance_m(
            lat1=left.points[-1].lat,
            lon1=left.points[-1].lon,
            lat2=right.points[-1].lat,
            lon2=right.points[-1].lon,
        )
        assert end_dist > 30, "Left/right climbs should diverge at endpoints"


class TestPathTracerGuards:
    """Edge guards."""

    def test_start_outside_dem_returns_none(self) -> None:
        """No elevation at the start point → trace returns None (early guard)."""
        from tests_workflow.conftest import MockDEMService

        dem = MockDEMService(base_elevation=2500.0, slope_ns_pct=20.0, slope_ew_pct=0.0)
        tracer = PathTracer(dem=dem)
        # Far outside the mock's (-1,-1,1,1) bounds → start elevation is None.
        result = tracer.trace_hill(
            start_lon=99.0, start_lat=99.0, target_grade_pct=15.0, side="left", target_length_m=300
        )
        assert result is None

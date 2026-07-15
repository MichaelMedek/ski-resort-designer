"""Unit tests for TerrainAnalyzer (core/terrain_analyzer.py).

Covers the real math — difficulty classification edges, multi-point weighted
gradient (slope magnitude + fall-line bearing), side-slope decomposition, and
the orientation MIN_SKIABLE_PCT branch — using the deterministic MockDEMService
(elevation is a known linear function of lat/lon, so gradients are predictable).
"""

import pytest

from skiresort_planner.core.terrain_analyzer import SideDirection, TerrainAnalyzer


class TestDifficultyClassification:
    """Tests for slope difficulty classification."""

    def test_classify_difficulty_all_thresholds(self) -> None:
        """Difficulty classification at all threshold boundaries."""
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

        # Black: 40%+ (including extreme, beyond MAX_SKIABLE)
        assert TerrainAnalyzer.classify_difficulty(slope_pct=40.0) == "black"
        assert TerrainAnalyzer.classify_difficulty(slope_pct=60.0) == "black"
        assert TerrainAnalyzer.classify_difficulty(slope_pct=100.0) == "black"

    def test_negative_slope_is_green(self) -> None:
        """Uphill (negative) slope classifies as green, not an error."""
        assert TerrainAnalyzer.classify_difficulty(slope_pct=-10.0) == "green"

    def test_difficulty_color_matches_classification(self) -> None:
        """get_difficulty_color returns the StyleConfig color for the class."""
        from skiresort_planner.constants import StyleConfig

        assert TerrainAnalyzer.get_difficulty_color(slope_pct=5.0) == StyleConfig.SLOPE_COLORS["green"]
        assert TerrainAnalyzer.get_difficulty_color(slope_pct=45.0) == StyleConfig.SLOPE_COLORS["black"]


class TestComputeGradient:
    """Weighted multi-point gradient on a known linear DEM.

    The 'Magic 8' two-ring weighting reduces raw slope magnitude, so we pin the
    fall-line BEARING exactly (the directionally meaningful output) and assert
    magnitude is positive and monotonic in DEM steepness, rather than asserting
    a specific reduced magnitude as if it were the raw grade.
    """

    def test_south_facing_slope_fall_line_points_south(self, mock_dem_blue_slope) -> None:
        """A DEM that drops going south → fall line points due south (180°)."""
        analyzer = TerrainAnalyzer(dem=mock_dem_blue_slope)
        grad = analyzer.compute_gradient(lon=0.0, lat=0.0)

        assert grad.bearing_deg == pytest.approx(180.0, abs=1.0), "fall line points south (downhill)"
        assert grad.slope_pct > 0.0, "a sloped DEM yields a positive gradient magnitude"

    @pytest.mark.parametrize(
        "ns_pct, ew_pct, expected_bearing",
        [
            (20.0, 0.0, 180.0),  # drops south → downhill points south
            (-20.0, 0.0, 0.0),  # rises south → downhill points north
            (0.0, 20.0, 90.0),  # drops east → downhill points east
            (0.0, -20.0, 270.0),  # rises east → downhill points west
        ],
    )
    def test_cardinal_slopes_have_exact_fall_line_bearing(
        self, ns_pct: float, ew_pct: float, expected_bearing: float
    ) -> None:
        """On a pure N/S or E/W plane the fall line is an EXACT cardinal bearing.

        Pins the bearing to 0.01° (not the ±1° = ~1km band used elsewhere): the weighted
        multi-point gradient must reproduce the plane's exact downhill direction, and any
        sign flip in the grad_x/grad_y decomposition would swing this by 90°+.
        """
        from tests_workflow.conftest import MockDEMService

        analyzer = TerrainAnalyzer(dem=MockDEMService(base_elevation=2500.0, slope_ns_pct=ns_pct, slope_ew_pct=ew_pct))
        grad = analyzer.compute_gradient(lon=0.0, lat=0.0)
        assert grad.bearing_deg == pytest.approx(expected_bearing, abs=0.01)
        assert grad.slope_pct > 0.0, "a tilted plane has a positive gradient magnitude"

    def test_flat_terrain_has_exactly_zero_gradient(self) -> None:
        """A perfectly level DEM yields exactly (0% slope, 0° bearing) — no phantom tilt."""
        from tests_workflow.conftest import MockDEMService

        analyzer = TerrainAnalyzer(dem=MockDEMService(base_elevation=2500.0, slope_ns_pct=0.0, slope_ew_pct=0.0))
        grad = analyzer.compute_gradient(lon=0.0, lat=0.0)
        assert grad.slope_pct == 0.0
        assert grad.bearing_deg == 0.0

    def test_gradient_magnitude_is_deterministic_on_a_known_plane(self) -> None:
        """The weighted 'Magic 8' gradient is a FIXED fraction of the plane's raw slope.

        On the mock's linear plane the two-ring weighting is fully deterministic, so the
        reduced magnitude is a known number — pin it. A 20% N-S plane reads 10.011%, and a
        30% plane reads 1.5× that. This guards the magnitude (not just its sign/monotonicity)
        against a re-weighting regression that a '> 0 and monotonic' check would miss.
        """
        from tests_workflow.conftest import MockDEMService

        g20 = TerrainAnalyzer(
            dem=MockDEMService(base_elevation=2500.0, slope_ns_pct=20.0, slope_ew_pct=0.0)
        ).compute_gradient(lon=0.0, lat=0.0)
        g30 = TerrainAnalyzer(
            dem=MockDEMService(base_elevation=2500.0, slope_ns_pct=30.0, slope_ew_pct=0.0)
        ).compute_gradient(lon=0.0, lat=0.0)
        assert g20.slope_pct == pytest.approx(10.011, abs=0.01), "20% plane → known reduced magnitude"
        # The plane is linear, so the reduced magnitude scales exactly with raw steepness.
        assert g30.slope_pct == pytest.approx(g20.slope_pct * 1.5, rel=1e-6)

    def test_diagonal_slope_bearing_in_se_quadrant_biased_south(self, mock_dem_red_slope_diagonal) -> None:
        """DEM dropping 30% south + 10% east → fall line in the SE quadrant, biased south."""
        analyzer = TerrainAnalyzer(dem=mock_dem_red_slope_diagonal)
        grad = analyzer.compute_gradient(lon=0.0, lat=0.0)

        assert 135.0 < grad.bearing_deg < 180.0, "steeper south component pulls the fall line past SE toward south"

    def test_steeper_dem_reports_steeper_gradient(self, mock_dem_blue_slope, mock_dem_black_slope) -> None:
        """Gradient magnitude is monotonic in DEM steepness (20% vs 45% south)."""
        gentle = TerrainAnalyzer(dem=mock_dem_blue_slope).compute_gradient(lon=0.0, lat=0.0)
        steep = TerrainAnalyzer(dem=mock_dem_black_slope).compute_gradient(lon=0.0, lat=0.0)
        assert steep.slope_pct > gentle.slope_pct, "the 45% DEM must read steeper than the 20% DEM"

    def test_out_of_bounds_returns_flat(self) -> None:
        """A center point with no elevation returns a zero gradient, not a crash."""
        from tests_workflow.conftest import MockDEMService

        dem = MockDEMService(base_elevation=2500.0, slope_ns_pct=20.0, slope_ew_pct=0.0)
        analyzer = TerrainAnalyzer(dem=dem)
        # Far outside the mock's (-1,-1,1,1) bounds → get_elevation returns None at center.
        grad = analyzer.compute_gradient(lon=99.0, lat=99.0)
        assert grad.slope_pct == 0.0 and grad.bearing_deg == 0.0


class TestComputeSideSlope:
    """Side slope = gradient component perpendicular to ski direction."""

    def test_skiing_down_fall_line_has_no_side_slope(self, mock_dem_blue_slope) -> None:
        """Skiing due south down a south-facing slope → side slope ≈ 0 (flat)."""
        analyzer = TerrainAnalyzer(dem=mock_dem_blue_slope)
        M = 111320.0
        side = TerrainAnalyzer.compute_side_slope(
            start_lon=0.0, start_lat=0.0, end_lon=0.0, end_lat=-100 / M, analyzer=analyzer
        )
        assert side.direction == SideDirection.FLAT
        assert abs(side.slope_pct) < 2.0

    def test_traversing_across_fall_line_has_strong_side_slope(self, mock_dem_blue_slope) -> None:
        """Skiing due east across a south-falling slope → large side slope to one side."""
        analyzer = TerrainAnalyzer(dem=mock_dem_blue_slope)
        M = 111320.0
        side = TerrainAnalyzer.compute_side_slope(
            start_lon=0.0, start_lat=0.0, end_lon=100 / M, end_lat=0.0, analyzer=analyzer
        )
        assert side.direction == SideDirection.RIGHT, "east across a south fall line leans right when looking downhill"
        assert side.slope_pct > 10.0, "crossing the fall line exposes most of the gradient sideways (positive = right)"

    def test_traversing_west_across_fall_line_leans_left(self, mock_dem_blue_slope) -> None:
        """Skiing due west across a south-falling slope → terrain leans LEFT with negative side slope.

        Mirror of the east case: the opposite traverse direction flips the side to LEFT and the
        signed side-slope percentage negative — guarding the sign/direction symmetry.
        """
        analyzer = TerrainAnalyzer(dem=mock_dem_blue_slope)
        M = 111320.0
        side = TerrainAnalyzer.compute_side_slope(
            start_lon=0.0, start_lat=0.0, end_lon=-100 / M, end_lat=0.0, analyzer=analyzer
        )
        assert side.direction == SideDirection.LEFT, "west across a south fall line leans left when looking downhill"
        assert side.slope_pct < -10.0, "the opposite traverse flips the side slope negative (left)"

    def test_excavator_warning_message_renders_plain_direction_word(self) -> None:
        """ExcavatorWarning renders the SideDirection as its plain value (e.g. 'left'), not 'SideDirection.LEFT'."""
        from skiresort_planner.model.warning import ExcavatorWarning

        warning = ExcavatorWarning(side_slope_pct=40.0, belt_width_m=20.0, side_slope_dir=SideDirection.LEFT)
        assert "terrain leans left" in warning.message
        assert "SideDirection" not in warning.message


class TestGetOrientation:
    """Orientation derives contour/half-slope bearings, or None when too flat."""

    def test_orientation_on_skiable_slope(self, mock_dem_blue_slope) -> None:
        analyzer = TerrainAnalyzer(dem=mock_dem_blue_slope)
        orient = analyzer.get_orientation(lon=0.0, lat=0.0)

        assert orient is not None
        assert orient.fall_line == pytest.approx(180.0, abs=2.0)
        # Contours are perpendicular to the fall line; half-slopes are ±45°.
        assert orient.contour_left == pytest.approx((orient.fall_line - 90) % 360, abs=0.01)
        assert orient.contour_right == pytest.approx((orient.fall_line + 90) % 360, abs=0.01)

    def test_flat_terrain_returns_none(self) -> None:
        """Below MIN_SKIABLE_PCT the terrain is unskiable → get_orientation returns None."""
        from tests_workflow.conftest import MockDEMService

        flat = MockDEMService(base_elevation=2000.0, slope_ns_pct=0.0, slope_ew_pct=0.0)
        analyzer = TerrainAnalyzer(dem=flat)
        assert analyzer.get_orientation(lon=0.0, lat=0.0) is None

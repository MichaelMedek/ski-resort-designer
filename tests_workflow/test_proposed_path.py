"""Unit tests for ProposedPathSegment computed metrics (model/proposed_path.py)."""

from skiresort_planner.constants import SlopeConfig
from skiresort_planner.model.path_point import PathPoint
from skiresort_planner.model.path_segment import SegmentKind
from skiresort_planner.model.proposed_path import ProposedPathSegment


class TestProposedSegmentComputedProperties:
    """Tests for ProposedPathSegment computed metrics."""

    def test_computed_metrics_from_path_points(self, path_points_blue) -> None:
        """ProposedPathSegment computes drop, length, slope, difficulty."""
        segment = ProposedPathSegment(
            points=path_points_blue,
            target_slope_pct=20.0,
            target_difficulty="blue",
            sector_name="Test",
        )

        assert segment.total_drop_m > 0
        assert 750 < segment.length_m < 850
        assert 15 < segment.avg_slope_pct < 25
        assert segment.difficulty == "blue"

        # ProposedPathSegment-specific fields (what this subclass adds over Path).
        assert segment.target_slope_pct == 20.0
        assert segment.target_difficulty == "blue"
        assert segment.sector_name == "Test"

        # Defaults left untouched by construction.
        assert segment.is_connector is False
        assert segment.target_node_id == ""
        assert segment.start_node_id == ""
        assert segment.kind == SegmentKind.SLOPE


class TestMaxSlopeRollingWindow:
    """Tests for max_slope_pct rolling window algorithm."""

    def test_detects_steep_section_in_variable_terrain(self) -> None:
        """max_slope_pct rolling window detects steep section within gradual terrain."""
        window_m = SlopeConfig.ROLLING_WINDOW_M
        step_m = 100
        steps_per_section = max(3, (window_m // step_m) + 1)

        base_lon = 10.27
        lat_per_step = 0.0009

        sections = [
            (steps_per_section, 10.0),
            (steps_per_section, 45.0),
            (steps_per_section, 10.0),
        ]

        points = []
        lat = 46.97
        elev = 2500.0
        points.append(PathPoint(lon=base_lon, lat=lat, elevation=elev))

        for num_steps, drop in sections:
            for _ in range(num_steps):
                lat -= lat_per_step
                elev -= drop
                points.append(PathPoint(lon=base_lon, lat=lat, elevation=elev))

        seg = ProposedPathSegment(points=points)

        total_steps = steps_per_section * 3
        expected_length = total_steps * step_m

        assert seg.length_m > window_m
        assert expected_length * 0.9 < seg.length_m < expected_length * 1.1

        avg = seg.avg_slope_pct
        assert 15 < avg < 30

        assert seg.max_slope_pct > 40
        assert seg.max_slope_pct < 50

    def test_ascending_path_keeps_max_slope_positive_magnitude(self) -> None:
        """For a climbing path, avg_slope_pct is negative but max_slope_pct is a positive magnitude."""
        # ~222m south, climbing 100m (end higher than start) -> shorter than ROLLING_WINDOW_M,
        # so max_slope_pct returns the abs(avg_slope_pct) seed.
        points = [
            PathPoint(lon=0.0, lat=0.0, elevation=2000.0),
            PathPoint(lon=0.0, lat=-0.002, elevation=2100.0),
        ]
        seg = ProposedPathSegment(points=points)

        assert seg.total_drop_m < 0  # climbs: end elevation exceeds start
        assert seg.avg_slope_pct < 0  # signed average is negative for a climb
        assert seg.max_slope_pct > 0  # magnitude seed survives the abs()
        assert seg.max_slope_pct == abs(seg.avg_slope_pct)

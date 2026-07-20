"""Unit tests for the Slope model (model/slope.py)."""

from skiresort_planner.constants import MapConfig
from skiresort_planner.model.path_point import PathPoint
from skiresort_planner.model.path_segment import PathSegment
from skiresort_planner.model.proposed_path import ProposedPathSegment


def _segment(id: str, length_m: float, grade_pct: float) -> PathSegment:
    """A 2-point south-running segment of the given length and (descending) grade."""
    dlat = length_m / MapConfig.METERS_PER_DEGREE_EQUATOR
    return PathSegment(
        id=id,
        start_node_id=f"{id}a",
        end_node_id=f"{id}b",
        points=[
            PathPoint(lon=0.0, lat=0.0, elevation=3000.0),
            PathPoint(lon=0.0, lat=-dlat, elevation=3000.0 - length_m * grade_pct / 100.0),
        ],
    )


class TestSlopeNaming:
    """Deterministic naming threshold branches (drop bands from NameConfig)."""

    def test_summit_name_for_large_drop(self) -> None:
        from skiresort_planner.model.slope import Slope

        # Drop above SUMMIT_RISE_M (500m) → "Summit" in the name.
        name = Slope.generate_name(
            difficulty="black", slope_id="SL1", start_elevation=3000.0, end_elevation=2400.0, avg_bearing=0.0
        )
        assert "Summit" in name

    def test_big_name_for_medium_drop(self) -> None:
        from skiresort_planner.model.slope import Slope

        # Drop between BIG_DROP_M (300m) and SUMMIT_RISE_M (500m) → "Big".
        name = Slope.generate_name(
            difficulty="red", slope_id="SL2", start_elevation=3000.0, end_elevation=2650.0, avg_bearing=90.0
        )
        assert "Big" in name

    def test_no_size_descriptor_for_small_drop(self) -> None:
        from skiresort_planner.model.slope import Slope

        # Drop below BIG_DROP_M (300m) → neither Summit nor Big.
        name = Slope.generate_name(
            difficulty="blue", slope_id="SL3", start_elevation=2200.0, end_elevation=2100.0, avg_bearing=180.0
        )
        assert "Summit" not in name and "Big" not in name

    def test_name_embeds_slope_number(self) -> None:
        from skiresort_planner.model.slope import Slope

        name = Slope.generate_name(
            difficulty="green", slope_id="SL7", start_elevation=2100.0, end_elevation=2050.0, avg_bearing=0.0
        )
        assert name.startswith("7 (")


class TestSlopeNumberFromId:
    def test_number_from_id_extracts_numeric_part(self) -> None:
        from skiresort_planner.model.slope import Slope

        assert Slope.number_from_id("SL1") == 1
        assert Slope.number_from_id("SL123") == 123


class TestSlopeGetDifficulty:
    """get_difficulty derives the rating from the steepest segment among the slope's segments."""

    def test_difficulty_matches_classifier_on_steepest_segment(self, empty_graph, path_points_blue) -> None:
        from skiresort_planner.core.terrain_analyzer import TerrainAnalyzer

        empty_graph.commit_paths(paths=[ProposedPathSegment(points=path_points_blue, target_difficulty="blue")])
        slope = empty_graph.finish_slope(segment_ids=list(empty_graph.segments.keys()))

        expected = TerrainAnalyzer.classify_difficulty(slope_pct=slope.get_max_gradient(segments=empty_graph.segments))
        assert slope.get_difficulty(segments=empty_graph.segments) == expected
        assert slope.get_difficulty(segments=empty_graph.segments) in {"green", "blue", "red", "black"}

    def test_short_but_steep_segment_drives_rating(self) -> None:
        """A steep pitch shorter than ROLLING_WINDOW_M (but ≥ SEGMENT_LENGTH_MIN_M) must count.

        Regression: slope 167 had two ~260m black pitches (52%/56%) yet read blue, because the
        steepest-section filter dropped every segment under 300m. Segments down to the builder's
        minimum length are real and rate the slope.
        """
        from skiresort_planner.model.segment_path import steepest_section_pct

        # A 260m 55% black wall plus longer blue segments — the wall must win.
        segs = [
            _segment("S1", length_m=260.0, grade_pct=55.0),  # black, < 300m
            _segment("S2", length_m=345.0, grade_pct=17.0),  # blue, > 300m
            _segment("S3", length_m=259.0, grade_pct=24.0),  # blue, < 300m
        ]
        assert steepest_section_pct(segments=segs) >= 50.0, "the short 55% wall must set the steepest section"

    def test_sub_minimum_sliver_ignored_when_a_real_segment_exists(self) -> None:
        """A segment below SEGMENT_LENGTH_MIN_M is a sliver — a longer real segment takes precedence."""
        from skiresort_planner.model.segment_path import steepest_section_pct

        segs = [
            _segment("S1", length_m=50.0, grade_pct=60.0),  # 50m sliver, below the 100m floor
            _segment("S2", length_m=345.0, grade_pct=17.0),  # real blue segment
        ]
        assert steepest_section_pct(segments=segs) < 25.0, "sub-minimum sliver excluded; blue segment rules"

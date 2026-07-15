"""Unit tests for the Slope model (model/slope.py)."""

from skiresort_planner.model.proposed_path import ProposedPathSegment


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

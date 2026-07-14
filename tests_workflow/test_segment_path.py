"""Unit tests for the SegmentPath base + subclasses (model/segment_path.py).

SegmentPath is the shared chain-of-segments base for Slope and Road. This
module covers the hierarchy, the shared geometry methods (exercised through a
committed Road), and the cross-model `number_from_id` ID parsing.
"""

import pytest

from skiresort_planner.enum_utils import enum_eq
from skiresort_planner.model.lift import Lift
from skiresort_planner.model.path_point import PathPoint
from skiresort_planner.model.path_segment import PathSegment, SegmentKind
from skiresort_planner.model.proposed_path import ProposedPathSegment
from skiresort_planner.model.resort_graph import ResortGraph
from skiresort_planner.model.road import Road
from skiresort_planner.model.segment_path import SegmentPath
from skiresort_planner.model.slope import Slope

M = 111320.0


def _commit_road(graph: ResortGraph, path_points: list[PathPoint]) -> Road:
    """Commit a path as a road (record_undo=False + finish_road), return the Road."""
    graph.commit_paths(
        paths=[ProposedPathSegment(points=path_points, is_connector=True, kind=SegmentKind.ROAD)], record_undo=False
    )
    road = graph.finish_road(segment_ids=[list(graph.segments.keys())[-1]])
    assert road is not None
    return road


class TestSegmentPathHierarchy:
    def test_slope_and_road_are_segment_paths(self) -> None:
        assert issubclass(Slope, SegmentPath)
        assert issubclass(Road, SegmentPath)

    def test_number_from_id_uses_subclass_prefix(self) -> None:
        assert Slope.number_from_id("SL7") == 7
        assert Road.number_from_id("R3") == 3


class TestSegmentPathBaseMethods:
    """Shared SegmentPath geometry, exercised through a committed Road."""

    def test_number_property_derives_from_id(self, empty_graph, path_points_blue) -> None:
        road = _commit_road(empty_graph, path_points_blue)
        assert road.number == Road.number_from_id(road.id)

    def test_total_length_and_drop_match_segment(self, empty_graph, path_points_blue) -> None:
        road = _commit_road(empty_graph, path_points_blue)
        seg = empty_graph.segments[road.segment_ids[0]]
        assert road.get_total_length(segments=empty_graph.segments) == pytest.approx(seg.length_m)
        assert road.get_total_drop(segments=empty_graph.segments) == pytest.approx(seg.total_drop_m)

    def test_get_all_points_returns_segment_points(self, empty_graph, path_points_blue) -> None:
        road = _commit_road(empty_graph, path_points_blue)
        points = road.get_all_points(segments=empty_graph.segments)
        assert len(points) == len(empty_graph.segments[road.segment_ids[0]].points)

    def test_get_all_points_raises_when_empty(self) -> None:
        # A road referencing no existing segments has no points → error.
        orphan = Road(id="R9", name="x", segment_ids=[], start_node_id="N1", end_node_id="N2")
        with pytest.raises(ValueError, match="at least one point"):
            orphan.get_all_points(segments={})

    def test_has_warnings_reflects_segments(self, empty_graph, path_points_blue) -> None:
        road = _commit_road(empty_graph, path_points_blue)
        seg = empty_graph.segments[road.segment_ids[0]]
        assert road.has_warnings(segments=empty_graph.segments) == bool(seg.warnings)


class TestSegmentKind:
    """A committed segment's kind (SLOPE/ROAD) is intrinsic — set at commit,
    persisted, and read straight off the segment (not reconstructed from owners).
    """

    def test_slope_commit_defaults_to_slope_kind(self, empty_graph, path_points_blue) -> None:
        empty_graph.commit_paths(paths=[ProposedPathSegment(points=path_points_blue, target_difficulty="blue")])
        seg = list(empty_graph.segments.values())[-1]
        assert enum_eq(a=seg.kind, b=SegmentKind.SLOPE)

    def test_road_commit_carries_road_kind(self, empty_graph, path_points_blue) -> None:
        empty_graph.commit_paths(
            paths=[ProposedPathSegment(points=path_points_blue, is_connector=True, kind=SegmentKind.ROAD)]
        )
        seg = list(empty_graph.segments.values())[-1]
        assert enum_eq(a=seg.kind, b=SegmentKind.ROAD)

    def test_from_dict_defaults_to_slope_when_kind_absent(self) -> None:
        # Pre-enum saves have no "kind" key → SLOPE (backward compatible).
        data: dict[str, object] = {
            "id": "S1",
            "name": "Segment 1",
            "points": [
                {"lon": 0.0, "lat": 0.0, "elevation": 2000.0},
                {"lon": 0.0, "lat": -0.001, "elevation": 1990.0},
            ],
            "start_node_id": "N1",
            "end_node_id": "N2",
        }
        assert enum_eq(a=PathSegment.from_dict(data=data).kind, b=SegmentKind.SLOPE)

    def test_from_dict_reads_road_kind(self) -> None:
        data: dict[str, object] = {
            "id": "S1",
            "name": "Segment 1",
            "points": [
                {"lon": 0.0, "lat": 0.0, "elevation": 2000.0},
                {"lon": 0.0, "lat": -0.001, "elevation": 1990.0},
            ],
            "start_node_id": "N1",
            "end_node_id": "N2",
            "kind": "road",
        }
        assert enum_eq(a=PathSegment.from_dict(data=data).kind, b=SegmentKind.ROAD)


class TestBeltWidth:
    """PathSegment.width_m: constant for roads, side-slope-adaptive for slopes."""

    @staticmethod
    def _segment(kind: SegmentKind, side_slope_pct: float) -> "PathSegment":
        return PathSegment(
            points=[
                PathPoint(lon=0.0, lat=0.0, elevation=2000.0),
                PathPoint(lon=0.0, lat=-0.005, elevation=1900.0),
            ],
            id="S1",
            kind=kind,
            side_slope_pct=side_slope_pct,
        )

    def test_road_width_is_constant_regardless_of_side_slope(self) -> None:
        from skiresort_planner.constants import EarthworkConfig

        gentle = self._segment(kind=SegmentKind.ROAD, side_slope_pct=2.0)
        steep = self._segment(kind=SegmentKind.ROAD, side_slope_pct=45.0)
        assert gentle.width_m == float(EarthworkConfig.ROAD_WIDTH_M)
        assert steep.width_m == gentle.width_m  # roads never vary with terrain

    def test_slope_width_narrows_on_steeper_side_slope(self) -> None:
        # Slopes stay adaptive: steeper cross-slope → narrower belt (excavation limit).
        gentle = self._segment(kind=SegmentKind.SLOPE, side_slope_pct=2.0)
        steep = self._segment(kind=SegmentKind.SLOPE, side_slope_pct=45.0)
        assert steep.width_m < gentle.width_m


class TestNaming:
    """Deterministic naming threshold branches (drop/rise bands from NameConfig)."""

    def test_slope_summit_name_for_large_drop(self) -> None:
        # Drop above SUMMIT_RISE_M (500m) → "Summit" in the name.
        name = Slope.generate_name(
            difficulty="black", slope_id="SL1", start_elevation=3000.0, end_elevation=2400.0, avg_bearing=0.0
        )
        assert "Summit" in name

    def test_slope_big_name_for_medium_drop(self) -> None:
        # Drop between BIG_DROP_M (300m) and SUMMIT_RISE_M (500m) → "Big".
        name = Slope.generate_name(
            difficulty="red", slope_id="SL2", start_elevation=3000.0, end_elevation=2650.0, avg_bearing=90.0
        )
        assert "Big" in name

    def test_lift_summit_name_for_large_rise(self) -> None:
        # Vertical rise above SUMMIT_RISE_M (500m) → "Summit" lift name.
        name = Lift.generate_name(
            lift_type="chairlift", lift_id="L1", length_m=800.0, vertical_rise_m=600.0, avg_bearing=0.0
        )
        assert "Summit" in name


class TestIdParsing:
    """Parametrized tests for ID number extraction (SegmentPath base + Lift)."""

    @pytest.mark.parametrize(
        "model_type,id_str,expected_number",
        [
            pytest.param("slope", "SL1", 1, id="slope_single_digit"),
            pytest.param("slope", "SL5", 5, id="slope_mid_digit"),
            pytest.param("slope", "SL10", 10, id="slope_double_digit"),
            pytest.param("slope", "SL123", 123, id="slope_triple_digit"),
            pytest.param("lift", "L1", 1, id="lift_single_digit"),
            pytest.param("lift", "L7", 7, id="lift_mid_digit"),
            pytest.param("lift", "L99", 99, id="lift_double_digit"),
        ],
    )
    def test_number_from_id(self, model_type: str, id_str: str, expected_number: int) -> None:
        """Slope/Lift.number_from_id() extracts numeric part from ID."""
        if model_type == "slope":
            result = Slope.number_from_id(entity_id=id_str)
        else:
            result = Lift.number_from_id(lift_id=id_str)
        assert result == expected_number

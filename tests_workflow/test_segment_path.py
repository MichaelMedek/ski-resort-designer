"""Unit tests for the SegmentPath base (model/segment_path.py)."""

import pytest

from skiresort_planner.model.path_point import PathPoint
from skiresort_planner.model.path_segment import PathSegment, SegmentKind
from skiresort_planner.model.proposed_path import ProposedPathSegment
from skiresort_planner.model.resort_graph import ResortGraph
from skiresort_planner.model.road import Road
from skiresort_planner.model.slope import Slope


def _commit_road(graph: ResortGraph, path_points: list[PathPoint]) -> Road:
    """Commit a path as a road (record_undo=False + finish_road), return the Road."""
    graph.commit_paths(
        paths=[ProposedPathSegment(points=path_points, is_connector=True, kind=SegmentKind.ROAD)], record_undo=False
    )
    road = graph.finish_road(segment_ids=[list(graph.segments.keys())[-1]])
    assert road is not None
    return road


class TestSegmentPathHierarchy:
    def test_number_from_id_uses_subclass_prefix(self) -> None:
        assert Slope.number_from_id("SL7") == 7
        assert Road.number_from_id("R3") == 3


class TestSegmentPathBaseMethods:
    """Shared SegmentPath geometry, exercised through a committed Road."""

    def test_number_property_derives_from_id(self, empty_graph, path_points_blue) -> None:
        road = _commit_road(empty_graph, path_points_blue)
        assert road.id == "R1"
        assert road.number == 1

    def test_total_length_and_drop_match_segment(self, empty_graph, path_points_blue) -> None:
        road = _commit_road(empty_graph, path_points_blue)
        seg = empty_graph.segments[road.segment_ids[0]]
        assert road.get_total_length(segments=empty_graph.segments) == pytest.approx(seg.length_m)
        assert road.get_total_drop(segments=empty_graph.segments) == pytest.approx(seg.total_drop_m)

    def test_get_all_points_returns_segment_points(self, empty_graph, path_points_blue) -> None:
        road = _commit_road(empty_graph, path_points_blue)
        points = road.get_all_points(segments=empty_graph.segments)
        assert len(points) == len(empty_graph.segments[road.segment_ids[0]].points)

    def test_get_all_points_dedups_shared_junction_across_two_segments(self) -> None:
        # Two contiguous segments sharing a junction node: get_all_points must
        # append the second segment's points[1:], dropping the duplicated junction.
        junction = PathPoint(lon=0.0, lat=-0.002, elevation=1960.0)
        seg_a = PathSegment(
            id="S1",
            name="a",
            kind=SegmentKind.ROAD,
            start_node_id="N1",
            end_node_id="N2",
            points=[
                PathPoint(lon=0.0, lat=0.0, elevation=2000.0),
                PathPoint(lon=0.0, lat=-0.001, elevation=1980.0),
                junction,
            ],
        )
        seg_b = PathSegment(
            id="S2",
            name="b",
            kind=SegmentKind.ROAD,
            start_node_id="N2",
            end_node_id="N3",
            points=[
                junction,
                PathPoint(lon=0.0, lat=-0.003, elevation=1940.0),
                PathPoint(lon=0.0, lat=-0.004, elevation=1920.0),
            ],
        )
        segments = {"S1": seg_a, "S2": seg_b}
        road = Road(id="R1", name="x", segment_ids=["S1", "S2"], start_node_id="N1", end_node_id="N3")

        points = road.get_all_points(segments=segments)

        # 3 + 3 - 1: the shared junction is not duplicated.
        assert len(points) == len(seg_a.points) + len(seg_b.points) - 1
        # Continuous sequence: seg_a in full, then seg_b skipping the junction.
        assert points == seg_a.points + seg_b.points[1:]
        # The junction appears exactly once at the seam.
        assert points.count(junction) == 1
        assert points[len(seg_a.points) - 1] == junction

    def test_get_all_points_raises_when_empty(self) -> None:
        # A road referencing no existing segments has no points → error.
        orphan = Road(id="R9", name="x", segment_ids=[], start_node_id="N1", end_node_id="N2")
        with pytest.raises(ValueError, match="at least one point"):
            orphan.get_all_points(segments={})

    def test_has_warnings_reflects_segments(self, empty_graph, path_points_blue) -> None:
        road = _commit_road(empty_graph, path_points_blue)
        seg = empty_graph.segments[road.segment_ids[0]]
        assert road.has_warnings(segments=empty_graph.segments) == bool(seg.warnings)

"""Unit tests for ResortGraph (model/resort_graph.py).

The graph is the single owner of nodes/segments/slopes/lifts/roads and the undo
stack. This module is the authoritative home for:
- commit_paths / finish_slope, node reuse, connector snapping
- per-entity add/delete + undo for slopes, lifts, roads (graph-level behavior)
- undo-stack semantics (order, size cap, empty-stack error)

Road graph-ops / parking / stats classes are added here in the road dissection.
"""

import pytest

from skiresort_planner.constants import MapConfig
from skiresort_planner.core.geo_calculator import GeoCalculator
from skiresort_planner.core.terrain_analyzer import TerrainAnalyzer
from skiresort_planner.model.actions import (
    AddLiftAction,
    AddSegmentsAction,
    DeleteLiftAction,
    DeleteRoadAction,
    DeleteSlopeAction,
    FinishRoadAction,
    FinishSlopeAction,
)
from skiresort_planner.model.node import Node
from skiresort_planner.model.path_point import PathPoint
from skiresort_planner.model.path_segment import PathSegment, SegmentKind
from skiresort_planner.model.proposed_path import ProposedPathSegment
from skiresort_planner.model.resort_graph import NodeDeletability, ResortGraph, _chain_node_sequence
from skiresort_planner.model.slope import Slope
from tests_workflow.conftest import MockDEMService

M = MapConfig.METERS_PER_DEGREE_EQUATOR


def _add_lift(graph: ResortGraph, dem, lift_type: str = "chairlift"):
    """Create two nodes + a lift on the graph and return the lift."""
    graph.nodes["N1"] = Node(
        id="N1",
        location=PathPoint(lon=0.0, lat=-1000 / M, elevation=dem.get_elevation_or_raise(lon=0.0, lat=-1000 / M)),
    )
    graph.nodes["N2"] = Node(
        id="N2", location=PathPoint(lon=0.0, lat=0.0, elevation=dem.get_elevation_or_raise(lon=0.0, lat=0.0))
    )
    return graph.add_lift(start_node_id="N1", end_node_id="N2", lift_type=lift_type, dem=dem)


def _commit_road(graph: ResortGraph, path_points):
    """Commit a path as a road segment (record_undo=True) + finish_road, return the Road.

    Mirrors production road building: each segment carries its own AddSegmentsAction
    undo entry, and finish_road records a FinishRoadAction on top.
    """
    graph.commit_paths(paths=[ProposedPathSegment(points=path_points, is_connector=True, kind=SegmentKind.ROAD)])
    road = graph.finish_road(segment_ids=[list(graph.segments.keys())[-1]])
    assert road is not None
    return road


# =============================================================================
# commit_paths / finish_slope
# =============================================================================


class TestCommitAndFinishWorkflow:
    """Tests for commit_paths and finish_slope operations."""

    def test_commit_paths_creates_nodes_and_segments(self, empty_graph, path_points_blue) -> None:
        """commit_paths creates nodes at endpoints and a segment, and pushes undo."""
        graph = empty_graph
        proposal = ProposedPathSegment(
            points=path_points_blue, target_slope_pct=20.0, target_difficulty="blue", sector_name="Test"
        )

        endpoint_ids = graph.commit_paths(paths=[proposal])

        assert len(graph.nodes) == 2, "Should create 2 nodes (start and end)"
        assert len(graph.segments) == 1, "Should create 1 segment"
        assert len(endpoint_ids) == 1, "Should return 1 endpoint ID"
        assert len(graph.undo_stack) == 1, "Should push undo action"
        assert isinstance(graph.undo_stack[0], AddSegmentsAction), "Undo action should be AddSegmentsAction"

    def test_finish_slope_groups_segments(self, empty_graph, path_points_blue) -> None:
        """finish_slope groups committed segments into a named slope, pushing undo."""
        graph = empty_graph
        proposal = ProposedPathSegment(
            points=path_points_blue, target_slope_pct=20.0, target_difficulty="blue", sector_name="Test"
        )

        graph.commit_paths(paths=[proposal])
        segment_ids = list(graph.segments.keys())

        slope = graph.finish_slope(segment_ids=segment_ids)

        assert slope is not None, "finish_slope should return a Slope"
        assert len(graph.slopes) == 1, "Should have 1 slope"
        assert slope.segment_ids == segment_ids, "Slope should contain all segments"
        assert slope.name is not None, "Slope should have a name"
        assert len(graph.undo_stack) == 2, "Should have 2 undo actions (commit + finish)"
        assert isinstance(graph.undo_stack[-1], FinishSlopeAction), "Last undo should be FinishSlopeAction"

    def test_finish_slope_raises_on_empty_segments(self, empty_graph) -> None:
        """Raise-fast: finishing with no segments is a programming error, not a soft None return."""
        with pytest.raises(ValueError, match="empty segment_ids"):
            empty_graph.finish_slope(segment_ids=[])

    def test_finish_slope_raises_on_missing_segment(self, empty_graph) -> None:
        """Raise-fast: a segment id that isn't in the graph must raise, never return None."""
        with pytest.raises(ValueError, match="missing segment"):
            empty_graph.finish_slope(segment_ids=["S999"])

    def test_finish_road_raises_on_missing_segment(self, empty_graph) -> None:
        """Raise-fast parity for roads (mirrors slope)."""
        with pytest.raises(ValueError, match="missing segment"):
            empty_graph.finish_road(segment_ids=["S999"])


class TestUndoStackIntegrityForMissingSegments:
    """Guards against the stale-segment undo crash.

    Committing a segment records an AddSegmentsAction; finishing records a FinishSlope/RoadAction —
    both reference their segment ids and assume they persist. Deleting the finished slope/road (or a
    merge that collapses one) removes those segments, so BOTH chained entries must be dropped from
    the stack the moment the segments die; otherwise undoing one restores/describes a phantom
    segment. With the scrub in place a live entry always has its segments, so describe() raises loud
    on a violation rather than masking it.
    """

    def test_add_segments_describe_is_empty_and_graph_independent(self, empty_graph) -> None:
        from skiresort_planner.model.actions import AddSegmentsAction
        from skiresort_planner.model.undo_handlers import UNDO_HANDLERS

        # AddSegments is skip_confirm (peeling a segment shows no dialog), so its describe is the
        # empty base — it never indexes the graph, hence never crashes on a stale/missing segment.
        action = AddSegmentsAction(segment_ids=("S172",), node_ids=("N1", "N2"))
        assert UNDO_HANDLERS[action.action_type.name].describe(action=action, graph=empty_graph) == ""

    @staticmethod
    def _chained_entries_are_live(graph) -> None:
        """Every AddSegments/Finish entry left on the stack still references at least one live segment."""
        from skiresort_planner.model.actions import ActionType

        chained = {ActionType.ADD_SEGMENTS.name, ActionType.FINISH_SLOPE.name, ActionType.FINISH_ROAD.name}
        for action in graph.undo_stack:
            if action.action_type.name in chained:
                assert any(sid in graph.segments for sid in action.segment_ids), (
                    f"stale {action.action_type.name} left on stack after segment removal: {action.segment_ids}"
                )

    def test_delete_slope_drops_stale_chain_entries(self, empty_graph, path_points_blue) -> None:
        from skiresort_planner.model.actions import ActionType

        graph = empty_graph
        graph.commit_paths(paths=[ProposedPathSegment(points=path_points_blue, target_difficulty="blue")])
        slope = graph.finish_slope(segment_ids=list(graph.segments.keys()))
        assert any(a.action_type == ActionType.ADD_SEGMENTS for a in graph.undo_stack), "commit recorded ADD_SEGMENTS"
        assert any(a.action_type == ActionType.FINISH_SLOPE for a in graph.undo_stack), "finish recorded FINISH_SLOPE"

        graph.delete_slope(slope_id=slope.id)

        # Both the AddSegments AND the FinishSlope entry referenced the now-deleted segments → gone.
        self._chained_entries_are_live(graph)
        assert not any(a.action_type == ActionType.FINISH_SLOPE for a in graph.undo_stack), (
            "the FinishSlopeAction for the deleted slope must be pruned (this was the actual crash path)"
        )

    def test_delete_road_drops_stale_chain_entries(self, empty_graph, path_points_blue) -> None:
        graph = empty_graph
        graph.commit_paths(
            paths=[ProposedPathSegment(points=path_points_blue, is_connector=True, kind=SegmentKind.ROAD)]
        )
        road = graph.finish_road(segment_ids=list(graph.segments.keys()))

        graph.delete_road(road_id=road.id)

        self._chained_entries_are_live(graph)

    def test_undo_after_delete_never_raises(self, empty_graph, path_points_blue) -> None:
        """The end-to-end guard: commit → finish → delete → undo the whole way down, no crash."""
        from skiresort_planner.model.undo_handlers import UNDO_HANDLERS

        graph = empty_graph
        graph.commit_paths(paths=[ProposedPathSegment(points=path_points_blue, target_difficulty="blue")])
        slope = graph.finish_slope(segment_ids=list(graph.segments.keys()))
        graph.delete_slope(slope_id=slope.id)

        # describe() every remaining entry (the undo dialog does this) and unwind — must not raise.
        while graph.undo_stack:
            top = graph.undo_stack[-1]
            UNDO_HANDLERS[top.action_type.name].describe(action=top, graph=graph)
            graph.undo_last()


class TestUndoActionUnionCompleteness:
    """The UndoAction union must stay 1:1 with ActionType — no member added without the other."""

    def test_union_member_count_matches_action_type(self) -> None:
        import typing

        from skiresort_planner.model.actions import ActionType, UndoAction

        assert len(typing.get_args(UndoAction)) == len(list(ActionType)), (
            "UndoAction union must have exactly one member class per ActionType — a new action "
            "type or class was added without updating the other."
        )


class TestNodeReuse:
    """Tests for node reuse when endpoints are close."""

    def test_nearby_endpoints_share_nodes(self, empty_graph, mock_dem_blue_slope) -> None:
        """Paths with nearby endpoints should share nodes."""
        graph = empty_graph
        dem = mock_dem_blue_slope

        path1_points = [
            PathPoint(lon=0.0, lat=0.0, elevation=dem.get_elevation_or_raise(lon=0.0, lat=0.0)),
            PathPoint(lon=0.0, lat=-500 / M, elevation=dem.get_elevation_or_raise(lon=0.0, lat=-500 / M)),
        ]
        graph.commit_paths(
            paths=[
                ProposedPathSegment(
                    points=path1_points, target_slope_pct=20.0, target_difficulty="blue", sector_name="P1"
                )
            ]
        )
        assert len(graph.nodes) == 2, "First path creates 2 nodes"

        # Second path starting very close to first path's end should reuse node.
        path2_points = [
            PathPoint(lon=0.00001, lat=-500 / M, elevation=dem.get_elevation_or_raise(lon=0.00001, lat=-500 / M)),
            PathPoint(lon=0.0, lat=-1000 / M, elevation=dem.get_elevation_or_raise(lon=0.0, lat=-1000 / M)),
        ]
        graph.commit_paths(
            paths=[
                ProposedPathSegment(
                    points=path2_points, target_slope_pct=20.0, target_difficulty="blue", sector_name="P2"
                )
            ]
        )
        assert len(graph.nodes) == 3, "Second path should reuse 1 node, create 1 new"


class TestConnectorGeometrySnapping:
    """Tests for connector path geometry snapping to target nodes."""

    def test_connector_path_snaps_to_target_node_coordinates(self, empty_graph, mock_dem_blue_slope) -> None:
        """A connector path's endpoint is snapped to the exact target node coords."""
        graph = empty_graph
        dem = mock_dem_blue_slope

        path1_points = [
            PathPoint(lon=0.0, lat=0.0, elevation=dem.get_elevation_or_raise(lon=0.0, lat=0.0)),
            PathPoint(lon=0.0, lat=-500 / M, elevation=dem.get_elevation_or_raise(lon=0.0, lat=-500 / M)),
        ]
        graph.commit_paths(paths=[ProposedPathSegment(points=path1_points, sector_name="P1")])

        first_segment = list(graph.segments.values())[0]
        target_node_id = first_segment.end_node_id
        target_node = graph.nodes[target_node_id]

        connector_points = [
            PathPoint(lon=0.001, lat=0.0, elevation=dem.get_elevation_or_raise(lon=0.001, lat=0.0)),
            PathPoint(
                lon=target_node.lon + 0.00005, lat=target_node.lat + 0.00003, elevation=target_node.elevation + 2.0
            ),
        ]
        connector = ProposedPathSegment(
            points=connector_points, sector_name="Connector", is_connector=True, target_node_id=target_node_id
        )
        graph.commit_paths(paths=[connector])

        committed_segment = list(graph.segments.values())[-1]
        snapped_point = committed_segment.points[-1]

        assert snapped_point.lon == target_node.lon, "Path end lon should match target node"
        assert snapped_point.lat == target_node.lat, "Path end lat should match target node"
        assert snapped_point.elevation == target_node.elevation, "Path end elevation should match target node"
        assert len(graph.nodes) == 3, "Should have 3 nodes (2 from first + 1 new start), not 4"
        assert committed_segment.end_node_id == target_node_id, "Segment should connect to target node"

    def test_start_node_id_reuses_node_and_never_duplicates(self, empty_graph, mock_dem_blue_slope) -> None:
        """A path carrying start_node_id reuses that exact node — even when the traced
        start point has drifted well past the snap threshold (spline smoothing can do
        this). Regression: extending from an existing node must NEVER spawn a duplicate.
        """
        graph = empty_graph
        dem = mock_dem_blue_slope

        path1 = [
            PathPoint(lon=0.0, lat=0.0, elevation=dem.get_elevation_or_raise(lon=0.0, lat=0.0)),
            PathPoint(lon=0.0, lat=-500 / M, elevation=dem.get_elevation_or_raise(lon=0.0, lat=-500 / M)),
        ]
        graph.commit_paths(paths=[ProposedPathSegment(points=path1, sector_name="P1")])
        first_segment = list(graph.segments.values())[0]
        start_node_id = first_segment.end_node_id
        start_node = graph.nodes[start_node_id]
        nodes_before = len(graph.nodes)

        # Next segment's traced start is 200m off the real node — far beyond STEP_SIZE_M
        # snapping — but start_node_id forces exact reuse.
        drifted = [
            PathPoint(lon=200 / M, lat=start_node.lat, elevation=start_node.elevation),
            PathPoint(lon=200 / M, lat=start_node.lat - 500 / M, elevation=start_node.elevation - 40),
        ]
        graph.commit_paths(paths=[ProposedPathSegment(points=drifted, start_node_id=start_node_id)])

        new_segment = list(graph.segments.values())[-1]
        assert new_segment.start_node_id == start_node_id, "must reuse the existing start node"
        assert len(graph.nodes) == nodes_before + 1, "only the END node is new — no duplicate start node"
        assert new_segment.points[0].lon == start_node.lon, "start geometry snapped to the exact node"
        assert new_segment.points[0].lat == start_node.lat


# =============================================================================
# Undo — per entity add/delete + stack semantics (authoritative home)
# =============================================================================


class TestSlopeUndo:
    """Undo for slope-related actions (AddSegments, FinishSlope, DeleteSlope)."""

    def test_undo_single_segment_removes_segment_and_nodes(self, empty_graph, path_points_blue) -> None:
        """Undo AddSegmentsAction removes segment and cleans up isolated nodes."""
        graph = empty_graph
        graph.commit_paths(paths=[ProposedPathSegment(points=path_points_blue, target_difficulty="blue")])

        assert len(graph.segments) == 1 and len(graph.nodes) == 2 and len(graph.undo_stack) == 1

        undone = graph.undo_last()

        assert isinstance(undone, AddSegmentsAction)
        assert len(graph.segments) == 0, "Segment should be removed"
        assert len(graph.undo_stack) == 0, "Stack should be empty"

    def test_undo_preserves_other_segments(self, empty_graph, mock_dem_blue_slope) -> None:
        """Undo removes only the undone segment, preserving others."""
        graph = empty_graph
        dem = mock_dem_blue_slope

        points1 = [
            PathPoint(lon=0.0, lat=0.0, elevation=dem.get_elevation_or_raise(lon=0.0, lat=0.0)),
            PathPoint(lon=0.0, lat=-500 / M, elevation=dem.get_elevation_or_raise(lon=0.0, lat=-500 / M)),
        ]
        graph.commit_paths(paths=[ProposedPathSegment(points=points1, target_difficulty="blue", sector_name="P1")])
        points2 = [
            PathPoint(lon=0.0, lat=-500 / M, elevation=dem.get_elevation_or_raise(lon=0.0, lat=-500 / M)),
            PathPoint(lon=0.0, lat=-1000 / M, elevation=dem.get_elevation_or_raise(lon=0.0, lat=-1000 / M)),
        ]
        graph.commit_paths(paths=[ProposedPathSegment(points=points2, target_difficulty="blue", sector_name="P2")])

        assert len(graph.segments) == 2
        seg_ids = list(graph.segments.keys())

        graph.undo_last()

        assert len(graph.segments) == 1, "Should have 1 segment remaining"
        assert seg_ids[0] in graph.segments, "First segment should remain"
        assert seg_ids[1] not in graph.segments, "Second segment should be removed"

    def test_undo_finish_slope_keeps_segments(self, empty_graph, path_points_blue) -> None:
        """Undo FinishSlopeAction removes slope but preserves segments."""
        graph = empty_graph
        graph.commit_paths(paths=[ProposedPathSegment(points=path_points_blue, target_difficulty="blue")])
        slope = graph.finish_slope(segment_ids=list(graph.segments.keys()))
        slope_name = slope.name

        assert len(graph.slopes) == 1 and len(graph.segments) == 1

        undone = graph.undo_last()

        assert isinstance(undone, FinishSlopeAction)
        assert undone.slope_name == slope_name
        assert len(graph.slopes) == 0, "Slope should be removed"
        assert len(graph.segments) == 1, "Segment should remain"

    def test_undo_delete_slope_restores_slope(self, empty_graph, path_points_blue) -> None:
        """Undo DeleteSlopeAction restores the slope and its segments."""
        graph = empty_graph
        graph.commit_paths(paths=[ProposedPathSegment(points=path_points_blue, target_difficulty="blue")])
        slope = graph.finish_slope(segment_ids=list(graph.segments.keys()))
        slope_id, slope_name = slope.id, slope.name

        graph.delete_slope(slope_id=slope_id)
        assert len(graph.slopes) == 0

        undone = graph.undo_last()

        assert isinstance(undone, DeleteSlopeAction)
        assert len(graph.slopes) == 1, "Slope should be restored"
        assert graph.slopes[slope_id].name == slope_name, "Name should match"


class TestLiftUndo:
    """Undo for lift actions (AddLift, DeleteLift)."""

    def test_undo_add_lift_removes_lift(self, empty_graph, mock_dem_blue_slope) -> None:
        """undo_last for AddLiftAction removes the lift."""
        graph = empty_graph
        _add_lift(graph, mock_dem_blue_slope, lift_type="gondola")

        assert len(graph.lifts) == 1
        assert isinstance(graph.undo_stack[0], AddLiftAction)

        graph.undo_last()
        assert len(graph.lifts) == 0, "Lift should be removed"

    def test_undo_delete_lift_restores_lift(self, empty_graph, mock_dem_blue_slope) -> None:
        """undo_last for DeleteLiftAction restores the lift."""
        graph = empty_graph
        lift = _add_lift(graph, mock_dem_blue_slope)
        lift_id, lift_name = lift.id, lift.name

        graph.delete_lift(lift_id=lift_id)
        assert len(graph.lifts) == 0

        undone = graph.undo_last()

        assert isinstance(undone, DeleteLiftAction)
        assert len(graph.lifts) == 1, "Lift should be restored"
        assert graph.lifts[lift_id].name == lift_name, "Lift name should be preserved"


class TestUndoStackSemantics:
    """Undo-stack ordering, size cap, and empty-stack behavior."""

    def test_multiple_consecutive_undos(self, empty_graph, mock_dem_blue_slope) -> None:
        """Multiple consecutive undos work correctly in sequence."""
        graph = empty_graph
        dem = mock_dem_blue_slope

        for i in range(3):
            start_lat, end_lat = -i * 500 / M, -(i + 1) * 500 / M
            points = [
                PathPoint(lon=0.0, lat=start_lat, elevation=dem.get_elevation_or_raise(lon=0.0, lat=start_lat)),
                PathPoint(lon=0.0, lat=end_lat, elevation=dem.get_elevation_or_raise(lon=0.0, lat=end_lat)),
            ]
            graph.commit_paths(
                paths=[ProposedPathSegment(points=points, target_difficulty="blue", sector_name=f"P{i}")]
            )

        assert len(graph.segments) == 3 and len(graph.undo_stack) == 3

        graph.undo_last()
        assert len(graph.segments) == 2
        graph.undo_last()
        assert len(graph.segments) == 1
        graph.undo_last()
        assert len(graph.segments) == 0 and len(graph.undo_stack) == 0

    def test_empty_undo_stack_raises_runtime_error(self, empty_graph) -> None:
        """undo_last on empty stack raises RuntimeError."""
        with pytest.raises(RuntimeError, match="empty"):
            empty_graph.undo_last()

    def test_undo_stack_has_max_size(self, empty_graph, mock_dem_blue_slope) -> None:
        """Undo stack enforces maximum size, discarding oldest actions."""
        from skiresort_planner.constants import UndoConfig

        graph = empty_graph
        dem = mock_dem_blue_slope

        for i in range(UndoConfig.MAX_UNDO_STACK_SIZE + 5):
            start_lat, end_lat = -i * 50 / M, -(i + 1) * 50 / M
            points = [
                PathPoint(lon=0.0, lat=start_lat, elevation=dem.get_elevation_or_raise(lon=0.0, lat=start_lat)),
                PathPoint(lon=0.0, lat=end_lat, elevation=dem.get_elevation_or_raise(lon=0.0, lat=end_lat)),
            ]
            graph.commit_paths(
                paths=[ProposedPathSegment(points=points, target_difficulty="blue", sector_name=f"P{i}")]
            )

        assert len(graph.undo_stack) <= UndoConfig.MAX_UNDO_STACK_SIZE, (
            f"Stack should not exceed {UndoConfig.MAX_UNDO_STACK_SIZE}"
        )


# =============================================================================
# Roads: finish_road, delete_road, undo, parking, stats
# =============================================================================


class TestRoadGraphOps:
    def test_finish_road_records_finish_action_on_top_of_segments(self, empty_graph, path_points_blue) -> None:
        road = _commit_road(empty_graph, path_points_blue)
        assert road.id in empty_graph.roads
        assert len(empty_graph.roads) == 1
        # One AddSegmentsAction (the committed segment) + one FinishRoadAction on top.
        assert len(empty_graph.undo_stack) == 2
        assert isinstance(empty_graph.undo_stack[0], AddSegmentsAction)
        assert isinstance(empty_graph.undo_stack[-1], FinishRoadAction)

    def test_undo_finish_road_ungroups_but_keeps_segments(self, empty_graph, path_points_blue) -> None:
        road = _commit_road(empty_graph, path_points_blue)
        seg_count = len(empty_graph.segments)
        # First undo: pop the FinishRoadAction → road ungrouped, segments stay.
        empty_graph.undo_last()
        assert road.id not in empty_graph.roads
        assert len(empty_graph.roads) == 0
        assert len(empty_graph.segments) == seg_count, "Segments remain after ungrouping the road"
        # Second undo: pop the AddSegmentsAction → segment (and orphan nodes) removed.
        empty_graph.undo_last()
        assert len(empty_graph.segments) == 0
        assert len(empty_graph.nodes) == 0

    def test_delete_road_removes_and_records_undo(self, empty_graph, path_points_blue) -> None:
        road = _commit_road(empty_graph, path_points_blue)
        assert empty_graph.delete_road(road_id=road.id) is True
        assert road.id not in empty_graph.roads
        assert len(empty_graph.segments) == 0
        assert isinstance(empty_graph.undo_stack[-1], DeleteRoadAction)

    def test_undo_delete_road_restores(self, empty_graph, path_points_blue) -> None:
        road = _commit_road(empty_graph, path_points_blue)
        seg_count = len(empty_graph.segments)
        empty_graph.delete_road(road_id=road.id)
        empty_graph.undo_last()
        assert road.id in empty_graph.roads
        assert len(empty_graph.segments) == seg_count

    def test_delete_missing_road_is_false(self, empty_graph) -> None:
        assert empty_graph.delete_road(road_id="R99") is False


class TestParkingNodes:
    """Parking nodes are computed where a road shares a node with a slope/lift."""

    def test_no_parking_without_roads(self, empty_graph, path_points_blue) -> None:
        empty_graph.commit_paths(paths=[ProposedPathSegment(points=path_points_blue, target_difficulty="blue")])
        empty_graph.finish_slope(segment_ids=list(empty_graph.segments.keys()))
        assert empty_graph.get_parking_nodes() == []

    def test_parking_appears_where_road_shares_slope_node(self, empty_graph, path_points_blue) -> None:
        # Slope along the blue points.
        empty_graph.commit_paths(paths=[ProposedPathSegment(points=path_points_blue, target_difficulty="blue")])
        empty_graph.finish_slope(segment_ids=list(empty_graph.segments.keys()))

        shared_start = path_points_blue[0]  # slope's top node
        # Road starting at the slope's top node, heading east (new distinct end).
        road_points = [
            PathPoint(lon=shared_start.lon, lat=shared_start.lat, elevation=shared_start.elevation),
            PathPoint(lon=300 / M, lat=shared_start.lat, elevation=shared_start.elevation),
        ]
        _commit_road(empty_graph, road_points)

        parking = empty_graph.get_parking_nodes()
        assert len(parking) == 1
        assert isinstance(parking[0], Node)
        assert parking[0].distance_to(lon=shared_start.lon, lat=shared_start.lat) < 1.0

    def test_road_touching_nothing_yields_no_parking(self, empty_graph) -> None:
        road_points = [
            PathPoint(lon=0.0, lat=0.0, elevation=2000.0),
            PathPoint(lon=300 / M, lat=0.0, elevation=2000.0),
        ]
        _commit_road(empty_graph, road_points)
        assert empty_graph.get_parking_nodes() == []


class TestStatsWithRoads:
    def test_stats_reports_road_count_and_length(self, empty_graph, path_points_blue) -> None:
        _commit_road(empty_graph, path_points_blue)
        stats = empty_graph.get_stats()
        assert stats["total_roads"] == 1
        assert stats["total_road_length_m"] > 0


class TestNodeSharingDeletes:
    """Deleting one entity must NOT orphan a node another entity still uses (real-world: a
    road and a slope meeting at a base node; delete one, the junction node must survive).
    """

    def _slope_plus_road_sharing_top_node(self, graph, path_points_blue):
        graph.commit_paths(paths=[ProposedPathSegment(points=path_points_blue, target_difficulty="blue")])
        slope = graph.finish_slope(segment_ids=list(graph.segments.keys()))
        shared = path_points_blue[0]
        road_points = [
            PathPoint(lon=shared.lon, lat=shared.lat, elevation=shared.elevation),
            PathPoint(lon=300 / M, lat=shared.lat, elevation=shared.elevation),
        ]
        road = _commit_road(graph, road_points)
        shared_node_id = graph.segments[slope.segment_ids[0]].start_node_id
        return slope, road, shared_node_id

    def test_delete_slope_keeps_node_shared_with_road(self, empty_graph, path_points_blue) -> None:
        slope, road, shared_id = self._slope_plus_road_sharing_top_node(empty_graph, path_points_blue)
        assert empty_graph.get_connection_count(node_id=shared_id) >= 2  # slope seg + road seg

        assert empty_graph.delete_slope(slope_id=slope.id) is True

        assert shared_id in empty_graph.nodes, "node shared with the road must survive slope deletion"
        assert road.id in empty_graph.roads, "road is untouched by slope deletion"
        assert empty_graph.get_connection_count(node_id=shared_id) == 1, "only the road connection remains"

    def test_delete_both_entities_cleans_shared_node(self, empty_graph, path_points_blue) -> None:
        slope, road, shared_id = self._slope_plus_road_sharing_top_node(empty_graph, path_points_blue)
        empty_graph.delete_slope(slope_id=slope.id)
        empty_graph.delete_road(road_id=road.id)
        assert shared_id not in empty_graph.nodes, "node is orphaned once BOTH users are gone"

    def test_undo_slope_delete_restores_without_duplicating_shared_node(self, empty_graph, path_points_blue) -> None:
        slope, road, shared_id = self._slope_plus_road_sharing_top_node(empty_graph, path_points_blue)
        node_count_before = len(empty_graph.nodes)
        empty_graph.delete_slope(slope_id=slope.id)
        empty_graph.undo_last()
        assert slope.id in empty_graph.slopes, "slope restored"
        assert len(empty_graph.nodes) == node_count_before, "no duplicate node created on restore"


class TestMultipleSlopesFromOneNode:
    """'Multiple ways down' — several slopes fanning off ONE shared hub node (DETAILS_UI.md
    Tips). The hub must count all connections and survive until its LAST user is deleted.
    """

    def _three_slopes_from_hub(self, graph, dem):
        hub = PathPoint(lon=0.0, lat=0.0, elevation=dem.get_elevation_or_raise(lon=0.0, lat=0.0))
        slope_ids = []
        # Ends 200 m apart in lon so no end-node snapping — each slope keeps a distinct end.
        for dlon in (0.0, 200 / M, 400 / M):
            end_lat = -500 / M
            start = PathPoint(lon=hub.lon, lat=hub.lat, elevation=hub.elevation)
            end = PathPoint(lon=dlon, lat=end_lat, elevation=dem.get_elevation_or_raise(lon=dlon, lat=end_lat))
            graph.commit_paths(paths=[ProposedPathSegment(points=[start, end], target_difficulty="blue")])
            slope = graph.finish_slope(segment_ids=[list(graph.segments.keys())[-1]])
            slope_ids.append(slope.id)
        hub_id = graph.segments[graph.slopes[slope_ids[0]].segment_ids[0]].start_node_id
        return slope_ids, hub_id

    def test_hub_counts_all_three_slopes(self, empty_graph, mock_dem_blue_slope) -> None:
        graph = empty_graph
        slope_ids, hub_id = self._three_slopes_from_hub(graph, mock_dem_blue_slope)
        assert len(graph.slopes) == 3
        assert len(graph.nodes) == 4, "one shared hub + three distinct ends"
        assert graph.get_connection_count(node_id=hub_id) == 3, "hub carries all three slope starts"

    def test_deleting_one_slope_keeps_hub_for_the_others(self, empty_graph, mock_dem_blue_slope) -> None:
        graph = empty_graph
        slope_ids, hub_id = self._three_slopes_from_hub(graph, mock_dem_blue_slope)

        graph.delete_slope(slope_id=slope_ids[0])
        assert hub_id in graph.nodes, "hub survives — two other slopes still use it"
        assert graph.get_connection_count(node_id=hub_id) == 2

        graph.delete_slope(slope_id=slope_ids[1])
        graph.delete_slope(slope_id=slope_ids[2])
        assert hub_id not in graph.nodes, "hub orphaned only once its LAST slope is gone"


class TestLiftMetricsInvariants:
    def test_vertical_rise_raises_on_missing_node(self, empty_graph, mock_dem_blue_slope) -> None:
        # A committed lift's endpoint nodes are a graph invariant; a missing one is a real bug,
        # so get_vertical_rise must raise (not silently return 0.0) — matches get_length_m.
        lift = _add_lift(empty_graph, mock_dem_blue_slope)
        with pytest.raises(KeyError):
            lift.get_vertical_rise(nodes={})


# =============================================================================
# Whole-path smoothing on finish (DETAILS.md §5.8)
# =============================================================================


def _leg(lon0, lat0, d_lon, d_lat, n, dem):
    """A straight leg of n points stepping (d_lon, d_lat) metres/point, elevation from DEM."""
    pts = []
    for i in range(n):
        lon = lon0 + d_lon * i / M
        lat = lat0 + d_lat * i / M
        pts.append(PathPoint(lon=lon, lat=lat, elevation=dem.get_elevation_or_raise(lon=lon, lat=lat)))
    return pts


def _turn_deg(a, b, c):
    """Absolute heading change (deg) at b for the polyline a->b->c."""
    h1 = GeoCalculator.initial_bearing_deg(lon1=a.lon, lat1=a.lat, lon2=b.lon, lat2=b.lat)
    h2 = GeoCalculator.initial_bearing_deg(lon1=b.lon, lat1=b.lat, lon2=c.lon, lat2=c.lat)
    d = abs(h1 - h2) % 360
    return d if d <= 180 else 360 - d


def _commit_L_slope(graph, dem):
    """Commit two descending segments meeting at a sharp ~45° junction; return their ids.

    Leg 1 heads due south, leg 2 turns south-east from the shared junction node. Both
    descend on the south-facing DEM, so finish_slope accepts them.
    """
    leg1 = _leg(0.0, 0.0, 0.0, -20.0, 25, dem)  # 480m south
    graph.commit_paths(paths=[ProposedPathSegment(points=leg1, target_difficulty="blue")])
    j = leg1[-1]
    leg2 = _leg(j.lon, j.lat, 20.0, -20.0, 25, dem)  # 480m south-east from the junction
    graph.commit_paths(paths=[ProposedPathSegment(points=leg2, target_difficulty="blue")])
    return list(graph.segments.keys())


class TestFinishSmoothing:
    """Whole-path smoothing runs on finish, rounding junction kinks without rejecting."""

    def test_finish_reduces_junction_turn_angle(self, empty_graph, mock_dem_blue_slope) -> None:
        graph = empty_graph
        seg_ids = _commit_L_slope(graph, mock_dem_blue_slope)
        s1, s2 = graph.segments[seg_ids[0]], graph.segments[seg_ids[1]]
        raw_turn = _turn_deg(s1.points[-2], s1.points[-1], s2.points[1])
        assert raw_turn > 30, "fixture should start with a sharp junction"

        graph.finish_slope(segment_ids=seg_ids)

        joined = graph.slopes[list(graph.slopes)[0]].get_all_points(segments=graph.segments)
        max_turn = max(_turn_deg(joined[i - 1], joined[i], joined[i + 1]) for i in range(1, len(joined) - 1))
        assert max_turn < raw_turn, f"junction should round (raw {raw_turn:.1f} -> {max_turn:.1f})"
        assert max_turn < 20, f"rounded junction should be gentle, got {max_turn:.1f}"

    def test_finish_endpoints_exact_junction_near_and_shared(self, empty_graph, mock_dem_blue_slope) -> None:
        # Outer endpoints land exactly on their nodes (entity termini). The internal junction
        # is shared by value between the two segments and stays within a few metres of its node
        # — NOT snapped back exactly, so a switchback stays a smooth radius rather than a cusp.
        graph = empty_graph
        seg_ids = _commit_L_slope(graph, mock_dem_blue_slope)
        start_node = graph.nodes[graph.segments[seg_ids[0]].start_node_id]
        junction_node = graph.nodes[graph.segments[seg_ids[0]].end_node_id]
        end_node = graph.nodes[graph.segments[seg_ids[-1]].end_node_id]

        graph.finish_slope(segment_ids=seg_ids)

        first_pt = graph.segments[seg_ids[0]].points[0]
        junction_pt = graph.segments[seg_ids[0]].points[-1]
        last_pt = graph.segments[seg_ids[-1]].points[-1]
        assert (first_pt.lon, first_pt.lat, first_pt.elevation) == (
            start_node.lon,
            start_node.lat,
            start_node.elevation,
        )
        assert (last_pt.lon, last_pt.lat, last_pt.elevation) == (end_node.lon, end_node.lat, end_node.elevation)
        assert junction_pt == graph.segments[seg_ids[1]].points[0], "junction shared by value across segments"
        assert junction_pt.distance_to(other=junction_node.location) < 15.0, "junction stays near its node"

    def test_finish_preserves_segment_count_and_ids(self, empty_graph, mock_dem_blue_slope) -> None:
        graph = empty_graph
        seg_ids = _commit_L_slope(graph, mock_dem_blue_slope)

        graph.finish_slope(segment_ids=seg_ids)

        assert list(graph.segments.keys()) == seg_ids, "same segment ids after smoothing"
        for sid in seg_ids:
            assert len(graph.segments[sid].points) >= 2, "each segment keeps >=2 points"

    def test_get_all_points_contiguous_after_finish(self, empty_graph, mock_dem_blue_slope) -> None:
        graph = empty_graph
        seg_ids = _commit_L_slope(graph, mock_dem_blue_slope)

        graph.finish_slope(segment_ids=seg_ids)

        s1, s2 = graph.segments[seg_ids[0]], graph.segments[seg_ids[1]]
        assert s1.points[-1] == s2.points[0], "adjacent segments must share the junction by value"
        slope = graph.slopes[list(graph.slopes)[0]]
        slope.get_all_points(segments=graph.segments)  # must not raise on the dedup

    def test_undo_after_smoothed_finish_keeps_segments(self, empty_graph, mock_dem_blue_slope) -> None:
        graph = empty_graph
        seg_ids = _commit_L_slope(graph, mock_dem_blue_slope)
        graph.finish_slope(segment_ids=seg_ids)
        after_first = [graph.segments[sid].max_slope_pct for sid in seg_ids]

        undone = graph.undo_last()

        assert isinstance(undone, FinishSlopeAction)
        assert list(graph.segments.keys()) == seg_ids, "segments survive finish-undo"
        # Re-finishing an already-smoothed path is idempotent (sub-meter drift, non-accumulating).
        graph.finish_slope(segment_ids=seg_ids)
        after_second = [graph.segments[sid].max_slope_pct for sid in seg_ids]
        for a, b in zip(after_first, after_second, strict=False):
            assert abs(a - b) < 1.0, "re-finish must not drift max_slope_pct meaningfully"

    def test_road_finish_may_exceed_15pct_but_never_none(self, empty_graph, mock_dem_black_slope) -> None:
        # Black terrain (45%) → smoothing a road's junction can push its steepest section
        # over the ±15% build cap. A finished road is allowed to exceed it and must not vanish.
        graph = empty_graph
        dem = mock_dem_black_slope
        leg1 = _leg(0.0, 0.0, 0.0, -20.0, 25, dem)
        graph.commit_paths(paths=[ProposedPathSegment(points=leg1, is_connector=True, kind=SegmentKind.ROAD)])
        j = leg1[-1]
        leg2 = _leg(j.lon, j.lat, 20.0, -20.0, 25, dem)
        graph.commit_paths(paths=[ProposedPathSegment(points=leg2, is_connector=True, kind=SegmentKind.ROAD)])
        seg_ids = list(graph.segments.keys())

        road = graph.finish_road(segment_ids=seg_ids)

        assert road is not None, "a finished road must never be rejected by smoothing"
        assert max(graph.segments[sid].max_slope_pct for sid in seg_ids) > 15.0

    def test_single_segment_finish_is_smoothed(self, empty_graph, path_points_blue) -> None:
        graph = empty_graph
        graph.commit_paths(paths=[ProposedPathSegment(points=path_points_blue, target_difficulty="blue")])
        seg_id = list(graph.segments.keys())[0]
        before = list(graph.segments[seg_id].points)

        slope = graph.finish_slope(segment_ids=[seg_id])

        assert slope is not None
        after = graph.segments[seg_id].points
        # Smoothing resamples the 5 raw points at RESAMPLE_STEP_M (~7m over 800m → many more),
        # with the entity endpoints pinned exactly.
        assert len(after) > len(before), "single-segment path is smoothed (resampled denser)"
        assert after[0] == before[0], "start endpoint pinned exactly"
        assert after[-1] == before[-1], "end endpoint pinned exactly"


class TestImportOSMBatch:
    """An OSM import is ONE undoable batch: it adds many slopes+lifts under a single undo entry,
    and one undo wipes the whole import (so the user can import a different selection).
    """

    def _pistes(self, dem, count):
        pistes = []
        for i in range(count):
            lon = 0.01 * i
            pts = [
                PathPoint(lon=lon, lat=0.0, elevation=dem.get_elevation_or_raise(lon=lon, lat=0.0)),
                PathPoint(lon=lon, lat=-500 / M, elevation=dem.get_elevation_or_raise(lon=lon, lat=-500 / M)),
            ]
            pistes.append((pts, f"Run {i}" if i else None))
        return pistes

    def _lift(self, dem):
        bottom = PathPoint(lon=0.05, lat=-500 / M, elevation=dem.get_elevation_or_raise(lon=0.05, lat=-500 / M))
        top = PathPoint(lon=0.05, lat=0.0, elevation=dem.get_elevation_or_raise(lon=0.05, lat=0.0))
        return (bottom, top, "chairlift", "Gipfelbahn")

    def test_import_adds_entities_as_single_undo(self, empty_graph, mock_dem_blue_slope) -> None:
        graph, dem = empty_graph, mock_dem_blue_slope
        slopes, lifts, duplicates = graph.import_osm(pistes=self._pistes(dem, 3), lifts=[self._lift(dem)], dem=dem)
        assert (slopes, lifts, duplicates) == (3, 1, 0)
        assert len(graph.slopes) == 3 and len(graph.lifts) == 1
        assert len(graph.undo_stack) == 1, "the whole import is ONE undo entry"

    def test_one_undo_reverts_the_whole_import(self, empty_graph, mock_dem_blue_slope) -> None:
        graph, dem = empty_graph, mock_dem_blue_slope
        graph.import_osm(pistes=self._pistes(dem, 3), lifts=[self._lift(dem)], dem=dem)

        graph.undo_last()

        assert len(graph.slopes) == 0 and len(graph.lifts) == 0
        assert len(graph.segments) == 0 and len(graph.nodes) == 0, "every imported entity + node is gone"
        assert len(graph.undo_stack) == 0

    def test_undo_import_keeps_pre_existing_entities(self, empty_graph, mock_dem_blue_slope, path_points_blue) -> None:
        graph, dem = empty_graph, mock_dem_blue_slope
        # A hand-built slope exists BEFORE the import.
        graph.commit_paths(paths=[ProposedPathSegment(points=path_points_blue, target_difficulty="blue")])
        pre_slope = graph.finish_slope(segment_ids=list(graph.segments.keys()))
        nodes_before = set(graph.nodes)

        graph.import_osm(pistes=self._pistes(dem, 2), lifts=[], dem=dem)
        graph.undo_last()  # undo ONLY the import

        assert pre_slope.id in graph.slopes, "the pre-existing slope survives the import-undo"
        assert set(graph.nodes) == nodes_before, "only import-created nodes were removed"

    def test_reimport_same_area_adds_nothing(self, empty_graph, mock_dem_blue_slope) -> None:
        """Re-importing identical pistes+lifts is idempotent: the second import adds zero."""
        graph, dem = empty_graph, mock_dem_blue_slope
        graph.import_osm(pistes=self._pistes(dem, 3), lifts=[self._lift(dem)], dem=dem)

        slopes, lifts, duplicates = graph.import_osm(pistes=self._pistes(dem, 3), lifts=[self._lift(dem)], dem=dem)

        assert (slopes, lifts, duplicates) == (0, 0, 4), "all 3 pistes + 1 lift recognised as already imported"
        assert len(graph.slopes) == 3 and len(graph.lifts) == 1, "no duplicates created"

    def test_reimport_adds_only_new_entities(self, empty_graph, mock_dem_blue_slope) -> None:
        """A second import over an overlapping area adds only the genuinely-new runs."""
        graph, dem = empty_graph, mock_dem_blue_slope
        graph.import_osm(pistes=self._pistes(dem, 2), lifts=[], dem=dem)

        # 3 pistes: the first 2 overlap the previous import, the 3rd is new.
        slopes, lifts, duplicates = graph.import_osm(pistes=self._pistes(dem, 3), lifts=[], dem=dem)

        assert (slopes, duplicates) == (1, 2)
        assert len(graph.slopes) == 3

    def test_imported_entities_expose_endpoints(self, empty_graph, mock_dem_blue_slope) -> None:
        graph, dem = empty_graph, mock_dem_blue_slope
        graph.import_osm(pistes=self._pistes(dem, 1), lifts=[self._lift(dem)], dem=dem)
        # endpoints() returns the two node locations, computed on demand (never stored).
        assert all(len(s.endpoints(nodes=graph.nodes)) == 2 for s in graph.slopes.values())
        assert all(len(lift.endpoints(nodes=graph.nodes)) == 2 for lift in graph.lifts.values())

    def test_imported_slope_and_lift_take_osm_name(self, empty_graph, mock_dem_blue_slope) -> None:
        graph, dem = empty_graph, mock_dem_blue_slope
        # piste index 1 is named "Run 1"; the lift is named "Gipfelbahn".
        graph.import_osm(pistes=self._pistes(dem, 2), lifts=[self._lift(dem)], dem=dem)
        assert any(s.name == "Run 1" for s in graph.slopes.values()), "OSM piste name kept verbatim"
        assert any(lift.name == "Gipfelbahn" for lift in graph.lifts.values()), "OSM lift name kept verbatim"

    def test_hand_built_slope_blocks_matching_import(self, empty_graph, mock_dem_blue_slope) -> None:
        """Dedup is source-agnostic: a run the user built by hand skips the matching OSM import."""
        graph, dem = empty_graph, mock_dem_blue_slope
        # Hand-build the same run the importer's first piste would create (same endpoints).
        points, name = self._pistes(dem, 1)[0]
        graph.commit_paths(paths=[ProposedPathSegment(points=points, kind=SegmentKind.SLOPE)])
        graph.finish_slope(segment_ids=list(graph.segments.keys()))
        assert len(graph.slopes) == 1

        slopes, _lifts, duplicates = graph.import_osm(pistes=[(points, name)], lifts=[], dem=dem)

        assert (slopes, duplicates) == (0, 1), "the hand-built run is recognised, not duplicated"
        assert len(graph.slopes) == 1

    def test_has_endpoint_duplicate_false_for_absent_run(self, empty_graph, mock_dem_blue_slope) -> None:
        dem = mock_dem_blue_slope
        a = PathPoint(lon=0.4, lat=0.4, elevation=dem.get_elevation_or_raise(lon=0.4, lat=0.4))
        b = PathPoint(lon=0.4, lat=0.3, elevation=dem.get_elevation_or_raise(lon=0.4, lat=0.3))
        assert empty_graph.has_endpoint_duplicate(a=a, b=b) is False

    def test_reimport_is_idempotent_with_snapped_shared_junctions(self, empty_graph, mock_dem_blue_slope) -> None:
        """Regression: real resorts share junctions, so import SNAPS endpoints onto common nodes.

        The re-import duplicate check must snap candidate endpoints the same way, or snapped runs
        key differently from their stored (snapped) form and are wrongly re-added. Build two pistes
        that meet at a shared top within snap range, import, then re-import the SAME two → 0 added.
        """
        graph, dem = empty_graph, mock_dem_blue_slope
        m = 111320.0

        def pt(lon, lat):
            return PathPoint(lon=lon, lat=lat, elevation=dem.get_elevation_or_raise(lon=lon, lat=lat))

        # Two runs whose TOP endpoints are ~5 m apart (< STEP_SIZE_M=30 m) → they snap to one node.
        shared_a = [pt(0.0, 0.0), pt(0.0, -600 / m)]
        shared_b = [pt(5 / m, 0.0), pt(0.02, -600 / m)]  # top ~5 m east of run A's top
        pistes = [(shared_a, "A"), (shared_b, "B")]

        r1 = graph.import_osm(pistes=pistes, lifts=[], dem=dem)
        assert r1.slopes_added == 2 and r1.duplicates_skipped == 0

        r2 = graph.import_osm(pistes=pistes, lifts=[], dem=dem)
        assert (r2.slopes_added, r2.duplicates_skipped) == (0, 2), "snapped re-import must be idempotent"
        assert len(graph.slopes) == 2, "no duplicate slopes created"


class TestRename:
    """graph.rename sets a custom name on a slope/lift/road by id (and its segments for slopes/roads)."""

    def test_rename_slope_also_renames_its_segments(self, empty_graph, path_points_blue) -> None:
        graph = empty_graph
        graph.commit_paths(paths=[ProposedPathSegment(points=path_points_blue, target_difficulty="blue")])
        slope = graph.finish_slope(segment_ids=list(graph.segments.keys()))

        graph.rename(entity_id=slope.id, new_name="My Run")

        assert graph.slopes[slope.id].name == "My Run"
        assert all(graph.segments[sid].name == "My Run" for sid in slope.segment_ids), "segments renamed too"

    def test_rename_road_also_renames_its_segments(self, empty_graph) -> None:
        graph = empty_graph
        pts = [PathPoint(lon=0.0, lat=0.0, elevation=2000.0), PathPoint(lon=300 / M, lat=0.0, elevation=1990.0)]
        graph.commit_paths(paths=[ProposedPathSegment(points=pts, is_connector=True, kind=SegmentKind.ROAD)])
        road = graph.finish_road(segment_ids=list(graph.segments.keys()))

        graph.rename(entity_id=road.id, new_name="Access Road")

        assert graph.roads[road.id].name == "Access Road"
        assert all(graph.segments[sid].name == "Access Road" for sid in road.segment_ids)

    def test_rename_lift(self, empty_graph, mock_dem_blue_slope) -> None:
        graph, dem = empty_graph, mock_dem_blue_slope
        bottom, _ = graph.get_or_create_node(
            lon=0.0, lat=-1000 / M, elevation=dem.get_elevation_or_raise(lon=0.0, lat=-1000 / M)
        )
        top, _ = graph.get_or_create_node(lon=0.0, lat=0.0, elevation=dem.get_elevation_or_raise(lon=0.0, lat=0.0))
        lift = graph.add_lift(start_node_id=bottom.id, end_node_id=top.id, lift_type="chairlift", dem=dem)

        graph.rename(entity_id=lift.id, new_name="Sunrise Express")

        assert graph.lifts[lift.id].name == "Sunrise Express"

    def test_rename_unknown_id_raises(self, empty_graph) -> None:
        with pytest.raises(KeyError):
            empty_graph.rename(entity_id="SL999", new_name="x")


class TestLiftTypeChangeKeepsName:
    """Changing a lift's type must NOT regenerate its name (it would clobber a custom/OSM name)."""

    def test_update_type_preserves_name_and_updates_geometry(self, empty_graph, mock_dem_blue_slope) -> None:
        graph, dem = empty_graph, mock_dem_blue_slope
        bottom, _ = graph.get_or_create_node(
            lon=0.0, lat=-1000 / M, elevation=dem.get_elevation_or_raise(lon=0.0, lat=-1000 / M)
        )
        top, _ = graph.get_or_create_node(lon=0.0, lat=0.0, elevation=dem.get_elevation_or_raise(lon=0.0, lat=0.0))
        lift = graph.add_lift(start_node_id=bottom.id, end_node_id=top.id, lift_type="chairlift", dem=dem)
        graph.rename(entity_id=lift.id, new_name="Keep Me")
        cable_before = lift.cable_points

        lift.update_type(new_type="gondola", start_node=graph.nodes[bottom.id], end_node=graph.nodes[top.id])

        assert lift.name == "Keep Me", "type change must not rename the lift"
        assert lift.lift_type == "gondola"
        assert lift.cable_points is not cable_before, "type-dependent geometry still recomputed"


class TestUndoActionBijection:
    """Tripwire: the ActionType ↔ *Action dataclass ↔ handler mapping must be an exact bijection.

    Three parallel structures must cover every ActionType with no gap and no duplicate:
    - each *Action dataclass (in the UndoAction union) reports a unique .action_type,
    - the model UNDO_HANDLERS registry (graph mutation + description),
    - the UI _UNDO_SIDE_EFFECTS registry (post-undo UI updates).

    Adding an ActionType (or an action dataclass) without wiring all three previously shipped a
    user-facing crash (the IMPORT_OSM undo-describe bug). The registries self-assert their keyset at
    import; this test additionally verifies the dataclass side of the bijection.
    """

    def _action_classes(self) -> list[type]:
        import typing

        from skiresort_planner.model import actions as actions_mod

        # The UndoAction union is the source of truth for "every action dataclass".
        return list(typing.get_args(actions_mod.UndoAction))

    def test_every_action_dataclass_reports_a_unique_action_type(self) -> None:
        import dataclasses

        from skiresort_planner.model.actions import ActionType

        classes = self._action_classes()
        # Each frozen dataclass exposes .action_type as a property; instantiate a zero-arg-free
        # dummy via object.__new__ to read it without constructing real field values.
        types: list[ActionType] = []
        for cls in classes:
            assert dataclasses.is_dataclass(cls), f"{cls.__name__} in UndoAction union is not a dataclass"
            inst = object.__new__(cls)
            # Every UndoAction member exposes `.action_type`; the union type is opaque to mypy here.
            types.append(inst.action_type)  # type: ignore[attr-defined]  # union member property

        # Surjective: every ActionType is claimed by some dataclass.
        assert set(types) == set(ActionType), (
            f"action dataclasses must cover every ActionType. "
            f"Missing: {set(ActionType) - set(types)}; extra: {set(types) - set(ActionType)}"
        )
        # Injective: no two dataclasses claim the same ActionType.
        assert len(types) == len(set(types)), f"duplicate .action_type across action dataclasses: {types}"

    def test_both_registries_cover_every_action_type(self) -> None:
        from skiresort_planner.model.actions import ActionType
        from skiresort_planner.model.undo_handlers import UNDO_HANDLERS
        from skiresort_planner.ui.actions import _UNDO_SIDE_EFFECTS

        names = {t.name for t in ActionType}
        assert set(UNDO_HANDLERS) == names, f"UNDO_HANDLERS keyset != ActionType: {set(UNDO_HANDLERS) ^ names}"
        assert set(_UNDO_SIDE_EFFECTS) == names, (
            f"_UNDO_SIDE_EFFECTS keyset != ActionType: {set(_UNDO_SIDE_EFFECTS) ^ names}"
        )


# =============================================================================
# Node merge (median collapse) + undo
# =============================================================================


class TestMergeNodes:
    """merge_nodes collapses several nodes to their median, repointing every segment/lift endpoint
    onto the survivor, as ONE undoable action. Undo restores the graph exactly.
    """

    M = MapConfig.METERS_PER_DEGREE_EQUATOR

    def _node(self, graph: ResortGraph, dem: MockDEMService, node_id: str, lon: float, lat: float) -> None:
        graph.nodes[node_id] = Node(
            id=node_id, location=PathPoint(lon=lon, lat=lat, elevation=dem.get_elevation_or_raise(lon=lon, lat=lat))
        )

    def test_merge_moves_survivor_to_median_and_deletes_others(self, empty_graph, mock_dem_blue_slope) -> None:
        dem = mock_dem_blue_slope
        # Three nodes a few metres apart (well within MAX_SPAN_M). A lift keeps the survivor from being
        # cleaned up as isolated after the merge.
        self._node(empty_graph, dem, "A", 0.0, 0.0)
        self._node(empty_graph, dem, "B", 10 / self.M, 0.0)
        self._node(empty_graph, dem, "C", 20 / self.M, 30 / self.M)
        self._node(empty_graph, dem, "T", 0.0, -1000 / self.M)
        empty_graph.add_lift(start_node_id="A", end_node_id="T", lift_type="chairlift", dem=dem)

        empty_graph.merge_nodes(node_ids=["A", "B", "C"], dem=dem)

        assert "B" not in empty_graph.nodes and "C" not in empty_graph.nodes, "merged-away nodes deleted"
        assert "A" in empty_graph.nodes, "survivor (first id) remains"
        survivor = empty_graph.nodes["A"]
        # Median of lons {0,10,20}/M and lats {0,0,30}/M is the middle value each.
        assert survivor.lon == pytest.approx(10 / self.M)
        assert survivor.lat == pytest.approx(0.0)
        # Elevation was re-sampled from the DEM at the median point.
        assert survivor.elevation == pytest.approx(dem.get_elevation_or_raise(lon=10 / self.M, lat=0.0))

    def test_merge_repoints_lift_endpoints_onto_survivor(self, empty_graph, mock_dem_blue_slope) -> None:
        dem = mock_dem_blue_slope
        # Bottom station split into two near-coincident nodes (A survivor, B merged); top is T.
        self._node(empty_graph, dem, "A", 0.0, 0.0)
        self._node(empty_graph, dem, "B", 5 / self.M, 0.0)
        self._node(empty_graph, dem, "T", 0.0, -1000 / self.M)
        lift = empty_graph.add_lift(start_node_id="B", end_node_id="T", lift_type="chairlift", dem=dem)

        empty_graph.merge_nodes(node_ids=["A", "B"], dem=dem)

        assert "B" not in empty_graph.nodes
        assert empty_graph.lifts[lift.id].start_node_id == "A", "lift start repointed onto survivor"
        assert empty_graph.lifts[lift.id].end_node_id == "T", "unrelated endpoint untouched"

    def test_merge_records_single_undo_entry(self, empty_graph, mock_dem_blue_slope) -> None:
        dem = mock_dem_blue_slope
        self._node(empty_graph, dem, "A", 0.0, 0.0)
        self._node(empty_graph, dem, "B", 8 / self.M, 0.0)
        before = len(empty_graph.undo_stack)
        empty_graph.merge_nodes(node_ids=["A", "B"], dem=dem)
        assert len(empty_graph.undo_stack) == before + 1

    def test_undo_restores_nodes_and_repointed_endpoints(self, empty_graph, mock_dem_blue_slope) -> None:
        dem = mock_dem_blue_slope
        self._node(empty_graph, dem, "A", 0.0, 0.0)
        self._node(empty_graph, dem, "B", 6 / self.M, 0.0)
        self._node(empty_graph, dem, "T", 0.0, -1000 / self.M)
        lift = empty_graph.add_lift(start_node_id="B", end_node_id="T", lift_type="chairlift", dem=dem)
        a_before = (empty_graph.nodes["A"].lon, empty_graph.nodes["A"].lat, empty_graph.nodes["A"].elevation)
        b_before = (empty_graph.nodes["B"].lon, empty_graph.nodes["B"].lat)

        empty_graph.merge_nodes(node_ids=["A", "B"], dem=dem)
        empty_graph.undo_last()

        assert set(empty_graph.nodes) >= {"A", "B", "T"}, "merged node restored"
        assert (empty_graph.nodes["A"].lon, empty_graph.nodes["A"].lat, empty_graph.nodes["A"].elevation) == a_before
        assert (empty_graph.nodes["B"].lon, empty_graph.nodes["B"].lat) == b_before
        assert empty_graph.lifts[lift.id].start_node_id == "B", "lift endpoint repointed back onto B"

    def test_merge_raises_when_nodes_too_far(self, empty_graph, mock_dem_blue_slope) -> None:
        from skiresort_planner.constants import MergeConfig

        dem = mock_dem_blue_slope
        far = (MergeConfig.MAX_SPAN_M + 100) / self.M
        self._node(empty_graph, dem, "A", 0.0, 0.0)
        self._node(empty_graph, dem, "B", 0.0, -far)
        with pytest.raises(ValueError, match="span"):
            empty_graph.merge_nodes(node_ids=["A", "B"], dem=dem)
        # No mutation on refusal.
        assert set(empty_graph.nodes) == {"A", "B"}

    def test_merge_raises_below_two_nodes(self, empty_graph, mock_dem_blue_slope) -> None:
        dem = mock_dem_blue_slope
        self._node(empty_graph, dem, "A", 0.0, 0.0)
        with pytest.raises(ValueError, match="at least two"):
            empty_graph.merge_nodes(node_ids=["A"], dem=dem)

    def test_max_node_span_m(self, empty_graph, mock_dem_blue_slope) -> None:
        dem = mock_dem_blue_slope
        self._node(empty_graph, dem, "A", 0.0, 0.0)
        self._node(empty_graph, dem, "B", 0.0, -100 / self.M)
        span = empty_graph.max_node_span_m(["A", "B"])
        assert span == pytest.approx(100.0, abs=1.0)

    # -- geometry re-stitch after merge (the split-station tangle fix) --------------------------

    def test_merge_resyncs_slope_boundary_ids_so_endpoints_do_not_dangle(
        self, empty_graph, mock_dem_blue_slope
    ) -> None:
        """Regression: merging a slope's boundary node used to leave slope.start/end_node_id pointing
        at a deleted node → `ValueError: Start or end node not found` on the next import. The slope's
        own boundary ids must be resynced to live nodes after merge.
        """
        dem = mock_dem_blue_slope
        graph = empty_graph
        seg_ids = _commit_L_slope(graph, dem)
        graph.finish_slope(segment_ids=seg_ids)
        slope = graph.slopes[next(iter(graph.slopes))]
        top_node_id = slope.start_node_id  # the run's top boundary node

        # A second node a few metres from the slope top, to merge the top into.
        self._node(graph, dem, "X", 6 / self.M, 6 / self.M)
        graph.merge_nodes(node_ids=[top_node_id, "X"], dem=dem)

        # The boundary ids must resolve to live nodes — endpoints() would raise otherwise.
        start_pt, end_pt = slope.endpoints(nodes=graph.nodes)
        assert start_pt is not None and end_pt is not None
        assert slope.start_node_id in graph.nodes and slope.end_node_id in graph.nodes

    def test_merge_restitches_segment_polyline_endpoint_to_survivor(self, empty_graph, mock_dem_blue_slope) -> None:
        """After merge, an affected segment's drawn polyline endpoint sits exactly on the survivor
        (not the pre-merge coordinate), so the slope actually reaches the merged node.
        """
        dem = mock_dem_blue_slope
        graph = empty_graph
        seg_ids = _commit_L_slope(graph, dem)
        graph.finish_slope(segment_ids=seg_ids)
        slope = graph.slopes[next(iter(graph.slopes))]
        top_seg = graph.segments[slope.segment_ids[0]]
        assert top_seg.start_node_id == slope.start_node_id
        top_node_id = slope.start_node_id

        self._node(graph, dem, "X", 8 / self.M, 8 / self.M)
        graph.merge_nodes(node_ids=[top_node_id, "X"], dem=dem)

        survivor = graph.nodes[top_node_id]
        # The segment's first drawn point is snapped onto the moved survivor node.
        assert top_seg.points[0].lon == pytest.approx(survivor.lon)
        assert top_seg.points[0].lat == pytest.approx(survivor.lat)
        assert top_seg.points[0].elevation == pytest.approx(survivor.elevation)

    def test_merge_rebuilds_lift_cable_from_moved_station(self, empty_graph, mock_dem_blue_slope) -> None:
        """Merging a lift station rebuilds the cable: first cable point lands on the survivor and the
        cached geometry actually changes (was left stale before).
        """
        dem = mock_dem_blue_slope
        graph = empty_graph
        self._node(graph, dem, "A", 0.0, 0.0)
        self._node(graph, dem, "B", 40 / self.M, 0.0)  # split bottom station, 40m from A
        self._node(graph, dem, "T", 0.0, -1000 / self.M)
        lift = graph.add_lift(start_node_id="B", end_node_id="T", lift_type="chairlift", dem=dem)
        cable_before = [(p.lon, p.lat) for p in graph.lifts[lift.id].cable_points]

        graph.merge_nodes(node_ids=["A", "B"], dem=dem)

        rebuilt = graph.lifts[lift.id]
        survivor = graph.nodes["A"]
        assert rebuilt.start_node_id == "A"
        assert rebuilt.cable_points[0].lon == pytest.approx(survivor.lon)
        assert rebuilt.cable_points[0].lat == pytest.approx(survivor.lat)
        cable_after = [(p.lon, p.lat) for p in rebuilt.cable_points]
        assert cable_after != cable_before, "cable geometry must be recomputed, not left stale"

    def test_undo_restores_segment_and_lift_geometry_exactly(self, empty_graph, mock_dem_blue_slope) -> None:
        """Undo of a merge restores the pre-merge drawn geometry byte-for-byte (segment polylines +
        lift cable) and the slope boundary ids.
        """
        dem = mock_dem_blue_slope
        graph = empty_graph
        seg_ids = _commit_L_slope(graph, dem)
        graph.finish_slope(segment_ids=seg_ids)
        slope = graph.slopes[next(iter(graph.slopes))]
        top_seg = graph.segments[slope.segment_ids[0]]
        top_node_id = slope.start_node_id

        self._node(graph, dem, "T", 0.0, -1200 / self.M)
        lift = graph.add_lift(start_node_id=top_node_id, end_node_id="T", lift_type="chairlift", dem=dem)

        seg_points_before = [(p.lon, p.lat, p.elevation) for p in top_seg.points]
        cable_before = [(p.lon, p.lat, p.elevation) for p in graph.lifts[lift.id].cable_points]
        slope_boundary_before = (slope.start_node_id, slope.end_node_id)

        self._node(graph, dem, "X", 8 / self.M, 8 / self.M)
        graph.merge_nodes(node_ids=[top_node_id, "X"], dem=dem)
        graph.undo_last()

        top_seg_after = graph.segments[slope.segment_ids[0]]
        restored_slope = graph.slopes[next(iter(graph.slopes))]
        assert [(p.lon, p.lat, p.elevation) for p in top_seg_after.points] == seg_points_before
        assert [(p.lon, p.lat, p.elevation) for p in graph.lifts[lift.id].cable_points] == cable_before
        assert (restored_slope.start_node_id, restored_slope.end_node_id) == slope_boundary_before

    def test_merge_after_slope_does_not_break_import_duplicate_check(self, empty_graph, mock_dem_blue_slope) -> None:
        """End-to-end regression for the crash: after merging a slope boundary node, has_endpoint_duplicate
        (called by import_osm) must not raise on the slope's now-updated endpoints.
        """
        dem = mock_dem_blue_slope
        graph = empty_graph
        seg_ids = _commit_L_slope(graph, dem)
        graph.finish_slope(segment_ids=seg_ids)
        slope = graph.slopes[next(iter(graph.slopes))]
        top_node_id = slope.start_node_id

        self._node(graph, dem, "X", 5 / self.M, 5 / self.M)
        graph.merge_nodes(node_ids=[top_node_id, "X"], dem=dem)

        # Would raise ValueError("Start or end node not found ...") before the fix.
        result = graph.has_endpoint_duplicate(
            a=PathPoint(lon=0.5, lat=0.5, elevation=0.0), b=PathPoint(lon=0.6, lat=0.6, elevation=0.0)
        )
        assert result is False

    # -- collapse to zero length deletes the entity (both endpoints merged onto the survivor) ----

    def test_merge_collapsing_lift_to_zero_length_deletes_it(self, empty_graph, mock_dem_blue_slope) -> None:
        """Merging both stations of a lift onto one node would give it 0 length (rebuild crashes on
        0 distance) — the lift is deleted instead, as part of the single merge undo entry.
        """
        dem = mock_dem_blue_slope
        graph = empty_graph
        self._node(graph, dem, "A", 0.0, 0.0)
        self._node(graph, dem, "B", 30 / self.M, 0.0)
        lift = graph.add_lift(start_node_id="A", end_node_id="B", lift_type="chairlift", dem=dem)
        before = len(graph.undo_stack)

        graph.merge_nodes(node_ids=["A", "B"], dem=dem)

        assert lift.id not in graph.lifts, "a lift collapsed to zero length is deleted, not rebuilt"
        assert len(graph.undo_stack) == before + 1, "still a single MERGE_NODES undo entry"

    def test_merge_collapsing_slope_deletes_it_and_its_segments(self, empty_graph, mock_dem_blue_slope) -> None:
        """Merging both boundary nodes of a single-segment slope collapses it — the slope and its
        segment are deleted.
        """
        dem = mock_dem_blue_slope
        graph = empty_graph
        graph.commit_paths(
            paths=[ProposedPathSegment(points=_leg(0.0, 0.0, 0.0, -20.0, 6, dem), target_difficulty="blue")]
        )
        slope = graph.finish_slope(segment_ids=list(graph.segments.keys()))
        seg_ids = list(slope.segment_ids)
        start_id, end_id = slope.start_node_id, slope.end_node_id

        graph.merge_nodes(node_ids=[start_id, end_id], dem=dem)

        assert slope.id not in graph.slopes, "a slope collapsed to zero length is deleted"
        assert all(sid not in graph.segments for sid in seg_ids), "its segments are removed too"

    def test_merge_collapsing_road_deletes_it_and_its_segments(self, empty_graph, mock_dem_blue_slope) -> None:
        """Same as the slope case, for a single-segment road."""
        dem = mock_dem_blue_slope
        graph = empty_graph
        road = _commit_road(graph, _leg(0.0, 0.0, 0.0, -20.0, 6, dem))
        seg_ids = list(road.segment_ids)
        start_id, end_id = road.start_node_id, road.end_node_id

        graph.merge_nodes(node_ids=[start_id, end_id], dem=dem)

        assert road.id not in graph.roads, "a road collapsed to zero length is deleted"
        assert all(sid not in graph.segments for sid in seg_ids), "its segments are removed too"

    def test_undo_restores_collapsed_lift(self, empty_graph, mock_dem_blue_slope) -> None:
        """Undo of the collapsing merge brings the deleted lift back with its original endpoints."""
        dem = mock_dem_blue_slope
        graph = empty_graph
        self._node(graph, dem, "A", 0.0, 0.0)
        self._node(graph, dem, "B", 30 / self.M, 0.0)
        lift = graph.add_lift(start_node_id="A", end_node_id="B", lift_type="chairlift", dem=dem)

        graph.merge_nodes(node_ids=["A", "B"], dem=dem)
        graph.undo_last()

        assert lift.id in graph.lifts, "undo restores the collapsed lift"
        restored = graph.lifts[lift.id]
        assert (restored.start_node_id, restored.end_node_id) == ("A", "B"), "endpoints restored"
        assert "B" in graph.nodes, "merged-away node restored"

    def test_undo_restores_collapsed_slope_and_segments(self, empty_graph, mock_dem_blue_slope) -> None:
        """Undo of the collapsing merge brings the slope and its segments back, endpoints distinct."""
        dem = mock_dem_blue_slope
        graph = empty_graph
        graph.commit_paths(
            paths=[ProposedPathSegment(points=_leg(0.0, 0.0, 0.0, -20.0, 6, dem), target_difficulty="blue")]
        )
        slope = graph.finish_slope(segment_ids=list(graph.segments.keys()))
        seg_ids = list(slope.segment_ids)
        boundary_before = (slope.start_node_id, slope.end_node_id)

        graph.merge_nodes(node_ids=[slope.start_node_id, slope.end_node_id], dem=dem)
        graph.undo_last()

        assert slope.id in graph.slopes, "undo restores the collapsed slope"
        assert all(sid in graph.segments for sid in seg_ids), "its segments are restored"
        restored = graph.slopes[slope.id]
        assert (restored.start_node_id, restored.end_node_id) == boundary_before, "boundary ids restored"
        assert restored.start_node_id != restored.end_node_id, "endpoints distinct again"

    def test_undo_collapsing_merge_after_survivor_swept_by_cleanup(self, empty_graph, mock_dem_blue_slope) -> None:
        """Regression: a merge that collapses the survivor's only slope leaves the survivor node
        isolated; a later cleanup_isolated_nodes removes it entirely. Undo must still restore the
        merge (survivor recreated wholesale), not crash with KeyError on the missing survivor.
        """
        dem = mock_dem_blue_slope
        graph = empty_graph
        graph.commit_paths(
            paths=[ProposedPathSegment(points=_leg(0.0, 0.0, 0.0, -20.0, 6, dem), target_difficulty="blue")]
        )
        slope = graph.finish_slope(segment_ids=list(graph.segments.keys()))
        survivor = slope.start_node_id

        graph.merge_nodes(node_ids=[slope.start_node_id, slope.end_node_id], dem=dem)
        assert graph.cleanup_isolated_nodes() >= 1, "the collapsed merge left the survivor isolated"
        assert survivor not in graph.nodes, "cleanup removed the isolated survivor"

        graph.undo_last()  # must not KeyError on the swept survivor

        assert survivor in graph.nodes, "undo recreates the swept survivor node"
        assert slope.id in graph.slopes, "undo restores the collapsed slope"

    def test_merge_partial_collapse_splices_out_the_zero_length_segment(self, empty_graph, mock_dem_blue_slope) -> None:
        """Merging both endpoints of ONE segment of a two-segment slope collapses that segment to
        zero length; it is SPLICED OUT of the chain (no zero-length curl) and the slope survives with
        the remaining segment. Undo restores the original two-segment chain verbatim.
        """
        dem = mock_dem_blue_slope
        graph = empty_graph
        seg_ids = _commit_L_slope(graph, dem)
        slope = graph.finish_slope(segment_ids=seg_ids)
        first_seg = graph.segments[slope.segment_ids[0]]
        original_chain = list(slope.segment_ids)
        # Merge the first segment's own endpoints (path start + interior junction).
        graph.merge_nodes(node_ids=[first_seg.start_node_id, first_seg.end_node_id], dem=dem)

        assert slope.id in graph.slopes, "a partially-collapsed multi-segment path is not deleted"
        assert first_seg.id not in graph.slopes[slope.id].segment_ids, "the collapsed segment is spliced out"
        assert len(graph.slopes[slope.id].segment_ids) == len(original_chain) - 1, "one segment removed"
        assert all(
            graph.segments[sid].start_node_id != graph.segments[sid].end_node_id
            for sid in graph.slopes[slope.id].segment_ids
        ), "no zero-length segment remains in the chain"
        assert slope.start_node_id != slope.end_node_id, "path boundary stays distinct"

        graph.undo_last()
        assert graph.slopes[slope.id].segment_ids == original_chain, "undo restores the original chain"


# =============================================================================
# Node delete / insert (merge-mode editing tools)
# =============================================================================


def _commit_straight_slope(graph, dem, n_segments: int):
    """Commit n straight due-south descending segments into one finished slope; return the slope.

    Each segment is its own commit so nodes materialise at every junction (interior nodes exist).
    Segments are ~480m so junctions sit well beyond the node-snap distance.
    """
    lat = 0.0
    for _ in range(n_segments):
        leg = _leg(0.0, lat, 0.0, -20.0, 25, dem)  # ~480m south per segment
        graph.commit_paths(paths=[ProposedPathSegment(points=leg, target_difficulty="blue")])
        lat = leg[-1].lat
    return graph.finish_slope(segment_ids=list(graph.segments.keys()))


class TestNodeDeletability:
    """node_deletability classifies why a node can/can't be deleted (single source for UI + op)."""

    def test_interior_node_is_deletable_interior(self, empty_graph, mock_dem_blue_slope) -> None:
        slope = _commit_straight_slope(empty_graph, mock_dem_blue_slope, n_segments=3)
        interior = empty_graph.segments[slope.segment_ids[0]].end_node_id  # junction of seg0/seg1
        assert empty_graph.node_deletability(interior) == NodeDeletability.DELETABLE_INTERIOR

    def test_clean_endpoint_of_multisegment_path_is_deletable_end(self, empty_graph, mock_dem_blue_slope) -> None:
        slope = _commit_straight_slope(empty_graph, mock_dem_blue_slope, n_segments=3)
        assert empty_graph.node_deletability(slope.start_node_id) == NodeDeletability.DELETABLE_END
        assert empty_graph.node_deletability(slope.end_node_id) == NodeDeletability.DELETABLE_END

    def test_endpoint_of_single_segment_path_is_last_segment(self, empty_graph, mock_dem_blue_slope) -> None:
        slope = _commit_straight_slope(empty_graph, mock_dem_blue_slope, n_segments=1)
        assert empty_graph.node_deletability(slope.start_node_id) == NodeDeletability.LAST_SEGMENT

    def test_lift_station_is_never_deletable(self, empty_graph, mock_dem_blue_slope) -> None:
        dem = mock_dem_blue_slope
        graph = empty_graph
        graph.nodes["A"] = Node(id="A", location=PathPoint(lon=0.0, lat=0.0, elevation=2000.0))
        graph.nodes["T"] = Node(id="T", location=PathPoint(lon=0.0, lat=-1000 / M, elevation=2400.0))
        graph.add_lift(start_node_id="A", end_node_id="T", lift_type="chairlift", dem=dem)
        assert graph.node_deletability("A") == NodeDeletability.IS_LIFT_STATION

    def test_branch_node_shared_by_two_paths_is_path_endpoint(self, empty_graph, mock_dem_blue_slope) -> None:
        """A node that is the boundary of one slope AND interior/boundary of another is a junction —
        not deletable; the user must delete a path first.
        """
        dem = mock_dem_blue_slope
        graph = empty_graph
        # Slope 1: due south to a junction node.
        leg1 = _leg(0.0, 0.0, 0.0, -20.0, 25, dem)
        graph.commit_paths(paths=[ProposedPathSegment(points=leg1, target_difficulty="blue")])
        slope1 = graph.finish_slope(segment_ids=list(graph.segments.keys()))
        junction = slope1.end_node_id
        # Slope 2 starts at that same junction and heads south-east (reuses the node via snap).
        j = graph.nodes[junction]
        leg2 = _leg(j.lon, j.lat, 20.0, -20.0, 25, dem)
        seg_before = set(graph.segments)
        graph.commit_paths(paths=[ProposedPathSegment(points=leg2, target_difficulty="blue")])
        new_seg = (set(graph.segments) - seg_before).pop()
        graph.finish_slope(segment_ids=[new_seg])
        assert graph.node_deletability(junction) == NodeDeletability.IS_PATH_ENDPOINT


class TestDeleteNodes:
    """delete_nodes fuses interior nodes / trims clean endpoints, as ONE undoable action."""

    def test_delete_interior_fuses_two_segments_into_one(self, empty_graph, mock_dem_blue_slope) -> None:
        slope = _commit_straight_slope(empty_graph, mock_dem_blue_slope, n_segments=3)
        interior = empty_graph.segments[slope.segment_ids[0]].end_node_id
        length_before = slope.get_total_length(empty_graph.segments)

        empty_graph.delete_nodes(node_ids=[interior], dem=mock_dem_blue_slope)

        assert interior not in empty_graph.nodes, "the interior node is gone"
        assert len(empty_graph.slopes[slope.id].segment_ids) == 2, "3 segments fused to 2"
        length_after = empty_graph.slopes[slope.id].get_total_length(empty_graph.segments)
        assert length_after == pytest.approx(length_before, rel=0.01), "path length preserved by the fuse"

    def test_delete_two_adjacent_interior_collapses_three_to_one(self, empty_graph, mock_dem_blue_slope) -> None:
        slope = _commit_straight_slope(empty_graph, mock_dem_blue_slope, n_segments=3)
        n0 = empty_graph.segments[slope.segment_ids[0]].end_node_id
        n1 = empty_graph.segments[slope.segment_ids[1]].end_node_id

        empty_graph.delete_nodes(node_ids=[n0, n1], dem=mock_dem_blue_slope)

        assert len(empty_graph.slopes[slope.id].segment_ids) == 1, "both interior nodes fused → 1 segment"
        assert n0 not in empty_graph.nodes and n1 not in empty_graph.nodes

    def test_delete_clean_endpoint_trims_boundary_segment(self, empty_graph, mock_dem_blue_slope) -> None:
        slope = _commit_straight_slope(empty_graph, mock_dem_blue_slope, n_segments=3)
        old_start = slope.start_node_id
        new_start_expected = empty_graph.segments[slope.segment_ids[1]].start_node_id

        empty_graph.delete_nodes(node_ids=[old_start], dem=mock_dem_blue_slope)

        assert old_start not in empty_graph.nodes, "the trimmed end node is freed"
        assert len(empty_graph.slopes[slope.id].segment_ids) == 2, "the boundary segment is trimmed"
        assert empty_graph.slopes[slope.id].start_node_id == new_start_expected, "boundary re-pointed"

    def test_delete_end_node_and_adjacent_node_leaves_no_dangling_boundary(
        self, empty_graph, mock_dem_blue_slope
    ) -> None:
        """Regression: deleting an end node AND its neighbour together must trim + fuse in one pass —
        the old two-phase logic left a segment pointing at the (also-deleted) neighbour node.
        """
        slope = _commit_straight_slope(empty_graph, mock_dem_blue_slope, n_segments=4)
        end = slope.start_node_id
        adjacent = empty_graph.segments[slope.segment_ids[0]].end_node_id  # neighbour of the end node

        empty_graph.delete_nodes(node_ids=[end, adjacent], dem=mock_dem_blue_slope)

        surviving = graph_slope = empty_graph.slopes[slope.id]
        assert end not in empty_graph.nodes and adjacent not in empty_graph.nodes, "both nodes freed"
        # Every surviving segment must reference nodes that still exist (no dangling boundary).
        for sid in surviving.segment_ids:
            seg = empty_graph.segments[sid]
            assert seg.start_node_id in empty_graph.nodes, f"{sid} start node missing"
            assert seg.end_node_id in empty_graph.nodes, f"{sid} end node missing"
        assert graph_slope.start_node_id in empty_graph.nodes, "path start node exists"
        assert graph_slope.end_node_id == empty_graph.segments[surviving.segment_ids[-1]].end_node_id

    def test_delete_records_single_undo_and_restores_verbatim(self, empty_graph, mock_dem_blue_slope) -> None:
        slope = _commit_straight_slope(empty_graph, mock_dem_blue_slope, n_segments=3)
        interior = empty_graph.segments[slope.segment_ids[0]].end_node_id
        chain_before = list(slope.segment_ids)

        empty_graph.delete_nodes(node_ids=[interior], dem=mock_dem_blue_slope)
        assert empty_graph.undo_stack[-1].action_type.name == "DELETE_NODES", "delete pushes one DELETE_NODES entry"

        empty_graph.undo_last()
        assert interior in empty_graph.nodes, "undo restores the deleted node"
        assert empty_graph.slopes[slope.id].segment_ids == chain_before, "undo restores the chain verbatim"

    def test_delete_non_deletable_raises(self, empty_graph, mock_dem_blue_slope) -> None:
        slope = _commit_straight_slope(empty_graph, mock_dem_blue_slope, n_segments=1)
        # A single-segment path's endpoint is LAST_SEGMENT — delete_nodes_rejection refuses it.
        with pytest.raises(ValueError, match="delete the path instead"):
            empty_graph.delete_nodes(node_ids=[slope.start_node_id], dem=mock_dem_blue_slope)

    def test_delete_never_orphans_a_segment_after_cross_path_merge(self, empty_graph, mock_dem_blue_slope) -> None:
        """Regression: a merge can make a node a junction shared by a second path. Deleting nodes on
        one path must never delete a node the other path's segment still references (crash: KeyError
        on a segment endpoint). Only truly-unreferenced nodes are removed.
        """
        dem = mock_dem_blue_slope
        graph = empty_graph
        s1 = _commit_straight_slope(graph, dem, n_segments=2)
        s1_interior = graph.segments[s1.segment_ids[0]].end_node_id
        # A second 2-segment slope, then move its interior node coincident with s1's interior node so
        # the cross-path merge is unconditionally accepted (independent of finish-smoothing jitter).
        before = set(graph.segments)
        lat = 0.0
        for _ in range(2):
            leg = _leg(80 / M, lat, 0.0, -20.0, 25, dem)
            graph.commit_paths(paths=[ProposedPathSegment(points=leg, target_difficulty="blue")])
            lat = leg[-1].lat
        s2 = graph.finish_slope(segment_ids=list(set(graph.segments) - before))
        s2_interior = graph.segments[s2.segment_ids[0]].end_node_id
        graph.nodes[s2_interior].location = graph.nodes[s1_interior].location  # make them coincident

        # Merge s2's interior node onto s1's interior node → a shared junction.
        graph.merge_nodes(node_ids=[s1_interior, s2_interior], dem=dem)

        # Deleting any still-deletable node must leave the graph referentially intact (no segment
        # pointing at a removed node) — whether the delete proceeds or is refused.
        for nid in list(graph.nodes):
            if graph.node_deletability(nid) in (NodeDeletability.DELETABLE_INTERIOR, NodeDeletability.DELETABLE_END):
                graph.delete_nodes(node_ids=[nid], dem=dem)
                break
        for seg in graph.segments.values():
            assert seg.start_node_id in graph.nodes, f"orphan: {seg.id} start {seg.start_node_id}"
            assert seg.end_node_id in graph.nodes, f"orphan: {seg.id} end {seg.end_node_id}"


class TestInsertNodeOnPath:
    """insert_node_on_path splits a segment at the clicked point, as ONE undoable action."""

    def test_insert_splits_segment_and_updates_chain(self, empty_graph, mock_dem_blue_slope) -> None:
        slope = _commit_straight_slope(empty_graph, mock_dem_blue_slope, n_segments=1)
        seg_id = slope.segment_ids[0]
        seg = empty_graph.segments[seg_id]
        mid = seg.points[len(seg.points) // 2]

        node_id = empty_graph.insert_node_on_path(segment_id=seg_id, lon=mid.lon, lat=mid.lat)

        assert node_id in empty_graph.nodes, "a new node was created"
        assert seg_id not in empty_graph.segments, "the original segment is replaced"
        assert len(empty_graph.slopes[slope.id].segment_ids) == 2, "[seg] → [A', B']"
        a, b = empty_graph.slopes[slope.id].segment_ids
        assert empty_graph.segments[a].end_node_id == node_id
        assert empty_graph.segments[b].start_node_id == node_id
        # The node snapped to an existing vertex, so it sits exactly on the drawn path.
        assert empty_graph.nodes[node_id].location == mid

    def test_insert_too_close_to_endpoint_raises(self, empty_graph, mock_dem_blue_slope) -> None:
        slope = _commit_straight_slope(empty_graph, mock_dem_blue_slope, n_segments=1)
        seg = empty_graph.segments[slope.segment_ids[0]]
        near_start = seg.points[0]
        with pytest.raises(ValueError, match="too close to an existing node"):
            empty_graph.insert_node_on_path(segment_id=seg.id, lon=near_start.lon, lat=near_start.lat)

    def test_insert_records_single_undo_and_restores_verbatim(self, empty_graph, mock_dem_blue_slope) -> None:
        slope = _commit_straight_slope(empty_graph, mock_dem_blue_slope, n_segments=1)
        seg_id = slope.segment_ids[0]
        mid = empty_graph.segments[seg_id].points[len(empty_graph.segments[seg_id].points) // 2]
        chain_before = list(slope.segment_ids)

        node_id = empty_graph.insert_node_on_path(segment_id=seg_id, lon=mid.lon, lat=mid.lat)
        assert empty_graph.undo_stack[-1].action_type.name == "INSERT_NODE", "insert pushes one INSERT_NODE entry"

        empty_graph.undo_last()
        assert node_id not in empty_graph.nodes, "undo removes the inserted node"
        assert seg_id in empty_graph.segments, "undo restores the original segment"
        assert empty_graph.slopes[slope.id].segment_ids == chain_before, "undo restores the chain verbatim"


class TestNodeEditHelpers:
    """Direct unit tests for the fragile private chain-surgery helpers, isolated from the graph ops
    that call them (delete_nodes / merge_nodes), so a break is pinpointed to the helper.
    """

    def _seg(self, sid: str, start: str, end: str, pts: list[tuple[float, float]]) -> PathSegment:
        points = [PathPoint(lon=x, lat=y, elevation=100.0) for x, y in pts]
        return PathSegment(id=sid, name=sid, points=points, start_node_id=start, end_node_id=end)

    # -- _chain_node_sequence: pure, the shared node-walk primitive -----------------------------

    def test_chain_node_sequence_single_segment(self) -> None:
        seg = self._seg("S1", "N1", "N2", [(0.0, 0.0), (1.0, 0.0)])
        assert _chain_node_sequence([seg]) == ["N1", "N2"]

    def test_chain_node_sequence_walks_the_whole_chain(self) -> None:
        segs = [
            self._seg("S1", "N1", "N2", [(0.0, 0.0), (1.0, 0.0)]),
            self._seg("S2", "N2", "N3", [(1.0, 0.0), (2.0, 0.0)]),
            self._seg("S3", "N3", "N4", [(2.0, 0.0), (3.0, 0.0)]),
        ]
        assert _chain_node_sequence(segs) == ["N1", "N2", "N3", "N4"]

    # -- _segments_touching / _owning_path: graph queries -----------------------------------------

    def test_segments_touching_finds_both_incident_segments(self, empty_graph, mock_dem_blue_slope) -> None:
        slope = _commit_straight_slope(empty_graph, mock_dem_blue_slope, n_segments=3)
        junction = empty_graph.segments[slope.segment_ids[0]].end_node_id
        touching = empty_graph._segments_touching(junction)
        assert {s.id for s in touching} == {slope.segment_ids[0], slope.segment_ids[1]}

    def test_segments_touching_empty_for_unknown_node(self, empty_graph) -> None:
        assert empty_graph._segments_touching("GHOST") == []

    def test_owning_path_returns_finished_owner(self, empty_graph, mock_dem_blue_slope) -> None:
        slope = _commit_straight_slope(empty_graph, mock_dem_blue_slope, n_segments=2)
        interior = empty_graph.segments[slope.segment_ids[0]].end_node_id
        assert empty_graph._owning_path(interior).id == slope.id

    def test_owning_path_none_for_unattached_node(self, empty_graph) -> None:
        assert empty_graph._owning_path("GHOST") is None

    # -- _nearest_vertex_index: geometry pick -----------------------------------------------------

    def test_nearest_vertex_index_picks_closest(self) -> None:
        graph = ResortGraph()
        seg = self._seg("S1", "N1", "N2", [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)])
        assert graph._nearest_vertex_index(seg, lon=0.9, lat=0.0) == 1  # nearest to the middle vertex
        assert graph._nearest_vertex_index(seg, lon=-5.0, lat=0.0) == 0  # off the start end
        assert graph._nearest_vertex_index(seg, lon=99.0, lat=0.0) == 2  # off the far end

    # -- _drop_collapsed_segments_in_chain: zero-length curl removal ------------------------------

    def test_drop_collapsed_removes_only_the_curl_and_rederives_boundaries(self, empty_graph) -> None:
        graph = empty_graph
        graph.nodes["N1"] = Node(id="N1", location=PathPoint(lon=0.0, lat=0.0, elevation=100.0))
        graph.nodes["S"] = Node(id="S", location=PathPoint(lon=1.0, lat=0.0, elevation=100.0))
        graph.nodes["N4"] = Node(id="N4", location=PathPoint(lon=2.0, lat=0.0, elevation=100.0))
        graph.segments["S1"] = self._seg("S1", "N1", "S", [(0.0, 0.0), (1.0, 0.0)])
        graph.segments["S2"] = self._seg("S2", "S", "S", [(1.0, 0.0), (1.0, 0.0)])  # zero-length curl
        graph.segments["S3"] = self._seg("S3", "S", "N4", [(1.0, 0.0), (2.0, 0.0)])
        slope = Slope(id="SL1", name="t", segment_ids=["S1", "S2", "S3"], start_node_id="N1", end_node_id="N4")
        graph.slopes["SL1"] = slope

        dropped = graph._drop_collapsed_segments_in_chain(slope)

        assert dropped == ["S2"], "only the zero-length segment is dropped"
        assert slope.segment_ids == ["S1", "S3"], "curl spliced out, neighbours kept"
        assert "S2" not in graph.segments, "the curl segment is removed from the graph"
        assert (slope.start_node_id, slope.end_node_id) == ("N1", "N4"), "boundaries preserved"

    def test_drop_collapsed_noop_when_no_curl(self, empty_graph, mock_dem_blue_slope) -> None:
        slope = _commit_straight_slope(empty_graph, mock_dem_blue_slope, n_segments=2)
        chain_before = list(slope.segment_ids)
        assert empty_graph._drop_collapsed_segments_in_chain(slope) == [], "no curl → nothing dropped"
        assert slope.segment_ids == chain_before, "chain unchanged when there is no curl"

    # -- _rebuild_chain_without_nodes: the fragile trim + fuse node-walk --------------------------

    def test_rebuild_fuses_a_single_interior_node(self, empty_graph, mock_dem_blue_slope) -> None:
        slope = _commit_straight_slope(empty_graph, mock_dem_blue_slope, n_segments=3)
        interior = empty_graph.segments[slope.segment_ids[0]].end_node_id
        empty_graph._rebuild_chain_without_nodes(path=slope, drop_nodes={interior}, dem=mock_dem_blue_slope)
        assert len(slope.segment_ids) == 2, "the two segments around the node fused into one"
        assert all(
            interior not in (empty_graph.segments[s].start_node_id, empty_graph.segments[s].end_node_id)
            for s in slope.segment_ids
        ), "the dropped node appears in no surviving segment"

    def test_rebuild_trims_a_boundary_node(self, empty_graph, mock_dem_blue_slope) -> None:
        slope = _commit_straight_slope(empty_graph, mock_dem_blue_slope, n_segments=3)
        old_start = slope.start_node_id
        new_start_expected = empty_graph.segments[slope.segment_ids[1]].start_node_id
        empty_graph._rebuild_chain_without_nodes(path=slope, drop_nodes={old_start}, dem=mock_dem_blue_slope)
        assert len(slope.segment_ids) == 2, "the lone boundary segment is trimmed"
        assert slope.start_node_id == new_start_expected, "boundary re-derived from the surviving chain"

    def test_rebuild_end_plus_adjacent_leaves_no_dangling_node(self, empty_graph, mock_dem_blue_slope) -> None:
        # The regression the node-walk fixed: trimming an end AND fusing its neighbour in one pass.
        slope = _commit_straight_slope(empty_graph, mock_dem_blue_slope, n_segments=4)
        end = slope.start_node_id
        adjacent = empty_graph.segments[slope.segment_ids[0]].end_node_id
        empty_graph._rebuild_chain_without_nodes(path=slope, drop_nodes={end, adjacent}, dem=mock_dem_blue_slope)
        for sid in slope.segment_ids:
            seg = empty_graph.segments[sid]
            assert seg.start_node_id in empty_graph.nodes and seg.end_node_id in empty_graph.nodes
        assert end not in _chain_node_sequence([empty_graph.segments[s] for s in slope.segment_ids])
        assert adjacent not in _chain_node_sequence([empty_graph.segments[s] for s in slope.segment_ids])


# =============================================================================
# Stats: get_segment_stats (running stats) + get_stats (whole-resort summary)
# =============================================================================


class TestSegmentStats:
    """get_segment_stats aggregates a set of committed segments into running build stats."""

    def test_two_segment_run_reports_computed_metrics(self, empty_graph, mock_dem_blue_slope) -> None:
        graph = empty_graph
        seg_ids = _commit_L_slope(graph, mock_dem_blue_slope)
        first_seg, last_seg = graph.segments[seg_ids[0]], graph.segments[seg_ids[-1]]
        exp_length = sum(graph.segments[sid].length_m for sid in seg_ids)
        exp_start_elev = first_seg.start.elevation
        exp_current_elev = last_seg.end.elevation
        exp_drop = exp_start_elev - exp_current_elev
        exp_max = max(graph.segments[sid].max_slope_pct for sid in seg_ids)

        stats = graph.get_segment_stats(segment_ids=seg_ids)

        assert stats["total_length"] == pytest.approx(exp_length)
        assert stats["start_elev"] == pytest.approx(exp_start_elev)
        assert stats["current_elev"] == pytest.approx(exp_current_elev)
        assert stats["total_drop"] == pytest.approx(exp_drop)
        assert stats["avg_gradient"] == pytest.approx(exp_drop / exp_length * 100)
        assert stats["max_gradient"] == pytest.approx(exp_max)
        # Difficulty is derived from the reported steepest section, not some other slope value.
        assert stats["difficulty"] == TerrainAnalyzer.classify_difficulty(slope_pct=stats["max_gradient"])

    def test_empty_segment_ids_returns_default_stats(self, empty_graph) -> None:
        stats = empty_graph.get_segment_stats(segment_ids=[])
        assert stats == {
            "total_drop": 0.0,
            "total_length": 0.0,
            "avg_gradient": 0.0,
            "max_gradient": 0.0,
            "difficulty": "green",
            "start_elev": 0.0,
            "current_elev": 0.0,
        }

    def test_missing_segment_id_raises(self, empty_graph) -> None:
        # A non-empty segment_ids list must reference real segments (internal build state);
        # a missing id is a bug and must fail loud, not silently return default stats.
        with pytest.raises(KeyError):
            empty_graph.get_segment_stats(segment_ids=["S_missing"])


class TestResortStats:
    """get_stats summarises the whole resort (slope/segment/lift/road counts + totals)."""

    def test_empty_graph_reports_all_zero(self, empty_graph) -> None:
        assert empty_graph.get_stats() == {
            "total_slopes": 0,
            "total_segments": 0,
            "total_vertical_m": 0,
            "total_length_m": 0,
            "longest_run_m": 0,
            "total_lifts": 0,
            "total_roads": 0,
            "total_road_length_m": 0,
        }

    def test_multi_segment_slope_reports_slope_totals(self, empty_graph, mock_dem_blue_slope) -> None:
        graph = empty_graph
        seg_ids = _commit_L_slope(graph, mock_dem_blue_slope)
        graph.finish_slope(segment_ids=seg_ids)
        slope = graph.slopes[next(iter(graph.slopes))]
        exp_vertical = sum(graph.segments[sid].total_drop_m for sid in seg_ids)
        exp_longest = slope.get_total_length(segments=graph.segments)
        max_single_seg = max(graph.segments[sid].length_m for sid in seg_ids)

        stats = graph.get_stats()

        assert stats["total_slopes"] == 1
        assert stats["total_segments"] == len(seg_ids)
        assert stats["total_vertical_m"] == pytest.approx(exp_vertical)
        assert stats["longest_run_m"] == pytest.approx(exp_longest)
        assert stats["longest_run_m"] > max_single_seg, "whole run is longer than any single segment"

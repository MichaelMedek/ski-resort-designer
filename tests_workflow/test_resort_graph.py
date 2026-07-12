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
from skiresort_planner.model.node import Node
from skiresort_planner.model.path_point import PathPoint
from skiresort_planner.model.path_segment import SegmentKind
from skiresort_planner.model.proposed_path import ProposedPathSegment
from skiresort_planner.model.resort_graph import (
    AddLiftAction,
    AddSegmentsAction,
    DeleteLiftAction,
    DeleteRoadAction,
    DeleteSlopeAction,
    FinishRoadAction,
    FinishSlopeAction,
    ResortGraph,
)

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

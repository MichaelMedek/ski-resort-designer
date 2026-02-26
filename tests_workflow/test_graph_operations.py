"""Unit tests for ResortGraph operations.

Tests commit_paths, finish_slope, add_lift, and undo operations.
"""

import pytest

from skiresort_planner.constants import MapConfig
from skiresort_planner.model.path_point import PathPoint
from skiresort_planner.model.proposed_path import ProposedSlopeSegment
from skiresort_planner.model.resort_graph import (
    AddSegmentsAction,
    FinishSlopeAction,
)


class TestCommitAndFinishWorkflow:
    """Tests for commit_paths and finish_slope operations."""

    def test_commit_paths_creates_nodes_and_segments(self, empty_graph, path_points_blue) -> None:
        """commit_paths creates nodes at endpoints and segments between them.

        Tests:
        - Nodes created for start and end points
        - Segment created connecting nodes
        - Undo action pushed to stack
        """
        graph = empty_graph
        proposal = ProposedSlopeSegment(
            points=path_points_blue,
            target_slope_pct=20.0,
            target_difficulty="blue",
            sector_name="Test",
        )

        endpoint_ids = graph.commit_paths(paths=[proposal])

        assert len(graph.nodes) == 2, "Should create 2 nodes (start and end)"
        assert len(graph.segments) == 1, "Should create 1 segment"
        assert len(endpoint_ids) == 1, "Should return 1 endpoint ID"
        assert len(graph.undo_stack) == 1, "Should push undo action"
        assert isinstance(graph.undo_stack[0], AddSegmentsAction), "Undo action should be AddSegmentsAction"

    def test_finish_slope_groups_segments(self, empty_graph, path_points_blue) -> None:
        """finish_slope groups committed segments into a named slope.

        Tests:
        - Slope created with segment IDs
        - Slope has generated name
        - Undo action pushed for finish
        """
        graph = empty_graph
        proposal = ProposedSlopeSegment(
            points=path_points_blue,
            target_slope_pct=20.0,
            target_difficulty="blue",
            sector_name="Test",
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
        """Paths with nearby endpoints should share nodes.

        Tests:
        - First path creates 2 nodes
        - Second path starting near first path's end reuses node
        """
        graph = empty_graph
        dem = mock_dem_blue_slope
        M = MapConfig.METERS_PER_DEGREE_EQUATOR

        # First path: from origin going south
        path1_points = [
            PathPoint(lon=0.0, lat=0.0, elevation=dem.get_elevation_or_raise(lon=0.0, lat=0.0)),
            PathPoint(lon=0.0, lat=-500 / M, elevation=dem.get_elevation_or_raise(lon=0.0, lat=-500 / M)),
        ]
        proposal1 = ProposedSlopeSegment(
            points=path1_points, target_slope_pct=20.0, target_difficulty="blue", sector_name="P1"
        )

        graph.commit_paths(paths=[proposal1])
        assert len(graph.nodes) == 2, "First path creates 2 nodes"

        # Second path: starting very close to first path's end (should reuse node)
        path2_points = [
            PathPoint(
                lon=0.00001,  # Very close to end of path1
                lat=-500 / M,
                elevation=dem.get_elevation_or_raise(lon=0.00001, lat=-500 / M),
            ),
            PathPoint(lon=0.0, lat=-1000 / M, elevation=dem.get_elevation_or_raise(lon=0.0, lat=-1000 / M)),
        ]
        proposal2 = ProposedSlopeSegment(
            points=path2_points, target_slope_pct=20.0, target_difficulty="blue", sector_name="P2"
        )

        graph.commit_paths(paths=[proposal2])
        assert len(graph.nodes) == 3, "Second path should reuse 1 node, create 1 new"


class TestConnectorGeometrySnapping:
    """Tests for connector path geometry snapping to target nodes."""

    def test_connector_path_snaps_to_target_node_coordinates(self, empty_graph, mock_dem_blue_slope) -> None:
        """Connector path endpoint is snapped to exact target node coordinates.

        When a path has target_node_id set (connector), the path's last point
        should be snapped to the target node's exact coordinates to avoid
        visual kinks in 3D rendering and topological inconsistencies.
        """
        graph = empty_graph
        dem = mock_dem_blue_slope
        M = MapConfig.METERS_PER_DEGREE_EQUATOR

        # Create first path to establish target node
        path1_points = [
            PathPoint(lon=0.0, lat=0.0, elevation=dem.get_elevation_or_raise(lon=0.0, lat=0.0)),
            PathPoint(lon=0.0, lat=-500 / M, elevation=dem.get_elevation_or_raise(lon=0.0, lat=-500 / M)),
        ]
        graph.commit_paths(paths=[ProposedSlopeSegment(points=path1_points, sector_name="P1")])

        # Get end node as target
        first_segment = list(graph.segments.values())[0]
        target_node_id = first_segment.end_node_id
        target_node = graph.nodes[target_node_id]

        # Create connector path with slightly offset end coordinates
        offset_lon = target_node.lon + 0.00005  # ~5m offset
        offset_lat = target_node.lat + 0.00003
        offset_elev = target_node.elevation + 2.0

        connector_points = [
            PathPoint(lon=0.001, lat=0.0, elevation=dem.get_elevation_or_raise(lon=0.001, lat=0.0)),
            PathPoint(lon=offset_lon, lat=offset_lat, elevation=offset_elev),  # Slightly off from target
        ]
        connector = ProposedSlopeSegment(
            points=connector_points,
            sector_name="Connector",
            is_connector=True,
            target_node_id=target_node_id,
        )

        # Commit connector
        graph.commit_paths(paths=[connector])

        # Verify: path endpoint was snapped to target node's exact coordinates
        committed_segment = list(graph.segments.values())[-1]
        snapped_point = committed_segment.points[-1]

        assert snapped_point.lon == target_node.lon, "Path end lon should match target node"
        assert snapped_point.lat == target_node.lat, "Path end lat should match target node"
        assert snapped_point.elevation == target_node.elevation, "Path end elevation should match target node"

        # Verify: no duplicate node created
        assert len(graph.nodes) == 3, "Should have 3 nodes (2 from first + 1 new start), not 4"
        assert committed_segment.end_node_id == target_node_id, "Segment should connect to target node"

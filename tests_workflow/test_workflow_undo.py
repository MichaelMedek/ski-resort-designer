"""Integration test for comprehensive undo operations.

Tests undo across all action types with intermediate assertions.
"""

import pytest

from skiresort_planner.constants import MapConfig
from skiresort_planner.model.path_point import PathPoint
from skiresort_planner.model.proposed_path import ProposedSlopeSegment
from skiresort_planner.model.resort_graph import (
    AddSegmentsAction,
    DeleteSlopeAction,
    FinishSlopeAction,
)


class TestUndoWorkflow:
    """Comprehensive tests for undo operations across all action types."""

    def test_undo_single_segment_removes_segment_and_nodes(self, empty_graph, path_points_blue) -> None:
        """Undo AddSegmentsAction removes segment and cleans up isolated nodes."""
        graph = empty_graph
        proposal = ProposedSlopeSegment(
            points=path_points_blue,
            target_slope_pct=20.0,
            target_difficulty="blue",
            sector_name="Test",
        )

        graph.commit_paths(paths=[proposal])

        assert len(graph.segments) == 1, "Should have 1 segment"
        assert len(graph.nodes) == 2, "Should have 2 nodes"
        assert len(graph.undo_stack) == 1

        undone = graph.undo_last()

        assert isinstance(undone, AddSegmentsAction)
        assert len(graph.segments) == 0, "Segment should be removed"
        assert len(graph.undo_stack) == 0, "Stack should be empty"

    def test_undo_preserves_other_segments(self, empty_graph, mock_dem_blue_slope) -> None:
        """Undo removes only the undone segment, preserving others."""
        graph = empty_graph
        dem = mock_dem_blue_slope
        M = MapConfig.METERS_PER_DEGREE_EQUATOR

        # First segment: 0 to -500m south
        points1 = [
            PathPoint(lon=0.0, lat=0.0, elevation=dem.get_elevation_or_raise(lon=0.0, lat=0.0)),
            PathPoint(lon=0.0, lat=-500 / M, elevation=dem.get_elevation_or_raise(lon=0.0, lat=-500 / M)),
        ]
        graph.commit_paths(
            paths=[
                ProposedSlopeSegment(points=points1, target_slope_pct=20.0, target_difficulty="blue", sector_name="P1")
            ]
        )

        # Second segment: -500 to -1000m south
        points2 = [
            PathPoint(lon=0.0, lat=-500 / M, elevation=dem.get_elevation_or_raise(lon=0.0, lat=-500 / M)),
            PathPoint(lon=0.0, lat=-1000 / M, elevation=dem.get_elevation_or_raise(lon=0.0, lat=-1000 / M)),
        ]
        graph.commit_paths(
            paths=[
                ProposedSlopeSegment(points=points2, target_slope_pct=20.0, target_difficulty="blue", sector_name="P2")
            ]
        )

        assert len(graph.segments) == 2
        seg_ids = list(graph.segments.keys())

        # Undo second segment
        graph.undo_last()

        assert len(graph.segments) == 1, "Should have 1 segment remaining"
        assert seg_ids[0] in graph.segments, "First segment should remain"
        assert seg_ids[1] not in graph.segments, "Second segment should be removed"

    def test_undo_finish_slope_keeps_segments(self, empty_graph, path_points_blue) -> None:
        """Undo FinishSlopeAction removes slope but preserves segments."""
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
        slope_name = slope.name

        assert len(graph.slopes) == 1
        assert len(graph.segments) == 1

        undone = graph.undo_last()

        assert isinstance(undone, FinishSlopeAction)
        assert undone.slope_name == slope_name
        assert len(graph.slopes) == 0, "Slope should be removed"
        assert len(graph.segments) == 1, "Segment should remain"

    def test_undo_delete_slope_restores_slope(self, empty_graph, path_points_blue) -> None:
        """Undo DeleteSlopeAction restores the slope and its segments."""
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
        slope_id = slope.id
        slope_name = slope.name

        # Delete slope
        graph.delete_slope(slope_id=slope_id)

        assert len(graph.slopes) == 0

        # Undo delete
        undone = graph.undo_last()

        assert isinstance(undone, DeleteSlopeAction)
        assert len(graph.slopes) == 1, "Slope should be restored"
        assert graph.slopes[slope_id].name == slope_name, "Name should match"

    def test_multiple_consecutive_undos(self, empty_graph, mock_dem_blue_slope) -> None:
        """Multiple consecutive undos work correctly in sequence."""
        graph = empty_graph
        dem = mock_dem_blue_slope
        M = MapConfig.METERS_PER_DEGREE_EQUATOR

        # Create 3 segments
        for i in range(3):
            start_lat = -i * 500 / M
            end_lat = -(i + 1) * 500 / M
            points = [
                PathPoint(lon=0.0, lat=start_lat, elevation=dem.get_elevation_or_raise(lon=0.0, lat=start_lat)),
                PathPoint(lon=0.0, lat=end_lat, elevation=dem.get_elevation_or_raise(lon=0.0, lat=end_lat)),
            ]
            graph.commit_paths(
                paths=[
                    ProposedSlopeSegment(
                        points=points, target_slope_pct=20.0, target_difficulty="blue", sector_name=f"P{i}"
                    )
                ]
            )

        assert len(graph.segments) == 3
        assert len(graph.undo_stack) == 3

        # Undo all three
        graph.undo_last()
        assert len(graph.segments) == 2

        graph.undo_last()
        assert len(graph.segments) == 1

        graph.undo_last()
        assert len(graph.segments) == 0
        assert len(graph.undo_stack) == 0

    def test_empty_undo_stack_raises_runtime_error(self, empty_graph) -> None:
        """undo_last on empty stack raises RuntimeError."""
        with pytest.raises(RuntimeError, match="empty"):
            empty_graph.undo_last()


class TestUndoStackSizeLimit:
    """Tests for undo stack size limiting."""

    def test_undo_stack_has_max_size(self, empty_graph, mock_dem_blue_slope) -> None:
        """Undo stack enforces maximum size, discarding oldest actions."""
        from skiresort_planner.constants import UndoConfig

        graph = empty_graph
        dem = mock_dem_blue_slope
        M = MapConfig.METERS_PER_DEGREE_EQUATOR

        # Push more actions than max size
        for i in range(UndoConfig.MAX_UNDO_STACK_SIZE + 5):
            start_lat = -i * 50 / M
            end_lat = -(i + 1) * 50 / M
            points = [
                PathPoint(lon=0.0, lat=start_lat, elevation=dem.get_elevation_or_raise(lon=0.0, lat=start_lat)),
                PathPoint(lon=0.0, lat=end_lat, elevation=dem.get_elevation_or_raise(lon=0.0, lat=end_lat)),
            ]
            graph.commit_paths(
                paths=[
                    ProposedSlopeSegment(
                        points=points, target_slope_pct=20.0, target_difficulty="blue", sector_name=f"P{i}"
                    )
                ]
            )

        # Stack should be capped at max size
        assert len(graph.undo_stack) <= UndoConfig.MAX_UNDO_STACK_SIZE, (
            f"Stack should not exceed {UndoConfig.MAX_UNDO_STACK_SIZE}"
        )

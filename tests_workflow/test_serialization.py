"""Integration tests for serialization (save/load roundtrip).

Tests that resort graphs can be saved and loaded without data loss.
"""

import json
import tempfile
from pathlib import Path


class TestResortGraphSerialization:
    """Tests for ResortGraph save/load operations."""

    def test_to_dict_and_from_dict_roundtrip(self, empty_graph, path_points_blue, mock_dem_blue_slope) -> None:
        """ResortGraph can be serialized to dict and restored.

        Tests:
        - All nodes preserved
        - All segments preserved with points
        - All slopes preserved with segment references
        - Counters restored correctly
        """
        from skiresort_planner.model.proposed_path import ProposedSlopeSegment
        from skiresort_planner.model.resort_graph import ResortGraph

        graph = empty_graph

        # Create some data
        proposal = ProposedSlopeSegment(
            points=path_points_blue,
            target_slope_pct=20.0,
            target_difficulty="blue",
            sector_name="Test",
        )
        graph.commit_paths(paths=[proposal])
        segment_ids = list(graph.segments.keys())
        slope = graph.finish_slope(segment_ids=segment_ids, name="Test Slope")

        # Record original state
        orig_nodes = len(graph.nodes)
        orig_segments = len(graph.segments)
        orig_slopes = len(graph.slopes)
        orig_slope_name = slope.name

        # Serialize
        data = graph.to_dict()

        # Should be JSON-serializable
        json_str = json.dumps(data)
        assert len(json_str) > 0, "Should produce JSON string"

        # Restore to new graph
        restored = ResortGraph.from_dict(data=data)

        # Verify restoration
        assert len(restored.nodes) == orig_nodes, "Nodes should match"
        assert len(restored.segments) == orig_segments, "Segments should match"
        assert len(restored.slopes) == orig_slopes, "Slopes should match"
        assert list(restored.slopes.values())[0].name == orig_slope_name, "Slope name should match"

    def test_roundtrip_preserves_segment_points(self, empty_graph, path_points_blue) -> None:
        """Segment points are preserved through serialization."""
        from skiresort_planner.model.proposed_path import ProposedSlopeSegment
        from skiresort_planner.model.resort_graph import ResortGraph

        graph = empty_graph
        proposal = ProposedSlopeSegment(
            points=path_points_blue,
            target_slope_pct=20.0,
            target_difficulty="blue",
            sector_name="Test",
        )
        graph.commit_paths(paths=[proposal])

        orig_segment = list(graph.segments.values())[0]
        orig_point_count = len(orig_segment.points)
        orig_first_point = orig_segment.points[0]

        # Roundtrip
        data = graph.to_dict()
        restored = ResortGraph.from_dict(data=data)

        restored_segment = list(restored.segments.values())[0]
        assert len(restored_segment.points) == orig_point_count, "Point count should match"
        assert abs(restored_segment.points[0].lon - orig_first_point.lon) < 0.0001
        assert abs(restored_segment.points[0].lat - orig_first_point.lat) < 0.0001
        assert abs(restored_segment.points[0].elevation - orig_first_point.elevation) < 0.1


class TestFileSaveLoad:
    """Tests for file-based save/load operations."""

    def test_save_and_load_from_file(self, empty_graph, path_points_blue) -> None:
        """ResortGraph can be serialized to JSON file and loaded back using to_dict/from_dict."""
        import json

        from skiresort_planner.model.proposed_path import ProposedSlopeSegment
        from skiresort_planner.model.resort_graph import ResortGraph

        graph = empty_graph
        proposal = ProposedSlopeSegment(
            points=path_points_blue,
            target_slope_pct=20.0,
            target_difficulty="blue",
            sector_name="Test",
        )
        graph.commit_paths(paths=[proposal])
        segment_ids = list(graph.segments.keys())
        graph.finish_slope(segment_ids=segment_ids, name="File Test Slope")

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            filepath = Path(f.name)
            # Manual save via to_dict + JSON
            json.dump(graph.to_dict(), f)

        try:
            with open(filepath, "r") as f:
                data = json.load(f)
            loaded = ResortGraph.from_dict(data=data)

            assert len(loaded.nodes) == len(graph.nodes)
            assert len(loaded.segments) == len(graph.segments)
            assert len(loaded.slopes) == len(graph.slopes)
        finally:
            filepath.unlink()  # Clean up


class TestLiftSerialization:
    """Tests for lift serialization including pylons and cable points."""

    def test_lift_roundtrip_preserves_pylons(self, mock_dem_blue_slope) -> None:
        """Lift pylons are preserved through serialization."""
        from skiresort_planner.constants import MapConfig
        from skiresort_planner.model.node import Node
        from skiresort_planner.model.path_point import PathPoint
        from skiresort_planner.model.resort_graph import ResortGraph

        dem = mock_dem_blue_slope
        M = MapConfig.METERS_PER_DEGREE_EQUATOR
        graph = ResortGraph()

        # Create nodes
        graph.nodes["N1"] = Node(
            id="N1",
            location=PathPoint(
                lon=0.0,
                lat=-1000 / M,
                elevation=dem.get_elevation_or_raise(lon=0.0, lat=-1000 / M),
            ),
        )
        graph.nodes["N2"] = Node(
            id="N2",
            location=PathPoint(
                lon=0.0,
                lat=0.0,
                elevation=dem.get_elevation_or_raise(lon=0.0, lat=0.0),
            ),
        )

        # Add lift
        lift = graph.add_lift(
            start_node_id="N1",
            end_node_id="N2",
            lift_type="chairlift",
            dem=dem,
        )
        orig_pylon_count = len(lift.pylons)

        # Roundtrip
        data = graph.to_dict()
        restored = ResortGraph.from_dict(data=data)

        restored_lift = list(restored.lifts.values())[0]
        assert len(restored_lift.pylons) == orig_pylon_count, "Pylon count should match"

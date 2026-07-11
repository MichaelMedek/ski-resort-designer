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
        from skiresort_planner.model.proposed_path import ProposedPathSegment
        from skiresort_planner.model.resort_graph import ResortGraph

        graph = empty_graph

        # Create some data
        proposal = ProposedPathSegment(
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
        from skiresort_planner.model.proposed_path import ProposedPathSegment
        from skiresort_planner.model.resort_graph import ResortGraph

        graph = empty_graph
        proposal = ProposedPathSegment(
            points=path_points_blue,
            target_slope_pct=20.0,
            target_difficulty="blue",
            sector_name="Test",
        )
        graph.commit_paths(paths=[proposal])
        graph.finish_slope(segment_ids=list(graph.segments.keys()))  # own the segment (orphans are discarded on load)

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

        from skiresort_planner.model.proposed_path import ProposedPathSegment
        from skiresort_planner.model.resort_graph import ResortGraph

        graph = empty_graph
        proposal = ProposedPathSegment(
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


class TestGPXExport:
    """Tests for the 'Export GPX' user action (ResortGraph.to_gpx)."""

    def _graph_with_slope_and_lift(self, dem):
        from skiresort_planner.constants import MapConfig
        from skiresort_planner.model.path_point import PathPoint
        from skiresort_planner.model.proposed_path import ProposedPathSegment
        from skiresort_planner.model.resort_graph import ResortGraph

        M = MapConfig.METERS_PER_DEGREE_EQUATOR
        graph = ResortGraph()
        # A finished slope.
        pts = [
            PathPoint(lon=0.0, lat=0.0, elevation=dem.get_elevation_or_raise(lon=0.0, lat=0.0)),
            PathPoint(lon=0.0, lat=-400 / M, elevation=dem.get_elevation_or_raise(lon=0.0, lat=-400 / M)),
        ]
        graph.commit_paths(paths=[ProposedPathSegment(points=pts, target_difficulty="blue")])
        graph.finish_slope(segment_ids=list(graph.segments.keys()))
        # A lift.
        bottom, _ = graph.get_or_create_node(
            lon=0.0, lat=-1000 / M, elevation=dem.get_elevation_or_raise(lon=0.0, lat=-1000 / M)
        )
        top, _ = graph.get_or_create_node(lon=0.0, lat=0.0, elevation=dem.get_elevation_or_raise(lon=0.0, lat=0.0))
        graph.add_lift(start_node_id=bottom.id, end_node_id=top.id, lift_type="chairlift", dem=dem)
        return graph

    def test_gpx_has_slope_and_lift_tracks(self, mock_dem_blue_slope) -> None:
        """to_gpx produces valid XML with one <trk> per slope and lift."""
        import xml.etree.ElementTree as ET

        graph = self._graph_with_slope_and_lift(mock_dem_blue_slope)
        gpx_str = graph.to_gpx()

        root = ET.fromstring(gpx_str)  # parses → well-formed XML
        ns = "{http://www.topografix.com/GPX/1/1}"
        tracks = root.findall(f"{ns}trk")
        assert len(tracks) == 2, "one track for the slope + one for the lift"

        types = {t.find(f"{ns}type").text for t in tracks}
        assert any(t.startswith("slope_") for t in types)
        assert any(t.startswith("lift_") for t in types)

        # Every track has at least one trackpoint with an elevation.
        for trk in tracks:
            pts = trk.findall(f"{ns}trkseg/{ns}trkpt")
            assert pts, "each track must have trackpoints"
            assert pts[0].find(f"{ns}ele") is not None

    def test_empty_graph_gpx_is_valid_with_no_tracks(self, empty_graph) -> None:
        """An empty resort still exports well-formed GPX (metadata, no tracks)."""
        import xml.etree.ElementTree as ET

        root = ET.fromstring(empty_graph.to_gpx())
        ns = "{http://www.topografix.com/GPX/1/1}"
        assert root.find(f"{ns}metadata") is not None
        assert root.findall(f"{ns}trk") == []

    def test_gpx_exports_roads(self, empty_graph, path_points_blue) -> None:
        """to_gpx emits a <trk type='road'> per road (roads are SegmentPaths like slopes)."""
        import xml.etree.ElementTree as ET

        from skiresort_planner.model.path_segment import SegmentKind
        from skiresort_planner.model.proposed_path import ProposedPathSegment

        empty_graph.commit_paths(
            paths=[ProposedPathSegment(points=path_points_blue, is_connector=True, kind=SegmentKind.ROAD)],
            record_undo=False,
        )
        road = empty_graph.finish_road(segment_ids=[list(empty_graph.segments.keys())[-1]])

        root = ET.fromstring(empty_graph.to_gpx())
        ns = "{http://www.topografix.com/GPX/1/1}"
        road_tracks = [t for t in root.findall(f"{ns}trk") if t.find(f"{ns}type").text == "road"]
        assert len(road_tracks) == 1, "the road must be exported as a GPX track"
        assert road_tracks[0].find(f"{ns}name").text == road.name
        pts = road_tracks[0].findall(f"{ns}trkseg/{ns}trkpt")
        assert pts and pts[0].find(f"{ns}ele") is not None


class TestRoadSerialization:
    """Roads round-trip through to_dict/from_dict with their counter preserved."""

    def test_roundtrip_preserves_roads(self, empty_graph, path_points_blue) -> None:
        from skiresort_planner.model.path_segment import SegmentKind
        from skiresort_planner.model.proposed_path import ProposedPathSegment
        from skiresort_planner.model.resort_graph import ResortGraph

        proposal = ProposedPathSegment(points=path_points_blue, is_connector=True, kind=SegmentKind.ROAD)
        empty_graph.commit_paths(paths=[proposal], record_undo=False)
        road = empty_graph.finish_road(segment_ids=[list(empty_graph.segments.keys())[-1]])
        road_seg_id = road.segment_ids[0]

        data = empty_graph.to_dict()
        restored = ResortGraph.from_dict(data=data)

        assert road.id in restored.roads
        assert restored.roads[road.id].name == road.name
        assert restored._road_counter == empty_graph._road_counter
        # The segment's road kind survives the round-trip (persisted, not recomputed).
        assert restored.segments[road_seg_id].kind is SegmentKind.ROAD

    def test_road_owned_slope_kind_segment_raises(self, empty_graph, path_points_blue) -> None:
        """A road owning a kind=SLOPE segment (corrupt/stale save) fails loudly on load."""
        import pytest

        from skiresort_planner.model.path_segment import SegmentKind
        from skiresort_planner.model.proposed_path import ProposedPathSegment
        from skiresort_planner.model.resort_graph import ResortGraph

        proposal = ProposedPathSegment(points=path_points_blue, is_connector=True, kind=SegmentKind.ROAD)
        empty_graph.commit_paths(paths=[proposal], record_undo=False)
        road = empty_graph.finish_road(segment_ids=[list(empty_graph.segments.keys())[-1]])

        data = empty_graph.to_dict()
        # Corrupt the persisted segment's kind to simulate a pre-SegmentKind save.
        data["segments"][road.segment_ids[0]]["kind"] = "slope"

        with pytest.raises(AssertionError, match="expected ROAD"):
            ResortGraph.from_dict(data=data)

    def test_orphan_segment_discarded_on_load(self, empty_graph, path_points_blue) -> None:
        """A segment owned by no slope/road (interrupted-build leftover) is discarded on
        load — it can never be re-associated (build context isn't persisted) and would
        otherwise render as an unopenable, undeletable 'Building …' ghost.
        """
        from skiresort_planner.model.path_point import PathPoint
        from skiresort_planner.model.path_segment import SegmentKind
        from skiresort_planner.model.proposed_path import ProposedPathSegment
        from skiresort_planner.model.resort_graph import ResortGraph

        # One finished road (kept) + one committed-but-unfinished road segment (orphan).
        empty_graph.commit_paths(
            paths=[ProposedPathSegment(points=path_points_blue, is_connector=True, kind=SegmentKind.ROAD)],
            record_undo=False,
        )
        road = empty_graph.finish_road(segment_ids=[list(empty_graph.segments.keys())[-1]])
        M = 111320.0
        orphan_pts = [
            PathPoint(lon=500 / M, lat=0.0, elevation=2000.0),
            PathPoint(lon=800 / M, lat=0.0, elevation=1990.0),
        ]
        empty_graph.commit_paths(
            paths=[ProposedPathSegment(points=orphan_pts, is_connector=True, kind=SegmentKind.ROAD)],
            record_undo=False,
        )
        orphan_id = list(empty_graph.segments.keys())[-1]
        assert orphan_id not in road.segment_ids

        restored = ResortGraph.from_dict(data=empty_graph.to_dict())
        assert orphan_id not in restored.segments, "orphan segment must be discarded on load"
        assert set(restored.roads[road.id].segment_ids) <= set(restored.segments), "owned road segments kept"
        assert len(restored.roads) == 1, "the finished road survives"

    def test_pre_roads_backup_loads(self, empty_graph, path_points_blue) -> None:
        """A backup written before roads existed (no 'roads' key, no 'road' counter)
        must still load — schema evolution, not a crash. Regression for KeyError.
        """
        from skiresort_planner.model.proposed_path import ProposedPathSegment
        from skiresort_planner.model.resort_graph import ResortGraph

        empty_graph.commit_paths(paths=[ProposedPathSegment(points=path_points_blue, target_difficulty="blue")])
        empty_graph.finish_slope(segment_ids=list(empty_graph.segments.keys()))

        data = empty_graph.to_dict()
        # Simulate an old on-disk backup: strip the road fields entirely.
        del data["roads"]
        del data["counters"]["road"]

        restored = ResortGraph.from_dict(data=data)
        assert restored.roads == {}
        assert restored._road_counter == 0
        assert len(restored.slopes) == 1, "existing slopes still load"

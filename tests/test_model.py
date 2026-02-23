"""Tests for skiresort_planner data model classes.

Tests: PathPoint, Node, SlopeSegment, Slope, Lift, ResortGraph
Focus: Data integrity, computed properties, serialization

Note: Fixtures are defined in conftest.py (MockDEMService, path points, nodes, graphs).
"""

import pytest

from skiresort_planner.model.lift import Lift
from skiresort_planner.model.node import Node
from skiresort_planner.model.path_point import PathPoint
from skiresort_planner.model.proposed_path import ProposedSlopeSegment
from skiresort_planner.model.resort_graph import FinishSlopeAction, ResortGraph
from skiresort_planner.model.slope import Slope
from skiresort_planner.model.slope_segment import SlopeSegment


# =============================================================================
# TESTS FOR MODEL CLASSES
# =============================================================================


class TestPathPoint:
    """PathPoint - the fundamental geometry atom."""

    def test_path_point_creation_and_distance(self) -> None:
        """PathPoint creation, properties, and distance calculation."""
        pt1 = PathPoint(lon=10.27, lat=46.97, elevation=2500.0)
        assert pt1.lon == 10.27 and pt1.lat == 46.97 and pt1.elevation == 2500.0

        pt2 = PathPoint(lon=10.27, lat=46.96, elevation=2400.0)  # ~1.1km south
        assert 1050 < pt1.distance_to(other=pt2) < 1150  # 0.01° ≈ 1100m


class TestNode:
    """Node - junction point wrapping PathPoint."""

    def test_node_distance_to_point(self, node_at_origin_summit: Node) -> None:
        """Node at origin, distance to 1000m south (0.009° at equator)."""
        assert 990 < node_at_origin_summit.distance_to(lon=0.0, lat=-0.009) < 1010


class TestProposedSlopeSegment:
    """ProposedSlopeSegment computed properties: drop, length, slope, difficulty."""

    def test_computed_properties(self, path_points_800m_south_20pct_drop: list[PathPoint]) -> None:
        """All computed properties from 5-point path (800m, 160m drop, 20% blue)."""
        seg = ProposedSlopeSegment(points=path_points_800m_south_20pct_drop)
        assert seg.total_drop_m == 160.0  # 2500 - 2340
        assert 780 < seg.length_m < 820  # 4 × 200m
        assert 18 < seg.avg_slope_pct < 22  # 160/800 = 20%
        assert seg.difficulty == "blue"


class TestResortGraph:
    """ResortGraph - central manager for all entities."""

    def test_empty_graph(self, empty_resort_graph: ResortGraph) -> None:
        """Fresh graph has no entities."""
        assert len(empty_resort_graph.nodes) == 0
        assert len(empty_resort_graph.segments) == 0
        assert len(empty_resort_graph.slopes) == 0
        assert len(empty_resort_graph.lifts) == 0

    def test_node_creation(self, empty_resort_graph: ResortGraph) -> None:
        """get_or_create_node creates new node, reuses existing if nearby."""
        node1, created1 = empty_resort_graph.get_or_create_node(lon=10.27, lat=46.97, elevation=2500.0)
        assert created1 and node1.id == "N1" and len(empty_resort_graph.nodes) == 1

        node2, created2 = empty_resort_graph.get_or_create_node(lon=10.2700005, lat=46.97, elevation=2500.0)
        assert not created2 and node2.id == node1.id and len(empty_resort_graph.nodes) == 1

    def test_commit_finish_undo_workflow(
        self, empty_resort_graph: ResortGraph, proposed_segment_blue_800m: ProposedSlopeSegment
    ) -> None:
        """Full workflow: commit path → finish slope → undo."""
        # Commit creates segment and nodes
        empty_resort_graph.commit_paths(paths=[proposed_segment_blue_800m])
        assert len(empty_resort_graph.segments) == 1 and len(empty_resort_graph.nodes) == 2
        segment = list(empty_resort_graph.segments.values())[0]
        assert segment.start_node_id == "N1" and segment.end_node_id == "N2"

        # Finish groups segments into slope
        slope = empty_resort_graph.finish_slope(segment_ids=[segment.id])
        assert slope and len(empty_resort_graph.slopes) == 1 and segment.id in slope.segment_ids

        # Undo removes slope
        undone = empty_resort_graph.undo_last()
        assert undone and isinstance(undone, FinishSlopeAction)

    def test_add_lift(
        self, resort_graph_with_3_nodes_vertical: ResortGraph, mock_dem_blue_slope_south: "MockDEMService"
    ) -> None:
        """Add lift between existing nodes."""
        lift = resort_graph_with_3_nodes_vertical.add_lift(
            start_node_id="N3", end_node_id="N1", lift_type="chairlift", dem=mock_dem_blue_slope_south
        )
        assert lift and lift.lift_type == "chairlift"
        assert lift.start_node_id == "N3" and lift.end_node_id == "N1"


class TestSlope:
    """Slope - number extraction and name generation."""

    def test_number_from_id_single_digit(self) -> None:
        """number_from_id extracts single digit from slope ID."""
        assert Slope.number_from_id(slope_id="SL1") == 1
        assert Slope.number_from_id(slope_id="SL5") == 5

    def test_number_from_id_multi_digit(self) -> None:
        """number_from_id extracts multi-digit numbers."""
        assert Slope.number_from_id(slope_id="SL12") == 12
        assert Slope.number_from_id(slope_id="SL999") == 999

    def test_number_property_matches_id(self) -> None:
        """Slope.number property derives from ID correctly."""
        slope = Slope(
            id="SL7",
            name="7 (Test Slope)",
            segment_ids=["S1"],
            start_node_id="N1",
            end_node_id="N2",
        )
        assert slope.number == 7

    def test_generate_name_format(self) -> None:
        """generate_name produces '{number} (creative name)' format."""
        name = Slope.generate_name(
            difficulty="blue",
            slope_id="SL3",
            start_elevation=2500.0,
            end_elevation=2200.0,
            avg_bearing=180.0,
        )
        assert name.startswith("3 (")
        assert name.endswith(")")


class TestLift:
    """Lift - number extraction and name generation."""

    def test_number_from_id_single_digit(self) -> None:
        """number_from_id extracts single digit from lift ID."""
        assert Lift.number_from_id(lift_id="L1") == 1
        assert Lift.number_from_id(lift_id="L5") == 5

    def test_number_from_id_multi_digit(self) -> None:
        """number_from_id extracts multi-digit numbers."""
        assert Lift.number_from_id(lift_id="L12") == 12
        assert Lift.number_from_id(lift_id="L999") == 999

    def test_number_property_matches_id(self) -> None:
        """Lift.number property derives from ID correctly."""
        start_point = PathPoint(lon=10.0, lat=46.0, elevation=1500)
        end_point = PathPoint(lon=10.0, lat=46.01, elevation=2000)
        lift = Lift(
            id="L7",
            name="7 (Test Lift)",
            lift_type="chairlift",
            terrain_points=[start_point, end_point],
            start_node_id="N1",
            end_node_id="N2",
            pylons=[],
            cable_points=[start_point, end_point],
        )
        assert lift.number == 7

    def test_generate_name_different_types(self) -> None:
        """generate_name works for all lift types."""
        for lift_type in ["surface_lift", "chairlift", "gondola", "aerial_tram"]:
            name = Lift.generate_name(
                lift_type=lift_type,
                lift_id="L1",
                length_m=500.0,
                vertical_rise_m=200.0,
                avg_bearing=90.0,
            )
            assert name.startswith("1 (")
            assert name.endswith(")")


class TestResortGraphSerialization:
    """ResortGraph save/load roundtrip."""

    def test_to_dict_and_from_dict(
        self,
        empty_resort_graph: ResortGraph,
        proposed_segment_blue_800m: ProposedSlopeSegment,
        mock_dem_blue_slope_south: "MockDEMService",
    ) -> None:
        """Serialize and deserialize preserves all entities."""
        graph = empty_resort_graph
        graph.commit_paths(paths=[proposed_segment_blue_800m])
        graph.finish_slope(segment_ids=list(graph.segments.keys()))

        # Add lift to make it a complete resort
        start_node = list(graph.nodes.values())[0]
        end_node = list(graph.nodes.values())[1]
        graph.add_lift(
            start_node_id=end_node.id,
            end_node_id=start_node.id,
            lift_type="chairlift",
            dem=mock_dem_blue_slope_south,
        )

        # Test to_dict
        data = graph.to_dict()
        assert data["version"] == "2.0"
        assert len(data["nodes"]) == 2
        assert len(data["segments"]) == 1
        assert len(data["slopes"]) == 1
        assert len(data["lifts"]) == 1

        # Test from_dict roundtrip
        loaded = ResortGraph.from_dict(data=data)
        assert len(loaded.nodes) == len(graph.nodes)
        assert len(loaded.segments) == len(graph.segments)
        assert len(loaded.slopes) == len(graph.slopes)
        assert len(loaded.lifts) == len(graph.lifts)
        assert loaded._node_counter == graph._node_counter
        assert loaded._lift_counter == graph._lift_counter

        # Verify lift data preserved
        orig_lift = list(graph.lifts.values())[0]
        loaded_lift = loaded.lifts[orig_lift.id]
        assert loaded_lift.name == orig_lift.name
        assert loaded_lift.lift_type == orig_lift.lift_type
        assert len(loaded_lift.terrain_points) == len(orig_lift.terrain_points)

        # Test to_gpx export
        gpx_str = graph.to_gpx()
        assert "<gpx" in gpx_str
        assert "<trk>" in gpx_str
        assert "<trkpt" in gpx_str
        assert "<ele>" in gpx_str
        assert list(graph.slopes.values())[0].name in gpx_str
        assert orig_lift.name in gpx_str

        # Test from_dict rejects invalid JSON
        with pytest.raises(KeyError):
            ResortGraph.from_dict(data={})


class TestSlopeSegment:
    """SlopeSegment with warnings and difficulty."""

    def test_blue_slope_difficulty(self, path_points_600m_south_3pts: list[PathPoint]) -> None:
        """Blue slope (20% gradient) correctly classified."""
        segment = SlopeSegment(
            id="S1",
            name="Test",
            points=path_points_600m_south_3pts,
            start_node_id="N1",
            end_node_id="N2",
            side_slope_pct=10.0,
            side_slope_dir="left",
        )
        assert segment.difficulty == "blue"


class TestCleanupIsolatedNodes:
    """Tests for ResortGraph.cleanup_isolated_nodes method."""

    def test_removes_isolated_nodes(self, empty_resort_graph: ResortGraph) -> None:
        """Isolated nodes (not connected to segments/lifts) are removed."""
        # Add some nodes
        n1, _ = empty_resort_graph.get_or_create_node(lon=10.0, lat=46.0, elevation=2000)
        n2, _ = empty_resort_graph.get_or_create_node(lon=10.1, lat=46.1, elevation=1900)
        n3, _ = empty_resort_graph.get_or_create_node(lon=10.2, lat=46.2, elevation=1800)  # isolated

        # Add a segment connecting N1 and N2
        segment = SlopeSegment(
            id="S1",
            name="Test",
            points=[PathPoint(lon=10.0, lat=46.0, elevation=2000)],
            start_node_id=n1.id,
            end_node_id=n2.id,
        )
        empty_resort_graph.segments["S1"] = segment

        assert len(empty_resort_graph.nodes) == 3

        # Run cleanup
        removed_count = empty_resort_graph.cleanup_isolated_nodes()

        assert removed_count == 1
        assert len(empty_resort_graph.nodes) == 2
        assert n1.id in empty_resort_graph.nodes
        assert n2.id in empty_resort_graph.nodes
        assert n3.id not in empty_resort_graph.nodes

    def test_no_nodes_removed_when_all_connected(self, empty_resort_graph: ResortGraph) -> None:
        """No nodes removed when all are connected."""
        # Add nodes
        n1, _ = empty_resort_graph.get_or_create_node(lon=10.0, lat=46.0, elevation=2000)
        n2, _ = empty_resort_graph.get_or_create_node(lon=10.1, lat=46.1, elevation=1900)

        # Connect them with a segment
        segment = SlopeSegment(
            id="S1",
            name="Test",
            points=[PathPoint(lon=10.0, lat=46.0, elevation=2000)],
            start_node_id=n1.id,
            end_node_id=n2.id,
        )
        empty_resort_graph.segments["S1"] = segment

        removed_count = empty_resort_graph.cleanup_isolated_nodes()

        assert removed_count == 0
        assert len(empty_resort_graph.nodes) == 2

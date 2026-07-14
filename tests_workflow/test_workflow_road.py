"""Integration test for the road building workflow (ui/state_machine.py).

Roads build segment-by-segment like slopes: idle → road_starting → road_building
(self-loop per segment) → idle_viewing_road, plus cancel. Mirrors
test_workflow_slope.py. Atomic road model/graph/serialization/planner/3D tests
live in their mirror modules (test_road, test_resort_graph, test_segment_path,
test_serialization, test_connection_planners, test_center_map).
"""

import pytest
from statemachine.exceptions import TransitionNotAllowed

from skiresort_planner.model.path_segment import SegmentKind
from skiresort_planner.model.proposed_path import ProposedPathSegment
from skiresort_planner.model.resort_graph import ResortGraph
from skiresort_planner.ui.state_machine import PlannerStateMachine


def _commit_road_segment(graph: ResortGraph, path_points) -> str:
    """Commit one path as a road segment, return its segment id."""
    graph.commit_paths(
        paths=[ProposedPathSegment(points=path_points, is_connector=True, kind=SegmentKind.ROAD)], record_undo=False
    )
    return list(graph.segments.keys())[-1]


class TestRoadBuildingWorkflow:
    """Road building state transitions (no Streamlit), mirroring slope building."""

    def _sm(self, graph: ResortGraph):
        return PlannerStateMachine.create(graph=graph, add_ui_listener=False)

    def test_build_two_segments_then_finish(self, empty_graph, path_points_blue) -> None:
        sm, ctx = self._sm(empty_graph)
        assert sm.current_state_value == "idle_ready"

        # First click sets the origin → road_starting.
        sm.start_road(node_id=None, location=path_points_blue[0])
        assert sm.current_state_value == "road_starting"
        assert ctx.build(SegmentKind.ROAD).start_location is path_points_blue[0]

        # First traced segment → road_building, accumulates.
        seg1 = _commit_road_segment(empty_graph, path_points_blue)
        sm.commit_road(segment_id=seg1, endpoint_node_id="N1")
        assert sm.current_state_value == "road_building"
        assert ctx.build(SegmentKind.ROAD).segments == [seg1]
        assert ctx.build(SegmentKind.ROAD).endpoints == ["N1"]

        # Second segment self-loops, still road_building.
        seg2 = _commit_road_segment(empty_graph, path_points_blue)
        sm.commit_road(segment_id=seg2, endpoint_node_id="N2")
        assert sm.current_state_value == "road_building"
        assert ctx.build(SegmentKind.ROAD).segments == [seg1, seg2]

        # Finish → idle_viewing_road, panel visible, road context cleared.
        road = empty_graph.finish_road(segment_ids=ctx.build(SegmentKind.ROAD).segments)
        sm.finish_road(road_id=road.id)
        assert sm.current_state_value == "idle_viewing_road"
        assert ctx.viewing.road_id == road.id
        assert ctx.viewing.panel_visible is True

    def test_cancel_from_starting_returns_to_idle(self, empty_graph, path_points_blue) -> None:
        sm, ctx = self._sm(empty_graph)
        sm.start_road(node_id=None, location=path_points_blue[0])
        assert sm.current_state_value == "road_starting"

        sm.cancel_road()
        assert sm.current_state_value == "idle_ready"
        assert ctx.build(SegmentKind.ROAD).start_location is None  # cleared on exit

    def test_cancel_from_building_returns_to_idle(self, empty_graph, path_points_blue) -> None:
        sm, ctx = self._sm(empty_graph)
        sm.start_road(node_id=None, location=path_points_blue[0])
        seg1 = _commit_road_segment(empty_graph, path_points_blue)
        sm.commit_road(segment_id=seg1, endpoint_node_id="N1")
        assert sm.current_state_value == "road_building"

        sm.cancel_road()
        assert sm.current_state_value == "idle_ready"
        assert ctx.build(SegmentKind.ROAD).segments == []  # cleared on cancel

    def test_branch_road_from_existing_node(self, empty_graph) -> None:
        """A road can start from an existing junction node (branch point)."""
        node, _ = empty_graph.get_or_create_node(lon=0.0, lat=0.0, elevation=2000.0)
        sm, ctx = self._sm(empty_graph)

        sm.start_road(node_id=node.id, location=None)
        assert sm.current_state_value == "road_starting"
        assert ctx.build(SegmentKind.ROAD).start_node_id == node.id


class TestRoadInvalidTransitions:
    """Invalid road transitions are blocked (mirrors slope TestInvalidTransitions)."""

    def _sm(self, graph: ResortGraph):
        return PlannerStateMachine.create(graph=graph, add_ui_listener=False)

    def test_cannot_finish_from_starting_state(self, empty_graph, path_points_blue) -> None:
        """finish_road is not allowed from road_starting (need at least 1 segment)."""
        sm, _ctx = self._sm(empty_graph)
        sm.start_road(node_id=None, location=path_points_blue[0])
        assert sm.current_state_value == "road_starting"

        with pytest.raises(TransitionNotAllowed):
            sm.finish_road(road_id="R1")


class TestRoadConnectorAutoFinish:
    """commit_road_finish auto-ends the road when a segment connects to an existing node.

    Mirrors commit_custom_finish for slopes: a single event fires from either
    road_starting or road_building straight to idle_viewing_road.
    """

    def _sm(self, graph: ResortGraph):
        return PlannerStateMachine.create(graph=graph, add_ui_listener=False)

    def test_connector_finish_from_starting(self, empty_graph, path_points_blue) -> None:
        sm, ctx = self._sm(empty_graph)
        sm.start_road(node_id=None, location=path_points_blue[0])
        assert sm.current_state_value == "road_starting"

        seg = _commit_road_segment(empty_graph, path_points_blue)
        road = empty_graph.finish_road(segment_ids=[seg])
        assert road is not None
        sm.commit_road_finish(segment_id=seg, road_id=road.id)

        # Connector segment ends the road immediately, straight from road_starting.
        assert sm.current_state_value == "idle_viewing_road"
        assert ctx.viewing.road_id == road.id
        assert ctx.viewing.panel_visible is True
        # enter_idle_viewing_road clears the build scratch.
        assert ctx.build(SegmentKind.ROAD).segments == []

    def test_connector_finish_from_building(self, empty_graph, path_points_blue) -> None:
        sm, ctx = self._sm(empty_graph)
        sm.start_road(node_id=None, location=path_points_blue[0])
        seg1 = _commit_road_segment(empty_graph, path_points_blue)
        sm.commit_road(segment_id=seg1, endpoint_node_id="N1")
        assert sm.current_state_value == "road_building"

        seg2 = _commit_road_segment(empty_graph, path_points_blue)
        road = empty_graph.finish_road(segment_ids=[seg1, seg2])
        assert road is not None
        sm.commit_road_finish(segment_id=seg2, road_id=road.id)

        assert sm.current_state_value == "idle_viewing_road"
        assert ctx.viewing.road_id == road.id

    def test_connector_finish_is_idempotent_on_already_tracked_segment(self, empty_graph, path_points_blue) -> None:
        """before_commit_road_finish only appends the segment if not already tracked."""
        sm, ctx = self._sm(empty_graph)
        sm.start_road(node_id=None, location=path_points_blue[0])
        seg1 = _commit_road_segment(empty_graph, path_points_blue)
        sm.commit_road(segment_id=seg1, endpoint_node_id="N1")
        assert ctx.build(SegmentKind.ROAD).segments == [seg1]

        # seg1 is already in road_build.segments; the finish hook must not double-append.
        road = empty_graph.finish_road(segment_ids=[seg1])
        assert road is not None
        sm.commit_road_finish(segment_id=seg1, road_id=road.id)

        assert sm.current_state_value == "idle_viewing_road"
        assert ctx.viewing.road_id == road.id

    def test_connector_finish_not_allowed_from_idle(self, empty_graph) -> None:
        """commit_road_finish has no transition out of idle_ready."""
        sm, _ctx = self._sm(empty_graph)
        assert sm.current_state_value == "idle_ready"
        with pytest.raises(TransitionNotAllowed):
            sm.commit_road_finish(segment_id="S1", road_id="R1")


class TestRoadForceStateMethods:
    """force_road_building / force_road_starting (undo helpers), mirroring slope force methods."""

    def _sm(self, graph: ResortGraph):
        return PlannerStateMachine.create(graph=graph, add_ui_listener=False)

    def test_force_road_building_when_segments_remain(self, empty_graph, path_points_blue) -> None:
        # Simulate an undo that leaves ≥1 committed road segment → force RoadBuilding.
        sm, ctx = self._sm(empty_graph)
        sm.start_road(node_id=None, location=path_points_blue[0])
        seg1 = _commit_road_segment(empty_graph, path_points_blue)
        sm.commit_road(segment_id=seg1, endpoint_node_id="N1")
        assert sm.current_state_value == "road_building"

        # Seed a stale viewing panel that force_road_building must clear (via viewing.clear()).
        ctx.viewing.set_road_id(road_id="R99")
        ctx.viewing.show_panel()

        sm.force_building(SegmentKind.ROAD)
        assert sm.current_state_value == "road_building"
        assert ctx.build(SegmentKind.ROAD).segments == [seg1]
        assert ctx.viewing.road_id is None  # force_road_building cleared the stale viewing state
        assert ctx.viewing.panel_visible is False

    def test_force_road_starting_when_only_origin_remains(self, empty_graph, path_points_blue) -> None:
        # Simulate undoing the last segment: origin still set, no segments → force RoadStarting.
        sm, ctx = self._sm(empty_graph)
        sm.start_road(node_id=None, location=path_points_blue[0])
        seg1 = _commit_road_segment(empty_graph, path_points_blue)
        sm.commit_road(segment_id=seg1, endpoint_node_id="N1")
        assert sm.current_state_value == "road_building"

        # Peel the segment back to the origin, then force RoadStarting.
        ctx.build(SegmentKind.ROAD).segments = []
        ctx.build(SegmentKind.ROAD).endpoints = []
        sm.force_starting(SegmentKind.ROAD)
        assert sm.current_state_value == "road_starting"

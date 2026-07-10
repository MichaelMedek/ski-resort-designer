"""Integration test for the road placement workflow (ui/state_machine.py).

The two-click road flow driven through the state machine: idle → road_placing →
idle_viewing_road, plus cancel. Atomic road model/graph/serialization/planner/3D
tests live in their mirror modules (test_road, test_resort_graph, test_segment_path,
test_serialization, test_connection_planners, test_center_map).
"""

from skiresort_planner.model.proposed_path import ProposedPathSegment
from skiresort_planner.model.resort_graph import ResortGraph
from skiresort_planner.ui.state_machine import PlannerStateMachine


def _commit_road(graph: ResortGraph, path_points):
    """Commit a path as a road (record_undo=False + finish_road), return the Road."""
    graph.commit_paths(paths=[ProposedPathSegment(points=path_points, is_connector=True)], record_undo=False)
    road = graph.finish_road(segment_ids=[list(graph.segments.keys())[-1]])
    assert road is not None
    return road


class TestRoadPlacementWorkflow:
    """Road placement state transitions (no Streamlit)."""

    def _sm(self, graph: ResortGraph):
        return PlannerStateMachine.create(graph=graph, add_ui_listener=False)

    def test_start_then_complete_road(self, empty_graph, path_points_blue) -> None:
        sm, ctx = self._sm(empty_graph)
        assert sm.current_state_value == "idle_ready"

        sm.start_road(node_id=None, location=path_points_blue[0])
        assert sm.current_state_value == "road_placing"
        assert ctx.road.start_location is path_points_blue[0]

        road = _commit_road(empty_graph, path_points_blue)
        sm.complete_road(road_id=road.id)
        assert sm.current_state_value == "idle_viewing_road"
        assert ctx.viewing.road_id == road.id
        assert ctx.viewing.panel_visible is True

    def test_cancel_road_returns_to_idle(self, empty_graph, path_points_blue) -> None:
        sm, ctx = self._sm(empty_graph)
        sm.start_road(node_id=None, location=path_points_blue[0])
        assert sm.current_state_value == "road_placing"

        sm.cancel_road()
        assert sm.current_state_value == "idle_ready"
        assert ctx.road.start_location is None  # cleared on exit

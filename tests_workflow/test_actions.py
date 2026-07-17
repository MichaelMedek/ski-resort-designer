"""Core-logic tests for the UI action functions (actions.py).

These functions read st.session_state.{state_machine, context, graph}; with the
fake `st` installed we seed those and call the action directly, asserting the
real effect (entity removed, panel closed when it was being viewed). Covers the
delete actions for slope/lift/road uniformly.
"""

import pytest

from skiresort_planner.constants import MapConfig
from skiresort_planner.model.node import Node
from skiresort_planner.model.path_point import PathPoint
from skiresort_planner.model.path_segment import SegmentKind
from skiresort_planner.model.proposed_path import ProposedPathSegment
from skiresort_planner.model.resort_graph import ResortGraph
from skiresort_planner.ui.state_machine import PlannerStateMachine
from tests_workflow.conftest import MockDEMService

M = 111320.0  # metres per degree near the equator


def _node_at(dem: MockDEMService, node_id: str, lon: float, lat: float) -> Node:
    """A Node at (lon, lat) with DEM elevation — for seeding lift stations in delete tests."""
    return Node(
        id=node_id, location=PathPoint(lon=lon, lat=lat, elevation=dem.get_elevation_or_raise(lon=lon, lat=lat))
    )


def _session(fake_st, graph, factory=None, dem=None):
    """Seed fake st.session_state with the objects action functions read."""
    sm, ctx = PlannerStateMachine.create(graph=graph, add_ui_listener=False)
    fake_st.session_state["state_machine"] = sm
    fake_st.session_state["context"] = ctx
    fake_st.session_state["graph"] = graph
    fake_st.session_state["camera_epoch"] = 0
    fake_st.session_state["dedup_epoch"] = 0
    if factory is not None:
        fake_st.session_state["path_factory"] = factory
    if dem is not None:
        fake_st.session_state["dem_service"] = dem
    return sm, ctx


def _make_slope(graph, path_points):
    graph.commit_paths(paths=[ProposedPathSegment(points=path_points, target_difficulty="blue")])
    return graph.finish_slope(segment_ids=list(graph.segments.keys()))


def _make_road(graph):
    M = 111320.0
    pts = [PathPoint(lon=0.0, lat=0.0, elevation=2000.0), PathPoint(lon=300 / M, lat=0.0, elevation=1990.0)]
    graph.commit_paths(
        paths=[ProposedPathSegment(points=pts, is_connector=True, kind=SegmentKind.ROAD)], record_undo=False
    )
    return graph.finish_road(segment_ids=[list(graph.segments.keys())[-1]])


class TestDeleteSlopeAction:
    def test_removes_slope(self, fake_st, empty_graph, path_points_blue) -> None:
        from skiresort_planner.ui.actions import delete_slope_action

        slope = _make_slope(empty_graph, path_points_blue)
        _session(fake_st, empty_graph)

        assert delete_slope_action(slope_id=slope.id) is True
        assert slope.id not in empty_graph.slopes

    def test_missing_slope_returns_false(self, fake_st, empty_graph) -> None:
        from skiresort_planner.ui.actions import delete_slope_action

        _session(fake_st, empty_graph)
        assert delete_slope_action(slope_id="SL999") is False

    def test_closes_panel_when_viewing_deleted_slope(self, fake_st, empty_graph, path_points_blue) -> None:
        from skiresort_planner.ui.actions import delete_slope_action

        slope = _make_slope(empty_graph, path_points_blue)
        sm, _ctx = _session(fake_st, empty_graph)
        sm.view_slope(slope_id=slope.id)

        delete_slope_action(slope_id=slope.id)
        assert not sm.is_idle_viewing_slope, "deleting the viewed slope must close its panel"


class TestRenameEntityAction:
    def test_sets_name_and_bumps_map(self, fake_st, empty_graph, path_points_blue) -> None:
        from skiresort_planner.ui.actions import rename_entity_action

        slope = _make_slope(empty_graph, path_points_blue)
        _session(fake_st, empty_graph)
        epoch_before = fake_st.session_state["dedup_epoch"]
        camera_before = fake_st.session_state["camera_epoch"]

        rename_entity_action(entity_id=slope.id, new_name="  Renamed  ")

        assert empty_graph.slopes[slope.id].name == "Renamed", "name is trimmed and applied"
        assert fake_st.session_state["dedup_epoch"] > epoch_before, "rename refreshes the label redraw"
        assert fake_st.session_state["camera_epoch"] == camera_before, "rename must NOT recenter"

    def test_empty_name_is_noop(self, fake_st, empty_graph, path_points_blue) -> None:
        from skiresort_planner.ui.actions import rename_entity_action

        slope = _make_slope(empty_graph, path_points_blue)
        original = slope.name
        _session(fake_st, empty_graph)

        rename_entity_action(entity_id=slope.id, new_name="   ")

        assert empty_graph.slopes[slope.id].name == original, "blank name must not overwrite"


class TestDeleteRoadAction:
    def test_removes_road(self, fake_st, empty_graph) -> None:
        from skiresort_planner.ui.actions import delete_road_action

        road = _make_road(empty_graph)
        _session(fake_st, empty_graph)

        assert delete_road_action(road_id=road.id) is True
        assert road.id not in empty_graph.roads

    def test_missing_road_returns_false(self, fake_st, empty_graph) -> None:
        from skiresort_planner.ui.actions import delete_road_action

        _session(fake_st, empty_graph)
        assert delete_road_action(road_id="R999") is False

    def test_closes_panel_when_viewing_deleted_road(self, fake_st, empty_graph) -> None:
        from skiresort_planner.ui.actions import delete_road_action

        road = _make_road(empty_graph)
        sm, _ctx = _session(fake_st, empty_graph)
        sm.view_road(road_id=road.id)

        delete_road_action(road_id=road.id)
        assert not sm.is_idle_viewing_road, "deleting the viewed road must close its panel"


class TestDeleteLiftAction:
    def test_removes_lift(self, fake_st, empty_graph, mock_dem_blue_slope) -> None:
        from skiresort_planner.ui.actions import delete_lift_action

        dem = mock_dem_blue_slope
        M = 111320.0
        bottom, _ = empty_graph.get_or_create_node(
            lon=0.0, lat=-1000 / M, elevation=dem.get_elevation_or_raise(lon=0.0, lat=-1000 / M)
        )
        top, _ = empty_graph.get_or_create_node(
            lon=0.0, lat=0.0, elevation=dem.get_elevation_or_raise(lon=0.0, lat=0.0)
        )
        lift = empty_graph.add_lift(start_node_id=bottom.id, end_node_id=top.id, lift_type="chairlift", dem=dem)
        _session(fake_st, empty_graph)

        assert delete_lift_action(lift_id=lift.id) is True
        assert lift.id not in empty_graph.lifts


def _two_segment_slope(graph: ResortGraph, dem: MockDEMService) -> ResortGraph:
    """Commit two contiguous 300m slope segments so the graph has 3 nodes with a junction.

    Nodes sit at lat 0, -300/M, -600/M (all lon 0). Adjacent nodes are 300m apart (< 500m,
    mergeable); the endpoints are 600m apart (> MergeConfig.MAX_SPAN_M, not mergeable).
    """
    mid = -300 / M
    bot = -600 / M
    seg_a = [
        PathPoint(lon=0.0, lat=0.0, elevation=dem.get_elevation_or_raise(lon=0.0, lat=0.0)),
        PathPoint(lon=0.0, lat=mid, elevation=dem.get_elevation_or_raise(lon=0.0, lat=mid)),
    ]
    seg_b = [
        PathPoint(lon=0.0, lat=mid, elevation=dem.get_elevation_or_raise(lon=0.0, lat=mid)),
        PathPoint(lon=0.0, lat=bot, elevation=dem.get_elevation_or_raise(lon=0.0, lat=bot)),
    ]
    graph.commit_paths(paths=[ProposedPathSegment(points=seg_a, target_difficulty="blue")])
    graph.commit_paths(paths=[ProposedPathSegment(points=seg_b, target_difficulty="blue")])
    return graph


class TestConfirmMergeAction:
    """confirm_merge_action validates the span, merges as one undoable action, returns to idle."""

    def test_close_nodes_merge_and_return_to_idle(self, fake_st, empty_graph, mock_dem_blue_slope) -> None:
        from skiresort_planner.ui.actions import confirm_merge_action

        dem = mock_dem_blue_slope
        _two_segment_slope(empty_graph, dem)
        by_lat = sorted(empty_graph.nodes.values(), key=lambda n: n.lat, reverse=True)
        top, mid = by_lat[0], by_lat[1]  # 300m apart, within MergeConfig.MAX_SPAN_M
        sm, ctx = _session(fake_st, empty_graph, dem=dem)
        count_before = len(empty_graph.nodes)
        sm.start_merge()
        sm.toggle_merge_node(node_id=top.id)
        sm.toggle_merge_node(node_id=mid.id)

        confirm_merge_action()

        assert len(empty_graph.nodes) == count_before - 1, "two close nodes collapsed into one"
        assert empty_graph.undo_stack[-1].action_type.name == "MERGE_NODES", "one undoable merge action"
        assert sm.is_idle_ready, "confirm returns to idle"
        assert ctx.merge.node_ids == [], "selection cleared by the before-hook"

    def test_far_nodes_refused_no_change(self, fake_st, empty_graph, mock_dem_blue_slope, monkeypatch) -> None:
        from skiresort_planner.ui.actions import confirm_merge_action

        dem = mock_dem_blue_slope
        _two_segment_slope(empty_graph, dem)
        by_lat = sorted(empty_graph.nodes.values(), key=lambda n: n.lat, reverse=True)
        top, bottom = by_lat[0], by_lat[-1]  # 600m apart, exceeds MergeConfig.MAX_SPAN_M
        sm, ctx = _session(fake_st, empty_graph, dem=dem)
        count_before = len(empty_graph.nodes)
        stack_before = len(empty_graph.undo_stack)
        sm.start_merge()
        sm.toggle_merge_node(node_id=top.id)
        sm.toggle_merge_node(node_id=bottom.id)

        # MergeTooFarMessage.display() does a function-local `import streamlit as st; st.toast(...)`,
        # so it hits the REAL streamlit module (not the fake `st`); capture it to prove the user is told.
        import streamlit

        toasts: list[str] = []
        monkeypatch.setattr(streamlit, "toast", lambda text, *a, **k: toasts.append(text))

        confirm_merge_action()

        assert len(empty_graph.nodes) == count_before, "nothing merged when the span is too large"
        assert len(empty_graph.undo_stack) == stack_before, "no undo action recorded on refusal"
        assert sm.is_merge_placing, "stays in merge so the user can adjust the selection"
        assert ctx.merge.node_ids == [top.id, bottom.id], "selection preserved for retry"
        assert any("too far" in t.lower() for t in toasts), "the user is told why the merge was refused"

    def test_fewer_than_two_nodes_raises(self, fake_st, empty_graph, mock_dem_blue_slope) -> None:
        from skiresort_planner.ui.actions import confirm_merge_action

        _session(fake_st, empty_graph, dem=mock_dem_blue_slope)
        with pytest.raises(RuntimeError, match="fewer than 2"):
            confirm_merge_action()

    def test_missing_lift_returns_false(self, fake_st, empty_graph) -> None:
        from skiresort_planner.ui.actions import delete_lift_action

        _session(fake_st, empty_graph)
        assert delete_lift_action(lift_id="L999") is False


class TestDeleteNodesAction:
    """delete_nodes_action deletes deletable nodes (return to idle) or refuses with a toast."""

    def test_interior_node_deletes_and_returns_to_idle(self, fake_st, empty_graph, mock_dem_blue_slope) -> None:
        from skiresort_planner.ui.actions import delete_nodes_action

        dem = mock_dem_blue_slope
        _two_segment_slope(empty_graph, dem)
        slope = empty_graph.finish_slope(segment_ids=list(empty_graph.segments.keys()))
        interior = empty_graph.segments[slope.segment_ids[0]].end_node_id
        sm, ctx = _session(fake_st, empty_graph, dem=dem)
        sm.start_merge()
        sm.toggle_merge_node(node_id=interior)

        delete_nodes_action()

        assert interior not in empty_graph.nodes, "the interior node was deleted"
        assert empty_graph.undo_stack[-1].action_type.name == "DELETE_NODES", "one DELETE_NODES undo action"
        assert sm.is_idle_ready, "delete returns to idle"
        assert ctx.merge.node_ids == [], "selection cleared by the before-hook"

    def test_lift_station_refused_no_change(self, fake_st, empty_graph, mock_dem_blue_slope, monkeypatch) -> None:
        from skiresort_planner.ui.actions import delete_nodes_action

        dem = mock_dem_blue_slope
        empty_graph.nodes["A"] = _node_at(dem, "A", 0.0, 0.0)
        empty_graph.nodes["T"] = _node_at(dem, "T", 0.0, -1000 / M)
        empty_graph.add_lift(start_node_id="A", end_node_id="T", lift_type="chairlift", dem=dem)
        sm, ctx = _session(fake_st, empty_graph, dem=dem)
        stack_before = len(empty_graph.undo_stack)
        sm.start_merge()
        sm.toggle_merge_node(node_id="A")

        import streamlit

        toasts: list[str] = []
        monkeypatch.setattr(streamlit, "toast", lambda text, *a, **k: toasts.append(text))

        delete_nodes_action()

        assert "A" in empty_graph.nodes, "a lift station is never deleted"
        assert len(empty_graph.undo_stack) == stack_before, "no undo action recorded on refusal"
        assert sm.is_merge_placing, "stays in merge so the user can adjust the selection"
        assert any("lift" in t.lower() for t in toasts), "the user is told why the delete was refused"

    def test_no_nodes_raises(self, fake_st, empty_graph, mock_dem_blue_slope) -> None:
        from skiresort_planner.ui.actions import delete_nodes_action

        _session(fake_st, empty_graph, dem=mock_dem_blue_slope)
        with pytest.raises(RuntimeError, match="no selected nodes"):
            delete_nodes_action()

    def test_deleting_whole_path_is_refused(self, fake_st, empty_graph, mock_dem_blue_slope, monkeypatch) -> None:
        """An end node + the sole interior node of a 2-segment slope are each individually deletable,
        but together they'd empty the path — refuse with a message and change nothing.
        """
        from skiresort_planner.ui.actions import delete_nodes_action

        dem = mock_dem_blue_slope
        _two_segment_slope(empty_graph, dem)
        slope = empty_graph.finish_slope(segment_ids=list(empty_graph.segments.keys()))
        end = slope.start_node_id
        interior = empty_graph.segments[slope.segment_ids[0]].end_node_id
        sm, ctx = _session(fake_st, empty_graph, dem=dem)
        stack_before = len(empty_graph.undo_stack)
        sm.start_merge()
        sm.toggle_merge_node(node_id=end)
        sm.toggle_merge_node(node_id=interior)

        import streamlit

        toasts: list[str] = []
        monkeypatch.setattr(streamlit, "toast", lambda text, *a, **k: toasts.append(text))

        delete_nodes_action()

        assert slope.id in empty_graph.slopes, "the path is not emptied"
        assert len(empty_graph.undo_stack) == stack_before, "no undo action recorded on refusal"
        assert sm.is_merge_placing, "stays in merge so the user can adjust the selection"
        assert any("whole path" in t.lower() for t in toasts), "the user is told the delete was refused"

    def test_branch_junction_refused_no_change(self, fake_st, empty_graph, mock_dem_blue_slope, monkeypatch) -> None:
        """A node shared by two slopes is a branch junction — deleting it is refused (delete a path
        first), nothing changes.
        """
        from skiresort_planner.ui.actions import delete_nodes_action

        dem = mock_dem_blue_slope
        # Slope 1 south to a junction; slope 2 branches south-east from that same node.
        leg1 = [
            PathPoint(lon=0.0, lat=0.0, elevation=dem.get_elevation_or_raise(lon=0.0, lat=0.0)),
            PathPoint(lon=0.0, lat=-400 / M, elevation=dem.get_elevation_or_raise(lon=0.0, lat=-400 / M)),
        ]
        empty_graph.commit_paths(paths=[ProposedPathSegment(points=leg1, target_difficulty="blue")])
        slope1 = empty_graph.finish_slope(segment_ids=list(empty_graph.segments.keys()))
        junction = slope1.end_node_id
        j = empty_graph.nodes[junction]
        leg2 = [
            PathPoint(lon=j.lon, lat=j.lat, elevation=j.elevation),
            PathPoint(
                lon=400 / M, lat=j.lat - 400 / M, elevation=dem.get_elevation_or_raise(lon=400 / M, lat=j.lat - 400 / M)
            ),
        ]
        before = set(empty_graph.segments)
        empty_graph.commit_paths(paths=[ProposedPathSegment(points=leg2, target_difficulty="blue")])
        empty_graph.finish_slope(segment_ids=list(set(empty_graph.segments) - before))
        sm, ctx = _session(fake_st, empty_graph, dem=dem)
        stack_before = len(empty_graph.undo_stack)
        sm.start_merge()
        sm.toggle_merge_node(node_id=junction)

        import streamlit

        toasts: list[str] = []
        monkeypatch.setattr(streamlit, "toast", lambda text, *a, **k: toasts.append(text))

        delete_nodes_action()

        assert junction in empty_graph.nodes, "a branch junction is not deleted"
        assert len(empty_graph.undo_stack) == stack_before, "no undo action recorded on refusal"
        assert sm.is_merge_placing, "stays in merge so the user can adjust the selection"
        assert any("delete that path" in t.lower() for t in toasts), "the user is told to delete a path first"


class TestAddNodeOnPathAction:
    """add_node_on_path_action returns True (inserted) / False (rejected) — the bool the click
    handlers gate their state transition on — and shows an InvalidClickMessage on rejection.
    """

    def test_success_returns_true_inserts_node_and_bumps_epoch(
        self, fake_st, empty_graph, mock_dem_blue_slope, path_points_blue
    ) -> None:
        from skiresort_planner.ui.actions import add_node_on_path_action

        slope = _make_slope(empty_graph, path_points_blue)
        seg_id = slope.segment_ids[0]
        mid = empty_graph.segments[seg_id].points[len(empty_graph.segments[seg_id].points) // 2]
        _session(fake_st, empty_graph, dem=mock_dem_blue_slope)
        nodes_before = len(empty_graph.nodes)
        epoch_before = fake_st.session_state["dedup_epoch"]

        result = add_node_on_path_action(segment_id=seg_id, lon=mid.lon, lat=mid.lat)

        assert result is True, "a successful insert returns True (callers gate the transition on it)"
        assert len(empty_graph.nodes) == nodes_before + 1, "one node inserted"
        assert seg_id not in empty_graph.segments, "the clicked segment was split"
        assert fake_st.session_state["dedup_epoch"] > epoch_before, "insert refreshes the map"

    def test_rejected_returns_false_changes_nothing_and_toasts(
        self, fake_st, empty_graph, mock_dem_blue_slope, path_points_blue, monkeypatch
    ) -> None:
        from skiresort_planner.ui.actions import add_node_on_path_action

        slope = _make_slope(empty_graph, path_points_blue)
        seg_id = slope.segment_ids[0]
        near_end = empty_graph.segments[seg_id].points[0]  # within STEP_SIZE_M of the endpoint node
        _session(fake_st, empty_graph, dem=mock_dem_blue_slope)
        nodes_before = len(empty_graph.nodes)

        import streamlit

        toasts: list[str] = []
        monkeypatch.setattr(streamlit, "toast", lambda text, *a, **k: toasts.append(text))

        result = add_node_on_path_action(segment_id=seg_id, lon=near_end.lon, lat=near_end.lat)

        assert result is False, "a rejected insert returns False (callers must NOT transition)"
        assert len(empty_graph.nodes) == nodes_before, "nothing inserted"
        assert seg_id in empty_graph.segments, "the segment is untouched"
        assert any("add a node" in t.lower() for t in toasts), "the user is told why"


class TestUndoLastActionDispatch:
    """undo_last_action ROUTING only — the per-entity graph undo end-state is
    owned by test_resort_graph. Here we assert the action layer pops the stack,
    dispatches to a handler, and honors the empty-stack + slope-cancel guards.
    """

    def test_dispatch_pops_the_stack(self, fake_st, empty_graph) -> None:
        """A committed road leaves an undo entry; undo_last_action consumes it."""
        from skiresort_planner.ui.actions import undo_last_action

        _make_road(empty_graph)
        _session(fake_st, empty_graph)
        assert len(empty_graph.undo_stack) == 1

        undo_last_action()
        assert empty_graph.undo_stack == [], "dispatch must pop the undone action"

    def test_dispatch_routes_finish_slope_needs_factory(
        self, fake_st, path_factory, mock_dem_red_slope_diagonal, empty_graph, path_points_blue
    ) -> None:
        """FINISH_SLOPE routes to the slope handler (which reads path_factory)."""
        from skiresort_planner.ui.actions import undo_last_action

        slope = _make_slope(empty_graph, path_points_blue)  # ADD_SEGMENTS + FINISH_SLOPE
        _session(fake_st, empty_graph, factory=path_factory, dem=mock_dem_red_slope_diagonal)

        undo_last_action()  # undo FINISH_SLOPE via the dispatch
        assert slope.id not in empty_graph.slopes, "routed to the finish-slope undo handler"

    def test_empty_stack_is_noop(self, fake_st, empty_graph) -> None:
        """Guard: undo_last_action on an empty stack does nothing and never raises."""
        from skiresort_planner.ui.actions import undo_last_action

        _session(fake_st, empty_graph)
        undo_last_action()
        assert empty_graph.undo_stack == []

    def test_undo_in_slope_starting_cancels_slope_not_stack(self, fake_st, empty_graph, path_points_blue) -> None:
        """In slope_starting (0 segments) Undo cancels the slope, NOT an unrelated stack entry."""
        from skiresort_planner.ui.actions import undo_last_action

        _make_road(empty_graph)  # an unrelated FINISH_ROAD entry sits on the stack
        sm, _ctx = _session(fake_st, empty_graph)
        sm.start_slope(lon=0.0, lat=0.0, elevation=2500.0, node_id=None)
        assert sm.is_slope_starting

        undo_last_action()
        assert sm.is_idle, "undo in slope_starting cancels the slope"
        assert len(empty_graph.undo_stack) == 1, "the unrelated road entry must NOT be consumed"

    def test_undo_in_road_starting_cancels_road_not_stack(self, fake_st, empty_graph, path_points_blue) -> None:
        """In road_starting (0 segments) Undo cancels the road, NOT an unrelated stack entry.

        Mirror of the slope guard — regression for the missing road short-circuit.
        """
        from skiresort_planner.ui.actions import undo_last_action

        _make_slope(empty_graph, path_points_blue)  # unrelated ADD_SEGMENTS + FINISH_SLOPE
        stack_before = len(empty_graph.undo_stack)
        sm, _ctx = _session(fake_st, empty_graph)
        sm.start_road(node_id=None, location=path_points_blue[0])
        assert sm.is_road_starting

        undo_last_action()
        assert sm.is_idle, "undo in road_starting cancels the road"
        assert len(empty_graph.undo_stack) == stack_before, "the unrelated slope entries must NOT be consumed"


class TestCenterHelpers:
    """center_on_* set the map to the entity midpoint at the given zoom."""

    def test_center_on_slope_sets_map(self, empty_graph, path_points_blue) -> None:
        from skiresort_planner.ui.actions import center_on_slope

        slope = _make_slope(empty_graph, path_points_blue)
        _sm, ctx = PlannerStateMachine.create(graph=empty_graph, add_ui_listener=False)
        center_on_slope(ctx=ctx, graph=empty_graph, slope=slope, zoom=15)

        start_pt = empty_graph.segments[slope.segment_ids[0]].points[0]
        end_pt = empty_graph.segments[slope.segment_ids[-1]].points[-1]
        assert ctx.map.zoom == 15
        assert ctx.map.lon == (start_pt.lon + end_pt.lon) / 2, "centered on the path midpoint"
        assert ctx.map.lat == (start_pt.lat + end_pt.lat) / 2
        assert ctx.map.pitch == MapConfig.VIEWING_PITCH

    def test_center_on_road_sets_map(self, empty_graph) -> None:
        from skiresort_planner.ui.actions import center_on_road

        road = _make_road(empty_graph)
        _sm, ctx = PlannerStateMachine.create(graph=empty_graph, add_ui_listener=False)
        center_on_road(ctx=ctx, graph=empty_graph, road=road, zoom=16)

        start_pt = empty_graph.segments[road.segment_ids[0]].points[0]
        end_pt = empty_graph.segments[road.segment_ids[-1]].points[-1]
        assert ctx.map.zoom == 16
        assert ctx.map.lon == (start_pt.lon + end_pt.lon) / 2, "centered on the road midpoint"
        assert ctx.map.lat == (start_pt.lat + end_pt.lat) / 2
        assert ctx.map.pitch == MapConfig.VIEWING_PITCH

    def test_center_on_lift_sets_map(self, empty_graph, mock_dem_blue_slope) -> None:
        from skiresort_planner.ui.actions import center_on_lift

        dem = mock_dem_blue_slope
        M = 111320.0
        bottom, _ = empty_graph.get_or_create_node(
            lon=0.0, lat=-1000 / M, elevation=dem.get_elevation_or_raise(lon=0.0, lat=-1000 / M)
        )
        top, _ = empty_graph.get_or_create_node(
            lon=0.0, lat=0.0, elevation=dem.get_elevation_or_raise(lon=0.0, lat=0.0)
        )
        lift = empty_graph.add_lift(start_node_id=bottom.id, end_node_id=top.id, lift_type="chairlift", dem=dem)
        _sm, ctx = PlannerStateMachine.create(graph=empty_graph, add_ui_listener=False)

        center_on_lift(ctx=ctx, graph=empty_graph, lift=lift, zoom=14)
        assert ctx.map.zoom == 14
        assert ctx.map.lon == (bottom.lon + top.lon) / 2, "centered on the lift-station midpoint"
        assert ctx.map.lat == (bottom.lat + top.lat) / 2
        assert ctx.map.pitch == MapConfig.VIEWING_PITCH


class TestSelectLiftTypeAction:
    """The sidebar lift-type buttons set the build mode, and re-type the viewed lift."""

    def test_sets_build_mode_when_not_viewing_a_lift(self, fake_st, empty_graph) -> None:
        from skiresort_planner.ui.actions import select_lift_type_action

        _sm, ctx = _session(fake_st, empty_graph)
        select_lift_type_action(lift_type="gondola")

        assert ctx.build_mode.mode == "gondola", "the next lift will be built as a gondola"

    def test_retypes_the_viewed_lift(self, fake_st, empty_graph, mock_dem_blue_slope) -> None:
        from skiresort_planner.ui.actions import select_lift_type_action

        dem = mock_dem_blue_slope
        bottom, _ = empty_graph.get_or_create_node(
            lon=0.0, lat=-1000 / M, elevation=dem.get_elevation_or_raise(lon=0.0, lat=-1000 / M)
        )
        top, _ = empty_graph.get_or_create_node(
            lon=0.0, lat=0.0, elevation=dem.get_elevation_or_raise(lon=0.0, lat=0.0)
        )
        lift = empty_graph.add_lift(start_node_id=bottom.id, end_node_id=top.id, lift_type="chairlift", dem=dem)
        sm, ctx = _session(fake_st, empty_graph, dem=dem)
        sm.view_lift(lift_id=lift.id)

        select_lift_type_action(lift_type="gondola")

        assert empty_graph.lifts[lift.id].lift_type == "gondola", "update_type re-typed the viewed lift"
        assert ctx.build_mode.mode == "gondola"


class TestSlopeBuildingActionFlow:
    """The slope-building action entry points, driven via the fake session.

    Uses a real PathFactory + DEM so commit/recompute/finish exercise the true
    generate → commit → finish path, not hand-built stubs.
    """

    def _start_building(self, fake_st, factory, dem):
        graph = ResortGraph()
        sm, ctx = _session(fake_st, graph, factory=factory, dem=dem)
        start_elev = dem.get_elevation_or_raise(lon=0.0, lat=0.0)
        sm.start_slope(lon=0.0, lat=0.0, elevation=start_elev, node_id=None)
        ctx.selection.set(lon=0.0, lat=0.0, elevation=start_elev)
        return sm, ctx, graph

    def test_recompute_then_commit_then_finish(self, fake_st, path_factory, mock_dem_red_slope_diagonal) -> None:
        from skiresort_planner.ui.actions import commit_selected_path, finish_current_slope, recompute_paths

        dem = mock_dem_red_slope_diagonal
        sm, ctx, graph = self._start_building(fake_st, path_factory, dem)

        recompute_paths()
        assert ctx.proposals.paths, "recompute must generate fan proposals"

        commit_selected_path(path_idx=0)
        assert ctx.build(SegmentKind.SLOPE).segments, "commit must add a segment to the building context"

        finish_current_slope()
        assert sm.is_idle_viewing_slope
        assert len(graph.slopes) == 1

    def test_cancel_current_slope_discards(self, fake_st, path_factory, mock_dem_red_slope_diagonal) -> None:
        from skiresort_planner.ui.actions import cancel_current_slope, commit_selected_path, recompute_paths

        dem = mock_dem_red_slope_diagonal
        sm, ctx, graph = self._start_building(fake_st, path_factory, dem)
        recompute_paths()
        commit_selected_path(path_idx=0)

        cancel_current_slope()
        assert sm.is_idle
        assert len(graph.slopes) == 0, "canceling discards the in-progress slope"

    def test_finish_then_undo_restores_slope_building(self, fake_st, path_factory, mock_dem_red_slope_diagonal) -> None:
        # Finishing then undoing the finish must return to slope_building with segments + a regenerated fan.
        from skiresort_planner.ui.actions import (
            commit_selected_path,
            finish_current_slope,
            process_path_generation_deferred,
            recompute_paths,
            undo_last_action,
        )

        dem = mock_dem_red_slope_diagonal
        sm, ctx, graph = self._start_building(fake_st, path_factory, dem)
        recompute_paths()
        commit_selected_path(path_idx=0)
        seg_id = ctx.build(SegmentKind.SLOPE).segments[-1]
        finish_current_slope()
        assert sm.is_idle_viewing_slope

        undo_last_action()  # undo FINISH_SLOPE
        assert sm.is_slope_building_only, "undo of finish returns to slope building"
        assert ctx.build(SegmentKind.SLOPE).segments == [seg_id], "segments are restored"
        # force_building arms the fan on the deferred pass (unified with the live flow).
        process_path_generation_deferred()
        assert ctx.proposals.paths, "the fan is regenerated from the restored endpoint"


class TestRoadBuildingActionFlow:
    """Roads commit through the SAME commit_selected_path as slopes (no fan, no
    connector auto-finish): a road-state commit fires the commit_road event and
    stays in road_building with a road-kind segment + per-segment undo.
    """

    def test_commit_selected_path_in_road_state_commits_segment(
        self, fake_st, path_factory, mock_dem_red_slope_diagonal, path_points_blue
    ) -> None:
        from skiresort_planner.model.path_segment import SegmentKind
        from skiresort_planner.ui.actions import commit_selected_path

        dem = mock_dem_red_slope_diagonal
        graph = ResortGraph()
        sm, ctx = _session(fake_st, graph, factory=path_factory, dem=dem)
        sm.start_road(node_id=None, location=path_points_blue[0])

        # Seed a road proposal (as handle_path_building_click would) and commit it.
        ctx.proposals.paths = [ProposedPathSegment(points=path_points_blue, is_connector=True, kind=SegmentKind.ROAD)]
        ctx.proposals.selected_idx = 0
        commit_selected_path(path_idx=0)

        assert sm.is_road_building_only, "road commit stays in road_building"
        assert len(ctx.build(SegmentKind.ROAD).segments) == 1
        assert len(graph.roads) == 0, "no Road entity until Finish Road"
        assert graph.segments[ctx.build(SegmentKind.ROAD).segments[-1]].kind == SegmentKind.ROAD
        assert graph.undo_stack[-1].action_type.name == "ADD_SEGMENTS", "per-segment undo recorded"

    def test_finish_then_undo_restores_road_building(
        self, fake_st, path_factory, mock_dem_red_slope_diagonal, path_points_blue
    ) -> None:
        # Finishing then undoing the finish must return to road_building with segments (no fan).
        from skiresort_planner.model.path_segment import SegmentKind
        from skiresort_planner.ui.actions import commit_selected_path, finish_current_road, undo_last_action

        dem = mock_dem_red_slope_diagonal
        graph = ResortGraph()
        sm, ctx = _session(fake_st, graph, factory=path_factory, dem=dem)
        sm.start_road(node_id=None, location=path_points_blue[0])
        ctx.proposals.paths = [ProposedPathSegment(points=path_points_blue, is_connector=True, kind=SegmentKind.ROAD)]
        ctx.proposals.selected_idx = 0
        commit_selected_path(path_idx=0)
        seg_id = ctx.build(SegmentKind.ROAD).segments[-1]
        finish_current_road()
        assert sm.is_idle_viewing_road

        undo_last_action()  # undo FINISH_ROAD
        assert sm.is_road_building_only, "undo of finish returns to road building"
        assert ctx.build(SegmentKind.ROAD).segments == [seg_id], "segments are restored"
        assert ctx.proposals.paths == [], "roads have no fan to regenerate"

    def test_connector_proposal_auto_finishes_to_viewing(
        self, fake_st, path_factory, mock_dem_red_slope_diagonal, path_points_blue
    ) -> None:
        # A proposal that IS a connector (is_connector AND target_node_id set), committed from the
        # custom-path state, must NOT stay in building — commit_selected_path routes it through
        # _finish_connector → the Road entity is created and the machine lands in idle_viewing_road.
        # This is the branch the other road tests miss (they leave target_node_id empty → continue).
        from skiresort_planner.model.path_segment import SegmentKind
        from skiresort_planner.ui.actions import commit_selected_path

        dem = mock_dem_red_slope_diagonal
        graph = ResortGraph()
        sm, ctx = _session(fake_st, graph, factory=path_factory, dem=dem)
        sm.start_road(node_id=None, location=path_points_blue[0])

        # Commit one real fan segment so we're in road_building with a target node to connect to.
        first = ProposedPathSegment(points=path_points_blue, kind=SegmentKind.ROAD)
        end_ids = graph.commit_paths(paths=[first])
        seg0 = list(graph.segments.keys())[-1]
        sm.commit_road(segment_id=seg0, endpoint_node_id=end_ids[0])
        assert sm.is_road_building_only

        # Route to a custom target → road_custom_path, then commit a CONNECTOR proposal onto an
        # existing node (target_node_id set) → auto-finish.
        target_node_id = end_ids[0]
        target = graph.nodes[target_node_id]
        sm.select_custom_target(target_location=(target.lon, target.lat, target.elevation))
        assert sm.is_road_custom_path

        connector = ProposedPathSegment(
            points=path_points_blue, is_connector=True, target_node_id=target_node_id, kind=SegmentKind.ROAD
        )
        ctx.proposals.paths = [connector]
        ctx.proposals.selected_idx = 0
        commit_selected_path(path_idx=0)

        assert sm.is_idle_viewing_road, "a real connector auto-finishes to the viewing state"
        assert len(graph.roads) == 1, "the Road entity was created by the connector auto-finish"
        assert ctx.viewing.road_id is not None, "the finished road is being viewed"


class TestDeferredProcessing:
    """Deferred-action processors read/clear ctx.deferred flags and act on them."""

    def test_process_path_generation_noop_when_not_pending(
        self, fake_st, path_factory, mock_dem_red_slope_diagonal
    ) -> None:
        from skiresort_planner.ui.actions import process_path_generation_deferred

        _sm, ctx = _session(fake_st, ResortGraph(), factory=path_factory, dem=mock_dem_red_slope_diagonal)
        ctx.deferred.fan_generation.discard(SegmentKind.SLOPE)
        assert process_path_generation_deferred() is False

    def test_process_path_generation_builds_fan_when_pending(
        self, fake_st, path_factory, mock_dem_red_slope_diagonal
    ) -> None:
        from skiresort_planner.ui.actions import process_path_generation_deferred

        dem = mock_dem_red_slope_diagonal
        graph = ResortGraph()
        sm, ctx = _session(fake_st, graph, factory=path_factory, dem=dem)
        start_elev = dem.get_elevation_or_raise(lon=0.0, lat=0.0)
        sm.start_slope(lon=0.0, lat=0.0, elevation=start_elev, node_id=None)
        ctx.selection.set(lon=0.0, lat=0.0, elevation=start_elev)
        ctx.deferred.fan_generation.add(SegmentKind.SLOPE)

        assert process_path_generation_deferred() is True
        assert SegmentKind.SLOPE not in ctx.deferred.fan_generation, "flag cleared after processing"
        assert ctx.proposals.paths, "fan proposals generated for the building state"

    def test_process_custom_connect_noop_when_not_pending(
        self, fake_st, path_factory, mock_dem_red_slope_diagonal
    ) -> None:
        from skiresort_planner.ui.actions import process_custom_connect_deferred

        _sm, ctx = _session(fake_st, ResortGraph(), factory=path_factory, dem=mock_dem_red_slope_diagonal)
        ctx.deferred.custom_connect = False
        assert process_custom_connect_deferred() is False

    def test_custom_connect_orders_shortest_first_straight_last(
        self, fake_st, monkeypatch, path_factory, mock_dem_red_slope_diagonal
    ) -> None:
        # Custom-connect sorts serpentine proposals SHORTEST→longest, appends the straight line LAST,
        # and pre-selects the shortest (index 0) — NOT the gradient-closest (that's the fan's rule).
        from skiresort_planner.ui import actions

        dem = mock_dem_red_slope_diagonal
        sm, ctx = _session(fake_st, ResortGraph(), factory=path_factory, dem=dem)
        start_elev = dem.get_elevation_or_raise(lon=0.0, lat=0.0)
        sm.start_slope(lon=0.0, lat=0.0, elevation=start_elev, node_id=None)
        ctx.selection.set(lon=0.0, lat=0.0, elevation=start_elev)

        # A run of N points stepping south by `step` metres each — length scales with (N-1)*step.
        def _seg(n: int, step: float) -> ProposedPathSegment:
            pts = [PathPoint(lon=0.0, lat=-(i * step) / M, elevation=start_elev - i * step * 0.1) for i in range(n)]
            return ProposedPathSegment(points=pts, is_connector=False)

        long_route, short_route = _seg(6, 100.0), _seg(3, 100.0)  # ~500m vs ~200m, out of order
        straight = _seg(2, 150.0)  # the straight line, appended last regardless of length
        monkeypatch.setattr(path_factory, "generate_manual_paths", lambda **_: [long_route, short_route])
        monkeypatch.setattr(path_factory, "straight_line", lambda **_: straight)

        ctx.custom_connect.target_location = (0.0, -500 / M, dem.get_elevation_or_raise(lon=0.0, lat=-500 / M))
        ctx.deferred.gradient_target = 99.0  # a stale fan target must be IGNORED by custom-connect
        ctx.deferred.custom_connect = True
        assert actions.process_custom_connect_deferred() is True

        paths = ctx.proposals.paths
        lengths = [p.length_m for p in paths]
        assert lengths[:2] == sorted(lengths[:2]), "serpentine proposals ordered shortest-first"
        assert paths[0] is short_route, "shortest route is first"
        assert paths[-1] is straight, "straight line appended last"
        assert ctx.proposals.selected_idx == 0, "shortest route pre-selected (not gradient-closest)"
        assert ctx.deferred.gradient_target is None, "stale fan gradient target consumed/ignored"


class TestGradientPreselection:
    """_preselect_by_rule — the fan passes the closest-gradient rule (grade continuity across
    committed segments); custom-connect passes a shortest-first (index 0) rule. Both always
    consume the one-shot gradient_target.
    """

    def _paths(self, *slopes: float) -> "list[ProposedPathSegment]":
        # Two-point segments whose avg_slope_pct is the given grade (100m run).
        out = []
        for s in slopes:
            pts = [PathPoint(lon=0.0, lat=0.0, elevation=1000.0), PathPoint(lon=0.001, lat=0.0, elevation=1000.0 - s)]
            out.append(ProposedPathSegment(points=pts, kind=SegmentKind.SLOPE))
        return out

    def test_gradient_rule_preselects_closest_and_consumes_target(self, fake_st, mock_dem_blue_slope) -> None:
        from skiresort_planner.ui.actions import _closest_gradient_rule, _preselect_by_rule

        _sm, ctx = _session(fake_st, ResortGraph(), dem=mock_dem_blue_slope)
        paths = self._paths(5.0, 18.0, 30.0)
        ctx.deferred.gradient_target = 17.0  # closest to the 18% path (index 1)
        _preselect_by_rule(ctx=ctx, paths=paths, rule=_closest_gradient_rule(ctx))
        assert ctx.proposals.selected_idx == 1, "pre-selects the proposal nearest the last committed grade"
        assert ctx.deferred.gradient_target is None, "one-shot: the target is consumed"

    def test_gradient_rule_defaults_to_first_without_target(self, fake_st, mock_dem_blue_slope) -> None:
        from skiresort_planner.ui.actions import _closest_gradient_rule, _preselect_by_rule

        _sm, ctx = _session(fake_st, ResortGraph(), dem=mock_dem_blue_slope)
        ctx.deferred.gradient_target = None
        _preselect_by_rule(ctx=ctx, paths=self._paths(5.0, 18.0), rule=_closest_gradient_rule(ctx))
        assert ctx.proposals.selected_idx == 0, "no target → first proposal"

    def test_shortest_rule_selects_index_zero(self, fake_st, mock_dem_blue_slope) -> None:
        from skiresort_planner.ui.actions import _preselect_by_rule, _shortest_rule

        _sm, ctx = _session(fake_st, ResortGraph(), dem=mock_dem_blue_slope)
        ctx.deferred.gradient_target = 17.0  # a stale fan target must still be consumed
        _preselect_by_rule(ctx=ctx, paths=self._paths(5.0, 18.0), rule=_shortest_rule)
        assert ctx.proposals.selected_idx == 0, "custom-connect shortest-first → index 0"
        assert ctx.deferred.gradient_target is None, "stale fan target consumed even for the shortest rule"

    def test_none_when_no_paths(self, fake_st, mock_dem_blue_slope) -> None:
        from skiresort_planner.ui.actions import _closest_gradient_rule, _preselect_by_rule

        _sm, ctx = _session(fake_st, ResortGraph(), dem=mock_dem_blue_slope)
        ctx.deferred.gradient_target = 17.0
        _preselect_by_rule(ctx=ctx, paths=[], rule=_closest_gradient_rule(ctx))
        assert ctx.proposals.selected_idx is None, "empty proposals → no selection"


class TestOSMImport:
    """Click-to-place import: start_import stores the box center; confirm_import_action flags the
    deferred fetch + returns to idle; process_osm_import_deferred runs it (mocked network) as one
    undoable batch centered on the placed box; undo removes the batch; re-import dedups.
    """

    def test_start_import_stores_center_and_confirm_flags_deferred(self, fake_st, mock_dem_blue_slope) -> None:
        from skiresort_planner.ui.actions import confirm_import_action

        sm, ctx = _session(fake_st, ResortGraph(), dem=mock_dem_blue_slope)
        sm.start_import(lon=0.1, lat=0.3)  # first map click places the box center
        assert sm.is_import_placing
        assert ctx.deferred.osm_import_center_lon == 0.1 and ctx.deferred.osm_import_center_lat == 0.3

        confirm_import_action()  # center-dot click / Confirm button
        assert ctx.deferred.osm_import is True
        assert sm.is_idle_ready, "confirm returns to idle so the deferred fetch runs under the spinner"

    def test_placed_center_reaches_fetch_as_bbox(self, fake_st, mock_dem_blue_slope, monkeypatch) -> None:
        """The placed box center + half-width must arrive at fetch() as a square bbox around it."""
        from skiresort_planner.generators.osm_importer import ImportSummary
        from skiresort_planner.ui import actions
        from skiresort_planner.ui.actions import confirm_import_action

        sm, ctx = _session(fake_st, ResortGraph(), dem=mock_dem_blue_slope)
        ctx.map.lat, ctx.map.lon = 0.9, 0.9  # deliberately NOT the placed center — must be ignored

        seen: dict[str, tuple[float, float, float, float]] = {}

        def _record_fetch(self: object, bbox: tuple[float, float, float, float]) -> list[object]:
            seen["bbox"] = bbox
            return []

        monkeypatch.setattr("skiresort_planner.ui.actions.OSMImporter.fetch", _record_fetch)
        monkeypatch.setattr(
            "skiresort_planner.ui.actions.OSMImporter.convert", lambda self, bbox, elements: ImportSummary()
        )

        sm.start_import(lon=0.1, lat=0.3)  # placed center
        ctx.deferred.osm_import_half_width_km = 3.5
        confirm_import_action()
        actions.process_osm_import_deferred()

        min_lon, min_lat, max_lon, max_lat = seen["bbox"]
        assert (min_lon + max_lon) / 2 == 0.1 and (min_lat + max_lat) / 2 == 0.3, "box centered on the PLACED center"
        assert max_lat - min_lat > 0 and max_lon - min_lon > 0, "3.5 km half-width → a real square box"

    def test_process_without_placed_center_raises(self, fake_st, mock_dem_blue_slope) -> None:
        """A pending import with no placed center is a bug — no silent map-center fallback."""
        from skiresort_planner.ui import actions

        _sm, ctx = _session(fake_st, ResortGraph(), dem=mock_dem_blue_slope)
        ctx.deferred.osm_import = True  # flagged, but no center placed
        with pytest.raises(RuntimeError, match="no placed center"):
            actions.process_osm_import_deferred()

    def test_process_import_adds_entities_and_bumps_map(self, fake_st, mock_dem_blue_slope, monkeypatch) -> None:
        from skiresort_planner.generators.osm_importer import ImportSummary, LiftImport, PisteImport
        from skiresort_planner.model.path_point import PathPoint
        from skiresort_planner.ui import actions

        dem = mock_dem_blue_slope
        graph = ResortGraph()
        _sm, ctx = _session(fake_st, graph, dem=dem)
        ctx.deferred.osm_import = True
        ctx.deferred.osm_import_center_lon = 0.0  # inside MockDEM bounds (-1..1)
        ctx.deferred.osm_import_center_lat = 0.0

        piste = PisteImport(
            points=[
                PathPoint(lon=0.0, lat=0.0, elevation=dem.get_elevation_or_raise(lon=0.0, lat=0.0)),
                PathPoint(lon=0.0, lat=-500 / M, elevation=dem.get_elevation_or_raise(lon=0.0, lat=-500 / M)),
            ],
            name="Imported Run",
        )
        lift = LiftImport(
            bottom=PathPoint(lon=0.02, lat=-500 / M, elevation=dem.get_elevation_or_raise(lon=0.02, lat=-500 / M)),
            top=PathPoint(lon=0.02, lat=0.0, elevation=dem.get_elevation_or_raise(lon=0.02, lat=0.0)),
            lift_type="chairlift",
            name=None,
        )

        # Mock the importer so no network happens: fetch returns nothing, convert returns our summary.
        monkeypatch.setattr("skiresort_planner.ui.actions.OSMImporter.fetch", lambda self, bbox: [])
        monkeypatch.setattr(
            "skiresort_planner.ui.actions.OSMImporter.convert",
            lambda self, bbox, elements: ImportSummary(pistes=[piste], lifts=[lift]),
        )
        epoch_before = fake_st.session_state["dedup_epoch"]

        handled = actions.process_osm_import_deferred()

        assert handled is True
        assert len(graph.slopes) == 1 and len(graph.lifts) == 1
        assert len(graph.undo_stack) == 1, "import is one undoable batch"
        assert fake_st.session_state["dedup_epoch"] > epoch_before, "import redraws new geometry (no recenter)"
        assert ctx.deferred.osm_import is False, "flag consumed"
        assert ctx.deferred.osm_import_center_lon is None, "placed center consumed"

    def test_process_import_network_error_reports_and_imports_nothing(
        self, fake_st, mock_dem_blue_slope, monkeypatch
    ) -> None:
        from skiresort_planner.ui import actions

        graph = ResortGraph()
        _sm, ctx = _session(fake_st, graph, dem=mock_dem_blue_slope)
        ctx.deferred.osm_import = True
        ctx.deferred.osm_import_center_lon = 0.0
        ctx.deferred.osm_import_center_lat = 0.0

        def boom(self, bbox):
            raise RuntimeError("overpass down")

        monkeypatch.setattr("skiresort_planner.ui.actions.OSMImporter.fetch", boom)

        handled = actions.process_osm_import_deferred()

        assert handled is True
        assert len(graph.slopes) == 0 and len(graph.lifts) == 0
        assert len(graph.undo_stack) == 0, "a network error imports nothing"

    def test_undo_last_action_reverts_whole_import(self, fake_st, mock_dem_blue_slope, monkeypatch) -> None:
        """The headline promise: one Undo (via the UI dispatcher) removes the entire import.

        Exercises undo_last_action() — the dispatcher a direct graph.undo_last() call bypasses.
        """
        from skiresort_planner.generators.osm_importer import ImportSummary, LiftImport, PisteImport
        from skiresort_planner.model.path_point import PathPoint
        from skiresort_planner.ui import actions
        from skiresort_planner.ui.actions import undo_last_action

        dem = mock_dem_blue_slope
        graph = ResortGraph()
        _sm, ctx = _session(fake_st, graph, dem=dem)
        ctx.deferred.osm_import = True
        ctx.deferred.osm_import_center_lon = 0.0
        ctx.deferred.osm_import_center_lat = 0.0

        piste = PisteImport(
            points=[
                PathPoint(lon=0.0, lat=0.0, elevation=dem.get_elevation_or_raise(lon=0.0, lat=0.0)),
                PathPoint(lon=0.0, lat=-500 / M, elevation=dem.get_elevation_or_raise(lon=0.0, lat=-500 / M)),
            ],
            name="Imported Run",
        )
        lift = LiftImport(
            bottom=PathPoint(lon=0.02, lat=-500 / M, elevation=dem.get_elevation_or_raise(lon=0.02, lat=-500 / M)),
            top=PathPoint(lon=0.02, lat=0.0, elevation=dem.get_elevation_or_raise(lon=0.02, lat=0.0)),
            lift_type="chairlift",
            name=None,
        )
        monkeypatch.setattr("skiresort_planner.ui.actions.OSMImporter.fetch", lambda self, bbox: [])
        monkeypatch.setattr(
            "skiresort_planner.ui.actions.OSMImporter.convert",
            lambda self, bbox, elements: ImportSummary(pistes=[piste], lifts=[lift]),
        )

        actions.process_osm_import_deferred()
        assert len(graph.slopes) == 1 and len(graph.lifts) == 1

        undo_last_action()  # dispatch IMPORT_OSM — must not raise, must wipe the batch

        assert len(graph.slopes) == 0 and len(graph.lifts) == 0
        assert len(graph.segments) == 0 and len(graph.nodes) == 0
        assert len(graph.undo_stack) == 0

    def test_reimport_same_area_adds_nothing(self, fake_st, mock_dem_blue_slope, monkeypatch) -> None:
        """Importing the same area twice adds entities once, then dedups the rest."""
        from skiresort_planner.generators.osm_importer import ImportSummary, LiftImport, PisteImport
        from skiresort_planner.model.path_point import PathPoint
        from skiresort_planner.ui import actions

        dem = mock_dem_blue_slope
        graph = ResortGraph()
        _sm, ctx = _session(fake_st, graph, dem=dem)

        piste = PisteImport(
            points=[
                PathPoint(lon=0.0, lat=0.0, elevation=dem.get_elevation_or_raise(lon=0.0, lat=0.0)),
                PathPoint(lon=0.0, lat=-500 / M, elevation=dem.get_elevation_or_raise(lon=0.0, lat=-500 / M)),
            ],
            name="Imported Run",
        )
        lift = LiftImport(
            bottom=PathPoint(lon=0.02, lat=-500 / M, elevation=dem.get_elevation_or_raise(lon=0.02, lat=-500 / M)),
            top=PathPoint(lon=0.02, lat=0.0, elevation=dem.get_elevation_or_raise(lon=0.02, lat=0.0)),
            lift_type="chairlift",
            name="Gipfelbahn",
        )
        monkeypatch.setattr("skiresort_planner.ui.actions.OSMImporter.fetch", lambda self, bbox: [])
        monkeypatch.setattr(
            "skiresort_planner.ui.actions.OSMImporter.convert",
            lambda self, bbox, elements: ImportSummary(pistes=[piste], lifts=[lift]),
        )

        def _flag_import() -> None:
            ctx.deferred.osm_import = True
            ctx.deferred.osm_import_center_lon = 0.0
            ctx.deferred.osm_import_center_lat = 0.0

        _flag_import()
        actions.process_osm_import_deferred()
        assert len(graph.slopes) == 1 and len(graph.lifts) == 1

        _flag_import()
        actions.process_osm_import_deferred()  # same area again
        assert len(graph.slopes) == 1 and len(graph.lifts) == 1, "no duplicates on re-import"


class TestSegmentOrigin:
    """_segment_origin resolves the point a fan radiates from.

    No origin node is materialised before commit, so start_node_id is either a LIVE node (existing
    junction / committed endpoint) or None (fresh terrain origin, carried as start_location). A
    non-None id must therefore resolve strictly — a dangling id is a bug and raises (fail-fast).
    """

    def test_falls_back_to_start_location_when_no_origin_node(self, empty_graph) -> None:
        from skiresort_planner.ui.actions import resolve_build_origin
        from skiresort_planner.ui.context import SegmentBuildContext

        # Fresh terrain origin: no node yet, carried as start_location.
        build = SegmentBuildContext(start_location=PathPoint(lon=8.019, lat=46.584, elevation=3065.0))
        lon, lat, elevation, start_node_id = resolve_build_origin(build=build, graph=empty_graph)

        assert (lon, lat, elevation) == (8.019, 46.584, 3065.0), "routes from the pending origin location"
        assert start_node_id is None, "no node yet — commit_paths mints it"

    def test_stale_origin_node_falls_back_to_start_location(self, empty_graph) -> None:
        from skiresort_planner.ui.actions import resolve_build_origin
        from skiresort_planner.ui.context import SegmentBuildContext

        # The origin node was cleaned when the last segment was undone, but start_location survives
        # (restored by _restore_build_context). The dangling id is ignored; the location is used.
        build = SegmentBuildContext(
            start_node_id="N999",  # cleaned as isolated
            start_location=PathPoint(lon=8.019, lat=46.584, elevation=3065.0),
        )
        lon, lat, elevation, start_node_id = resolve_build_origin(build=build, graph=empty_graph)
        assert (lon, lat, elevation) == (8.019, 46.584, 3065.0), "falls back to the origin location"
        assert start_node_id is None, "the stale id is not reused"

    def test_raises_when_no_origin_at_all(self, empty_graph) -> None:
        import pytest

        from skiresort_planner.ui.actions import resolve_build_origin
        from skiresort_planner.ui.context import SegmentBuildContext

        # No endpoint, a dangling origin id, and NO location fallback → genuine programming error.
        build = SegmentBuildContext(start_node_id="N999")
        with pytest.raises(ValueError, match="no start node or location"):
            resolve_build_origin(build=build, graph=empty_graph)

    def test_endpoint_must_be_live(self, empty_graph) -> None:
        import pytest

        from skiresort_planner.ui.actions import resolve_build_origin
        from skiresort_planner.ui.context import SegmentBuildContext

        # A committed endpoint must exist — a missing one is an invariant violation (strict []).
        build = SegmentBuildContext(endpoints=["N999"])
        with pytest.raises(KeyError):
            resolve_build_origin(build=build, graph=empty_graph)

    def test_uses_node_when_present(self, empty_graph, path_points_blue) -> None:
        from skiresort_planner.ui.actions import resolve_build_origin
        from skiresort_planner.ui.context import SegmentBuildContext

        node, _ = empty_graph.get_or_create_node(lon=8.02, lat=46.58, elevation=3000.0)
        build = SegmentBuildContext(start_node_id=node.id)
        lon, lat, elevation, start_node_id = resolve_build_origin(build=build, graph=empty_graph)

        assert (lon, lat, elevation) == (node.lon, node.lat, node.elevation)
        assert start_node_id == node.id, "an existing origin node is returned for reuse on commit"


class TestUndoToZeroAfterFinish:
    """Regression: build a road, finish, undo the finish, then undo each segment back to zero.

    The final undo cleans the origin node (now isolated); the build must stay resolvable — the fan
    regenerates from the origin location, not a dangling start_node_id (was 'KeyError: N###').
    """

    def test_undo_all_segments_after_finish_regenerates_without_crash(
        self, fake_st, empty_graph, path_factory, mock_dem_red_slope_diagonal
    ) -> None:
        from skiresort_planner.ui.actions import (
            process_path_generation_deferred,
            resolve_build_origin,
            undo_last_action,
        )

        dem = mock_dem_red_slope_diagonal
        m = 111320.0
        sm, ctx = _session(fake_st, empty_graph, path_factory, dem)
        ctx.build_mode.mode = SegmentKind.ROAD.value

        # Start a road from fresh terrain and commit two segments (undo actions recorded).
        sm.start_road(node_id=None, location=PathPoint(lon=0.0, lat=0.0, elevation=2000.0))
        for i in range(1, 3):
            pts = [
                PathPoint(lon=(i - 1) * 300 / m, lat=0.0, elevation=2000.0 - (i - 1) * 10),
                PathPoint(lon=i * 300 / m, lat=0.0, elevation=2000.0 - i * 10),
            ]
            endpoint_ids = empty_graph.commit_paths(paths=[ProposedPathSegment(points=pts, kind=SegmentKind.ROAD)])
            seg = list(empty_graph.segments.keys())[-1]
            sm.commit_road(segment_id=seg, endpoint_node_id=endpoint_ids[0])

        road = empty_graph.finish_road(segment_ids=ctx.build(SegmentKind.ROAD).segments)
        sm.finish_road(entity_id=road.id)
        assert sm.is_idle_viewing_road

        # Undo everything: finish, then each segment. The last undo cleans the origin node.
        while empty_graph.undo_stack:
            undo_last_action()

        # The build must still resolve its origin without a dangling id (the crash was here).
        build = ctx.build(SegmentKind.ROAD)
        if build.segments or build.start_location or build.start_node_id:
            resolve_build_origin(build=build, graph=empty_graph)  # must not raise
        process_path_generation_deferred()  # the deferred fan pass must not raise either


class TestMapEpochs:
    """camera_epoch (remount → recenter) moves ONLY on finish; dedup_epoch (click-id) moves on
    proposal regeneration. Neither commit nor cancel nor start recenters (keeps the user's pan).
    """

    def _road_building(self, fake_st, empty_graph, path_factory, dem):
        sm, ctx = _session(fake_st, empty_graph, path_factory, dem)
        ctx.build_mode.mode = SegmentKind.ROAD.value
        sm.start_road(node_id=None, location=PathPoint(lon=0.0, lat=0.0, elevation=2000.0))
        return sm, ctx

    def test_commit_does_not_recenter(self, fake_st, empty_graph, path_factory, mock_dem_red_slope_diagonal) -> None:
        from skiresort_planner.ui.actions import commit_selected_path

        dem = mock_dem_red_slope_diagonal
        sm, ctx = self._road_building(fake_st, empty_graph, path_factory, dem)
        pts = [PathPoint(lon=0.0, lat=0.0, elevation=2000.0), PathPoint(lon=300 / M, lat=0.0, elevation=1990.0)]
        ctx.proposals.paths = [ProposedPathSegment(points=pts, kind=SegmentKind.ROAD)]
        ctx.proposals.selected_idx = 0
        camera_before = fake_st.session_state["camera_epoch"]

        commit_selected_path(path_idx=0)

        assert fake_st.session_state["camera_epoch"] == camera_before, "commit must NOT recenter"

    def test_finish_recenters(self, fake_st, empty_graph, path_factory, mock_dem_red_slope_diagonal) -> None:
        from skiresort_planner.ui.actions import finish_current_build

        dem = mock_dem_red_slope_diagonal
        sm, ctx = self._road_building(fake_st, empty_graph, path_factory, dem)
        pts = [PathPoint(lon=0.0, lat=0.0, elevation=2000.0), PathPoint(lon=300 / M, lat=0.0, elevation=1990.0)]
        endpoint_ids = empty_graph.commit_paths(paths=[ProposedPathSegment(points=pts, kind=SegmentKind.ROAD)])
        sm.commit_road(segment_id=list(empty_graph.segments.keys())[-1], endpoint_node_id=endpoint_ids[0])
        camera_before = fake_st.session_state["camera_epoch"]

        finish_current_build(kind=SegmentKind.ROAD)

        assert sm.is_idle_viewing_road
        assert fake_st.session_state["camera_epoch"] > camera_before, "finish recenters on the entity"

    def test_cancel_does_not_recenter(self, fake_st, empty_graph, path_factory, mock_dem_red_slope_diagonal) -> None:
        from skiresort_planner.ui.actions import cancel_current_build

        dem = mock_dem_red_slope_diagonal
        sm, ctx = self._road_building(fake_st, empty_graph, path_factory, dem)
        pts = [PathPoint(lon=0.0, lat=0.0, elevation=2000.0), PathPoint(lon=300 / M, lat=0.0, elevation=1990.0)]
        endpoint_ids = empty_graph.commit_paths(paths=[ProposedPathSegment(points=pts, kind=SegmentKind.ROAD)])
        sm.commit_road(segment_id=list(empty_graph.segments.keys())[-1], endpoint_node_id=endpoint_ids[0])
        camera_before = fake_st.session_state["camera_epoch"]

        cancel_current_build(kind=SegmentKind.ROAD)

        assert fake_st.session_state["camera_epoch"] == camera_before, "cancel must NOT recenter"

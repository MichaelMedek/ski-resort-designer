"""Core-logic tests for the UI action functions (actions.py).

These functions read st.session_state.{state_machine, context, graph}; with the
fake `st` installed we seed those and call the action directly, asserting the
real effect (entity removed, panel closed when it was being viewed). Covers the
delete actions for slope/lift/road uniformly.
"""

import pytest

from skiresort_planner.model.path_point import PathPoint
from skiresort_planner.model.path_segment import SegmentKind
from skiresort_planner.model.proposed_path import ProposedPathSegment
from skiresort_planner.model.resort_graph import ResortGraph
from skiresort_planner.ui.state_machine import PlannerStateMachine


def _session(fake_st, graph, factory=None, dem=None):
    """Seed fake st.session_state with the objects action functions read."""
    sm, ctx = PlannerStateMachine.create(graph=graph, add_ui_listener=False)
    fake_st.session_state["state_machine"] = sm
    fake_st.session_state["context"] = ctx
    fake_st.session_state["graph"] = graph
    fake_st.session_state["map_version"] = 0
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
        sm.show_slope_info_panel(slope_id=slope.id)

        delete_slope_action(slope_id=slope.id)
        assert not sm.is_idle_viewing_slope, "deleting the viewed slope must close its panel"


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
        sm.show_road_info_panel(road_id=road.id)

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

    def test_missing_lift_returns_false(self, fake_st, empty_graph) -> None:
        from skiresort_planner.ui.actions import delete_lift_action

        _session(fake_st, empty_graph)
        assert delete_lift_action(lift_id="L999") is False


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
        assert ctx.map.zoom == 15

    def test_center_on_road_sets_map(self, empty_graph) -> None:
        from skiresort_planner.ui.actions import center_on_road

        road = _make_road(empty_graph)
        _sm, ctx = PlannerStateMachine.create(graph=empty_graph, add_ui_listener=False)
        center_on_road(ctx=ctx, graph=empty_graph, road=road, zoom=16)
        assert ctx.map.zoom == 16

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
        assert ctx.slope_build.segments, "commit must add a segment to the building context"

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
            recompute_paths,
            undo_last_action,
        )

        dem = mock_dem_red_slope_diagonal
        sm, ctx, graph = self._start_building(fake_st, path_factory, dem)
        recompute_paths()
        commit_selected_path(path_idx=0)
        seg_id = ctx.slope_build.segments[-1]
        finish_current_slope()
        assert sm.is_idle_viewing_slope

        undo_last_action()  # undo FINISH_SLOPE
        assert sm.is_slope_building_only, "undo of finish returns to slope building"
        assert ctx.slope_build.segments == [seg_id], "segments are restored"
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
        assert sm.is_road_starting

        # Seed a road proposal (as handle_road_building_click would) and commit it.
        ctx.proposals.paths = [ProposedPathSegment(points=path_points_blue, is_connector=True, kind=SegmentKind.ROAD)]
        ctx.proposals.selected_idx = 0
        commit_selected_path(path_idx=0)

        assert sm.is_road_building_only, "road commit stays in road_building"
        assert len(ctx.road_build.segments) == 1
        assert len(graph.roads) == 0, "no Road entity until Finish Road"
        assert graph.segments[ctx.road_build.segments[-1]].kind is SegmentKind.ROAD
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
        seg_id = ctx.road_build.segments[-1]
        finish_current_road()
        assert sm.is_idle_viewing_road

        undo_last_action()  # undo FINISH_ROAD
        assert sm.is_road_building_only, "undo of finish returns to road building"
        assert ctx.road_build.segments == [seg_id], "segments are restored"
        assert ctx.proposals.paths == [], "roads have no fan to regenerate"


class TestDeferredProcessing:
    """Deferred-action processors read/clear ctx.deferred flags and act on them."""

    def test_process_path_generation_noop_when_not_pending(
        self, fake_st, path_factory, mock_dem_red_slope_diagonal
    ) -> None:
        from skiresort_planner.ui.actions import process_path_generation_deferred

        _sm, ctx = _session(fake_st, ResortGraph(), factory=path_factory, dem=mock_dem_red_slope_diagonal)
        ctx.deferred.path_generation = False
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
        ctx.deferred.path_generation = True

        assert process_path_generation_deferred() is True
        assert ctx.deferred.path_generation is False, "flag cleared after processing"
        assert ctx.proposals.paths, "fan proposals generated for the building state"

    def test_process_custom_connect_noop_when_not_pending(
        self, fake_st, path_factory, mock_dem_red_slope_diagonal
    ) -> None:
        from skiresort_planner.ui.actions import process_custom_connect_deferred

        _sm, ctx = _session(fake_st, ResortGraph(), factory=path_factory, dem=mock_dem_red_slope_diagonal)
        ctx.deferred.custom_connect = False
        assert process_custom_connect_deferred() is False

    def test_fast_deferred_start_building_from_node(self, fake_st, path_factory, mock_dem_red_slope_diagonal) -> None:
        from skiresort_planner.ui.actions import handle_fast_deferred_actions

        dem = mock_dem_red_slope_diagonal
        graph = ResortGraph()
        node, _ = graph.get_or_create_node(lon=0.0, lat=0.0, elevation=dem.get_elevation_or_raise(lon=0.0, lat=0.0))
        sm, ctx = _session(fake_st, graph, factory=path_factory, dem=dem)
        ctx.deferred.start_building_from_node_id = node.id

        handle_fast_deferred_actions()
        assert sm.is_slope_starting, "deferred start-building-from-node begins a slope"
        assert ctx.deferred.start_building_from_node_id is None, "flag consumed"

    def test_fast_deferred_start_lift_from_node(self, fake_st, path_factory, mock_dem_red_slope_diagonal) -> None:
        from skiresort_planner.ui.actions import handle_fast_deferred_actions

        dem = mock_dem_red_slope_diagonal
        graph = ResortGraph()
        node, _ = graph.get_or_create_node(lon=0.0, lat=0.0, elevation=dem.get_elevation_or_raise(lon=0.0, lat=0.0))
        sm, ctx = _session(fake_st, graph, factory=path_factory, dem=dem)
        ctx.deferred.start_lift_from_node_id = node.id

        handle_fast_deferred_actions()
        assert sm.is_lift_placing, "deferred start-lift-from-node begins lift placement"
        assert ctx.deferred.start_lift_from_node_id is None, "flag consumed"

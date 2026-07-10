"""Core-logic tests for the UI action functions (actions.py).

These functions read st.session_state.{state_machine, context, graph}; with the
fake `st` installed we seed those and call the action directly, asserting the
real effect (entity removed, panel closed when it was being viewed). Covers the
delete actions for slope/lift/road uniformly.
"""

from skiresort_planner.model.path_point import PathPoint
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
    graph.commit_paths(paths=[ProposedPathSegment(points=pts, is_connector=True)], record_undo=False)
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
    """undo_last_action dispatches to the right handler by action type."""

    def test_undo_add_road_removes_it(self, fake_st, empty_graph) -> None:
        from skiresort_planner.ui.actions import undo_last_action

        road = _make_road(empty_graph)
        _session(fake_st, empty_graph)
        assert road.id in empty_graph.roads

        undo_last_action()
        assert road.id not in empty_graph.roads

    def test_undo_delete_road_restores_it(self, fake_st, empty_graph) -> None:
        from skiresort_planner.ui.actions import undo_last_action

        road = _make_road(empty_graph)
        empty_graph.delete_road(road_id=road.id)  # records a DELETE_ROAD undo
        _session(fake_st, empty_graph)

        undo_last_action()
        assert road.id in empty_graph.roads

    def test_undo_finish_slope_keeps_segments(
        self, fake_st, path_factory, mock_dem_red_slope_diagonal, empty_graph, path_points_blue
    ) -> None:
        from skiresort_planner.ui.actions import undo_last_action

        slope = _make_slope(empty_graph, path_points_blue)  # ADD_SEGMENTS + FINISH_SLOPE
        _session(fake_st, empty_graph, factory=path_factory, dem=mock_dem_red_slope_diagonal)
        n_segments = len(empty_graph.segments)

        undo_last_action()  # undo FINISH_SLOPE
        assert slope.id not in empty_graph.slopes
        assert len(empty_graph.segments) == n_segments  # segments survive

    def test_undo_empty_stack_is_noop(self, fake_st, empty_graph) -> None:
        from skiresort_planner.ui.actions import undo_last_action

        _session(fake_st, empty_graph)
        undo_last_action()  # must not raise
        assert not empty_graph.undo_stack


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
        assert ctx.building.segments, "commit must add a segment to the building context"

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

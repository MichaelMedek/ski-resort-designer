"""Tests for the heavy right/left panel + sidebar render logic.

Uses the shared `fake_st` fixture (conftest) to install a no-op Streamlit so
each panel's render() executes every widget call without a browser. Two flavors:
- render tests assert the panel *runs* (no raise) across slope/lift/road;
- `fake_st.clicked_keys` fires a specific button so its body executes and the
  real state change (3D toggle, close panel, mode switch) is asserted.
Also covers the sidebar undo-action labels (`_describe_undo_action`).
"""

import pytest

from skiresort_planner.model.proposed_path import ProposedPathSegment
from skiresort_planner.model.resort_graph import ResortGraph
from skiresort_planner.ui.context import BuildMode
from skiresort_planner.ui.right_panel import (
    LiftStatsPanel,
    RoadStatsPanel,
    SlopeStatsPanel,
    render_control_panel,
)
from skiresort_planner.ui.state_machine import PlannerStateMachine


# =============================================================================
# Entity builders (commit a segment, then group into slope / road)
# =============================================================================


def _build_slope(graph: ResortGraph, path_points: list) -> str:
    proposal = ProposedPathSegment(points=path_points, target_difficulty="blue")
    graph.commit_paths(paths=[proposal])
    slope = graph.finish_slope(segment_ids=list(graph.segments.keys()))
    assert slope is not None
    return slope.id


def _build_road(graph: ResortGraph, path_points: list) -> str:
    proposal = ProposedPathSegment(points=path_points, is_connector=True)
    graph.commit_paths(paths=[proposal], record_undo=False)
    seg_id = list(graph.segments.keys())[-1]
    road = graph.finish_road(segment_ids=[seg_id])
    assert road is not None
    return road.id


def _build_lift(graph: ResortGraph, dem) -> str:
    M = 111320.0
    bottom, _ = graph.get_or_create_node(
        lon=0.0, lat=-1000 / M, elevation=dem.get_elevation_or_raise(lon=0.0, lat=-1000 / M)
    )
    top, _ = graph.get_or_create_node(lon=0.0, lat=0.0, elevation=dem.get_elevation_or_raise(lon=0.0, lat=0.0))
    lift = graph.add_lift(start_node_id=bottom.id, end_node_id=top.id, lift_type="chairlift", dem=dem)
    return lift.id


# =============================================================================
# Stats panels (render() must run without raising)
# =============================================================================


class TestStatsPanelsRun:
    def test_slope_stats_panel_runs(self, fake_st, empty_graph, path_points_blue) -> None:
        slope_id = _build_slope(empty_graph, path_points_blue)
        SlopeStatsPanel(graph=empty_graph).render(slope_id=slope_id)

    def test_road_stats_panel_runs(self, fake_st, empty_graph, path_points_blue) -> None:
        road_id = _build_road(empty_graph, path_points_blue)
        RoadStatsPanel(graph=empty_graph).render(road_id=road_id)

    def test_lift_stats_panel_runs(self, fake_st, empty_graph, mock_dem_blue_slope) -> None:
        lift_id = _build_lift(empty_graph, mock_dem_blue_slope)
        LiftStatsPanel(graph=empty_graph).render(lift_id=lift_id)


# =============================================================================
# Button-click branches (fake_st.clicked_keys fires a specific button's body)
# =============================================================================


class TestPanelButtonClicks:
    """Drive the button-body logic that a plain no-raise render never touches.

    fake_st.button() returns True only for keys in clicked_keys, so these tests
    exercise the close/3D-toggle handlers and assert the real state changes.
    """

    def _bump_ready(self, fake_st, sm, ctx, graph) -> None:
        # reload_map()/bump_map_version() read these off session_state.
        fake_st.session_state["state_machine"] = sm
        fake_st.session_state["context"] = ctx
        fake_st.session_state["graph"] = graph
        fake_st.session_state["map_version"] = 0

    def test_enable_3d_from_slope_panel(self, fake_st, empty_graph, path_points_blue) -> None:
        from skiresort_planner.ui.right_panel import _render_slope_info_panel

        slope_id = _build_slope(empty_graph, path_points_blue)
        sm, ctx = PlannerStateMachine.create(graph=empty_graph, add_ui_listener=False)
        sm.show_slope_info_panel(slope_id=slope_id)
        self._bump_ready(fake_st, sm, ctx, empty_graph)

        assert not ctx.viewing.view_3d
        fake_st.clicked_keys = {"slope_3d_view"}
        _render_slope_info_panel(sm=sm, ctx=ctx, graph=empty_graph)
        assert ctx.viewing.view_3d, "clicking 'View in 3D' must enable 3D"

    def test_disable_3d_recenters_road(self, fake_st, empty_graph, path_points_blue) -> None:
        from skiresort_planner.ui.right_panel import _render_road_info_panel

        road_id = _build_road(empty_graph, path_points_blue)
        sm, ctx = PlannerStateMachine.create(graph=empty_graph, add_ui_listener=False)
        sm.show_road_info_panel(road_id=road_id)
        ctx.viewing.enable_3d()
        self._bump_ready(fake_st, sm, ctx, empty_graph)

        fake_st.clicked_keys = {"road_2d_view"}
        _render_road_info_panel(sm=sm, ctx=ctx, graph=empty_graph)
        assert not ctx.viewing.view_3d, "clicking 'Return to 2D' must disable 3D"

    def test_close_slope_panel_returns_to_idle(self, fake_st, empty_graph, path_points_blue) -> None:
        from skiresort_planner.ui.right_panel import _render_slope_info_panel

        slope_id = _build_slope(empty_graph, path_points_blue)
        sm, ctx = PlannerStateMachine.create(graph=empty_graph, add_ui_listener=False)
        sm.show_slope_info_panel(slope_id=slope_id)
        self._bump_ready(fake_st, sm, ctx, empty_graph)

        assert sm.is_idle_viewing_slope
        fake_st.clicked_keys = {"close_slope"}
        _render_slope_info_panel(sm=sm, ctx=ctx, graph=empty_graph)
        assert not sm.is_idle_viewing_slope, "clicking Close must leave the viewing state"

    def test_enable_3d_from_lift_panel_recenters(self, fake_st, empty_graph, mock_dem_blue_slope) -> None:
        from skiresort_planner.ui.right_panel import _render_lift_info_panel

        lift_id = _build_lift(empty_graph, mock_dem_blue_slope)
        sm, ctx = PlannerStateMachine.create(graph=empty_graph, add_ui_listener=False)
        sm.show_lift_info_panel(lift_id=lift_id)
        self._bump_ready(fake_st, sm, ctx, empty_graph)

        fake_st.clicked_keys = {"lift_3d_view"}
        _render_lift_info_panel(sm=sm, ctx=ctx, graph=empty_graph)
        assert ctx.viewing.view_3d, "clicking 'View in 3D' on a lift must enable 3D"

    def test_disable_3d_recenters_slope(self, fake_st, empty_graph, path_points_blue) -> None:
        from skiresort_planner.ui.right_panel import _render_slope_info_panel

        slope_id = _build_slope(empty_graph, path_points_blue)
        sm, ctx = PlannerStateMachine.create(graph=empty_graph, add_ui_listener=False)
        sm.show_slope_info_panel(slope_id=slope_id)
        ctx.viewing.enable_3d()
        self._bump_ready(fake_st, sm, ctx, empty_graph)

        fake_st.clicked_keys = {"slope_2d_view"}
        _render_slope_info_panel(sm=sm, ctx=ctx, graph=empty_graph)
        assert not ctx.viewing.view_3d, "clicking 'Return to 2D' on a slope must disable 3D"

    def test_click_road_mode_button_switches_mode(self, fake_st, empty_graph) -> None:
        from skiresort_planner.ui.context import BuildMode
        from skiresort_planner.ui.left_panel import SidebarRenderer

        sm, ctx = PlannerStateMachine.create(graph=empty_graph, add_ui_listener=False)
        ctx.build_mode.mode = BuildMode.SLOPE
        self._bump_ready(fake_st, sm, ctx, empty_graph)

        fake_st.clicked_keys = {"build_btn_road"}
        SidebarRenderer(state_machine=sm, context=ctx, graph=empty_graph).render()
        assert ctx.build_mode.mode == BuildMode.ROAD, "clicking the Road button must switch build mode"


# =============================================================================
# Full control-panel dispatch in each viewing state
# =============================================================================


class TestControlPanelDispatch:
    def _noop(self, *args: object, **kwargs: object) -> None:
        return None

    def _render(self, graph: ResortGraph) -> None:
        sm, ctx = PlannerStateMachine.create(graph=graph, add_ui_listener=False)
        # Drive to the relevant viewing state below via sm; here just dispatch.
        render_control_panel(
            sm=sm,
            ctx=ctx,
            graph=graph,
            on_commit=self._noop,
            on_custom_direction=self._noop,
            on_cancel_custom=self._noop,
            on_cancel_connection=self._noop,
        )

    def test_idle_ready_panel_runs(self, fake_st, empty_graph) -> None:
        self._render(empty_graph)

    def test_viewing_slope_panel_runs(self, fake_st, empty_graph, path_points_blue) -> None:
        slope_id = _build_slope(empty_graph, path_points_blue)
        sm, ctx = PlannerStateMachine.create(graph=empty_graph, add_ui_listener=False)
        sm.show_slope_info_panel(slope_id=slope_id)
        render_control_panel(
            sm=sm,
            ctx=ctx,
            graph=empty_graph,
            on_commit=self._noop,
            on_custom_direction=self._noop,
            on_cancel_custom=self._noop,
            on_cancel_connection=self._noop,
        )

    def test_viewing_road_panel_runs(self, fake_st, empty_graph, path_points_blue) -> None:
        road_id = _build_road(empty_graph, path_points_blue)
        sm, ctx = PlannerStateMachine.create(graph=empty_graph, add_ui_listener=False)
        sm.show_road_info_panel(road_id=road_id)
        render_control_panel(
            sm=sm,
            ctx=ctx,
            graph=empty_graph,
            on_commit=self._noop,
            on_custom_direction=self._noop,
            on_cancel_custom=self._noop,
            on_cancel_connection=self._noop,
        )

    def test_viewing_lift_panel_runs(self, fake_st, empty_graph, mock_dem_blue_slope) -> None:
        lift_id = _build_lift(empty_graph, mock_dem_blue_slope)
        sm, ctx = PlannerStateMachine.create(graph=empty_graph, add_ui_listener=False)
        sm.show_lift_info_panel(lift_id=lift_id)
        render_control_panel(
            sm=sm,
            ctx=ctx,
            graph=empty_graph,
            on_commit=self._noop,
            on_custom_direction=self._noop,
            on_cancel_custom=self._noop,
            on_cancel_connection=self._noop,
        )

    def test_road_placing_panel_runs(self, fake_st, empty_graph, path_points_blue) -> None:
        # Regression: render_control_panel must handle road_placing (not raise).
        sm, ctx = PlannerStateMachine.create(graph=empty_graph, add_ui_listener=False)
        sm.start_road(node_id=None, location=path_points_blue[0])
        assert sm.is_road_placing
        render_control_panel(
            sm=sm,
            ctx=ctx,
            graph=empty_graph,
            on_commit=self._noop,
            on_custom_direction=self._noop,
            on_cancel_custom=self._noop,
            on_cancel_connection=self._noop,
        )

    def test_road_placing_panel_from_node_runs(self, fake_st, empty_graph) -> None:
        # Start from an existing node → exercises the node-start message branch.
        node, _ = empty_graph.get_or_create_node(lon=0.0, lat=0.0, elevation=2000.0)
        sm, ctx = PlannerStateMachine.create(graph=empty_graph, add_ui_listener=False)
        sm.start_road(node_id=node.id, location=None)
        render_control_panel(
            sm=sm,
            ctx=ctx,
            graph=empty_graph,
            on_commit=self._noop,
            on_custom_direction=self._noop,
            on_cancel_custom=self._noop,
            on_cancel_connection=self._noop,
        )


# =============================================================================
# Sidebar (left panel) in each build mode
# =============================================================================


class TestSidebarRuns:
    @pytest.mark.parametrize("mode", [BuildMode.SLOPE, BuildMode.ROAD, BuildMode.CHAIRLIFT])
    def test_sidebar_runs_in_each_mode(self, fake_st, empty_graph, mode: str) -> None:
        from skiresort_planner.ui.left_panel import SidebarRenderer

        sm, ctx = PlannerStateMachine.create(graph=empty_graph, add_ui_listener=False)
        ctx.build_mode.mode = mode
        actions = SidebarRenderer(state_machine=sm, context=ctx, graph=empty_graph).render()
        assert isinstance(actions, dict)

    def test_sidebar_runs_with_content(self, fake_st, empty_graph, path_points_blue, mock_dem_blue_slope) -> None:
        # A resort with a slope + lift + road exercises every summary section.
        _build_slope(empty_graph, path_points_blue)
        _build_lift(empty_graph, mock_dem_blue_slope)
        _build_road(empty_graph, path_points_blue)
        from skiresort_planner.ui.left_panel import SidebarRenderer

        sm, ctx = PlannerStateMachine.create(graph=empty_graph, add_ui_listener=False)
        SidebarRenderer(state_machine=sm, context=ctx, graph=empty_graph).render()

    def test_sidebar_during_slope_building(self, fake_st, empty_graph, mock_dem_blue_slope) -> None:
        # Building state renders the building controls + undo/reset buttons.
        sm, ctx = PlannerStateMachine.create(graph=empty_graph, add_ui_listener=False)
        sm.start_building(lon=0.0, lat=0.0, elevation=mock_dem_blue_slope.get_elevation_or_raise(lon=0.0, lat=0.0))
        from skiresort_planner.ui.left_panel import SidebarRenderer

        SidebarRenderer(state_machine=sm, context=ctx, graph=empty_graph).render()

    def test_sidebar_while_viewing_slope(self, fake_st, empty_graph, path_points_blue) -> None:
        # Viewing state renders the close-panel button path.
        slope_id = _build_slope(empty_graph, path_points_blue)
        sm, ctx = PlannerStateMachine.create(graph=empty_graph, add_ui_listener=False)
        sm.show_slope_info_panel(slope_id=slope_id)
        from skiresort_planner.ui.left_panel import SidebarRenderer

        SidebarRenderer(state_machine=sm, context=ctx, graph=empty_graph).render()

    def test_sidebar_during_road_placing(self, fake_st, empty_graph) -> None:
        # Road placing state renders the road cancel button.
        from skiresort_planner.model.path_point import PathPoint

        sm, ctx = PlannerStateMachine.create(graph=empty_graph, add_ui_listener=False)
        sm.start_road(node_id=None, location=PathPoint(lon=0.0, lat=0.0, elevation=2000.0))
        assert sm.is_road_placing
        from skiresort_planner.ui.left_panel import SidebarRenderer

        SidebarRenderer(state_machine=sm, context=ctx, graph=empty_graph).render()

    def test_sidebar_while_viewing_road(self, fake_st, empty_graph, path_points_blue) -> None:
        # Viewing a road renders the close-panel button path and road summary.
        road_id = _build_road(empty_graph, path_points_blue)
        sm, ctx = PlannerStateMachine.create(graph=empty_graph, add_ui_listener=False)
        sm.show_road_info_panel(road_id=road_id)
        from skiresort_planner.ui.left_panel import SidebarRenderer

        SidebarRenderer(state_machine=sm, context=ctx, graph=empty_graph).render()


# =============================================================================
# Building / placing control panels (state-driven)
# =============================================================================


class TestBuildingPanelsRun:
    def _noop(self, *args: object, **kwargs: object) -> None:
        return None

    def _render(self, sm, ctx, graph) -> None:
        render_control_panel(
            sm=sm,
            ctx=ctx,
            graph=graph,
            on_commit=self._noop,
            on_custom_direction=self._noop,
            on_cancel_custom=self._noop,
            on_cancel_connection=self._noop,
        )

    def test_slope_starting_panel_runs(self, fake_st, empty_graph, mock_dem_blue_slope) -> None:
        sm, ctx = PlannerStateMachine.create(graph=empty_graph, add_ui_listener=False)
        sm.start_building(lon=0.0, lat=0.0, elevation=mock_dem_blue_slope.get_elevation_or_raise(lon=0.0, lat=0.0))
        assert sm.is_slope_starting
        self._render(sm, ctx, empty_graph)

    def test_lift_placing_panel_runs(self, fake_st, empty_graph, mock_dem_blue_slope) -> None:
        from skiresort_planner.model.path_point import PathPoint

        sm, ctx = PlannerStateMachine.create(graph=empty_graph, add_ui_listener=False)
        loc = PathPoint(lon=0.0, lat=0.0, elevation=mock_dem_blue_slope.get_elevation_or_raise(lon=0.0, lat=0.0))
        sm.start_lift(node_id=None, location=loc)
        assert sm.is_lift_placing
        self._render(sm, ctx, empty_graph)


# =============================================================================
# Path selection panel (proposal browsing)
# =============================================================================


class TestPathSelectionPanelRuns:
    def _noop(self, *args: object, **kwargs: object) -> None:
        return None

    def _panel(self, ctx, graph):
        from skiresort_planner.ui.right_panel import PathSelectionPanel

        return PathSelectionPanel(
            context=ctx,
            graph=graph,
            on_commit=self._noop,
            on_custom_direction=self._noop,
            on_cancel_custom=self._noop,
            on_cancel_connection=self._noop,
        )

    def test_no_proposals_runs(self, fake_st, empty_graph) -> None:
        _sm, ctx = PlannerStateMachine.create(graph=empty_graph, add_ui_listener=False)
        self._panel(ctx, empty_graph).render()

    def test_with_selected_proposal_runs(self, fake_st, empty_graph, path_points_blue) -> None:
        _sm, ctx = PlannerStateMachine.create(graph=empty_graph, add_ui_listener=False)
        ctx.proposals.paths = [ProposedPathSegment(points=path_points_blue, target_difficulty="blue")]
        ctx.proposals.selected_idx = 0
        self._panel(ctx, empty_graph).render()

    def test_custom_connect_enabled_runs(self, fake_st, empty_graph) -> None:
        _sm, ctx = PlannerStateMachine.create(graph=empty_graph, add_ui_listener=False)
        ctx.custom_connect.enabled = True
        self._panel(ctx, empty_graph).render()


# =============================================================================
# Sidebar undo-action labels (_describe_undo_action, one per action type)
# =============================================================================


class TestDescribeUndoAction:
    """_describe_undo_action labels every undo type. Each action comes from a
    real graph mutation that pushes it onto graph.undo_stack — no stubs."""

    def _describe_top(self, graph: ResortGraph) -> str:
        from skiresort_planner.ui.left_panel import _describe_undo_action

        return _describe_undo_action(graph.undo_stack[-1], graph)

    def test_add_segments_label(self, empty_graph, path_points_blue) -> None:
        empty_graph.commit_paths(paths=[ProposedPathSegment(points=path_points_blue, target_difficulty="blue")])
        assert "segment" in self._describe_top(empty_graph).lower()

    def test_finish_slope_label(self, empty_graph, path_points_blue) -> None:
        slope_id = _build_slope(empty_graph, path_points_blue)
        assert empty_graph.slopes[slope_id].name in self._describe_top(empty_graph)

    def test_add_lift_label(self, empty_graph, mock_dem_blue_slope) -> None:
        lift_id = _build_lift(empty_graph, mock_dem_blue_slope)
        label = self._describe_top(empty_graph)
        assert "Delete lift" in label and empty_graph.lifts[lift_id].name in label

    def test_add_road_label(self, empty_graph, path_points_blue) -> None:
        road_id = _build_road(empty_graph, path_points_blue)  # finish_road records ADD_ROAD
        label = self._describe_top(empty_graph)
        assert "Delete road" in label and empty_graph.roads[road_id].name in label

    def test_delete_slope_label(self, empty_graph, path_points_blue) -> None:
        slope_id = _build_slope(empty_graph, path_points_blue)
        empty_graph.delete_slope(slope_id=slope_id)
        assert "Restore deleted slope" in self._describe_top(empty_graph)

    def test_delete_lift_label(self, empty_graph, mock_dem_blue_slope) -> None:
        lift_id = _build_lift(empty_graph, mock_dem_blue_slope)
        empty_graph.delete_lift(lift_id=lift_id)
        assert "Restore deleted lift" in self._describe_top(empty_graph)

    def test_delete_road_label(self, empty_graph, path_points_blue) -> None:
        road_id = _build_road(empty_graph, path_points_blue)
        empty_graph.delete_road(road_id=road_id)
        assert "Restore deleted road" in self._describe_top(empty_graph)

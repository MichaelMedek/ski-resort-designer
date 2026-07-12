"""Tests for the right control panel (ui/right_panel.py).

Uses the shared `fake_st` fixture so each panel's render() runs without a
browser. Two flavors: render tests assert the panel runs across slope/lift/road;
`fake_st.clicked_keys` fires a specific button so its body executes and the real
state change (3D toggle, close panel) is asserted.
"""

from skiresort_planner.model.path_segment import SegmentKind
from skiresort_planner.model.proposed_path import ProposedPathSegment
from skiresort_planner.model.resort_graph import ResortGraph
from skiresort_planner.ui.right_panel import (
    LiftStatsPanel,
    RoadStatsPanel,
    SlopeStatsPanel,
    render_control_panel,
)
from skiresort_planner.ui.state_machine import PlannerStateMachine

M = 111320.0


def _build_slope(graph: ResortGraph, path_points: list) -> str:
    graph.commit_paths(paths=[ProposedPathSegment(points=path_points, target_difficulty="blue")])
    slope = graph.finish_slope(segment_ids=list(graph.segments.keys()))
    assert slope is not None
    return slope.id


def _build_road(graph: ResortGraph, path_points: list) -> str:
    graph.commit_paths(
        paths=[ProposedPathSegment(points=path_points, is_connector=True, kind=SegmentKind.ROAD)], record_undo=False
    )
    road = graph.finish_road(segment_ids=[list(graph.segments.keys())[-1]])
    assert road is not None
    return road.id


def _build_lift(graph: ResortGraph, dem) -> str:
    bottom, _ = graph.get_or_create_node(
        lon=0.0, lat=-1000 / M, elevation=dem.get_elevation_or_raise(lon=0.0, lat=-1000 / M)
    )
    top, _ = graph.get_or_create_node(lon=0.0, lat=0.0, elevation=dem.get_elevation_or_raise(lon=0.0, lat=0.0))
    lift = graph.add_lift(start_node_id=bottom.id, end_node_id=top.id, lift_type="chairlift", dem=dem)
    return lift.id


def _noop(*args: object, **kwargs: object) -> None:
    return None


def _dispatch(sm, ctx, graph) -> None:
    render_control_panel(
        sm=sm,
        ctx=ctx,
        graph=graph,
        on_commit=_noop,
        on_cancel_connection=_noop,
    )


# =============================================================================
# Stats panels (render() must run without raising)
# =============================================================================


class TestStatsPanelsRun:
    """Each stats panel renders its OWN metric labels — no kind shares another's
    layout by accident (the per-kind drift that hit the sidebar). Metric labels are
    captured to assert the distinguishing fields actually render."""

    @staticmethod
    def _capture_labels(fake_st) -> list[str]:
        labels: list[str] = []
        fake_st.metric = lambda label, *a, **k: labels.append(label)
        fake_st.subheader = lambda text, *a, **k: labels.append(text)
        return labels

    def test_slope_stats_panel_shows_slope_metrics(self, fake_st, empty_graph, path_points_blue) -> None:
        labels = self._capture_labels(fake_st)
        SlopeStatsPanel(graph=empty_graph).render(slope_id=_build_slope(empty_graph, path_points_blue))
        assert {"Top Elevation", "Drop", "Overall Gradient", "Steepest Section"} <= set(labels)

    def test_road_stats_panel_shows_road_metrics(self, fake_st, empty_graph, path_points_blue) -> None:
        labels = self._capture_labels(fake_st)
        RoadStatsPanel(graph=empty_graph).render(road_id=_build_road(empty_graph, path_points_blue))
        # Roads report signed elevation change + average gradient, not slope "Drop".
        assert {"Start Elevation", "Elevation Change", "Average Gradient", "Steepest Section"} <= set(labels)
        assert "Drop" not in labels

    def test_lift_stats_panel_shows_lift_metrics(self, fake_st, empty_graph, mock_dem_blue_slope) -> None:
        labels = self._capture_labels(fake_st)
        LiftStatsPanel(graph=empty_graph).render(lift_id=_build_lift(empty_graph, mock_dem_blue_slope))
        assert {"Vertical Rise", "Pylons", "Inclined Length", "Steepest Section"} <= set(labels)


# =============================================================================
# Info-panel button clicks (fake_st.clicked_keys fires a specific button body)
# =============================================================================


class TestInfoPanelButtonClicks:
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

    def test_disable_3d_from_slope_panel(self, fake_st, empty_graph, path_points_blue) -> None:
        from skiresort_planner.ui.right_panel import _render_slope_info_panel

        slope_id = _build_slope(empty_graph, path_points_blue)
        sm, ctx = PlannerStateMachine.create(graph=empty_graph, add_ui_listener=False)
        sm.show_slope_info_panel(slope_id=slope_id)
        ctx.viewing.enable_3d()
        self._bump_ready(fake_st, sm, ctx, empty_graph)

        fake_st.clicked_keys = {"slope_2d_view"}
        _render_slope_info_panel(sm=sm, ctx=ctx, graph=empty_graph)
        assert not ctx.viewing.view_3d, "clicking 'Return to 2D' on a slope must disable 3D"

    def test_disable_3d_from_road_panel(self, fake_st, empty_graph, path_points_blue) -> None:
        from skiresort_planner.ui.right_panel import _render_road_info_panel

        road_id = _build_road(empty_graph, path_points_blue)
        sm, ctx = PlannerStateMachine.create(graph=empty_graph, add_ui_listener=False)
        sm.show_road_info_panel(road_id=road_id)
        ctx.viewing.enable_3d()
        self._bump_ready(fake_st, sm, ctx, empty_graph)

        fake_st.clicked_keys = {"road_2d_view"}
        _render_road_info_panel(sm=sm, ctx=ctx, graph=empty_graph)
        assert not ctx.viewing.view_3d, "clicking 'Return to 2D' must disable 3D"

    def test_enable_3d_from_lift_panel(self, fake_st, empty_graph, mock_dem_blue_slope) -> None:
        from skiresort_planner.ui.right_panel import _render_lift_info_panel

        lift_id = _build_lift(empty_graph, mock_dem_blue_slope)
        sm, ctx = PlannerStateMachine.create(graph=empty_graph, add_ui_listener=False)
        sm.show_lift_info_panel(lift_id=lift_id)
        self._bump_ready(fake_st, sm, ctx, empty_graph)

        fake_st.clicked_keys = {"lift_3d_view"}
        _render_lift_info_panel(sm=sm, ctx=ctx, graph=empty_graph)
        assert ctx.viewing.view_3d, "clicking 'View in 3D' on a lift must enable 3D"

    def test_enable_3d_from_road_panel(self, fake_st, empty_graph, path_points_blue) -> None:
        from skiresort_planner.ui.right_panel import _render_road_info_panel

        road_id = _build_road(empty_graph, path_points_blue)
        sm, ctx = PlannerStateMachine.create(graph=empty_graph, add_ui_listener=False)
        sm.show_road_info_panel(road_id=road_id)
        self._bump_ready(fake_st, sm, ctx, empty_graph)

        assert not ctx.viewing.view_3d
        fake_st.clicked_keys = {"road_3d_view"}
        _render_road_info_panel(sm=sm, ctx=ctx, graph=empty_graph)
        assert ctx.viewing.view_3d, "clicking 'View in 3D' on a road must enable 3D"

    def test_disable_3d_from_lift_panel(self, fake_st, empty_graph, mock_dem_blue_slope) -> None:
        from skiresort_planner.ui.right_panel import _render_lift_info_panel

        lift_id = _build_lift(empty_graph, mock_dem_blue_slope)
        sm, ctx = PlannerStateMachine.create(graph=empty_graph, add_ui_listener=False)
        sm.show_lift_info_panel(lift_id=lift_id)
        ctx.viewing.enable_3d()
        self._bump_ready(fake_st, sm, ctx, empty_graph)

        fake_st.clicked_keys = {"lift_2d_view"}
        _render_lift_info_panel(sm=sm, ctx=ctx, graph=empty_graph)
        assert not ctx.viewing.view_3d, "clicking 'Return to 2D' on a lift must disable 3D"

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


# =============================================================================
# render_control_panel dispatch across states
# =============================================================================


class TestControlPanelDispatch:
    def test_idle_ready_panel_runs(self, fake_st, empty_graph) -> None:
        sm, ctx = PlannerStateMachine.create(graph=empty_graph, add_ui_listener=False)
        _dispatch(sm, ctx, empty_graph)

    def test_viewing_slope_panel_runs(self, fake_st, empty_graph, path_points_blue) -> None:
        slope_id = _build_slope(empty_graph, path_points_blue)
        sm, ctx = PlannerStateMachine.create(graph=empty_graph, add_ui_listener=False)
        sm.show_slope_info_panel(slope_id=slope_id)
        _dispatch(sm, ctx, empty_graph)

    def test_viewing_road_panel_runs(self, fake_st, empty_graph, path_points_blue) -> None:
        road_id = _build_road(empty_graph, path_points_blue)
        sm, ctx = PlannerStateMachine.create(graph=empty_graph, add_ui_listener=False)
        sm.show_road_info_panel(road_id=road_id)
        _dispatch(sm, ctx, empty_graph)

    def test_viewing_lift_panel_runs(self, fake_st, empty_graph, mock_dem_blue_slope) -> None:
        lift_id = _build_lift(empty_graph, mock_dem_blue_slope)
        sm, ctx = PlannerStateMachine.create(graph=empty_graph, add_ui_listener=False)
        sm.show_lift_info_panel(lift_id=lift_id)
        _dispatch(sm, ctx, empty_graph)

    def test_road_starting_panel_runs(self, fake_st, empty_graph, path_points_blue) -> None:
        # Regression: render_control_panel must handle road_starting (not raise).
        sm, ctx = PlannerStateMachine.create(graph=empty_graph, add_ui_listener=False)
        sm.start_road(node_id=None, location=path_points_blue[0])
        assert sm.is_road_starting
        _dispatch(sm, ctx, empty_graph)

    def test_road_starting_panel_from_node_runs(self, fake_st, empty_graph) -> None:
        # Start from an existing node → exercises the node-start message branch.
        node, _ = empty_graph.get_or_create_node(lon=0.0, lat=0.0, elevation=2000.0)
        sm, ctx = PlannerStateMachine.create(graph=empty_graph, add_ui_listener=False)
        sm.start_road(node_id=node.id, location=None)
        _dispatch(sm, ctx, empty_graph)

    def test_road_building_panel_runs(self, fake_st, empty_graph, path_points_blue) -> None:
        # After committing a segment, the panel shows the building progress message.
        sm, ctx = PlannerStateMachine.create(graph=empty_graph, add_ui_listener=False)
        sm.start_road(node_id=None, location=path_points_blue[0])
        empty_graph.commit_paths(
            paths=[ProposedPathSegment(points=path_points_blue, is_connector=True, kind=SegmentKind.ROAD)],
            record_undo=False,
        )
        seg = list(empty_graph.segments.keys())[-1]
        sm.commit_road(segment_id=seg, endpoint_node_id=empty_graph.segments[seg].end_node_id)
        assert sm.is_road_building_only
        _dispatch(sm, ctx, empty_graph)

    def test_slope_starting_panel_runs(self, fake_st, empty_graph, mock_dem_blue_slope) -> None:
        sm, ctx = PlannerStateMachine.create(graph=empty_graph, add_ui_listener=False)
        sm.start_building(lon=0.0, lat=0.0, elevation=mock_dem_blue_slope.get_elevation_or_raise(lon=0.0, lat=0.0))
        assert sm.is_slope_starting
        _dispatch(sm, ctx, empty_graph)

    def test_lift_placing_panel_runs(self, fake_st, empty_graph, mock_dem_blue_slope) -> None:
        from skiresort_planner.model.path_point import PathPoint

        sm, ctx = PlannerStateMachine.create(graph=empty_graph, add_ui_listener=False)
        loc = PathPoint(lon=0.0, lat=0.0, elevation=mock_dem_blue_slope.get_elevation_or_raise(lon=0.0, lat=0.0))
        sm.start_lift(node_id=None, location=loc)
        assert sm.is_lift_placing
        _dispatch(sm, ctx, empty_graph)


# =============================================================================
# Path selection panel (proposal browsing)
# =============================================================================


class TestPathSelectionPanelRuns:
    def _panel(self, ctx, graph):
        from skiresort_planner.ui.right_panel import PathSelectionPanel

        return PathSelectionPanel(
            context=ctx,
            graph=graph,
            on_commit=_noop,
            on_cancel_connection=_noop,
        )

    @staticmethod
    def _capture_buttons(fake_st) -> list[str]:
        """Record every button label rendered this pass (label is the 1st positional arg)."""
        labels: list[str] = []
        orig = fake_st.button

        def spy(*args: object, **kwargs: object) -> bool:
            if args:
                labels.append(str(args[0]))
            return orig(*args, **kwargs)

        fake_st.button = spy
        return labels

    def test_no_proposals_runs(self, fake_st, empty_graph) -> None:
        _sm, ctx = PlannerStateMachine.create(graph=empty_graph, add_ui_listener=False)
        self._panel(ctx, empty_graph).render()

    def test_with_selected_proposal_runs(self, fake_st, empty_graph, path_points_blue) -> None:
        _sm, ctx = PlannerStateMachine.create(graph=empty_graph, add_ui_listener=False)
        ctx.proposals.paths = [ProposedPathSegment(points=path_points_blue, target_difficulty="blue")]
        ctx.proposals.selected_idx = 0
        self._panel(ctx, empty_graph).render()

    def test_custom_target_shows_cancel_custom_path(self, fake_st, empty_graph, path_points_blue) -> None:
        # A plain custom target (no connector node) → "Cancel Custom Path", never "Cancel Connection".
        _sm, ctx = PlannerStateMachine.create(graph=empty_graph, add_ui_listener=False)
        ctx.custom_connect.force_mode = True
        ctx.proposals.paths = [ProposedPathSegment(points=path_points_blue, target_difficulty="blue")]
        ctx.proposals.selected_idx = 0
        labels = self._capture_buttons(fake_st)
        self._panel(ctx, empty_graph).render()
        assert any("Cancel Custom Path" in b for b in labels)
        assert not any("Cancel Connection" in b for b in labels)

    def test_connector_target_shows_cancel_connection(self, fake_st, empty_graph, path_points_blue) -> None:
        # A connector (routing to an existing node) → "Cancel Connection", matching the finish label.
        _sm, ctx = PlannerStateMachine.create(graph=empty_graph, add_ui_listener=False)
        ctx.custom_connect.force_mode = True
        ctx.proposals.paths = [
            ProposedPathSegment(
                points=path_points_blue, target_difficulty="blue", is_connector=True, target_node_id="N3"
            )
        ]
        ctx.proposals.selected_idx = 0
        labels = self._capture_buttons(fake_st)
        self._panel(ctx, empty_graph).render()
        assert any("Cancel Connection" in b for b in labels)
        assert not any("Cancel Custom Path" in b for b in labels)

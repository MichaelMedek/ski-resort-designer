"""Tests for the right control panel (ui/right_panel.py).

Uses the shared `fake_st` fixture so each panel's render() runs without a
browser. Two flavors: render tests assert the panel runs across slope/lift/road;
`fake_st.clicked_keys` fires a specific button so its body executes and the real
state change (3D toggle, close panel) is asserted.
"""

from typing import Literal

from skiresort_planner.model.path_point import PathPoint
from skiresort_planner.model.path_segment import SegmentKind
from skiresort_planner.model.proposed_path import ProposedPathSegment
from skiresort_planner.model.resort_graph import ResortGraph
from skiresort_planner.ui.context import EntityKind, PlannerContext
from skiresort_planner.ui.mode_registry import ENTITY_KIND_SPECS, render_control_panel
from skiresort_planner.ui.right_panel import (
    EntityInfoControlPanel,
    ImportPlacingControlPanel,
    LiftStatsPanel,
    MergePlacingControlPanel,
    RoadStatsPanel,
    SlopeStatsPanel,
)
from skiresort_planner.ui.state_machine import PlannerStateMachine

M = 111320.0


def _info_panel(kind: EntityKind, sm: PlannerStateMachine, ctx: PlannerContext, graph: ResortGraph) -> None:
    """Build the viewing-info ControlPanel for a kind and render it (on_commit/on_cancel unused)."""
    EntityInfoControlPanel(
        sm=sm,
        ctx=ctx,
        graph=graph,
        on_commit=lambda _i: None,
        on_cancel_connection=lambda: None,
        spec=ENTITY_KIND_SPECS[kind],
    ).render()


def _build_slope(graph: ResortGraph, path_points: list[PathPoint]) -> str:
    graph.commit_paths(paths=[ProposedPathSegment(points=path_points, target_difficulty="blue")])
    slope = graph.finish_slope(segment_ids=list(graph.segments.keys()))
    assert slope is not None
    return slope.id


def _build_road(graph: ResortGraph, path_points: list[PathPoint]) -> str:
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


def _capture_buttons(fake_st) -> list[str]:
    """Record every button label rendered this pass (label is the 1st positional arg)."""
    labels: list[str] = []
    orig = fake_st.button

    def spy(*args: object, **kwargs: object) -> bool:
        if args:
            labels.append(str(args[0]))
        return bool(orig(*args, **kwargs))

    fake_st.button = spy
    return labels


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
    captured to assert the distinguishing fields actually render.
    """

    @staticmethod
    def _capture_labels(fake_st) -> list[str]:
        labels: list[str] = []
        fake_st.metric = lambda label, *a, **k: labels.append(label)
        fake_st.subheader = lambda text, *a, **k: labels.append(text)
        return labels

    @staticmethod
    def _capture_metrics(fake_st) -> dict[str, str]:
        """Map each rendered metric LABEL -> its VALUE string (value is the 2nd positional arg)."""
        metrics: dict[str, str] = {}

        def _record(label: str, value: str = "", *a: object, **k: object) -> None:
            metrics[label] = value

        fake_st.metric = _record
        return metrics

    def test_slope_overall_gradient_value(self, fake_st, empty_graph, path_points_blue) -> None:
        # drop/length*100 rounded: 160m over ~799m = 20% (the 800m south blue path).
        metrics = self._capture_metrics(fake_st)
        SlopeStatsPanel(graph=empty_graph).render(slope_id=_build_slope(empty_graph, path_points_blue))
        assert metrics["Overall Gradient"] == "20%"
        assert metrics["Drop"] == "160m"

    def test_road_elevation_change_is_signed_negative(self, fake_st, empty_graph, path_points_blue) -> None:
        # The path drops 2500m -> 2340m, so end-start = -160m and the metric carries the sign.
        metrics = self._capture_metrics(fake_st)
        RoadStatsPanel(graph=empty_graph).render(road_id=_build_road(empty_graph, path_points_blue))
        assert metrics["Elevation Change"] == "-160m"

    def test_lift_rise_and_inclined_length_values(self, fake_st, empty_graph, mock_dem_blue_slope) -> None:
        # Rise = 2500-2300 = 200m; inclined = sqrt(200^2 + horizontal^2) with horizontal ~= 999m -> 1019m.
        metrics = self._capture_metrics(fake_st)
        LiftStatsPanel(graph=empty_graph).render(lift_id=_build_lift(empty_graph, mock_dem_blue_slope))
        assert metrics["Vertical Rise"] == "200m"
        assert metrics["Inclined Length"] == "1019m"

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
        slope_id = _build_slope(empty_graph, path_points_blue)
        sm, ctx = PlannerStateMachine.create(graph=empty_graph, add_ui_listener=False)
        sm.show_slope_info_panel(slope_id=slope_id)
        self._bump_ready(fake_st, sm, ctx, empty_graph)

        assert not ctx.viewing.view_3d
        fake_st.clicked_keys = {"slope_3d_view"}
        _info_panel(EntityKind.SLOPE, sm, ctx, empty_graph)
        assert ctx.viewing.view_3d, "clicking 'View in 3D' must enable 3D"

    def test_disable_3d_from_slope_panel(self, fake_st, empty_graph, path_points_blue) -> None:
        slope_id = _build_slope(empty_graph, path_points_blue)
        sm, ctx = PlannerStateMachine.create(graph=empty_graph, add_ui_listener=False)
        sm.show_slope_info_panel(slope_id=slope_id)
        ctx.viewing.enable_3d()
        self._bump_ready(fake_st, sm, ctx, empty_graph)

        fake_st.clicked_keys = {"slope_2d_view"}
        _info_panel(EntityKind.SLOPE, sm, ctx, empty_graph)
        assert not ctx.viewing.view_3d, "clicking 'Return to 2D' on a slope must disable 3D"

    def test_disable_3d_from_road_panel(self, fake_st, empty_graph, path_points_blue) -> None:
        road_id = _build_road(empty_graph, path_points_blue)
        sm, ctx = PlannerStateMachine.create(graph=empty_graph, add_ui_listener=False)
        sm.show_road_info_panel(road_id=road_id)
        ctx.viewing.enable_3d()
        self._bump_ready(fake_st, sm, ctx, empty_graph)

        fake_st.clicked_keys = {"road_2d_view"}
        _info_panel(EntityKind.ROAD, sm, ctx, empty_graph)
        assert not ctx.viewing.view_3d, "clicking 'Return to 2D' must disable 3D"

    def test_entity_actions_are_2x2_grid_and_ordered(self, fake_st, empty_graph, path_points_blue, monkeypatch) -> None:
        """The viewed-entity actions render as a 2x2 grid (two st.columns(2) rows) in the fixed
        order 3D-toggle + Rename on top, Close + Delete on the bottom; each fills its column.
        """
        from skiresort_planner.ui import right_panel

        slope_id = _build_slope(empty_graph, path_points_blue)
        sm, ctx = PlannerStateMachine.create(graph=empty_graph, add_ui_listener=False)
        sm.show_slope_info_panel(slope_id=slope_id)
        self._bump_ready(fake_st, sm, ctx, empty_graph)

        calls: list[dict[str, object]] = []

        def _record(*_args: object, **kwargs: object) -> bool:
            calls.append(kwargs)
            return False  # nothing clicked

        monkeypatch.setattr("skiresort_planner.ui.right_panel.st.button", _record)

        # Count st.columns(2) rows and provide context-manager column stubs.
        columns_calls: list[int] = []

        class _Col:
            def __enter__(self) -> "_Col":
                return self

            def __exit__(self, *_exc: object) -> Literal[False]:
                return False

        def _fake_columns(spec: int, **_k: object) -> tuple[_Col, ...]:
            columns_calls.append(spec)
            return tuple(_Col() for _ in range(spec))

        monkeypatch.setattr("skiresort_planner.ui.right_panel.st.columns", _fake_columns)

        right_panel._render_entity_actions(
            sm=sm,
            ctx=ctx,
            graph=empty_graph,
            kind=EntityKind.SLOPE,
            entity_id=slope_id,
            entity=empty_graph.slopes[slope_id],
            delete_fn=lambda _id: True,
        )

        assert columns_calls == [2, 2], "two rows of st.columns(2) → a 2x2 grid"
        keys = [c["key"] for c in calls]
        assert keys == ["slope_3d_view", "rename_slope", "close_slope", "delete_slope"], "fixed grid order"
        assert all(c.get("width") == "stretch" for c in calls), "each grid button fills its column"

    def test_enable_3d_from_lift_panel(self, fake_st, empty_graph, mock_dem_blue_slope) -> None:
        lift_id = _build_lift(empty_graph, mock_dem_blue_slope)
        sm, ctx = PlannerStateMachine.create(graph=empty_graph, add_ui_listener=False)
        sm.show_lift_info_panel(lift_id=lift_id)
        self._bump_ready(fake_st, sm, ctx, empty_graph)

        fake_st.clicked_keys = {"lift_3d_view"}
        _info_panel(EntityKind.LIFT, sm, ctx, empty_graph)
        assert ctx.viewing.view_3d, "clicking 'View in 3D' on a lift must enable 3D"

    def test_enable_3d_from_road_panel(self, fake_st, empty_graph, path_points_blue) -> None:
        road_id = _build_road(empty_graph, path_points_blue)
        sm, ctx = PlannerStateMachine.create(graph=empty_graph, add_ui_listener=False)
        sm.show_road_info_panel(road_id=road_id)
        self._bump_ready(fake_st, sm, ctx, empty_graph)

        assert not ctx.viewing.view_3d
        fake_st.clicked_keys = {"road_3d_view"}
        _info_panel(EntityKind.ROAD, sm, ctx, empty_graph)
        assert ctx.viewing.view_3d, "clicking 'View in 3D' on a road must enable 3D"

    def test_disable_3d_from_lift_panel(self, fake_st, empty_graph, mock_dem_blue_slope) -> None:
        lift_id = _build_lift(empty_graph, mock_dem_blue_slope)
        sm, ctx = PlannerStateMachine.create(graph=empty_graph, add_ui_listener=False)
        sm.show_lift_info_panel(lift_id=lift_id)
        ctx.viewing.enable_3d()
        self._bump_ready(fake_st, sm, ctx, empty_graph)

        fake_st.clicked_keys = {"lift_2d_view"}
        _info_panel(EntityKind.LIFT, sm, ctx, empty_graph)
        assert not ctx.viewing.view_3d, "clicking 'Return to 2D' on a lift must disable 3D"

    def test_close_slope_panel_returns_to_idle(self, fake_st, empty_graph, path_points_blue) -> None:
        slope_id = _build_slope(empty_graph, path_points_blue)
        sm, ctx = PlannerStateMachine.create(graph=empty_graph, add_ui_listener=False)
        sm.show_slope_info_panel(slope_id=slope_id)
        self._bump_ready(fake_st, sm, ctx, empty_graph)

        assert sm.is_idle_viewing_slope
        fake_st.clicked_keys = {"close_slope"}
        _info_panel(EntityKind.SLOPE, sm, ctx, empty_graph)
        assert not sm.is_idle_viewing_slope, "clicking Close must leave the viewing state"

    def test_rename_button_offered_in_each_panel(
        self, fake_st, empty_graph, path_points_blue, mock_dem_blue_slope
    ) -> None:
        """The ✏️ Rename button renders in all three detail panels (key rename_<kind>).

        We capture the button keys rather than click it — clicking would invoke the real
        @st.dialog. The dialog body's effect is covered by TestRenameEntityAction.
        """
        keys: list[str] = []
        orig_button = fake_st.button

        def _spy_button(*a: object, **k: object) -> bool:
            keys.append(str(k.get("key")))
            return bool(orig_button(*a, **k))

        fake_st.button = _spy_button

        slope_id = _build_slope(empty_graph, path_points_blue)
        road_id = _build_road(empty_graph, path_points_blue)
        lift_id = _build_lift(empty_graph, mock_dem_blue_slope)
        sm, ctx = PlannerStateMachine.create(graph=empty_graph, add_ui_listener=False)
        self._bump_ready(fake_st, sm, ctx, empty_graph)

        sm.show_slope_info_panel(slope_id=slope_id)
        _info_panel(EntityKind.SLOPE, sm, ctx, empty_graph)
        assert "rename_slope" in keys

        sm.hide_info_panel()
        sm.show_road_info_panel(road_id=road_id)
        _info_panel(EntityKind.ROAD, sm, ctx, empty_graph)
        assert "rename_road" in keys

        sm.hide_info_panel()
        sm.show_lift_info_panel(lift_id=lift_id)
        _info_panel(EntityKind.LIFT, sm, ctx, empty_graph)
        assert "rename_lift" in keys


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
        sm.commit_road(segment_id=seg, endpoint_node_id=empty_graph.segments[seg].end_node_id)  # type: ignore[attr-defined]  # dynamic python-statemachine event
        assert sm.is_road_building_only
        _dispatch(sm, ctx, empty_graph)

    def test_road_connector_shows_finish_label(self, fake_st, empty_graph, path_points_blue) -> None:
        # A road proposal onto an existing node is a connector → the shared _commit_button_label
        # gives the road panel "🏁 Finish → {node}", never plain "Commit Road Segment".
        sm, ctx = PlannerStateMachine.create(graph=empty_graph, add_ui_listener=False)
        sm.start_road(node_id=None, location=path_points_blue[0])
        ctx.proposals.paths = [
            ProposedPathSegment(points=path_points_blue, is_connector=True, target_node_id="N7", kind=SegmentKind.ROAD)
        ]
        ctx.proposals.selected_idx = 0

        labels = _capture_buttons(fake_st)
        _dispatch(sm, ctx, empty_graph)
        assert any("🏁 Finish → N7" in b for b in labels), "road connector commit shows the Finish label"
        assert not any("Commit Road Segment" in b for b in labels)

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
        labels = _capture_buttons(fake_st)
        self._panel(ctx, empty_graph).render()
        assert any("Cancel Custom Path" in b for b in labels)
        assert not any("Cancel Connection" in b for b in labels)

    def test_connector_target_shows_cancel_connection(self, fake_st, empty_graph, path_points_blue) -> None:
        # A connector (routing to an existing node) → "Cancel Connection" + the shared
        # "🏁 Finish → {node}" commit label (from _commit_button_label), never plain Commit.
        _sm, ctx = PlannerStateMachine.create(graph=empty_graph, add_ui_listener=False)
        ctx.custom_connect.force_mode = True
        ctx.proposals.paths = [
            ProposedPathSegment(
                points=path_points_blue, target_difficulty="blue", is_connector=True, target_node_id="N3"
            )
        ]
        ctx.proposals.selected_idx = 0
        labels = _capture_buttons(fake_st)
        self._panel(ctx, empty_graph).render()
        assert any("Cancel Connection" in b for b in labels)
        assert not any("Cancel Custom Path" in b for b in labels)
        assert any("🏁 Finish → N3" in b for b in labels), "slope connector commit shows the Finish label"
        assert not any("Commit This Path" in b for b in labels)


# =============================================================================
# Merge + Import placing panels (deferred-confirm buttons)
# =============================================================================


class TestMergeAndImportPanels:
    """The Merge/Import panels each render ONE deferred-confirm button. These drive the
    button-body logic and disabled-state, asserting the real action fires and the guard raises.
    """

    def _merge_panel(self, sm, ctx, graph) -> MergePlacingControlPanel:
        return MergePlacingControlPanel(sm=sm, ctx=ctx, graph=graph, on_commit=_noop, on_cancel_connection=_noop)

    def _import_panel(self, sm, ctx, graph) -> ImportPlacingControlPanel:
        return ImportPlacingControlPanel(sm=sm, ctx=ctx, graph=graph, on_commit=_noop, on_cancel_connection=_noop)

    def test_confirm_merge_disabled_below_two_nodes(self, fake_st, empty_graph, monkeypatch) -> None:
        # Confirm Merge is disabled with 0 or 1 selected nodes, enabled at 2 (see buttons()).
        sm, ctx = PlannerStateMachine.create(graph=empty_graph, add_ui_listener=False)
        seen: dict[str, object] = {}

        def _btn(label: str, **kwargs: object) -> bool:
            seen["label"] = label
            seen["disabled"] = kwargs.get("disabled")
            return False

        monkeypatch.setattr("skiresort_planner.ui.right_panel.st.button", _btn)
        panel = self._merge_panel(sm, ctx, empty_graph)

        ctx.merge.node_ids = []
        panel.buttons()
        assert seen["label"] == "🔗 Confirm Merge"
        assert seen["disabled"] is True

        ctx.merge.node_ids = ["A"]
        panel.buttons()
        assert seen["disabled"] is True

        ctx.merge.node_ids = ["A", "B"]
        panel.buttons()
        assert seen["disabled"] is False, "two selected nodes must enable Confirm Merge"

    def test_confirm_merge_fires_action(self, fake_st, empty_graph, monkeypatch) -> None:
        from skiresort_planner.ui import right_panel

        sm, ctx = PlannerStateMachine.create(graph=empty_graph, add_ui_listener=False)
        fired: list[bool] = []
        monkeypatch.setattr(right_panel, "confirm_merge_action", lambda: fired.append(True))
        # Fire only the enabled button (mirror the real disabled guard).
        monkeypatch.setattr(
            "skiresort_planner.ui.right_panel.st.button", lambda label, **k: not bool(k.get("disabled", False))
        )
        ctx.merge.node_ids = ["A", "B"]
        self._merge_panel(sm, ctx, empty_graph).buttons()
        assert fired == [True], "clicking Confirm Merge must invoke confirm_merge_action"

    def test_confirm_import_fires_action(self, fake_st, empty_graph, monkeypatch) -> None:
        from skiresort_planner.ui import right_panel

        sm, ctx = PlannerStateMachine.create(graph=empty_graph, add_ui_listener=False)
        fired: list[bool] = []
        monkeypatch.setattr(right_panel, "confirm_import_action", lambda: fired.append(True))
        monkeypatch.setattr("skiresort_planner.ui.right_panel.st.button", lambda label, **k: True)
        self._import_panel(sm, ctx, empty_graph).buttons()
        assert fired == [True], "clicking Confirm Import must invoke confirm_import_action"

    def test_import_context_message_requires_placed_center(self, fake_st, empty_graph) -> None:
        # A fresh context has no osm_import_center_* → context_message() raises the guard.
        import pytest

        sm, ctx = PlannerStateMachine.create(graph=empty_graph, add_ui_listener=False)
        panel = self._import_panel(sm, ctx, empty_graph)
        assert ctx.deferred.osm_import_center_lon is None
        with pytest.raises(RuntimeError, match="requires a placed box center"):
            panel.context_message()

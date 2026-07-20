"""Tests for the right control panel (ui/right_panel.py).

Uses the shared `fake_st` fixture so each panel's render() runs without a
browser. Two flavors: render tests assert the panel runs across slope/lift/road;
`fake_st.clicked_keys` fires a specific button so its body executes and the real
state change (3D toggle, close panel) is asserted.
"""

from contextlib import nullcontext
from typing import Literal

from skiresort_planner.constants import MapConfig, RoutePlannerConfig, StyleConfig
from skiresort_planner.model.path_point import PathPoint
from skiresort_planner.model.path_segment import SegmentKind
from skiresort_planner.model.proposed_path import ProposedPathSegment
from skiresort_planner.model.resort_graph import ResortGraph
from skiresort_planner.model.routing import RouteStep
from skiresort_planner.ui.context import BuildMode, EntityKind, PlannerContext
from skiresort_planner.ui.mode_registry import ENTITY_KIND_SPECS, render_control_panel
from skiresort_planner.ui.right_panel import (
    EntityInfoControlPanel,
    ImportSelectingControlPanel,
    LiftStatsPanel,
    NodeEditingControlPanel,
    PathStatsPanel,
    route_legs,
)
from skiresort_planner.ui.state_machine import PlannerStateMachine


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
        lon=0.0,
        lat=-1000 / MapConfig.METERS_PER_DEGREE_EQUATOR,
        elevation=dem.get_elevation_or_raise(lon=0.0, lat=-1000 / MapConfig.METERS_PER_DEGREE_EQUATOR),
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
    """Render the control panel for the current state. Guards dispatch-completeness: every build /
    viewing state must have a panel that renders without raising (a missing state → RuntimeError).
    """
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
    """PathStatsPanel (slope/road) + LiftStatsPanel each render their OWN distinguishing metric
    labels. Slope and road share one kind-parameterized class but must still show kind-correct
    labels (Drop vs Elevation Change, difficulty vs none); these capture the labels to assert that.
    """

    @staticmethod
    def _capture_labels(fake_st) -> list[str]:
        labels: list[str] = []
        fake_st.metric = lambda label, *a, **k: labels.append(label)
        fake_st.subheader = lambda text, *a, **k: labels.append(text)
        fake_st.markdown = lambda text, *a, **k: labels.append(text)
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
        # drop/length*100 rounded: 160m over ~799m = 20% (the 800m south blue path). Labels are
        # unified across kinds ("Average Gradient"/"Elevation Change") — only difficulty is slope-only.
        metrics = self._capture_metrics(fake_st)
        PathStatsPanel(graph=empty_graph, kind=SegmentKind.SLOPE).render(
            entity_id=_build_slope(empty_graph, path_points_blue)
        )
        assert metrics["Average Gradient"] == "20%"
        assert metrics["Elevation Change"] == "160m"

    def test_road_elevation_change_is_absolute(self, fake_st, empty_graph, path_points_blue) -> None:
        # The path drops 2500m -> 2340m; roads are bidirectional, so the metric shows the
        # magnitude with no sign (regression: was "-160m").
        metrics = self._capture_metrics(fake_st)
        PathStatsPanel(graph=empty_graph, kind=SegmentKind.ROAD).render(
            entity_id=_build_road(empty_graph, path_points_blue)
        )
        assert metrics["Elevation Change"] == "160m"

    def test_lift_rise_and_inclined_length_values(self, fake_st, empty_graph, mock_dem_blue_slope) -> None:
        # Rise = 2500-2300 = 200m; inclined = sqrt(200^2 + horizontal^2) with horizontal ~= 999m -> 1019m.
        metrics = self._capture_metrics(fake_st)
        LiftStatsPanel(graph=empty_graph).render(entity_id=_build_lift(empty_graph, mock_dem_blue_slope))
        assert metrics["Vertical Rise"] == "200m"
        assert metrics["Inclined Length"] == "1019m"

    def test_slope_stats_panel_shows_difficulty_and_unified_metrics(
        self, fake_st, empty_graph, path_points_blue
    ) -> None:
        labels = self._capture_labels(fake_st)
        PathStatsPanel(graph=empty_graph, kind=SegmentKind.SLOPE).render(
            entity_id=_build_slope(empty_graph, path_points_blue)
        )
        # Unified metric labels for every kind; the ski-specific bit is the Difficulty line (subheader-captured).
        assert {"Start Elevation", "Elevation Change", "Average Gradient", "Steepest Section"} <= set(labels)
        assert any("Difficulty" in label for label in labels), "a slope shows its ski difficulty"

    def test_road_stats_panel_shows_metrics_without_difficulty(self, fake_st, empty_graph, path_points_blue) -> None:
        labels = self._capture_labels(fake_st)
        PathStatsPanel(graph=empty_graph, kind=SegmentKind.ROAD).render(
            entity_id=_build_road(empty_graph, path_points_blue)
        )
        assert {"Start Elevation", "Elevation Change", "Average Gradient", "Steepest Section"} <= set(labels)
        assert not any("Difficulty" in label for label in labels), "a road has no ski difficulty"

    def test_lift_stats_panel_shows_lift_metrics(self, fake_st, empty_graph, mock_dem_blue_slope) -> None:
        labels = self._capture_labels(fake_st)
        LiftStatsPanel(graph=empty_graph).render(entity_id=_build_lift(empty_graph, mock_dem_blue_slope))
        assert {"Vertical Rise", "Pylons", "Inclined Length", "Steepest Section"} <= set(labels)

    def test_slope_and_road_both_show_segment_details_expander(self, fake_st, empty_graph, path_points_blue) -> None:
        """Both slope and road stats panels offer the 📋 Segment Details fold-out (roads used to lack it)."""
        expanders: list[str] = []

        def _expander(label, *a, **k):
            expanders.append(label)
            return nullcontext()

        fake_st.expander = _expander

        PathStatsPanel(graph=empty_graph, kind=SegmentKind.SLOPE).render(
            entity_id=_build_slope(empty_graph, path_points_blue)
        )
        assert any("Segment Details" in e for e in expanders)

        expanders.clear()
        PathStatsPanel(graph=empty_graph, kind=SegmentKind.ROAD).render(
            entity_id=_build_road(empty_graph, path_points_blue)
        )
        assert any("Segment Details" in e for e in expanders), "road panel must also show Segment Details"


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
        sm.view_slope(slope_id=slope_id)
        self._bump_ready(fake_st, sm, ctx, empty_graph)

        assert not ctx.viewing.view_3d
        fake_st.clicked_keys = {"slope_3d_view"}
        _info_panel(EntityKind.SLOPE, sm, ctx, empty_graph)
        assert ctx.viewing.view_3d, "clicking 'View in 3D' must enable 3D"

    def test_disable_3d_from_slope_panel(self, fake_st, empty_graph, path_points_blue) -> None:
        slope_id = _build_slope(empty_graph, path_points_blue)
        sm, ctx = PlannerStateMachine.create(graph=empty_graph, add_ui_listener=False)
        sm.view_slope(slope_id=slope_id)
        ctx.viewing.enable_3d()
        self._bump_ready(fake_st, sm, ctx, empty_graph)

        fake_st.clicked_keys = {"slope_2d_view"}
        _info_panel(EntityKind.SLOPE, sm, ctx, empty_graph)
        assert not ctx.viewing.view_3d, "clicking 'Return to 2D' on a slope must disable 3D"

    def test_disable_3d_from_road_panel(self, fake_st, empty_graph, path_points_blue) -> None:
        road_id = _build_road(empty_graph, path_points_blue)
        sm, ctx = PlannerStateMachine.create(graph=empty_graph, add_ui_listener=False)
        sm.view_road(road_id=road_id)
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
        sm.view_slope(slope_id=slope_id)
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
        sm.view_lift(lift_id=lift_id)
        self._bump_ready(fake_st, sm, ctx, empty_graph)

        fake_st.clicked_keys = {"lift_3d_view"}
        _info_panel(EntityKind.LIFT, sm, ctx, empty_graph)
        assert ctx.viewing.view_3d, "clicking 'View in 3D' on a lift must enable 3D"

    def test_enable_3d_from_road_panel(self, fake_st, empty_graph, path_points_blue) -> None:
        road_id = _build_road(empty_graph, path_points_blue)
        sm, ctx = PlannerStateMachine.create(graph=empty_graph, add_ui_listener=False)
        sm.view_road(road_id=road_id)
        self._bump_ready(fake_st, sm, ctx, empty_graph)

        assert not ctx.viewing.view_3d
        fake_st.clicked_keys = {"road_3d_view"}
        _info_panel(EntityKind.ROAD, sm, ctx, empty_graph)
        assert ctx.viewing.view_3d, "clicking 'View in 3D' on a road must enable 3D"

    def test_disable_3d_from_lift_panel(self, fake_st, empty_graph, mock_dem_blue_slope) -> None:
        lift_id = _build_lift(empty_graph, mock_dem_blue_slope)
        sm, ctx = PlannerStateMachine.create(graph=empty_graph, add_ui_listener=False)
        sm.view_lift(lift_id=lift_id)
        ctx.viewing.enable_3d()
        self._bump_ready(fake_st, sm, ctx, empty_graph)

        fake_st.clicked_keys = {"lift_2d_view"}
        _info_panel(EntityKind.LIFT, sm, ctx, empty_graph)
        assert not ctx.viewing.view_3d, "clicking 'Return to 2D' on a lift must disable 3D"

    def test_close_slope_panel_returns_to_idle(self, fake_st, empty_graph, path_points_blue) -> None:
        slope_id = _build_slope(empty_graph, path_points_blue)
        sm, ctx = PlannerStateMachine.create(graph=empty_graph, add_ui_listener=False)
        sm.view_slope(slope_id=slope_id)
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

        sm.view_slope(slope_id=slope_id)
        _info_panel(EntityKind.SLOPE, sm, ctx, empty_graph)
        assert "rename_slope" in keys

        sm.close_panel()  # type: ignore[attr-defined]  # dynamic python-statemachine event
        sm.view_road(road_id=road_id)
        _info_panel(EntityKind.ROAD, sm, ctx, empty_graph)
        assert "rename_road" in keys

        sm.close_panel()  # type: ignore[attr-defined]  # dynamic python-statemachine event
        sm.view_lift(lift_id=lift_id)
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
        sm.view_slope(slope_id=slope_id)
        _dispatch(sm, ctx, empty_graph)

    def test_viewing_road_panel_runs(self, fake_st, empty_graph, path_points_blue) -> None:
        road_id = _build_road(empty_graph, path_points_blue)
        sm, ctx = PlannerStateMachine.create(graph=empty_graph, add_ui_listener=False)
        sm.view_road(road_id=road_id)
        _dispatch(sm, ctx, empty_graph)

    def test_viewing_lift_panel_runs(self, fake_st, empty_graph, mock_dem_blue_slope) -> None:
        lift_id = _build_lift(empty_graph, mock_dem_blue_slope)
        sm, ctx = PlannerStateMachine.create(graph=empty_graph, add_ui_listener=False)
        sm.view_lift(lift_id=lift_id)
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
        sm.start_slope(lon=0.0, lat=0.0, elevation=mock_dem_blue_slope.get_elevation_or_raise(lon=0.0, lat=0.0))
        assert sm.is_slope_starting
        _dispatch(sm, ctx, empty_graph)

    def test_building_panel_surfaces_committed_segment_warning(
        self, fake_st, empty_graph, path_points_blue, monkeypatch
    ) -> None:
        """A committed segment's warning must surface as a ⚠️ warning MESSAGE in the building panel
        (not as a plot annotation) — regression for moving warnings off the elevation chart.
        """
        import streamlit as real_st

        from skiresort_planner.model.path_segment import PathSegment
        from skiresort_planner.model.warning import TooFlatWarning

        # SegmentWarningMessage.display() does a local `import streamlit` (WARNING → st.warning),
        # so capture on the real streamlit module, not the per-module fake.
        warnings: list[str] = []
        monkeypatch.setattr(real_st, "warning", lambda text, *a, **k: warnings.append(text), raising=False)
        monkeypatch.setattr(
            PathSegment, "warnings", property(lambda self: [TooFlatWarning(slope_pct=1.0, min_threshold_pct=5.0)])
        )

        sm, ctx = PlannerStateMachine.create(graph=empty_graph, add_ui_listener=False)
        sm.start_slope(
            lon=path_points_blue[0].lon, lat=path_points_blue[0].lat, elevation=path_points_blue[0].elevation
        )
        empty_graph.commit_paths(paths=[ProposedPathSegment(points=path_points_blue, target_difficulty="blue")])
        seg_id = list(empty_graph.segments.keys())[-1]
        sm.commit_path(segment_id=seg_id, endpoint_node_id=empty_graph.segments[seg_id].end_node_id)  # type: ignore[attr-defined]  # dynamic python-statemachine event

        _dispatch(sm, ctx, empty_graph)

        assert any("Too Flat" in w for w in warnings), (
            f"building panel must show the ⚠️ too-flat warning; got {warnings}"
        )

    def test_lift_placing_panel_runs(self, fake_st, empty_graph, mock_dem_blue_slope) -> None:
        from skiresort_planner.model.path_point import PathPoint

        sm, ctx = PlannerStateMachine.create(graph=empty_graph, add_ui_listener=False)
        ctx.build_mode.mode = BuildMode.CHAIRLIFT  # lift type selected before entering LIFT_PLACING
        loc = PathPoint(lon=0.0, lat=0.0, elevation=mock_dem_blue_slope.get_elevation_or_raise(lon=0.0, lat=0.0))
        sm.start_lift(node_id=None, location=loc)
        assert sm.is_lift_placing
        _dispatch(sm, ctx, empty_graph)


# =============================================================================
# Path selection panel (proposal browsing)
# =============================================================================


class TestPathSelectionPanelRuns:
    def _panel(self, ctx, graph, kind=SegmentKind.SLOPE):
        from skiresort_planner.ui.right_panel import PathSelectionPanel

        return PathSelectionPanel(
            context=ctx,
            graph=graph,
            kind=kind,
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

    def test_road_proposal_runs_without_difficulty_emoji(self, fake_st, empty_graph, path_points_blue) -> None:
        # Roads carry difficulty="" — the panel must NOT KeyError on the difficulty-emoji lookup
        # (the bug that forced roads onto a separate inline panel). One shared kind-aware panel.
        _sm, ctx = PlannerStateMachine.create(graph=empty_graph, add_ui_listener=False)
        ctx.proposals.paths = [
            ProposedPathSegment(points=path_points_blue, target_difficulty="", kind=SegmentKind.ROAD)
        ]
        ctx.proposals.selected_idx = 0
        self._panel(ctx, empty_graph, kind=SegmentKind.ROAD).render()  # must not raise

    def test_custom_target_shows_cancel_custom_path(self, fake_st, empty_graph, path_points_blue) -> None:
        # A plain custom target (no connector node) → "Cancel Custom Path", never "Cancel Connection".
        _sm, ctx = PlannerStateMachine.create(graph=empty_graph, add_ui_listener=False)
        ctx.custom_connect.target_location = (0.0, 0.0, 2000.0)  # force_mode derives from this
        ctx.proposals.paths = [ProposedPathSegment(points=path_points_blue, target_difficulty="blue")]
        ctx.proposals.selected_idx = 0
        labels = _capture_buttons(fake_st)
        self._panel(ctx, empty_graph).render()
        assert any("Cancel Custom Path" in b for b in labels)
        assert not any("Cancel Connection" in b for b in labels)

    def test_road_custom_target_shows_cancel_custom_path(self, fake_st, empty_graph, path_points_blue) -> None:
        # Parity: a ROAD custom target also gets the Cancel-Custom-Path affordance (the bug where the
        # road build panel had no way back to the fan). Same shared PathSelectionPanel.
        _sm, ctx = PlannerStateMachine.create(graph=empty_graph, add_ui_listener=False)
        ctx.custom_connect.target_location = (0.0, 0.0, 2000.0)  # force_mode derives from this
        ctx.proposals.paths = [
            ProposedPathSegment(points=path_points_blue, target_difficulty="", kind=SegmentKind.ROAD)
        ]
        ctx.proposals.selected_idx = 0
        labels = _capture_buttons(fake_st)
        self._panel(ctx, empty_graph, kind=SegmentKind.ROAD).render()
        assert any("Cancel Custom Path" in b for b in labels)

    def test_connector_target_shows_cancel_connection(self, fake_st, empty_graph, path_points_blue) -> None:
        # A connector (routing to an existing node) → "Cancel Connection" + the shared
        # "🏁 Finish → {node}" commit label (from _commit_button_label), never plain Commit.
        _sm, ctx = PlannerStateMachine.create(graph=empty_graph, add_ui_listener=False)
        ctx.custom_connect.target_location = (0.0, 0.0, 2000.0)  # force_mode derives from this
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
        assert not any("Commit This Slope" in b for b in labels)

    def test_custom_target_too_steep_still_offers_escape(self, fake_st, empty_graph) -> None:
        # Bug: clicking a too-steep custom target wipes proposals to empty. The panel used to
        # early-return the "No Paths Available" message with NO way back, trapping the user in
        # *_CUSTOM_PATH (force_mode) — only Undo escaped. Even with 0 proposals, force_mode must
        # still render the Cancel-Custom-Path escape so the user can return to the fan.
        _sm, ctx = PlannerStateMachine.create(graph=empty_graph, add_ui_listener=False)
        ctx.custom_connect.target_location = (0.0, 0.0, 2000.0)  # force_mode derives from this
        ctx.proposals.paths = []  # too-steep target → generator produced nothing
        labels = _capture_buttons(fake_st)
        self._panel(ctx, empty_graph, kind=SegmentKind.ROAD).render()
        assert any("Cancel Custom Path" in b for b in labels), (
            "an empty custom-connect result must still show the escape button, not trap the user"
        )


# =============================================================================
# Merge + Import placing panels (deferred-confirm buttons)
# =============================================================================


class TestMergeAndImportPanels:
    """The Merge/Import panels each render ONE deferred-confirm button. These drive the
    button-body logic and disabled-state, asserting the real action fires and the guard raises.
    """

    def _merge_panel(self, sm, ctx, graph) -> NodeEditingControlPanel:
        return NodeEditingControlPanel(sm=sm, ctx=ctx, graph=graph, on_commit=_noop, on_cancel_connection=_noop)

    def _import_panel(self, sm, ctx, graph) -> ImportSelectingControlPanel:
        return ImportSelectingControlPanel(sm=sm, ctx=ctx, graph=graph, on_commit=_noop, on_cancel_connection=_noop)

    def test_confirm_merge_disabled_below_two_nodes(self, fake_st, empty_graph, monkeypatch) -> None:
        # Confirm Merge is disabled with 0 or 1 selected nodes, enabled at 2; Delete needs only 1.
        sm, ctx = PlannerStateMachine.create(graph=empty_graph, add_ui_listener=False)
        panel = self._merge_panel(sm, ctx, empty_graph)

        def _disabled_for(node_ids: list[str]) -> dict[str, bool]:
            captured: dict[str, bool] = {}

            def _btn(label: str, **k: object) -> bool:
                captured[label] = bool(k.get("disabled"))
                return False

            monkeypatch.setattr("skiresort_planner.ui.right_panel.st.button", _btn)
            ctx.node_edit.node_ids = node_ids
            panel.buttons()
            return captured

        at_zero = _disabled_for([])
        assert at_zero["🔗 Confirm Merge"] and at_zero["🗑️ Delete Node(s)"], "both disabled at 0 selected"

        at_one = _disabled_for(["A"])
        assert at_one["🔗 Confirm Merge"], "Confirm Merge still disabled at 1 selected"
        assert not at_one["🗑️ Delete Node(s)"], "one node enables Delete"

        at_two = _disabled_for(["A", "B"])
        assert not at_two["🔗 Confirm Merge"], "two selected nodes must enable Confirm Merge"

    def test_confirm_merge_fires_action(self, fake_st, empty_graph, monkeypatch) -> None:
        from skiresort_planner.ui import right_panel

        sm, ctx = PlannerStateMachine.create(graph=empty_graph, add_ui_listener=False)
        fired: list[bool] = []
        monkeypatch.setattr(right_panel, "confirm_merge_action", lambda: fired.append(True))
        monkeypatch.setattr(right_panel, "delete_nodes_action", lambda: None)  # only assert the merge button
        # Fire only the enabled Confirm Merge button (mirror the real disabled guard).
        monkeypatch.setattr(
            "skiresort_planner.ui.right_panel.st.button",
            lambda label, **k: label == "🔗 Confirm Merge" and not bool(k.get("disabled", False)),
        )
        ctx.node_edit.node_ids = ["A", "B"]
        self._merge_panel(sm, ctx, empty_graph).buttons()
        assert fired == [True], "clicking Confirm Merge must invoke confirm_merge_action"

    def test_delete_button_fires_action(self, fake_st, empty_graph, monkeypatch) -> None:
        from skiresort_planner.ui import right_panel

        sm, ctx = PlannerStateMachine.create(graph=empty_graph, add_ui_listener=False)
        fired: list[bool] = []
        monkeypatch.setattr(right_panel, "delete_nodes_action", lambda: fired.append(True))
        monkeypatch.setattr(right_panel, "confirm_merge_action", lambda: None)  # only assert the delete button
        monkeypatch.setattr(
            "skiresort_planner.ui.right_panel.st.button",
            lambda label, **k: label == "🗑️ Delete Node(s)" and not bool(k.get("disabled", False)),
        )
        ctx.node_edit.node_ids = ["A"]
        self._merge_panel(sm, ctx, empty_graph).buttons()
        assert fired == [True], "clicking Delete Node(s) must invoke delete_nodes_action"

    def test_confirm_import_fires_action(self, fake_st, empty_graph, monkeypatch) -> None:
        from skiresort_planner.constants import OSMImportMode
        from skiresort_planner.ui import right_panel

        sm, ctx = PlannerStateMachine.create(graph=empty_graph, add_ui_listener=False)
        fired: list[OSMImportMode] = []
        monkeypatch.setattr(right_panel, "confirm_import_action", lambda mode: fired.append(mode))
        monkeypatch.setattr("skiresort_planner.ui.right_panel.st.button", lambda label, **k: True)
        self._import_panel(sm, ctx, empty_graph).buttons()
        # Both buttons render True here, so both modes fire — proving each is wired to its mode.
        assert fired == [OSMImportMode.LIFTS_AND_SLOPES, OSMImportMode.LIFTS_ONLY], (
            "the two import buttons must invoke confirm_import_action with their modes"
        )

    def test_import_context_message_requires_placed_center(self, fake_st, empty_graph) -> None:
        # A fresh context has no osm_import_center_* → context_message() raises the guard.
        import pytest

        sm, ctx = PlannerStateMachine.create(graph=empty_graph, add_ui_listener=False)
        panel = self._import_panel(sm, ctx, empty_graph)
        assert ctx.pending.osm_import_center_lon is None
        with pytest.raises(RuntimeError, match="requires a placed box center"):
            panel.context_message()


class TestRouteLegs:
    """route_legs collapses a route's flat steps into readable legs (lifts kept, slopes grouped)."""

    @staticmethod
    def _slope(name: str, difficulty: str) -> RouteStep:
        return RouteStep(is_lift=False, entity_id=name, name=name, detail=difficulty)

    @staticmethod
    def _lift(name: str) -> RouteStep:
        return RouteStep(is_lift=True, entity_id=name, name=name, detail="chairlift")

    def test_lift_is_its_own_leg_with_type_icon(self) -> None:
        legs = route_legs((self._lift("Gondi"),))
        assert legs == [f"{StyleConfig.LIFT_ICONS['chairlift']} **Gondi**"]

    def test_consecutive_slopes_fold_into_one_leg(self) -> None:
        steps = (self._slope("A", "blue"), self._slope("B", "red"))
        legs = route_legs(steps)
        assert len(legs) == 1, "two consecutive slopes collapse to a single leg"
        assert "A" in legs[0] and "B" in legs[0]

    def test_difficulty_shown_as_colour_emoji_not_text(self) -> None:
        legs = route_legs((self._slope("Steep", "black"),))
        assert StyleConfig.DIFFICULTY_EMOJIS["black"] in legs[0]
        assert "black" not in legs[0], "difficulty is the colour emoji, never the '(black)' word"

    def test_more_than_preview_slopes_truncate_with_ellipsis(self) -> None:
        n = RoutePlannerConfig.ROUTE_STEP_SLOPE_PREVIEW
        steps = tuple(self._slope(f"S{i}", "blue") for i in range(n + 2))
        legs = route_legs(steps)
        assert legs[0].endswith("…"), "a leg with more than the preview count ends with an ellipsis"
        assert "S0" in legs[0] and f"S{n - 1}" in legs[0]
        assert f"S{n}" not in legs[0], "slopes beyond the preview count are not named"

    def test_exactly_preview_slopes_have_no_ellipsis(self) -> None:
        n = RoutePlannerConfig.ROUTE_STEP_SLOPE_PREVIEW
        legs = route_legs(tuple(self._slope(f"S{i}", "blue") for i in range(n)))
        assert not legs[0].endswith("…"), "exactly the preview count fits without an ellipsis"

    def test_lift_slope_lift_alternation_splits_legs(self) -> None:
        steps = (self._lift("Up1"), self._slope("Run", "red"), self._lift("Up2"))
        legs = route_legs(steps)
        assert len(legs) == 3, "a lift boundary flushes the slope run into its own leg"
        assert "Up1" in legs[0] and "Run" in legs[1] and "Up2" in legs[2]

    def test_empty_steps_yield_no_legs(self) -> None:
        assert route_legs(()) == []

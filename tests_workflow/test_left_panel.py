"""Tests for the sidebar / left panel (ui/left_panel.py).

Covers SidebarRenderer across build modes and viewing states, the build-mode
button click, and the `_describe_undo_action` label logic for every undo type.
Uses the shared `fake_st` fixture (no browser).
"""

from contextlib import nullcontext

import pytest

from skiresort_planner.model.path_point import PathPoint
from skiresort_planner.model.path_segment import SegmentKind
from skiresort_planner.model.proposed_path import ProposedPathSegment
from skiresort_planner.model.resort_graph import ResortGraph
from skiresort_planner.ui.context import BuildMode
from skiresort_planner.ui.left_panel import SidebarRenderer
from skiresort_planner.ui.state_machine import PlannerStateMachine

M = 111320.0


def _build_slope(graph: ResortGraph, path_points: list[PathPoint]) -> str:
    graph.commit_paths(paths=[ProposedPathSegment(points=path_points, target_difficulty="blue")])
    slope = graph.finish_slope(segment_ids=list(graph.segments.keys()))
    assert slope is not None
    return slope.id


def _build_road(graph: ResortGraph, path_points: list[PathPoint]) -> str:
    graph.commit_paths(paths=[ProposedPathSegment(points=path_points, is_connector=True, kind=SegmentKind.ROAD)])
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


# =============================================================================
# Sidebar render across modes / states
# =============================================================================


class TestSidebarRuns:
    @pytest.mark.parametrize("mode", [BuildMode.SLOPE, BuildMode.ROAD, BuildMode.CHAIRLIFT])
    def test_sidebar_runs_in_each_mode(self, fake_st, empty_graph, mode: str) -> None:
        sm, ctx = PlannerStateMachine.create(graph=empty_graph, add_ui_listener=False)
        ctx.build_mode.mode = mode
        # render() is fire-and-forget (returns None); it just needs to run without raising.
        SidebarRenderer(state_machine=sm, context=ctx, graph=empty_graph).render()

    def test_sidebar_runs_with_content(self, fake_st, empty_graph, path_points_blue, mock_dem_blue_slope) -> None:
        # A resort with a slope + lift + road exercises every summary section.
        _build_slope(empty_graph, path_points_blue)
        _build_lift(empty_graph, mock_dem_blue_slope)
        _build_road(empty_graph, path_points_blue)

        sm, ctx = PlannerStateMachine.create(graph=empty_graph, add_ui_listener=False)
        SidebarRenderer(state_machine=sm, context=ctx, graph=empty_graph).render()

    def test_sidebar_during_slope_building(self, fake_st, empty_graph, mock_dem_blue_slope) -> None:
        # Building state renders the building controls + undo/reset buttons.
        sm, ctx = PlannerStateMachine.create(graph=empty_graph, add_ui_listener=False)
        sm.start_slope(lon=0.0, lat=0.0, elevation=mock_dem_blue_slope.get_elevation_or_raise(lon=0.0, lat=0.0))
        SidebarRenderer(state_machine=sm, context=ctx, graph=empty_graph).render()

    def test_sidebar_during_road_building(self, fake_st, empty_graph) -> None:
        # Road building state renders the Finish Road / Cancel Road controls; clicking Cancel Road
        # fires cancel_current_road directly (fire-and-forget) and returns to idle.
        sm, ctx = PlannerStateMachine.create(graph=empty_graph, add_ui_listener=False)
        sm.start_road(node_id=None, location=PathPoint(lon=0.0, lat=0.0, elevation=2000.0))
        assert sm.is_road_starting
        fake_st.session_state["state_machine"] = sm
        fake_st.session_state["context"] = ctx
        fake_st.session_state["graph"] = empty_graph
        fake_st.session_state["map_version"] = 0

        fake_st.clicked_keys = {"cancel_road_btn"}
        SidebarRenderer(state_machine=sm, context=ctx, graph=empty_graph).render()
        assert sm.is_idle_ready, "clicking Cancel Road must discard the road and return to idle"

    @pytest.mark.parametrize("kind", ["slope", "road", "lift"])
    def test_sidebar_viewing_header_and_body_match_kind(
        self, fake_st, empty_graph, path_points_blue, mock_dem_blue_slope, kind: str
    ) -> None:
        # Every viewed kind gets its OWN header + body — no kind falls through to the
        # generic idle text (the drift that once left the road body wrong).
        sm, ctx = PlannerStateMachine.create(graph=empty_graph, add_ui_listener=False)
        if kind == "slope":
            sm.view_slope(slope_id=_build_slope(empty_graph, path_points_blue))
        elif kind == "road":
            sm.view_road(road_id=_build_road(empty_graph, path_points_blue))
        elif kind == "lift":
            sm.view_lift(lift_id=_build_lift(empty_graph, mock_dem_blue_slope))
        else:
            raise ValueError

        # The header (icon + label) is now the expander title; the bullets are its markdown body.
        # Capture both so the kind-specific header + body are visible to the assertions.
        captured: list[str] = []

        def _capture_expander(label: str, *a: object, **k: object) -> nullcontext[None]:
            captured.append(label)
            return nullcontext()

        fake_st.markdown = lambda text, *a, **k: captured.append(text)
        fake_st.expander = _capture_expander
        SidebarRenderer(state_machine=sm, context=ctx, graph=empty_graph).render()
        joined = "\n".join(captured)

        assert f"Viewing {kind.capitalize()}" in joined  # kind-specific header
        assert f"new {kind}" in joined  # kind-specific body, not "start building"
        assert "Select **Slope**" not in joined  # generic idle body must NOT appear
        # Only lifts show the change-type hint.
        assert ("Use lift buttons to change type" in joined) is (kind == "lift")

    def test_sidebar_building_state_renders_consolidated_block(self, fake_st, empty_graph) -> None:
        # Building/placing states render the SAME collapsed info block as idle/viewing (header label
        # + the "complete or cancel" bullet), not the old plain markdown/caption pair.
        sm, ctx = PlannerStateMachine.create(graph=empty_graph, add_ui_listener=False)
        sm.start_import(lon=0.0, lat=0.0)
        assert sm.is_import_placing

        captured: list[str] = []
        fake_st.markdown = lambda text, *a, **k: captured.append(text)
        SidebarRenderer(state_machine=sm, context=ctx, graph=empty_graph).render()
        joined = "\n".join(captured)

        assert "Complete or cancel current build to change type" in joined


class TestSidebarButtonHelpCompleteness:
    """Every registered state must render the mode-selector without a button's help text raising.

    `_get_button_help` / `_disabled_button_reason` must resolve a help string in EVERY state.
    """

    def test_every_state_renders_all_button_help_without_raising(self, fake_st, empty_graph) -> None:
        from skiresort_planner.ui.mode_registry import BUILD_STATES, OPERATIONS

        for state_id in BUILD_STATES:
            sm, ctx = PlannerStateMachine.create(graph=empty_graph, add_ui_listener=False)
            sm.current_state = getattr(sm, state_id)  # force the state (bypasses guards — test only)
            renderer = SidebarRenderer(state_machine=sm, context=ctx, graph=empty_graph)
            for mode, op in OPERATIONS.items():
                label = BuildMode.display_name(mode)
                # Whatever the state, the help text must resolve (disabled-reason or enabled-action) —
                # never hit the fall-through raise.
                renderer._get_button_help(
                    mode=mode,
                    label=label,
                    is_disabled=not op.enabled(sm),
                    is_building_or_placing=BUILD_STATES[state_id].blocks_build_buttons(),
                )


class TestModeSelectorButton:
    def test_click_road_mode_button_switches_mode(self, fake_st, empty_graph) -> None:
        sm, ctx = PlannerStateMachine.create(graph=empty_graph, add_ui_listener=False)
        ctx.build_mode.mode = BuildMode.SLOPE
        fake_st.session_state["state_machine"] = sm
        fake_st.session_state["context"] = ctx
        fake_st.session_state["graph"] = empty_graph
        fake_st.session_state["map_version"] = 0

        fake_st.clicked_keys = {"build_btn_road"}
        SidebarRenderer(state_machine=sm, context=ctx, graph=empty_graph).render()
        assert ctx.build_mode.mode == BuildMode.ROAD, "clicking the Road button must switch build mode"


class TestImportOSMButton:
    def test_click_import_button_selects_import_mode(self, fake_st, empty_graph) -> None:
        sm, ctx = PlannerStateMachine.create(graph=empty_graph, add_ui_listener=False)
        fake_st.session_state["state_machine"] = sm
        fake_st.session_state["context"] = ctx
        fake_st.session_state["graph"] = empty_graph
        fake_st.session_state["map_version"] = 0

        fake_st.clicked_keys = {"build_btn_import"}
        SidebarRenderer(state_machine=sm, context=ctx, graph=empty_graph).render()
        # Selecting Import only arms the click-to-place mode; it stays idle and does NOT flag a fetch.
        assert ctx.build_mode.mode == BuildMode.IMPORT, "clicking Import must select import mode"
        assert sm.is_idle_ready, "selecting a mode must not leave idle"
        assert ctx.pending.osm_import_mode is None, "import is not flagged until the box is placed + confirmed"


class TestPathSettingsVisibility:
    """The ⚙️ Path Settings block only applies to fan-out proposals, so it is hidden
    while routing a custom-connect path (force_mode).
    """

    @staticmethod
    def _capture_markdown(fake_st) -> list[str]:
        seen: list[str] = []
        fake_st.markdown = lambda text, *a, **k: seen.append(text)
        return seen

    def test_path_settings_shown_in_fan_out(self, fake_st, empty_graph, mock_dem_blue_slope) -> None:
        sm, ctx = PlannerStateMachine.create(graph=empty_graph, add_ui_listener=False)
        sm.start_slope(lon=0.0, lat=0.0, elevation=mock_dem_blue_slope.get_elevation_or_raise(lon=0.0, lat=0.0))
        seen = self._capture_markdown(fake_st)
        SidebarRenderer(state_machine=sm, context=ctx, graph=empty_graph).render()
        assert any("Path Settings" in m for m in seen), "fan-out mode shows the Path Settings block"

    def test_path_settings_hidden_in_custom_mode(self, fake_st, empty_graph, mock_dem_blue_slope) -> None:
        sm, ctx = PlannerStateMachine.create(graph=empty_graph, add_ui_listener=False)
        sm.start_slope(lon=0.0, lat=0.0, elevation=mock_dem_blue_slope.get_elevation_or_raise(lon=0.0, lat=0.0))
        ctx.custom_connect.target_location = (0.0, 0.0, 2000.0)  # showing custom-connect proposals (force_mode)
        seen = self._capture_markdown(fake_st)
        SidebarRenderer(state_machine=sm, context=ctx, graph=empty_graph).render()
        assert not any("Path Settings" in m for m in seen), "custom mode hides the Path Settings block"


# =============================================================================
# Undo-action labels (_describe_undo_action, one per action type)
# =============================================================================


class TestDescribeUndoAction:
    """_describe_undo_action labels every undo type. Each action comes from a
    real graph mutation that pushes it onto graph.undo_stack — no stubs.
    """

    def _describe_top(self, graph: ResortGraph) -> str:
        from skiresort_planner.ui.left_panel import _describe_undo_action

        return _describe_undo_action(action=graph.undo_stack[-1], graph=graph)

    def test_add_segments_has_no_describe_text(self, empty_graph, path_points_blue) -> None:
        # AddSegments is skip_confirm (peeling a segment shows no dialog), so its describe is empty.
        empty_graph.commit_paths(paths=[ProposedPathSegment(points=path_points_blue, target_difficulty="blue")])
        assert self._describe_top(empty_graph) == ""

    def test_finish_slope_label(self, empty_graph, path_points_blue) -> None:
        slope_id = _build_slope(empty_graph, path_points_blue)
        assert empty_graph.slopes[slope_id].name in self._describe_top(empty_graph)

    def test_add_lift_label(self, empty_graph, mock_dem_blue_slope) -> None:
        lift_id = _build_lift(empty_graph, mock_dem_blue_slope)
        label = self._describe_top(empty_graph)
        assert "Delete lift" in label and empty_graph.lifts[lift_id].name in label

    def test_finish_road_label(self, empty_graph, path_points_blue) -> None:
        road_id = _build_road(empty_graph, path_points_blue)  # finish_road records FINISH_ROAD on top
        label = self._describe_top(empty_graph)
        assert "Restore road" in label and empty_graph.roads[road_id].name in label

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

    def test_import_osm_label(self, empty_graph, mock_dem_blue_slope) -> None:
        from skiresort_planner.generators.osm_importer import ImportResult

        dem = mock_dem_blue_slope
        m = 111320.0
        slope_points = [
            PathPoint(lon=0.0, lat=0.0, elevation=dem.get_elevation_or_raise(lon=0.0, lat=0.0)),
            PathPoint(lon=0.0, lat=-500 / m, elevation=dem.get_elevation_or_raise(lon=0.0, lat=-500 / m)),
        ]
        empty_graph.import_osm(ImportResult(slope_chains=[([slope_points], "Run")]), dem=dem)
        assert "OSM import" in self._describe_top(empty_graph)

    def test_merge_nodes_label(self, empty_graph, mock_dem_blue_slope) -> None:
        # The 9th ActionType. Two nodes ~200m apart merge (< MergeConfig.MAX_SPAN_M=500m,
        # > STEP_SIZE_M=30m so they don't snap into one). deleted_nodes has 1 entry, so
        # _MergeNodesHandler.describe reports len(deleted_nodes) + 1 == 2 nodes.
        dem = mock_dem_blue_slope
        a, _ = empty_graph.get_or_create_node(lon=0.0, lat=0.0, elevation=dem.get_elevation_or_raise(lon=0.0, lat=0.0))
        b, _ = empty_graph.get_or_create_node(
            lon=0.0, lat=-200 / M, elevation=dem.get_elevation_or_raise(lon=0.0, lat=-200 / M)
        )
        empty_graph.merge_nodes(node_ids=[a.id, b.id], dem=dem)
        assert self._describe_top(empty_graph) == "Un-merge 2 nodes"

    def test_build_cancel_undo_skips_dialog_instead_of_describing(
        self, empty_graph, path_points_blue, mock_dem_blue_slope
    ) -> None:
        """In a build state with no committed segments, the next undo CANCELS the build — a routine
        one-tap step that skips the confirmation dialog, so it is never described (no off-by-one
        describe of the stale stack entry either).
        """
        from skiresort_planner.ui.left_panel import _next_undo_skips_confirm

        # A prior committed slope sits on the undo stack (the stale entry the old code would describe).
        _build_slope(empty_graph, path_points_blue)
        sm, ctx = PlannerStateMachine.create(graph=empty_graph, add_ui_listener=False)
        # Now START a new slope build with no committed segments yet.
        sm.start_slope(lon=0.0, lat=0.0, elevation=mock_dem_blue_slope.get_elevation_or_raise(lon=0.0, lat=0.0))
        assert sm.is_slope_starting

        assert _next_undo_skips_confirm(sm=sm, ctx=ctx, graph=empty_graph), (
            "cancelling a just-started build is a routine step → no confirmation dialog"
        )


class TestNextUndoSkipsConfirm:
    """The undo dialog is skipped only for routine builder steps: peeling a just-committed segment
    or cancelling a not-yet-committed build. Destructive actions (finish/delete/merge/import) confirm.
    """

    def test_add_segments_skips_confirm(self, empty_graph, path_points_blue) -> None:
        from skiresort_planner.ui.left_panel import _next_undo_skips_confirm

        empty_graph.commit_paths(paths=[ProposedPathSegment(points=path_points_blue, target_difficulty="blue")])
        sm, ctx = PlannerStateMachine.create(graph=empty_graph, add_ui_listener=False)
        assert _next_undo_skips_confirm(sm=sm, ctx=ctx, graph=empty_graph), "peeling a segment is a normal step"

    def test_finish_slope_requires_confirm(self, empty_graph, path_points_blue) -> None:
        from skiresort_planner.ui.left_panel import _next_undo_skips_confirm

        _build_slope(empty_graph, path_points_blue)  # top of stack is now FinishSlopeAction
        sm, ctx = PlannerStateMachine.create(graph=empty_graph, add_ui_listener=False)
        assert not _next_undo_skips_confirm(sm=sm, ctx=ctx, graph=empty_graph), "finishing a slope must confirm"

    def test_build_cancel_skips_confirm(self, empty_graph, path_points_blue, mock_dem_blue_slope) -> None:
        from skiresort_planner.ui.left_panel import _next_undo_skips_confirm

        _build_slope(empty_graph, path_points_blue)  # a stale FinishSlopeAction sits on the stack
        sm, ctx = PlannerStateMachine.create(graph=empty_graph, add_ui_listener=False)
        sm.start_slope(lon=0.0, lat=0.0, elevation=mock_dem_blue_slope.get_elevation_or_raise(lon=0.0, lat=0.0))
        # No committed segments yet → undo cancels the build, a routine one-tap step.
        assert _next_undo_skips_confirm(sm=sm, ctx=ctx, graph=empty_graph)


# =============================================================================
# Dialog action helpers (extracted from @st.dialog bodies to be testable)
# =============================================================================


class TestDialogHelpers:
    def test_request_pending_undo_sets_flag(self, fake_st) -> None:
        from skiresort_planner.ui.left_panel import _request_pending_undo

        _request_pending_undo()
        assert fake_st.session_state["_pending_undo"] is True

    def test_perform_reset_resort_deletes_backup_and_clears_session(self, fake_st, monkeypatch) -> None:
        from skiresort_planner.ui import left_panel

        deleted: list[str] = []
        monkeypatch.setattr(
            "skiresort_planner.ui.left_panel.backup_store.delete", lambda resort_id: deleted.append(resort_id)
        )
        monkeypatch.setattr("skiresort_planner.ui.left_panel.backup_store.new_resort_id", lambda: "fresh999")

        # Seed a full session that reset must tear down.
        for key in ("resort_id", "graph", "state_machine", "context", "map_renderer", "_saved_token"):
            fake_st.session_state[key] = object()
        fake_st.session_state["resort_id"] = "old123"

        left_panel._perform_reset_resort()

        assert deleted == ["old123"], "current backup must be deleted"
        assert fake_st.query_params["resort"] == "fresh999", "a fresh resort id is routed"
        for key in ("resort_id", "graph", "state_machine", "context", "map_renderer", "_saved_token"):
            assert key not in fake_st.session_state, f"{key} must be dropped so init rebuilds fresh"


# =============================================================================
# Load-from-file guard: only an empty resort may be overwritten by an upload
# =============================================================================


class _FakeUpload:
    """Minimal stand-in for Streamlit's UploadedFile: a JSON-readable, named handle."""

    def __init__(self, payload: dict[str, object]) -> None:
        import io
        import json

        self._buf = io.StringIO(json.dumps(payload))
        self.name = "uploaded.json"

    def read(self, size: int = -1) -> str:
        return self._buf.read(size)


class TestLoadFromFileGuard:
    def _render(self, fake_st, monkeypatch, graph, upload) -> list[str]:
        """Render the sidebar with `upload` returned by the file uploader; capture toasts."""
        import streamlit

        toasts: list[str] = []
        monkeypatch.setattr(streamlit, "toast", lambda text, *a, **k: toasts.append(text))
        fake_st.file_uploader = lambda *a, **k: upload
        fake_st.session_state["resort_id"] = "r1"

        sm, ctx = PlannerStateMachine.create(graph=graph, add_ui_listener=False)
        fake_st.session_state["graph"] = graph
        SidebarRenderer(state_machine=sm, context=ctx, graph=graph).render()
        return toasts

    def test_upload_rejected_when_resort_has_content(self, fake_st, monkeypatch, empty_graph, path_points_blue) -> None:
        # Regression: a non-empty resort must NOT be overwritten by an upload. It stays intact and
        # the user is told to clear first.
        _build_slope(empty_graph, path_points_blue)
        payload = ResortGraph().to_dict()  # a valid but empty file — still refused

        toasts = self._render(fake_st, monkeypatch, empty_graph, _FakeUpload(payload))

        assert fake_st.session_state["graph"] is empty_graph, "current resort must be untouched"
        assert empty_graph.slopes, "the existing slope must survive"
        assert any("Clear the resort first" in t for t in toasts), "must warn to clear first"

    def test_upload_loads_into_empty_resort(self, fake_st, monkeypatch, empty_graph, path_points_blue) -> None:
        # Happy path: an empty resort accepts the upload and is replaced by the file's content.
        source = ResortGraph()
        _build_slope(source, path_points_blue)
        payload = source.to_dict()

        toasts = self._render(fake_st, monkeypatch, empty_graph, _FakeUpload(payload))

        loaded = fake_st.session_state["graph"]
        assert loaded is not empty_graph, "the empty graph is replaced by the loaded one"
        assert loaded.slopes, "loaded resort carries the file's slope"
        assert not any("Clear the resort first" in t for t in toasts), "no rejection on empty resort"

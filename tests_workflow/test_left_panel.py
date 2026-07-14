"""Tests for the sidebar / left panel (ui/left_panel.py).

Covers SidebarRenderer across build modes and viewing states, the build-mode
button click, and the `_describe_undo_action` label logic for every undo type.
Uses the shared `fake_st` fixture (no browser).
"""

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
        sm.start_building(lon=0.0, lat=0.0, elevation=mock_dem_blue_slope.get_elevation_or_raise(lon=0.0, lat=0.0))
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
            sm.show_slope_info_panel(slope_id=_build_slope(empty_graph, path_points_blue))
        elif kind == "road":
            sm.show_road_info_panel(road_id=_build_road(empty_graph, path_points_blue))
        elif kind == "lift":
            sm.show_lift_info_panel(lift_id=_build_lift(empty_graph, mock_dem_blue_slope))
        else:
            raise ValueError

        captured: list[str] = []
        fake_st.markdown = lambda text, *a, **k: captured.append(text)
        SidebarRenderer(state_machine=sm, context=ctx, graph=empty_graph).render()
        joined = "\n".join(captured)

        assert f"Viewing {kind.capitalize()}" in joined  # kind-specific header
        assert f"new {kind}" in joined  # kind-specific body, not "start building"
        assert "Select **Slope**" not in joined  # generic idle body must NOT appear
        # Only lifts show the change-type hint.
        assert ("Use lift buttons to change type" in joined) is (kind == "lift")


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
        assert ctx.deferred.osm_import is False, "import is not flagged until the box is placed + confirmed"


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
        sm.start_building(lon=0.0, lat=0.0, elevation=mock_dem_blue_slope.get_elevation_or_raise(lon=0.0, lat=0.0))
        seen = self._capture_markdown(fake_st)
        SidebarRenderer(state_machine=sm, context=ctx, graph=empty_graph).render()
        assert any("Path Settings" in m for m in seen), "fan-out mode shows the Path Settings block"

    def test_path_settings_hidden_in_custom_mode(self, fake_st, empty_graph, mock_dem_blue_slope) -> None:
        sm, ctx = PlannerStateMachine.create(graph=empty_graph, add_ui_listener=False)
        sm.start_building(lon=0.0, lat=0.0, elevation=mock_dem_blue_slope.get_elevation_or_raise(lon=0.0, lat=0.0))
        ctx.custom_connect.force_mode = True  # showing custom-connect proposals
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

    def test_add_segments_label(self, empty_graph, path_points_blue) -> None:
        empty_graph.commit_paths(paths=[ProposedPathSegment(points=path_points_blue, target_difficulty="blue")])
        label = self._describe_top(empty_graph)
        assert "segment" in label.lower() and "slope" in label.lower()

    def test_add_segments_label_says_road_for_road_kind(self, empty_graph, path_points_blue) -> None:
        # Roads commit via the same AddSegmentsAction — the label must say "road", not "slope".
        empty_graph.commit_paths(
            paths=[ProposedPathSegment(points=path_points_blue, is_connector=True, kind=SegmentKind.ROAD)]
        )
        assert "road" in self._describe_top(empty_graph).lower()

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
        dem = mock_dem_blue_slope
        m = 111320.0
        piste = (
            [
                PathPoint(lon=0.0, lat=0.0, elevation=dem.get_elevation_or_raise(lon=0.0, lat=0.0)),
                PathPoint(lon=0.0, lat=-500 / m, elevation=dem.get_elevation_or_raise(lon=0.0, lat=-500 / m)),
            ],
            "Run",
        )
        empty_graph.import_osm(pistes=[piste], lifts=[], dem=dem)
        assert "OSM import" in self._describe_top(empty_graph)


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

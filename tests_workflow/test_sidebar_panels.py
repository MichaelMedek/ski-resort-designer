"""Unit tests for the left-sidebar mode-specific panels (ui/sidebar_panels.py).

Each build state owns a SidebarPanel whose `controls()` renders its buttons/sliders and fires the
action DIRECTLY on click (fire-and-forget) — there is no return value for app.py to act on. Driven
via the shared `fake_st`: `fake_st.clicked_keys` fires one specific button so its body runs, and we
assert the real state change (cancel → idle_ready, slider → ctx mutation). The dispatch (which panel
a state maps to) is asserted in test_mode_registry; here we assert each panel's own behavior.
"""

from skiresort_planner.model.path_point import PathPoint
from skiresort_planner.model.proposed_path import ProposedPathSegment
from skiresort_planner.model.resort_graph import ResortGraph
from skiresort_planner.ui.context import BuildMode
from skiresort_planner.ui.sidebar_panels import (
    IdleSidebarPanel,
    ImportSidebarPanel,
    LiftSidebarPanel,
    MergeSidebarPanel,
    RoadSidebarPanel,
    SlopeSidebarPanel,
    ViewingSidebarPanel,
)
from skiresort_planner.ui.state_machine import PlannerStateMachine

M = 111320.0  # metres per degree near the equator


def _session(fake_st, graph, factory, dem):
    """Seed fake st.session_state with the objects the panels' action functions read."""
    sm, ctx = PlannerStateMachine.create(graph=graph, add_ui_listener=False)
    fake_st.session_state["state_machine"] = sm
    fake_st.session_state["context"] = ctx
    fake_st.session_state["graph"] = graph
    fake_st.session_state["path_factory"] = factory
    fake_st.session_state["dem_service"] = dem
    fake_st.session_state["map_version"] = 0
    return sm, ctx


def _capture_buttons(fake_st) -> list[dict[str, object]]:
    """Record every st.button call's kwargs PLUS its positional label under 'label' (so we can
    assert `disabled`/`key`/label) without firing.

    fake_st.button ignores `disabled` and would fire a disabled button, so a disabled-state test must
    inspect the computed kwargs rather than click. Returns a list that fills as the panel renders.
    """
    calls: list[dict[str, object]] = []

    def record(*args: object, **kwargs: object) -> bool:
        entry = dict(kwargs)
        if args:
            entry["label"] = args[0]
        calls.append(entry)
        return False

    fake_st.button = record
    return calls


class TestIdleSidebarPanel:
    def test_renders_nothing(self, fake_st, path_factory, mock_dem_red_slope_diagonal) -> None:
        # idle_ready has no mode-specific controls — controls() is a no-op and must not raise.
        sm, ctx = _session(fake_st, ResortGraph(), path_factory, mock_dem_red_slope_diagonal)
        IdleSidebarPanel(sm=sm, ctx=ctx, graph=ResortGraph()).controls()  # no-op: must not raise


class TestViewingSidebarPanel:
    def test_close_button_hides_the_info_panel(self, fake_st, path_factory, mock_dem_red_slope_diagonal) -> None:
        # The one control is "Close Right Panel" → sm.hide_info_panel() → back to idle_ready.
        graph = ResortGraph()
        graph.commit_paths(
            paths=[
                ProposedPathSegment(
                    points=[
                        PathPoint(lon=0.0, lat=0.0, elevation=2500.0),
                        PathPoint(lon=0.0, lat=-0.01, elevation=2400.0),
                    ],
                    target_difficulty="blue",
                )
            ]
        )
        slope = graph.finish_slope(segment_ids=list(graph.segments.keys()))
        assert slope is not None
        sm, ctx = _session(fake_st, graph, path_factory, mock_dem_red_slope_diagonal)
        sm.show_slope_info_panel(slope_id=slope.id)
        assert sm.is_idle_viewing_slope

        fake_st.clicked_keys = {"close_panel_btn"}
        ViewingSidebarPanel(sm=sm, ctx=ctx, graph=graph).controls()
        assert sm.is_idle_ready, "Close Right Panel returns to idle_ready"

    def test_no_click_leaves_the_panel_open(self, fake_st, path_factory, mock_dem_blue_slope, path_points_blue) -> None:
        graph = ResortGraph()
        graph.commit_paths(paths=[ProposedPathSegment(points=path_points_blue, target_difficulty="blue")])
        slope = graph.finish_slope(segment_ids=list(graph.segments.keys()))
        assert slope is not None
        sm, ctx = _session(fake_st, graph, path_factory, mock_dem_blue_slope)
        sm.show_slope_info_panel(slope_id=slope.id)

        ViewingSidebarPanel(sm=sm, ctx=ctx, graph=graph).controls()  # no clicked_keys → button not fired
        assert sm.is_idle_viewing_slope, "rendering without a click does not close the panel"


class TestSlopeSidebarPanel:
    def _building(self, fake_st, dem, factory):
        graph = ResortGraph()
        sm, ctx = _session(fake_st, graph, factory, dem)
        sm.start_building(lon=0.0, lat=0.0, elevation=dem.get_elevation_or_raise(lon=0.0, lat=0.0))
        return sm, ctx, graph

    def test_finish_disabled_with_no_committed_segments(self, fake_st, path_factory, mock_dem_blue_slope) -> None:
        # With no segments the Finish button must render DISABLED. (fake_st.button ignores `disabled`
        # and would fire it, so we assert the computed `disabled=True` kwarg rather than firing it.)
        sm, ctx, graph = self._building(fake_st, mock_dem_blue_slope, path_factory)
        assert not ctx.has_committed_segments()
        buttons = _capture_buttons(fake_st)

        SlopeSidebarPanel(sm=sm, ctx=ctx, graph=graph).controls()
        finish = next(b for b in buttons if b.get("key") == "finish_slope_btn")
        assert finish["disabled"] is True, "Finish is disabled until at least one segment is committed"

    def test_cancel_discards_and_returns_to_idle(self, fake_st, path_factory, mock_dem_blue_slope) -> None:
        sm, ctx, graph = self._building(fake_st, mock_dem_blue_slope, path_factory)

        fake_st.clicked_keys = {"cancel_slope_btn"}
        SlopeSidebarPanel(sm=sm, ctx=ctx, graph=graph).controls()
        assert sm.is_idle_ready, "Cancel Full Slope discards the build and returns to idle"

    def test_segment_length_slider_change_updates_ctx(self, fake_st, path_factory, mock_dem_blue_slope) -> None:
        # Moving the slider must write the new length into ctx. Override the fake slider to return a
        # CHANGED value (the default fake echoes the current value, which never trips the change
        # branch). Note: pending_recompute is set then CONSUMED by recompute_paths() in the same
        # controls() call, so we assert the durable outcome — the new segment length on ctx.
        sm, ctx, graph = self._building(fake_st, mock_dem_blue_slope, path_factory)
        new_length = ctx.segment_length_m + 100
        fake_st.slider = lambda *a, **k: new_length

        SlopeSidebarPanel(sm=sm, ctx=ctx, graph=graph).controls()
        assert ctx.segment_length_m == new_length, "slider change is written to ctx"

    def test_path_settings_hidden_in_custom_connect_force_mode(
        self, fake_st, path_factory, mock_dem_blue_slope
    ) -> None:
        # While routing a custom-connect path, the Path Settings block (slider + Recompute) is hidden.
        sm, ctx, graph = self._building(fake_st, mock_dem_blue_slope, path_factory)
        ctx.custom_connect.force_mode = True
        seen: list[str] = []
        fake_st.markdown = lambda text, *a, **k: seen.append(text)

        SlopeSidebarPanel(sm=sm, ctx=ctx, graph=graph).controls()
        assert not any("Path Settings" in m for m in seen), "custom-connect force_mode hides Path Settings"


class TestRoadSidebarPanel:
    def _road_building(self, fake_st, dem, factory):
        sm, ctx = _session(fake_st, ResortGraph(), factory, dem)
        ctx.build_mode.mode = BuildMode.ROAD
        sm.select_road_start(location=PathPoint(lon=0.0, lat=0.0, elevation=2500.0))
        assert sm.is_road_starting
        return sm, ctx, fake_st.session_state["graph"]

    def test_cancel_road_returns_to_idle(self, fake_st, path_factory, mock_dem_red_slope_diagonal) -> None:
        sm, ctx, graph = self._road_building(fake_st, mock_dem_red_slope_diagonal, path_factory)

        fake_st.clicked_keys = {"cancel_road_btn"}
        RoadSidebarPanel(sm=sm, ctx=ctx, graph=graph).controls()
        assert sm.is_idle_ready, "Cancel Road discards the build and returns to idle"

    def test_finish_disabled_with_no_committed_segments(
        self, fake_st, path_factory, mock_dem_red_slope_diagonal
    ) -> None:
        sm, ctx, graph = self._road_building(fake_st, mock_dem_red_slope_diagonal, path_factory)
        assert not ctx.road_build.has_committed_segments()
        buttons = _capture_buttons(fake_st)

        RoadSidebarPanel(sm=sm, ctx=ctx, graph=graph).controls()
        finish = next(b for b in buttons if b.get("key") == "finish_road_btn")
        assert finish["disabled"] is True, "Finish Road is disabled until a segment is committed"


class TestLiftSidebarPanel:
    def test_renders_the_cancel_button_while_placing(self, fake_st, path_factory, mock_dem_blue_slope) -> None:
        # LiftSidebarPanel's only control is a shared `_cancel_button` (no explicit key), so it cannot
        # be *fired* through fake_st (which keys on `key=`). We assert it renders the labelled button
        # without crashing and stays in placement — the cancel TRANSITION itself is covered by the
        # click-handler/state-machine suites.
        dem = mock_dem_blue_slope
        sm, ctx = _session(fake_st, ResortGraph(), path_factory, dem)
        sm.start_lift(node_id=None, location=PathPoint(lon=0.0, lat=-0.01, elevation=2400.0))
        assert sm.is_lift_placing

        labels = _capture_buttons(fake_st)
        LiftSidebarPanel(sm=sm, ctx=ctx, graph=ResortGraph()).controls()
        assert any("Cancel Lift Placement" in str(b.get("label", "")) for b in labels), (
            "the lift panel renders its cancel button"
        )
        assert sm.is_lift_placing, "rendering the cancel button does not itself cancel"


class TestImportSidebarPanel:
    def test_half_width_slider_change_updates_deferred(
        self, fake_st, path_factory, mock_dem_red_slope_diagonal
    ) -> None:
        sm, ctx = _session(fake_st, ResortGraph(), path_factory, mock_dem_red_slope_diagonal)
        ctx.build_mode.mode = BuildMode.IMPORT
        sm.start_import(lon=0.0, lat=0.0)
        new_half_width = ctx.deferred.osm_import_half_width_km + 1.5
        fake_st.slider = lambda *a, **k: new_half_width

        ImportSidebarPanel(sm=sm, ctx=ctx, graph=ResortGraph()).controls()
        assert ctx.deferred.osm_import_half_width_km == new_half_width, "slider change resizes the import box"

    def test_renders_without_confirming(self, fake_st, path_factory, mock_dem_red_slope_diagonal) -> None:
        sm, ctx = _session(fake_st, ResortGraph(), path_factory, mock_dem_red_slope_diagonal)
        ctx.build_mode.mode = BuildMode.IMPORT
        sm.start_import(lon=0.0, lat=0.0)

        ImportSidebarPanel(sm=sm, ctx=ctx, graph=ResortGraph()).controls()  # no button fired
        assert sm.is_import_placing and ctx.deferred.osm_import is False, "rendering does not confirm the import"


class TestMergeSidebarPanel:
    def test_renders_while_placing_without_leaving(self, fake_st, path_factory, mock_dem_red_slope_diagonal) -> None:
        sm, ctx = _session(fake_st, ResortGraph(), path_factory, mock_dem_red_slope_diagonal)
        sm.start_merge()
        assert sm.is_merge_placing

        MergeSidebarPanel(sm=sm, ctx=ctx, graph=ResortGraph()).controls()  # no button fired
        assert sm.is_merge_placing, "rendering the merge cancel button does not itself cancel"

"""Tests for the top-level app orchestration (app.py).

Everything here runs against the shared `fake_st` (no browser). Session/routing
helpers are tested directly; the full render loop (main → _run_app_ui →
_render_map_fragment_inner) is driven end-to-end with a seeded session and a
stubbed st_deckgl so the deck.gl component call returns no event.
"""

import skiresort_planner.app as app
import skiresort_planner.ui.infra as infra
import skiresort_planner.ui.pydeck_click_handler as pch
from skiresort_planner.constants import ChartConfig
from skiresort_planner.model.path_segment import SegmentKind
from skiresort_planner.model.proposed_path import ProposedPathSegment
from skiresort_planner.model.resort_graph import ResortGraph
from skiresort_planner.ui.state_machine import PlannerStateMachine

M = 111320.0


def _seed_full_session(fake_st, dem):
    """Populate st.session_state with everything the render loop reads."""
    from skiresort_planner.generators.path_factory import PathFactory
    from skiresort_planner.ui.center_map import MapRenderer

    graph = ResortGraph()
    sm, ctx = PlannerStateMachine.create(graph=graph, add_ui_listener=False)
    ss = fake_st.session_state
    ss["resort_id"] = "test1234"
    ss["graph"] = graph
    ss["state_machine"] = sm
    ss["context"] = ctx
    ss["dem_service"] = dem
    ss["path_factory"] = PathFactory(dem_service=dem)
    ss["map_renderer"] = MapRenderer(graph=graph)
    ss["map_version"] = 0
    ss["_upload_counter"] = 0
    return graph, sm, ctx


def _stub_deckgl(monkeypatch, event=None):
    """Stub st_deckgl + terrain layer + window-height JS so the map render loop needs no browser."""
    monkeypatch.setattr(pch, "st_deckgl", lambda *a, **k: event)
    monkeypatch.setattr(app, "create_aws_terrain_layer", lambda *a, **k: object())
    monkeypatch.setattr(infra, "streamlit_js_eval", lambda *a, **k: 1080)  # browser height resolved


class TestSessionHelpers:
    def test_init_session_state_seeds_everything(self, fake_st, monkeypatch) -> None:
        # No resort param, no backups → fresh id + fresh graph/sm/renderer.
        monkeypatch.setattr(app.backup_store, "largest_resort_id", lambda: None)
        monkeypatch.setattr(app.backup_store, "new_resort_id", lambda: "fresh567")

        app.init_session_state()

        ss = fake_st.session_state
        assert ss["resort_id"] == "fresh567"
        assert isinstance(ss["graph"], ResortGraph)
        assert ss["state_machine"] is not None
        assert ss["map_renderer"] is not None
        assert ss["map_version"] == 0

    def test_reset_ui_state_preserves_graph(self, fake_st, path_points_blue) -> None:
        graph = ResortGraph()
        graph.commit_paths(paths=[ProposedPathSegment(points=path_points_blue, target_difficulty="blue")])
        graph.finish_slope(segment_ids=list(graph.segments.keys()))
        fake_st.session_state["graph"] = graph
        fake_st.session_state["map_version"] = 0

        app.reset_ui_state()

        # Graph preserved; a fresh state machine + context installed.
        assert fake_st.session_state["graph"] is graph
        assert len(graph.slopes) == 1
        assert fake_st.session_state["state_machine"] is not None

    def test_load_dem_data_returns_true_when_loaded(self, fake_st, mock_dem_blue_slope) -> None:
        fake_st.session_state["dem_service"] = mock_dem_blue_slope  # already loaded
        assert app.load_dem_data() is True


class TestMapHeight:
    """viewport_map_height: js-eval read + session_state cache + reserve/floor math."""

    def test_none_before_first_resolve(self, fake_st, monkeypatch) -> None:
        # js-eval returns None and nothing cached yet → None (caller shows placeholder).
        monkeypatch.setattr(infra, "streamlit_js_eval", lambda *a, **k: None)
        assert infra.viewport_map_height() is None

    def test_resolves_and_caches_window_height(self, fake_st, monkeypatch) -> None:
        monkeypatch.setattr(infra, "streamlit_js_eval", lambda *a, **k: 1080.0)
        assert infra.viewport_map_height() == 1080 - ChartConfig.MAP_TOP_OFFSET_PX
        assert fake_st.session_state["window_height_px"] == 1080

    def test_uses_cache_when_js_returns_none(self, fake_st, monkeypatch) -> None:
        # After a real value is cached, a later None-returning rerun keeps the map sized.
        fake_st.session_state["window_height_px"] = 900
        monkeypatch.setattr(infra, "streamlit_js_eval", lambda *a, **k: None)
        assert infra.viewport_map_height() == 900 - ChartConfig.MAP_TOP_OFFSET_PX

    def test_short_window_clamped_to_min(self, fake_st, monkeypatch) -> None:
        monkeypatch.setattr(infra, "streamlit_js_eval", lambda *a, **k: 300)
        assert infra.viewport_map_height() == ChartConfig.MAP_MIN_HEIGHT_PX

    def test_reserved_space_shrinks_map(self, fake_st, monkeypatch) -> None:
        # Reserving room for a profile below the map subtracts from its height.
        monkeypatch.setattr(infra, "streamlit_js_eval", lambda *a, **k: 1080.0)
        full = infra.viewport_map_height(reserved_below_px=0)
        reserved = infra.viewport_map_height(reserved_below_px=ChartConfig.PROFILE_HEIGHT_PX)
        assert reserved == full - ChartConfig.PROFILE_HEIGHT_PX

    def test_first_render_shows_message_and_skips_map(self, fake_st, monkeypatch, mock_dem_blue_slope) -> None:
        # window height None and nothing cached → return early, never call st_deckgl.
        _stub_deckgl(monkeypatch)
        monkeypatch.setattr(infra, "streamlit_js_eval", lambda *a, **k: None)
        _seed_full_session(fake_st, mock_dem_blue_slope)
        called = {"deck": False}
        monkeypatch.setattr(pch, "st_deckgl", lambda *a, **k: called.__setitem__("deck", True))

        app._render_map_fragment_inner()

        assert called["deck"] is False  # map render skipped until height known


class TestRenderLoop:
    """Drive the full UI render loop for each build state (no raise, real path)."""

    def test_run_app_ui_idle(self, fake_st, monkeypatch, mock_dem_blue_slope) -> None:
        _stub_deckgl(monkeypatch)
        _seed_full_session(fake_st, mock_dem_blue_slope)
        app._run_app_ui()  # idle_ready → renders sidebar + map + control panel

    def test_run_app_ui_viewing_slope(self, fake_st, monkeypatch, mock_dem_blue_slope, path_points_blue) -> None:
        _stub_deckgl(monkeypatch)
        graph, sm, ctx = _seed_full_session(fake_st, mock_dem_blue_slope)
        graph.commit_paths(paths=[ProposedPathSegment(points=path_points_blue, target_difficulty="blue")])
        slope = graph.finish_slope(segment_ids=list(graph.segments.keys()))
        sm.show_slope_info_panel(slope_id=slope.id)
        keys: list[str] = []
        fake_st.plotly_chart = lambda *a, **k: keys.append(k.get("key"))

        app._run_app_ui()

        # The viewing profile renders below the map (same slot as the in-build profile).
        assert "viewing_profile" in keys

    def test_run_app_ui_viewing_road(self, fake_st, monkeypatch, mock_dem_blue_slope, path_points_blue) -> None:
        _stub_deckgl(monkeypatch)
        graph, sm, ctx = _seed_full_session(fake_st, mock_dem_blue_slope)
        proposal = ProposedPathSegment(points=path_points_blue, is_connector=True, kind=SegmentKind.ROAD)
        graph.commit_paths(paths=[proposal], record_undo=False)
        road = graph.finish_road(segment_ids=[list(graph.segments.keys())[-1]])
        sm.show_road_info_panel(road_id=road.id)

        app._run_app_ui()

    def test_run_app_ui_lift_placing_renders_marker(self, fake_st, monkeypatch, mock_dem_blue_slope) -> None:
        from skiresort_planner.model.path_point import PathPoint

        _stub_deckgl(monkeypatch)
        _graph, sm, ctx = _seed_full_session(fake_st, mock_dem_blue_slope)
        loc = PathPoint(
            lon=0.0, lat=-1000 / M, elevation=mock_dem_blue_slope.get_elevation_or_raise(lon=0.0, lat=-1000 / M)
        )
        sm.start_lift(node_id=None, location=loc)  # exercises pending-lift marker layers

        app._run_app_ui()

    def test_render_map_terrain_click_dispatches(self, fake_st, monkeypatch, mock_dem_blue_slope) -> None:
        # A terrain click event flows through detector → dispatch_click.
        _stub_deckgl(monkeypatch, event={"coordinate": [0.0, 0.0], "eventType": "click"})
        _graph, sm, ctx = _seed_full_session(fake_st, mock_dem_blue_slope)
        ctx.build_mode.mode = "slope"

        app._render_map_fragment_inner()  # must parse + dispatch without raising

    def test_main_runs_full_cycle(self, fake_st, monkeypatch, mock_dem_blue_slope) -> None:
        _stub_deckgl(monkeypatch)
        monkeypatch.setattr(app.backup_store, "largest_resort_id", lambda: None)
        monkeypatch.setattr(app.backup_store, "new_resort_id", lambda: "main1234")
        # DEM already loaded so load_dem_data() returns True and main proceeds.
        fake_st.session_state["dem_service"] = mock_dem_blue_slope

        app.main()

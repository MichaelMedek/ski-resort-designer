"""Tests for the top-level app orchestration (app.py).

Everything here runs against the shared `fake_st` (no browser). Session/routing
helpers are tested directly; the full render loop (main → _run_app_ui →
_render_map_fragment_inner) is driven end-to-end with a seeded session and a
stubbed st_deckgl so the deck.gl component call returns no event.
"""

import pytest

import skiresort_planner.ui.pydeck_click_handler as pch
from skiresort_planner import app
from skiresort_planner.constants import ChartConfig, DEMConfig
from skiresort_planner.model.click_info import ClickInfo
from skiresort_planner.model.path_segment import SegmentKind
from skiresort_planner.model.proposed_path import ProposedPathSegment
from skiresort_planner.model.resort_graph import ResortGraph
from skiresort_planner.ui import infra
from skiresort_planner.ui.context import BuildMode
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
    ss["camera_epoch"] = 0
    ss["dedup_epoch"] = 0
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
        monkeypatch.setattr("skiresort_planner.app.backup_store.largest_resort_id", lambda: None)
        monkeypatch.setattr("skiresort_planner.app.backup_store.new_resort_id", lambda: "fresh567")

        app.init_session_state()

        ss = fake_st.session_state
        assert ss["resort_id"] == "fresh567"
        assert isinstance(ss["graph"], ResortGraph)
        assert ss["state_machine"] is not None
        assert ss["map_renderer"] is not None
        assert ss["camera_epoch"] == 0
        assert ss["dedup_epoch"] == 0

    def test_reset_ui_state_preserves_graph(self, fake_st, path_points_blue) -> None:
        graph = ResortGraph()
        graph.commit_paths(paths=[ProposedPathSegment(points=path_points_blue, target_difficulty="blue")])
        graph.finish_slope(segment_ids=list(graph.segments.keys()))
        fake_st.session_state["graph"] = graph
        fake_st.session_state["camera_epoch"] = 0

        app.reset_ui_state()

        # Graph preserved; a fresh state machine + context installed; camera remounts (recovery).
        assert fake_st.session_state["graph"] is graph
        assert len(graph.slopes) == 1
        assert fake_st.session_state["state_machine"] is not None
        assert fake_st.session_state["camera_epoch"] == 1

    def test_load_dem_data_returns_early_when_loaded(self, fake_st, mock_dem_blue_slope, monkeypatch) -> None:
        fake_st.session_state["dem_service"] = mock_dem_blue_slope  # already loaded
        reframed: list[object] = []
        monkeypatch.setattr(app, "reload_map", lambda **k: reframed.append(k))

        assert app.load_dem_data() is None, "already-loaded path returns early (no value)"
        assert reframed == [], "no reframe/rerun when the DEM is already loaded"

    def test_load_dem_data_builds_services_and_reframes_while_loading(self, fake_st, monkeypatch) -> None:
        # No dem_service in session and the DEM file already present locally: skip download,
        # build DEMService + PathFactory, then reframe to the start view (which reruns).
        from pathlib import Path

        from skiresort_planner.constants import MapConfig

        class _FakeDEM:
            is_loaded = True

            def get_elevation(self, *, lon: float, lat: float) -> float:
                return 1000.0

        class _PresentPath(Path):
            # A Path subclass that always reports the DEM file as present (can't patch .exists on
            # a bare PosixPath instance — it's read-only — so swap the whole EURODEM_PATH).
            _flavour = type(Path())._flavour  # type: ignore[attr-defined]  # inherit the OS flavour

            def exists(self, *, follow_symlinks: bool = True) -> bool:
                return True

        monkeypatch.setattr(DEMConfig, "EURODEM_PATH", _PresentPath("/tmp/fake_eurodem.tif"))
        monkeypatch.setattr(app, "DEMService", lambda *a, **k: _FakeDEM())
        monkeypatch.setattr(app, "PathFactory", lambda *a, **k: object())
        # reload_map reruns in prod (StopExecution); here it records the frame and raises so the flow
        # stops exactly where the real rerun would — load_dem_data never returns after loading.
        reframes: list[tuple[tuple[float, float], int]] = []

        class _Rerun(Exception):
            pass

        def _fake_reload(*, center: tuple[float, float], zoom: int, pitch: float) -> None:
            reframes.append((center, zoom))
            raise _Rerun

        monkeypatch.setattr(app, "reload_map", _fake_reload)

        with pytest.raises(_Rerun):
            app.load_dem_data()

        assert isinstance(fake_st.session_state["dem_service"], _FakeDEM)
        assert fake_st.session_state["path_factory"] is not None
        assert reframes == [((MapConfig.START_CENTER_LON, MapConfig.START_CENTER_LAT), MapConfig.DEFAULT_ZOOM)], (
            "DEM load reframes to the start view via the shared slow-load helper"
        )


class TestReloadMapSignature:
    """reload_map must require an EXPLICIT frame so no caller can remount on a stale view.

    Guards the design rule: `center` and `zoom` are keyword-only with NO defaults — a bare
    keep-current-view remount uses bump_camera_epoch() instead.
    """

    def test_center_and_zoom_are_required(self) -> None:
        import inspect

        params = inspect.signature(infra.reload_map).parameters
        assert params["center"].default is inspect.Parameter.empty, "center must be required"
        assert params["zoom"].default is inspect.Parameter.empty, "zoom must be required"

    def test_reload_map_frames_and_bumps(self, fake_st, monkeypatch) -> None:
        # Seed a context so reload_map can write ctx.map, and stub the rerun (raises in prod).
        from skiresort_planner.ui.state_machine import PlannerStateMachine

        _sm, ctx = PlannerStateMachine.create(graph=ResortGraph(), add_ui_listener=False)
        fake_st.session_state["context"] = ctx
        fake_st.session_state["camera_epoch"] = 0
        monkeypatch.setattr(infra, "trigger_rerun", lambda *a, **k: None)

        infra.reload_map(center=(10.5, 46.5), zoom=13)

        assert (ctx.map.lon, ctx.map.lat) == (10.5, 46.5)
        assert ctx.map.zoom == 13
        assert fake_st.session_state["camera_epoch"] == 1  # remount so deck re-reads the frame


class TestPendingOSMImportGate:
    """The pending OSM import shows a blocking loading message + progress bar, returns early (no map
    this pass), and ALWAYS reframes on the placed import box center at the import-overview zoom.
    """

    def test_pending_import_gates_and_recenters(self, fake_st, monkeypatch, mock_dem_blue_slope) -> None:
        from skiresort_planner.constants import MapConfig, OSMImportMode
        from skiresort_planner.generators.osm_importer import ProgressFn

        graph, _sm, ctx = _seed_full_session(fake_st, mock_dem_blue_slope)
        ctx.pending.osm_import_mode = OSMImportMode.LIFTS_AND_SLOPES
        ctx.pending.osm_import_center_lon = 10.5  # the placed box center — reframe target
        ctx.pending.osm_import_center_lat = 46.5

        # Stub the heavy import: no network; it must receive the progress reporter and drive it.
        reported: list[float] = []

        def _fake_import(report: ProgressFn) -> bool:
            report(0.5, "working…")
            reported.append(0.5)
            return True

        monkeypatch.setattr(app, "process_osm_import_pending", _fake_import)
        monkeypatch.setattr(infra, "trigger_rerun", lambda *a, **k: None)
        rendered: list[bool] = []
        monkeypatch.setattr(app, "_render_map_fragment", lambda: rendered.append(True))

        app._run_app_ui()

        assert reported == [0.5], "the import work was driven with the progress reporter"
        assert rendered == [], "returns before the normal UI renders (no frozen map)"
        assert (ctx.map.lon, ctx.map.lat) == (10.5, 46.5), "reframed on the placed import box center"
        assert ctx.map.zoom == MapConfig.IMPORT_OVERVIEW_ZOOM, "one step further out than building zoom"
        assert fake_st.session_state["camera_epoch"] == 1, "remount so deck re-reads the frame"


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

    def test_first_render_shows_message_and_skips_map(self, fake_st, monkeypatch, mock_dem_blue_slope) -> None:
        # window height None and nothing cached → return early, never call st_deckgl.
        _stub_deckgl(monkeypatch)
        monkeypatch.setattr(infra, "streamlit_js_eval", lambda *a, **k: None)
        _seed_full_session(fake_st, mock_dem_blue_slope)
        called = {"deck": False}

        def _mark_called(*_a: object, **_k: object) -> None:
            called["deck"] = True

        monkeypatch.setattr(pch, "st_deckgl", _mark_called)

        app._render_map_fragment_inner()

        assert called["deck"] is False  # map render skipped until height known


class TestRenderLoop:
    """Drive the full UI render loop for each build state (no raise, real path)."""

    def test_run_app_ui_idle(self, fake_st, monkeypatch, mock_dem_blue_slope) -> None:
        _stub_deckgl(monkeypatch)
        _seed_full_session(fake_st, mock_dem_blue_slope)
        # Capture the single deck.gl render call (deck + key) instead of hitting the real component.
        calls: list[dict[str, object]] = []

        def _record_render(*, deck: object, height: int, key: str) -> pch.PydeckClickResult:
            calls.append({"deck": deck, "height": height, "key": key})
            return pch.PydeckClickResult.empty()

        monkeypatch.setattr(app, "render_pydeck_map", _record_render)
        profile_keys: list[str] = []
        fake_st.plotly_chart = lambda *a, **k: profile_keys.append(k.get("key"))

        app._run_app_ui()  # idle_ready → renders sidebar + map + control panel

        assert len(calls) == 1  # map rendered exactly once
        assert calls[0]["deck"] is not None  # a real pdk.Deck was built and handed to the component
        assert isinstance(calls[0]["key"], str) and calls[0]["key"].startswith("main_map_0_")
        # idle_ready has no bottom_profile, so no profile chart renders in the right column.
        assert profile_keys == []

    def test_run_app_ui_viewing_slope(self, fake_st, monkeypatch, mock_dem_blue_slope, path_points_blue) -> None:
        _stub_deckgl(monkeypatch)
        graph, sm, ctx = _seed_full_session(fake_st, mock_dem_blue_slope)
        graph.commit_paths(paths=[ProposedPathSegment(points=path_points_blue, target_difficulty="blue")])
        slope = graph.finish_slope(segment_ids=list(graph.segments.keys()))
        sm.view_slope(slope_id=slope.id)
        keys: list[str] = []
        fake_st.plotly_chart = lambda *a, **k: keys.append(k.get("key"))

        app._run_app_ui()

        # The viewing profile renders in the right column (render_control_panel), not below the map.
        assert "viewing_profile" in keys

    def test_run_app_ui_viewing_road(self, fake_st, monkeypatch, mock_dem_blue_slope, path_points_blue) -> None:
        _stub_deckgl(monkeypatch)
        graph, sm, ctx = _seed_full_session(fake_st, mock_dem_blue_slope)
        proposal = ProposedPathSegment(points=path_points_blue, is_connector=True, kind=SegmentKind.ROAD)
        graph.commit_paths(paths=[proposal], record_undo=False)
        road = graph.finish_road(segment_ids=[list(graph.segments.keys())[-1]])
        sm.view_road(road_id=road.id)
        keys: list[str] = []
        fake_st.plotly_chart = lambda *a, **k: keys.append(k.get("key"))

        app._run_app_ui()

        # The road viewing profile must actually render (regression for the "Unknown viewing
        # kind ROAD" crash — the app.py fragment → render_viewing_profile wiring for a road).
        assert "viewing_profile" in keys

    def test_run_app_ui_lift_placing_renders_marker(self, fake_st, monkeypatch, mock_dem_blue_slope) -> None:
        from skiresort_planner.model.path_point import PathPoint

        _stub_deckgl(monkeypatch)
        _graph, sm, ctx = _seed_full_session(fake_st, mock_dem_blue_slope)
        ctx.build_mode.mode = BuildMode.CHAIRLIFT  # lift type selected before entering LIFT_PLACING
        loc = PathPoint(
            lon=0.0, lat=-1000 / M, elevation=mock_dem_blue_slope.get_elevation_or_raise(lon=0.0, lat=-1000 / M)
        )
        sm.start_lift(node_id=None, location=loc)  # exercises pending-lift marker layers

        # Capture the extra_layers the state feeds into the renderer.
        renderer = fake_st.session_state["map_renderer"]
        original_render = renderer.render
        captured: dict[str, object] = {}

        def _spy_render(*args: object, **kwargs: object) -> object:
            captured["extra_layers"] = kwargs.get("extra_layers")
            return original_render(*args, **kwargs)

        monkeypatch.setattr(renderer, "render", _spy_render)

        app._run_app_ui()

        extra_layers = captured["extra_layers"]
        assert extra_layers  # pending-lift markers were produced and passed through
        assert isinstance(extra_layers, list)
        assert any(getattr(layer, "id", None) == "pending_lift_station" for layer in extra_layers)

    def test_render_map_terrain_click_dispatches(self, fake_st, monkeypatch, mock_dem_blue_slope) -> None:
        # A terrain click event flows through detector → dispatch_click.
        _stub_deckgl(monkeypatch, event={"coordinate": [0.0, 0.0], "eventType": "click"})
        _graph, sm, ctx = _seed_full_session(fake_st, mock_dem_blue_slope)
        ctx.build_mode.mode = "slope"
        dispatched: list[ClickInfo] = []
        monkeypatch.setattr(app, "dispatch_click", lambda *, click_info: dispatched.append(click_info))

        app._render_map_fragment_inner()  # must parse + dispatch without raising

        assert len(dispatched) == 1  # the terrain click reached the dispatcher exactly once
        click_info = dispatched[0]
        assert click_info.lon == 0.0 and click_info.lat == 0.0  # coordinate carried through detector

    def test_main_runs_full_cycle(self, fake_st, monkeypatch, mock_dem_blue_slope) -> None:
        _stub_deckgl(monkeypatch)
        monkeypatch.setattr("skiresort_planner.app.backup_store.largest_resort_id", lambda: None)
        monkeypatch.setattr("skiresort_planner.app.backup_store.new_resort_id", lambda: "main1234")
        # DEM already loaded so load_dem_data() returns True and main proceeds.
        fake_st.session_state["dem_service"] = mock_dem_blue_slope

        app.main()

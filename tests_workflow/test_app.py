"""Tests for the top-level app orchestration (app.py).

Everything here runs against the shared `fake_st` (no browser). Session/routing
helpers are tested directly; the full render loop (main → _run_app_ui →
_render_map_fragment_inner) is driven end-to-end with a seeded session and a
stubbed st_deckgl so the deck.gl component call returns no event.
"""

import pytest

import skiresort_planner.ui.pydeck_click_handler as pch
from skiresort_planner import app
from skiresort_planner.constants import ChartConfig, DEMConfig, MapConfig
from skiresort_planner.model.click_info import ClickInfo
from skiresort_planner.model.path_segment import SegmentKind
from skiresort_planner.model.proposed_path import ProposedPathSegment
from skiresort_planner.model.resort_graph import ResortGraph
from skiresort_planner.ui import infra
from skiresort_planner.ui.context import BuildMode
from skiresort_planner.ui.state_machine import PlannerStateMachine


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

        app.load_dem_data()  # already loaded → returns early, no work
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
        assert reframes == [((MapConfig.START_CENTER_LON, MapConfig.START_CENTER_LAT), MapConfig.VIEWING_ZOOM)], (
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

    def test_reload_map_frames_in_place(self, fake_st, monkeypatch) -> None:
        # Seed a context so reload_map can write ctx.map, and stub the rerun (raises in prod).
        from skiresort_planner.ui.state_machine import PlannerStateMachine

        _sm, ctx = PlannerStateMachine.create(graph=ResortGraph(), add_ui_listener=False)
        fake_st.session_state["context"] = ctx
        fake_st.session_state["camera_epoch"] = 0
        monkeypatch.setattr(infra, "trigger_rerun", lambda *a, **k: None)

        infra.reload_map(center=(10.5, 46.5), zoom=13)

        assert (ctx.map.lon, ctx.map.lat) == (10.5, 46.5)
        assert ctx.map.zoom == 13
        # In-place reframe: the new view flows via ctx.map → initialViewState (deck.gl applies it to the
        # mounted component). camera_epoch is NOT bumped — bumping it would remount and gray-out the map.
        assert fake_st.session_state["camera_epoch"] == 0


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
        # In-place reframe (reload_map): view moves via ctx.map → initialViewState; no camera_epoch bump
        # (a bump would remount the deck.gl iframe → gray-out). See tests_workflow/test_map_reframe.py.
        assert fake_st.session_state["camera_epoch"] == 0, "in-place reframe: no remount bump"


class TestRunPendingLoadFailure:
    """run_pending_load's strict contract: success reframes; a CAUGHT exception shows a pre-given
    WarningToast WITHOUT reframing; an uncaught type always propagates; `catch`+`failure_message` pair.
    """

    def _seed_ctx(self, fake_st):
        _sm, ctx = PlannerStateMachine.create(graph=ResortGraph(), add_ui_listener=False)
        fake_st.session_state["context"] = ctx
        fake_st.session_state["camera_epoch"] = 0
        return ctx

    def test_caught_failure_shows_warning_and_does_not_reframe(self, fake_st, monkeypatch) -> None:
        from skiresort_planner.model.message import DEMLoadingMessage, OSMImportErrorMessage

        ctx = self._seed_ctx(fake_st)
        reframed: list[object] = []
        reran: list[int] = []
        monkeypatch.setattr(app, "reload_map", lambda **k: reframed.append(k))
        monkeypatch.setattr(app, "trigger_rerun", lambda *a, **k: reran.append(1))

        def boom(_report):
            raise RuntimeError("kaput")

        app.run_pending_load(
            message=DEMLoadingMessage(),
            work=boom,
            reset_center=(1.0, 2.0),
            reset_zoom=12,
            catch=RuntimeError,
            failure_message=OSMImportErrorMessage(error="nope"),
        )

        assert reframed == [], "a caught failure must NOT reframe (would bury the warning)"
        assert reran == [1], "it still reruns so the warning toast paints"
        assert ctx.map.zoom != 12, "the reset frame was discarded on failure"

    def test_uncaught_exception_type_propagates(self, fake_st, monkeypatch) -> None:
        import pytest

        from skiresort_planner.model.message import DEMLoadingMessage, OSMImportErrorMessage

        self._seed_ctx(fake_st)
        monkeypatch.setattr(app, "reload_map", lambda **k: None)

        def boom(_report):
            raise KeyError("not the caught type")

        # catch=ValueError only → a KeyError is NOT soft-handled, it propagates.
        with pytest.raises(KeyError):
            app.run_pending_load(
                message=DEMLoadingMessage(),
                work=boom,
                reset_center=(1.0, 2.0),
                reset_zoom=12,
                catch=ValueError,
                failure_message=OSMImportErrorMessage(error="nope"),
            )

    def test_no_catch_hard_fails(self, fake_st, monkeypatch) -> None:
        import pytest

        from skiresort_planner.model.message import DEMLoadingMessage

        self._seed_ctx(fake_st)
        monkeypatch.setattr(app, "reload_map", lambda **k: None)

        def boom(_report):
            raise RuntimeError("kaput")

        # No catch/failure_message → nothing soft-handled (DEM-style hard fail).
        with pytest.raises(RuntimeError, match="kaput"):
            app.run_pending_load(message=DEMLoadingMessage(), work=boom, reset_center=(1.0, 2.0), reset_zoom=12)

    def test_catch_and_failure_message_must_be_paired(self, fake_st, monkeypatch) -> None:
        import pytest

        from skiresort_planner.model.message import DEMLoadingMessage

        self._seed_ctx(fake_st)
        monkeypatch.setattr(app, "reload_map", lambda **k: None)

        # catch given but no failure_message → the pairing assert fires.
        with pytest.raises(AssertionError, match="both"):
            app.run_pending_load(
                message=DEMLoadingMessage(),
                work=lambda _r: None,
                reset_center=(1.0, 2.0),
                reset_zoom=12,
                catch=RuntimeError,
            )

    def test_failure_message_must_be_a_warning_toast(self, fake_st, monkeypatch) -> None:
        import pytest

        from skiresort_planner.model.message import DEMLoadingMessage, ToastMessage

        # A non-WarningToast toast must be rejected as a failure_message.
        class _PlainToast(ToastMessage):
            @property
            def message(self) -> str:
                return "info"

            @property
            def icon(self) -> str:
                return "ℹ️"

        self._seed_ctx(fake_st)
        monkeypatch.setattr(app, "reload_map", lambda **k: None)

        with pytest.raises(AssertionError, match="WarningToast"):
            app.run_pending_load(
                message=DEMLoadingMessage(),
                work=lambda _r: None,
                reset_center=(1.0, 2.0),
                reset_zoom=12,
                catch=RuntimeError,
                failure_message=_PlainToast(),  # type: ignore[arg-type]  # not a WarningToast — must be rejected
            )


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
            lon=0.0,
            lat=-1000 / MapConfig.METERS_PER_DEGREE_EQUATOR,
            elevation=mock_dem_blue_slope.get_elevation_or_raise(
                lon=0.0, lat=-1000 / MapConfig.METERS_PER_DEGREE_EQUATOR
            ),
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


def _render_key(fake_st):
    """Run one map render pass and return the map_key computed this rerun."""
    app._render_map_fragment_inner()
    return fake_st.session_state.get("_last_map_key")


class TestReframeInPlace:
    """Regression for the reframe gray-out: a same-pitch 2D reframe (finish/close/search/reset) must NOT
    change the component key (which would remount the deck.gl iframe → WebGL teardown + tile re-fetch =
    the ~0.5s gray-out). Only a pitch change (2D↔3D) legitimately remounts.
    """

    def test_same_pitch_reframe_does_not_remount(self, fake_st, monkeypatch, mock_dem_blue_slope) -> None:
        _stub_deckgl(monkeypatch)
        _graph, _sm, ctx = _seed_full_session(fake_st, mock_dem_blue_slope)

        key_before = _render_key(fake_st)
        assert key_before is not None and "_2d_" in key_before

        # Reframe exactly as finish/close/search/reset do — must stay in place (new initialViewState, same key).
        infra.reload_map(center=(10.30, 46.90), zoom=MapConfig.VIEWING_ZOOM, pitch=MapConfig.VIEWING_PITCH)
        key_after = _render_key(fake_st)

        assert key_after == key_before, f"same-pitch reframe must not remount; {key_before!r} -> {key_after!r}"

    def test_2d_to_3d_toggle_still_remounts(self, fake_st, monkeypatch, mock_dem_blue_slope) -> None:
        _stub_deckgl(monkeypatch)
        _graph, _sm, ctx = _seed_full_session(fake_st, mock_dem_blue_slope)

        key_2d = _render_key(fake_st)
        ctx.viewing.enable_3d()  # pitch change → intentional remount preserved
        key_3d = _render_key(fake_st)

        assert key_2d is not None and "_2d_" in key_2d
        assert key_3d != key_2d and "_3d_" in key_3d, "2D↔3D toggle must remount (key changes)"

    def test_flythrough_frames_keep_key_constant(
        self, fake_st, monkeypatch, mock_dem_blue_slope, path_points_blue
    ) -> None:
        """The flythrough camera advances frames on a CONSTANT key (in-place camera move, no per-frame
        remount) — the same mechanism as the reframe fix.
        """
        _stub_deckgl(monkeypatch)
        monkeypatch.setattr(infra, "trigger_rerun", lambda *a, **k: None)
        graph, sm, ctx = _seed_full_session(fake_st, mock_dem_blue_slope)
        graph.commit_paths(paths=[ProposedPathSegment(points=path_points_blue, target_difficulty="blue")])
        slope = graph.finish_slope(segment_ids=list(graph.segments.keys()))
        sm.view_slope(slope_id=slope.id)

        ctx.viewing.enable_3d()
        ctx.viewing.start_flythrough()  # a single slope → 2 keyframes (start, end)

        key0 = _render_key(fake_st)
        ctx.viewing.advance_flythrough()
        key1 = _render_key(fake_st)

        assert key0 == key1, f"flythrough frames must render on a constant key; {key0!r}/{key1!r}"

    def test_flythrough_highlight_ribbon_only_while_flying(
        self, fake_st, monkeypatch, mock_dem_blue_slope, path_points_blue
    ) -> None:
        """The hot-orange current-element ribbon (id 'flythrough_highlight') is in the built deck ONLY
        while flying — applied once at the render choke-point, never per viewing-state.
        """
        _stub_deckgl(monkeypatch)
        monkeypatch.setattr(infra, "trigger_rerun", lambda *a, **k: None)
        graph, sm, ctx = _seed_full_session(fake_st, mock_dem_blue_slope)
        graph.commit_paths(paths=[ProposedPathSegment(points=path_points_blue, target_difficulty="blue")])
        slope = graph.finish_slope(segment_ids=list(graph.segments.keys()))
        sm.view_slope(slope_id=slope.id)
        ctx.viewing.enable_3d()

        decks: list[object] = []
        monkeypatch.setattr(
            app, "render_pydeck_map", lambda *, deck, height, key: decks.append(deck) or pch.PydeckClickResult.empty()
        )

        def highlight_layer_ids() -> list[str]:
            return [layer.id for layer in decks[-1].layers if layer.id == "flythrough_highlight"]

        app._render_map_fragment_inner()  # not flying
        assert highlight_layer_ids() == [], "no highlight ribbon when idle in 3D"

        ctx.viewing.start_flythrough()
        app._render_map_fragment_inner()  # flying
        assert highlight_layer_ids() == ["flythrough_highlight"], "highlight ribbon shown while flying"

    def test_flythrough_dwells_at_end_before_stopping(
        self, fake_st, monkeypatch, mock_dem_blue_slope, path_points_blue
    ) -> None:
        """At the final keyframe the driver DWELLS (glide finishes + the viewer takes the finish in) before
        stopping — so the camera doesn't snap back to the entry view the instant it arrives.
        """
        import time

        _stub_deckgl(monkeypatch)
        monkeypatch.setattr(infra, "trigger_rerun", lambda *a, **k: None)
        monkeypatch.setattr(app, "trigger_rerun", lambda *a, **k: None)
        sleeps: list[float] = []
        monkeypatch.setattr(time, "sleep", lambda s: sleeps.append(s))
        graph, sm, ctx = _seed_full_session(fake_st, mock_dem_blue_slope)
        graph.commit_paths(paths=[ProposedPathSegment(points=path_points_blue, target_difficulty="blue")])
        slope = graph.finish_slope(segment_ids=list(graph.segments.keys()))
        sm.view_slope(slope_id=slope.id)
        ctx.viewing.enable_3d()

        ctx.viewing.start_flythrough()
        for _ in range(50):
            if not ctx.viewing.flythrough_active:
                break
            app._advance_flythrough_if_playing()

        assert not ctx.viewing.flythrough_active, "the flythrough terminates"
        assert MapConfig.FLYTHROUGH_END_DWELL_S in sleeps, "must dwell at the final keyframe before stopping"

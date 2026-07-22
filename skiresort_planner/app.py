"""Ski Resort Planner - Interactive terrain planning application.

Design ski resort layouts on real terrain using Digital Elevation Model (DEM) data.
Features fan-pattern path generation, lift placement, and elevation profiles.

Run: streamlit run skiresort_planner/app.py
"""

import time
import traceback
import uuid
from collections.abc import Callable
from typing import TYPE_CHECKING

import pydeck as pdk
import requests
import streamlit as st

from skiresort_planner.constants import (
    AppConfig,
    DEMConfig,
    MapConfig,
)
from skiresort_planner.core.dem_service import DEMService, download_dem_from_huggingface
from skiresort_planner.generators.osm_importer import ProgressFn
from skiresort_planner.generators.path_factory import PathFactory
from skiresort_planner.logging_setup import configure_logging
from skiresort_planner.model.message import (
    ClickingDisabledIn3DToast,
    DEMLoadingMessage,
    Message,
    OSMImportErrorMessage,
    OSMImportLoadingMessage,
    SizingMapMessage,
    WarningToast,
)
from skiresort_planner.model.resort_graph import ResortGraph
from skiresort_planner.persistence import backup_store
from skiresort_planner.ui import (
    ClickDetector,
    MapRenderer,
    PlannerContext,
    PlannerStateMachine,
    SidebarRenderer,
    active_flythrough_groups,
    bump_camera_epoch,
    cancel_custom_path,
    commit_selected_path,
    dispatch_click,
    process_custom_connect_pending,
    process_osm_import_pending,
    process_path_generation_pending,
    process_route_plan_pending,
    reload_map,
    render_control_panel,
    trigger_rerun,
    viewport_map_height,
)
from skiresort_planner.ui.mode_registry import BUILD_STATES
from skiresort_planner.ui.pydeck_click_handler import render_pydeck_map
from skiresort_planner.ui.terrain_layer import create_aws_terrain_layer

if TYPE_CHECKING:
    from skiresort_planner.core.terrain_analyzer import TerrainAnalyzer

# app.py runs as the Streamlit entry script (__name__ == "__main__", outside our package tree), so
# take the configured package logger and derive app's child from it — no hardcoded name.
logger = configure_logging().getChild("app")


# CSS to give the map the full window height. Streamlit exposes no API for any of these,
# so injection is the only option (the standard community pattern). Each rule is load-bearing:
_FULLSCREEN_CSS = """
<style>
/* Reclaim the top toolbar's height WITHOUT hiding the whole header/toolbar. In Streamlit 1.59
   the arrow that re-opens a collapsed sidebar (stExpandSidebarButton) is rendered INSIDE the
   toolbar (stToolbar), next to the Deploy/hamburger menu — so display:none on the header or
   the toolbar hides that arrow too, leaving no way to reopen the sidebar. Instead: flatten the
   header, hide only the Deploy menu + status widget, and keep the toolbar/expand arrow visible. */
header[data-testid="stHeader"] { height: 0; background: transparent; }
div[data-testid="stDecoration"], div[data-testid="stMainMenu"], div[data-testid="stStatusWidget"] { display: none; }
/* Collapse the default ~6rem top padding so the map starts near the top. */
.block-container { padding-top: 1rem; padding-bottom: 0; }
/* Hide the streamlit-js-eval helper iframe: it only reads the window height and should
   occupy no space (this is the "blue bar"). Most version-fragile rule — keyed on its title. */
div[data-testid="stElementContainer"]:has(iframe[title="streamlit_js_eval.streamlit_js_eval"]) { display: none; }
</style>
"""


# =============================================================================
# SESSION STATE
# =============================================================================


def init_session_state() -> None:
    """Initialize session state with DEM, graph, and UI components."""
    if "resort_id" not in st.session_state:
        _init_resort_from_url_or_new()

    if "graph" not in st.session_state:
        st.session_state.graph = ResortGraph()

    if "state_machine" not in st.session_state:
        sm, ctx = PlannerStateMachine.create(graph=st.session_state.graph)
        st.session_state.state_machine = sm
        st.session_state.context = ctx

    if "map_renderer" not in st.session_state:
        center = st.session_state.pop("_loaded_map_center", None)
        center_lon = center[0] if center else MapConfig.START_CENTER_LON
        center_lat = center[1] if center else MapConfig.START_CENTER_LAT
        st.session_state.map_renderer = MapRenderer(
            center_lon=center_lon,
            center_lat=center_lat,
            zoom=MapConfig.VIEWING_ZOOM,
            pitch=MapConfig.DEFAULT_PITCH,
            bearing=MapConfig.DEFAULT_BEARING,
        )
        if center is not None:
            st.session_state.context.map.set_center(lon=center_lon, lat=center_lat)

    if "_upload_counter" not in st.session_state:
        st.session_state._upload_counter = 0

    # Two independent counters (see infra.py): camera_epoch keys the map component (remount → recenter),
    # dedup_epoch keys click ids (proposal/marker regeneration) without moving the camera.
    if "camera_epoch" not in st.session_state:
        st.session_state.camera_epoch = 0
    if "dedup_epoch" not in st.session_state:
        st.session_state.dedup_epoch = 0


def _init_resort_from_url_or_new() -> None:
    """Resolve resort_id and prime the graph: load the ?resort=<id> backup (reload path — F5/
    bookmarks keep the URL), else the largest existing backup (the user's own work), else fresh.
    """
    param_id = st.query_params.get("resort")
    if not param_id:
        param_id = backup_store.largest_resort_id()

    if param_id:
        loaded = backup_store.load(resort_id=param_id)
        if loaded is not None:
            st.session_state.resort_id = param_id
            st.session_state.graph = loaded
            st.session_state._saved_token = loaded.change_token()
            st.query_params["resort"] = param_id
            # Renderer/context don't exist yet; stash center for MapRenderer init.
            st.session_state._loaded_map_center = loaded.get_center()
            logger.info(f"Loaded resort {param_id} from backup")
            return

    st.session_state.resort_id = backup_store.new_resort_id()
    st.query_params["resort"] = st.session_state.resort_id
    logger.info(f"Created new resort {st.session_state.resort_id} (no backup found)")


def reset_ui_state() -> None:
    """Reset state machine + context and remount the map on error recovery.

    Preserves the resort graph, DEM service, path factory, and map renderer (re-linked to graph).
    """
    logger.info("Resetting UI state due to error recovery")

    # Create fresh state machine and context
    sm, ctx = PlannerStateMachine.create(graph=st.session_state.graph)
    st.session_state.state_machine = sm
    st.session_state.context = ctx

    # Remount the map component for a clean slate after error recovery.
    bump_camera_epoch()

    logger.debug("UI state reset complete - graph preserved")


def _handle_error_with_recovery(e: Exception, context_tag: str) -> None:
    """Log the traceback, show the error + a reset button, and recover UI state (graph preserved).

    Args:
        e: The caught exception.
        context_tag: Short tag for the failing region ("RENDER" / "UI"), used in log + message.
    """
    error_msg = f"{type(e).__name__}: {e}"
    logger.error(f"[{context_tag}] error caught: {error_msg}\n{traceback.format_exc()}")
    st.error(f"⚠️ [{context_tag}] Something went wrong: {error_msg}")
    reset_ui_state()
    if st.button("🔄 Reset and Continue", type="primary"):
        trigger_rerun()


def run_pending_load(
    message: Message,
    work: Callable[[ProgressFn], object],
    *,
    reset_center: tuple[float, float],
    reset_zoom: float,
    catch: type[Exception] | tuple[type[Exception], ...] | None = None,
    failure_message: WarningToast | None = None,
) -> None:
    """SLOW pending action: blocking loading message + mandatory progress bar around `work`, then EITHER
    reframe the map to (reset_center, reset_zoom) on success OR, if `work` raises one of `catch`, show
    `failure_message` and rerun WITHOUT reframing (so the warning isn't buried by a camera remount).
    reset_center/reset_zoom are always required (discarded on failure). `catch` names EXACTLY the
    exception type(s) to soft-handle — anything else always propagates; None catches nothing (hard fail).
    `catch` and `failure_message` are given together or not at all.
    """
    assert (catch is None) == (failure_message is None), "pass both `catch` and `failure_message`, or neither"
    assert (failure_message is None) or isinstance(failure_message, WarningToast), (
        "failure_message must be a WarningToast (a yellow toast), not an info/inline message"
    )
    message.display()
    bar = st.progress(0.0, text="Starting…")

    def report(frac: float, text: str) -> None:
        bar.progress(frac, text=text)

    caught: tuple[type[Exception], ...] = () if catch is None else (catch if isinstance(catch, tuple) else (catch,))
    try:
        work(report)
    except caught as exc:  # only the named type(s); () catches nothing → any exception propagates
        logger.warning(f"pending load failed: {exc}")
        assert failure_message is not None  # paired with `catch` by the assert above
        failure_message.display()
        trigger_rerun()  # rerun WITHOUT reframing so the warning survives
        return
    reload_map(center=reset_center, zoom=reset_zoom, pitch=MapConfig.VIEWING_PITCH)  # success: reframe + rerun


def load_dem_data() -> None:
    """Return early once DEM data is loaded; otherwise load it and reframe to the start view (which
    reruns), so control never returns here after loading. Downloads from Hugging Face if absent. Uses
    DEMService.is_loaded to survive Streamlit module reloads that reset class-level singleton state.
    """
    # Check if DEM service exists AND is actually loaded (handles module reimport)
    dem_service = st.session_state.get("dem_service")
    if dem_service is not None and dem_service.is_loaded:
        return

    def _load(report: ProgressFn) -> None:
        dem_path = DEMConfig.EURODEM_PATH
        # dem_path.exists() is legitimate external-file handling: download only when missing.
        if not dem_path.exists():
            logger.info(f"Downloading DEM from Hugging Face to {dem_path}")
            download_dem_from_huggingface(
                target_path=dem_path, progress_callback=lambda f: report(f, f"Downloading terrain… {f * 100:.0f}%")
            )
        report(1.0, "Loading terrain…")
        svc = DEMService(dem_path=dem_path)
        _ = svc.get_elevation(lon=10.0, lat=47.0)  # force _ensure_loaded now
        st.session_state.dem_service = svc
        st.session_state.path_factory = PathFactory(dem_service=svc)

    run_pending_load(
        message=DEMLoadingMessage(),
        work=_load,
        reset_center=(MapConfig.START_CENTER_LON, MapConfig.START_CENTER_LAT),
        reset_zoom=MapConfig.VIEWING_ZOOM,
        catch=None,  # DEM load has no soft-failure — any error is a hard fail
        failure_message=None,
    )


# =============================================================================
# MAP RENDERING
# =============================================================================


# NOTE: @st.fragment intentionally NOT used: isolated render contexts race with session_state
# updates and break key-based deck.gl 2D/3D remounts. Full reruns + st.cache_data give equivalent
# perf without the state-sync issues.
def _render_map_fragment() -> None:
    """Render map and handle clicks.

    Named for backwards compat but NOT a fragment; full reruns + UUID keys force deterministic
    2D/3D deck.gl remounts.
    """
    try:
        _render_map_fragment_inner()
    except Exception as e:
        _handle_error_with_recovery(e, "RENDER")


def _render_map_fragment_inner() -> None:
    """Inner implementation of map fragment rendering."""
    sm: PlannerStateMachine = st.session_state.state_machine
    ctx: PlannerContext = st.session_state.context
    graph: ResortGraph = st.session_state.graph
    renderer: MapRenderer = st.session_state.map_renderer
    terrain_analyzer: TerrainAnalyzer = st.session_state.path_factory.terrain_analyzer
    dem: DEMService = st.session_state.dem_service
    build_state = BUILD_STATES[sm.get_current_state_id()]

    camera_epoch = st.session_state.get("camera_epoch", 0)
    logger.debug(f"[RENDER] Map fragment: state={sm.get_state_name()}, camera_epoch={camera_epoch}")

    # Determine 2D/3D mode early so all layers use consistent z-handling
    use_3d = ctx.viewing.view_3d

    # The current state owns its whole map surface: overlay layers, camera framing, and profile.
    extra_layers: list[pdk.Layer] = build_state.overlay_layers(
        ctx=ctx, graph=graph, renderer=renderer, terrain_analyzer=terrain_analyzer, dem=dem, use_3d=use_3d
    )

    # 3D mode: TerrainLayer with AWS tiles + OpenTopoMap texture. 2D mode: render() uses the
    # OPENTOPOMAP_STYLE map_style dict (TileLayer can't — pydeck doesn't expose renderSubLayers).
    basemap_layer = create_aws_terrain_layer() if use_3d else None

    # Camera framing for the current state (3D fit when viewing, else the stored 2D view).
    view_lat, view_lon, view_bearing, view_zoom, view_pitch = build_state.view_state(
        ctx=ctx, graph=graph, use_3d=use_3d
    )

    # Flythrough ("Play"): resolve the viewed element's groups ONCE, here at the single render
    # choke-point. Groups drive both the camera keyframes (deck.gl GLIDES between them client-side) and
    # the hot-orange highlight over the current element — so the highlight is applied in ONE place, never
    # scattered per viewing-state. Groups resolve LIVE (single source, never a stale snapshot).
    fly_groups = active_flythrough_groups()
    fly_keyframes = MapRenderer.flythrough_keyframes(groups=fly_groups)
    assert len(fly_keyframes) != 1, "flythrough must yield 0 (nothing to fly) or ≥2 keyframes, never 1"
    flying = bool(fly_keyframes)
    if flying:
        # ONE index off the current frame (keyframes are the authoritative count the driver advances).
        # The highlighted group maps through that same index — clamped into the (shorter) group list, so
        # the ribbon and the camera can never point at different elements.
        keyframe_index = ctx.viewing.flythrough_index(count=len(fly_keyframes))
        current = fly_groups[min(keyframe_index, len(fly_groups) - 1)]
        extra_layers = extra_layers + renderer.create_highlight_ribbon(polyline=current.actual_polyline, use_3d=use_3d)
        view_lat, view_lon, view_bearing, view_zoom, view_pitch = MapRenderer.flythrough_view_state(
            keyframes=fly_keyframes, index=keyframe_index
        )

    # Detect a view change by comparing the current framing to the last rendered one.
    last_view_3d = st.session_state.get("last_rendered_view_3d", False)
    last_pitch = st.session_state.get("last_rendered_pitch", 0.0)
    last_bearing = st.session_state.get("last_rendered_bearing", 0.0)
    is_view_change = not flying and (
        use_3d != last_view_3d or abs(view_pitch - last_pitch) > 0.1 or abs(view_bearing - last_bearing) > 0.1
    )

    if is_view_change:
        # A fresh UUID key forces React to remount the deck.gl component.
        new_key = str(uuid.uuid4())
        st.session_state.force_remount_key = new_key
        logger.info(
            f"[REMOUNT] View change: 3D={last_view_3d}->{use_3d}, pitch={last_pitch:.1f}->{view_pitch:.1f}, key={new_key[:8]}..."
        )

    # Store current state for next comparison — but NOT while flying, so the pitch/bearing the flythrough
    # sweeps through don't trip is_view_change on the frame Stop restores the entry fit.
    if not flying:
        st.session_state.last_rendered_view_3d = use_3d
        st.session_state.last_rendered_pitch = view_pitch
        st.session_state.last_rendered_bearing = view_bearing

    # Update renderer with calculated view state BEFORE creating deck. While flying, request deck.gl
    # easing so the camera glides between keyframes instead of jumping.
    renderer.update_view(lat=view_lat, lon=view_lon, zoom=view_zoom, pitch=view_pitch, bearing=view_bearing)
    renderer.set_flythrough_easing(flying=flying)

    def _build_deck() -> pdk.Deck:
        return renderer.render(
            proposals=ctx.proposals.paths,
            selected_proposal_idx=ctx.proposals.selected_idx,
            highlight_segment_ids=[sid for build in ctx.builds.values() for sid in build.segments],
            is_custom_path=build_state.renders_custom_path(ctx),
            extra_layers=extra_layers,
            terrain_layer=basemap_layer,
            use_3d=use_3d,
            selected_node_ids=build_state.selected_node_ids(ctx),
        )

    deck = _build_deck()

    # Map height fills the browser window — CONSTANT across every lifecycle state so the pydeck
    # component key never changes from a height shift. The elevation profile renders in the RIGHT column.
    # None only on first load, before the js-eval round-trip resolves (cached thereafter, so reruns keep the size).
    height = viewport_map_height()
    if height is None:
        # A None height mid-session blanks the map for this rerun (placeholder instead of the deck) — a
        # flicker source distinct from a key remount. Flagged so a reproduction shows it in the log.
        logger.debug("[MAP] height=None → SizingMapMessage placeholder (map blank this rerun, not a remount)")
        SizingMapMessage().display()
        return

    # force_remount_key AND height are in the key: st_deckgl only applies height on first mount, so
    # height must change the key to force a remount when it changes.
    force_key = st.session_state.get("force_remount_key", "init")
    map_key = f"main_map_{st.session_state.camera_epoch}_{force_key}_{'3d' if use_3d else '2d'}_h{height}"
    # A CHANGED map_key remounts the deck.gl iframe (WebGL teardown + tile re-fetch = the ~0.5s gray-out);
    # a same-pitch reframe must stay key_changed=False (in-place). is_view_change is the intended 2D↔3D remount.
    last_map_key = st.session_state.get("_last_map_key")
    logger.debug(
        f"[MAP] key={map_key} changed={last_map_key is not None and last_map_key != map_key} "
        f"is_view_change={is_view_change} height={height} camera_epoch={st.session_state.camera_epoch} "
        f"view=({view_lat:.5f},{view_lon:.5f},z{view_zoom},p{view_pitch:.1f},b{view_bearing:.1f})"
    )
    st.session_state._last_map_key = map_key
    click_result = render_pydeck_map(deck=deck, height=height, key=map_key)

    # Clicks are disabled in 3D (deck.gl picking is unreliable under pitch); warn instead.
    if use_3d:
        # 3D mode: show warning if user clicks terrain
        if click_result.clicked_coordinate:
            ClickingDisabledIn3DToast().display()
    else:
        detector = ClickDetector(dedup=ctx.click_dedup)
        click_info = detector.detect(
            clicked_object=click_result.clicked_object,
            clicked_coordinate=click_result.clicked_coordinate,
        )
        if click_info:
            dispatch_click(click_info=click_info)


def _advance_flythrough_if_playing() -> None:
    """Flythrough frame driver — call AFTER the control panel renders so the Stop button is drawn before
    this rerun fires (trigger_rerun raises StopExecution). Advances one keyframe per rerun; the camera
    glides IN PLACE (constant key). At the last keyframe it PARKS (stops advancing, no rerun) so the view
    stays on the finish until the user presses Stop or closes the panel.
    """
    ctx: PlannerContext = st.session_state.context
    viewing = ctx.viewing
    keyframes = MapRenderer.flythrough_keyframes(groups=active_flythrough_groups())
    if not keyframes:
        return  # nothing to fly (active_flythrough_groups is empty unless playing in 3D)
    if viewing.flythrough_frame >= len(keyframes) - 1:
        return  # parked on the final keyframe — hold here (no rerun) until Stop/Close
    time.sleep(MapConfig.FLYTHROUGH_STEP_S)
    # The sleep IS the interactive window (Stop renders before this driver). A Stop click during it flips
    # flythrough_active on the shared session state — re-check so we don't advance a just-stopped playback.
    if not viewing.flythrough_active:
        return
    viewing.advance_flythrough()
    trigger_rerun()


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:
    """Application entry point."""
    st.set_page_config(page_title=AppConfig.TITLE, page_icon=AppConfig.ICON, layout=AppConfig.LAYOUT)
    init_session_state()

    # Streamlit has no API to reclaim vertical chrome, so a full-height map needs CSS.
    st.markdown(_FULLSCREEN_CSS, unsafe_allow_html=True)

    # Block until DEM is loaded: returns early if already loaded, else shows the loading screen and
    # reruns (never returns here), so control only continues once the DEM is ready.
    load_dem_data()

    try:
        _run_app_ui()
    except Exception as e:
        _handle_error_with_recovery(e, "UI")


def _run_app_ui() -> None:
    """Run the main application UI. Separated for error handling wrapper."""
    sm: PlannerStateMachine = st.session_state.state_machine
    ctx: PlannerContext = st.session_state.context
    graph: ResortGraph = st.session_state.graph
    renderer: MapRenderer = st.session_state.map_renderer
    renderer.graph = graph

    camera_epoch = st.session_state.get("camera_epoch", 0)
    logger.debug(
        f"[MAIN] ===== rerun ===== state={sm.get_state_name()} camera_epoch={camera_epoch} "
        f"dedup_epoch={st.session_state.get('dedup_epoch', 0)} "
        f"deferred(osm={ctx.pending.osm_import_mode},custom={ctx.pending.custom_connect},"
        f"fan={bool(ctx.pending.fan_generation)})"
    )

    # Pending actions (once per render). OSM import is heavy (network + graph build): the slow helper
    # shows a blocking loading message + progress bar and returns early (no map iframe this pass, so the
    # bar is layout-safe), then reframes on the import box center on success — or shows a warning toast
    # WITHOUT reframing on failure. Custom/fan are fast: the fast helper just runs them.
    if ctx.pending.osm_import_mode is not None:
        # Capture the box center BEFORE process_* consumes (nulls) it — the reframe target on success.
        lon, lat = ctx.pending.osm_import_center_lon, ctx.pending.osm_import_center_lat
        assert lon is not None and lat is not None, "a pending OSM import always has a placed center"
        # The framed box IS the extent: side = 2 * half-width → adaptive overview zoom (same scale as builds).
        box_side_m = 2 * ctx.pending.osm_import_half_width_km * 1000.0
        run_pending_load(
            message=OSMImportLoadingMessage(mode=ctx.pending.osm_import_mode),
            work=process_osm_import_pending,
            reset_center=(lon, lat),
            reset_zoom=MapConfig.zoom_for_span_m(span_m=box_side_m),
            catch=requests.RequestException,  # only a network failure is soft; a parse/logic bug must raise
            failure_message=OSMImportErrorMessage(error="the area could not be imported — network error"),
        )
        return  # slow helper reframed/warned + reran; skip the normal UI this render
    if ctx.pending.custom_connect:
        process_custom_connect_pending()
    elif ctx.pending.fan_generation:
        process_path_generation_pending()
    elif ctx.pending.route_plan_generation:
        process_route_plan_pending()

    # Sidebar (fire-and-forget: its panels call actions directly on button clicks)
    sidebar = SidebarRenderer(state_machine=sm, context=ctx, graph=graph)
    sidebar.render()

    # Main content
    col_map, col_ctrl = st.columns([3, 1])

    with col_map:
        _render_map_fragment()

    with col_ctrl:
        render_control_panel(
            sm=sm,
            ctx=ctx,
            graph=graph,
            on_commit=commit_selected_path,
            on_cancel_connection=cancel_custom_path,
        )

    # Advance the flythrough LAST — after the panel (incl. its Stop button) has rendered, so this rerun
    # doesn't preempt the Stop control. Only reruns while a flythrough is playing.
    _advance_flythrough_if_playing()


if __name__ == "__main__":
    main()

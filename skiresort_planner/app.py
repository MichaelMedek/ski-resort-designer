"""Ski Resort Planner - Interactive terrain planning application.

Design ski resort layouts on real terrain using Digital Elevation Model (DEM) data.
Features fan-pattern path generation, lift placement, and elevation profiles.

Run: streamlit run skiresort_planner/app.py
"""

import traceback
import uuid
from typing import TYPE_CHECKING

import pydeck as pdk
import streamlit as st

from skiresort_planner.constants import (
    AppConfig,
    DEMConfig,
    MapConfig,
)
from skiresort_planner.core.dem_service import DEMService, download_dem_from_huggingface
from skiresort_planner.generators.path_factory import PathFactory
from skiresort_planner.logging_setup import configure_logging
from skiresort_planner.model.message import DEMLoadingMessage
from skiresort_planner.model.resort_graph import ResortGraph
from skiresort_planner.persistence import backup_store
from skiresort_planner.ui import (
    ClickDetector,
    MapRenderer,
    PlannerContext,
    PlannerStateMachine,
    SidebarRenderer,
    bump_camera_epoch,
    cancel_custom_path,
    commit_selected_path,
    dispatch_click,
    process_custom_connect_deferred,
    process_osm_import_deferred,
    process_path_generation_deferred,
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
            zoom=MapConfig.DEFAULT_ZOOM,
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


def load_dem_data() -> bool:
    """Load DEM data. Returns True when loaded, False while loading.

    Downloads from Hugging Face if not present locally. Uses DEMService.is_loaded to survive
    Streamlit module reloads that reset class-level singleton state.
    """
    # Check if DEM service exists AND is actually loaded (handles module reimport)
    dem_service = st.session_state.get("dem_service")
    if dem_service is not None and dem_service.is_loaded:
        return True

    # Show loading screen with centered message
    DEMLoadingMessage().display()

    dem_path = DEMConfig.EURODEM_PATH

    # Download from Hugging Face if not present locally
    if not dem_path.exists():
        st.info("🗺️ Downloading Alps terrain data from Hugging Face (~285MB)...")
        progress_bar = st.progress(0, text="Starting download...")

        def update_progress(progress: float) -> None:
            progress_bar.progress(progress, text=f"Downloading... {progress * 100:.0f}%")

        logger.info(f"Downloading DEM from Hugging Face to {dem_path}")
        download_dem_from_huggingface(target_path=dem_path, progress_callback=update_progress)
        progress_bar.progress(1.0, text="Download complete!")

    with st.spinner("Loading terrain elevation data..."):
        dem_service = DEMService(dem_path=dem_path)
        # Force immediate loading by querying a point (triggers _ensure_loaded)
        _ = dem_service.get_elevation(lon=10.0, lat=47.0)
        st.session_state.dem_service = dem_service
        st.session_state.path_factory = PathFactory(dem_service=dem_service)

    trigger_rerun()  # Raises StopExecution in production; returns in tests
    return False  # Reached only under a mocked rerun: DEM not ready yet


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

    # Detect a view change by comparing the current framing to the last rendered one.
    last_view_3d = st.session_state.get("last_rendered_view_3d", False)
    last_pitch = st.session_state.get("last_rendered_pitch", 0.0)
    last_bearing = st.session_state.get("last_rendered_bearing", 0.0)
    is_view_change = (
        use_3d != last_view_3d or abs(view_pitch - last_pitch) > 0.1 or abs(view_bearing - last_bearing) > 0.1
    )

    if is_view_change:
        # A fresh UUID key forces React to remount the deck.gl component.
        new_key = str(uuid.uuid4())
        st.session_state.force_remount_key = new_key
        logger.info(
            f"[REMOUNT] View change: 3D={last_view_3d}->{use_3d}, pitch={last_pitch:.1f}->{view_pitch:.1f}, key={new_key[:8]}..."
        )

    # Store current state for next comparison
    st.session_state.last_rendered_view_3d = use_3d
    st.session_state.last_rendered_pitch = view_pitch
    st.session_state.last_rendered_bearing = view_bearing

    # Update renderer with calculated view state BEFORE creating deck
    renderer.update_view(lat=view_lat, lon=view_lon, zoom=view_zoom, pitch=view_pitch, bearing=view_bearing)

    def _build_deck() -> pdk.Deck:
        return renderer.render(
            proposals=ctx.proposals.paths,
            selected_proposal_idx=ctx.proposals.selected_idx,
            highlight_segment_ids=[sid for build in ctx.builds.values() for sid in build.segments],
            is_custom_path=build_state.renders_custom_path(ctx),
            extra_layers=extra_layers,
            terrain_layer=basemap_layer,
            use_3d=use_3d,
            merge_node_ids=build_state.merge_highlight_node_ids(ctx),
        )

    deck = _build_deck()

    # Map height fills the browser window — CONSTANT across every lifecycle state so the pydeck
    # component key never changes from a height shift. The elevation profile renders in the RIGHT column.
    # None only on first load, before the js-eval round-trip resolves (cached thereafter, so reruns keep the size).
    height = viewport_map_height()
    if height is None:
        st.info("📐 Sizing map to your window…")
        return

    # force_remount_key AND height are in the key: st_deckgl only applies height on first mount, so
    # height must change the key to force a remount when it changes.
    force_key = st.session_state.get("force_remount_key", "init")
    map_key = f"main_map_{st.session_state.camera_epoch}_{force_key}_{'3d' if use_3d else '2d'}_h{height}"
    # Diagnostic: a CHANGED map_key remounts the deck.gl iframe (camera snaps to initial_view_state).
    last_map_key = st.session_state.get("_last_map_key")
    logger.debug(
        f"[MAP] key={map_key} (changed={last_map_key != map_key}) height={height} "
        f"camera_epoch={st.session_state.camera_epoch} force_key={force_key} "
        f"view=({view_lat:.5f},{view_lon:.5f},z{view_zoom},p{view_pitch:.1f},b{view_bearing:.1f})"
    )
    st.session_state._last_map_key = map_key
    click_result = render_pydeck_map(deck=deck, height=height, key=map_key)

    # Clicks are disabled in 3D (deck.gl picking is unreliable under pitch); warn instead.
    if use_3d:
        # 3D mode: show warning if user clicks terrain
        if click_result.clicked_coordinate:
            st.toast("Clicking disabled in 3D view. Return to 2D to interact with the map.", icon="⚠️")
    else:
        detector = ClickDetector(dedup=ctx.click_dedup)
        click_info = detector.detect(
            clicked_object=click_result.clicked_object,
            clicked_coordinate=click_result.clicked_coordinate,
        )
        if click_info:
            dispatch_click(click_info=click_info)


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:
    """Application entry point."""
    st.set_page_config(page_title=AppConfig.TITLE, page_icon=AppConfig.ICON, layout=AppConfig.LAYOUT)
    init_session_state()

    # Streamlit has no API to reclaim vertical chrome, so a full-height map needs CSS.
    st.markdown(_FULLSCREEN_CSS, unsafe_allow_html=True)

    # Block until DEM is loaded - shows loading message and prevents map interaction
    if not load_dem_data():
        return

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
        f"deferred(osm={ctx.deferred.osm_import_mode},custom={ctx.deferred.custom_connect},"
        f"fan={bool(ctx.deferred.fan_generation)})"
    )

    # Deferred actions (once per render). Progress uses st.toast, NOT st.spinner: a body spinner
    # shifts the body element order between reruns, re-creating the map iframe (flash + camera reset);
    # a toast is a transient overlay that never touches the layout. Fan is fast (no cue).
    if ctx.deferred.osm_import_mode is not None:
        st.toast("🗺️ Importing lifts & pistes from OpenStreetMap…")
        process_osm_import_deferred()
    elif ctx.deferred.custom_connect:
        st.toast("🎯 Computing custom path options…")
        process_custom_connect_deferred()
    elif ctx.deferred.fan_generation:
        process_path_generation_deferred()  # fast — no cue needed

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


if __name__ == "__main__":
    main()

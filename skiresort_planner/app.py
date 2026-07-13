"""Ski Resort Planner - Interactive terrain planning application.

Design ski resort layouts on real terrain using Digital Elevation Model (DEM) data.
Features fan-pattern path generation, lift placement, and elevation profiles.

Run: streamlit run skiresort_planner/app.py
"""

import logging
import traceback
from typing import TYPE_CHECKING

import pydeck as pdk
import streamlit as st

from skiresort_planner.constants import (
    AppConfig,
    ChartConfig,
    DEMConfig,
    MapConfig,
)
from skiresort_planner.core.dem_service import DEMService, download_dem_from_huggingface
from skiresort_planner.generators.path_factory import PathFactory
from skiresort_planner.model.message import DEMLoadingMessage
from skiresort_planner.model.resort_graph import ResortGraph
from skiresort_planner.persistence import backup_store
from skiresort_planner.ui import (
    ClickDetector,
    MapRenderer,
    PlannerContext,
    PlannerStateMachine,
    SidebarRenderer,
    bump_map_version,
    cancel_current_road,
    cancel_current_slope,
    cancel_custom_path,
    commit_selected_path,
    dispatch_click,
    finish_current_road,
    finish_current_slope,
    handle_fast_deferred_actions,
    process_custom_connect_deferred,
    process_osm_import_deferred,
    process_path_generation_deferred,
    recompute_paths,
    render_building_profile,
    render_control_panel,
    render_viewing_profile,
    trigger_rerun,
    viewport_map_height,
)
from skiresort_planner.ui.pydeck_click_handler import render_pydeck_map
from skiresort_planner.ui.terrain_layer import create_aws_terrain_layer

if TYPE_CHECKING:
    from skiresort_planner.core.terrain_analyzer import TerrainAnalyzer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# CSS to give the map the full window height. Streamlit exposes no API for any of these,
# so injection is the only option (the standard community pattern). Each rule is load-bearing:
_FULLSCREEN_CSS = """
<style>
/* Hide the top toolbar (Deploy/menu strip) — reclaims its height for the map. */
header[data-testid="stHeader"] { display: none; }
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

    if "map_version" not in st.session_state:
        st.session_state.map_version = 0


def _init_resort_from_url_or_new() -> None:
    """Resolve the session's resort_id and prime the graph from a backup.

    - If the URL has ?resort=<id> and a backup exists, load it. This is the
      reload path — F5, brief outage, and reopened bookmarks keep the URL.
    - Otherwise (bare link) fall back to the biggest existing backup by node
      count — almost always the user's own work — and adopt its id.
    - If no backups exist, start a fresh empty resort.
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


def reset_ui_state() -> None:
    """Reset UI state to initial while preserving the resort graph.

    Called when an error occurs to recover gracefully. Resets:
    - State machine to Idle state
    - Context to fresh instance
    - Map version (to clear any stale map state)

    Preserves:
    - Resort graph (all slopes, lifts, nodes, segments)
    - DEM service and path factory
    - Map renderer (re-linked to graph)
    """
    logger.info("Resetting UI state due to error recovery")

    # Create fresh state machine and context
    sm, ctx = PlannerStateMachine.create(graph=st.session_state.graph)
    st.session_state.state_machine = sm
    st.session_state.context = ctx

    # Increment map version to force fresh map component
    bump_map_version()

    logger.info("UI state reset complete - graph preserved")


def load_dem_data() -> bool:
    """Load DEM data. Returns True when loaded, False while loading.

    Downloads the DEM from Hugging Face if not present locally, then loads
    into memory. Uses DEMService.is_loaded property to handle Streamlit
    module reloads that can reset class-level singleton state.
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


# NOTE: @st.fragment intentionally NOT used here.
# Fragments create isolated render contexts that can cause race conditions with
# session_state updates, preventing proper key-based remounts for deck.gl 2D/3D
# view transitions. Full app reruns with st.cache_data for heavy computations
# (DEM loading, path generation) provide equivalent performance without the
# state synchronization issues. See: Streamlit docs on fragment limitations.
def _render_map_fragment() -> None:
    """Render map and handle clicks.

    Despite the name (kept for backwards compatibility), this is NOT a fragment.
    Full app reruns ensure deterministic 2D/3D view transitions via UUID-based
    key changes that force deck.gl component remounts.
    """
    try:
        _render_map_fragment_inner()
        logger.debug("[RENDER] _render_map_fragment_inner() completed successfully")
    except Exception as e:
        # Log full traceback for debugging
        error_msg = f"{type(e).__name__}: {e}"
        full_traceback = traceback.format_exc()
        logger.error(f"[RENDER] Map fragment error caught: {error_msg}\n{full_traceback}")

        # Show user-friendly error message
        st.error(f"⚠️ [RENDER] Something went wrong: {error_msg}")

        # Reset UI state while preserving the graph
        reset_ui_state()

        # Add a button to manually recover
        if st.button("🔄 Reset and Continue", type="primary"):
            trigger_rerun()


def _render_map_fragment_inner() -> None:
    """Inner implementation of map fragment rendering."""
    sm: PlannerStateMachine = st.session_state.state_machine
    ctx: PlannerContext = st.session_state.context
    graph: ResortGraph = st.session_state.graph
    renderer: MapRenderer = st.session_state.map_renderer
    terrain_analyzer: TerrainAnalyzer = st.session_state.path_factory.terrain_analyzer

    map_version = st.session_state.get("map_version", 0)
    logger.info(f"[RENDER] Map fragment: state={sm.get_state_name()}, map_version={map_version}")

    # Determine 2D/3D mode early so all layers use consistent z-handling
    use_3d = ctx.viewing.view_3d

    # Collect extra layers for overlays
    extra_layers: list[pdk.Layer] = []

    # Add orientation arrows in Building state.
    sel = ctx.selection
    if sm.is_any_slope_state and sel.lon is not None and sel.lat is not None and sel.elevation is not None:
        orientation = terrain_analyzer.get_orientation(lon=sel.lon, lat=sel.lat)
        if orientation:
            arrow_layers = renderer.create_orientation_arrows_layers(
                lat=sel.lat,
                lon=sel.lon,
                elevation=sel.elevation,
                orientation=orientation,
                use_3d=use_3d,
            )
            extra_layers.extend(arrow_layers)

    # Add direction arrow while routing a custom-connect path (targeting a clicked point)
    if ctx.custom_connect.force_mode and ctx.custom_connect.start_node:
        start_node = graph.nodes.get(ctx.custom_connect.start_node)
        if start_node:
            gradient = terrain_analyzer.compute_gradient(lon=start_node.lon, lat=start_node.lat)
            arrow_layer = renderer.create_direction_arrow_layer(
                start_lat=start_node.lat,
                start_lon=start_node.lon,
                bearing_deg=gradient.bearing_deg,
                direction="downhill",
                use_3d=use_3d,
            )
            extra_layers.append(arrow_layer)

    # Add lift marker in LiftPlacing state
    if sm.is_lift_placing and (ctx.lift.start_node_id or ctx.lift.start_location):
        if ctx.lift.start_node_id:
            lift_start_node = graph.nodes.get(ctx.lift.start_node_id)
            if lift_start_node is None:
                raise ValueError(f"Lift start node {ctx.lift.start_node_id} not found in graph")
            gradient = terrain_analyzer.compute_gradient(lon=lift_start_node.lon, lat=lift_start_node.lat)
            lift_layers = renderer.create_pending_lift_marker_layers(
                lat=lift_start_node.lat,
                lon=lift_start_node.lon,
                elevation=lift_start_node.elevation,
                fall_line_bearing=gradient.bearing_deg,
                use_3d=use_3d,
            )
            extra_layers.extend(lift_layers)
        elif ctx.lift.start_location:
            loc = ctx.lift.start_location
            gradient = terrain_analyzer.compute_gradient(lon=loc.lon, lat=loc.lat)
            lift_layers = renderer.create_pending_lift_marker_layers(
                lat=loc.lat,
                lon=loc.lon,
                elevation=loc.elevation,
                fall_line_bearing=gradient.bearing_deg,
                use_3d=use_3d,
            )
            extra_layers.extend(lift_layers)

    # Add road origin marker in RoadStarting state (no segments committed yet, so
    # the click point needs a visible dot — like the lift bottom station, minus the
    # direction arrow since a road has no fall-line orientation).
    if sm.is_road_starting and (ctx.road_build.start_node_id or ctx.road_build.start_location):
        if ctx.road_build.start_node_id:
            road_start_node = graph.nodes.get(ctx.road_build.start_node_id)
            if road_start_node is None:
                raise ValueError(f"Road start node {ctx.road_build.start_node_id} not found in graph")
            extra_layers.extend(
                renderer.create_pending_road_marker_layers(
                    lat=road_start_node.lat,
                    lon=road_start_node.lon,
                    elevation=road_start_node.elevation,
                    use_3d=use_3d,
                )
            )
        elif ctx.road_build.start_location:
            loc = ctx.road_build.start_location
            extra_layers.extend(
                renderer.create_pending_road_marker_layers(
                    lat=loc.lat,
                    lon=loc.lon,
                    elevation=loc.elevation,
                    use_3d=use_3d,
                )
            )

    # Add the OSM import box (rectangle + pickable center dot) while placing an import area.
    if sm.is_import_placing and ctx.deferred.osm_import_center_lon is not None:
        center_lon = ctx.deferred.osm_import_center_lon
        center_lat = ctx.deferred.osm_import_center_lat
        assert center_lat is not None  # set together with lon by start_import
        dem_service: DEMService = st.session_state.dem_service
        center_elev = dem_service.get_elevation(lon=center_lon, lat=center_lat) or 0.0
        extra_layers.extend(
            renderer.create_import_bbox_layers(
                center_lon=center_lon,
                center_lat=center_lat,
                half_width_m=ctx.deferred.osm_import_half_width_km * 1000.0,
                elevation=center_elev,
                use_3d=use_3d,
            )
        )
    # 3D mode: TerrainLayer with AWS tiles + OpenTopoMap texture
    # 2D mode: No terrain_layer needed - render() uses OPENTOPOMAP_STYLE map_style dict
    #          (TileLayer doesn't work because pydeck doesn't expose renderSubLayers)
    basemap_layer = create_aws_terrain_layer() if use_3d else None

    # Calculate view state BEFORE creating deck (fixes inconsistent 2D/3D toggle)
    # Update renderer's internal state so deck is created with correct values
    if use_3d and sm.is_info_panel_visible:
        # Calculate optimal 3D camera position for viewing slope/lift
        if sm.is_idle_viewing_slope and ctx.viewing.slope_id:
            view_lat, view_lon, view_bearing, view_zoom, view_pitch = MapRenderer.calculate_3d_view_for_slope(
                graph=graph, slope_id=ctx.viewing.slope_id
            )
        elif sm.is_idle_viewing_lift and ctx.viewing.lift_id:
            view_lat, view_lon, view_bearing, view_zoom, view_pitch = MapRenderer.calculate_3d_view_for_lift(
                graph=graph, lift_id=ctx.viewing.lift_id
            )
        elif sm.is_idle_viewing_road and ctx.viewing.road_id:
            view_lat, view_lon, view_bearing, view_zoom, view_pitch = MapRenderer.calculate_3d_view_for_road(
                graph=graph, road_id=ctx.viewing.road_id
            )
        else:
            # 3D enabled but not viewing - shouldn't happen, disable 3D
            ctx.viewing.disable_3d()
            view_lat, view_lon, view_bearing, view_zoom, view_pitch = (
                ctx.map.lat,
                ctx.map.lon,
                ctx.map.bearing,
                ctx.map.zoom,
                ctx.map.pitch,
            )
    else:
        # Normal 2D view - use stored view state
        view_lat, view_lon, view_bearing, view_zoom, view_pitch = (
            ctx.map.lat,
            ctx.map.lon,
            ctx.map.bearing,
            ctx.map.zoom,
            ctx.map.pitch,
        )

    # SIMPLE view change detection: compare current state to last rendered state
    # This replaces complex callback injection with direct comparison
    last_view_3d = st.session_state.get("last_rendered_view_3d", False)
    last_pitch = st.session_state.get("last_rendered_pitch", 0.0)
    last_bearing = st.session_state.get("last_rendered_bearing", 0.0)

    is_view_change = (
        use_3d != last_view_3d or abs(view_pitch - last_pitch) > 0.1 or abs(view_bearing - last_bearing) > 0.1
    )

    if is_view_change:
        # UUID guarantees unique key - forces React to remount deck.gl component
        import uuid

        new_key = str(uuid.uuid4())
        st.session_state.force_remount_key = new_key
        logger.info(
            f"[REMOUNT] View change detected: 3D={last_view_3d}->{use_3d}, pitch={last_pitch:.1f}->{view_pitch:.1f}, key={new_key[:8]}..."
        )

    # Store current state for next comparison
    st.session_state.last_rendered_view_3d = use_3d
    st.session_state.last_rendered_pitch = view_pitch
    st.session_state.last_rendered_bearing = view_bearing

    # Update renderer with calculated view state BEFORE creating deck
    renderer.update_view(lat=view_lat, lon=view_lon, zoom=view_zoom, pitch=view_pitch, bearing=view_bearing)

    # Render deck with all layers - deck is created with correct view state
    # Use spinner during view changes (2D/3D toggle, Reset View) to show progress
    if is_view_change:
        with st.spinner("🔄 Switching view..."):
            deck = renderer.render(
                proposals=ctx.proposals.paths,
                selected_proposal_idx=ctx.proposals.selected_idx,
                highlight_segment_ids=ctx.slope_build.segments,
                is_custom_path=ctx.custom_connect.force_mode or sm.is_any_road_state,
                extra_layers=extra_layers,
                terrain_layer=basemap_layer,
                use_3d=use_3d,
            )
    else:
        deck = renderer.render(
            proposals=ctx.proposals.paths,
            selected_proposal_idx=ctx.proposals.selected_idx,
            highlight_segment_ids=ctx.slope_build.segments,
            is_custom_path=ctx.custom_connect.force_mode or sm.is_any_road_state,
            extra_layers=extra_layers,
            terrain_layer=basemap_layer,
            use_3d=use_3d,
        )

    # One elevation profile renders directly below the map: the in-build profile while
    # building a slope/road, or the finished entity's profile while viewing it. Reserve
    # room for it so it stays visible without scrolling; idle mode lets the map fill all.
    show_slope_profile = sm.is_any_slope_state and bool(ctx.slope_build.segments)
    show_road_profile = sm.is_any_road_state and bool(ctx.road_build.segments)
    viewing = sm.viewing_entity  # (EntityKind, id) or None
    reserved = ChartConfig.PROFILE_HEIGHT_PX if (show_slope_profile or show_road_profile or viewing) else 0

    # Map height that fills the browser window. None only on first load, before the
    # js-eval round-trip resolves (cached thereafter, so reruns keep the size).
    height = viewport_map_height(reserved_below_px=reserved)
    if height is None:
        st.info("📐 Sizing map to your window…")
        return

    # Render with click handling
    # Include force_remount_key AND height in key: st_deckgl only applies height on
    # first mount, so height must change the key to force a remount when it changes.
    force_key = st.session_state.get("force_remount_key", "init")
    map_key = f"main_map_{st.session_state.map_version}_{force_key}_{'3d' if use_3d else '2d'}_h{height}"
    click_result = render_pydeck_map(
        deck=deck,
        height=height,
        key=map_key,
    )

    # The single profile below the map — building (kind-driven) or viewing (kind-driven).
    # No else: idle-ready has no profile (matches `reserved == 0`); the map fills the viewport.
    if show_slope_profile:
        fig = render_building_profile(
            building_segments=ctx.slope_build.segments, building_name=ctx.slope_build.name, graph=graph
        )
        st.plotly_chart(fig, width="stretch", key="combined_profile")
    elif show_road_profile:
        fig = render_building_profile(
            building_segments=ctx.road_build.segments, building_name=ctx.road_build.name, graph=graph
        )
        st.plotly_chart(fig, width="stretch", key="combined_road_profile")
    elif viewing is not None:
        kind, entity_id = viewing
        fig = render_viewing_profile(kind=kind, entity_id=entity_id, graph=graph)
        st.plotly_chart(fig, width="stretch", key="viewing_profile")

    # Detect clicks from Pydeck result - disabled in 3D mode
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
        # Log full traceback for debugging
        error_msg = f"{type(e).__name__}: {e}"
        full_traceback = traceback.format_exc()
        logger.error(f"[UI] UI error caught: {error_msg}\n{full_traceback}")

        # Show user-friendly error message
        st.error(f"⚠️ [UI] Something went wrong: {error_msg}")

        # Reset UI state while preserving the graph
        reset_ui_state()

        # Add a button to manually recover
        if st.button("🔄 Reset and Continue", type="primary"):
            trigger_rerun()


def _run_app_ui() -> None:
    """Run the main application UI. Separated for error handling wrapper."""
    sm: PlannerStateMachine = st.session_state.state_machine
    ctx: PlannerContext = st.session_state.context
    graph: ResortGraph = st.session_state.graph
    renderer: MapRenderer = st.session_state.map_renderer
    renderer.graph = graph

    map_version = st.session_state.get("map_version", 0)
    logger.info(f"[MAIN] Render cycle starting: state={sm.get_state_name()}, map_version={map_version}")

    # Handle deferred actions from previous transitions
    # Slow ops get spinners, fast ops run directly
    if ctx.deferred.osm_import:
        with st.spinner("🗺️ Importing lifts & pistes from OpenStreetMap..."):
            process_osm_import_deferred()
    elif ctx.deferred.custom_connect:
        with st.spinner("🎯 Computing custom path options..."):
            process_custom_connect_deferred()
    elif ctx.deferred.path_generation:
        with st.spinner("🗺️ Generating path options..."):
            process_path_generation_deferred()
    else:
        handle_fast_deferred_actions()

    # Sidebar
    sidebar = SidebarRenderer(state_machine=sm, context=ctx, graph=graph)
    actions = sidebar.render()

    # Handle actions
    if actions.get("finish_slope"):
        finish_current_slope()
    if actions.get("cancel_slope"):
        cancel_current_slope()
    if actions.get("finish_road"):
        finish_current_road()
    if actions.get("cancel_road"):
        cancel_current_road()
    if actions.get("recompute") or ctx.click_dedup.pending_recompute:
        recompute_paths()

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

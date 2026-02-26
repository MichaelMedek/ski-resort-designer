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
from skiresort_planner.ui import (
    ClickDetector,
    MapRenderer,
    PlannerContext,
    PlannerStateMachine,
    ProfileChart,
    SidebarRenderer,
    cancel_current_slope,
    cancel_custom_direction_mode,
    cancel_custom_path,
    commit_selected_path,
    dispatch_click,
    enter_custom_direction_mode,
    finish_current_slope,
    handle_deferred_actions,
    recompute_paths,
    render_building_profiles,
    render_control_panel,
    render_proposal_preview,
)
from skiresort_planner.ui.pydeck_click_handler import render_pydeck_map
from skiresort_planner.ui.terrain_layer import create_aws_terrain_layer

if TYPE_CHECKING:
    from skiresort_planner.core.terrain_analyzer import TerrainAnalyzer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# SESSION STATE
# =============================================================================


def init_session_state() -> None:
    """Initialize session state with DEM, graph, and UI components."""
    if "graph" not in st.session_state:
        st.session_state.graph = ResortGraph()

    if "state_machine" not in st.session_state:
        sm, ctx = PlannerStateMachine.create(graph=st.session_state.graph)
        st.session_state.state_machine = sm
        st.session_state.context = ctx

    if "map_renderer" not in st.session_state:
        st.session_state.map_renderer = MapRenderer(
            center_lon=MapConfig.START_CENTER_LON,
            center_lat=MapConfig.START_CENTER_LAT,
            zoom=MapConfig.DEFAULT_ZOOM,
            pitch=MapConfig.DEFAULT_PITCH,
            bearing=MapConfig.DEFAULT_BEARING,
        )

    if "_upload_counter" not in st.session_state:
        st.session_state._upload_counter = 0

    if "map_version" not in st.session_state:
        st.session_state.map_version = 0


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
    st.session_state.map_version = st.session_state.get("map_version", 0) + 1

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

    st.rerun()  # Raises StopExecution, never returns


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
            st.rerun()


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

    # Add orientation arrows in Building state
    if sm.is_any_slope_state and ctx.selection.lon is not None and ctx.selection.lat is not None:
        orientation = terrain_analyzer.get_orientation(lon=ctx.selection.lon, lat=ctx.selection.lat)
        if orientation:
            arrow_layers = renderer.create_orientation_arrows_layers(
                lat=ctx.selection.lat,
                lon=ctx.selection.lon,
                elevation=ctx.selection.elevation or 0.0,
                orientation=orientation,
                use_3d=use_3d,
            )
            extra_layers.extend(arrow_layers)

    # Add direction arrow in custom connect mode
    if ctx.custom_connect.enabled and ctx.custom_connect.start_node:
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
                highlight_segment_ids=ctx.building.segments,
                is_custom_path=ctx.custom_connect.force_mode,
                extra_layers=extra_layers,
                terrain_layer=basemap_layer,
                use_3d=use_3d,
            )
    else:
        deck = renderer.render(
            proposals=ctx.proposals.paths,
            selected_proposal_idx=ctx.proposals.selected_idx,
            highlight_segment_ids=ctx.building.segments,
            is_custom_path=ctx.custom_connect.force_mode,
            extra_layers=extra_layers,
            terrain_layer=basemap_layer,
            use_3d=use_3d,
        )

    # Render with click handling
    # Include force_remount_key in key to force component hard remount on view changes
    force_key = st.session_state.get("force_remount_key", "init")
    map_key = f"main_map_{st.session_state.map_version}_{force_key}_{'3d' if use_3d else '2d'}"
    click_result = render_pydeck_map(
        deck=deck,
        height=ChartConfig.PROFILE_HEIGHT_LARGE,
        key=map_key,
    )

    # Elevation profiles below map
    if sm.is_any_slope_state and ctx.building.segments:
        fig = render_building_profiles(
            building_segments=ctx.building.segments,
            building_name=ctx.building.name,
            graph=graph,
        )
        st.plotly_chart(fig, width="stretch", key="combined_profile")

    if ctx.proposals.paths and ctx.proposals.selected_idx is not None:
        fig = render_proposal_preview(proposals=ctx.proposals.paths, selected_idx=ctx.proposals.selected_idx)
        st.plotly_chart(fig, width="stretch", key="preview_profile")

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

    st.title(AppConfig.TITLE)

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
            st.rerun()


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
    handle_deferred_actions()

    # Sidebar
    sidebar = SidebarRenderer(state_machine=sm, context=ctx, graph=graph)
    actions = sidebar.render()

    # Handle actions
    if actions.get("finish_slope"):
        finish_current_slope()
    if actions.get("cancel_slope"):
        cancel_current_slope()
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
            on_custom_direction=enter_custom_direction_mode,
            on_cancel_custom=cancel_custom_direction_mode,
            on_cancel_connection=cancel_custom_path,
        )

    # Full-width profile for viewing slope
    if sm.is_idle_viewing_slope and ctx.viewing.slope_id:
        chart = ProfileChart(height=ChartConfig.PROFILE_HEIGHT_MEDIUM, width=ChartConfig.WIDE_WIDTH)
        slope = graph.slopes.get(ctx.viewing.slope_id)
        if slope is None:
            raise ValueError(f"Slope {ctx.viewing.slope_id} must exist when panel shows slope")
        fig = chart.render_slope(slope=slope, graph=graph)
        st.plotly_chart(fig, key="slope_full_profile")

    # Full-width profile for viewing lift
    if sm.is_idle_viewing_lift and ctx.viewing.lift_id:
        lift = graph.lifts.get(ctx.viewing.lift_id)
        if lift is None:
            raise ValueError(f"Lift {ctx.viewing.lift_id} must exist when panel shows lift")
        chart = ProfileChart(height=ChartConfig.LIFT_PROFILE_HEIGHT, width=ChartConfig.WIDE_WIDTH)
        fig = chart.render_lift(lift=lift, graph=graph)
        st.plotly_chart(fig, key="lift_full_profile")


if __name__ == "__main__":
    main()

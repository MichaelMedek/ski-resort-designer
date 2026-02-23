"""Ski Resort Planner - Interactive terrain planning application.

Design ski resort layouts on real terrain using Digital Elevation Model (DEM) data.
Features fan-pattern path generation, lift placement, and elevation profiles.

Run: streamlit run skiresort_planner/app.py
"""

import logging
import traceback
from typing import TYPE_CHECKING

import streamlit as st
from streamlit_folium import st_folium

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
    cancel_connection_mode,
    cancel_current_slope,
    cancel_custom_direction_mode,
    commit_selected_path,
    dispatch_click,
    enter_custom_direction_mode,
    finish_current_slope,
    handle_deferred_actions,
    recompute_paths,
    render_building_profiles,
    render_control_panel,
    render_proposal_preview,
    undo_last_action,
)

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
        sm, ctx = PlannerStateMachine.create()
        st.session_state.state_machine = sm
        st.session_state.context = ctx

    if "map_renderer" not in st.session_state:
        st.session_state.map_renderer = MapRenderer(
            center_lat=MapConfig.START_CENTER_LAT,
            center_lon=MapConfig.START_CENTER_LON,
            zoom=MapConfig.DEFAULT_ZOOM,
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
    sm, ctx = PlannerStateMachine.create()
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

    st.rerun()
    return False


# =============================================================================
# MAP FRAGMENT
# =============================================================================


@st.fragment
def _render_map_fragment() -> None:
    """Render map and handle clicks in an isolated fragment.

    With returned_objects limited to click fields only, pan/zoom don't trigger
    reruns at all. The fragment isolates map interactions from the rest of the UI.
    """
    try:
        _render_map_fragment_inner()
    except Exception as e:
        # Log full traceback for debugging
        error_msg = f"{type(e).__name__}: {e}"
        logger.error(f"Map fragment error caught: {error_msg}\n{traceback.format_exc()}")

        # Reset UI state while preserving the graph
        reset_ui_state()

        # Rerun to show clean UI
        st.rerun()


def _render_map_fragment_inner() -> None:
    """Inner implementation of map fragment rendering."""
    sm: PlannerStateMachine = st.session_state.state_machine
    ctx: PlannerContext = st.session_state.context
    graph: ResortGraph = st.session_state.graph
    renderer: MapRenderer = st.session_state.map_renderer

    m = renderer.render(
        proposals=ctx.proposals.paths,
        selected_proposal_idx=ctx.proposals.selected_idx,
        highlight_segment_ids=ctx.building.segments,
        is_custom_path=ctx.custom_connect.force_mode,
    )

    # Add orientation arrows in Building state
    if sm.is_slope_building and ctx.selection.lon is not None and ctx.selection.lat is not None:
        terrain_analyzer: TerrainAnalyzer = st.session_state.path_factory.terrain_analyzer
        orientation = terrain_analyzer.get_orientation(lon=ctx.selection.lon, lat=ctx.selection.lat)
        if orientation:
            renderer.add_orientation_arrows(
                m=m,
                lat=ctx.selection.lat,
                lon=ctx.selection.lon,
                orientation=orientation,
            )

    # Add direction arrow in custom connect mode
    if ctx.custom_connect.enabled and ctx.custom_connect.start_node:
        start_node = graph.nodes.get(ctx.custom_connect.start_node)
        if start_node:
            terrain_analyzer: TerrainAnalyzer = st.session_state.path_factory.terrain_analyzer
            gradient = terrain_analyzer.compute_gradient(lon=start_node.lon, lat=start_node.lat)
            renderer.add_direction_arrow(
                m=m,
                start_lat=start_node.lat,
                start_lon=start_node.lon,
                bearing_deg=gradient.bearing_deg,
                direction="downhill",
                tooltip="🎯 Click DOWNHILL to create path",
            )

    # Add lift marker in LiftPlacing state
    if sm.is_lift_placing and (ctx.lift.start_node_id or ctx.lift.start_location):
        terrain_analyzer: TerrainAnalyzer = st.session_state.path_factory.terrain_analyzer
        if ctx.lift.start_node_id:
            start_node = graph.nodes.get(ctx.lift.start_node_id)
            gradient = terrain_analyzer.compute_gradient(lon=start_node.lon, lat=start_node.lat)
            renderer.add_pending_lift_marker(
                m=m, fall_line_bearing=gradient.bearing_deg, node_id=ctx.lift.start_node_id
            )
        elif ctx.lift.start_location:
            loc = ctx.lift.start_location
            gradient = terrain_analyzer.compute_gradient(lon=loc.lon, lat=loc.lat)
            renderer.add_pending_lift_marker(m=m, fall_line_bearing=gradient.bearing_deg, location=loc)

    # Render map - dynamic key resets st_folium state when map_version changes
    # This eliminates ghost clicks by creating a fresh component instance
    map_data = st_folium(
        m,
        width=None,
        height=ChartConfig.PROFILE_HEIGHT_LARGE,
        center=ctx.map.center,
        zoom=ctx.map.zoom,
        returned_objects=["last_clicked", "last_object_clicked", "last_object_clicked_tooltip"],
        key=f"main_map_{st.session_state.map_version}",
    )

    # Elevation profiles below map
    if sm.is_slope_building and ctx.building.segments:
        fig = render_building_profiles(
            building_segments=ctx.building.segments,
            building_name=ctx.building.name,
            graph=graph,
        )
        st.plotly_chart(fig, width="stretch", key="combined_profile")

    if ctx.proposals.paths and ctx.proposals.selected_idx is not None:
        fig = render_proposal_preview(proposals=ctx.proposals.paths, selected_idx=ctx.proposals.selected_idx)
        st.plotly_chart(fig, width="stretch", key="preview_profile")

    # Detect clicks - only fires on actual clicks (not pan/zoom)
    detector = ClickDetector(dedup=ctx.click_dedup)
    click_info = detector.detect(map_data=map_data)
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
        logger.error(f"UI error caught: {error_msg}\n{traceback.format_exc()}")

        # Reset UI state while preserving the graph
        reset_ui_state()

        # Rerun to show clean UI
        st.rerun()


def _run_app_ui() -> None:
    """Run the main application UI. Separated for error handling wrapper."""
    sm: PlannerStateMachine = st.session_state.state_machine
    ctx: PlannerContext = st.session_state.context
    graph: ResortGraph = st.session_state.graph
    renderer: MapRenderer = st.session_state.map_renderer
    renderer.graph = graph

    # Handle deferred actions from previous transitions
    handle_deferred_actions()

    # Sidebar
    sidebar = SidebarRenderer(state_machine=sm, context=ctx, graph=graph)
    actions = sidebar.render()

    # Handle actions
    if actions.get("finish_slope"):
        finish_current_slope()
    if actions.get("undo"):
        undo_last_action()
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
            on_cancel_connection=cancel_connection_mode,
        )

    # Full-width profile for viewing slope (panel visible with slope_id set)
    if ctx.viewing.panel_visible and ctx.viewing.slope_id:
        chart = ProfileChart(height=ChartConfig.PROFILE_HEIGHT_MEDIUM, width=ChartConfig.WIDE_WIDTH)
        slope = graph.slopes.get(ctx.viewing.slope_id)
        if slope is None:
            raise ValueError(f"Slope {ctx.viewing.slope_id} must exist when panel shows slope")
        fig = chart.render_slope(slope=slope, graph=graph)
        st.plotly_chart(fig, key="slope_full_profile")

    # Full-width profile for viewing lift (panel visible with lift_id set)
    if ctx.viewing.panel_visible and ctx.viewing.lift_id:
        lift = graph.lifts.get(ctx.viewing.lift_id)
        if lift is None:
            raise ValueError(f"Lift {ctx.viewing.lift_id} must exist when panel shows lift")
        chart = ProfileChart(height=ChartConfig.LIFT_PROFILE_HEIGHT, width=ChartConfig.WIDE_WIDTH)
        fig = chart.render_lift(lift=lift, graph=graph)
        st.plotly_chart(fig, key="lift_full_profile")


if __name__ == "__main__":
    main()

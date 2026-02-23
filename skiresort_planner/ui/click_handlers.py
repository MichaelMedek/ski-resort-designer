"""Click handlers for ski resort planner.

Uses ClickDetector to detect clicks, then dispatches to state-specific handlers.
Each handler processes ClickInfo objects from the unified click detection system.

Design Principles:
- ClickDetector handles ALL detection, logging, and UI display
- One handler per state (no if-else chains)
- Handlers raise exceptions for invalid states (fail-fast)
- STRICT: Unknown/unhandled clicks raise RuntimeError immediately
"""

import logging
from typing import TYPE_CHECKING

import streamlit as st

from skiresort_planner.constants import MapConfig
from skiresort_planner.model.click_info import ClickInfo, MapClickType, MarkerType
from skiresort_planner.model.message import InvalidClickMessage, OutsideTerrainMessage
from skiresort_planner.model.path_point import PathPoint
from skiresort_planner.ui.actions import (
    bump_map_version,
    center_on_lift,
    center_on_slope,
    commit_selected_path,
)
from skiresort_planner.ui.validators import (
    validate_custom_target_distance,
    validate_custom_target_downhill,
    validate_lift_different_nodes,
    validate_lift_goes_uphill,
)

if TYPE_CHECKING:
    from skiresort_planner.core.dem_service import DEMService
    from skiresort_planner.model.resort_graph import ResortGraph
    from skiresort_planner.ui.state_machine import (
        PlannerContext,
        PlannerStateMachine,
    )

logger = logging.getLogger(__name__)


# =============================================================================
# CLICK DISPATCH
# =============================================================================


def get_click_handler(state_name: str):
    """Get the appropriate click handler for the given state.

    Args:
        state_name: Current state machine state name

    Returns:
        Handler function (click_info, elevation) -> None

    Raises:
        RuntimeError: If state has no registered handler
    """
    handlers = {
        "Idle": handle_idle_click,
        "SlopeBuilding": handle_slope_building_click,
        "LiftPlacing": handle_lift_placing_click,
    }

    handler = handlers.get(state_name)
    if handler is None:
        raise RuntimeError(
            f"No click handler registered for state '{state_name}'. "
            f"Available states: {list(handlers.keys())}. "
            f"Add handler for new state."
        )
    return handler


def dispatch_click(click_info: ClickInfo) -> None:
    """Dispatch click to appropriate state handler.

    Called by app.py after ClickDetector returns a ClickInfo.

    Args:
        click_info: Unified click information from ClickDetector

    Raises:
        RuntimeError: If state has no handler or click is outside terrain
    """
    sm: "PlannerStateMachine" = st.session_state.state_machine
    dem: "DEMService" = st.session_state.dem_service

    # Get elevation ONLY for terrain clicks (markers don't have lat/lon)
    elevation: float | None = None
    if click_info.click_type == MapClickType.TERRAIN:
        elevation = dem.get_elevation(lon=click_info.lon, lat=click_info.lat)
        if elevation is None:
            OutsideTerrainMessage(lat=click_info.lat, lon=click_info.lon).display()
            return

    state_name = sm.get_state_name()
    logger.info(f"Dispatching {click_info.display_name} in state {state_name}")

    handler = get_click_handler(state_name)
    handler(click_info=click_info, elevation=elevation)


# =============================================================================
# STATE-SPECIFIC HANDLERS
# =============================================================================


def handle_idle_click(click_info: ClickInfo, elevation: float | None) -> None:
    """Handle click in IDLE state - start building/placing or show info panel.

    Build behavior depends on ctx.build_mode.mode:
        - SLOPE: Start building ski slope
        - CHAIRLIFT/GONDOLA/etc: Start placing lift
        - None: Only view panels, no building

    Valid Click Types:
        NODE → Start building from junction (uses build_mode)
        TERRAIN → Start building at new point (uses build_mode)
        SLOPE → Show slope info panel
        SEGMENT → Show parent slope info panel
        LIFT → Show lift info panel
        PYLON → Show parent lift info panel

    Invalid Click Types:
        PROPOSAL_* → Programming error (no proposals in idle)
    """
    sm: "PlannerStateMachine" = st.session_state.state_machine
    ctx: "PlannerContext" = st.session_state.context
    graph: "ResortGraph" = st.session_state.graph

    build_mode = ctx.build_mode.mode

    # TERRAIN click → start building based on mode
    if click_info.click_type == MapClickType.TERRAIN:
        lat, lon = click_info.lat, click_info.lon

        if build_mode is None:
            # No build mode selected - ignore terrain clicks
            logger.info("[IDLE] Terrain click ignored: no build mode selected")
            InvalidClickMessage(
                action="click terrain",
                reason="Select a build mode first (slope or lift type) to start building.",
            ).display()
            return

        if ctx.build_mode.is_slope():
            # Start building slope
            logger.info(f"[IDLE] Terrain click: starting new slope at ({lat:.6f}, {lon:.6f})")
            ctx.set_selection(lon=lon, lat=lat, elevation=elevation)
            ctx.map.center = (lat, lon)
            ctx.map.zoom = MapConfig.BUILDING_ZOOM
            ctx.deferred.path_generation = True
            sm.start_building(
                lon=lon,
                lat=lat,
                elevation=elevation,
                node_id=None,
                slope_number=graph._slope_counter + 1,
            )
        elif ctx.build_mode.is_lift():
            # Start placing lift
            logger.info(f"[IDLE] Terrain click: starting {build_mode} at ({lat:.6f}, {lon:.6f})")
            ctx.map.center = (lat, lon)
            ctx.map.zoom = MapConfig.BUILDING_ZOOM
            sm.select_lift_start(
                node_id=None,
                location=PathPoint(lon=lon, lat=lat, elevation=elevation),
            )
        else:
            raise RuntimeError(f"[IDLE] Unknown build_mode '{build_mode}'. Expected SLOPE or a lift type.")
        return

    # MARKER clicks
    if click_info.click_type == MapClickType.MARKER:
        marker_type = click_info.marker_type

        # NODE → Start building from junction (uses build_mode)
        if marker_type == MarkerType.NODE:
            node = graph.nodes.get(click_info.node_id)
            if not node:
                raise RuntimeError(f"Node {click_info.node_id} not found in graph")

            if build_mode is None:
                # No build mode - ignore node clicks
                logger.info("[IDLE] Node click ignored: no build mode selected")
                InvalidClickMessage(
                    action="click junction",
                    reason="Select a build mode first (slope or lift type) to start building.",
                ).display()
                return

            if ctx.build_mode.is_slope():
                # Start building slope from node
                logger.info(f"[IDLE] Node click: starting slope from {node.id}")
                ctx.set_selection(lon=node.lon, lat=node.lat, elevation=node.elevation)
                ctx.map.center = (node.lat, node.lon)
                ctx.map.zoom = MapConfig.BUILDING_ZOOM
                ctx.deferred.path_generation = True
                sm.start_building(
                    lon=node.lon,
                    lat=node.lat,
                    elevation=node.elevation,
                    node_id=node.id,
                    slope_number=graph._slope_counter + 1,
                )
            elif ctx.build_mode.is_lift():
                # Start placing lift from node
                logger.info(f"[IDLE] Node click: starting {build_mode} from {node.id}")
                ctx.map.center = (node.lat, node.lon)
                ctx.map.zoom = MapConfig.BUILDING_ZOOM
                sm.select_lift_start(node_id=node.id)
            else:
                raise RuntimeError(f"[IDLE] Unknown build_mode '{build_mode}'. Expected SLOPE or a lift type.")
            return

        # SLOPE → Show slope panel (always works regardless of build_mode)
        if marker_type == MarkerType.SLOPE:
            slope = graph.slopes.get(click_info.slope_id)
            if not slope:
                raise RuntimeError(f"Slope {click_info.slope_id} not found in graph")
            logger.info(f"[IDLE] Slope click: showing panel for {slope.name}")
            center_on_slope(ctx=ctx, graph=graph, slope=slope, zoom=MapConfig.VIEWING_ZOOM)
            sm.show_slope_info_panel(slope_id=slope.id)
            st.rerun()  # Refresh sidebar with viewing state
            return

        # SEGMENT → Show parent slope panel
        if marker_type == MarkerType.SEGMENT:
            seg_id = click_info.segment_id
            parent_slope = graph.get_slope_by_segment_id(segment_id=seg_id)
            if not parent_slope:
                logger.info(f"[IDLE] Segment {seg_id} click: orphan segment, ignoring")
                return
            logger.info(f"[IDLE] Segment click: showing panel for {parent_slope.name}")
            center_on_slope(ctx=ctx, graph=graph, slope=parent_slope, zoom=MapConfig.VIEWING_ZOOM)
            sm.show_slope_info_panel(slope_id=parent_slope.id)
            st.rerun()  # Refresh sidebar with viewing state
            return

        # LIFT → Show lift panel and sync build mode
        if marker_type == MarkerType.LIFT:
            lift = graph.lifts.get(click_info.lift_id)
            if not lift:
                raise RuntimeError(f"Lift {click_info.lift_id} not found in graph")
            logger.info(f"[IDLE] Lift click: showing panel for {lift.name}")
            # Sync build_mode and lift.type to the viewed lift's type
            ctx.build_mode.mode = lift.lift_type
            ctx.lift.type = lift.lift_type
            center_on_lift(ctx=ctx, graph=graph, lift=lift, zoom=MapConfig.VIEWING_ZOOM)
            sm.show_lift_info_panel(lift_id=lift.id)
            st.rerun()  # Refresh sidebar with viewing state
            return

        # PYLON → Show parent lift panel and sync build mode
        if marker_type == MarkerType.PYLON:
            lift = graph.lifts.get(click_info.lift_id)
            if not lift:
                raise RuntimeError(f"Lift {click_info.lift_id} not found in graph")
            logger.info(f"[IDLE] Pylon click: showing panel for {lift.name}")
            # Sync build_mode and lift.type to the viewed lift's type
            ctx.build_mode.mode = lift.lift_type
            ctx.lift.type = lift.lift_type
            center_on_lift(ctx=ctx, graph=graph, lift=lift, zoom=MapConfig.VIEWING_ZOOM)
            sm.show_lift_info_panel(lift_id=lift.id)
            st.rerun()  # Refresh sidebar with viewing state
            return

        # PROPOSAL clicks in idle = programming error
        if marker_type in {MarkerType.PROPOSAL_ENDPOINT, MarkerType.PROPOSAL_BODY}:
            raise RuntimeError(
                "[IDLE] Proposal click detected but no proposals exist in idle state. "
                "This indicates a bug - proposal markers should not be on the map."
            )

        raise RuntimeError(f"[IDLE] Unhandled marker type {marker_type.value}. Add explicit handling.")

    raise RuntimeError(f"[IDLE] Unknown click_type {click_info.click_type}. Expected MARKER or TERRAIN.")


def handle_slope_building_click(click_info: ClickInfo, elevation: float | None) -> None:
    """Handle click in SLOPE_BUILDING state - commit path, select path, or custom connect.

    Valid Click Types:
        PROPOSAL_ENDPOINT → Commit the path
        PROPOSAL_BODY → Select the path variant (no commit)
        TERRAIN → Target for custom connect mode (if enabled)
        NODE → Target for custom connect mode (snap to node)

    Invalid Click Types (during building):
        SLOPE → Cannot view while building
        LIFT → Cannot view while building
        PYLON → Cannot view while building
    """
    ctx: "PlannerContext" = st.session_state.context

    # Custom connect mode - generate path to clicked point
    if ctx.custom_connect.enabled:
        _handle_custom_connect_click(click_info=click_info, elevation=elevation)
        return

    # TERRAIN click without custom connect = user error
    if click_info.click_type == MapClickType.TERRAIN:
        InvalidClickMessage(
            action="click terrain",
            reason='Enable "Custom Connect" to select a target point.',
        ).display()
        return

    # MARKER clicks
    if click_info.click_type == MapClickType.MARKER:
        marker_type = click_info.marker_type

        # PROPOSAL_ENDPOINT → Commit path
        if marker_type == MarkerType.PROPOSAL_ENDPOINT:
            idx = click_info.proposal_number - 1  # Convert 1-indexed to 0-indexed
            if 0 <= idx < len(ctx.proposals.paths):
                path = ctx.proposals.paths[idx]
                # Don't commit connector paths via endpoint click
                if not path.is_connector:
                    logger.info(f"[BUILDING] Proposal endpoint click: committing path {click_info.proposal_number}")
                    ctx.proposals.selected_idx = idx
                    commit_selected_path(path_idx=idx)
            return

        # PROPOSAL_BODY → Select path variant
        if marker_type == MarkerType.PROPOSAL_BODY:
            idx = click_info.proposal_number - 1  # Convert 1-indexed to 0-indexed
            if 0 <= idx < len(ctx.proposals.paths):
                logger.info(f"[BUILDING] Proposal body click: selecting path {click_info.proposal_number}")
                ctx.proposals.selected_idx = idx
                st.rerun()
            return

        # NODE without custom connect = user error
        if marker_type == MarkerType.NODE:
            InvalidClickMessage(
                action="click node",
                reason='Enable "Custom Connect" to select a target point.',
            ).display()
            return

        # SLOPE during building = user error
        if marker_type == MarkerType.SLOPE:
            InvalidClickMessage(
                action="view slope",
                reason="Finish or cancel the current slope first.",
            ).display()
            return

        # LIFT/PYLON during building = user error
        if marker_type in {MarkerType.LIFT, MarkerType.PYLON}:
            InvalidClickMessage(
                action="view lift",
                reason="Finish or cancel the current slope first.",
            ).display()
            return

        # SEGMENT during building = user error (same as SLOPE)
        if marker_type == MarkerType.SEGMENT:
            InvalidClickMessage(
                action="view segment",
                reason="Finish or cancel the current slope first.",
            ).display()
            return

        # STRICT: Unknown marker type
        raise RuntimeError(f"[BUILDING] Unhandled marker type {marker_type.value}. Add explicit handling.")

    # STRICT: Unknown click type
    raise RuntimeError(f"[BUILDING] Unknown click_type {click_info.click_type}. Expected MARKER or TERRAIN.")


def _handle_custom_connect_click(click_info: ClickInfo, elevation: float | None) -> None:
    """Handle click in custom connect mode - validate target and defer path generation."""
    ctx: "PlannerContext" = st.session_state.context
    graph: "ResortGraph" = st.session_state.graph

    # Get target coordinates - from terrain click or from node lookup
    if click_info.click_type == MapClickType.TERRAIN:
        target_lon, target_lat = click_info.lon, click_info.lat
        target_elevation = elevation
        logger.info(f"Custom connect terrain click at ({target_lat:.6f}, {target_lon:.6f})")
    elif click_info.click_type == MapClickType.MARKER and click_info.marker_type == MarkerType.NODE:
        node = graph.nodes.get(click_info.node_id)
        if not node:
            raise RuntimeError(f"Node {click_info.node_id} not found in graph")
        target_lon, target_lat = node.lon, node.lat
        target_elevation = node.elevation
        logger.info(f"Custom connect snapped to existing node {node.id}")
    else:
        # Other marker types during custom connect = user error
        InvalidClickMessage(
            action="click marker",
            reason="Click on terrain or a node to select custom connect target.",
        ).display()
        return

    if target_elevation is None:
        OutsideTerrainMessage(lat=target_lat, lon=target_lon).display()
        return

    # Get start node
    start_node = None
    if ctx.building.endpoints:
        start_node = graph.nodes.get(ctx.building.endpoints[0])
    elif ctx.custom_connect.start_node:
        start_node = graph.nodes.get(ctx.custom_connect.start_node)

    if not start_node:
        ctx.clear_custom_connect()
        raise RuntimeError(
            f"Start node not found in custom connect mode: "
            f"building.endpoints={ctx.building.endpoints}, "
            f"custom_connect.start_node={ctx.custom_connect.start_node}"
        )

    # Validate target is downhill and within range
    if error := validate_custom_target_downhill(
        start_elevation=start_node.elevation,
        target_elevation=target_elevation,
    ):
        error.display()
        return

    if error := validate_custom_target_distance(
        start_lat=start_node.lat,
        start_lon=start_node.lon,
        target_lat=target_lat,
        target_lon=target_lon,
        max_distance_m=ctx.segment_length_m,
    ):
        error.display()
        return

    logger.info(
        f"Custom connect from {start_node.id} ({start_node.elevation:.0f}m) "
        f"to ({target_lat:.6f}, {target_lon:.6f}, {target_elevation:.0f}m)"
    )

    # Store target and trigger deferred path generation
    ctx.custom_connect.target_location = (target_lon, target_lat, target_elevation)
    ctx.deferred.custom_connect = True
    st.rerun()


def handle_lift_placing_click(click_info: ClickInfo, elevation: float | None) -> None:
    """Handle click in LIFT_PLACING state - complete lift placement.

    Valid Click Types:
        NODE → Complete lift to existing node
        TERRAIN → Create new node and complete lift

    Invalid Click Types (during placement):
        SLOPE → Cannot view while placing
        LIFT → Cannot view while placing
        PYLON → Cannot view while placing
        PROPOSAL_* → No proposals in lift mode
    """
    sm: "PlannerStateMachine" = st.session_state.state_machine
    ctx: "PlannerContext" = st.session_state.context
    graph: "ResortGraph" = st.session_state.graph
    dem: "DEMService" = st.session_state.dem_service

    # Check for invalid marker clicks first
    if click_info.click_type == MapClickType.MARKER:
        marker_type = click_info.marker_type

        # SLOPE during placement = user error
        if marker_type == MarkerType.SLOPE:
            InvalidClickMessage(
                action="view slope",
                reason="Finish placing the lift first (click uphill for top station).",
            ).display()
            return

        # SEGMENT during placement = user error (same as SLOPE)
        if marker_type == MarkerType.SEGMENT:
            InvalidClickMessage(
                action="view segment",
                reason="Finish placing the lift first (click uphill for top station).",
            ).display()
            return

        # LIFT/PYLON during placement = user error
        if marker_type in {MarkerType.LIFT, MarkerType.PYLON}:
            InvalidClickMessage(
                action="view lift",
                reason="Finish placing the lift first (click uphill for top station).",
            ).display()
            return

        # PROPOSAL during lift placement = programming error (no proposals exist)
        if marker_type in {MarkerType.PROPOSAL_ENDPOINT, MarkerType.PROPOSAL_BODY}:
            raise RuntimeError(
                "[LIFT_PLACING] Proposal click detected but no proposals exist in lift mode. "
                "This indicates a bug - proposal markers should not be on the map."
            )

    # Create start node if starting from empty terrain
    if ctx.lift.start_node_id is None and ctx.lift.start_location:
        loc = ctx.lift.start_location
        start_node, _ = graph.get_or_create_node(lon=loc.lon, lat=loc.lat, elevation=loc.elevation)
        ctx.lift.start_node_id = start_node.id
        ctx.lift.start_location = None
        logger.info(f"Created start node {start_node.id} for lift at ({loc.lat:.6f}, {loc.lon:.6f})")

    # Determine end node
    end_node = None

    # NODE click → use existing node
    if click_info.click_type == MapClickType.MARKER and click_info.marker_type == MarkerType.NODE:
        end_node = graph.nodes.get(click_info.node_id)
        if end_node:
            logger.info(f"[LIFT_PLACING] Node click: completing lift to {end_node.id}")

    # TERRAIN click → create new node
    if end_node is None:
        if click_info.click_type != MapClickType.TERRAIN:
            raise RuntimeError(f"Expected TERRAIN click but got {click_info.click_type}")
        lat, lon = click_info.lat, click_info.lon
        if elevation is None:
            OutsideTerrainMessage(lat=lat, lon=lon).display()
            return
        end_node, _ = graph.get_or_create_node(lon=lon, lat=lat, elevation=elevation)
        logger.info(f"Created end node {end_node.id} for lift at ({lat:.6f}, {lon:.6f})")

    start_node = graph.nodes.get(ctx.lift.start_node_id)
    if start_node is None:
        raise RuntimeError(f"Start node {ctx.lift.start_node_id} must exist but was not found")

    # Validate lift placement
    if error := validate_lift_different_nodes(
        start_node_id=ctx.lift.start_node_id,
        end_node_id=end_node.id,
    ):
        error.display()
        return

    if error := validate_lift_goes_uphill(start_node=start_node, end_node=end_node):
        error.display()
        return

    logger.info(
        f"Creating lift from {ctx.lift.start_node_id} ({start_node.elevation:.0f}m) "
        f"to {end_node.id} ({end_node.elevation:.0f}m)"
    )

    lift = graph.add_lift(
        start_node_id=ctx.lift.start_node_id,
        end_node_id=end_node.id,
        lift_type=ctx.lift.type,
        dem=dem,
    )

    logger.info(f"Lift {lift.name} created successfully")
    center_on_lift(ctx=ctx, graph=graph, lift=lift, zoom=MapConfig.VIEWING_ZOOM)
    bump_map_version()  # Clear stale click state
    sm.complete_lift(lift_id=lift.id)

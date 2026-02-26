"""UI Actions - All action functions for ski resort planner.

Centralizes all action functions that modify UI state, trigger state
machine transitions, or perform business logic operations.

This module handles:
- Map centering (center_on_slope, center_on_lift)
- Path operations (commit_selected_path, recompute_paths)
- Slope operations (finish_current_slope, cancel_current_slope)
- Undo operations (undo_last_action)
- Custom direction mode (enter/cancel)
- Deferred action handling (handle_deferred_actions)
"""

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, cast

import streamlit as st

from skiresort_planner.constants import MapConfig, PathConfig
from skiresort_planner.generators.path_factory import PathFactory
from skiresort_planner.model.lift import Lift
from skiresort_planner.model.resort_graph import (
    ActionType,
    AddLiftAction,
    AddSegmentsAction,
    DeleteLiftAction,
    DeleteSlopeAction,
    FinishSlopeAction,
    ResortGraph,
)
from skiresort_planner.model.slope import Slope
from skiresort_planner.ui.state_machine import PlannerContext, PlannerStateMachine

if TYPE_CHECKING:
    from skiresort_planner.model.proposed_path import ProposedSlopeSegment

logger = logging.getLogger(__name__)


# =============================================================================
# MAP RELOAD ABSTRACTION
# =============================================================================


def reload_map(before: "Callable[[], None] | None" = None) -> None:
    """Reload map with optional pre-reload callback.

    This is the canonical way to reload the map. It provides a single point
    for all map reloads, making the pattern explicit and consistent.

    The flow is:
    1. Execute before callback (if provided) - runs BEFORE st.rerun()
    2. Bump map version to clear stale click state
    3. Call st.rerun() which raises StopExecution

    For actions that need to run AFTER the reload, use the deferred action
    pattern (set ctx.deferred.* flags before calling this).

    Args:
        before: Optional callback to execute before rerun.
                Use for state updates that must happen before reload.

    Example:
        # Simple reload
        reload_map()

        # Reload with pre-action
        def setup_for_reload():
            ctx.set_selection(lon=x, lat=y, elevation=e)
            ctx.deferred.path_generation = True
        reload_map(before=setup_for_reload)
    """
    if before is not None:
        before()
    bump_map_version()
    st.rerun()


# =============================================================================
# MAP VERSION HELPER
# =============================================================================


def bump_map_version() -> None:
    """Increment map_version to create fresh Pydeck component.

    This eliminates ghost clicks by creating a new component instance
    with no memory of previous click events. Call this when completing
    actions that should clear stale click state.
    """
    old_version = st.session_state.get("map_version", 0)
    new_version = old_version + 1
    st.session_state.map_version = new_version
    logger.info(f"[MAP] Bumped map_version: {old_version} -> {new_version}")


# =============================================================================
# MAP CENTERING
# =============================================================================


def center_on_slope(
    ctx: PlannerContext,
    graph: ResortGraph,
    slope: Slope,
    zoom: int,
    pitch: float = MapConfig.VIEWING_PITCH,
) -> None:
    """Center map on slope midpoint with specified zoom and pitch."""
    slope_segments = [graph.segments.get(sid) for sid in slope.segment_ids]
    if slope_segments and slope_segments[0] and slope_segments[-1]:
        first_seg, last_seg = slope_segments[0], slope_segments[-1]
        if first_seg.points and last_seg.points:
            start_pt, end_pt = first_seg.points[0], last_seg.points[-1]
            ctx.map.set_center(
                lon=(start_pt.lon + end_pt.lon) / 2,
                lat=(start_pt.lat + end_pt.lat) / 2,
            )
            ctx.map.zoom = zoom
            ctx.map.pitch = pitch


def center_on_lift(
    ctx: PlannerContext,
    graph: ResortGraph,
    lift: Lift,
    zoom: int,
    pitch: float = MapConfig.VIEWING_PITCH,
) -> None:
    """Center map on lift midpoint with specified zoom and pitch."""
    start_node = graph.nodes.get(lift.start_node_id)
    end_node = graph.nodes.get(lift.end_node_id)
    if start_node and end_node:
        ctx.map.set_center(
            lon=(start_node.lon + end_node.lon) / 2,
            lat=(start_node.lat + end_node.lat) / 2,
        )
        ctx.map.zoom = zoom
        ctx.map.pitch = pitch


# =============================================================================
# DEFERRED ACTIONS
# =============================================================================


def handle_deferred_actions() -> None:
    """Execute pending work deferred from previous state transition.

    Called at start of main() every render. Handles:
    - Auto-finish for connector paths
    - Custom connect path generation (2-stage)
    - Regular path generation

    Note: Panel view switching is now handled by explicit transitions
    (switch_to_lift_view, switch_to_slope_view, switch_slope, switch_lift)
    instead of deferred actions.
    """
    sm: PlannerStateMachine = st.session_state.state_machine
    ctx: PlannerContext = st.session_state.context

    # Handle auto-finish for connector paths (must be checked first)
    if ctx.deferred.auto_finish:
        ctx.deferred.auto_finish = False
        finish_current_slope()
        return

    # Custom connect - generate paths immediately (no two-stage delay)
    if ctx.deferred.custom_connect:
        ctx.deferred.custom_connect = False
        with st.spinner("🎯 Computing path options..."):
            _generate_custom_connect_paths()
        bump_map_version()  # Clear stale click state so proposal 1 can be clicked
        return

    if ctx.deferred.start_building_from_node_id:
        node_id = ctx.deferred.start_building_from_node_id
        ctx.deferred.start_building_from_node_id = None
        graph: ResortGraph = st.session_state.graph
        node = graph.nodes.get(node_id)
        if node and sm.is_idle:
            ctx.deferred.path_generation = True
            sm.start_building(
                lon=node.lon,
                lat=node.lat,
                elevation=node.elevation,
                node_id=node.id,
            )
        return

    if ctx.deferred.start_lift_from_node_id:
        node_id = ctx.deferred.start_lift_from_node_id
        ctx.deferred.start_lift_from_node_id = None
        if sm.is_idle:
            sm.select_lift_start(node_id=node_id)
        return

    if not ctx.deferred.path_generation:
        return

    if sm.is_any_slope_state:
        with st.spinner("🗺️ Generating path options..."):
            _generate_paths_for_building_state()
        bump_map_version()  # Clear stale click state so proposal 1 can be clicked

    ctx.deferred.path_generation = False


def _generate_paths_for_building_state() -> None:
    """Generate path proposals for current building position."""
    ctx: PlannerContext = st.session_state.context
    factory: PathFactory = st.session_state.path_factory

    if ctx.selection.lon is None or ctx.selection.lat is None or ctx.selection.elevation is None:
        logger.warning("Cannot generate paths: no current position set")
        return

    ctx.proposals.paths = list(
        factory.generate_fan(
            lon=ctx.selection.lon,
            lat=ctx.selection.lat,
            elevation=ctx.selection.elevation,
            target_length_m=PathConfig.SEGMENT_LENGTH_DEFAULT_M,
        )
    )

    # Smart recommendation: match gradient if we have a target
    if ctx.proposals.paths and ctx.deferred.gradient_target is not None:
        best_idx = _find_closest_gradient_path(
            paths=ctx.proposals.paths,
            target_gradient=ctx.deferred.gradient_target,
        )
        ctx.proposals.selected_idx = best_idx
        logger.info(
            f"Generated {len(ctx.proposals.paths)} fan paths from ({ctx.selection.lat:.6f}, {ctx.selection.lon:.6f}), "
            f"recommending path {best_idx} (closest to {ctx.deferred.gradient_target:.1f}%)"
        )
        ctx.deferred.gradient_target = None
    else:
        ctx.proposals.selected_idx = 0 if ctx.proposals.paths else None
        logger.info(
            f"Generated {len(ctx.proposals.paths)} fan paths from ({ctx.selection.lat:.6f}, {ctx.selection.lon:.6f})"
        )


def _generate_custom_connect_paths() -> None:
    """Generate paths from custom connect start to target location."""
    ctx: PlannerContext = st.session_state.context
    graph: ResortGraph = st.session_state.graph
    factory: PathFactory = st.session_state.path_factory

    if not ctx.custom_connect.target_location:
        logger.warning("No custom target location set")
        ctx.clear_custom_connect()
        return

    target_lon, target_lat, target_elevation = ctx.custom_connect.target_location

    start_node = None
    if ctx.building.endpoints:
        start_node = graph.nodes.get(ctx.building.endpoints[0])
    elif ctx.custom_connect.start_node:
        start_node = graph.nodes.get(ctx.custom_connect.start_node)

    if not start_node:
        logger.warning("Cannot find start node for custom connect")
        ctx.clear_custom_connect()
        return

    paths = list(
        factory.generate_manual_paths(
            start_lon=start_node.lon,
            start_lat=start_node.lat,
            start_elevation=start_node.elevation,
            target_lon=target_lon,
            target_lat=target_lat,
            target_elevation=target_elevation,
        )
    )

    if not paths:
        raise ValueError("generate_manual_paths should always return at least one path (fallback straight line)")

    # Always have at least one path (fallback straight line is created if needed)
    target_node = graph.find_nearest_node(
        lon=target_lon, lat=target_lat, threshold_m=MapConfig.LIFT_END_NODE_THRESHOLD_M
    )
    if target_node:
        for p in paths:
            p.is_connector = True
            p.target_node_id = target_node.id
            p.sector_name = f"🔗 {p.sector_name}"

    ctx.proposals.paths = paths
    ctx.proposals.selected_idx = 0
    # Note: enabled, force_mode, target_location already set by before_select_custom_target hook
    # No cleanup here - before_cancel_* and before_commit_* hooks handle it on exit
    logger.info(f"Generated {len(paths)} custom paths from {start_node.id} to ({target_lat:.6f}, {target_lon:.6f})")


def _find_closest_gradient_path(paths: "list[ProposedSlopeSegment]", target_gradient: float) -> int:
    """Find index of path with gradient closest to target."""
    if not paths:
        return 0
    best_idx = 0
    best_diff = float("inf")
    for i, path in enumerate(paths):
        diff = abs(path.avg_slope_pct - target_gradient)
        if diff < best_diff:
            best_diff = diff
            best_idx = i
    return best_idx


# =============================================================================
# PATH OPERATIONS
# =============================================================================


def _commit_path_transition(
    sm: PlannerStateMachine,
    segment_id: str,
    endpoint_node_id: str,
    is_connector: bool,
    slope: "Slope | None" = None,
) -> None:
    """Dispatch to appropriate state machine event for path commit.

    Uses events instead of direct transition calls - the state machine
    resolves which transition to use based on current state:
    - SlopeStarting + commit_path event → commit_first_path transition
    - SlopeBuilding + commit_path event → commit_continue_path transition
    - SlopeCustomPath: separate events for different intents (finish vs continue)

    Args:
        sm: State machine instance
        segment_id: ID of committed segment
        endpoint_node_id: ID of endpoint node
        is_connector: Whether this path connects to an existing node
        slope: Finalized slope object (required for connector from SlopeCustomPath)
    """
    if sm.is_slope_custom_path:
        # Custom path has different intents: finish (connector) vs continue
        if is_connector:
            if slope is None:
                raise RuntimeError("slope required for connector commit_custom_finish transition")
            sm.commit_custom_finish(segment_id=segment_id, slope_id=slope.id)
        else:
            sm.commit_custom_continue(segment_id=segment_id, endpoint_node_id=endpoint_node_id)
    else:
        # Use event - state machine resolves to commit_first_path or commit_continue_path
        sm.commit_path(segment_id=segment_id, endpoint_node_id=endpoint_node_id)


def commit_selected_path(path_idx: int) -> None:
    """Commit selected path to graph and trigger next path generation.

    Handles state-specific transitions:
    - From SlopeStarting/SlopeBuilding: uses commit_path
    - From SlopeCustomPath + connector: uses commit_custom_finish (direct finish)
    - From SlopeCustomPath + continue: uses commit_custom_continue
    """
    sm: PlannerStateMachine = st.session_state.state_machine
    ctx: PlannerContext = st.session_state.context
    graph: ResortGraph = st.session_state.graph

    if path_idx < 0 or path_idx >= len(ctx.proposals.paths):
        raise RuntimeError(f"Invalid path index {path_idx}, valid range: 0-{len(ctx.proposals.paths) - 1}")

    path = ctx.proposals.paths[path_idx]
    is_connector = path.is_connector and path.target_node_id

    ctx.custom_connect.clear()
    committed_gradient = path.avg_slope_pct

    end_node_ids = graph.commit_paths(paths=[path])

    if not end_node_ids:
        raise RuntimeError(f"graph.commit_paths() returned empty for path {path_idx + 1}")

    segment_id = list(graph.segments.keys())[-1]
    endpoint_node_id = end_node_ids[0]
    logger.info(
        f"Committed path {path_idx + 1} as segment {segment_id}: "
        f"{path.length_m:.0f}m, {path.avg_slope_pct:.1f}%, endpoint={endpoint_node_id}"
    )

    if is_connector:
        logger.info(f"Connector to {path.target_node_id}")
        # For connectors from SlopeCustomPath, finish the slope and transition to viewing
        if sm.is_slope_custom_path:
            # Add segment to building context before finishing
            ctx.building.segments.append(segment_id)
            # Finish the slope
            slope = graph.finish_slope(segment_ids=ctx.building.segments)
            if not slope:
                raise RuntimeError(f"graph.finish_slope() failed for connector: {ctx.building.segments}")
            logger.info(f"Slope {slope.name} (id={slope.id}) auto-finished from connector")
            center_on_slope(ctx=ctx, graph=graph, slope=slope, zoom=MapConfig.VIEWING_ZOOM)
            bump_map_version()
            sm.commit_custom_finish(segment_id=segment_id, slope_id=slope.id)
        else:
            # For other states, use commit_path
            sm.commit_path(segment_id=segment_id, endpoint_node_id=endpoint_node_id)
        return

    end_node = graph.nodes.get(endpoint_node_id)
    if end_node:
        ctx.set_selection(lon=end_node.lon, lat=end_node.lat, elevation=end_node.elevation)
        ctx.map.set_center(lon=end_node.lon, lat=end_node.lat)
        ctx.deferred.path_generation = True
        ctx.deferred.gradient_target = committed_gradient

    _commit_path_transition(
        sm=sm, segment_id=segment_id, endpoint_node_id=endpoint_node_id, is_connector=False, slope=None
    )


def recompute_paths() -> None:
    """Regenerate path proposals from current position."""
    ctx: PlannerContext = st.session_state.context
    graph: ResortGraph = st.session_state.graph
    factory: PathFactory = st.session_state.path_factory
    dem = st.session_state.dem_service

    ctx.click_dedup.pending_recompute = False
    segment_length = ctx.segment_length_m

    # Custom target mode - regenerate to stored target
    if ctx.custom_connect.force_mode and ctx.custom_connect.target_location and ctx.custom_connect.start_node:
        target_lon, target_lat, target_elevation = ctx.custom_connect.target_location
        start_node = graph.nodes.get(ctx.custom_connect.start_node)
        if start_node:
            paths = list(
                factory.generate_manual_paths(
                    start_lon=start_node.lon,
                    start_lat=start_node.lat,
                    start_elevation=start_node.elevation,
                    target_lon=target_lon,
                    target_lat=target_lat,
                    target_elevation=target_elevation,
                )
            )
            if paths:
                ctx.proposals.paths = paths
                ctx.proposals.selected_idx = 0
                logger.info(
                    f"Recomputed {len(paths)} custom paths from {start_node.id} (segment_length={segment_length}m)"
                )
                reload_map()  # Clear stale click state so proposal 1 can be clicked
            return

    # Clear custom connect when generating fan paths
    if ctx.custom_connect.enabled or ctx.custom_connect.force_mode:
        ctx.clear_custom_connect()

    if ctx.building.endpoints:
        node = graph.nodes.get(ctx.building.endpoints[0])
        if node:
            ctx.proposals.paths = list(
                factory.generate_fan(
                    lon=node.lon, lat=node.lat, elevation=node.elevation, target_length_m=segment_length
                )
            )
            ctx.proposals.selected_idx = 0 if ctx.proposals.paths else None
            logger.info(
                f"Recomputed {len(ctx.proposals.paths)} fan paths from node {node.id} (segment_length={segment_length}m)"
            )
            reload_map()  # Clear stale click state so proposal 1 can be clicked
    elif ctx.selection.has_selection():
        lon, lat = ctx.selection.get_lon_lat()
        elev = dem.get_elevation(lon=lon, lat=lat)
        if elev:
            ctx.proposals.paths = list(
                factory.generate_fan(lon=lon, lat=lat, elevation=elev, target_length_m=segment_length)
            )
            ctx.proposals.selected_idx = 0 if ctx.proposals.paths else None
            logger.info(
                f"Recomputed {len(ctx.proposals.paths)} fan paths from click (segment_length={segment_length}m)"
            )
            reload_map()  # Clear stale click state so proposal 1 can be clicked


# =============================================================================
# SLOPE OPERATIONS
# =============================================================================


def finish_current_slope() -> None:
    """Finish building and create finalized slope."""
    sm: PlannerStateMachine = st.session_state.state_machine
    ctx: PlannerContext = st.session_state.context
    graph: ResortGraph = st.session_state.graph

    if not ctx.building.segments:
        raise RuntimeError("finish_current_slope called with no segments")

    logger.info(f"Finishing slope {ctx.building.name} with {len(ctx.building.segments)} segments")
    slope = graph.finish_slope(segment_ids=ctx.building.segments)

    if not slope:
        raise RuntimeError(f"graph.finish_slope() failed for segments: {ctx.building.segments}")

    logger.info(f"Slope {slope.name} (id={slope.id}) created successfully")
    center_on_slope(ctx=ctx, graph=graph, slope=slope, zoom=MapConfig.VIEWING_ZOOM)
    bump_map_version()  # Clear stale click state
    sm.finish_slope(slope_id=slope.id)


def cancel_current_slope() -> None:
    """Cancel slope building and discard segments."""
    sm: PlannerStateMachine = st.session_state.state_machine
    ctx: PlannerContext = st.session_state.context
    graph: ResortGraph = st.session_state.graph

    logger.info(f"Canceling slope {ctx.building.name}, discarding {len(ctx.building.segments)} segments")

    start_node = graph.nodes.get(ctx.building.start_node) if ctx.building.start_node else None
    if start_node:
        ctx.map.set_building_view(lon=start_node.lon, lat=start_node.lat)

    # Clear undo entries for segments being canceled (they become invalid)
    canceled_segment_ids = set(ctx.building.segments)
    graph.undo_stack = [
        action
        for action in graph.undo_stack
        if not (
            action.action_type == ActionType.ADD_SEGMENTS
            and any(sid in canceled_segment_ids for sid in cast(AddSegmentsAction, action).segment_ids)
        )
    ]

    for seg_id in ctx.building.segments:
        if seg_id in graph.segments:
            del graph.segments[seg_id]

    graph.cleanup_isolated_nodes()  # Remove orphaned nodes from canceled building
    bump_map_version()  # Clear stale click state
    sm.cancel_slope()


def _undo_add_segments(undone: AddSegmentsAction) -> None:
    """Handle undo of ADD_SEGMENTS action.

    Handles state-specific transitions:
    - From slope building states: uses undo_segment transition
    - From idle states (e.g., after error recovery): reload map
      Note: This case happens after error recovery resets state. The graph undo
      already removed the segment. Transitioning to building mode requires more
      context than we have at this point (which slope was being built, etc.).
    """
    sm: PlannerStateMachine = st.session_state.state_machine
    ctx: PlannerContext = st.session_state.context
    graph: ResortGraph = st.session_state.graph
    factory: PathFactory = st.session_state.path_factory

    removed_segment_id = undone.segment_ids[-1] if undone.segment_ids else ""

    # Handle case where we're in idle state (e.g., after error recovery reset)
    # The graph undo already happened, just reload the map
    if sm.is_idle:
        logger.info(f"[ACTION] Undo from idle state (segment={removed_segment_id})")
        logger.warning(
            "[ACTION] Undo from idle doesn't restore building mode - click on a node or terrain to start a new slope"
        )
        reload_map()
        return

    # Check remaining segments BEFORE state machine modifies ctx.building.segments
    # The state machine hooks will remove the segment from ctx.building.segments
    remaining_segments = [s for s in ctx.building.segments if s not in undone.segment_ids]

    if remaining_segments:
        # Get the new endpoint from the last remaining segment
        last_seg = graph.segments.get(remaining_segments[-1])
        if last_seg:
            new_endpoint_node_id = last_seg.end_node_id

            # IMPORTANT: Regenerate paths BEFORE state transition!
            # The state machine triggers st.rerun() via listener, which raises StopExecution.
            # Any code after sm.undo_segment() will NOT execute.
            if last_seg.points:
                last_pt = last_seg.points[-1]
                ctx.set_selection(lon=last_pt.lon, lat=last_pt.lat, elevation=last_pt.elevation)
                ctx.proposals.paths = list(
                    factory.generate_fan(
                        lon=last_pt.lon,
                        lat=last_pt.lat,
                        elevation=last_pt.elevation,
                        target_length_m=ctx.segment_length_m,
                    )
                )
                ctx.proposals.selected_idx = 0 if ctx.proposals.paths else None
                logger.info(f"Regenerated {len(ctx.proposals.paths)} paths from previous endpoint")

            bump_map_version()
            # State machine triggers st.rerun() - code after this won't execute
            sm.undo_segment(removed_segment_id=removed_segment_id, new_endpoint_node_id=new_endpoint_node_id)
        else:
            # Segment not in graph - force state machine to handle cleanup
            logger.warning(f"Segment {remaining_segments[-1]} not found in graph during undo")
            bump_map_version()
            sm.undo_segment(removed_segment_id=removed_segment_id)
    else:
        logger.info(f"[ACTION] No segments left after undo, returning to idle (segment={removed_segment_id})")
        bump_map_version()
        logger.info("[ACTION] Calling sm.undo_segment() -> will trigger st.rerun() via listener")
        sm.undo_segment(removed_segment_id=removed_segment_id)


def _undo_finish_slope(undone: FinishSlopeAction) -> None:
    """Handle undo of FINISH_SLOPE action."""
    sm: PlannerStateMachine = st.session_state.state_machine
    ctx: PlannerContext = st.session_state.context
    graph: ResortGraph = st.session_state.graph
    factory: PathFactory = st.session_state.path_factory

    logger.info(f"Undone slope finish, restoring {len(undone.segment_ids)} segments")

    # Restore building context BEFORE state transition
    ctx.building.segments = list(undone.segment_ids)
    ctx.building.name = undone.slope_name
    ctx.building.start_node = undone.start_node_id

    if not ctx.building.segments:
        return

    last_seg = graph.segments.get(ctx.building.segments[-1])
    if not last_seg or not last_seg.points:
        return

    last_pt = last_seg.points[-1]
    # Set selection and regenerate paths BEFORE state transition
    ctx.set_selection(lon=last_pt.lon, lat=last_pt.lat, elevation=last_pt.elevation)
    ctx.building.endpoints = [last_seg.end_node_id]
    ctx.proposals.paths = list(
        factory.generate_fan(
            lon=last_pt.lon,
            lat=last_pt.lat,
            elevation=last_pt.elevation,
            target_length_m=ctx.segment_length_m,
        )
    )
    ctx.proposals.selected_idx = 0 if ctx.proposals.paths else None
    logger.info(f"Regenerated {len(ctx.proposals.paths)} paths for restored slope")

    bump_map_version()
    # State machine triggers st.rerun() - code after this won't execute
    # Uses restore_building event - SM resolves based on state + guards
    sm.restore_building()


def _undo_add_lift(undone: AddLiftAction) -> None:
    """Handle undo of ADD_LIFT action."""
    sm: PlannerStateMachine = st.session_state.state_machine

    logger.info(f"Undone lift addition: {undone.lift_id}")

    # Close panel if we were showing the deleted lift (uses close_panel event)
    if sm.is_idle_viewing_lift:
        bump_map_version()
        sm.hide_info_panel()  # Triggers st.rerun()
    else:
        reload_map()


def _undo_delete_slope(undone: DeleteSlopeAction) -> None:
    """Handle undo of DELETE_SLOPE action."""
    logger.info(f"Restored deleted slope {undone.slope_id}")
    reload_map()


def _undo_delete_lift(undone: DeleteLiftAction) -> None:
    """Handle undo of DELETE_LIFT action."""
    logger.info(f"Restored deleted lift {undone.lift_id}")
    reload_map()


def undo_last_action() -> None:
    """Undo the most recent action.

    Dispatches to type-specific handlers based on the undone action type.
    Each handler is responsible for:
    - Restoring context state (if needed)
    - Regenerating paths (if needed)
    - Triggering state machine transition or reload

    Note: Undo confirmation is handled by UI dialog before calling this function.
    """
    sm: PlannerStateMachine = st.session_state.state_machine
    ctx: PlannerContext = st.session_state.context
    graph: ResortGraph = st.session_state.graph

    # Guard: nothing to undo (should not happen - UI disables button when empty)
    if not graph.undo_stack:
        return

    logger.info(f"[ACTION] Undo requested, state={sm.get_state_name()}, undo_stack_size={len(graph.undo_stack)}")

    # Special case: in slope state with no segments → cancel slope
    if sm.is_any_slope_state and not ctx.building.segments:
        logger.info("[ACTION] No segments in building state, canceling slope via undo")
        sm.cancel_slope()
        return

    undone = graph.undo_last()
    action_type = undone.action_type
    logger.info(f"[ACTION] Undone: {action_type.name}")

    # Dispatch to type-specific handlers.
    # IMPORTANT: We compare by .name (string) instead of direct enum equality because
    # Streamlit's module reloading creates NEW enum class instances on each rerun.
    # Objects in st.session_state.graph.undo_stack hold references to the OLD enum
    # values, which fail `==` comparison against the NEW enum class values.
    # Using .name ensures stable comparison across module reloads.
    if action_type.name == ActionType.ADD_SEGMENTS.name:
        _undo_add_segments(undone=cast(AddSegmentsAction, undone))
    elif action_type.name == ActionType.FINISH_SLOPE.name:
        _undo_finish_slope(undone=cast(FinishSlopeAction, undone))
    elif action_type.name == ActionType.ADD_LIFT.name:
        _undo_add_lift(undone=cast(AddLiftAction, undone))
    elif action_type.name == ActionType.DELETE_SLOPE.name:
        _undo_delete_slope(undone=cast(DeleteSlopeAction, undone))
    elif action_type.name == ActionType.DELETE_LIFT.name:
        _undo_delete_lift(undone=cast(DeleteLiftAction, undone))
    else:
        raise RuntimeError(f"Unknown action type: {action_type}")


# =============================================================================
# CUSTOM DIRECTION MODE
# =============================================================================


def enter_custom_direction_mode() -> None:
    """Enter custom direction mode via state machine transition.

    Triggers enable_custom_from_starting or enable_custom_from_building
    depending on current state. All context setup is handled by before_* hooks.
    """
    sm: PlannerStateMachine = st.session_state.state_machine
    logger.info("[ACTION] Custom Direction button clicked - triggering state transition")
    sm.enable_custom_connect()


def cancel_custom_direction_mode() -> None:
    """Cancel custom direction mode via state machine transition.

    Triggers cancel_custom_to_starting or cancel_custom_to_building.
    Path regeneration is triggered by before_* hooks.
    """
    sm: PlannerStateMachine = st.session_state.state_machine
    logger.info("[ACTION] Cancel Custom Direction - triggering state transition")
    sm.cancel_custom_connect()


def cancel_custom_path() -> None:
    """Cancel custom path mode (from SLOPE_CUSTOM_PATH) via state machine transition.

    Triggers cancel_path_to_starting or cancel_path_to_building.
    Path regeneration is triggered by before_* hooks.
    """
    sm: PlannerStateMachine = st.session_state.state_machine
    logger.info("[ACTION] Cancel Custom Path - triggering state transition")
    sm.cancel_custom_connect()

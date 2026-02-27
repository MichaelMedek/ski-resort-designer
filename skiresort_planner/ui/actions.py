"""UI Actions - All action functions for ski resort planner.

Centralizes all action functions that modify UI state, trigger state
machine transitions, or perform business logic operations.

This module handles:
- Map centering (center_on_slope, center_on_lift)
- Path operations (commit_selected_path, recompute_paths)
- Slope operations (finish_current_slope, cancel_current_slope)
- Undo operations (undo_last_action)
- Custom direction mode (enter/cancel)
- Deferred action handling (handle_fast_deferred_actions, process_*_deferred)
"""

import logging
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
from skiresort_planner.ui.infra import bump_map_version, reload_map, trigger_rerun
from skiresort_planner.ui.state_machine import PlannerContext, PlannerStateMachine

if TYPE_CHECKING:
    from skiresort_planner.model.proposed_path import ProposedSlopeSegment

logger = logging.getLogger(__name__)


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


def handle_fast_deferred_actions() -> None:
    """Execute fast deferred actions that don't need spinners.

    Called at start of main() for quick state transitions:
    - Auto-finish for connector paths
    - Start building from node (triggers path generation)
    - Start lift from node

    NOTE: Slow operations (custom_connect, path_generation) are handled
    separately in app.py with spinners around process_*_deferred() calls.
    """
    sm: PlannerStateMachine = st.session_state.state_machine
    ctx: PlannerContext = st.session_state.context

    # Handle auto-finish for connector paths (must be checked first)
    if ctx.deferred.auto_finish:
        ctx.deferred.auto_finish = False
        finish_current_slope()
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


def process_custom_connect_deferred() -> bool:
    """Process pending custom connect path generation.

    Call this wrapped in st.spinner() from app.py.

    Returns:
        True if processed, False if nothing pending.
    """
    ctx: PlannerContext = st.session_state.context

    if not ctx.deferred.custom_connect:
        return False

    ctx.deferred.custom_connect = False
    _generate_custom_connect_paths()
    bump_map_version()  # Clear stale click state so proposal 1 can be clicked
    return True


def process_path_generation_deferred() -> bool:
    """Process pending path generation.

    Call this wrapped in st.spinner() from app.py.

    Returns:
        True if processed, False if nothing pending.
    """
    sm: PlannerStateMachine = st.session_state.state_machine
    ctx: PlannerContext = st.session_state.context

    if not ctx.deferred.path_generation:
        return False

    if sm.is_any_slope_state:
        _generate_paths_for_building_state()
        bump_map_version()  # Clear stale click state so proposal 1 can be clicked

    ctx.deferred.path_generation = False
    return True


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

    Uses force_idle/force_building instead of state machine transitions.
    This follows the expert recommendation to treat undo as history management,
    not core workflow state transitions.
    """
    sm: PlannerStateMachine = st.session_state.state_machine
    ctx: PlannerContext = st.session_state.context
    graph: ResortGraph = st.session_state.graph
    factory: PathFactory = st.session_state.path_factory

    removed_segment_id = undone.segment_ids[-1] if undone.segment_ids else ""

    # Calculate remaining segments after undo
    remaining_segments = [s for s in ctx.building.segments if s not in undone.segment_ids]

    # Step 3: Update context and force appropriate state
    if remaining_segments:
        # Update building context
        ctx.building.segments = remaining_segments

        # Get the new endpoint from the last remaining segment
        last_seg = graph.segments.get(remaining_segments[-1])
        if last_seg:
            ctx.building.endpoints = [last_seg.end_node_id]

            # Regenerate paths from new endpoint
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

        # Force to building state (exit hooks handle cleanup automatically)
        logger.info(f"[ACTION] Undo leaves {len(remaining_segments)} segments, forcing SlopeBuilding")
        sm.force_building()
    else:
        # No segments left - return to idle
        logger.info(f"[ACTION] No segments left after undo (segment={removed_segment_id}), forcing IdleReady")
        sm.force_idle()

    bump_map_version()
    trigger_rerun()


def _undo_finish_slope(undone: FinishSlopeAction) -> None:
    """Handle undo of FINISH_SLOPE action.

    Uses force_building instead of restore_building event.
    This allows undo from any state (including LiftPlacing).
    """
    sm: PlannerStateMachine = st.session_state.state_machine
    ctx: PlannerContext = st.session_state.context
    graph: ResortGraph = st.session_state.graph
    factory: PathFactory = st.session_state.path_factory

    logger.info(f"Undone slope finish, restoring {len(undone.segment_ids)} segments")

    # Restore building context
    ctx.building.segments = list(undone.segment_ids)
    ctx.building.name = undone.slope_name
    ctx.building.start_node = undone.start_node_id

    if not ctx.building.segments:
        # Edge case: finished slope had no segments (shouldn't happen)
        sm.force_idle()
        bump_map_version()
        trigger_rerun()
        return

    last_seg = graph.segments.get(ctx.building.segments[-1])
    if not last_seg or not last_seg.points:
        # Segment data missing - go to idle
        sm.force_idle()
        bump_map_version()
        trigger_rerun()
        return

    # Set selection and regenerate paths
    last_pt = last_seg.points[-1]
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

    # Force to building state (exit hooks handle cleanup automatically)
    sm.force_building()
    bump_map_version()
    trigger_rerun()


def _undo_add_lift(undone: AddLiftAction) -> None:
    """Handle undo of ADD_LIFT action."""
    sm: PlannerStateMachine = st.session_state.state_machine

    logger.info(f"Undone lift addition: {undone.lift_id}")

    # If in LiftPlacing state, force to idle (lift placement context is now stale)
    if sm.is_lift_placing:
        sm.force_idle()
        bump_map_version()
        trigger_rerun()
        return

    # If we were viewing the deleted lift, force to idle (exit hooks handle cleanup)
    if sm.is_idle_viewing_lift and st.session_state.context.viewing.lift_id == undone.lift_id:
        sm.force_idle()
        bump_map_version()
        trigger_rerun()
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


# =============================================================================
# DELETE OPERATIONS
# =============================================================================


def delete_slope_action(slope_id: str) -> bool:
    """Delete a slope and trigger UI updates.

    This is the canonical function to delete a slope. It handles:
    - Graph deletion (with undo support)
    - State machine transition if viewing the deleted slope
    - Map version bump for UI refresh

    Args:
        slope_id: ID of slope to delete

    Returns:
        True if deleted, False if not found.
    """
    sm: PlannerStateMachine = st.session_state.state_machine
    ctx: PlannerContext = st.session_state.context
    graph: ResortGraph = st.session_state.graph

    result = graph.delete_slope(slope_id=slope_id)
    if not result:
        logger.warning(f"[ACTION] delete_slope_action: slope {slope_id} not found")
        return False

    logger.info(f"[ACTION] Deleted slope {slope_id}")

    # If viewing the deleted slope, return to idle
    if sm.is_idle_viewing_slope and ctx.viewing.slope_id == slope_id:
        sm.close_panel()

    bump_map_version()
    return True


def delete_lift_action(lift_id: str) -> bool:
    """Delete a lift and trigger UI updates.

    This is the canonical function to delete a lift. It handles:
    - Graph deletion (with undo support)
    - State machine transition if viewing the deleted lift
    - Map version bump for UI refresh

    Args:
        lift_id: ID of lift to delete

    Returns:
        True if deleted, False if not found.
    """
    sm: PlannerStateMachine = st.session_state.state_machine
    ctx: PlannerContext = st.session_state.context
    graph: ResortGraph = st.session_state.graph

    result = graph.delete_lift(lift_id=lift_id)
    if not result:
        logger.warning(f"[ACTION] delete_lift_action: lift {lift_id} not found")
        return False

    logger.info(f"[ACTION] Deleted lift {lift_id}")

    # If viewing the deleted lift, return to idle
    if sm.is_idle_viewing_lift and ctx.viewing.lift_id == lift_id:
        sm.close_panel()

    bump_map_version()
    return True

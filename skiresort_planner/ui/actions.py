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
- Deferred toast messages (survive st.rerun)
"""

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, cast

import streamlit as st

from skiresort_planner.constants import MapConfig, PathConfig
from skiresort_planner.generators.path_factory import PathFactory
from skiresort_planner.model.lift import Lift
from skiresort_planner.model.message import (
    ToastMessage,
    UndoCancelSlopeMessage,
    UndoDeleteLiftMessage,
    UndoDeleteSlopeMessage,
    UndoFinishSlopeMessage,
    UndoLiftMessage,
    UndoSegmentMessage,
)
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
    st.session_state.map_version = st.session_state.get("map_version", 0) + 1


# =============================================================================
# DEFERRED TOAST MESSAGES
# =============================================================================


def queue_toast(message: str, icon: str = "ℹ️") -> None:
    """Queue a toast message to display after the next st.rerun().

    Toast messages are transient notifications that appear briefly. However,
    st.rerun() raises StopExecution and prevents st.toast() from being shown
    if called before rerun. This function stores toasts in session state
    to be displayed on the next run.

    Args:
        message: Toast message text
        icon: Icon to show (emoji)
    """
    if "pending_toasts" not in st.session_state:
        st.session_state.pending_toasts = []
    st.session_state.pending_toasts.append({"message": message, "icon": icon})
    logger.info(f"[TOAST QUEUED] {icon} {message}")


def display_pending_toasts() -> None:
    """Display and clear any pending toast messages.

    Call this at the start of each app run to show toasts queued
    from the previous run (before st.rerun was called).
    """
    pending = st.session_state.get("pending_toasts", [])
    for toast in pending:
        st.toast(f"{toast['icon']} {toast['message']}")
    st.session_state.pending_toasts = []


def _queue_toast_from_message(msg: ToastMessage) -> None:
    """Queue a ToastMessage for display after st.rerun()."""
    queue_toast(message=msg.message, icon=msg.icon)


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
    ctx.custom_connect.enabled = False
    ctx.custom_connect.force_mode = True
    ctx.custom_connect.target_location = None
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


def commit_selected_path(path_idx: int) -> None:
    """Commit selected path to graph and trigger next path generation."""
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
        logger.info(f"Connector to {path.target_node_id}, auto-finish enabled")
        ctx.deferred.auto_finish = True
        sm.commit_path(segment_id=segment_id, endpoint_node_id=endpoint_node_id)
        return

    end_node = graph.nodes.get(endpoint_node_id)
    if end_node:
        ctx.set_selection(lon=end_node.lon, lat=end_node.lat, elevation=end_node.elevation)
        ctx.map.set_center(lon=end_node.lon, lat=end_node.lat)
        ctx.deferred.path_generation = True
        ctx.deferred.gradient_target = committed_gradient

    sm.commit_path(segment_id=segment_id, endpoint_node_id=endpoint_node_id)


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
                bump_map_version()  # Clear stale click state so proposal 1 can be clicked
                st.rerun()
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
            bump_map_version()  # Clear stale click state so proposal 1 can be clicked
            st.rerun()
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
            bump_map_version()  # Clear stale click state so proposal 1 can be clicked
            st.rerun()


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

    graph.cleanup_isolated_nodes()  # Remove orphaned nodes
    bump_map_version()  # Clear stale click state
    sm.cancel_slope()


def undo_last_action() -> None:
    """Undo the most recent action."""
    sm: PlannerStateMachine = st.session_state.state_machine
    ctx: PlannerContext = st.session_state.context
    graph: ResortGraph = st.session_state.graph
    factory: PathFactory = st.session_state.path_factory

    logger.info(f"Undo requested, state={sm.get_state_name()}, undo_stack_size={len(graph.undo_stack)}")

    if sm.is_any_slope_state and not ctx.building.segments:
        logger.info("No segments in building state, canceling slope via undo")
        # Queue toast BEFORE state transition (sm.cancel_slope triggers st.rerun)
        _queue_toast_from_message(UndoCancelSlopeMessage())
        sm.cancel_slope()
        # NOTE: Code here won't execute - st.rerun() is called by state machine listener
        return

    undone = graph.undo_last()
    # undo_last() raises RuntimeError if stack is empty, so no None check needed

    # Use action_type property (enum) for reliable dispatch across module reloads
    action_type = undone.action_type
    logger.info(f"Undone action: {action_type.name}")

    if action_type == ActionType.ADD_SEGMENTS:
        add_seg = cast(AddSegmentsAction, undone)
        removed_segment_id = add_seg.segment_ids[-1] if add_seg.segment_ids else ""

        # Check remaining segments BEFORE state machine modifies ctx.building.segments
        # The state machine hooks will remove the segment from ctx.building.segments
        remaining_segments = [s for s in ctx.building.segments if s not in add_seg.segment_ids]

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

                # Queue toast BEFORE state transition (st.rerun prevents direct display)
                _queue_toast_from_message(UndoSegmentMessage(segment_id=removed_segment_id, was_last_segment=False))

                bump_map_version()
                # State machine triggers st.rerun() - code after this won't execute
                sm.undo_segment(removed_segment_id=removed_segment_id, new_endpoint_node_id=new_endpoint_node_id)
            else:
                # Segment not in graph - force state machine to handle cleanup
                logger.warning(f"Segment {remaining_segments[-1]} not found in graph during undo")
                _queue_toast_from_message(UndoSegmentMessage(segment_id=removed_segment_id, was_last_segment=False))
                bump_map_version()
                sm.undo_segment(removed_segment_id=removed_segment_id)
        else:
            logger.info("No segments left after undo, returning to idle")
            _queue_toast_from_message(UndoSegmentMessage(segment_id=removed_segment_id, was_last_segment=True))
            bump_map_version()
            sm.undo_segment(removed_segment_id=removed_segment_id)
        # NOTE: Code here won't execute - st.rerun() is called by state machine listener

    elif action_type == ActionType.FINISH_SLOPE:
        finish_slope = cast(FinishSlopeAction, undone)
        logger.info(f"Undone slope finish, restoring {len(finish_slope.segment_ids)} segments")

        # Restore building context BEFORE state transition
        ctx.building.segments = list(finish_slope.segment_ids)
        ctx.building.name = finish_slope.slope_name
        ctx.building.start_node = finish_slope.start_node_id

        if ctx.building.segments:
            last_seg = graph.segments.get(ctx.building.segments[-1])
            if last_seg and last_seg.points:
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

                # Queue toast BEFORE state transition
                _queue_toast_from_message(
                    UndoFinishSlopeMessage(
                        slope_name=finish_slope.slope_name or "Unnamed",
                        num_segments=len(finish_slope.segment_ids),
                    )
                )

                bump_map_version()
                # State machine triggers st.rerun() - code after this won't execute
                if sm.is_idle_viewing_slope:
                    sm.resume_building()
                elif sm.is_idle_ready:
                    sm.undo_restore_to_building()
                elif sm.is_idle_viewing_lift:
                    sm.undo_restore_from_lift_to_building()
                else:
                    raise RuntimeError(f"Cannot undo FINISH_SLOPE from state {sm.get_state_name()}")
        # NOTE: Code here won't execute - st.rerun() is called by state machine listener

    elif action_type == ActionType.ADD_LIFT:
        add_lift = cast(AddLiftAction, undone)
        # The lift was already removed by graph.undo_last(), so we can't get its name
        # Use the lift_id for the toast message
        logger.info(f"Undone lift addition: {add_lift.lift_id}")
        # Queue toast BEFORE state transition
        _queue_toast_from_message(UndoLiftMessage(lift_name=add_lift.lift_id))
        bump_map_version()
        # Hide panel if we were showing the deleted lift
        if sm.is_idle_viewing_lift:
            sm.hide_info_panel()  # Triggers st.rerun()
        else:
            st.rerun()

    elif action_type == ActionType.DELETE_SLOPE:
        del_slope = cast(DeleteSlopeAction, undone)
        logger.info(f"Restored deleted slope {del_slope.slope_id}")
        # Get restored slope name from graph
        restored_slope = graph.slopes.get(del_slope.slope_id)
        slope_name = restored_slope.name if restored_slope else del_slope.slope_id
        _queue_toast_from_message(UndoDeleteSlopeMessage(slope_name=slope_name))
        bump_map_version()
        st.rerun()

    elif action_type == ActionType.DELETE_LIFT:
        del_lift = cast(DeleteLiftAction, undone)
        logger.info(f"Restored deleted lift {del_lift.lift_id}")
        # Get restored lift name from graph
        restored_lift = graph.lifts.get(del_lift.lift_id)
        lift_name = restored_lift.name if restored_lift else del_lift.lift_id
        _queue_toast_from_message(UndoDeleteLiftMessage(lift_name=lift_name))
        bump_map_version()
        st.rerun()


# =============================================================================
# CUSTOM DIRECTION MODE
# =============================================================================


def enter_custom_direction_mode() -> None:
    """Enter custom direction mode - backup proposals and wait for click."""
    ctx: PlannerContext = st.session_state.context
    graph: ResortGraph = st.session_state.graph

    start_node_id = None
    if ctx.building.endpoints:
        start_node_id = ctx.building.endpoints[0]
    elif ctx.building.start_node:
        start_node_id = ctx.building.start_node
    elif ctx.proposals.paths:
        first_path = ctx.proposals.paths[0]
        if first_path.points:
            start_pt = first_path.points[0]
            existing = graph.find_nearest_node(
                lon=start_pt.lon, lat=start_pt.lat, threshold_m=MapConfig.LIFT_END_NODE_THRESHOLD_M
            )
            if existing:
                start_node_id = existing.id
            else:
                new_node, _ = graph.get_or_create_node(lon=start_pt.lon, lat=start_pt.lat, elevation=start_pt.elevation)
                start_node_id = new_node.id

    if not start_node_id:
        logger.warning("Cannot enter custom direction - no start node")
        return

    logger.info(f"Entering custom direction mode from node {start_node_id}")
    ctx.custom_connect.start_node = start_node_id
    ctx.custom_connect.enabled = True
    ctx.proposals.clear()
    st.rerun()


def cancel_custom_direction_mode() -> None:
    """Cancel custom direction mode and recompute fan paths."""
    ctx: PlannerContext = st.session_state.context

    logger.info("Cancelling custom direction mode, will recompute fan paths")
    ctx.clear_custom_connect()
    ctx.proposals.clear()
    ctx.deferred.path_generation = True
    st.rerun()


def cancel_connection_mode() -> None:
    """Cancel force connection mode and recompute fan paths."""
    ctx: PlannerContext = st.session_state.context

    logger.info("Cancelling connection mode, will recompute fan paths")
    ctx.custom_connect.force_mode = False
    ctx.proposals.clear()
    ctx.deferred.path_generation = True
    st.rerun()

"""State Lifecycle Functions - Entry and exit handlers for state machine states.

This module provides the enter_* handlers (one per state) plus the three exit_* handlers for the
states with real teardown (lift/import/merge). States with no teardown have NO exit handler — they
exit as no-ops (no on_exit_* hook is wired), so cleanup can't accidentally break undo/self-loops.

Architecture:
- enter_* is called by the state machine's on_enter_* hook; the 3 exit_* by on_exit_lift_placing
  and by the force/undo path via EXIT_HOOKS.
- Functions receive the PlannerContext to modify UI state; they are idempotent.

Usage in state machine:
    def on_enter_idle_ready(self) -> None:
        enter_idle_ready(self.context)

State Definitions:
    1. IDLE_READY: No panel visible, ready to start building
    2. IDLE_VIEWING_SLOPE: Panel showing slope details, profile visible, 3D available
    3. IDLE_VIEWING_LIFT: Panel showing lift details, profile visible, 3D available
    4. IDLE_VIEWING_ROAD: Panel showing road details, profile visible, 3D available
    5. SLOPE_STARTING: 0 segments committed, picking first fan direction
    6. SLOPE_BUILDING: 1+ segments committed, continuing slope picking next fan direction
    7. SLOPE_CUSTOM_PATH: Showing custom path options routed to a clicked target
    8. LIFT_PLACING: Start selected, waiting for end station
    9. ROAD_STARTING: 0 road segments committed, picking first target
    10. ROAD_BUILDING: 1+ road segments committed, extending the road

Design Philosophy:
    → NO workflow mutations here (force_mode, target_location, proposals, etc.)
    → All state cleanup lives in before_* transition hooks in state_machine.py
    → Single source of truth principle
    → Lifecycle functions handle UI & context side-effects only
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from skiresort_planner.model.path_segment import SegmentKind

if TYPE_CHECKING:
    from collections.abc import Callable

    from skiresort_planner.ui.context import PlannerContext

logger = logging.getLogger(__name__)


# =============================================================================
# Shared enter bodies — the *_starting / *_building / *_custom_path / idle_viewing
# groups each collapse to one of these, so slope and road (and future kinds) cannot drift.
# =============================================================================


def _enter_fan_state(ctx: PlannerContext, kind: SegmentKind) -> None:
    """Enter a fan state (*_starting or *_building): hide the panel, free the last-clicked node
    marker so it can be re-clicked, and arm the fan for this kind. Same for starting and building —
    building just keeps its already-committed segments in the build context.
    """
    ctx.viewing.hide_panel()
    ctx.click_dedup.clear_marker()
    ctx.deferred.fan_generation.add(kind)


def _enter_custom_path(ctx: PlannerContext) -> None:
    """Enter a *_custom_path state: free the last-clicked node so re-clicking retargets, and flag
    deferred custom-connect generation to the stored target. Kind-agnostic.
    """
    ctx.click_dedup.clear_marker()
    ctx.deferred.custom_connect = True


def _enter_viewing_panel(ctx: PlannerContext) -> None:
    """Enter an idle_viewing_* state: show the info panel and clear all stale build/placement state.
    Identical for slope/lift/road — the before-hook already recorded which entity to view.
    """
    ctx.viewing.show_panel()
    ctx.clear_proposals()
    ctx.clear_builds()
    ctx.clear_custom_connect()
    ctx.clear_lift()
    ctx.selection.clear()
    ctx.click_dedup.clear_marker()


# =============================================================================
# 1. IDLE_READY - No panel visible, ready to start building
# =============================================================================


def enter_idle_ready(ctx: PlannerContext) -> None:
    """Enter IDLE_READY: Clear all building state and hide panels.

    What needs to be cleared:
    - All path proposals (no proposals should be visible)
    - All building context (segments, start node, name)
    - All custom connect state
    - All lift placement state
    - Node selection marker (click dedup)
    - Viewing state (panel should be hidden)

    What should NOT be touched:
    - Map center/zoom (preserve user's view position)
    - Build mode selection (user's preference)
    - Segment length setting (user's preference)

    End state: Clean slate ready for any action (view slope/lift, start building)
    """
    logger.debug("[LIFECYCLE] ENTER: idle_ready - clearing all building state")
    ctx.clear_proposals()
    ctx.clear_builds()
    ctx.clear_custom_connect()
    ctx.clear_lift()
    ctx.selection.clear()
    ctx.click_dedup.clear_marker()
    ctx.viewing.clear()
    logger.info(
        f"[LIFECYCLE] idle_ready complete: map_center=({ctx.map.lat:.4f}, {ctx.map.lon:.4f}), zoom={ctx.map.zoom}"
    )


# =============================================================================
# 2. IDLE_VIEWING_SLOPE - Panel showing slope details
# =============================================================================


def enter_idle_viewing_slope(ctx: PlannerContext) -> None:
    """Enter IDLE_VIEWING_SLOPE: Make slope panel visible (Single Point of Truth).

    SINGLE POINT OF TRUTH PRINCIPLE:
    This function GUARANTEES panel is visible, regardless of which transition
    brought us here (view_slope, switch_slope, undo_finish_slope, etc.).

    Prior to this (in before_* hooks):
    - ctx.viewing.set_slope_id(slope_id) stored WHICH slope to view
    - Map centering may have been triggered

    This function is responsible for:
    - Making panel visible via show_panel()
    - Clearing all building/placement state (defensive cleanup)

    End state: Panel visible showing slope details
    """
    logger.debug("ENTER: idle_viewing_slope - showing panel, clearing building state")
    _enter_viewing_panel(ctx)


# =============================================================================
# 3. IDLE_VIEWING_LIFT - Panel showing lift details
# =============================================================================


def enter_idle_viewing_lift(ctx: PlannerContext) -> None:
    """Enter IDLE_VIEWING_LIFT: Make lift panel visible (Single Point of Truth).

    SINGLE POINT OF TRUTH PRINCIPLE:
    This function GUARANTEES panel is visible, regardless of which transition
    brought us here (view_lift, complete_lift, switch_lift, etc.).

    Prior to this (in before_* hooks):
    - ctx.viewing.set_lift_id(lift_id) stored WHICH lift to view
    - Map centering may have been triggered

    This function is responsible for:
    - Making panel visible via show_panel()
    - Clearing all building/placement state (defensive cleanup)

    End state: Panel visible showing lift details
    """
    logger.debug("ENTER: idle_viewing_lift - showing panel, clearing building state")
    _enter_viewing_panel(ctx)


# =============================================================================
# 4. SLOPE_STARTING - 0 segments committed, picking first direction
# =============================================================================


def enter_slope_starting(ctx: PlannerContext) -> None:
    """Enter SLOPE_STARTING: Begin slope building (Single Point of Truth).

    SINGLE POINT OF TRUTH PRINCIPLE:
    This function GUARANTEES panel is hidden and map is in building mode,
    regardless of which transition brought us here.

    Prior to this (in before_* hooks):
    - selection is set with start point (lon, lat, elevation)
    - building.start_node is set if starting from existing node
    - building.name is assigned (e.g., "Slope 5")
    - deferred.fan_generation gains this kind to trigger path generation

    This function is responsible for:
    - Hiding any viewing panel
    - Clearing click dedup marker for fresh clicks

    End state: Panel hidden, ready for path proposals
    """
    logger.debug("ENTER: slope_starting - hiding panel, clearing marker dedup, arming slope fan")
    _enter_fan_state(ctx, SegmentKind.SLOPE)


# =============================================================================
# 5. SLOPE_BUILDING - 1+ segments committed, continuing slope
# =============================================================================


def enter_slope_building(ctx: PlannerContext) -> None:
    """Enter SLOPE_BUILDING: Continue building slope (Single Point of Truth).

    SINGLE POINT OF TRUTH PRINCIPLE:
    This function GUARANTEES panel is hidden and we're in building mode,
    regardless of which transition brought us here (commit_path event,
    undo event, resume_to_building, cancel_custom event).

    Sources:
    - From SLOPE_STARTING: First path committed via commit_path event
    - From SLOPE_CUSTOM_PATH: Custom path committed (commit_custom_continue), or
      canceled back to building when segments exist (cancel_custom → cancel_path_to_building)
    - From SLOPE_BUILDING: Self-loop (more segments committed, undo)
    - From IDLE_VIEWING_SLOPE: Resume building (undo finish)

    This function is responsible for:
    - Hiding any viewing panel
    - Preserving building context (has committed segments!)
    - Preserving proposals (may be in use or being generated)

    End state: Panel hidden, continuing to build
    """
    logger.debug("ENTER: slope_building - hiding panel, preserving building context, arming slope fan")
    _enter_fan_state(ctx, SegmentKind.SLOPE)


# =============================================================================
# 7. SLOPE_CUSTOM_PATH - Showing custom path options
# =============================================================================


def enter_slope_custom_path(ctx: PlannerContext) -> None:
    """Enter SLOPE_CUSTOM_PATH: Show path options to custom target.

    Sources:
    - From SLOPE_STARTING / SLOPE_BUILDING: terrain/node clicked as target
    - From SLOPE_CUSTOM_PATH (self-loop): a new target clicked → re-route

    What happens:
    - The target before-hook set start_node, target_location and force_mode
    - This function sets deferred.custom_connect to trigger path generation
    - On next render, deferred handler generates proposals with spinner

    Fires on the retarget self-loop too (external self-transition), so a new
    target click regenerates proposals here.

    End state: Path proposals shown from start to custom target
    """
    logger.debug("ENTER: slope_custom_path - clearing marker, triggering deferred path generation")
    _enter_custom_path(ctx)


# =============================================================================
# 8. LIFT_PLACING - Start selected, waiting for end station
# =============================================================================


def enter_lift_placing(ctx: PlannerContext) -> None:
    """Enter LIFT_PLACING: First lift station selected (Single Point of Truth).

    SINGLE POINT OF TRUTH PRINCIPLE:
    This function GUARANTEES panel is hidden and we're in placement mode,
    regardless of which transition brought us here.

    Sources:
    - From IDLE_READY: Click node/terrain in lift mode
    - From IDLE_VIEWING_SLOPE: Click node/terrain in lift mode
    - From IDLE_VIEWING_LIFT: Click node/terrain in lift mode

    Prior to this (in before_* hooks):
    - lift.start_node_id is set (or lift.start_location for new node)
    - lift.type is set based on build_mode

    This function is responsible for:
    - Hiding any viewing panel
    - Clearing click dedup marker for fresh clicks

    End state: Panel hidden, ready for end station click
    """
    logger.debug("ENTER: lift_placing - hiding panel")
    # SINGLE POINT OF TRUTH: Hide panel for placement mode
    ctx.viewing.hide_panel()
    ctx.click_dedup.clear_marker()


def exit_lift_placing(ctx: PlannerContext) -> None:
    """Exit LIFT_PLACING: Clean up lift placement state.

    Possible destinations:
    - IDLE_VIEWING_LIFT: Lift completed successfully
    - IDLE_READY: Cancel pressed

    What needs to be cleared:
    - lift context (start_node_id, start_location)

    Note: before_complete_lift and before_cancel_lift handle showing/hiding panel.
    The lift context should be cleared since placement is done.
    """
    logger.debug("EXIT: lift_placing - clearing lift context")
    ctx.lift.clear()


def exit_import_placing(ctx: PlannerContext) -> None:
    """Exit IMPORT_PLACING: clear the placed import-box center.

    Destinations: IDLE_READY (cancel or confirm). The placed box center lives in
    ctx.deferred.osm_import_center_lon/lat; clearing it here guarantees no stale box survives the
    exit no matter which transition (or a force_* during undo) leaves the state. The osm_import
    fetch flag is deliberately left alone — a confirmed import sets it just before this runs and
    consumes it in process_osm_import_deferred.
    """
    logger.debug("EXIT: import_placing - clearing placed import-box center")
    ctx.deferred.osm_import_center_lon = None
    ctx.deferred.osm_import_center_lat = None


def exit_merge_placing(ctx: PlannerContext) -> None:
    """Exit MERGE_PLACING: clear the node-merge selection.

    Destinations: IDLE_READY (cancel or confirm). The selected node ids live in ctx.merge; clearing
    here guarantees no stale selection survives the exit regardless of transition (or a force_*
    during undo).
    """
    logger.debug("EXIT: merge_placing - clearing merge selection")
    ctx.merge.clear()


def enter_import_placing(ctx: PlannerContext) -> None:
    """Enter IMPORT_PLACING: an import box center has been placed (Single Point of Truth).

    Reached from any idle state on the first map click in import mode, and re-entered on
    each retarget (self-loop) when the user clicks a new center. The placed center lives in
    ctx.deferred.osm_import_center_lon/lat (set by before_start_import) — it must NOT be cleared
    here, or the self-loop would wipe it. Cancel clears it via before_cancel_import; a confirmed
    import clears it in process_osm_import_deferred after building the bbox.

    End state: Panel hidden, box drawn from the stored center, ready for confirm.
    """
    logger.debug("ENTER: import_placing - hiding panel")
    ctx.viewing.hide_panel()
    ctx.click_dedup.clear_marker()


def enter_merge_placing(ctx: PlannerContext) -> None:
    """Enter MERGE_PLACING: the user selects node markers to collapse (Single Point of Truth).

    Reached from any idle state on entering merge mode, and re-entered on each node-toggle
    (self-loop). The selected node ids live in ctx.merge.node_ids (toggled by the click handler) —
    they must NOT be cleared here, or the self-loop would wipe the selection on every click.
    Cancel clears it via before_cancel_merge; a confirmed merge clears it in complete_merge.

    End state: Panel hidden, selected nodes drawn red, ready for confirm.
    """
    logger.debug("ENTER: merge_placing - hiding panel")
    ctx.viewing.hide_panel()
    ctx.click_dedup.clear_marker()


# =============================================================================
# 9. IDLE_VIEWING_ROAD - Panel showing road details
# =============================================================================


def enter_idle_viewing_road(ctx: PlannerContext) -> None:
    """Enter IDLE_VIEWING_ROAD: Make road panel visible (Single Point of Truth).

    Mirrors enter_idle_viewing_lift: the road_id was set by a before_* hook;
    this guarantees the panel is visible and clears any stale building state.
    """
    logger.debug("ENTER: idle_viewing_road - showing panel, clearing building state")
    _enter_viewing_panel(ctx)


# =============================================================================
# 10. ROAD_STARTING / ROAD_BUILDING - segment-by-segment, like a slope
# =============================================================================


def enter_road_starting(ctx: PlannerContext) -> None:
    """Enter ROAD_STARTING: begin road building (Single Point of Truth).

    Mirrors enter_slope_starting. The origin point was stored by
    before_start_road. Guarantees the panel is hidden and the click dedup
    marker is fresh, and triggers the road fan from the origin, regardless of
    which transition brought us here.
    """
    logger.debug("ENTER: road_starting - hiding panel, clearing marker dedup, arming road fan")
    _enter_fan_state(ctx, SegmentKind.ROAD)


def enter_road_building(ctx: PlannerContext) -> None:
    """Enter ROAD_BUILDING: continue building road (Single Point of Truth).

    Mirrors enter_slope_building. Sources: first segment committed
    (commit_road_first), self-loop (commit_road_continue), undo. Preserves the road
    context (it holds the committed segments!), hides the panel, and triggers the
    road fan from the new endpoint.
    """
    logger.debug("ENTER: road_building - hiding panel, preserving road context, arming road fan")
    _enter_fan_state(ctx, SegmentKind.ROAD)


def enter_road_custom_path(ctx: PlannerContext) -> None:
    """Enter ROAD_CUSTOM_PATH: show path options routed to a clicked target.

    Mirror of enter_slope_custom_path. The target before-hook set start_node,
    target_location and force_mode; this triggers the shared deferred custom-connect
    generation, which resolves the active build (road) and routes to the target.
    Fires on the retarget self-loop too, so a new target click regenerates proposals.
    """
    logger.debug("ENTER: road_custom_path - clearing marker, triggering deferred custom-connect generation")
    _enter_custom_path(ctx)


# The ONLY states with real exit teardown → their exit function.
EXIT_HOOKS: dict[str, Callable[[PlannerContext], None]] = {
    "lift_placing": exit_lift_placing,
    "import_placing": exit_import_placing,
    "merge_placing": exit_merge_placing,
}

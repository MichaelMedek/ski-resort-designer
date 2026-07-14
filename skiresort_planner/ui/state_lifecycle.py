"""State Lifecycle Functions - Entry and exit handlers for each state.

This module provides all lifecycle functions (all states × 2 = enter + exit) that
define exactly what happens when transitioning into or out of each state.

Architecture:
- Each function is called by the state machine's on_enter_* / on_exit_* hooks
- Functions receive the PlannerContext to modify UI state
- Functions are idempotent and safe to call multiple times
- All functions are implemented even if they do nothing (pass)

Usage in state machine:
    def on_enter_idle_ready(self) -> None:
        enter_idle_ready(self.context)

    def on_exit_idle_ready(self) -> None:
        exit_idle_ready(self.context)

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
    from skiresort_planner.ui.context import PlannerContext

logger = logging.getLogger(__name__)


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


def exit_idle_ready(ctx: PlannerContext) -> None:
    """Exit IDLE_READY: Nothing needed.

    We're leaving idle ready state to either:
    - View a slope/lift (viewing state will set up panel)
    - Start building (building state will initialize)

    The destination state's enter function handles all setup.
    No cleanup needed since idle_ready is a clean state.
    """
    logger.debug("EXIT: idle_ready - no cleanup needed")
    pass


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
    # SINGLE POINT OF TRUTH: Make panel visible
    ctx.viewing.show_panel()
    # Defensive cleanup - clear any stale building state
    ctx.clear_proposals()
    ctx.clear_builds()
    ctx.clear_custom_connect()
    ctx.clear_lift()
    ctx.selection.clear()
    ctx.click_dedup.clear_marker()


def exit_idle_viewing_slope(ctx: PlannerContext) -> None:
    """Exit IDLE_VIEWING_SLOPE: No cleanup needed.

    SINGLE POINT OF TRUTH PRINCIPLE:
    We do NOT touch any viewing state here. The destination state's enter
    function handles all necessary changes:
    - enter_idle_ready: calls ctx.viewing.clear() to reset everything
    - enter_idle_viewing_lift: before_* calls set_lift_id() which clears slope_id
    - enter_slope_starting: hides panel for building mode

    For self-loop transitions (switch_slope), clearing here would erase the
    slope_id set by before_switch_slope, causing errors.
    """
    logger.debug("EXIT: idle_viewing_slope - no cleanup needed")
    pass


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
    # SINGLE POINT OF TRUTH: Make panel visible
    ctx.viewing.show_panel()
    # Defensive cleanup - clear any stale building state
    ctx.clear_proposals()
    ctx.clear_builds()
    ctx.clear_custom_connect()
    ctx.clear_lift()
    ctx.selection.clear()
    ctx.click_dedup.clear_marker()


def exit_idle_viewing_lift(ctx: PlannerContext) -> None:
    """Exit IDLE_VIEWING_LIFT: No cleanup needed.

    SINGLE POINT OF TRUTH PRINCIPLE:
    We do NOT touch any viewing state here. The destination state's enter
    function handles all necessary changes:
    - enter_idle_ready: calls ctx.viewing.clear() to reset everything
    - enter_idle_viewing_slope: before_* calls set_slope_id() which clears lift_id
    - enter_lift_placing: hides panel for placement mode

    For self-loop transitions (switch_lift), clearing here would erase the
    lift_id set by before_switch_lift, causing errors.
    """
    logger.debug("EXIT: idle_viewing_lift - no cleanup needed")
    pass


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
    logger.debug("ENTER: slope_starting - hiding panel, clearing marker dedup")
    # SINGLE POINT OF TRUTH: Hide panel for building mode
    ctx.viewing.hide_panel()
    ctx.click_dedup.clear_marker()
    # Arm the fan here (mirrors enter_road_starting) so EVERY entry to slope_starting — first click,
    # undo back to starting, cancel-custom-to-starting — regenerates proposals. Single source of truth.
    ctx.deferred.fan_generation.add(SegmentKind.SLOPE)


def exit_slope_starting(ctx: PlannerContext) -> None:
    """Exit SLOPE_STARTING: Minimal cleanup.

    Possible destinations:
    - SLOPE_BUILDING: before_commit_path event hook clears proposals
    - SLOPE_CUSTOM_PATH: enter_slope_custom_path regenerates proposals to the target
    - IDLE_READY: before_cancel_slope event hook clears building state

    All destinations handle their own cleanup, so no action needed here.
    """
    logger.debug("EXIT: slope_starting - no cleanup needed")
    pass


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
    logger.debug("ENTER: slope_building - hiding panel, preserving building context")
    # SINGLE POINT OF TRUTH: Hide panel for building mode
    ctx.viewing.hide_panel()
    # Arm the fan on every entry (mirrors enter_road_building) so a fresh proposal set is generated
    # from the new endpoint — after a commit, a custom-continue, or an undo. fan_generation is a set,
    # so a redundant add (the commit flow also arms) is a no-op; the deferred pass regenerates once.
    ctx.deferred.fan_generation.add(SegmentKind.SLOPE)


def exit_slope_building(ctx: PlannerContext) -> None:
    """Exit SLOPE_BUILDING: Minimal cleanup for non-self-loop destinations.

    Possible destinations:
    - SLOPE_BUILDING (self-loop): commit_path/undo_continue
      - before_commit_path: clears proposals (new segment)
      - before_undo_continue: PRESERVES proposals (set by undo_last_action)
    - SLOPE_CUSTOM_PATH: enter_slope_custom_path regenerates proposals to the target
    - IDLE_VIEWING_SLOPE: enter_idle_viewing_slope clears proposals
    - IDLE_READY: enter_idle_ready clears proposals

    IMPORTANT: Do NOT clear proposals here!
    For undo_continue self-loops, proposals are set by undo_last_action() BEFORE
    the state transition. Clearing here would destroy them.
    All other destinations clear proposals in their own hooks.
    """
    logger.debug("EXIT: slope_building - no cleanup (destinations handle it)")
    # Do NOT clear proposals - undo_continue needs them preserved!
    pass


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
    logger.debug("ENTER: slope_custom_path - triggering deferred path generation")
    ctx.deferred.custom_connect = True


def exit_slope_custom_path(ctx: PlannerContext) -> None:
    """Exit SLOPE_CUSTOM_PATH: No-op.

    Cleanup is intentionally NOT done here because:
    - Different transitions need different cleanup
    - before_commit_* and before_cancel_* hooks handle specific cases
    - force_idle()/force_building() handle cleanup explicitly

    This follows the "destination controls cleanup" pattern.
    """
    logger.debug("EXIT: slope_custom_path - no cleanup (destination handles it)")
    pass


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
    ctx.viewing.show_panel()
    ctx.clear_proposals()
    ctx.clear_builds()
    ctx.clear_custom_connect()
    ctx.clear_lift()
    ctx.selection.clear()
    ctx.click_dedup.clear_marker()


def exit_idle_viewing_road(ctx: PlannerContext) -> None:
    """Exit IDLE_VIEWING_ROAD: No cleanup needed.

    The destination state's enter function handles all necessary changes
    (same Single Point of Truth pattern as exit_idle_viewing_lift).
    """
    logger.debug("EXIT: idle_viewing_road - no cleanup needed")
    pass


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
    logger.debug("ENTER: road_starting - hiding panel, clearing marker dedup, triggering road fan")
    ctx.viewing.hide_panel()
    ctx.click_dedup.clear_marker()
    ctx.deferred.fan_generation.add(SegmentKind.ROAD)


def exit_road_starting(ctx: PlannerContext) -> None:
    """Exit ROAD_STARTING: minimal cleanup.

    Destinations handle their own cleanup:
    - ROAD_BUILDING: before_commit_road_first clears proposals
    - IDLE_READY: before_cancel_road / enter_idle_ready clears road state
    """
    logger.debug("EXIT: road_starting - no cleanup needed")


def enter_road_building(ctx: PlannerContext) -> None:
    """Enter ROAD_BUILDING: continue building road (Single Point of Truth).

    Mirrors enter_slope_building. Sources: first segment committed
    (commit_road_first), self-loop (commit_road_continue), undo. Preserves the road
    context (it holds the committed segments!), hides the panel, and triggers the
    road fan from the new endpoint.
    """
    logger.debug("ENTER: road_building - hiding panel, preserving road context, triggering road fan")
    ctx.viewing.hide_panel()
    ctx.deferred.fan_generation.add(SegmentKind.ROAD)


def exit_road_building(ctx: PlannerContext) -> None:
    """Exit ROAD_BUILDING: no cleanup here.

    Destinations own their cleanup (self-loop clears proposals via
    before_commit_road_continue; finish_road/cancel clear road context). Clearing
    road state here would erase the committed segments on the self-loop.
    """
    logger.debug("EXIT: road_building - no cleanup needed")


def enter_road_custom_path(ctx: PlannerContext) -> None:
    """Enter ROAD_CUSTOM_PATH: show path options routed to a clicked target.

    Mirror of enter_slope_custom_path. The target before-hook set start_node,
    target_location and force_mode; this triggers the shared deferred custom-connect
    generation, which resolves the active build (road) and routes to the target.
    Fires on the retarget self-loop too, so a new target click regenerates proposals.
    """
    logger.debug("ENTER: road_custom_path - triggering deferred custom-connect generation")
    ctx.deferred.custom_connect = True


def exit_road_custom_path(ctx: PlannerContext) -> None:
    """Exit ROAD_CUSTOM_PATH: no-op (destinations handle their own cleanup)."""
    logger.debug("EXIT: road_custom_path - no cleanup (destination handles it)")

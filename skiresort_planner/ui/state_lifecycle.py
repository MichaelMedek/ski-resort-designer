"""State Lifecycle Functions - Entry and exit handlers for each state.

This module provides 16 lifecycle functions (8 states × 2 = enter + exit) that
define exactly what happens when transitioning into or out of each state.

Architecture:
- Each function is called by the state machine's on_enter_* / on_exit_* hooks
- Functions receive the PlannerContext to modify UI state
- Functions are idempotent and safe to call multiple times
- All 16 functions are implemented even if they do nothing (pass)

Usage in state machine:
    def on_enter_idle_ready(self) -> None:
        enter_idle_ready(self.context)

    def on_exit_idle_ready(self) -> None:
        exit_idle_ready(self.context)

State Definitions:
    1. IDLE_READY: No panel visible, ready to start building
    2. IDLE_VIEWING_SLOPE: Panel showing slope details, profile visible, 3D available
    3. IDLE_VIEWING_LIFT: Panel showing lift details, profile visible, 3D available
    4. SLOPE_STARTING: 0 segments committed, picking first fan direction
    5. SLOPE_BUILDING: 1+ segments committed, continuing slope picking next fan direction
    6. SLOPE_CUSTOM_PICKING: Waiting for custom target click or cancel to return to building/starting
    7. SLOPE_CUSTOM_PATH: Showing custom path options
    8. LIFT_PLACING: Start selected, waiting for end station
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

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
    logger.debug("ENTER: idle_ready - clearing all building state")
    ctx.clear_proposals()
    ctx.clear_building()
    ctx.clear_custom_connect()
    ctx.clear_lift()
    ctx.selection.node_id = None
    ctx.click_dedup.clear_marker()
    ctx.viewing.clear()


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
    ctx.clear_building()
    ctx.clear_custom_connect()
    ctx.clear_lift()
    ctx.selection.node_id = None
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
    ctx.clear_building()
    ctx.clear_custom_connect()
    ctx.clear_lift()
    ctx.selection.node_id = None
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
    - deferred.path_generation is set to trigger path generation

    This function is responsible for:
    - Hiding any viewing panel
    - Clearing click dedup marker for fresh clicks

    End state: Panel hidden, ready for path proposals
    """
    logger.debug("ENTER: slope_starting - hiding panel, clearing marker dedup")
    # SINGLE POINT OF TRUTH: Hide panel for building mode
    ctx.viewing.hide_panel()
    ctx.click_dedup.clear_marker()


def exit_slope_starting(ctx: PlannerContext) -> None:
    """Exit SLOPE_STARTING: Minimal cleanup.

    Possible destinations:
    - SLOPE_BUILDING: before_commit_first_path clears proposals
    - SLOPE_CUSTOM_PICKING: enter_slope_custom_picking clears proposals
    - IDLE_READY: enter_idle_ready clears proposals

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
    regardless of which transition brought us here (commit_first_path,
    commit_path, commit_custom_path, undo_continue, resume_building).

    Sources:
    - From SLOPE_STARTING: First path committed
    - From SLOPE_CUSTOM_PATH: Custom path committed (continue)
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


def exit_slope_building(ctx: PlannerContext) -> None:
    """Exit SLOPE_BUILDING: Minimal cleanup for non-self-loop destinations.

    Possible destinations:
    - SLOPE_BUILDING (self-loop): commit_path/undo_continue
      - before_commit_path: clears proposals (new segment)
      - before_undo_continue: PRESERVES proposals (set by undo_last_action)
    - SLOPE_CUSTOM_PICKING: enter_slope_custom_picking clears proposals
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
# 6. SLOPE_CUSTOM_PICKING - Waiting for custom target click
# =============================================================================


def enter_slope_custom_picking(ctx: PlannerContext) -> None:
    """Enter SLOPE_CUSTOM_PICKING: Wait for user to click target location.

    Sources:
    - From SLOPE_STARTING: Custom connect button pressed (0 segments)
    - From SLOPE_BUILDING: Custom connect button pressed (1+ segments)

    What happens externally (handled by before_* transition actions):
    - custom_connect.enabled is set to True
    - custom_connect.start_node is set to current endpoint

    What this function ensures:
    - Clear proposals (no fan proposals in picking mode)

    What should NOT be cleared:
    - building context (has committed segments if from BUILDING)
    - custom_connect state (being set up)

    End state: Map ready to receive target click, no proposals shown
    """
    logger.debug("ENTER: slope_custom_picking - clearing proposals")
    ctx.clear_proposals()


def exit_slope_custom_picking(ctx: PlannerContext) -> None:
    """Exit SLOPE_CUSTOM_PICKING: Clean up picking state.

    Possible destinations:
    - SLOPE_CUSTOM_PATH: Target selected (generate paths to target)
    - SLOPE_STARTING: Cancel (back to 0 segments)
    - SLOPE_BUILDING: Cancel (back to 1+ segments)

    What needs to be cleared:
    - custom_connect.enabled flag (no longer picking)

    Note: custom_connect.target_location is set by transition when going to PATH.
    """
    logger.debug("EXIT: slope_custom_picking - disabling custom connect picking")
    ctx.custom_connect.enabled = False


# =============================================================================
# 7. SLOPE_CUSTOM_PATH - Showing custom path options
# =============================================================================


def enter_slope_custom_path(ctx: PlannerContext) -> None:
    """Enter SLOPE_CUSTOM_PATH: Show path options to custom target.

    Sources:
    - From SLOPE_CUSTOM_PICKING: Target location selected

    What happens externally (handled by deferred action):
    - custom_connect.target_location is set
    - deferred.custom_connect triggers path generation
    - proposals are populated with paths to target

    What this function ensures:
    - Nothing to clear (proposals being generated)

    End state: Path proposals shown from start to custom target
    """
    logger.debug("ENTER: slope_custom_path - proposals being generated by deferred action")
    pass


def exit_slope_custom_path(ctx: PlannerContext) -> None:
    """Exit SLOPE_CUSTOM_PATH: Clean up custom path state.

    Possible destinations:
    - SLOPE_BUILDING: Path committed (continue building)
    - IDLE_VIEWING_SLOPE: Connector path finishes slope (auto-finish)
    - SLOPE_STARTING: Cancel (back to 0 segments)
    - SLOPE_BUILDING: Cancel (back to 1+ segments)

    What needs to be cleared:
    - custom_connect.force_mode flag
    - custom_connect.target_location

    Note: proposals are cleared by destination's enter or transition action.
    """
    logger.debug("EXIT: slope_custom_path - clearing custom connect force mode")
    ctx.custom_connect.force_mode = False
    ctx.custom_connect.target_location = None


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

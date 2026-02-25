"""State machine for ski resort planner UI.

Uses python-statemachine for robust state management with:
- Clear state definitions (8 explicit states)
- Guarded transitions (conditions)
- Entry/exit hooks for side effects
- Explicit event-driven transitions
- Dispatch-table pattern for UI rendering

Architecture Overview
---------------------
This module implements a UI state machine integrated with Streamlit's reactive model.
The key pattern is:

1. User action triggers state transition (e.g., click map → start_slope)
2. StreamlitUIListener fires after_transition and calls graph.perform_cleanup() + st.rerun()
3. On the next render cycle, handle_deferred_actions() checks pending flags
4. Deferred work (e.g., path generation) executes with access to full context

This separates state transitions (instant) from business logic (deferred), ensuring
the state machine remains focused on state management while expensive operations
run after the UI refresh.

States (8 explicit states)
--------------------------
    1. IDLE_READY: No panel visible, ready to start building
    2. IDLE_VIEWING_SLOPE: Panel showing slope details (3D toggle available)
    3. IDLE_VIEWING_LIFT: Panel showing lift details (3D toggle available)
    4. SLOPE_STARTING: 0 segments committed, picking first direction
    5. SLOPE_BUILDING: 1+ segments committed, continuing slope
    6. SLOPE_CUSTOM_PICKING: Waiting for custom target click
    7. SLOPE_CUSTOM_PATH: Showing custom path options
    8. LIFT_PLACING: Start selected, waiting for end station

Orthogonal State (flags, not formal states)
-------------------------------------------
    - view_3d: 3D terrain view toggle (only in IDLE_VIEWING_* states)
              When True, map clicks are BLOCKED (UI enforces, not state machine)
    - build_mode: Determines what type of element to build (slope/lift type)

    Design Decision - Why view_3d is NOT a separate state:
    - Would require 4 viewing states instead of 2 (VIEWING_SLOPE_2D, VIEWING_SLOPE_3D, etc.)
    - Would increase transition matrix from 64 to 100 combinations
    - Click blocking is a UI concern, not a workflow state concern
    - The 3D view doesn't change WHAT actions are available, only HOW the map is rendered
    - Current approach: UI checks view_3d flag and ignores map clicks when True

Complete Transition Matrix (8x8 = 64 combinations)
==================================================

# 1. Transitions: From IDLE_READY
# --------------------------------
# 1.1. → IDLE_READY: NOT ALLOWED (no-op, nothing to transition)
# 1.2. → IDLE_VIEWING_SLOPE: view_slope (click slope icon/centerline)
# 1.3. → IDLE_VIEWING_LIFT: view_lift (click lift icon/cable)
# 1.4. → SLOPE_STARTING: start_slope (click terrain/node in slope mode)
# 1.5. → SLOPE_BUILDING: undo_restore_to_building (undo finish_slope after error recovery)
# 1.6. → SLOPE_CUSTOM_PICKING: NOT ALLOWED (must go through SLOPE_STARTING first)
# 1.7. → SLOPE_CUSTOM_PATH: NOT ALLOWED (must go through SLOPE_CUSTOM_PICKING first)
# 1.8. → LIFT_PLACING: start_lift (click terrain/node in lift mode)

# 2. Transitions: From IDLE_VIEWING_SLOPE
# ----------------------------------------
# 2.1. → IDLE_READY: close_slope_panel (close button or click elsewhere)
# 2.2. → IDLE_VIEWING_SLOPE: switch_slope (self-loop: click different slope)
# 2.3. → IDLE_VIEWING_LIFT: switch_to_lift_view (click lift in panel or on map)
# 2.4. → SLOPE_STARTING: start_slope_from_slope_view (click terrain/node),
#                        resume_to_starting (undo finish when 0 segments)
# 2.5. → SLOPE_BUILDING: resume_to_building (undo finish when 1+ segments)
# 2.6. → SLOPE_CUSTOM_PICKING: NOT ALLOWED (must start slope first)
# 2.7. → SLOPE_CUSTOM_PATH: NOT ALLOWED (must go through picking first)
# 2.8. → LIFT_PLACING: start_lift_from_slope_view (click terrain/node in lift mode)

# 3. Transitions: From IDLE_VIEWING_LIFT
# ---------------------------------------
# 3.1. → IDLE_READY: close_lift_panel (close button or click elsewhere)
# 3.2. → IDLE_VIEWING_SLOPE: switch_to_slope_view (click connected slope in panel)
# 3.3. → IDLE_VIEWING_LIFT: switch_lift (self-loop: click different lift)
# 3.4. → SLOPE_STARTING: start_slope_from_lift_view (click terrain/node),
#                        undo_restore_from_lift_to_starting (undo finish_slope)
# 3.5. → SLOPE_BUILDING: undo_restore_from_lift_to_building (undo finish_slope)
# 3.6. → SLOPE_CUSTOM_PICKING: NOT ALLOWED (must start slope first)
# 3.7. → SLOPE_CUSTOM_PATH: NOT ALLOWED (must go through picking first)
# 3.8. → LIFT_PLACING: start_lift_from_lift_view (click terrain/node in lift mode)

# 4. Transitions: From SLOPE_STARTING
# ------------------------------------
# 4.1. → IDLE_READY: cancel_from_starting (cancel button)
# 4.2. → IDLE_VIEWING_SLOPE: NOT ALLOWED (must commit or cancel first)
# 4.3. → IDLE_VIEWING_LIFT: NOT ALLOWED (must commit or cancel first)
# 4.4. → SLOPE_STARTING: NOT ALLOWED (no self-loop, proposal selection is internal)
# 4.5. → SLOPE_BUILDING: commit_first_path (click proposal endpoint)
# 4.6. → SLOPE_CUSTOM_PICKING: enable_custom_from_starting (custom connect button)
# 4.7. → SLOPE_CUSTOM_PATH: NOT ALLOWED (must go through picking first)
# 4.8. → LIFT_PLACING: NOT ALLOWED (must cancel slope first)

# 5. Transitions: From SLOPE_BUILDING
# ------------------------------------
# 5.1. → IDLE_READY: cancel_from_building (cancel button),
#                    undo_to_idle (undo when only 1 segment left)
# 5.2. → IDLE_VIEWING_SLOPE: finish_slope (finish button)
# 5.3. → IDLE_VIEWING_LIFT: NOT ALLOWED (must finish/cancel slope first)
# 5.4. → SLOPE_STARTING: NOT ALLOWED (would lose committed segments)
# 5.5. → SLOPE_BUILDING: commit_continue_path (self-loop: commit more segments),
#                        undo_continue (self-loop: undo keeps 2+ segments)
# 5.6. → SLOPE_CUSTOM_PICKING: enable_custom_from_building (custom connect button)
# 5.7. → SLOPE_CUSTOM_PATH: NOT ALLOWED (must go through picking first)
# 5.8. → LIFT_PLACING: NOT ALLOWED (must finish/cancel slope first)

# 6. Transitions: From SLOPE_CUSTOM_PICKING
# ------------------------------------------
# 6.1. → IDLE_READY: NOT ALLOWED (must cancel to starting/building first)
# 6.2. → IDLE_VIEWING_SLOPE: NOT ALLOWED (must cancel or select first)
# 6.3. → IDLE_VIEWING_LIFT: NOT ALLOWED (must cancel or select first)
# 6.4. → SLOPE_STARTING: cancel_custom_to_starting (cancel when has_no_segments)
# 6.5. → SLOPE_BUILDING: cancel_custom_to_building (cancel when has segments)
# 6.6. → SLOPE_CUSTOM_PICKING: NOT ALLOWED (waiting for click, no self-loop)
# 6.7. → SLOPE_CUSTOM_PATH: select_custom_target (click target location)
# 6.8. → LIFT_PLACING: NOT ALLOWED (must cancel first)

# 7. Transitions: From SLOPE_CUSTOM_PATH
# ---------------------------------------
# 7.1. → IDLE_READY: NOT ALLOWED (must go through cancel to starting/building)
# 7.2. → IDLE_VIEWING_SLOPE: commit_custom_finish (auto-finish when connecting to node)
# 7.3. → IDLE_VIEWING_LIFT: NOT ALLOWED (must finish/cancel slope first)
# 7.4. → SLOPE_STARTING: cancel_path_to_starting (cancel when has_no_segments)
# 7.5. → SLOPE_BUILDING: commit_custom_continue (commit and keep building),
#                        cancel_path_to_building (cancel when has segments)
# 7.6. → SLOPE_CUSTOM_PICKING: NOT ALLOWED (can't go back to picking, must cancel)
# 7.7. → SLOPE_CUSTOM_PATH: NOT ALLOWED (no self-loop, different target → cancel+repick)
# 7.8. → LIFT_PLACING: NOT ALLOWED (must finish/cancel slope first)

# 8. Transitions: From LIFT_PLACING
# ----------------------------------
# 8.1. → IDLE_READY: cancel_lift (cancel button)
# 8.2. → IDLE_VIEWING_SLOPE: NOT ALLOWED (must cancel or complete first)
# 8.3. → IDLE_VIEWING_LIFT: complete_lift (click end station location)
# 8.4. → SLOPE_STARTING: NOT ALLOWED (must cancel lift first)
# 8.5. → SLOPE_BUILDING: NOT ALLOWED (must cancel lift first)
# 8.6. → SLOPE_CUSTOM_PICKING: NOT ALLOWED (must cancel lift first)
# 8.7. → SLOPE_CUSTOM_PATH: NOT ALLOWED (must cancel lift first)
# 8.8. → LIFT_PLACING: NOT ALLOWED (no self-loop needed)

Transition Summary Table
------------------------
    ALLOWED (28 transitions + 4 self-loops = 32 total):
    - From IDLE_READY (5): view_slope, view_lift, start_slope, start_lift, undo_restore_to_building
    - From IDLE_VIEWING_SLOPE (6+1): close, switch_to_lift, start_slope, start_lift, resume_to_starting, resume_to_building, switch_slope (loop)
    - From IDLE_VIEWING_LIFT (6+1): close, switch_to_slope, start_slope, start_lift, undo_restore_to_starting, undo_restore_to_building, switch_lift (loop)
    - From SLOPE_STARTING (3): cancel, commit_first_path, enable_custom
    - From SLOPE_BUILDING (5+2): cancel, undo_to_idle, finish, enable_custom, commit_path (loop), undo_continue (loop)
    - From SLOPE_CUSTOM_PICKING (3): cancel_to_starting, cancel_to_building, select_target
    - From SLOPE_CUSTOM_PATH (4): commit_continue, commit_finish, cancel_to_starting, cancel_to_building
    - From LIFT_PLACING (2): cancel, complete

    NOT ALLOWED (32 combinations): All other transitions that would bypass required workflow steps

Undo Transitions (special cases)
--------------------------------
    - undo_to_idle: SLOPE_BUILDING → IDLE_READY (when undoing last segment)
    - undo_continue: SLOPE_BUILDING → SLOPE_BUILDING (when 2+ segments remain)
    - resume_to_starting: IDLE_VIEWING_SLOPE → SLOPE_STARTING (undo finish_slope, 0 segments)
    - resume_to_building: IDLE_VIEWING_SLOPE → SLOPE_BUILDING (undo finish_slope, 1+ segments)
    - undo_restore_to_starting: IDLE_READY → SLOPE_STARTING (undo after error recovery)
    - undo_restore_to_building: IDLE_READY → SLOPE_BUILDING (undo after error recovery)
    - undo_restore_from_lift_to_*: IDLE_VIEWING_LIFT → SLOPE_* (undo after building lift from slope)

Cleanup on Transition
---------------------
StreamlitUIListener.after_transition() calls graph.perform_cleanup() before st.rerun().
This ensures the resort graph is always in a clean state by:
- Removing isolated nodes (nodes not connected to any segment or lift)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import streamlit as st
from statemachine import State, StateMachine
from statemachine.exceptions import TransitionNotAllowed

from skiresort_planner.model.path_point import PathPoint
from skiresort_planner.model.resort_graph import ResortGraph
from skiresort_planner.ui.context import (
    BuildingContext,
    BuildMode,
    BuildModeContext,
    ClickDeduplicationContext,
    CustomConnectContext,
    DeferredContext,
    LiftContext,
    LonLat,
    LonLatElev,
    MapContext,
    PlannerContext,
    ProposalContext,
    SelectionContext,
    UIMessagesContext,
    ViewingContext,
)
from skiresort_planner.ui.state_lifecycle import (
    enter_idle_ready,
    enter_idle_viewing_lift,
    enter_idle_viewing_slope,
    enter_lift_placing,
    enter_slope_building,
    enter_slope_custom_path,
    enter_slope_custom_picking,
    enter_slope_starting,
    exit_idle_ready,
    exit_idle_viewing_lift,
    exit_idle_viewing_slope,
    exit_lift_placing,
    exit_slope_building,
    exit_slope_custom_path,
    exit_slope_custom_picking,
    exit_slope_starting,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    pass


# Re-export context classes for backward compatibility
__all__ = [
    "BuildMode",
    "BuildModeContext",
    "BuildingContext",
    "ClickDeduplicationContext",
    "CustomConnectContext",
    "DeferredContext",
    "LiftContext",
    "LonLat",
    "LonLatElev",
    "MapContext",
    "PlannerContext",
    "PlannerStateMachine",
    "ProposalContext",
    "SelectionContext",
    "StreamlitUIListener",
    "UIMessagesContext",
    "ViewingContext",
]


class StreamlitUIListener:
    """Listener that handles Streamlit UI side effects after state transitions.

    This listener follows the python-statemachine best practice of using
    listeners for side effects. It runs after every state transition to:

    1. Perform cleanup (remove isolated nodes, create auto-backup)
    2. Trigger st.rerun() to refresh the UI

    The separation ensures the state machine focuses purely on state logic
    while this listener handles UI integration and maintenance tasks.

    Usage:
        sm = PlannerStateMachine(context=context)
        sm.add_listener(StreamlitUIListener())
    """

    def after_transition(self, event: str, source: State, target: State) -> None:
        """Run cleanup and trigger Streamlit rerun after state transitions.

        NOTE: We do NOT modify click deduplication here. The dedup is simple:
        same click key = duplicate. When user clicks elsewhere, key changes,
        so they can click back to original element.
        """
        logger.info(f"[STATE] {source.name} --({event})--> {target.name}")

        # Perform graph cleanup before rerun (isolated nodes, auto-backup)
        graph: ResortGraph | None = st.session_state.get("graph")
        if graph is not None:
            graph.perform_cleanup()

        st.rerun()


class PlannerStateMachine(StateMachine):
    """State machine for ski resort planner workflow.

    Manages transitions between 8 planning states with guards
    and hooks for validation and side effects. See module docstring
    for complete transition documentation.

    States (8 explicit states):
        IDLE_READY: No panel visible, ready to build
        IDLE_VIEWING_SLOPE: Panel showing slope details
        IDLE_VIEWING_LIFT: Panel showing lift details
        SLOPE_STARTING: 0 segments, picking first direction
        SLOPE_BUILDING: 1+ segments, continuing
        SLOPE_CUSTOM_PICKING: Waiting for custom target click
        SLOPE_CUSTOM_PATH: Showing custom path options
        LIFT_PLACING: Waiting for end station

    Using explicit states eliminates impossible state combinations
    and enables dispatch-table UI rendering pattern.
    """

    # ==========================================================================
    # State Definitions (8 explicit states)
    # ==========================================================================

    # IDLE states (no building in progress)
    idle_ready = State("IdleReady", initial=True)
    idle_viewing_slope = State("IdleViewingSlope")
    idle_viewing_lift = State("IdleViewingLift")

    # SLOPE states (building in progress)
    slope_starting = State("SlopeStarting")
    slope_building = State("SlopeBuilding")
    slope_custom_picking = State("SlopeCustomPicking")
    slope_custom_path = State("SlopeCustomPath")

    # LIFT state
    lift_placing = State("LiftPlacing")

    # ==========================================================================
    # 1. Transitions: From IDLE_READY
    # ==========================================================================
    # 1.2. view_slope: Click slope icon/centerline to view details
    # 1.3. view_lift: Click lift icon/cable to view details
    # 1.4. start_slope: Click terrain/node in slope mode
    # 1.8. start_lift: Click terrain/node in lift mode

    start_slope = idle_ready.to(slope_starting)  # 1.4
    start_lift = idle_ready.to(lift_placing)  # 1.8
    view_slope = idle_ready.to(idle_viewing_slope)  # 1.2
    view_lift = idle_ready.to(idle_viewing_lift)  # 1.3

    # ==========================================================================
    # 2. Transitions: From IDLE_VIEWING_SLOPE
    # ==========================================================================
    # 2.1. close_slope_panel: Close button or click elsewhere
    # 2.3. NOT ALLOWED: click connected lift → close+view (no direct switch)
    # 2.4. start_slope_from_slope_view: Click terrain/node to start new slope
    # 2.8. start_lift_from_slope_view: Click terrain/node in lift mode

    close_slope_panel = idle_viewing_slope.to(idle_ready)  # 2.1
    switch_slope = idle_viewing_slope.to(idle_viewing_slope)  # 2.2 self-loop
    switch_to_lift_view = idle_viewing_slope.to(idle_viewing_lift)  # 2.3
    start_slope_from_slope_view = idle_viewing_slope.to(slope_starting)  # 2.4
    start_lift_from_slope_view = idle_viewing_slope.to(lift_placing)  # 2.8
    resume_building = idle_viewing_slope.to(slope_building)  # legacy alias for resume_to_building

    # ==========================================================================
    # 3. Transitions: From IDLE_VIEWING_LIFT
    # ==========================================================================
    # 3.1. close_lift_panel: Close button or click elsewhere
    # 3.2. switch_to_slope_view: Click connected slope in panel
    # 3.4. start_slope_from_lift_view: Click terrain/node in slope mode
    # 3.8. start_lift_from_lift_view: Click terrain/node in lift mode

    close_lift_panel = idle_viewing_lift.to(idle_ready)  # 3.1
    switch_to_slope_view = idle_viewing_lift.to(idle_viewing_slope)  # 3.2
    switch_lift = idle_viewing_lift.to(idle_viewing_lift)  # 3.3 self-loop
    start_slope_from_lift_view = idle_viewing_lift.to(slope_starting)  # 3.4
    start_lift_from_lift_view = idle_viewing_lift.to(lift_placing)  # 3.8

    # ==========================================================================
    # 4. Transitions: From SLOPE_STARTING (0 segments)
    # ==========================================================================
    # 4.1. cancel_from_starting: Cancel button
    # 4.5. commit_first_path: Click proposal endpoint to commit first segment
    # 4.6. enable_custom_from_starting: Custom connect button

    commit_first_path = slope_starting.to(slope_building, event="commit_path")  # 4.5
    cancel_from_starting = slope_starting.to(idle_ready)  # 4.1
    enable_custom_from_starting = slope_starting.to(slope_custom_picking)  # 4.6

    # ==========================================================================
    # 5. Transitions: From SLOPE_BUILDING (1+ segments)
    # ==========================================================================
    # 5.1. cancel_from_building: Cancel button (discard all)
    # 5.1. undo_to_idle: Undo when only 1 segment (discards slope)
    # 5.2. finish_slope: Finish button
    # 5.5. commit_continue_path: Self-loop, commit more segments
    # 5.5. undo_continue: Self-loop, undo keeps 2+ segments
    # 5.6. enable_custom_from_building: Custom connect button

    commit_continue_path = slope_building.to(slope_building, event="commit_path")  # 5.5 self-loop
    finish_slope = slope_building.to(idle_viewing_slope)  # 5.2
    cancel_from_building = slope_building.to(idle_ready)  # 5.1
    enable_custom_from_building = slope_building.to(slope_custom_picking)  # 5.6
    undo_to_idle = slope_building.to(idle_ready, cond="undo_leaves_no_segments")  # 5.1 conditional
    undo_continue = slope_building.to(slope_building, unless="undo_leaves_no_segments")  # 5.5 self-loop

    # ==========================================================================
    # 6. Transitions: From SLOPE_CUSTOM_PICKING
    # ==========================================================================
    # 6.4. cancel_custom_to_starting: Cancel when has_no_segments
    # 6.5. cancel_custom_to_building: Cancel when has segments
    # 6.7. select_custom_target: Click target location

    select_custom_target = slope_custom_picking.to(slope_custom_path)  # 6.7
    cancel_custom_to_starting = slope_custom_picking.to(slope_starting, cond="has_no_segments")  # 6.4
    cancel_custom_to_building = slope_custom_picking.to(slope_building, unless="has_no_segments")  # 6.5

    # ==========================================================================
    # 7. Transitions: From SLOPE_CUSTOM_PATH
    # ==========================================================================
    # 7.2. commit_custom_finish: Auto-finish when connecting to existing node
    # 7.4. cancel_path_to_starting: Cancel when has_no_segments
    # 7.5. commit_custom_continue: Commit and keep building
    # 7.5. cancel_path_to_building: Cancel when has segments

    commit_custom_continue = slope_custom_path.to(slope_building)  # 7.5
    commit_custom_finish = slope_custom_path.to(idle_viewing_slope)  # 7.2 auto-finish connector
    cancel_path_to_starting = slope_custom_path.to(slope_starting, cond="has_no_segments")  # 7.4
    cancel_path_to_building = slope_custom_path.to(slope_building, unless="has_no_segments")  # 7.5

    # ==========================================================================
    # 8. Transitions: From LIFT_PLACING
    # ==========================================================================
    # 8.1. cancel_lift: Cancel button
    # 8.3. complete_lift: Click end station location

    complete_lift = lift_placing.to(idle_viewing_lift)  # 8.3
    cancel_lift = lift_placing.to(idle_ready)  # 8.1

    # ==========================================================================
    # UNDO Transitions: Resume building after finish_slope undo
    # ==========================================================================
    # These transitions restore slope building state after undo of finish_slope.
    # The source state depends on what the user did after finishing the slope.
    #
    # 2.4. resume_to_starting: From IDLE_VIEWING_SLOPE when has_no_segments
    # 2.5. resume_to_building: From IDLE_VIEWING_SLOPE when has segments
    # 1.5. undo_restore_to_building: From IDLE_READY (after error recovery)
    # 3.4. undo_restore_from_lift_to_starting: From IDLE_VIEWING_LIFT
    # 3.5. undo_restore_from_lift_to_building: From IDLE_VIEWING_LIFT

    # From idle_viewing_slope (normal case: user is still viewing the slope they finished)
    resume_to_starting = idle_viewing_slope.to(slope_starting, cond="has_no_segments")  # 2.4
    resume_to_building = idle_viewing_slope.to(slope_building, unless="has_no_segments")  # 2.5

    # From idle_ready (rare: undo after closing panel or error recovery)
    undo_restore_to_starting = idle_ready.to(slope_starting, cond="has_no_segments")  # 1.4 (undo variant)
    undo_restore_to_building = idle_ready.to(slope_building, unless="has_no_segments")  # 1.5

    # From idle_viewing_lift (undo finish_slope after user built a lift from the slope endpoint)
    undo_restore_from_lift_to_starting = idle_viewing_lift.to(slope_starting, cond="has_no_segments")  # 3.4 (undo)
    undo_restore_from_lift_to_building = idle_viewing_lift.to(slope_building, unless="has_no_segments")  # 3.5

    # ==========================================================================
    # Guards (Conditions)
    # ==========================================================================

    def has_no_segments(self) -> bool:
        """Guard: Check if there are no committed segments."""
        return len(self.context.building.segments) == 0

    def undo_leaves_no_segments(self) -> bool:
        """Guard: Check if undo would leave zero segments."""
        return len(self.context.building.segments) <= 1

    # ==========================================================================
    # State Check Properties
    # ==========================================================================

    @property
    def is_idle(self) -> bool:
        """Check if in any idle state (not building)."""
        return self.is_idle_ready or self.is_idle_viewing_slope or self.is_idle_viewing_lift

    @property
    def is_idle_ready(self) -> bool:
        """Check if in idle ready state (no panel)."""
        return bool(self.idle_ready.is_active)

    @property
    def is_idle_viewing_slope(self) -> bool:
        """Check if viewing a slope."""
        return bool(self.idle_viewing_slope.is_active)

    @property
    def is_idle_viewing_lift(self) -> bool:
        """Check if viewing a lift."""
        return bool(self.idle_viewing_lift.is_active)

    @property
    def is_slope_starting(self) -> bool:
        """Check if starting a slope (0 segments)."""
        return bool(self.slope_starting.is_active)

    @property
    def is_slope_building_only(self) -> bool:
        """Check if in slope_building state specifically (1+ segments)."""
        return bool(self.slope_building.is_active)

    @property
    def is_slope_custom_picking(self) -> bool:
        """Check if picking custom target."""
        return bool(self.slope_custom_picking.is_active)

    @property
    def is_slope_custom_path(self) -> bool:
        """Check if showing custom path options."""
        return bool(self.slope_custom_path.is_active)

    @property
    def is_lift_placing(self) -> bool:
        """Check if placing a lift."""
        return bool(self.lift_placing.is_active)

    # Composite state checks
    @property
    def is_any_slope_state(self) -> bool:
        """Check if in any slope-related state.

        Returns True for: slope_starting, slope_building, slope_custom_picking, slope_custom_path
        """
        return (
            self.is_slope_starting
            or self.is_slope_building_only
            or self.is_slope_custom_picking
            or self.is_slope_custom_path
        )

    @property
    def is_info_panel_visible(self) -> bool:
        """Check if info panel is visible (viewing slope or lift)."""
        return self.is_idle_viewing_slope or self.is_idle_viewing_lift

    def is_slope_mode(self) -> bool:
        """Check if build mode is set to slope."""
        return BuildMode.is_slope(self.context.build_mode.mode)

    def is_lift_mode(self) -> bool:
        """Check if build mode is set to any lift type."""
        return BuildMode.is_lift(self.context.build_mode.mode)

    # ==========================================================================
    # Entry Hooks - Using lifecycle functions
    # ==========================================================================

    def on_enter_idle_ready(self) -> None:
        """Hook: Entering idle ready state."""
        enter_idle_ready(self.context)

    def on_enter_idle_viewing_slope(self) -> None:
        """Hook: Entering slope viewing state."""
        enter_idle_viewing_slope(self.context)

    def on_enter_idle_viewing_lift(self) -> None:
        """Hook: Entering lift viewing state."""
        enter_idle_viewing_lift(self.context)

    def on_enter_slope_starting(self) -> None:
        """Hook: Entering slope starting state."""
        enter_slope_starting(self.context)

    def on_enter_slope_building(self) -> None:
        """Hook: Entering slope building state."""
        enter_slope_building(self.context)

    def on_enter_slope_custom_picking(self) -> None:
        """Hook: Entering custom picking state."""
        enter_slope_custom_picking(self.context)

    def on_enter_slope_custom_path(self) -> None:
        """Hook: Entering custom path state."""
        enter_slope_custom_path(self.context)

    def on_enter_lift_placing(self) -> None:
        """Hook: Entering lift placing state."""
        enter_lift_placing(self.context)

    # ==========================================================================
    # Exit Hooks - Using lifecycle functions
    # ==========================================================================

    def on_exit_idle_ready(self) -> None:
        """Hook: Exiting idle ready state."""
        exit_idle_ready(self.context)

    def on_exit_idle_viewing_slope(self) -> None:
        """Hook: Exiting slope viewing state."""
        exit_idle_viewing_slope(self.context)

    def on_exit_idle_viewing_lift(self) -> None:
        """Hook: Exiting lift viewing state."""
        exit_idle_viewing_lift(self.context)

    def on_exit_slope_starting(self) -> None:
        """Hook: Exiting slope starting state."""
        exit_slope_starting(self.context)

    def on_exit_slope_building(self) -> None:
        """Hook: Exiting slope building state."""
        exit_slope_building(self.context)

    def on_exit_slope_custom_picking(self) -> None:
        """Hook: Exiting custom picking state."""
        exit_slope_custom_picking(self.context)

    def on_exit_slope_custom_path(self) -> None:
        """Hook: Exiting custom path state."""
        exit_slope_custom_path(self.context)

    def on_exit_lift_placing(self) -> None:
        """Hook: Exiting lift placing state."""
        exit_lift_placing(self.context)

    # ==========================================================================
    # Transition Actions (before_* hooks)
    # ==========================================================================

    def before_start_slope(
        self,
        lon: float,
        lat: float,
        elevation: float,
        node_id: str | None = None,
    ) -> None:
        """Action before starting to build a slope."""
        self.context.set_selection(lon=lon, lat=lat, elevation=elevation)
        self.context.building.start_node = node_id
        self.context.selection.node_id = node_id
        slope_number = self._resort_graph._slope_counter + 1
        self.context.building.name = f"Slope {slope_number}"

    # Reuse start_slope logic for other entry points
    before_start_slope_from_slope_view = before_start_slope
    before_start_slope_from_lift_view = before_start_slope

    def before_commit_first_path(self, segment_id: str, endpoint_node_id: str) -> None:
        """Action before committing the first path segment."""
        self.context.building.segments.append(segment_id)
        self.context.building.endpoints = [endpoint_node_id]
        self.context.clear_proposals()

    def before_commit_path(self, segment_id: str, endpoint_node_id: str) -> None:
        """Action before committing a path segment."""
        self.context.building.segments.append(segment_id)
        self.context.building.endpoints = [endpoint_node_id]
        self.context.clear_proposals()

    def before_commit_custom_continue(self, segment_id: str, endpoint_node_id: str) -> None:
        """Action before committing custom path and continuing."""
        self.context.building.segments.append(segment_id)
        self.context.building.endpoints = [endpoint_node_id]
        self.context.clear_proposals()
        self.context.custom_connect.clear()

    def before_commit_custom_finish(self, segment_id: str, slope_id: str) -> None:
        """Action before committing custom connector and finishing."""
        self.context.building.segments.append(segment_id)
        self.context.viewing.set_slope_id(slope_id=slope_id)
        self.context.custom_connect.clear()

    def before_finish_slope(self, slope_id: str) -> None:
        """Action before finishing a slope."""
        self.context.viewing.set_slope_id(slope_id=slope_id)

    def before_undo_to_idle(self, removed_segment_id: str) -> None:
        """Action before undoing to idle (last segment removed, cancels slope)."""
        if removed_segment_id in self.context.building.segments:
            self.context.building.segments.remove(removed_segment_id)
        self.context.building.clear()

    def before_undo_continue(self, removed_segment_id: str, new_endpoint_node_id: str) -> None:
        """Action before undoing but continuing in building state.

        Note: Does NOT clear proposals - undo_last_action() sets them BEFORE calling this.
        """
        if removed_segment_id in self.context.building.segments:
            self.context.building.segments.remove(removed_segment_id)
        self.context.building.endpoints = [new_endpoint_node_id]

    def before_view_slope(self, slope_id: str) -> None:
        """Set slope_id before entering viewing state. Panel visibility set by enter function."""
        self.context.viewing.set_slope_id(slope_id=slope_id)

    def before_view_lift(self, lift_id: str) -> None:
        """Set lift_id before entering viewing state. Panel visibility set by enter function."""
        self.context.viewing.set_lift_id(lift_id=lift_id)

    def before_switch_to_slope_view(self, slope_id: str) -> None:
        """Set slope_id when switching from lift view. Panel visibility set by enter function."""
        self.context.viewing.set_slope_id(slope_id=slope_id)

    def before_switch_to_lift_view(self, lift_id: str) -> None:
        """Set lift_id when switching from slope view. Panel visibility set by enter function."""
        self.context.viewing.set_lift_id(lift_id=lift_id)

    def before_switch_slope(self, slope_id: str) -> None:
        """Set slope_id for different slope (self-loop). Panel visibility set by enter function."""
        self.context.viewing.set_slope_id(slope_id=slope_id)

    def before_switch_lift(self, lift_id: str) -> None:
        """Set lift_id for different lift (self-loop). Panel visibility set by enter function."""
        self.context.viewing.set_lift_id(lift_id=lift_id)

    def before_close_slope_panel(self) -> None:
        """Before closing slope panel. Panel hidden by enter_idle_ready."""
        pass  # Visibility handled by enter_idle_ready

    def before_close_lift_panel(self) -> None:
        """Before closing lift panel. Panel hidden by enter_idle_ready."""
        pass  # Visibility handled by enter_idle_ready

    def before_start_lift(self, node_id: str | None = None, location: PathPoint | None = None) -> None:
        """Action before starting lift placement."""
        self.context.lift.start_node_id = node_id
        self.context.lift.start_location = location

    # Reuse start_lift logic for other entry points
    before_start_lift_from_slope_view = before_start_lift
    before_start_lift_from_lift_view = before_start_lift

    def before_complete_lift(self, lift_id: str) -> None:
        """Set lift_id before completing. Panel visibility set by enter_idle_viewing_lift."""
        self.context.viewing.set_lift_id(lift_id=lift_id)
        self.context.lift.clear()

    def before_enable_custom_from_starting(self) -> None:
        """Action before enabling custom connect from starting."""
        self.context.custom_connect.enabled = True

    def before_enable_custom_from_building(self) -> None:
        """Action before enabling custom connect from building."""
        self.context.custom_connect.enabled = True

    def before_select_custom_target(self, target_location: LonLatElev) -> None:
        """Action before selecting custom target."""
        self.context.custom_connect.target_location = target_location
        self.context.custom_connect.enabled = False
        self.context.custom_connect.force_mode = True

    def before_resume_to_starting(self) -> None:
        """Action before resuming to starting state (undo finish)."""
        # Building context is restored by the action caller
        pass

    def before_resume_to_building(self) -> None:
        """Action before resuming to building state (undo finish)."""
        # Building context is restored by the action caller
        pass

    def before_undo_restore_to_starting(self) -> None:
        """Action before restoring to starting state from idle (undo finish_slope)."""
        # Building context is restored by the action caller
        pass

    def before_undo_restore_to_building(self) -> None:
        """Action before restoring to building state from idle (undo finish_slope)."""
        # Building context is restored by the action caller
        pass

    def before_undo_restore_from_lift_to_starting(self) -> None:
        """Action before restoring to starting state from lift view (undo finish_slope)."""
        self.context.viewing.clear()

    def before_undo_restore_from_lift_to_building(self) -> None:
        """Action before restoring to building state from lift view (undo finish_slope)."""
        self.context.viewing.clear()

    # ==========================================================================
    # Initialization
    # ==========================================================================

    def __init__(
        self,
        graph: ResortGraph,
        context: PlannerContext | None = None,
        start_value: str | None = None,
    ) -> None:
        """Initialize state machine with model pattern.

        Args:
            graph: ResortGraph instance for accessing slope counter
            context: Shared context/model (creates new if None)
            start_value: Optional initial state value (for restoring state)
        """
        self._resort_graph = graph
        model = context or PlannerContext()
        super().__init__(model=model, start_value=start_value)

    # ==========================================================================
    # Utility Methods
    # ==========================================================================

    @property
    def context(self) -> PlannerContext:
        """Access to the context/model (PlannerContext)."""
        return self.model  # type: ignore[return-value]

    def can_finish_slope(self) -> bool:
        """Check if slope can be finished (in building state with segments)."""
        return self.is_slope_building_only and len(self.context.building.segments) > 0

    def can_undo(self) -> bool:
        """Check if undo is available in current slope state."""
        return self.is_any_slope_state and len(self.context.building.segments) > 0

    def get_state_name(self) -> str:
        """Get current state name for display."""
        return str(self.current_state.name)

    def get_available_actions(self) -> list[str]:
        """Get list of available transition names (for UI display only)."""
        return [t.event for t in self.current_state.transitions]

    def __repr__(self) -> str:
        """Return string representation of state machine."""
        return f"PlannerStateMachine(state={self.get_state_name()}, model={self.context!r})"

    def try_transition(self, event: str, **kwargs: Any) -> bool:
        """Attempt a transition, returning success/failure.

        Args:
            event: Transition event name
            **kwargs: Arguments for transition

        Returns:
            True if transition succeeded, False otherwise.
        """
        try:
            self.send(event=event, **kwargs)
            return True
        except (TransitionNotAllowed, ValueError):
            logger.warning(f"Transition '{event}' not allowed from {self.get_state_name()}")
            return False

    # ==========================================================================
    # Convenience Methods for Common Transitions
    # ==========================================================================

    def start_building(
        self,
        lon: float,
        lat: float,
        elevation: float,
        node_id: str | None = None,
    ) -> None:
        """Start building a slope from any idle state.

        Dispatches to the appropriate transition based on current state.
        """
        if self.is_idle_ready:
            self.start_slope(lon=lon, lat=lat, elevation=elevation, node_id=node_id)
        elif self.is_idle_viewing_slope:
            self.start_slope_from_slope_view(lon=lon, lat=lat, elevation=elevation, node_id=node_id)
        elif self.is_idle_viewing_lift:
            self.start_slope_from_lift_view(lon=lon, lat=lat, elevation=elevation, node_id=node_id)
        else:
            raise ValueError(f"Cannot start building from state {self.get_state_name()}")

    def select_lift_start(self, node_id: str | None = None, location: PathPoint | None = None) -> None:
        """Start placing a lift from any idle state.

        Dispatches to the appropriate transition based on current state.
        """
        if self.is_idle_ready:
            self.start_lift(node_id=node_id, location=location)
        elif self.is_idle_viewing_slope:
            self.start_lift_from_slope_view(node_id=node_id, location=location)
        elif self.is_idle_viewing_lift:
            self.start_lift_from_lift_view(node_id=node_id, location=location)
        else:
            raise ValueError(f"Cannot start lift from state {self.get_state_name()}")

    def commit_segment(self, segment_id: str, endpoint_node_id: str) -> None:
        """Commit a path segment (dispatches to appropriate transition)."""
        if self.is_slope_starting:
            self.commit_first_path(segment_id=segment_id, endpoint_node_id=endpoint_node_id)
        elif self.is_slope_building_only:
            self.commit_path(segment_id=segment_id, endpoint_node_id=endpoint_node_id)
        else:
            raise ValueError(f"Cannot commit segment from state {self.get_state_name()}")

    def show_slope_info_panel(self, slope_id: str) -> None:
        """Show slope info panel using explicit transitions.

        Uses direct transitions:
        - From IDLE_READY: view_slope
        - From IDLE_VIEWING_LIFT: switch_to_slope_view
        - From IDLE_VIEWING_SLOPE: switch_slope (self-loop to switch which slope)
        """
        if self.is_idle_ready:
            self.view_slope(slope_id=slope_id)
        elif self.is_idle_viewing_lift:
            self.switch_to_slope_view(slope_id=slope_id)
        elif self.is_idle_viewing_slope:
            self.switch_slope(slope_id=slope_id)
        else:
            logger.warning(f"Cannot show slope panel from state {self.get_state_name()}")

    def show_lift_info_panel(self, lift_id: str) -> None:
        """Show lift info panel using explicit transitions.

        Uses direct transitions:
        - From IDLE_READY: view_lift
        - From IDLE_VIEWING_SLOPE: switch_to_lift_view
        - From IDLE_VIEWING_LIFT: switch_lift (self-loop to switch which lift)
        """
        if self.is_idle_ready:
            self.view_lift(lift_id=lift_id)
        elif self.is_idle_viewing_slope:
            self.switch_to_lift_view(lift_id=lift_id)
        elif self.is_idle_viewing_lift:
            self.switch_lift(lift_id=lift_id)
        else:
            logger.warning(f"Cannot show lift panel from state {self.get_state_name()}")

    def hide_info_panel(self) -> None:
        """Hide info panel (transitions to idle_ready if viewing)."""
        if self.is_idle_viewing_slope:
            self.close_slope_panel()
        elif self.is_idle_viewing_lift:
            self.close_lift_panel()
        else:
            logger.warning(f"No panel to hide from state {self.get_state_name()}")

    def cancel_slope(self) -> None:
        """Cancel slope building from any slope state.

        Dispatches to the appropriate transition based on current state.
        """
        if self.is_slope_starting:
            self.cancel_from_starting()
        elif self.is_slope_building_only:
            self.cancel_from_building()
        elif self.is_slope_custom_picking:
            if self.has_no_segments():
                self.cancel_custom_to_starting()
                self.cancel_from_starting()
            else:
                self.cancel_custom_to_building()
                self.cancel_from_building()
        elif self.is_slope_custom_path:
            if self.has_no_segments():
                self.cancel_path_to_starting()
                self.cancel_from_starting()
            else:
                self.cancel_path_to_building()
                self.cancel_from_building()
        else:
            raise ValueError(f"Cannot cancel slope from state {self.get_state_name()}")

    def undo_segment(self, removed_segment_id: str, new_endpoint_node_id: str | None = None) -> None:
        """Undo last segment, transitioning appropriately.

        Args:
            removed_segment_id: The segment being removed
            new_endpoint_node_id: The new endpoint after undo (None if undoing to start)
        """
        if self.is_slope_building_only:
            if self.undo_leaves_no_segments():
                self.undo_to_idle(removed_segment_id=removed_segment_id)
            else:
                if new_endpoint_node_id is None:
                    raise ValueError("new_endpoint_node_id required when not undoing to starting")
                self.undo_continue(removed_segment_id=removed_segment_id, new_endpoint_node_id=new_endpoint_node_id)
        else:
            raise ValueError(f"Cannot undo from state {self.get_state_name()}")

    def enable_custom_connect(self) -> None:
        """Enable custom connect mode from current slope state."""
        if self.is_slope_starting:
            self.enable_custom_from_starting()
        elif self.is_slope_building_only:
            self.enable_custom_from_building()
        else:
            raise ValueError(f"Cannot enable custom connect from state {self.get_state_name()}")

    def cancel_custom_connect(self) -> None:
        """Cancel custom connect mode, returning to previous slope state."""
        if self.is_slope_custom_picking:
            if self.has_no_segments():
                self.cancel_custom_to_starting()
            else:
                self.cancel_custom_to_building()
        elif self.is_slope_custom_path:
            if self.has_no_segments():
                self.cancel_path_to_starting()
            else:
                self.cancel_path_to_building()
        else:
            raise ValueError(f"Cannot cancel custom from state {self.get_state_name()}")

    @staticmethod
    def create(
        graph: ResortGraph,
        add_ui_listener: bool = True,
    ) -> tuple["PlannerStateMachine", PlannerContext]:
        """Factory method to create state machine with context and optional UI listener.

        Args:
            graph: ResortGraph instance for accessing slope counter
            add_ui_listener: If True, adds StreamlitUIListener for auto st.rerun().
                             Set to False for testing or non-Streamlit usage.

        Returns:
            Tuple of (PlannerStateMachine, PlannerContext)
        """
        context = PlannerContext()
        sm = PlannerStateMachine(graph=graph, context=context)
        if add_ui_listener:
            sm.add_listener(StreamlitUIListener())  # type: ignore[no-untyped-call]
            logger.info("Created PlannerStateMachine with StreamlitUIListener")
        else:
            logger.info("Created PlannerStateMachine without UI listener")
        return sm, context

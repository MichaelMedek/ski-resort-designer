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
2. StreamlitUIListener fires after_transition and calls st.rerun()
3. On the next render cycle, handle_fast_deferred_actions() checks pending flags
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

Undo Architecture (Meta-Feature - NOT State Machine Transitions)
================================================================
Undo is handled as a META-FEATURE at the action layer, NOT as state machine transitions.
This simplifies the state machine and separates concerns:

    State Machine: Manages WORKFLOW states (what the user is doing NOW)
    Action Layer:  Manages HISTORY (undo/redo stack, restoring previous states)

When undo is triggered:
    1. Action layer (_exit_active_mode_for_undo) cancels any active mode (custom picking, lift placing)
    2. History manager reverts the graph changes
    3. Action layer uses force_idle() or force_building() to set the target state
    4. These force methods BYPASS the state machine transitions entirely

Available force methods:
    - force_idle(): Jump to IdleReady, clearing all building/viewing state
    - force_building(): Jump to SlopeBuilding, preserving building context

This design means:
    - No undo transitions in the state machine (simpler, fewer edge cases)
    - Undo can work from ANY state (not limited by transition definitions)
    - Action layer has full control over compound undo operations

Events Reference (API for UI/Actions layer)
============================================
Events are the external API for triggering state transitions. The state machine
resolves which specific transition fires based on current state and guards.

IMPORTANT: Direct transition calls are BLOCKED at runtime via __getattribute__.
           Only event calls are allowed:

           sm.commit_path(...)          # allowed
           sm.send("commit_path", ...)  # allowed
           sm.commit_first_path(...)    # raises RuntimeError

    commit_path - Commit a path segment
        Args: segment_id, endpoint_node_id
        Hook: before_commit_path (event-level only)
        Resolves to:
        - commit_first_path: SLOPE_STARTING → SLOPE_BUILDING
        - commit_continue_path: SLOPE_BUILDING → SLOPE_BUILDING (self-loop)

    cancel_slope - Cancel entire slope building
        Args: none
        Hook: before_cancel_slope (event-level only)
        Resolves to:
        - cancel_from_starting: SLOPE_STARTING → IDLE_READY
        - cancel_from_building: SLOPE_BUILDING → IDLE_READY
        - cancel_slope_from_custom_picking: SLOPE_CUSTOM_PICKING → IDLE_READY
        - cancel_slope_from_custom_path: SLOPE_CUSTOM_PATH → IDLE_READY

    cancel_custom_connect - Cancel custom connect mode (return to normal building)
        Args: none
        Hook: before_cancel_custom_connect (event-level only)
        Guards: has_no_segments
        Resolves to:
        - cancel_custom_to_starting: SLOPE_CUSTOM_PICKING → SLOPE_STARTING
        - cancel_custom_to_building: SLOPE_CUSTOM_PICKING → SLOPE_BUILDING
        - cancel_path_to_starting: SLOPE_CUSTOM_PATH → SLOPE_STARTING
        - cancel_path_to_building: SLOPE_CUSTOM_PATH → SLOPE_BUILDING

    enable_custom - Enable custom connect mode
        Args: none
        Hook: before_enable_custom (event-level), plus transition-specific hooks
        Resolves to:
        - enable_custom_from_starting: SLOPE_STARTING → SLOPE_CUSTOM_PICKING
        - enable_custom_from_building: SLOPE_BUILDING → SLOPE_CUSTOM_PICKING

Complete Transition Matrix (8x8 = 64 combinations)
==================================================

# 1. Transitions: From IDLE_READY
# --------------------------------
# 1.1. → IDLE_READY: NOT ALLOWED (no-op, nothing to transition)
# 1.2. → IDLE_VIEWING_SLOPE: view_slope [direct] (click slope icon/centerline)
# 1.3. → IDLE_VIEWING_LIFT: view_lift [direct] (click lift icon/cable)
# 1.4. → SLOPE_STARTING: start_slope [direct] (click terrain/node in slope mode)
# 1.5. → SLOPE_BUILDING: NOT ALLOWED (must go through SLOPE_STARTING first)
# 1.6. → SLOPE_CUSTOM_PICKING: NOT ALLOWED (must go through SLOPE_STARTING first)
# 1.7. → SLOPE_CUSTOM_PATH: NOT ALLOWED (must go through SLOPE_CUSTOM_PICKING first)
# 1.8. → LIFT_PLACING: start_lift [direct] (click terrain/node in lift mode)

# 2. Transitions: From IDLE_VIEWING_SLOPE
# ----------------------------------------
# 2.1. → IDLE_READY: close_slope_panel [direct] (close button or click elsewhere)
# 2.2. → IDLE_VIEWING_SLOPE: switch_slope [direct, self-loop] (click different slope)
# 2.3. → IDLE_VIEWING_LIFT: switch_to_lift_view [direct] (click lift in panel or on map)
# 2.4. → SLOPE_STARTING: start_slope_from_slope_view [direct] (click terrain/node)
# 2.5. → SLOPE_BUILDING: NOT ALLOWED (must go through SLOPE_STARTING first)
# 2.6. → SLOPE_CUSTOM_PICKING: NOT ALLOWED (must start slope first)
# 2.7. → SLOPE_CUSTOM_PATH: NOT ALLOWED (must go through picking first)
# 2.8. → LIFT_PLACING: start_lift_from_slope_view [direct] (click terrain/node in lift mode)

# 3. Transitions: From IDLE_VIEWING_LIFT
# ---------------------------------------
# 3.1. → IDLE_READY: close_lift_panel [direct] (close button or click elsewhere)
# 3.2. → IDLE_VIEWING_SLOPE: switch_to_slope_view [direct] (click connected slope in panel)
# 3.3. → IDLE_VIEWING_LIFT: switch_lift [direct, self-loop] (click different lift)
# 3.4. → SLOPE_STARTING: start_slope_from_lift_view [direct] (click terrain/node)
# 3.5. → SLOPE_BUILDING: NOT ALLOWED (must go through SLOPE_STARTING first)
# 3.6. → SLOPE_CUSTOM_PICKING: NOT ALLOWED (must start slope first)
# 3.7. → SLOPE_CUSTOM_PATH: NOT ALLOWED (must go through picking first)
# 3.8. → LIFT_PLACING: start_lift_from_lift_view [direct] (click terrain/node in lift mode)

# 4. Transitions: From SLOPE_STARTING
# ------------------------------------
# 4.1. → IDLE_READY: cancel_from_starting [event: cancel_slope] (cancel button)
# 4.2. → IDLE_VIEWING_SLOPE: NOT ALLOWED (must commit or cancel first)
# 4.3. → IDLE_VIEWING_LIFT: NOT ALLOWED (must commit or cancel first)
# 4.4. → SLOPE_STARTING: NOT ALLOWED (no self-loop, proposal selection is internal)
# 4.5. → SLOPE_BUILDING: commit_first_path [event: commit_path] (click proposal endpoint)
# 4.6. → SLOPE_CUSTOM_PICKING: enable_custom_from_starting [event: enable_custom]
# 4.7. → SLOPE_CUSTOM_PATH: NOT ALLOWED (must go through picking first)
# 4.8. → LIFT_PLACING: NOT ALLOWED (must cancel slope first)

# 5. Transitions: From SLOPE_BUILDING
# ------------------------------------
# 5.1. → IDLE_READY: cancel_from_building [event: cancel_slope] (cancel button)
# 5.2. → IDLE_VIEWING_SLOPE: finish_slope [direct] (finish button)
# 5.3. → IDLE_VIEWING_LIFT: NOT ALLOWED (must finish/cancel slope first)
# 5.4. → SLOPE_STARTING: NOT ALLOWED (would lose committed segments)
# 5.5. → SLOPE_BUILDING: commit_continue_path [event: commit_path, self-loop]
# 5.6. → SLOPE_CUSTOM_PICKING: enable_custom_from_building [event: enable_custom]
# 5.7. → SLOPE_CUSTOM_PATH: NOT ALLOWED (must go through picking first)
# 5.8. → LIFT_PLACING: NOT ALLOWED (must finish/cancel slope first)

# 6. Transitions: From SLOPE_CUSTOM_PICKING
# ------------------------------------------
# 6.1. → IDLE_READY: cancel_slope_from_custom_picking [event: cancel_slope]
# 6.2. → IDLE_VIEWING_SLOPE: NOT ALLOWED (must cancel or select first)
# 6.3. → IDLE_VIEWING_LIFT: NOT ALLOWED (must cancel or select first)
# 6.4. → SLOPE_STARTING: cancel_custom_to_starting [event: cancel_custom_connect, guard: has_no_segments]
# 6.5. → SLOPE_BUILDING: cancel_custom_to_building [event: cancel_custom_connect, guard: !has_no_segments]
# 6.6. → SLOPE_CUSTOM_PICKING: NOT ALLOWED (waiting for click, no self-loop)
# 6.7. → SLOPE_CUSTOM_PATH: select_custom_target [direct] (click target location)
# 6.8. → LIFT_PLACING: NOT ALLOWED (must cancel first)

# 7. Transitions: From SLOPE_CUSTOM_PATH
# ---------------------------------------
# 7.1. → IDLE_READY: cancel_slope_from_custom_path [event: cancel_slope]
# 7.2. → IDLE_VIEWING_SLOPE: commit_custom_finish [direct] (auto-finish when connecting to node)
# 7.3. → IDLE_VIEWING_LIFT: NOT ALLOWED (must finish/cancel slope first)
# 7.4. → SLOPE_STARTING: cancel_path_to_starting [event: cancel_custom_connect, guard: has_no_segments]
# 7.5. → SLOPE_BUILDING: commit_custom_continue [direct] (commit and keep building),
#                        cancel_path_to_building [event: cancel_custom_connect, guard: !has_no_segments]
# 7.6. → SLOPE_CUSTOM_PICKING: NOT ALLOWED (can't go back to picking, must cancel)
# 7.7. → SLOPE_CUSTOM_PATH: NOT ALLOWED (no self-loop, different target → cancel+repick)
# 7.8. → LIFT_PLACING: NOT ALLOWED (must finish/cancel slope first)

# 8. Transitions: From LIFT_PLACING
# ----------------------------------
# 8.1. → IDLE_READY: cancel_lift [direct] (cancel button)
# 8.2. → IDLE_VIEWING_SLOPE: NOT ALLOWED (must cancel or complete first)
# 8.3. → IDLE_VIEWING_LIFT: complete_lift [direct] (click end station location)
# 8.4. → SLOPE_STARTING: NOT ALLOWED (must cancel lift first)
# 8.5. → SLOPE_BUILDING: NOT ALLOWED (must cancel lift first)
# 8.6. → SLOPE_CUSTOM_PICKING: NOT ALLOWED (must cancel lift first)
# 8.7. → SLOPE_CUSTOM_PATH: NOT ALLOWED (must cancel lift first)
# 8.8. → LIFT_PLACING: NOT ALLOWED (no self-loop needed)

Transition Summary Table
------------------------
    ALLOWED (20 transitions + 3 self-loops = 23 total):
    - From IDLE_READY (4): view_slope, view_lift, start_slope, start_lift [all direct]
    - From IDLE_VIEWING_SLOPE (4+1): close, switch_to_lift, start_slope, start_lift [all direct], switch_slope (loop)
    - From IDLE_VIEWING_LIFT (4+1): close, switch_to_slope, start_slope, start_lift [all direct], switch_lift (loop)
    - From SLOPE_STARTING (3): cancel [cancel_slope], commit_first_path [commit_path], enable_custom [enable_custom]
    - From SLOPE_BUILDING (3+1): cancel [cancel_slope], finish [direct], enable_custom [enable_custom], commit_path (loop)
    - From SLOPE_CUSTOM_PICKING (4): cancel_slope [cancel_slope], cancel_custom_to_* [cancel_custom_connect], select_target [direct]
    - From SLOPE_CUSTOM_PATH (5): commit_continue [direct], commit_finish [direct], cancel_slope [cancel_slope], cancel_path_to_* [cancel_custom_connect]
    - From LIFT_PLACING (2): cancel [direct], complete [direct]

    Event-triggered transitions use [event_name] notation.
    Direct transitions are called by their transition name directly.

    NOT ALLOWED (41 combinations): All other transitions that would bypass required workflow steps

    NOTE: Undo is handled via force_idle()/force_building() methods, NOT via transitions.
          See "Undo Architecture" section above.

Cleanup Policy
--------------
Orphaned node cleanup is NOT called on every transition. Instead, cleanup_isolated_nodes()
is called explicitly when entities are removed:
- undo ADD_SEGMENTS (segment deleted)
- undo ADD_LIFT (lift deleted)
- delete_slope (slope and segments deleted)
- delete_lift (lift deleted)
- cancel_current_slope (building segments discarded)

This prevents premature deletion of nodes that are still needed (e.g., start node
in custom connect mode from SlopeStarting state, before any segment is committed).
"""

from __future__ import annotations

import logging
from typing import Any, Callable

import streamlit as st
from statemachine import State, StateMachine
from statemachine.exceptions import TransitionNotAllowed

from skiresort_planner.model.path_point import PathPoint
from skiresort_planner.model.resort_graph import ResortGraph
from skiresort_planner.ui.context import (
    BuildMode,
    LonLatElev,
    PlannerContext,
)
from skiresort_planner.ui.infra import trigger_rerun
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

        Supports deferred rerun for compound operations (e.g., undo from custom state).
        When _defer_rerun flag is set in session_state, the rerun is skipped to allow
        multiple state transitions before a single UI refresh.
        """
        logger.info(f"[STATE] {source.name} --({event})--> {target.name}")

        # NOTE: Orphaned node cleanup is NOT called here. It's called explicitly
        # in operations that remove entities (undo, delete, cancel). This prevents
        # premature deletion of nodes still in use (e.g., start nodes in custom
        # connect mode before any segment is committed).

        # Check if rerun should be deferred (used during compound operations)
        if st.session_state.get("_defer_rerun"):
            logger.info(f'[STATE] Deferring st.rerun() after {event} transition (compound operation)"')
            return

        logger.info(f'[STATE] Calling st.rerun() after {event} transition"')
        trigger_rerun()


def _forbidden_call(name: str):
    """Create a function that raises RuntimeError when called.

    Used to block direct calls to event-triggered transitions.
    """

    def wrapper(*args, **kwargs):
        raise RuntimeError(
            f"Direct transition '{name}' call forbidden. Use the corresponding event instead (e.g., sm.event_name())."
        )

    return wrapper


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
    # Events: start_slope, start_lift, view_slope, view_lift
    # 1.2. view_slope [event: view_slope]: Click slope icon/centerline to view details
    # 1.3. view_lift [event: view_lift]: Click lift icon/cable to view details
    # 1.4. start_slope [event: start_slope]: Click terrain/node in slope mode
    # 1.8. start_lift [event: start_lift]: Click terrain/node in lift mode

    start_slope = idle_ready.to(slope_starting, event="start_slope")  # 1.4 [event: start_slope]
    start_lift = idle_ready.to(lift_placing, event="start_lift")  # 1.8 [event: start_lift]
    view_slope = idle_ready.to(idle_viewing_slope, event="view_slope")  # 1.2 [event: view_slope]
    view_lift = idle_ready.to(idle_viewing_lift, event="view_lift")  # 1.3 [event: view_lift]

    # ==========================================================================
    # 2. Transitions: From IDLE_VIEWING_SLOPE
    # ==========================================================================
    # Events: close_panel, view_slope (self-loop), view_lift, start_slope, start_lift
    # 2.1. close_slope_panel [event: close_panel]: Close button or click elsewhere
    # 2.2. switch_slope [event: view_slope, self-loop]: Click different slope
    # 2.3. switch_to_lift_view [event: view_lift]: Click lift in panel or on map
    # 2.4. start_slope_from_slope_view [event: start_slope]: Click terrain/node to start new slope
    # 2.8. start_lift_from_slope_view [event: start_lift]: Click terrain/node in lift mode

    close_slope_panel = idle_viewing_slope.to(idle_ready, event="close_panel")  # 2.1 [event: close_panel]
    switch_slope = idle_viewing_slope.to(idle_viewing_slope, event="view_slope")  # 2.2 [event: view_slope] self-loop
    switch_to_lift_view = idle_viewing_slope.to(idle_viewing_lift, event="view_lift")  # 2.3 [event: view_lift]
    start_slope_from_slope_view = idle_viewing_slope.to(slope_starting, event="start_slope")  # 2.4 [event: start_slope]
    start_lift_from_slope_view = idle_viewing_slope.to(lift_placing, event="start_lift")  # 2.8 [event: start_lift]

    # ==========================================================================
    # 3. Transitions: From IDLE_VIEWING_LIFT
    # ==========================================================================
    # Events: close_panel, view_slope, view_lift (self-loop), start_slope, start_lift
    # 3.1. close_lift_panel [event: close_panel]: Close button or click elsewhere
    # 3.2. switch_to_slope_view [event: view_slope]: Click connected slope in panel
    # 3.3. switch_lift [event: view_lift, self-loop]: Click different lift
    # 3.4. start_slope_from_lift_view [event: start_slope]: Click terrain/node in slope mode
    # 3.8. start_lift_from_lift_view [event: start_lift]: Click terrain/node in lift mode

    close_lift_panel = idle_viewing_lift.to(idle_ready, event="close_panel")  # 3.1 [event: close_panel]
    switch_to_slope_view = idle_viewing_lift.to(idle_viewing_slope, event="view_slope")  # 3.2 [event: view_slope]
    switch_lift = idle_viewing_lift.to(idle_viewing_lift, event="view_lift")  # 3.3 [event: view_lift] self-loop
    start_slope_from_lift_view = idle_viewing_lift.to(slope_starting, event="start_slope")  # 3.4 [event: start_slope]
    start_lift_from_lift_view = idle_viewing_lift.to(lift_placing, event="start_lift")  # 3.8 [event: start_lift]

    # ==========================================================================
    # 4. Transitions: From SLOPE_STARTING (0 segments)
    # ==========================================================================
    # Events available: commit_path, cancel_slope, enable_custom
    # 4.1. cancel_from_starting [event: cancel_slope]: Cancel button
    # 4.5. commit_first_path [event: commit_path]: Click proposal endpoint
    # 4.6. enable_custom_from_starting [event: enable_custom]: Custom connect button

    commit_first_path = slope_starting.to(slope_building, event="commit_path")  # 4.5 [event: commit_path]
    cancel_from_starting = slope_starting.to(idle_ready, event="cancel_slope")  # 4.1 [event: cancel_slope]
    enable_custom_from_starting = slope_starting.to(
        slope_custom_picking, event="enable_custom", before="_enable_custom_from_starting"
    )  # 4.6 [event: enable_custom]

    # ==========================================================================
    # 5. Transitions: From SLOPE_BUILDING (1+ segments)
    # ==========================================================================
    # Events available: commit_path, cancel_slope, enable_custom
    # NOTE: Undo is handled via force_idle()/force_building(), NOT transitions
    # 5.1. cancel_from_building [event: cancel_slope]: Cancel button (discard all)
    # 5.2. finish_slope [direct]: Finish button
    # 5.5. commit_continue_path [event: commit_path, self-loop]: Commit more segments
    # 5.6. enable_custom_from_building [event: enable_custom]: Custom connect button

    commit_continue_path = slope_building.to(slope_building, event="commit_path")  # 5.5 [event: commit_path] self-loop
    finish_slope = slope_building.to(idle_viewing_slope)  # 5.2 [direct]
    cancel_from_building = slope_building.to(idle_ready, event="cancel_slope")  # 5.1 [event: cancel_slope]
    enable_custom_from_building = slope_building.to(
        slope_custom_picking, event="enable_custom", before="_enable_custom_from_building"
    )  # 5.6 [event: enable_custom]

    # ==========================================================================
    # 6. Transitions: From SLOPE_CUSTOM_PICKING
    # ==========================================================================
    # Events available: cancel_custom, cancel_slope
    # 6.1. cancel_slope_from_custom_picking [event: cancel_slope]: Cancel entire slope
    # 6.4. cancel_custom_to_starting [event: cancel_custom, guard]: Cancel when has_no_segments
    # 6.5. cancel_custom_to_building [event: cancel_custom, guard]: Cancel when has segments
    # 6.7. select_custom_target [direct]: Click target location

    select_custom_target = slope_custom_picking.to(slope_custom_path)  # 6.7 [direct]
    cancel_custom_to_starting = slope_custom_picking.to(
        slope_starting, cond="has_no_segments", event="cancel_custom"
    )  # 6.4 [event: cancel_custom, guard]
    cancel_custom_to_building = slope_custom_picking.to(
        slope_building, unless="has_no_segments", event="cancel_custom"
    )  # 6.5 [event: cancel_custom, guard]
    cancel_slope_from_custom_picking = slope_custom_picking.to(
        idle_ready, event="cancel_slope"
    )  # 6.1 [event: cancel_slope]

    # ==========================================================================
    # 7. Transitions: From SLOPE_CUSTOM_PATH
    # ==========================================================================
    # Events available: cancel_custom, cancel_slope
    # 7.1. cancel_slope_from_custom_path [event: cancel_slope]: Cancel entire slope
    # 7.2. commit_custom_finish [direct]: Auto-finish when connecting to existing node
    # 7.4. cancel_path_to_starting [event: cancel_custom, guard]: Cancel when has_no_segments
    # 7.5. commit_custom_continue [direct]: Commit and keep building
    # 7.5. cancel_path_to_building [event: cancel_custom, guard]: Cancel when has segments

    commit_custom_continue = slope_custom_path.to(slope_building)  # 7.5 [direct]
    commit_custom_finish = slope_custom_path.to(idle_viewing_slope)  # 7.2 [direct] auto-finish connector
    cancel_path_to_starting = slope_custom_path.to(
        slope_starting, cond="has_no_segments", event="cancel_custom"
    )  # 7.4 [event: cancel_custom, guard]
    cancel_path_to_building = slope_custom_path.to(
        slope_building, unless="has_no_segments", event="cancel_custom"
    )  # 7.5 [event: cancel_custom, guard]
    cancel_slope_from_custom_path = slope_custom_path.to(idle_ready, event="cancel_slope")  # 7.1 [event: cancel_slope]

    # ==========================================================================
    # 8. Transitions: From LIFT_PLACING
    # ==========================================================================
    # All transitions from LIFT_PLACING are direct (no shared events)
    # 8.1. cancel_lift [direct]: Cancel button
    # 8.3. complete_lift [direct]: Click end station location

    complete_lift = lift_placing.to(idle_viewing_lift)  # 8.3 [direct]
    cancel_lift = lift_placing.to(idle_ready)  # 8.1 [direct]

    # ==========================================================================
    # Guards (Conditions)
    # ==========================================================================

    def has_no_segments(self) -> bool:
        """Guard: Check if there are no committed segments."""
        return len(self.context.building.segments) == 0

    # ==========================================================================
    # Event-Only Access Control
    # ==========================================================================
    # Block direct calls to event-triggered transitions. Only allow event calls.
    # This prevents bypassing the event dispatch mechanism.
    #
    # Example:
    #   sm.commit_path(...)     # allowed (event)
    #   sm.commit_first_path()  # raises RuntimeError

    _EVENT_ONLY_TRANSITIONS: frozenset[str] = frozenset(
        {
            # commit_path event
            "commit_first_path",
            "commit_continue_path",
            # cancel_slope event
            "cancel_from_starting",
            "cancel_from_building",
            "cancel_slope_from_custom_picking",
            "cancel_slope_from_custom_path",
            # cancel_custom_connect event
            "cancel_custom_to_starting",
            "cancel_custom_to_building",
            "cancel_path_to_starting",
            "cancel_path_to_building",
            # enable_custom event
            "enable_custom_from_starting",
            "enable_custom_from_building",
            # start_slope event (NOT start_slope - that IS the event entry point)
            "start_slope_from_slope_view",
            "start_slope_from_lift_view",
            # start_lift event (NOT start_lift - that IS the event entry point)
            "start_lift_from_slope_view",
            "start_lift_from_lift_view",
            # view_slope event (NOT view_slope - that IS the event entry point)
            "switch_to_slope_view",
            "switch_slope",
            # view_lift event (NOT view_lift - that IS the event entry point)
            "switch_to_lift_view",
            "switch_lift",
            # close_panel event (both are variants, event is "close_panel")
            "close_slope_panel",
            "close_lift_panel",
        }
    )

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

    def _add_segment_to_building(self, segment_id: str, endpoint_node_id: str) -> None:
        """Common logic for adding segment to building context."""
        self.context.building.segments.append(segment_id)
        self.context.building.endpoints = [endpoint_node_id]
        self.context.clear_proposals()

    def before_commit_path(self, segment_id: str, endpoint_node_id: str) -> None:
        """Action before committing a path segment (event hook only)."""
        self._add_segment_to_building(segment_id=segment_id, endpoint_node_id=endpoint_node_id)

    def before_commit_custom_continue(self, segment_id: str, endpoint_node_id: str) -> None:
        """Action before committing custom path and continuing."""
        self.context.building.segments.append(segment_id)
        self.context.building.endpoints = [endpoint_node_id]
        self.context.clear_proposals()
        self.context.custom_connect.clear()

    def before_commit_custom_finish(self, segment_id: str, slope_id: str) -> None:
        """Action before committing custom connector and finishing.

        Note: segment_id may already be in building.segments if added before
        calling graph.finish_slope(). This hook is idempotent.
        """
        if segment_id not in self.context.building.segments:
            self.context.building.segments.append(segment_id)
        self.context.viewing.set_slope_id(slope_id=slope_id)
        self.context.custom_connect.clear()

    def before_finish_slope(self, slope_id: str) -> None:
        """Action before finishing a slope."""
        self.context.viewing.set_slope_id(slope_id=slope_id)

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

    # ──────────────────────────────────────────────────────────────────────────────
    # Custom Connect Transitions (Single Source of Truth for ctx.custom_connect.*)
    # ──────────────────────────────────────────────────────────────────────────────
    # All custom_connect state mutations happen ONLY in these hooks:
    # - _enable_custom_from_*: Attached via before= in .to() for path-specific logic
    # - select_custom_target: Sets target_location, enabled=False, force_mode=True
    # - cancel_custom/cancel_path: Clears state via clear_custom_connect() or field resets
    # ──────────────────────────────────────────────────────────────────────────────

    def _enable_custom_from_starting(self) -> None:
        """Transition action: From STARTING, get or create start node, enable custom connect.

        Attached via before= parameter to enable_custom_from_starting transition.
        """
        start_node_id = self.context.building.start_node
        if start_node_id is None:
            sel = self.context.selection
            node, _ = self._resort_graph.get_or_create_node(lon=sel.lon, lat=sel.lat, elevation=sel.elevation)
            start_node_id = node.id
            self.context.building.start_node = start_node_id
        self.context.custom_connect.enabled = True
        self.context.custom_connect.start_node = start_node_id

    def _enable_custom_from_building(self) -> None:
        """Transition action: From BUILDING, use current endpoint, enable custom connect.

        Attached via before= parameter to enable_custom_from_building transition.
        """
        self.context.custom_connect.enabled = True
        if self.context.building.endpoints:
            self.context.custom_connect.start_node = self.context.building.endpoints[0]

    def before_select_custom_target(self, target_location: LonLatElev) -> None:
        """Action before selecting custom target."""
        self.context.custom_connect.target_location = target_location
        self.context.custom_connect.enabled = False
        self.context.custom_connect.force_mode = True

    def before_cancel_custom(self) -> None:
        """Event hook for cancel_custom. Clears custom state and triggers path regeneration."""
        self.context.clear_custom_connect()
        self.context.clear_proposals()
        self.context.deferred.path_generation = True

    def before_cancel_slope(self) -> None:
        """Event hook for cancel_slope. Clears all building and custom state."""
        self.context.clear_custom_connect()
        self.context.clear_proposals()
        self.context.building.clear()

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

        # Block direct calls to variant transitions (setattr is more performant
        # than __getattribute__ and doesn't interfere with library internals)
        for trans_name in PlannerStateMachine._EVENT_ONLY_TRANSITIONS:
            setattr(self, trans_name, _forbidden_call(trans_name))

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

    # ==========================================================================
    # Force State Methods (for Undo - bypasses transitions)
    # ==========================================================================
    # These methods allow the action layer to reset the state machine to a
    # stable state after graph undo operations. This follows the expert
    # recommendation to treat undo as a "meta-feature" (history management)
    # rather than core workflow state transitions.

    # Map state names to their exit hooks (for dynamic dispatch)
    _EXIT_HOOKS: dict[str, Callable[[PlannerContext], None]] = {
        "idle_ready": exit_idle_ready,
        "idle_viewing_slope": exit_idle_viewing_slope,
        "idle_viewing_lift": exit_idle_viewing_lift,
        "slope_starting": exit_slope_starting,
        "slope_building": exit_slope_building,
        "slope_custom_picking": exit_slope_custom_picking,
        "slope_custom_path": exit_slope_custom_path,
        "lift_placing": exit_lift_placing,
    }

    def force_idle(self) -> None:
        """Force state machine to IdleReady state without transition.

        Used after undo operations when no building context remains.
        Clears all building, custom, and viewing state.
        Does NOT trigger st.rerun() - caller is responsible for UI refresh.
        """
        logger.info(f"[STATE] Forcing state from {self.get_state_name()} to IdleReady")
        # Clear all context state (state-specific cleanup via exit hook in _set_current_state)
        self.context.building.clear()
        self.context.clear_custom_connect()
        self.context.clear_proposals()
        self.context.viewing.clear()
        # Force state machine internal state (calls exit hook for current state)
        self._set_current_state(state=self.idle_ready)
        # Run entry hook to ensure consistent state
        enter_idle_ready(self.context)

    def force_building(self) -> None:
        """Force state machine to SlopeBuilding state without transition.

        Used after undo operations when building context should be restored.
        Assumes caller has already set up ctx.building with the restored segments.
        Does NOT trigger st.rerun() - caller is responsible for UI refresh.
        """
        logger.info(f"[STATE] Forcing state from {self.get_state_name()} to SlopeBuilding")
        # Clear non-building state (state-specific cleanup via exit hook in _set_current_state)
        self.context.clear_custom_connect()
        self.context.viewing.clear()
        # Force state machine internal state (calls exit hook for current state)
        self._set_current_state(state=self.slope_building)
        # Run entry hook to ensure consistent state
        enter_slope_building(self.context)

    def _set_current_state(self, state: State) -> None:
        """Force state change with proper exit hook lifecycle.

        Implements the 'Safe Dynamic Exit' pattern per expert recommendation:
        1. Call exit hook for CURRENT state (dynamic dispatch)
        2. Set the new state value (in finally block - MUST happen)

        The try-finally ensures the state change ALWAYS happens even if the
        exit hook raises an exception. This prevents the app from getting
        stuck in an inconsistent state.

        Important: This method bypasses the normal transition mechanism and should only be used for undo operations!
                   Also the method does only handle exit hooks, but entry hooks must be called separately by the caller after setting the state.

        Raises:
            KeyError: If current state has no exit hook registered in _EXIT_HOOKS. Adding a new state requires adding its hook.
        """
        # Use .value (snake_case identifier) not .name (CamelCase display name)
        current_state_value = str(self.current_state.value)
        # Direct access - raises KeyError if state not in _EXIT_HOOKS (fail fast)
        exit_hook = PlannerStateMachine._EXIT_HOOKS[current_state_value]

        try:
            # 1. Dynamic exit hook dispatch for current state
            logger.info(f"[STATE] Calling exit_{current_state_value} before force")
            exit_hook(self.context)
        except Exception as e:
            # Log but don't block - availability over perfect cleanup
            logger.error(f"[STATE] Exit hook exit_{current_state_value} failed during force: {e}")
        finally:
            # 2. State change MUST happen regardless of exit hook success
            setattr(self.model, self.state_field, state.value)

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

        Uses start_slope event - SM resolves to appropriate transition.
        """
        self.start_slope(lon=lon, lat=lat, elevation=elevation, node_id=node_id)

    def select_lift_start(self, node_id: str | None = None, location: PathPoint | None = None) -> None:
        """Start placing a lift from any idle state.

        Uses start_lift event - SM resolves to appropriate transition.
        """
        self.start_lift(node_id=node_id, location=location)

    def commit_segment(self, segment_id: str, endpoint_node_id: str) -> None:
        """Commit a path segment. SM resolves to commit_first_path or commit_continue_path."""
        self.commit_path(segment_id=segment_id, endpoint_node_id=endpoint_node_id)

    def show_slope_info_panel(self, slope_id: str) -> None:
        """Show slope info panel from any idle state.

        Uses view_slope event - SM resolves to appropriate transition.
        """
        self.view_slope(slope_id=slope_id)

    def show_lift_info_panel(self, lift_id: str) -> None:
        """Show lift info panel from any idle state.

        Uses view_lift event - SM resolves to appropriate transition.
        """
        self.view_lift(lift_id=lift_id)

    def hide_info_panel(self) -> None:
        """Hide info panel (transitions to idle_ready if viewing).

        Uses close_panel event - SM resolves to appropriate transition.
        """
        self.close_panel()

    # NOTE: No restore_building() wrapper - call sm.restore_building() event directly.
    # The event is defined by transitions with event="restore_building" parameter.

    def cancel_slope(self) -> None:
        """Cancel slope building from any slope state. SM resolves transition atomically."""
        self.send("cancel_slope")

    # NOTE: undo_segment() removed - undo handled via force_idle()/force_building()

    def enable_custom_connect(self) -> None:
        """Enable custom connect mode. SM resolves based on current state."""
        self.enable_custom()

    def cancel_custom_connect(self) -> None:
        """Cancel custom connect mode. SM resolves based on current state and guards."""
        self.cancel_custom()

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

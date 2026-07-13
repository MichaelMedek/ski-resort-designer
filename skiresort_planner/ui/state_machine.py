"""State machine for ski resort planner UI.

Uses python-statemachine for robust state management with:
- Clear state definitions (all explicit states)
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

States (all explicit states)
--------------------------
    1. IDLE_READY: No panel visible, ready to start building
    2. IDLE_VIEWING_SLOPE: Panel showing slope details (3D toggle available)
    3. IDLE_VIEWING_LIFT: Panel showing lift details (3D toggle available)
    4. IDLE_VIEWING_ROAD: Panel showing road details (3D toggle available)
    5. SLOPE_STARTING: 0 segments committed, picking first direction
    6. SLOPE_BUILDING: 1+ segments committed, continuing slope
    7. SLOPE_CUSTOM_PATH: Showing custom path options routed to a clicked target
    8. LIFT_PLACING: Start selected, waiting for end station
    9. ROAD_STARTING: 0 road segments committed, picking first target
    10. ROAD_BUILDING: 1+ road segments committed, extending the road

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
        - cancel_slope_from_custom_path: SLOPE_CUSTOM_PATH → IDLE_READY

    cancel_custom - Leave custom targeting, return to fan-out proposals
        Args: none
        Hook: before_cancel_custom (event-level only, regenerates the fan)
        Guards: has_no_segments
        Resolves to:
        - cancel_path_to_starting: SLOPE_CUSTOM_PATH → SLOPE_STARTING
        - cancel_path_to_building: SLOPE_CUSTOM_PATH → SLOPE_BUILDING
        (The cancel_custom_connect() method is a thin wrapper that sends this event.)

    select_custom_target - Route custom-connect proposals to a clicked target
        Args: target_location, target_node
        Hook: per-transition before hooks (NO event-level hook, to avoid double-fire)
        Resolves to (based on active state):
        - select_target_from_starting: SLOPE_STARTING → SLOPE_CUSTOM_PATH
        - select_target_from_building: SLOPE_BUILDING → SLOPE_CUSTOM_PATH
        - retarget_custom: SLOPE_CUSTOM_PATH → SLOPE_CUSTOM_PATH (self-loop, re-target)
        Fired directly from the click handler on a VALID terrain/node target click;
        there is no button — targeting is map-only, mirroring roads.

Complete Transition Matrix (all states)
==================================================

# 1. Transitions: From IDLE_READY
# --------------------------------
# 1.1. → IDLE_READY: NOT ALLOWED (no-op, nothing to transition)
# 1.2. → IDLE_VIEWING_SLOPE: view_slope [direct] (click slope icon/centerline)
# 1.3. → IDLE_VIEWING_LIFT: view_lift [direct] (click lift icon/cable)
# 1.4. → IDLE_VIEWING_ROAD: view_road [direct] (click road)
# 1.5. → SLOPE_STARTING: start_slope [direct] (click terrain/node in slope mode)
# 1.6. → SLOPE_CUSTOM_PATH: NOT ALLOWED (must go through SLOPE_STARTING first)
# 1.7. → LIFT_PLACING: start_lift [direct] (click terrain/node in lift mode)
# 1.8. → ROAD_STARTING: start_road [direct] (click terrain/node in road mode)

# 2. Transitions: From IDLE_VIEWING_SLOPE
# ----------------------------------------
# 2.1. → IDLE_READY: close_slope_panel [direct] (close button or click elsewhere)
# 2.2. → IDLE_VIEWING_SLOPE: switch_slope [direct, self-loop] (click different slope)
# 2.3. → IDLE_VIEWING_LIFT: switch_to_lift_view [direct] (click lift in panel or on map)
# 2.4. → IDLE_VIEWING_ROAD: switch_slope_to_road_view [direct] (click a road)
# 2.5. → SLOPE_STARTING: start_slope_from_slope_view [direct] (click terrain/node)
# 2.6. → SLOPE_CUSTOM_PATH: NOT ALLOWED (must go through SLOPE_STARTING first)
# 2.7. → LIFT_PLACING: start_lift_from_slope_view [direct] (click terrain/node in lift mode)
# 2.8. → ROAD_STARTING: start_road_from_slope_view [direct] (click terrain/node in road mode)

# 3. Transitions: From IDLE_VIEWING_LIFT
# ---------------------------------------
# 3.1. → IDLE_READY: close_lift_panel [direct] (close button or click elsewhere)
# 3.2. → IDLE_VIEWING_SLOPE: switch_to_slope_view [direct] (click connected slope in panel)
# 3.3. → IDLE_VIEWING_LIFT: switch_lift [direct, self-loop] (click different lift)
# 3.4. → IDLE_VIEWING_ROAD: switch_lift_to_road_view [direct] (click a road)
# 3.5. → SLOPE_STARTING: start_slope_from_lift_view [direct] (click terrain/node)
# 3.6. → SLOPE_CUSTOM_PATH: NOT ALLOWED (must go through SLOPE_STARTING first)
# 3.7. → LIFT_PLACING: start_lift_from_lift_view [direct] (click terrain/node in lift mode)
# 3.8. → ROAD_STARTING: start_road_from_lift_view [direct] (click terrain/node in road mode)

# 3b. Transitions: From IDLE_VIEWING_ROAD
# ----------------------------------------
# 3b.1. → IDLE_READY: close_road_panel [direct] (close button or click elsewhere)
# 3b.2. → IDLE_VIEWING_SLOPE: switch_road_to_slope_view [direct] (click a slope)
# 3b.3. → IDLE_VIEWING_LIFT: switch_road_to_lift_view [direct] (click a lift)
# 3b.4. → IDLE_VIEWING_ROAD: switch_road [direct, self-loop] (click a different road)
# 3b.5. → SLOPE_STARTING: start_slope_from_road_view [direct] (click terrain/node)
# 3b.6. → LIFT_PLACING: start_lift_from_road_view [direct] (click terrain/node in lift mode)
# 3b.7. → ROAD_STARTING: start_road_from_road_view [direct] (click terrain/node in road mode)

# 4. Transitions: From SLOPE_STARTING
# ------------------------------------
# 4.1. → IDLE_READY: cancel_from_starting [event: cancel_slope] (cancel button)
# 4.2. → IDLE_VIEWING_SLOPE: NOT ALLOWED (must commit or cancel first)
# 4.3. → SLOPE_STARTING: NOT ALLOWED (no self-loop, proposal selection is internal)
# 4.4. → SLOPE_BUILDING: commit_first_path [event: commit_path] (click proposal endpoint)
# 4.5. → SLOPE_CUSTOM_PATH: select_target_from_starting [event: select_custom_target] (click a target)
# 4.6. → LIFT_PLACING: NOT ALLOWED (must cancel slope first)

# 5. Transitions: From SLOPE_BUILDING
# ------------------------------------
# 5.1. → IDLE_READY: cancel_from_building [event: cancel_slope] (cancel button)
# 5.2. → IDLE_VIEWING_SLOPE: finish_slope [direct] (finish button)
# 5.3. → SLOPE_STARTING: NOT ALLOWED (would lose committed segments)
# 5.4. → SLOPE_BUILDING: commit_continue_path [event: commit_path, self-loop]
# 5.5. → SLOPE_CUSTOM_PATH: select_target_from_building [event: select_custom_target] (click a target)
# 5.6. → LIFT_PLACING: NOT ALLOWED (must finish/cancel slope first)

# 6. Transitions: From SLOPE_CUSTOM_PATH
# ---------------------------------------
# 6.1. → IDLE_READY: cancel_slope_from_custom_path [event: cancel_slope]
# 6.2. → IDLE_VIEWING_SLOPE: commit_custom_finish [direct] (auto-finish when connecting to node)
# 6.3. → SLOPE_STARTING: cancel_path_to_starting [event: cancel_custom, guard: has_no_segments]
# 6.4. → SLOPE_BUILDING: commit_custom_continue [direct] (commit and keep building),
#                        cancel_path_to_building [event: cancel_custom, guard: !has_no_segments]
# 6.5. → SLOPE_CUSTOM_PATH: retarget_custom [event: select_custom_target, self-loop] (click a new target)
# 6.6. → LIFT_PLACING: NOT ALLOWED (must finish/cancel slope first)

# 7. Transitions: From LIFT_PLACING
# ----------------------------------
# 7.1. → IDLE_READY: cancel_lift [direct] (cancel button)
# 7.2. → IDLE_VIEWING_LIFT: complete_lift [direct] (click end station location)
# 7.3. → any other state: NOT ALLOWED (must cancel or complete the lift first)

# 9. Transitions: From ROAD_STARTING / ROAD_BUILDING
# ---------------------------------------------------
# 9.1. → IDLE_READY: cancel_road_from_starting / cancel_road_from_building [event: cancel_road]
# 9.2. → IDLE_VIEWING_ROAD: finish_road [direct] (finish button, from ROAD_BUILDING)
# 9.3. → ROAD_BUILDING: commit_road_first (from ROAD_STARTING) / commit_road_continue (self-loop) [event: commit_road]

Transition Summary Table
------------------------
    Slopes are targeted map-only (click a downhill point/node) via the select_custom_target
    event — there is no picking state and no enter button. Roads are always segment-by-segment.

    - From IDLE_READY (5): view_slope, view_lift, view_road, start_slope, start_lift, start_road [all direct]
    - From IDLE_VIEWING_SLOPE (5+1): close, switch_to_lift, switch_to_road, start_slope, start_lift, start_road, switch_slope (loop)
    - From IDLE_VIEWING_LIFT (5+1): close, switch_to_slope, switch_to_road, start_slope, start_lift, start_road, switch_lift (loop)
    - From IDLE_VIEWING_ROAD (5+1): close, switch_to_slope, switch_to_lift, start_slope, start_lift, start_road, switch_road (loop)
    - From SLOPE_STARTING (3): cancel [cancel_slope], commit_first_path [commit_path], select_target [select_custom_target]
    - From SLOPE_BUILDING (3+1): cancel [cancel_slope], finish [direct], select_target [select_custom_target], commit_path (loop)
    - From SLOPE_CUSTOM_PATH (4+1): commit_continue [direct], commit_finish [direct], cancel_slope [cancel_slope], cancel_path_to_* [cancel_custom], retarget (loop) [select_custom_target]
    - From LIFT_PLACING (2): cancel [direct], complete [direct]
    - From ROAD_STARTING (2): cancel [cancel_road], commit_road_first [commit_road]
    - From ROAD_BUILDING (2+1): cancel [cancel_road], finish [direct], commit_road_continue (loop) [commit_road]

    Event-triggered transitions use [event_name] notation.
    Direct transitions are called by their transition name directly.

    All other (state, event) combinations are NOT ALLOWED — they would bypass required workflow steps.

    NOTE: Undo is handled via force_idle()/force_building() methods, NOT via transitions.
          See "Undo Architecture" section above.

Cleanup Policy
--------------
Orphaned node cleanup is NOT called on every transition. Instead, cleanup_isolated_nodes()
is called explicitly when entities are removed or operations are canceled:
- undo ADD_SEGMENTS (segment deleted)
- undo ADD_LIFT (lift deleted)
- delete_slope (slope and segments deleted)
- delete_lift (lift deleted)
- cancel_current_slope (building segments discarded)
- Reset View button (manual fallback for any edge cases)

This prevents premature deletion of nodes that are still needed (e.g., start node
in custom connect mode from SlopeStarting state, before any segment is committed).

Lift Placement Pattern
----------------------
Lift placement validates using elevations BEFORE creating nodes for terrain clicks.
This prevents orphan nodes from repeated failed uphill validation attempts.
Both start and end nodes are only created AFTER validation passes.
"""

from __future__ import annotations

import logging
from typing import Callable

import streamlit as st
from statemachine import State, StateMachine

from skiresort_planner.model.path_point import PathPoint
from skiresort_planner.model.resort_graph import ResortGraph
from skiresort_planner.ui.context import (
    BuildMode,
    EntityKind,
    LonLatElev,
    PlannerContext,
)
from skiresort_planner.ui.infra import trigger_rerun
from skiresort_planner.ui.state_lifecycle import (
    enter_idle_ready,
    enter_idle_viewing_lift,
    enter_idle_viewing_road,
    enter_idle_viewing_slope,
    enter_import_placing,
    enter_lift_placing,
    enter_road_building,
    enter_road_starting,
    enter_slope_building,
    enter_slope_custom_path,
    enter_slope_starting,
    exit_idle_ready,
    exit_idle_viewing_lift,
    exit_idle_viewing_road,
    exit_idle_viewing_slope,
    exit_lift_placing,
    exit_road_building,
    exit_road_starting,
    exit_slope_building,
    exit_slope_custom_path,
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

    States (all explicit states):
        IDLE_READY: No panel visible, ready to build
        IDLE_VIEWING_SLOPE: Panel showing slope details
        IDLE_VIEWING_LIFT: Panel showing lift details
        IDLE_VIEWING_ROAD: Panel showing road details
        SLOPE_STARTING: 0 segments, picking first direction
        SLOPE_BUILDING: 1+ segments, continuing
        SLOPE_CUSTOM_PATH: Showing custom path options routed to a clicked target
        LIFT_PLACING: Waiting for end station
        ROAD_STARTING: 0 road segments, picking first target
        ROAD_BUILDING: 1+ road segments, extending the road

    Using explicit states eliminates impossible state combinations
    and enables dispatch-table UI rendering pattern.
    """

    # ==========================================================================
    # State Definitions (all explicit states)
    # ==========================================================================

    # IDLE states (no building in progress)
    idle_ready = State("IdleReady", initial=True)
    idle_viewing_slope = State("IdleViewingSlope")
    idle_viewing_lift = State("IdleViewingLift")
    idle_viewing_road = State("IdleViewingRoad")

    # SLOPE states (building in progress)
    slope_starting = State("SlopeStarting")
    slope_building = State("SlopeBuilding")
    slope_custom_path = State("SlopeCustomPath")

    # LIFT state
    lift_placing = State("LiftPlacing")

    # IMPORT state (click-to-place an OSM import bounding box, then confirm)
    import_placing = State("ImportPlacing")

    # ROAD states (segment-by-segment, like a slope: build then finish)
    road_starting = State("RoadStarting")
    road_building = State("RoadBuilding")

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
    start_road = idle_ready.to(road_starting, event="start_road")  # 1.9 [event: start_road]
    start_import = idle_ready.to(import_placing, event="start_import")  # 1.10 [event: start_import]
    view_slope = idle_ready.to(idle_viewing_slope, event="view_slope")  # 1.2 [event: view_slope]
    view_lift = idle_ready.to(idle_viewing_lift, event="view_lift")  # 1.3 [event: view_lift]
    view_road = idle_ready.to(idle_viewing_road, event="view_road")  # 1.5 [event: view_road]

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
    switch_slope_to_road_view = idle_viewing_slope.to(idle_viewing_road, event="view_road")  # 2.5 [event: view_road]
    start_slope_from_slope_view = idle_viewing_slope.to(slope_starting, event="start_slope")  # 2.4 [event: start_slope]
    start_lift_from_slope_view = idle_viewing_slope.to(lift_placing, event="start_lift")  # 2.8 [event: start_lift]
    start_road_from_slope_view = idle_viewing_slope.to(road_starting, event="start_road")  # 2.9 [event: start_road]
    start_import_from_slope_view = idle_viewing_slope.to(
        import_placing, event="start_import"
    )  # 2.10 [event: start_import]

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
    switch_lift_to_road_view = idle_viewing_lift.to(idle_viewing_road, event="view_road")  # 3.5 [event: view_road]
    start_slope_from_lift_view = idle_viewing_lift.to(slope_starting, event="start_slope")  # 3.4 [event: start_slope]
    start_lift_from_lift_view = idle_viewing_lift.to(lift_placing, event="start_lift")  # 3.8 [event: start_lift]
    start_road_from_lift_view = idle_viewing_lift.to(road_starting, event="start_road")  # 3.9 [event: start_road]
    start_import_from_lift_view = idle_viewing_lift.to(
        import_placing, event="start_import"
    )  # 3.10 [event: start_import]

    # ==========================================================================
    # 3b. Transitions: From IDLE_VIEWING_ROAD
    # ==========================================================================
    # Events: close_panel, view_road (self-loop), view_slope, view_lift, start_*
    # 3b.1. close_road_panel [event: close_panel]: Close button or click elsewhere
    # 3b.5. switch_road [event: view_road, self-loop]: Click a different road
    # 3b.2. switch_road_to_slope_view [event: view_slope]: Click a slope
    # 3b.3. switch_road_to_lift_view [event: view_lift]: Click a lift
    # 3b.4/8/9. start_* : Click terrain/node to start building in the active mode

    close_road_panel = idle_viewing_road.to(idle_ready, event="close_panel")  # 3b.1 [event: close_panel]
    switch_road = idle_viewing_road.to(idle_viewing_road, event="view_road")  # 3b.5 [event: view_road] self-loop
    switch_road_to_slope_view = idle_viewing_road.to(idle_viewing_slope, event="view_slope")  # 3b.2 [event: view_slope]
    switch_road_to_lift_view = idle_viewing_road.to(idle_viewing_lift, event="view_lift")  # 3b.3 [event: view_lift]
    start_slope_from_road_view = idle_viewing_road.to(slope_starting, event="start_slope")  # 3b.4 [event: start_slope]
    start_lift_from_road_view = idle_viewing_road.to(lift_placing, event="start_lift")  # 3b.8 [event: start_lift]
    start_road_from_road_view = idle_viewing_road.to(road_starting, event="start_road")  # 3b.9 [event: start_road]
    start_import_from_road_view = idle_viewing_road.to(
        import_placing, event="start_import"
    )  # 3b.10 [event: start_import]

    # ==========================================================================
    # 4. Transitions: From SLOPE_STARTING (0 segments)
    # ==========================================================================
    # Events available: commit_path, cancel_slope, select_custom_target
    # 4.1. cancel_from_starting [event: cancel_slope]: Cancel button
    # 4.5. commit_first_path [event: commit_path]: Click proposal endpoint
    # 4.6. select_target_from_starting [event: select_custom_target]: Click a target on the map

    commit_first_path = slope_starting.to(slope_building, event="commit_path")  # 4.5 [event: commit_path]
    cancel_from_starting = slope_starting.to(idle_ready, event="cancel_slope")  # 4.1 [event: cancel_slope]
    select_target_from_starting = slope_starting.to(
        slope_custom_path, event="select_custom_target", before="_before_target_from_starting"
    )  # 4.6 [event: select_custom_target]

    # ==========================================================================
    # 5. Transitions: From SLOPE_BUILDING (1+ segments)
    # ==========================================================================
    # Events available: commit_path, cancel_slope, select_custom_target
    # NOTE: Undo is handled via force_idle()/force_building(), NOT transitions
    # 5.1. cancel_from_building [event: cancel_slope]: Cancel button (discard all)
    # 5.2. finish_slope [direct]: Finish button
    # 5.5. commit_continue_path [event: commit_path, self-loop]: Commit more segments
    # 5.6. select_target_from_building [event: select_custom_target]: Click a target on the map

    commit_continue_path = slope_building.to(slope_building, event="commit_path")  # 5.5 [event: commit_path] self-loop
    finish_slope = slope_building.to(idle_viewing_slope, event="finish_slope")  # 5.2 [event: finish_slope]
    cancel_from_building = slope_building.to(idle_ready, event="cancel_slope")  # 5.1 [event: cancel_slope]
    select_target_from_building = slope_building.to(
        slope_custom_path, event="select_custom_target", before="_before_target_from_building"
    )  # 5.6 [event: select_custom_target]

    # ==========================================================================
    # 6. Transitions: From SLOPE_CUSTOM_PATH
    # ==========================================================================
    # Events available: cancel_custom, cancel_slope, select_custom_target (self-loop), finish_slope
    # 6.1. cancel_slope_from_custom_path [event: cancel_slope]: Cancel entire slope
    # 6.2. commit_custom_finish [direct]: Auto-finish when connecting to existing node
    # 6.3. commit_custom_continue [direct]: Commit and keep building
    # 6.4. cancel_path_to_starting [event: cancel_custom, guard]: Back to fan-out when has_no_segments
    # 6.5. cancel_path_to_building [event: cancel_custom, guard]: Back to fan-out when has segments
    # 6.6. retarget_custom [event: select_custom_target, self-loop]: Click a new target → re-route
    # 6.7. finish_slope_from_custom [event: finish_slope]: Sidebar Finish during targeting —
    #      finalize committed segments, drop the in-progress proposal

    commit_custom_continue = slope_custom_path.to(slope_building)  # 6.3 [direct]
    commit_custom_finish = slope_custom_path.to(idle_viewing_slope)  # 6.2 [direct] auto-finish connector
    retarget_custom = slope_custom_path.to(
        slope_custom_path, event="select_custom_target", before="_before_retarget_custom"
    )  # 6.6 [event: select_custom_target] self-loop
    finish_slope_from_custom = slope_custom_path.to(
        idle_viewing_slope, event="finish_slope", before="_before_finish_slope_from_custom"
    )  # 6.7 [event: finish_slope]
    cancel_path_to_starting = slope_custom_path.to(
        slope_starting, cond="has_no_segments", event="cancel_custom"
    )  # 6.4 [event: cancel_custom, guard]
    cancel_path_to_building = slope_custom_path.to(
        slope_building, unless="has_no_segments", event="cancel_custom"
    )  # 6.5 [event: cancel_custom, guard]
    cancel_slope_from_custom_path = slope_custom_path.to(idle_ready, event="cancel_slope")  # 6.1 [event: cancel_slope]

    # ==========================================================================
    # 8. Transitions: From LIFT_PLACING
    # ==========================================================================
    # All transitions from LIFT_PLACING are direct (no shared events)
    # 8.1. cancel_lift [direct]: Cancel button
    # 8.3. complete_lift [direct]: Click end station location

    complete_lift = lift_placing.to(idle_viewing_lift)  # 8.3 [direct]
    cancel_lift = lift_placing.to(idle_ready)  # 8.1 [direct]

    # ==========================================================================
    # 8b. Transitions: From IMPORT_PLACING
    # ==========================================================================
    # All transitions from IMPORT_PLACING are direct (no shared events), mirroring LIFT_PLACING.
    # 8b.1. cancel_import [direct]: Cancel button
    # 8b.2. complete_import [direct]: Confirm button or center-dot click → run the deferred fetch
    # 8b.3. retarget_import [direct, self-loop]: click a new point to re-place the box center

    complete_import = import_placing.to(idle_ready)  # 8b.2 [direct]
    cancel_import = import_placing.to(idle_ready)  # 8b.1 [direct]
    retarget_import = import_placing.to(import_placing)  # 8b.3 [direct] self-loop

    # ==========================================================================
    # 9. Transitions: From ROAD_STARTING (0 segments) / ROAD_BUILDING (1+ segments)
    # ==========================================================================
    # Roads build segment-by-segment like slopes: each click traces one gentle
    # (gentle-gradient) segment to the clicked point. No custom-connect (every segment is
    # already point-to-point).
    # 9.1. cancel_road [direct]: Cancel button, from either state
    # 9.4. commit_road_first [event: commit_road]: first traced segment
    # 9.5. commit_road_continue [event: commit_road, self-loop]: extend the road
    # 9.2. finish_road [direct]: Finish button
    # 9.6. commit_road_finish [event: commit_road_finish]: a connector segment (target is an
    #      existing node) ends the road immediately, from either state. Mirrors commit_custom_finish.

    commit_road_first = road_starting.to(road_building, event="commit_road")  # 9.4 [event: commit_road]
    commit_road_continue = road_building.to(road_building, event="commit_road")  # 9.5 [event: commit_road] self-loop
    finish_road = road_building.to(idle_viewing_road)  # 9.2 [direct]
    commit_road_finish_from_starting = road_starting.to(
        idle_viewing_road, event="commit_road_finish"
    )  # 9.6 [event: commit_road_finish]
    commit_road_finish_from_building = road_building.to(
        idle_viewing_road, event="commit_road_finish"
    )  # 9.6 [event: commit_road_finish]
    cancel_road_from_starting = road_starting.to(idle_ready, event="cancel_road")  # 9.1 [event: cancel_road]
    cancel_road_from_building = road_building.to(idle_ready, event="cancel_road")  # 9.1 [event: cancel_road]

    # ==========================================================================
    # Guards (Conditions)
    # ==========================================================================

    def has_no_segments(self) -> bool:
        """Guard: Check if there are no committed segments."""
        return len(self.context.slope_build.segments) == 0

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
            # commit_road event
            "commit_road_first",
            "commit_road_continue",
            # commit_road_finish event
            "commit_road_finish_from_starting",
            "commit_road_finish_from_building",
            # cancel_slope event
            "cancel_from_starting",
            "cancel_from_building",
            "cancel_slope_from_custom_path",
            # cancel_road event
            "cancel_road_from_starting",
            "cancel_road_from_building",
            # cancel_custom event
            "cancel_path_to_starting",
            "cancel_path_to_building",
            # select_custom_target event
            "select_target_from_starting",
            "select_target_from_building",
            "retarget_custom",
            # start_slope event (NOT start_slope - that IS the event entry point)
            "start_slope_from_slope_view",
            "start_slope_from_lift_view",
            "start_slope_from_road_view",
            # start_lift event (NOT start_lift - that IS the event entry point)
            "start_lift_from_slope_view",
            "start_lift_from_lift_view",
            "start_lift_from_road_view",
            # start_road event (NOT start_road - that IS the event entry point)
            "start_road_from_slope_view",
            "start_road_from_lift_view",
            "start_road_from_road_view",
            # view_slope event (NOT view_slope - that IS the event entry point)
            "switch_to_slope_view",
            "switch_slope",
            "switch_road_to_slope_view",
            # view_lift event (NOT view_lift - that IS the event entry point)
            "switch_to_lift_view",
            "switch_lift",
            "switch_road_to_lift_view",
            # view_road event (NOT view_road - that IS the event entry point)
            "switch_slope_to_road_view",
            "switch_lift_to_road_view",
            "switch_road",
            # close_panel event (both are variants, event is "close_panel")
            "close_slope_panel",
            "close_lift_panel",
            "close_road_panel",
        }
    )

    # ==========================================================================
    # State Check Properties
    # ==========================================================================

    @property
    def is_idle(self) -> bool:
        """Check if in any idle state (not building)."""
        return (
            self.is_idle_ready or self.is_idle_viewing_slope or self.is_idle_viewing_lift or self.is_idle_viewing_road
        )

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
    def is_idle_viewing_road(self) -> bool:
        """Check if viewing a road."""
        return bool(self.idle_viewing_road.is_active)

    @property
    def is_slope_starting(self) -> bool:
        """Check if starting a slope (0 segments)."""
        return bool(self.slope_starting.is_active)

    @property
    def is_slope_building_only(self) -> bool:
        """Check if in slope_building state specifically (1+ segments)."""
        return bool(self.slope_building.is_active)

    @property
    def is_slope_custom_path(self) -> bool:
        """Check if showing custom path options."""
        return bool(self.slope_custom_path.is_active)

    @property
    def is_lift_placing(self) -> bool:
        """Check if placing a lift."""
        return bool(self.lift_placing.is_active)

    @property
    def is_import_placing(self) -> bool:
        """Check if placing an OSM import bounding box."""
        return bool(self.import_placing.is_active)

    @property
    def is_road_starting(self) -> bool:
        """Check if starting a road (0 segments)."""
        return bool(self.road_starting.is_active)

    @property
    def is_road_building_only(self) -> bool:
        """Check if in road_building state specifically (1+ segments)."""
        return bool(self.road_building.is_active)

    @property
    def is_any_road_state(self) -> bool:
        """Check if in any road-building state (starting or building)."""
        return self.is_road_starting or self.is_road_building_only

    # Composite state checks
    @property
    def is_any_slope_state(self) -> bool:
        """Check if in any slope-related state.

        Returns True for: slope_starting, slope_building, slope_custom_path
        """
        return self.is_slope_starting or self.is_slope_building_only or self.is_slope_custom_path

    @property
    def is_info_panel_visible(self) -> bool:
        """Check if info panel is visible (viewing slope, lift, or road)."""
        return self.is_idle_viewing_slope or self.is_idle_viewing_lift or self.is_idle_viewing_road

    @property
    def viewing_entity(self) -> tuple[EntityKind, str] | None:
        """The (kind, id) of the slope/road/lift being viewed, or None if not viewing."""
        v = self.context.viewing
        if self.is_idle_viewing_slope and v.slope_id:
            return EntityKind.SLOPE, v.slope_id
        if self.is_idle_viewing_road and v.road_id:
            return EntityKind.ROAD, v.road_id
        if self.is_idle_viewing_lift and v.lift_id:
            return EntityKind.LIFT, v.lift_id
        return None

    def is_slope_mode(self) -> bool:
        """Check if build mode is set to slope."""
        return BuildMode.is_slope(self.context.build_mode.mode)

    def is_lift_mode(self) -> bool:
        """Check if build mode is set to any lift type."""
        return BuildMode.is_lift(self.context.build_mode.mode)

    def is_road_mode(self) -> bool:
        """Check if build mode is set to road."""
        return BuildMode.is_road(self.context.build_mode.mode)

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

    def on_enter_idle_viewing_road(self) -> None:
        """Hook: Entering road viewing state."""
        enter_idle_viewing_road(self.context)

    def on_enter_slope_starting(self) -> None:
        """Hook: Entering slope starting state."""
        enter_slope_starting(self.context)

    def on_enter_slope_building(self) -> None:
        """Hook: Entering slope building state."""
        enter_slope_building(self.context)

    def on_enter_slope_custom_path(self) -> None:
        """Hook: Entering custom path state."""
        enter_slope_custom_path(self.context)

    def on_enter_lift_placing(self) -> None:
        """Hook: Entering lift placing state."""
        enter_lift_placing(self.context)

    def on_enter_import_placing(self) -> None:
        """Hook: Entering import placing state (also fires on retarget self-loop)."""
        enter_import_placing(self.context)

    def on_enter_road_starting(self) -> None:
        """Hook: Entering road starting state."""
        enter_road_starting(self.context)

    def on_enter_road_building(self) -> None:
        """Hook: Entering road building state."""
        enter_road_building(self.context)

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

    def on_exit_idle_viewing_road(self) -> None:
        """Hook: Exiting road viewing state."""
        exit_idle_viewing_road(self.context)

    def on_exit_slope_starting(self) -> None:
        """Hook: Exiting slope starting state."""
        exit_slope_starting(self.context)

    def on_exit_slope_building(self) -> None:
        """Hook: Exiting slope building state."""
        exit_slope_building(self.context)

    def on_exit_slope_custom_path(self) -> None:
        """Hook: Exiting custom path state."""
        exit_slope_custom_path(self.context)

    def on_exit_lift_placing(self) -> None:
        """Hook: Exiting lift placing state."""
        exit_lift_placing(self.context)

    def on_exit_road_starting(self) -> None:
        """Hook: Exiting road starting state."""
        exit_road_starting(self.context)

    def on_exit_road_building(self) -> None:
        """Hook: Exiting road building state."""
        exit_road_building(self.context)

    # ==========================================================================
    # Transition Actions (before_* hooks)
    # ==========================================================================
    # Naming convention (enforced repo-wide):
    #   before_<event>   → auto-discovered event-level hook; fires for EVERY transition
    #                      of that event. Use when all transitions share one action.
    #   _before_<name>   → private; wired explicitly via before="..." on ONE transition.
    #                      Use when transitions sharing an event need DIFFERENT actions
    #                      (e.g. select_custom_target: starting vs building vs retarget).

    def before_start_slope(
        self,
        lon: float,
        lat: float,
        elevation: float,
        node_id: str | None = None,
    ) -> None:
        """Action before starting to build a slope."""
        self.context.set_selection(lon=lon, lat=lat, elevation=elevation)
        self.context.slope_build.start_node_id = node_id
        self.context.selection.node_id = node_id
        slope_number = self._resort_graph._slope_counter + 1
        self.context.slope_build.name = f"Slope {slope_number}"

    def _add_segment_to_building(self, segment_id: str, endpoint_node_id: str) -> None:
        """Common logic for adding segment to building context."""
        self.context.slope_build.segments.append(segment_id)
        self.context.slope_build.endpoints = [endpoint_node_id]
        self.context.clear_proposals()

    def before_commit_path(self, segment_id: str, endpoint_node_id: str) -> None:
        """Action before committing a path segment (event hook only)."""
        self._add_segment_to_building(segment_id=segment_id, endpoint_node_id=endpoint_node_id)

    def _add_segment_to_road(self, segment_id: str, endpoint_node_id: str) -> None:
        """Common logic for adding a traced segment to the road context."""
        self.context.road_build.segments.append(segment_id)
        self.context.road_build.endpoints = [endpoint_node_id]
        self.context.clear_proposals()

    def before_commit_road(self, segment_id: str, endpoint_node_id: str) -> None:
        """Action before committing a road segment (event hook; both first + continue)."""
        self._add_segment_to_road(segment_id=segment_id, endpoint_node_id=endpoint_node_id)

    def before_commit_custom_continue(self, segment_id: str, endpoint_node_id: str) -> None:
        """Action before committing custom path and continuing."""
        self.context.slope_build.segments.append(segment_id)
        self.context.slope_build.endpoints = [endpoint_node_id]
        self.context.clear_proposals()
        self.context.custom_connect.clear()

    def before_commit_custom_finish(self, segment_id: str, slope_id: str) -> None:
        """Action before committing custom connector and finishing.

        Note: segment_id may already be in building.segments if added before
        calling graph.finish_slope(). This hook is idempotent.
        """
        if segment_id not in self.context.slope_build.segments:
            self.context.slope_build.segments.append(segment_id)
        self.context.viewing.set_slope_id(slope_id=slope_id)
        self.context.custom_connect.clear()

    def before_commit_road_finish(self, segment_id: str, road_id: str) -> None:
        """Road connector auto-finish (mirrors before_commit_custom_finish).

        Idempotent on segment_id (caller appends before graph.finish_road()).
        enter_idle_viewing_road clears road_build, so no clear here.
        """
        if segment_id not in self.context.road_build.segments:
            self.context.road_build.segments.append(segment_id)
        self.context.viewing.set_road_id(road_id=road_id)

    def _before_finish_slope_from_custom(self, slope_id: str) -> None:
        """Sidebar Finish during targeting: drop the in-progress proposal (never in
        segments) and clear custom-connect + proposals. set_slope_id via before_finish_slope.
        """
        self.context.clear_custom_connect()
        self.context.clear_proposals()

    def before_finish_slope(self, slope_id: str) -> None:
        """Action before finishing a slope."""
        self.context.viewing.set_slope_id(slope_id=slope_id)

    def before_view_slope(self, slope_id: str) -> None:
        """Set slope_id before entering viewing state. Panel visibility set by enter function."""
        self.context.viewing.set_slope_id(slope_id=slope_id)

    def before_view_lift(self, lift_id: str) -> None:
        """Set lift_id before entering viewing state. Panel visibility set by enter function."""
        self.context.viewing.set_lift_id(lift_id=lift_id)

    def before_view_road(self, road_id: str) -> None:
        """Set road_id before entering viewing state. Panel visibility set by enter function."""
        self.context.viewing.set_road_id(road_id=road_id)

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

    def before_switch_road(self, road_id: str) -> None:
        """Set road_id for a different road (self-loop). Panel visibility set by enter function."""
        self.context.viewing.set_road_id(road_id=road_id)

    def before_close_slope_panel(self) -> None:
        """Before closing slope panel. Panel hidden by enter_idle_ready."""
        pass  # Visibility handled by enter_idle_ready

    def before_close_lift_panel(self) -> None:
        """Before closing lift panel. Panel hidden by enter_idle_ready."""
        pass  # Visibility handled by enter_idle_ready

    def before_close_road_panel(self) -> None:
        """Before closing road panel. Panel hidden by enter_idle_ready."""
        pass  # Visibility handled by enter_idle_ready

    def before_start_lift(self, node_id: str | None = None, location: PathPoint | None = None) -> None:
        """Action before starting lift placement."""
        self.context.lift.start_node_id = node_id
        self.context.lift.start_location = location

    # Reuse start_lift logic for other entry points
    before_start_lift_from_slope_view = before_start_lift
    before_start_lift_from_lift_view = before_start_lift

    def before_start_road(self, node_id: str | None = None, location: PathPoint | None = None) -> None:
        """Action before starting road placement: store the first clicked point."""
        self.context.road_build.start_node_id = node_id
        self.context.road_build.start_location = location

    # Reuse start_road logic for other entry points
    before_start_road_from_slope_view = before_start_road
    before_start_road_from_lift_view = before_start_road
    before_start_road_from_road_view = before_start_road

    def before_start_import(self, lon: float, lat: float) -> None:
        """Action before starting import placement: store the first clicked center."""
        self.context.deferred.osm_import_center_lon = lon
        self.context.deferred.osm_import_center_lat = lat

    # Reuse start_import logic for the other idle entry points
    before_start_import_from_slope_view = before_start_import
    before_start_import_from_lift_view = before_start_import
    before_start_import_from_road_view = before_start_import

    def before_complete_lift(self, lift_id: str) -> None:
        """Set lift_id before completing. Panel visibility set by enter_idle_viewing_lift."""
        self.context.viewing.set_lift_id(lift_id=lift_id)
        self.context.lift.clear()

    def before_cancel_import(self) -> None:
        """Discard a placed-but-unconfirmed import: clear the box center and the pending flag."""
        self.context.deferred.osm_import = False
        self.context.deferred.osm_import_center_lon = None
        self.context.deferred.osm_import_center_lat = None

    def before_finish_road(self, road_id: str) -> None:
        """Set road_id before finishing. Panel visibility set by enter_idle_viewing_road."""
        self.context.viewing.set_road_id(road_id=road_id)
        self.context.road_build.clear()

    # ──────────────────────────────────────────────────────────────────────────────
    # Custom Connect Transitions (Single Source of Truth for ctx.custom_connect.*)
    # ──────────────────────────────────────────────────────────────────────────────
    # All custom_connect state mutations happen ONLY in these hooks:
    # - _before_target_from_*: per-transition before= hooks for select_custom_target (set
    #   start_node/target/force_mode). NO event-level hook — it would double-fire on top.
    # - cancel_custom/cancel_slope: Clears state via clear_custom_connect().
    # ──────────────────────────────────────────────────────────────────────────────

    def _before_target_from_starting(self, target_location: LonLatElev, target_node: str | None = None) -> None:
        """From SLOPE_STARTING: get-or-create the origin node, then route to the target.

        Attached via before= to select_target_from_starting. The origin has no
        committed segment yet, so materialise it from the current selection.
        """
        start_node_id = self.context.slope_build.start_node_id
        if start_node_id is None:
            sel = self.context.selection
            node, _ = self._resort_graph.get_or_create_node(lon=sel.lon, lat=sel.lat, elevation=sel.elevation)
            start_node_id = node.id
            self.context.slope_build.start_node_id = start_node_id
        self.context.custom_connect.start_node = start_node_id
        self.context.custom_connect.target_location = target_location
        self.context.custom_connect.target_node = target_node
        self.context.custom_connect.force_mode = True

    def _before_target_from_building(self, target_location: LonLatElev, target_node: str | None = None) -> None:
        """From SLOPE_BUILDING: route from the current endpoint to the clicked target.

        Attached via before= to select_target_from_building.
        """
        self.context.custom_connect.start_node = self.context.slope_build.endpoints[0]
        self.context.custom_connect.target_location = target_location
        self.context.custom_connect.target_node = target_node
        self.context.custom_connect.force_mode = True

    def _before_retarget_custom(self, target_location: LonLatElev, target_node: str | None = None) -> None:
        """From SLOPE_CUSTOM_PATH: re-route to a newly clicked target (self-loop).

        Attached via before= to retarget_custom. The start node is unchanged; only
        the target moves. enter_slope_custom_path (fired on the self-loop) regenerates
        proposals, so no deferred flag is set here.
        """
        self.context.custom_connect.target_location = target_location
        self.context.custom_connect.target_node = target_node
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
        self.context.slope_build.clear()

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
        "idle_viewing_road": exit_idle_viewing_road,
        "slope_starting": exit_slope_starting,
        "slope_building": exit_slope_building,
        "slope_custom_path": exit_slope_custom_path,
        "lift_placing": exit_lift_placing,
        "road_starting": exit_road_starting,
        "road_building": exit_road_building,
    }

    def force_idle(self) -> None:
        """Force state machine to IdleReady state without transition.

        Used after undo operations when no building context remains.
        Clears all building, custom, and viewing state.
        Does NOT trigger st.rerun() - caller is responsible for UI refresh.
        """
        logger.info(f"[STATE] Forcing state from {self.get_state_name()} to IdleReady")
        # Clear all context state (state-specific cleanup via exit hook in _set_current_state)
        self.context.slope_build.clear()
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
        Assumes caller has already set up ctx.slope_build with the restored segments.
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

    def force_road_building(self) -> None:
        """Force state machine to RoadBuilding without transition (undo helper).

        Used after undoing a road segment/finish when road segments remain.
        Assumes caller has already set up ctx.road_build with the restored segments.
        """
        logger.info(f"[STATE] Forcing state from {self.get_state_name()} to RoadBuilding")
        self.context.viewing.clear()
        self._set_current_state(state=self.road_building)
        enter_road_building(self.context)

    def force_road_starting(self) -> None:
        """Force state machine to RoadStarting without transition (undo helper).

        Used after undoing the last road segment when the origin is still set
        but no segments remain.
        """
        logger.info(f"[STATE] Forcing state from {self.get_state_name()} to RoadStarting")
        self.context.viewing.clear()
        self._set_current_state(state=self.road_starting)
        enter_road_starting(self.context)

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

    def __repr__(self) -> str:
        """Return string representation of state machine."""
        return f"PlannerStateMachine(state={self.get_state_name()}, model={self.context!r})"

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

    def select_road_start(self, node_id: str | None = None, location: PathPoint | None = None) -> None:
        """Start placing a road from any idle state.

        Uses start_road event - SM resolves to appropriate transition.
        """
        self.start_road(node_id=node_id, location=location)

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

    def show_road_info_panel(self, road_id: str) -> None:
        """Show road info panel from any idle state.

        Uses view_road event - SM resolves to appropriate transition.
        """
        self.view_road(road_id=road_id)

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

    def cancel_custom_connect(self) -> None:
        """Leave custom targeting, back to fan-out. SM resolves based on guards."""
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

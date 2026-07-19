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
3. On the next render cycle, the deferred dispatch in app.py checks pending flags
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
    - From IDLE_VIEWING_SLOPE (5+1): close, switch_to_lift, switch_to_road, start_slope, start_lift, start_road,
      switch_slope (loop)
    - From IDLE_VIEWING_LIFT (5+1): close, switch_to_slope, switch_to_road, start_slope, start_lift, start_road,
      switch_lift (loop)
    - From IDLE_VIEWING_ROAD (5+1): close, switch_to_slope, switch_to_lift, start_slope, start_lift, start_road,
      switch_road (loop)
    - From SLOPE_STARTING (3): cancel [cancel_slope], commit_first_path [commit_path],
      select_target [select_custom_target]
    - From SLOPE_BUILDING (3+1): cancel [cancel_slope], finish [direct], select_target [select_custom_target],
      commit_path (loop)
    - From SLOPE_CUSTOM_PATH (5+1): commit_continue [direct], commit_finish [direct], finish [finish_slope],
      cancel_slope [cancel_slope], cancel_path_to_* [cancel_custom], retarget (loop) [select_custom_target]
    - From LIFT_PLACING (2): cancel [direct], complete [direct]
    - From ROAD_STARTING (3): cancel [cancel_road], commit_road_first [commit_road],
      select_target [select_custom_target]
    - From ROAD_BUILDING (3+1): cancel [cancel_road], finish [direct], select_target [select_custom_target],
      commit_road_continue (loop) [commit_road]
    - From ROAD_CUSTOM_PATH (5+1): commit_road_custom_continue [direct], commit_road_custom_finish [direct],
      finish [finish_road], cancel_road [cancel_road], cancel_road_path_to_* [cancel_custom],
      retarget (loop) [select_custom_target]

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
from collections.abc import Iterator
from contextlib import contextmanager
from typing import NoReturn, Protocol, cast

import streamlit as st
from statemachine import State, StateMachine

from skiresort_planner.model.path_point import PathPoint
from skiresort_planner.model.path_segment import SegmentKind
from skiresort_planner.model.resort_graph import ResortGraph
from skiresort_planner.ui.context import (
    BuildMode,
    EntityKind,
    LonLatElev,
    PlannerContext,
    SegmentBuildContext,
)
from skiresort_planner.ui.infra import trigger_rerun
from skiresort_planner.ui.kind_spec import KIND_SPECS
from skiresort_planner.ui.state_lifecycle import (
    EXIT_HOOKS,
    enter_idle_ready,
    enter_idle_viewing_lift,
    enter_idle_viewing_road,
    enter_idle_viewing_route,
    enter_idle_viewing_slope,
    enter_import_placing,
    enter_lift_placing,
    enter_merge_placing,
    enter_road_building,
    enter_road_custom_path,
    enter_road_starting,
    enter_route_placing,
    enter_slope_building,
    enter_slope_custom_path,
    enter_slope_starting,
    exit_lift_placing,
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

    def after_transition(self, event: str, source: State, target: State, machine: StateMachine) -> None:
        """Run cleanup and trigger Streamlit rerun after state transitions.

        NOTE: We do NOT modify click deduplication here. The dedup is simple:
        same click key = duplicate. When user clicks elsewhere, key changes,
        so they can click back to original element.

        Supports deferred rerun for compound operations (e.g., undo from custom state).
        When _defer_rerun flag is set in session_state, the rerun is skipped to allow
        multiple state transitions before a single UI refresh.
        """
        logger.debug(f"[STATE] {source.name} --({event})--> {target.name}")

        # NOTE: Orphaned node cleanup is NOT called here. It's called explicitly
        # in operations that remove entities (undo, delete, cancel). This prevents
        # premature deletion of nodes still in use (e.g., start nodes in custom
        # connect mode before any segment is committed).

        # Check if rerun should be deferred (used during compound operations)
        if st.session_state.get("_defer_rerun", False):
            logger.debug(f'[STATE] Deferring st.rerun() after {event} transition (compound operation)"')
            return

        logger.debug(f'[STATE] Calling st.rerun() after {event} transition"')
        trigger_rerun()


class _ForbiddenCall(Protocol):
    """A blocked transition stand-in: any call raises. Variadic-object so it can replace any bound
    transition method regardless of that transition's real signature.
    """

    def __call__(self, *args: object, **kwargs: object) -> NoReturn: ...


def _forbidden_call(name: str) -> _ForbiddenCall:
    """Create a function that raises RuntimeError when called.

    Used to block direct calls to event-triggered transitions.
    """

    def wrapper(*args: object, **kwargs: object) -> NoReturn:
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
    idle_viewing_route = State("IdleViewingRoute")

    # SLOPE states (building in progress)
    slope_starting = State("SlopeStarting")
    slope_building = State("SlopeBuilding")
    slope_custom_path = State("SlopeCustomPath")

    # LIFT state
    lift_placing = State("LiftPlacing")

    # IMPORT state (click-to-place an OSM import bounding box, then confirm)
    import_placing = State("ImportPlacing")

    # MERGE state (click-to-select node markers to collapse, then confirm)
    merge_placing = State("MergePlacing")

    # ROUTE state (click a start node then an end node)
    route_placing = State("RoutePlacing")

    # ROAD states (segment-by-segment, like a slope: build then finish)
    road_starting = State("RoadStarting")
    road_building = State("RoadBuilding")
    road_custom_path = State("RoadCustomPath")

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
    start_merge = idle_ready.to(merge_placing, event="start_merge")  # 1.11 [event: start_merge]
    start_route = idle_ready.to(route_placing, event="start_route")  # 1.12 [event: start_route]
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
    start_merge_from_slope_view = idle_viewing_slope.to(merge_placing, event="start_merge")  # 2.11 [event: start_merge]
    start_route_from_slope_view = idle_viewing_slope.to(route_placing, event="start_route")  # 2.12 [event: start_route]

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
    start_merge_from_lift_view = idle_viewing_lift.to(merge_placing, event="start_merge")  # 3.11 [event: start_merge]
    start_route_from_lift_view = idle_viewing_lift.to(route_placing, event="start_route")  # 3.12 [event: start_route]

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
    start_merge_from_road_view = idle_viewing_road.to(merge_placing, event="start_merge")  # 3b.11 [event: start_merge]
    start_route_from_road_view = idle_viewing_road.to(route_placing, event="start_route")  # 3b.12 [event: start_route]

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
        idle_viewing_slope, event="finish_slope", before="_before_finish_from_custom"
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
    # 8c. Transitions: From MERGE_PLACING (click-to-select nodes, then confirm)
    # ==========================================================================
    # All transitions from MERGE_PLACING are direct (no shared events), mirroring IMPORT_PLACING.
    # 8c.1. cancel_merge [direct]: Cancel button
    # 8c.2. complete_merge [direct]: Confirm button → collapse the selected nodes to their median
    # 8c.3. toggle_merge_node [direct, self-loop]: click a node marker to add/remove it

    complete_merge = merge_placing.to(idle_ready)  # 8c.2 [direct]
    cancel_merge = merge_placing.to(idle_ready)  # 8c.1 [direct]
    toggle_merge_node = merge_placing.to(merge_placing)  # 8c.3 [direct] self-loop

    # route_placing / idle_viewing_route (all direct, mirroring MERGE_PLACING)
    complete_route = route_placing.to(idle_viewing_route)  # 8d.2 [direct]
    cancel_route_placing = route_placing.to(idle_ready)  # 8d.1 [direct]
    close_route_panel = idle_viewing_route.to(idle_ready, event="close_panel")  # 8d.3 [event: close_panel]

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
    # A connector road (target is an existing node) auto-finishes via commit_road_custom_finish
    # from ROAD_CUSTOM_PATH (§9b) — mirroring slope's commit_custom_finish. There is deliberately
    # NO connector-finish from the fan states: fan proposals are never connectors (is_connector is
    # only set in the custom-connect generator), exactly like slopes.

    commit_road_first = road_starting.to(road_building, event="commit_road")  # 9.4 [event: commit_road]
    commit_road_continue = road_building.to(road_building, event="commit_road")  # 9.5 [event: commit_road] self-loop
    finish_road = road_building.to(idle_viewing_road)  # 9.2 [direct]
    cancel_road_from_starting = road_starting.to(idle_ready, event="cancel_road")  # 9.1 [event: cancel_road]
    cancel_road_from_building = road_building.to(idle_ready, event="cancel_road")  # 9.1 [event: cancel_road]

    # ==========================================================================
    # 9b. Transitions: From ROAD_CUSTOM_PATH (mirror of SLOPE_CUSTOM_PATH §6)
    # ==========================================================================
    # A road target click routes a custom-connect path to that point, exactly like a
    # slope. Same event vocabulary (select_custom_target / cancel_custom / finish_road),
    # wired to the SAME kind-agnostic before-hooks the slope transitions use, so one
    # shared handler/generator serves both entities.
    # 9b.1. select_target_from_road_starting [event: select_custom_target]: click a target from ROAD_STARTING
    # 9b.2. select_target_from_road_building [event: select_custom_target]: click a target from ROAD_BUILDING
    # 9b.3. commit_road_custom_continue [direct]: commit the custom segment and keep building
    # 9b.4. commit_road_custom_finish [direct]: auto-finish when the target is an existing node
    # 9b.5. retarget_road_custom [event: select_custom_target, self-loop]: click a new target → re-route
    # 9b.6. finish_road_from_custom [event: finish_road]: sidebar Finish during targeting
    # 9b.7. cancel_road_path_to_starting [event: cancel_custom, guard]: back to fan when has_no_segments
    # 9b.8. cancel_road_path_to_building [event: cancel_custom, guard]: back to fan when segments exist
    # 9b.9. cancel_road_from_custom_path [event: cancel_road]: cancel the whole road
    select_target_from_road_starting = road_starting.to(
        road_custom_path, event="select_custom_target", before="_before_target_from_starting"
    )  # 9b.1 [event: select_custom_target]
    select_target_from_road_building = road_building.to(
        road_custom_path, event="select_custom_target", before="_before_target_from_building"
    )  # 9b.2 [event: select_custom_target]
    commit_road_custom_continue = road_custom_path.to(road_building)  # 9b.3 [direct]
    commit_road_custom_finish = road_custom_path.to(idle_viewing_road)  # 9b.4 [direct] auto-finish connector
    retarget_road_custom = road_custom_path.to(
        road_custom_path, event="select_custom_target", before="_before_retarget_custom"
    )  # 9b.5 [event: select_custom_target] self-loop
    finish_road_from_custom = road_custom_path.to(
        idle_viewing_road, event="finish_road", before="_before_finish_from_custom"
    )  # 9b.6 [event: finish_road]
    cancel_road_path_to_starting = road_custom_path.to(
        road_starting, cond="has_no_segments", event="cancel_custom"
    )  # 9b.7 [event: cancel_custom, guard]
    cancel_road_path_to_building = road_custom_path.to(
        road_building, unless="has_no_segments", event="cancel_custom"
    )  # 9b.8 [event: cancel_custom, guard]
    cancel_road_from_custom_path = road_custom_path.to(idle_ready, event="cancel_road")  # 9b.9 [event: cancel_road]

    # ==========================================================================
    # Guards (Conditions)
    # ==========================================================================

    def has_no_segments(self) -> bool:
        """Guard: the active build has no committed segments yet (any kind).

        Used by cancel_custom to decide whether to return to *_STARTING (no segments)
        or *_BUILDING. Keyed to the active kind, so one guard serves slope + road + any
        future kind — no per-kind duplicate.
        """
        return len(self._active_build().segments) == 0

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
            # cancel_slope event
            "cancel_from_starting",
            "cancel_from_building",
            "cancel_slope_from_custom_path",
            # cancel_road event
            "cancel_road_from_starting",
            "cancel_road_from_building",
            "cancel_road_from_custom_path",
            # cancel_custom event
            "cancel_path_to_starting",
            "cancel_path_to_building",
            "cancel_road_path_to_starting",
            "cancel_road_path_to_building",
            # finish_slope / finish_road event (the _from_custom variants; the base
            # finish_slope/finish_road transitions ARE the event entry points and stay callable)
            "finish_slope_from_custom",
            "finish_road_from_custom",
            # select_custom_target event
            "select_target_from_starting",
            "select_target_from_building",
            "retarget_custom",
            "select_target_from_road_starting",
            "select_target_from_road_building",
            "retarget_road_custom",
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
            # start_import event (NOT start_import - that IS the event entry point)
            "start_import_from_slope_view",
            "start_import_from_lift_view",
            "start_import_from_road_view",
            # start_merge event (NOT start_merge - that IS the event entry point)
            "start_merge_from_slope_view",
            "start_merge_from_lift_view",
            "start_merge_from_road_view",
            # start_route event (NOT start_route - that IS the event entry point)
            "start_route_from_slope_view",
            "start_route_from_lift_view",
            "start_route_from_road_view",
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
    def is_merge_placing(self) -> bool:
        """Check if selecting nodes to merge."""
        return bool(self.merge_placing.is_active)

    @property
    def is_route_placing(self) -> bool:
        """Check if picking the route start/end nodes."""
        return bool(self.route_placing.is_active)

    @property
    def is_idle_viewing_route(self) -> bool:
        """Check if browsing the computed routes."""
        return bool(self.idle_viewing_route.is_active)

    @property
    def is_road_starting(self) -> bool:
        """Check if starting a road (0 segments)."""
        return bool(self.road_starting.is_active)

    @property
    def is_road_building_only(self) -> bool:
        """Check if in road_building state specifically (1+ segments)."""
        return bool(self.road_building.is_active)

    @property
    def is_road_custom_path(self) -> bool:
        """Check if showing road custom path options (mirror of is_slope_custom_path)."""
        return bool(self.road_custom_path.is_active)

    @property
    def is_any_road_state(self) -> bool:
        """Check if in any road-building state (starting, building, or custom path)."""
        return self.is_road_starting or self.is_road_building_only or self.is_road_custom_path

    # Composite state checks
    @property
    def is_any_slope_state(self) -> bool:
        """Check if in any slope-related state.

        Returns True for: slope_starting, slope_building, slope_custom_path
        """
        return self.is_slope_starting or self.is_slope_building_only or self.is_slope_custom_path

    @property
    def is_any_path_state(self) -> bool:
        """True in ANY segment-path build state (slope, road, or a future kind)."""
        current = self.get_current_state_id()
        return any(
            current in {spec.starting_state, spec.building_state, spec.custom_path_state}
            for spec in KIND_SPECS.values()
        )

    @property
    def active_build_kind(self) -> SegmentKind:
        """The SegmentKind currently being built, resolved from the active state id.

        The single source that maps build state → kind, so callers dispatch on the
        SegmentKind enum instead of an is_road boolean. Driven by KIND_SPECS (each kind
        lists its 3 build-state ids), so a new kind needs no edit here. Raises if not in
        a build state.
        """
        current = self.get_current_state_id()
        for spec in KIND_SPECS.values():
            if current in {spec.starting_state, spec.building_state, spec.custom_path_state}:
                return spec.kind
        raise RuntimeError(f"active_build_kind called outside a build state: {current}")

    def commit_active_segment(self, segment_id: str, endpoint_node_id: str) -> None:
        """Fire the active kind's non-connector commit event, resolved from the current state.

        From a fan state (*_starting / *_building) this is the fan commit self-loop; from the
        custom-path state it is the custom-continue event. Keeps the state→event mapping inside
        the state machine so the action layer just says "commit the active segment".
        """
        spec = KIND_SPECS[self.active_build_kind]
        in_custom_path = self.get_current_state_id() == spec.custom_path_state
        event = spec.custom_continue_event if in_custom_path else spec.fan_commit_event
        self.send(event, segment_id=segment_id, endpoint_node_id=endpoint_node_id)

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

    def on_enter_merge_placing(self) -> None:
        """Hook: Entering merge placing state (also fires on toggle self-loop)."""
        enter_merge_placing(self.context)

    def on_enter_route_placing(self) -> None:
        """Hook: Entering route_placing — the start node was set by the completing click handler."""
        enter_route_placing(self.context)

    def on_enter_idle_viewing_route(self) -> None:
        """Hook: Entering idle_viewing_route (routes computed by the completing click handler)."""
        enter_idle_viewing_route(self.context)

    def on_enter_road_starting(self) -> None:
        """Hook: Entering road starting state."""
        enter_road_starting(self.context)

    def on_enter_road_building(self) -> None:
        """Hook: Entering road building state."""
        enter_road_building(self.context)

    def on_enter_road_custom_path(self) -> None:
        """Hook: Entering road custom-path state."""
        enter_road_custom_path(self.context)

    # ==========================================================================
    # Exit Hooks - only states with real teardown need one; the rest exit as no-ops.
    # (force/undo runs the same teardown via EXIT_HOOKS. import/merge clear their scratch
    # in their before_cancel_*/before_complete_* hooks, so they need no on_exit here.)
    # ==========================================================================

    def on_exit_lift_placing(self) -> None:
        """Hook: Exiting lift placing state — clears the lift scratch context."""
        exit_lift_placing(self.context)

    # ==========================================================================
    # Transition Actions (before_* hooks)
    # ==========================================================================
    # Naming convention (enforced repo-wide):
    #   before_<event>   → auto-discovered event-level hook; fires for EVERY transition
    #                      of that event. Use when all transitions share one action.
    #   _before_<name>   → private; wired explicitly via before="..." on ONE transition.
    #                      Use when transitions sharing an event need DIFFERENT actions
    #                      (e.g. select_custom_target: starting vs building vs retarget).

    def _init_build(self, kind: SegmentKind, *, node_id: str | None, location: PathPoint | None, name: str) -> None:
        """Initialise a build's origin, name, and selection — the SHARED body for every kind."""
        build = self.context.build(kind)
        build.start_node_id = node_id
        build.start_location = None if node_id else location
        build.name = name
        if node_id is None:
            logger.debug(f"[STATE] _init_build({kind.value}): no node_id, using start_location={location}")

        origin = self._resort_graph.nodes[node_id] if node_id else None
        if origin is not None:
            self.context.set_selection(lon=origin.lon, lat=origin.lat, elevation=origin.elevation)
        elif location is not None:
            self.context.set_selection(lon=location.lon, lat=location.lat, elevation=location.elevation)

    def before_start_slope(
        self,
        lon: float,
        lat: float,
        elevation: float,
        node_id: str | None = None,
    ) -> None:
        """Action before starting to build a slope (thin adapter over _init_build)."""
        self._init_build(
            kind=SegmentKind.SLOPE,
            node_id=node_id,
            location=PathPoint(lon=lon, lat=lat, elevation=elevation),
            name=f"Slope {self._resort_graph._slope_counter + 1}",
        )

    def _add_segment_to_active_build(self, segment_id: str, endpoint_node_id: str) -> None:
        """Append a committed segment to the active build (any kind) and clear proposals."""
        assert endpoint_node_id, "endpoint_node_id must be non-empty after segment commit"
        build = self._active_build()
        build.segments.append(segment_id)
        build.endpoints = [endpoint_node_id]
        self.context.clear_proposals()

    def before_commit_path(self, segment_id: str, endpoint_node_id: str) -> None:
        """Action before committing a slope path segment (event hook only)."""
        self._add_segment_to_active_build(segment_id=segment_id, endpoint_node_id=endpoint_node_id)

    def before_commit_road(self, segment_id: str, endpoint_node_id: str) -> None:
        """Action before committing a road segment (event hook; both first + continue)."""
        self._add_segment_to_active_build(segment_id=segment_id, endpoint_node_id=endpoint_node_id)

    def before_commit_custom_continue(self, segment_id: str, endpoint_node_id: str) -> None:
        """Commit a custom-path segment and keep building (any kind)."""
        self._add_segment_to_active_build(segment_id=segment_id, endpoint_node_id=endpoint_node_id)
        self.context.custom_connect.clear()

    def before_commit_custom_finish(self, segment_id: str, entity_id: str) -> None:
        """Commit a custom connector and finish the entity (any kind). Idempotent on segment_id."""
        build = self._active_build()
        if segment_id not in build.segments:
            build.segments.append(segment_id)
        self.context.viewing.set_viewed(kind=self.active_build_kind, entity_id=entity_id)
        self.context.custom_connect.clear()

    def before_commit_road_custom_continue(self, segment_id: str, endpoint_node_id: str) -> None:
        """Road custom-path commit + keep building — same body as the slope custom continue."""
        self.before_commit_custom_continue(segment_id=segment_id, endpoint_node_id=endpoint_node_id)

    def before_commit_road_custom_finish(self, segment_id: str, entity_id: str) -> None:
        """Road custom-path connector auto-finish — same body as the slope custom finish."""
        self.before_commit_custom_finish(segment_id=segment_id, entity_id=entity_id)

    def before_finish_slope(self, entity_id: str) -> None:
        """Action before finishing a slope."""
        self.context.viewing.set_viewed(kind=SegmentKind.SLOPE, entity_id=entity_id)

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
        self.context.lift.first_node_id = node_id
        self.context.lift.first_location = location

    # Reuse start_lift logic for other entry points
    before_start_lift_from_slope_view = before_start_lift
    before_start_lift_from_lift_view = before_start_lift

    def before_start_road(self, node_id: str | None = None, location: PathPoint | None = None) -> None:
        """Action before starting road placement (thin adapter over _init_build).

        Same shared body as before_start_slope — origin + in-build "Road N" name + selection — so
        the two kinds cannot drift. The finish-time bearing name overrides the temporary name.
        """
        self._init_build(
            kind=SegmentKind.ROAD,
            node_id=node_id,
            location=location,
            name=f"Road {self._resort_graph._road_counter + 1}",
        )

    # Reuse start_road logic for other entry points
    before_start_road_from_slope_view = before_start_road
    before_start_road_from_lift_view = before_start_road
    before_start_road_from_road_view = before_start_road

    def before_start_import(self, lon: float, lat: float) -> None:
        """Action before starting import placement: store the first clicked center."""
        self.context.pending.osm_import_center_lon = lon
        self.context.pending.osm_import_center_lat = lat

    # Reuse start_import logic for the other idle entry points
    before_start_import_from_slope_view = before_start_import
    before_start_import_from_lift_view = before_start_import
    before_start_import_from_road_view = before_start_import

    def before_complete_lift(self, lift_id: str) -> None:
        """Set lift_id before completing. Panel visibility set by enter_idle_viewing_lift."""
        self.context.viewing.set_lift_id(lift_id=lift_id)
        self.context.lift.clear()

    def before_cancel_import(self) -> None:
        """Discard a placed-but-unconfirmed import: clear the box center and the pending mode."""
        self.context.pending.osm_import_mode = None
        self.context.pending.osm_import_center_lon = None
        self.context.pending.osm_import_center_lat = None

    def before_toggle_merge_node(self, node_id: str) -> None:
        """Self-loop in merge_placing: add/remove the clicked node from the selection."""
        self.context.merge.toggle(node_id)

    def before_cancel_merge(self) -> None:
        """Discard an unconfirmed merge: clear the selected-node set."""
        self.context.merge.clear()

    def before_complete_merge(self) -> None:
        """Merge confirmed: clear the selection (the graph mutation runs in the action)."""
        self.context.merge.clear()

    def before_finish_road(self, entity_id: str) -> None:
        """Set the viewed road before finishing (mirrors before_finish_slope).

        The build is cleared by enter_idle_viewing_road (Single Point of Truth), same as slopes —
        no explicit clear here, so slope and road finish identically.
        """
        self.context.viewing.set_viewed(kind=SegmentKind.ROAD, entity_id=entity_id)

    # ──────────────────────────────────────────────────────────────────────────────
    # Custom Connect Transitions (Single Source of Truth for ctx.custom_connect.*)
    # ──────────────────────────────────────────────────────────────────────────────
    # All custom_connect state mutations happen ONLY in these hooks:
    # - _before_target_from_*: per-transition before= hooks for select_custom_target (set
    #   start_node/target/force_mode). NO event-level hook — it would double-fire on top.
    # - cancel_custom/cancel_slope: Clears state via clear_custom_connect().
    # ──────────────────────────────────────────────────────────────────────────────

    def _active_build(self) -> SegmentBuildContext:
        """The build context for the active kind — one accessor, keyed by kind (no is_road)."""
        return self.context.build(self.active_build_kind)

    def _before_target_from_starting(self, target_location: LonLatElev, target_node: str | None = None) -> None:
        """From *_STARTING: route to the target from the build's origin WITHOUT minting a node.

        The origin node is created only at commit (commit_paths), mirroring lift placement: an
        uncommitted origin is carried as build.start_location and never materialised early.

        start_node is the origin id ONLY when the build began from an existing (connected) node
        (build.start_node_id set by _init_build); a fresh terrain origin stays None and is routed
        from build.start_location by the custom-connect generator.
        """
        self.context.custom_connect.start_node = self._active_build().start_node_id
        self.context.custom_connect.target_location = target_location
        self.context.custom_connect.target_node = target_node

    def _before_target_from_building(self, target_location: LonLatElev, target_node: str | None = None) -> None:
        """From *_BUILDING: route from the active build's current endpoint to the clicked target."""
        endpoints = self._active_build().endpoints
        assert endpoints, "endpoints must be non-empty in *_BUILDING state (segment committed before routing)"
        self.context.custom_connect.start_node = endpoints[0]
        self.context.custom_connect.target_location = target_location
        self.context.custom_connect.target_node = target_node

    def _before_retarget_custom(self, target_location: LonLatElev, target_node: str | None = None) -> None:
        """From *_CUSTOM_PATH: re-route to a newly clicked target (self-loop).

        The start node is unchanged; only the target moves. enter_*_custom_path (fired on
        the self-loop) regenerates proposals, so no deferred flag is set here.
        """
        self.context.custom_connect.target_location = target_location
        self.context.custom_connect.target_node = target_node

    def _before_finish_from_custom(self, entity_id: str) -> None:
        """Sidebar Finish during targeting (any kind): drop the in-progress proposal.

        The finish event carries the entity id; it is unused here — the destination
        viewing state's own before-hook records it.
        """
        self.context.clear_custom_connect()
        self.context.clear_proposals()

    def before_cancel_custom(self) -> None:
        """Event hook for cancel_custom (any kind). Clears custom state and regenerates the fan."""
        self.context.pending.fan_generation.add(self.active_build_kind)
        self.context.clear_custom_connect()
        self.context.clear_proposals()

    def before_cancel_build(self) -> None:
        """Clear the active build + custom state (shared body for cancel_slope / cancel_road)."""
        self.context.clear_custom_connect()
        self.context.clear_proposals()
        self._active_build().clear()

    # Event-named hooks python-statemachine auto-wires to the cancel_* events
    def before_cancel_slope(self) -> None:
        """Event hook for cancel_slope."""
        self.before_cancel_build()

    def before_cancel_road(self) -> None:
        """Event hook for cancel_road."""
        self.before_cancel_build()

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

        # force_* (transition bypass) is legal ONLY inside an undo. undo_running() sets this for the
        # duration of the undo side-effect; force_* asserts it.
        self._undo_in_progress = False

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
        return cast(PlannerContext, self.model)

    @property
    def _active_state(self) -> State:
        """The single active State. This SM is non-parallel, so current_state is one State (the
        library types it as a State | set-of-States union); narrow it once here.
        """
        return cast(State, self.current_state)

    # ==========================================================================
    # Force State Methods (for Undo - bypasses transitions)
    # ==========================================================================
    # These methods allow the action layer to reset the state machine to a
    # stable state after graph undo operations. This follows the expert
    # recommendation to treat undo as a "meta-feature" (history management)
    # rather than core workflow state transitions.

    @contextmanager
    def undo_running(self) -> Iterator[None]:
        """Mark that an undo is in progress so force_* (the transition bypass) is permitted.

        The undo dispatcher wraps its side-effect in `with sm.undo_running():`. Outside this scope
        force_idle/force_building/force_starting raise — the bypass is undo-only by construction, so
        no one can use it as a shortcut in normal flow (which would skip guards/validation).
        """
        self._undo_in_progress = True
        try:
            yield
        finally:
            self._undo_in_progress = False

    def force_idle(self) -> None:
        """Force state machine to IdleReady state without transition.

        Used after undo operations when no building context remains. Clears ALL build
        kinds, custom, and viewing state. Does NOT trigger st.rerun().
        """
        logger.debug(f"[STATE] Forcing state from {self.get_state_name()} to IdleReady")
        self.context.clear_builds()
        self.context.clear_custom_connect()
        self.context.clear_proposals()
        self.context.viewing.clear()
        # Force state machine internal state (calls exit hook for current state)
        self._set_current_state(state=self.idle_ready)
        # Run entry hook to ensure consistent state
        enter_idle_ready(self.context)

    def force_building(self, kind: SegmentKind) -> None:
        """Force state machine to the kind's BUILDING state without transition (undo helper).

        Used after undoing a segment/finish when segments remain. Assumes the caller has
        restored ctx.build(kind). Kind-generic: resolves the State + enter hook from the
        kind's building_state id.
        """
        self._force_fan_state(kind=kind, state_id=KIND_SPECS[kind].building_state)

    def force_starting(self, kind: SegmentKind) -> None:
        """Force state machine to the kind's STARTING state without transition (undo helper).

        Used after undoing the last segment when the origin remains but no segments do.
        """
        self._force_fan_state(kind=kind, state_id=KIND_SPECS[kind].starting_state)

    def _force_fan_state(self, kind: SegmentKind, state_id: str) -> None:
        """Force the machine into one of the kind's fan states (STARTING/BUILDING) after undo.

        force_* bypasses transitions, but it still runs the state's enter hook below, and every
        kind's enter_*_starting/building arms the fan (Single Point of Truth) — so the next deferred
        pass regenerates proposals. Both undo callers rely on that enter-hook arming.
        """
        logger.debug(f"[STATE] Forcing state from {self.get_state_name()} to {kind.value} {state_id}")
        self.context.clear_custom_connect()
        self.context.viewing.clear()
        state: State = getattr(self, state_id)
        assert state is not None, f"state_id {state_id} must resolve to a State object for kind {kind.value}"
        self._set_current_state(state=state)
        self._run_enter_hook(state)

    def _run_enter_hook(self, state: State) -> None:
        """Invoke the on_enter_<state> hook for a force-set state (undo helpers)."""
        getattr(self, f"on_enter_{state.id}")()

    def _set_current_state(self, state: State) -> None:
        """Force the state value directly, running the current state's real exit teardown first.

        Bypasses the normal transition mechanism — undo helpers only. Enter hooks are the
        caller's job (run separately after this). The registered exit hooks only clear in-memory
        context fields, so they cannot fail.
        """
        if not self._undo_in_progress:
            raise RuntimeError(
                "_set_current_state (transition bypass) is undo-only — call it inside `with "
                "sm.undo_running():`. Normal flow must use events/transitions, not force_*."
            )
        # Use .value (snake_case identifier) not .name (CamelCase display name)
        current_state_value = str(self._active_state.value)
        # Only states with real exit cleanup are in EXIT_HOOKS; the rest have no teardown.
        exit_hook = EXIT_HOOKS.get(current_state_value)
        if exit_hook is not None:
            logger.debug(f"[STATE] Calling exit_{current_state_value} before force")
            exit_hook(self.context)
        setattr(self.model, self.state_field, state.value)

    def get_state_name(self) -> str:
        """Get current state name for display."""
        return str(self._active_state.name)

    def get_current_state_id(self) -> str:
        """The active state's snake_case id (matches BUILD_STATES keys / KIND_SPECS state ids)."""
        return str(self._active_state.id)

    def as_mermaid(self) -> str:
        """The full state graph as a Mermaid stateDiagram (dependency-free — no Graphviz needed).

        The library renders the whole state/transition network via its `format()` protocol and marks
        the CURRENT state, so this doubles as a live debug snapshot. Wire it behind a debug flag to
        log "where am I + the whole map" when diagnosing a stuck/again-only-once click (see docs).
        """
        return format(self, "mermaid")

    def __repr__(self) -> str:
        """Return string representation of state machine."""
        return f"PlannerStateMachine(state={self.get_state_name()}, model={self.context!r})"

    @staticmethod
    def create(
        graph: ResortGraph,
        *,
        add_ui_listener: bool = True,
    ) -> tuple[PlannerStateMachine, PlannerContext]:
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
            logger.debug("Created PlannerStateMachine with StreamlitUIListener")
        else:
            logger.debug("Created PlannerStateMachine without UI listener")
        # The state graph is fixed at class-definition time, so dump it ONCE here.
        # Paste this block into https://mermaid.live to see the diagram.
        logger.info("[STATE][GRAPH] Full state machine (paste into mermaid.live):\n%s", sm.as_mermaid())
        return sm, context


def _validate_registries_against_machine() -> None:
    """Fail LOUD at import if EXIT_HOOKS or KIND_SPECS name a state/event this machine lacks.

    Both hold plain strings the action/undo layers dispatch on by name (getattr/send); a typo'd or
    forgotten state/event would otherwise only crash at runtime, deep in a build flow.
    """
    state_ids = {s.id for s in PlannerStateMachine.states}
    event_names = {name for s in PlannerStateMachine.states for t in s.transitions for name in t.event.split()}

    spec_states = {
        sid
        for spec in KIND_SPECS.values()
        for sid in (spec.starting_state, spec.building_state, spec.custom_path_state)
    }
    spec_events = {
        ev
        for spec in KIND_SPECS.values()
        for ev in (
            spec.fan_commit_event,
            spec.custom_continue_event,
            spec.connector_finish_event,
            spec.finish_event,
            spec.cancel_event,
        )
    }
    assert set(EXIT_HOOKS) <= state_ids, f"EXIT_HOOKS names unknown states: {set(EXIT_HOOKS) - state_ids}"
    assert spec_states <= state_ids, f"KIND_SPECS names unknown states: {spec_states - state_ids}"
    assert spec_events <= event_names, f"KIND_SPECS names unknown events: {spec_events - event_names}"


# Import validation
_validate_registries_against_machine()

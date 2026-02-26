"""State Machine Transition Matrix - Parameterized validation of all event/state combinations.

Uses pytest.mark.parametrize to create a data-driven truth table for state transitions.
This serves as executable documentation of the state machine contract.

Test Categories:
    1. Valid transitions: Event fires successfully from allowed source states
    2. Invalid transitions: Event raises TransitionNotAllowed from forbidden states

Matrix Reference (from state_machine.py docstring):
    8 states × 12 events = 96 combinations
    32 valid transitions (28 + 4 self-loops)
    64 invalid transitions
"""

import pytest

from skiresort_planner.ui.state_machine import PlannerStateMachine
from statemachine.exceptions import TransitionNotAllowed


# =============================================================================
# TRUTH TABLE: Valid Transitions
# =============================================================================
# Format: (event_name, valid_source_states, setup_function_name)
# setup_function_name refers to a fixture/helper that prepares guards/context

VALID_TRANSITIONS: list[tuple[str, list[str], str | None]] = [
    # From IDLE_READY
    ("start_slope", ["idle_ready"], "setup_slope_start"),
    ("start_lift", ["idle_ready"], "setup_lift_mode"),
    ("view_slope", ["idle_ready"], "setup_viewed_slope"),
    ("view_lift", ["idle_ready"], "setup_viewed_lift"),
    # From IDLE_VIEWING_SLOPE
    ("close_panel", ["idle_viewing_slope"], None),
    ("view_slope", ["idle_viewing_slope"], "setup_viewed_slope"),  # self-loop (switch)
    ("view_lift", ["idle_viewing_slope"], "setup_viewed_lift"),
    ("start_slope", ["idle_viewing_slope"], "setup_slope_start"),
    ("start_lift", ["idle_viewing_slope"], "setup_lift_mode"),
    # From IDLE_VIEWING_LIFT
    ("close_panel", ["idle_viewing_lift"], None),
    ("view_slope", ["idle_viewing_lift"], "setup_viewed_slope"),
    ("view_lift", ["idle_viewing_lift"], "setup_viewed_lift"),  # self-loop (switch)
    ("start_slope", ["idle_viewing_lift"], "setup_slope_start"),
    ("start_lift", ["idle_viewing_lift"], "setup_lift_mode"),
    # From SLOPE_STARTING
    ("cancel_slope", ["slope_starting"], None),
    ("commit_path", ["slope_starting"], "setup_commit_first"),
    ("enable_custom", ["slope_starting"], None),
    # From SLOPE_BUILDING (with guards)
    ("cancel_slope", ["slope_building"], None),
    ("commit_path", ["slope_building"], "setup_commit_continue"),  # self-loop
    ("enable_custom", ["slope_building"], None),
    # From SLOPE_CUSTOM_PICKING
    ("cancel_slope", ["slope_custom_picking"], None),
    ("cancel_custom", ["slope_custom_picking"], None),
    # From SLOPE_CUSTOM_PATH
    ("cancel_slope", ["slope_custom_path"], None),
    ("cancel_custom", ["slope_custom_path"], None),
    # From LIFT_PLACING
    ("cancel_lift", ["lift_placing"], None),
]


# =============================================================================
# TRUTH TABLE: Invalid Transitions (Events from forbidden states)
# =============================================================================
# Format: (event_name, invalid_source_states)

INVALID_TRANSITIONS: list[tuple[str, list[str]]] = [
    # Cannot start slope from building states
    ("start_slope", ["slope_starting", "slope_building", "slope_custom_picking", "slope_custom_path", "lift_placing"]),
    # Cannot start lift from building states
    ("start_lift", ["slope_starting", "slope_building", "slope_custom_picking", "slope_custom_path", "lift_placing"]),
    # Cannot view slope from building states
    ("view_slope", ["slope_starting", "slope_building", "slope_custom_picking", "slope_custom_path", "lift_placing"]),
    # Cannot view lift from building states
    ("view_lift", ["slope_starting", "slope_building", "slope_custom_picking", "slope_custom_path", "lift_placing"]),
    # Cannot close panel when no panel open
    (
        "close_panel",
        ["idle_ready", "slope_starting", "slope_building", "slope_custom_picking", "slope_custom_path", "lift_placing"],
    ),
    # Cannot cancel slope from non-slope states
    ("cancel_slope", ["idle_ready", "idle_viewing_slope", "idle_viewing_lift", "lift_placing"]),
    # Cannot cancel custom from non-custom states
    (
        "cancel_custom",
        ["idle_ready", "idle_viewing_slope", "idle_viewing_lift", "slope_starting", "slope_building", "lift_placing"],
    ),
    # Cannot commit path from non-building states
    (
        "commit_path",
        [
            "idle_ready",
            "idle_viewing_slope",
            "idle_viewing_lift",
            "slope_custom_picking",
            "slope_custom_path",
            "lift_placing",
        ],
    ),
    # Cannot enable custom from non-slope states
    (
        "enable_custom",
        [
            "idle_ready",
            "idle_viewing_slope",
            "idle_viewing_lift",
            "slope_custom_picking",
            "slope_custom_path",
            "lift_placing",
        ],
    ),
    # Cannot cancel lift from non-lift states
    (
        "cancel_lift",
        [
            "idle_ready",
            "idle_viewing_slope",
            "idle_viewing_lift",
            "slope_starting",
            "slope_building",
            "slope_custom_picking",
            "slope_custom_path",
        ],
    ),
]


class TestTransitionMatrix:
    """Parameterized tests validating the complete state machine transition matrix."""

    @pytest.fixture
    def sm_ctx(self, sm_and_ctx: tuple) -> tuple:
        """Get state machine and context from conftest fixture."""
        return sm_and_ctx

    @pytest.mark.parametrize("event,invalid_states", INVALID_TRANSITIONS)
    def test_invalid_transitions_raise_error(
        self,
        sm_ctx: tuple,
        event: str,
        invalid_states: list[str],
    ) -> None:
        """Invalid transitions raise TransitionNotAllowed.

        Matrix Test: Verifies that calling an event from a forbidden state
        raises the appropriate exception. This is the "Safety Net" that
        prevents impossible state combinations.
        """
        sm, ctx = sm_ctx

        for state_name in invalid_states:
            # Force state machine to the test state
            _force_state(sm=sm, state_name=state_name)

            # Attempt the event - should raise TransitionNotAllowed
            event_func = getattr(sm, event)
            with pytest.raises(TransitionNotAllowed):
                # Call with minimal args (events accept **kwargs)
                event_func()


def _force_state(sm: PlannerStateMachine, state_name: str) -> None:
    """Force state machine to a specific state for testing.

    WARNING: This bypasses normal transition guards. Use only for testing.
    Direct assignment to current_state is supported by python-statemachine v2.
    """
    target_state = getattr(sm, state_name)
    sm.current_state = target_state


# NOTE: Undo transitions removed from state machine.
# Undo is now handled via force_idle()/force_building() methods in the action layer.
# See state_machine.py "Undo Architecture" section for details.


class TestCancelCustomGuards:
    """Tests for cancel_custom event guard resolution.

    The cancel_custom event uses guards to determine destination:
    - cancel_custom_to_starting: when 0 segments → SLOPE_STARTING
    - cancel_custom_to_building: when 1+ segments → SLOPE_BUILDING
    """

    def test_cancel_custom_with_no_segments_goes_to_starting(self, sm_and_ctx: tuple) -> None:
        """Cancel custom with 0 segments returns to SLOPE_STARTING."""
        sm, ctx = sm_and_ctx

        # Setup: Force to slope_custom_picking with no committed segments
        _force_state(sm=sm, state_name="slope_custom_picking")
        ctx.building.segments = []  # No segments committed

        # Act: Call cancel_custom event
        sm.cancel_custom()

        # Assert: Should transition to slope_starting
        assert sm.current_state == sm.slope_starting

    def test_cancel_custom_with_segments_goes_to_building(self, sm_and_ctx: tuple) -> None:
        """Cancel custom with segments returns to SLOPE_BUILDING."""
        sm, ctx = sm_and_ctx

        # Setup: Force to slope_custom_picking with committed segments
        _force_state(sm=sm, state_name="slope_custom_picking")
        ctx.building.segments = ["S1"]  # Has committed segments

        # Act: Call cancel_custom event
        sm.cancel_custom()

        # Assert: Should transition to slope_building
        assert sm.current_state == sm.slope_building

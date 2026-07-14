"""State Machine Transition Matrix - Parameterized validation of all event/state combinations.

Uses pytest.mark.parametrize to create a data-driven truth table for state transitions.
This serves as executable documentation of the state machine contract.

Test Categories:
    1. Valid transitions: Event fires successfully from allowed source states
    2. Invalid transitions: Event raises TransitionNotAllowed from forbidden states

Matrix Reference (from state_machine.py docstring):
    all states, covering slope / lift / road build flows plus viewing states.
    Valid transitions are enumerated in VALID_TRANSITIONS; forbidden ones in
    INVALID_TRANSITIONS. Both are exercised by TestTransitionMatrix.
"""

import pytest
from statemachine.exceptions import TransitionNotAllowed

from skiresort_planner.model.path_segment import SegmentKind
from skiresort_planner.ui.state_machine import PlannerStateMachine
from tests_workflow.conftest import SMAndCtx

# =============================================================================
# TRUTH TABLE: Valid Transitions
# =============================================================================
# Format: (event_name, valid_source_states, expected_target_state_ids)
# expected_target_state_ids is the SET of destination state ids that firing the
# event from each source is allowed to reach (guarded events like cancel_custom
# reach more than one). The test asserts the transition graph maps exactly to it.

VALID_TRANSITIONS: list[tuple[str, list[str], set[str]]] = [
    # From IDLE_READY
    ("start_slope", ["idle_ready"], {"slope_starting"}),
    ("start_lift", ["idle_ready"], {"lift_placing"}),
    ("view_slope", ["idle_ready"], {"idle_viewing_slope"}),
    ("view_lift", ["idle_ready"], {"idle_viewing_lift"}),
    # From IDLE_VIEWING_SLOPE
    ("close_panel", ["idle_viewing_slope"], {"idle_ready"}),
    ("view_slope", ["idle_viewing_slope"], {"idle_viewing_slope"}),  # self-loop (switch)
    ("view_lift", ["idle_viewing_slope"], {"idle_viewing_lift"}),
    ("start_slope", ["idle_viewing_slope"], {"slope_starting"}),
    ("start_lift", ["idle_viewing_slope"], {"lift_placing"}),
    # From IDLE_VIEWING_LIFT
    ("close_panel", ["idle_viewing_lift"], {"idle_ready"}),
    ("view_slope", ["idle_viewing_lift"], {"idle_viewing_slope"}),
    ("view_lift", ["idle_viewing_lift"], {"idle_viewing_lift"}),  # self-loop (switch)
    ("start_slope", ["idle_viewing_lift"], {"slope_starting"}),
    ("start_lift", ["idle_viewing_lift"], {"lift_placing"}),
    # From SLOPE_STARTING
    ("cancel_slope", ["slope_starting"], {"idle_ready"}),
    ("commit_path", ["slope_starting"], {"slope_building"}),
    ("select_custom_target", ["slope_starting"], {"slope_custom_path"}),
    # From SLOPE_BUILDING (with guards)
    ("cancel_slope", ["slope_building"], {"idle_ready"}),
    ("commit_path", ["slope_building"], {"slope_building"}),  # self-loop
    ("select_custom_target", ["slope_building"], {"slope_custom_path"}),
    ("finish_slope", ["slope_building"], {"idle_viewing_slope"}),
    # From SLOPE_CUSTOM_PATH
    ("cancel_slope", ["slope_custom_path"], {"idle_ready"}),
    # cancel_custom is guarded: has_no_segments → slope_starting, else → slope_building
    ("cancel_custom", ["slope_custom_path"], {"slope_starting", "slope_building"}),
    ("select_custom_target", ["slope_custom_path"], {"slope_custom_path"}),  # self-loop (re-target)
    ("finish_slope", ["slope_custom_path"], {"idle_viewing_slope"}),  # sidebar Finish during targeting
    # From LIFT_PLACING
    ("cancel_lift", ["lift_placing"], {"idle_ready"}),
    # From IDLE (road build entry) — road mirrors slope
    (
        "start_road",
        ["idle_ready", "idle_viewing_slope", "idle_viewing_lift", "idle_viewing_road"],
        {"road_starting"},
    ),
    ("view_road", ["idle_ready"], {"idle_viewing_road"}),
    # From ROAD_STARTING / ROAD_BUILDING
    ("cancel_road", ["road_starting"], {"idle_ready"}),
    ("commit_road", ["road_starting"], {"road_building"}),
    ("cancel_road", ["road_building"], {"idle_ready"}),
    ("commit_road", ["road_building"], {"road_building"}),  # self-loop
    ("commit_road_finish", ["road_starting", "road_building"], {"idle_viewing_road"}),  # connector auto-finish
]


# =============================================================================
# TRUTH TABLE: Invalid Transitions (Events from forbidden states)
# =============================================================================
# Format: (event_name, invalid_source_states)

INVALID_TRANSITIONS: list[tuple[str, list[str]]] = [
    # Cannot start slope from building states
    ("start_slope", ["slope_starting", "slope_building", "slope_custom_path", "lift_placing"]),
    # Cannot start lift from building states
    ("start_lift", ["slope_starting", "slope_building", "slope_custom_path", "lift_placing"]),
    # Cannot view slope from building states
    ("view_slope", ["slope_starting", "slope_building", "slope_custom_path", "lift_placing"]),
    # Cannot view lift from building states
    ("view_lift", ["slope_starting", "slope_building", "slope_custom_path", "lift_placing"]),
    # Cannot close panel when no panel open
    (
        "close_panel",
        ["idle_ready", "slope_starting", "slope_building", "slope_custom_path", "lift_placing"],
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
            "slope_custom_path",
            "lift_placing",
        ],
    ),
    # Cannot select a custom target outside slope-building states
    (
        "select_custom_target",
        [
            "idle_ready",
            "idle_viewing_slope",
            "idle_viewing_lift",
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
            "slope_custom_path",
        ],
    ),
    # Cannot cancel road from non-road states
    (
        "cancel_road",
        ["idle_ready", "idle_viewing_slope", "idle_viewing_lift", "slope_starting", "slope_building", "lift_placing"],
    ),
    # Cannot commit road from non-road-build states
    (
        "commit_road",
        ["idle_ready", "idle_viewing_slope", "slope_starting", "slope_building", "lift_placing"],
    ),
    # Cannot fire the road connector-finish outside road-build states
    (
        "commit_road_finish",
        ["idle_ready", "idle_viewing_slope", "slope_starting", "slope_building", "slope_custom_path", "lift_placing"],
    ),
    # Finish_slope is valid from slope_building + slope_custom_path only
    (
        "finish_slope",
        ["idle_ready", "idle_viewing_slope", "idle_viewing_lift", "slope_starting", "lift_placing"],
    ),
]


class TestTransitionMatrix:
    """Parameterized tests validating the complete state machine transition matrix."""

    @pytest.fixture
    def sm_ctx(self, sm_and_ctx: SMAndCtx) -> SMAndCtx:
        """Get state machine and context from conftest fixture."""
        return sm_and_ctx

    @pytest.mark.parametrize("event,valid_states,expected_targets", VALID_TRANSITIONS)
    def test_valid_transitions_are_allowed_from_source(
        self,
        sm_ctx: SMAndCtx,
        event: str,
        valid_states: list[str],
        expected_targets: set[str],
    ) -> None:
        """Each VALID_TRANSITIONS event routes from its source states to the expected targets.

        Asserts the transition graph (not context-guarded firing): from every listed
        source, the set of destination state ids reachable via `event` must equal
        the expected target set exactly. This catches a mis-wired destination
        (e.g. finish_slope routed to idle_ready), not just a missing event.
        """
        sm, _ctx = sm_ctx
        assert hasattr(sm, event), f"event {event} must be defined on the state machine"

        for state_name in valid_states:
            state = getattr(sm, state_name)
            # A transition's `event` may bundle multiple aliases separated by spaces.
            reached_targets = {t.target.id for t in state.transitions for name in t.event.split() if name == event}
            assert reached_targets == expected_targets, (
                f"{event} from {state_name} must reach {expected_targets}, got {reached_targets}"
            )

    @pytest.mark.parametrize("event,invalid_states", INVALID_TRANSITIONS)
    def test_invalid_transitions_raise_error(
        self,
        sm_ctx: SMAndCtx,
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

    @pytest.mark.parametrize("variant", sorted(PlannerStateMachine._EVENT_ONLY_TRANSITIONS))
    def test_direct_variant_calls_are_forbidden(self, sm_ctx: SMAndCtx, variant: str) -> None:
        """Each _EVENT_ONLY_TRANSITIONS variant is blocked at runtime.

        __init__ replaces every listed variant with _forbidden_call, so calling it
        directly (e.g. sm.commit_first_path()) raises RuntimeError telling the caller
        to use the event instead — preventing bypass of the event dispatch.
        """
        sm, _ctx = sm_ctx
        with pytest.raises(RuntimeError, match="forbidden"):
            getattr(sm, variant)()

    def test_event_entry_point_fires_while_its_variant_is_forbidden(self, sm_ctx: SMAndCtx) -> None:
        """commit_first_path is forbidden, but the commit_path EVENT it backs is allowed.

        From slope_starting the commit_path event resolves to commit_first_path and
        drives the SM to slope_building (recording the segment), proving the block is
        scoped to the variant name, not the shared event entry point.
        """
        sm, ctx = sm_ctx
        _force_state(sm=sm, state_name="slope_starting")

        with pytest.raises(RuntimeError, match="forbidden"):
            sm.commit_first_path()

        sm.commit_path(segment_id="S1", endpoint_node_id="N1")  # type: ignore[attr-defined]  # dynamic python-statemachine event
        assert sm.current_state_value == "slope_building"
        assert ctx.build(SegmentKind.SLOPE).segments == ["S1"]


def _force_state(sm: PlannerStateMachine, state_name: str) -> None:
    """Force state machine to a specific state for testing.

    WARNING: This bypasses normal transition guards. Use only for testing.
    Direct assignment to current_state is supported by python-statemachine v2.
    """
    target_state = getattr(sm, state_name)
    sm.current_state = target_state


# NOTE: Undo transitions removed from state machine.
# Undo is now handled via force_idle()/force_building(SegmentKind.SLOPE) methods in the action layer.
# See state_machine.py "Undo Architecture" section for details.


class TestCancelCustomGuards:
    """Tests for cancel_custom event guard resolution.

    The cancel_custom event uses guards to determine destination:
    - cancel_custom_to_starting: when 0 segments → SLOPE_STARTING
    - cancel_custom_to_building: when 1+ segments → SLOPE_BUILDING
    """

    def test_cancel_custom_with_no_segments_goes_to_starting(self, sm_and_ctx: SMAndCtx) -> None:
        """Cancel custom with 0 segments returns to SLOPE_STARTING."""
        sm, ctx = sm_and_ctx

        # Setup: Force to slope_custom_path with no committed segments
        _force_state(sm=sm, state_name="slope_custom_path")
        ctx.build(SegmentKind.SLOPE).segments = []  # No segments committed

        # Act: Call cancel_custom event
        sm.cancel_custom()  # type: ignore[attr-defined]  # dynamic python-statemachine event

        # Assert: Should transition to slope_starting
        assert sm.current_state == sm.slope_starting

    def test_cancel_custom_with_segments_goes_to_building(self, sm_and_ctx: SMAndCtx) -> None:
        """Cancel custom with segments returns to SLOPE_BUILDING."""
        sm, ctx = sm_and_ctx

        # Setup: Force to slope_custom_path with committed segments
        _force_state(sm=sm, state_name="slope_custom_path")
        ctx.build(SegmentKind.SLOPE).segments = ["S1"]  # Has committed segments

        # Act: Call cancel_custom event
        sm.cancel_custom()  # type: ignore[attr-defined]  # dynamic python-statemachine event

        # Assert: Should transition to slope_building
        assert sm.current_state == sm.slope_building


class TestViewingEntity:
    """viewing_entity is the single (EntityKind, id) source for kind-dispatch."""

    def test_none_when_not_viewing(self) -> None:
        from skiresort_planner.model.resort_graph import ResortGraph

        sm, _ = PlannerStateMachine.create(graph=ResortGraph(), add_ui_listener=False)
        assert sm.viewing_entity is None  # idle_ready

    def test_returns_kind_and_id_per_viewed_entity(self) -> None:
        from skiresort_planner.model.resort_graph import ResortGraph
        from skiresort_planner.ui.context import EntityKind

        sm, ctx = PlannerStateMachine.create(graph=ResortGraph(), add_ui_listener=False)

        _force_state(sm=sm, state_name="idle_viewing_slope")
        ctx.viewing.set_slope_id(slope_id="SL1")
        assert sm.viewing_entity == (EntityKind.SLOPE, "SL1")

        _force_state(sm=sm, state_name="idle_viewing_road")
        ctx.viewing.set_road_id(road_id="R2")
        assert sm.viewing_entity == (EntityKind.ROAD, "R2")

        _force_state(sm=sm, state_name="idle_viewing_lift")
        ctx.viewing.set_lift_id(lift_id="L3")
        assert sm.viewing_entity == (EntityKind.LIFT, "L3")


class TestImportPlacing:
    """The click-to-place OSM import mode: start_import from any idle state stores the box center;
    retarget re-places it; cancel/complete return to idle (cancel also clears the center).
    """

    def _sm(self):
        from skiresort_planner.model.resort_graph import ResortGraph

        return PlannerStateMachine.create(graph=ResortGraph(), add_ui_listener=False)

    @pytest.mark.parametrize(
        "idle_state",
        ["idle_ready", "idle_viewing_slope", "idle_viewing_lift", "idle_viewing_road"],
    )
    def test_start_import_from_every_idle_state(self, idle_state) -> None:
        sm, ctx = self._sm()
        _force_state(sm=sm, state_name=idle_state)
        sm.start_import(lon=10.3, lat=47.0)
        assert sm.is_import_placing
        assert ctx.deferred.osm_import_center_lon == 10.3 and ctx.deferred.osm_import_center_lat == 47.0

    def test_retarget_keeps_placing(self) -> None:
        sm, ctx = self._sm()
        sm.start_import(lon=1.0, lat=2.0)
        sm.retarget_import()
        assert sm.is_import_placing, "retarget is a self-loop"

    def test_cancel_returns_to_idle_and_clears_center(self) -> None:
        sm, ctx = self._sm()
        sm.start_import(lon=1.0, lat=2.0)
        sm.cancel_import()
        assert sm.is_idle_ready
        assert ctx.deferred.osm_import_center_lon is None and ctx.deferred.osm_import_center_lat is None
        assert ctx.deferred.osm_import is False

    def test_complete_returns_to_idle(self) -> None:
        sm, ctx = self._sm()
        sm.start_import(lon=1.0, lat=2.0)
        sm.complete_import()
        assert sm.is_idle_ready

    def test_import_placing_is_not_idle(self) -> None:
        sm, _ = self._sm()
        sm.start_import(lon=1.0, lat=2.0)
        assert not sm.is_idle and not sm.is_any_slope_state and not sm.is_any_road_state

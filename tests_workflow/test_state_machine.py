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

from skiresort_planner.model.path_point import PathPoint
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


class TestEventOnlyTransitionsCompleteness:
    """Structural guard: _EVENT_ONLY_TRANSITIONS must list EXACTLY the variant transitions.

    A "variant" is a transition whose attribute name differs from the event string it fires
    (e.g. start_slope_from_slope_view fires the "start_slope" event; select_target_from_building
    fires "select_custom_target"). Every variant MUST be blocked from direct calls so callers go
    through the shared event — otherwise a direct call bypasses the event dispatch and its guards.

    Transitions whose attribute name IS the event string (start_slope, commit_path, close_panel,
    finish_road, commit_custom_continue, …) are the event entry points / direct transitions and
    must stay callable — they must NOT appear in the set.

    This computes the variant set by introspecting the state machine itself, so adding a new
    kind/utility with `_from_*` entry variants (like the import/merge ones) can never again ship
    unblocked: the frozenset must equal the computed set exactly, or this test fails.
    """

    @staticmethod
    def _variant_transition_names() -> set[str]:
        """Attribute names of transitions whose name != the event string they trigger.

        Introspects the class Event descriptors: each attribute-defined transition carries its
        triggering event string(s) on ``_transitions``. Shared event-id entries (start_slope,
        commit_path, …) have no ``_transitions`` and are skipped — those are the entry points.
        """
        from statemachine.event import Event

        variants: set[str] = set()
        for attr, value in vars(PlannerStateMachine).items():
            if not isinstance(value, Event):
                continue
            transitions = getattr(value, "_transitions", None)
            if transitions is None:
                continue  # shared event entry point (e.g. start_slope), not a variant attr
            event_strings = {t.event for t in transitions}
            if attr not in event_strings:
                variants.add(attr)
        return variants

    def test_frozenset_lists_exactly_the_variant_transitions(self) -> None:
        computed = self._variant_transition_names()
        listed = set(PlannerStateMachine._EVENT_ONLY_TRANSITIONS)
        missing = computed - listed
        stray = listed - computed
        assert not missing, (
            f"_EVENT_ONLY_TRANSITIONS is missing variant transitions that could be called "
            f"directly, bypassing event dispatch: {sorted(missing)}"
        )
        assert not stray, (
            f"_EVENT_ONLY_TRANSITIONS lists names that are NOT variant transitions (they are "
            f"event entry points / direct transitions and must stay callable): {sorted(stray)}"
        )


def _force_state(sm: PlannerStateMachine, state_name: str) -> None:
    """Force state machine to a specific state for testing.

    WARNING: This bypasses normal transition guards. Use only for testing.
    Direct assignment to current_state is supported by python-statemachine v2.
    """
    target_state = getattr(sm, state_name)
    sm.current_state = target_state


class TestStartHookParity:
    """Slope and road are built identically (one SegmentBuildContext, one KIND_SPECS entry, one
    unified state class + click handler). Their `before_start_*` hooks must therefore leave the
    context in the SAME shape, or the shared overlay/panel code silently diverges between kinds.

    These guards pin the two invariants the shared code depends on:
      1. An in-build name is set at start (the panel reads build.name; a None → "Unnamed X").
      2. ctx.selection is populated at start (the unified overlay draws orientation arrows from it).
    """

    def test_slope_and_road_start_both_set_an_in_build_name(self, sm_and_ctx: SMAndCtx) -> None:
        sm, ctx = sm_and_ctx
        sm.start_slope(lon=0.0, lat=0.0, elevation=2000.0, node_id=None)
        assert ctx.build(SegmentKind.SLOPE).name, "slope start must set an in-build name"

        sm.cancel_slope()  # back to idle
        sm.start_road(location=PathPoint(lon=0.0, lat=0.0, elevation=2000.0))
        assert ctx.build(SegmentKind.ROAD).name, "road start must set an in-build name (parity with slope)"

    def test_slope_and_road_start_both_populate_selection(self, sm_and_ctx: SMAndCtx) -> None:
        """before_start_* must set ctx.selection so the unified overlay can draw orientation arrows.

        Regression: only the slope path set the selection (in the hook AND in the click handler),
        so a road started via the SM event left selection empty and drew no arrows.
        """
        sm, ctx = sm_and_ctx
        sm.start_slope(lon=1.0, lat=2.0, elevation=2000.0, node_id=None)
        assert ctx.selection.has_selection(), "slope start must populate selection"
        assert (ctx.selection.lon, ctx.selection.lat) == (1.0, 2.0)

        sm.cancel_slope()
        sm.start_road(location=PathPoint(lon=3.0, lat=4.0, elevation=2000.0))
        assert ctx.selection.has_selection(), "road start must populate selection (parity with slope)"
        assert (ctx.selection.lon, ctx.selection.lat) == (3.0, 4.0)


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


class TestViewSwitching:
    """Switching between viewed entities (slope↔road↔lift) fires the before_switch_* hooks that
    set the newly-viewed id. Exercises the full switch chain including the road-view transitions.
    """

    @staticmethod
    def _sm() -> SMAndCtx:
        from skiresort_planner.model.resort_graph import ResortGraph

        return PlannerStateMachine.create(graph=ResortGraph(), add_ui_listener=False)

    def test_switch_chain_updates_viewed_id_and_state(self) -> None:
        from skiresort_planner.ui.context import EntityKind

        sm, _ctx = self._sm()
        # idle_ready → view a slope → switch to a road → self-loop to another road → switch to a lift.
        # viewing_entity encodes both the state's kind and the id set by the before_switch_* hook.
        sm.show_slope_info_panel(slope_id="SL1")
        assert sm.viewing_entity == (EntityKind.SLOPE, "SL1")

        sm.show_road_info_panel(road_id="R1")  # switch_slope_to_road_view
        assert sm.viewing_entity == (EntityKind.ROAD, "R1")

        sm.show_road_info_panel(road_id="R2")  # switch_road self-loop
        assert sm.viewing_entity == (EntityKind.ROAD, "R2")

        sm.show_lift_info_panel(lift_id="L1")  # switch_road_to_lift_view
        assert sm.viewing_entity == (EntityKind.LIFT, "L1")

        sm.show_road_info_panel(road_id="R3")  # switch_lift_to_road_view
        assert sm.viewing_entity == (EntityKind.ROAD, "R3")

    def test_switch_road_to_slope_then_close(self) -> None:
        from skiresort_planner.ui.context import EntityKind

        sm, _ctx = self._sm()
        sm.show_road_info_panel(road_id="R1")
        sm.show_slope_info_panel(slope_id="SL9")  # switch_road_to_slope_view
        assert sm.viewing_entity == (EntityKind.SLOPE, "SL9")
        sm.hide_info_panel()  # close_panel → idle_ready
        assert sm.viewing_entity is None

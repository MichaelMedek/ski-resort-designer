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
from skiresort_planner.ui.context import BuildMode
from skiresort_planner.ui.kind_spec import KIND_SPECS
from skiresort_planner.ui.mode_registry import OPERATIONS
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


class TestKindSpecResolvesAgainstStateMachine:
    """Every KIND_SPECS state id / event name must resolve to a REAL state/event on the machine.

    KIND_SPECS (ui/kind_spec.py) stores state ids ("road_starting") and event names ("commit_road")
    as plain strings; the action + undo layers dispatch on them by name. A typo or a forgotten
    transition when adding a new SegmentKind would otherwise pass import and only crash at runtime
    deep in a build flow. The import-time asserts at the bottom of state_machine.py catch it; these
    tests exercise the same invariant explicitly (and prove the check itself is wired to reality).
    """

    @staticmethod
    def _sm_state_ids() -> set[str]:
        return {s.id for s in PlannerStateMachine.states}

    @staticmethod
    def _sm_event_names() -> set[str]:
        # A transition's `event` bundles space-separated aliases (e.g. "commit_path commit_first_path").
        return {name for s in PlannerStateMachine.states for t in s.transitions for name in t.event.split()}

    def test_every_kindspec_state_id_is_a_real_state(self) -> None:
        from skiresort_planner.ui.kind_spec import KIND_SPECS

        state_ids = self._sm_state_ids()
        for kind, spec in KIND_SPECS.items():
            for attr in ("starting_state", "building_state", "custom_path_state"):
                sid = getattr(spec, attr)
                assert sid in state_ids, f"KIND_SPECS[{kind}].{attr}={sid!r} is not a state-machine state"

    def test_every_kindspec_event_name_is_a_real_event(self) -> None:
        from skiresort_planner.ui.kind_spec import KIND_SPECS

        events = self._sm_event_names()
        for kind, spec in KIND_SPECS.items():
            for attr in (
                "fan_commit_event",
                "custom_continue_event",
                "connector_finish_event",
                "finish_event",
                "cancel_event",
            ):
                ev = getattr(spec, attr)
                assert ev in events, f"KIND_SPECS[{kind}].{attr}={ev!r} is not a state-machine event"

    def test_is_any_path_state_covers_every_kind_build_state(self) -> None:
        """is_any_path_state must be True in EVERY kind's 3 build states and False elsewhere.

        Guards the drift `is_any_slope_state or is_any_road_state` had: a new SegmentKind's states are
        covered automatically because is_any_path_state derives from KIND_SPECS.
        """
        from skiresort_planner.model.resort_graph import ResortGraph
        from skiresort_planner.ui.kind_spec import KIND_SPECS

        sm, _ = PlannerStateMachine.create(graph=ResortGraph(), add_ui_listener=False)
        build_state_ids = {
            sid
            for spec in KIND_SPECS.values()
            for sid in (spec.starting_state, spec.building_state, spec.custom_path_state)
        }
        for sid in self._sm_state_ids():
            _force_state(sm, sid)
            expected = sid in build_state_ids
            assert sm.is_any_path_state is expected, f"is_any_path_state wrong in state {sid!r} (expected {expected})"


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

        sm.send("cancel_slope")  # back to idle
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

        sm.send("cancel_slope")
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


class TestImportSelecting:
    """The click-to-place OSM import mode: start_import from any idle state stores the box center;
    retarget re-places it; cancel/complete return to idle (cancel also clears the center).
    """

    def _sm(self):
        from skiresort_planner.model.resort_graph import ResortGraph

        return PlannerStateMachine.create(graph=ResortGraph(), add_ui_listener=False)

    def test_start_import_from_idle_ready(self) -> None:
        # Import is a utility: it starts ONLY from idle_ready (no "own kind" to click from a view).
        sm, ctx = self._sm()
        sm.start_import(lon=10.3, lat=47.0)
        assert sm.is_import_selecting
        assert ctx.pending.osm_import_center_lon == 10.3 and ctx.pending.osm_import_center_lat == 47.0

    @pytest.mark.parametrize("view_state", ["idle_viewing_slope", "idle_viewing_lift", "idle_viewing_road"])
    def test_utility_starts_not_allowed_from_a_view(self, view_state) -> None:
        # Regression: the utilities (import/merge/route) must NOT be enterable from a viewing state —
        # unlike the builders (slope/lift/road), they have no own kind to click, so they are idle-only.
        # The transitions simply do not exist, so the SM refuses the event.
        from statemachine.exceptions import TransitionNotAllowed

        for event in ("start_import", "start_merge", "start_route"):
            sm, _ = self._sm()
            _force_state(sm=sm, state_name=view_state)
            with pytest.raises(TransitionNotAllowed):
                sm.send(event)

    def test_retarget_keeps_placing(self) -> None:
        sm, ctx = self._sm()
        sm.start_import(lon=1.0, lat=2.0)
        sm.retarget_import()
        assert sm.is_import_selecting, "retarget is a self-loop"

    def test_cancel_returns_to_idle_and_clears_center(self) -> None:
        sm, ctx = self._sm()
        sm.start_import(lon=1.0, lat=2.0)
        sm.cancel_import()
        assert sm.is_idle_ready
        assert ctx.pending.osm_import_center_lon is None and ctx.pending.osm_import_center_lat is None
        assert ctx.pending.osm_import_mode is None

    def test_complete_returns_to_idle(self) -> None:
        sm, ctx = self._sm()
        sm.start_import(lon=1.0, lat=2.0)
        sm.complete_import()
        assert sm.is_idle_ready

    def test_import_selecting_is_not_idle(self) -> None:
        sm, _ = self._sm()
        sm.start_import(lon=1.0, lat=2.0)
        assert not sm.is_idle and not sm.is_any_slope_state and not sm.is_any_road_state

    def test_force_idle_from_import_runs_exit_teardown(self) -> None:
        # force_idle (the undo path) bypasses transitions but MUST still run the real exit teardown
        # via _set_current_state → _EXIT_HOOKS. Undoing an OSM import from import_selecting has to clear
        # the placed box center, or a stale box resurfaces. Guards the force/undo exit dispatch.
        sm, ctx = self._sm()
        sm.start_import(lon=1.0, lat=2.0)
        assert ctx.pending.osm_import_center_lon == 1.0

        with sm.undo_running():  # force_* is undo-only
            sm.force_idle()

        assert sm.is_idle_ready
        assert ctx.pending.osm_import_center_lon is None, "force_idle must run exit_import_selecting (clears center)"
        assert ctx.pending.osm_import_center_lat is None


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
        sm.view_slope(slope_id="SL1")
        assert sm.viewing_entity == (EntityKind.SLOPE, "SL1")

        sm.view_road(road_id="R1")  # switch_slope_to_road_view
        assert sm.viewing_entity == (EntityKind.ROAD, "R1")

        sm.view_road(road_id="R2")  # switch_road self-loop
        assert sm.viewing_entity == (EntityKind.ROAD, "R2")

        sm.view_lift(lift_id="L1")  # switch_road_to_lift_view
        assert sm.viewing_entity == (EntityKind.LIFT, "L1")

        sm.view_road(road_id="R3")  # switch_lift_to_road_view
        assert sm.viewing_entity == (EntityKind.ROAD, "R3")

    def test_switch_road_to_slope_then_close(self) -> None:
        from skiresort_planner.ui.context import EntityKind

        sm, _ctx = self._sm()
        sm.view_road(road_id="R1")
        sm.view_slope(slope_id="SL9")  # switch_road_to_slope_view
        assert sm.viewing_entity == (EntityKind.SLOPE, "SL9")
        sm.close_panel()  # type: ignore[attr-defined]  # dynamic python-statemachine event → idle_ready
        assert sm.viewing_entity is None


class TestStateGraphIsComplete:
    """The workflow graph must be sound: every state reachable from the initial state, and every
    state able to reach idle_ready (no dead-ends the user can get stuck in). Uses the library's own
    graph (state.transitions with .source/.target) so this tracks the real machine, not a copy.

    NOTE: this checks the FORWARD workflow graph only. Undo deliberately jumps to prior states the
    forward graph doesn't connect (e.g. idle_viewing_slope → slope_building after undoing a finish),
    which is why undo uses force_* instead of transitions — see the module docstring / docs/workflows.
    """

    def _edges(self, sm) -> dict[str, set[str]]:
        return {s.id: {t.target.id for t in s.transitions} for s in sm.states}

    def _reachable(self, edges: dict[str, set[str]], start: str) -> set[str]:
        seen: set[str] = set()
        stack = [start]
        while stack:
            n = stack.pop()
            if n in seen:
                continue
            seen.add(n)
            stack.extend(edges.get(n, ()))
        return seen

    def test_every_state_reachable_from_initial(self) -> None:
        from skiresort_planner.model.resort_graph import ResortGraph

        sm, _ = PlannerStateMachine.create(graph=ResortGraph(), add_ui_listener=False)
        edges = self._edges(sm)
        initial = next(s.id for s in sm.states if s.initial)
        reachable = self._reachable(edges, initial)
        assert set(edges) == reachable, f"states unreachable from {initial}: {set(edges) - reachable}"

    def test_every_state_can_return_to_idle_ready(self) -> None:
        # No workflow dead-end: from any state the user can always get back to idle_ready via
        # transitions (cancel/close/finish). Check on the reversed graph: idle_ready must reach all.
        from skiresort_planner.model.resort_graph import ResortGraph

        sm, _ = PlannerStateMachine.create(graph=ResortGraph(), add_ui_listener=False)
        edges = self._edges(sm)
        reverse: dict[str, set[str]] = {s: set() for s in edges}
        for src, tgts in edges.items():
            for t in tgts:
                reverse[t].add(src)
        can_reach_idle = self._reachable(reverse, "idle_ready")
        assert set(edges) == can_reach_idle, f"states with no path back to idle_ready: {set(edges) - can_reach_idle}"

    def test_graph_is_strongly_connected(self) -> None:
        # The strongest "complete network" property: every state can reach every other state. For a
        # workflow with idle_ready as the hub this follows from (reachable-from-initial) +
        # (all-can-return-to-idle), but asserting it directly catches any future island of states
        # that link among themselves yet detach from the rest. (Kosaraju: one SCC == the whole graph.)
        from skiresort_planner.model.resort_graph import ResortGraph

        sm, _ = PlannerStateMachine.create(graph=ResortGraph(), add_ui_listener=False)
        edges = self._edges(sm)
        reverse: dict[str, set[str]] = {s: set() for s in edges}
        for src, tgts in edges.items():
            for t in tgts:
                reverse[t].add(src)
        any_state = next(iter(edges))
        forward = self._reachable(edges, any_state)
        backward = self._reachable(reverse, any_state)
        # A graph is strongly connected iff every node is both reachable from and can reach one node.
        assert forward == set(edges) and backward == set(edges), (
            f"state graph is NOT strongly connected — from {any_state}: "
            f"cannot reach {set(edges) - forward}; cannot be reached by {set(edges) - backward}"
        )


class TestAsMermaid:
    """as_mermaid() is dumped ONCE at startup by PlannerStateMachine.create() (the state graph is
    fixed at class-definition time, so re-dumping per transition would only spam the log). It
    delegates to python-statemachine's `format(sm, "mermaid")` spec (added in 3.1.0). Two things
    must hold: the call must never raise (a crash here took down the whole render fragment — the
    3.0.0 library had no mermaid spec, so format() hit object.__format__ and raised TypeError), and
    it must stay dependency-free (no Graphviz/pydot). These guard the pinned floor (>=3.1.0).
    """

    def _sm(self) -> PlannerStateMachine:
        from skiresort_planner.model.resort_graph import ResortGraph

        sm, _ = PlannerStateMachine.create(graph=ResortGraph(), add_ui_listener=False)
        return sm

    def test_produces_a_mermaid_state_diagram(self) -> None:
        out = self._sm().as_mermaid()
        assert out.startswith("stateDiagram-v2"), "must emit a Mermaid stateDiagram header"
        assert "[*] --> idle_ready" in out, "initial state must be marked"

    def test_lists_every_state(self) -> None:
        sm = self._sm()
        out = sm.as_mermaid()
        for state in sm.states:
            assert state.id in out, f"state {state.id} missing from mermaid dump"

    def test_highlights_the_current_state(self) -> None:
        sm = self._sm()
        sm.start_slope(lon=0.0, lat=0.0, elevation=2000.0, node_id=None)
        out = sm.as_mermaid()
        # The library marks the active state with a `:::active` class and defines that classDef.
        assert f"{sm.current_state_value}:::active" in out, "active state must be highlighted"
        assert "classDef active" in out, "active classDef must be defined"

    def test_never_raises_across_transitions(self) -> None:
        # create() calls as_mermaid() at startup; it must never raise. This is the direct
        # regression guard for the 3.0.0 TypeError. Re-check after transitions for good measure.
        sm = self._sm()
        sm.start_slope(lon=0.0, lat=0.0, elevation=2000.0, node_id=None)
        sm.send("cancel_slope")
        sm.start_road(location=PathPoint(lon=0.0, lat=0.0, elevation=2000.0))
        assert sm.as_mermaid().startswith("stateDiagram-v2")


# =============================================================================
# OPERATION TAXONOMY vs STATE MACHINE STRUCTURE
# =============================================================================
# Every left-panel OPERATION maps onto the state machine in ONE of three shapes. These tests derive
# the expected structure from the registries (OPERATIONS / KIND_SPECS) and assert the live state
# graph matches, so an inconsistent machine fails here until fixed.
#
#   PATH ops           (slope, road)   builder, multi-segment
#       -> states: <p>_starting, <p>_building, <p>_custom_path  + viewing idle_viewing_<p>
#   POINT-TO-POINT ops (lift, route)   two picks placed on the map, then a result to view
#       -> state:  <p>_placing                                  + viewing idle_viewing_<p>
#   POINT-ONLY ops     (import, merge) SELECT an area/nodes, mutate the graph, return to idle
#       -> state:  <p>_selecting                                + NO viewing state


def _op_prefix(mode: str) -> str:
    """The state-id prefix an operation's states use. The four lift types collapse to 'lift' (one
    lift_placing/idle_viewing_lift), matching the SM naming; every other op is its own mode string.
    """
    return "lift" if BuildMode.is_lift(mode) else mode


# Category prefixes, DERIVED from the registries (not hand-listed):
#  - path: the KIND_SPECS kinds (the only multi-segment builders).
#  - point-to-point: a builder that is not a path kind (lift), plus route (a utility that yields a view).
#  - point-only: the remaining utilities (import, merge) — mutate then return to idle.
_PATH_PREFIXES = {spec.kind.value for spec in KIND_SPECS.values()}
_POINT_TO_POINT_PREFIXES = {
    _op_prefix(m)
    for m, op in OPERATIONS.items()
    if _op_prefix(m) not in _PATH_PREFIXES and (op.group == "builder" or m == BuildMode.ROUTE)
}
_POINT_ONLY_PREFIXES = {
    _op_prefix(m) for m in OPERATIONS if _op_prefix(m) not in (_PATH_PREFIXES | _POINT_TO_POINT_PREFIXES)
}


def _sm_state_ids_set() -> set[str]:
    return {s.id for s in PlannerStateMachine.states}


def _sm_edges() -> dict[str, set[str]]:
    """State id -> set of directly reachable target state ids (live library graph)."""
    return {s.id: {t.target.id for t in s.transitions} for s in PlannerStateMachine.states}


class TestOperationTaxonomy:
    """The three categories partition the operation set, and each op maps to the right SM states."""

    def test_categories_partition_all_operations(self) -> None:
        all_prefixes = {_op_prefix(m) for m in OPERATIONS}
        union = _PATH_PREFIXES | _POINT_TO_POINT_PREFIXES | _POINT_ONLY_PREFIXES
        assert union == all_prefixes, f"categories miss/extra vs OPERATIONS: {union ^ all_prefixes}"
        assert not (_PATH_PREFIXES & _POINT_TO_POINT_PREFIXES)
        assert not (_PATH_PREFIXES & _POINT_ONLY_PREFIXES)
        assert not (_POINT_TO_POINT_PREFIXES & _POINT_ONLY_PREFIXES)

    def test_expected_membership(self) -> None:
        # Pin concrete membership so a mis-categorised op (e.g. route slipping into point-only) fails.
        assert {"slope", "road"} == _PATH_PREFIXES
        assert {"lift", "route"} == _POINT_TO_POINT_PREFIXES
        assert {"import", "merge"} == _POINT_ONLY_PREFIXES

    @pytest.mark.parametrize("prefix", sorted(_PATH_PREFIXES))
    def test_path_op_has_starting_building_custompath_and_viewing(self, prefix: str) -> None:
        states = _sm_state_ids_set()
        for suffix in ("starting", "building", "custom_path"):
            assert f"{prefix}_{suffix}" in states, f"path op {prefix} missing {prefix}_{suffix}"
        assert f"idle_viewing_{prefix}" in states, f"path op {prefix} missing its viewing state"
        assert f"{prefix}_placing" not in states, f"path op {prefix} must NOT have a *_placing state"
        assert f"{prefix}_selecting" not in states, f"path op {prefix} must NOT have a *_selecting state"

    @pytest.mark.parametrize("prefix", sorted(_POINT_TO_POINT_PREFIXES))
    def test_point_to_point_op_has_placing_and_viewing(self, prefix: str) -> None:
        states = _sm_state_ids_set()
        assert f"{prefix}_placing" in states, f"point-to-point op {prefix} missing {prefix}_placing"
        assert f"idle_viewing_{prefix}" in states, f"point-to-point op {prefix} missing its viewing state"
        for suffix in ("starting", "building", "custom_path", "selecting"):
            assert f"{prefix}_{suffix}" not in states, f"point-to-point op {prefix} must not have {prefix}_{suffix}"

    @pytest.mark.parametrize("prefix", sorted(_POINT_ONLY_PREFIXES))
    def test_point_only_op_has_selecting_and_no_viewing(self, prefix: str) -> None:
        # Point-only utilities SELECT an area/nodes (they don't place map points), so their state is
        # <p>_selecting, NOT <p>_placing — and they have nothing to view afterwards.
        states = _sm_state_ids_set()
        assert f"{prefix}_selecting" in states, f"point-only op {prefix} missing {prefix}_selecting"
        assert f"idle_viewing_{prefix}" not in states, (
            f"point-only op {prefix} mutates then returns to idle — it must have NO viewing state"
        )
        for suffix in ("starting", "building", "custom_path", "placing"):
            assert f"{prefix}_{suffix}" not in states, f"point-only op {prefix} must NOT have {prefix}_{suffix}"


class TestOperationTransitions:
    """Each operation's entry/exit wiring, derived from its category and asserted on the live graph."""

    def test_idle_ready_enters_every_operation(self) -> None:
        # idle_ready is the single hub: it must start EVERY operation (its entry state).
        targets = _sm_edges()["idle_ready"]
        for prefix in _PATH_PREFIXES:
            assert f"{prefix}_starting" in targets, f"idle_ready cannot start path op {prefix}"
        for prefix in _POINT_TO_POINT_PREFIXES:
            assert f"{prefix}_placing" in targets, f"idle_ready cannot start point-to-point op {prefix}"
        for prefix in _POINT_ONLY_PREFIXES:
            assert f"{prefix}_selecting" in targets, f"idle_ready cannot start point-only op {prefix}"

    def test_idle_ready_opens_every_entity_viewing_state(self) -> None:
        # The entity viewers (slope/road/lift) open directly from idle by clicking an entity. Route's
        # viewing state is reached via its placing flow (complete_route), NOT from idle.
        targets = _sm_edges()["idle_ready"]
        for prefix in _PATH_PREFIXES | {"lift"}:
            assert f"idle_viewing_{prefix}" in targets, f"idle_ready cannot view {prefix}"

    @pytest.mark.parametrize("prefix", sorted(_PATH_PREFIXES | _POINT_TO_POINT_PREFIXES))
    def test_op_reaches_its_viewing_state(self, prefix: str) -> None:
        # Every result-producing op (path + point-to-point) lands in ITS viewing state when it
        # finishes/completes — from its terminal build/placing state.
        edges = _sm_edges()
        viewing = f"idle_viewing_{prefix}"
        source = f"{prefix}_building" if prefix in _PATH_PREFIXES else f"{prefix}_placing"
        assert viewing in edges[source], f"{source} cannot finish into {viewing}"

    @pytest.mark.parametrize("prefix", sorted(_PATH_PREFIXES))
    def test_path_op_custom_path_reaches_its_viewing_state(self, prefix: str) -> None:
        # Every custom path op lands in ITS viewing state when it
        # finishes/completes — from its alternative terminal custom path state.
        edges = _sm_edges()
        viewing = f"idle_viewing_{prefix}"
        source = f"{prefix}_custom_path"
        assert viewing in edges[source], f"{source} cannot finish into {viewing}"

    @pytest.mark.parametrize("prefix", sorted(_PATH_PREFIXES))
    def test_path_op_custom_path_reachable_from_its_build_state(self, prefix: str) -> None:
        # The custom_path state is entered (select_custom_target) from BOTH fan states — the
        # 0-segment starting state and the 1+-segment building state.
        edges = _sm_edges()
        custom = f"{prefix}_custom_path"
        assert custom in edges[f"{prefix}_starting"], f"{prefix}_starting cannot reach {custom}"
        assert custom in edges[f"{prefix}_building"], f"{prefix}_building cannot reach {custom}"

    @pytest.mark.parametrize("prefix", sorted(_PATH_PREFIXES))
    def test_path_op_custom_path_reaches_its_build_state(self, prefix: str) -> None:
        # From custom_path, committing/continuing (or cancel-with-segments) returns to the building state.
        edges = _sm_edges()
        building = f"{prefix}_building"
        assert building in edges[f"{prefix}_custom_path"], f"{prefix}_custom_path cannot reach {building}"

    def test_every_state_returns_to_idle_ready(self) -> None:
        # No dead-ends: every state can reach idle_ready (transitively) via cancel/close/finish.
        edges = _sm_edges()
        reverse: dict[str, set[str]] = {s: set() for s in edges}
        for src, tgts in edges.items():
            for t in tgts:
                reverse[t].add(src)
        seen: set[str] = set()
        stack = ["idle_ready"]
        while stack:
            n = stack.pop()
            if n in seen:
                continue
            seen.add(n)
            stack.extend(reverse[n])
        assert set(edges) == seen, f"states with no path back to idle_ready: {set(edges) - seen}"

    @pytest.mark.parametrize("prefix", sorted(_PATH_PREFIXES | _POINT_TO_POINT_PREFIXES))
    def test_viewing_state_closes_to_idle_ready(self, prefix: str) -> None:
        assert "idle_ready" in _sm_edges()[f"idle_viewing_{prefix}"], f"idle_viewing_{prefix} can't close to idle"

    def test_entity_viewers_switch_among_each_other(self) -> None:
        # The three entity viewers can switch directly to each other (click a different entity while a
        # panel is open). Route-view can switch INTO them too.
        edges = _sm_edges()
        entity_viewers = ["idle_viewing_slope", "idle_viewing_road", "idle_viewing_lift"]
        for src in entity_viewers:
            for dst in entity_viewers:
                if src != dst:
                    assert dst in edges[src], f"{src} cannot switch to {dst}"
        for dst in entity_viewers:
            assert dst in edges["idle_viewing_route"], f"idle_viewing_route cannot switch to {dst}"

"""Tests for the dispatch registries (ui/mode_registry.py).

Every dispatch axis is covered EXACTLY, so a new state/mode/kind can't silently forget a part.
The bijection asserts run at import (so a gap crashes on import), and these tests re-assert them.
"""

import pytest

from skiresort_planner.model.resort_graph import ResortGraph
from skiresort_planner.ui.context import BuildMode, EntityKind, PlannerContext
from skiresort_planner.ui.mode_registry import (
    BUILD_STATES,
    ENTITY_KIND_SPECS,
    OPERATIONS,
    OperationGroup,
    StateHeader,
)
from skiresort_planner.ui.state_machine import PlannerStateMachine


class TestBuildStateHeader:
    """Every state must supply a non-empty header + a bool blocks_build_buttons (abstract methods,
    so this can't be forgotten; this test also pins that headers render for all states).
    """

    def test_every_state_has_a_valid_header(self) -> None:
        ctx = PlannerContext()
        for key, bs in BUILD_STATES.items():
            head = bs.header(ctx)
            assert isinstance(head, StateHeader), key
            assert head.icon and head.label, f"{key} header has empty icon/label"
            assert isinstance(bs.blocks_build_buttons(), bool), key

    def test_placing_and_building_states_block_buttons(self) -> None:
        # Every non-idle state blocks the build-mode buttons; idle_* states do not.
        for key, bs in BUILD_STATES.items():
            expected = not key.startswith("idle_")
            assert bs.blocks_build_buttons() is expected, key


class TestBuildStateBijection:
    def test_keys_match_sm_state_ids_exactly(self) -> None:
        sm_ids = {s.id for s in PlannerStateMachine.states}
        assert set(BUILD_STATES) == sm_ids

    def test_merge_placing_is_registered(self) -> None:
        assert "merge_placing" in BUILD_STATES


class TestExitHookBijection:
    """Every SM state must have an exit hook; _set_current_state does a direct [] lookup, so a
    missing hook KeyError-crashes the render during any force_* (e.g. undoing an OSM import from
    import_placing — the bug this guards). Mirrors the import-time assert in state_machine.py.
    """

    def test_exit_hooks_cover_every_state(self) -> None:
        sm_ids = {s.id for s in PlannerStateMachine.states}
        assert set(PlannerStateMachine._EXIT_HOOKS) == sm_ids

    def test_import_and_merge_placing_have_exit_hooks(self) -> None:
        # These two were the states missing from _EXIT_HOOKS (the live bug).
        assert "import_placing" in PlannerStateMachine._EXIT_HOOKS
        assert "merge_placing" in PlannerStateMachine._EXIT_HOOKS


class TestOperationBijection:
    def test_keys_match_buildmode_values_exactly(self) -> None:
        modes = {
            BuildMode.SLOPE,
            BuildMode.ROAD,
            BuildMode.CHAIRLIFT,
            BuildMode.GONDOLA,
            BuildMode.SURFACE_LIFT,
            BuildMode.AERIAL_TRAM,
            BuildMode.IMPORT,
            BuildMode.MERGE,
        }
        assert set(OPERATIONS) == modes

    def test_import_and_merge_are_utility_group(self) -> None:
        assert OPERATIONS[BuildMode.IMPORT].group == OperationGroup.UTILITY
        assert OPERATIONS[BuildMode.MERGE].group == OperationGroup.UTILITY

    def test_builders_are_builder_group(self) -> None:
        for mode in (BuildMode.SLOPE, BuildMode.ROAD, BuildMode.CHAIRLIFT, BuildMode.AERIAL_TRAM):
            assert OPERATIONS[mode].group == OperationGroup.BUILDER

    def test_every_operation_has_a_first_instruction(self) -> None:
        # Abstract on BuilderOperation, so a new mode button can't forget the idle first-click hint.
        for mode, op in OPERATIONS.items():
            assert op.first_instruction.strip(), f"{mode} has an empty first_instruction"

    def test_on_select_highlights_the_mode_for_every_operation(self, fake_st, empty_graph) -> None:
        # The shared invariant lives on the base class: on_select highlights its OWN mode (no state
        # entry — the first map click does that). Even the lift ops (which override) must satisfy it.
        sm, ctx = PlannerStateMachine.create(graph=empty_graph, add_ui_listener=False)
        fake_st.session_state["state_machine"] = sm
        fake_st.session_state["context"] = ctx
        fake_st.session_state["graph"] = empty_graph
        fake_st.session_state["map_version"] = 0

        for mode, op in OPERATIONS.items():
            ctx.build_mode.mode = "__unset__"  # sentinel: prove on_select actually writes the mode
            op.on_select(ctx=ctx, sm=sm)
            assert ctx.build_mode.mode == mode, f"{mode}: on_select must highlight its own mode"
            assert sm.is_idle_ready, f"{mode}: on_select must NOT enter a build state (highlight only)"

    def test_lift_operation_on_select_retypes_via_select_lift_type_action(
        self, fake_st, empty_graph, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # _LiftOperation overrides on_select to route through actions.select_lift_type_action (which,
        # when viewing a lift, re-types it via Lift.update_type). Spy that the delegation fires once
        # with the button's own mode. mode_registry calls it as `actions.select_lift_type_action`.
        sm, ctx = PlannerStateMachine.create(graph=empty_graph, add_ui_listener=False)
        fake_st.session_state["state_machine"] = sm
        fake_st.session_state["context"] = ctx
        fake_st.session_state["graph"] = empty_graph
        fake_st.session_state["map_version"] = 0

        calls: list[str] = []
        monkeypatch.setattr(
            "skiresort_planner.ui.actions.select_lift_type_action", lambda lift_type: calls.append(lift_type)
        )

        OPERATIONS[BuildMode.GONDOLA].on_select(ctx=ctx, sm=sm)
        assert calls == [BuildMode.GONDOLA]
        assert ctx.build_mode.mode == BuildMode.GONDOLA, "base on_select still highlights the mode"


class TestEntityKindSpecBijection:
    def test_keys_match_entity_kind_members_exactly(self) -> None:
        assert set(ENTITY_KIND_SPECS) == set(EntityKind)


class TestGreyoutRule:
    """The bug the refactor fixed: while viewing an entity, the OTHER-kind builders grey out, but
    Import AND Merge must ALSO be enabled (they're whole-resort utilities usable from any idle
    state). Before the fix Import stayed clickable via copy-pasted logic that forgot nothing; now
    every button's enabled() is the same rule, so this is asserted uniformly.
    """

    def _sm(self, empty_graph: ResortGraph) -> PlannerStateMachine:
        sm, _ctx = PlannerStateMachine.create(graph=empty_graph, add_ui_listener=False)
        return sm

    def test_utilities_enabled_from_idle_ready(self, empty_graph) -> None:
        sm = self._sm(empty_graph)
        assert OPERATIONS[BuildMode.IMPORT].enabled(sm)
        assert OPERATIONS[BuildMode.MERGE].enabled(sm)

    def test_all_buttons_disabled_while_placing_import(self, empty_graph) -> None:
        sm = self._sm(empty_graph)
        sm.start_import(lon=0.0, lat=0.0)
        for op in OPERATIONS.values():
            assert not op.enabled(sm), f"{op.mode} must be disabled while placing an import box"

    def test_all_buttons_disabled_while_merging(self, empty_graph) -> None:
        sm = self._sm(empty_graph)
        sm.start_merge()
        for op in OPERATIONS.values():
            assert not op.enabled(sm), f"{op.mode} must be disabled while selecting nodes to merge"

    def test_import_and_merge_still_enabled_while_viewing_a_slope(self, empty_graph, path_points_blue) -> None:
        from skiresort_planner.model.proposed_path import ProposedPathSegment

        empty_graph.commit_paths(paths=[ProposedPathSegment(points=path_points_blue, target_difficulty="blue")])
        slope = empty_graph.finish_slope(segment_ids=list(empty_graph.segments.keys()))
        assert slope is not None
        sm = self._sm(empty_graph)
        sm.show_slope_info_panel(slope_id=slope.id)

        # Import + Merge stay enabled (the regression) …
        assert OPERATIONS[BuildMode.IMPORT].enabled(sm), "Import must stay enabled while viewing a slope"
        assert OPERATIONS[BuildMode.MERGE].enabled(sm), "Merge must stay enabled while viewing a slope"
        # … while the road + lift builders grey out (must close the slope panel first).
        assert not OPERATIONS[BuildMode.ROAD].enabled(sm)
        assert not OPERATIONS[BuildMode.CHAIRLIFT].enabled(sm)
        # Slope itself stays enabled (switch straight into building a new slope).
        assert OPERATIONS[BuildMode.SLOPE].enabled(sm)

    def test_greyout_while_viewing_a_lift(self, empty_graph) -> None:
        # view_lift only sets ctx.viewing.lift_id + enters idle_viewing_lift (no graph lookup), so an
        # id is enough to exercise the greyout rule. Each builder stays enabled on its OWN kind and
        # greys out on the two other kinds; Import + Merge are always enabled from any idle state.
        sm = self._sm(empty_graph)
        sm.view_lift(lift_id="L1")
        assert sm.is_idle_viewing_lift

        assert OPERATIONS[BuildMode.IMPORT].enabled(sm), "Import must stay enabled while viewing a lift"
        assert OPERATIONS[BuildMode.MERGE].enabled(sm), "Merge must stay enabled while viewing a lift"
        # Lift stays enabled (re-type the viewed lift); slope + road grey out.
        assert OPERATIONS[BuildMode.CHAIRLIFT].enabled(sm), "Lift builders stay enabled while viewing a lift"
        assert not OPERATIONS[BuildMode.SLOPE].enabled(sm)
        assert not OPERATIONS[BuildMode.ROAD].enabled(sm)

    def test_greyout_while_viewing_a_road(self, empty_graph) -> None:
        sm = self._sm(empty_graph)
        sm.view_road(road_id="R1")
        assert sm.is_idle_viewing_road

        assert OPERATIONS[BuildMode.IMPORT].enabled(sm), "Import must stay enabled while viewing a road"
        assert OPERATIONS[BuildMode.MERGE].enabled(sm), "Merge must stay enabled while viewing a road"
        # Road stays enabled; slope + lift grey out.
        assert OPERATIONS[BuildMode.ROAD].enabled(sm), "Road builder stays enabled while viewing a road"
        assert not OPERATIONS[BuildMode.SLOPE].enabled(sm)
        assert not OPERATIONS[BuildMode.CHAIRLIFT].enabled(sm)


class TestRegistryReturnsRealObjects:
    def test_click_handlers_are_callable(self, empty_graph) -> None:
        for bs in BUILD_STATES.values():
            handler = bs.click_handler()
            assert callable(handler)

    def test_control_panels_construct(self, empty_graph) -> None:
        sm, ctx = PlannerStateMachine.create(graph=empty_graph, add_ui_listener=False)
        for bs in BUILD_STATES.values():
            panel = bs.control_panel(
                sm=sm, ctx=ctx, graph=empty_graph, on_commit=lambda _i: None, on_cancel_connection=lambda: None
            )
            # Every panel must implement the three-part contract.
            assert hasattr(panel, "context_message")
            assert hasattr(panel, "action_message")
            assert hasattr(panel, "buttons")

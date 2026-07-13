"""Tests for the dispatch registries (ui/mode_registry.py).

Every dispatch axis is covered EXACTLY, so a new state/mode/kind can't silently forget a part.
The bijection asserts run at import (so a gap crashes on import), and these tests re-assert them.
"""

from skiresort_planner.model.resort_graph import ResortGraph
from skiresort_planner.ui.context import BuildMode, EntityKind
from skiresort_planner.ui.mode_registry import (
    BUILD_STATES,
    ENTITY_KIND_SPECS,
    OPERATIONS,
    OperationGroup,
)
from skiresort_planner.ui.state_machine import PlannerStateMachine


class TestBuildStateBijection:
    def test_keys_match_sm_state_ids_exactly(self) -> None:
        sm_ids = {s.id for s in PlannerStateMachine.states}
        assert set(BUILD_STATES) == sm_ids

    def test_every_build_state_key_matches_its_registry_key(self) -> None:
        for key, bs in BUILD_STATES.items():
            assert bs.state_key == key

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


class TestEntityKindSpecBijection:
    def test_keys_match_entity_kind_members_exactly(self) -> None:
        assert set(ENTITY_KIND_SPECS) == set(EntityKind)

    def test_spec_kind_matches_its_key(self) -> None:
        for kind, spec in ENTITY_KIND_SPECS.items():
            assert spec.kind == kind


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

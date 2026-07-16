"""Tests for the dispatch registries (ui/mode_registry.py).

Every dispatch axis is covered EXACTLY, so a new state/mode/kind can't silently forget a part.
The bijection asserts run at import (so a gap crashes on import), and these tests re-assert them.
"""

import pytest

from skiresort_planner.model.path_segment import SegmentKind
from skiresort_planner.model.resort_graph import ResortGraph
from skiresort_planner.ui.context import BuildMode, EntityKind, PlannerContext
from skiresort_planner.ui.mode_registry import (
    BUILD_STATES,
    ENTITY_KIND_SPECS,
    OPERATIONS,
    InfoBlock,
    OperationGroup,
)
from skiresort_planner.ui.state_machine import PlannerStateMachine


class TestBuildStateInfoBlock:
    """Every state must supply a non-empty info block (icon + label + bullets) and a bool
    blocks_build_buttons (abstract methods, so this can't be forgotten; this test also pins that
    info blocks render for all states).
    """

    def test_every_state_has_a_valid_info_block(self) -> None:
        ctx = PlannerContext()
        for key, bs in BUILD_STATES.items():
            block = bs.info_block(ctx)
            assert isinstance(block, InfoBlock), key
            assert block.icon and block.label, f"{key} info block has empty icon/label"
            assert block.bullets, f"{key} info block has no bullets"
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
    """EXIT_HOOKS (owned by state_lifecycle, used by the state machine's force/undo dispatch) lists
    only states whose exit does real cleanup; no-op states are simply absent (dispatch uses .get).
    The states with real cleanup MUST be present or a force_* would skip their teardown — e.g.
    undoing an OSM import from import_placing would leak the placed box (the class of bug this guards).
    """

    def test_exit_hook_keys_are_all_real_states(self) -> None:
        from skiresort_planner.ui.state_lifecycle import EXIT_HOOKS

        sm_ids = {s.id for s in PlannerStateMachine.states}
        assert set(EXIT_HOOKS) <= sm_ids, "no stray/non-existent state ids"

    def test_states_with_real_cleanup_are_registered(self) -> None:
        # These three have non-trivial exit teardown and MUST be dispatched on force/undo.
        from skiresort_planner.ui.state_lifecycle import EXIT_HOOKS

        for state_id in ("lift_placing", "import_placing", "merge_placing"):
            assert state_id in EXIT_HOOKS, f"{state_id} exit cleanup must run on force_*"


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
            fake_st.session_state["map_version"] = 0
            op.on_select(ctx=ctx, sm=sm)
            assert ctx.build_mode.mode == mode, f"{mode}: on_select must highlight its own mode"
            assert sm.is_idle_ready, f"{mode}: on_select must NOT enter a build state (highlight only)"
            # Pure pre-selection from idle → no map content change → no deck.gl remount.
            assert fake_st.session_state["map_version"] == 0, f"{mode}: highlight must not bump map_version"

    def test_lift_operation_on_select_retypes_via_select_lift_type_action(
        self, fake_st, empty_graph, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # _LiftOperation.on_select routes through actions.select_lift_type_action (which sets the
        # build mode AND, when viewing a lift, re-types it via Lift.update_type). Spy that the
        # delegation fires once with the button's own mode. mode_registry calls it as
        # `actions.select_lift_type_action`. (The spy replaces the real setter, so build_mode.mode
        # isn't set here — that is select_lift_type_action's own tested job.)
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
        # Not viewing a lift → pure highlight → no map remount (map_version unchanged).
        assert fake_st.session_state["map_version"] == 0, "lift-type pre-selection must not remount the map"


class TestEntityKindSpecBijection:
    def test_keys_match_entity_kind_members_exactly(self) -> None:
        assert set(ENTITY_KIND_SPECS) == set(EntityKind)


class TestGreyoutRule:
    """While viewing an entity, EVERY build button greys out except the viewed kind's own builder:
    the other-kind builders AND the Import / Node Merge utilities all disable (only their own view
    panel can be re-opened). Every button's enabled() goes through the same rule, so this is
    asserted uniformly across kinds.
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

    def test_import_and_merge_disabled_while_viewing_a_slope(self, empty_graph, path_points_blue) -> None:
        from skiresort_planner.model.proposed_path import ProposedPathSegment

        empty_graph.commit_paths(paths=[ProposedPathSegment(points=path_points_blue, target_difficulty="blue")])
        slope = empty_graph.finish_slope(segment_ids=list(empty_graph.segments.keys()))
        assert slope is not None
        sm = self._sm(empty_graph)
        sm.show_slope_info_panel(slope_id=slope.id)

        # Import + Merge now grey out while viewing (close the panel first) …
        assert not OPERATIONS[BuildMode.IMPORT].enabled(sm), "Import must grey out while viewing a slope"
        assert not OPERATIONS[BuildMode.MERGE].enabled(sm), "Merge must grey out while viewing a slope"
        # … alongside the road + lift builders.
        assert not OPERATIONS[BuildMode.ROAD].enabled(sm)
        assert not OPERATIONS[BuildMode.CHAIRLIFT].enabled(sm)
        # Slope itself stays enabled (switch straight into building a new slope).
        assert OPERATIONS[BuildMode.SLOPE].enabled(sm)

    def test_greyout_while_viewing_a_lift(self, empty_graph) -> None:
        # view_lift only sets ctx.viewing.lift_id + enters idle_viewing_lift (no graph lookup), so an
        # id is enough to exercise the greyout rule. Only the lift builders stay enabled (re-type the
        # viewed lift); slope, road, Import + Merge all grey out.
        sm = self._sm(empty_graph)
        sm.view_lift(lift_id="L1")
        assert sm.is_idle_viewing_lift

        assert not OPERATIONS[BuildMode.IMPORT].enabled(sm), "Import must grey out while viewing a lift"
        assert not OPERATIONS[BuildMode.MERGE].enabled(sm), "Merge must grey out while viewing a lift"
        # Lift stays enabled (re-type the viewed lift); slope + road grey out.
        assert OPERATIONS[BuildMode.CHAIRLIFT].enabled(sm), "Lift builders stay enabled while viewing a lift"
        assert not OPERATIONS[BuildMode.SLOPE].enabled(sm)
        assert not OPERATIONS[BuildMode.ROAD].enabled(sm)

    def test_greyout_while_viewing_a_road(self, empty_graph) -> None:
        sm = self._sm(empty_graph)
        sm.view_road(road_id="R1")
        assert sm.is_idle_viewing_road

        assert not OPERATIONS[BuildMode.IMPORT].enabled(sm), "Import must grey out while viewing a road"
        assert not OPERATIONS[BuildMode.MERGE].enabled(sm), "Merge must grey out while viewing a road"
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


class TestRendersCustomPath:
    """The fan-vs-freehand distinction that governs whether orange commit-endpoint dots draw.

    The rule is identical for every kind: only a custom-path state (routing to a clicked target,
    ``force_mode`` set) draws a single freehand route with no endpoints; the fan states show the
    endpoint dots. This pins the road-parity fix — roads used to hide endpoints in EVERY road
    state, so the fan showed no orange markers.
    """

    def test_fan_states_show_endpoints_custom_path_hides_them(self) -> None:
        ctx = PlannerContext()
        # force_mode is only ever True inside a custom-path state (derived from a set target_location).
        assert ctx.custom_connect.force_mode is False
        for key in ("slope_starting", "slope_building", "road_starting", "road_building"):
            assert BUILD_STATES[key].renders_custom_path(ctx) is False, f"{key} must show orange endpoints"

        ctx.custom_connect.target_location = (0.0, 0.0, 2000.0)
        for key in ("slope_custom_path", "road_custom_path"):
            assert BUILD_STATES[key].renders_custom_path(ctx) is True, f"{key} must hide endpoints (freehand)"

    def test_non_build_states_never_render_custom_path(self) -> None:
        ctx = PlannerContext()
        ctx.custom_connect.target_location = (0.0, 0.0, 2000.0)  # even with force_mode set, non-build states ignore it
        for key in ("idle_ready", "idle_viewing_slope", "idle_viewing_road", "lift_placing", "merge_placing"):
            assert BUILD_STATES[key].renders_custom_path(ctx) is False, key


class TestBuildStateMapSurface:
    """The build states own overlay layers and a bottom profile; exercise both branches (with and
    without an origin / committed segments) so the road-parity surface is covered like slopes.
    """

    def test_overlay_and_profile_empty_before_any_origin(self, empty_graph, path_factory) -> None:
        from skiresort_planner.ui.center_map import MapRenderer

        _sm, ctx = PlannerStateMachine.create(graph=empty_graph, add_ui_listener=False)
        renderer = MapRenderer(center_lon=0.0, center_lat=0.0, zoom=13, pitch=0, bearing=0)
        analyzer = path_factory.terrain_analyzer
        dem = path_factory.dem_service
        for key in ("slope_starting", "road_starting"):
            bs = BUILD_STATES[key]
            # No origin yet → no overlay dot, no in-build profile.
            assert (
                bs.overlay_layers(
                    ctx=ctx, graph=empty_graph, renderer=renderer, terrain_analyzer=analyzer, dem=dem, use_3d=False
                )
                == []
            )
            assert bs.bottom_profile(ctx=ctx, graph=empty_graph) is None

    @pytest.mark.parametrize("state_key", ["slope_building", "road_building"])
    def test_overlay_draws_orientation_and_custom_direction_arrows(self, empty_graph, path_factory, state_key) -> None:
        """Slope AND road build states draw the same overlays (they share _PathBuildingState).

        Parity guard: since the merge into one kind-parameterized build state, roads must draw the
        fall-line orientation arrows at the origin and the custom-connect direction arrow while
        routing — exactly like slopes. Running the identical assertions for both keys locks that in.
        """
        from skiresort_planner.model.node import Node
        from skiresort_planner.model.path_point import PathPoint
        from skiresort_planner.ui.center_map import MapRenderer

        _sm, ctx = PlannerStateMachine.create(graph=empty_graph, add_ui_listener=False)
        renderer = MapRenderer(center_lon=0.0, center_lat=0.0, zoom=13, pitch=0, bearing=0)
        analyzer = path_factory.terrain_analyzer
        dem = path_factory.dem_service
        bs = BUILD_STATES[state_key]

        # A selection on real (sloped) terrain → orientation arrows are drawn.
        elev = dem.get_elevation_or_raise(lon=0.0, lat=0.0)
        ctx.selection.set(lon=0.0, lat=0.0, elevation=elev)
        layers_with_selection = bs.overlay_layers(
            ctx=ctx, graph=empty_graph, renderer=renderer, terrain_analyzer=analyzer, dem=dem, use_3d=False
        )
        assert layers_with_selection, f"{state_key}: a selection on sloped terrain must draw orientation arrows"

        # Custom-connect routing (force_mode + start_node) adds the downhill direction arrow.
        empty_graph.nodes["N1"] = Node(id="N1", location=PathPoint(lon=0.0, lat=0.0, elevation=elev))
        ctx.custom_connect.target_location = (0.01, 0.0, elev)  # force_mode derives from this
        ctx.custom_connect.start_node = "N1"
        layers_with_arrow = bs.overlay_layers(
            ctx=ctx, graph=empty_graph, renderer=renderer, terrain_analyzer=analyzer, dem=dem, use_3d=False
        )
        assert len(layers_with_arrow) > len(layers_with_selection), (
            f"{state_key}: custom-connect adds a direction arrow"
        )

    @pytest.mark.parametrize(
        "state_key,kind,expected_key",
        [
            ("slope_building", SegmentKind.SLOPE, "combined_slope_profile"),
            ("road_building", SegmentKind.ROAD, "combined_road_profile"),
        ],
    )
    def test_build_profile_renders_once_segments_exist(
        self, empty_graph, path_points_blue, state_key, kind, expected_key
    ) -> None:
        from skiresort_planner.model.proposed_path import ProposedPathSegment

        _sm, ctx = PlannerStateMachine.create(graph=empty_graph, add_ui_listener=False)
        empty_graph.commit_paths(paths=[ProposedPathSegment(points=path_points_blue, target_difficulty="blue")])
        ctx.build(kind).segments = list(empty_graph.segments.keys())
        spec = BUILD_STATES[state_key].bottom_profile(ctx=ctx, graph=empty_graph)
        assert spec is not None and spec.key == expected_key


class TestBottomProfileRendersInRightColumn:
    """The elevation profile must render in the RIGHT column, not below the map.

    Map height stays constant across every lifecycle state (the profile no longer shifts it), so the
    pydeck component key never changes from a height shift — which would remount the deck.gl component
    and reset the camera. This pins that render_control_panel is the single site that draws the
    profile, and that viewport_map_height is state-independent (takes no reserved-space argument).
    """

    def test_viewport_map_height_is_state_independent(self) -> None:
        import inspect

        from skiresort_planner.ui.infra import viewport_map_height

        # No parameters → height cannot depend on state/profile (a reserved-space arg is what used to
        # shrink the map when a profile appeared, remounting the deck).
        assert list(inspect.signature(viewport_map_height).parameters) == []

    def test_render_control_panel_draws_the_profile(self) -> None:
        import inspect

        from skiresort_planner.ui import mode_registry

        src = inspect.getsource(mode_registry.render_control_panel)
        assert "bottom_profile" in src and "plotly_chart" in src, (
            "the profile must render inside render_control_panel (right column), not below the map"
        )

"""Integration test for complete slope building workflow.

Tests the full slope lifecycle with STRICT state machine contract validation.
Verifies the Four Pillars (see TEST_REFACTORING_DESIGN.md Section 0) at each step.
"""

import pytest
from statemachine.exceptions import TransitionNotAllowed

from skiresort_planner.constants import MapConfig
from skiresort_planner.model.path_segment import SegmentKind
from tests_workflow.conftest import SMAndCtx, WorkflowSetup


class TestSlopeBuildingWorkflow:
    """Tests for complete slope building workflow with state machine validation."""

    def test_complete_slope_workflow(self, workflow_setup: WorkflowSetup) -> None:
        """Build a slope through all states: start → commit → finish → view.

        This test verifies the STATE MACHINE CONTRACT at each step:
        - Pillar 1: enter_* is Single Point of Truth for state guarantees
        - Pillar 2: before_* hooks only store data
        - Pillar 3: Self-loops run exit+enter
        - Pillar 4: Guards control conditional transitions
        """
        sm, ctx, graph, factory, dem = workflow_setup

        # === Phase 1: Start Slope (IdleReady → SlopeStarting) ===
        assert sm.current_state_value == "idle_ready", "Should start in idle_ready"
        panel_at_idle = ctx.viewing.panel_visible
        assert panel_at_idle is False, "No panel in idle_ready"

        start_elev = dem.get_elevation_or_raise(lon=0.0, lat=0.0)
        sm.start_slope(lon=0.0, lat=0.0, elevation=start_elev, node_id=None)

        assert sm.current_state_value == "slope_starting", "After start_slope: slope_starting"
        assert ctx.build(SegmentKind.SLOPE).name is not None, "Building context should have slope name"

        # === Phase 2: Generate proposals and commit first (SlopeStarting → SlopeBuilding) ===
        proposals = list(factory.generate_fan(kind=SegmentKind.SLOPE, lon=0.0, lat=0.0, elevation=start_elev))
        assert len(proposals) > 0, "Should generate at least one proposal"

        # Commit the first proposal
        endpoint_ids = graph.commit_paths(paths=[proposals[0]])
        seg_id = list(graph.segments.keys())[0]

        sm.commit_path(segment_id=seg_id, endpoint_node_id=endpoint_ids[0])  # type: ignore[attr-defined]  # dynamic python-statemachine event

        assert sm.current_state_value == "slope_building", "After commit_path: slope_building"
        assert seg_id in ctx.build(SegmentKind.SLOPE).segments, "Segment should be in building context"

        # === Phase 3: Finish Slope (SlopeBuilding → IdleViewingSlope) ===
        slope = graph.finish_slope(segment_ids=ctx.build(SegmentKind.SLOPE).segments)
        assert slope is not None, "finish_slope should return Slope"

        sm.finish_slope(entity_id=slope.id)

        # VERIFY Pillar 1: enter_idle_viewing_slope guarantees panel visible
        assert sm.current_state_value == "idle_viewing_slope", "After finish: idle_viewing_slope"
        panel_at_view = ctx.viewing.panel_visible
        assert panel_at_view is True, "Panel should be visible in viewing state"
        assert ctx.viewing.slope_id == slope.id, "Viewing context should have slope ID"

        # === Phase 4: Close Panel (IdleViewingSlope → IdleReady) ===
        sm.close_panel()  # type: ignore[attr-defined]  # dynamic python-statemachine event

        assert sm.current_state_value == "idle_ready", "After close: idle_ready"
        panel_after_close = ctx.viewing.panel_visible
        assert panel_after_close is False, "Panel should be hidden"
        assert ctx.viewing.slope_id is None, "Viewing slope should be cleared"


class TestSelfLoopBehavior:
    """Tests for self-loop transitions (Pillar 3 of state machine contract)."""

    def test_switch_slope_refreshes_viewed_slope(self, workflow_setup: WorkflowSetup) -> None:
        """Self-loop switch_slope must trigger exit+enter to refresh state."""
        sm, ctx, graph, factory, dem = workflow_setup

        # Create first slope
        start_elev = dem.get_elevation_or_raise(lon=0.0, lat=0.0)
        sm.start_slope(lon=0.0, lat=0.0, elevation=start_elev, node_id=None)

        proposals = list(factory.generate_fan(kind=SegmentKind.SLOPE, lon=0.0, lat=0.0, elevation=start_elev))
        endpoint_ids = graph.commit_paths(paths=[proposals[0]])
        seg1_id = list(graph.segments.keys())[0]

        sm.commit_path(segment_id=seg1_id, endpoint_node_id=endpoint_ids[0])  # type: ignore[attr-defined]  # dynamic python-statemachine event
        slope1 = graph.finish_slope(segment_ids=ctx.build(SegmentKind.SLOPE).segments)
        assert slope1 is not None
        sm.finish_slope(entity_id=slope1.id)

        assert ctx.viewing.slope_id == slope1.id, "Viewing first slope"

        # Create second slope
        sm.start_slope(lon=0.001, lat=0.0, elevation=start_elev - 10, node_id=None)
        proposals2 = list(factory.generate_fan(kind=SegmentKind.SLOPE, lon=0.001, lat=0.0, elevation=start_elev - 10))
        endpoint_ids2 = graph.commit_paths(paths=[proposals2[0]])
        seg2_id = [s for s in graph.segments if s != seg1_id][0]

        sm.commit_path(segment_id=seg2_id, endpoint_node_id=endpoint_ids2[0])  # type: ignore[attr-defined]  # dynamic python-statemachine event
        slope2 = graph.finish_slope(segment_ids=ctx.build(SegmentKind.SLOPE).segments)
        assert slope2 is not None
        sm.finish_slope(entity_id=slope2.id)

        # Self-loop: switch to first slope
        sm.view_slope(slope_id=slope1.id)

        assert sm.current_state_value == "idle_viewing_slope", "Still in viewing state"
        assert ctx.viewing.slope_id == slope1.id, "Should view first slope after switch"
        assert ctx.viewing.panel_visible is True, "Panel should remain visible after switch"


class TestForceStateMethods:
    """Tests for force_idle() and force_building(SegmentKind.SLOPE) methods used by action-layer undo.

    These methods bypass the normal state machine transitions to reset state after graph undo
    operations. They are undo-only: force_* raises unless called inside `with sm.undo_running():`.
    """

    def test_force_idle_outside_undo_raises(self, workflow_setup: WorkflowSetup) -> None:
        # The bypass is undo-only. Calling force_* in normal flow (no undo_running scope) must raise,
        # so nobody uses it as a shortcut that skips guards/validation.
        sm, _ctx, _graph, _factory, _dem = workflow_setup
        sm.start_lift(node_id=None, location=None)
        with pytest.raises(RuntimeError, match="undo-only"):
            sm.force_idle()

    def test_force_idle_from_building_clears_context(self, workflow_setup: WorkflowSetup) -> None:
        """force_idle() from SlopeBuilding clears building and goes to IdleReady."""
        sm, ctx, graph, factory, dem = workflow_setup

        start_elev = dem.get_elevation_or_raise(lon=0.0, lat=0.0)
        sm.start_slope(lon=0.0, lat=0.0, elevation=start_elev, node_id=None)

        # Commit first segment to get into SlopeBuilding
        proposals = list(factory.generate_fan(kind=SegmentKind.SLOPE, lon=0.0, lat=0.0, elevation=start_elev))
        endpoint_ids = graph.commit_paths(paths=[proposals[0]])
        seg1_id = list(graph.segments.keys())[0]
        sm.commit_path(segment_id=seg1_id, endpoint_node_id=endpoint_ids[0])  # type: ignore[attr-defined]  # dynamic python-statemachine event

        assert sm.current_state_value == "slope_building"
        assert len(ctx.build(SegmentKind.SLOPE).segments) == 1

        # Force to idle (simulates undo removing all segments)
        with sm.undo_running():
            sm.force_idle()

        assert sm.current_state_value == "idle_ready"
        assert len(ctx.build(SegmentKind.SLOPE).segments) == 0, "Building context should be cleared"

    def test_force_building_from_custom_path(self, workflow_setup: WorkflowSetup) -> None:
        """force_building(SegmentKind.SLOPE) from SlopeCustomPath goes to SlopeBuilding."""
        sm, ctx, graph, factory, dem = workflow_setup

        start_elev = dem.get_elevation_or_raise(lon=0.0, lat=0.0)
        sm.start_slope(lon=0.0, lat=0.0, elevation=start_elev, node_id=None)

        # Commit a segment, then click a target to enter custom path.
        proposals = list(factory.generate_fan(kind=SegmentKind.SLOPE, lon=0.0, lat=0.0, elevation=start_elev))
        endpoint_ids = graph.commit_paths(paths=[proposals[0]])
        seg1_id = list(graph.segments.keys())[0]
        sm.commit_path(segment_id=seg1_id, endpoint_node_id=endpoint_ids[0])  # type: ignore[attr-defined]  # dynamic python-statemachine event

        sm.select_custom_target(  # type: ignore[attr-defined]  # dynamic python-statemachine event
            target_location=(
                0.0,
                -500 / MapConfig.METERS_PER_DEGREE_EQUATOR,
                dem.get_elevation_or_raise(lon=0.0, lat=-500 / MapConfig.METERS_PER_DEGREE_EQUATOR),
            )
        )

        assert sm.current_state_value == "slope_custom_path"

        # Force back to building (simulates undo while in custom path)
        with sm.undo_running():
            sm.force_building(SegmentKind.SLOPE)

        assert sm.current_state_value == "slope_building"
        assert ctx.custom_connect.force_mode is False, "Custom connect should be cleared"
        assert len(ctx.build(SegmentKind.SLOPE).segments) == 1, "Committed segment must survive force_building"

    def test_force_idle_from_lift_placing_clears_lift_context(self, workflow_setup: WorkflowSetup) -> None:
        """force_idle() from LiftPlacing calls exit_lift_placing which clears lift context."""
        sm, ctx, _graph, _factory, _dem = workflow_setup

        # Enter lift placing mode
        sm.start_lift(node_id=None, location=None)
        # Manually set some lift state to verify it gets cleared
        ctx.lift.first_node_id = "test_node"

        assert sm.current_state_value == "lift_placing"
        assert ctx.lift.first_node_id == "test_node"

        # Force to idle - exit_lift_placing should clear lift context
        with sm.undo_running():
            sm.force_idle()

        assert sm.current_state_value == "idle_ready"
        assert ctx.lift.first_node_id is None, "Lift context should be cleared by exit hook"


class TestCancelSlope:
    """Tests for cancel_slope event from different states."""

    def test_cancel_from_starting_state(self, workflow_setup: WorkflowSetup) -> None:
        """cancel_slope from SlopeStarting returns to IdleReady."""
        sm, ctx, graph, factory, dem = workflow_setup

        start_elev = dem.get_elevation_or_raise(lon=0.0, lat=0.0)
        sm.start_slope(lon=0.0, lat=0.0, elevation=start_elev, node_id=None)

        assert sm.current_state_value == "slope_starting"

        # Use cancel_slope EVENT
        sm.send("cancel_slope")

        assert sm.current_state_value == "idle_ready", "Should return to IdleReady"

    def test_cancel_from_building_state(self, workflow_setup: WorkflowSetup) -> None:
        """cancel_slope from SlopeBuilding discards work and returns to IdleReady."""
        sm, ctx, graph, factory, dem = workflow_setup

        start_elev = dem.get_elevation_or_raise(lon=0.0, lat=0.0)
        sm.start_slope(lon=0.0, lat=0.0, elevation=start_elev, node_id=None)

        proposals = list(factory.generate_fan(kind=SegmentKind.SLOPE, lon=0.0, lat=0.0, elevation=start_elev))
        endpoint_ids = graph.commit_paths(paths=[proposals[0]])
        seg_id = list(graph.segments.keys())[0]
        sm.commit_path(segment_id=seg_id, endpoint_node_id=endpoint_ids[0])  # type: ignore[attr-defined]  # dynamic python-statemachine event

        assert sm.current_state_value == "slope_building"

        # Use cancel_slope EVENT
        sm.send("cancel_slope")

        assert sm.current_state_value == "idle_ready", "Should return to IdleReady"


class TestCustomPathBranch:
    """Tests for select_custom_target / cancel_custom (has_no_segments guard both ways)."""

    def test_select_custom_target_from_starting_sets_context(self, workflow_setup: WorkflowSetup) -> None:
        """select_custom_target from SlopeStarting records the route and enters SlopeCustomPath."""
        sm, ctx, graph, factory, dem = workflow_setup

        start_elev = dem.get_elevation_or_raise(lon=0.0, lat=0.0)
        sm.start_slope(lon=0.0, lat=0.0, elevation=start_elev, node_id=None)
        assert sm.current_state_value == "slope_starting"

        target = (
            0.0,
            -500 / MapConfig.METERS_PER_DEGREE_EQUATOR,
            dem.get_elevation_or_raise(lon=0.0, lat=-500 / MapConfig.METERS_PER_DEGREE_EQUATOR),
        )
        sm.select_custom_target(target_location=target)  # type: ignore[attr-defined]  # dynamic python-statemachine event

        assert sm.current_state_value == "slope_custom_path"
        assert ctx.custom_connect.force_mode is True, "force_mode set by _before_target_from_starting"
        assert ctx.custom_connect.target_location == target, "target_location recorded verbatim"
        assert ctx.custom_connect.target_node is None, "terrain target has no node id"
        # A fresh terrain origin is NOT materialised as a node here — carried as a pending location,
        # minted only at commit, so no isolated node can be swept out from under a stored id.
        assert ctx.custom_connect.start_node is None, "fresh terrain origin carries no node id yet"
        assert ctx.build(SegmentKind.SLOPE).start_location is not None, "origin carried as a location"

    def test_cancel_custom_with_no_segments_returns_to_starting(self, workflow_setup: WorkflowSetup) -> None:
        """cancel_custom with 0 committed segments takes the has_no_segments guard to SlopeStarting."""
        sm, ctx, graph, factory, dem = workflow_setup

        start_elev = dem.get_elevation_or_raise(lon=0.0, lat=0.0)
        sm.start_slope(lon=0.0, lat=0.0, elevation=start_elev, node_id=None)

        sm.select_custom_target(  # type: ignore[attr-defined]  # dynamic python-statemachine event
            target_location=(
                0.0,
                -500 / MapConfig.METERS_PER_DEGREE_EQUATOR,
                dem.get_elevation_or_raise(lon=0.0, lat=-500 / MapConfig.METERS_PER_DEGREE_EQUATOR),
            )
        )
        assert sm.current_state_value == "slope_custom_path"
        assert len(ctx.build(SegmentKind.SLOPE).segments) == 0

        sm.cancel_custom()  # type: ignore[attr-defined]  # dynamic python-statemachine event

        assert sm.current_state_value == "slope_starting", "has_no_segments guard routes back to starting"
        assert ctx.custom_connect.force_mode is False, "custom connect cleared by before_cancel_custom"

    def test_cancel_custom_with_one_segment_returns_to_building(self, workflow_setup: WorkflowSetup) -> None:
        """cancel_custom with 1 committed segment takes the !has_no_segments arm to SlopeBuilding."""
        sm, ctx, graph, factory, dem = workflow_setup

        start_elev = dem.get_elevation_or_raise(lon=0.0, lat=0.0)
        sm.start_slope(lon=0.0, lat=0.0, elevation=start_elev, node_id=None)

        proposals = list(factory.generate_fan(kind=SegmentKind.SLOPE, lon=0.0, lat=0.0, elevation=start_elev))
        endpoint_ids = graph.commit_paths(paths=[proposals[0]])
        seg_id = list(graph.segments.keys())[0]
        sm.commit_path(segment_id=seg_id, endpoint_node_id=endpoint_ids[0])  # type: ignore[attr-defined]  # dynamic python-statemachine event
        assert sm.current_state_value == "slope_building"

        sm.select_custom_target(  # type: ignore[attr-defined]  # dynamic python-statemachine event
            target_location=(
                0.0,
                -1000 / MapConfig.METERS_PER_DEGREE_EQUATOR,
                dem.get_elevation_or_raise(lon=0.0, lat=-1000 / MapConfig.METERS_PER_DEGREE_EQUATOR),
            )
        )
        assert sm.current_state_value == "slope_custom_path"
        assert len(ctx.build(SegmentKind.SLOPE).segments) == 1

        sm.cancel_custom()  # type: ignore[attr-defined]  # dynamic python-statemachine event

        assert sm.current_state_value == "slope_building", "one segment routes back to building, not starting"
        assert len(ctx.build(SegmentKind.SLOPE).segments) == 1, "committed segment survives cancel_custom"


class TestInvalidTransitions:
    """Tests that invalid transitions are properly blocked."""

    def test_cannot_finish_from_starting_state(self, sm_and_ctx: SMAndCtx) -> None:
        """finish_slope is not allowed from SlopeStarting (need at least 1 segment)."""
        sm, ctx = sm_and_ctx

        sm.start_slope(lon=0.0, lat=0.0, elevation=2500.0, node_id=None)
        assert sm.current_state_value == "slope_starting"

        # Try to call finish_slope - should raise or be blocked

        with pytest.raises(TransitionNotAllowed):
            sm.finish_slope(entity_id="SL1")

    def test_cannot_view_slope_from_building_state(self, workflow_setup: WorkflowSetup) -> None:
        """view_slope is not allowed from SlopeBuilding (must finish/cancel first)."""
        sm, ctx, graph, factory, dem = workflow_setup

        start_elev = dem.get_elevation_or_raise(lon=0.0, lat=0.0)
        sm.start_slope(lon=0.0, lat=0.0, elevation=start_elev, node_id=None)

        proposals = list(factory.generate_fan(kind=SegmentKind.SLOPE, lon=0.0, lat=0.0, elevation=start_elev))
        endpoint_ids = graph.commit_paths(paths=[proposals[0]])
        seg_id = list(graph.segments.keys())[0]
        sm.commit_path(segment_id=seg_id, endpoint_node_id=endpoint_ids[0])  # type: ignore[attr-defined]  # dynamic python-statemachine event

        assert sm.current_state_value == "slope_building"

        with pytest.raises(TransitionNotAllowed):
            sm.view_slope(slope_id="SL1")

    def test_direct_variant_transition_calls_are_forbidden(self, sm_and_ctx: SMAndCtx) -> None:
        """Event-only transitions raise RuntimeError when called directly (bypass guard).

        Both commit_first_path and cancel_from_building are in _EVENT_ONLY_TRANSITIONS
        and are replaced with _forbidden_call in __init__; the commit_path event entry
        point remains callable.
        """
        sm, ctx = sm_and_ctx

        sm.start_slope(lon=0.0, lat=0.0, elevation=2500.0, node_id=None)
        assert sm.current_state_value == "slope_starting"

        # Direct call to the resolved transition is blocked...
        with pytest.raises(RuntimeError, match="forbidden"):
            sm.commit_first_path(segment_id="S1", endpoint_node_id="N1")
        with pytest.raises(RuntimeError, match="forbidden"):
            sm.cancel_from_building()

        # ...and the block did not change state.
        assert sm.current_state_value == "slope_starting", "Blocked call must not transition"

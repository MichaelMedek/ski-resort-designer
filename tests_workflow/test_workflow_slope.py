"""Integration test for complete slope building workflow.

Tests the full slope lifecycle with STRICT state machine contract validation.
Verifies the Four Pillars (see TEST_REFACTORING_DESIGN.md Section 0) at each step.
"""

import pytest
from statemachine.exceptions import TransitionNotAllowed

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
        assert ctx.viewing.panel_visible is False, "No panel in idle_ready"

        start_elev = dem.get_elevation_or_raise(lon=0.0, lat=0.0)
        sm.start_slope(lon=0.0, lat=0.0, elevation=start_elev, node_id=None)

        assert sm.current_state_value == "slope_starting", "After start_slope: slope_starting"
        assert ctx.building.name is not None, "Building context should have slope name"

        # === Phase 2: Generate proposals and commit first (SlopeStarting → SlopeBuilding) ===
        proposals = list(factory.generate_fan(lon=0.0, lat=0.0, elevation=start_elev))
        assert len(proposals) > 0, "Should generate at least one proposal"

        # Commit the first proposal
        endpoint_ids = graph.commit_paths(paths=[proposals[0]])
        seg_id = list(graph.segments.keys())[0]

        sm.commit_path(segment_id=seg_id, endpoint_node_id=endpoint_ids[0])

        assert sm.current_state_value == "slope_building", "After commit_path: slope_building"
        assert seg_id in ctx.building.segments, "Segment should be in building context"

        # === Phase 3: Finish Slope (SlopeBuilding → IdleViewingSlope) ===
        slope = graph.finish_slope(segment_ids=ctx.building.segments)
        assert slope is not None, "finish_slope should return Slope"

        sm.finish_slope(slope_id=slope.id)

        # VERIFY Pillar 1: enter_idle_viewing_slope guarantees panel visible
        assert sm.current_state_value == "idle_viewing_slope", "After finish: idle_viewing_slope"
        assert ctx.viewing.panel_visible is True, "Panel should be visible in viewing state"
        assert ctx.viewing.slope_id == slope.id, "Viewing context should have slope ID"

        # === Phase 4: Close Panel (IdleViewingSlope → IdleReady) ===
        sm.send("close_panel")

        assert sm.current_state_value == "idle_ready", "After close: idle_ready"
        assert ctx.viewing.panel_visible is False, "Panel should be hidden"
        assert ctx.viewing.slope_id is None, "Viewing slope should be cleared"


class TestSelfLoopBehavior:
    """Tests for self-loop transitions (Pillar 3 of state machine contract)."""

    def test_switch_slope_refreshes_viewed_slope(self, workflow_setup: WorkflowSetup) -> None:
        """Self-loop switch_slope must trigger exit+enter to refresh state."""
        sm, ctx, graph, factory, dem = workflow_setup

        # Create first slope
        start_elev = dem.get_elevation_or_raise(lon=0.0, lat=0.0)
        sm.start_slope(lon=0.0, lat=0.0, elevation=start_elev, node_id=None)

        proposals = list(factory.generate_fan(lon=0.0, lat=0.0, elevation=start_elev))
        endpoint_ids = graph.commit_paths(paths=[proposals[0]])
        seg1_id = list(graph.segments.keys())[0]

        sm.commit_path(segment_id=seg1_id, endpoint_node_id=endpoint_ids[0])
        slope1 = graph.finish_slope(segment_ids=ctx.building.segments)
        sm.finish_slope(slope_id=slope1.id)

        assert ctx.viewing.slope_id == slope1.id, "Viewing first slope"

        # Create second slope
        sm.start_slope(lon=0.001, lat=0.0, elevation=start_elev - 10, node_id=None)
        proposals2 = list(factory.generate_fan(lon=0.001, lat=0.0, elevation=start_elev - 10))
        endpoint_ids2 = graph.commit_paths(paths=[proposals2[0]])
        seg2_id = [s for s in graph.segments.keys() if s != seg1_id][0]

        sm.commit_path(segment_id=seg2_id, endpoint_node_id=endpoint_ids2[0])
        slope2 = graph.finish_slope(segment_ids=ctx.building.segments)
        sm.finish_slope(slope_id=slope2.id)

        # Self-loop: switch to first slope
        sm.send("view_slope", slope_id=slope1.id)

        assert sm.current_state_value == "idle_viewing_slope", "Still in viewing state"
        assert ctx.viewing.slope_id == slope1.id, "Should view first slope after switch"
        assert ctx.viewing.panel_visible is True, "Panel should remain visible after switch"


class TestForceStateMethods:
    """Tests for force_idle() and force_building() methods used by action-layer undo.

    These methods bypass the normal state machine transitions to reset state
    after graph undo operations. They follow the 'Safe Dynamic Exit' pattern
    which calls the exit hook for the current state before forcing the new state.
    """

    def test_force_idle_from_building_clears_context(self, workflow_setup: WorkflowSetup) -> None:
        """force_idle() from SlopeBuilding clears building and goes to IdleReady."""
        sm, ctx, graph, factory, dem = workflow_setup

        start_elev = dem.get_elevation_or_raise(lon=0.0, lat=0.0)
        sm.start_slope(lon=0.0, lat=0.0, elevation=start_elev, node_id=None)

        # Commit first segment to get into SlopeBuilding
        proposals = list(factory.generate_fan(lon=0.0, lat=0.0, elevation=start_elev))
        endpoint_ids = graph.commit_paths(paths=[proposals[0]])
        seg1_id = list(graph.segments.keys())[0]
        sm.commit_path(segment_id=seg1_id, endpoint_node_id=endpoint_ids[0])

        assert sm.current_state_value == "slope_building"
        assert len(ctx.building.segments) == 1

        # Force to idle (simulates undo removing all segments)
        sm.force_idle()

        assert sm.current_state_value == "idle_ready"
        assert len(ctx.building.segments) == 0, "Building context should be cleared"

    def test_force_building_from_custom_picking(self, workflow_setup: WorkflowSetup) -> None:
        """force_building() from SlopeCustomPicking goes to SlopeBuilding."""
        sm, ctx, graph, factory, dem = workflow_setup

        start_elev = dem.get_elevation_or_raise(lon=0.0, lat=0.0)
        sm.start_slope(lon=0.0, lat=0.0, elevation=start_elev, node_id=None)

        # Commit segment and enable custom mode
        proposals = list(factory.generate_fan(lon=0.0, lat=0.0, elevation=start_elev))
        endpoint_ids = graph.commit_paths(paths=[proposals[0]])
        seg1_id = list(graph.segments.keys())[0]
        sm.commit_path(segment_id=seg1_id, endpoint_node_id=endpoint_ids[0])
        sm.enable_custom()

        assert sm.current_state_value == "slope_custom_picking"

        # Force back to building (simulates undo while in custom picking)
        sm.force_building()

        assert sm.current_state_value == "slope_building"
        assert ctx.custom_connect.enabled is False, "Custom connect should be cleared"

    def test_force_idle_from_lift_placing_clears_lift_context(self, workflow_setup: WorkflowSetup) -> None:
        """force_idle() from LiftPlacing calls exit_lift_placing which clears lift context."""
        sm, ctx, _graph, _factory, _dem = workflow_setup

        # Enter lift placing mode
        sm.send("start_lift", node_id=None, location=None)
        # Manually set some lift state to verify it gets cleared
        ctx.lift.start_node_id = "test_node"

        assert sm.current_state_value == "lift_placing"
        assert ctx.lift.start_node_id == "test_node"

        # Force to idle - exit_lift_placing should clear lift context
        sm.force_idle()

        assert sm.current_state_value == "idle_ready"
        assert ctx.lift.start_node_id is None, "Lift context should be cleared by exit hook"

    def test_force_idle_succeeds_even_if_exit_hook_fails(
        self, workflow_setup: WorkflowSetup, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """force_idle() completes even when exit hook raises exception (try-finally guarantee)."""
        from skiresort_planner.ui import state_machine

        sm, ctx, _graph, _factory, _dem = workflow_setup

        # Enter lift placing mode
        sm.send("start_lift", node_id=None, location=None)
        assert sm.current_state_value == "lift_placing"

        # Patch exit_lift_placing to raise an exception
        def failing_exit_hook(ctx: "PlannerContext") -> None:
            raise RuntimeError("Simulated exit hook failure")

        original_hooks = state_machine.PlannerStateMachine._EXIT_HOOKS
        patched_hooks = dict(original_hooks)
        patched_hooks["lift_placing"] = failing_exit_hook
        monkeypatch.setattr(state_machine.PlannerStateMachine, "_EXIT_HOOKS", patched_hooks)

        # Force to idle - should succeed despite exit hook failure
        sm.force_idle()  # Should NOT raise

        # State change MUST have happened (finally block guarantee)
        assert sm.current_state_value == "idle_ready", "State change must happen even if exit hook fails"


class TestCancelSlope:
    """Tests for cancel_slope event from different states."""

    def test_cancel_from_starting_state(self, workflow_setup: WorkflowSetup) -> None:
        """cancel_slope from SlopeStarting returns to IdleReady."""
        sm, ctx, graph, factory, dem = workflow_setup

        start_elev = dem.get_elevation_or_raise(lon=0.0, lat=0.0)
        sm.start_slope(lon=0.0, lat=0.0, elevation=start_elev, node_id=None)

        assert sm.current_state_value == "slope_starting"

        # Use cancel_slope EVENT
        sm.cancel_slope()

        assert sm.current_state_value == "idle_ready", "Should return to IdleReady"

    def test_cancel_from_building_state(self, workflow_setup: WorkflowSetup) -> None:
        """cancel_slope from SlopeBuilding discards work and returns to IdleReady."""
        sm, ctx, graph, factory, dem = workflow_setup

        start_elev = dem.get_elevation_or_raise(lon=0.0, lat=0.0)
        sm.start_slope(lon=0.0, lat=0.0, elevation=start_elev, node_id=None)

        proposals = list(factory.generate_fan(lon=0.0, lat=0.0, elevation=start_elev))
        endpoint_ids = graph.commit_paths(paths=[proposals[0]])
        seg_id = list(graph.segments.keys())[0]
        sm.commit_path(segment_id=seg_id, endpoint_node_id=endpoint_ids[0])

        assert sm.current_state_value == "slope_building"

        # Use cancel_slope EVENT
        sm.cancel_slope()

        assert sm.current_state_value == "idle_ready", "Should return to IdleReady"


class TestInvalidTransitions:
    """Tests that invalid transitions are properly blocked."""

    def test_cannot_finish_from_starting_state(self, sm_and_ctx: SMAndCtx) -> None:
        """finish_slope is not allowed from SlopeStarting (need at least 1 segment)."""
        sm, ctx = sm_and_ctx

        sm.start_slope(lon=0.0, lat=0.0, elevation=2500.0, node_id=None)
        assert sm.current_state_value == "slope_starting"

        # Try to call finish_slope - should raise or be blocked

        with pytest.raises(TransitionNotAllowed):
            sm.finish_slope(slope_id="SL1")

    def test_cannot_view_slope_from_building_state(self, workflow_setup: WorkflowSetup) -> None:
        """view_slope is not allowed from SlopeBuilding (must finish/cancel first)."""
        sm, ctx, graph, factory, dem = workflow_setup

        start_elev = dem.get_elevation_or_raise(lon=0.0, lat=0.0)
        sm.start_slope(lon=0.0, lat=0.0, elevation=start_elev, node_id=None)

        proposals = list(factory.generate_fan(lon=0.0, lat=0.0, elevation=start_elev))
        endpoint_ids = graph.commit_paths(paths=[proposals[0]])
        seg_id = list(graph.segments.keys())[0]
        sm.commit_path(segment_id=seg_id, endpoint_node_id=endpoint_ids[0])

        assert sm.current_state_value == "slope_building"

        with pytest.raises(TransitionNotAllowed):
            sm.send("view_slope", slope_id="SL1")

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


class TestGuardedTransitions:
    """Tests for guard-based flow control (Pillar 4)."""

    def test_undo_uses_guards_for_destination(self, workflow_setup: WorkflowSetup) -> None:
        """Guard undo_leaves_no_segments determines undo destination.

        Tests:
        - With 2+ segments: undo stays in SlopeBuilding
        - With 1 segment: undo goes to IdleReady
        """
        sm, ctx, graph, factory, dem = workflow_setup

        start_elev = dem.get_elevation_or_raise(lon=0.0, lat=0.0)
        sm.start_slope(lon=0.0, lat=0.0, elevation=start_elev, node_id=None)

        # Commit first segment
        proposals = list(factory.generate_fan(lon=0.0, lat=0.0, elevation=start_elev))
        endpoint_ids = graph.commit_paths(paths=[proposals[0]])
        seg1_id = list(graph.segments.keys())[0]
        sm.commit_path(segment_id=seg1_id, endpoint_node_id=endpoint_ids[0])

        # Get endpoint for second segment
        end_node = graph.nodes[endpoint_ids[0]]

        # Commit second segment from endpoint
        proposals2 = list(factory.generate_fan(lon=end_node.lon, lat=end_node.lat, elevation=end_node.elevation))
        if len(proposals2) > 0:
            endpoint_ids2 = graph.commit_paths(paths=[proposals2[0]])
            seg2_id = [s for s in graph.segments.keys() if s != seg1_id][0]
            sm.commit_path(segment_id=seg2_id, endpoint_node_id=endpoint_ids2[0])

            assert len(ctx.building.segments) == 2, "Should have 2 segments"

            # Undo with 2 segments: guard keeps us in SlopeBuilding
            # Note: Use undo EVENT, not direct transition
            sm.undo(removed_segment_id=seg2_id, new_endpoint_node_id=endpoint_ids[0])

            assert sm.current_state_value == "slope_building", "Guard kept us in building"
            assert len(ctx.building.segments) == 1, "Should have 1 segment after undo"

        # Undo with 1 segment: guard sends to IdleReady
        sm.undo(removed_segment_id=ctx.building.segments[0])

        assert sm.current_state_value == "idle_ready", "Guard sent to idle when no segments"


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

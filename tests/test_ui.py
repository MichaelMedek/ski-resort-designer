"""Tests for skiresort_planner UI state machine and full workflow simulation.

Tests: PlannerStateMachine, PlannerContext, Full Workflow Integration
Focus: State transitions, guards, context updates, end-to-end planning flow

Note: Fixtures are defined in conftest.py (MockDEMService, state machine, workflow setup).
"""

import pytest
from statemachine.exceptions import TransitionNotAllowed

from skiresort_planner.generators.path_factory import PathFactory
from skiresort_planner.model.path_point import PathPoint
from skiresort_planner.model.resort_graph import ResortGraph
from skiresort_planner.ui.state_machine import (
    BuildMode,
    PlannerContext,
    PlannerStateMachine,
)


class TestPlannerContext:
    """PlannerContext - shared state for the planner."""

    def test_context_defaults(self) -> None:
        """Context initializes with reasonable defaults."""
        ctx = PlannerContext()
        assert ctx.building.name is None
        assert ctx.building.segments == []
        assert ctx.proposals.paths == []
        assert ctx.lift.type == "chairlift"

    def test_context_clear_methods(self) -> None:
        """Context clear methods reset state."""
        ctx = PlannerContext()
        ctx.selection.lon = 10.27
        ctx.selection.lat = 46.97
        ctx.proposals.paths = [1, 2, 3]  # Dummy data
        ctx.building.segments = ["S1", "S2"]

        ctx.clear_selection()
        assert ctx.selection.lon is None

        ctx.clear_proposals()
        assert ctx.proposals.paths == []

        ctx.clear_building()
        assert ctx.building.segments == []

    def test_has_selection(self) -> None:
        """has_selection checks for lon/lat presence."""
        ctx = PlannerContext()
        assert ctx.has_selection() is False

        ctx.selection.lon = 10.27
        ctx.selection.lat = 46.97
        assert ctx.has_selection() is True

    def test_has_committed_segments(self) -> None:
        """has_committed_segments checks segment list."""
        ctx = PlannerContext()
        assert ctx.has_committed_segments() is False

        ctx.building.segments = ["S1"]
        assert ctx.has_committed_segments() is True


class TestStateMachineInitialization:
    """State machine creation and initial state."""

    def test_create_returns_tuple(self) -> None:
        """create() returns (machine, context) tuple."""
        result = PlannerStateMachine.create()
        assert isinstance(result, tuple)
        assert len(result) == 2
        sm, ctx = result
        assert isinstance(sm, PlannerStateMachine)
        assert isinstance(ctx, PlannerContext)

    def test_initial_state_is_idle(self, state_machine_and_context: tuple) -> None:
        """State machine starts in IDLE state with is_idle=True."""
        sm, ctx = state_machine_and_context
        assert sm.current_state == sm.idle
        assert sm.get_state_name() == "Idle"
        assert sm.is_idle is True
        assert sm.is_slope_building is False
        assert ctx.viewing.panel_visible is False
        # Default build mode is slope
        ctx.build_mode.mode = BuildMode.SLOPE
        assert sm.is_slope_mode() is True
        assert sm.is_lift_mode() is False


class TestSlopeModeTransitions:
    """Test slope design mode transitions."""

    def test_idle_to_building(self, state_machine_and_context: tuple) -> None:
        """IDLE -> SLOPE_BUILDING via start_building, is_slope_building=True."""
        sm, ctx = state_machine_and_context
        sm.start_building(lon=10.27, lat=46.97, elevation=2500.0, node_id=None, slope_number=1)
        assert sm.current_state == sm.slope_building
        assert ctx.building.name == "Slope 1"
        assert sm.is_idle is False
        assert sm.is_slope_building is True
        assert ctx.viewing.panel_visible is False

    def test_building_commit_stays_in_building(self, state_machine_and_context: tuple) -> None:
        """SLOPE_BUILDING -> SLOPE_BUILDING via commit_path."""
        sm, ctx = state_machine_and_context
        sm.start_building(lon=10.27, lat=46.97, elevation=2500.0, node_id=None, slope_number=1)
        sm.commit_path(segment_id="S1", endpoint_node_id="N2")

        assert sm.current_state == sm.slope_building
        assert "S1" in ctx.building.segments

    def test_building_to_idle_via_finish(self, state_machine_and_context: tuple) -> None:
        """SLOPE_BUILDING -> IDLE via finish_slope (panel shows automatically)."""
        sm, ctx = state_machine_and_context
        sm.start_building(lon=10.27, lat=46.97, elevation=2500.0, node_id=None, slope_number=1)
        sm.commit_path(segment_id="S1", endpoint_node_id="N2")
        sm.finish_slope(slope_id="Slope1")

        assert sm.current_state == sm.idle
        assert ctx.viewing.panel_visible is True  # Panel shown automatically after finish
        assert ctx.viewing.slope_id == "Slope1"  # viewing the just-finished slope
        # Building context cleared on transition
        assert ctx.building.segments == []

    def test_building_to_idle_via_cancel(self, state_machine_and_context: tuple) -> None:
        """SLOPE_BUILDING -> IDLE via cancel_slope."""
        sm, ctx = state_machine_and_context
        sm.start_building(lon=10.27, lat=46.97, elevation=2500.0, node_id=None, slope_number=1)
        sm.commit_path(segment_id="S1", endpoint_node_id="N2")
        sm.cancel_slope()

        assert sm.current_state == sm.idle

    def test_show_slope_info_panel_in_idle(self, state_machine_and_context: tuple) -> None:
        """Show slope panel in IDLE (no state change - panel is orthogonal)."""
        sm, ctx = state_machine_and_context
        sm.show_slope_info_panel(slope_id="ExistingSlope")

        assert sm.current_state == sm.idle  # State unchanged
        assert ctx.viewing.slope_id == "ExistingSlope"
        assert ctx.viewing.panel_visible is True
        assert sm.is_idle is True
        assert sm.is_slope_building is False

    def test_hide_slope_info_panel_in_idle(self, state_machine_and_context: tuple) -> None:
        """Hide slope panel in IDLE (no state change - panel is orthogonal)."""
        sm, ctx = state_machine_and_context
        sm.show_slope_info_panel(slope_id="ExistingSlope")
        sm.hide_info_panel()

        assert sm.current_state == sm.idle  # State unchanged
        # NOTE: hide() keeps slope_id for potential re-show, only clears panel_visible
        assert ctx.viewing.panel_visible is False


class TestLiftModeTransitions:
    """Test lift design mode transitions.

    Build mode (slope vs lift) is now stored in ctx.build_mode.mode.
    The state machine has a unified 'idle' state instead of separate slope_idle/lift_idle.
    """

    def test_lift_mode_via_build_mode(self, state_machine_and_context: tuple) -> None:
        """Setting build_mode to lift type enables lift mode behavior."""
        sm, ctx = state_machine_and_context
        ctx.build_mode.mode = BuildMode.CHAIRLIFT

        assert sm.current_state == sm.idle
        assert sm.is_lift_mode() is True
        assert sm.is_slope_mode() is False
        assert sm.is_idle is True
        assert sm.is_lift_placing is False
        assert ctx.viewing.panel_visible is False

    def test_idle_to_placing(self, state_machine_and_context: tuple) -> None:
        """IDLE -> LIFT_PLACING via select_lift_start (with lift build mode)."""
        sm, ctx = state_machine_and_context
        ctx.build_mode.mode = BuildMode.CHAIRLIFT
        sm.select_lift_start(node_id="N1")

        assert sm.current_state == sm.lift_placing
        assert ctx.lift.start_node_id == "N1"
        assert sm.is_idle is False
        assert sm.is_lift_placing is True
        assert ctx.viewing.panel_visible is False

    def test_lift_placing_to_idle_via_complete(self, state_machine_and_context: tuple) -> None:
        """LIFT_PLACING -> IDLE via complete_lift (panel shows automatically)."""
        sm, ctx = state_machine_and_context
        ctx.build_mode.mode = BuildMode.CHAIRLIFT
        sm.select_lift_start(node_id="N1")
        sm.complete_lift(lift_id="L1")

        # After completing lift, go to idle with panel showing
        assert sm.current_state == sm.idle
        assert ctx.viewing.panel_visible is True  # Panel shown automatically after complete
        assert ctx.viewing.lift_id == "L1"

    def test_show_lift_info_panel_in_idle(self, state_machine_and_context: tuple) -> None:
        """Show lift panel in IDLE (no state change - panel is orthogonal)."""
        sm, ctx = state_machine_and_context
        ctx.build_mode.mode = BuildMode.CHAIRLIFT
        sm.show_lift_info_panel(lift_id="ExistingLift")

        assert sm.current_state == sm.idle  # State unchanged
        assert ctx.viewing.lift_id == "ExistingLift"
        assert ctx.viewing.panel_visible is True
        assert sm.is_idle is True
        assert sm.is_lift_placing is False

    def test_switch_build_mode_slope_to_lift(self, state_machine_and_context: tuple) -> None:
        """Build mode can be switched from slope to lift via context."""
        sm, ctx = state_machine_and_context
        ctx.build_mode.mode = BuildMode.SLOPE
        assert sm.is_slope_mode() is True

        ctx.build_mode.mode = BuildMode.CHAIRLIFT
        assert sm.is_lift_mode() is True
        assert sm.current_state == sm.idle  # State unchanged


class TestUndoTransitions:
    """Test undo behavior with guards."""

    def test_undo_with_multiple_segments_stays_building(self, state_machine_and_context: tuple) -> None:
        """Undo with >1 segment stays in SLOPE_BUILDING."""
        sm, ctx = state_machine_and_context
        sm.start_building(lon=10.27, lat=46.97, elevation=2500.0, node_id=None, slope_number=1)
        sm.commit_path(segment_id="S1", endpoint_node_id="N2")
        sm.commit_path(segment_id="S2", endpoint_node_id="N3")

        # Now have 2 segments
        assert len(ctx.building.segments) == 2

        # Undo removes one and stays in SLOPE_BUILDING
        sm.undo_segment(removed_segment_id="S2")
        assert sm.current_state == sm.slope_building
        assert len(ctx.building.segments) == 1

    def test_undo_with_one_segment_goes_idle(self, state_machine_and_context: tuple) -> None:
        """Undo with 1 segment goes to IDLE."""
        sm, ctx = state_machine_and_context
        sm.start_building(lon=10.27, lat=46.97, elevation=2500.0, node_id=None, slope_number=1)
        sm.commit_path(segment_id="S1", endpoint_node_id="N2")

        # Only 1 segment
        assert len(ctx.building.segments) == 1

        # Undo goes to IDLE
        sm.undo_segment(removed_segment_id="S1")
        assert sm.current_state == sm.idle


class TestInvalidTransitions:
    """Test that invalid transitions are blocked."""

    def test_cannot_finish_from_idle(self, state_machine_and_context: tuple) -> None:
        """Cannot call finish_slope from IDLE."""
        sm, ctx = state_machine_and_context
        with pytest.raises(TransitionNotAllowed):
            sm.finish_slope(slope_id="X")

    def test_cannot_commit_from_idle(self, state_machine_and_context: tuple) -> None:
        """Cannot call commit_path from IDLE."""
        sm, ctx = state_machine_and_context
        with pytest.raises(TransitionNotAllowed):
            sm.commit_path(segment_id="S1", endpoint_node_id="N1")

    def test_cannot_complete_lift_from_idle(self, state_machine_and_context: tuple) -> None:
        """Cannot call complete_lift from IDLE (must be in LIFT_PLACING)."""
        sm, ctx = state_machine_and_context
        ctx.build_mode.mode = BuildMode.CHAIRLIFT
        with pytest.raises(TransitionNotAllowed):
            sm.complete_lift(lift_id="L1")

    def test_can_change_build_mode_in_idle_with_panel(self, state_machine_and_context: tuple) -> None:
        """CAN change build mode while in idle even with panel showing.

        Build mode is now stored in context, not as separate states.
        Panel visibility is orthogonal to build mode.
        """
        sm, ctx = state_machine_and_context
        ctx.build_mode.mode = BuildMode.SLOPE
        sm.show_slope_info_panel(slope_id="ExistingSlope")
        assert ctx.viewing.panel_visible is True
        assert sm.is_slope_mode() is True

        # Changing build mode doesn't affect state or panel
        ctx.build_mode.mode = BuildMode.CHAIRLIFT
        assert sm.is_idle is True
        assert sm.is_lift_mode() is True
        # Panel is NOT auto-hidden (caller must handle panel)
        assert ctx.viewing.panel_visible is True


class TestTryTransition:
    """Test try_transition safe transition method."""

    def test_try_transition_success(self, state_machine_and_context: tuple) -> None:
        """try_transition returns True on valid transition."""
        sm, ctx = state_machine_and_context
        # Use a valid transition: start_building from idle
        result = sm.try_transition(
            "start_building", lon=10.27, lat=46.97, elevation=2500.0, node_id=None, slope_number=1
        )
        assert result is True
        assert sm.get_state_name() == "SlopeBuilding"  # State changed

    def test_try_transition_failure(self, state_machine_and_context: tuple) -> None:
        """try_transition returns False on invalid transition (no exception)."""
        sm, ctx = state_machine_and_context
        sm.start_building(lon=10.27, lat=46.97, elevation=2500.0, node_id=None, slope_number=1)

        # Cannot select_lift_start from slope building state
        result = sm.try_transition("select_lift_start", node_id="N1")
        assert result is False
        assert sm.get_state_name() == "SlopeBuilding"  # State unchanged


class TestStateMachineHelpers:
    """Test helper methods on state machine."""

    def test_get_state_name(self, state_machine_and_context: tuple) -> None:
        """get_state_name returns current state name."""
        sm, ctx = state_machine_and_context
        assert sm.get_state_name() == "Idle"

        sm.start_building(lon=10.27, lat=46.97, elevation=2500.0, node_id=None, slope_number=1)
        assert sm.get_state_name() == "SlopeBuilding"

        # Return to Idle via cancel
        sm.cancel_slope()
        assert sm.get_state_name() == "Idle"

        # Enter lift placing state
        ctx.build_mode.mode = BuildMode.CHAIRLIFT
        sm.select_lift_start(node_id="N1")
        assert sm.get_state_name() == "LiftPlacing"


# =============================================================================
# FULL WORKFLOW INTEGRATION TESTS
# Simulates complete ski resort planning flow with mock DEM.
# Fixtures (MockDEMService, workflow_complete_setup) are in conftest.py.
# =============================================================================


class TestFullWorkflow:
    """Full ski resort planning workflow simulation.

    Tests the complete flow:
    1. Start slope at summit
    2. Generate proposals
    3. Select and commit segments
    4. Finish slope
    5. Branch from intermediate node
    6. Place lift
    """

    def test_generate_proposals_at_start_point(self, workflow_complete_setup: tuple) -> None:
        """Start at summit, generate path proposals."""
        sm, ctx, graph, factory, dem = workflow_complete_setup

        # Start at summit (equator coordinates)
        start_lon, start_lat = 0.0, 0.0
        start_elev = dem.get_elevation(lon=start_lon, lat=start_lat)
        assert start_elev == 2500.0

        # Generate fan of proposals
        proposals = list(
            factory.generate_fan(lon=start_lon, lat=start_lat, elevation=start_elev, target_length_m=500.0)
        )

        # Should generate multiple paths (green, blue, red, black variants)
        assert len(proposals) >= 4
        difficulties = {p.target_difficulty for p in proposals}
        assert "green" in difficulties
        assert "blue" in difficulties

        # All paths should go downhill
        for p in proposals:
            assert p.total_drop_m > 0
            assert p.end.elevation < p.start.elevation

    def test_commit_path_creates_segment_and_nodes(self, workflow_complete_setup: tuple) -> None:
        """Committing a path creates segment and nodes in graph."""
        sm, ctx, graph, factory, dem = workflow_complete_setup

        # Generate proposals
        proposals = list(factory.generate_fan(lon=0.0, lat=0.0, elevation=2500.0, target_length_m=500.0))
        assert len(proposals) > 0

        # Commit first proposal
        path = proposals[0]
        end_node_ids = graph.commit_paths(paths=[path])

        # Should create nodes and segment
        assert len(graph.nodes) == 2  # Start and end
        assert len(graph.segments) == 1
        assert len(end_node_ids) == 1

        seg = list(graph.segments.values())[0]
        assert seg.start_node_id in graph.nodes
        assert seg.end_node_id in graph.nodes

    def test_full_slope_building_workflow(self, workflow_complete_setup: tuple) -> None:
        """Complete workflow: start → commit x3 → finish slope."""
        sm, ctx, graph, factory, dem = workflow_complete_setup

        # === Step 1: Start building from summit ===
        start_lon, start_lat = 0.0, 0.0
        start_elev = dem.get_elevation(lon=start_lon, lat=start_lat)

        sm.start_building(lon=start_lon, lat=start_lat, elevation=start_elev, node_id=None, slope_number=1)
        assert sm.get_state_name() == "SlopeBuilding"
        assert ctx.building.name == "Slope 1"

        # Generate initial proposals
        proposals = list(
            factory.generate_fan(lon=start_lon, lat=start_lat, elevation=start_elev, target_length_m=400.0)
        )
        ctx.proposals.paths = proposals

        # === Step 2: Commit first segment ===
        path1 = proposals[0]
        end_nodes = graph.commit_paths(paths=[path1])
        seg1_id = list(graph.segments.keys())[0]
        sm.commit_path(segment_id=seg1_id, endpoint_node_id=end_nodes[0])

        assert sm.get_state_name() == "SlopeBuilding"
        assert seg1_id in ctx.building.segments
        assert len(graph.segments) == 1

        # === Step 3: Continue from endpoint ===
        end_node = graph.nodes[end_nodes[0]]
        proposals2 = list(
            factory.generate_fan(
                lon=end_node.lon, lat=end_node.lat, elevation=end_node.elevation, target_length_m=400.0
            )
        )
        assert len(proposals2) > 0

        # Commit second segment
        path2 = proposals2[0]
        end_nodes2 = graph.commit_paths(paths=[path2])
        seg2_id = list(graph.segments.keys())[-1]
        sm.commit_path(segment_id=seg2_id, endpoint_node_id=end_nodes2[0])

        assert len(graph.segments) == 2
        assert len(ctx.building.segments) == 2

        # === Step 4: One more segment ===
        end_node2 = graph.nodes[end_nodes2[0]]
        proposals3 = list(
            factory.generate_fan(
                lon=end_node2.lon, lat=end_node2.lat, elevation=end_node2.elevation, target_length_m=400.0
            )
        )
        path3 = proposals3[0]
        end_nodes3 = graph.commit_paths(paths=[path3])
        seg3_id = list(graph.segments.keys())[-1]
        sm.commit_path(segment_id=seg3_id, endpoint_node_id=end_nodes3[0])

        assert len(graph.segments) == 3
        assert len(ctx.building.segments) == 3

        # === Step 5: Finish the slope ===
        slope = graph.finish_slope(segment_ids=ctx.building.segments)
        sm.finish_slope(slope_id=slope.id)

        assert sm.get_state_name() == "Idle"  # Panel shown via ctx.viewing.panel_visible
        assert ctx.viewing.panel_visible is True
        assert ctx.viewing.slope_id == slope.id
        assert len(graph.slopes) == 1
        assert slope.name is not None

        # Verify total drop (should be cumulative)
        final_node = graph.nodes[end_nodes3[0]]
        total_drop = start_elev - final_node.elevation
        assert total_drop > 100  # Dropped over multiple segments

    def test_branch_from_intermediate_node(self, workflow_complete_setup: tuple) -> None:
        """Create a branch: build slope, then start new slope from mid-node."""
        sm, ctx, graph, factory, dem = workflow_complete_setup

        # === Build first slope (2 segments) ===
        proposals = list(factory.generate_fan(lon=0.0, lat=0.0, elevation=2500.0, target_length_m=400.0))
        sm.start_building(lon=0.0, lat=0.0, elevation=2500.0, node_id=None, slope_number=1)

        # Commit segment 1
        end1 = graph.commit_paths(paths=[proposals[0]])
        seg1_id = list(graph.segments.keys())[0]
        sm.commit_path(segment_id=seg1_id, endpoint_node_id=end1[0])

        # Remember mid-point node for branching
        mid_node_id = end1[0]
        mid_node = graph.nodes[mid_node_id]

        # Commit segment 2
        proposals2 = list(
            factory.generate_fan(
                lon=mid_node.lon, lat=mid_node.lat, elevation=mid_node.elevation, target_length_m=400.0
            )
        )
        end2 = graph.commit_paths(paths=[proposals2[0]])
        seg2_id = list(graph.segments.keys())[-1]
        sm.commit_path(segment_id=seg2_id, endpoint_node_id=end2[0])

        # Finish first slope
        slope1 = graph.finish_slope(segment_ids=[seg1_id, seg2_id])
        sm.finish_slope(slope_id=slope1.id)
        assert sm.get_state_name() == "Idle"  # Panel shown via ctx.viewing.panel_visible
        sm.hide_info_panel()  # Close panel to continue building
        assert sm.get_state_name() == "Idle"  # Still idle after hiding panel

        # === Branch: Start new slope from mid-node ===
        sm.start_building(
            lon=mid_node.lon, lat=mid_node.lat, elevation=mid_node.elevation, node_id=mid_node_id, slope_number=2
        )
        assert sm.get_state_name() == "SlopeBuilding"
        assert ctx.building.name == "Slope 2"

        # Generate proposals from mid-node (they will go different direction)
        branch_proposals = list(
            factory.generate_fan(
                lon=mid_node.lon, lat=mid_node.lat, elevation=mid_node.elevation, target_length_m=400.0
            )
        )
        assert len(branch_proposals) > 0

        # Commit branch segment (pick different path for variety)
        branch_path = branch_proposals[-1]  # Last one often has different bearing
        end_branch = graph.commit_paths(paths=[branch_path])
        branch_seg_id = list(graph.segments.keys())[-1]
        sm.commit_path(segment_id=branch_seg_id, endpoint_node_id=end_branch[0])

        # Finish branch slope
        slope2 = graph.finish_slope(segment_ids=[branch_seg_id])
        sm.finish_slope(slope_id=slope2.id)
        assert sm.get_state_name() == "Idle"  # Panel shown via ctx.viewing.panel_visible

        # Verify we have 2 slopes, 3 segments total
        assert len(graph.slopes) == 2
        assert len(graph.segments) == 3

        # Mid-node should be connected to 3 segments (1 incoming from summit, 2 outgoing)
        connections = graph.get_connection_count(node_id=mid_node_id)
        assert connections == 3

    def test_place_lift_from_valley_to_summit(self, workflow_complete_setup: tuple) -> None:
        """Place lift connecting valley (low) to summit (high)."""
        sm, ctx, graph, factory, dem = workflow_complete_setup

        # === Build a slope first to create nodes ===
        proposals = list(factory.generate_fan(lon=0.0, lat=0.0, elevation=2500.0, target_length_m=600.0))
        sm.start_building(lon=0.0, lat=0.0, elevation=2500.0, node_id=None, slope_number=1)

        path = proposals[0]
        end_nodes = graph.commit_paths(paths=[path])
        seg_id = list(graph.segments.keys())[0]
        sm.commit_path(segment_id=seg_id, endpoint_node_id=end_nodes[0])

        slope = graph.finish_slope(segment_ids=[seg_id])
        sm.finish_slope(slope_id=slope.id)
        sm.hide_info_panel()  # Close panel to place lift

        # Get summit and valley nodes
        summit_node_id = graph.segments[seg_id].start_node_id
        valley_node_id = graph.segments[seg_id].end_node_id
        summit = graph.nodes[summit_node_id]
        valley = graph.nodes[valley_node_id]

        assert summit.elevation > valley.elevation

        # === Set lift mode and place lift ===
        ctx.build_mode.mode = BuildMode.CHAIRLIFT
        assert sm.get_state_name() == "Idle"
        assert sm.is_lift_mode() is True

        # Select valley as start (bottom station)
        sm.select_lift_start(node_id=valley_node_id)
        assert sm.get_state_name() == "LiftPlacing"
        assert ctx.lift.start_node_id == valley_node_id

        # Add lift to summit
        lift = graph.add_lift(
            start_node_id=valley_node_id,
            end_node_id=summit_node_id,
            lift_type="chairlift",
            dem=dem,
        )
        assert lift is not None
        sm.complete_lift(lift_id=lift.id)

        # After placing lift, go to Idle with panel visible
        assert sm.get_state_name() == "Idle"
        assert ctx.viewing.panel_visible is True
        assert ctx.viewing.lift_id == lift.id
        assert len(graph.lifts) == 1

        # Verify lift properties
        assert lift.start_node_id == valley_node_id
        assert lift.end_node_id == summit_node_id
        vertical_rise = lift.get_vertical_rise(nodes=graph.nodes)
        assert vertical_rise > 0

    def test_undo_segment_in_workflow(self, workflow_complete_setup: tuple) -> None:
        """Test undo during slope building removes segment correctly."""
        sm, ctx, graph, factory, dem = workflow_complete_setup

        # Start building
        proposals = list(factory.generate_fan(lon=0.0, lat=0.0, elevation=2500.0, target_length_m=400.0))
        sm.start_building(lon=0.0, lat=0.0, elevation=2500.0, node_id=None, slope_number=1)

        # Commit THREE segments (so undo leaves 2, staying in Building)
        end1 = graph.commit_paths(paths=[proposals[0]])
        seg1_id = list(graph.segments.keys())[0]
        sm.commit_path(segment_id=seg1_id, endpoint_node_id=end1[0])

        end_node = graph.nodes[end1[0]]
        proposals2 = list(
            factory.generate_fan(
                lon=end_node.lon, lat=end_node.lat, elevation=end_node.elevation, target_length_m=400.0
            )
        )
        end2 = graph.commit_paths(paths=[proposals2[0]])
        seg2_id = list(graph.segments.keys())[-1]
        sm.commit_path(segment_id=seg2_id, endpoint_node_id=end2[0])

        end_node2 = graph.nodes[end2[0]]
        proposals3 = list(
            factory.generate_fan(
                lon=end_node2.lon, lat=end_node2.lat, elevation=end_node2.elevation, target_length_m=400.0
            )
        )
        end3 = graph.commit_paths(paths=[proposals3[0]])
        seg3_id = list(graph.segments.keys())[-1]
        sm.commit_path(segment_id=seg3_id, endpoint_node_id=end3[0])

        assert len(ctx.building.segments) == 3
        assert len(graph.segments) == 3

        # Undo last segment (leaves 2 segments, stays in Building)
        undone = graph.undo_last()
        assert undone is not None
        ctx.building.segments.remove(seg3_id)
        sm.undo_segment(removed_segment_id=seg3_id)

        # Should stay in Building with 2 segments (undo with >1 stays in Building)
        assert sm.get_state_name() == "SlopeBuilding"
        assert len(ctx.building.segments) == 2
        assert seg1_id in ctx.building.segments
        assert seg2_id in ctx.building.segments

    def test_complete_resort_with_multiple_slopes_and_lift(self, workflow_complete_setup: tuple) -> None:
        """Full resort: 2 slopes from summit + connecting lift."""
        sm, ctx, graph, factory, dem = workflow_complete_setup

        # === Slope 1: Summit to Valley ===
        proposals = list(factory.generate_fan(lon=0.0, lat=0.0, elevation=2500.0, target_length_m=500.0))
        sm.start_building(lon=0.0, lat=0.0, elevation=2500.0, node_id=None, slope_number=1)

        end1 = graph.commit_paths(paths=[proposals[0]])
        sm.commit_path(segment_id="S1", endpoint_node_id=end1[0])

        slope1 = graph.finish_slope(segment_ids=["S1"], name="Main Run")
        sm.finish_slope(slope_id=slope1.id)
        sm.hide_info_panel()  # Close panel before building next slope

        summit_id = "N1"
        valley1_id = end1[0]

        # === Slope 2: Different direction from summit ===
        # Use a different proposal that goes a different direction
        proposals2 = list(factory.generate_fan(lon=0.0, lat=0.0, elevation=2500.0, target_length_m=500.0))
        sm.start_building(lon=0.0, lat=0.0, elevation=2500.0, node_id=summit_id, slope_number=2)

        # Pick last proposal (usually different direction)
        path2 = proposals2[-1] if len(proposals2) > 1 else proposals2[0]
        end2 = graph.commit_paths(paths=[path2])
        seg2_id = list(graph.segments.keys())[-1]
        sm.commit_path(segment_id=seg2_id, endpoint_node_id=end2[0])

        slope2 = graph.finish_slope(segment_ids=[seg2_id], name="Side Run")
        sm.finish_slope(slope_id=slope2.id)
        sm.hide_info_panel()  # Close panel before placing lift

        valley2_id = end2[0]

        # === Add lift from Valley 1 back to Summit ===
        ctx.build_mode.mode = BuildMode.CHAIRLIFT
        sm.select_lift_start(node_id=valley1_id)
        lift = graph.add_lift(start_node_id=valley1_id, end_node_id=summit_id, lift_type="chairlift", dem=dem)
        sm.complete_lift(lift_id=lift.id)

        # === Verify complete resort ===
        assert len(graph.slopes) == 2
        assert len(graph.segments) == 2
        assert len(graph.lifts) == 1
        assert len(graph.nodes) == 3  # Summit (N1), Valley1 (N2), Valley2 (N3)

        # Verify node connections
        # Summit (N1): 2 slopes going down + 1 lift arriving = 3 connections
        assert graph.get_connection_count(node_id=summit_id) == 3
        # Valley1 (N2): 1 slope arriving + 1 lift departing = 2 connections
        assert graph.get_connection_count(node_id=valley1_id) == 2
        # Valley2 (N3): 1 slope arriving = 1 connection
        assert graph.get_connection_count(node_id=valley2_id) == 1

        # Verify elevations
        summit = graph.nodes[summit_id]
        valley1 = graph.nodes[valley1_id]
        assert summit.elevation > valley1.elevation

    def test_path_selection_in_context(self, workflow_complete_setup: tuple) -> None:
        """Test proposal selection/switching in context."""
        sm, ctx, graph, factory, dem = workflow_complete_setup

        proposals = list(factory.generate_fan(lon=0.0, lat=0.0, elevation=2500.0, target_length_m=400.0))
        ctx.proposals.paths = proposals

        # Initially no selection
        ctx.proposals.selected_idx = None
        assert ctx.proposals.selected_idx is None

        # Select first proposal
        ctx.proposals.selected_idx = 0
        assert ctx.proposals.selected_idx == 0

        # Switch to different proposal
        ctx.proposals.selected_idx = len(proposals) - 1
        assert ctx.proposals.paths[ctx.proposals.selected_idx] == proposals[-1]

    def test_view_existing_slope(self, workflow_complete_setup: tuple) -> None:
        """View an existing slope transitions correctly."""
        sm, ctx, graph, factory, dem = workflow_complete_setup

        # Build and finish a slope
        proposals = list(factory.generate_fan(lon=0.0, lat=0.0, elevation=2500.0, target_length_m=400.0))
        sm.start_building(lon=0.0, lat=0.0, elevation=2500.0, node_id=None, slope_number=1)
        end = graph.commit_paths(paths=[proposals[0]])
        sm.commit_path(segment_id="S1", endpoint_node_id=end[0])
        slope = graph.finish_slope(segment_ids=["S1"])
        sm.finish_slope(slope_id=slope.id)

        # finish_slope goes to Idle with panel visible
        assert sm.get_state_name() == "Idle"
        assert ctx.viewing.panel_visible is True
        assert ctx.viewing.slope_id == slope.id

        # Close panel
        sm.hide_info_panel()
        assert sm.get_state_name() == "Idle"  # State unchanged
        assert ctx.viewing.panel_visible is False


# =============================================================================
# DEFERRED ACTION FLAG TESTS
# Test the pending_* flags on PlannerContext
# =============================================================================


class TestDeferredActionFlags:
    """Test deferred action flags in PlannerContext."""

    def test_pending_path_generation_default_false(self) -> None:
        """pending_path_generation starts as False."""
        ctx = PlannerContext()
        assert ctx.deferred.path_generation is False

    def test_pending_gradient_target_default_none(self) -> None:
        """pending_gradient_target starts as None."""
        ctx = PlannerContext()
        assert ctx.deferred.gradient_target is None

    def test_pending_auto_finish_default_false(self) -> None:
        """pending_auto_finish starts as False."""
        ctx = PlannerContext()
        assert ctx.deferred.auto_finish is False

    def test_pending_flags_can_be_set(self) -> None:
        """Pending flags can be set for deferred work."""
        ctx = PlannerContext()
        ctx.deferred.path_generation = True
        ctx.deferred.gradient_target = 25.0
        ctx.deferred.auto_finish = True

        assert ctx.deferred.path_generation is True
        assert ctx.deferred.gradient_target == 25.0
        assert ctx.deferred.auto_finish is True

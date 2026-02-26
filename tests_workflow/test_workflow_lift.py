"""Integration test for lift placement workflow.

Tests the full lift lifecycle: start → select end → complete.
"""

from skiresort_planner.constants import MapConfig
from skiresort_planner.model.node import Node
from skiresort_planner.model.path_point import PathPoint
from skiresort_planner.model.resort_graph import AddLiftAction, DeleteLiftAction, ResortGraph
from skiresort_planner.ui.state_machine import PlannerStateMachine


class TestLiftPlacementWorkflow:
    """Tests for complete lift placement workflow."""

    def test_complete_lift_workflow(self, mock_dem_blue_slope) -> None:
        """Place a lift: IdleReady → LiftPlacing → IdleViewingLift.

        Tests:
        - start_lift transitions to LiftPlacing
        - complete_lift creates lift and transitions to viewing
        - Lift connects start and end nodes
        """
        dem = mock_dem_blue_slope
        M = MapConfig.METERS_PER_DEGREE_EQUATOR
        graph = ResortGraph()
        sm, ctx = PlannerStateMachine.create(graph=graph)

        # Create start node at valley (lower elevation)
        start_loc = PathPoint(
            lon=0.0,
            lat=-1000 / M,
            elevation=dem.get_elevation_or_raise(lon=0.0, lat=-1000 / M),
        )
        start_node = Node(id="N1", location=start_loc)
        graph.nodes["N1"] = start_node

        # Create end node at summit (higher elevation)
        end_loc = PathPoint(
            lon=0.0,
            lat=0.0,
            elevation=dem.get_elevation_or_raise(lon=0.0, lat=0.0),
        )
        end_node = Node(id="N2", location=end_loc)
        graph.nodes["N2"] = end_node

        # === Phase 1: Start lift placement ===
        assert sm.current_state_value == "idle_ready"

        sm.start_lift(
            node_id="N1",
            lon=start_loc.lon,
            lat=start_loc.lat,
            elevation=start_loc.elevation,
        )

        assert sm.current_state_value == "lift_placing", "Should be in LiftPlacing"
        assert ctx.lift.start_node_id == "N1", "Start node should be stored"

        # === Phase 2: Complete lift ===
        lift = graph.add_lift(
            start_node_id="N1",
            end_node_id="N2",
            lift_type="chairlift",
            dem=dem,
        )

        sm.complete_lift(lift_id=lift.id)

        assert sm.current_state_value == "idle_viewing_lift", "Should be viewing lift"
        assert ctx.viewing.lift_id == lift.id, "Should be viewing created lift"
        assert ctx.viewing.panel_visible is True, "Panel should be visible"

        # Verify lift was created correctly
        assert len(graph.lifts) == 1, "Should have 1 lift"
        created_lift = graph.lifts[lift.id]
        assert created_lift.start_node_id == "N1"
        assert created_lift.end_node_id == "N2"
        assert created_lift.lift_type == "chairlift"

    def test_cancel_lift_returns_to_idle(self, mock_dem_blue_slope) -> None:
        """cancel_lift from LiftPlacing returns to IdleReady."""
        dem = mock_dem_blue_slope
        M = MapConfig.METERS_PER_DEGREE_EQUATOR
        graph = ResortGraph()
        sm, ctx = PlannerStateMachine.create(graph=graph)

        # Setup node
        start_loc = PathPoint(
            lon=0.0,
            lat=-1000 / M,
            elevation=dem.get_elevation_or_raise(lon=0.0, lat=-1000 / M),
        )
        graph.nodes["N1"] = Node(id="N1", location=start_loc)

        sm.start_lift(
            node_id="N1",
            lon=start_loc.lon,
            lat=start_loc.lat,
            elevation=start_loc.elevation,
        )
        assert sm.current_state_value == "lift_placing"

        sm.cancel_lift()

        assert sm.current_state_value == "idle_ready", "Should return to IdleReady"


class TestLiftUndoOperations:
    """Tests for lift undo operations."""

    def test_undo_add_lift_removes_lift(self, mock_dem_blue_slope) -> None:
        """undo_last for AddLiftAction removes the lift."""
        dem = mock_dem_blue_slope
        M = MapConfig.METERS_PER_DEGREE_EQUATOR
        graph = ResortGraph()

        # Create nodes
        graph.nodes["N1"] = Node(
            id="N1",
            location=PathPoint(
                lon=0.0,
                lat=-1000 / M,
                elevation=dem.get_elevation_or_raise(lon=0.0, lat=-1000 / M),
            ),
        )
        graph.nodes["N2"] = Node(
            id="N2",
            location=PathPoint(
                lon=0.0,
                lat=0.0,
                elevation=dem.get_elevation_or_raise(lon=0.0, lat=0.0),
            ),
        )

        # Add lift
        graph.add_lift(
            start_node_id="N1",
            end_node_id="N2",
            lift_type="gondola",
            dem=dem,
        )

        assert len(graph.lifts) == 1
        assert len(graph.undo_stack) == 1
        assert isinstance(graph.undo_stack[0], AddLiftAction)

        # Undo
        graph.undo_last()

        assert len(graph.lifts) == 0, "Lift should be removed"

    def test_undo_delete_lift_restores_lift(self, mock_dem_blue_slope) -> None:
        """undo_last for DeleteLiftAction restores the lift."""
        dem = mock_dem_blue_slope
        M = MapConfig.METERS_PER_DEGREE_EQUATOR
        graph = ResortGraph()

        # Create nodes
        graph.nodes["N1"] = Node(
            id="N1",
            location=PathPoint(
                lon=0.0,
                lat=-1000 / M,
                elevation=dem.get_elevation_or_raise(lon=0.0, lat=-1000 / M),
            ),
        )
        graph.nodes["N2"] = Node(
            id="N2",
            location=PathPoint(
                lon=0.0,
                lat=0.0,
                elevation=dem.get_elevation_or_raise(lon=0.0, lat=0.0),
            ),
        )

        # Add and delete lift
        lift = graph.add_lift(
            start_node_id="N1",
            end_node_id="N2",
            lift_type="chairlift",
            dem=dem,
        )
        lift_name = lift.name
        lift_id = lift.id

        graph.delete_lift(lift_id=lift_id)
        assert len(graph.lifts) == 0

        # Undo delete
        undone = graph.undo_last()

        assert isinstance(undone, DeleteLiftAction)
        assert len(graph.lifts) == 1, "Lift should be restored"
        assert graph.lifts[lift_id].name == lift_name, "Lift name should be preserved"


class TestSwitchLiftSelfLoop:
    """Tests for self-loop switch_lift transition."""

    def test_switch_lift_updates_viewed_lift(self, mock_dem_blue_slope) -> None:
        """Self-loop switch_lift updates the viewed lift."""
        dem = mock_dem_blue_slope
        M = MapConfig.METERS_PER_DEGREE_EQUATOR
        graph = ResortGraph()
        sm, ctx = PlannerStateMachine.create(graph=graph)

        # Create nodes for two lifts
        for i, lat_offset in enumerate([0, 100]):
            start_lat = (-1000 - lat_offset) / M
            end_lat = -lat_offset / M
            graph.nodes[f"N{2 * i + 1}"] = Node(
                id=f"N{2 * i + 1}",
                location=PathPoint(
                    lon=0.0,
                    lat=start_lat,
                    elevation=dem.get_elevation_or_raise(lon=0.0, lat=start_lat),
                ),
            )
            graph.nodes[f"N{2 * i + 2}"] = Node(
                id=f"N{2 * i + 2}",
                location=PathPoint(
                    lon=0.0,
                    lat=end_lat,
                    elevation=dem.get_elevation_or_raise(lon=0.0, lat=end_lat),
                ),
            )

        # Create two lifts
        lift1 = graph.add_lift(start_node_id="N1", end_node_id="N2", lift_type="chairlift", dem=dem)
        lift2 = graph.add_lift(start_node_id="N3", end_node_id="N4", lift_type="gondola", dem=dem)

        # View first lift
        sm.send("view_lift", lift_id=lift1.id)
        assert ctx.viewing.lift_id == lift1.id

        # Switch to second lift (self-loop)
        sm.send("view_lift", lift_id=lift2.id)

        assert sm.current_state_value == "idle_viewing_lift", "Still in viewing state"
        assert ctx.viewing.lift_id == lift2.id, "Should now view second lift"
        assert ctx.viewing.panel_visible is True, "Panel should remain visible"

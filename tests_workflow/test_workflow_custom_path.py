"""Integration test for custom path connection workflow.

Tests the custom connect feature: enable → pick target → view options → commit.
"""

from tests_workflow.conftest import WorkflowSetup


class TestCustomConnectWorkflow:
    """Tests for custom path connection workflow."""

    def test_custom_connect_from_starting_state(self, workflow_setup: WorkflowSetup) -> None:
        """Enable custom connect mode from SlopeStarting state.

        Workflow: SlopeStarting → enable_custom → SlopeCustomPicking
        """
        sm, ctx, graph, factory, dem = workflow_setup

        # Start slope
        start_elev = dem.get_elevation_or_raise(lon=0.0, lat=0.0)
        sm.start_slope(lon=0.0, lat=0.0, elevation=start_elev, node_id=None)
        assert sm.current_state_value == "slope_starting"

        # Enable custom connect
        sm.enable_custom()

        assert sm.current_state_value == "slope_custom_picking", "Should be in custom picking"

    def test_custom_connect_from_building_state(self, workflow_setup: WorkflowSetup) -> None:
        """Enable custom connect mode from SlopeBuilding state.

        Workflow: SlopeBuilding → enable_custom → SlopeCustomPicking
        """
        sm, ctx, graph, factory, dem = workflow_setup

        # Build one segment first
        start_elev = dem.get_elevation_or_raise(lon=0.0, lat=0.0)
        sm.start_slope(lon=0.0, lat=0.0, elevation=start_elev, node_id=None)

        proposals = list(factory.generate_fan(lon=0.0, lat=0.0, elevation=start_elev))
        endpoint_ids = graph.commit_paths(paths=[proposals[0]])
        seg_id = list(graph.segments.keys())[0]
        sm.commit_path(segment_id=seg_id, endpoint_node_id=endpoint_ids[0])

        assert sm.current_state_value == "slope_building"

        # Enable custom connect
        sm.enable_custom()

        assert sm.current_state_value == "slope_custom_picking", "Should be in custom picking"


class TestCancelCustomConnect:
    """Tests for canceling custom connect mode."""

    def test_cancel_custom_returns_to_starting(self, workflow_setup: WorkflowSetup) -> None:
        """cancel_custom from SlopeCustomPicking returns to SlopeStarting when no segments."""
        sm, ctx, graph, factory, dem = workflow_setup

        # Start slope and enable custom (no segments committed)
        start_elev = dem.get_elevation_or_raise(lon=0.0, lat=0.0)
        sm.start_slope(lon=0.0, lat=0.0, elevation=start_elev, node_id=None)
        sm.enable_custom()

        assert sm.current_state_value == "slope_custom_picking"
        assert len(ctx.building.segments) == 0, "No segments committed"

        # Cancel custom - should return to starting (no segments)
        sm.cancel_custom()

        assert sm.current_state_value == "slope_starting", "Should return to starting"

    def test_cancel_custom_returns_to_building(self, workflow_setup: WorkflowSetup) -> None:
        """cancel_custom from SlopeCustomPicking returns to SlopeBuilding when has segments."""
        sm, ctx, graph, factory, dem = workflow_setup

        # Start and build one segment
        start_elev = dem.get_elevation_or_raise(lon=0.0, lat=0.0)
        sm.start_slope(lon=0.0, lat=0.0, elevation=start_elev, node_id=None)

        proposals = list(factory.generate_fan(lon=0.0, lat=0.0, elevation=start_elev))
        endpoint_ids = graph.commit_paths(paths=[proposals[0]])
        seg_id = list(graph.segments.keys())[0]
        sm.commit_path(segment_id=seg_id, endpoint_node_id=endpoint_ids[0])

        # Enable custom
        sm.enable_custom()
        assert sm.current_state_value == "slope_custom_picking"
        assert len(ctx.building.segments) == 1, "Has 1 segment"

        # Cancel custom - should return to building (has segments)
        sm.cancel_custom()

        assert sm.current_state_value == "slope_building", "Should return to building"


class TestCancelSlopeFromCustom:
    """Tests for cancel_slope from custom connect states."""

    def test_cancel_slope_from_custom_picking(self, workflow_setup: WorkflowSetup) -> None:
        """cancel_slope from SlopeCustomPicking returns to IdleReady."""
        sm, ctx, graph, factory, dem = workflow_setup

        start_elev = dem.get_elevation_or_raise(lon=0.0, lat=0.0)
        sm.start_slope(lon=0.0, lat=0.0, elevation=start_elev, node_id=None)
        sm.enable_custom()

        assert sm.current_state_value == "slope_custom_picking"

        # Cancel entire slope
        sm.cancel_slope()

        assert sm.current_state_value == "idle_ready", "Should return to IdleReady"


class TestSelectCustomTarget:
    """Tests for selecting custom target location."""

    def test_select_target_transitions_to_custom_path(self, workflow_setup: WorkflowSetup) -> None:
        """select_custom_target transitions from SlopeCustomPicking to SlopeCustomPath."""
        sm, ctx, graph, factory, dem = workflow_setup
        from skiresort_planner.constants import MapConfig

        M = MapConfig.METERS_PER_DEGREE_EQUATOR

        start_elev = dem.get_elevation_or_raise(lon=0.0, lat=0.0)
        sm.start_slope(lon=0.0, lat=0.0, elevation=start_elev, node_id=None)
        sm.enable_custom()

        assert sm.current_state_value == "slope_custom_picking"

        # Select a target (downhill from start)
        target_lat = -500 / M
        target_elev = dem.get_elevation_or_raise(lon=0.0, lat=target_lat)
        target_location = (0.0, target_lat, target_elev)

        sm.select_custom_target(target_location=target_location)

        assert sm.current_state_value == "slope_custom_path", "Should be in custom path state"
        assert ctx.custom_connect.target_location is not None
        assert ctx.custom_connect.target_location[0] == 0.0  # lon
        assert abs(ctx.custom_connect.target_location[1] - target_lat) < 0.0001  # lat


class TestCommitCustomContinue:
    """Tests for commit_custom_continue transition (slope_custom_path → slope_building)."""

    def test_commit_custom_continue_transitions_to_building(self, workflow_setup: WorkflowSetup) -> None:
        """commit_custom_continue from SlopeCustomPath returns to SlopeBuilding."""
        sm, ctx, graph, factory, dem = workflow_setup
        from skiresort_planner.constants import MapConfig

        M = MapConfig.METERS_PER_DEGREE_EQUATOR

        # 1. Start slope
        start_elev = dem.get_elevation_or_raise(lon=0.0, lat=0.0)
        sm.start_slope(lon=0.0, lat=0.0, elevation=start_elev, node_id=None)

        # 2. Commit first segment to get to slope_building
        proposals = list(factory.generate_fan(lon=0.0, lat=0.0, elevation=start_elev))
        endpoint_ids = graph.commit_paths(paths=[proposals[0]])
        seg_id_1 = list(graph.segments.keys())[0]
        sm.commit_path(segment_id=seg_id_1, endpoint_node_id=endpoint_ids[0])
        assert sm.current_state_value == "slope_building"

        # 3. Enable custom and select target
        sm.enable_custom()
        target_lat = -500 / M
        target_elev = dem.get_elevation_or_raise(lon=0.0, lat=target_lat)
        sm.select_custom_target(target_location=(0.0, target_lat, target_elev))
        assert sm.current_state_value == "slope_custom_path"

        # 4. Simulate committing a custom path segment (continue building)
        end_node = graph.nodes[endpoint_ids[0]]
        proposals_2 = list(factory.generate_fan(lon=end_node.lon, lat=end_node.lat, elevation=end_node.elevation))
        endpoint_ids_2 = graph.commit_paths(paths=[proposals_2[0]])
        seg_id_2 = list(graph.segments.keys())[-1]

        # 5. Call commit_custom_continue
        sm.commit_custom_continue(segment_id=seg_id_2, endpoint_node_id=endpoint_ids_2[0])

        assert sm.current_state_value == "slope_building", "Should return to slope_building"
        assert seg_id_2 in ctx.building.segments, "New segment should be tracked"
        assert ctx.custom_connect.target_location is None, "Custom connect should be cleared"


class TestCommitCustomFinish:
    """Tests for commit_custom_finish transition (slope_custom_path → idle_viewing_slope)."""

    def test_commit_custom_finish_transitions_to_viewing(self, workflow_setup: WorkflowSetup) -> None:
        """commit_custom_finish from SlopeCustomPath transitions to IdleViewingSlope."""
        sm, ctx, graph, factory, dem = workflow_setup
        from skiresort_planner.constants import MapConfig

        M = MapConfig.METERS_PER_DEGREE_EQUATOR

        # 1. Start slope
        start_elev = dem.get_elevation_or_raise(lon=0.0, lat=0.0)
        sm.start_slope(lon=0.0, lat=0.0, elevation=start_elev, node_id=None)

        # 2. Commit first segment
        proposals = list(factory.generate_fan(lon=0.0, lat=0.0, elevation=start_elev))
        endpoint_ids = graph.commit_paths(paths=[proposals[0]])
        seg_id_1 = list(graph.segments.keys())[0]
        sm.commit_path(segment_id=seg_id_1, endpoint_node_id=endpoint_ids[0])
        assert sm.current_state_value == "slope_building"

        # 3. Enable custom and select target
        sm.enable_custom()
        target_lat = -500 / M
        target_elev = dem.get_elevation_or_raise(lon=0.0, lat=target_lat)
        sm.select_custom_target(target_location=(0.0, target_lat, target_elev))
        assert sm.current_state_value == "slope_custom_path"

        # 4. Simulate committing a connector segment and finishing slope
        end_node = graph.nodes[endpoint_ids[0]]
        proposals_2 = list(factory.generate_fan(lon=end_node.lon, lat=end_node.lat, elevation=end_node.elevation))
        endpoint_ids_2 = graph.commit_paths(paths=[proposals_2[0]])
        seg_id_2 = list(graph.segments.keys())[-1]

        # Add segment to building context before finishing
        ctx.building.segments.append(seg_id_2)

        # Finish the slope
        slope = graph.finish_slope(segment_ids=ctx.building.segments)
        assert slope is not None, "Slope should be created"

        # 5. Call commit_custom_finish
        sm.commit_custom_finish(segment_id=seg_id_2, slope_id=slope.id)

        assert sm.current_state_value == "idle_viewing_slope", "Should transition to viewing slope"
        assert ctx.viewing.slope_id == slope.id, "Viewing context should have slope ID"
        assert ctx.custom_connect.target_location is None, "Custom connect should be cleared"

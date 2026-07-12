"""Integration test for custom path connection workflow.

Tests the custom-connect feature: a terrain/node click routes a path to that target
(select_custom_target → SlopeCustomPath), re-targeting is a self-loop, and Cancel
Connection (cancel_custom) returns to the fan-out. There is no button and no picking
state — targeting is map-only, mirroring roads.
"""

from skiresort_planner.constants import MapConfig
from skiresort_planner.ui.context import CustomConnectContext
from tests_workflow.conftest import WorkflowSetup

M = MapConfig.METERS_PER_DEGREE_EQUATOR


def _commit_first_segment(sm, graph, factory, start_elev) -> str:
    """Commit one fan segment from the origin so the SM reaches slope_building."""
    proposals = list(factory.generate_fan(lon=0.0, lat=0.0, elevation=start_elev))
    endpoint_ids = graph.commit_paths(paths=[proposals[0]])
    seg_id = list(graph.segments.keys())[0]
    sm.commit_path(segment_id=seg_id, endpoint_node_id=endpoint_ids[0])
    return endpoint_ids[0]


class TestSelectCustomTargetWorkflow:
    """A target click routes a custom-connect path (SlopeCustomPath) from any slope state."""

    def test_select_target_from_starting_state(self, workflow_setup: WorkflowSetup) -> None:
        """SlopeStarting → select_custom_target → SlopeCustomPath (no button, no picking)."""
        sm, ctx, graph, factory, dem = workflow_setup

        start_elev = dem.get_elevation_or_raise(lon=0.0, lat=0.0)
        sm.start_slope(lon=0.0, lat=0.0, elevation=start_elev, node_id=None)
        assert sm.current_state_value == "slope_starting"

        target_lat = -500 / M
        target_elev = dem.get_elevation_or_raise(lon=0.0, lat=target_lat)
        sm.select_custom_target(target_location=(0.0, target_lat, target_elev))

        assert sm.current_state_value == "slope_custom_path", "Should route to custom path"
        assert ctx.custom_connect.force_mode, "force_mode set while showing custom proposals"
        # The origin node was materialised from the selection and captured as start_node.
        assert ctx.custom_connect.start_node is not None
        assert ctx.slope_build.start_node_id == ctx.custom_connect.start_node

    def test_select_target_from_building_state(self, workflow_setup: WorkflowSetup) -> None:
        """SlopeBuilding → select_custom_target → SlopeCustomPath."""
        sm, ctx, graph, factory, dem = workflow_setup

        start_elev = dem.get_elevation_or_raise(lon=0.0, lat=0.0)
        sm.start_slope(lon=0.0, lat=0.0, elevation=start_elev, node_id=None)
        endpoint_id = _commit_first_segment(sm, graph, factory, start_elev)
        assert sm.current_state_value == "slope_building"

        target_lat = -500 / M
        target_elev = dem.get_elevation_or_raise(lon=0.0, lat=target_lat)
        sm.select_custom_target(target_location=(0.0, target_lat, target_elev))

        assert sm.current_state_value == "slope_custom_path"
        assert ctx.custom_connect.start_node == endpoint_id, "routes from the current endpoint"

    def test_retarget_is_a_self_loop(self, workflow_setup: WorkflowSetup) -> None:
        """SlopeCustomPath → select_custom_target → SlopeCustomPath (re-target)."""
        sm, ctx, graph, factory, dem = workflow_setup

        start_elev = dem.get_elevation_or_raise(lon=0.0, lat=0.0)
        sm.start_slope(lon=0.0, lat=0.0, elevation=start_elev, node_id=None)
        sm.select_custom_target(target_location=(0.0, -400 / M, dem.get_elevation_or_raise(lon=0.0, lat=-400 / M)))
        start_node_first = ctx.custom_connect.start_node
        assert sm.current_state_value == "slope_custom_path"

        # Click a new target → stays in custom path, target moves, start node unchanged.
        new_lat = -600 / M
        new_elev = dem.get_elevation_or_raise(lon=0.0, lat=new_lat)
        sm.select_custom_target(target_location=(0.0, new_lat, new_elev))

        assert sm.current_state_value == "slope_custom_path", "re-target stays in custom path"
        assert ctx.custom_connect.start_node == start_node_first, "start node preserved on re-target"
        assert abs(ctx.custom_connect.target_location[1] - new_lat) < 0.0001, "target moved to new point"

    def test_target_node_captured_for_identity_reuse(self, workflow_setup: WorkflowSetup) -> None:
        """When the target is an existing node, its id is captured so commit reuses it
        by identity (not an 80m proximity guess that a drifted end could miss). A
        terrain target leaves it None (proximity fallback).
        """
        sm, ctx, graph, factory, dem = workflow_setup

        start_elev = dem.get_elevation_or_raise(lon=0.0, lat=0.0)
        sm.start_slope(lon=0.0, lat=0.0, elevation=start_elev, node_id=None)

        # Node target → its id is captured for exact reuse on commit.
        sm.select_custom_target(target_location=(0.0, -600 / M, 1880.0), target_node="N7")
        assert ctx.custom_connect.target_node == "N7", "clicked node's identity is captured"

        # Default (terrain target, no node kwarg) leaves it None → proximity fallback.
        assert CustomConnectContext().target_node is None


class TestCancelCustomConnect:
    """Cancel Connection (cancel_custom) leaves targeting and returns to the fan-out."""

    def test_cancel_custom_returns_to_starting(self, workflow_setup: WorkflowSetup) -> None:
        """cancel_custom from SlopeCustomPath returns to SlopeStarting when no segments."""
        sm, ctx, graph, factory, dem = workflow_setup

        start_elev = dem.get_elevation_or_raise(lon=0.0, lat=0.0)
        sm.start_slope(lon=0.0, lat=0.0, elevation=start_elev, node_id=None)
        sm.select_custom_target(target_location=(0.0, -500 / M, dem.get_elevation_or_raise(lon=0.0, lat=-500 / M)))

        assert sm.current_state_value == "slope_custom_path"
        assert len(ctx.slope_build.segments) == 0, "No segments committed"

        sm.cancel_custom()

        assert sm.current_state_value == "slope_starting", "Should return to starting"
        assert not ctx.custom_connect.force_mode, "custom state cleared"

    def test_cancel_custom_returns_to_building(self, workflow_setup: WorkflowSetup) -> None:
        """cancel_custom from SlopeCustomPath returns to SlopeBuilding when has segments."""
        sm, ctx, graph, factory, dem = workflow_setup

        start_elev = dem.get_elevation_or_raise(lon=0.0, lat=0.0)
        sm.start_slope(lon=0.0, lat=0.0, elevation=start_elev, node_id=None)
        _commit_first_segment(sm, graph, factory, start_elev)
        sm.select_custom_target(target_location=(0.0, -500 / M, dem.get_elevation_or_raise(lon=0.0, lat=-500 / M)))

        assert sm.current_state_value == "slope_custom_path"
        assert len(ctx.slope_build.segments) == 1, "Has 1 segment"

        sm.cancel_custom()

        assert sm.current_state_value == "slope_building", "Should return to building"


class TestCancelSlopeFromCustom:
    """cancel_slope discards the whole slope from the custom-path state."""

    def test_cancel_slope_from_custom_path(self, workflow_setup: WorkflowSetup) -> None:
        """cancel_slope from SlopeCustomPath returns to IdleReady."""
        sm, ctx, graph, factory, dem = workflow_setup

        start_elev = dem.get_elevation_or_raise(lon=0.0, lat=0.0)
        sm.start_slope(lon=0.0, lat=0.0, elevation=start_elev, node_id=None)
        sm.select_custom_target(target_location=(0.0, -500 / M, dem.get_elevation_or_raise(lon=0.0, lat=-500 / M)))

        assert sm.current_state_value == "slope_custom_path"

        sm.cancel_slope()

        assert sm.current_state_value == "idle_ready", "Should return to IdleReady"


class TestFinishSlopeFromCustom:
    """Sidebar Finish during targeting: finish_slope must be valid from slope_custom_path
    (regression for the TransitionNotAllowed crash), finalizing the committed segments and
    dropping the in-progress target proposal.
    """

    def test_finish_slope_from_custom_path_no_crash(self, workflow_setup: WorkflowSetup) -> None:
        sm, ctx, graph, factory, dem = workflow_setup

        # Commit one real segment, then start targeting a new point (SlopeCustomPath).
        start_elev = dem.get_elevation_or_raise(lon=0.0, lat=0.0)
        sm.start_slope(lon=0.0, lat=0.0, elevation=start_elev, node_id=None)
        _commit_first_segment(sm, graph, factory, start_elev)
        sm.select_custom_target(target_location=(0.0, -500 / M, dem.get_elevation_or_raise(lon=0.0, lat=-500 / M)))
        assert sm.current_state_value == "slope_custom_path"

        # Sidebar Finish fires the finish_slope event — must resolve, not raise.
        slope = graph.finish_slope(segment_ids=ctx.slope_build.segments)
        assert slope is not None
        sm.finish_slope(slope_id=slope.id)

        assert sm.current_state_value == "idle_viewing_slope", "Finish during targeting lands in viewing"
        assert not ctx.custom_connect.force_mode, "in-progress target cleared"


class TestCommitCustomContinue:
    """Tests for commit_custom_continue transition (slope_custom_path → slope_building)."""

    def test_commit_custom_continue_transitions_to_building(self, workflow_setup: WorkflowSetup) -> None:
        """commit_custom_continue from SlopeCustomPath returns to SlopeBuilding."""
        sm, ctx, graph, factory, dem = workflow_setup

        # 1. Start slope and commit first segment to reach slope_building.
        start_elev = dem.get_elevation_or_raise(lon=0.0, lat=0.0)
        sm.start_slope(lon=0.0, lat=0.0, elevation=start_elev, node_id=None)
        endpoint_id = _commit_first_segment(sm, graph, factory, start_elev)
        assert sm.current_state_value == "slope_building"

        # 2. Click a target → custom path.
        target_lat = -500 / M
        target_elev = dem.get_elevation_or_raise(lon=0.0, lat=target_lat)
        sm.select_custom_target(target_location=(0.0, target_lat, target_elev))
        assert sm.current_state_value == "slope_custom_path"

        # 3. Simulate committing a custom path segment (continue building).
        end_node = graph.nodes[endpoint_id]
        proposals_2 = list(factory.generate_fan(lon=end_node.lon, lat=end_node.lat, elevation=end_node.elevation))
        endpoint_ids_2 = graph.commit_paths(paths=[proposals_2[0]])
        seg_id_2 = list(graph.segments.keys())[-1]

        # 4. Call commit_custom_continue.
        sm.commit_custom_continue(segment_id=seg_id_2, endpoint_node_id=endpoint_ids_2[0])

        assert sm.current_state_value == "slope_building", "Should return to slope_building"
        assert seg_id_2 in ctx.slope_build.segments, "New segment should be tracked"
        assert ctx.custom_connect.target_location is None, "Custom connect should be cleared"


class TestCommitCustomFinish:
    """Tests for commit_custom_finish transition (slope_custom_path → idle_viewing_slope)."""

    def test_commit_custom_finish_transitions_to_viewing(self, workflow_setup: WorkflowSetup) -> None:
        """commit_custom_finish from SlopeCustomPath transitions to IdleViewingSlope."""
        sm, ctx, graph, factory, dem = workflow_setup

        # 1. Start slope and commit first segment.
        start_elev = dem.get_elevation_or_raise(lon=0.0, lat=0.0)
        sm.start_slope(lon=0.0, lat=0.0, elevation=start_elev, node_id=None)
        endpoint_id = _commit_first_segment(sm, graph, factory, start_elev)
        assert sm.current_state_value == "slope_building"

        # 2. Click a target → custom path.
        target_lat = -500 / M
        target_elev = dem.get_elevation_or_raise(lon=0.0, lat=target_lat)
        sm.select_custom_target(target_location=(0.0, target_lat, target_elev))
        assert sm.current_state_value == "slope_custom_path"

        # 3. Simulate committing a connector segment and finishing the slope.
        end_node = graph.nodes[endpoint_id]
        proposals_2 = list(factory.generate_fan(lon=end_node.lon, lat=end_node.lat, elevation=end_node.elevation))
        endpoint_ids_2 = graph.commit_paths(paths=[proposals_2[0]])
        seg_id_2 = list(graph.segments.keys())[-1]
        ctx.slope_build.segments.append(seg_id_2)

        slope = graph.finish_slope(segment_ids=ctx.slope_build.segments)
        assert slope is not None, "Slope should be created"

        # 4. Call commit_custom_finish.
        sm.commit_custom_finish(segment_id=seg_id_2, slope_id=slope.id)

        assert sm.current_state_value == "idle_viewing_slope", "Should transition to viewing slope"
        assert ctx.viewing.slope_id == slope.id, "Viewing context should have slope ID"
        assert ctx.custom_connect.target_location is None, "Custom connect should be cleared"

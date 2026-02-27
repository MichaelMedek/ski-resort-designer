"""End-to-End Integration Test using Streamlit AppTest Framework.

This is the "Grand Resort Tour" - ONE comprehensive integration test that simulates
a complete user workflow through the ski resort planner application.

CRITICAL: This test exercises the ACTION LAYER (actions.py, click_handlers.py)
by simulating CLICKS and BUTTON PRESSES - not just state machine methods!

Test Flow Mirrors Real User Interaction:
    1. Click terrain → dispatch_click(ClickInfo)
    2. Run deferred actions → execute_deferred_actions() helper
    3. Click path proposal → commit_selected_path(idx)
    4. Click Finish button → finish_current_slope()
    5. Click Undo button → undo_last_action()
    6. Click node → dispatch_click(ClickInfo with MarkerType.NODE)

Architecture:
- Uses AppTest.from_function with COMMAND-BASED execution
- Each at.run() processes ONE user action from command queue
- Uses deferred action functions for path generation (True E2E)
- Uses delete_slope_action/delete_lift_action for deletions

WHY THIS PATTERN MATTERS FOR COVERAGE:
- Direct SM/graph calls bypass the coordinator layer (actions.py)
- Using dispatch_click + action functions tests the real code paths
- Using deferred action functions tests the deferred action chain
- This catches bugs in the glue logic between state machine + graph + context
"""

from __future__ import annotations

import pytest
from streamlit.testing.v1 import AppTest

from skiresort_planner.constants import MapConfig
from skiresort_planner.core.path_tracer import PathTracer
from skiresort_planner.core.terrain_analyzer import TerrainAnalyzer
from skiresort_planner.generators.path_factory import PathFactory
from skiresort_planner.model.click_info import ClickInfo, MapClickType, MarkerType
from skiresort_planner.model.path_point import PathPoint
from skiresort_planner.model.resort_graph import ResortGraph
from tests_workflow.conftest import MockDEMService


# =============================================================================
# TEST CONSTANTS
# =============================================================================

M = MapConfig.METERS_PER_DEGREE_EQUATOR


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def apptest_dem() -> MockDEMService:
    """Mock DEM with 25% slope for E2E tests (between blue and red)."""
    return MockDEMService(base_elevation=3000.0, slope_ns_pct=25.0, slope_ew_pct=5.0)


@pytest.fixture
def apptest_graph() -> ResortGraph:
    """Fresh empty graph for E2E testing."""
    return ResortGraph()


@pytest.fixture
def apptest_factory(apptest_dem: MockDEMService) -> PathFactory:
    """PathFactory configured with mock DEM."""
    analyzer = TerrainAnalyzer(dem=apptest_dem)
    tracer = PathTracer(dem=apptest_dem, analyzer=analyzer)
    return PathFactory(dem_service=apptest_dem, path_tracer=tracer, terrain_analyzer=analyzer)


# =============================================================================
# COMMAND EXECUTOR - Simulates User Actions via Action Layer
# =============================================================================


def create_command_executor() -> None:
    """Streamlit app that executes commands from session_state.command_queue.

    This function runs inside AppTest and processes ONE command per at.run().

    COMMAND TYPES:
    Terrain/Marker clicks (via dispatch_click):
        ("click_terrain", lon, lat) → dispatch_click with terrain ClickInfo
        ("click_node", node_id) → dispatch_click with node marker ClickInfo
        ("click_slope", slope_id) → dispatch_click with slope marker ClickInfo
        ("click_lift", lift_id) → dispatch_click with lift marker ClickInfo

    Path operations (actions.py):
        ("commit_path", idx) → commit_selected_path(idx)
        ("handle_deferred",) → execute_deferred_actions() - TRUE E2E path gen!
        ("recompute_paths",) → recompute_paths()

    Slope operations (actions.py):
        ("finish_slope",) → finish_current_slope()
        ("cancel_slope",) → cancel_current_slope()

    Custom connect (actions.py):
        ("enter_custom",) → enter_custom_direction_mode()
        ("cancel_custom",) → cancel_custom_direction_mode()
        ("select_custom_target", lon, lat) → sm.select_custom_target()

    Delete operations (actions.py):
        ("delete_slope", slope_id) → delete_slope_action(slope_id)
        ("delete_lift", lift_id) → delete_lift_action(lift_id)

    Control operations:
        ("undo",) → undo_last_action()
        ("set_build_mode", mode) → ctx.build_mode.mode = mode
        ("close_panel",) → sm.send("close_panel")
        ("noop",) → do nothing

    HYBRID: Renders buttons that can be clicked via at.button().click()
    """
    import streamlit as st

    from skiresort_planner.model.click_info import ClickInfo, MapClickType, MarkerType
    from skiresort_planner.model.resort_graph import ResortGraph
    from skiresort_planner.ui.click_handlers import dispatch_click
    from skiresort_planner.ui.actions import (
        commit_selected_path,
        finish_current_slope,
        cancel_current_slope,
        cancel_custom_direction_mode,
        undo_last_action,
        enter_custom_direction_mode,
        handle_fast_deferred_actions,
        process_custom_connect_deferred,
        process_path_generation_deferred,
        recompute_paths,
        delete_slope_action,
        delete_lift_action,
    )
    from skiresort_planner.ui.state_machine import PlannerStateMachine
    from skiresort_planner.ui.context import PlannerContext, BuildMode
    from skiresort_planner.generators.path_factory import PathFactory
    from tests_workflow.conftest import MockDEMService

    def execute_deferred_actions() -> None:
        """Test helper: Execute all pending deferred actions.

        Mirrors app.py logic but without spinners for testing.
        """
        ctx: PlannerContext = st.session_state.context
        if ctx.deferred.custom_connect:
            process_custom_connect_deferred()
        elif ctx.deferred.path_generation:
            process_path_generation_deferred()
        else:
            handle_fast_deferred_actions()

    # Initialize state machine if not already done
    if "state_machine" not in st.session_state:
        graph: ResortGraph = st.session_state.graph
        sm, ctx = PlannerStateMachine.create(graph=graph, add_ui_listener=False)
        st.session_state.state_machine = sm
        st.session_state.context = ctx

    # Get references
    sm: PlannerStateMachine = st.session_state.state_machine
    ctx: PlannerContext = st.session_state.context
    graph: ResortGraph = st.session_state.graph
    dem: MockDEMService = st.session_state.dem_service

    # =========================================================================
    # COMMAND QUEUE EXECUTION - MUST RUN BEFORE UI RENDERING
    # =========================================================================
    # Commands are processed first so that build_mode changes take effect
    # before the radio widget tries to sync.
    command_queue: list = st.session_state.get("command_queue", [])
    executed: list = st.session_state.get("executed_commands", [])

    command_processed = False
    if command_queue:
        # Pop and execute ONE command
        cmd = command_queue.pop(0)
        cmd_type = cmd[0]
        executed.append(cmd)
        st.session_state.executed_commands = executed
        st.session_state.command_queue = command_queue
        command_processed = True

        # -------------------------------------------------------------------------
        # Terrain/Marker clicks (via dispatch_click)
        # -------------------------------------------------------------------------
        if cmd_type == "click_terrain":
            _, lon, lat = cmd
            click_info = ClickInfo(click_type=MapClickType.TERRAIN, lon=lon, lat=lat)
            dispatch_click(click_info=click_info)

        elif cmd_type == "click_node":
            _, node_id = cmd
            click_info = ClickInfo(
                click_type=MapClickType.MARKER,
                marker_type=MarkerType.NODE,
                node_id=node_id,
            )
            dispatch_click(click_info=click_info)

        elif cmd_type == "click_slope":
            _, slope_id = cmd
            click_info = ClickInfo(
                click_type=MapClickType.MARKER,
                marker_type=MarkerType.SLOPE,
                slope_id=slope_id,
            )
            dispatch_click(click_info=click_info)

        elif cmd_type == "click_lift":
            _, lift_id = cmd
            click_info = ClickInfo(
                click_type=MapClickType.MARKER,
                marker_type=MarkerType.LIFT,
                lift_id=lift_id,
            )
            dispatch_click(click_info=click_info)

        # -------------------------------------------------------------------------
        # Path operations (actions.py) - TRUE E2E!
        # -------------------------------------------------------------------------
        elif cmd_type == "commit_path":
            _, idx = cmd
            commit_selected_path(path_idx=idx)

        elif cmd_type == "handle_deferred":
            execute_deferred_actions()

        elif cmd_type == "recompute_paths":
            recompute_paths()

        # -------------------------------------------------------------------------
        # Slope operations (actions.py)
        # -------------------------------------------------------------------------
        elif cmd_type == "finish_slope":
            finish_current_slope()

        elif cmd_type == "cancel_slope":
            cancel_current_slope()

        # -------------------------------------------------------------------------
        # Custom connect (actions.py) - THE ENDBOSS LOGIC!
        # -------------------------------------------------------------------------
        elif cmd_type == "enter_custom":
            enter_custom_direction_mode()

        elif cmd_type == "cancel_custom":
            cancel_custom_direction_mode()

        elif cmd_type == "select_custom_target":
            _, lon, lat = cmd
            elevation = dem.get_elevation_or_raise(lon=lon, lat=lat)
            sm.select_custom_target(target_location=(lon, lat, elevation))

        # -------------------------------------------------------------------------
        # Delete operations (actions.py) - Uses new action functions!
        # -------------------------------------------------------------------------
        elif cmd_type == "delete_slope":
            _, slope_id = cmd
            delete_slope_action(slope_id=slope_id)

        elif cmd_type == "delete_lift":
            _, lift_id = cmd
            delete_lift_action(lift_id=lift_id)

        # -------------------------------------------------------------------------
        # Control operations
        # -------------------------------------------------------------------------
        elif cmd_type == "undo":
            undo_last_action()

        elif cmd_type == "set_build_mode":
            _, mode = cmd
            mode_constants = {
                "SLOPE": BuildMode.SLOPE,
                "CHAIRLIFT": BuildMode.CHAIRLIFT,
                "GONDOLA": BuildMode.GONDOLA,
                "SURFACE_LIFT": BuildMode.SURFACE_LIFT,
                "AERIAL_TRAM": BuildMode.AERIAL_TRAM,
            }
            if mode not in mode_constants:
                raise ValueError(f"Unknown build mode: {mode}")
            ctx.build_mode.mode = mode_constants[mode]

        elif cmd_type == "close_panel":
            sm.send("close_panel")

        elif cmd_type == "noop":
            pass

        else:
            raise ValueError(f"Unknown command type: {cmd_type}")

    # =========================================================================
    # HYBRID: Render buttons that AppTest can interact with
    # =========================================================================

    # Get current build mode for radio initial selection (AFTER command processing)
    current_mode_str = {
        BuildMode.SLOPE: "SLOPE",
        BuildMode.CHAIRLIFT: "CHAIRLIFT",
        BuildMode.GONDOLA: "GONDOLA",
    }.get(ctx.build_mode.mode, "SLOPE")
    mode_options = ["SLOPE", "CHAIRLIFT", "GONDOLA"]
    current_index = mode_options.index(current_mode_str) if current_mode_str in mode_options else 0

    with st.sidebar:
        # Build mode radio - index synced from context
        st.radio(
            "Build Mode",
            options=mode_options,
            key="rad_build_mode",
            index=current_index,
        )

        # Slope building buttons (only shown when building)
        if sm.is_any_slope_state:
            if st.button("Finish Slope", key="btn_finish_slope"):
                finish_current_slope()
            if st.button("Cancel Slope", key="btn_cancel_slope"):
                cancel_current_slope()
            if not ctx.custom_connect.enabled and not ctx.custom_connect.force_mode:
                if st.button("Custom Connect", key="btn_custom_connect"):
                    enter_custom_direction_mode()
            if ctx.custom_connect.enabled:
                if st.button("Cancel Custom", key="btn_cancel_custom"):
                    cancel_custom_direction_mode()

        # Undo button (always visible if undo stack)
        if graph.undo_stack:
            if st.button("Undo", key="btn_undo"):
                undo_last_action()

    # Sync build mode from radio button (radio → context) - ONLY if no command processed
    # This prevents radio overwriting command-set build mode
    if not command_processed:
        radio_mode = st.session_state.get("rad_build_mode", "SLOPE")
        mode_map = {
            "SLOPE": BuildMode.SLOPE,
            "CHAIRLIFT": BuildMode.CHAIRLIFT,
            "GONDOLA": BuildMode.GONDOLA,
        }
        if radio_mode in mode_map and ctx.build_mode.mode != mode_map[radio_mode]:
            ctx.build_mode.mode = mode_map[radio_mode]

    if not command_processed:
        st.write(f"State: {sm.get_state_name()}")
        st.write(f"Lifts: {len(graph.lifts)}, Slopes: {len(graph.slopes)}")

    st.session_state.command_queue = command_queue


# =============================================================================
# THE GRAND RESORT TOUR - ONE COMPREHENSIVE E2E TEST
# =============================================================================


@pytest.mark.apptest
class TestGrandResortTour:
    """THE comprehensive E2E test simulating complete resort building workflow."""

    def test_complete_resort_lifecycle(
        self,
        apptest_dem: MockDEMService,
        apptest_graph: ResortGraph,
        apptest_factory: PathFactory,
    ) -> None:
        """Build complete resort using ACTION LAYER functions.

        This test exercises:
        - dispatch_click() for terrain/node clicks (click_handlers.py)
        - execute_deferred_actions() for path generation (actions.py) - TRUE E2E!
        - commit_selected_path() for path commits (actions.py)
        - finish_current_slope() for finishing (actions.py)
        - undo_last_action() for undo (actions.py)
        - enter_custom_direction_mode() for custom connect (actions.py)
        - delete_slope_action/delete_lift_action for deletions (actions.py)

        PHASE 1: SLOPE_1 (Terrain → Terrain)
            - Click terrain to start
            - execute_deferred_actions() generates paths
            - Commit 3 path segments
            - Finish slope

        PHASE 2: LIFT_1 (Node → Terrain)
            - Set build mode to chairlift
            - Click on SLOPE_1 summit node
            - Click terrain for lift end

        PHASE 3: SLOPE_2 with Undo
            - Start new slope
            - Commit 2 segments
            - Undo last segment
            - Continue and finish

        PHASE 4: Delete and Undo
            - Delete SLOPE_2 via delete_slope_action()
            - Undo deletion
            - Delete LIFT_1 via delete_lift_action()
            - Undo deletion

        PHASE 5: CUSTOM CONNECT (The Endboss!)
            - Start new slope
            - Enter custom direction mode
            - Select target (downhill)
            - execute_deferred_actions() generates custom paths
            - Commit connector path → Auto-finish

        FINAL: Verify 1 lift, 3 slopes
        """
        # Create AppTest with command executor
        at = AppTest.from_function(create_command_executor, default_timeout=30)

        # Inject dependencies
        at.session_state["graph"] = apptest_graph
        at.session_state["dem_service"] = apptest_dem
        at.session_state["path_factory"] = apptest_factory
        at.session_state["map_version"] = 0
        at.session_state["command_queue"] = []
        at.session_state["executed_commands"] = []

        # Initialize - run once to set up state machine
        at.run()

        graph = at.session_state["graph"]

        # ================================================================
        # PHASE 1: SLOPE_1 (Terrain → Terrain)
        # ================================================================

        # Start: Click terrain to begin slope building
        start_lat = 0.0
        start_lon = 0.0
        at.session_state["command_queue"] = [("click_terrain", start_lon, start_lat)]
        at.run()

        # Verify we're in slope building state
        sm = at.session_state["state_machine"]
        assert sm.is_any_slope_state, f"Should be in slope state, got {sm.get_state_name()}"

        # TRUE E2E: Use execute_deferred_actions() instead of manual generation!
        at.session_state["command_queue"] = [("handle_deferred",)]
        at.run()

        ctx = at.session_state["context"]
        assert len(ctx.proposals.paths) > 0, "Should have path proposals"

        # Commit 3 segments - each commit triggers deferred path generation flag
        for _ in range(3):
            at.session_state["command_queue"] = [("commit_path", 0)]
            at.run()
            # Handle deferred actions (path generation)
            at.session_state["command_queue"] = [("handle_deferred",)]
            at.run()

        # Finish slope
        at.session_state["command_queue"] = [("finish_slope",)]
        at.run()

        # Verify slope created
        graph = at.session_state["graph"]
        assert len(graph.slopes) == 1, f"Expected 1 slope, got {len(graph.slopes)}"
        slope1_id = list(graph.slopes.keys())[0]
        slope1 = graph.slopes[slope1_id]

        # Get summit node for PHASE 2
        first_seg = graph.segments[slope1.segment_ids[0]]
        summit_node_id = first_seg.start_node_id

        # ================================================================
        # PHASE 2: LIFT_1 (Node → Terrain)
        # ================================================================

        # Close panel first
        at.session_state["command_queue"] = [("close_panel",)]
        at.run()

        # Set build mode to chairlift
        at.session_state["command_queue"] = [("set_build_mode", "CHAIRLIFT")]
        at.run()

        # Click summit node to start lift
        at.session_state["command_queue"] = [("click_node", summit_node_id)]
        at.run()

        sm = at.session_state["state_machine"]
        assert sm.is_lift_placing, f"Should be placing lift, got {sm.get_state_name()}"

        # Click terrain for lift end (higher elevation = north)
        lift_end_lat = 500 / M  # 500m north (higher)
        lift_end_lon = start_lon
        at.session_state["command_queue"] = [("click_terrain", lift_end_lon, lift_end_lat)]
        at.run()

        # Verify lift created
        graph = at.session_state["graph"]
        assert len(graph.lifts) == 1, f"Expected 1 lift, got {len(graph.lifts)}"
        lift1_id = list(graph.lifts.keys())[0]

        # ================================================================
        # PHASE 3: SLOPE_2 with Undo during Building
        # ================================================================

        # Close panel
        at.session_state["command_queue"] = [("close_panel",)]
        at.run()

        # Set build mode back to slope
        at.session_state["command_queue"] = [("set_build_mode", "SLOPE")]
        at.run()

        # Start new slope at different location
        slope2_start_lat = -500 / M
        slope2_start_lon = 0.01
        at.session_state["command_queue"] = [("click_terrain", slope2_start_lon, slope2_start_lat)]
        at.run()

        # Handle deferred path generation
        at.session_state["command_queue"] = [("handle_deferred",)]
        at.run()

        # Commit 2 segments
        for _ in range(2):
            at.session_state["command_queue"] = [("commit_path", 0)]
            at.run()
            at.session_state["command_queue"] = [("handle_deferred",)]
            at.run()

        ctx = at.session_state["context"]
        segments_before_undo = len(ctx.building.segments)
        assert segments_before_undo == 2, f"Should have 2 segments, got {segments_before_undo}"

        # UNDO last segment
        at.session_state["command_queue"] = [("undo",)]
        at.run()

        ctx = at.session_state["context"]
        segments_after_undo = len(ctx.building.segments)
        assert segments_after_undo == segments_before_undo - 1, (
            f"After undo: expected {segments_before_undo - 1} segments, got {segments_after_undo}"
        )

        # Continue building - commit 2 more segments
        for _ in range(2):
            at.session_state["command_queue"] = [("commit_path", 0)]
            at.run()
            at.session_state["command_queue"] = [("handle_deferred",)]
            at.run()

        # Finish slope 2
        at.session_state["command_queue"] = [("finish_slope",)]
        at.run()

        graph = at.session_state["graph"]
        assert len(graph.slopes) == 2, f"Expected 2 slopes, got {len(graph.slopes)}"
        slope2_id = [sid for sid in graph.slopes.keys() if sid != slope1_id][0]

        # ================================================================
        # PHASE 4: Delete and Undo Operations (Using action functions!)
        # ================================================================

        # Delete SLOPE_2 via delete_slope_action
        at.session_state["command_queue"] = [("delete_slope", slope2_id)]
        at.run()

        graph = at.session_state["graph"]
        assert slope2_id not in graph.slopes, "SLOPE_2 should be deleted"
        assert len(graph.slopes) == 1, f"Expected 1 slope after delete, got {len(graph.slopes)}"

        # Undo deletion
        at.session_state["command_queue"] = [("undo",)]
        at.run()

        graph = at.session_state["graph"]
        assert slope2_id in graph.slopes, "SLOPE_2 should be restored after undo"
        assert len(graph.slopes) == 2, f"Expected 2 slopes after undo, got {len(graph.slopes)}"

        # Delete LIFT_1 via delete_lift_action
        at.session_state["command_queue"] = [("delete_lift", lift1_id)]
        at.run()

        graph = at.session_state["graph"]
        assert lift1_id not in graph.lifts, "LIFT_1 should be deleted"
        assert len(graph.lifts) == 0, f"Expected 0 lifts after delete, got {len(graph.lifts)}"

        # Undo deletion
        at.session_state["command_queue"] = [("undo",)]
        at.run()

        graph = at.session_state["graph"]
        assert lift1_id in graph.lifts, "LIFT_1 should be restored after undo"
        assert len(graph.lifts) == 1, f"Expected 1 lift after undo, got {len(graph.lifts)}"

        # ================================================================
        # PHASE 5: CUSTOM CONNECT (The Endboss!)
        # ================================================================
        # This tests the most complex logic path in the app:
        # - Two-phase commit
        # - Temporary node creation
        # - Custom path generation to target
        # - Auto-finish on connector commit

        # Close panel first
        at.session_state["command_queue"] = [("close_panel",)]
        at.run()

        # Start new slope for custom connect test
        custom_start_lat = 1000 / M  # Start high (north)
        custom_start_lon = 0.02
        at.session_state["command_queue"] = [("click_terrain", custom_start_lon, custom_start_lat)]
        at.run()

        # Handle deferred path generation
        at.session_state["command_queue"] = [("handle_deferred",)]
        at.run()

        # Commit one segment first
        at.session_state["command_queue"] = [("commit_path", 0)]
        at.run()
        at.session_state["command_queue"] = [("handle_deferred",)]
        at.run()

        # Enter custom direction mode
        at.session_state["command_queue"] = [("enter_custom",)]
        at.run()

        ctx = at.session_state["context"]
        assert ctx.custom_connect.enabled, "Custom connect should be enabled"

        sm = at.session_state["state_machine"]
        assert sm.is_slope_custom_picking, f"Should be in custom picking state, got {sm.get_state_name()}"

        # Select custom target (downhill = south)
        custom_target_lat = 800 / M  # 200m south of start (lower)
        custom_target_lon = custom_start_lon + 0.001  # Slightly east
        at.session_state["command_queue"] = [("select_custom_target", custom_target_lon, custom_target_lat)]
        at.run()

        sm = at.session_state["state_machine"]
        assert sm.is_slope_custom_path, f"Should be in custom path state, got {sm.get_state_name()}"

        # Handle deferred custom connect path generation
        at.session_state["command_queue"] = [("handle_deferred",)]
        at.run()

        ctx = at.session_state["context"]
        assert len(ctx.proposals.paths) > 0, "Should have custom path proposals"

        # Commit the custom path - this should auto-finish if it's a connector
        # (connecting to an existing node) or continue building if not
        at.session_state["command_queue"] = [("commit_path", 0)]
        at.run()

        # Check if we're done or need to finish manually
        sm = at.session_state["state_machine"]
        if sm.is_any_slope_state:
            # Not a connector - finish manually
            at.session_state["command_queue"] = [("finish_slope",)]
            at.run()

        # Verify we have 3 slopes now
        graph = at.session_state["graph"]
        assert len(graph.slopes) == 3, f"Expected 3 slopes after custom connect, got {len(graph.slopes)}"

        # ================================================================
        # FINAL VALIDATION
        # ================================================================
        graph = at.session_state["graph"]
        assert len(graph.lifts) == 1, f"FINAL: Expected 1 lift, got {len(graph.lifts)}"
        assert len(graph.slopes) == 3, f"FINAL: Expected 3 slopes, got {len(graph.slopes)}"
        assert len(graph.nodes) > 0, "FINAL: Graph should have nodes"
        assert len(graph.segments) > 0, "FINAL: Graph should have segments"

        # Verify IDs are correct
        assert lift1_id in graph.lifts, f"FINAL: Lift {lift1_id} should exist"
        assert slope1_id in graph.slopes, f"FINAL: Slope {slope1_id} should exist"
        assert slope2_id in graph.slopes, f"FINAL: Slope {slope2_id} should exist"


@pytest.mark.apptest
class TestButtonInteractions:
    """Test real button clicks via AppTest for hybrid coverage."""

    def test_button_finish_slope(
        self,
        apptest_dem: MockDEMService,
        apptest_graph: ResortGraph,
        apptest_factory: PathFactory,
    ) -> None:
        """Test clicking finish button via AppTest."""
        at = AppTest.from_function(create_command_executor, default_timeout=30)

        # Inject dependencies
        at.session_state["graph"] = apptest_graph
        at.session_state["dem_service"] = apptest_dem
        at.session_state["path_factory"] = apptest_factory
        at.session_state["map_version"] = 0
        at.session_state["command_queue"] = []
        at.session_state["executed_commands"] = []

        at.run()

        # Start slope building
        at.session_state["command_queue"] = [("click_terrain", 0.0, 0.0)]
        at.run()
        at.session_state["command_queue"] = [("handle_deferred",)]
        at.run()

        # Commit one segment
        at.session_state["command_queue"] = [("commit_path", 0)]
        at.run()
        at.session_state["command_queue"] = [("handle_deferred",)]
        at.run()

        # Now click the finish button via AppTest!
        at.button(key="btn_finish_slope").click().run()

        # Verify slope was created
        graph = at.session_state["graph"]
        assert len(graph.slopes) == 1, "Should have 1 slope after button click"

    def test_button_undo(
        self,
        apptest_dem: MockDEMService,
        apptest_graph: ResortGraph,
        apptest_factory: PathFactory,
    ) -> None:
        """Test clicking undo button via AppTest."""
        at = AppTest.from_function(create_command_executor, default_timeout=30)

        at.session_state["graph"] = apptest_graph
        at.session_state["dem_service"] = apptest_dem
        at.session_state["path_factory"] = apptest_factory
        at.session_state["map_version"] = 0
        at.session_state["command_queue"] = []
        at.session_state["executed_commands"] = []

        at.run()

        # Build a slope
        at.session_state["command_queue"] = [("click_terrain", 0.0, 0.0)]
        at.run()
        at.session_state["command_queue"] = [("handle_deferred",)]
        at.run()
        at.session_state["command_queue"] = [("commit_path", 0)]
        at.run()
        at.session_state["command_queue"] = [("handle_deferred",)]
        at.run()
        at.session_state["command_queue"] = [("finish_slope",)]
        at.run()

        graph = at.session_state["graph"]
        assert len(graph.slopes) == 1, "Should have 1 slope before undo"

        # Click undo button - this undoes the finish_slope
        at.button(key="btn_undo").click().run()

        # We should be back in building state
        sm = at.session_state["state_machine"]
        assert sm.is_any_slope_state, f"Should be in slope state after undo, got {sm.get_state_name()}"

    def test_radio_build_mode(
        self,
        apptest_dem: MockDEMService,
        apptest_graph: ResortGraph,
        apptest_factory: PathFactory,
    ) -> None:
        """Test changing build mode via radio button."""
        at = AppTest.from_function(create_command_executor, default_timeout=30)

        at.session_state["graph"] = apptest_graph
        at.session_state["dem_service"] = apptest_dem
        at.session_state["path_factory"] = apptest_factory
        at.session_state["map_version"] = 0
        at.session_state["command_queue"] = []
        at.session_state["executed_commands"] = []

        at.run()

        # Change to CHAIRLIFT via radio button
        at.sidebar.radio(key="rad_build_mode").set_value("CHAIRLIFT").run()

        ctx = at.session_state["context"]
        from skiresort_planner.ui.context import BuildMode

        assert ctx.build_mode.mode == BuildMode.CHAIRLIFT, "Build mode should be CHAIRLIFT"


@pytest.mark.apptest
class TestCancelOperations:
    """Test cancel operations for better coverage."""

    def test_cancel_slope(
        self,
        apptest_dem: MockDEMService,
        apptest_graph: ResortGraph,
        apptest_factory: PathFactory,
    ) -> None:
        """Test canceling slope building."""
        at = AppTest.from_function(create_command_executor, default_timeout=30)

        at.session_state["graph"] = apptest_graph
        at.session_state["dem_service"] = apptest_dem
        at.session_state["path_factory"] = apptest_factory
        at.session_state["map_version"] = 0
        at.session_state["command_queue"] = []
        at.session_state["executed_commands"] = []

        at.run()

        # Start slope and commit a segment
        at.session_state["command_queue"] = [("click_terrain", 0.0, 0.0)]
        at.run()
        at.session_state["command_queue"] = [("handle_deferred",)]
        at.run()
        at.session_state["command_queue"] = [("commit_path", 0)]
        at.run()

        # Cancel slope
        at.session_state["command_queue"] = [("cancel_slope",)]
        at.run()

        sm = at.session_state["state_machine"]
        assert sm.is_idle, f"Should be idle after cancel, got {sm.get_state_name()}"

        graph = at.session_state["graph"]
        assert len(graph.slopes) == 0, "No slopes should exist after cancel"

    def test_cancel_custom_direction(
        self,
        apptest_dem: MockDEMService,
        apptest_graph: ResortGraph,
        apptest_factory: PathFactory,
    ) -> None:
        """Test canceling custom direction mode."""
        at = AppTest.from_function(create_command_executor, default_timeout=30)

        at.session_state["graph"] = apptest_graph
        at.session_state["dem_service"] = apptest_dem
        at.session_state["path_factory"] = apptest_factory
        at.session_state["map_version"] = 0
        at.session_state["command_queue"] = []
        at.session_state["executed_commands"] = []

        at.run()

        # Start slope
        at.session_state["command_queue"] = [("click_terrain", 0.0, 0.0)]
        at.run()
        at.session_state["command_queue"] = [("handle_deferred",)]
        at.run()

        # Enter custom mode
        at.session_state["command_queue"] = [("enter_custom",)]
        at.run()

        ctx = at.session_state["context"]
        assert ctx.custom_connect.enabled, "Custom connect should be enabled"

        # Cancel custom mode
        at.session_state["command_queue"] = [("cancel_custom",)]
        at.run()

        ctx = at.session_state["context"]
        assert not ctx.custom_connect.enabled, "Custom connect should be disabled after cancel"

        sm = at.session_state["state_machine"]
        assert sm.is_any_slope_state, f"Should still be in slope state, got {sm.get_state_name()}"

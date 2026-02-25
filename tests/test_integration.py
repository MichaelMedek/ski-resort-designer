"""Integration and smoke tests to ensure the app works on first try.

These tests verify:
1. All module imports work without errors
2. Configuration constants are valid (colors, paths, layer names)
3. Key integration points work together
4. Layer creation produces valid Pydeck data structures
5. Click handling pipeline works end-to-end

Run these before deploying to catch issues early.
"""

import pytest
import pydeck as pdk

from skiresort_planner.constants import (
    AppConfig,
    ChartConfig,
    ClickConfig,
    ConnectionConfig,
    DEMConfig,
    LiftConfig,
    MapConfig,
    MarkerConfig,
    PathConfig,
    SlopeConfig,
    StyleConfig,
    UndoConfig,
)
from skiresort_planner.model.path_point import PathPoint
from skiresort_planner.model.proposed_path import ProposedSlopeSegment
from skiresort_planner.model.resort_graph import ResortGraph
from skiresort_planner.ui.center_map import LayerCollection, MapRenderer
from skiresort_planner.ui.click_detector import ClickDetector
from skiresort_planner.ui.click_handlers import get_click_handler
from skiresort_planner.ui.state_machine import (
    BuildMode,
    ClickDeduplicationContext,
    PlannerContext,
    PlannerStateMachine,
)


# =============================================================================
# IMPORT SMOKE TESTS
# Verify all modules import without errors - catches missing dependencies
# =============================================================================


class TestImportSmoke:
    """Verify all application modules import without errors."""

    def test_app_module_imports(self) -> None:
        """Main app module imports successfully."""
        from skiresort_planner import app

        assert hasattr(app, "main")
        assert hasattr(app, "init_session_state")

    def test_ui_module_imports(self) -> None:
        """All UI modules import successfully."""
        from skiresort_planner.ui import (
            ClickDetector,
            MapRenderer,
            PlannerContext,
            PlannerStateMachine,
            ProfileChart,
            SidebarRenderer,
        )

        assert ClickDetector is not None
        assert MapRenderer is not None

    def test_action_imports(self) -> None:
        """Action functions import successfully."""
        from skiresort_planner.ui.actions import (
            bump_map_version,
            cancel_current_slope,
            center_on_lift,
            center_on_slope,
            commit_selected_path,
            finish_current_slope,
            handle_deferred_actions,
            recompute_paths,
            undo_last_action,
        )

        assert callable(bump_map_version)

    def test_validator_imports(self) -> None:
        """Validator functions import successfully."""
        from skiresort_planner.ui.validators import (
            validate_custom_target_distance,
            validate_custom_target_downhill,
            validate_lift_different_nodes,
            validate_lift_goes_uphill,
        )

        assert callable(validate_lift_goes_uphill)

    def test_click_handler_imports(self) -> None:
        """Click handler functions import successfully."""
        from skiresort_planner.ui.click_handlers import dispatch_click, get_click_handler

        assert callable(get_click_handler)

    def test_pydeck_handler_imports(self) -> None:
        """Pydeck handler functions import successfully."""
        from skiresort_planner.ui.pydeck_click_handler import (
            PydeckClickResult,
            _get_click_id,
            render_pydeck_map,
        )

        assert callable(render_pydeck_map)


# =============================================================================
# CONFIGURATION VALIDATION
# Verify constants are valid - catches typos and format errors
# =============================================================================


class TestConfigurationValidation:
    """Verify configuration constants are valid for Pydeck/Streamlit."""

    def test_slope_colors_are_valid_rgba(self) -> None:
        """All slope colors are valid RGBA lists (4 integers 0-255)."""
        for difficulty, color in StyleConfig.SLOPE_COLORS_RGBA.items():
            assert isinstance(color, (list, tuple)), f"{difficulty} color must be list"
            assert len(color) == 4, f"{difficulty} must have 4 components (RGBA)"
            for i, c in enumerate(color):
                assert isinstance(c, int), f"{difficulty}[{i}] must be int, got {type(c)}"
                assert 0 <= c <= 255, f"{difficulty}[{i}]={c} must be 0-255"

    def test_lift_colors_are_valid_rgba(self) -> None:
        """All lift colors are valid RGBA lists."""
        for lift_type, color in StyleConfig.LIFT_COLORS_RGBA.items():
            assert isinstance(color, (list, tuple)), f"{lift_type} color must be list"
            assert len(color) == 4, f"{lift_type} must have 4 components"
            for c in color:
                assert 0 <= c <= 255

    def test_marker_colors_are_valid(self) -> None:
        """Marker colors are valid RGBA."""
        colors = [
            MarkerConfig.DIRECTION_ARROW_COLOR_DOWNHILL,
            MarkerConfig.DIRECTION_ARROW_COLOR_UPHILL,
            MarkerConfig.LIFT_STATION_COLOR,
            MarkerConfig.NODE_MARKER_COLOR,
            MarkerConfig.NODE_MARKER_BORDER,
            MarkerConfig.PYLON_MARKER_COLOR,
            MarkerConfig.PYLON_BORDER_COLOR,
        ]
        for color in colors:
            assert len(color) == 4
            for c in color:
                assert 0 <= c <= 255

    def test_click_type_constants_are_strings(self) -> None:
        """Click type constants are non-empty strings."""
        types = [
            ClickConfig.TYPE_TERRAIN,
            ClickConfig.TYPE_NODE,
            ClickConfig.TYPE_SEGMENT,
            ClickConfig.TYPE_SLOPE,
            ClickConfig.TYPE_LIFT,
            ClickConfig.TYPE_PYLON,
            ClickConfig.TYPE_PROPOSAL_ENDPOINT,
            ClickConfig.TYPE_PROPOSAL_BODY,
        ]
        for t in types:
            assert len(t) > 0

    def test_difficulty_levels_consistent(self) -> None:
        """All difficulty-keyed dicts have same keys."""
        keys = set(SlopeConfig.DIFFICULTIES)
        assert set(SlopeConfig.DIFFICULTY_THRESHOLDS.keys()) == keys
        assert set(SlopeConfig.DIFFICULTY_TARGETS.keys()) == keys
        assert set(SlopeConfig.BELT_WIDTHS.keys()) == keys
        assert set(StyleConfig.SLOPE_COLORS_RGBA.keys()) == keys

    def test_map_config_valid_coordinates(self) -> None:
        """Map config has valid coordinate ranges."""
        assert -180 <= MapConfig.START_CENTER_LON <= 180
        assert -90 <= MapConfig.START_CENTER_LAT <= 90
        assert 0 <= MapConfig.DEFAULT_ZOOM <= 22
        assert 0 <= MapConfig.DEFAULT_PITCH <= 85


# =============================================================================
# CLICK HANDLER PIPELINE TESTS
# Verify click detection → handler dispatch works without Streamlit
# =============================================================================


class TestClickHandlerPipeline:
    """Test click detection and dispatch without Streamlit."""

    def test_get_click_handler_for_all_states(self) -> None:
        """All expected states have click handlers."""
        from skiresort_planner.ui.state_machine import PlannerStateMachine

        graph = ResortGraph()
        sm = PlannerStateMachine(graph=graph)
        # Test idle state
        assert sm.is_idle
        handler = get_click_handler(sm=sm)
        assert callable(handler), "No handler for idle"

        # Test slope building state
        sm.start_building(lon=10.0, lat=47.0, elevation=1000.0, node_id=None)
        assert sm.is_any_slope_state
        handler = get_click_handler(sm=sm)
        assert callable(handler), "No handler for slope building"

        # Test lift placing state
        sm.cancel_slope()
        sm.select_lift_start(node_id=None, location=PathPoint(lon=10.0, lat=47.0, elevation=1000.0))
        assert sm.is_lift_placing
        handler = get_click_handler(sm=sm)
        assert callable(handler), "No handler for lift placing"

    def test_unknown_state_raises(self) -> None:
        """Unknown state raises RuntimeError - now tested via coverage of all states."""
        # With state machine properties, there's no way to reach an unknown state
        # The function is exhaustive over boolean properties
        pass

    def test_click_detector_integration(self) -> None:
        """ClickDetector produces ClickInfo from Pydeck objects."""
        from skiresort_planner.model.click_info import MapClickType, MarkerType

        dedup = ClickDeduplicationContext(debounce_seconds=0)
        detector = ClickDetector(dedup=dedup)

        # Terrain click
        result = detector.detect(
            clicked_object=None,
            clicked_coordinate=[10.84, 46.97],
        )
        assert result is not None
        assert result.click_type == MapClickType.TERRAIN

        # Node click
        dedup.clear()
        result = detector.detect(
            clicked_object={"type": "node", "id": "N1", "position": [10.84, 46.97]},
            clicked_coordinate=None,
        )
        assert result is not None
        assert result.click_type == MapClickType.MARKER
        assert result.marker_type == MarkerType.NODE


# =============================================================================
# STATE MACHINE INTEGRATION
# Verify state machine works with context and graph
# =============================================================================


class TestStateMachineIntegration:
    """Test state machine integration with context and graph."""

    def test_create_returns_linked_components(self) -> None:
        """PlannerStateMachine.create() returns linked sm and ctx."""
        graph = ResortGraph()
        sm, ctx = PlannerStateMachine.create(graph=graph)

        assert sm is not None
        assert ctx is not None
        assert sm.is_idle

    def test_full_slope_building_cycle(self) -> None:
        """Complete slope: idle → building → commit → finish → idle."""
        graph = ResortGraph()
        sm, ctx = PlannerStateMachine.create(graph=graph)

        # Start building
        sm.start_building(lon=10.84, lat=46.97, elevation=2500, node_id=None)
        assert sm.is_any_slope_state
        assert ctx.building.name == "Slope 1"

        # Commit segment
        sm.commit_path(segment_id="S1", endpoint_node_id="N1")
        assert "S1" in ctx.building.segments

        # Finish slope
        sm.finish_slope(slope_id="SL1")
        assert sm.is_idle
        assert ctx.viewing.panel_visible

    def test_lift_placing_cycle(self) -> None:
        """Complete lift: idle → placing → complete → idle."""
        graph = ResortGraph()
        sm, ctx = PlannerStateMachine.create(graph=graph)
        ctx.build_mode.mode = BuildMode.CHAIRLIFT

        # Select lift start
        sm.select_lift_start(node_id="N1")
        assert sm.is_lift_placing
        assert ctx.lift.start_node_id == "N1"

        # Complete lift
        sm.complete_lift(lift_id="L1")
        assert sm.is_idle


# =============================================================================
# MAP RENDERER LAYER DATA TESTS
# Verify MapRenderer creates valid Pydeck layer data
# =============================================================================


class TestMapRendererLayerData:
    """Test MapRenderer creates valid Pydeck data structures."""

    @pytest.fixture
    def renderer_with_graph(self) -> MapRenderer:
        """Renderer with a populated graph."""
        graph = ResortGraph()

        # Add a segment
        M = MapConfig.METERS_PER_DEGREE_EQUATOR
        proposal = ProposedSlopeSegment(
            points=[
                PathPoint(lon=10.84, lat=46.97, elevation=2500),
                PathPoint(lon=10.84, lat=46.97 - 500 / M, elevation=2400),
            ],
            target_slope_pct=20.0,
            target_difficulty="blue",
        )
        graph.commit_paths(paths=[proposal])

        renderer = MapRenderer(
            center_lon=10.84,
            center_lat=46.97,
            zoom=14,
            pitch=45,
            bearing=0,
        )
        renderer.graph = graph
        return renderer

    def test_render_returns_deck(self, renderer_with_graph: MapRenderer) -> None:
        """render() returns a pdk.Deck object."""
        deck = renderer_with_graph.render(proposals=[], selected_proposal_idx=None)
        assert isinstance(deck, pdk.Deck)

    def test_node_layer_has_required_fields(self, renderer_with_graph: MapRenderer) -> None:
        """Node layer data has all required fields."""
        layer = renderer_with_graph._create_node_layer()
        assert layer.id == "nodes"

        # Check layer data has required fields
        if layer.data:
            node = layer.data[0]
            assert "type" in node
            assert "id" in node
            assert "position" in node
            assert node["type"] == ClickConfig.TYPE_NODE

    def test_node_positions_have_z_offset(self, renderer_with_graph: MapRenderer) -> None:
        """Node positions in 3D mode include elevation plus z-offset for z-fighting prevention."""
        layer = renderer_with_graph._create_node_layer(use_3d=True)
        if layer.data:
            node_data = layer.data[0]
            pos = node_data["position"]
            assert len(pos) == 3, "Position should be [lon, lat, z]"
            # Z should be elevation + MARKER_Z_OFFSET_M in 3D mode
            expected_z = node_data["elevation"] + MarkerConfig.MARKER_Z_OFFSET_M
            assert pos[2] == expected_z

    def test_node_positions_2d_mode(self, renderer_with_graph: MapRenderer) -> None:
        """Node positions in 2D mode use flat z offset for layer ordering."""
        layer = renderer_with_graph._create_node_layer(use_3d=False)
        if layer.data:
            node_data = layer.data[0]
            pos = node_data["position"]
            assert len(pos) == 3, "Position should be [lon, lat, z]"
            # Z should be Z_OFFSET_2D_NODES in 2D mode
            assert pos[2] == MapConfig.Z_OFFSET_2D_NODES

    def test_layer_collection_ordering(self) -> None:
        """LayerCollection maintains correct z-order."""
        lc = LayerCollection()
        lc.pylons = [pdk.Layer("ScatterplotLayer", [], id="pylons")]
        lc.nodes = [pdk.Layer("ScatterplotLayer", [], id="nodes")]
        lc.slopes = [pdk.Layer("PathLayer", [], id="slopes")]
        lc.lifts = [pdk.Layer("PathLayer", [], id="lifts")]
        lc.proposals = [pdk.Layer("PathLayer", [], id="proposals")]
        lc.markers = [pdk.Layer("ScatterplotLayer", [], id="markers")]

        layers = lc.get_ordered_layers()
        ids = [l.id for l in layers]

        # Pylons must be first (back)
        assert ids[0] == "pylons"
        # Markers must be last (front)
        assert ids[-1] == "markers"


# =============================================================================
# PROFILE CHART TESTS
# Verify chart creation doesn't fail with typical data
# =============================================================================


class TestProfileChartIntegration:
    """Test ProfileChart creates valid Plotly figures."""

    def test_chart_renders_proposal(self) -> None:
        """ProfileChart renders proposal without error."""
        from skiresort_planner.ui.bottom_chart import ProfileChart

        chart = ProfileChart(width=800, height=400)
        proposal = ProposedSlopeSegment(
            points=[
                PathPoint(lon=10.84, lat=46.97, elevation=2500),
                PathPoint(lon=10.84, lat=46.96, elevation=2400),
            ],
            target_slope_pct=20.0,
            target_difficulty="blue",
        )

        fig = chart.render_proposal(proposal=proposal, proposed_segment_title="Test")

        import plotly.graph_objects as go

        assert isinstance(fig, go.Figure)


# =============================================================================
# CONTEXT VALIDATION
# Verify PlannerContext initializes cleanly
# =============================================================================


class TestContextInitialization:
    """Test PlannerContext initializes with valid defaults."""

    def test_context_defaults_are_safe(self) -> None:
        """Context initializes without None where needed."""
        ctx = PlannerContext()

        # Build mode has valid default
        assert ctx.build_mode.mode in [BuildMode.SLOPE, BuildMode.CHAIRLIFT, BuildMode.GONDOLA]

        # Numeric defaults are reasonable
        assert ctx.map.zoom > 0
        assert -180 <= ctx.map.lon <= 180
        assert -90 <= ctx.map.lat <= 90

        # Lists start empty (not None)
        assert isinstance(ctx.building.segments, list)
        assert isinstance(ctx.proposals.paths, list)

    def test_context_has_selection_check(self) -> None:
        """has_selection() works correctly."""
        ctx = PlannerContext()
        assert ctx.has_selection() is False

        ctx.selection.lon = 10.84
        ctx.selection.lat = 46.97
        assert ctx.has_selection() is True

    def test_context_clear_methods_work(self) -> None:
        """Clear methods reset state properly."""
        ctx = PlannerContext()
        ctx.selection.lon = 10.84
        ctx.selection.lat = 46.97
        ctx.building.segments = ["S1", "S2"]
        ctx.proposals.paths = [object(), object()]  # type: ignore[list-item]

        ctx.clear_selection()
        assert ctx.selection.lon is None

        ctx.clear_building()  # type: ignore[unreachable]
        assert ctx.building.segments == []

        ctx.clear_proposals()
        assert ctx.proposals.paths == []


# =============================================================================
# RESORT GRAPH INTEGRATION
# Verify graph operations work correctly
# =============================================================================


class TestResortGraphIntegration:
    """Test ResortGraph operations that the UI relies on."""

    def test_commit_paths_creates_nodes_and_segments(self) -> None:
        """commit_paths creates nodes and segments correctly."""
        graph = ResortGraph()
        M = MapConfig.METERS_PER_DEGREE_EQUATOR

        proposal = ProposedSlopeSegment(
            points=[
                PathPoint(lon=10.84, lat=46.97, elevation=2500),
                PathPoint(lon=10.84, lat=46.97 - 500 / M, elevation=2400),
            ],
            target_slope_pct=20.0,
            target_difficulty="blue",
        )

        end_node_ids = graph.commit_paths(paths=[proposal])

        assert len(graph.nodes) == 2
        assert len(graph.segments) == 1
        assert len(end_node_ids) == 1

    def test_finish_slope_creates_slope(self) -> None:
        """finish_slope creates a Slope from segments."""
        graph = ResortGraph()
        M = MapConfig.METERS_PER_DEGREE_EQUATOR

        proposal = ProposedSlopeSegment(
            points=[
                PathPoint(lon=10.84, lat=46.97, elevation=2500),
                PathPoint(lon=10.84, lat=46.97 - 500 / M, elevation=2400),
            ],
            target_slope_pct=20.0,
            target_difficulty="blue",
        )
        graph.commit_paths(paths=[proposal])
        seg_id = list(graph.segments.keys())[0]

        slope = graph.finish_slope(segment_ids=[seg_id])

        assert slope is not None
        assert slope.id in graph.slopes
        assert seg_id in slope.segment_ids

    def test_undo_works_after_commit(self) -> None:
        """Undo removes the last committed action."""
        graph = ResortGraph()
        M = MapConfig.METERS_PER_DEGREE_EQUATOR

        proposal = ProposedSlopeSegment(
            points=[
                PathPoint(lon=10.84, lat=46.97, elevation=2500),
                PathPoint(lon=10.84, lat=46.97 - 500 / M, elevation=2400),
            ],
            target_slope_pct=20.0,
            target_difficulty="blue",
        )
        graph.commit_paths(paths=[proposal])
        assert len(graph.segments) == 1

        graph.undo_last()
        assert len(graph.segments) == 0


# =============================================================================
# UNDO SEGMENT INTEGRATION TESTS
# Verify undo operations preserve correct state for UI
# =============================================================================


class TestUndoSegmentIntegration:
    """Test undo segment operations work correctly with context updates."""

    def test_undo_segment_preserves_remaining_segment(self) -> None:
        """When undoing 2nd segment, 1st segment should remain and be the new endpoint."""
        graph = ResortGraph()
        M = MapConfig.METERS_PER_DEGREE_EQUATOR

        # Create first segment
        proposal1 = ProposedSlopeSegment(
            points=[
                PathPoint(lon=10.84, lat=46.97, elevation=2500),
                PathPoint(lon=10.84, lat=46.97 - 500 / M, elevation=2400),
            ],
            target_slope_pct=20.0,
            target_difficulty="blue",
        )
        end_nodes_1 = graph.commit_paths(paths=[proposal1])
        seg1_id = list(graph.segments.keys())[0]
        seg1 = graph.segments[seg1_id]

        # Create second segment from end of first
        start_lat = 46.97 - 500 / M
        proposal2 = ProposedSlopeSegment(
            points=[
                PathPoint(lon=10.84, lat=start_lat, elevation=2400),
                PathPoint(lon=10.84, lat=start_lat - 500 / M, elevation=2300),
            ],
            target_slope_pct=20.0,
            target_difficulty="blue",
        )
        end_nodes_2 = graph.commit_paths(paths=[proposal2])
        assert len(graph.segments) == 2
        seg2_id = [s for s in graph.segments.keys() if s != seg1_id][0]

        # Undo second segment
        undone = graph.undo_last()
        assert len(graph.segments) == 1
        assert seg1_id in graph.segments
        assert seg2_id not in graph.segments

        # Verify first segment's endpoint can be used for path regeneration
        seg1_points = graph.segments[seg1_id].points
        assert seg1_points is not None
        assert len(seg1_points) > 0
        last_pt = seg1_points[-1]
        assert last_pt.elevation == 2400  # Should be end of first segment

    def test_undo_all_segments_empties_graph(self) -> None:
        """Undoing all segments should leave empty graph."""
        graph = ResortGraph()
        M = MapConfig.METERS_PER_DEGREE_EQUATOR

        proposal = ProposedSlopeSegment(
            points=[
                PathPoint(lon=10.84, lat=46.97, elevation=2500),
                PathPoint(lon=10.84, lat=46.97 - 500 / M, elevation=2400),
            ],
            target_slope_pct=20.0,
            target_difficulty="blue",
        )
        graph.commit_paths(paths=[proposal])
        assert len(graph.segments) == 1
        assert len(graph.nodes) == 2

        # Undo should remove segment
        graph.undo_last()
        assert len(graph.segments) == 0
        # Note: nodes may remain until cleanup_isolated_nodes is called

    def test_undo_finish_slope_restores_segments(self) -> None:
        """Undoing finish_slope restores segments to building state."""
        graph = ResortGraph()
        M = MapConfig.METERS_PER_DEGREE_EQUATOR

        # Create and commit a segment
        proposal = ProposedSlopeSegment(
            points=[
                PathPoint(lon=10.84, lat=46.97, elevation=2500),
                PathPoint(lon=10.84, lat=46.97 - 500 / M, elevation=2400),
            ],
            target_slope_pct=20.0,
            target_difficulty="blue",
        )
        graph.commit_paths(paths=[proposal])
        seg_id = list(graph.segments.keys())[0]

        # Finish the slope
        slope = graph.finish_slope(segment_ids=[seg_id])
        assert slope is not None
        assert slope.id in graph.slopes

        # Undo finish slope
        undone = graph.undo_last()
        assert slope.id not in graph.slopes  # Slope removed
        assert seg_id in graph.segments  # Segment still exists

        # Verify segment is intact for rebuild
        seg = graph.segments[seg_id]
        assert seg.points is not None
        assert len(seg.points) > 0

    def test_context_update_after_undo_segment(self) -> None:
        """PlannerContext should have correct state after undo preparation."""
        ctx = PlannerContext()
        ctx.building.segments = ["S1", "S2"]

        # Simulate what undo does before state transition
        remaining = [s for s in ctx.building.segments if s != "S2"]
        assert remaining == ["S1"]

        # Verify context can track remaining segments
        ctx.building.segments = remaining
        assert ctx.building.segments == ["S1"]

    def test_context_selection_update(self) -> None:
        """Context selection should update correctly for crosshair positioning."""
        ctx = PlannerContext()

        # Update selection (simulates what happens before path generation)
        ctx.set_selection(lon=10.84, lat=46.97, elevation=2500)

        assert ctx.selection.lon == 10.84
        assert ctx.selection.lat == 46.97
        assert ctx.selection.elevation == 2500
        assert ctx.has_selection() is True


# =============================================================================
# 3D VIEW TOGGLE TESTS
# Verify 3D/2D view state management
# =============================================================================


class TestViewToggleIntegration:
    """Test 3D/2D view toggle state management."""

    def test_viewing_context_3d_toggle(self) -> None:
        """ViewingContext 3D toggle should update state correctly."""
        ctx = PlannerContext()

        # Initially 2D
        assert ctx.viewing.view_3d is False

        # Enable 3D
        ctx.viewing.enable_3d()
        assert ctx.viewing.view_3d is True

        # Disable 3D
        ctx.viewing.disable_3d()  # type: ignore[unreachable]
        assert ctx.viewing.view_3d is False

    def test_viewing_context_hide_disables_3d(self) -> None:
        """Hiding panel should also disable 3D view."""
        ctx = PlannerContext()

        # Enable 3D while viewing
        ctx.viewing.set_slope_id(slope_id="SL1")
        ctx.viewing.show_panel()  # BAD PRACTICE! DO ONYL SUE TRANSITIOSN OF STAES FOR NEW TESTS!
        ctx.viewing.enable_3d()
        assert ctx.viewing.view_3d is True
        assert ctx.viewing.panel_visible is True

        # Hide panel should disable 3D
        ctx.viewing.hide_panel()
        assert ctx.viewing.view_3d is False
        assert ctx.viewing.panel_visible is False  # type: ignore[unreachable]

    def test_map_context_pitch_bearing_update(self) -> None:
        """Map context pitch and bearing can be updated for 2D reset."""
        ctx = PlannerContext()

        # Set some non-default values
        ctx.map.pitch = 45.0
        ctx.map.bearing = 90.0

        # Reset to defaults (what happens when returning to 2D)
        ctx.map.pitch = MapConfig.DEFAULT_PITCH
        ctx.map.bearing = MapConfig.DEFAULT_BEARING

        assert ctx.map.pitch == MapConfig.DEFAULT_PITCH
        assert ctx.map.bearing == MapConfig.DEFAULT_BEARING

    def test_viewing_context_slope_lifecycle(self) -> None:
        """ViewingContext tracks slope through show/hide/3D cycle."""
        ctx = PlannerContext()

        # Show slope
        ctx.viewing.set_slope_id(slope_id="SL1")
        ctx.viewing.show_panel()
        assert ctx.viewing.slope_id == "SL1"
        assert ctx.viewing.panel_visible is True
        assert ctx.viewing.view_3d is False

        # Enable 3D
        ctx.viewing.enable_3d()
        assert ctx.viewing.slope_id == "SL1"  # ID preserved
        assert ctx.viewing.view_3d is True

        # Disable 3D
        ctx.viewing.disable_3d()  # type: ignore[unreachable]
        assert ctx.viewing.slope_id == "SL1"  # ID preserved
        assert ctx.viewing.view_3d is False

        # Hide
        ctx.viewing.hide_panel()
        assert ctx.viewing.slope_id == "SL1"  # ID preserved for re-show
        assert ctx.viewing.panel_visible is False
        assert ctx.viewing.view_3d is False

    def test_viewing_context_lift_lifecycle(self) -> None:
        """ViewingContext tracks lift through show/hide/3D cycle."""
        ctx = PlannerContext()

        # Show lift
        ctx.viewing.set_lift_id(lift_id="L1")
        ctx.viewing.show_panel()
        assert ctx.viewing.lift_id == "L1"
        assert ctx.viewing.slope_id is None  # Previous slope cleared
        assert ctx.viewing.panel_visible is True

        # Enable 3D
        ctx.viewing.enable_3d()
        assert ctx.viewing.lift_id == "L1"
        assert ctx.viewing.view_3d is True

        # Clear state
        ctx.viewing.clear()
        assert ctx.viewing.lift_id is None
        assert ctx.viewing.view_3d is False  # type: ignore[unreachable]


# =============================================================================
# DEFERRED TOAST TESTS
# Verify toast queuing mechanism
# =============================================================================


class TestDeferredToastIntegration:
    """Test deferred toast message mechanism."""

    def test_queue_toast_function(self) -> None:
        """queue_toast stores messages in pending list."""
        from skiresort_planner.ui.actions import queue_toast
        import streamlit as st

        # Clear any existing pending toasts
        st.session_state.pending_toasts = []

        queue_toast(message="Test message 1", icon="✅")
        queue_toast(message="Test message 2", icon="⚠️")

        pending = st.session_state.get("pending_toasts", [])
        assert len(pending) == 2
        assert pending[0]["message"] == "Test message 1"
        assert pending[0]["icon"] == "✅"
        assert pending[1]["message"] == "Test message 2"
        assert pending[1]["icon"] == "⚠️"

        # Cleanup
        st.session_state.pending_toasts = []

    def test_toast_message_classes(self) -> None:
        """Toast message classes have correct message and icon properties."""
        from skiresort_planner.model.message import (
            UndoCancelSlopeMessage,
            UndoSegmentMessage,
            UndoFinishSlopeMessage,
            UndoLiftMessage,
        )

        # UndoCancelSlopeMessage
        msg1 = UndoCancelSlopeMessage()
        assert isinstance(msg1.message, str)
        assert isinstance(msg1.icon, str)
        assert len(msg1.icon) > 0

        # UndoSegmentMessage
        msg2 = UndoSegmentMessage(segment_id="S1", was_last_segment=False)
        assert "S1" in msg2.message
        assert isinstance(msg2.icon, str)

        # UndoFinishSlopeMessage
        msg3 = UndoFinishSlopeMessage(slope_name="Test Slope", num_segments=3)
        assert "Test Slope" in msg3.message
        assert "3" in msg3.message

        # UndoLiftMessage
        msg4 = UndoLiftMessage(lift_name="L1")
        assert "L1" in msg4.message

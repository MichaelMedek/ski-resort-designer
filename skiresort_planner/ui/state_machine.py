"""State machine for ski resort planner UI.

Uses python-statemachine for robust state management with:
- Clear state definitions
- Guarded transitions (conditions)
- Entry/exit hooks for side effects
- Explicit event-driven transitions

Architecture Overview
---------------------
This module implements a UI state machine integrated with Streamlit's reactive model.
The key pattern is:

1. User action triggers state transition (e.g., click map → start_building)
2. StreamlitUIListener fires after_transition and calls graph.perform_cleanup() + st.rerun()
3. On the next render cycle, handle_deferred_actions() checks pending flags
4. Deferred work (e.g., path generation) executes with access to full context

This separates state transitions (instant) from business logic (deferred), ensuring
the state machine remains focused on state management while expensive operations
run after the UI refresh.

States (3 states - unified idle):
    IDLE: Ready to build (user selects slope or lift type first)
    SLOPE_BUILDING: Proposals visible + 0 or more committed segments
    LIFT_PLACING: Start node/location selected, waiting for end node

Build Mode (stored in context.build_mode):
    Determines what clicking terrain/nodes creates:
    - None: Just viewing, clicks on terrain ignored
    - "slope": Start slope building
    - "chairlift"/"gondola"/"drag_lift"/"funicular": Start lift placing

Panel System (orthogonal to state):
    The info panel (ViewingContext.panel_visible) can show slope/lift details
    in IDLE state. It's hidden automatically when building starts.
    Users can always start building without explicitly closing the panel.

Transitions:
    IDLE -> SLOPE_BUILDING: start_building (click terrain/node with slope mode)
    IDLE -> LIFT_PLACING: select_lift_start (click terrain/node with lift mode)
    SLOPE_BUILDING -> SLOPE_BUILDING: commit_path (commit segment, continue)
    SLOPE_BUILDING -> IDLE: finish_slope, cancel_slope, undo_segment (when no segments left)
    LIFT_PLACING -> IDLE: complete_lift, cancel_lift

Cleanup on Transition
---------------------
StreamlitUIListener.after_transition() calls graph.perform_cleanup() before st.rerun().
This ensures the resort graph is always in a clean state by:
- Removing isolated nodes (nodes not connected to any segment or lift)
- Creating automatic backups to output/skiresort_planner/

Reference: DETAILS_STATEMACHINE.md for state diagram
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import streamlit as st
from statemachine import State, StateMachine
from statemachine.exceptions import TransitionNotAllowed

from skiresort_planner.constants import LiftConfig, MapConfig
from skiresort_planner.model.path_point import PathPoint

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from skiresort_planner.core import TerrainOrientation
    from skiresort_planner.model import ProposedSlopeSegment
    from skiresort_planner.model.resort_graph import ResortGraph


# Coordinate tuple type aliases for clarity
# Folium requires (lat, lon) order for map operations
FoliumLatLon = tuple[float, float]  # (lat, lon) - for map_center, clicked_location
# Internal geometry uses (lon, lat) to match PyProj/Shapely conventions
GeoLonLatElev = tuple[float, float, float | None]  # (lon, lat, elev) - for custom_target_location


@dataclass
class SelectionContext:
    """Current click/selection state."""

    lon: float | None = None
    lat: float | None = None
    elevation: float | None = None
    location: FoliumLatLon | None = None  # (lat, lon) for Folium
    node_id: str | None = None

    def clear(self) -> None:
        self.lon = None
        self.lat = None
        self.elevation = None
        self.location = None
        self.node_id = None

    def set(self, lon: float, lat: float, elevation: float) -> None:
        self.lon = lon
        self.lat = lat
        self.elevation = elevation
        self.location = (lat, lon)

    def has_selection(self) -> bool:
        return self.lon is not None


@dataclass
class ProposalContext:
    """Path proposals state."""

    paths: list[ProposedSlopeSegment] = field(default_factory=list)
    selected_idx: int | None = None
    terrain_orientation: TerrainOrientation | None = None

    def clear(self) -> None:
        self.paths = []
        self.selected_idx = None
        self.terrain_orientation = None


@dataclass
class BuildingContext:
    """Slope building state."""

    name: str | None = None
    segments: list[str] = field(default_factory=list)
    start_node: str | None = None
    endpoints: list[str] = field(default_factory=list)

    def clear(self) -> None:
        self.name = None
        self.segments = []
        self.start_node = None
        self.endpoints = []

    def has_committed_segments(self) -> bool:
        return len(self.segments) > 0


@dataclass
class LiftContext:
    """Lift placement state."""

    start_node_id: str | None = None
    start_location: PathPoint | None = None  # For new node creation
    type: str = "chairlift"

    def clear(self) -> None:
        self.start_node_id = None
        self.start_location = None


class BuildMode:
    """What type of element user wants to build.

    Uses internal lift type names matching StyleConfig.LIFT_ICONS.
    """

    SLOPE = "slope"
    CHAIRLIFT = "chairlift"
    GONDOLA = "gondola"
    SURFACE_LIFT = "surface_lift"
    AERIAL_TRAM = "aerial_tram"

    assert CHAIRLIFT in LiftConfig.TYPES, f"Invalid lift type '{CHAIRLIFT}'."
    assert GONDOLA in LiftConfig.TYPES, f"Invalid lift type '{GONDOLA}'."
    assert SURFACE_LIFT in LiftConfig.TYPES, f"Invalid lift type '{SURFACE_LIFT}'."
    assert AERIAL_TRAM in LiftConfig.TYPES, f"Invalid lift type '{AERIAL_TRAM}'."

    # All lift types for iteration (matches StyleConfig.LIFT_ICONS keys)
    LIFT_TYPES = [CHAIRLIFT, GONDOLA, SURFACE_LIFT, AERIAL_TRAM]

    @staticmethod
    def is_slope(mode: str) -> bool:
        """Check if mode is slope building."""
        return mode == BuildMode.SLOPE

    @staticmethod
    def is_lift(mode: str) -> bool:
        """Check if mode is a lift type (not slope)."""
        return mode in BuildMode.LIFT_TYPES

    @staticmethod
    def display_name(mode: str) -> str:
        """Human-friendly name for display."""
        from skiresort_planner.constants import StyleConfig

        return StyleConfig.LIFT_DISPLAY_NAMES[mode]

    @staticmethod
    def icon(mode: str) -> str:
        """Emoji icon for mode."""
        from skiresort_planner.constants import StyleConfig

        if mode == BuildMode.SLOPE:
            return "⛷️"
        # Lift icons are defined in StyleConfig.LIFT_ICONS with keys matching mode
        return StyleConfig.LIFT_ICONS[mode]


@dataclass
class BuildModeContext:
    """Build mode selection state.

    Tracks what type of element the user wants to build next.
    Defaults to SLOPE - there is always a mode selected.
    """

    mode: str = BuildMode.SLOPE  # One of BuildMode values, defaults to slope

    def clear(self) -> None:
        """Reset to default slope mode."""
        self.mode = BuildMode.SLOPE

    def is_slope(self) -> bool:
        """Check if mode is slope building."""
        return self.mode == BuildMode.SLOPE

    def is_lift(self) -> bool:
        """Check if mode is any lift type."""
        return self.mode is not None and BuildMode.is_lift(self.mode)


@dataclass
class ViewingContext:
    """Info panel state for viewing slopes/lifts.

    The panel is orthogonal to state - it can be shown in any IDLE state.
    Panel is auto-hidden when building starts, or manually via hide_panel().
    """

    slope_id: str | None = None
    lift_id: str | None = None
    panel_visible: bool = False

    def show_slope(self, slope_id: str) -> None:
        """Show slope info panel."""
        self.slope_id = slope_id
        self.lift_id = None
        self.panel_visible = True

    def show_lift(self, lift_id: str) -> None:
        """Show lift info panel."""
        self.slope_id = None
        self.lift_id = lift_id
        self.panel_visible = True

    def hide(self) -> None:
        """Hide info panel (keeps IDs for potential re-show)."""
        self.panel_visible = False

    def clear(self) -> None:
        """Clear all viewing state."""
        self.slope_id = None
        self.lift_id = None
        self.panel_visible = False


@dataclass
class CustomConnectContext:
    """Custom connect mode state."""

    enabled: bool = False
    start_node: str | None = None
    force_mode: bool = False
    target_location: GeoLonLatElev | None = None  # (lon, lat, elev) internal order

    def clear(self) -> None:
        self.enabled = False
        self.start_node = None
        self.force_mode = False
        self.target_location = None


@dataclass
class MapContext:
    """Map UI state (Folium format)."""

    center: FoliumLatLon = (MapConfig.START_CENTER_LAT, MapConfig.START_CENTER_LON)
    zoom: int = MapConfig.DEFAULT_ZOOM


@dataclass
class ClickDetectionResult:
    """Result of click detection - replaces confusing tuple returns.

    Attributes:
        click_type: "marker", "terrain", or None (no click/ghost)
        data: The click coordinate dict ({"lat": ..., "lng": ...}) or None
        tooltip: Marker tooltip string (only for marker clicks) or None
    """

    click_type: str | None
    data: dict | None
    tooltip: str | None

    @property
    def is_valid(self) -> bool:
        """True if a real click was detected."""
        return self.click_type is not None

    @property
    def is_marker(self) -> bool:
        """True if this is a marker click."""
        return self.click_type == "marker"

    @property
    def is_terrain(self) -> bool:
        """True if this is a terrain click."""
        return self.click_type == "terrain"

    @staticmethod
    def no_click() -> "ClickDetectionResult":
        """Factory for no-click/ghost result."""
        return ClickDetectionResult(click_type=None, data=None, tooltip=None)


@dataclass
class ClickDeduplicationContext:
    """Simple click deduplication by tracking last-seen coordinates.

    With returned_objects limited to click fields only, pan/zoom don't trigger
    reruns at all. We only need to track the last coordinates to prevent
    re-processing the same click on subsequent reruns (e.g., button clicks).
    """

    last_marker_data: dict | None = None
    last_terrain_data: dict | None = None
    pending_recompute: bool = False

    def detect_new_click(
        self,
        marker_data: dict | None,
        marker_tooltip: str | None,
        terrain_data: dict | None,
    ) -> ClickDetectionResult:
        """Detect if there's a new click by comparing coordinates."""
        marker_new = marker_data is not None and marker_data != self.last_marker_data
        terrain_new = terrain_data is not None and terrain_data != self.last_terrain_data

        if marker_new:
            self.last_marker_data = marker_data
            if terrain_data is not None:
                self.last_terrain_data = terrain_data
            return ClickDetectionResult(click_type="marker", data=marker_data, tooltip=marker_tooltip)

        if terrain_new:
            self.last_terrain_data = terrain_data
            return ClickDetectionResult(click_type="terrain", data=terrain_data, tooltip=None)

        return ClickDetectionResult.no_click()

    def clear(self) -> None:
        """Clear dedup state."""
        self.last_marker_data = None
        self.last_terrain_data = None

    def clear_marker(self) -> None:
        """Clear only marker dedup state.

        Called on state transitions to allow clicking the same marker in a new state.
        Terrain dedup is preserved to prevent ghost clicks from st.rerun().
        """
        self.last_marker_data = None


@dataclass
class DeferredContext:
    """Deferred action flags for work that runs after st.rerun().

    When set, handle_deferred_actions() in app.py performs the work on the
    next render cycle. This ensures expensive operations (path generation)
    run with full access to session state after the UI refresh.
    """

    path_generation: bool = False
    gradient_target: float | None = None  # For smart path recommendation
    auto_finish: bool = False  # Auto-finish slope after connector commit
    custom_connect: bool = False  # Generate paths to custom target location
    start_building_from_node_id: str | None = None  # Deferred start_building from node
    start_lift_from_node_id: str | None = None  # Deferred start_lift from node

    def clear_custom_connect(self) -> None:
        self.custom_connect = False


@dataclass
class UIMessagesContext:
    """User-facing messages and errors."""

    message: str = ""
    error: str = ""

    def clear(self) -> None:
        self.message = ""
        self.error = ""


@dataclass
class PlannerContext:
    """Shared context/model for state machine.

    Holds all mutable state that persists across transitions including
    current selection, path proposals, slope building progress, lift
    placement, and UI state like map center and messages.

    Architecture: State is organized into logical sub-contexts for clarity.
    Backward-compatible properties map old field names to new sub-contexts
    so existing code like `ctx.current_lon` continues working.

    Sub-contexts:
        selection: Current click/selection data
        proposals: Generated path proposals
        building: Slope building progress
        lift: Lift placement state
        viewing: Which slope/lift is being viewed
        custom_connect: Custom target connection mode
        map: Map center and zoom
        click_dedup: Click deduplication tracking
        deferred: Flags for deferred actions
        messages: User-facing messages/errors

    Note: The 'state' field is managed by python-statemachine when this
    object is passed as the model. It stores the current state value.
    """

    # State managed by python-statemachine (model pattern)
    state: str | None = None

    # Organized sub-contexts
    selection: SelectionContext = field(default_factory=SelectionContext)
    proposals: ProposalContext = field(default_factory=ProposalContext)
    building: BuildingContext = field(default_factory=BuildingContext)
    lift: LiftContext = field(default_factory=LiftContext)
    viewing: ViewingContext = field(default_factory=ViewingContext)
    custom_connect: CustomConnectContext = field(default_factory=CustomConnectContext)
    map: MapContext = field(default_factory=MapContext)
    click_dedup: ClickDeduplicationContext = field(default_factory=ClickDeduplicationContext)
    deferred: DeferredContext = field(default_factory=DeferredContext)
    messages: UIMessagesContext = field(default_factory=UIMessagesContext)
    build_mode: BuildModeContext = field(default_factory=BuildModeContext)

    # Settings
    segment_length_m: int = 500

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    def clear_selection(self) -> None:
        """Clear current selection state."""
        self.selection.clear()
        self.messages.clear()

    def clear_proposals(self) -> None:
        """Clear path proposals."""
        self.proposals.clear()

    def clear_building(self) -> None:
        """Clear slope building state."""
        self.building.clear()

    def clear_lift(self) -> None:
        """Clear lift placement state."""
        self.lift.clear()

    def clear_viewing(self) -> None:
        """Clear viewing state."""
        self.viewing.clear()

    def clear_custom_connect(self) -> None:
        """Clear custom connect mode."""
        self.custom_connect.clear()
        self.deferred.clear_custom_connect()

    def set_selection(
        self,
        lon: float,
        lat: float,
        elevation: float,
    ) -> None:
        """Set current selection."""
        self.selection.set(lon=lon, lat=lat, elevation=elevation)

    def __repr__(self) -> str:
        """Return string representation of context."""
        return (
            f"PlannerContext(state={self.state}, "
            f"location={self.selection.location}, "
            f"slope={self.building.name}, "
            f"segments={len(self.building.segments)}, "
            f"lift_start={self.lift.start_node_id})"
        )

    def has_selection(self) -> bool:
        """Check if a point is selected."""
        return self.selection.has_selection()

    def has_committed_segments(self) -> bool:
        """Check if there are committed segments in current slope."""
        return self.building.has_committed_segments()


class StreamlitUIListener:
    """Listener that handles Streamlit UI side effects after state transitions.

    This listener follows the python-statemachine best practice of using
    listeners for side effects. It runs after every state transition to:

    1. Perform cleanup (remove isolated nodes, create auto-backup)
    2. Trigger st.rerun() to refresh the UI

    The separation ensures the state machine focuses purely on state logic
    while this listener handles UI integration and maintenance tasks.

    Usage:
        sm = PlannerStateMachine(context=context)
        sm.add_listener(StreamlitUIListener())
    """

    def after_transition(self, event: str, source: State, target: State) -> None:
        """Run cleanup and trigger Streamlit rerun after state transitions.

        NOTE: We do NOT modify click deduplication here. The dedup is simple:
        same click key = duplicate. When user clicks elsewhere, key changes,
        so they can click back to original element.
        """
        logger.info(f"[STATE] {source.name} --({event})--> {target.name}")

        # Perform graph cleanup before rerun (isolated nodes, auto-backup)
        graph: ResortGraph = st.session_state.get("graph")
        if graph is not None:
            graph.perform_cleanup()

        st.rerun()


class PlannerStateMachine(StateMachine):
    """State machine for ski resort planner workflow.

    Manages transitions between 3 planning states with guards
    and hooks for validation and side effects. See module docstring
    for complete transition documentation.

    States:
        idle: Ready to build (user selects build type first)
        slope_building: Building a slope with committed segments
        lift_placing: Placing a lift (start selected, waiting for end)

    Panel visibility (ctx.viewing.panel_visible) is orthogonal to state
    and managed via show_slope_info_panel()/show_lift_info_panel()/hide_info_panel() methods.
    """

    # ==========================================================================
    # State Definitions (3 states: unified idle + 2 building states)
    # ==========================================================================

    # Unified idle state (initial)
    idle = State("Idle", initial=True)

    # Building states
    slope_building = State("SlopeBuilding")
    lift_placing = State("LiftPlacing")

    # ==========================================================================
    # Transitions: IDLE (unified entry state)
    # ==========================================================================

    # Start building new slope
    start_building = idle.to(slope_building)
    # Resume building (for undo of finish_slope)
    resume_building = idle.to(slope_building)
    # Select bottom station for lift
    select_lift_start = idle.to(lift_placing)

    # ==========================================================================
    # Transitions: SLOPE_BUILDING
    # ==========================================================================

    # Commit path and continue building
    commit_path = slope_building.to(slope_building)
    # Finish slope and return to idle (with panel showing the new slope)
    finish_slope = slope_building.to(idle)
    # Cancel slope (discard all segments) and return to idle
    cancel_slope = slope_building.to(idle)
    # Undo last segment (may stay in building or go to idle)
    undo_segment = slope_building.to(slope_building, cond="has_more_segments") | slope_building.to(
        idle, unless="has_more_segments"
    )

    # ==========================================================================
    # Transitions: LIFT_PLACING
    # ==========================================================================

    # Complete lift placement -> return to idle (with panel showing the new lift)
    complete_lift = lift_placing.to(idle)
    # Cancel lift placement
    cancel_lift = lift_placing.to(idle)

    # ==========================================================================
    # Guards (Conditions)
    # ==========================================================================

    def has_more_segments(self) -> bool:
        """Guard: Check if there are committed segments to undo to."""
        return len(self.context.building.segments) > 1

    # ==========================================================================
    # State Check Properties (idiomatic python-statemachine pattern)
    # ==========================================================================

    @property
    def is_idle(self) -> bool:
        """Check if in idle state (ready to build)."""
        return self.idle.is_active

    @property
    def is_slope_building(self) -> bool:
        """Check if in slope building state."""
        return self.slope_building.is_active

    @property
    def is_lift_placing(self) -> bool:
        """Check if in lift placing state."""
        return self.lift_placing.is_active

    # ==========================================================================
    # Info Panel Helpers (not state changes, just context updates)
    # ==========================================================================

    def show_slope_info_panel(self, slope_id: str) -> None:
        """Show slope info panel (no state change, no st.rerun)."""
        self.context.viewing.show_slope(slope_id=slope_id)
        logger.info(f"Info panel: showing slope {slope_id}")

    def show_lift_info_panel(self, lift_id: str) -> None:
        """Show lift info panel (no state change, no st.rerun)."""
        self.context.viewing.show_lift(lift_id=lift_id)
        logger.info(f"Info panel: showing lift {lift_id}")

    def hide_info_panel(self) -> None:
        """Hide info panel (no state change, no st.rerun).

        Note: bump_map_version() is called by the button handlers in
        right_panel.py to clear stale click state.
        """
        self.context.viewing.hide()
        logger.info("Info panel: hidden")

    @property
    def is_info_panel_visible(self) -> bool:
        """Check if info panel is currently visible."""
        return self.context.viewing.panel_visible

    # ==========================================================================
    # Mode Helpers
    # ==========================================================================

    def is_slope_mode(self) -> bool:
        """Check if in slope-related state (idle with slope mode or building)."""
        return self.is_slope_building or (self.is_idle and self.context.build_mode.is_slope())

    def is_lift_mode(self) -> bool:
        """Check if in lift-related state (idle with lift mode or placing)."""
        return self.is_lift_placing or (self.is_idle and self.context.build_mode.is_lift())

    # ==========================================================================
    # Entry Hooks
    # ==========================================================================

    def on_enter_idle(self) -> None:
        """Hook: Entering idle state."""
        self.context.clear_proposals()
        self.context.clear_building()
        self.context.clear_custom_connect()
        self.context.clear_lift()
        self.context.selection.node_id = None
        # NOTE: Don't clear viewing here - panel stays visible if it was open
        # NOTE: Don't clear build_mode here - user keeps their selection
        # Clear marker dedup so user can click same marker in new state (e.g., branch from same node)
        # Terrain dedup is preserved to prevent ghost clicks from st.rerun()
        self.context.click_dedup.clear_marker()

    def on_enter_slope_building(self) -> None:
        """Hook: Entering slope building state."""
        # Auto-hide panel when starting to build
        self.context.viewing.hide()

    def on_enter_lift_placing(self) -> None:
        """Hook: Entering lift placing state."""
        # Auto-hide panel when starting to place lift
        self.context.viewing.hide()

    # ==========================================================================
    # Exit Hooks
    # ==========================================================================

    def on_exit_slope_building(self) -> None:
        """Hook: Exiting slope building state."""
        self.context.clear_proposals()
        self.context.clear_custom_connect()

    def on_exit_lift_placing(self) -> None:
        """Hook: Exiting lift placing state."""
        pass  # Keep lift_start_node_id until explicitly cleared

    # Cleanup and st.rerun() are handled by StreamlitUIListener.after_transition()
    # to separate UI concerns from state machine logic (python-statemachine best practice)

    # ==========================================================================
    # Transition Actions (before_* hooks)
    # ==========================================================================

    def before_start_building(
        self,
        lon: float,
        lat: float,
        elevation: float,
        node_id: str | None,
        slope_number: int,
    ) -> None:
        """Action before starting to build a slope."""
        self.context.set_selection(lon=lon, lat=lat, elevation=elevation)
        self.context.building.start_node = node_id
        self.context.selection.node_id = node_id
        # Set slope name (working name using slope number)
        self.context.building.name = f"Slope {slope_number}"

    def before_commit_path(self, segment_id: str, endpoint_node_id: str) -> None:
        """Action before committing a path segment."""
        self.context.building.segments.append(segment_id)
        self.context.building.endpoints = [endpoint_node_id]
        self.context.clear_proposals()

    def before_finish_slope(self, slope_id: str) -> None:
        """Action before finishing a slope - show panel with the new slope."""
        self.context.viewing.show_slope(slope_id=slope_id)

    def before_cancel_slope(self) -> None:
        """Action before canceling a slope."""
        pass

    def before_undo_segment(self, removed_segment_id: str) -> None:
        """Action before undoing a segment."""
        if removed_segment_id in self.context.building.segments:
            self.context.building.segments.remove(removed_segment_id)

    def before_select_lift_start(self, node_id: str | None = None, location: PathPoint | None = None) -> None:
        """Action before selecting lift start node or location."""
        self.context.lift.start_node_id = node_id
        self.context.lift.start_location = location

    def before_complete_lift(self, lift_id: str) -> None:
        """Action before completing lift - show panel with the new lift."""
        self.context.viewing.show_lift(lift_id=lift_id)
        self.context.lift.start_node_id = None

    def before_cancel_lift(self) -> None:
        """Action before canceling lift."""
        self.context.clear_lift()

    # ==========================================================================
    # Initialization
    # ==========================================================================

    def __init__(self, context: PlannerContext | None = None, start_value: str | None = None) -> None:
        """Initialize state machine with model pattern.

        Args:
            context: Shared context/model (creates new if None)
            start_value: Optional initial state value (for restoring state)
        """
        model = context or PlannerContext()
        super().__init__(model=model, start_value=start_value)

    # ==========================================================================
    # Utility Methods
    # ==========================================================================

    @property
    def context(self) -> PlannerContext:
        """Alias for model - provides backward compatibility."""
        return self.model

    def can_finish_slope(self) -> bool:
        """Check if slope can be finished (has committed segments)."""
        return self.is_slope_building and len(self.context.building.segments) > 0

    def can_undo(self) -> bool:
        """Check if undo is available in current state."""
        return self.is_slope_building and len(self.context.building.segments) > 0

    def get_state_name(self) -> str:
        """Get current state name for display."""
        return self.current_state.name

    def get_available_actions(self) -> list[str]:
        """Get list of available transition names (for UI display only)."""
        return [t.event for t in self.current_state.transitions]

    def __repr__(self) -> str:
        """Return string representation of state machine."""
        return f"PlannerStateMachine(state={self.get_state_name()}, model={self.context!r})"

    def try_transition(self, event: str, **kwargs: Any) -> bool:
        """Attempt a transition, returning success/failure.

        Args:
            event: Transition event name
            **kwargs: Arguments for transition

        Returns:
            True if transition succeeded, False otherwise.
        """
        try:
            self.send(event=event, **kwargs)
            return True
        except TransitionNotAllowed:
            logger.warning(f"Transition '{event}' not allowed from {self.get_state_name()}")
            return False

    @staticmethod
    def create(add_ui_listener: bool = True) -> tuple["PlannerStateMachine", PlannerContext]:
        """Factory method to create state machine with context and optional UI listener.

        Args:
            add_ui_listener: If True, adds StreamlitUIListener for auto st.rerun().
                             Set to False for testing or non-Streamlit usage.

        Returns:
            Tuple of (PlannerStateMachine, PlannerContext)
        """
        context = PlannerContext()
        sm = PlannerStateMachine(context=context)
        if add_ui_listener:
            sm.add_listener(StreamlitUIListener())
            logger.info("Created PlannerStateMachine with StreamlitUIListener")
        else:
            logger.info("Created PlannerStateMachine without UI listener")
        return sm, context

"""Context Classes for Ski Resort Planner State Machine.

This module contains all context dataclasses that hold mutable state
for the ski resort planner application. The state machine uses these
contexts to track UI state, building progress, and user selections.

Architecture:
- All contexts inherit from BaseContext (provides clear() interface)
- PlannerContext composes all sub-contexts
- Contexts are pure data holders - no business logic
- State machine owns the context, UI reads from it

Sub-contexts:
    SelectionContext: Current click/selection data
    ProposalContext: Generated path proposals
    BuildingContext: Slope building progress
    LiftContext: Lift placement state
    BuildModeContext: What type of element to build
    ViewingContext: Which slope/lift is being viewed
    CustomConnectContext: Custom target connection mode
    MapContext: Map center and zoom
    ClickDeduplicationContext: Click deduplication tracking
    DeferredContext: Flags for deferred actions
    UIMessagesContext: User-facing messages/errors
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from skiresort_planner.constants import ClickConfig, LiftConfig, MapConfig, PathConfig

if TYPE_CHECKING:
    from skiresort_planner.core import TerrainOrientation
    from skiresort_planner.model import PathPoint, ProposedSlopeSegment


# Coordinate type aliases for clarity
# Pydeck uses [lon, lat] order (GeoJSON standard) for all coordinates
LonLat = tuple[float, float]  # (lon, lat) - for map operations
LonLatElev = tuple[float, float, float | None]  # (lon, lat, elev) - for locations with elevation


class BaseContext(ABC):
    """Abstract base class for all context dataclasses.

    All contexts should be clearable to reset to their initial state.
    This enables clean state transitions without residual data.
    """

    @abstractmethod
    def clear(self) -> None:
        """Reset context to initial state."""
        ...


@dataclass
class SelectionContext(BaseContext):
    """Current click/selection state."""

    lon: float | None = None
    lat: float | None = None
    elevation: float | None = None
    node_id: str | None = None

    def clear(self) -> None:
        self.lon = None
        self.lat = None
        self.elevation = None
        self.node_id = None

    def set(self, lon: float, lat: float, elevation: float) -> None:
        """Set selection coordinates. Use this setter, don't set fields directly."""
        self.lon = lon
        self.lat = lat
        self.elevation = elevation

    def has_selection(self) -> bool:
        """Check if a valid selection exists."""
        return self.lon is not None and self.lat is not None

    def get_lon_lat(self) -> tuple[float, float]:
        """Get (lon, lat) tuple. Raises if no selection exists.

        Use has_selection() first to check, then call this for strict access.

        Returns:
            Tuple of (longitude, latitude)

        Raises:
            ValueError: If no selection exists (lon or lat is None)
        """
        if self.lon is None or self.lat is None:
            raise ValueError("No selection exists. Call has_selection() first.")
        return (self.lon, self.lat)

    @property
    def coordinate(self) -> list[float] | None:
        """Return [lon, lat] for display/logging. Returns None if no selection."""
        if self.lon is not None and self.lat is not None:
            return [self.lon, self.lat]
        return None


@dataclass
class ProposalContext(BaseContext):
    """Path proposals state."""

    paths: list[ProposedSlopeSegment] = field(default_factory=list)
    selected_idx: int | None = None
    terrain_orientation: TerrainOrientation | None = None

    def clear(self) -> None:
        self.paths = []
        self.selected_idx = None
        self.terrain_orientation = None


@dataclass
class BuildingContext(BaseContext):
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
class LiftContext(BaseContext):
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
class BuildModeContext(BaseContext):
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
        return BuildMode.is_lift(self.mode)


@dataclass
class ViewingContext(BaseContext):
    """Info panel state for viewing slopes/lifts.

    State Machine Integration:
    - before_* hooks call set_slope_id() or set_lift_id() to store WHAT to view
    - enter_* functions call show_panel() to make panel visible
    - exit functions going to non-viewing states call hide_panel()

    This separation ensures:
    - The ID is set by the triggering action (before_*)
    - Panel visibility is guaranteed by entering the state (enter_*)
    - No matter how you enter IDLE_VIEWING_*, the panel will be visible

    Attributes:
        slope_id: ID of slope being viewed (None if viewing lift or nothing)
        lift_id: ID of lift being viewed (None if viewing slope or nothing)
        panel_visible: Whether info panel should be rendered
        view_3d: When True, renders map in 3D with terrain
    """

    slope_id: str | None = None
    lift_id: str | None = None
    panel_visible: bool = False
    view_3d: bool = False

    # =========================================================================
    # SETTER METHODS (called by state machine before_* hooks)
    # =========================================================================

    def set_slope_id(self, slope_id: str) -> None:
        """Set the slope to view. Called by before_* hooks.

        Sets slope_id and clears lift_id. Does NOT change panel_visible.
        The enter_* function will call show_panel() to make it visible.

        Args:
            slope_id: ID of slope to view (e.g., "SL1")
        """
        self.slope_id = slope_id
        self.lift_id = None

    def set_lift_id(self, lift_id: str) -> None:
        """Set the lift to view. Called by before_* hooks.

        Sets lift_id and clears slope_id. Does NOT change panel_visible.
        The enter_* function will call show_panel() to make it visible.

        Args:
            lift_id: ID of lift to view (e.g., "L1")
        """
        self.lift_id = lift_id
        self.slope_id = None

    # =========================================================================
    # STATE CONTROL METHODS (called by enter_*/exit_* lifecycle functions)
    # =========================================================================

    def show_panel(self) -> None:
        """Make info panel visible. Called by enter_idle_viewing_* functions.

        This is called by the enter function to ensure panel is visible
        regardless of which transition brought us to the viewing state.
        """
        self.panel_visible = True

    def hide_panel(self) -> None:
        """Hide info panel. Called when entering non-viewing states.

        Also disables 3D view since 3D is only available when viewing.
        Called by enter_* functions of building states (slope_*, lift_placing).
        """
        self.panel_visible = False
        self.view_3d = False

    # =========================================================================
    # 3D VIEW CONTROL
    # =========================================================================

    def enable_3d(self) -> None:
        """Enable 3D view with terrain."""
        self.view_3d = True

    def disable_3d(self) -> None:
        """Disable 3D view, return to flat 2D map."""
        self.view_3d = False

    # =========================================================================
    # QUERY METHODS (for UI to check state)
    # =========================================================================

    def is_viewing_slope(self) -> bool:
        """Check if currently viewing a slope (panel visible with slope_id set)."""
        return self.panel_visible and self.slope_id is not None

    def is_viewing_lift(self) -> bool:
        """Check if currently viewing a lift (panel visible with lift_id set)."""
        return self.panel_visible and self.lift_id is not None

    def clear(self) -> None:
        """Clear all viewing state. Called when entering IDLE_READY."""
        self.slope_id = None
        self.lift_id = None
        self.panel_visible = False
        self.view_3d = False


@dataclass
class CustomConnectContext(BaseContext):
    """Custom connect mode state."""

    enabled: bool = False
    start_node: str | None = None
    force_mode: bool = False
    target_location: LonLatElev | None = None  # (lon, lat, elev)

    def clear(self) -> None:
        self.enabled = False
        self.start_node = None
        self.force_mode = False
        self.target_location = None


@dataclass
class MapContext(BaseContext):
    """Map UI state for Pydeck.

    State Machine Integration:
    - Use set_building_view() when entering building states (zoomed in, top-down)
    - Use set_viewing_view() when entering viewing states (zoomed out, overview)
    - Use set_3d_view() for 3D terrain viewing

    Note: center is [lon, lat] order (GeoJSON/Pydeck standard).
    """

    lon: float = MapConfig.START_CENTER_LON
    lat: float = MapConfig.START_CENTER_LAT
    zoom: int = MapConfig.DEFAULT_ZOOM
    pitch: float = MapConfig.DEFAULT_PITCH
    bearing: float = MapConfig.DEFAULT_BEARING

    # =========================================================================
    # GETTER PROPERTIES (for UI to read)
    # =========================================================================

    @property
    def lat_lon(self) -> tuple[float, float]:
        """Return (lat, lon) tuple - standard geographic order."""
        return (self.lat, self.lon)

    @property
    def lon_lat(self) -> tuple[float, float]:
        """Return (lon, lat) tuple - GeoJSON/Pydeck order."""
        return (self.lon, self.lat)

    @property
    def lon_lat_list(self) -> list[float]:
        """Return [lon, lat] list for Pydeck ViewState."""
        return [self.lon, self.lat]

    # =========================================================================
    # SETTER METHODS (called by state machine or actions)
    # =========================================================================

    def set_center(self, lon: float, lat: float) -> None:
        """Set map center coordinates.

        Args:
            lon: Longitude (x coordinate)
            lat: Latitude (y coordinate)
        """
        self.lon = lon
        self.lat = lat

    def set_building_view(self, lon: float, lat: float) -> None:
        """Set map to building mode: centered, zoomed in, top-down.

        Use when starting slope building or lift placement.
        Provides precise placement view for accurate clicking.

        Args:
            lon: Center longitude
            lat: Center latitude
        """
        self.lon = lon
        self.lat = lat
        self.zoom = MapConfig.BUILDING_ZOOM
        self.pitch = MapConfig.BUILDING_PITCH

    def set_viewing_view(self, lon: float, lat: float) -> None:
        """Set map to viewing mode: centered, zoomed out, top-down overview.

        Use when viewing completed slopes/lifts.
        Provides good overview of the entire element.

        Args:
            lon: Center longitude
            lat: Center latitude
        """
        self.lon = lon
        self.lat = lat
        self.zoom = MapConfig.VIEWING_ZOOM
        self.pitch = MapConfig.VIEWING_PITCH

    def set_3d_view(self, lon: float, lat: float, pitch: float, zoom: int) -> None:
        """Set map to 3D terrain view with specified camera angle.

        Args:
            lon: Center longitude
            lat: Center latitude
            pitch: Camera tilt angle in degrees (0=top-down, higher=more tilted)
            zoom: Zoom level (adjusted for elevation to prevent camera clipping)
        """
        self.lon = lon
        self.lat = lat
        self.pitch = pitch
        self.zoom = zoom

    def reset_view(self) -> None:
        """Reset zoom, pitch, and bearing to defaults for 2D viewing."""
        self.zoom = MapConfig.DEFAULT_ZOOM
        self.pitch = MapConfig.DEFAULT_PITCH
        self.bearing = MapConfig.DEFAULT_BEARING

    def clear(self) -> None:
        """Reset to default map position and view settings."""
        self.lon = MapConfig.START_CENTER_LON
        self.lat = MapConfig.START_CENTER_LAT
        self.zoom = MapConfig.DEFAULT_ZOOM
        self.pitch = MapConfig.DEFAULT_PITCH
        self.bearing = MapConfig.DEFAULT_BEARING


@dataclass
class ClickDeduplicationContext(BaseContext):
    """Click deduplication for Pydeck by tracking last-seen coordinates and object IDs.

    Pydeck provides coordinate and picked object data. We track both to prevent
    re-processing the same click on subsequent reruns (e.g., button clicks).

    Also includes timestamp debounce to prevent rapid double-clicks from
    triggering duplicate actions.
    """

    last_coord: tuple[float, float] | None = None
    last_object_id: str | None = None
    last_click_timestamp: float = 0.0
    pending_recompute: bool = False
    debounce_seconds: float = ClickConfig.DEBOUNCE_TIME_DELAY  # Minimum time between clicks (150ms debounce)

    def is_new_click(
        self,
        coord: tuple[float, ...] | None,
        obj_id: str | None,
    ) -> bool:
        """Check if this is a new click by comparing coordinates, object ID, and timing.

        Args:
            coord: Click coordinate tuple (lon, lat) or None
            obj_id: Unique object identifier string or None for terrain

        Returns:
            True if this is a new click that should be processed
        """
        import time

        # No click data at all
        if coord is None and obj_id is None:
            return False

        # Check timestamp debounce (skip if debounce_seconds is 0, e.g., in tests)
        now = time.time()
        if self.debounce_seconds > 0 and now - self.last_click_timestamp < self.debounce_seconds:
            return False

        # Object click - check object ID
        if obj_id is not None:
            if obj_id != self.last_object_id:
                self.last_object_id = obj_id
                self.last_click_timestamp = now
                if coord is not None:
                    self.last_coord = (coord[0], coord[1])
                return True
            return False

        # Terrain click - check coordinates
        if coord is not None:
            coord_2d = (coord[0], coord[1])
            if coord_2d != self.last_coord:
                self.last_coord = coord_2d
                self.last_click_timestamp = now
                return True
            return False

        return False

    def clear(self) -> None:
        """Clear dedup state."""
        self.last_coord = None
        self.last_object_id = None

    def clear_marker(self) -> None:
        """Clear only object dedup state.

        Called on state transitions to allow clicking the same object in a new state.
        Coordinate dedup is preserved to prevent ghost clicks from st.rerun().
        """
        self.last_object_id = None


@dataclass
class DeferredContext(BaseContext):
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

    def clear(self) -> None:
        """Clear all deferred flags."""
        self.path_generation = False
        self.gradient_target = None
        self.auto_finish = False
        self.custom_connect = False
        self.start_building_from_node_id = None
        self.start_lift_from_node_id = None


@dataclass
class UIMessagesContext(BaseContext):
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
    segment_length_m: int = PathConfig.SEGMENT_LENGTH_DEFAULT_M

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
            f"coordinate={self.selection.coordinate}, "
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

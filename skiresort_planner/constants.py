"""Configuration constants for Ski Resort Planner.

All configurable parameters are centralized here for easy tuning.
Values are referenced in DETAILS.md.

Classes:
    AppConfig: UI application settings
    MapConfig: Default map view parameters
    DEMConfig: Elevation data file paths
    SlopeConfig: Difficulty thresholds and targets
    PathConfig: Path generation domain values (segment length, road cap)
    GeometricTuningConfig: Machine-tunable knobs shaping generated route geometry
    EarthworkConfig: Excavation warning thresholds and belt width limits
    ConnectionConfig: Connection path parameters
    PlannerConfig: Grid-based Dijkstra grid connectivity
    MarkerConfig: Map marker styling
    LiftConfig: Lift types and catenary parameters
    StyleConfig: Visual colors and styling
    NameConfig: Creative naming components
    ChartConfig: Chart rendering dimensions
    OSMConfig: OpenStreetMap import (Overpass query, aerialway→lift-type map)
    MergeConfig: Manual node-merge tool
"""

from pathlib import Path
from typing import Literal

# Package root directory (where skiresort_planner/ lives)
PACKAGE_DIR = Path(__file__).parent

# Project root directory (parent of skiresort_planner/)
PROJECT_ROOT = PACKAGE_DIR.parent

# Data directory outside package (downloaded separately, not shipped with package)
DATA_DIR = PROJECT_ROOT / "data"

# Output directory for saved graphs
OUTPUT_DIR = PROJECT_ROOT / "output"

# Auto-saved resort backups (per-session crash/outage safety net)
BACKUP_DIR = PROJECT_ROOT / "backups"


class AppConfig:
    """UI application settings."""

    TITLE = "Ski Resort Planner"
    ICON = "⛷️"
    LAYOUT: Literal["centered", "wide"] = "wide"


class EntityPrefixes:
    """ID prefixes for graph entities."""

    NODE = "N"
    SEGMENT = "S"
    SLOPE = "SL"
    LIFT = "L"
    ROAD = "R"


class MapConfig:
    """Default map view parameters for Pydeck."""

    # Initial center for program start: Idalp, Ischgl, Austria
    START_CENTER_LAT = 46.982  # Latitude
    START_CENTER_LON = 10.317  # Longitude

    # Zoom levels for different modes
    # Higher number = more zoomed in, lower = more zoomed out
    # Reduced zoom levels to prevent camera going underground with 3D terrain
    BUILDING_ZOOM = 14  # Working zoom for building slopes/lifts
    VIEWING_ZOOM = 13  # Overview after finishing slope/lift (zoomed out)
    VIEW_3D_ZOOM = 14  # 3D side view - balanced zoom
    VIEW_3D_MIN_ZOOM = 12  # Minimum zoom for high elevation (prevents camera under terrain)
    DEFAULT_ZOOM = VIEWING_ZOOM  # Start zoomed out to prevent camera clipping terrain

    # Pitch angles for different modes
    # Use 0 (top-down) for all modes to ensure accurate terrain clicks
    BUILDING_PITCH = 0  # Top-down view for precise placement during building
    VIEWING_PITCH = 0  # Top-down view for viewing (tilted views cause terrain click issues)
    VIEW_3D_PITCH = 25  # 25° angle for 3D - more from above to avoid mountains blocking view
    DEFAULT_PITCH = 0  # Always start top-down
    DEFAULT_BEARING = 0  # Map rotation in degrees (0 = north up)

    # Node snapping threshold for lift placement (used when creating end nodes)
    LIFT_END_NODE_THRESHOLD_M = 80  # Extra generous for lift top station placement

    # At equator, 1 degree of latitude or longitude ≈ 111,320 meters
    # Used by MockDEMService in tests for coordinate calculations
    METERS_PER_DEGREE_EQUATOR = 111320.0

    # 2D mode z-offsets (relative layer ordering, no terrain)
    # Small offsets prevent z-fighting while keeping flat appearance
    # Z-offsets for 2D mode - small values for proper layer ordering
    Z_OFFSET_2D_SLOPES = 1  # Slope polygons at base
    Z_OFFSET_2D_LIFTS = 2  # Lift cables above slopes
    Z_OFFSET_2D_PYLONS = 3  # Pylons slightly above lift cables
    Z_OFFSET_2D_ICONS = 4  # Slope/lift icons above pylons
    Z_OFFSET_2D_NODES = 10  # Nodes above icons
    Z_OFFSET_2D_MARKERS = 20  # Interactive markers (commit/select) on top


class DEMConfig:
    """Elevation data file paths and Hugging Face hosting."""

    # Path to Alps DEM GeoTIFF (60m resolution, cropped to Alps region)
    # Full EuroDEM available from: https://www.mapsforeurope.org/datasets/euro-dem
    EURODEM_PATH = DATA_DIR / "alps_dem.tif"

    # Hugging Face hosting (auto-download if local file missing)
    HF_REPO_ID = "MichaelMedek/alps_eurodem"
    HF_FILENAME = "alps_dem.tif"
    HF_DOWNLOAD_URL = f"https://huggingface.co/datasets/{HF_REPO_ID}/resolve/main/{HF_FILENAME}"


class SlopeConfig:
    """Slope classification thresholds and difficulty targets."""

    # Core slope limits - single source of truth
    MIN_SKIABLE_PCT = 5  # Below this: need to push poles
    MAX_SKIABLE_PCT = 70  # Above this: dangerously steep

    # Side-slope classification thresholds (compute_side_slope)
    FLAT_GRADIENT_EPS_PCT = 0.5  # Terrain gradient below this → no meaningful side slope
    SIDE_SLOPE_FLAT_PCT = 2  # |side slope| below this → "flat" direction (not left/right)

    # European slope difficulty thresholds, classified by the steepest ROLLING_WINDOW_M
    # (300m) section (max_slope_pct) — see DETAILS.md §6.
    DIFFICULTY_THRESHOLDS = {
        "green": (0, 15),  # Beginner: 0-15%
        "blue": (15, 25),  # Intermediate: 15-25%
        "red": (25, 40),  # Advanced: 25-40%
        "black": (40, MAX_SKIABLE_PCT),  # Expert: 40%+
    }
    DIFFICULTIES = list(DIFFICULTY_THRESHOLDS.keys())

    # Target effective slopes for path generation (DETAILS.md Section 5.2)
    # Targets set 2-3% inside threshold bounds to prevent accidental misclassification
    DIFFICULTY_TARGETS = {
        "green": {"gentle": 7.0, "steep": 12.0},  # Threshold: 0-15%
        "blue": {"gentle": 17.0, "steep": 22.0},  # Threshold: 15-25%
        "red": {"gentle": 28.0, "steep": 37.0},  # Threshold: 25-40%
        "black": {"gentle": 45.0, "steep": 60.0},  # Threshold: 40%+
    }
    assert set(DIFFICULTY_TARGETS.keys()) == set(DIFFICULTIES)

    # Rolling window for steepness calculation
    # Used to find the steepest section within a segment or across the full path
    ROLLING_WINDOW_M = 300  # Window length in meters (few ski turns)


# Validate targets are within thresholds (module-level assertion)
assert all(
    SlopeConfig.DIFFICULTY_TARGETS[diff]["gentle"] < SlopeConfig.DIFFICULTY_THRESHOLDS[diff][1]
    for diff in SlopeConfig.DIFFICULTY_TARGETS
), "Gentle targets must be below upper threshold"
assert all(
    SlopeConfig.DIFFICULTY_TARGETS[diff]["steep"] > SlopeConfig.DIFFICULTY_THRESHOLDS[diff][0]
    for diff in SlopeConfig.DIFFICULTY_TARGETS
), "Steep targets must be above lower threshold"


class PathConfig:
    """Path generation domain values the user owns (not algorithm tuning knobs)."""

    # Segment length controls (slider range in UI)
    SEGMENT_LENGTH_MIN_M = 100
    SEGMENT_LENGTH_MAX_M = 1000
    SEGMENT_LENGTH_DEFAULT_M = 500

    # Minimum path points for valid path (less = terrain edge or error)
    MIN_PATH_POINTS = 4

    # Roads for cars: the hard gradient cap enforced at build time.
    # Cars may climb, descend, or run flat, but a proposal over this is refused.
    # (Roads AIM for the green grades 7%/12% — see PathFactory.generate_manual_paths.)
    ROAD_MAX_GRADIENT_PCT = 15


class GeometricTuningConfig:
    """Machine-tunable knobs that shape generated route geometry."""

    # --- Grid-Dijkstra planner (connection_planners.py) ---
    GRID_RESOLUTION_M = 15.0  # Grid cell size in meters
    GRID_BUFFER_FACTOR = 1.0  # Lateral room around the direct start→target line, as a fraction of that distance
    MAX_GRID_SIZE = 100  # Maximum grid cells per dimension (performance cap)
    COST_SIGMA = 8.0  # Slope-deviation sensitivity in the edge cost (lower = stricter grade matching)
    PATH_SIMILARITY_TOLERANCE = 0.0001  # Overlap-dedup tolerance (~0.0001° ≈ 10m at mid-latitudes)

    # --- Fan tracer (path_tracer.py) + fan breadth (path_factory.py) ---
    STEP_SIZE_M = 30  # Path trace / terrain-sample / node-snap step (smaller = smoother, slower)
    MIN_TRAVERSE_ANGLE_DEG = 2  # Ensures left/right paths diverge on gentle terrain
    MAX_TRAVERSE_ANGLE_DEG = 89  # Physical limit (near-horizontal traverse)
    MAX_TURN_PER_STEP_DEG = 40.0  # Max angular change per step to prevent self-intersection
    BEARING_SMOOTHING_WINDOW = 4  # Number of recent bearings to average when smoothing
    FLAT_TERRAIN_THRESHOLD_PCT = 15.0  # Below this slope %, use bearing smoothing (no clear fall line)
    BEARING_SMOOTHING_WEIGHT = 0.8  # Weight of the averaged bearing vs terrain bearing on flat terrain
    MAX_CENTER_PATHS = 4  # Center-stop rule: stop after this many center paths (DETAILS.md §5.4)
    TRACER_NOISE_BASE = 5.0  # Base Gaussian bearing noise (deg), scaled down by traverse angle
    STEP_TARGET_CLAMP_FACTOR = 2.5  # Upper clamp on per-step target grade = FACTOR × target (self-correction bound)

    # --- Whole-path finish smoothing (path_smoothing.py) ---
    RESAMPLE_STEP_M = 7.0  # Output point spacing of the smoothed polyline (tuned: 10m+ aliases sharp turns)
    # splprep s = FACTOR * point_count; higher averages corridor jitter into a broad radius
    # Roads need smooth curves for cars; slopes hug terrain more so they smooth less.
    ROAD_SMOOTHING_FACTOR = 50.0
    SLOPE_SMOOTHING_FACTOR = 15.0
    # Node weight vs corridor weight in the weighted spline fit.
    NODE_WEIGHT = 10.0  # Smooth spline should mathc very well at nodes
    CORRIDOR_WEIGHT = 1.0  # In between path points are less stricly used for attraction


class EarthworkConfig:
    """Earthwork warning thresholds and belt width limits (DETAILS.md Section 4)."""

    # Side cut excavation threshold (meters of vertical cut)
    # H_edge = (S_side × W_belt) / 200 > threshold triggers warning
    EXCAVATOR_THRESHOLD_M = 2.5

    # Belt width limits per difficulty (min_m, max_m)
    # Varies by difficulty to match typical ski run widths
    BELT_WIDTH_LIMITS = {
        "green": (10, 25),  # Narrow beginner runs
        "blue": (20, 35),  # Standard intermediate width
        "red": (25, 40),  # Wide advanced runs for carving
        "black": (20, 35),  # Narrower expert terrain
    }

    # Roads are a fixed-width vehicle ribbon — unlike ski pistes
    ROAD_WIDTH_M = 12


assert set(EarthworkConfig.BELT_WIDTH_LIMITS.keys()) == set(SlopeConfig.DIFFICULTIES)


class ConnectionConfig:
    """Connection path parameters for manual "Connect to Custom Point" feature.

    User workflow:
    1. User clicks "Connect to Custom Point" button
    2. User clicks target on map (node or free point)
    3. System validates: downhill by MIN_DROP_M + within segment_length
    4. Connection paths are generated if valid
    """

    # Minimum elevation drop to target (must go meaningfully downhill)
    MIN_DROP_M = 5


class PlannerConfig:
    """Structural constants for the grid-based Dijkstra planner.

    Reference: DETAILS.md Section 7 for algorithm details.
    """

    # 8-connected grid neighbor directions
    NEIGHBORS_8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


class MarkerConfig:
    """Static marker parameters for Pydeck map UI feedback.

    Controls directional arrows, target markers, and station indicators.
    Colors are RGBA lists for Pydeck GPU rendering.
    """

    # Direction arrow for custom connect (downhill) and lift placement (uphill)
    DIRECTION_ARROW_COLOR_DOWNHILL = [34, 197, 94, 230]  # Green - going down (slopes)
    DIRECTION_ARROW_COLOR_UPHILL = [168, 85, 247, 230]  # Purple - going up (lifts)
    DIRECTION_ARROW_LENGTH_M = 300  # Arrow length in meters
    DIRECTION_ARROW_WIDTH = 8  # Line width for PathLayer
    DIRECTION_CENTER_MARKER_RADIUS = 12  # Radius for center marker at click point (meters for ScatterplotLayer)

    # Lift station marker
    LIFT_STATION_COLOR = [168, 85, 247, 230]  # Purple
    LIFT_STATION_RADIUS = 25  # Meters for ScatterplotLayer

    # Orientation arrows (fall line compass at selection point)
    ORIENTATION_ARROW_LENGTH_M = 80
    ORIENTATION_CONTOUR_COLOR = [156, 163, 175, 200]  # Light gray

    # Node marker styling
    NODE_MARKER_COLOR = [229, 231, 235, 220]  # Near-white for nodes (most visible)
    NODE_MARKER_BORDER = [100, 100, 100, 255]  # Gray border

    # Pylon marker styling
    PYLON_MARKER_COLOR = [107, 114, 128, 230]  # Gray-500 fill
    PYLON_BORDER_COLOR = [31, 41, 55, 255]  # Gray-800 border

    # Cable line styling
    CABLE_WIDTH = 10  # Width in meters for better visibility
    CABLE_MIN_PIXELS = 3  # Minimum visible width when zoomed out

    # Z-offset for marker elevation to prevent z-fighting with terrain
    # Markers/paths rendered this height above DEM elevation (meters)
    # Smaller offsets (10m) work with top-down view; nodes slightly higher for clickability
    MARKER_Z_OFFSET_M = 20

    # Z-offset for paths/lines above terrain to prevent z-fighting
    PATH_Z_OFFSET_M = 10


class ClickConfig:
    """Click detection configuration for Pydeck picking.

    Pydeck uses object picking instead of tooltips for click detection.
    Objects contain type and ID fields for identification.
    """

    # Pydeck picking configuration
    PICKING_RADIUS_PX = 8  # Pixels radius for click detection (5-10 ideal for nodes on lines)

    # Object type identifiers (used in layer data for picking)
    TYPE_TERRAIN = "terrain"  # Invisible layer for terrain clicks
    TYPE_NODE = "node"
    TYPE_SEGMENT = "segment"
    TYPE_SLOPE = "slope"
    TYPE_ROAD = "road"
    TYPE_LIFT = "lift"
    TYPE_PYLON = "pylon"
    TYPE_PROPOSAL_ENDPOINT = "proposal_endpoint"
    TYPE_PROPOSAL_BODY = "proposal_body"
    TYPE_START_MARKER = "start_marker"  # non-interactive origin dot on a proposal
    TYPE_IMPORT_CENTER = "import_center"  # clickable center dot of the OSM import box (re-click = confirm)

    # Clickable marker radii (meters for Pydeck ScatterplotLayer)
    NODE_MARKER_RADIUS = 35
    PARKING_MARKER_RADIUS = 50  # Parking nodes render bigger than plain nodes
    SLOPE_ICON_MARKER_RADIUS = 30
    ROAD_ICON_MARKER_RADIUS = 30
    LIFT_ICON_MARKER_RADIUS = 30
    PYLON_MARKER_RADIUS = 15
    PROPOSAL_BODY_RADIUS = 20
    PROPOSAL_ENDPOINT_RADIUS = 25

    # Colors for interactive elements (RGBA for Pydeck)
    PROPOSAL_ENDPOINT_COLOR = [249, 115, 22, 230]  # Orange-500

    DEBOUNCE_TIME_DELAY = 0.15  # Minimum time between clicks (150ms debounce)


class LiftConfig:
    """Lift types and catenary/pylon parameters."""

    # Terrain sampling step size for lift paths (meters)
    TERRAIN_SAMPLE_STEP_M = 30

    PYLON_CONFIG = {
        "surface_lift": {
            "pylon_height_m": 15,
            "station_height_m": 5,
            "min_spacing_m": 10,
            "max_spacing_m": 100,
            "min_clearance_m": 10,
            "sag_factor": 0.05,
        },
        "chairlift": {
            "pylon_height_m": 25,
            "station_height_m": 6,
            "min_spacing_m": 15,
            "max_spacing_m": 200,
            "min_clearance_m": 15,
            "sag_factor": 0.06,
        },
        "gondola": {
            "pylon_height_m": 35,
            "station_height_m": 6,
            "min_spacing_m": 20,
            "max_spacing_m": 300,
            "min_clearance_m": 20,
            "sag_factor": 0.06,
        },
        "aerial_tram": {
            "pylon_height_m": 60,
            "station_height_m": 10,
            "min_spacing_m": 30,
            "max_spacing_m": 1e6,  # Can span very long distances
            "min_clearance_m": 30,
            "sag_factor": 0.06,
        },
    }
    TYPES = list(PYLON_CONFIG.keys())


class StyleConfig:
    """Visual colors and styling."""

    # Slope colors - Hex for Plotly charts
    SLOPE_COLORS = {
        "green": "#22C55E",  # green-500
        "blue": "#3B82F6",  # blue-500
        "red": "#EF4444",  # red-500
        "black": "#1F2937",  # gray-800
    }
    assert set(SLOPE_COLORS.keys()) == set(SlopeConfig.DIFFICULTIES)

    # Slope colors - RGBA lists for Pydeck (GPU-compatible format)
    SLOPE_COLORS_RGBA = {
        "green": [34, 197, 94, 200],  # #22C55E with alpha
        "blue": [59, 130, 246, 200],  # #3B82F6
        "red": [239, 68, 68, 200],  # #EF4444
        "black": [31, 41, 55, 255],  # #1F2937 (full opacity for contrast)
    }
    assert set(SLOPE_COLORS_RGBA.keys()) == set(SlopeConfig.DIFFICULTIES)

    # Difficulty emoji mapping
    DIFFICULTY_EMOJIS = {
        "green": "🟢",
        "blue": "🔵",
        "red": "🔴",
        "black": "⚫",
    }
    assert set(DIFFICULTY_EMOJIS.keys()) == set(SlopeConfig.DIFFICULTIES)

    # Slope icon for map display and build-mode selector
    SLOPE_ICON = "⛷️"

    # Lift colors - Hex for Plotly
    LIFT_COLORS = {
        "surface_lift": "#D8B4FE",  # Light purple
        "chairlift": "#A855F7",  # Bright purple
        "gondola": "#6B21A8",  # Dark plum
        "aerial_tram": "#7C3AED",  # Vibrant purple
    }
    assert set(LIFT_COLORS.keys()) == set(LiftConfig.TYPES)

    # Lift colors - RGBA lists for Pydeck
    LIFT_COLORS_RGBA = {
        "surface_lift": [216, 180, 254, 200],
        "chairlift": [168, 85, 247, 200],
        "gondola": [107, 33, 168, 200],
        "aerial_tram": [124, 58, 237, 200],
    }
    assert set(LIFT_COLORS_RGBA.keys()) == set(LiftConfig.TYPES)

    # Lift icons for map display
    LIFT_ICONS = {
        "surface_lift": "🎿",
        "chairlift": "🪑",
        "gondola": "🚡",
        "aerial_tram": "🚠",
    }
    assert set(LIFT_ICONS.keys()) == set(LiftConfig.TYPES)

    # Road (for cars): warm brown-orange, clearly visible against terrain and
    # distinct from difficulty-colored slopes and purple lifts.
    ROAD_COLOR = "#B45309"  # amber-700, for Plotly charts
    ROAD_COLOR_RGBA = [180, 83, 9, 230]  # amber-700 for Pydeck
    # Road PROPOSAL (dashed browse path before commit): lighter, translucent amber-600
    # so "proposed" reads as distinct from a committed solid amber-700 road.
    ROAD_PROPOSAL_COLOR_RGBA = [217, 119, 6, 150]
    ROAD_ICON = "🛣️"

    # OSM import mode icon (build-mode selector + placement markers)
    IMPORT_ICON = "🗺️"
    # Node-merge mode icon (build-mode selector).
    MERGE_ICON = "🔗"
    # Generic sidebar-header icons: one "in-progress" glyph for every building/placing state, one for
    # viewing a finished entity. Shared so all state headers stay consistent from one source.
    BUILDING_ICON = "🏗️"
    VIEWING_ICON = "👁️"
    # Nodes selected for merging render solid red so the collapse set is unmistakable.
    MERGE_SELECTED_RGBA = [239, 68, 68, 235]
    # OSM import overlay (RGBA for Pydeck): one blue for the box, one for the center dot.
    IMPORT_BOX_RGBA = [33, 150, 243, 60]  # translucent square (fill + outline)
    IMPORT_CENTER_RGBA = [33, 150, 243, 230]  # solid center dot (click to confirm)

    # Parking place (auto-shown where a road meets a slope or lift)
    PARKING_ICON = "🅿️"
    PARKING_COLOR_RGBA = [96, 165, 250, 180]  # soft blue, gently shown

    # Human-friendly lift display names (includes "slope" for unified build type selector)
    LIFT_DISPLAY_NAMES = {
        "slope": "Slope",
        "surface_lift": "Surface Lift",
        "chairlift": "Chairlift",
        "gondola": "Gondola",
        "aerial_tram": "Aerial Tram",
    }
    assert set(LIFT_DISPLAY_NAMES.keys()) == set(LiftConfig.TYPES) | {"slope"}


class NameConfig:
    """Creative naming components for slopes and lifts."""

    # Slope name prefixes by difficulty
    SLOPE_PREFIXES = {
        "green": ["Gentle", "Easy", "Nursery", "Bunny", "Beginner's", "Soft"],
        "blue": ["Cruiser", "Scenic", "Alpine", "Mountain", "Valley", "Classic"],
        "red": ["Bold", "Thunder", "Steep", "Expert's", "Challenge", "Pro"],
        "black": ["Extreme", "Death", "Kamikaze", "Insane", "Devil's", "Daredevil"],
    }
    assert set(SLOPE_PREFIXES.keys()) == set(SlopeConfig.DIFFICULTIES)

    SLOPE_SUFFIXES = [
        "Run",
        "Trail",
        "Slope",
        "Descent",
        "Pass",
        "Chute",
        "Face",
        "Bowl",
        "Gully",
        "Ridge",
        "Drop",
        "Way",
        "Line",
        "Route",
        "Pitch",
        "Section",
    ]

    # Road name components (roads have no difficulty — creative geography words)
    ROAD_PREFIXES = [
        "Alpine",
        "Valley",
        "Ridgeline",
        "Serpentine",
        "Highland",
        "Forest",
        "Mountain",
        "Meadow",
        "Glacier",
        "Summit",
    ]

    ROAD_SUFFIXES = [
        "Road",
        "Route",
        "Way",
        "Drive",
        "Pass",
        "Trail",
        "Access",
        "Lane",
    ]

    # Lift name prefixes by type
    LIFT_PREFIXES = {
        "surface_lift": ["Bunny", "Beginner's", "Practice", "Easy", "Learner's", "First"],
        "chairlift": ["Alpine", "Mountain", "Scenic", "Valley", "Ridge", "Summit"],
        "gondola": ["Panorama", "Vista", "Sky", "Peak", "Grand", "Majestic"],
        "aerial_tram": ["Eagle", "Falcon", "Summit", "Peak", "Apex", "Pinnacle"],
    }
    assert set(LIFT_PREFIXES.keys()) == set(LiftConfig.TYPES)

    LIFT_SUFFIXES = {
        "surface_lift": ["Tow", "Lift", "Pull", "Rope", "Drag", "Line"],
        "chairlift": ["Chair", "Lift", "Express", "Quad", "Six", "Flyer"],
        "gondola": ["Gondola", "Cabin", "Tram", "Link", "Connect", "Cruiser"],
        "aerial_tram": ["Tram", "Cable Car", "Aerial", "Skyway", "Tramway", "Rise"],
    }
    assert set(LIFT_SUFFIXES.keys()) == set(LiftConfig.TYPES)

    # Length descriptors for lift naming
    LENGTH_DESCRIPTORS = {
        "short": ["Little", "Mini", "Short"],  # < LENGTH_SHORT_MAX_M
        "medium": ["Classic", "Standard", "Regular"],  # between short and long
        "long": ["Grand", "Big", "Giant"],  # > LENGTH_LONG_MIN_M
    }
    # Length/rise bands for entity naming (single source for lift + slope generate_name).
    LENGTH_SHORT_MAX_M = 500  # below → "short" descriptor
    LENGTH_LONG_MIN_M = 1500  # above → "long" descriptor
    SUMMIT_RISE_M = 500  # lift rise / slope drop above this → "Summit" name
    BIG_DROP_M = 300  # slope drop above this → "big" descriptor

    # 8-point compass directions for naming
    COMPASS_DIRECTIONS = {
        "N": (337.5, 22.5),
        "NE": (22.5, 67.5),
        "E": (67.5, 112.5),
        "SE": (112.5, 157.5),
        "S": (157.5, 202.5),
        "SW": (202.5, 247.5),
        "W": (247.5, 292.5),
        "NW": (292.5, 337.5),
    }
    assert len(COMPASS_DIRECTIONS) == 8

    @staticmethod
    def get_compass_direction(bearing_deg: float) -> str:
        """Get compass direction name from bearing.

        Args:
            bearing_deg: Bearing in degrees (0-360)

        Returns:
            Compass direction string (N, NE, E, SE, S, SW, W, NW)
        """
        brg = bearing_deg % 360
        for direction, (low, high) in NameConfig.COMPASS_DIRECTIONS.items():
            if direction == "N":
                if brg >= low or brg < high:
                    return direction
            elif low <= brg < high:
                return direction
        raise ValueError(f"Invalid bearing: {bearing_deg}")


class ChartConfig:
    """Chart rendering dimensions and settings."""

    # The one elevation-profile chart below the map — same height whether building or
    # viewing a slope/road/lift (it is the same chart in the same place).
    PROFILE_HEIGHT_PX = 260

    # Plot margin shared by every profile figure (slope/road/lift) — one look for all.
    PROFILE_MARGIN = dict(l=50, r=30, t=50, b=50)

    # Map height = browser window (parent.innerHeight) minus the chrome above the map,
    # minus the profile's own height when a profile is shown below it.
    MAP_TOP_OFFSET_PX = 40  # Where the map starts: Streamlit block padding + column gap
    MAP_MIN_HEIGHT_PX = 400  # Never shrink the map below this, even on short windows

    # Y-axis padding settings
    ELEVATION_PADDING_FACTOR = 0.1  # 10% padding above/below
    ELEVATION_PADDING_MIN_M = 20  # Minimum padding in meters
    LIFT_ELEVATION_PADDING_FACTOR = 0.15
    LIFT_ELEVATION_PADDING_MIN_M = 30


class UndoConfig:
    """Undo system configuration."""

    # Maximum number of actions to keep in undo stack
    # Older actions are discarded when limit is reached
    MAX_UNDO_STACK_SIZE = 50


class MergeConfig:
    """Manual node-merge tool (collapse scattered station nodes into one)."""

    # Refuse to merge if any two selected nodes are farther apart than this
    MAX_SPAN_M = 500.0


class OSMConfig:
    """OpenStreetMap import (generators/osm_importer.py).

    We take GEOMETRY ONLY from OSM (where lifts/pistes are); elevation, difficulty, and pylons
    are all recomputed from our own DEM + physics. OSM's difficulty/pylon/elevation tags are
    deliberately ignored.
    """

    OVERPASS_URL = "https://overpass-api.de/api/interpreter"
    OVERPASS_STATUS_URL = "https://overpass-api.de/api/status"
    OVERPASS_TIMEOUT_S = 30
    # Overpass returns HTTP 406 without a User-Agent (verified live) — always send one.
    USER_AGENT = "ski-resort-designer/0.1"

    # Nominatim free-text place search (generators/geocoder.py) — powers the sidebar search box.
    # Policy: max 1 req/s and a custom User-Agent (we send USER_AGENT); search-on-submit only.
    NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
    NOMINATIM_TIMEOUT_S = 10

    # A lift/piste-only query is light, so the whole region is fetched in ONE query. Overpass gives
    # a few slots per IP; on a transient 429/504 we wait for a free slot (from /api/status) and retry
    # once. SLOT_WAIT_MAX_S caps that wait; SLOT_WAIT_FALLBACK_S is used when status can't be read.
    SLOT_WAIT_MAX_S = 30.0
    SLOT_WAIT_FALLBACK_S = 3.0

    # Output spacing when resampling an OSM polyline onto DEM-sampled points.
    RESAMPLE_STEP_M = 30.0

    # Import region: a square centered on the current map center, half-width chosen on a slider (km).
    HALF_WIDTH_MIN_KM = 0.5
    HALF_WIDTH_MAX_KM = 5.0
    HALF_WIDTH_DEFAULT_KM = 2.0

    # Minimum imported length: shorter entities are ignored (nursery/kiddie lifts, stub runs).
    MIN_LIFT_LENGTH_M = 300.0
    MIN_PISTE_LENGTH_M = 200.0

    # OSM aerialway value → our LiftConfig.TYPES. ONLY these values import; every other aerialway
    # value (station, pylon, zip_line, magic_carpet, rope_tow, yes, …) is silently ignored.
    AERIALWAY_TO_LIFT_TYPE = {
        "drag_lift": "surface_lift",
        "t-bar": "surface_lift",
        "j-bar": "surface_lift",
        "platter": "surface_lift",
        "chair_lift": "chairlift",
        "gondola": "gondola",
        "mixed_lift": "gondola",
        "cable_car": "aerial_tram",
    }

    # piste:type value marking an alpine downhill run — the only kind we import.
    PISTE_TYPE_DOWNHILL = "downhill"

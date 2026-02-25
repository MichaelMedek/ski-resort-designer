"""Configuration constants for Ski Resort Planner.

All configurable parameters are centralized here for easy tuning.
Values are referenced in DETAILS.md.

Classes:
    AppConfig: UI application settings
    MapConfig: Default map view parameters
    DEMConfig: Elevation data file paths
    SlopeConfig: Difficulty thresholds, targets, belt widths
    PathConfig: Path generation algorithm parameters
    EarthworkConfig: Excavation warning thresholds
    ConnectionConfig: Connection path parameters
    PlannerConfig: Grid-based Dijkstra path planner parameters
    MarkerConfig: Map marker styling
    LiftConfig: Lift types and catenary parameters
    StyleConfig: Visual colors and styling
    NameConfig: Creative naming components
    ChartConfig: Chart rendering dimensions
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


class AppConfig:
    """UI application settings."""

    TITLE = "Ski Resort Planner - Design Your Dream Resort"
    ICON = "⛷️"
    LAYOUT: Literal["centered", "wide"] = "wide"


class EntityPrefixes:
    """ID prefixes for graph entities."""

    NODE = "N"
    SEGMENT = "S"
    SLOPE = "SL"
    LIFT = "L"


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
    """Slope classification thresholds, targets, and belt widths."""

    # Core slope limits - single source of truth
    MIN_SKIABLE_PCT = 5  # Below this: need to push poles
    MAX_SKIABLE_PCT = 70  # Above this: dangerously steep

    # European slope difficulty thresholds (by average gradient percentage)
    # Classification: avg_slope = total_drop / total_length * 100
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

    # Belt widths for polygon visualization (meters)
    BELT_WIDTHS = {
        "green": 8,  # Cat track - narrow
        "blue": 25,  # Cruiser - medium
        "red": 40,  # Expert - wide
        "black": 30,  # Variable terrain
    }
    assert set(BELT_WIDTHS.keys()) == set(DIFFICULTIES)


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
    """Path generation algorithm parameters."""

    # Step size for path tracing (meters)
    STEP_SIZE_M = 30  # Smaller = smoother paths, larger = faster computation

    # Segment length controls (slider range in UI)
    SEGMENT_LENGTH_MIN_M = 100
    SEGMENT_LENGTH_MAX_M = 1000
    SEGMENT_LENGTH_DEFAULT_M = 500

    # Traverse angle limits (degrees)
    MIN_TRAVERSE_ANGLE_DEG = 2  # Ensures left/right paths diverge on gentle terrain
    MAX_TRAVERSE_ANGLE_DEG = 89  # Physical limit (near-horizontal traverse)

    # Center-stop rule: stop after this many center paths (DETAILS.md Section 5.4)
    MAX_CENTER_PATHS = 4

    # Minimum path points for valid path (less = terrain edge or error)
    MIN_PATH_POINTS = 4

    # Path tracing behavior parameters (self-intersection and smoothing)
    MAX_TURN_PER_STEP_DEG = 40.0  # Max angular change per step to prevent self-intersection
    BEARING_SMOOTHING_WINDOW = 4  # Number of recent bearings to average for momentum
    FLAT_TERRAIN_THRESHOLD_PCT = 15.0  # Below this slope %, use momentum-based bearing smoothing
    MOMENTUM_WEIGHT_FACTOR = 0.8  # Weight factor for momentum bearing on flat terrain


class EarthworkConfig:
    """Earthwork warning thresholds (DETAILS.md Section 4)."""

    # Side cut excavation threshold (meters of vertical cut)
    # H_edge = (S_side × W_belt) / 200 > threshold triggers warning
    EXCAVATOR_THRESHOLD_M = 2.5


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
    """Grid-based Dijkstra path planner configuration parameters.

    Controls the grid-based path planning algorithm that finds optimal
    routes considering slope preferences and terrain features.

    Uses SciPy's C-optimized Dijkstra followed by cubic spline smoothing.

    Reference: DETAILS.md Section 7 for algorithm details.
    """

    # Grid search parameters
    GRID_RESOLUTION_M = 15.0  # Grid cell size in meters
    GRID_BUFFER_FACTOR = 0.5  # Extra buffer around start-target line (as fraction)
    MAX_GRID_SIZE = 100  # Maximum grid cells per dimension (performance cap)

    # Cost function parameters
    # Cost = distance × exp(slope_deviation / COST_SIGMA) × uphill_penalty
    COST_SIGMA = 8.0  # Slope deviation sensitivity (lower = stricter matching)

    # Path deduplication for overlapping path removal
    # ~0.0001 degrees ≈ ~10 meters at mid-latitudes
    PATH_SIMILARITY_TOLERANCE = 0.0001

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
    CABLE_WIDTH = 4

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
    TYPE_LIFT = "lift"
    TYPE_PYLON = "pylon"
    TYPE_PROPOSAL_ENDPOINT = "proposal_endpoint"
    TYPE_PROPOSAL_BODY = "proposal_body"

    # Clickable marker radii (meters for Pydeck ScatterplotLayer)
    NODE_MARKER_RADIUS = 25
    SLOPE_ICON_MARKER_RADIUS = 20
    PYLON_MARKER_RADIUS = 15
    PROPOSAL_BODY_RADIUS = 20
    PROPOSAL_ENDPOINT_RADIUS = 28

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
            "min_spacing_m": 10,
            "max_spacing_m": 200,
            "min_clearance_m": 15,
            "sag_factor": 0.06,
        },
        "gondola": {
            "pylon_height_m": 35,
            "station_height_m": 6,
            "min_spacing_m": 10,
            "max_spacing_m": 300,
            "min_clearance_m": 20,
            "sag_factor": 0.06,
        },
        "aerial_tram": {
            "pylon_height_m": 60,
            "station_height_m": 10,
            "min_spacing_m": 10,
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
        "short": ["Little", "Mini", "Short"],  # < 500m
        "medium": ["Classic", "Standard", "Regular"],  # 500-1500m
        "long": ["Grand", "Big", "Giant"],  # > 1500m
    }

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

    # Profile chart heights for different contexts
    PROFILE_HEIGHT_LARGE = 550  # Full-width main panel
    PROFILE_HEIGHT_MEDIUM = 320  # Side panel
    PROFILE_HEIGHT_SMALL = 250  # Compact view
    PROFILE_HEIGHT_MINI = 200  # Minimal view

    # Lift profile specific
    LIFT_PROFILE_HEIGHT = 320

    # Chart widths
    DEFAULT_WIDTH = 800  # Default chart width
    WIDE_WIDTH = 1000  # Wide layout width

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


class CoordinateConfig:
    """Configuration for coordinate handling and comparison.

    STRICT: All coordinate comparisons must use these methods,
    NEVER use == for lat/lon floats directly!
    """

    # Decimal places for dedup key generation (6 decimals ≈ 10cm precision)
    DEDUP_KEY_DECIMALS: int = 6

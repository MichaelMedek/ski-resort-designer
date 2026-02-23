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
    LAYOUT = "wide"


class EntityPrefixes:
    """ID prefixes for graph entities."""

    NODE = "N"
    SEGMENT = "S"
    SLOPE = "SL"
    LIFT = "L"


class MapConfig:
    """Default map view parameters."""

    # Initial center for program start: Idalp, Ischgl, Austria
    START_CENTER_LAT = 46.982  # Latitude
    START_CENTER_LON = 10.317  # Longitude

    # Zoom levels for different modes
    # Higher number = more zoomed in, lower = more zoomed out
    BUILDING_ZOOM = 15  # Working zoom for building slopes/lifts (close for selection)
    VIEWING_ZOOM = 14  # Overview after finishing slope/lift (slightly zoomed out)
    DEFAULT_ZOOM = BUILDING_ZOOM  # Default to building zoom to avoid animation during commits

    # Node snapping threshold for lift placement (used when creating end nodes)
    LIFT_END_NODE_THRESHOLD_M = 80  # Extra generous for lift top station placement
    # At equator, 1 degree of latitude or longitude ≈ 111,320 meters
    # (Earth circumference 40,075 km / 360 degrees)
    METERS_PER_DEGREE_EQUATOR = 111320.0


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
    """Static marker parameters for map UI feedback.

    Controls directional arrows, target markers, and station indicators.
    All markers are static (no CSS animations) for simplicity and performance.
    """

    # Direction arrow for custom connect (downhill) and lift placement (uphill)
    DIRECTION_ARROW_COLOR_DOWNHILL = "#22C55E"  # Green - going down (slopes)
    DIRECTION_ARROW_COLOR_UPHILL = "#A855F7"  # Purple - going up (lifts)
    DIRECTION_ARROW_LENGTH_M = 300  # Arrow length in meters
    DIRECTION_ARROW_WIDTH = 4  # Line weight

    # Lift station marker
    LIFT_STATION_COLOR = "#A855F7"  # Purple
    LIFT_STATION_RADIUS = 14

    # Orientation arrows (fall line compass at selection point)
    ORIENTATION_ARROW_LENGTH_M = 80
    ORIENTATION_CONTOUR_COLOR = "#9CA3AF"  # Light gray

    # Node marker styling
    NODE_MARKER_COLOR = "#E5E7EB"  # Near-white for nodes (most visible)

    # Pylon marker styling
    PYLON_MARKER_COLOR = "#6B7280"  # Gray-500 fill
    PYLON_BORDER_COLOR = "#1F2937"  # Gray-800 border


class ClickConfig:
    """Click detection and marker tooltip configuration.

    Direct marker click detection uses machine-readable tooltips embedded
    in user-friendly display text.
    """

    # Tooltip prefixes for marker identification (user-friendly display)
    # Format: "{prefix} {id}" for single ID or "{prefix} {id} on {parent}" for nested
    # IDs use consistent short format: N1 (node), SL1 (slope), L1 (lift), S1 (segment)
    TOOLTIP_PREFIX_NODE = "Build From Node"  # "Build From Node N1"
    TOOLTIP_PREFIX_SLOPE_ICON = "View Slope"  # "View Slope SL1"
    TOOLTIP_PREFIX_LIFT_ICON = "View Lift"  # "View Lift L1"
    TOOLTIP_PREFIX_PYLON = "View Pylon"  # "View Pylon 1 on L1" (1-indexed for user display)
    TOOLTIP_PREFIX_PROPOSAL_END = "Commit Proposal"  # "Commit Proposal 1" (1-indexed)
    TOOLTIP_PREFIX_PROPOSAL_BODY = "Select Proposal"  # "Select Proposal 1" (1-indexed)

    # Separator for nested IDs (e.g., "Pylon 3 on Lift 1")
    TOOLTIP_SEPARATOR_ON = " on "

    # Clickable marker radii (pixels) - Colors are slope colors if applicable
    NODE_MARKER_RADIUS = 7
    SLOPE_ICON_MARKER_RADIUS = 8
    PYLON_MARKER_RADIUS = 5
    PROPOSAL_BODY_RADIUS = 6
    PROPOSAL_ENDPOINT_RADIUS = 8
    PROPOSAL_ENDPOINT_COLOR = "#F97316"  # Orange-500


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
            "max_spacing_m": None,  # Can span very long distances
            "min_clearance_m": 30,
            "sag_factor": 0.06,
        },
    }
    TYPES = list(PYLON_CONFIG.keys())


class StyleConfig:
    """Visual colors and styling."""

    # Slope colors (Tailwind CSS palette)
    SLOPE_COLORS = {
        "green": "#22C55E",  # green-500
        "blue": "#3B82F6",  # blue-500
        "red": "#EF4444",  # red-500
        "black": "#1F2937",  # gray-800
    }
    assert set(SLOPE_COLORS.keys()) == set(SlopeConfig.DIFFICULTIES)

    # Difficulty emoji mapping
    DIFFICULTY_EMOJIS = {
        "green": "🟢",
        "blue": "🔵",
        "red": "🔴",
        "black": "⚫",
    }
    assert set(DIFFICULTY_EMOJIS.keys()) == set(SlopeConfig.DIFFICULTIES)

    # Lift colors
    LIFT_COLORS = {
        "surface_lift": "#D8B4FE",  # Light purple
        "chairlift": "#A855F7",  # Bright purple
        "gondola": "#6B21A8",  # Dark plum
        "aerial_tram": "#7C3AED",  # Vibrant purple
    }
    assert set(LIFT_COLORS.keys()) == set(LiftConfig.TYPES)

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

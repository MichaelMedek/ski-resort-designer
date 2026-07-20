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

from enum import StrEnum
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
    VIEWING_ZOOM = 13  # Overview after finishing slope/lift (zoomed out)
    VIEW_3D_ZOOM = 14  # 3D side view - balanced zoom
    VIEW_3D_MIN_ZOOM = 12  # Minimum zoom for high elevation (prevents camera under terrain)
    DEFAULT_ZOOM = VIEWING_ZOOM  # Start zoomed out to prevent camera clipping terrain
    IMPORT_OVERVIEW_ZOOM = 12  # Post-import overview: one step further out than building zoom

    # Pitch angles for different modes
    # Use 0 (top-down) for all modes to ensure accurate terrain clicks
    VIEWING_PITCH = 0  # Top-down view for viewing (tilted views cause terrain click issues)
    VIEW_3D_PITCH = 25  # 25° angle for 3D - more from above to avoid mountains blocking view
    DEFAULT_PITCH = 0  # Always start top-down
    DEFAULT_BEARING = 0  # Map rotation in degrees (0 = north up)

    # Node snapping threshold for lift placement (used when creating end nodes)
    LIFT_END_NODE_THRESHOLD_M = 80  # Extra generous for lift top station placement

    # Metres per degree of latitude (and of longitude at the equator), on the WGS84 spherical earth
    # (EARTH_RADIUS_M · π/180 ≈ 111195 m). Single source for all lat/lon↔metre conversions codebase-wide.
    # test_geo_calculator asserts this equals GeoCalculator.haversine_distance_m(0,0,1,0) exactly.
    METERS_PER_DEGREE_EQUATOR = 111320.0

    # 2D mode z-offsets (relative layer ordering, no terrain)
    # Small offsets prevent z-fighting while keeping flat appearance
    # Z-offsets for 2D mode - small values for proper layer ordering
    Z_OFFSET_2D_SLOPES = 1  # Slope polygons at base
    Z_OFFSET_2D_ROADS = 2  # Road polygons just above slopes (matches slopes→roads z-order)
    Z_OFFSET_2D_LIFTS = 3  # Lift cables above roads
    Z_OFFSET_2D_PYLONS = 4  # Pylons slightly above lift cables
    Z_OFFSET_2D_ICONS = 5  # Slope/road/lift icons above pylons
    Z_OFFSET_2D_NODE_BIG = 9  # Merge-selected or parking place (big red or blue) nodes just BELOW plain nodes
    Z_OFFSET_2D_NODES = 10  # Nodes above icons
    Z_OFFSET_2D_MARKERS = 20  # Interactive markers (commit/select) on top

    # Flat-mode z-offset per committed-segment kind, keyed by SegmentKind value. Keyed by
    # the string value (not the enum) to stay reload-safe and avoid a model import here.
    # Kept in sync with SegmentKind by an assert in ui/kind_spec.py.
    SEGMENT_FLAT_Z = {
        "slope": Z_OFFSET_2D_SLOPES,
        "road": Z_OFFSET_2D_ROADS,
    }


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

    # Safety bias added to a PROPOSAL's steepness before classifying, so its previewed
    # difficulty rounds toward the harder band.
    SLOPE_DIFFICULTY_MARGIN_PCT = 2.0

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
    # Every target sub-dict must define exactly the two grade knobs path generation reads.
    assert all(set(v.keys()) == {"gentle", "steep"} for v in DIFFICULTY_TARGETS.values()), (
        "each DIFFICULTY_TARGETS entry must define exactly {'gentle', 'steep'}"
    )

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

# DIFFICULTIES order is LOAD-BEARING: path_factory uses DIFFICULTIES[:-1] ("all but the hardest"),
# so the last entry MUST be the steepest. Enforce strictly-ascending thresholds.
_ordered_lowers = [SlopeConfig.DIFFICULTY_THRESHOLDS[d][0] for d in SlopeConfig.DIFFICULTIES]
_ordered_uppers = [SlopeConfig.DIFFICULTY_THRESHOLDS[d][1] for d in SlopeConfig.DIFFICULTIES]
assert _ordered_lowers == sorted(_ordered_lowers) and _ordered_uppers == sorted(_ordered_uppers), (
    f"DIFFICULTIES must be ordered easiest→hardest (strictly ascending thresholds); "
    f"got lowers={_ordered_lowers}, uppers={_ordered_uppers}"
)


class PathConfig:
    """Path generation domain values the user owns (not algorithm tuning knobs)."""

    # Segment length controls (slider range in UI)
    SEGMENT_LENGTH_MIN_M = 100
    SEGMENT_LENGTH_MAX_M = 1500
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
    MAX_GRID_SIZE = 320  # Max grid cells per dimension — covers a 1500m path at 15m resolution
    COST_SIGMA = 8.0  # Slope-deviation sensitivity in the edge cost (lower = stricter grade matching)
    PATH_SIMILARITY_TOLERANCE = 0.0001  # Overlap-dedup tolerance (~0.0001° ≈ 10m at mid-latitudes)

    # --- Fan tracer (path_tracer.py) + fan breadth (path_factory.py) ---
    STEP_SIZE_M = 30  # Path trace / terrain-sample / node-snap step (smaller = smoother, slower)
    # Weighted "Magic 8" gradient (terrain_analyzer.compute_gradient): two concentric rings sampled at
    # 8 compass bearings. (radius_factor × STEP_SIZE_M, weight) — inner ring weighted double.
    GRADIENT_RINGS = [(0.5, 2.0), (1.0, 1.0)]
    GRADIENT_SAMPLE_ANGLES_DEG = [0, 45, 90, 135, 180, 225, 270, 315]
    MIN_TRAVERSE_ANGLE_DEG = 2  # Keeps left/right paths diverging and the traverse off straight up/down
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
    SLOPE_SMOOTHING_FACTOR = 30.0
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
    """Connection path parameters for manual "Connect to Custom Point".

    User clicks a target (node/free point); a path is generated if it's downhill by MIN_DROP_M
    and within segment_length.
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
    NODE_MARKER_RADIUS_BIG = 50  # Parking and merge nodes render bigger than plain nodes
    SLOPE_ICON_MARKER_RADIUS = 30
    ROAD_ICON_MARKER_RADIUS = 30
    LIFT_ICON_MARKER_RADIUS = 30
    PYLON_MARKER_RADIUS = 15
    PROPOSAL_BODY_RADIUS = 20
    PROPOSAL_ENDPOINT_RADIUS = 25

    # Colors for interactive elements (RGBA for Pydeck)
    PROPOSAL_ENDPOINT_COLOR = [249, 115, 22, 230]  # Orange-500

    DEBOUNCE_TIME_DELAY = 0.15  # Minimum time between clicks (150ms debounce)


class LiftType(StrEnum):
    """The kinds of lift the app builds. A StrEnum, so a member IS its string value — every dict
    keyed/valued by these (PYLON_CONFIG, LIFT_ICONS, AERIALWAY_TO_LIFT_TYPE, …), every JSON round-trip,
    and str(LiftType.CHAIRLIFT) == "chairlift" all work transparently.
    """

    SURFACE_LIFT = "surface_lift"
    CHAIRLIFT = "chairlift"
    GONDOLA = "gondola"
    AERIAL_TRAM = "aerial_tram"


class EntitySource(StrEnum):
    """Provenance tag for an imported entity. A StrEnum, so a member IS its string value — it stores
    directly on Lift/Slope/PathSegment.source and JSON round-trips transparently. Hidden from the
    user; used to recognise (and skip re-importing) OSM-sourced lifts/slopes.
    """

    OSM = "OSM"


class OSMImportMode(StrEnum):
    """Which OSM import the user requested: lifts only (raw, fast) or the full connected graph."""

    LIFTS_ONLY = "lifts_only"
    LIFTS_AND_SLOPES = "lifts_and_slopes"


class LiftConfig:
    """Lift types and catenary/pylon parameters."""

    # Terrain sampling step size for lift paths (meters)
    TERRAIN_SAMPLE_STEP_M = 30

    PYLON_CONFIG = {
        LiftType.SURFACE_LIFT: {
            "pylon_height_m": 15,
            "station_height_m": 5,
            "min_spacing_m": 10,
            "max_spacing_m": 100,
            "min_clearance_m": 10,
            "sag_factor": 0.05,
        },
        LiftType.CHAIRLIFT: {
            "pylon_height_m": 25,
            "station_height_m": 6,
            "min_spacing_m": 15,
            "max_spacing_m": 200,
            "min_clearance_m": 15,
            "sag_factor": 0.06,
        },
        LiftType.GONDOLA: {
            "pylon_height_m": 35,
            "station_height_m": 6,
            "min_spacing_m": 20,
            "max_spacing_m": 300,
            "min_clearance_m": 20,
            "sag_factor": 0.06,
        },
        LiftType.AERIAL_TRAM: {
            "pylon_height_m": 60,
            "station_height_m": 10,
            "min_spacing_m": 30,
            "max_spacing_m": 1e6,  # Can span very long distances
            "min_clearance_m": 30,
            "sag_factor": 0.06,
        },
    }
    # Lift-type strings, in canonical order — derived from the authoritative LiftType enum. Members
    # are str-Enum so this list also behaves as plain strings for callers that compare/serialize.
    TYPES = [t.value for t in LiftType]

    # Whether a lift type carries riders one way (uphill only) or both ways. The single source the
    # directed ski-graph reads for edge direction — gondolas and trams run both ways, drags/chairs up.
    UPHILL_ONLY = {
        LiftType.SURFACE_LIFT: True,
        LiftType.CHAIRLIFT: True,
        LiftType.GONDOLA: False,
        LiftType.AERIAL_TRAM: False,
    }

    # Every lift type must define the full set of pylon-placement knobs the builder reads.
    assert all(
        set(v.keys())
        == {"pylon_height_m", "station_height_m", "min_spacing_m", "max_spacing_m", "min_clearance_m", "sag_factor"}
        for v in PYLON_CONFIG.values()
    ), "each PYLON_CONFIG entry must define exactly the 6 pylon-placement keys"
    # PYLON_CONFIG must be keyed by every LiftType member (bijection: no type missing, none stray).
    assert set(PYLON_CONFIG) == set(LiftType), "PYLON_CONFIG must have one entry per LiftType member"
    # UPHILL_ONLY must cover every LiftType member (same bijection guarantee as PYLON_CONFIG).
    assert set(UPHILL_ONLY) == set(LiftType), "UPHILL_ONLY must have one entry per LiftType member"


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
        "chairlift": "💺",
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
    # Route-planner mode icon (build-mode selector).
    ROUTE_ICON = "🧭"
    # Generic sidebar-header icons: one "in-progress" glyph for every building/placing state, one for
    # viewing a finished entity. Shared so all state headers stay consistent from one source.
    BUILDING_ICON = "🏗️"
    VIEWING_ICON = "👁️"
    # A selected node (merge/delete candidate or route start) renders solid red so the selection is clear.
    SELECTED_NODE_RGBA = [239, 68, 68, 235]
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


class ConnectivityConfig:
    """Core-resort connectivity thresholds (model/connectivity.py, ResortGraph.get_core_resort)."""

    # A strongly-connected component must hold at least this many lifts before it counts as the
    # core resort — below it we assume the resort is still being started and flag nothing.
    MIN_CORE_LIFTS = 5


class RoutePlannerConfig:
    """Route planner (model/routing.py + ui route views): overlay colours + line geometry."""

    # RGBA per route criterion, keyed by RouteCriterion's string value.
    # Hue = the metric (cyan = fewest lifts, gold = shortest slope); the SCENIC tour of each metric
    # is a DARKER tone of the same hue, so path length alone reads shortest-vs-scenic. Semi-transparent so
    # the slope colour shows through. model/routing.py asserts this covers every RouteCriterion.
    ROUTE_COLORS = {
        "fewest_lifts": [0, 200, 210, 150],  # bright cyan
        "shortest_slope": [240, 200, 20, 150],  # bright gold
        "scenic_fewest_lifts": [0, 110, 120, 170],  # deep teal — cyan, darker tone
        "scenic_shortest_slope": [150, 120, 10, 170],  # dark amber — gold, darker tone
    }

    # "Shortest slope" is primarily least slope distance; drop is folded in with a light weight so a
    # gentler descent breaks near-ties (distance and drop are usually the same route anyway).
    SHORTEST_SLOPE_DROP_WEIGHT = 0.1

    # The route line is drawn WIDER than any slope belt (EarthworkConfig max ≈ 35m) so it reads as an
    # overlay on top of the pistes, not another run. PathLayer get_width is in metres (deck.gl default).
    ROUTE_WIDTH_M = 60

    # In 3D, the route floats this far above the terrain/piste points so it hovers clearly above the
    # slopes and lifts it traces, rather than z-fighting with them.
    ROUTE_FLOAT_ABOVE_M = 100

    # Route Steps groups the slopes between two lifts into one leg; a leg names at most this many slopes,
    # then a trailing "…" if it has more. Keeps the step list short, especially on scenic tours.
    ROUTE_STEP_SLOPE_PREVIEW = 3


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
    MIN_PISTE_LENGTH_M = 30.0

    # OSM aerialway value → our LiftConfig.TYPES. ONLY these values import; every other aerialway
    # value (station, pylon, zip_line, magic_carpet, rope_tow, yes, …) is silently ignored.
    AERIALWAY_TO_LIFT_TYPE = {
        "drag_lift": LiftType.SURFACE_LIFT,
        "t-bar": LiftType.SURFACE_LIFT,
        "j-bar": LiftType.SURFACE_LIFT,
        "platter": LiftType.SURFACE_LIFT,
        "chair_lift": LiftType.CHAIRLIFT,
        "gondola": LiftType.GONDOLA,
        "mixed_lift": LiftType.GONDOLA,
        "cable_car": LiftType.AERIAL_TRAM,
    }

    # piste:type value marking an alpine downhill run — the only kind we import.
    PISTE_TYPE_DOWNHILL = "downhill"
    # piste:type value for a connector run — kept for connectivity, not difficulty-filtered.
    PISTE_TYPE_CONNECTION = "connection"

    # Standard groomed downhill grades we import (green→black).
    PISTE_DIFFICULTY_ALLOWED = frozenset({"novice", "easy", "intermediate", "advanced", "expert"})

    # Re-import dedup radius: an incoming entity whose endpoints match an existing one within this is skipped.
    OSM_DEDUP_TOL_M = 100.0

    # --- Connected-graph build (generators/osm_graph_builder.py). Distances in metres. ---
    DEDUP_TOL_M = 18.0  # near-coincidence band for the duplicate-piste test
    PARALLEL_TOL_M = 60.0  # near-but-offset band for the redundant-parallel
    PARALLEL_TWIN_FRAC = 0.70  # a same-name run parallel to a longer sibling ≥this of ITS length is a twin
    DEDUP_COVER_FRAC = 0.78  # covered fraction to call a piste a duplicate
    MIN_NODE_DIST_M = 100.0  # min hub spacing (closer nodes merge)
    RELAXED_MERGE_DIST_M = 200.0  # slope-node→lift pull radius
    MAX_BACKCLIMB_M = 30.0  # max uphill RISE over any BACKCLIMB_WINDOW_M span (60m-DEM sampling-noise tolerance)
    BACKCLIMB_WINDOW_M = 80.0  # window for the strict per-span uphill check
    SLOPE_ON_SOURCE_TOL_M = 30.0  # strict on-piste band (slope body hugs OSM)
    PISTE_TOL_M = 40.0  # off-piste threshold (~a wide piste's half-width)
    MAX_PULL_M = 300.0  # max straight hub connector (longer → drop the segment)
    MAX_STRAIGHT_M = 100.0  # max single straight leg between consecutive points
    TRIM_END_M = 50.0  # trim off each slope end before the hub connector
    SNAP_GRID_M = 12.0  # snap-round grid before noding (collapse near-coincident ends)
    NODE_TERRAIN_TOL_M = 10.0  # max node vs DEM deviation; also the carve depth + descent-carry cap
    SLOPE_TERRAIN_TOL_M = 50.0  # max slope-point vs DEM deviation

    # Consistency
    assert SLOPE_ON_SOURCE_TOL_M < PISTE_TOL_M <= SLOPE_TERRAIN_TOL_M
    assert NODE_TERRAIN_TOL_M < SLOPE_TERRAIN_TOL_M
    assert DEDUP_TOL_M < PARALLEL_TOL_M < MIN_NODE_DIST_M < RELAXED_MERGE_DIST_M

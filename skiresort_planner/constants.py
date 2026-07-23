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

import math
import os
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import platformdirs

if TYPE_CHECKING:
    from skiresort_planner.model.segment_profile import SegmentProfile

# Package root directory (where skiresort_planner/ lives)
PACKAGE_DIR = Path(__file__).parent

# Project root directory (parent of skiresort_planner/)
PROJECT_ROOT = PACKAGE_DIR.parent


def _user_data_root() -> Path:
    """Writable root for DEM/output/backups — a per-user OS app-data dir.
    platformdirs resolves the canonical location per OS (Application Support / %LOCALAPPDATA% / XDG).
    SKIRESORT_DATA_ROOT overrides it (CI and power users) — an external condition, not an internal invariant.
    """
    override = os.environ.get("SKIRESORT_DATA_ROOT")
    if override:
        return Path(override)
    return Path(platformdirs.user_data_dir("AlpinArchitect", appauthor=False))


_DATA_ROOT = _user_data_root()

# Data directory (DEM downloads here on first run — 285MB, not shipped with the package)
DATA_DIR = _DATA_ROOT / "data"

# Output directory for saved graphs / OSM import artifacts
OUTPUT_DIR = _DATA_ROOT / "output"

# Auto-saved resort backups (per-session crash/outage safety net)
BACKUP_DIR = _DATA_ROOT / "backups"

# Timestamped INFO log files (one per run) alongside data/backups
LOG_DIR = _DATA_ROOT / "logs"


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

    # Zoom levels (higher = more zoomed in). Reduced to keep the camera above ground with 3D terrain.
    VIEWING_ZOOM = 13  # Base 2D overview: fits a ~4km build (see zoom_for_span_m) + program start
    VIEW_3D_ZOOM = 13.0  # 3D side view + flythrough — one tuned value, can also be float
    VIEW_3D_ROUTE_ZOOM_OUT = 0.5  # Static 3D route view sits this much further out than VIEW_3D_ZOOM

    # 2D overview zoom adapts to build size around VIEWING_ZOOM; span anchors the ~4km-fits-at-13 base.
    ZOOM_SPAN_ANCHOR_M = 4000.0  # a build this long fits at VIEWING_ZOOM
    ZOOM_STEPS_IN = 2  # clamp: at most this many levels further IN (smaller builds)
    ZOOM_STEPS_OUT = 1  # clamp: at most this many levels further OUT (bigger builds)

    # Pitch angles for different modes
    # Use 0 (top-down) for all modes to ensure accurate terrain clicks
    VIEWING_PITCH = 0  # Top-down view for viewing (tilted views cause terrain click issues)
    VIEW_3D_PITCH = 25  # 25° angle for 3D - more from above to avoid mountains blocking view
    DEFAULT_PITCH = 0  # Always start top-down
    DEFAULT_BEARING = 0  # Map rotation in degrees (0 = north up)

    # Flythrough ("Play") camera: few element-anchored keyframes, glided between by deck.gl client-side.
    FLYTHROUGH_ANCHOR_FRACTION = 0.5  # route: camera sits this far along each element
    FLYTHROUGH_LOOK_AHEAD_M = 900.0  # center this far ahead (nav-style: "here" below centre)
    FLYTHROUGH_STEP_S = 1.0  # sleep between keyframes ≈ glide duration
    FLYTHROUGH_TRANSITION_MS = 1600  # deck.gl glide per keyframe (> STEP_S so it never stalls)
    # Current-element highlight ribbon while flying (hot orange, distinct from route/arrow palettes).
    FLYTHROUGH_HIGHLIGHT_FLOAT_ABOVE_M = 110.0  # 10m above the 100m route overlay (no z-fight)
    FLYTHROUGH_HIGHLIGHT_COLOR = [255, 120, 0, 235]

    # Node snapping threshold for lift placement (used when creating end nodes)
    LIFT_END_NODE_THRESHOLD_M = 80  # Extra generous for lift top station placement

    # Earth's radius in metres (WGS84 spherical approximation) — single source for all geodesy
    # (haversine/destination in GeoCalculator) and the metres-per-degree constant below.
    EARTH_RADIUS_M = 6_371_000

    # Metres per degree of latitude (≈ per degree longitude at equator). Single source for all
    # lat/lon↔metre conversions; a round nominal — exact geodesics use GeoCalculator.haversine_distance_m.
    METERS_PER_DEGREE_EQUATOR = 111320.0

    # 2D-mode z-offsets: small values for relative layer ordering (no terrain), preventing z-fighting
    # while keeping a flat appearance.
    Z_OFFSET_2D_SLOPES = 1  # Slope polygons at base
    Z_OFFSET_2D_ROADS = 2  # Road polygons just above slopes (matches slopes→roads z-order)
    Z_OFFSET_2D_LIFTS = 3  # Lift cables above roads
    Z_OFFSET_2D_PYLONS = 4  # Pylons slightly above lift cables
    Z_OFFSET_2D_ICONS = 5  # Slope/road/lift icons above pylons
    Z_OFFSET_2D_NODE_BIG = 9  # Merge-selected or parking place (big red or blue) nodes just BELOW plain nodes
    Z_OFFSET_2D_NODES = 10  # Nodes above icons
    Z_OFFSET_2D_MARKERS = 20  # Interactive markers (commit/select) on top

    # Flat-mode z-offset per committed-segment kind, keyed by SegmentKind's string value (reload-safe,
    # no model import). Kept in sync with SegmentKind by an assert in ui/kind_spec.py.
    SEGMENT_FLAT_Z = {
        "slope": Z_OFFSET_2D_SLOPES,
        "road": Z_OFFSET_2D_ROADS,
    }

    @staticmethod
    def zoom_for_span_m(span_m: float) -> float:
        """2D overview zoom for a build of the given characteristic length (rough size, in metres).

        Logarithmic around the ~4km-fits-at-VIEWING_ZOOM anchor: each halving of span → +1 zoom.
        Clamped to [VIEWING_ZOOM-STEPS_OUT, VIEWING_ZOOM+STEPS_IN]; both clamp joints meet the log
        curve with no jump. `span_m>0` is an internal invariant (real builds have real length).
        """
        assert span_m > 0, f"zoom_for_span_m needs a positive span, got {span_m}"
        raw = MapConfig.VIEWING_ZOOM + math.log2(MapConfig.ZOOM_SPAN_ANCHOR_M / span_m)
        return max(
            float(MapConfig.VIEWING_ZOOM - MapConfig.ZOOM_STEPS_OUT),
            min(float(MapConfig.VIEWING_ZOOM + MapConfig.ZOOM_STEPS_IN), raw),
        )


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

    # Hard gradient cap for car roads, enforced at build time (a proposal over this is refused). Roads
    # AIM for green grades 7%/12% — see PathFactory.generate_manual_paths.
    ROAD_MAX_GRADIENT_PCT = 15


class GeometricTuningConfig:
    """Machine-tunable knobs that shape generated route geometry."""

    # --- Grid-Dijkstra planner (connection_planners.py) ---
    # Grid is sized from the REQUIRED path length L = 100·drop/g (not the straight chord), so a gentle
    # grade on steep ground gets room to serpentine: `along` spans the chord, `across` twice its bow.
    MAX_GRID_SIZE = 220  # hard cap; resolution coarsens under it
    GRID_RES_DIVISOR = 175.0  # target cells along length L
    GRID_RES_MIN_M = 4.0  # finest cell (short switchbacks)
    GRID_ALONG_MARGIN = 1.2  # along extent = this × chord
    GRID_PADDING_M = 120.0  # slack so endpoints leave the edge
    GRID_ACROSS_MIN_M = 600.0  # lateral floor (turning room)
    GRID_ACROSS_MAX_M = 13500.0  # lateral ceiling (widest serpentine)
    MIN_GRADE_PCT_FOR_LENGTH = 1.0  # below this, size from chord
    COST_SIGMA = 2.0  # slope-deviation sensitivity (lower = stricter)
    # Lateral momentum: a switchback REVERSAL costs this × cell size. Without it Dijkstra can't tell a
    # clean switchback from a micro-sawtooth, and smoothing rounds the sawtooth back to the fall line.
    SWITCHBACK_REVERSAL_PENALTY = 500.0  # per-reversal cost, × cell
    # Cap on the planner's own finish-smoothing. A heavy factor over switchback apexes over-rounds
    # (shortens) and overshoots vertically across the gaps (dips below ground), so keep it light.
    PLANNER_SMOOTHING_FACTOR = 5.0
    PATH_SIMILARITY_TOLERANCE = 0.0001  # overlap-dedup (~10m mid-latitude)

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
    # Horizontal Douglas–Peucker tolerance after finish-smoothing: drop interior points within this of the
    # line between kept neighbours. 3.5m is the coarsest that keeps junction turns gentle (<20°); >4m kinks.
    FINISH_SIMPLIFY_TOLERANCE_M = 3.5
    # Douglas–Peucker "turned around" for lifts: DP in the (along-track distance, elevation) plane of the
    # terrain profile (horizontal is straight). 10m sheds ~half the points, keeping pylons near raw.
    TERRAIN_SIMPLIFY_TOLERANCE_M = 10.0


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

    # Bridge/tunnel bar: a smoothed deck floats a few m off ground routinely, max ground deviation.
    BRIDGE_TUNNEL_THRESHOLD_M = 50.0


assert set(EarthworkConfig.BELT_WIDTH_LIMITS.keys()) == set(SlopeConfig.DIFFICULTIES)


class ConnectionConfig:
    """Connection path parameters for manual "Connect to Custom Point".

    User clicks a target (node/free point); a path is generated if it's downhill by MIN_DROP_M
    and within segment_length.
    """

    # Minimum elevation drop to target (must go meaningfully downhill)
    MIN_DROP_M = 5


def _coprime_neighbors(radius: int) -> list[tuple[int, int]]:
    """All coprime grid offsets within `radius` (gcd==1 drops collinear duplicates like (2,4)≡(1,2)),
    row-major for determinism. These are the search-grid neighbor directions.
    """
    return [
        (dr, dc)
        for dr in range(-radius, radius + 1)
        for dc in range(-radius, radius + 1)
        if (dr, dc) != (0, 0) and math.gcd(abs(dr), abs(dc)) == 1
    ]


class PlannerConfig:
    """Structural constants for the grid-based Dijkstra planner.

    Reference: DETAILS.md Section 7 for algorithm details.
    """

    # Neighbor connectivity radius. On planar ground an edge's grade is S·cos(bearing-from-fall-line),
    # so plain 8-connectivity only offers grades at 45° bearing steps — far too coarse to hold a gentle
    # target on steep ground. A radius-R knight-style neighborhood exposes many intermediate bearings
    # (e.g. (1,9) ≈ 6° off-contour) so the search can quantize any target grade below the fall line.
    NEIGHBOR_RADIUS = 9
    # Connectivity derived from the radius so the two never drift (single source of truth).
    NEIGHBORS = _coprime_neighbors(NEIGHBOR_RADIUS)


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
    MARKER_Z_OFFSET_M = 30

    # Z-offset for paths/lines above terrain to prevent z-fighting (below markers)
    PATH_Z_OFFSET_M = 20


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
    # Cable sampling is curvature-adaptive: each span is a parabola with max sag sag_factor*span, so an
    # n-segment chord errs ~sag/n². Pick n=ceil(sqrt(sag/tol)) — short spans get few points, long spans more.
    CABLE_SAG_TOLERANCE_M = 6.0
    # A buildable lift must be at least this many min_spacing_m long, so a pylon has comfortable room to
    # each station. Below it the lift is refused at placement.
    MIN_LENGTH_SPACING_FACTOR = 3

    PYLON_CONFIG = {
        LiftType.SURFACE_LIFT: {
            "pylon_height_m": 15,
            "station_height_m": 5,
            "min_spacing_m": 30,
            "max_spacing_m": 100,
            "min_clearance_m": 10,
            "sag_factor": 0.05,
        },
        LiftType.CHAIRLIFT: {
            "pylon_height_m": 25,
            "station_height_m": 6,
            "min_spacing_m": 50,
            "max_spacing_m": 200,
            "min_clearance_m": 15,
            "sag_factor": 0.06,
        },
        LiftType.GONDOLA: {
            "pylon_height_m": 35,
            "station_height_m": 6,
            "min_spacing_m": 75,
            "max_spacing_m": 300,
            "min_clearance_m": 20,
            "sag_factor": 0.06,
        },
        LiftType.AERIAL_TRAM: {
            "pylon_height_m": 60,
            "station_height_m": 10,
            "min_spacing_m": 100,
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

    # Human-friendly lift-type display names. Non-lift build modes (Slope, Road, …) get their names
    # from BuildMode.display_name — this dict is lift types ONLY (asserted below).
    LIFT_DISPLAY_NAMES = {
        "surface_lift": "Surface Lift",
        "chairlift": "Chairlift",
        "gondola": "Gondola",
        "aerial_tram": "Aerial Tram",
    }
    assert set(LIFT_DISPLAY_NAMES.keys()) == set(LiftConfig.TYPES)

    # How far a connectivity-defect color is pulled toward mid-gray (0 = unchanged, 1 = full gray).
    GRAY_RGB = [128, 128, 128]  # gray
    DEFECT_GRAY_BLEND = 0.75

    # Bridge/tunnel tint keyed by SegmentProfile value (reload-safe, no model import; bijection
    # asserted in segment_profile.py). Bridge → lighter, tunnel → darker; GROUND has none.
    STRUCTURE_TINT_RGB = {
        "bridge": [255, 255, 255],  # blend toward white
        "tunnel": [20, 20, 20],  # blend toward near-black
    }
    BRIDGE_TUNNEL_BLEND = 0.45  # how far the base hue is pulled toward the structure tint
    BRIDGE_TUNNEL_WIDTH_MULT = 2.0  # bridge/tunnel ribbons render this much wider than a normal segment

    @staticmethod
    def _blend(rgba: list[int], target_rgb: list[int], t: float) -> list[int]:
        """Blend rgb toward target_rgb by fraction t (0 = unchanged, 1 = full target); alpha preserved."""
        r, g, b, a = rgba
        return [round(c * (1 - t) + tc * t) for c, tc in zip((r, g, b), target_rgb, strict=True)] + [a]

    @staticmethod
    def gray_out(rgba: list[int]) -> list[int]:
        """Mute an entity color strongly toward gray — the "half-dead" tone for a connectivity-defect
        slope/lift: the difficulty/type hue is just barely readable but clearly demoted. Alpha kept.
        """
        return StyleConfig._blend(rgba=rgba, target_rgb=StyleConfig.GRAY_RGB, t=StyleConfig.DEFECT_GRAY_BLEND)

    @staticmethod
    def structure_tint(rgba: list[int], profile: "SegmentProfile") -> list[int]:
        """Blend a segment color toward its bridge/tunnel tint (bridge lighter, tunnel darker), alpha
        kept — mirrors gray_out. GROUND has no tint and never reaches here (KeyError = fail loud).
        """
        return StyleConfig._blend(
            rgba=rgba, target_rgb=StyleConfig.STRUCTURE_TINT_RGB[profile.value], t=StyleConfig.BRIDGE_TUNNEL_BLEND
        )


class NameConfig:
    """Creative naming components for slopes and lifts."""

    # Slope name prefixes by difficulty — GENERIC Austrian terrain/theme words (no real peaks).
    SLOPE_PREFIXES = {
        "green": ["Sonnen", "Wiesen", "Kids", "Übungs", "Almen", "Wald"],
        "blue": ["Panorama", "Berg", "Genuss", "Familien", "Tal", "See"],
        "red": ["Gams", "Adler", "Steil", "Gipfel", "Fels", "Wilde"],
        "black": ["Teufels", "Höllen", "Todes", "Wahnsinns", "Donner", "Schwindel"],
    }
    assert set(SLOPE_PREFIXES.keys()) == set(SlopeConfig.DIFFICULTIES)

    SLOPE_SUFFIXES = [
        "Abfahrt",
        "Piste",
        "Hang",
        "Schuss",
        "Rinne",
        "Kante",
        "Mulde",
        "Steilhang",
        "Kar",
        "Wand",
        "Buckel",
        "Schneise",
        "Trasse",
        "Route",
        "Strecke",
        "Latschen",
    ]

    # Road name components — GENERIC Alpine geography common nouns (no real place names)
    ROAD_PREFIXES = [
        "Alpen",
        "Tal",
        "Serpentinen",
        "Höhen",
        "Wald",
        "Berg",
        "Almen",
        "Gletscher",
        "Gipfel",
        "Wiesen",
    ]

    ROAD_SUFFIXES = [
        "Straße",
        "Route",
        "Weg",
        "Zufahrt",
        "Pass",
        "Steig",
        "Allee",
        "Gasse",
    ]

    # Lift name prefixes by type — GENERIC terrain/landform common nouns that recur across real
    # Austrian lift names (Alm-, Sonnen-, Wald-, Gipfel-, Kreuz-, Sattel-…). NO real mountains.
    LIFT_PREFIXES = {
        "surface_lift": ["Hasen", "Übungs", "Zwergen", "Wiesen", "Moos"],
        "chairlift": ["Alm", "Gams", "Sonnen", "Wald", "Kreuz", "Sattel"],
        "gondola": ["Panorama", "Gipfel", "Berg", "Kristall", "Wolken", "Adler"],
        "aerial_tram": ["Gletscher", "Fels", "Sonnen", "Grat", "Joch", "Kar"],
    }
    assert set(LIFT_PREFIXES.keys()) == set(LiftConfig.TYPES)

    LIFT_SUFFIXES = {
        "surface_lift": ["Schlepplift", "Tellerlift", "Bügellift", "Übungslift"],
        "chairlift": ["Sesselbahn", "Sessellift", "Express", "Jet", "Flyer"],
        "gondola": ["Gondelbahn", "Kabinenbahn", "Umlaufbahn", "Gondel", "Express"],
        "aerial_tram": ["Seilbahn", "Pendelbahn", "Luftseilbahn", "Schwebebahn", "Bahn"],
    }
    assert set(LIFT_SUFFIXES.keys()) == set(LiftConfig.TYPES)

    # Length descriptors for lift naming
    LENGTH_DESCRIPTORS = {
        "short": ["Kleine", "Mini", "Kurze"],  # < LENGTH_SHORT_MAX_M
        "medium": ["Klassische", "Standard", "Normale"],  # between short and long
        "long": ["Große", "Lange", "Riesen"],  # > LENGTH_LONG_MIN_M
    }
    # Length/rise bands for entity naming (single source for lift + slope generate_name).
    LENGTH_SHORT_MAX_M = 500  # below → "short" descriptor
    LENGTH_LONG_MIN_M = 1500  # above → "long" descriptor
    SUMMIT_RISE_M = 500  # lift rise / slope drop above this → "Summit" name
    BIG_DROP_M = 300  # slope drop above this → "big" descriptor

    # 8-point compass directions for naming (German labels)
    COMPASS_DIRECTIONS = {
        "Nord": (337.5, 22.5),
        "Nordost": (22.5, 67.5),
        "Ost": (67.5, 112.5),
        "Südost": (112.5, 157.5),
        "Süd": (157.5, 202.5),
        "Südwest": (202.5, 247.5),
        "West": (247.5, 292.5),
        "Nordwest": (292.5, 337.5),
    }
    assert len(COMPASS_DIRECTIONS) == 8

    @staticmethod
    def get_compass_direction(bearing_deg: float) -> str:
        """Get compass direction name from bearing.

        Args:
            bearing_deg: Bearing in degrees (0-360)

        Returns:
            Compass direction string (Nord, Nordost, Ost, Südost, Süd, Südwest, West, Nordwest)
        """
        brg = bearing_deg % 360
        for direction, (low, high) in NameConfig.COMPASS_DIRECTIONS.items():
            if direction == "Nord":
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

    # RGBA per route criterion (keyed by RouteCriterion's string value): hue = metric (cyan=fewest lifts,
    # gold=shortest), SCENIC = darker tone of same hue. Semi-transparent; routing.py asserts full coverage.
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

    # Whole region fetched in ONE query (light). On a transient 429/504, wait for a free slot (/api/status)
    # and retry once: SLOT_WAIT_MAX_S caps the wait; SLOT_WAIT_FALLBACK_S applies when status is unreadable.
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
    DEDUP_TOL_M = 20.0  # near-coincidence band for the duplicate-piste test
    PARALLEL_TOL_M = 60.0  # near-but-offset band for the redundant-parallel
    PARALLEL_TWIN_FRAC = 0.70  # a same-name run parallel to a longer sibling ≥this of ITS length is a twin
    DEDUP_COVER_FRAC = 0.90  # covered fraction to call a piste a duplicate (below this = genuinely distinct)
    MIN_NODE_DIST_M = 100.0  # min hub spacing (closer nodes merge)
    RELAXED_MERGE_DIST_M = 200.0  # slope-node→lift pull radius
    MAX_BACKCLIMB_M = 30.0  # max uphill RISE over any BACKCLIMB_WINDOW_M span (60m-DEM sampling-noise tolerance)
    BACKCLIMB_WINDOW_M = 80.0  # window for the strict per-span uphill check
    SLOPE_ON_SOURCE_TOL_M = 30.0  # strict on-piste band (slope body hugs OSM)
    PISTE_TOL_M = 40.0  # off-piste threshold (~a wide piste's half-width)
    PISTE_VERTEX_TOL_M = 45.0  # R12 fidelity: max point→nearest-source-VERTEX gap (~1 piste width)
    MAX_PULL_M = 300.0  # max straight hub connector (longer → drop the segment)
    MAX_STRAIGHT_M = 100.0  # max single straight leg between consecutive points
    TRIM_END_M = 50.0  # trim off each slope end before the hub connector
    SNAP_GRID_M = 10.0  # snap-round grid before noding (collapse near-coincident ends)
    COORD_GRID_M = 1.0  # integer-metre projection grid (_to_m + post-noding set_precision) → bit-exact vertices
    NODE_TERRAIN_TOL_M = 10.0  # max node vs DEM deviation; also the carve depth + descent-carry cap
    SLOPE_TERRAIN_TOL_M = 50.0  # max slope-point vs DEM deviation

    # Consistency
    assert SLOPE_ON_SOURCE_TOL_M < PISTE_TOL_M <= SLOPE_TERRAIN_TOL_M
    assert PISTE_TOL_M < PISTE_VERTEX_TOL_M  # vertex gap is looser than the line off-piste band
    assert NODE_TERRAIN_TOL_M < SLOPE_TERRAIN_TOL_M
    assert DEDUP_TOL_M < PARALLEL_TOL_M < MIN_NODE_DIST_M < RELAXED_MERGE_DIST_M
    assert COORD_GRID_M < DEDUP_TOL_M  # the integer grid must sit below every planar tolerance

"""PathFactory — route generation for ski slopes and roads.

Two ways to generate proposals, shared by slopes and roads:

**Fan generation (generate_fan):**
    Radiates candidate routes from a point via the terrain-following tracer.
    A fan is a list of signed grade targets, traced as left/right traverses or a
    center path depending on terrain steepness. Slopes fan positive difficulty
    targets (green→black, descend only); roads fan the signed green targets
    (descend/climb/flat). Both run through the one shared engine `_generate_fan`.

**Manual path generation (generate_manual_paths):**
    When the user clicks a target point, a grid-based Dijkstra (SciPy) routes to it
    holding a target grade, smoothed with a cubic spline. Slopes fall back to a
    straight line when nothing is viable; the road-side straight-line fallback lives
    in the click handler, gated on the ±15% cap.

Reference: DETAILS.md Sections 5 and 7 for algorithm details
"""

import logging
import math
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from enum import Enum

from skiresort_planner.constants import (
    GeometricTuningConfig,
    PathConfig,
    SlopeConfig,
    StyleConfig,
)
from skiresort_planner.core.dem_service import DEMService
from skiresort_planner.core.geo_calculator import GeoCalculator
from skiresort_planner.core.path_tracer import PathTracer
from skiresort_planner.core.terrain_analyzer import TerrainAnalyzer
from skiresort_planner.enum_utils import enum_eq
from skiresort_planner.generators.connection_planners import GradientMode, LeastCostPathPlanner
from skiresort_planner.model.path_point import PathPoint
from skiresort_planner.model.path_segment import SegmentKind
from skiresort_planner.model.proposed_path import ProposedPathSegment

logger = logging.getLogger(__name__)


class Side(Enum):
    """Traverse direction relative to fall line."""

    LEFT = "left"
    RIGHT = "right"
    CENTER = "center"


@dataclass(frozen=True)
class FanTarget:
    """One signed grade the fan should attempt, with its labelling.

    A fan is a list of these: slopes build positive difficulty targets
    (green→black), roads build signed green targets (descend/climb/flat).

    Attributes:
        grade_pct: Signed target grade (+ descends, − climbs, 0 contours).
        difficulty: Slope difficulty (green/blue/red/black), or "" for roads.
        grade_name: Steepness label (gentle/steep, or "flat" for a contour road).
        kind: SLOPE or ROAD — stamped onto each proposal.
    """

    grade_pct: float
    difficulty: str
    grade_name: str
    kind: SegmentKind


@dataclass
class GradeConfig:
    """Configuration for a single traced fan variant.

    Attributes:
        difficulty: Slope difficulty (green/blue/red/black), or "" for roads.
        grade: Steepness variant (gentle/steep/flat).
        target_slope_pct: Signed target grade (+ descends, − climbs, 0 contours).
        side: Traverse direction — only meaningful for the fan (trace_hill). The
            grid planner ignores side, so planner-path configs leave it at CENTER.
    """

    difficulty: str
    grade: str
    target_slope_pct: float
    side: Side = Side.CENTER

    def sector_name(self, kind: SegmentKind) -> str:
        """Fan display name for this variant, dispatched by kind.

        Each SegmentKind supplies its own label builder via _SECTOR_NAME_BUILDERS, so a
        new kind (e.g. a nordic trail) is added by registering a builder.
        """
        return _SECTOR_NAME_BUILDERS[kind](self)


# Per-kind fan label builders. Each is symmetric — it reads the raw config fields and formats its own label.
_SECTOR_NAME_BUILDERS: dict[SegmentKind, Callable[[GradeConfig], str]] = {
    SegmentKind.SLOPE: lambda cfg: f"{cfg.difficulty.capitalize()} {cfg.side.value.capitalize()} ({cfg.grade.capitalize()})",
    SegmentKind.ROAD: lambda cfg: f"Road {cfg.side.value.capitalize()} ({cfg.grade.capitalize()})",
}
assert set(_SECTOR_NAME_BUILDERS) == set(SegmentKind), "every SegmentKind must have a fan sector-name builder"


def _slope_fan_targets() -> tuple[list["FanTarget"], bool]:
    """Slope fan: every difficulty-grade target, descending (positive). Center-stop applies."""
    targets = [
        FanTarget(grade_pct=target_slope, difficulty=difficulty, grade_name=grade_name, kind=SegmentKind.SLOPE)
        for difficulty in SlopeConfig.DIFFICULTIES
        for grade_name, target_slope in SlopeConfig.DIFFICULTY_TARGETS[difficulty].items()
    ]
    return targets, True


def _road_fan_targets() -> tuple[list["FanTarget"], bool]:
    """Road fan: signed GREEN grades — descend/climb/contour {+7,+12,−7,−12,0}. No center-stop.

    Grades are single-sourced from DIFFICULTY_TARGETS["green"]; every proposal is
    hard-capped at ±ROAD_MAX_GRADIENT_PCT by the caller (which may keep none).
    """
    green = SlopeConfig.DIFFICULTY_TARGETS["green"]
    targets = [
        FanTarget(grade_pct=sign * grade, difficulty="", grade_name=grade_name, kind=SegmentKind.ROAD)
        for sign in (+1.0, -1.0)
        for grade_name, grade in green.items()
    ]
    targets.append(FanTarget(grade_pct=0.0, difficulty="", grade_name="flat", kind=SegmentKind.ROAD))
    return targets, False


# Per-kind fan target builders → (targets, apply_center_stop). Adding a kind = one entry.
_FAN_TARGETS: dict[SegmentKind, Callable[[], tuple[list["FanTarget"], bool]]] = {
    SegmentKind.SLOPE: _slope_fan_targets,
    SegmentKind.ROAD: _road_fan_targets,
}
assert set(_FAN_TARGETS) == set(SegmentKind), "every SegmentKind must have a fan target builder"


class PathFactory:
    """Factory for generating proposed ski paths using nested difficulty loops.

    Generates paths by iterating through all difficulty-grade-side combinations:
    - Difficulty: green → blue → red → black
    - Grade: gentle → steep
    - Side: left/right (steep terrain) or center (flat terrain)

    Green paths work on ALL terrain because shallow traverse angles (7-12%
    target slope) can always be achieved regardless of terrain steepness.

    Example:
        factory = PathFactory(dem_service=dem_service)
        # Generate up to 16 fan paths
        for path in factory.generate_fan(kind=SegmentKind.SLOPE, lon=12.5, lat=47.0, elevation=2400.0):
            print(path.sector_name)  # "Green Left (Gentle)", etc.

        # Generate connection paths to target
        for path in factory.generate_manual_paths(...):
            print(f"Slope: {path.avg_slope_pct}%")

    Configuration: See GeometricTuningConfig in constants.py for tunable parameters.
    """

    def __init__(
        self,
        dem_service: DEMService | None = None,
        path_tracer: PathTracer | None = None,
        terrain_analyzer: TerrainAnalyzer | None = None,
    ) -> None:
        """Initialize path factory with required services."""
        self.dem_service = dem_service or DEMService()
        # Create terrain_analyzer first so it can be shared with other components
        self.terrain_analyzer = terrain_analyzer or TerrainAnalyzer(dem=self.dem_service)
        # Pass terrain_analyzer to path_tracer to avoid creating another instance
        self.path_tracer = path_tracer or PathTracer(dem=self.dem_service, analyzer=self.terrain_analyzer)

        # Initialize connection path planner with shared terrain_analyzer
        self._planner = LeastCostPathPlanner(dem_service=self.dem_service, terrain_analyzer=self.terrain_analyzer)

    def generate_fan(
        self,
        kind: SegmentKind,
        lon: float,
        lat: float,
        elevation: float | None = None,
        target_length_m: float = PathConfig.SEGMENT_LENGTH_DEFAULT_M,
    ) -> Iterator[ProposedPathSegment]:
        """Generate the fan of proposals for a build kind — one dispatch for every kind.

        Each kind supplies its target set + center-stop policy via _FAN_TARGETS[kind];
        the shared engine (_generate_fan) traces them. Slopes descend the difficulty
        targets (green→black); roads fan the signed green targets (descend/climb/flat).
        A new kind is a new _FAN_TARGETS entry — no new method.
        """
        targets, apply_center_stop = _FAN_TARGETS[kind]()
        yield from self._generate_fan(
            lon=lon,
            lat=lat,
            elevation=elevation,
            targets=targets,
            target_length_m=target_length_m,
            apply_center_stop=apply_center_stop,
        )

    def _generate_fan(
        self,
        lon: float,
        lat: float,
        elevation: float | None,
        targets: list[FanTarget],
        target_length_m: float,
        *,
        apply_center_stop: bool,
    ) -> Iterator[ProposedPathSegment]:
        """Shared fan engine: trace each target as left/right traverses or a center path.

        For each target, choose CENTER (straight along the reference bearing) when the
        target grade magnitude meets or exceeds the terrain steepness — no traverse is
        needed — otherwise LEFT and RIGHT traverses. When apply_center_stop is set
        (slopes), stop after MAX_CENTER_PATHS redundant fall-line paths (DETAILS.md §5.4).

        Args:
            lon, lat: Starting coordinates
            elevation: Starting elevation (queries DEM if None)
            targets: The signed grade targets to attempt, in fan order.
            target_length_m: Target path length in meters.
            apply_center_stop: Slopes cap redundant center paths; roads do not.

        Yields:
            ProposedPathSegment for each successfully traced variant.
        """
        if elevation is None:
            elevation = self.dem_service.get_elevation(lon=lon, lat=lat)
        if elevation is None:
            logger.warning(f"No elevation at ({lon}, {lat})")
            return

        # Terrain steepness (magnitude) decides left/right vs center per target.
        gradient = self.terrain_analyzer.compute_gradient(lon=lon, lat=lat)
        terrain_slope_pct = gradient.slope_pct
        fall_line_bearing = gradient.bearing_deg

        logger.debug(
            f"_generate_fan: start=({lon:.5f}, {lat:.5f}, {elevation:.0f}m), "
            f"terrain_slope={terrain_slope_pct:.1f}%, fall_line={fall_line_bearing:.0f}°, "
            f"{len(targets)} targets, center_stop={apply_center_stop}"
        )

        # Track statistics
        count_by_diff = {d: 0 for d in SlopeConfig.DIFFICULTIES}
        center_count = 0
        paths_generated = 0

        for target in targets:
            # Center (no traverse) when the target grade meets/exceeds terrain steepness;
            # otherwise a left/right traverse holds the gentler grade on steeper ground.
            needs_center = abs(target.grade_pct) >= terrain_slope_pct

            if needs_center:
                center_count += 1
                # Slopes: once every non-hardest difficulty has a path, extra center
                # paths (same fall line) are redundant — stop after the cap.
                if apply_center_stop:
                    all_diffs_seen = all(count_by_diff[d] > 0 for d in SlopeConfig.DIFFICULTIES[:-1])
                    if center_count > GeometricTuningConfig.MAX_CENTER_PATHS and all_diffs_seen:
                        break
                side_variants = [Side.CENTER]
            else:
                side_variants = [Side.LEFT, Side.RIGHT]

            for side in side_variants:
                config = GradeConfig(
                    difficulty=target.difficulty,
                    grade=target.grade_name,
                    target_slope_pct=target.grade_pct,
                    side=side,
                )
                path = self._trace_path_for_config(
                    lon=lon,
                    lat=lat,
                    config=config,
                    target_length_m=target_length_m,
                    kind=target.kind,
                )
                if path is None:
                    continue
                if target.difficulty:
                    assert target.difficulty in count_by_diff, (
                        f"difficulty {target.difficulty!r} not in count_by_diff (predefined from DIFFICULTIES)"
                    )
                    count_by_diff[target.difficulty] += 1
                paths_generated += 1
                yield path

        logger.debug(f"_generate_fan complete: {paths_generated} paths (by difficulty: {count_by_diff})")

    def _trace_path_for_config(
        self,
        lon: float,
        lat: float,
        config: GradeConfig,
        target_length_m: float,
        kind: SegmentKind,
    ) -> ProposedPathSegment | None:
        """Trace a single fan variant and wrap it as a proposal of the given kind."""
        traced = self.path_tracer.trace_hill(
            start_lon=lon,
            start_lat=lat,
            target_grade_pct=config.target_slope_pct,
            side=config.side.value,
            target_length_m=target_length_m,
        )

        if not traced or not traced.points:
            logger.debug(
                f"_trace_path_for_config: no path from ({lon:.5f}, {lat:.5f}) "
                f"target_grade={config.target_slope_pct:.1f}%, side={config.side.value}, kind={kind.value}"
            )
            return None

        return ProposedPathSegment(
            points=traced.points,
            target_slope_pct=config.target_slope_pct,
            target_difficulty=config.difficulty,
            sector_name=config.sector_name(kind),
            is_connector=False,
            kind=kind,
        )

    @staticmethod
    def filter_by_max_grade(
        paths: list[ProposedPathSegment], cap_pct: float
    ) -> tuple[list[ProposedPathSegment], float | None]:
        """Keep proposals whose steepest section is within the cap; report the gentlest seen.

        The one cap-filter for both slopes (MAX_SKIABLE_PCT) and roads (ROAD_MAX_GRADIENT_PCT).
        `max_slope_pct` is a magnitude, so a single `<= cap` catches climbs and descents alike.

        Returns:
            (kept, gentlest) where kept are the in-cap proposals and gentlest is the
            smallest max_slope_pct over ALL inputs (for the too-steep message), or None
            when `paths` is empty.
        """
        kept = [p for p in paths if p.max_slope_pct <= cap_pct]
        gentlest = min((p.max_slope_pct for p in paths), default=None)
        return kept, gentlest

    def _are_paths_similar(self, path1: ProposedPathSegment, path2: ProposedPathSegment) -> bool:
        """Check if two paths are similar by comparing points at percentile positions.

        Since start and end points are always the same, compares intermediate points
        at 10%, 20%, ..., 90% positions along each path. Calculates average distance
        across all percentiles.

        Args:
            path1, path2: Paths to compare

        Returns:
            True if average distance across percentiles is below threshold.
        """
        # If either path has too few points, consider them not similar (can't compare)
        if not path1.points or not path2.points:
            return False

        len1, len2 = len(path1.points), len(path2.points)
        if len1 < 3 or len2 < 3:
            return False

        # Compare at 10%, 20%, ..., 90% positions (skip 0% and 100% - same start/end)
        percentiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        total_distance = 0.0

        for pct in percentiles:
            idx1 = int(pct * (len1 - 1))
            idx2 = int(pct * (len2 - 1))
            p1 = path1.points[idx1]
            p2 = path2.points[idx2]
            # Sum squared differences (faster than sqrt for comparison)
            total_distance += (p1.lon - p2.lon) ** 2 + (p1.lat - p2.lat) ** 2

        avg_distance = total_distance / len(percentiles)
        threshold_sq = GeometricTuningConfig.PATH_SIMILARITY_TOLERANCE**2
        return avg_distance < threshold_sq

    def _deduplicate_paths(self, paths: list[ProposedPathSegment]) -> list[ProposedPathSegment]:
        """Remove duplicate/overlapping paths, keeping gentlest slope.

        When multiple paths follow nearly the same trajectory, keeps only
        the one with lowest avg_slope_pct (gentlest actual slope).

        Args:
            paths: List of paths to deduplicate

        Returns:
            Deduplicated list with similar paths removed.
        """
        if not paths:
            return []

        # Sort by actual measured slope (gentlest first)
        sorted_paths = sorted(paths, key=lambda p: p.avg_slope_pct)

        unique: list[ProposedPathSegment] = []

        for path in sorted_paths:
            # Check if this path is similar to any already-kept path
            is_duplicate = False
            for kept_path in unique:
                if self._are_paths_similar(path1=path, path2=kept_path):
                    is_duplicate = True
                    break

            if not is_duplicate:
                unique.append(path)

        removed_count = len(paths) - len(unique)
        if removed_count > 0:
            logger.debug(f"Deduplicated paths: removed {removed_count} similar paths")

        return unique

    @staticmethod
    def _road_target_grades(signed_drop: float) -> list[float]:
        """The signed GREEN grades a road may hold, sign from signed_drop.

        signed_drop = start_elev − target_elev (+ descent, − climb). Same magnitudes as
        a green slope; sign is the only road-vs-slope difference.
        """
        return [math.copysign(target, signed_drop) for target in SlopeConfig.DIFFICULTY_TARGETS["green"].values()]

    def generate_manual_paths(
        self,
        kind: SegmentKind,
        start_lon: float,
        start_lat: float,
        start_elevation: float,
        target_lon: float,
        target_lat: float,
        target_elevation: float | None = None,
    ) -> Iterator[ProposedPathSegment]:
        """Generate grid-planner paths connecting the start to a user-clicked target.

        SLOPE: tries all difficulty-grade combinations for viable ski routes (descend
        only), deduplicated to keep the gentlest per trajectory.

        ROAD: holds a GREEN grade (7%/12%) signed for the endpoints' direction so it may
        climb or descend, serpentining on steep ground.

        No straight-line fabrication here: when the planner finds nothing this yields
        nothing (the caller applies the kind's cap and, for roads, the direct fallback).

        Args:
            kind: The SegmentKind being routed (SLOPE difficulty fan, ROAD signed green).
            start_lon, start_lat, start_elevation: Starting point.
            target_lon, target_lat: Target coordinates (user click).
            target_elevation: Target elevation (queries DEM if None).

        Yields:
            ProposedPathSegment for each unique path, sorted by avg_slope_pct.
        """
        if target_elevation is None:
            target_elevation = self.dem_service.get_elevation(lon=target_lon, lat=target_lat)
        if target_elevation is None:
            logger.warning(f"No elevation at target ({target_lon}, {target_lat})")
            return

        # The grid planner ignores `side` (grade-only cost), so every config leaves
        # GradeConfig.side at its CENTER default — no dead LEFT/RIGHT duplication here.
        if enum_eq(a=kind, b=SegmentKind.ROAD):
            # A road holds a GREEN grade, signed by the endpoints' direction (climb or
            # descend). Same routing as a green slope; sign is the only difference. On
            # steep ground the planner serpentines to hold it (§7.3).
            signed_drop = start_elevation - target_elevation
            gradient_mode = GradientMode.DOWNHILL if signed_drop >= 0 else GradientMode.UPHILL
            configs: list[GradeConfig] = [
                GradeConfig(difficulty="", grade="road", target_slope_pct=grade)
                for grade in self._road_target_grades(signed_drop=signed_drop)
            ]
        elif enum_eq(a=kind, b=SegmentKind.SLOPE):
            gradient_mode = GradientMode.DOWNHILL  # slopes always descend
            configs = [
                GradeConfig(difficulty=difficulty, grade=grade_name, target_slope_pct=target_slope)
                for difficulty in SlopeConfig.DIFFICULTIES
                for grade_name, target_slope in SlopeConfig.DIFFICULTY_TARGETS[difficulty].items()
            ]
        else:
            raise ValueError(f"Unexpected {kind=}")

        all_paths: list[ProposedPathSegment] = []
        for config in configs:
            path = self._planner.plan(
                start_lon=start_lon,
                start_lat=start_lat,
                start_elevation=start_elevation,
                target_lon=target_lon,
                target_lat=target_lat,
                target_elevation=target_elevation,
                target_grade_pct=config.target_slope_pct,
                gradient_mode=gradient_mode,
            )
            if path is None:
                logger.debug(
                    f"generate_manual_paths: planner found no path for kind={kind.value}, "
                    f"difficulty={config.difficulty!r}, grade={config.grade!r}, "
                    f"target_grade={config.target_slope_pct:.1f}% from ({start_lon:.5f}, {start_lat:.5f}) "
                    f"to ({target_lon:.5f}, {target_lat:.5f})"
                )
                continue
            path.kind = kind
            path.target_difficulty = config.difficulty  # "" for roads
            path.sector_name = f"🎯 {config.sector_name(kind)}"
            all_paths.append(path)

        # Deduplicate paths (keep gentlest slope for overlapping paths)
        unique_paths = self._deduplicate_paths(paths=all_paths)

        logger.debug(f"generate_manual_paths: {len(all_paths)} raw → {len(unique_paths)} unique paths")

        # No straight-line fabrication here. When the planner finds nothing viable,
        # this yields nothing: a slope is refused above MAX_SKIABLE_PCT, a road above
        # ±ROAD_MAX_GRADIENT_PCT (the road-side direct-line fallback lives in the
        # shared custom-connect generator, gated on the cap).
        yield from unique_paths

    def straight_line(
        self,
        kind: SegmentKind,
        start_lon: float,
        start_lat: float,
        start_elevation: float,
        target_lon: float,
        target_lat: float,
        target_elevation: float,
    ) -> ProposedPathSegment:
        """A direct straight-line connector (bridge/cut) between two points, of the given kind.

        The fallback when the grid planner finds no in-grade route; the caller still
        hard-caps it at the kind's max grade. Densified to RESAMPLE_STEP_M by linear
        interpolation (matching the planner/finish density) but stays a straight 3D line,
        not DEM-sampled. Only kinds whose KindSpec.has_direct_fallback is True use this.
        """
        distance_m = GeoCalculator.haversine_distance_m(
            lat1=start_lat, lon1=start_lon, lat2=target_lat, lon2=target_lon
        )
        n_steps = max(1, int(distance_m / GeometricTuningConfig.RESAMPLE_STEP_M))
        points = [
            PathPoint(
                lon=start_lon + (target_lon - start_lon) * (i / n_steps),
                lat=start_lat + (target_lat - start_lat) * (i / n_steps),
                elevation=start_elevation + (target_elevation - start_elevation) * (i / n_steps),
            )
            for i in range(n_steps + 1)
        ]
        return ProposedPathSegment(
            points=points,
            target_slope_pct=0.0,  # No target grade for a direct line
            target_difficulty="",  # A direct connector carries no ski difficulty
            sector_name=f"{StyleConfig.ROAD_ICON} Direct {kind.value.capitalize()} (bridge/cut)",
            is_connector=True,
            kind=kind,
        )

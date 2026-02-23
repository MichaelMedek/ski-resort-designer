"""PathFactory - Path generation algorithms for ski slope planning.

Generates proposed paths using a nested loop structure to cover all difficulty variants:

**Fan Pattern Generation (generate_fan):**
    Generates up to 16 paths by iterating:
    1. Difficulty: green → blue → red → black
    2. Grade: gentle → steep (2 variants per difficulty)
    3. Side: left/right on steep terrain, center on flat terrain

    Green paths (traverses) ALWAYS work on ALL terrain steepness because they
    use shallow traverse angles that work even on very steep terrain.

**Manual Path Generation (generate_manual_paths):**
    When user clicks a target point, uses grid-based Dijkstra algorithm:

    GRID-BASED DIJKSTRA:
       - Creates a grid (15m spacing) covering the search area
       - Uses SciPy's C-optimized Dijkstra with slope-preference cost function
       - Smooths output with cubic spline interpolation
       - Terrain-adaptive path planning

    The algorithm generates paths for all difficulty combinations, then
    deduplicates overlapping paths (keeping gentlest measured slope).

Reference: DETAILS.md Section 7 for algorithm details
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Iterator, Optional

from skiresort_planner.constants import (
    PathConfig,
    PlannerConfig,
    SlopeConfig,
    StyleConfig,
)
from skiresort_planner.core.dem_service import DEMService
from skiresort_planner.core.geo_calculator import GeoCalculator
from skiresort_planner.core.path_tracer import PathTracer
from skiresort_planner.core.terrain_analyzer import TerrainAnalyzer
from skiresort_planner.generators.connection_planners import LeastCostPathPlanner
from skiresort_planner.model.proposed_path import ProposedSlopeSegment

logger = logging.getLogger(__name__)


class Side(Enum):
    """Traverse direction relative to fall line."""

    LEFT = "left"
    RIGHT = "right"
    CENTER = "center"


@dataclass
class GradeConfig:
    """Configuration for a single difficulty-grade variant.

    Attributes:
        difficulty: Slope difficulty (green/blue/red/black)
        grade: Steepness variant (gentle/steep)
        target_slope_pct: Target slope percentage
        side: Traverse direction (left/right/center)
    """

    difficulty: str
    grade: str
    target_slope_pct: float
    side: Side

    @property
    def name(self) -> str:
        """Display name like 'Green Left (Gentle)' or 'Blue Center (Steep)'."""
        side_str = self.side.value.capitalize()
        return f"{self.difficulty.capitalize()} {side_str} ({self.grade.capitalize()})"

    @property
    def color(self) -> str:
        """Get color for this difficulty."""
        return StyleConfig.SLOPE_COLORS[self.difficulty]


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
        for path in factory.generate_fan(lon=12.5, lat=47.0, elevation=2400.0):
            print(path.sector_name)  # "Green Left (Gentle)", etc.

        # Generate connection paths to target
        for path in factory.generate_manual_paths(...):
            print(f"Slope: {path.avg_slope_pct}%")

    Configuration: See PlannerConfig in constants.py for tunable parameters.
    """

    def __init__(
        self,
        dem_service: Optional[DEMService] = None,
        path_tracer: Optional[PathTracer] = None,
        terrain_analyzer: Optional[TerrainAnalyzer] = None,
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
        lon: float,
        lat: float,
        elevation: Optional[float] = None,
        target_length_m: float = PathConfig.SEGMENT_LENGTH_DEFAULT_M,
    ) -> Iterator[ProposedSlopeSegment]:
        """Generate fan of paths iterating through all difficulty-grade-side combinations.

        Nested loop structure:
        1. Difficulty: green → blue → red → black
        2. Grade: gentle → steep (per DIFFICULTY_TARGETS)
        3. Side: left/right if terrain >= target slope, center otherwise

        On steep terrain (e.g., 30% slope):
        - Green paths use traverse angles to achieve 7-12% effective slope
        - Blue/Red/Black paths descend more directly

        On flat terrain (e.g., 10% slope):
        - Paths that need steeper terrain than available use center (fall line)
        - Generation stops after MAX_CENTER_PATHS to avoid redundant paths

        Args:
            lon, lat: Starting coordinates
            elevation: Starting elevation (queries DEM if None)
            target_length_m: Target path length in meters (default 500m)

        Yields:
            ProposedSlopeSegment for each successfully traced path.
            Order: Green Left (Gentle), Green Right (Gentle), Green Left (Steep),
                   Green Right (Steep), Blue Left (Gentle), ... etc.
        """
        if elevation is None:
            elevation = self.dem_service.get_elevation(lon=lon, lat=lat)
        if elevation is None:
            logger.warning(f"No elevation at ({lon}, {lat})")
            return

        # Get terrain slope to determine left/right vs center generation
        gradient = self.terrain_analyzer.compute_gradient(lon=lon, lat=lat)
        terrain_slope_pct = gradient.slope_pct
        fall_line_bearing = gradient.bearing_deg

        logger.info(
            f"generate_fan: start=({lon:.5f}, {lat:.5f}, {elevation:.0f}m), "
            f"terrain_slope={terrain_slope_pct:.1f}%, fall_line={fall_line_bearing:.0f}°"
        )

        # Track statistics
        count_by_diff = {"green": 0, "blue": 0, "red": 0, "black": 0}
        center_count = 0
        paths_generated = 0
        stop_generation = False

        # Nested loop: Difficulty → Grade → Side
        for difficulty in ["green", "blue", "red", "black"]:
            if stop_generation:
                break

            targets = SlopeConfig.DIFFICULTY_TARGETS[difficulty]

            for grade_name, target_slope in targets.items():
                if stop_generation:
                    break

                # Determine if we need center (target >= terrain) or left/right (target < terrain)
                # Center = straight down fall line (no traverse angle needed)
                # Left/Right = traverse at angle to achieve target slope on steep terrain
                needs_center = target_slope >= terrain_slope_pct

                if needs_center:
                    center_count += 1
                    # Stop if we've generated paths at multiple difficulties AND hit limit
                    all_diffs_seen = all(count_by_diff[d] > 0 for d in ["green", "blue", "red"])
                    if center_count > PathConfig.MAX_CENTER_PATHS and all_diffs_seen:
                        stop_generation = True
                        break
                    side_variants = [Side.CENTER]
                else:
                    side_variants = [Side.LEFT, Side.RIGHT]

                for side in side_variants:
                    config = GradeConfig(
                        difficulty=difficulty,
                        grade=grade_name,
                        target_slope_pct=target_slope,
                        side=side,
                    )

                    path = self._trace_path_for_config(
                        lon=lon,
                        lat=lat,
                        config=config,
                        target_length_m=target_length_m,
                    )

                    if path is None:
                        continue

                    count_by_diff[difficulty] += 1
                    paths_generated += 1
                    yield path

        logger.info(f"generate_fan complete: {paths_generated} paths (by difficulty: {count_by_diff})")

    def _trace_path_for_config(
        self,
        lon: float,
        lat: float,
        config: GradeConfig,
        target_length_m: float,
    ) -> Optional[ProposedSlopeSegment]:
        """Trace a single path for a given configuration."""
        traced = self.path_tracer.trace_downhill(
            start_lon=lon,
            start_lat=lat,
            target_slope_pct=config.target_slope_pct,
            side=config.side.value,
            target_length_m=target_length_m,
        )

        if not traced or not traced.points:
            return None

        return ProposedSlopeSegment(
            points=traced.points,
            target_slope_pct=config.target_slope_pct,
            target_difficulty=config.difficulty,
            sector_name=config.name,
            is_connector=False,
        )

    def _are_paths_similar(self, path1: ProposedSlopeSegment, path2: ProposedSlopeSegment) -> bool:
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
        threshold_sq = PlannerConfig.PATH_SIMILARITY_TOLERANCE**2
        return avg_distance < threshold_sq

    def _deduplicate_paths(self, paths: list[ProposedSlopeSegment]) -> list[ProposedSlopeSegment]:
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

        unique: list[ProposedSlopeSegment] = []

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
            logger.info(f"Deduplicated paths: removed {removed_count} similar paths")

        return unique

    def generate_manual_paths(
        self,
        start_lon: float,
        start_lat: float,
        start_elevation: float,
        target_lon: float,
        target_lat: float,
        target_elevation: Optional[float] = None,
    ) -> Iterator[ProposedSlopeSegment]:
        """Generate paths to a user-clicked target, trying all difficulty-grade combinations.

        When user clicks a target point on the map, this method generates paths
        using all difficulty-grade combinations to find viable routes. Paths are
        deduplicated: when multiple paths follow nearly the same trajectory,
        only the gentlest measured slope is kept.

        Args:
            start_lon, start_lat, start_elevation: Starting point
            target_lon, target_lat: Target coordinates (user click)
            target_elevation: Target elevation (queries DEM if None)

        Yields:
            ProposedSlopeSegment for each unique path, sorted by avg_slope_pct.
        """
        if target_elevation is None:
            target_elevation = self.dem_service.get_elevation(lon=target_lon, lat=target_lat)
        if target_elevation is None:
            logger.warning(f"No elevation at target ({target_lon}, {target_lat})")
            return

        # Build all 8 grade configs
        all_grades = []
        for difficulty in ["green", "blue", "red", "black"]:
            for grade_name, target_slope in SlopeConfig.DIFFICULTY_TARGETS[difficulty].items():
                all_grades.append((difficulty, grade_name, target_slope))

        # Collect all valid paths first
        all_paths: list[ProposedSlopeSegment] = []

        for difficulty, grade_name, target_slope in all_grades:
            for side_enum in [Side.LEFT, Side.RIGHT]:
                config = GradeConfig(
                    difficulty=difficulty,
                    grade=grade_name,
                    target_slope_pct=target_slope,
                    side=side_enum,
                )

                path = self._planner.plan(
                    start_lon=start_lon,
                    start_lat=start_lat,
                    start_elevation=start_elevation,
                    target_lon=target_lon,
                    target_lat=target_lat,
                    target_elevation=target_elevation,
                    target_slope_pct=config.target_slope_pct,
                    side=config.side.value,
                )

                if path is None:
                    continue

                # Add difficulty metadata
                path.target_difficulty = config.difficulty
                path.sector_name = f"🎯 {config.name}"
                all_paths.append(path)

        # Deduplicate paths (keep gentlest slope for overlapping paths)
        unique_paths = self._deduplicate_paths(paths=all_paths)

        logger.info(f"generate_manual_paths: {len(all_paths)} raw → {len(unique_paths)} unique paths")

        # If no paths found, create a fallback straight-line path
        # This ensures the user can always connect points in the network
        if not unique_paths:
            logger.info("No optimized paths found, creating straight-line fallback")
            unique_paths = [
                self._create_straight_line_path(
                    start_lon=start_lon,
                    start_lat=start_lat,
                    start_elevation=start_elevation,
                    target_lon=target_lon,
                    target_lat=target_lat,
                    target_elevation=target_elevation,
                )
            ]
            logger.info(f"Created fallback straight-line path: {unique_paths[0].sector_name}")

        yield from unique_paths

    def _create_straight_line_path(
        self,
        start_lon: float,
        start_lat: float,
        start_elevation: float,
        target_lon: float,
        target_lat: float,
        target_elevation: float,
    ) -> ProposedSlopeSegment:
        """Create a simple straight-line path as fallback when Dijkstra finds nothing.

        This allows the user to always connect two points, even if the terrain
        is difficult. The path is marked as a direct connection.
        """
        from skiresort_planner.model.path_point import PathPoint

        # Create simple 2-point path
        points = [
            PathPoint(lon=start_lon, lat=start_lat, elevation=start_elevation),
            PathPoint(lon=target_lon, lat=target_lat, elevation=target_elevation),
        ]

        # Calculate actual slope
        drop = start_elevation - target_elevation
        dist = GeoCalculator.haversine_distance_m(lat1=start_lat, lon1=start_lon, lat2=target_lat, lon2=target_lon)

        actual_slope = (drop / dist) * 100

        return ProposedSlopeSegment(
            points=points,
            target_slope_pct=0.0,  # No target slope for direct line
            target_difficulty=TerrainAnalyzer.classify_difficulty(slope_pct=actual_slope),
            sector_name="📍 Direct Line (fallback)",
            is_connector=True,
        )

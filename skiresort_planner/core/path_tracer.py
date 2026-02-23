"""Path tracing algorithms for ski slope generation.

Implements the Cumulative Drop Tracking algorithm:
- Pre-calculates target total drop based on path length and target slope
- Tracks accumulated drop as path is traced
- Dynamically adjusts step target to converge on final average

This self-correcting approach eliminates DEM grid artifacts.

Reference: DETAILS.md Sections 5, 6
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from math import acos, degrees
from typing import TYPE_CHECKING, Optional

import numpy as np

from skiresort_planner.constants import (
    PathConfig,
    SlopeConfig,
)
from skiresort_planner.core.dem_service import DEMService
from skiresort_planner.core.geo_calculator import GeoCalculator
from skiresort_planner.core.terrain_analyzer import (
    TerrainAnalyzer,
)

if TYPE_CHECKING:
    from skiresort_planner.model.path_point import PathPoint

logger = logging.getLogger(__name__)


@dataclass
class TracedPath:
    """Result of path tracing (raw path data before conversion to ProposedSlopeSegment).

    Attributes:
        points: List of PathPoint instances
        avg_slope_pct: Average slope percentage
        total_drop_m: Total vertical drop in meters
        length_m: Total path length in meters
        difficulty: Classified difficulty (green/blue/red/black)
        target_slope_pct: Target slope used for tracing
    """

    points: list[PathPoint]
    avg_slope_pct: float
    total_drop_m: float
    length_m: float
    difficulty: str
    target_slope_pct: float


class PathTracer:
    """Traces ski paths using cumulative drop tracking.

    The algorithm works by:
    1. Setting a target total drop based on path length and target slope
    2. At each step, computing how much drop is remaining
    3. Adjusting traverse angle to achieve the remaining drop target

    This creates paths that naturally converge to the target average slope.
    """

    def __init__(
        self,
        dem: Optional[DEMService] = None,
        analyzer: Optional[TerrainAnalyzer] = None,
    ):
        """Initialize path tracer.

        Args:
            dem: DEM service for elevation queries
            analyzer: Terrain analyzer for gradient calculations
        """
        self._dem = dem or DEMService()
        self._analyzer = analyzer or TerrainAnalyzer(dem=self._dem)

    @property
    def dem(self) -> DEMService:
        """Access the DEM service."""
        return self._dem

    @property
    def analyzer(self) -> TerrainAnalyzer:
        """Access the terrain analyzer."""
        return self._analyzer

    def trace_downhill(
        self,
        start_lon: float,
        start_lat: float,
        target_slope_pct: float,
        side: str,
        target_length_m: float,
    ) -> Optional[TracedPath]:
        """Trace a downhill path using cumulative drop tracking.

        Args:
            start_lon: Starting longitude
            start_lat: Starting latitude
            target_slope_pct: Target effective slope percentage
            side: "left", "right", or "center" (traverse direction)
            target_length_m: Target path length in meters

        Returns:
            TracedPath if successful, None if path cannot be traced.
        """
        from skiresort_planner.model.path_point import PathPoint

        start_elev = self._dem.get_elevation(lon=start_lon, lat=start_lat)
        if start_elev is None:
            logger.warning(f"Cannot trace path: invalid elevation at start ({start_lon:.6f}, {start_lat:.6f})")
            return None

        bounds = self._dem.bounds
        step_size = PathConfig.STEP_SIZE_M
        side_sign = -1 if side == "left" else +1  # Left = negative offset

        # Initialize tracking
        points: list[PathPoint] = [PathPoint(lon=start_lon, lat=start_lat, elevation=start_elev)]
        current_lon, current_lat, current_elev = start_lon, start_lat, start_elev
        total_dist = 0.0

        # Self-intersection prevention
        max_turn_per_step = PathConfig.MAX_TURN_PER_STEP_DEG
        previous_bearing: Optional[float] = None

        # Bearing smoothing for flat terrain
        recent_bearings: list[float] = []
        smoothing_window = PathConfig.BEARING_SMOOTHING_WINDOW
        flat_terrain_threshold = PathConfig.FLAT_TERRAIN_THRESHOLD_PCT

        # Cumulative drop tracking
        target_total_drop = (target_slope_pct / 100.0) * target_length_m
        accumulated_drop = 0.0

        while total_dist < target_length_m:
            # Calculate remaining drop and distance
            remaining_drop = target_total_drop - accumulated_drop
            remaining_distance = target_length_m - total_dist

            # Dynamic target slope for this step
            if remaining_distance > step_size:
                step_target = (remaining_drop / remaining_distance) * 100.0
                # Asymmetric clamping for self-correction
                upper_clamp = target_slope_pct * 2.5
                lower_clamp = SlopeConfig.MIN_SKIABLE_PCT
                step_target = max(lower_clamp, min(upper_clamp, step_target))
            else:
                step_target = target_slope_pct

            # Get terrain gradient
            gradient = self._analyzer.compute_gradient(
                lon=current_lon,
                lat=current_lat,
            )
            terrain_slope = gradient.slope_pct
            fall_line = gradient.bearing_deg

            # Calculate traverse angle
            if terrain_slope > 0 and step_target < terrain_slope:
                cos_theta = step_target / terrain_slope
                cos_theta = max(-1.0, min(1.0, cos_theta))
                traverse_angle = degrees(acos(cos_theta))
                traverse_angle = min(
                    max(traverse_angle, PathConfig.MIN_TRAVERSE_ANGLE_DEG),
                    PathConfig.MAX_TRAVERSE_ANGLE_DEG,
                )
            else:
                traverse_angle = PathConfig.MIN_TRAVERSE_ANGLE_DEG

            # Add noise scaled by traverse angle
            noise_factor = (90.0 - traverse_angle) / 90.0
            noise_factor = max(0.0, noise_factor)
            base_noise = 5.0
            noise = random.gauss(0, base_noise * noise_factor)

            # Calculate terrain-derived bearing
            terrain_bearing = (fall_line + side_sign * traverse_angle + noise) % 360

            # Bearing smoothing for flat terrain
            if terrain_slope < flat_terrain_threshold and len(recent_bearings) >= 2:
                sin_sum = sum(np.sin(np.radians(b)) for b in recent_bearings)
                cos_sum = sum(np.cos(np.radians(b)) for b in recent_bearings)
                momentum_bearing = np.degrees(np.arctan2(sin_sum, cos_sum)) % 360
                momentum_weight = PathConfig.MOMENTUM_WEIGHT_FACTOR * (1.0 - terrain_slope / flat_terrain_threshold)
                diff = terrain_bearing - momentum_bearing
                if diff > 180:
                    diff -= 360
                elif diff < -180:
                    diff += 360
                target_bearing = (momentum_bearing + (1.0 - momentum_weight) * diff) % 360
            else:
                target_bearing = terrain_bearing

            # Self-intersection prevention
            if previous_bearing is not None:
                turn_angle = target_bearing - previous_bearing
                while turn_angle > 180:
                    turn_angle -= 360
                while turn_angle < -180:
                    turn_angle += 360
                if abs(turn_angle) > max_turn_per_step:
                    clamped_turn = max_turn_per_step if turn_angle > 0 else -max_turn_per_step
                    target_bearing = (previous_bearing + clamped_turn) % 360

            # Track bearing for smoothing
            recent_bearings.append(target_bearing)
            if len(recent_bearings) > smoothing_window:
                recent_bearings.pop(0)

            previous_bearing = target_bearing

            # Step forward
            next_lon, next_lat = GeoCalculator.destination(
                lon=current_lon,
                lat=current_lat,
                bearing_deg=target_bearing,
                distance_m=step_size,
            )

            # Check bounds - break if outside DEM coverage
            if not (bounds[0] <= next_lon <= bounds[2] and bounds[1] <= next_lat <= bounds[3]):
                break

            next_elev = self._dem.get_elevation(lon=next_lon, lat=next_lat)
            if next_elev is None:
                logger.warning(f"Cannot trace path: invalid elevation at ({next_lon:.6f}, {next_lat:.6f})")
                break

            # Update tracking
            step_dist = GeoCalculator.haversine_distance_m(
                lat1=current_lat,
                lon1=current_lon,
                lat2=next_lat,
                lon2=next_lon,
            )
            step_drop = current_elev - next_elev  # Positive = downhill

            # Update cumulative tracking
            accumulated_drop += step_drop
            total_dist += step_dist

            points.append(PathPoint(lon=next_lon, lat=next_lat, elevation=next_elev))
            current_lon, current_lat, current_elev = next_lon, next_lat, next_elev

        # Calculate final metrics
        if len(points) < PathConfig.MIN_PATH_POINTS:
            logger.warning(f"Path too short: {len(points)} points < {PathConfig.MIN_PATH_POINTS} minimum")
            return None

        total_drop = points[0].elevation - points[-1].elevation
        avg_slope = (total_drop / total_dist * 100) if total_dist > 0 else 0.0
        difficulty = TerrainAnalyzer.classify_difficulty(slope_pct=avg_slope)

        return TracedPath(
            points=points,
            avg_slope_pct=avg_slope,
            total_drop_m=total_drop,
            length_m=total_dist,
            difficulty=difficulty,
            target_slope_pct=target_slope_pct,
        )

"""Path tracing for slope and road generation.

Traces a route that holds a SIGNED target grade, following the terrain:
- Positive target descends the fall line (ski slopes, descending roads)
- Negative target climbs against the fall line (climbing roads)
- Zero target contours across the slope (flat roads)

Implements the Cumulative Drop Tracking algorithm:
- Pre-calculates the (signed) target total drop from path length and target grade
- Tracks accumulated drop as the path is traced
- Dynamically adjusts the step target to converge on the final average

This self-correcting approach eliminates DEM grid artifacts. Sign handling lives
in exactly two places: the reference bearing (fall line for descent, +180° for a
climb) and the step-target clamp; all traverse-angle trig runs on magnitudes.

Reference: DETAILS.md Sections 5, 6
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from math import acos, degrees
from typing import TYPE_CHECKING

import numpy as np

from skiresort_planner.constants import (
    GeometricTuningConfig,
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
    """Result of path tracing (raw path data before conversion to ProposedPathSegment).

    Attributes:
        points: List of PathPoint instances
        avg_slope_pct: Average grade (SIGNED: + descends, − climbs)
        total_drop_m: Total vertical drop in meters (SIGNED: + descends, − climbs)
        length_m: Total path length in meters
        difficulty: Classified difficulty (green/blue/red/black), on grade magnitude
        target_grade_pct: Signed target grade used for tracing
    """

    points: list[PathPoint]
    avg_slope_pct: float
    total_drop_m: float
    length_m: float
    difficulty: str
    target_grade_pct: float


class PathTracer:
    """Traces routes using cumulative drop tracking, holding a signed target grade.

    The algorithm works by:
    1. Setting a signed target total drop from path length and target grade
    2. At each step, computing how much drop is remaining
    3. Adjusting the traverse angle to achieve the remaining drop target

    This creates paths that naturally converge to the target average grade,
    whether descending, climbing, or contouring.
    """

    def __init__(
        self,
        dem: DEMService | None = None,
        analyzer: TerrainAnalyzer | None = None,
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

    def trace_hill(
        self,
        start_lon: float,
        start_lat: float,
        target_grade_pct: float,
        side: str,
        target_length_m: float,
    ) -> TracedPath | None:
        """Trace a route holding a signed target grade, using cumulative drop tracking.

        Args:
            start_lon: Starting longitude
            start_lat: Starting latitude
            target_grade_pct: Signed target grade. Positive descends the fall line,
                negative climbs against it, zero contours across the slope.
            side: "left", "right", or "center" (traverse direction). Only affects the
                path when a traverse angle exists (steeper terrain than the target);
                on a straight fall-line descent or contour, left/right are symmetric.
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
        step_size = GeometricTuningConfig.STEP_SIZE_M
        side_sign = -1 if side == "left" else +1  # Left = negative offset

        # Initialize tracking
        points: list[PathPoint] = [PathPoint(lon=start_lon, lat=start_lat, elevation=start_elev)]
        current_lon, current_lat, current_elev = start_lon, start_lat, start_elev
        total_dist = 0.0

        # Self-intersection prevention
        max_turn_per_step = GeometricTuningConfig.MAX_TURN_PER_STEP_DEG
        previous_bearing: float | None = None

        # Bearing smoothing for flat terrain
        recent_bearings: list[float] = []
        smoothing_window = GeometricTuningConfig.BEARING_SMOOTHING_WINDOW
        flat_terrain_threshold = GeometricTuningConfig.FLAT_TERRAIN_THRESHOLD_PCT

        # Cumulative drop tracking (signed: + descends, − climbs)
        target_total_drop = (target_grade_pct / 100.0) * target_length_m
        accumulated_drop = 0.0

        while total_dist < target_length_m:
            # Calculate remaining drop and distance
            remaining_drop = target_total_drop - accumulated_drop
            remaining_distance = target_length_m - total_dist

            # Dynamic step target for self-correction, clamped to a band that keeps
            # the step running in the target's direction (a descent step never climbs,
            # a climb step never descends, a contour step stays near level).
            if remaining_distance > step_size:
                step_target = (remaining_drop / remaining_distance) * 100.0
                floor = SlopeConfig.MIN_SKIABLE_PCT
                span = abs(target_grade_pct) * GeometricTuningConfig.STEP_TARGET_CLAMP_FACTOR
                if target_grade_pct > 0:  # descend: never gentler than floor, never past span
                    step_target = max(floor, min(span, step_target))
                elif target_grade_pct < 0:  # climb: mirror of the descend band
                    step_target = max(-span, min(-floor, step_target))
                else:  # contour: hold within ±floor of level
                    step_target = max(-floor, min(floor, step_target))
            else:
                step_target = target_grade_pct

            # Get terrain gradient
            gradient = self._analyzer.compute_gradient(
                lon=current_lon,
                lat=current_lat,
            )
            terrain_slope = gradient.slope_pct
            fall_line = gradient.bearing_deg

            # Reference bearing = the direction we progress ALONG the path's length:
            # descend along the fall line, climb against it, contour along either
            # (the ~90° traverse below makes it a contour regardless).
            reference_bearing = (fall_line + 180.0) % 360.0 if target_grade_pct < 0 else fall_line

            # Traverse angle from grade MAGNITUDES (sign-independent): how far off the
            # reference bearing to hold the target grade on this terrain steepness.
            # A zero target → cos_theta 0 → ~90° traverse → a contour.
            step_target_mag = abs(step_target)
            if terrain_slope > 0 and step_target_mag < terrain_slope:
                cos_theta = step_target_mag / terrain_slope
                cos_theta = max(-1.0, min(1.0, cos_theta))
                traverse_angle = degrees(acos(cos_theta))
                traverse_angle = min(
                    max(traverse_angle, GeometricTuningConfig.MIN_TRAVERSE_ANGLE_DEG),
                    GeometricTuningConfig.MAX_TRAVERSE_ANGLE_DEG,
                )
            else:
                traverse_angle = GeometricTuningConfig.MIN_TRAVERSE_ANGLE_DEG

            # Add noise scaled by traverse angle
            noise_factor = (90.0 - traverse_angle) / 90.0
            noise_factor = max(0.0, noise_factor)
            noise = random.gauss(0, GeometricTuningConfig.TRACER_NOISE_BASE * noise_factor)

            # Calculate terrain-derived bearing
            terrain_bearing = (reference_bearing + side_sign * traverse_angle + noise) % 360

            # Bearing smoothing for flat terrain
            if terrain_slope < flat_terrain_threshold and len(recent_bearings) >= 2:
                sin_sum = sum(np.sin(np.radians(b)) for b in recent_bearings)
                cos_sum = sum(np.cos(np.radians(b)) for b in recent_bearings)
                smoothed_bearing = np.degrees(np.arctan2(sin_sum, cos_sum)) % 360
                smoothing_weight = GeometricTuningConfig.BEARING_SMOOTHING_WEIGHT * (
                    1.0 - terrain_slope / flat_terrain_threshold
                )
                diff = terrain_bearing - smoothed_bearing
                if diff > 180:
                    diff -= 360
                elif diff < -180:
                    diff += 360
                target_bearing = (smoothed_bearing + (1.0 - smoothing_weight) * diff) % 360
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

        # Calculate final metrics (signed drop/grade; difficulty on magnitude)
        if len(points) < PathConfig.MIN_PATH_POINTS:
            logger.warning(f"Path too short: {len(points)} points < {PathConfig.MIN_PATH_POINTS} minimum")
            return None

        total_drop = points[0].elevation - points[-1].elevation
        avg_slope = (total_drop / total_dist * 100) if total_dist > 0 else 0.0
        difficulty = TerrainAnalyzer.classify_difficulty(slope_pct=abs(avg_slope))

        return TracedPath(
            points=points,
            avg_slope_pct=avg_slope,
            total_drop_m=total_drop,
            length_m=total_dist,
            difficulty=difficulty,
            target_grade_pct=target_grade_pct,
        )

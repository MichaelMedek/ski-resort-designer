"""Terrain analysis for ski slope planning.

Provides gradient calculation and terrain orientation analysis:
- Weighted multi-point gradient sampling (reduces DEM noise)
- Fall line detection (direction of steepest descent)
- Side slope calculation (cross-slope perpendicular to ski direction)
- Earthwork warning checks (excavation, grading)

Based on Zevenbergen & Thorne (1987) and Horn (1981) algorithms.
Reference: DETAILS.md Sections 2, 3, 4
"""

import logging
from dataclasses import dataclass
from math import atan2, cos, degrees, radians, sin, sqrt
from typing import Optional

from skiresort_planner.constants import (
    PathConfig,
    SlopeConfig,
    StyleConfig,
)
from skiresort_planner.core.dem_service import DEMService
from skiresort_planner.core.geo_calculator import GeoCalculator

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TerrainGradient:
    """Result of terrain gradient calculation.

    Attributes:
        slope_pct: Gradient magnitude as percentage (rise/run * 100)
        bearing_deg: Direction of steepest descent (0-360°, clockwise from North)
    """

    slope_pct: float
    bearing_deg: float


@dataclass(frozen=True)
class TerrainOrientation:
    """Complete terrain orientation at a point.

    Attributes:
        fall_line: Bearing of steepest descent (0-360°)
        slope_pct: Terrain steepness as percentage
        contour_left: Bearing perpendicular to fall line (left)
        contour_right: Bearing perpendicular to fall line (right)
        half_slope_left: 45° diagonal (left of fall line)
        half_slope_right: 45° diagonal (right of fall line)
        difficulty_color: Color based on slope_pct classification
    """

    fall_line: float
    slope_pct: float
    contour_left: float
    contour_right: float
    half_slope_left: float
    half_slope_right: float
    difficulty_color: str


@dataclass(frozen=True)
class SideSlope:
    """Side slope calculation result.

    Attributes:
        slope_pct: Side slope percentage (perpendicular to ski direction)
        direction: "left", "right", or "flat"
    """

    slope_pct: float
    direction: str


class TerrainAnalyzer:
    """Analyzes terrain characteristics for ski slope planning.

    Uses multi-point gradient estimation with weighted sampling for smooth,
    noise-resistant slope calculations.

    Example:
        analyzer = TerrainAnalyzer()
        gradient = analyzer.compute_gradient(lon=10.3, lat=47.0)
        print(f"Slope: {gradient.slope_pct:.1f}%, Fall line: {gradient.bearing_deg:.0f}°")
    """

    def __init__(self, dem: Optional[DEMService] = None):
        """Initialize with optional DEM service.

        Args:
            dem: DEMService instance (creates new if not provided)
        """
        self._dem = dem or DEMService()

    @property
    def dem(self) -> DEMService:
        """Access the DEM service."""
        return self._dem

    @staticmethod
    def classify_difficulty(slope_pct: float) -> str:
        """Classify slope percentage into difficulty level.

        Args:
            slope_pct: Terrain slope as percentage (rise/run * 100)

        Returns:
            Difficulty string: "green", "blue", "red", or "black"
            Extreme slopes (>MAX_SKIABLE_PCT) are classified as "black"
            with a TooSteep warning expected separately.
        """
        for difficulty, (low, high) in SlopeConfig.DIFFICULTY_THRESHOLDS.items():
            if low <= slope_pct < high:
                return difficulty
        # Extreme slopes beyond MAX_SKIABLE_PCT are still "black"
        # The TooSteep warning will flag them separately
        if slope_pct >= SlopeConfig.MAX_SKIABLE_PCT:
            return "black"
        # Negative slopes (uphill) are green
        return "green"

    @staticmethod
    def get_difficulty_color(slope_pct: float) -> str:
        """Get color code for a slope percentage.

        Args:
            slope_pct: Terrain slope as percentage

        Returns:
            Hex color string from StyleConfig.SLOPE_COLORS
        """
        difficulty = TerrainAnalyzer.classify_difficulty(slope_pct=slope_pct)
        return StyleConfig.SLOPE_COLORS[difficulty]

    @staticmethod
    def compute_side_slope(
        start_lon: float,
        start_lat: float,
        end_lon: float,
        end_lat: float,
        analyzer: Optional["TerrainAnalyzer"] = None,
    ) -> SideSlope:
        """Calculate side slope at a point given ski direction.

        Side slope is the terrain gradient component perpendicular to ski direction.
        High side slope means terrain falls steeply to one side.

        Reference: DETAILS.md Section 3.2

        Args:
            start_lon, start_lat: Start point coordinates
            end_lon, end_lat: Next point (defines ski direction)
            analyzer: TerrainAnalyzer instance (creates new if not provided)

        Returns:
            SideSlope with slope_pct and direction.
        """
        if analyzer is None:
            analyzer = TerrainAnalyzer()

        ski_bearing = GeoCalculator.initial_bearing_deg(
            lon1=start_lon,
            lat1=start_lat,
            lon2=end_lon,
            lat2=end_lat,
        )

        gradient = analyzer.compute_gradient(lon=start_lon, lat=start_lat)

        if gradient.slope_pct < 0.5:
            return SideSlope(slope_pct=0.0, direction="flat")

        # Side slope = gradient × sin(angle between ski direction and fall line)
        angle_diff = radians(gradient.bearing_deg - ski_bearing)
        side_slope = gradient.slope_pct * sin(angle_diff)

        # Determine direction
        if abs(side_slope) < 2:
            direction = "flat"
        elif side_slope > 0:
            direction = "right"  # Terrain falls to right when looking downhill
        else:
            direction = "left"

        return SideSlope(slope_pct=side_slope, direction=direction)

    def compute_gradient(self, lon: float, lat: float) -> TerrainGradient:
        """Calculate smoothed terrain gradient using weighted multi-point sampling.

        Uses the "Magic 8" algorithm with two concentric rings:
        - Inner ring (0.5 × step_size): weight 2
        - Outer ring (1.0 × step_size): weight 1

        This reduces DEM noise while maintaining good local terrain sensitivity.
        Based on Zevenbergen & Thorne (1987) and Horn (1981).

        Reference: DETAILS.md Section 2.2

        Args:
            lon: Longitude in decimal degrees
            lat: Latitude in decimal degrees

        Returns:
            TerrainGradient with slope_pct and bearing_deg
        """
        center_elev = self._dem.get_elevation(lon=lon, lat=lat)
        if center_elev is None:
            logger.warning(f"Elevation query returned None at ({lon:.6f}, {lat:.6f}) - outside DEM bounds")
            return TerrainGradient(slope_pct=0.0, bearing_deg=0.0)

        bounds = self._dem.bounds

        # Sample at 8 compass directions on two rings
        samples_x: list[float] = []  # East-West gradient contributions
        samples_y: list[float] = []  # North-South gradient contributions
        total_weight = 0.0

        inner_radius = 0.5 * PathConfig.STEP_SIZE_M  # 15m with 30m step
        outer_radius = 1.0 * PathConfig.STEP_SIZE_M  # 30m with 30m step

        ring_configs = [
            (inner_radius, 2.0),
            (outer_radius, 1.0),
        ]

        for radius_m, weight in ring_configs:
            for angle_deg in [0, 45, 90, 135, 180, 225, 270, 315]:
                sample_lon, sample_lat = GeoCalculator.destination(
                    lon=lon,
                    lat=lat,
                    bearing_deg=angle_deg,
                    distance_m=radius_m,
                )

                # Check bounds
                if not (bounds[0] <= sample_lon <= bounds[2] and bounds[1] <= sample_lat <= bounds[3]):
                    continue

                sample_elev = self._dem.get_elevation(lon=sample_lon, lat=sample_lat)
                if sample_elev is None:
                    continue

                # Calculate slope from center to sample point
                drop = center_elev - sample_elev
                slope = (drop / radius_m) * 100  # As percentage

                # Decompose into x (east) and y (north) components
                angle_rad = radians(angle_deg)
                samples_x.append(slope * sin(angle_rad) * weight)
                samples_y.append(slope * cos(angle_rad) * weight)
                total_weight += weight

        if not samples_x or total_weight == 0:
            return TerrainGradient(slope_pct=0.0, bearing_deg=0.0)

        # Average weighted gradients
        grad_x = sum(samples_x) / total_weight
        grad_y = sum(samples_y) / total_weight

        # Calculate magnitude and direction
        magnitude = sqrt(grad_x**2 + grad_y**2)
        steepest_bearing = (degrees(atan2(grad_x, grad_y)) + 360) % 360

        return TerrainGradient(slope_pct=magnitude, bearing_deg=steepest_bearing)

    def get_orientation(self, lon: float, lat: float) -> Optional[TerrainOrientation]:
        """Get complete terrain orientation at a point.

        Returns orientation info including fall line, slope percentage,
        and derived directions (contours, half-slopes).

        Args:
            lon: Longitude in decimal degrees
            lat: Latitude in decimal degrees

        Returns:
            TerrainOrientation, or None if terrain too flat for skiing.
        """
        gradient = self.compute_gradient(lon=lon, lat=lat)

        if gradient.slope_pct < SlopeConfig.MIN_SKIABLE_PCT:
            return None

        fall_line = gradient.bearing_deg

        return TerrainOrientation(
            fall_line=fall_line,
            slope_pct=gradient.slope_pct,
            contour_left=(fall_line - 90) % 360,
            contour_right=(fall_line + 90) % 360,
            half_slope_left=(fall_line - 45) % 360,
            half_slope_right=(fall_line + 45) % 360,
            difficulty_color=TerrainAnalyzer.get_difficulty_color(gradient.slope_pct),
        )

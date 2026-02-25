"""SlopeSegment - A committed path section between two nodes.

A SlopeSegment is created when a proposed path is committed.
It connects two nodes and stores the full path geometry.

Inherits computed metrics from BaseSlopePath. Adds node connections,
side slope data, and warnings.

Reference: DETAILS.md
"""

from dataclasses import dataclass
from math import floor
from typing import Any

import pyproj
from shapely.geometry import LineString
from shapely.ops import transform as shapely_transform

from skiresort_planner.constants import EarthworkConfig, SlopeConfig
from skiresort_planner.model.base_slope_path import BaseSlopePath
from skiresort_planner.model.path_point import PathPoint
from skiresort_planner.model.warning import (
    ExcavatorWarning,
    TooFlatWarning,
    TooSteepWarning,
    Warning,
)


def _get_utm_zone(lon: float, lat: float) -> str:
    """Get UTM zone EPSG code for given coordinates."""
    zone_number = floor((lon + 180) / 6) + 1
    if lat >= 0:
        return f"EPSG:326{zone_number:02d}"
    return f"EPSG:327{zone_number:02d}"


@dataclass
class SlopeSegment(BaseSlopePath):
    """A committed slope segment between two nodes.

    Inherits points and geometric metrics from BaseSlopePath.

    Attributes:
        id: Unique identifier (e.g., "S1", "S2", ...)
        name: Display name for the segment
        start_node_id: ID of the starting node
        end_node_id: ID of the ending node
        side_slope_pct: Cross-slope percentage at start (terrain-dependent)
        side_slope_dir: "left", "right", or "flat"

    Properties:
        warnings: List of Warning objects based on slope metrics
        has_warnings: Whether segment has any warnings
        all_warnings: List of warning messages as strings
    """

    id: str = ""
    name: str = ""
    start_node_id: str = ""
    end_node_id: str = ""
    side_slope_pct: float = 0.0
    side_slope_dir: str = "flat"

    @property
    def warnings(self) -> list[Warning]:
        """Compute all warnings based on segment metrics.

        Excavator warning triggers when side slope is so steep that even at
        MIN_BELT_WIDTH_M, excavation would exceed threshold.
        """
        result: list[Warning] = []

        # Excavator warning: side slope exceeds what MIN width can handle
        # Formula: H_edge = (side_slope_pct * width) / 200
        # Warning when: (side_slope_pct * MIN_WIDTH) / 200 > threshold
        side_slope_limit = (EarthworkConfig.EXCAVATOR_THRESHOLD_M * 200) / EarthworkConfig.MIN_BELT_WIDTH_M
        if abs(self.side_slope_pct) > side_slope_limit:
            result.append(
                ExcavatorWarning(
                    side_slope_pct=abs(self.side_slope_pct),
                    belt_width_m=self.width_m,
                    side_slope_dir=self.side_slope_dir,
                )
            )

        # Too steep warning
        if self.avg_slope_pct >= SlopeConfig.MAX_SKIABLE_PCT:
            result.append(
                TooSteepWarning(
                    slope_pct=self.avg_slope_pct,
                    max_threshold_pct=SlopeConfig.MAX_SKIABLE_PCT,
                )
            )

        # Too flat warning
        if self.avg_slope_pct < SlopeConfig.MIN_SKIABLE_PCT:
            result.append(
                TooFlatWarning(
                    slope_pct=self.avg_slope_pct,
                    min_threshold_pct=SlopeConfig.MIN_SKIABLE_PCT,
                )
            )

        return result

    @property
    def has_warnings(self) -> bool:
        """Check if segment has any warnings."""
        return len(self.warnings) > 0

    def get_linestring(self) -> LineString:
        """Get Shapely LineString for path geometry.

        Returns:
            Shapely LineString of the path.
        """
        return LineString([(p.lon, p.lat) for p in self.points])

    @property
    def width_m(self) -> float:
        """Belt width in meters based on side slope steepness.

        Width is determined by side slope to keep excavation within threshold:
        width = (EXCAVATOR_THRESHOLD_M * 200) / abs(side_slope_pct)

        Returns:
            Width in meters, clamped to [MIN_BELT_WIDTH_M, MAX_BELT_WIDTH_M].
            Returns MAX_BELT_WIDTH_M for flat terrain (side slope < 1%).
        """
        # Flat terrain: use maximum width to avoid zero division
        if abs(self.side_slope_pct) < 1.0:
            return float(EarthworkConfig.MAX_BELT_WIDTH_M)

        # Calculate width from side slope to stay within excavation threshold
        adaptive_width = (EarthworkConfig.EXCAVATOR_THRESHOLD_M * 200) / abs(self.side_slope_pct)

        # Clamp to allowed range
        return max(EarthworkConfig.MIN_BELT_WIDTH_M, min(EarthworkConfig.MAX_BELT_WIDTH_M, adaptive_width))

    def get_belt_polygon(self) -> list[tuple[float, float]]:
        """Get belt polygon coordinates (buffered ribbon in meters).

        Uses adaptive width based on side slope to stay within excavation
        threshold. UTM projection used for accurate meter-based widths.
        Buffer uses round cap/join for smooth turns.

        Returns:
            List of (lon, lat) tuples for polygon boundary.
        """
        line = self.get_linestring()
        if line.is_empty or len(line.coords) < 2:
            return []

        belt_width = self.width_m

        # Get center point for UTM zone
        center_lon = (line.bounds[0] + line.bounds[2]) / 2
        center_lat = (line.bounds[1] + line.bounds[3]) / 2
        utm_crs = _get_utm_zone(lon=center_lon, lat=center_lat)

        # Create transformers
        wgs84 = pyproj.CRS("EPSG:4326")
        utm = pyproj.CRS(utm_crs)
        to_utm = pyproj.Transformer.from_crs(wgs84, utm, always_xy=True).transform
        to_wgs84 = pyproj.Transformer.from_crs(utm, wgs84, always_xy=True).transform

        # Buffer in UTM (meters)
        line_utm = shapely_transform(to_utm, line)
        buffered_utm = line_utm.buffer(
            belt_width / 2,
            cap_style="round",
            join_style="round",
        )

        if buffered_utm.is_empty:
            return []

        buffered_wgs84 = shapely_transform(to_wgs84, buffered_utm)

        if hasattr(buffered_wgs84, "exterior"):
            return [(float(c[0]), float(c[1])) for c in buffered_wgs84.exterior.coords]
        return []

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SlopeSegment":
        """Create SlopeSegment from dictionary."""
        return cls(
            id=data["id"],
            name=data["name"],
            points=[PathPoint(**p) for p in data["points"]],
            start_node_id=data["start_node_id"],
            end_node_id=data["end_node_id"],
            side_slope_pct=data.get("side_slope_pct", 0.0),
            side_slope_dir=data.get("side_slope_dir", "flat"),
        )

    def __repr__(self) -> str:
        return f"SlopeSegment({self.id}, {self.difficulty}, {self.avg_slope_pct:.1f}%, {self.length_m:.0f}m)"

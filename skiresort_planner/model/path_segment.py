"""PathSegment - A committed path section between two nodes.

A PathSegment is created when a proposed path is committed.
It connects two nodes and stores the full path geometry.

Inherits computed metrics from Path. Adds node connections,
side slope data, and warnings.

Reference: DETAILS.md
"""

from dataclasses import dataclass
from enum import StrEnum
from math import floor
from typing import TYPE_CHECKING, cast

import pyproj
from shapely.geometry import LineString
from shapely.ops import transform as shapely_transform

from skiresort_planner.constants import EarthworkConfig, SlopeConfig
from skiresort_planner.core.terrain_analyzer import SideDirection
from skiresort_planner.enum_utils import enum_eq
from skiresort_planner.model.path_geometry import Path
from skiresort_planner.model.path_point import PathPoint
from skiresort_planner.model.warning import (
    ExcavatorWarning,
    TooFlatWarning,
    TooSteepWarning,
    Warning,
)

if TYPE_CHECKING:
    from skiresort_planner.core.dem_service import DEMService
    from skiresort_planner.model.node import Node


def _get_utm_zone(lon: float, lat: float) -> str:
    """Get UTM zone EPSG code for given coordinates."""
    zone_number = floor((lon + 180) / 6) + 1
    if lat >= 0:
        return f"EPSG:326{zone_number:02d}"
    return f"EPSG:327{zone_number:02d}"


class SegmentKind(StrEnum):
    """What a committed segment IS — a ski slope or a vehicle road."""

    SLOPE = "slope"
    ROAD = "road"


@dataclass
class PathSegment(Path):
    """A committed slope or road segment between two nodes.

    Inherits points and geometric metrics from Path.

    Attributes:
        id: Unique identifier (e.g., "S1", "S2", ...)
        name: Display name for the segment
        start_node_id: ID of the starting node
        end_node_id: ID of the ending node
        side_slope_pct: Cross-slope percentage at start (terrain-dependent)
        side_slope_dir: Cross-slope lean direction (SideDirection)
        kind: Whether this segment is a ski slope or a vehicle road

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
    side_slope_dir: SideDirection = SideDirection.FLAT
    kind: SegmentKind = SegmentKind.SLOPE

    @property
    def warnings(self) -> list[Warning]:
        """Compute all warnings based on segment metrics.

        Excavator warning triggers when side slope is so steep that even at
        minimum belt width for this difficulty, excavation would exceed threshold.
        """
        result: list[Warning] = []

        # Excavator warning: side slope exceeds what MIN width can handle
        # Formula: H_edge = (side_slope_pct * width) / 200
        # Warning when: (side_slope_pct * MIN_WIDTH) / 200 > threshold
        min_width, _ = EarthworkConfig.BELT_WIDTH_LIMITS[self.difficulty]
        side_slope_limit = (EarthworkConfig.EXCAVATOR_THRESHOLD_M * 200) / min_width
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
        """Belt width in meters.

        Roads are a fixed-width vehicle ribbon. Slopes adapt width to side slope
        to keep excavation within threshold: width = (EXCAVATOR_THRESHOLD_M * 200)
        / abs(side_slope_pct), clamped to difficulty-specific limits.

        Returns:
            Width in meters. Constant for roads; for slopes, clamped to difficulty
            limits (max width on flat terrain, side slope < 1%).
        """
        if enum_eq(a=self.kind, b=SegmentKind.ROAD):
            return float(EarthworkConfig.ROAD_WIDTH_M)

        # Get difficulty-specific limits
        min_width, max_width = EarthworkConfig.BELT_WIDTH_LIMITS[self.difficulty]

        # Flat terrain: use maximum width to avoid zero division
        if abs(self.side_slope_pct) < 1.0:
            return float(max_width)

        # Calculate width from side slope to stay within excavation threshold
        adaptive_width = (EarthworkConfig.EXCAVATOR_THRESHOLD_M * 200) / abs(self.side_slope_pct)

        # Clamp to allowed range for this difficulty
        return max(min_width, min(max_width, adaptive_width))

    def restitch(self, start_node: "Node", end_node: "Node", dem: "DEMService") -> None:
        """Re-anchor this segment's drawn polyline after an endpoint node moved.

        Snaps the first point to `start_node` and the last to `end_node` (the same exact-coordinate
        snap that commit does), then re-drapes every point's elevation from the DEM so the whole
        polyline sits on current terrain. Keeps identity + styling (id, name, kind, side slope);
        derived metrics (length/drop/slope/difficulty/belt) are computed from `points`, so they
        refresh automatically. Route is preserved — this re-drapes existing geometry, it does not
        re-plan (mirrors OSM import's re-sample-in-place).

        Args:
            start_node: The (possibly moved) node this segment starts at.
            end_node: The (possibly moved) node this segment ends at.
            dem: DEM service for elevation re-draping.

        Raises:
            ValueError: If any point falls on DEM nodata.
        """
        redraped: list[PathPoint] = []
        for i, p in enumerate(self.points):
            if i == 0:
                lon, lat = start_node.lon, start_node.lat
            elif i == len(self.points) - 1:
                lon, lat = end_node.lon, end_node.lat
            else:
                lon, lat = p.lon, p.lat
            elevation = dem.get_elevation(lon=lon, lat=lat)
            if elevation is None:
                raise ValueError(f"restitch of segment {self.id}: point ({lat:.5f}, {lon:.5f}) has no DEM elevation")
            redraped.append(PathPoint(lon=lon, lat=lat, elevation=elevation))
        self.points = redraped

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
    def from_dict(cls, data: dict[str, object]) -> "PathSegment":
        """Create PathSegment from dictionary."""
        return cls(
            id=cast(str, data["id"]),
            name=cast(str, data["name"]),
            points=[PathPoint(**p) for p in cast(list[dict[str, float]], data["points"])],
            start_node_id=cast(str, data["start_node_id"]),
            end_node_id=cast(str, data["end_node_id"]),
            side_slope_pct=cast(float, data.get("side_slope_pct", 0.0)),
            side_slope_dir=SideDirection(cast(str, data.get("side_slope_dir", SideDirection.FLAT.value))),
            # Pre-enum saves have no "kind" → default to SLOPE.
            kind=SegmentKind(cast(str, data.get("kind", SegmentKind.SLOPE.value))),
        )

    def __repr__(self) -> str:
        return f"PathSegment({self.id}, {self.difficulty}, {self.avg_slope_pct:.1f}%, {self.length_m:.0f}m)"

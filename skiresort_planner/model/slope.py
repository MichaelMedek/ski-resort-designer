"""Slope - A complete ski run composed of multiple segments.

A Slope is created when the user clicks "Finish Slope".
It groups multiple SlopeSegments into a single named run.

Difficulty is derived from maximum segment avg_slope.
Creative naming generates memorable descriptive names.

Reference: DETAILS.md
"""

import logging
import random
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from skiresort_planner.constants import EntityPrefixes, NameConfig
from skiresort_planner.core.terrain_analyzer import TerrainAnalyzer

if TYPE_CHECKING:
    from skiresort_planner.model.path_point import PathPoint
    from skiresort_planner.model.slope_segment import SlopeSegment

logger = logging.getLogger(__name__)


@dataclass
class Slope:
    """A complete ski slope composed of one or more segments.

    Created when user finalizes a slope. Groups segments into a single
    named run with unified difficulty classification.

    Attributes:
        id: Unique identifier (e.g., "SL1")
        name: Display name with number prefix
        number: Slope number for easy reference
        segment_ids: Ordered list of segment IDs
        start_node_id: ID of first node (top of slope)
        end_node_id: ID of last node (bottom of slope)

    Example:
        slope = Slope(
            id="SL1",
            name="1 (Thunder Ridge)",
            segment_ids=["S1", "S2", "S3"],
            start_node_id="N1",
            end_node_id="N4",
        )
    """

    id: str
    name: str
    segment_ids: list[str]
    start_node_id: str
    end_node_id: str

    @property
    def number(self) -> int:
        """Slope number derived from ID."""
        return Slope.number_from_id(slope_id=self.id)

    @staticmethod
    def number_from_id(slope_id: str) -> int:
        """Extract slope number from slope ID.

        Args:
            slope_id: Slope ID (e.g., "SL1", "SL5")

        Returns:
            Numeric part of the ID.
        """
        return int(slope_id[len(EntityPrefixes.SLOPE) :])

    @staticmethod
    def generate_name(
        difficulty: str,
        slope_id: str,
        start_elevation: float,
        end_elevation: float,
        avg_bearing: float,
    ) -> str:
        """Generate a creative, descriptive slope name.

        Args:
            difficulty: Slope difficulty (green, blue, red, black)
            slope_id: Slope ID (e.g., "SL1")
            start_elevation: Starting elevation in meters
            end_elevation: Ending elevation in meters
            avg_bearing: Average bearing in degrees

        Returns:
            Creative slope name like "1 (Thunder Ridge)"
        """
        slope_number = Slope.number_from_id(slope_id=slope_id)
        prefixes = NameConfig.SLOPE_PREFIXES[difficulty]
        prefix = random.choice(prefixes)

        direction = NameConfig.get_compass_direction(bearing_deg=avg_bearing) + " "

        suffix = random.choice(NameConfig.SLOPE_SUFFIXES)

        name = f"{prefix} {direction}{suffix}"

        drop = start_elevation - end_elevation
        if drop > 500:
            name = f"{prefix} {direction}Summit {suffix}"
        elif drop > 300:
            name = f"{prefix} {direction}Big {suffix}"

        return f"{slope_number} ({name})"

    def get_difficulty(self, segments: dict[str, "SlopeSegment"]) -> str:
        """Derive difficulty from maximum avg_slope among segments.

        Classification uses the steepest segment to determine the
        overall slope rating (most challenging section defines difficulty).

        Args:
            segments: Dict of segment_id -> SlopeSegment

        Returns:
            Difficulty string: green, blue, red, or black
        """
        return TerrainAnalyzer.classify_difficulty(slope_pct=self.get_steepest_segment_slope(segments=segments))

    def get_total_length(self, segments: dict[str, "SlopeSegment"]) -> float:
        """Calculate total length in meters.

        Args:
            segments: Dict of segment_id -> SlopeSegment

        Returns:
            Total length in meters.
        """
        return sum(segments[sid].length_m for sid in self.segment_ids if sid in segments)

    def get_total_drop(self, segments: dict[str, "SlopeSegment"]) -> float:
        """Calculate total vertical drop in meters.

        Args:
            segments: Dict of segment_id -> SlopeSegment

        Returns:
            Total vertical drop in meters.
        """
        return sum(segments[sid].total_drop_m for sid in self.segment_ids if sid in segments)

    def get_steepest_segment_slope(self, segments: dict[str, "SlopeSegment"]) -> float:
        """Get the avg_slope of the steepest segment.

        Used for difficulty classification.

        Args:
            segments: Dict of segment_id -> SlopeSegment

        Returns:
            Maximum avg_slope_pct among segments.
        """
        max_slope = 0.0
        for seg_id in self.segment_ids:
            seg = segments.get(seg_id)
            if seg and seg.avg_slope_pct > max_slope:
                max_slope = seg.avg_slope_pct
        return max_slope

    def get_all_points(self, segments: dict[str, "SlopeSegment"]) -> list["PathPoint"]:
        """Get all points from all segments, deduplicated at junctions.

        Args:
            segments: Dict of segment_id -> SlopeSegment

        Returns:
            List of PathPoints for the entire slope.
        """
        all_points: list["PathPoint"] = []
        for seg_id in self.segment_ids:
            seg = segments.get(seg_id)
            if seg:
                if all_points and seg.points:
                    all_points.extend(seg.points[1:])  # Skip duplicate junction
                else:
                    all_points.extend(seg.points)
        if len(all_points) == 0:
            raise ValueError("Slope must have at least one point")
        return all_points

    def has_warnings(self, segments: dict[str, "SlopeSegment"]) -> bool:
        """Check if any segment has warnings.

        Args:
            segments: Dict of segment_id -> SlopeSegment

        Returns:
            True if any segment has warnings.
        """
        return any(segments[sid].has_warnings for sid in self.segment_ids if sid in segments)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Slope":
        """Create Slope from dictionary."""
        return cls(
            id=data["id"],
            name=data["name"],
            segment_ids=data["segment_ids"],
            start_node_id=data["start_node_id"],
            end_node_id=data["end_node_id"],
        )

    def __repr__(self) -> str:
        return f"Slope({self.id}, {len(self.segment_ids)} segments)"

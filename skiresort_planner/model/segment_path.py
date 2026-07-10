"""SegmentPath - shared base for entities built from a chain of segments.

Both a Slope (ski run) and a Road (vehicle road) are an ordered group of
PathSegments running between a start node and an end node. All geometry
derived from that chain — length, drop, steepest gradient, point list,
warnings — is identical for both and lives here. Subclasses add only what
is unique to them (e.g. Slope difficulty/naming, Road elevation gain).

Reference: DETAILS.md
"""

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, TypeVar

from skiresort_planner.model.path_geometry import Path

if TYPE_CHECKING:
    from skiresort_planner.model.path_point import PathPoint
    from skiresort_planner.model.path_segment import PathSegment

logger = logging.getLogger(__name__)

T = TypeVar("T", bound="SegmentPath")


@dataclass
class SegmentPath:
    """A named chain of segments between two nodes.

    Attributes:
        id: Unique identifier (prefix defined by subclass ID_PREFIX).
        name: Display name with number prefix.
        segment_ids: Ordered list of segment IDs.
        start_node_id: ID of first node.
        end_node_id: ID of last node.
    """

    id: str
    name: str
    segment_ids: list[str]
    start_node_id: str
    end_node_id: str

    # Subclasses set their entity id prefix (e.g. "SL", "R").
    ID_PREFIX: ClassVar[str] = ""

    @property
    def number(self) -> int:
        """Entity number derived from ID."""
        return type(self).number_from_id(self.id)

    @classmethod
    def number_from_id(cls, entity_id: str) -> int:
        """Extract the numeric part from an entity ID (e.g. 'SL5' -> 5)."""
        return int(entity_id[len(cls.ID_PREFIX) :])

    def get_total_length(self, segments: dict[str, "PathSegment"]) -> float:
        """Total horizontal length in meters across all segments."""
        return sum(segments[sid].length_m for sid in self.segment_ids if sid in segments)

    def get_total_drop(self, segments: dict[str, "PathSegment"]) -> float:
        """Total vertical drop in meters (sum of signed segment drops)."""
        return sum(segments[sid].total_drop_m for sid in self.segment_ids if sid in segments)

    def get_max_gradient(self, segments: dict[str, "PathSegment"]) -> float:
        """Steepest-section gradient magnitude across the whole path.

        Rolls the ROLLING_WINDOW_M steepest-section window over the ENTIRE point
        chain (across segment boundaries), not per-segment — so a steep stretch
        spanning two segments is measured as one window, and short segments do not
        each fall back to their own average. `max_slope_pct` is already a
        magnitude (climb or descent), so this reflects steepness in either
        direction. This is the authoritative value for difficulty and the road cap.
        """
        return Path(points=self.get_all_points(segments=segments)).max_slope_pct

    def get_all_points(self, segments: dict[str, "PathSegment"]) -> list["PathPoint"]:
        """All points across segments, deduplicated at shared junction nodes."""
        all_points: list["PathPoint"] = []
        for seg_id in self.segment_ids:
            seg = segments.get(seg_id)
            if seg:
                if all_points and seg.points:
                    all_points.extend(seg.points[1:])  # Skip duplicate junction
                else:
                    all_points.extend(seg.points)
        if len(all_points) == 0:
            raise ValueError(f"{type(self).__name__} must have at least one point")
        return all_points

    def has_warnings(self, segments: dict[str, "PathSegment"]) -> bool:
        """True if any segment carries a warning."""
        return any(segments[sid].has_warnings for sid in self.segment_ids if sid in segments)

    @classmethod
    def from_dict(cls: type[T], data: dict[str, Any]) -> T:
        """Create an instance from a serialized dict."""
        return cls(
            id=data["id"],
            name=data["name"],
            segment_ids=data["segment_ids"],
            start_node_id=data["start_node_id"],
            end_node_id=data["end_node_id"],
        )

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.id}, {len(self.segment_ids)} segments)"

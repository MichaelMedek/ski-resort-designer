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
from typing import TYPE_CHECKING, ClassVar, TypeVar, cast

from skiresort_planner.constants import PathConfig
from skiresort_planner.model.node_connected import NodeConnected
from skiresort_planner.model.path_segment import SegmentKind

if TYPE_CHECKING:
    from skiresort_planner.model.path_point import PathPoint
    from skiresort_planner.model.path_segment import PathSegment

logger = logging.getLogger(__name__)

T = TypeVar("T", bound="SegmentPath")


def steepest_section_pct(segments: list["PathSegment"]) -> float:
    """Steepest-section gradient magnitude across a chain of segments.

    Max of each segment's own rolling-window steepest section (`max_slope_pct`). Counts every
    segment at least SEGMENT_LENGTH_MIN_M long (the builder's minimum) — a short-but-steep pitch
    (e.g. a 260m 55% wall) is a real black section and must drive the rating. If none reach that,
    fall back to all so the value is never empty.
    """
    long_enough = [s for s in segments if s.length_m >= PathConfig.SEGMENT_LENGTH_MIN_M]
    counted = long_enough or segments
    return max((s.max_slope_pct for s in counted), default=0.0)


@dataclass
class SegmentPath(NodeConnected):
    """A named chain of segments between two nodes.

    Attributes:
        id: Unique identifier (prefix defined by subclass ID_PREFIX).
        name: Display name with number prefix.
        segment_ids: Ordered list of segment IDs.
        start_node_id: Boundary node the chain starts at (the first segment's start node).
        end_node_id: Boundary node the chain ends at (the last segment's end node).
    """

    id: str
    name: str
    segment_ids: list[str]
    start_node_id: str
    end_node_id: str
    source: str | None = None  # provenance tag (e.g. EntitySource.OSM); None for hand-built paths

    # Subclasses set their entity id prefix (e.g. "SL", "R").
    ID_PREFIX: ClassVar[str] = ""
    # Reload-safe entity discriminator (StrEnum): concrete subclasses set it (Slope→SLOPE, Road→ROAD).
    kind: ClassVar[SegmentKind]

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
        return sum(segments[sid].length_m for sid in self.segment_ids)

    def get_total_drop(self, segments: dict[str, "PathSegment"]) -> float:
        """Total vertical drop in meters (sum of signed segment drops)."""
        return sum(segments[sid].total_drop_m for sid in self.segment_ids)

    def get_max_gradient(self, segments: dict[str, "PathSegment"]) -> float:
        """Steepest-section gradient magnitude, measured PER SEGMENT.

        Returns the max of each segment's own rolling-window steepest section
        (`max_slope_pct`, already a magnitude), NOT a window rolled across the whole
        chain. A window spanning a junction would fold several independently-validated
        segments into one figure that can exceed the per-segment cap even though every
        segment is legal and the junctions are continuous — deliberately avoided here to
        keep the metric simple and consistent with how segments are validated at commit.

        Only segments at least SEGMENT_LENGTH_MIN_M long are counted (a sub-minimum sliver has
        no meaningful section); a real 260m 55% wall counts and rightly makes the slope black. If
        NO segment reaches that length, fall back to the max over all segments so it's never empty.
        """
        present = [segments[sid] for sid in self.segment_ids]
        return steepest_section_pct(segments=present)

    def get_all_points(self, segments: dict[str, "PathSegment"]) -> list["PathPoint"]:
        """All points across segments, deduplicated at shared junction nodes."""
        all_points: list[PathPoint] = []
        for seg_id in self.segment_ids:
            seg = segments[seg_id]
            if all_points and seg.points:
                all_points.extend(seg.points[1:])  # Skip duplicate junction
            else:
                all_points.extend(seg.points)
        if len(all_points) == 0:
            raise ValueError(f"{type(self).__name__} must have at least one point")
        return all_points

    def has_warnings(self, segments: dict[str, "PathSegment"]) -> bool:
        """True if any segment carries a warning."""
        return any(segments[sid].has_warnings for sid in self.segment_ids)

    def center(self, segments: dict[str, "PathSegment"]) -> tuple[float, float]:
        """(lon, lat) midpoint of the group's first-start → last-end point.

        Args:
            segments: Dict of segment_id -> PathSegment.

        Returns:
            (lon, lat) midpoint.
        """
        first, last = segments[self.segment_ids[0]], segments[self.segment_ids[-1]]
        start_pt, end_pt = first.points[0], last.points[-1]
        return ((start_pt.lon + end_pt.lon) / 2, (start_pt.lat + end_pt.lat) / 2)

    @classmethod
    def from_dict(cls: type[T], data: dict[str, object]) -> T:
        """Create an instance from a serialized dict."""
        return cls(
            id=cast(str, data["id"]),
            name=cast(str, data["name"]),
            segment_ids=cast(list[str], data["segment_ids"]),
            start_node_id=cast(str, data["start_node_id"]),
            end_node_id=cast(str, data["end_node_id"]),
            source=cast("str | None", data.get("source")),
        )

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.id}, {len(self.segment_ids)} segments)"

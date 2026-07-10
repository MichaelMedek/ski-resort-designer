"""Road - A vehicle road composed of one or more segments.

A Road is a route for cars: it connects two points on the terrain while
keeping the gradient within a gentle band (PathConfig.ROAD_MAX_GRADIENT_PCT),
so vehicles can climb, descend, or run flat but never traverse steep ground.

Shared segment-chain geometry lives in SegmentPath; this class adds only
road-specific naming.

Reference: DETAILS.md
"""

import logging
from dataclasses import dataclass
from typing import ClassVar

from skiresort_planner.constants import EntityPrefixes, NameConfig
from skiresort_planner.model.segment_path import SegmentPath

logger = logging.getLogger(__name__)


@dataclass
class Road(SegmentPath):
    """A vehicle road composed of one or more gentle-gradient segments.

    Created when the user finishes connecting two points in Road mode.
    Unlike a slope, a road has no difficulty rating and may climb uphill.

    Example:
        road = Road(
            id="R1",
            name="1 (North Access)",
            segment_ids=["S1", "S2"],
            start_node_id="N1",
            end_node_id="N3",
        )
    """

    ID_PREFIX: ClassVar[str] = EntityPrefixes.ROAD

    @staticmethod
    def generate_name(road_id: str, avg_bearing: float) -> str:
        """Generate a road name like '1 (North Access)'.

        Args:
            road_id: Road ID (e.g., "R1")
            avg_bearing: Average bearing in degrees, for a compass label.

        Returns:
            Road name with number prefix.
        """
        road_number = Road.number_from_id(road_id)
        direction = NameConfig.get_compass_direction(bearing_deg=avg_bearing)
        return f"{road_number} ({direction} Access)"

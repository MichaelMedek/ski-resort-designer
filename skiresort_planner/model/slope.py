"""Slope - A complete ski run composed of multiple segments.

Created on "Finish Slope"; groups PathSegments into a named run. Difficulty derives from the
steepest section; naming is creative. Shared chain geometry lives in SegmentPath; this class
adds only slope-specific difficulty classification and naming. Reference: DETAILS.md
"""

import logging
import random
from dataclasses import dataclass
from typing import ClassVar

from skiresort_planner.constants import EntityPrefixes, NameConfig
from skiresort_planner.core.terrain_analyzer import TerrainAnalyzer
from skiresort_planner.model.path_segment import PathSegment, SegmentKind
from skiresort_planner.model.segment_path import SegmentPath

logger = logging.getLogger(__name__)


@dataclass
class Slope(SegmentPath):
    """A complete ski slope composed of one or more segments.

    Created when user finalizes a slope. Groups segments into a single
    named run with unified difficulty classification.

    Example:
        slope = Slope(
            id="SL1",
            name="1 (Thunder Ridge)",
            segment_ids=["S1", "S2", "S3"],
            start_node_id="N1",
            end_node_id="N4",
        )
    """

    ID_PREFIX: ClassVar[str] = EntityPrefixes.SLOPE
    kind: ClassVar[SegmentKind] = SegmentKind.SLOPE

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
        slope_number = Slope.number_from_id(slope_id)
        prefixes = NameConfig.SLOPE_PREFIXES[difficulty]
        prefix = random.choice(prefixes)

        direction = NameConfig.get_compass_direction(bearing_deg=avg_bearing) + " "

        suffix = random.choice(NameConfig.SLOPE_SUFFIXES)

        name = f"{prefix} {direction}{suffix}"

        drop = start_elevation - end_elevation
        if drop > NameConfig.SUMMIT_RISE_M:
            name = f"{prefix} {direction}Summit {suffix}"
        elif drop > NameConfig.BIG_DROP_M:
            name = f"{prefix} {direction}Big {suffix}"

        return f"{slope_number} ({name})"

    def get_difficulty(self, segments: dict[str, "PathSegment"]) -> str:
        """Derive difficulty from the steepest section among segments.

        Classification uses the steepest segment to determine the
        overall slope rating (most challenging section defines difficulty).

        Args:
            segments: Dict of segment_id -> PathSegment

        Returns:
            Difficulty string: green, blue, red, or black
        """
        return TerrainAnalyzer.classify_difficulty(slope_pct=self.get_max_gradient(segments=segments))

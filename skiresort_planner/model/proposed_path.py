"""ProposedPathSegment - A slope segment proposal before committing to the graph.

ProposedPathSegment represents a potential slope segment that has been traced
but not yet committed. It inherits computed metrics from Path.

Created by PathFactory, consumed by ResortGraph.commit_paths().

Reference: DETAILS.md
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from skiresort_planner.constants import SlopeConfig
from skiresort_planner.core.terrain_analyzer import TerrainAnalyzer
from skiresort_planner.model.path_geometry import Path
from skiresort_planner.model.path_segment import SegmentKind

if TYPE_CHECKING:
    pass


@dataclass
class ProposedPathSegment(Path):
    """A proposed slope or road segment before committing to the graph.

    Inherits points and geometric metrics from Path.

    Attributes:
        target_slope_pct: Target slope used during tracing
        target_difficulty: Requested difficulty level
        sector_name: Name of sector (for multi-sector generation)
        is_connector: Whether this is a connection path
        target_node_id: Target node ID for connections
        start_node_id: Start node ID when extending from an existing node — commit
            reuses it exactly (never creates a near-duplicate). Empty when the start
            is a brand-new point.
        kind: Whether this proposal becomes a ski slope or a vehicle road
    """

    target_slope_pct: float = 0.0
    target_difficulty: str = ""
    sector_name: str = ""
    is_connector: bool = False
    target_node_id: str = ""
    start_node_id: str = ""
    kind: SegmentKind = SegmentKind.SLOPE

    @property
    def difficulty(self) -> str:
        """Difficulty with a safety margin, so the preview never reads softer than the finished
        slope: finish-time smoothing may nudge the steepest section a hair. Committed entities
        classify margin-free (the honest value); only proposals bias toward the harder band.
        """
        return TerrainAnalyzer.classify_difficulty(
            slope_pct=self.max_slope_pct, margin_pct=SlopeConfig.SLOPE_DIFFICULTY_MARGIN_PCT
        )

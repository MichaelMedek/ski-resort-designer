"""ProposedSlopeSegment - A slope segment proposal before committing to the graph.

ProposedSlopeSegment represents a potential slope segment that has been traced
but not yet committed. It inherits computed metrics from BaseSlopePath.

Created by PathFactory, consumed by ResortGraph.commit_paths().

Reference: DETAILS.md
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from skiresort_planner.model.base_slope_path import BaseSlopePath

if TYPE_CHECKING:
    pass


@dataclass
class ProposedSlopeSegment(BaseSlopePath):
    """A proposed slope segment before committing to the graph.

    Inherits points and geometric metrics from BaseSlopePath.

    Attributes:
        target_slope_pct: Target slope used during tracing
        target_difficulty: Requested difficulty level
        sector_name: Name of sector (for multi-sector generation)
        is_connector: Whether this is a connection path
        target_node_id: Target node ID for connections
    """

    target_slope_pct: float = 0.0
    target_difficulty: str = ""
    sector_name: str = ""
    is_connector: bool = False
    target_node_id: str = ""

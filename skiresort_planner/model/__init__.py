"""Data model classes for ski resort graph representation.

Follows the separation of Geometry (where things are) vs Topology (how things connect):
- PathPoint: Geometry atom (lon, lat, elevation)
- Node: Junction point (wraps PathPoint, has ID)
- BaseSlopePath: Base class for slope paths with computed metrics
- SlopeSegment: Committed path section between nodes
- ProposedSlopeSegment: Path proposal before committing
- Slope: Complete ski run (collection of segments)
- Lift: Uphill connection between nodes
- Pylon: Support structure for lifts
- Warning: Construction warnings for segments
- ResortGraph: Central manager owning all entities

Data structure details documented in DETAILS.md.
"""

from skiresort_planner.model.base_slope_path import BaseSlopePath
from skiresort_planner.model.lift import Lift
from skiresort_planner.model.node import Node
from skiresort_planner.model.path_point import PathPoint
from skiresort_planner.model.proposed_path import ProposedSlopeSegment
from skiresort_planner.model.pylon import Pylon
from skiresort_planner.model.resort_graph import (
    ResortGraph,
    UndoAction,
)
from skiresort_planner.model.slope import Slope
from skiresort_planner.model.slope_segment import SlopeSegment
from skiresort_planner.model.warning import (
    ExcavatorWarning,
    TooFlatWarning,
    TooSteepWarning,
    Warning,
)

__all__ = [
    "PathPoint",
    "Node",
    "BaseSlopePath",
    "Warning",
    "ExcavatorWarning",
    "TooSteepWarning",
    "TooFlatWarning",
    "SlopeSegment",
    "ProposedSlopeSegment",
    "Slope",
    "Pylon",
    "Lift",
    "ResortGraph",
    "UndoAction",
]

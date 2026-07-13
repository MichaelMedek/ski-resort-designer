"""Data model classes for ski resort graph representation.

Follows the separation of Geometry (where things are) vs Topology (how things connect):
- PathPoint: Geometry atom (lon, lat, elevation)
- Node: Junction point (wraps PathPoint, has ID)
- Path: Base class for slope paths with computed metrics
- PathSegment: Committed path section between nodes
- ProposedPathSegment: Path proposal before committing
- Slope: Complete ski run (collection of segments)
- Lift: Uphill connection between nodes
- Pylon: Support structure for lifts
- Warning: Construction warnings for segments
- ResortGraph: Central manager owning all entities

Data structure details documented in DETAILS.md.
"""

from skiresort_planner.model.actions import UndoAction
from skiresort_planner.model.lift import Lift
from skiresort_planner.model.node import Node
from skiresort_planner.model.path_geometry import Path
from skiresort_planner.model.path_point import PathPoint
from skiresort_planner.model.path_segment import PathSegment
from skiresort_planner.model.proposed_path import ProposedPathSegment
from skiresort_planner.model.pylon import Pylon
from skiresort_planner.model.resort_graph import ResortGraph
from skiresort_planner.model.slope import Slope
from skiresort_planner.model.warning import (
    ExcavatorWarning,
    TooFlatWarning,
    TooSteepWarning,
    Warning,
)

__all__ = [
    "PathPoint",
    "Node",
    "Path",
    "Warning",
    "ExcavatorWarning",
    "TooSteepWarning",
    "TooFlatWarning",
    "PathSegment",
    "ProposedPathSegment",
    "Slope",
    "Pylon",
    "Lift",
    "ResortGraph",
    "UndoAction",
]

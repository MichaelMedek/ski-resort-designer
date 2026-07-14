"""Undo action types for the resort graph.

Each user action that mutates the graph pushes a frozen ``*Action`` dataclass onto the undo stack.
Every action carries an ``.action_type`` (an :class:`ActionType` member) used for reload-safe
dispatch — see :mod:`skiresort_planner.model.undo_handlers` for the handler registry that turns an
action back into a graph mutation and a human-readable description.

These are pure data containers over model-layer types (Node/Slope/Road/Lift/PathSegment/
SegmentPath); they import no graph or UI code, so they can live in the model layer with no cycle.
"""

from dataclasses import dataclass
from enum import Enum, auto

from skiresort_planner.model.lift import Lift
from skiresort_planner.model.node import Node
from skiresort_planner.model.path_segment import PathSegment
from skiresort_planner.model.road import Road
from skiresort_planner.model.segment_path import SegmentPath
from skiresort_planner.model.slope import Slope


class ActionType(Enum):
    """Enum identifying the type of (undo) action for reliable dispatch."""

    ADD_SEGMENTS = auto()
    FINISH_SLOPE = auto()
    ADD_LIFT = auto()
    FINISH_ROAD = auto()
    DELETE_SLOPE = auto()
    DELETE_LIFT = auto()
    DELETE_ROAD = auto()
    IMPORT_OSM = auto()
    MERGE_NODES = auto()


@dataclass(frozen=True)
class AddSegmentsAction:
    """Undo action for committed path segments."""

    segment_ids: tuple[str, ...]
    node_ids: tuple[str, ...]

    @property
    def action_type(self) -> ActionType:
        """Return the enum type for dispatch."""
        return ActionType.ADD_SEGMENTS


@dataclass(frozen=True)
class FinishSlopeAction:
    """Undo action for finishing a slope."""

    slope_id: str
    segment_ids: tuple[str, ...]
    slope_name: str
    start_node_id: str | None

    @property
    def action_type(self) -> ActionType:
        """Return the enum type for dispatch."""
        return ActionType.FINISH_SLOPE


@dataclass(frozen=True)
class AddLiftAction:
    """Undo action for creating a lift."""

    lift_id: str

    @property
    def action_type(self) -> ActionType:
        """Return the enum type for dispatch."""
        return ActionType.ADD_LIFT


@dataclass(frozen=True)
class FinishRoadAction:
    """Undo action for finishing a road (mirrors FinishSlopeAction).

    Ungroups the Road but keeps its segments, so undo returns to road building
    with the segments intact; further undos peel each segment (AddSegmentsAction).
    """

    road_id: str
    segment_ids: tuple[str, ...]
    road_name: str
    start_node_id: str | None

    @property
    def action_type(self) -> ActionType:
        """Return the enum type for dispatch."""
        return ActionType.FINISH_ROAD


@dataclass(frozen=True)
class DeleteSlopeAction:
    """Undo action for deleting a slope (stores data for restore)."""

    slope_id: str
    deleted_slope: Slope
    deleted_segments: tuple[PathSegment, ...]
    deleted_nodes: tuple[Node, ...] = ()  # Nodes orphaned by segment removal

    @property
    def action_type(self) -> ActionType:
        """Return the enum type for dispatch."""
        return ActionType.DELETE_SLOPE


@dataclass(frozen=True)
class DeleteLiftAction:
    """Undo action for deleting a lift (stores data for restore)."""

    lift_id: str
    deleted_lift: Lift
    deleted_nodes: tuple[Node, ...] = ()  # Nodes orphaned by lift removal

    @property
    def action_type(self) -> ActionType:
        """Return the enum type for dispatch."""
        return ActionType.DELETE_LIFT


@dataclass(frozen=True)
class DeleteRoadAction:
    """Undo action for deleting a road (stores data for restore)."""

    road_id: str
    deleted_road: Road
    deleted_segments: tuple[PathSegment, ...]
    deleted_nodes: tuple[Node, ...] = ()  # Nodes orphaned by segment removal

    @property
    def action_type(self) -> ActionType:
        """Return the enum type for dispatch."""
        return ActionType.DELETE_ROAD


@dataclass(frozen=True)
class ImportOSMAction:
    """Undo action for a single OSM import batch.

    One import (any number of pistes + lifts) is ONE undoable unit: undo removes every entity
    the import created — its slopes, lifts, segments, and the nodes it newly created — so the
    user can then import a different selection.
    """

    slope_ids: tuple[str, ...]
    lift_ids: tuple[str, ...]
    segment_ids: tuple[str, ...]
    node_ids: tuple[str, ...]  # nodes CREATED by this import (not pre-existing shared ones)

    @property
    def action_type(self) -> ActionType:
        """Return the enum type for dispatch."""
        return ActionType.IMPORT_OSM

    def removed_entity(self, entity_id: str) -> bool:
        """True if the given slope/lift id was one this import created (now removed on undo)."""
        return entity_id in self.slope_ids or entity_id in self.lift_ids


@dataclass(frozen=True)
class MergeNodesAction:
    """Undo action for merging several nodes into one survivor.

    Restoring is exact: put the deleted nodes back, move the survivor to its old location, and restore
    the pre-merge snapshot of every builder the merge touched — segments (repointed + re-stitched
    geometry), lifts (repointed + rebuilt cable), and slopes/roads (repointed boundary ids). Each
    snapshot carries the original endpoint ids, so replacing it in place also undoes the repoint.
    """

    survivor_id: str
    survivor_before: Node  # survivor's location before it moved to the median
    deleted_nodes: tuple[Node, ...]  # the merged-away nodes, to restore verbatim
    # Pre-merge snapshots of every touched builder, restored verbatim on undo.
    segments_before: tuple[PathSegment, ...]
    lifts_before: tuple[Lift, ...]
    paths_before: tuple[SegmentPath, ...]  # slopes + roads whose boundary ids were repointed

    @property
    def action_type(self) -> ActionType:
        """Return the enum type for dispatch."""
        return ActionType.MERGE_NODES


UndoAction = (
    AddSegmentsAction
    | FinishSlopeAction
    | AddLiftAction
    | FinishRoadAction
    | DeleteSlopeAction
    | DeleteLiftAction
    | DeleteRoadAction
    | ImportOSMAction
    | MergeNodesAction
)

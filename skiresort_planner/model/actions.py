"""Undo action types for the resort graph.

Each user action that mutates the graph pushes a frozen ``*Action`` dataclass onto the undo stack.
Every action carries an ``.action_type`` (an :class:`ActionType` member) used for reload-safe
dispatch — see :mod:`skiresort_planner.model.undo_handlers` for the handler registry that turns an
action back into a graph mutation and a human-readable description.

These are pure data containers over model-layer types (Node/Slope/Road/Lift/PathSegment/
SegmentPath); they import no graph or UI code, so they can live in the model layer with no cycle.
"""

from dataclasses import dataclass
from enum import StrEnum
from typing import ClassVar

from skiresort_planner.model.lift import Lift
from skiresort_planner.model.node import Node
from skiresort_planner.model.path_segment import PathSegment
from skiresort_planner.model.road import Road
from skiresort_planner.model.segment_path import SegmentPath
from skiresort_planner.model.slope import Slope


class ActionType(StrEnum):
    """Enum identifying the type of (undo) action for reliable dispatch.

    StrEnum (value == lowercased name) so comparisons/dispatch survive Streamlit module reloads.
    """

    ADD_SEGMENTS = "add_segments"
    FINISH_SLOPE = "finish_slope"
    ADD_LIFT = "add_lift"
    FINISH_ROAD = "finish_road"
    DELETE_SLOPE = "delete_slope"
    DELETE_LIFT = "delete_lift"
    DELETE_ROAD = "delete_road"
    IMPORT_OSM = "import_osm"
    MERGE_NODES = "merge_nodes"
    DELETE_NODES = "delete_nodes"
    INSERT_NODE = "insert_node"


@dataclass(frozen=True)
class AddSegmentsAction:
    """Undo action for committed path segments."""

    action_type: ClassVar[ActionType] = ActionType.ADD_SEGMENTS
    segment_ids: tuple[str, ...]
    node_ids: tuple[str, ...]


@dataclass(frozen=True)
class FinishSlopeAction:
    """Undo action for finishing a slope."""

    action_type: ClassVar[ActionType] = ActionType.FINISH_SLOPE
    slope_id: str
    segment_ids: tuple[str, ...]
    slope_name: str
    start_node_id: str | None


@dataclass(frozen=True)
class AddLiftAction:
    """Undo action for creating a lift."""

    action_type: ClassVar[ActionType] = ActionType.ADD_LIFT
    lift_id: str


@dataclass(frozen=True)
class FinishRoadAction:
    """Undo action for finishing a road (mirrors FinishSlopeAction).

    Ungroups the Road but keeps its segments, so undo returns to road building
    with the segments intact; further undos peel each segment (AddSegmentsAction).
    """

    action_type: ClassVar[ActionType] = ActionType.FINISH_ROAD
    road_id: str
    segment_ids: tuple[str, ...]
    road_name: str
    start_node_id: str | None


@dataclass(frozen=True)
class DeleteSlopeAction:
    """Undo action for deleting a slope (stores data for restore)."""

    action_type: ClassVar[ActionType] = ActionType.DELETE_SLOPE
    slope_id: str
    deleted_slope: Slope
    deleted_segments: tuple[PathSegment, ...]
    deleted_nodes: tuple[Node, ...] = ()  # Nodes orphaned by segment removal


@dataclass(frozen=True)
class DeleteLiftAction:
    """Undo action for deleting a lift (stores data for restore)."""

    action_type: ClassVar[ActionType] = ActionType.DELETE_LIFT
    lift_id: str
    deleted_lift: Lift
    deleted_nodes: tuple[Node, ...] = ()  # Nodes orphaned by lift removal


@dataclass(frozen=True)
class DeleteRoadAction:
    """Undo action for deleting a road (stores data for restore)."""

    action_type: ClassVar[ActionType] = ActionType.DELETE_ROAD
    road_id: str
    deleted_road: Road
    deleted_segments: tuple[PathSegment, ...]
    deleted_nodes: tuple[Node, ...] = ()  # Nodes orphaned by segment removal


@dataclass(frozen=True)
class ImportOSMAction:
    """Undo action for a single OSM import batch.

    One import (any number of pistes + lifts) is ONE undoable unit: undo removes every entity
    the import created — its slopes, lifts, segments, and the nodes it newly created — so the
    user can then import a different selection.
    """

    action_type: ClassVar[ActionType] = ActionType.IMPORT_OSM
    slope_ids: tuple[str, ...]
    lift_ids: tuple[str, ...]
    segment_ids: tuple[str, ...]
    node_ids: tuple[str, ...]  # nodes CREATED by this import (not pre-existing shared ones)

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

    action_type: ClassVar[ActionType] = ActionType.MERGE_NODES
    survivor_id: str
    survivor_before: Node  # survivor's location before it moved to the median
    deleted_nodes: tuple[Node, ...]  # the merged-away nodes, to restore verbatim
    # Pre-merge snapshots of every touched builder, restored verbatim on undo.
    segments_before: tuple[PathSegment, ...]
    lifts_before: tuple[Lift, ...]
    paths_before: tuple[SegmentPath, ...]  # slopes + roads whose boundary ids were repointed


@dataclass(frozen=True)
class DeleteNodesAction:
    """Undo a batch delete of path nodes (interior fusion + clean-endpoint trim).

    Restores each affected path's pre-delete segment chain + boundary ids, every segment of those
    chains verbatim (some were mutated in place, some dropped), and the deleted nodes. The delete
    reuses existing segment ids (fuses into the first segment of a run), so there is nothing extra
    to drop on undo — restoring the snapshots is a complete reversal.
    """

    action_type: ClassVar[ActionType] = ActionType.DELETE_NODES
    deleted_nodes: tuple[Node, ...]
    paths_before: tuple[SegmentPath, ...]  # affected slopes/roads with their original segment chain
    segments_before: tuple[PathSegment, ...]  # every segment of each affected chain, pre-delete


@dataclass(frozen=True)
class InsertNodeAction:
    """Undo a node insert on a path: delete the created node + the two split segments, restore the
    original pre-split segment and the owning path's original segment_ids.
    """

    action_type: ClassVar[ActionType] = ActionType.INSERT_NODE
    created_node_id: str
    created_segment_ids: tuple[str, ...]  # the two segments the split produced (A', B')
    path_before: SegmentPath  # owning path with its original segment chain
    segment_before: PathSegment  # the pre-split segment


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
    | DeleteNodesAction
    | InsertNodeAction
)

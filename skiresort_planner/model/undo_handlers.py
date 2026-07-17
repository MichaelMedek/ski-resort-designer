"""Undo handler registry — one handler per :class:`ActionType`.

Each :class:`UndoHandler` folds together the two model-level things an undo action needs: how to
mutate the graph to reverse it (``apply_undo``) and a human-readable description of what undoing
will do (``describe``). Both are model-safe (graph mutation + pure text — no Streamlit, no UI).

The registry is keyed by ``ActionType.name`` (a plain str), NOT the enum member: the undo stack
lives in ``st.session_state`` and, after a Streamlit module reload, holds actions built against an
OLD ``ActionType`` class. An enum-keyed dict (identity hash) would miss those; ``.name`` is a stable
string across reloads. ActionType is a StrEnum, so ``==`` comparisons are reload-safe too.

An import-time bijection assert guarantees every ActionType has exactly one handler — a new action
type that forgets to register here fails the first time this module is imported.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, cast

from skiresort_planner.model.actions import (
    ActionType,
    AddLiftAction,
    AddSegmentsAction,
    DeleteLiftAction,
    DeleteNodesAction,
    DeleteRoadAction,
    DeleteSlopeAction,
    FinishRoadAction,
    FinishSlopeAction,
    ImportOSMAction,
    InsertNodeAction,
    MergeNodesAction,
    UndoAction,
)
from skiresort_planner.model.path_segment import SegmentKind

if TYPE_CHECKING:
    from skiresort_planner.model.node import Node
    from skiresort_planner.model.path_segment import PathSegment
    from skiresort_planner.model.resort_graph import ResortGraph
    from skiresort_planner.model.segment_path import SegmentPath

import logging

logger = logging.getLogger(__name__)


def _ungroup_finished_entity(
    graph: "ResortGraph", kind: SegmentKind, entity_id: str, segment_ids: tuple[str, ...], entity_name: str
) -> None:
    """Undo a finish: drop the grouped entity, rename its segments back to building names.

    Shared by the FINISH_SLOPE / FINISH_ROAD handlers. Segments survive (their own AddSegmentsAction
    handles per-segment removal on further undo).
    """
    del graph.entity_dict_for_kind(kind)[entity_id]
    for seg_id in segment_ids:
        graph.segments[seg_id].name = f"Segment {seg_id[1:]}"
    logger.info(f"Undid FINISH_{kind.value.upper()}: ungrouped {entity_name} to {len(segment_ids)} building segment(s)")


def _restore_deleted_path_entity(
    graph: "ResortGraph",
    kind: SegmentKind,
    entity_id: str,
    entity: "SegmentPath",
    deleted_nodes: tuple["Node", ...],
    deleted_segments: tuple["PathSegment", ...],
) -> None:
    """Undo a slope/road delete: restore orphaned nodes first, then the entity and its segments."""
    for node in deleted_nodes:
        graph.nodes[node.id] = node
    graph.entity_dict_for_kind(kind)[entity_id] = entity
    for seg in deleted_segments:
        graph.segments[seg.id] = seg
    logger.info(
        f"Restored {kind.value} {entity_id} with {len(deleted_segments)} segments and {len(deleted_nodes)} nodes"
    )


class UndoHandler(ABC):
    """Everything the model must do to reverse one action type: mutate the graph + describe it.

    Concrete subclasses are stateless descriptors keyed by ``action_type``. The abstract methods
    are the complete per-action surface, so a handler cannot be registered with a part missing.
    """

    #: The ActionType this handler reverses.
    action_type: ActionType

    #: Skip the undo confirmation dialog for this action type. Default False (confirm).
    skip_confirm: bool = False

    @abstractmethod
    def apply_undo(self, graph: "ResortGraph", action: UndoAction) -> None:
        """Mutate the graph to undo ``action``."""

    @abstractmethod
    def describe(self, action: UndoAction, graph: "ResortGraph") -> str:
        """Human-readable description of what undoing ``action`` will do."""


class _AddSegmentsHandler(UndoHandler):
    action_type = ActionType.ADD_SEGMENTS
    skip_confirm = True  # peeling a just-committed segment while building is a normal step

    def apply_undo(self, graph: "ResortGraph", action: UndoAction) -> None:
        add_seg = cast(AddSegmentsAction, action)
        for seg_id in add_seg.segment_ids:
            del graph.segments[seg_id]
        graph.cleanup_isolated_nodes()  # Remove orphaned nodes from segment removal
        logger.info(f"Undid ADD_SEGMENTS: removed {len(add_seg.segment_ids)} segment(s)")

    def describe(self, action: UndoAction, graph: "ResortGraph") -> str:
        return ""  # skip_confirm handlers show no dialog, so no description text


class _FinishSlopeHandler(UndoHandler):
    action_type = ActionType.FINISH_SLOPE

    def apply_undo(self, graph: "ResortGraph", action: UndoAction) -> None:
        # Ungroup the slope but keep its segments; their own AddSegmentsAction
        # entries handle per-segment removal on further undo.
        finish = cast(FinishSlopeAction, action)
        _ungroup_finished_entity(graph, SegmentKind.SLOPE, finish.slope_id, finish.segment_ids, finish.slope_name)

    def describe(self, action: UndoAction, graph: "ResortGraph") -> str:
        finish_slope_act = cast(FinishSlopeAction, action)
        return f"Restore slope **{finish_slope_act.slope_name}** to building mode"


class _AddLiftHandler(UndoHandler):
    action_type = ActionType.ADD_LIFT

    def apply_undo(self, graph: "ResortGraph", action: UndoAction) -> None:
        add_lift = cast(AddLiftAction, action)
        del graph.lifts[add_lift.lift_id]
        graph.cleanup_isolated_nodes()  # Remove orphaned station nodes
        logger.info(f"Undid ADD_LIFT: removed lift {add_lift.lift_id}")

    def describe(self, action: UndoAction, graph: "ResortGraph") -> str:
        add_lift_act = cast(AddLiftAction, action)
        return f"Delete lift **{graph.lifts[add_lift_act.lift_id].name}**"


class _FinishRoadHandler(UndoHandler):
    action_type = ActionType.FINISH_ROAD

    def apply_undo(self, graph: "ResortGraph", action: UndoAction) -> None:
        finish_road = cast(FinishRoadAction, action)
        _ungroup_finished_entity(
            graph, SegmentKind.ROAD, finish_road.road_id, finish_road.segment_ids, finish_road.road_name
        )

    def describe(self, action: UndoAction, graph: "ResortGraph") -> str:
        finish_road_act = cast(FinishRoadAction, action)
        return f"Restore road **{finish_road_act.road_name}** to building mode"


class _DeleteSlopeHandler(UndoHandler):
    action_type = ActionType.DELETE_SLOPE

    def apply_undo(self, graph: "ResortGraph", action: UndoAction) -> None:
        del_slope = cast(DeleteSlopeAction, action)
        _restore_deleted_path_entity(
            graph,
            SegmentKind.SLOPE,
            del_slope.slope_id,
            del_slope.deleted_slope,
            del_slope.deleted_nodes,
            del_slope.deleted_segments,
        )

    def describe(self, action: UndoAction, graph: "ResortGraph") -> str:
        delete_slope_act = cast(DeleteSlopeAction, action)
        return f"Restore deleted slope **{delete_slope_act.deleted_slope.name}**"


class _DeleteLiftHandler(UndoHandler):
    action_type = ActionType.DELETE_LIFT

    def apply_undo(self, graph: "ResortGraph", action: UndoAction) -> None:
        del_lift = cast(DeleteLiftAction, action)
        # Restore orphaned nodes first (they're needed by lift)
        for node in del_lift.deleted_nodes:
            graph.nodes[node.id] = node
        # Restore deleted lift
        graph.lifts[del_lift.lift_id] = del_lift.deleted_lift
        logger.info(f"Restored lift {del_lift.lift_id} with {len(del_lift.deleted_nodes)} nodes")

    def describe(self, action: UndoAction, graph: "ResortGraph") -> str:
        delete_lift_act = cast(DeleteLiftAction, action)
        return f"Restore deleted lift **{delete_lift_act.deleted_lift.name}**"


class _DeleteRoadHandler(UndoHandler):
    action_type = ActionType.DELETE_ROAD

    def apply_undo(self, graph: "ResortGraph", action: UndoAction) -> None:
        del_road = cast(DeleteRoadAction, action)
        _restore_deleted_path_entity(
            graph,
            SegmentKind.ROAD,
            del_road.road_id,
            del_road.deleted_road,
            del_road.deleted_nodes,
            del_road.deleted_segments,
        )

    def describe(self, action: UndoAction, graph: "ResortGraph") -> str:
        delete_road_act = cast(DeleteRoadAction, action)
        return f"Restore deleted road **{delete_road_act.deleted_road.name}**"


class _ImportOSMHandler(UndoHandler):
    action_type = ActionType.IMPORT_OSM

    def apply_undo(self, graph: "ResortGraph", action: UndoAction) -> None:
        # One import = one undo: drop every entity the batch created (slopes, lifts,
        # segments) and the nodes it newly made; reused/shared nodes are left untouched.
        imp = cast(ImportOSMAction, action)
        for slope_id in imp.slope_ids:
            del graph.slopes[slope_id]
        for lift_id in imp.lift_ids:
            del graph.lifts[lift_id]
        for seg_id in imp.segment_ids:
            del graph.segments[seg_id]
        for node_id in imp.node_ids:
            del graph.nodes[node_id]
        graph.cleanup_isolated_nodes()
        logger.info(
            f"Reverted OSM import: {len(imp.slope_ids)} slopes, {len(imp.lift_ids)} lifts, "
            f"{len(imp.segment_ids)} segments, {len(imp.node_ids)} nodes"
        )

    def describe(self, action: UndoAction, graph: "ResortGraph") -> str:
        import_act = cast(ImportOSMAction, action)
        return f"Remove OSM import ({len(import_act.slope_ids)} slopes, {len(import_act.lift_ids)} lifts)"


class _MergeNodesHandler(UndoHandler):
    action_type = ActionType.MERGE_NODES

    def apply_undo(self, graph: "ResortGraph", action: UndoAction) -> None:
        # Restore the merged-away nodes, the touched builders (each snapshot carries the original
        # endpoint ids, so replacing it undoes the repoint), and the survivor WHOLESALE — a merge that
        # collapsed the survivor's only path can leave it isolated, so it may be gone from graph.nodes.
        merge = cast(MergeNodesAction, action)
        for node in merge.deleted_nodes:
            graph.nodes[node.id] = node
        graph.nodes[merge.survivor_id] = merge.survivor_before
        for seg_before in merge.segments_before:
            graph.segments[seg_before.id] = seg_before
        for lift_before in merge.lifts_before:
            graph.lifts[lift_before.id] = lift_before
        for path_before in merge.paths_before:
            graph.entity_dict_for_kind(path_before.kind)[path_before.id] = path_before
        logger.info(f"Reverted merge into {merge.survivor_id}: restored {len(merge.deleted_nodes)} nodes")

    def describe(self, action: UndoAction, graph: "ResortGraph") -> str:
        merge = cast(MergeNodesAction, action)
        return f"Un-merge {len(merge.deleted_nodes) + 1} nodes"


class _DeleteNodesHandler(UndoHandler):
    action_type = ActionType.DELETE_NODES

    def apply_undo(self, graph: "ResortGraph", action: UndoAction) -> None:
        # Restore nodes + every segment of the affected chains verbatim (some were mutated in place
        # by the fusion, some dropped), then restore each path's original segment_ids/boundaries
        # (paths reference the segments, so segments come first).
        delete = cast(DeleteNodesAction, action)
        for node in delete.deleted_nodes:
            graph.nodes[node.id] = node
        for seg_before in delete.segments_before:
            graph.segments[seg_before.id] = seg_before
        for path_before in delete.paths_before:
            graph.entity_dict_for_kind(path_before.kind)[path_before.id] = path_before
        logger.info(f"Reverted delete of {len(delete.deleted_nodes)} node(s)")

    def describe(self, action: UndoAction, graph: "ResortGraph") -> str:
        delete = cast(DeleteNodesAction, action)
        return f"Restore {len(delete.deleted_nodes)} deleted node(s)"


class _InsertNodeHandler(UndoHandler):
    action_type = ActionType.INSERT_NODE

    def apply_undo(self, graph: "ResortGraph", action: UndoAction) -> None:
        # Drop the two split segments, restore the original segment + the path's chain, then remove
        # the created node (now unreferenced).
        insert = cast(InsertNodeAction, action)
        for seg_id in insert.created_segment_ids:
            del graph.segments[seg_id]
        graph.segments[insert.segment_before.id] = insert.segment_before
        graph.entity_dict_for_kind(insert.path_before.kind)[insert.path_before.id] = insert.path_before
        del graph.nodes[insert.created_node_id]
        logger.info(f"Reverted insert of node {insert.created_node_id}")

    def describe(self, action: UndoAction, graph: "ResortGraph") -> str:
        return "Remove the inserted node"


_UNDO_HANDLER_LIST: list[UndoHandler] = [
    _AddSegmentsHandler(),
    _FinishSlopeHandler(),
    _AddLiftHandler(),
    _FinishRoadHandler(),
    _DeleteSlopeHandler(),
    _DeleteLiftHandler(),
    _DeleteRoadHandler(),
    _ImportOSMHandler(),
    _MergeNodesHandler(),
    _DeleteNodesHandler(),
    _InsertNodeHandler(),
]

# Keyed by ActionType.name (str) — reload-safe (see module docstring).
UNDO_HANDLERS: dict[str, UndoHandler] = {h.action_type.name: h for h in _UNDO_HANDLER_LIST}

# Import-time bijection guard: every ActionType has exactly one handler, and no stray handlers.
_action_names = {t.name for t in ActionType}
assert set(UNDO_HANDLERS) == _action_names, (
    f"UNDO_HANDLERS keys must match ActionType members exactly. "
    f"Missing: {_action_names - set(UNDO_HANDLERS)}; stray: {set(UNDO_HANDLERS) - _action_names}"
)
assert len(UNDO_HANDLERS) == len(_UNDO_HANDLER_LIST), "duplicate action_type across UndoHandler subclasses"

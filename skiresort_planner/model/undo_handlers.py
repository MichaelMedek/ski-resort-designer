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
    DeleteRoadAction,
    DeleteSlopeAction,
    FinishRoadAction,
    FinishSlopeAction,
    ImportOSMAction,
    MergeNodesAction,
    UndoAction,
)
from skiresort_planner.model.path_segment import SegmentKind

if TYPE_CHECKING:
    from skiresort_planner.model.resort_graph import ResortGraph
    from skiresort_planner.model.road import Road
    from skiresort_planner.model.slope import Slope

import logging

logger = logging.getLogger(__name__)


class UndoHandler(ABC):
    """Everything the model must do to reverse one action type: mutate the graph + describe it.

    Concrete subclasses are stateless descriptors keyed by ``action_type``. The abstract methods
    are the complete per-action surface, so a handler cannot be registered with a part missing.
    """

    #: The ActionType this handler reverses.
    action_type: ActionType

    @abstractmethod
    def apply_undo(self, graph: "ResortGraph", action: UndoAction) -> None:
        """Mutate the graph to undo ``action``."""

    @abstractmethod
    def describe(self, action: UndoAction, graph: "ResortGraph") -> str:
        """Human-readable description of what undoing ``action`` will do."""


class _AddSegmentsHandler(UndoHandler):
    action_type = ActionType.ADD_SEGMENTS

    def apply_undo(self, graph: "ResortGraph", action: UndoAction) -> None:
        add_seg = cast(AddSegmentsAction, action)
        for seg_id in add_seg.segment_ids:
            graph.segments.pop(seg_id, None)
        graph.cleanup_isolated_nodes()  # Remove orphaned nodes from segment removal
        logger.info(f"Undid ADD_SEGMENTS: removed {len(add_seg.segment_ids)} segment(s)")

    def describe(self, action: UndoAction, graph: "ResortGraph") -> str:
        segments_act = cast(AddSegmentsAction, action)
        n_segments = len(segments_act.segment_ids)
        # segment_ids is never empty (commit_paths pushes this only for ≥1 new segment)
        first_seg = graph.segments[segments_act.segment_ids[0]]
        # SegmentKind is a str-Enum, so .value ("slope"/"road") is reload-safe.
        return f"Remove {n_segments} segment(s) from current {first_seg.kind.value}"


class _FinishSlopeHandler(UndoHandler):
    action_type = ActionType.FINISH_SLOPE

    def apply_undo(self, graph: "ResortGraph", action: UndoAction) -> None:
        # Ungroup the slope but keep its segments; their own AddSegmentsAction
        # entries handle per-segment removal on further undo.
        finish = cast(FinishSlopeAction, action)
        graph.slopes.pop(finish.slope_id, None)
        for seg_id in finish.segment_ids:
            graph.segments[seg_id].name = f"Segment {seg_id[1:]}"
        logger.info(
            f"Undid FINISH_SLOPE: ungrouped {finish.slope_name} back to {len(finish.segment_ids)} building segment(s)"
        )

    def describe(self, action: UndoAction, graph: "ResortGraph") -> str:
        finish_slope_act = cast(FinishSlopeAction, action)
        return f"Restore slope **{finish_slope_act.slope_name}** to building mode"


class _AddLiftHandler(UndoHandler):
    action_type = ActionType.ADD_LIFT

    def apply_undo(self, graph: "ResortGraph", action: UndoAction) -> None:
        add_lift = cast(AddLiftAction, action)
        graph.lifts.pop(add_lift.lift_id, None)
        graph.cleanup_isolated_nodes()  # Remove orphaned station nodes
        logger.info(f"Undid ADD_LIFT: removed lift {add_lift.lift_id}")

    def describe(self, action: UndoAction, graph: "ResortGraph") -> str:
        add_lift_act = cast(AddLiftAction, action)
        lift = graph.lifts.get(add_lift_act.lift_id)
        name = lift.name if lift else add_lift_act.lift_id
        return f"Delete lift **{name}**"


class _FinishRoadHandler(UndoHandler):
    action_type = ActionType.FINISH_ROAD

    def apply_undo(self, graph: "ResortGraph", action: UndoAction) -> None:
        # Mirror FINISH_SLOPE: ungroup the road, keep its segments.
        finish_road = cast(FinishRoadAction, action)
        graph.roads.pop(finish_road.road_id, None)
        for seg_id in finish_road.segment_ids:
            graph.segments[seg_id].name = f"Segment {seg_id[1:]}"
        logger.info(
            f"Undid FINISH_ROAD: ungrouped {finish_road.road_name} back to {len(finish_road.segment_ids)} building segment(s)"
        )

    def describe(self, action: UndoAction, graph: "ResortGraph") -> str:
        finish_road_act = cast(FinishRoadAction, action)
        return f"Restore road **{finish_road_act.road_name}** to building mode"


class _DeleteSlopeHandler(UndoHandler):
    action_type = ActionType.DELETE_SLOPE

    def apply_undo(self, graph: "ResortGraph", action: UndoAction) -> None:
        del_slope = cast(DeleteSlopeAction, action)
        # Restore orphaned nodes first (they're needed by segments)
        for node in del_slope.deleted_nodes:
            graph.nodes[node.id] = node
        # Restore deleted slope and its segments
        graph.slopes[del_slope.slope_id] = del_slope.deleted_slope
        for seg in del_slope.deleted_segments:
            graph.segments[seg.id] = seg
        logger.info(
            f"Restored slope {del_slope.slope_id} with {len(del_slope.deleted_segments)} segments "
            f"and {len(del_slope.deleted_nodes)} nodes"
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
        # Restore orphaned nodes first (they're needed by segments)
        for node in del_road.deleted_nodes:
            graph.nodes[node.id] = node
        graph.roads[del_road.road_id] = del_road.deleted_road
        for seg in del_road.deleted_segments:
            graph.segments[seg.id] = seg
        logger.info(
            f"Restored road {del_road.road_id} with {len(del_road.deleted_segments)} segments "
            f"and {len(del_road.deleted_nodes)} nodes"
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
            graph.slopes.pop(slope_id, None)
        for lift_id in imp.lift_ids:
            graph.lifts.pop(lift_id, None)
        for seg_id in imp.segment_ids:
            graph.segments.pop(seg_id, None)
        for node_id in imp.node_ids:
            graph.nodes.pop(node_id, None)
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
        # Restore the merged-away nodes, move the survivor back, and restore every touched builder
        # from its pre-merge snapshot. Each snapshot carries the original endpoint/boundary ids, so
        # replacing it in place also undoes the repoint.
        merge = cast(MergeNodesAction, action)
        for node in merge.deleted_nodes:
            graph.nodes[node.id] = node
        graph.nodes[merge.survivor_id].location = merge.survivor_before.location
        for seg_before in merge.segments_before:
            graph.segments[seg_before.id] = seg_before
        for lift_before in merge.lifts_before:
            graph.lifts[lift_before.id] = lift_before
        for path_before in merge.paths_before:
            if path_before.kind == SegmentKind.SLOPE:
                graph.slopes[path_before.id] = cast("Slope", path_before)
            elif path_before.kind == SegmentKind.ROAD:
                graph.roads[path_before.id] = cast("Road", path_before)
            else:
                raise RuntimeError(f"merge undo: unexpected path kind {path_before.kind}")
        logger.info(f"Reverted merge into {merge.survivor_id}: restored {len(merge.deleted_nodes)} nodes")

    def describe(self, action: UndoAction, graph: "ResortGraph") -> str:
        merge = cast(MergeNodesAction, action)
        return f"Un-merge {len(merge.deleted_nodes) + 1} nodes"


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

"""Node-editing vocabulary for the merge-mode Delete / add-node-on-path tools.

Classifies WHY a node can or can't be edited and builds the exact toast sentences the UI renders
(UnableToDeleteMessage etc.). Kept out of resort_graph.py (graph bookkeeping) and out of message.py
(so message.py stays free of graph-tool enums). Imports nothing but StrEnum → cycle-free.
"""

from enum import StrEnum


class NodeDeletability(StrEnum):
    """Why a node can or can't be deleted by the merge-mode Delete tool (reload-safe StrEnum)."""

    DELETABLE_INTERIOR = "deletable_interior"  # degree-2 head-to-tail (one chain OR two paths) -> fuse
    DELETABLE_END = "deletable_end"  # clean endpoint of a >1-segment path -> trim its boundary segment
    IS_BRANCH = "is_branch"  # 3+ segments of its kind meet here -> delete a path first
    IS_CONFLUENCE = "is_confluence"  # degree-2 but a peak/valley (both paths run the same way) -> reject
    IS_PARKING = "is_parking"  # a road meets a slope here -> immutable, like a lift station
    IS_LIFT_STATION = "is_lift_station"  # a lift station -> delete the lift first
    LAST_SEGMENT = "last_segment"  # sole segment of its path -> delete the whole path instead
    NOT_INTERIOR = "not_interior"  # none of the deletable shapes (isolated / unfinished-only)


# Human sentence per non-deletable reason (one place, so the toast text can't drift from the enum).
_DELETABILITY_REASONS: dict[NodeDeletability, str] = {
    NodeDeletability.IS_BRANCH: "it is a branch of 3+ segments — delete a path first",
    NodeDeletability.IS_CONFLUENCE: "the two paths meet here going the same way (a peak or valley) — delete a path first",
    NodeDeletability.IS_PARKING: "it is a parking place (road meets slope) — delete a path first",
    NodeDeletability.IS_LIFT_STATION: "it is a lift station — delete the lift first",
    NodeDeletability.LAST_SEGMENT: "it is the only segment of its path — delete the path instead",
    NodeDeletability.NOT_INTERIOR: "it is not an interior or end node of a single path",
}


def deletability_reason(node_id: str, reason: NodeDeletability) -> str:
    """Human sentence for why a node can't be deleted (used by the UnableToDeleteMessage toast)."""
    return f"{node_id} {_DELETABILITY_REASONS[reason]}"


# Add-node-on-path rejection sentences (one place, mirroring _DELETABILITY_REASONS).
INSERT_REJECT_NOT_FINISHED = "click a finished path to add a node"
INSERT_REJECT_TOO_CLOSE = "too close to an existing node (within {gap:.0f}m)"

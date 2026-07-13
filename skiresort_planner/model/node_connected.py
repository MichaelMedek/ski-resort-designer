"""NodeConnected - shared endpoint interface for entities anchored between two graph nodes.

A Slope, a Road, and a Lift all connect a start node to an end node and answer the same three
questions: what is your `id`, your `start_node_id`, your `end_node_id`, and (from those) your two
endpoint locations. All three store the two boundary node ids as plain string fields (like every
other reference in the model — a segment stores its node ids, a slope stores its segment ids); this
base unifies the shared surface with one `endpoints(nodes)`.

The contract — every concrete subclass exposes `id` / `start_node_id` / `end_node_id` — is verified
by a completeness guard test (test_completeness_guards).
"""

from abc import ABC
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from skiresort_planner.model.node import Node
    from skiresort_planner.model.path_point import PathPoint


class NodeConnected(ABC):
    """An entity anchored between two graph nodes (Slope, Road, Lift).

    Subclasses store `id` / `start_node_id` / `end_node_id` as plain dataclass fields; the shared
    `endpoints()` resolves those ids against a passed-in `nodes` table.
    """

    # The endpoint contract every subclass provides as dataclass fields. Declared for mypy + readers.
    id: str
    start_node_id: str
    end_node_id: str

    def endpoints(self, nodes: dict[str, "Node"]) -> tuple["PathPoint", "PathPoint"]:
        """The start and end node locations, for geometric duplicate matching (see endpoints_match)."""
        start = nodes.get(self.start_node_id)
        end = nodes.get(self.end_node_id)
        if not start or not end:
            raise ValueError(f"Start or end node not found for {type(self).__name__} {self.id}")
        return start.location, end.location

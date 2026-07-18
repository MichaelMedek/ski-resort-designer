"""Graph-connectivity primitive shared by OSM import and the core-resort computation.

Pure model layer — no streamlit/generators/ui imports. Backed by scipy's csgraph (already a
project dependency, used the same way in generators/connection_planners.py), so there is ONE
component-labeller for the whole codebase rather than the hand-rolled BFS blocks that used to
live in the OSM builder.
"""

from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from typing import TypeVar

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

NodeT = TypeVar("NodeT")  # int for OSM hub indices, str for ResortGraph node ids


def component_labels(
    nodes: Iterable[NodeT],
    edges: Iterable[tuple[NodeT, NodeT]],
    *,
    strong: bool,
) -> dict[NodeT, int]:
    """Component id per node via scipy.sparse.csgraph.connected_components.

    Args:
        nodes: Every node to label. Isolated nodes (no incident edge) each get their own
            component — callers relying on that must pass the full node set, not just edge endpoints.
        edges: Directed (u, v) pairs. Treated as undirected when strong=False.
        strong: False → weak/undirected components; True → strongly-connected components.

    Returns:
        node → opaque component id (only equality/grouping is meaningful, not the id value).
    """
    index: dict[NodeT, int] = {}
    order: list[NodeT] = []
    for node in nodes:
        if node not in index:
            index[node] = len(order)
            order.append(node)

    rows: list[int] = []
    cols: list[int] = []
    for u, v in edges:
        # Edge endpoints are always graph nodes; missing means the caller built an inconsistent
        # node set — let the KeyError surface (fail fast) rather than silently dropping the edge.
        rows.append(index[u])
        cols.append(index[v])

    n = len(order)
    graph = csr_matrix((np.ones(len(rows), dtype=np.int8), (rows, cols)), shape=(n, n))
    connection = "strong" if strong else "weak"
    n_components, labels = connected_components(graph, directed=strong, connection=connection)
    # scipy returns exactly one label per node; the node→label mapping below depends on it.
    assert len(labels) == n, f"scipy returned {len(labels)} labels for {n} nodes"
    assert n_components <= n, f"more components ({n_components}) than nodes ({n})"
    return {node: int(labels[i]) for node, i in index.items()}


@dataclass(frozen=True)
class CoreResort:
    """The core skiable area — the largest strongly-connected component of the ski graph.

    Derived, never stored: recomputed from the current slopes/lifts each render.
    """

    node_ids: frozenset[str]
    longest_lift_name: str  # longest in-core lift (by Lift.get_length_m) — named in the warning


class CoreMembership(StrEnum):
    """Where a slope/lift sits relative to the core resort. StrEnum → reload-safe `==`."""

    IN_CORE = "in_core"
    DISCONNECTED = "disconnected"
    NO_CORE_YET = "no_core_yet"  # no core exists yet → never warn


def both_in(node_ids: set[str] | frozenset[str], a: str, b: str) -> bool:
    """Both endpoints of an entity lie inside `node_ids` — the "fully in the core" test."""
    return a in node_ids and b in node_ids

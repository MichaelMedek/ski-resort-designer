"""Route planner — the best A→B ski routes by five criteria.

Pure model layer (stdlib + numpy + scipy only, like model.connectivity): no streamlit/ui imports, so
the core is unit-testable without a browser. We compute FIVE single-objective optimal routes (one per
criterion) rather than enumerating all paths: single-objective shortest/best paths are polynomial,
whereas one path optimal across several objectives at once is NP-hard (a Pareto front). Criteria 1–3
are additive shortest paths (scipy Dijkstra on a re-weighted CSR); criterion 4 is a minimax
(bottleneck) path and 5 a max-min over node elevation — small best-first searches over the same graph.
"""

import heapq
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from skiresort_planner.constants import SlopeConfig
from skiresort_planner.model.path_segment import PathSegment
from skiresort_planner.model.resort_graph import ResortGraph, SkiEdge

_NO_PREDECESSOR = -9999  # scipy dijkstra's "unreachable" sentinel in the predecessor array

# A route-finding cost is a totally-ordered tuple (lower is better); its first field is the metric,
# the last is edge-count (tie-break toward fewer edges). Folder: (cost, from_idx, to_idx, edge) -> cost.
_Cost = tuple[float, int]
_Fold = Callable[[_Cost, int, int, tuple[str, str]], _Cost]


class RouteCriterion(StrEnum):
    """The metric a route is optimal under (reload-safe StrEnum)."""

    FEWEST_LIFTS = "fewest_lifts"
    LEAST_DISTANCE = "least_distance"  # least total slope distance
    LEAST_DROP = "least_drop"  # least total vertical descent skied
    EASIEST = "easiest"  # minimises the hardest slope difficulty on the route
    MOST_SCENIC = "most_scenic"  # reaches the highest peak point en route


@dataclass(frozen=True)
class RouteStep:
    """One entity on a route: the slope or lift traversed, with what the UI shows for it."""

    is_lift: bool
    entity_id: str
    name: str
    detail: str  # lift type for a lift; difficulty for a slope


@dataclass(frozen=True)
class Route:
    """A computed A→B route: the entities to traverse + per-route totals + which criteria it wins.

    node_path is the ordered node ids (for map drawing). criteria lists every RouteCriterion this
    exact route is optimal under (a route can win several — routes are deduped by node_path).
    """

    node_path: tuple[str, ...]
    steps: tuple[RouteStep, ...]
    total_slope_length_m: float
    total_slope_drop_m: float
    lift_count: int
    max_difficulty: str  # hardest slope band on the route ("green" if lift-only)
    highest_elev_m: float
    criteria: tuple[RouteCriterion, ...]


def filter_routes(routes: list[Route], *, max_difficulty: str | None, allowed_lift_types: set[str]) -> list[Route]:
    """The routes passing the UI filters: hardest slope ≤ max_difficulty (None = no cap) AND every
    lift on the route is of an allowed type. Pure — shared by the compute step and the sidebar.
    """
    cap = (
        SlopeConfig.DIFFICULTIES.index(max_difficulty) if max_difficulty is not None else len(SlopeConfig.DIFFICULTIES)
    )

    def ok(route: Route) -> bool:
        if SlopeConfig.DIFFICULTIES.index(route.max_difficulty) > cap:
            return False
        return all(step.detail in allowed_lift_types for step in route.steps if step.is_lift)

    return [r for r in routes if ok(r)]


class RoutePlanner:
    """Computes the best A→B ski routes over a ResortGraph's directed skiable graph.

    Built once per graph — indexes nodes and the edge→owner map from graph.ski_digraph(); each
    best_routes() call runs the five criteria and dedupes coincident winners.
    """

    def __init__(self, graph: ResortGraph) -> None:
        self.graph = graph
        edges, self._owner = graph.ski_digraph()
        self._edges = edges
        # Contiguous int indices for scipy; forward map + reverse list.
        self._index: dict[str, int] = {}
        self._nodes: list[str] = []
        for a, b in edges:
            for nid in (a, b):
                if nid not in self._index:
                    self._index[nid] = len(self._nodes)
                    self._nodes.append(nid)
        # node index -> [(neighbour index, edge)] for the best-first (non-additive) criteria.
        self._adj: dict[int, list[tuple[int, tuple[str, str]]]] = {i: [] for i in range(len(self._nodes))}
        for edge in edges:
            self._adj[self._index[edge[0]]].append((self._index[edge[1]], edge))

    def best_routes(self, start_node_id: str, end_node_id: str) -> list[Route]:
        """The optimal A→B routes, one per criterion, deduped (each kept route wins ≥1 criterion).

        Empty when the two nodes aren't both in the ski graph or B is unreachable from A.
        """
        if start_node_id not in self._index or end_node_id not in self._index:
            return []
        src, dst = self._index[start_node_id], self._index[end_node_id]

        by_path: dict[tuple[str, ...], list[RouteCriterion]] = {}
        for criterion in RouteCriterion:
            path = self._best_path(criterion, src, dst)
            # A single-node path (start == end reached with no edge) isn't a route — you must move.
            if path is not None and len(path) >= 2:
                by_path.setdefault(tuple(path), []).append(criterion)

        return [self._build_route(path, criteria) for path, criteria in by_path.items()]

    # --- per-criterion path finding -------------------------------------------------

    def _best_path(self, criterion: RouteCriterion, src: int, dst: int) -> list[str] | None:
        """Node-id path optimal under `criterion`, or None if B is unreachable from A."""
        if criterion in (RouteCriterion.FEWEST_LIFTS, RouteCriterion.LEAST_DISTANCE, RouteCriterion.LEAST_DROP):
            return self._dijkstra_path(criterion, src, dst)
        if criterion == RouteCriterion.EASIEST:
            return self._minimax_path(src, dst)
        return self._scenic_path(src, dst)  # MOST_SCENIC

    def _segment_of(self, owner: SkiEdge) -> PathSegment:
        """The PathSegment behind a slope edge. Asserts it's a slope edge (segment_id set) — a lift
        edge here is a caller bug, so fail loud rather than silently mis-weight it.
        """
        assert not owner.is_lift and owner.segment_id is not None, "expected a slope edge with a segment"
        return self.graph.segments[owner.segment_id]

    def _edge_weight(self, criterion: RouteCriterion, owner: SkiEdge) -> float:
        """Additive edge cost for the Dijkstra criteria. A tiny epsilon keeps every weight strictly
        positive (Dijkstra needs non-negative) and makes the search prefer fewer edges on ties.
        """
        eps = 1e-6
        if criterion == RouteCriterion.FEWEST_LIFTS:
            return 1.0 if owner.is_lift else eps
        if owner.is_lift:
            return eps
        seg = self._segment_of(owner)
        if criterion == RouteCriterion.LEAST_DISTANCE:
            return seg.length_m + eps
        return abs(seg.total_drop_m) + eps  # LEAST_DROP

    def _dijkstra_path(self, criterion: RouteCriterion, src: int, dst: int) -> list[str] | None:
        """Additive shortest path via scipy Dijkstra on a CSR weighted for `criterion`."""
        n = len(self._nodes)
        rows = [self._index[a] for a, _ in self._edges]
        cols = [self._index[b] for _, b in self._edges]
        data = np.array([self._edge_weight(criterion, self._owner[e]) for e in self._edges], dtype=np.float64)
        graph = csr_matrix((data, (rows, cols)), shape=(n, n))
        _dist, pred = dijkstra(graph, directed=True, indices=src, return_predecessors=True)
        if dst != src and pred[dst] == _NO_PREDECESSOR:
            return None
        return self._walk_predecessors({i: int(pred[i]) for i in range(n)}, src, dst)

    def _minimax_path(self, src: int, dst: int) -> list[str] | None:
        """Easiest route: minimise the hardest slope-difficulty band crossed (a bottleneck path),
        tie-broken by fewer edges. Cost per node = (max band so far, edge count); a lift edge is band 0.
        """

        def band(owner: SkiEdge) -> int:
            return 0 if owner.is_lift else SlopeConfig.DIFFICULTIES.index(self._segment_of(owner).difficulty)

        def fold(cost: _Cost, _u: int, _v: int, edge: tuple[str, str]) -> _Cost:
            max_band, edges = cost
            return (max(max_band, band(self._owner[edge])), edges + 1)

        return self._best_first(src, dst, start_cost=(0.0, 0), fold=fold)

    def _scenic_path(self, src: int, dst: int) -> list[str] | None:
        """Most scenic: reach the highest node elevation en route. Cost per node =
        (-highest elevation reached so far, edge count) — negated so min-cost is the highest peak.
        """
        elev = {i: self.graph.nodes[nid].elevation for i, nid in enumerate(self._nodes)}

        def fold(cost: _Cost, _u: int, v: int, _edge: tuple[str, str]) -> _Cost:
            neg_peak, edges = cost
            return (min(neg_peak, -elev[v]), edges + 1)

        return self._best_first(src, dst, start_cost=(-elev[src], 0), fold=fold)

    def _best_first(
        self,
        src: int,
        dst: int,
        *,
        start_cost: _Cost,
        fold: _Fold,
    ) -> list[str] | None:
        """Dijkstra-shaped best-first search over a totally-ordered cost tuple (min is best).

        `fold(cost, u, v, edge)` returns v's candidate cost when reached from u; the search keeps the
        minimum cost per node. Works for any cost whose optimal substructure is preserved by min.
        """
        best: dict[int, _Cost] = {src: start_cost}
        pred: dict[int, int] = {}
        heap: list[tuple[_Cost, int]] = [(start_cost, src)]
        while heap:
            cost, u = heapq.heappop(heap)
            if cost > best.get(u, cost):
                continue  # stale entry superseded by a better cost
            if u == dst:
                break
            for v, edge in self._adj[u]:
                cand = fold(cost, u, v, edge)
                if v not in best or cand < best[v]:
                    best[v] = cand
                    pred[v] = u
                    heapq.heappush(heap, (cand, v))
        if dst not in best:
            return None
        return self._walk_predecessors(pred, src, dst)

    def _walk_predecessors(self, pred: dict[int, int], src: int, dst: int) -> list[str]:
        """Reconstruct the node-id path from a {node: predecessor} map (dst back to src)."""
        idx_path = [dst]
        while idx_path[-1] != src:
            idx_path.append(pred[idx_path[-1]])
        idx_path.reverse()
        return [self._nodes[i] for i in idx_path]

    # --- route assembly -------------------------------------------------------------

    def _build_route(self, node_path: tuple[str, ...], criteria: list[RouteCriterion]) -> Route:
        """Turn a node path into a Route: map each edge to its owning entity, then aggregate stats."""
        steps: list[RouteStep] = []
        slope_len = slope_drop = 0.0
        lift_count = 0
        hardest_idx = -1
        highest = max(self.graph.nodes[nid].elevation for nid in node_path)
        for a, b in zip(node_path, node_path[1:], strict=False):  # consecutive node pairs
            owner = self._owner[(a, b)]
            if owner.is_lift:
                lift = self.graph.lifts[owner.entity_id]
                lift_count += 1
                steps.append(RouteStep(is_lift=True, entity_id=lift.id, name=lift.name, detail=lift.lift_type))
            else:
                seg = self._segment_of(owner)
                slope_len += seg.length_m
                slope_drop += abs(seg.total_drop_m)
                hardest_idx = max(hardest_idx, SlopeConfig.DIFFICULTIES.index(seg.difficulty))
                slope = self.graph.slopes[owner.entity_id]
                # Collapse consecutive segments of the same slope into a single step (reads once).
                if not (steps and not steps[-1].is_lift and steps[-1].entity_id == slope.id):
                    steps.append(RouteStep(is_lift=False, entity_id=slope.id, name=slope.name, detail=seg.difficulty))
        max_difficulty = SlopeConfig.DIFFICULTIES[hardest_idx] if hardest_idx >= 0 else SlopeConfig.DIFFICULTIES[0]
        return Route(
            node_path=node_path,
            steps=tuple(steps),
            total_slope_length_m=slope_len,
            total_slope_drop_m=slope_drop,
            lift_count=lift_count,
            max_difficulty=max_difficulty,
            highest_elev_m=highest,
            criteria=tuple(criteria),
        )

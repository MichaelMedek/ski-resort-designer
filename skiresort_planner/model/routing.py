"""Route planner — the best A→B ski routes, precomputed per difficulty cap.

Pure model layer (stdlib + numpy + scipy only, like model.connectivity): no streamlit/ui imports, so
the core is unit-testable without a browser. Difficulty is an HONEST computation input, not a
post-filter: for each cap (green→black) we prune every slope harder than the cap OUT of the graph and
recompute, so "shortest path with max red" really is the shortest path over green/blue/red slopes —
never a harder route hidden by a filter. Under each cap we compute TWO single-objective optima
(polynomial shortest paths; multi-objective-in-one-path is NP-hard):

  - FEWEST_LIFTS   — fewest lift rides.
  - SHORTEST_SLOPE — least slope distance, with a light drop weight breaking near-ties.

Both are additive scipy-Dijkstra shortest paths on a re-weighted CSR, so all are stable and cycle-safe
even on bidirectional-lift loops.
"""

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from skiresort_planner.constants import RoutePlannerConfig, SlopeConfig
from skiresort_planner.model.path_segment import PathSegment
from skiresort_planner.model.resort_graph import ResortGraph, SkiEdge

_NO_PREDECESSOR = -9999  # scipy dijkstra's "unreachable" sentinel in the predecessor array


class RouteCriterion(StrEnum):
    """The metric a route is optimal under (reload-safe StrEnum)."""

    FEWEST_LIFTS = "fewest_lifts"
    SHORTEST_SLOPE = "shortest_slope"  # least slope distance (+ light drop weight)


# The overlay palette must cover every criterion (routes are coloured by criterion index).
assert len(RoutePlannerConfig.ROUTE_COLORS) == len(RouteCriterion), "ROUTE_COLORS must cover every RouteCriterion"


@dataclass(frozen=True)
class RouteStep:
    """One entity on a route: the slope or lift traversed, with what the UI shows for it."""

    is_lift: bool
    entity_id: str
    name: str
    detail: str  # lift type for a lift; difficulty for a slope


@dataclass(frozen=True)
class Route:
    """A computed A→B route: the entities to traverse + per-route totals + its computation premise.

    node_path is the ordered node ids; path_points is the drawable polyline (lon, lat, elevation)
    following the actual slope geometry (straight across lifts). difficulty_cap is the hardest band
    ALLOWED when this route was computed (the premise, e.g. "red"); criteria lists every RouteCriterion
    this exact route wins under that cap (routes are deduped by node_path within a cap).
    """

    node_path: tuple[str, ...]
    path_points: tuple[tuple[float, float, float], ...]  # (lon, lat, elevation) along the pistes
    steps: tuple[RouteStep, ...]
    total_slope_length_m: float
    total_slope_drop_m: float
    lift_count: int
    max_difficulty: str  # hardest slope band actually on the route ("green" if lift-only)
    difficulty_cap: str  # the premise: hardest band allowed when computing this route
    criteria: tuple[RouteCriterion, ...]


def routes_for_cap(routes: list[Route], *, max_difficulty: str) -> list[Route]:
    """The precomputed routes whose computation premise (difficulty_cap) is exactly `max_difficulty`.

    Pure — shared by the compute step and the UI selector. This is an honest SELECT over the
    per-cap-precomputed set, NOT a post-filter that could hide a reachable harder route.
    """
    return [r for r in routes if r.difficulty_cap == max_difficulty]


class RoutePlanner:
    """Computes the best A→B ski routes over a ResortGraph's directed skiable graph.

    Built once per graph — indexes nodes and the edge→owner map from graph.ski_digraph(); each
    best_routes() call computes, for every difficulty cap × criterion, one shortest path, deduped.
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

    def best_routes(self, start_node_id: str, end_node_id: str) -> list[Route]:
        """The optimal A→B routes for every difficulty cap × criterion, deduped within each cap.

        For each cap (green→black), slopes harder than the cap are pruned from the graph before the
        shortest-path search, so the result honestly answers "the best route using only slopes up to
        this band". Empty when the two nodes aren't both in the ski graph.
        """
        if start_node_id not in self._index or end_node_id not in self._index:
            return []
        src, dst = self._index[start_node_id], self._index[end_node_id]

        out: list[Route] = []
        for cap in SlopeConfig.DIFFICULTIES:
            by_path: dict[tuple[str, ...], list[RouteCriterion]] = {}
            for criterion in RouteCriterion:
                path = self._shortest_path(criterion, src, dst, cap=cap)
                # A single-node path (start == end reached with no edge) isn't a route — you must move.
                if path is not None and len(path) >= 2:
                    by_path.setdefault(tuple(path), []).append(criterion)
            out.extend(self._build_route(path, criteria, cap=cap) for path, criteria in by_path.items())
        return out

    # --- per-criterion path finding -------------------------------------------------

    def _segment_of(self, owner: SkiEdge) -> PathSegment:
        """The PathSegment behind a slope edge. Asserts it's a slope edge (segment_id set) — a lift
        edge here is a caller bug, so fail loud rather than silently mis-weight it.
        """
        assert not owner.is_lift and owner.segment_id is not None, "expected a slope edge with a segment"
        return self.graph.segments[owner.segment_id]

    def _edge_weight(self, criterion: RouteCriterion, owner: SkiEdge) -> float:
        """Additive edge cost for `criterion`. A tiny epsilon keeps every weight strictly positive
        (Dijkstra needs non-negative) and makes the search prefer fewer edges on ties.
        """
        eps = 1e-6
        if criterion == RouteCriterion.FEWEST_LIFTS:
            return 1.0 if owner.is_lift else eps
        if owner.is_lift:
            return eps
        seg = self._segment_of(owner)  # SHORTEST_SLOPE: distance + a light drop weight for near-ties
        return seg.length_m + RoutePlannerConfig.SHORTEST_SLOPE_DROP_WEIGHT * abs(seg.total_drop_m) + eps

    def _within_cap(self, owner: SkiEdge, cap_idx: int) -> bool:
        """Whether an edge is allowed under a difficulty cap: lifts always; slopes only up to the cap."""
        return owner.is_lift or SlopeConfig.DIFFICULTIES.index(self._segment_of(owner).difficulty) <= cap_idx

    def _shortest_path(self, criterion: RouteCriterion, src: int, dst: int, *, cap: str) -> list[str] | None:
        """Shortest path via scipy Dijkstra on a CSR weighted for `criterion`, with slopes harder than
        `cap` pruned OUT of the graph (so the route is honestly restricted to that band).
        """
        cap_idx = SlopeConfig.DIFFICULTIES.index(cap)
        allowed = [e for e in self._edges if self._within_cap(self._owner[e], cap_idx)]
        n = len(self._nodes)
        rows = [self._index[a] for a, _ in allowed]
        cols = [self._index[b] for _, b in allowed]
        data = np.array([self._edge_weight(criterion, self._owner[e]) for e in allowed], dtype=np.float64)
        graph = csr_matrix((data, (rows, cols)), shape=(n, n))
        _dist, pred = dijkstra(graph, directed=True, indices=src, return_predecessors=True)
        if dst != src and pred[dst] == _NO_PREDECESSOR:
            return None
        return self._walk_predecessors({i: int(pred[i]) for i in range(n)}, src, dst)

    def _walk_predecessors(self, pred: dict[int, int], src: int, dst: int) -> list[str]:
        """Reconstruct the node-id path from a {node: predecessor} map (dst back to src)."""
        idx_path = [dst]
        while idx_path[-1] != src:
            idx_path.append(pred[idx_path[-1]])
        idx_path.reverse()
        return [self._nodes[i] for i in idx_path]

    # --- route assembly -------------------------------------------------------------

    def _build_route(self, node_path: tuple[str, ...], criteria: list[RouteCriterion], *, cap: str) -> Route:
        """Turn a node path into a Route: map each edge to its owning entity, aggregate stats, and
        trace the drawable polyline (slope segment geometry oriented per traversal; straight for lifts).
        """
        steps: list[RouteStep] = []
        slope_len = slope_drop = 0.0
        lift_count = 0
        hardest_idx = -1
        points: list[tuple[float, float, float]] = []

        def add_point(lon: float, lat: float, elev: float) -> None:
            # Dedupe the shared junction point between consecutive edges.
            p = (lon, lat, elev)
            if not points or points[-1] != p:
                points.append(p)

        for a, b in zip(node_path, node_path[1:], strict=False):  # consecutive node pairs
            owner = self._owner[(a, b)]
            if owner.is_lift:
                lift = self.graph.lifts[owner.entity_id]
                lift_count += 1
                steps.append(RouteStep(is_lift=True, entity_id=lift.id, name=lift.name, detail=lift.lift_type))
                # A lift is a straight line between its two stations.
                for nid in (a, b):
                    node = self.graph.nodes[nid]
                    add_point(node.lon, node.lat, node.elevation)
            else:
                seg = self._segment_of(owner)
                slope_len += seg.length_m
                slope_drop += abs(seg.total_drop_m)
                hardest_idx = max(hardest_idx, SlopeConfig.DIFFICULTIES.index(seg.difficulty))
                slope = self.graph.slopes[owner.entity_id]
                # Collapse consecutive segments of the same slope into a single step (reads once).
                if not (steps and not steps[-1].is_lift and steps[-1].entity_id == slope.id):
                    steps.append(RouteStep(is_lift=False, entity_id=slope.id, name=slope.name, detail=seg.difficulty))
                # Trace the segment's real geometry, oriented so it runs a→b (points are stored
                # start_node→end_node; reverse when the route skis it the other way).
                seg_points = seg.points if seg.start_node_id == a else list(reversed(seg.points))
                for p in seg_points:
                    add_point(p.lon, p.lat, p.elevation)
        max_difficulty = SlopeConfig.DIFFICULTIES[hardest_idx] if hardest_idx >= 0 else SlopeConfig.DIFFICULTIES[0]
        return Route(
            node_path=node_path,
            path_points=tuple(points),
            steps=tuple(steps),
            total_slope_length_m=slope_len,
            total_slope_drop_m=slope_drop,
            lift_count=lift_count,
            max_difficulty=max_difficulty,
            difficulty_cap=cap,
            criteria=tuple(criteria),
        )

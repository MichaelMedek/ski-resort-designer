"""Route planner — the best A→B ski routes, precomputed per difficulty cap.

Pure model layer (stdlib + numpy + scipy + networkx only, like model.connectivity): no streamlit/ui
imports, so the core is unit-testable without a browser. Difficulty is an HONEST computation input, not
a post-filter: for each cap (green→black) we prune every slope harder than the cap OUT of the graph and
recompute, so "shortest path with max red" really is the shortest path over green/blue/red slopes.

Two route CATEGORIES, per cap:
  - Point-to-point (start ≠ end): the two shortest optima —
      FEWEST_LIFTS (fewest lift rides) and SHORTEST_SLOPE (least slope distance + light drop weight).
      Additive scipy-Dijkstra shortest paths on a re-weighted CSR; stable and cycle-safe.
  - Scenic closed tour (start == end): visit EVERY reachable lift and return, by the same two metrics
      (SCENIC_FEWEST_LIFTS / SCENIC_SHORTEST_SLOPE). This is an asymmetric TSP over the reachable lifts
      (arc-routing → ATSP; Rural-Postman is NP-hard so the ORDER is networkx's approximate ATSP, ~a few %
      of optimal). COMPLETENESS is exact — every reachable lift is a TSP city — and asserted.
"""

from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from itertools import groupby
from typing import cast

import networkx as nx
import numpy as np
import numpy.typing as npt
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from skiresort_planner.constants import RoutePlannerConfig, SlopeConfig
from skiresort_planner.model.connectivity import component_labels
from skiresort_planner.model.path_segment import PathSegment
from skiresort_planner.model.resort_graph import ResortGraph, SkiEdge

_NO_PREDECESSOR = -9999  # scipy dijkstra's "unreachable" sentinel in the predecessor array
_ATSP_SEED = 20260720  # fixed seed → deterministic scenic tour order (same graph ⇒ same route)


class RouteCriterion(StrEnum):
    """The metric a route is optimal under (reload-safe StrEnum). The SCENIC_* variants are the
    closed-tour "visit every lift" category; the others are point-to-point shortest paths.
    """

    FEWEST_LIFTS = "fewest_lifts"
    SHORTEST_SLOPE = "shortest_slope"  # least slope distance (+ light drop weight)
    SCENIC_FEWEST_LIFTS = "scenic_fewest_lifts"  # closed tour of every reachable lift, fewest rides
    SCENIC_SHORTEST_SLOPE = "scenic_shortest_slope"  # closed tour of every reachable lift, least slope

    @property
    def is_scenic(self) -> bool:
        """Whether this criterion is a scenic visit-every-lift tour (vs a point-to-point shortest path)."""
        return self in (RouteCriterion.SCENIC_FEWEST_LIFTS, RouteCriterion.SCENIC_SHORTEST_SLOPE)

    @property
    def base_metric(self) -> "RouteCriterion":
        """The point-to-point metric a scenic criterion optimises with (its edge weights). Identity for
        the non-scenic criteria.
        """
        if self.is_scenic:
            if self == RouteCriterion.SCENIC_FEWEST_LIFTS:
                return RouteCriterion.FEWEST_LIFTS
            if self == RouteCriterion.SCENIC_SHORTEST_SLOPE:
                return RouteCriterion.SHORTEST_SLOPE
            raise ValueError(f"Unexpected {self} for {self.is_scenic=}")
        return self


# Point-to-point criteria (start ≠ end) and scenic criteria (start == end), in browser/colour order.
_SHORTEST_CRITERIA = (RouteCriterion.FEWEST_LIFTS, RouteCriterion.SHORTEST_SLOPE)
_SCENIC_CRITERIA = (RouteCriterion.SCENIC_FEWEST_LIFTS, RouteCriterion.SCENIC_SHORTEST_SLOPE)

# The overlay palette must name a colour for every criterion (routes are coloured by their criterion).
assert set(RoutePlannerConfig.ROUTE_COLORS) == {c.value for c in RouteCriterion}, (
    "ROUTE_COLORS must key every RouteCriterion"
)


@dataclass(frozen=True)
class RouteStep:
    """One entity on a route: the slope or lift traversed, with what the UI shows for it."""

    is_lift: bool
    entity_id: str
    name: str
    detail: str  # lift type for a lift; difficulty for a slope


def _append_deduped(pts: list[tuple[float, float, float]], p: tuple[float, float, float]) -> None:
    """Append `p` unless it repeats the running tail — drops the junction shared by adjacent geometry."""
    if not pts or pts[-1] != p:
        pts.append(p)


def _concat_deduped(
    polylines: Iterable[Iterable[tuple[float, float, float]]],
) -> tuple[tuple[float, float, float], ...]:
    """Join polylines end-to-end into one, deduping the junction each shares with its predecessor. Single
    source for both `Route.path_points` (all elements) and a viewing group's folded run.
    """
    pts: list[tuple[float, float, float]] = []
    for poly in polylines:
        for p in poly:
            _append_deduped(pts, p)
    return tuple(pts)


@dataclass(frozen=True)
class ViewingGroup:
    """One viewing unit for the flythrough / panel legs: a lift, or a run of consecutive slopes between
    two lifts folded into one. `actual_polyline` is the run's real geometry (curves/cable); the camera
    faces its `straight_line` (start→end) so the sweep is steady lift→lift, not curve-tracking.
    """

    is_lift: bool
    actual_polyline: tuple[tuple[float, float, float], ...]
    steps: tuple["RouteStep", ...] = ()  # the route steps this group covers (empty for a standalone entity)

    @property
    def straight_line(self) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        """The (start, end) endpoints of the group — the gross sightline the camera faces along."""
        return (self.actual_polyline[0], self.actual_polyline[-1])


def build_viewing_groups(
    steps: tuple["RouteStep", ...],
    element_polylines: tuple[tuple[tuple[float, float, float], ...], ...],
) -> tuple[ViewingGroup, ...]:
    """Fold parallel steps + element polylines into viewing groups: each LIFT is its own group; a run of
    consecutive SLOPES between lifts merges into one. Single source for the panel legs, the flythrough
    keyframes, and the current-element highlight (they must never disagree on what "one unit" is).
    """

    def group_of(run: list[tuple["RouteStep", tuple[tuple[float, float, float], ...]]]) -> ViewingGroup:
        return ViewingGroup(
            is_lift=run[0][0].is_lift,
            steps=tuple(s for s, _ in run),
            actual_polyline=_concat_deduped(poly for _, poly in run),
        )

    groups: list[ViewingGroup] = []
    for is_lift, run_iter in groupby(zip(steps, element_polylines, strict=True), key=lambda sp: sp[0].is_lift):
        run = list(run_iter)
        if is_lift:
            groups.extend(group_of([pair]) for pair in run)  # each lift is its own group (never folded)
        else:
            groups.append(group_of(run))  # consecutive slopes fold into one
    return tuple(groups)


@dataclass(frozen=True)
class Route:
    """A computed route: the entities to traverse + per-route totals + its computation premise.

    node_path is the ordered node ids; element_polylines is one oriented polyline per traversed element
    (slope/lift) and path_points (derived) is those joined. difficulty_cap is the hardest band ALLOWED
    when computed; criteria lists every RouteCriterion this route wins under that cap. For a scenic tour,
    scenic_lifts_visited/target report lift coverage.
    """

    node_path: tuple[str, ...]
    # One oriented polyline per traversed element (parallel to `steps`), each running in ski-travel
    # direction. The drawable `path_points` is DERIVED from this (single geometry source, no drift); the
    # flythrough camera anchors one keyframe per element from it.
    element_polylines: tuple[tuple[tuple[float, float, float], ...], ...]
    steps: tuple[RouteStep, ...]
    total_slope_length_m: float
    total_slope_drop_m: float
    lift_count: int
    max_difficulty: str  # hardest slope band actually on the route ("green" if lift-only)
    difficulty_cap: str  # the premise: hardest band allowed when computing this route
    criteria: tuple[RouteCriterion, ...]
    scenic_lifts_visited: int = 0  # distinct lifts ridden on a scenic tour (0 for point-to-point)
    scenic_lifts_target: int = 0  # distinct lifts reachable under the cap (== visited: coverage is exact)

    @property
    def path_points(self) -> tuple[tuple[float, float, float], ...]:
        """The drawable (lon, lat, elevation) polyline along the pistes — the elements joined end-to-end,
        deduping the shared junction point between consecutive elements. Derived from element_polylines.
        """
        return _concat_deduped(self.element_polylines)

    @property
    def viewing_groups(self) -> tuple[ViewingGroup, ...]:
        """Between-lift viewing units (lift, or a folded run of consecutive slopes) — shared by the panel
        legs, the flythrough keyframes, and the current-element highlight so they can never drift.
        """
        return build_viewing_groups(steps=self.steps, element_polylines=self.element_polylines)

    @property
    def is_scenic(self) -> bool:
        """Whether this is a scenic visit-every-lift tour (any of its criteria is scenic)."""
        return any(c.is_scenic for c in self.criteria)

    @property
    def color(self) -> list[int]:
        """The overlay RGBA for this route, keyed by its first (representative) criterion — so a colour
        always means the same metric, regardless of how many routes a cap yields or their order.
        """
        return RoutePlannerConfig.ROUTE_COLORS[self.criteria[0].value]


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
        """The best routes for every difficulty cap, deduped within each cap.

        For each cap (green→black), slopes harder than the cap are pruned before search. Two categories:
        - start ≠ end → the two point-to-point shortest routes (fewest-lifts / shortest-slope).
        - start == end → the two scenic closed tours visiting every reachable lift and returning.
        Empty when the two nodes aren't both in the ski graph.
        """
        if start_node_id not in self._index or end_node_id not in self._index:
            return []
        src, dst = self._index[start_node_id], self._index[end_node_id]
        closed = src == dst

        out: list[Route] = []
        for cap in SlopeConfig.DIFFICULTIES:
            by_path: dict[tuple[str, ...], list[RouteCriterion]] = {}
            counts: dict[tuple[str, ...], tuple[int, int]] = {}  # node_path -> (visited, target) for scenic
            criteria = _SCENIC_CRITERIA if closed else _SHORTEST_CRITERIA
            for criterion in criteria:
                path: list[str] | None
                if criterion.is_scenic:
                    result = self._scenic_tour(src, cap=cap, criterion=criterion)
                    path = result[0] if result is not None else None
                    if result is not None:
                        counts[tuple(result[0])] = (result[1], result[1])  # closed-tour coverage is exact
                else:
                    path = self._shortest_path(criterion, src, dst, cap=cap)
                # A single-node path (start == end with no edge) isn't a route — you must move.
                if path is not None and len(path) >= 2:
                    by_path.setdefault(tuple(path), []).append(criterion)
            out.extend(
                self._build_route(
                    path,
                    crits,
                    cap=cap,
                    # Scenic paths MUST have a recorded count (strict); shortest paths carry none.
                    scenic_counts=counts[path] if any(c.is_scenic for c in crits) else (0, 0),
                )
                for path, crits in by_path.items()
            )
        return out

    # --- per-criterion path finding -------------------------------------------------

    def _segment_of(self, owner: SkiEdge) -> PathSegment:
        """The PathSegment behind a slope edge. Asserts it's a slope edge (segment_id set) — a lift
        edge here is a caller bug, so fail loud rather than silently mis-weight it.
        """
        assert not owner.is_lift and owner.segment_id is not None, "expected a slope edge with a segment"
        return self.graph.segments[owner.segment_id]

    def _edge_weight(self, criterion: RouteCriterion, owner: SkiEdge) -> float:
        """Additive edge cost for `criterion` (scenic criteria use their base_metric). A tiny epsilon
        keeps every weight strictly positive (Dijkstra needs non-negative) and prefers fewer edges on ties.
        """
        metric = criterion.base_metric
        eps = 1e-6
        if metric == RouteCriterion.FEWEST_LIFTS:
            return 1.0 if owner.is_lift else eps
        if owner.is_lift:
            return eps
        seg = self._segment_of(owner)  # SHORTEST_SLOPE: distance + a light drop weight for near-ties
        return seg.length_m + RoutePlannerConfig.SHORTEST_SLOPE_DROP_WEIGHT * abs(seg.total_drop_m) + eps

    def _within_cap(self, owner: SkiEdge, cap_idx: int) -> bool:
        """Whether an edge is allowed under a difficulty cap: lifts always; slopes only up to the cap."""
        return owner.is_lift or SlopeConfig.DIFFICULTIES.index(self._segment_of(owner).difficulty) <= cap_idx

    def _cap_edges(self, cap: str) -> list[tuple[str, str]]:
        """The edges allowed under `cap` (slopes harder than the cap pruned OUT). Shared by the
        shortest-path search, the reachable-lift SCC, and the scenic cost matrix.
        """
        cap_idx = SlopeConfig.DIFFICULTIES.index(cap)
        return [e for e in self._edges if self._within_cap(self._owner[e], cap_idx)]

    def _dijkstra(
        self, criterion: RouteCriterion, src: int, allowed: list[tuple[str, str]]
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int32]]:
        """Single-source scipy Dijkstra over `allowed` edges weighted for `criterion`. Returns
        (dist, pred) arrays over all node indices (dist == inf / pred == sentinel where unreachable).
        """
        n = len(self._nodes)
        rows = [self._index[a] for a, _ in allowed]
        cols = [self._index[b] for _, b in allowed]
        data = np.array([self._edge_weight(criterion, self._owner[e]) for e in allowed], dtype=np.float64)
        graph = csr_matrix((data, (rows, cols)), shape=(n, n))
        dist, pred = dijkstra(graph, directed=True, indices=src, return_predecessors=True)
        return cast(npt.NDArray[np.float64], dist), cast(npt.NDArray[np.int32], pred)

    def _shortest_path(self, criterion: RouteCriterion, src: int, dst: int, *, cap: str) -> list[str] | None:
        """Shortest path via scipy Dijkstra on the cap-pruned graph (route honestly restricted to `cap`)."""
        _dist, pred = self._dijkstra(criterion, src, self._cap_edges(cap))
        if dst != src and pred[dst] == _NO_PREDECESSOR:
            return None
        n = len(self._nodes)
        return self._walk_predecessors({i: int(pred[i]) for i in range(n)}, src, dst)

    def _walk_predecessors(self, pred: dict[int, int], src: int, dst: int) -> list[str]:
        """Reconstruct the node-id path from a {node: predecessor} map (dst back to src)."""
        idx_path = [dst]
        while idx_path[-1] != src:
            idx_path.append(pred[idx_path[-1]])
        idx_path.reverse()
        return [self._nodes[i] for i in idx_path]

    # --- scenic closed-tour (visit every reachable lift) ----------------------------

    def _reachable_lifts(self, src: int, allowed: list[tuple[str, str]]) -> list[SkiEdge]:
        """Lifts whose BOTH stations sit in the start node's strongly-connected component of the
        cap-pruned graph — exactly the lifts you can ride AND still return from (a provable set; a
        disconnected sub-resort's lifts fall out). Uses the shared SCC primitive (component_labels),
        the same one get_core_resort/can_loop_back derive from, just seeded from `src` instead of the core.
        """
        start_node = self._nodes[src]
        # Label every ski node in the cap-pruned graph; isolated nodes (start with no allowed edge)
        # land in their own component → no lift shares it → empty, which is correct.
        labels = component_labels(self._nodes, allowed, strong=True)
        scc = labels[start_node]
        seen: set[str] = set()
        lifts: list[SkiEdge] = []
        for a, b in allowed:
            owner = self._owner[(a, b)]
            # A lift is "ridden" on its UP edge (start_node→end_node); the down edge is only a connector.
            if owner.is_lift and owner.entity_id not in seen:
                lift = self.graph.lifts[owner.entity_id]
                if labels[lift.start_node_id] == scc and labels[lift.end_node_id] == scc:
                    seen.add(owner.entity_id)
                    lifts.append(owner)
        return lifts

    def _scenic_tour(self, src: int, *, cap: str, criterion: RouteCriterion) -> tuple[list[str], int] | None:
        """A closed walk from `src` visiting EVERY reachable lift (under `cap`) and returning, ordered
        near-optimally for `criterion`. Returns (node_path, distinct_lifts_visited) or None if no lift is
        reachable. Order is networkx's approximate ATSP (Rural-Postman is NP-hard); COMPLETENESS is exact
        (every reachable lift is a TSP city) and asserted.
        """
        allowed = self._cap_edges(cap)
        lifts = self._reachable_lifts(src, allowed)
        if not lifts:
            return None
        # TSP cities: 0 = start node; city i+1 = lift i (represented by its TOP node, reached by riding up).
        tops = [self._index[self.graph.lifts[lf.entity_id].end_node_id] for lf in lifts]
        cities = [src, *tops]
        bottoms = [src, *[self._index[self.graph.lifts[lf.entity_id].start_node_id] for lf in lifts]]
        # Asymmetric cost city_i → city_j = shortest cost (city_i → lift_j bottom) + the lift-j ride.
        big = float(len(allowed) + 1) * 1e9  # sentinel; never chosen (all cities share one SCC)
        m = len(cities)
        cost = np.full((m, m), big)
        ride = {lf.entity_id: self._edge_weight(criterion, lf) for lf in lifts}
        for i, ci in enumerate(cities):
            dist, _pred = self._dijkstra(criterion, ci, allowed)  # one sweep serves every target j
            cost[i][0] = dist[src] if np.isfinite(dist[src]) else big  # return to the anchor (no ride)
            for j in range(1, m):  # ski to lift-j's bottom, then ride it up
                d = dist[bottoms[j]]
                cost[i][j] = (d + ride[lifts[j - 1].entity_id]) if np.isfinite(d) else big
        cost[np.diag_indices(m)] = 0.0
        order = self._atsp_order(cost)
        node_path = self._stitch_tour(order, cities, bottoms, criterion, allowed)
        visited = {
            self._owner[(a, b)].entity_id
            for a, b in zip(node_path, node_path[1:], strict=False)
            if self._owner[(a, b)].is_lift
        }
        required = {lf.entity_id for lf in lifts}
        assert required <= visited, f"scenic tour dropped reachable lifts: {required - visited}"
        return node_path, len(required)

    @staticmethod
    def _atsp_order(cost: npt.NDArray[np.float64]) -> list[int]:
        """Near-optimal closed-tour visiting order over an asymmetric cost matrix, via networkx's
        approximate ATSP (threshold-accepting from a greedy start), seeded for determinism. City 0 is
        the tour anchor. Returns the city sequence (may repeat the anchor at the end).
        """
        n = cost.shape[0]
        weighted_edges = [(i, j, float(cost[i][j])) for i in range(n) for j in range(n) if i != j]
        g = nx.DiGraph((a, b, {"weight": w}) for a, b, w in weighted_edges)
        # networkx's approximate-ATSP returns Any; the call is deterministic (seeded greedy init).
        tour = nx.approximation.traveling_salesman_problem(
            g,
            weight="weight",
            cycle=True,
            method=lambda gg, weight: nx.approximation.threshold_accepting_tsp(
                gg, init_cycle="greedy", weight=weight, seed=_ATSP_SEED
            ),
        )
        return cast(list[int], tour)

    def _stitch_tour(
        self,
        order: list[int],
        cities: list[int],
        bottoms: list[int],
        criterion: RouteCriterion,
        allowed: list[tuple[str, str]],
    ) -> list[str]:
        """Expand a city visiting order into a real edge walk: for each city hop, ski to the next lift's
        bottom (shortest path) then ride its UP edge; the anchor city (0) contributes no ride.
        """
        node_path: list[str] = [self._nodes[cities[order[0]]]]

        def append_path(a: int, b: int) -> None:
            _dist, pred = self._dijkstra(criterion, a, allowed)
            seg = self._walk_predecessors({i: int(pred[i]) for i in range(len(self._nodes))}, a, b)
            node_path.extend(seg[1:])  # skip the first node (already the current tail)

        for city in order[1:]:
            if city == 0:  # returning to the anchor: just ski there, no lift
                append_path(self._index[node_path[-1]], cities[0])
            else:
                append_path(self._index[node_path[-1]], bottoms[city])  # ski to lift bottom
                node_path.append(self._nodes[cities[city]])  # ride the lift UP (bottom→top edge)
        return node_path

    # --- route assembly -------------------------------------------------------------

    def _build_route(
        self,
        node_path: tuple[str, ...],
        criteria: list[RouteCriterion],
        *,
        cap: str,
        scenic_counts: tuple[int, int] = (0, 0),
    ) -> Route:
        """Turn a node path into a Route: map each edge to its owning entity, aggregate stats, and
        trace the drawable polyline (slope segment + lift cable geometry, oriented per traversal).
        scenic_counts = (visited, target) distinct lifts for a scenic tour; (0, 0) for point-to-point.
        """
        steps: list[RouteStep] = []
        slope_len = slope_drop = 0.0
        lift_count = 0
        hardest_idx = -1
        # One polyline per element, parallel to `steps`; consecutive same-slope segments extend the last.
        element_polylines: list[list[tuple[float, float, float]]] = []

        for a, b in zip(node_path, node_path[1:], strict=False):  # consecutive node pairs
            owner = self._owner[(a, b)]
            if owner.is_lift:
                lift = self.graph.lifts[owner.entity_id]
                lift_count += 1
                steps.append(RouteStep(is_lift=True, entity_id=lift.id, name=lift.name, detail=lift.lift_type))
                # Trace the actual sagged cable geometry (like slopes trace their segment points), oriented
                # a→b: cable_points run start_node→end_node; reverse when riding the lift the other way.
                cable = lift.cable_points if lift.start_node_id == a else list(reversed(lift.cable_points))
                element_polylines.append([p.lon_lat_elev for p in cable])
            else:
                seg = self._segment_of(owner)
                slope_len += seg.length_m
                slope_drop += abs(seg.total_drop_m)
                hardest_idx = max(hardest_idx, SlopeConfig.DIFFICULTIES.index(seg.difficulty))
                slope = self.graph.slopes[owner.entity_id]
                # Collapse consecutive segments of the same slope into a single step + element polyline.
                same_slope = steps and not steps[-1].is_lift and steps[-1].entity_id == slope.id
                if not same_slope:
                    steps.append(RouteStep(is_lift=False, entity_id=slope.id, name=slope.name, detail=seg.difficulty))
                    element_polylines.append([])
                # Trace the segment's real geometry, oriented so it runs a→b (points are stored
                # start_node→end_node; reverse when the route skis it the other way). Dedupe the shared
                # junction against the element's running tail so a collapsed slope reads as one line.
                seg_points = seg.points if seg.start_node_id == a else list(reversed(seg.points))
                for p in seg_points:
                    _append_deduped(element_polylines[-1], p.lon_lat_elev)
        max_difficulty = SlopeConfig.DIFFICULTIES[hardest_idx] if hardest_idx >= 0 else SlopeConfig.DIFFICULTIES[0]
        visited, target = scenic_counts
        return Route(
            node_path=node_path,
            element_polylines=tuple(tuple(poly) for poly in element_polylines),
            steps=tuple(steps),
            total_slope_length_m=slope_len,
            total_slope_drop_m=slope_drop,
            lift_count=lift_count,
            max_difficulty=max_difficulty,
            difficulty_cap=cap,
            criteria=tuple(criteria),
            scenic_lifts_visited=visited,
            scenic_lifts_target=target,
        )

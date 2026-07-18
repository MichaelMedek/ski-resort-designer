"""Build a CONNECTED resort graph from raw OSM ways (validated on Ischgl).

Geometry comes from OSM; elevation/difficulty/pylons are recomputed from our DEM. The pipeline
enforces the strict node rules: no two nodes within MIN_NODE_DIST_M (100 m), no slope node within
RELAXED_MERGE_DIST_M (200 m) of a lift (it is pulled INTO the lift hub), lifts authoritative, no
duplicate runs, ≥90% connectivity:

  1. ways_to_lines: filter to standard groomed downhill/connection pistes + skiable lifts (splitting
     a lift way at interior stations), fully in-box, ≥ min length.
  2. DEDUP: drop a piste mostly covered by a longer one (OSM redraws = double runs).
  3. FULL-SPLIT: shapely unary_union planar-nodes every piste — split at EVERY crossing.
  4. HUB MERGE (iterated to fixpoint): cluster segment endpoints + lift stations so NO two hubs stay
     within MIN_NODE_DIST_M. Two passes: general 100 m (lift-authoritative — a cluster with a lift
     takes the lift coord), then a 200 m lift-pull that only pulls SLOPE nodes onto lift hubs.
  5. RE-CUT ON THE PISTE: each split segment is trimmed to the arc between its two hubs ALONG the
     original piste (never a straight chord to a far hub), so geometry stays on the real OSM piste.
  6. ORIENT + DROP: orient each run downhill; drop sustained climbs, off-source runs, and runs whose
     component holds no lift. One run per hub-pair (keep longest). Lifts ALWAYS kept.
  7. DEM-drape each kept run; endpoints pinned to hub nodes.

Output is an ImportGraph the ResortGraph materializes with shared nodes as one undoable batch.
"""

import heapq
import logging
import math
import time
from collections import defaultdict, deque
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from shapely import set_precision
from shapely.geometry import LineString, MultiLineString, Point
from shapely.ops import substring, unary_union

from skiresort_planner.constants import OSMConfig, SlopeConfig
from skiresort_planner.core.dem_service import DEMService
from skiresort_planner.core.terrain_analyzer import TerrainAnalyzer
from skiresort_planner.generators.osm_graph_plot import render_png
from skiresort_planner.generators.osm_importer import (
    BaseOSMImporter,
    ImportResult,
    OverpassElement,
    OverpassVertex,
    ProgressFn,
    split_lift_way_at_stations,
    station_node_ids,
    suffixed_lift_name,
)
from skiresort_planner.model.path_point import PathPoint

logger = logging.getLogger(__name__)

Vertex = tuple[float, float]  # (lon, lat)
XY = tuple[float, float]  # local planar metres


@dataclass
class SlopeRun:
    """One downhill run between two hub nodes: DEM-sampled points + its two node keys + optional name."""

    points: list[PathPoint]
    node_a: int
    node_b: int
    name: str | None = None
    fabricated: list[bool] = field(default_factory=list)  # per-point: True where the point off source OSM piste


@dataclass
class ImportSlope:
    """A whole slope = an ordered group of segment runs sharing one name/identity. The segments stay
    exactly as built; this only records which of them form one named piste + its (steepest) difficulty.
    """

    name: str | None
    run_indices: list[int]  # indices into ImportGraph.slope_runs
    difficulty: str


@dataclass
class LiftLine:
    """One lift: bottom/top stations (DEM-sampled) + type + its two node keys + optional name."""

    bottom: PathPoint
    top: PathPoint
    lift_type: str
    node_a: int
    node_b: int
    name: str | None = None


@dataclass
class ImportGraph:
    """A connected import result: shared node points, downhill slope runs, and lifts."""

    node_points: dict[int, PathPoint] = field(default_factory=dict)
    slope_runs: list[SlopeRun] = field(default_factory=list)
    lifts: list[LiftLine] = field(default_factory=list)
    slopes: list[ImportSlope] = field(default_factory=list)  # segment runs grouped into named whole slopes
    deduped: int = 0
    dropped_uphill: int = 0
    dropped_isolated: int = 0

    def to_slope_chains(self) -> list[tuple[list[list[PathPoint]], str | None]]:
        """Materialisation-ready app-slopes: (ordered segment point-lists, name), one per maximal LINEAR
        chain of a named piste.

        An app Slope is rendered as a single ordered point-list, so it MUST be linear — a branch or a
        disconnected arm cannot share one slope without the spline drawing a straight belt across the
        junction/void. So each ImportSlope is decomposed by longest-path into linear chains and every
        chain becomes its own contiguous app-slope. A piste that branches therefore yields a few slopes
        (the long main run + its side branches), all sharing the OSM name — never fragmented at EVERY
        junction (longest-path keeps the main descent whole), and never spliced across a gap.
        """
        elev = {k: v.elevation for k, v in self.node_points.items()}
        out: list[tuple[list[list[PathPoint]], str | None]] = []
        for sl in self.slopes:
            for chain in _linear_chains([self.slope_runs[ri] for ri in sl.run_indices], elev):
                out.append(([r.points for r in chain], sl.name))
        return out


def _linear_chains(group: list[SlopeRun], elev: dict[int, float]) -> list[list[SlopeRun]]:
    """Decompose a named piste's runs into contiguous top→bottom chains (one app Slope each), keeping
    the MAIN descent whole.

    Orient every run downhill (hi→lo) — that's a DAG. Treat each RUN as a node with a "can follow" edge
    A→B when A's bottom hub is B's top hub. We want each chain to be the longest continuous line, so at a
    fork the trunk continues down its LONGEST remaining branch and short side-runs spin off as their own
    (short) slopes. We compute each run's longest-downstream length by DAG DP, then greedily link every
    run to its unmatched successor with the greatest downstream length (each run linked once in, once
    out). Every chain is contiguous (shares hubs end-to-end — no spline belt); the trunk stays intact
    instead of being cut in half by a short parallel hop.
    """
    oriented = [(r.node_a, r.node_b, r) if elev[r.node_a] >= elev[r.node_b] else (r.node_b, r.node_a, r) for r in group]
    n = len(oriented)
    hi = [o[0] for o in oriented]
    lo = [o[1] for o in oriented]
    length = [OSMGraphBuilder._polylen_m(o[2].points) for o in oriented]
    by_hi: dict[int, list[int]] = defaultdict(list)
    for j in range(n):
        by_hi[hi[j]].append(j)
    succ = [by_hi.get(lo[a], []) for a in range(n)]

    # Longest-downstream length per run (memoised DAG DP; downhill orientation guarantees acyclic).
    downstream: dict[int, float] = {}

    def down_len(a: int) -> float:
        if a not in downstream:
            downstream[a] = length[a] + max((down_len(b) for b in succ[a]), default=0.0)
        return downstream[a]

    for a in range(n):
        down_len(a)

    # Greedy max-weight linking: link each run (longest-downstream trunk first) to its still-free
    # successor that itself has the longest downstream — the trunk keeps flowing down its main branch.
    match_next = [-1] * n
    match_prev = [-1] * n
    for a in sorted(range(n), key=lambda a: -down_len(a)):
        if match_next[a] != -1:
            continue
        candidates = sorted((b for b in succ[a] if match_prev[b] == -1), key=lambda b: -down_len(b))
        if candidates:
            b = candidates[0]
            match_next[a] = b
            match_prev[b] = a

    chains: list[list[SlopeRun]] = []
    for a in range(n):
        if match_prev[a] == -1:  # a chain HEAD (nothing precedes it)
            chain: list[SlopeRun] = []
            cur = a
            while cur != -1:
                chain.append(oriented[cur][2])
                cur = match_next[cur]
            chains.append(chain)
    return chains


def _is_importable_piste(tags: dict[str, str]) -> bool:
    """True for a piste we import: a connector (kept for connectivity, any difficulty) or a standard
    groomed downhill run (difficulty in the allow-list).
    """
    ptype = tags.get("piste:type")
    pdifficulty = tags.get("piste:difficulty")
    # connector kept for connectivity
    if ptype == OSMConfig.PISTE_TYPE_CONNECTION:
        return True
    # Anything else than groomed downhill run is discarded
    if ptype != OSMConfig.PISTE_TYPE_DOWNHILL:
        return False
    # Only keep standard difficulty runs
    return pdifficulty in OSMConfig.PISTE_DIFFICULTY_ALLOWED


def _is_importable_lift(tags: dict[str, str]) -> bool:
    """True for a skiable aerialway lift we import (a mapped aerialway type); everything else
    (station/pylon/zip_line/…) is ignored — the counterpart to `_is_importable_piste`.
    """
    aerialway = tags.get("aerialway")
    return aerialway in OSMConfig.AERIALWAY_TO_LIFT_TYPE


def ways_to_lines(
    elements: list[OverpassElement], bbox: tuple[float, float, float, float]
) -> tuple[list[tuple[list[Vertex], str | None]], list[tuple[list[Vertex], str, str | None]]]:
    """Split raw Overpass ways into (piste (verts, name), lift (verts, lift_type, name)).

    Keeps standard groomed downhill + connection pistes and skiable-aerialway lifts fully inside the
    box with ≥2 vertices; a lift way with interior `aerialway=station` nodes splits into per-section
    lifts. Names carried through for display; dedup/merge are geometric, not name-based.
    """
    min_lon, min_lat, max_lon, max_lat = bbox
    station_ids = station_node_ids(elements)

    def inside(geom: list[OverpassVertex]) -> bool:
        return all(min_lon <= v["lon"] <= max_lon and min_lat <= v["lat"] <= max_lat for v in geom)

    def name_of(tags: dict[str, str]) -> str | None:
        for key in ("name", "piste:name", "piste:ref", "ref"):
            if tags.get(key):
                return str(tags[key])
        return None

    pistes: list[tuple[list[Vertex], str | None]] = []
    lifts: list[tuple[list[Vertex], str, str | None]] = []
    for el in elements:
        if el.get("type") != "way":
            continue
        geom = el.get("geometry", [])
        tags = el.get("tags", {})
        if len(geom) < 2 or not inside(geom):
            continue
        verts = [(v["lon"], v["lat"]) for v in geom]
        if _is_importable_piste(tags):
            pistes.append((verts, name_of(tags)))
        elif _is_importable_lift(tags):
            lift_type = str(OSMConfig.AERIALWAY_TO_LIFT_TYPE[tags["aerialway"]])
            sections = split_lift_way_at_stations(vertices=verts, node_ids=el.get("nodes", []), station_ids=station_ids)
            name = name_of(tags)
            for i, section in enumerate(sections):
                lifts.append((section, lift_type, suffixed_lift_name(name, i, len(sections))))
    return pistes, lifts


class OSMGraphBuilder:
    """Turns raw Overpass ways into a connected ImportGraph. Pure geometry — no graph mutation."""

    def __init__(self, dem: DEMService, bbox: tuple[float, float, float, float]) -> None:
        self.dem = dem
        self.bbox = bbox
        lat0 = (bbox[1] + bbox[3]) / 2
        self._mlat = 111_320.0
        self._mlon = 111_320.0 * math.cos(math.radians(lat0))
        self._source_lines: list[LineString] = []  # source pistes (metres) for piste-following pulls
        self._named_sources: list[tuple[LineString, str]] = []  # (piste metres, OSM name) for run naming

    def _to_m(self, lon: float, lat: float) -> XY:
        return ((lon - self.bbox[0]) * self._mlon, (lat - self.bbox[1]) * self._mlat)

    def _to_deg(self, x: float, y: float) -> Vertex:
        return (self.bbox[0] + x / self._mlon, self.bbox[1] + y / self._mlat)

    def _hub_elev(self, hubs: list[XY], keys: Iterable[int]) -> dict[int, float]:
        """Hub id → DEM elevation for each key in `keys` (nodata hubs omitted)."""
        out: dict[int, float] = {}
        for h in keys:
            lon, lat = self._to_deg(*hubs[h])
            e = self.dem.get_elevation(lon=lon, lat=lat)
            if e is not None:
                out[h] = e
        return out

    def build(
        self,
        pistes: list[tuple[list[Vertex], str | None]],
        lifts: list[tuple[list[Vertex], str, str | None]],
        on_progress: ProgressFn,
    ) -> ImportGraph:
        """Build the connected graph from piste (verts,name) + lift (verts,type,name) inputs.

        `on_progress` fires one coarse marker before each major stage (the build runs several seconds
        over ~8 stages, so markers land roughly every 1-3 s).
        """
        t0 = time.perf_counter()
        on_progress(0.0, "Preparing pistes…")
        piste_lines = [LineString([self._to_m(lon, lat) for lon, lat in vs]) for vs, _nm in pistes if len(vs) >= 2]
        piste_lines = [ls for ls in piste_lines if ls.length >= OSMConfig.MIN_PISTE_LENGTH_M]
        # Named source pistes (metres) — used to re-attach the ORIGINAL OSM slope name to each final run.
        self._named_sources = []
        for vs, nm in pistes:
            if len(vs) < 2 or not nm:
                continue
            ls = LineString([self._to_m(lon, lat) for lon, lat in vs])
            if ls.length >= OSMConfig.MIN_PISTE_LENGTH_M:
                self._named_sources.append((ls, nm))
        lift_lines = [
            (LineString([self._to_m(lon, lat) for lon, lat in vs]), lt, nm) for vs, lt, nm in lifts if len(vs) >= 2
        ]
        lift_lines = [(ls, lt, nm) for ls, lt, nm in lift_lines if ls.length >= OSMConfig.MIN_LIFT_LENGTH_M]

        on_progress(0.12, "Deduplicating pistes…")
        kept, deduped = self._dedup(piste_lines)
        logger.debug(f"[IMPORT] dedup: {len(piste_lines)} pistes → {len(kept)} kept ({deduped} dropped)")
        on_progress(0.25, "Splitting at crossings…")
        segments = self._full_split(kept)
        segments = self._split_at_lift_stations(segments, lift_lines)
        logger.debug(f"[IMPORT] split: {len(segments)} segments from {len(kept)} pistes + {len(lift_lines)} lifts")
        on_progress(0.4, "Draping runs on terrain…")
        t_assemble = time.perf_counter()
        graph = self._assemble(segments, lift_lines, source=kept)
        assemble_ms = (time.perf_counter() - t_assemble) * 1000
        on_progress(0.7, "Naming runs…")
        self._name_runs(graph)
        # Drop redundant same-name parallel twins (R34) — needs run names, so AFTER _name_runs and
        # BEFORE _group_slopes (grouping must partition the FINAL run set). Reachability-guarded.
        on_progress(0.8, "Merging parallel runs…")
        self._drop_parallel_twins(graph)
        # Cross-pair coverage dedup (R2): a run ≥DEDUP_COVER within DEDUP_TOL of a LONGER run on a
        # DIFFERENT hub-pair (by_pair_final only dedups within one pair). AFTER the twin drop so the
        # reachability guard sees the repaired graph; then re-prune any dead-end a drop exposes.
        on_progress(0.9, "Removing duplicate runs…")
        self._dedup_final_runs(graph)
        self._prune_dead_end_slopes(graph)
        self._drop_isolated_slopes(graph)
        on_progress(0.97, "Grouping into slopes…")
        self._group_slopes(graph)
        graph.deduped = deduped
        named = sum(1 for r in graph.slope_runs if r.name)
        logger.debug(f"[IMPORT] grouped: {len(graph.slope_runs)} segments ({named} named) → {len(graph.slopes)} slopes")
        logger.info(
            f"[IMPORT] OSM graph: {len(graph.node_points)} nodes, {len(graph.slope_runs)} slopes, "
            f"{len(graph.lifts)} lifts (deduped {graph.deduped}, dropped uphill {graph.dropped_uphill}, "
            f"isolated {graph.dropped_isolated}) — build {(time.perf_counter() - t0) * 1000:.0f}ms "
            f"(assemble/DEM {assemble_ms:.0f}ms)"
        )
        return graph

    # -- step 2: dedup duplicate pistes -----------------------------------------------------------

    def _dedup(self, lines: list[LineString]) -> tuple[list[LineString], int]:
        """Drop a piste ≥DEDUP_COVER of whose length lies within DEDUP_TOL_M of a LONGER kept piste."""
        tol, cover = OSMConfig.DEDUP_TOL_M, OSMConfig.DEDUP_COVER_FRAC
        order = sorted(range(len(lines)), key=lambda i: lines[i].length, reverse=True)
        kept: list[LineString] = []
        for i in order:
            ls = lines[i]
            n = max(2, int(ls.length // 15))
            if any(
                sum(1 for t in range(n + 1) if kl.distance(ls.interpolate(t / n, normalized=True)) < tol) / (n + 1)
                >= cover
                for kl in kept
            ):
                continue
            kept.append(ls)
        return kept, len(lines) - len(kept)

    # -- step 3: full-split (planar-node at every crossing) ---------------------------------------

    def _full_split(self, lines: list[LineString]) -> list[LineString]:
        """Planar-node every piste (shapely unary_union): split at EVERY crossing. The connectivity
        engine — endpoint-only touching → ~47% connected; noding → ~98%. Snap-round vertices to a small
        grid FIRST (set_precision) so pistes that meet near-coincidentally (a run ending where another
        begins, ±a few m) collapse to identical coords and thus NODE together — otherwise unary_union
        leaves them disconnected and descent chains break across piste boundaries.
        """
        if not lines:
            return []
        snapped = [set_precision(ls, OSMConfig.SNAP_GRID_M) for ls in lines]
        noded = unary_union([s for s in snapped if not s.is_empty and s.length > 0])
        if isinstance(noded, MultiLineString):
            return [LineString(g.coords) for g in noded.geoms if g.length > 0]
        if isinstance(noded, LineString) and noded.length > 0:
            return [LineString(noded.coords)]
        return []

    def _split_at_lift_stations(
        self, segments: list[LineString], lifts: list[tuple[LineString, str, str | None]]
    ) -> list[LineString]:
        """Split each piste segment where a LIFT STATION projects onto its interior within
        RELAXED_MERGE_DIST_M, and move that split vertex ONTO the station coordinate. The station sits
        0–10 m from its feeder piste (verified on Ischgl), so the move stays on-piste — and it makes the
        feeder piste and the lift base share ONE hub after merge, so you can ski INTO the base (R21).
        """
        stations = [Point(lf.coords[0]) for lf, _lt, _nm in lifts] + [Point(lf.coords[-1]) for lf, _lt, _nm in lifts]
        if not stations:
            return segments
        # Only split where the station genuinely TOUCHES the piste (≤ SLOPE_ON_SOURCE_TOL_M). Snapping a
        # Split where a station is within MIN_NODE_DIST of the piste so the lift base/top shares a node
        # with its feeder (the station will merge onto that split vertex). Guarded below against creating
        # a zero-length (a==b) piece that would collapse the run.
        tol = OSMConfig.MIN_NODE_DIST_M
        snap_tol = OSMConfig.SLOPE_ON_SOURCE_TOL_M  # only MOVE the vertex onto the station within this
        out: list[LineString] = []
        for s in segments:
            cuts: list[tuple[float, XY | None]] = []
            for st in stations:
                dist = s.distance(st)
                if dist < tol:
                    d = s.project(st)
                    if 1.0 < d < s.length - 1.0:
                        # within snap_tol: snap the split vertex ONTO the station (they coincide). Farther
                        # (up to tol): split at the on-piste projection only; the station merges to it later.
                        cuts.append((d, (st.x, st.y) if dist < snap_tol else None))
            ends: list[tuple[float, XY | None]] = [(0.0, None), (s.length, None)] + cuts
            ends.sort(key=lambda t: t[0])
            for (d0, p0), (d1, p1) in zip(ends, ends[1:], strict=False):
                if d1 - d0 <= 1.0:
                    continue
                piece = substring(s, d0, d1)
                if not isinstance(piece, LineString) or piece.is_empty or len(piece.coords) < 2:
                    continue
                cs = [(x, y) for x, y in piece.coords]
                if p0 is not None:
                    cs[0] = p0  # snap the cut vertex exactly onto the lift station (0–30 m, on-piste)
                if p1 is not None:
                    cs[-1] = p1
                out.append(LineString(cs))
        return out

    # -- step 4: hub merge (iterated to fixpoint; strict spacing; lift-authoritative) -------------

    def _merge_hubs(self, seg_pts: list[XY], lift_pts: list[XY]) -> tuple[list[int], list[XY]]:
        """Cluster all endpoints so NO two hubs stay within MIN_NODE_DIST_M, then pull slope nodes onto
        lift hubs within RELAXED_MERGE_DIST_M. Returns (point→hub index, hub coords).

        A cluster with any lift member takes the lift centroid (lifts authoritative). Iterated to a
        fixpoint so a merged centroid can't re-create a sub-threshold pair.
        """
        pts = list(seg_pts) + list(lift_pts)
        is_lift = [False] * len(seg_pts) + [True] * len(lift_pts)
        assign = list(range(len(pts)))  # original point index -> current cluster index (into `pts`)

        def cluster_pass(tol: float, *, lift_only: bool, lift_lift_only: bool = False) -> bool:
            """One LEADER-clustering pass at `tol`: each point joins the nearest already-chosen leader
            within `tol`, else becomes a new leader. Unlike single-linkage this does NOT chain (A–B–C
            collapsing 200 m apart) — every member stays within `tol` of its leader, so no hub ends up
            far from the endpoints it represents. Lift points are preferred as leaders (authoritative);
            lift_only=True keeps distinct lift stations apart (a lift never joins another lift);
            lift_lift_only=True merges ONLY lift↔lift pairs (slope nodes untouched — used to consolidate
            a nearby lift-station complex without collapsing slope spacing).
            """
            n = len(pts)
            order = sorted(range(n), key=lambda i: not is_lift[i])  # lifts first → they become leaders
            leaders: list[int] = []
            grid: dict[tuple[int, int], list[int]] = defaultdict(list)
            assign_leader = [-1] * n
            changed = False
            for i in order:
                if lift_lift_only and not is_lift[i]:
                    leaders.append(i)
                    gx0, gy0 = int(pts[i][0] // tol), int(pts[i][1] // tol)
                    grid[(gx0, gy0)].append(i)
                    assign_leader[i] = i
                    continue
                gx, gy = int(pts[i][0] // tol), int(pts[i][1] // tol)
                best, best_d = -1, tol
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        for ld in grid.get((gx + dx, gy + dy), []):
                            if lift_only and is_lift[i] and is_lift[ld]:
                                continue  # a lift never joins another lift
                            if lift_lift_only and not (is_lift[i] and is_lift[ld]):
                                continue  # only lift↔lift merges in this mode
                            d = math.dist(pts[i], pts[ld])
                            if d < best_d:
                                best, best_d = ld, d
                if best == -1:
                    leaders.append(i)
                    grid[(gx, gy)].append(i)
                    assign_leader[i] = i
                else:
                    assign_leader[i] = best
                    changed = True
            if not changed:
                return False
            groups: dict[int, list[int]] = defaultdict(list)
            for i in range(n):
                groups[assign_leader[i]].append(i)
            new_pts: list[XY] = []
            new_lift: list[bool] = []
            leader_to_new: dict[int, int] = {}
            for leader, members in groups.items():
                lifts_here = [m for m in members if is_lift[m]]
                # Hub takes a REPRESENTATIVE real coordinate (a lift station if any, else the leader
                # endpoint) — NOT a centroid; a real endpoint keeps every hub ON the piste/lift.
                rep = lifts_here[0] if lifts_here else leader
                new_pts.append(pts[rep])
                new_lift.append(bool(lifts_here))
                leader_to_new[leader] = len(new_pts) - 1
            remap = {old: leader_to_new[assign_leader[old]] for old in range(n)}
            for i in range(len(assign)):
                assign[i] = remap[assign[i]]
            pts[:] = new_pts
            is_lift[:] = new_lift
            return True

        # ONE leader pass per phase — a single pass already guarantees every member is within tol of its
        # leader (no chaining); iterating would re-cluster leaders and re-introduce the A–B–C chain.
        cluster_pass(OSMConfig.MIN_NODE_DIST_M, lift_only=False)  # strict spacing
        cluster_pass(OSMConfig.RELAXED_MERGE_DIST_M, lift_only=True)  # relaxed lift-pull (ALWAYS, strict)
        return assign, pts

    def _build_premerge_graph(
        self, segments: list[LineString], seg_ab: list[tuple[int, int]]
    ) -> tuple[list[XY], list[int], list[float], dict[int, list[tuple[int, int]]]]:
        """Pre-merge node graph over the raw split segments: unique node per rounded metre coord, each
        segment an edge carrying its index. Returns (pn_xy, pn_hub, pn_elev, padj) where pn_hub[p] is the
        merged-hub id of pre-merge node p and pn_elev[p] its DEM height (math.inf for a nodata point, so
        the climb-cap rejects it). Shared by the contraction and the stranded-sink reconnection.
        """
        pn_id: dict[tuple[float, float], int] = {}
        pn_xy: list[XY] = []
        pn_hub: list[int] = []
        pn_elev: list[float] = []

        def pnode(xy: XY, hub: int) -> int:
            k = (round(xy[0], 1), round(xy[1], 1))
            if k not in pn_id:
                pn_id[k] = len(pn_xy)
                pn_xy.append(xy)
                pn_hub.append(hub)
                lon, lat = self._to_deg(*xy)
                e = self.dem.get_elevation(lon=lon, lat=lat)
                pn_elev.append(e if e is not None else math.inf)
            return pn_id[k]

        padj: dict[int, list[tuple[int, int]]] = defaultdict(list)  # pnode -> [(pnode, segment index)]
        for i, s in enumerate(segments):
            cs = list(s.coords)
            a = pnode((cs[0][0], cs[0][1]), seg_ab[i][0])
            b = pnode((cs[-1][0], cs[-1][1]), seg_ab[i][1])
            if a != b:
                padj[a].append((b, i))
                padj[b].append((a, i))
        return pn_xy, pn_hub, pn_elev, padj

    def _contract_collapsed_descents(
        self,
        segments: list[LineString],
        seg_ab: list[tuple[int, int]],
        lift_ab: list[tuple[int, int]],
        hubs: list[XY],
    ) -> list[tuple[LineString, int, int]]:
        """One through-segment per lift TOP whose descent collapsed into self-loops.

        A lift top fed only by pistes SHORTER than the merge distance keeps no descending edge — every
        piece has both ends on the top hub and is dropped. We rebuild the pre-merge node graph, and from
        each such top walk the real geometry downhill (min-climb ≤ NODE_TERRAIN_TOL_M per hop) to the DISTINCT,
        strictly-lower, non-lift hub NEAREST the lift's OWN base; the concatenated arc becomes one
        contracted run top→that-hub. No new node is created (R5 safe); the geometry is real OSM (R19).
        """
        helev = self._hub_elev(hubs, {a for ab in seg_ab for a in ab} | {a for ab in lift_ab for a in ab})
        # which hubs already have a descending slope edge out of them?
        has_descent_out: set[int] = set()
        for a, b in seg_ab:
            if a == b or helev.get(a) is None or helev.get(b) is None:
                continue
            has_descent_out.add(a if helev[a] >= helev[b] else b)

        pn_xy, pn_hub, pn_elev, padj = self._build_premerge_graph(segments, seg_ab)

        lift_hubs = {h for ab in lift_ab for h in ab}
        min_lift_base = min((helev[h] for h in lift_hubs if h in helev), default=0.0)
        out: list[tuple[LineString, int, int]] = []
        # Reconnect every hub that has NO descending slope edge yet sits ABOVE some lift base — a lift top
        # whose descent collapsed OR a mid-mountain sink where the continuing piste was dropped. Both are
        # stranding dead-ends (R22); walk the real OSM geometry downhill to a lower hub that drains onward.
        # A hub BELOW every lift base is a genuine valley terminus (return lift out of bbox) — left alone.
        seed_hubs: list[tuple[int, int | None]] = []
        for a, b in lift_ab:
            if a == b or helev.get(a) is None or helev.get(b) is None:
                continue
            top = a if helev[a] >= helev[b] else b
            if top not in has_descent_out:
                seed_hubs.append((top, b if top == a else a))  # (stranded hub, its lift base for aiming)
        for h in helev:
            if h in lift_hubs or h in has_descent_out or helev[h] <= min_lift_base:
                continue
            seed_hubs.append((h, None))  # mid-mountain sink — aim at the nearest reachable lower hub

        for top, base_hub in seed_hubs:
            starts = [p for p in range(len(pn_xy)) if pn_hub[p] == top]
            if not starts:
                continue
            # min-climb walk (≤NODE_TERRAIN_TOL per hop) over ALL reachable pre-merge nodes; collect valid exit
            # hubs (distinct, strictly lower, non-lift) and pick the one nearest the aim point.
            best: dict[int, float] = dict.fromkeys(starts, 0.0)
            par: dict[int, tuple[int, int]] = {}
            pq: list[tuple[float, int]] = [(0.0, p) for p in starts]
            heapq.heapify(pq)
            aim_xy = hubs[base_hub] if base_hub is not None else hubs[top]
            exits: list[tuple[float, int]] = []
            while pq:
                c, x = heapq.heappop(pq)
                if c > best.get(x, 1e18):
                    continue
                xh = pn_hub[x]
                if xh != top and xh not in lift_hubs and helev.get(xh) is not None and helev[xh] < helev[top]:
                    exits.append((math.dist(hubs[xh], aim_xy), x))
                    continue  # record, but keep walking for exits nearer the aim
                for y, si in padj[x]:
                    climb = max(0.0, pn_elev[y] - pn_elev[x])
                    if climb > OSMConfig.NODE_TERRAIN_TOL_M:  # nodata (inf) rejected here
                        continue
                    nc = c + climb
                    if nc < best.get(y, 1e18):
                        best[y] = nc
                        par[y] = (x, si)
                        heapq.heappush(pq, (nc, y))
            if not exits:
                continue
            target = min(exits, key=lambda t: t[0])[1]
            exit_hub = pn_hub[target]
            chain: list[int] = [target]
            n = target
            while n in par:
                n = par[n][0]
                chain.append(n)
            chain.reverse()
            clean = self._concat_chain_geometry(chain, hubs[top], hubs[exit_hub], segments, padj, pn_xy)
            if len(clean) < 2:
                continue
            logger.debug(
                f"[IMPORT] contract: stranded hub {top} → through-run to hub {exit_hub}, "
                f"{len(clean)} pts, min-climb {best[target]:.1f}m"
            )
            out.append((LineString(clean), top, exit_hub))
        return out

    @staticmethod
    def _concat_chain_geometry(
        chain: list[int],
        start: XY,
        end: XY,
        segments: list[LineString],
        padj: dict[int, list[tuple[int, int]]],
        pn_xy: list[XY],
    ) -> list[XY]:
        """Concatenate the real OSM geometry of the segments along a pre-merge node `chain`, oriented
        start→end, deduping consecutive near-identical points. Endpoints pinned to `start`/`end` hubs.
        """
        coords: list[XY] = [start]
        for u, v in zip(chain, chain[1:], strict=False):
            si = next(si for (w, si) in padj[u] if w == v)
            cs = list(segments[si].coords)
            if math.dist((cs[0][0], cs[0][1]), pn_xy[u]) > math.dist((cs[-1][0], cs[-1][1]), pn_xy[u]):
                cs = cs[::-1]  # orient piece u→v
            coords.extend((x, y) for x, y in cs)
        coords.append(end)
        clean: list[XY] = [coords[0]]
        for p in coords[1:]:
            if math.dist(p, clean[-1]) >= 1.0:
                clean.append(p)
        return clean

    # -- step 5: assemble -------------------------------------------------------------------------

    def _assemble(
        self, segments: list[LineString], lifts: list[tuple[LineString, str, str | None]], source: list[LineString]
    ) -> ImportGraph:
        seg_pts: list[XY] = []
        for s in segments:
            cs = list(s.coords)
            seg_pts += [(cs[0][0], cs[0][1]), (cs[-1][0], cs[-1][1])]
        lift_pts: list[XY] = []
        for lf, _lt, _nm in lifts:
            cs = list(lf.coords)
            lift_pts += [(cs[0][0], cs[0][1]), (cs[-1][0], cs[-1][1])]

        assign, hubs = self._merge_hubs(seg_pts, lift_pts)
        seg_ab = [(assign[2 * i], assign[2 * i + 1]) for i in range(len(segments))]
        base = len(seg_pts)
        lift_ab = [(assign[base + 2 * j], assign[base + 2 * j + 1]) for j in range(len(lifts))]

        # Contract collapsed descent chains: a lift top whose descending pistes are all SHORTER than the
        # merge distance sees every piece become a self-loop (both ends → the top hub) and dropped — its
        # whole descent vanishes (Lange Wandbahn). Walk the real pre-merge geometry from such a top out to
        # the first DISTINCT downhill hub and emit ONE through-segment carrying the concatenated OSM arc.
        contracted = self._contract_collapsed_descents(segments, seg_ab, lift_ab, hubs)
        for seg, ca, cb in contracted:
            segments.append(seg)
            seg_ab.append((ca, cb))
        # keep the pre-merge geometry so the post-drop reconnection pass can rebuild descents for sinks
        # that only STRAND after the later drop/dedup/gate passes (not visible at contraction time).
        self._pre_segments, self._pre_seg_ab, self._pre_hubs, self._pre_lift_ab = segments, seg_ab, hubs, lift_ab

        source_union = MultiLineString([s for s in source if s.length > 0]) if source else None
        self._source_lines = [s for s in source if s.length > 0]  # for piste-following connectors
        # R12 measures a run point's distance to the nearest source-piste VERTEX (not the line). Build a
        # vertex STRtree of ALL piste vertices so the builder's final gate matches R12 exactly.
        from shapely import STRtree

        self._piste_vertices = [Point(c) for s in self._source_lines for c in s.coords]
        self._vtree = STRtree(self._piste_vertices) if self._piste_vertices else None

        # component per hub; keep only segments reaching a lift
        adj: dict[int, set[int]] = defaultdict(set)
        for a, b in seg_ab + lift_ab:
            if a != b:
                adj[a].add(b)
                adj[b].add(a)
        comp = self._components(len(hubs), adj)
        lift_comps = {comp[a] for a, b in lift_ab} | {comp[b] for a, b in lift_ab}

        # dedup: one run per hub-pair (keep the longest) — kills copies the split creates.
        by_pair: dict[frozenset[int], tuple[LineString, int, int]] = {}
        for seg, (a, b) in zip(segments, seg_ab, strict=True):
            if a == b:
                continue
            key = frozenset((a, b))
            if key not in by_pair or seg.length > by_pair[key][0].length:
                by_pair[key] = (seg, a, b)

        graph = ImportGraph()
        kept: list[tuple[LineString, int, int]] = []
        used: set[int] = set()
        for seg, a, b in by_pair.values():
            if comp[a] not in lift_comps:
                graph.dropped_isolated += 1
                continue
            if not self._on_source(seg, source_union):
                graph.dropped_isolated += 1
                continue
            kept.append((seg, a, b))
            used.update((a, b))
        for a, b in lift_ab:
            used.update((a, b))

        for h in used:
            lon, lat = self._to_deg(*hubs[h])
            elev = self.dem.get_elevation(lon=lon, lat=lat)
            if elev is not None:
                graph.node_points[h] = PathPoint(lon=lon, lat=lat, elevation=elev)

        kept = self._split_at_interior_hubs(kept, hubs, used)

        for seg, a, b in kept:
            run = self._build_slope_run(seg, a, b, graph, source_union)
            if run is not None:
                graph.slope_runs.append(run)

        # Drop floaters BEFORE the per-pair dedup so dedup only chooses among runs that survive the R6
        # floating-hub gate — otherwise it may keep a run for a pair, then have it floating-dropped,
        # losing the pair (and the descent) when a shorter sibling would have survived.
        self._drop_slopes_floating_past_hubs(graph)

        # Dedup again AFTER splitting: at most one slope per hub-pair. Splitting at interior hubs can
        # recreate same-pair duplicates (a parallel run + a split piece) — collapse them, keeping the longest.
        by_pair_final: dict[frozenset[int], SlopeRun] = {}
        for run in graph.slope_runs:
            key = frozenset((run.node_a, run.node_b))
            cur = by_pair_final.get(key)
            if cur is None or self._polylen_m(run.points) > self._polylen_m(cur.points):
                by_pair_final[key] = run
        graph.slope_runs = list(by_pair_final.values())

        for (_lf, lt, nm), (a, b) in zip(lifts, lift_ab, strict=True):
            if a == b or a not in graph.node_points or b not in graph.node_points:
                continue
            pa, pb = graph.node_points[a], graph.node_points[b]
            if pa.elevation <= pb.elevation:
                graph.lifts.append(LiftLine(bottom=pa, top=pb, lift_type=lt, node_a=a, node_b=b, name=nm))
            else:
                graph.lifts.append(LiftLine(bottom=pb, top=pa, lift_type=lt, node_a=b, node_b=a, name=nm))

        self._prune_dead_end_slopes(graph)
        self._drop_isolated_slopes(graph)
        self._reconnect_stranded_sinks(graph)  # rebuild descents for sinks stranded after drops (never uphill, R3)
        self._prune_dead_end_slopes(graph)  # final: a drop above can leave a fresh dead-end
        self._drop_isolated_slopes(graph)
        self._mark_fabricated(graph, source_union)  # tag off-piste points for the red reference overlay
        return graph

    def _mark_fabricated(self, graph: ImportGraph, source_union: MultiLineString | None) -> None:
        """Flag each run point that lies > PISTE_TOL_M from every source OSM piste — invented geometry
        (a pull/connector), the same off-piste test the pull-shape gate uses. Drives the plot's red
        overlay so a reader sees clean OSM (blue) vs fabricated-by-us (red). No source → nothing marked.
        """
        tol = OSMConfig.PISTE_TOL_M
        for r in graph.slope_runs:
            if source_union is None:
                r.fabricated = [False] * len(r.points)
            else:
                r.fabricated = [source_union.distance(Point(self._to_m(p.lon, p.lat))) > tol for p in r.points]

    def _name_runs(self, graph: ImportGraph) -> None:
        """Attach the ORIGINAL OSM piste name to each run: the name of the named source piste that best
        covers the run's geometry (≥ half the run's sample points within SLOPE_ON_SOURCE_TOL_M of it).
        A run following two named pistes takes the one covering more of it; a run on no named piste
        (pure connector) stays unnamed. Preserves the real slope identity through the full-split.
        """
        if not self._named_sources:
            return
        tol = OSMConfig.SLOPE_ON_SOURCE_TOL_M + 10.0
        for r in graph.slope_runs:
            samples = [Point(self._to_m(p.lon, p.lat)) for p in r.points]
            best_name, best_cover = None, 0.0
            for line, name in self._named_sources:
                cover = sum(1 for q in samples if line.distance(q) <= tol) / len(samples)
                if cover > best_cover:
                    best_cover, best_name = cover, name
            if best_cover >= 0.5:
                r.name = best_name

    def _run_max_slope_pct(self, run: SlopeRun) -> float:
        """Steepest-section slope magnitude (%) of a run, rolled over ROLLING_WINDOW_M — the same metric
        the app's difficulty classifier uses. Falls back to the average grade on a short run.
        """
        pts = run.points
        if len(pts) < 2:
            return 0.0
        cum = [0.0]
        for i in range(1, len(pts)):
            cum.append(cum[-1] + pts[i - 1].distance_to(other=pts[i]))
        total = cum[-1]
        if total <= 0:
            return 0.0
        avg = abs(pts[0].elevation - pts[-1].elevation) / total * 100.0
        win = SlopeConfig.ROLLING_WINDOW_M
        if total < win:
            return avg
        best = avg
        for i in range(len(pts)):
            j = i
            while j < len(pts) and cum[j] - cum[i] < win:
                j += 1
            if j >= len(pts):
                break
            run_m = cum[j] - cum[i]
            if run_m > 0:
                best = max(best, abs(pts[i].elevation - pts[j].elevation) / run_m * 100.0)
        return best

    def _group_slopes(self, graph: ImportGraph) -> None:
        """Group segment runs into whole named slopes (segments stay exactly as built).

        1. Runs sharing an OSM name form one slope; its difficulty = the STEEPEST member's band.
        2. An unnamed run (connector) is folded into an adjacent slope it touches ONLY when doing so does
           NOT raise that slope's difficulty band (never turn a red slope black by grafting a short steep
           link). Among slopes it could join, it picks the one whose difficulty is closest to its own —
           preferring a gentle connector into a gentle slope. A connector that would upgrade every
           neighbour is left as its own single-run slope (the black expert short-cut stays separate).
        3. Every run lands in exactly one slope (referential completeness for the ratio test).
        """
        runs = graph.slope_runs
        band = SlopeConfig.DIFFICULTIES  # ["green","blue","red","black"], ascending
        rank = {d: i for i, d in enumerate(band)}
        run_diff = [TerrainAnalyzer.classify_difficulty(slope_pct=self._run_max_slope_pct(r)) for r in runs]

        # 1) named groups
        by_name: dict[str, list[int]] = defaultdict(list)
        unnamed: list[int] = []
        for i, r in enumerate(runs):
            if r.name:
                by_name[r.name].append(i)
            else:
                unnamed.append(i)
        slopes: list[ImportSlope] = [
            ImportSlope(name=nm, run_indices=list(idxs), difficulty=band[max(rank[run_diff[i]] for i in idxs)])
            for nm, idxs in by_name.items()
        ]

        # 2) fold each unnamed connector into the best adjacent slope that it does NOT upgrade
        node_slopes: dict[int, list[int]] = defaultdict(list)  # node -> slope indices touching it
        for si, sl in enumerate(slopes):
            for ri in sl.run_indices:
                node_slopes[runs[ri].node_a].append(si)
                node_slopes[runs[ri].node_b].append(si)
        for ci in unnamed:
            r = runs[ci]
            cdiff = rank[run_diff[ci]]
            cands = {si for n in (r.node_a, r.node_b) for si in node_slopes.get(n, [])}
            # eligible: joining does not raise the slope's band (connector no steeper than the slope)
            eligible = [si for si in cands if cdiff <= rank[slopes[si].difficulty]]
            if eligible:
                best = min(eligible, key=lambda si: abs(rank[slopes[si].difficulty] - cdiff))
                slopes[best].run_indices.append(ci)
            else:
                slopes.append(ImportSlope(name=None, run_indices=[ci], difficulty=band[cdiff]))
        graph.slopes = slopes

    @staticmethod
    def _skier_reachable(graph: ImportGraph, elev: dict[int, float]) -> set[int]:
        """Every node a skier can stand on: from each lift TOP, ski DOWN slope edges and ride lifts UP,
        to a fixpoint. A slope whose HIGH node is NOT in this set can never be entered (an unreachable
        top — the mirror of a downhill sink).
        """
        down = OSMGraphBuilder._down_adjacency(graph, elev)
        lift_up: dict[int, set[int]] = defaultdict(set)
        seeds: set[int] = set()
        for lf in graph.lifts:
            base, top = (lf.node_a, lf.node_b) if elev[lf.node_a] <= elev[lf.node_b] else (lf.node_b, lf.node_a)
            lift_up[base].add(top)
            seeds.add(top)
        reach = set(seeds)
        q = deque(seeds)
        while q:
            x = q.popleft()
            for y in list(down.get(x, ())) + list(lift_up.get(x, ())):
                if y not in reach:
                    reach.add(y)
                    q.append(y)
        return reach

    def _reconnect_stranded_sinks(self, graph: ImportGraph) -> None:
        """Rebuild a descending run for every slope node that, in the FINAL graph, cannot reach a lift
        going down yet sits ABOVE some lift base (a sink created by the drop/dedup/gate passes, not
        visible at contraction time). Walks the stored pre-merge OSM geometry from the sink downhill to a
        hub that DOES reach a lift, and adds that run. Never drops; a sink BELOW all lift bases (valley
        terminus, return lift out of bbox) is left alone. Iterated to a fixpoint.
        """
        segments, seg_ab, hubs = self._pre_segments, self._pre_seg_ab, self._pre_hubs
        elev = self._node_elevations(graph)
        lift_nodes = {n for lf in graph.lifts for n in (lf.node_a, lf.node_b)}
        if not lift_nodes:
            return
        min_lift_base = min(elev[n] for n in lift_nodes)

        pn_xy, pn_hub, pn_elev, padj = self._build_premerge_graph(segments, seg_ab)

        for _ in range(20):
            good = self._hubs_reaching_lift(graph, lift_nodes)
            sinks = [
                n
                for n in {a for r in graph.slope_runs for a in (r.node_a, r.node_b)}
                if n not in good and n not in lift_nodes and elev[n] > min_lift_base
            ]
            if not sinks:
                break
            added = False
            for sink in sinks:
                if self._reconnect_one_sink(sink, good, graph, elev, segments, hubs, pn_xy, pn_hub, pn_elev, padj):
                    added = True
            if not added:
                break

        # Per-lift OWN-base descent (R21): a lift top may reach SOME lift going down yet not its own
        # base. Walk the pre-merge OSM geometry from the top to a node that strict-descends to its base
        # and add the missing runs (raw OSM proven to have such a path). Iterated to a fixpoint.
        for _ in range(20):
            added = False
            for lf in graph.lifts:
                elev = self._node_elevations(graph)
                top = lf.node_a if elev[lf.node_a] >= elev[lf.node_b] else lf.node_b
                base = lf.node_b if top == lf.node_a else lf.node_a
                reach_base = self._down_reaches(graph, {base}, elev)  # nodes that strict-descend to base
                if top in reach_base:
                    continue
                if self._reconnect_one_sink(top, reach_base, graph, elev, segments, hubs, pn_xy, pn_hub, pn_elev, padj):
                    added = True
            if not added:
                break

        # Unreachable slope TOPS (mirror of sinks): a slope whose high node no skier can reach — no lift
        # arrives and no piste descends INTO it — is a phantom (you can't get onto it). Walk the pre-merge
        # OSM geometry UP a real descending feeder to a reachable higher node and add it. Iterated.
        for _ in range(20):
            elev = self._node_elevations(graph)
            reachable = self._skier_reachable(graph, elev)
            tops = {(r.node_a if elev[r.node_a] >= elev[r.node_b] else r.node_b) for r in graph.slope_runs} - reachable
            if not tops:
                break
            added = False
            for top in tops:
                if self._reconnect_one_sink(
                    top, reachable, graph, elev, segments, hubs, pn_xy, pn_hub, pn_elev, padj, want_higher=True
                ):
                    added = True
            if not added:
                break

    @staticmethod
    def _node_elevations(graph: ImportGraph) -> dict[int, float]:
        """Node id → elevation for the current graph nodes."""
        return {k: v.elevation for k, v in graph.node_points.items()}

    @staticmethod
    def _down_adjacency(graph: ImportGraph, elev: dict[int, float]) -> dict[int, set[int]]:
        """Hi → {lo}: descending slope adjacency, each run oriented by its endpoints' elevations."""
        down: dict[int, set[int]] = defaultdict(set)
        for r in graph.slope_runs:
            hi, lo = (r.node_a, r.node_b) if elev[r.node_a] >= elev[r.node_b] else (r.node_b, r.node_a)
            down[hi].add(lo)
        return down

    @staticmethod
    def _down_reaches(graph: ImportGraph, targets: set[int], elev: dict[int, float]) -> set[int]:
        """Every node that reaches any of `targets` following DESCENDING slope edges (targets included)."""
        down = OSMGraphBuilder._down_adjacency(graph, elev)
        up: dict[int, set[int]] = defaultdict(set)
        for hi, los in down.items():
            for lo in los:
                up[lo].add(hi)  # from lo you can be REACHED by hi going down
        good = set(targets)
        q = deque(targets)
        while q:
            x = q.popleft()
            for y in up[x]:
                if y not in good:
                    good.add(y)
                    q.append(y)
        return good

    @staticmethod
    def _hubs_reaching_lift(graph: ImportGraph, lift_nodes: set[int]) -> set[int]:
        """Every node from which a lift station is reachable following DESCENDING slope edges."""
        down = OSMGraphBuilder._down_adjacency(graph, OSMGraphBuilder._node_elevations(graph))
        good: set[int] = set()
        for start in graph.node_points:
            seen, q = {start}, deque([start])
            while q:
                x = q.popleft()
                if x in lift_nodes:
                    good.add(start)
                    break
                for y in down[x]:
                    if y not in seen:
                        seen.add(y)
                        q.append(y)
        return good

    def _reconnect_one_sink(
        self,
        sink: int,
        good: set[int],
        graph: ImportGraph,
        elev: dict[int, float],
        segments: list[LineString],
        hubs: list[XY],
        pn_xy: list[XY],
        pn_hub: list[int],
        pn_elev: list[float],
        padj: dict[int, list[tuple[int, int]]],
        *,
        want_higher: bool = False,
    ) -> bool:
        """Walk the pre-merge OSM geometry from `sink` to the nearest node in `good`, and add the
        connecting run(s) — split at intermediate existing hubs so nothing floats (R6) or duplicates (R2).

        Direction: by default (downhill sink) walk DOWN — cost is upward movement, target must be LOWER;
        with want_higher (an unreachable slope TOP) walk UP a real descending feeder — cost is downward
        movement, target must be HIGHER. Either way each hop's against-grain move is capped at
        NODE_TERRAIN_TOL (a DEM-sampling dip, not a real wall). Returns True if any run was added.
        """
        starts = [p for p in range(len(pn_xy)) if pn_hub[p] == sink]
        if not starts:
            return False
        best: dict[int, float] = dict.fromkeys(starts, 0.0)
        par: dict[int, tuple[int, int]] = {}
        pq: list[tuple[float, int]] = [(0.0, p) for p in starts]
        heapq.heapify(pq)
        # Collect candidate targets in ascending min-climb order (do NOT stop at the first): the nearest
        # good hub may fail the R3 backclimb gate below (its real OSM arc dips then rises), while a
        # slightly farther one descends cleanly. Try each until one yields runs that all pass.
        candidates: list[int] = []
        while pq and len(candidates) < 8:
            c, x = heapq.heappop(pq)
            if c > best.get(x, 1e18):
                continue
            hub_ok = pn_hub[x] in good and (
                elev[pn_hub[x]] > elev[sink] if want_higher else elev[pn_hub[x]] < elev[sink]
            )
            if hub_ok:  # `good ⊆ node_points`, so elev[..] is present
                candidates.append(x)
                continue  # record, but keep walking for other reachable good hubs
            for y, si in padj[x]:
                # cost = movement AGAINST the desired direction (down when climbing a feeder up, else up)
                against = max(0.0, pn_elev[x] - pn_elev[y]) if want_higher else max(0.0, pn_elev[y] - pn_elev[x])
                if against > OSMConfig.NODE_TERRAIN_TOL_M:
                    continue
                nc = c + against
                if nc < best.get(y, 1e18):
                    best[y] = nc
                    par[y] = (x, si)
                    heapq.heappush(pq, (nc, y))
        for target in candidates:
            if self._try_add_sink_run(target, par, graph, segments, hubs, pn_hub, pn_xy, padj):
                return True
        return False

    def _try_add_sink_run(
        self,
        target: int,
        par: dict[int, tuple[int, int]],
        graph: ImportGraph,
        segments: list[LineString],
        hubs: list[XY],
        pn_hub: list[int],
        pn_xy: list[XY],
        padj: dict[int, list[tuple[int, int]]],
    ) -> bool:
        """Rebuild the pre-merge chain to `target`, split it at interior existing hubs, drape each
        sub-chain, and append every sub-run that passes the R3 backclimb gate. Returns True if any run
        was added (i.e. this target yields at least one clean descending run).
        """
        chain: list[int] = [target]
        n = target
        while n in par:
            n = par[n][0]
            chain.append(n)
        chain.reverse()
        # Split the chain at every intermediate pre-merge node whose hub is a DISTINCT existing graph
        # node — one run per hub-to-hub sub-chain. Shares existing nodes (no floating, R6) and yields
        # short pieces that collapse against existing runs (no long overlapping duplicate, R2).
        existing = set(graph.node_points)
        existing_pairs = {frozenset((r.node_a, r.node_b)) for r in graph.slope_runs}
        boundaries = [0]
        for idx in range(1, len(chain) - 1):
            h = pn_hub[chain[idx]]
            if h in existing and h != pn_hub[chain[boundaries[-1]]]:
                boundaries.append(idx)
        boundaries.append(len(chain) - 1)
        added = False
        for bi in range(len(boundaries) - 1):
            lo_i, hi_i = boundaries[bi], boundaries[bi + 1]
            ha, hb = pn_hub[chain[lo_i]], pn_hub[chain[hi_i]]
            if ha == hb or ha not in existing or hb not in existing or frozenset((ha, hb)) in existing_pairs:
                continue
            clean = self._concat_chain_geometry(chain[lo_i : hi_i + 1], hubs[ha], hubs[hb], segments, padj, pn_xy)
            if len(clean) < 2:
                continue
            pa, pb = graph.node_points[ha], graph.node_points[hb]
            pts = self._drape(clean, pa, pb)  # `clean` is already metre XY
            if pts is None or len(pts) < 2:
                continue
            self._snap_interior_to_source(pts)  # pull any bend-chord point back onto the real OSM piste (R19)
            if pts[0].elevation < pts[-1].elevation:
                pts = pts[::-1]
            na, nb = (ha, hb) if pts[0] is pa else (hb, ha)
            pts[0], pts[-1] = graph.node_points[na], graph.node_points[nb]
            if self._backclimb(pts) > OSMConfig.MAX_BACKCLIMB_M:
                continue  # never re-add a run that climbs (R3) — the pre-merge arc dips then rises
            if any(pts[k].distance_to(other=pts[k + 1]) > OSMConfig.MAX_STRAIGHT_M for k in range(len(pts) - 1)):
                continue  # a long straight chord would tunnel through terrain (R7)
            if any(pts[k].distance_to(other=pts[k + 1]) > OSMConfig.MAX_STRAIGHT_M for k in range(len(pts) - 1)):
                continue  # a long straight chord would tunnel through terrain (R7)
            logger.debug(f"[IMPORT] reconnect: sub-run {na}→{nb}, {len(pts)} pts")
            graph.slope_runs.append(SlopeRun(points=pts, node_a=na, node_b=nb))
            added = True
        return added

    def _snap_interior_to_source(self, pts: list[PathPoint]) -> None:
        """Pull each INTERIOR point that strays > SLOPE_ON_SOURCE_TOL_M from every source piste back onto
        the nearest one (project + re-sample DEM z), in place. Removes bend-chord bows the drape leaves
        between coarse concatenated vertices, keeping the run on real OSM geometry (R19). Endpoints (hubs)
        untouched. No-op when there are no source lines.
        """
        if not self._source_lines:
            return
        tol = OSMConfig.SLOPE_ON_SOURCE_TOL_M
        for i in range(1, len(pts) - 1):
            q = Point(self._to_m(pts[i].lon, pts[i].lat))
            nearest = min(self._source_lines, key=lambda ln: ln.distance(q))
            if nearest.distance(q) <= tol:
                continue
            proj = nearest.interpolate(nearest.project(q))
            lon, lat = self._to_deg(proj.x, proj.y)
            elev = self.dem.get_elevation(lon=lon, lat=lat)
            if elev is not None:
                pts[i] = PathPoint(lon=lon, lat=lat, elevation=elev)

    def _prune_dead_end_slopes(self, graph: ImportGraph) -> None:
        """R22 (frozen, no bbox-edge exception): every slope endpoint must connect onward — be a vertex
        of another segment or a lift. Iteratively drop any slope with a degree-1 non-lift endpoint (a
        dead-end); pruning one can expose another, so repeat to a fixpoint.
        """
        while True:
            deg: dict[int, int] = defaultdict(int)
            for r in graph.slope_runs:
                deg[r.node_a] += 1
                deg[r.node_b] += 1
            lift_nodes = {n for lf in graph.lifts for n in (lf.node_a, lf.node_b)}
            keep = [
                r
                for r in graph.slope_runs
                if not any(deg[n] == 1 and n not in lift_nodes for n in (r.node_a, r.node_b))
            ]
            if len(keep) == len(graph.slope_runs):
                break
            graph.dropped_isolated += len(graph.slope_runs) - len(keep)
            graph.slope_runs = keep

    def _build_slope_run(
        self, seg: LineString, a: int, b: int, graph: ImportGraph, source_union: MultiLineString | None
    ) -> SlopeRun | None:
        """Materialize one kept segment into a downhill SlopeRun, or None if it must be dropped.

        Pulls each end to its hub by tier (see _connector), DEM-drapes, orients downhill, and enforces
        the pull/fidelity model + no-uphill. Bumps graph.dropped_* counters on a drop.
        """
        if a not in graph.node_points or b not in graph.node_points:
            return None
        pa, pb = graph.node_points[a], graph.node_points[b]
        # Trim ~TRIM_END_M off each end of the OSM body (if long enough), then connect the trimmed body
        # straight to the hub. On the first/last stretch, matching the hub's exact height + direction
        # matters more than terrain fidelity. A short segment is kept untrimmed (never dropped for length).
        seg = self._trim_ends(seg, OSMConfig.TRIM_END_M)
        body = [self._to_deg(x, y) for x, y in seg.coords]  # (lon,lat) on the real piste (ends trimmed)
        head = self._connector(pa, body[0])  # hub pa -> trimmed piste start
        tail_c = self._connector(pb, body[-1])  # hub pb -> trimmed piste end
        if head is None or tail_c is None:  # a pull exceeded MAX_PULL — discard the segment
            graph.dropped_isolated += 1
            return None
        tail = [(p[0], p[1]) for p in reversed(tail_c)]
        # Include the hub coords as the true first/last points so the drape densifies the hub→connector
        # legs too (no long straight jump when the endpoint is later pinned).
        latlon = [(pa.lon, pa.lat)] + head + body + tail + [(pb.lon, pb.lat)]
        pts = self._drape([self._to_m(lon, lat) for lon, lat in latlon], pa, pb)
        if pts is None or len(pts) < 2:
            return None
        if pts[0].elevation < pts[-1].elevation:  # orient downhill
            pts = pts[::-1]
            a, b = b, a
        pts[0] = graph.node_points[a]
        pts[-1] = graph.node_points[b]
        # Enforce the pull/fidelity model on the FINAL geometry: OSM body on-piste, off-piste points only
        # a contiguous END connector ≤ MAX_PULL_M (no mid-run tunnel). Wrong slope worse than missing.
        if not self._valid_pull_shape(pts, source_union):
            graph.dropped_isolated += 1
            return None
        # Match R12 exactly: ≤15% of points may lie >45 m from the nearest source-piste VERTEX. A
        # connector that pushes a run past this is invented geometry — drop it rather than keep off-piste.
        if self._vtree is not None:
            off = 0
            for p in pts:
                q = Point(self._to_m(p.lon, p.lat))
                nearest = self._piste_vertices[self._vtree.nearest(q)]
                if q.distance(nearest) > 45.0:
                    off += 1
            if off / len(pts) > 0.15:
                graph.dropped_isolated += 1
                return None
        # A real piste descends monotonically; the run keeps its real DEM elevations. DEM point-sampling
        # noise on a genuinely-descending piste is absorbed by the smoothed backclimb gate — a run whose
        # SMOOTHED profile still climbs > MAX_BACKCLIMB_M is a real uphill and is dropped (no elevation faked).
        if self._backclimb(pts) > OSMConfig.MAX_BACKCLIMB_M:
            graph.dropped_uphill += 1
            return None
        return SlopeRun(points=pts, node_a=a, node_b=b)

    def _drop_parallel_twins(self, graph: ImportGraph) -> None:
        """R34: drop a run that is a redundant parallel TWIN of a longer same-named sibling — hugging it
        within the near band (DEDUP_TOL..PARALLEL_TOL) for ≥ PARALLEL_TWIN_FRAC of its own length (one
        wide piste double-drawn as two offset ribbons). Mirrors R34's detection exactly. Reachability-
        guarded: a twin whose drop would strand a node from a lift is kept. Iterated to a fixpoint.
        """
        lo, hi, frac = OSMConfig.DEDUP_TOL_M, OSMConfig.PARALLEL_TOL_M, OSMConfig.PARALLEL_TWIN_FRAC
        keep_unsafe: set[int] = set()  # run identities that would strand a node if dropped — leave them
        while True:
            runs = graph.slope_runs
            pm = [[Point(self._to_m(p.lon, p.lat)) for p in r.points] for r in runs]
            plen = [self._polylen_m(r.points) for r in runs]
            victim = self._find_parallel_twin(runs, pm, plen, lo, hi, frac, keep_unsafe)
            if victim is None:
                return
            cand = ImportGraph(
                node_points=graph.node_points, slope_runs=runs[:victim] + runs[victim + 1 :], lifts=graph.lifts
            )
            if self._newly_stranded(cand):
                keep_unsafe.add(id(runs[victim]))  # dropping would strand a skier (R22 wins over R34)
                continue
            logger.debug(f"[IMPORT] drop redundant parallel twin '{runs[victim].name}'")
            graph.slope_runs = cand.slope_runs
            graph.dropped_isolated += 1

    @staticmethod
    def _find_parallel_twin(
        runs: list[SlopeRun],
        pm: list[list[Point]],
        plen: list[float],
        lo: float,
        hi: float,
        frac: float,
        keep_unsafe: set[int],
    ) -> int | None:
        """Index of a run that is a redundant parallel twin of a longer same-named sibling, else None."""
        for i in range(len(runs)):
            if id(runs[i]) in keep_unsafe or plen[i] == 0 or not runs[i].name:
                continue
            for j in range(len(runs)):
                if i == j or len(pm[j]) < 2 or runs[i].name != runs[j].name or plen[i] > plen[j]:
                    continue
                best = cur = 0.0  # longest contiguous stretch of i inside j's near band
                for k in range(len(pm[i])):
                    d = min(pm[i][k].distance(q) for q in pm[j])
                    if lo < d <= hi:
                        cur += pm[i][k - 1].distance(pm[i][k]) if k > 0 else 0.0
                        best = max(best, cur)
                    else:
                        cur = 0.0
                if best >= frac * plen[i]:
                    return i
        return None

    def _dedup_final_runs(self, graph: ImportGraph) -> None:
        """R2: drop a run whose vertices are ≥DEDUP_COVER_FRAC within DEDUP_TOL_M of a LONGER run — a
        near-duplicate double-draw that survived the per-hub-pair dedup because it sits on a DIFFERENT
        hub-pair. Mirrors the R2 test predicate exactly (vertex-to-vertex distance, shared-hub pull
        exemption ≤ MAX_PULL_M). Reachability-guarded like the twin drop; iterated to a fixpoint.
        """
        tol, cover = OSMConfig.DEDUP_TOL_M, OSMConfig.DEDUP_COVER_FRAC
        keep_unsafe: set[int] = set()
        while True:
            runs = graph.slope_runs
            pm = [[Point(self._to_m(p.lon, p.lat)) for p in r.points] for r in runs]
            plen = [self._polylen_m(r.points) for r in runs]
            victim = self._find_covered_run(runs, pm, plen, tol, cover, keep_unsafe)
            if victim is None:
                return
            cand = ImportGraph(
                node_points=graph.node_points, slope_runs=runs[:victim] + runs[victim + 1 :], lifts=graph.lifts
            )
            if self._newly_stranded(cand):
                keep_unsafe.add(id(runs[victim]))  # a load-bearing coverage-dup — keep it (R22 wins)
                continue
            logger.debug(f"[IMPORT] drop near-duplicate run '{runs[victim].name}'")
            graph.slope_runs = cand.slope_runs
            graph.dropped_isolated += 1

    def _find_covered_run(
        self,
        runs: list[SlopeRun],
        pm: list[list[Point]],
        plen: list[float],
        tol: float,
        cover: float,
        keep_unsafe: set[int],
    ) -> int | None:
        """Index of a run i ≥`cover`-covered (vertex-within-`tol`) by a longer run j, else None. Skips a
        short shared-hub coincident prefix ≤ MAX_PULL_M (a mere pull artifact, not a double-draw).
        """
        max_pull = OSMConfig.MAX_PULL_M
        for i in range(len(runs)):
            if id(runs[i]) in keep_unsafe or len(pm[i]) < 2:
                continue
            for j in range(len(runs)):
                if i == j or len(pm[j]) < 2 or plen[i] > plen[j]:
                    continue
                shared = {runs[i].node_a, runs[i].node_b} & {runs[j].node_a, runs[j].node_b}
                if shared:
                    seq = pm[i] if runs[i].node_a in shared else pm[i][::-1]
                    coincident = 0.0
                    for k in range(1, len(seq)):
                        if min(seq[k].distance(vb) for vb in pm[j]) > tol:
                            break
                        coincident += seq[k - 1].distance(seq[k])
                    if coincident <= max_pull:
                        continue  # short coincidence at a shared hub → pull artifact, not a duplicate
                near = sum(1 for va in pm[i] if any(va.distance(vb) < tol for vb in pm[j]))
                if near / len(pm[i]) >= cover:
                    return i
        return None

    def _newly_stranded(self, after: ImportGraph) -> bool:
        """True if `after` leaves a slope node stranded, guarding BOTH R22 clauses so a drop never orphans
        a node: (a) DOWN — a node above the lowest lift base that cannot reach a lift descending; or
        (b) UP — a node that cannot be reached FROM a lift descending (an orphaned top whose feeder was
        dropped). Used to veto a twin/duplicate drop that would strand a skier.
        """
        elev = self._node_elevations(after)
        lift_nodes = {n for lf in after.lifts for n in (lf.node_a, lf.node_b)}
        if not lift_nodes:
            return False
        min_base = min(elev[n] for n in lift_nodes)
        good_down = self._hubs_reaching_lift(after, lift_nodes)  # nodes that reach a lift going DOWN
        reachable = self._skier_reachable(after, elev)  # nodes reachable FROM a lift (down slopes + up lifts)
        after_nodes = {n for r in after.slope_runs for n in (r.node_a, r.node_b)}
        stranded_down = any(n not in good_down and n not in lift_nodes and elev[n] > min_base for n in after_nodes)
        stranded_up = any(n not in reachable for n in after_nodes)
        return stranded_down or stranded_up

    def _drop_slopes_floating_past_hubs(self, graph: ImportGraph) -> None:
        """R6: drop any slope that passes within MIN_NODE_DIST_M of a hub that is NOT one of its own
        endpoints — such a slope should have split there but couldn't. A wrong slope is worse than a
        floating-node violation, so it is removed rather than left floating.
        """
        tol = OSMConfig.MIN_NODE_DIST_M
        hub_m = {k: self._to_m(v.lon, v.lat) for k, v in graph.node_points.items()}
        elev = self._node_elevations(graph)
        keep: list[SlopeRun] = []
        for r in graph.slope_runs:
            line = LineString([self._to_m(p.lon, p.lat) for p in r.points])
            hi_end = max(elev[r.node_a], elev[r.node_b])
            # a run legitimately passes BELOW a nearby higher peak — floating past a hub higher than both
            # its ends is not a missed split (it could not share that node without climbing). Ignore those.
            floats = any(
                h not in (r.node_a, r.node_b) and elev[h] <= hi_end and line.distance(Point(pt)) < tol
                for h, pt in hub_m.items()
            )
            if floats:
                graph.dropped_isolated += 1
            else:
                keep.append(r)
        graph.slope_runs = keep

    @staticmethod
    def _drop_isolated_slopes(graph: ImportGraph) -> None:
        """Drop slope runs whose component (over the FINAL kept slopes + lifts) contains no lift. Runs
        after pull-shape drops so a segment orphaned by an earlier drop is removed too (keeps R4 strict).
        """
        while True:
            adj: dict[int, set[int]] = defaultdict(set)
            for r in graph.slope_runs:
                adj[r.node_a].add(r.node_b)
                adj[r.node_b].add(r.node_a)
            for lf in graph.lifts:
                adj[lf.node_a].add(lf.node_b)
                adj[lf.node_b].add(lf.node_a)
            comp: dict[int, int] = {}
            cid = 0
            for start in list(adj):
                if start in comp:
                    continue
                q = deque([start])
                comp[start] = cid
                while q:
                    x = q.popleft()
                    for y in adj[x]:
                        if y not in comp:
                            comp[y] = cid
                            q.append(y)
                cid += 1
            lift_comps = {comp[lf.node_a] for lf in graph.lifts} | {comp[lf.node_b] for lf in graph.lifts}
            keep = [r for r in graph.slope_runs if comp.get(r.node_a) in lift_comps or comp.get(r.node_b) in lift_comps]
            if len(keep) == len(graph.slope_runs):
                break
            graph.dropped_isolated += len(graph.slope_runs) - len(keep)
            graph.slope_runs = keep
        # drop now-unused slope-only nodes (keep lift stations)
        used = {n for r in graph.slope_runs for n in (r.node_a, r.node_b)} | {
            n for lf in graph.lifts for n in (lf.node_a, lf.node_b)
        }
        graph.node_points = {k: v for k, v in graph.node_points.items() if k in used}

    def _connector(self, frm: PathPoint, to_lonlat: Vertex) -> list[Vertex] | None:
        """Interior (lon,lat) points pulling hub `frm` to the on-piste body point `to_lonlat`:

          - a shared source piste (substring) when hub + body sit on ONE piste and the arc is short — the
            connector stays on the real piste.
          - else a STRAIGHT densified pull, up to MAX_PULL_M. (The custom-path fan is slow AND loops
            without tuning, so a straight line is the robust choice.)
          - gap > MAX_PULL_M (300 m): return None → the caller DISCARDS the segment (a straight pull that
            long no longer credibly follows terrain).

        Endpoints excluded (caller adds them). Returns [] when no connector points are needed.
        """
        h_m, b_m = self._to_m(frm.lon, frm.lat), self._to_m(*to_lonlat)
        gap = math.dist(h_m, b_m)
        if gap < 1.0:
            return []
        if gap > OSMConfig.MAX_PULL_M:  # > 300 m: a straight pull no longer credibly follows terrain → drop
            return None
        # A shared source piste, if hub and body sit on one and the along-arc is short (≤1.5× the straight
        # gap), keeps the connector ON the real piste — preferred over a straight chord.
        along = self._piste_substring(h_m, b_m)
        if along is not None:
            arc = sum(math.dist(along[i], along[i + 1]) for i in range(len(along) - 1)) if len(along) > 1 else 0.0
            arc += (math.dist(h_m, along[0]) + math.dist(along[-1], b_m)) if along else gap
            if arc <= 1.5 * gap:
                return [self._to_deg(x, y) for x, y in along]
        # Otherwise a STRAIGHT densified pull. The custom-path fan was tried but is slow AND loop-prone
        # without per-connector tuning; a straight line up to MAX_PULL_M is the honest, robust choice.
        n = max(1, int(gap // OSMConfig.RESAMPLE_STEP_M))
        return [
            (frm.lon + (to_lonlat[0] - frm.lon) * k / n, frm.lat + (to_lonlat[1] - frm.lat) * k / n)
            for k in range(1, n)
        ]

    def _piste_substring(self, h_m: XY, b_m: XY) -> list[XY] | None:
        """If hub and body point both lie within tol of ONE source piste, return the along-piste
        substring between them (metres, endpoints excluded); else None.
        """
        tol = OSMConfig.SLOPE_ON_SOURCE_TOL_M + 15.0
        hp, bp = Point(h_m), Point(b_m)
        best: LineString | None = None
        for ln in self._source_lines:
            if ln.distance(hp) < tol and ln.distance(bp) < tol:
                d0, d1 = ln.project(hp), ln.project(bp)
                sub = substring(ln, min(d0, d1), max(d0, d1))
                if isinstance(sub, LineString) and not sub.is_empty and (best is None or sub.length < best.length):
                    best = sub
        if best is None:
            return None
        coords = list(best.coords)
        return [(x, y) for x, y in coords[1:-1]] if len(coords) > 2 else []

    def _split_at_interior_hubs(
        self, kept: list[tuple[LineString, int, int]], hubs: list[XY], used: set[int]
    ) -> list[tuple[LineString, int, int]]:
        """Split each segment at every hub that projects onto its INTERIOR within MIN_NODE_DIST_M, so a
        slope passes through (shares) that node instead of floating beside it (R6). The end hubs a/b are
        kept as the outer cut points; interior cuts use the hub whose projection is on the segment.
        """
        onseg_tol = OSMConfig.MIN_NODE_DIST_M  # R6 gate: split where a hub is within this of the segment
        hub_pts = {h: Point(hubs[h]) for h in used}
        hub_elev: dict[int, float] = {}
        for h in used:
            lon, lat = self._to_deg(*hubs[h])
            e = self.dem.get_elevation(lon=lon, lat=lat)
            hub_elev[h] = e if e is not None else -1e9
        out: list[tuple[LineString, int, int]] = []
        for seg, a, b in kept:
            length = seg.length
            hi_end = max(hub_elev[a], hub_elev[b])  # the run's higher endpoint
            cuts: list[tuple[float, int]] = [(0.0, a), (length, b)]
            for h, hp in hub_pts.items():
                if h in (a, b):
                    continue
                # never cut a run at an interior hub HIGHER than both its ends — a descending run passes
                # legitimately BELOW a nearby peak (Greitspitz summit beside Lange Wandbahn's descent);
                # cutting there would splice in an uphill leg and invert the run.
                if hub_elev[h] > hi_end:
                    continue
                d_along = seg.project(hp)
                if 1.0 < d_along < length - 1.0 and seg.distance(hp) < onseg_tol:
                    cuts.append((d_along, h))
            if len(cuts) == 2:
                out.append((seg, a, b))
                continue
            cuts.sort()
            # dedupe near-coincident cut positions (~5 m); ensure the true end hub is last
            dedup: list[tuple[float, int]] = [cuts[0]]
            for d, h in cuts[1:]:
                if d - dedup[-1][0] >= 5.0:
                    dedup.append((d, h))
            if dedup[-1][1] != b:
                dedup[-1] = (length, b)
            for (d0, h0), (d1, h1) in zip(dedup, dedup[1:], strict=False):
                if h0 == h1 or d1 - d0 < 5.0:
                    continue
                piece = substring(seg, d0, d1)
                if isinstance(piece, LineString) and not piece.is_empty and len(piece.coords) >= 2:
                    out.append((piece, h0, h1))
        return out

    @staticmethod
    def _trim_ends(seg: LineString, trim_m: float) -> LineString:
        """Trim `trim_m` off both ends of a segment (its terrain-following middle survives; ends become a
        clean straight connector to the hub). A segment too short to trim is returned UNCHANGED (never
        dropped for length — dropping severs descent chains).
        """
        if trim_m <= 0 or seg.length <= 2 * trim_m + 5.0:
            return seg
        piece = substring(seg, trim_m, seg.length - trim_m)
        if not isinstance(piece, LineString) or piece.is_empty or len(piece.coords) < 2:
            return seg
        return piece

    def _valid_pull_shape(self, pts: list[PathPoint], source_union: MultiLineString | None) -> bool:
        """True if the run obeys the pull model: OSM body on-piste, off-piste points only in a contiguous
        END connector (never mid-run = tunnel), each connector ≤ MAX_PULL_M. Mirrors the R19 test EXACTLY
        (same PISTE_TOL_M = 40 m off-piste threshold and same MAX_PULL_M) so the builder never drops a
        run the test would accept, nor keeps one it would reject.
        """
        if source_union is None:
            return True
        tol = OSMConfig.PISTE_TOL_M
        max_pull = OSMConfig.MAX_PULL_M
        off = [source_union.distance(Point(self._to_m(p.lon, p.lat))) > tol for p in pts]
        if not any(off):
            return True
        first_on = off.index(False) if False in off else len(off)
        last_on = len(off) - 1 - off[::-1].index(False) if False in off else -1
        if any(off[first_on : last_on + 1]):  # off-piste point between the on-piste body → tunnel
            return False
        head = self._polylen_m(pts[: first_on + 1]) if first_on > 0 else 0.0
        tail = self._polylen_m(pts[last_on:]) if last_on < len(off) - 1 else 0.0
        return max(head, tail) <= max_pull

    @staticmethod
    def _polylen_m(pts: list[PathPoint]) -> float:
        """Length (m) of a PathPoint polyline."""
        return sum(pts[i].distance_to(other=pts[i + 1]) for i in range(len(pts) - 1))

    @staticmethod
    def _on_source(piece: LineString, source_union: MultiLineString | None) -> bool:
        """True if ≥85% of a piece lies within SLOPE_ON_SOURCE_TOL_M of a source OSM piste."""
        if source_union is None:
            return True
        tol = OSMConfig.SLOPE_ON_SOURCE_TOL_M
        n = max(2, int(piece.length // 15))
        off = sum(1 for k in range(n + 1) if source_union.distance(piece.interpolate(k / n, normalized=True)) > tol)
        return off / (n + 1) <= 0.15

    @staticmethod
    def _components(n_nodes: int, adj: dict[int, set[int]]) -> dict[int, int]:
        """Connected-component id per node index (0..n_nodes-1) over adjacency `adj`."""
        comp: dict[int, int] = {}
        cid = 0
        for start in range(n_nodes):
            if start in comp:
                continue
            q = deque([start])
            comp[start] = cid
            while q:
                x = q.popleft()
                for y in adj[x]:
                    if y not in comp:
                        comp[y] = cid
                        q.append(y)
            cid += 1
        return comp

    def _drape(self, coords: list[XY], start: PathPoint, end: PathPoint) -> list[PathPoint] | None:
        """DEM-drape a segment, keeping every original vertex plus uniform RESAMPLE_STEP_M infill."""
        line = LineString(coords)
        total = line.length
        if total <= 0:
            return None
        vert_d = [0.0]
        for i in range(1, len(coords)):
            vert_d.append(vert_d[-1] + math.dist(coords[i - 1], coords[i]))
        step = OSMConfig.RESAMPLE_STEP_M
        n = max(1, int(round(total / step)))
        grid = [total * k / n for k in range(n + 1)]
        merged = sorted(set(vert_d) | set(grid))
        dists = [merged[0]]
        for d in merged[1:]:
            if d - dists[-1] >= 1.0:
                dists.append(d)
        if dists[-1] < total - 1e-6:
            dists.append(total)
        out: list[PathPoint] = []
        for d in dists:
            p = line.interpolate(d)
            lon, lat = self._to_deg(p.x, p.y)
            elev = self.dem.get_elevation(lon=lon, lat=lat)
            if elev is None:
                return None
            out.append(PathPoint(lon=lon, lat=lat, elevation=elev))
        out[0] = start
        out[-1] = end
        return out

    def _backclimb(self, pts: list[PathPoint]) -> float:
        """ULTRA-STRICT uphill metric (R3): the largest elevation RISE over any BACKCLIMB_WINDOW_M span
        of the descending-oriented run. A real piste must go strictly DOWN — over every ~80 m window the
        far end must sit no higher than the near end. Returns the worst window's rise (0 for a clean
        descent); raw DEM elevations, no smoothing/clamping (nothing faked). Matches the R3 test exactly.
        """
        es = [p.elevation for p in pts]
        if len(es) < 2:
            return 0.0
        if es[0] < es[-1]:
            es = es[::-1]
            pts = pts[::-1]
        cum = [0.0]
        for i in range(1, len(pts)):
            cum.append(cum[-1] + pts[i - 1].distance_to(other=pts[i]))
        win = OSMConfig.BACKCLIMB_WINDOW_M
        worst = 0.0
        for i in range(len(es)):
            j = i + 1
            while j < len(es) and cum[j] - cum[i] < win:
                j += 1
            j = min(j, len(es) - 1)  # clamp; the final short tail still checked against its window start
            worst = max(worst, es[j] - es[i])  # oriented downhill → a positive value is an uphill span
        return worst


class GraphImporter(BaseOSMImporter):
    """Imports lifts + slopes via the connected-graph algorithm: the lifts and slopes are reported
    as OSMGraphBuilder preprocessed them (hub-merged, descent-carved, grouped). The graph's own
    hub-aligned lifts are used (not raw OSM), so slopes and lift stations share nodes.
    """

    def _assemble(self, elements: list[OverpassElement], on_progress: ProgressFn) -> ImportResult:
        pistes, lifts = ways_to_lines(elements, self.bbox)
        self._graph = OSMGraphBuilder(dem=self.dem, bbox=self.bbox).build(pistes, lifts, on_progress=on_progress)
        logger.info(
            f"OSM graph import: {len(self._graph.slope_runs)} runs, {len(self._graph.lifts)} lifts, "
            f"{len(self._graph.slopes)} slopes"
        )
        return ImportResult(
            lifts=[(lf.bottom, lf.top, lf.lift_type, lf.name) for lf in self._graph.lifts],
            slope_chains=self._graph.to_slope_chains(),
            source=self.SOURCE,
        )

    def _dump(self, elements: list[OverpassElement], dump_dir: Path) -> None:
        """Raw fetch (base) + the built-graph PNG for reference (never read back). Writes directly;
        skips the PNG only for an empty graph (nothing to plot — render_png would raise on it).
        """
        super()._dump(elements, dump_dir)
        if not self._graph.node_points:
            logger.debug("OSM graph import: empty graph, no reference PNG")
            return
        out = dump_dir / "osm_import.png"
        render_png(self._graph, out)
        logger.debug(f"OSM graph import: wrote reference PNG to {out}")

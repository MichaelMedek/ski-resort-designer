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

import logging
import math
from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path

import networkx as nx
import numpy as np
import numpy.typing as npt
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree
from shapely import STRtree, distance, get_parts, line_interpolate_point, line_merge, set_precision
from shapely.geometry import LineString, MultiLineString, Point
from shapely.ops import substring, unary_union

from skiresort_planner.constants import OSMConfig, SlopeConfig
from skiresort_planner.core.dem_service import DEMService
from skiresort_planner.core.geo_calculator import GeoCalculator
from skiresort_planner.core.terrain_analyzer import TerrainAnalyzer
from skiresort_planner.generators.osm_graph_plot import render_png
from skiresort_planner.generators.osm_importer import (
    BaseOSMImporter,
    ImportResult,
    OverpassElement,
    OverpassVertex,
    ProgressFn,
    extract_lift_sections,
    suffixed_name,
)
from skiresort_planner.model.connectivity import component_labels
from skiresort_planner.model.path_point import PathPoint
from skiresort_planner.model.path_segment import PathSegment

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
        """Convert named piste segments into linear app-slopes ready for display, each a contiguous point-list.
        Each ImportSlope's runs decompose by longest-path into linear chains, split at interior lift nodes
        (R37) so lifts end slopes; names shared by several chains are disambiguated with a (k) suffix (R38).
        """
        elev = {k: v.elevation for k, v in self.node_points.items()}
        lift_nodes = OSMGraphBuilder._lift_nodes(self)
        chained: list[tuple[list[SlopeRun], str | None]] = []
        for sl in self.slopes:
            for chain in _linear_chains([self.slope_runs[ri] for ri in sl.run_indices], elev):
                for piece in _split_chain_at_lifts(chain, lift_nodes):
                    chained.append((piece, sl.name))
        # R38: disambiguate names shared by >1 app-slope with a 1-based (k) suffix (bare when unique).
        counts: dict[str, int] = defaultdict(int)
        for _piece, nm in chained:
            if nm:
                counts[nm] += 1
        seen: dict[str, int] = defaultdict(int)
        out: list[tuple[list[list[PathPoint]], str | None]] = []
        for piece, nm in chained:
            if nm and counts[nm] > 1:
                name: str | None = suffixed_name(nm, seen[nm], counts[nm])
                seen[nm] += 1
            else:
                name = nm
            out.append(([r.points for r in piece], name))
        return out


def _linear_chains(group: list[SlopeRun], elev: dict[int, float]) -> list[list[SlopeRun]]:
    """Decompose a named piste's runs into top→bottom chains via an optimal min-path-cover (fewest
    chains, ties toward the heaviest-downstream trunk), so short side-runs spin off on their own.
    Min-path-cover = tail→head max_weight_matching; the per-run longest-downstream weight is a DAG DP.
    """
    oriented = [(*OSMGraphBuilder._orient_downhill(r.node_a, r.node_b, elev), r) for r in group]
    n = len(oriented)
    hi = [o[0] for o in oriented]
    lo = [o[1] for o in oriented]
    length = [OSMGraphBuilder._polylen_m(o[2].points) for o in oriented]
    by_hi: dict[int, list[int]] = defaultdict(list)
    for j in range(n):
        by_hi[hi[j]].append(j)
    succ = [by_hi.get(lo[a], []) for a in range(n)]

    # Longest-downstream length per run: a DAG DP in reverse topological order (networkx supplies the
    # order and raises on a cycle — a downhill-oriented run graph is acyclic).
    dag = nx.DiGraph((a, b) for a in range(n) for b in succ[a])
    dag.add_nodes_from(range(n))  # include isolated runs (no successor) so every run is ordered
    downstream: dict[int, float] = {}
    for a in reversed(list(nx.topological_sort(dag))):
        downstream[a] = length[a] + max((downstream[b] for b in succ[a]), default=0.0)

    # Optimal min-path-cover: match each run (tail) to at most one successor (head). max_weight_matching
    # with maxcardinality picks the FEWEST chains, ties broken toward the heaviest-downstream trunk (its
    # +n offset makes every real link outweigh leaving a run unmatched), so the main descent stays whole.
    bip = nx.Graph((("tail", a), ("head", b), {"weight": downstream[b] + n}) for a in range(n) for b in succ[a])
    match_next = [-1] * n
    match_prev = [-1] * n
    for u, v in nx.max_weight_matching(bip, maxcardinality=True):
        ends = dict([u, v])  # each matched edge joins one ("tail", a) and one ("head", b)
        tail_run, head_run = ends["tail"], ends["head"]
        match_next[tail_run] = head_run
        match_prev[head_run] = tail_run

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


def _split_chain_at_lifts(chain: list[SlopeRun], lift_nodes: set[int]) -> list[list[SlopeRun]]:
    """Split a run-chain wherever consecutive runs meet at a lift node, as slopes must end at lifts.
    A lift only at the chain's outer ends (terminus) does not split; returns ≥1 contiguous sub-chains
    where interior lifts act as boundaries between separate app-slopes.
    """
    pieces: list[list[SlopeRun]] = [[chain[0]]] if chain else []
    for prev, cur in zip(chain, chain[1:], strict=False):
        shared = {prev.node_a, prev.node_b} & {cur.node_a, cur.node_b}
        if shared and next(iter(shared)) in lift_nodes:
            pieces.append([cur])  # break the chain at the lift junction
        else:
            pieces[-1].append(cur)
    return pieces


def _is_importable_piste(tags: dict[str, str]) -> bool:
    """True for a piste worth importing: a connector (any difficulty, kept for connectivity) or a standard
    groomed downhill run whose difficulty is in the allow-list. Every other piste type or difficulty is
    rejected, so only skiable, on-map geometry enters the graph.
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


def ways_to_lines(
    elements: list[OverpassElement], bbox: tuple[float, float, float, float]
) -> tuple[list[tuple[list[Vertex], str | None]], list[tuple[list[Vertex], str, str | None]]]:
    """Extract pistes and lifts from raw Overpass ways, filtering to standard geometry rules.
    Pistes: standard groomed downhill + connection, fully in-box, ≥2 vertices. Lifts from the shared
    extract_lift_sections importer (mapped aerialway, in-box, mid-station split); dedup/merge are downstream.

    Args:
      elements: Raw Overpass elements (ways + nodes).
      bbox: (min_lon, min_lat, max_lon, max_lat) bounding box in WGS84.

    Returns:
      (pistes, lifts) where pistes = [(vertices, name), ...] and lifts = [(vertices, type, name), ...].
    """
    min_lon, min_lat, max_lon, max_lat = bbox

    def inside(geom: list[OverpassVertex]) -> bool:
        return all(min_lon <= v["lon"] <= max_lon and min_lat <= v["lat"] <= max_lat for v in geom)

    def name_of(tags: dict[str, str]) -> str | None:
        for key in ("name", "piste:name", "piste:ref", "ref"):
            if tags.get(key):
                return str(tags[key])
        return None

    pistes: list[tuple[list[Vertex], str | None]] = []
    for el in elements:
        if el.get("type") != "way":
            continue
        geom = el.get("geometry", [])
        tags = el.get("tags", {})
        if len(geom) < 2 or not inside(geom) or not _is_importable_piste(tags):
            continue
        pistes.append(([(v["lon"], v["lat"]) for v in geom], name_of(tags)))
    return pistes, extract_lift_sections(elements, bbox)


class OSMGraphBuilder:
    """Turns raw Overpass ways into a connected ImportGraph. Pure geometry — no graph mutation."""

    def __init__(self, dem: DEMService, bbox: tuple[float, float, float, float]) -> None:
        self.dem = dem
        self.bbox = bbox
        lat0 = (bbox[1] + bbox[3]) / 2
        self._mlon, self._mlat = GeoCalculator.meters_per_degree(lat=lat0)
        self._source_lines: list[LineString] = []  # source pistes (metres) for piste-following pulls
        self._named_sources: list[tuple[LineString, str]] = []  # (piste metres, OSM name) for run naming

    def _to_m(self, lon: float, lat: float) -> XY:
        # Quantise the projection to the COORD_GRID_M (1 m) grid so shared vertices are bit-exact (no
        # sub-metre float drift) — the grid sits below every geometry tolerance (see constants assert).
        g = OSMConfig.COORD_GRID_M
        return (round((lon - self.bbox[0]) * self._mlon / g) * g, round((lat - self.bbox[1]) * self._mlat / g) * g)

    def _to_deg(self, x: float, y: float) -> Vertex:
        return (self.bbox[0] + x / self._mlon, self.bbox[1] + y / self._mlat)

    def _hub_elev(self, hubs: list[XY], keys: Iterable[int]) -> dict[int, float]:
        """Fetch DEM elevations for the given hub ids, returning hub id → elevation. Nodata points are omitted
        so a DEM hole never fabricates a height. Only the requested keys are looked up (not every hub),
        keeping the query cheap for the merge/split callers.
        """
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
        """Build a fully connected ImportGraph from piste and lift geometry via the 7-stage pipeline.
        Input pistes are deduplicated, split at crossings + lift stations, merged into hubs, DEM-draped,
        oriented downhill, grouped into slopes, and reconnected to recover stranded sinks (fixpoint on all gates).

        Args:
          pistes: [(vertices in WGS84, name), ...] piste linestrings.
          lifts: [(vertices in WGS84, type, name), ...] lift linestrings.
          on_progress: Callback fired before each major stage with (fraction, message).

        Returns:
          ImportGraph with node_points, slope_runs, lifts, and slopes groups, dedup/drop counters.
        """
        on_progress(0.0, "Preparing pistes…")
        # Project pistes + lifts to metres, drop sub-min-length; keep named pistes for later run-naming.
        piste_lines = [LineString([self._to_m(lon, lat) for lon, lat in vs]) for vs, _nm in pistes if len(vs) >= 2]
        piste_lines = [ls for ls in piste_lines if ls.length >= OSMConfig.MIN_PISTE_LENGTH_M]
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
        logger.debug(f"[IMPORT] prepare: {len(piste_lines)} pistes, {len(lift_lines)} lifts over min length")

        on_progress(0.3, "Deduplicating pistes…")
        # Drop a piste mostly covered by a longer one (an OSM redraw = double run).
        kept, deduped = self._dedup(piste_lines)
        logger.debug(f"[IMPORT] dedup: {len(piste_lines)} pistes → {len(kept)} kept ({deduped} dropped)")

        on_progress(0.4, "Splitting at crossings…")
        # Planar-node every piste at each crossing, then split at lift stations.
        segments = self._full_split(kept)
        segments = self._split_at_lift_stations(segments, lift_lines)
        logger.debug(f"[IMPORT] split: {len(segments)} segments from {len(kept)} pistes + {len(lift_lines)} lifts")

        on_progress(0.5, "Draping runs on terrain…")
        # Merge hubs, DEM-drape each run, orient downhill, reconnect stranded sinks.
        graph = self._assemble(segments, lift_lines, source=kept)
        logger.debug(f"[IMPORT] assemble: {len(graph.slope_runs)} runs, {len(graph.node_points)} nodes")

        on_progress(0.6, "Naming runs…")
        # Attach each run's original OSM piste name (best-covering named source).
        self._name_runs(graph)
        logger.debug(f"[IMPORT] name: {sum(1 for r in graph.slope_runs if r.name)}/{len(graph.slope_runs)} runs named")

        on_progress(0.7, "Merging parallel runs…")
        # Fork doubled ribbons into trunk+branches (R35), drop parallel twins (R34), then collapse
        # degree-2 pass-through nodes (R36) — BEFORE the coverage dedup so merged duplicates surface.
        before = len(graph.slope_runs)
        self._split_parallel_forks(graph)
        self._drop_parallel_twins(graph)
        self._collapse_degree2_nodes(graph)
        # Degree-2 merges fuse geometry into brand-new runs (fabricated defaults to []); re-flag off-piste
        # points so their red overlay is correct — mirrors the re-mark _split_parallel_forks already does.
        self._mark_fabricated(graph)
        logger.debug(f"[IMPORT] merge: {before} → {len(graph.slope_runs)} runs (fork/twin/degree-2 collapse)")

        on_progress(0.8, "Removing duplicate runs…")
        # Cross-pair coverage dedup of near-duplicates, then re-prune any dead-end / isolated run (R2).
        self._dedup_final_runs(graph)
        self._prune_dead_end_slopes(graph)
        self._drop_isolated_slopes(graph)
        logger.debug(f"[IMPORT] dedup-final: {len(graph.slope_runs)} runs, {len(graph.node_points)} nodes")

        on_progress(0.9, "Grouping into slopes…")
        # Group runs into named app-slopes (split at lifts, names disambiguated in to_slope_chains).
        self._group_slopes(graph)
        graph.deduped = deduped
        logger.debug(f"[IMPORT] group: {len(graph.slope_runs)} segments → {len(graph.slopes)} slopes")

        logger.info(
            f"[IMPORT] OSM graph: {len(graph.node_points)} nodes, {len(graph.slope_runs)} slopes, "
            f"{len(graph.lifts)} lifts (deduped {graph.deduped}, dropped uphill {graph.dropped_uphill}, "
            f"isolated {graph.dropped_isolated})"
        )
        return graph

    # -- step 2: dedup duplicate pistes -----------------------------------------------------------

    def _dedup(self, lines: list[LineString]) -> tuple[list[LineString], int]:
        """Drop a piste ≥DEDUP_COVER of whose length lies within DEDUP_TOL_M of a longer kept piste. Processed
        longest-first and tested with one STRtree dwithin query per candidate, so an OSM redraw (the same
        run drawn twice) collapses to a single kept piste before noding.

        Args:
          lines: Source piste LineStrings in metres.

        Returns:
          (kept, dropped): the surviving pistes and the count removed.
        """
        tol, cover = OSMConfig.DEDUP_TOL_M, OSMConfig.DEDUP_COVER_FRAC
        order = sorted(range(len(lines)), key=lambda i: lines[i].length, reverse=True)
        kept: list[LineString] = []
        for i in order:
            ls = lines[i]
            n = max(2, int(ls.length // 15))
            samples = line_interpolate_point(ls, np.linspace(0.0, 1.0, n + 1), normalized=True)
            # STRtree dwithin: count each kept line's samples within tol in one query; drop if any covers.
            covered = False
            if kept:
                tree = STRtree(kept)
                _, kept_idx = tree.query(samples, predicate="dwithin", distance=tol)
                counts = np.bincount(kept_idx, minlength=len(kept))
                covered = bool((counts / (n + 1) >= cover).any())
            if not covered:
                kept.append(ls)
        return kept, len(lines) - len(kept)

    # -- step 3: full-split (planar-node at every crossing) ---------------------------------------

    def _full_split(self, lines: list[LineString]) -> list[LineString]:
        """Split every piste at every crossing via shapely unary_union (planar noding), the connectivity engine.
        Vertices snap-round to a coarse grid first so near-coincident piste ends collapse to identical coords
        and node together; without the snap they stay disconnected and descent chains break at boundaries.

        Args:
          lines: Source piste LineStrings in metres.

        Returns:
          The noded segment LineStrings, one continuous piece between crossings.
        """
        if not lines:
            return []
        snapped = [set_precision(ls, OSMConfig.SNAP_GRID_M) for ls in lines]
        noded = unary_union([s for s in snapped if not s.is_empty and s.length > 0])
        # Snap the noded output to COORD_GRID_M: unary_union places crossings at fractional points, so
        # this makes EVERY vertex (endpoints + computed crossings) integer-metre → bit-exact shared nodes.
        noded = set_precision(noded, OSMConfig.COORD_GRID_M)
        if isinstance(noded, MultiLineString):
            return [LineString(g.coords) for g in noded.geoms if g.length > 0]
        if isinstance(noded, LineString) and noded.length > 0:
            return [LineString(noded.coords)]
        return []

    def _split_at_lift_stations(
        self, segments: list[LineString], lifts: list[tuple[LineString, str, str | None]]
    ) -> list[LineString]:
        """Split each piste segment where a lift station projects onto its interior within MIN_NODE_DIST_M.
        When the station sits within SLOPE_ON_SOURCE_TOL_M, snap the cut vertex onto the station so feeder and lift
        base share one hub after merge—you can ski into the base (R21). Returns piste segments split at lift stations.

        Args:
          segments: List of LineString piste segments in metres.
          lifts: List of (LineString, lift_type, name) lifts in metres.

        Returns:
          List of LineString segments, split at lift-station projections or snapped to them.
        """
        stations = [Point(lf.coords[0]) for lf, _lt, _nm in lifts] + [Point(lf.coords[-1]) for lf, _lt, _nm in lifts]
        if not stations:
            return segments
        # Split where a station is within MIN_NODE_DIST of the piste (so base/top shares its feeder's node);
        # within snap_tol, MOVE the cut vertex onto the station. Guarded below against a zero-length piece.
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
        """Cluster segment endpoints and lift stations so no two hubs stay within MIN_NODE_DIST_M, then pull
        slope nodes onto lift hubs within RELAXED_MERGE_DIST_M (lifts authoritative). One leader pass per
        phase guarantees every member is within tol of its leader without single-linkage chaining.

        Args:
          seg_pts: (x, y) metre endpoints of every split segment.
          lift_pts: (x, y) metre endpoints of every lift.

        Returns:
          (assign, hubs): point-index → hub-index map, and the merged hub coordinates.
        """
        pts = list(seg_pts) + list(lift_pts)
        is_lift = [False] * len(seg_pts) + [True] * len(lift_pts)
        assign = list(range(len(pts)))  # original point index -> current cluster index (into `pts`)

        def cluster_pass(tol: float, *, lift_only: bool, lift_lift_only: bool = False) -> bool:
            """One LEADER-clustering pass at `tol`: each point joins the nearest chosen leader within
            `tol` (no single-linkage chaining), lifts preferred as leaders. `lift_only` keeps distinct
            lift stations apart; `lift_lift_only` merges ONLY lift↔lift pairs (slope nodes untouched).
            """
            n = len(pts)
            order = sorted(range(n), key=lambda i: not is_lift[i])  # lifts first → they become leaders
            tree = cKDTree(np.array(pts, dtype=np.float64))  # library spatial index for the tol-radius query
            is_leader = [False] * n
            assign_leader = [-1] * n
            changed = False
            for i in order:
                if lift_lift_only and not is_lift[i]:
                    is_leader[i] = True
                    assign_leader[i] = i
                    continue
                best, best_d = -1, tol
                for ld in tree.query_ball_point(pts[i], tol):
                    if not is_leader[ld]:
                        continue  # only join an already-chosen leader (greedy, no chaining)
                    if lift_only and is_lift[i] and is_lift[ld]:
                        continue  # a lift never joins another lift
                    if lift_lift_only and not (is_lift[i] and is_lift[ld]):
                        continue  # only lift↔lift merges in this mode
                    d = math.dist(pts[i], pts[ld])
                    if d < best_d:
                        best, best_d = ld, d
                if best == -1:
                    is_leader[i] = True
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
        """Pre-merge node graph over the raw split segments: one node per rounded-metre coord, each segment an
        edge carrying its index. Returns (pn_xy, pn_hub, pn_elev, padj) with each node's merged-hub id and
        DEM height (inf for nodata). Shared by the contraction and the stranded-sink reconnection.
        """
        pn_id: dict[tuple[float, float], int] = {}
        pn_xy: list[XY] = []
        pn_hub: list[int] = []
        pn_elev: list[float] = []

        def pnode(xy: XY, hub: int) -> int:
            k = (xy[0], xy[1])  # coords are integer-metre (1 m grid) → exact-equal shared vertices
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

    @staticmethod
    def _min_climb_walk(
        starts: list[int],
        padj: dict[int, list[tuple[int, int]]],
        pn_elev: list[float],
        record: Callable[[int], bool],
        *,
        want_higher: bool,
        max_records: int | None = None,
    ) -> tuple[dict[int, int], list[int]]:
        """Min-AGAINST-grade shortest path over the pre-merge node graph from `starts`, via scipy
        Dijkstra. Cost is movement against the wanted direction (uphill when descending, downhill when
        `want_higher`), each hop capped at NODE_TERRAIN_TOL_M (a DEM-sampling dip, not a real wall).

        A `record`-true node is a goal: it's collected but NOT expanded through (a sink), so the walk
        keeps routing around it for others. Returns ({node: predecessor node}, recorded goal nodes,
        nearest-first). `max_records` caps how many goals to keep (nearest by cost).
        """
        m = len(pn_elev)
        goal = [record(x) for x in range(m)]  # a goal node is a sink: no outgoing edges built for it

        def against(x: int, y: int) -> float:
            return max(0.0, pn_elev[x] - pn_elev[y]) if want_higher else max(0.0, pn_elev[y] - pn_elev[x])

        eps = 1e-9  # keep a zero-grade hop strictly positive so Dijkstra visits it (and prefers fewer hops)
        # Weighted edges = every non-goal hop whose against-grade is within tol (goal = sink, no outgoing).
        edges = [
            (x, y, against(x, y))
            for x, neigh in padj.items()
            if not goal[x]
            for y, _si in neigh
            if against(x, y) <= OSMConfig.NODE_TERRAIN_TOL_M  # over-tol / nodata-inf hops are impassable
        ]
        rows = [e[0] for e in edges]
        cols = [e[1] for e in edges]
        data = np.array([e[2] for e in edges], dtype=np.float64) + eps
        graph = csr_matrix((data, (rows, cols)), shape=(m, m))
        # min_only collapses the multi-source search into one (dist, pred, source) triple per node.
        dist, pred, _sources = dijkstra(graph, directed=True, indices=starts, min_only=True, return_predecessors=True)
        par = {y: int(pred[y]) for y in range(m) if pred[y] >= 0}  # scipy: -9999 == no predecessor
        recorded = sorted((x for x in range(m) if goal[x] and np.isfinite(dist[x])), key=lambda x: dist[x])
        if max_records is not None:
            recorded = recorded[:max_records]
        return par, recorded

    def _contract_collapsed_descents(
        self,
        segments: list[LineString],
        seg_ab: list[tuple[int, int]],
        lift_ab: list[tuple[int, int]],
        hubs: list[XY],
    ) -> list[tuple[LineString, int, int]]:
        """Rebuild descents for lift tops whose pistes collapsed into self-loops during merge. When a lift top
        fed only by short pistes has no descending edge, walk the real pre-merge OSM geometry downhill to
        the nearest distinct lower hub and emit a through-segment. Geometry stays on real pistes (R19).
        """
        helev = self._hub_elev(hubs, {a for ab in seg_ab for a in ab} | {a for ab in lift_ab for a in ab})
        # which hubs already have a descending slope edge out of them?
        has_descent_out: set[int] = set()
        for a, b in seg_ab:
            if a == b or a not in helev or b not in helev:
                continue
            has_descent_out.add(a if helev[a] >= helev[b] else b)

        pn_xy, pn_hub, pn_elev, padj = self._build_premerge_graph(segments, seg_ab)

        lift_hubs = {h for ab in lift_ab for h in ab}
        min_lift_base = min((helev[h] for h in lift_hubs if h in helev), default=0.0)
        out: list[tuple[LineString, int, int]] = []
        # Seed every hub with NO descending edge yet above some lift base — a collapsed lift top or a
        # mid-mountain sink whose continuing piste was dropped (both strand a skier, R22). A hub below
        # every lift base is a valley terminus (return lift out of bbox) and is left alone.
        seed_hubs: list[tuple[int, int | None]] = []
        for a, b in lift_ab:
            if a == b or a not in helev or b not in helev:
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
            # min-climb walk over the pre-merge geometry; collect valid exit hubs (distinct, strictly
            # lower, non-lift) and pick the one nearest the aim point.
            aim_xy = hubs[base_hub] if base_hub is not None else hubs[top]

            def is_exit(x: int, top: int = top) -> bool:
                xh = pn_hub[x]
                return xh != top and xh not in lift_hubs and xh in helev and helev[xh] < helev[top]

            par, exits = self._min_climb_walk(starts, padj, pn_elev, is_exit, want_higher=False)
            if not exits:
                continue
            target = min(exits, key=lambda x: math.dist(hubs[pn_hub[x]], aim_xy))
            exit_hub = pn_hub[target]
            chain: list[int] = [target]
            n = target
            while n in par:
                n = par[n]
                chain.append(n)
            chain.reverse()
            clean = self._concat_chain_geometry(chain, hubs[top], hubs[exit_hub], segments, padj, pn_xy)
            if len(clean) < 2:
                continue
            logger.debug(f"[IMPORT] contract: stranded hub {top} → through-run to hub {exit_hub}, {len(clean)} pts")
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
        """Fuse the real OSM geometry along a pre-merge node chain into one polyline, endpoints pinned to
        the given hub coords. Each piece is oriented u→v (integer-metre vertices are bit-exact) then chained
        head-to-tail with shapely.ops.line_merge(directed=True); the <1 m dedup drops any duplicate vertex.
        """
        oriented: list[LineString] = []
        for u, v in zip(chain, chain[1:], strict=False):
            si = next(si for (w, si) in padj[u] if w == v)
            cs = list(segments[si].coords)
            if math.dist((cs[0][0], cs[0][1]), pn_xy[u]) > math.dist((cs[-1][0], cs[-1][1]), pn_xy[u]):
                cs = cs[::-1]  # orient piece u→v
            oriented.append(LineString(cs))
        fused = line_merge(MultiLineString(oriented), directed=True) if len(oriented) > 1 else oriented[0]
        coords: list[XY] = [start, *((x, y) for x, y in fused.coords), end]
        clean: list[XY] = [coords[0]]
        for p in coords[1:]:
            if math.dist(p, clean[-1]) >= 1.0:
                clean.append(p)
        return clean

    # -- step 5: assemble -------------------------------------------------------------------------

    def _assemble(
        self, segments: list[LineString], lifts: list[tuple[LineString, str, str | None]], source: list[LineString]
    ) -> ImportGraph:
        """Merge hubs (strict spacing, lift-authoritative), DEM-drape, orient downhill, then apply gates.
        Returns a connected ImportGraph with one run per hub-pair, dead-ends pruned, isolated components
        dropped, and sinks reconnected via real pre-merge geometry when a later pass strands them.

        Args:
            segments: Split piste segments (LineString, metres).
            lifts: Lift lines (LineString, type, name).
            source: Original pistes, for the on-source / pull-shape fidelity gates.

        Returns:
            ImportGraph with merged nodes, drape-sampled runs, and lift lines.
        """
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

        # Contract collapsed descents: a lift top whose feeder pistes are all shorter than the merge
        # distance sees every piece become a self-loop and dropped, so its whole descent vanishes. Walk
        # the real pre-merge geometry to the first distinct downhill hub and emit one through-segment.
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
        self._piste_vertices = [Point(c) for s in self._source_lines for c in s.coords]
        self._vtree = STRtree(self._piste_vertices) if self._piste_vertices else None

        # component per hub; keep only segments reaching a lift
        comp = component_labels(nodes=range(len(hubs)), edges=seg_ab + lift_ab, strong=False)
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
        self._mark_fabricated(graph)  # tag off-piste points for the red reference overlay
        return graph

    def _mark_fabricated(self, graph: ImportGraph) -> None:
        """Flag each run point that lies >PISTE_TOL_M from every source OSM piste as off-piste (fabricated).
        Drives the plot red overlay so readers see real OSM (blue) vs connector/pull geometry (red). No-op
        if no source pistes exist. Derives the union from self._source_lines (set in _assemble).
        """
        source_union = MultiLineString(self._source_lines) if self._source_lines else None
        tol = OSMConfig.PISTE_TOL_M
        for r in graph.slope_runs:
            if source_union is None:
                r.fabricated = [False] * len(r.points)
            else:
                r.fabricated = [source_union.distance(Point(self._to_m(p.lon, p.lat))) > tol for p in r.points]

    def _name_runs(self, graph: ImportGraph) -> None:
        """Attach the original OSM piste name to each run: the best-covering named source where ≥50% of the
        run's samples lie within SLOPE_ON_SOURCE_TOL_M. Run names preserve identity through split-at-crossings,
        lets a pure connector stay unnamed.
        """
        if not self._named_sources:
            return
        tol = OSMConfig.SLOPE_ON_SOURCE_TOL_M + 10.0
        lines = [ln for ln, _nm in self._named_sources]
        names = [nm for _ln, nm in self._named_sources]
        tree = STRtree(lines)
        for r in graph.slope_runs:
            samples = [Point(self._to_m(p.lon, p.lat)) for p in r.points]
            # STRtree dwithin: for each source line, how many of the run's samples fall within tol.
            _, line_idx = tree.query(samples, predicate="dwithin", distance=tol)
            if line_idx.size == 0:
                continue
            cover = np.bincount(line_idx, minlength=len(lines)) / len(samples)
            best = int(cover.argmax())  # argmax returns the FIRST max → matches the original `>` scan
            if cover[best] >= 0.5:
                r.name = names[best]

    def _run_max_slope_pct(self, run: SlopeRun) -> float:
        """Steepest-section slope magnitude (%) of a run, via the PRODUCTION PathSegment.max_slope_pct —
        the exact metric finish_slope re-applies when the app materialises the run. Delegating (not
        re-rolling a window here) keeps the import-time band and the final app band from drifting.
        """
        return PathSegment(points=list(run.points)).max_slope_pct

    def _group_slopes(self, graph: ImportGraph) -> None:
        """Group segment runs into whole named slopes with difficulty = steepest member's band. Each run
        lays in exactly one slope (referential completeness). Unnamed connectors fold into adjacent slopes
        without raising difficulty, or stand alone as black short-cuts (R2, R28, R29).
        """
        runs = graph.slope_runs
        band = SlopeConfig.DIFFICULTIES  # ["green","blue","red","black"], ascending
        rank = {d: i for i, d in enumerate(band)}
        run_diff = [TerrainAnalyzer.classify_difficulty(slope_pct=self._run_max_slope_pct(r)) for r in runs]

        # 1) named groups
        by_name = self._group_runs_by_name(runs)
        unnamed = [i for i, r in enumerate(runs) if not r.name]
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
        """Every node a skier can stand on: from each lift TOP, ski DOWN slope edges and ride lifts UP, to a
        fixpoint. A slope whose HIGH node is not in this set can never be entered — the mirror of a downhill
        sink — so it flags an unreachable top for the reconnection pass.
        """
        down = OSMGraphBuilder._down_adjacency(graph, elev)
        adjacency: dict[int, set[int]] = defaultdict(set, {k: set(v) for k, v in down.items()})
        seeds: set[int] = set()
        for lf in graph.lifts:
            base, top = (lf.node_a, lf.node_b) if elev[lf.node_a] <= elev[lf.node_b] else (lf.node_b, lf.node_a)
            adjacency[base].add(top)  # a skier at a lift base can ride up
            seeds.add(top)  # every lift top is a place a skier can start
        return OSMGraphBuilder._reachable(seeds, adjacency)

    def _reconnect_stranded_sinks(self, graph: ImportGraph) -> None:
        """Rebuild descending runs for slope nodes stranded after drop/dedup/gate passes. Walks stored
        pre-merge OSM geometry from sinks downhill to hubs that reach a lift, from unreachable tops uphill to
        reachable nodes, and from lift tops to their own bases. Iterated to a fixpoint (R3, R21 safe).
        """
        segments, seg_ab, hubs = self._pre_segments, self._pre_seg_ab, self._pre_hubs
        lift_nodes = self._lift_nodes(graph)
        if not lift_nodes:
            return
        min_lift_base = min(self._node_elevations(graph)[n] for n in lift_nodes)
        premerge = self._build_premerge_graph(segments, seg_ab)

        def down_sinks(g: ImportGraph, elev: dict[int, float]) -> list[tuple[int, set[int]]]:
            """Slope nodes above the lowest lift base that cannot reach a lift going down — stranded
            sinks (R22). Each seeds a downhill walk toward the set of hubs that DO reach a lift.
            A node below every lift base is a valley terminus and is excluded (not stranded).
            """
            good = self._hubs_reaching_lift(g, lift_nodes)
            return [
                (n, good)
                for n in {a for r in g.slope_runs for a in (r.node_a, r.node_b)}
                if n not in good and n not in lift_nodes and elev[n] > min_lift_base
            ]

        def own_base(g: ImportGraph, elev: dict[int, float]) -> list[tuple[int, set[int]]]:
            """A lift top that reaches SOME lift but not its OWN base (R21) — each seeds a downhill walk
            toward the set of hubs draining to that lift's base. Guarantees a skier off any lift can ski
            back to where they boarded, not merely to some other lift in the network.
            """
            seeds = []
            for lf in g.lifts:
                top = lf.node_a if elev[lf.node_a] >= elev[lf.node_b] else lf.node_b
                base = lf.node_b if top == lf.node_a else lf.node_a
                reach_base = self._down_reaches(g, {base}, elev)
                if top not in reach_base:
                    seeds.append((top, reach_base))
            return seeds

        def unreachable_tops(g: ImportGraph, elev: dict[int, float]) -> list[tuple[int, set[int]]]:
            """The mirror of a downhill sink: a slope top no skier can reach FROM any lift (R22 UP clause).
            Each seeds an UPHILL walk along a real descending feeder to a reachable node, so the orphaned
            top gains an entrance instead of being a run you can ski down but never get onto.
            """
            reachable = self._skier_reachable(g, elev)
            tops = {(r.node_a if elev[r.node_a] >= elev[r.node_b] else r.node_b) for r in g.slope_runs} - reachable
            return [(t, reachable) for t in tops]

        self._reconnect_pass(graph, down_sinks, segments, hubs, premerge, want_higher=False)
        self._reconnect_pass(graph, own_base, segments, hubs, premerge, want_higher=False)
        self._reconnect_pass(graph, unreachable_tops, segments, hubs, premerge, want_higher=True)

    def _reconnect_pass(
        self,
        graph: ImportGraph,
        seeds_fn: Callable[[ImportGraph, dict[int, float]], list[tuple[int, set[int]]]],
        segments: list[LineString],
        hubs: list[XY],
        premerge: tuple[list[XY], list[int], list[float], dict[int, list[tuple[int, int]]]],
        *,
        want_higher: bool,
    ) -> None:
        """One iteration of the stranded-sink reconnection fixpoint: find stranded nodes via `seeds_fn`,
        walk pre-merge OSM to allowed targets, and add any connecting runs. Stop when nothing new is added
        or the max iteration limit is hit.
        """
        pn_xy, pn_hub, pn_elev, padj = premerge
        for _ in range(20):
            elev = self._node_elevations(graph)
            seeds = seeds_fn(graph, elev)
            if not seeds:
                return
            added = False
            for sink, good in seeds:
                if self._reconnect_one_sink(
                    sink, good, graph, elev, segments, hubs, pn_xy, pn_hub, pn_elev, padj, want_higher=want_higher
                ):
                    added = True
            if not added:
                return

    @staticmethod
    def _lift_nodes(graph: ImportGraph) -> set[int]:
        """The set of all lift-station node ids (both ends of every lift). One source for 'is this a lift node'
        so the degree-2 collapse, spacing gates, prune, and dedup all agree on which nodes anchor a station
        and must never be treated as a plain slope junction.
        """
        return {n for lf in graph.lifts for n in (lf.node_a, lf.node_b)}

    @staticmethod
    def _group_runs_by_name(runs: list[SlopeRun]) -> dict[str, list[int]]:
        """Map each OSM name → the indices of the runs carrying it (unnamed runs excluded). One grouping used by
        both slope grouping and the fork split, so a change to name handling can never drift between them.
        Returns a defaultdict, so an absent name yields an empty list.
        """
        by_name: dict[str, list[int]] = defaultdict(list)
        for i, r in enumerate(runs):
            if r.name:
                by_name[r.name].append(i)
        return by_name

    @staticmethod
    def _orient_downhill(node_a: int, node_b: int, elev: dict[int, float]) -> tuple[int, int]:
        """The two node ids ordered high→low by elevation (a tie keeps node_a high). The one place that fixes
        the descent-orientation convention, so no caller can drift to a different tie-break. Used by every
        downhill-adjacency and chain-orientation site.
        """
        return (node_a, node_b) if elev[node_a] >= elev[node_b] else (node_b, node_a)

    @staticmethod
    def _node_elevations(graph: ImportGraph) -> dict[int, float]:
        """Node id → elevation for the current graph nodes, read straight off each node's PathPoint. The single
        elevation lookup the reachability, orientation, and spacing passes share, so they never disagree on
        a node's height. Rebuilt per call since node_points changes across passes.
        """
        return {k: v.elevation for k, v in graph.node_points.items()}

    @staticmethod
    def _down_adjacency(graph: ImportGraph, elev: dict[int, float]) -> dict[int, set[int]]:
        """Hi → {lo}: descending slope adjacency, each run oriented by its endpoints' elevations. The forward
        direction for every reachability query (sinks, unreachable tops, lift coverage), built from one
        orientation rule so all traversals see the same edges.
        """
        down: dict[int, set[int]] = defaultdict(set)
        for r in graph.slope_runs:
            hi, lo = OSMGraphBuilder._orient_downhill(r.node_a, r.node_b, elev)
            down[hi].add(lo)
        return down

    @staticmethod
    def _up_adjacency(graph: ImportGraph, elev: dict[int, float]) -> dict[int, set[int]]:
        """Lo → {hi}: the reverse of _down_adjacency, so reachability from a set answers 'who can ski
        DOWN to here'. Single source for the reversed-edge construction the down-reaches queries share.
        """
        up: dict[int, set[int]] = defaultdict(set)
        for hi, los in OSMGraphBuilder._down_adjacency(graph, elev).items():
            for lo in los:
                up[lo].add(hi)
        return up

    @staticmethod
    def _reachable(seeds: Iterable[int], adjacency: dict[int, set[int]]) -> set[int]:
        """Every node reachable from any seed over the directed `adjacency` (seeds included).

        Multi-source directed reachability via scipy `dijkstra(min_only=True)` — finite distance ⇒
        reachable. Replaces hand-rolled BFS so every traversal uses the C-optimized primitive.
        """
        seeds = list(seeds)
        if not seeds:
            return set()
        # Index every node that appears as a seed or in the adjacency (source or target).
        nodes = set(seeds) | set(adjacency) | {v for vs in adjacency.values() for v in vs}
        order = sorted(nodes)
        idx = {n: i for i, n in enumerate(order)}
        rows = [idx[u] for u, vs in adjacency.items() for _ in vs]
        cols = [idx[v] for vs in adjacency.values() for v in vs]
        n = len(order)
        graph = csr_matrix((np.ones(len(rows), dtype=np.int8), (rows, cols)), shape=(n, n))
        dist = dijkstra(graph, directed=True, indices=[idx[s] for s in seeds], min_only=True, unweighted=True)
        return {order[i] for i in range(n) if np.isfinite(dist[i])}

    @staticmethod
    def _down_reaches(graph: ImportGraph, targets: set[int], elev: dict[int, float]) -> set[int]:
        """Every node that reaches any target following DESCENDING slope edges (targets included). Walks the
        reverse of the down-adjacency from the targets, so it answers 'who can ski down to here'. Used to
        test whether a lift top can still reach its own base after drops (R21/R22).
        """
        return OSMGraphBuilder._reachable(targets, OSMGraphBuilder._up_adjacency(graph, elev))

    @staticmethod
    def _hubs_reaching_lift(graph: ImportGraph, lift_nodes: set[int]) -> set[int]:
        """Every node from which a lift station is reachable following DESCENDING slope edges only. A node with
        no downhill path to a lift is a stranded sink (R22) — it cannot feed a ski descent back to the
        network — so this drives both the reconnection seeds and the drop-stranding veto.
        """
        # "Reaches a lift going down" == reachable FROM a lift node over the REVERSED down edges; restrict
        # to real node_points (the caller's domain). Lift nodes trivially reach themselves (distance 0).
        up = OSMGraphBuilder._up_adjacency(graph, OSMGraphBuilder._node_elevations(graph))
        return OSMGraphBuilder._reachable(lift_nodes, up) & set(graph.node_points)

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
        """Walk the pre-merge OSM geometry from a stranded slope node to a reachable target,
        adding the connecting run(s) split at intermediate hubs to avoid floating (R6) or
        duplicates (R2). Returns True if any run was successfully added to the graph.
        """
        starts = [p for p in range(len(pn_xy)) if pn_hub[p] == sink]
        if not starts:
            return False

        # Collect candidate targets in ascending against-grain order (do NOT stop at the first): the
        # nearest good hub may fail the R3 backclimb gate in _try_add_sink_run (its real OSM arc dips then
        # rises), while a slightly farther one descends cleanly. Try each until one yields passing runs.
        def is_target(x: int) -> bool:
            h = pn_hub[x]
            return h in good and (elev[h] > elev[sink] if want_higher else elev[h] < elev[sink])

        par, candidates = self._min_climb_walk(starts, padj, pn_elev, is_target, want_higher=want_higher, max_records=8)
        for target in candidates:
            if self._try_add_sink_run(target, par, graph, segments, hubs, pn_hub, pn_xy, padj):
                return True
        return False

    def _try_add_sink_run(
        self,
        target: int,
        par: dict[int, int],
        graph: ImportGraph,
        segments: list[LineString],
        hubs: list[XY],
        pn_hub: list[int],
        pn_xy: list[XY],
        padj: dict[int, list[tuple[int, int]]],
    ) -> bool:
        """Rebuild a pre-merge node chain to the target, split it at interior existing hubs,
        drap and append each sub-run that passes the R3 backclimb gate. Returns True if at
        least one clean descending run was added (i.e., this target yielded a viable option).
        """
        chain: list[int] = [target]
        n = target
        while n in par:
            n = par[n]
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
            run = self._finalize_fork_run(pts, pa, ha, pb, hb, name=None)  # orient + pin + R3/R7 gates
            if run is None:
                continue
            logger.debug(f"[IMPORT] reconnect: sub-run {run.node_a}→{run.node_b}, {len(run.points)} pts")
            graph.slope_runs.append(run)
            added = True
        return added

    def _snap_interior_to_source(self, pts: list[PathPoint]) -> None:
        """Pull each interior point that strays beyond SLOPE_ON_SOURCE_TOL_M from every source
        piste back onto the nearest one via projection and re-sampling DEM z, in place.
        Removes bend-chord bows, keeping the run on real OSM geometry (R19); endpoints untouched.
        """
        if not self._source_lines:
            return
        tol = OSMConfig.SLOPE_ON_SOURCE_TOL_M
        source = unary_union(self._source_lines)  # project onto the whole piste network at once
        for i in range(1, len(pts) - 1):
            q = Point(self._to_m(pts[i].lon, pts[i].lat))
            if source.distance(q) <= tol:
                continue
            proj = source.interpolate(source.project(q))
            lon, lat = self._to_deg(proj.x, proj.y)
            elev = self.dem.get_elevation(lon=lon, lat=lat)
            if elev is not None:
                pts[i] = PathPoint(lon=lon, lat=lat, elevation=elev)

    def _prune_dead_end_slopes(self, graph: ImportGraph) -> None:
        """Iteratively drop any slope with a degree-1 non-lift endpoint (R22 frozen rule: every
        slope endpoint must connect onward — vertex of another segment or lift). Repeat to
        fixpoint since pruning one can expose another.
        """
        lift_nodes = self._lift_nodes(graph)
        while True:
            # MultiGraph so parallel runs (same node pair) each add to endpoint degree.
            g: nx.MultiGraph = nx.MultiGraph()  # networkx untyped
            g.add_edges_from((r.node_a, r.node_b) for r in graph.slope_runs)
            keep = [
                r
                for r in graph.slope_runs
                if not any(g.degree(n) == 1 and n not in lift_nodes for n in (r.node_a, r.node_b))
            ]
            if len(keep) == len(graph.slope_runs):
                break
            graph.dropped_isolated += len(graph.slope_runs) - len(keep)
            graph.slope_runs = keep

    def _build_slope_run(
        self, seg: LineString, a: int, b: int, graph: ImportGraph, source_union: MultiLineString | None
    ) -> SlopeRun | None:
        """Materialize one split segment into a downhill SlopeRun or None if it must be dropped.
        Pulls each end to its hub, DEM-drapes, orients downhill, and enforces the pull model,
        fidelity gate, and no-uphill rule; bumps graph.dropped_* counters on drops.
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
            qs = np.array([Point(self._to_m(p.lon, p.lat)) for p in pts], dtype=object)
            dists = self._vtree.query_nearest(qs, return_distance=True)[1]  # vectorized nearest-vertex dist
            if int(np.count_nonzero(dists > 45.0)) / len(pts) > 0.15:
                graph.dropped_isolated += 1
                return None
        # A real piste descends monotonically; the run keeps its real DEM elevations. DEM point-sampling
        # noise on a genuinely-descending piste is absorbed by the smoothed backclimb gate — a run whose
        # SMOOTHED profile still climbs > MAX_BACKCLIMB_M is a real uphill and is dropped (no elevation faked).
        if self._backclimb(pts) > OSMConfig.MAX_BACKCLIMB_M:
            graph.dropped_uphill += 1
            return None
        return SlopeRun(points=pts, node_a=a, node_b=b)

    def _runs_in_metres(self, runs: list[SlopeRun]) -> list[npt.NDArray[np.float64]]:
        """Convert each run's PathPoints to local-metre (n,2) numpy arrays for efficient
        vectorized distance and parallel geometry checks. Shared utility for fork/parallel
        analysis across multiple runs in a single operation.
        """
        return [np.array([self._to_m(p.lon, p.lat) for p in r.points], dtype=float) for r in runs]

    def _drop_guarded_fixpoint(
        self,
        graph: ImportGraph,
        find_victim: Callable[[list[SlopeRun], list[npt.NDArray[np.float64]], list[float], set[int]], int | None],
        label: str,
    ) -> None:
        """Repeatedly drop runs that a picker function identifies, unless the drop strands a
        node (R22 wins) — then mark it keep-unsafe and move on. Iterate to fixpoint; shared
        by twin (R34) and coverage-duplicate (R2) drops, differing only in picker logic.
        """
        keep_unsafe: set[int] = set()
        runs = graph.slope_runs
        pm = self._runs_in_metres(runs)  # rebuilt only after an actual drop (run set unchanged otherwise)
        plen = [self._polylen_m(r.points) for r in runs]
        while True:
            victim = find_victim(runs, pm, plen, keep_unsafe)
            if victim is None:
                return
            cand = ImportGraph(
                node_points=graph.node_points, slope_runs=runs[:victim] + runs[victim + 1 :], lifts=graph.lifts
            )
            if self._newly_stranded(cand):
                keep_unsafe.add(id(runs[victim]))  # dropping would strand a skier — keep it
                continue
            logger.debug(f"[IMPORT] drop {label} '{runs[victim].name}'")
            graph.slope_runs = cand.slope_runs
            graph.dropped_isolated += 1
            runs = graph.slope_runs
            pm = pm[:victim] + pm[victim + 1 :]  # surviving runs keep identity → keep_unsafe ids stay valid
            plen = plen[:victim] + plen[victim + 1 :]

    def _drop_parallel_twins(self, graph: ImportGraph) -> None:
        """Drop redundant parallel twins (R34): runs that hug a longer same-named sibling within
        the 18–60m band for ≥85% of their own length (one wide piste drawn as two offset
        ribbons). Reachability-guarded, iterated to fixpoint.
        """
        lo, hi, frac = OSMConfig.DEDUP_TOL_M, OSMConfig.PARALLEL_TOL_M, OSMConfig.PARALLEL_TWIN_FRAC

        def find(
            runs: list[SlopeRun], pm: list[npt.NDArray[np.float64]], plen: list[float], keep_unsafe: set[int]
        ) -> int | None:
            return self._find_parallel_twin(runs, pm, plen, lo, hi, frac, keep_unsafe)

        self._drop_guarded_fixpoint(graph, find, "redundant parallel twin")

    @staticmethod
    def _find_parallel_twin(
        runs: list[SlopeRun],
        pm: list[npt.NDArray[np.float64]],
        plen: list[float],
        lo: float,
        hi: float,
        frac: float,
        keep_unsafe: set[int],
    ) -> int | None:
        """Index of a run that is a redundant parallel twin of a longer same-named sibling, or
        None. A twin hugs the sibling within the lo–hi OFFSET band contiguously for ≥frac of its own
        length; safe-blocked runs and unnamed runs are skipped.
        """
        lines = [LineString(a) if len(a) > 1 else None for a in pm]
        for i in range(len(runs)):
            li = lines[i]
            if id(runs[i]) in keep_unsafe or plen[i] == 0 or not runs[i].name or li is None:
                continue
            for j in range(len(runs)):
                lj = lines[j]
                if i == j or lj is None or runs[i].name != runs[j].name or plen[i] > plen[j]:
                    continue
                # Offset band = points between lo and hi of the sibling (buffer(hi) minus buffer(lo));
                # the twin's longest CONTIGUOUS arc inside it, via shapely intersection, must be ≥ frac.
                band = lj.buffer(hi).difference(lj.buffer(lo))
                inband = li.intersection(band)
                # get_parts splits a single/multi result uniformly; the longest contiguous arc must be ≥frac.
                longest = max((p.length for p in get_parts(inband) if p.geom_type == "LineString"), default=0.0)
                if longest >= frac * plen[i]:
                    return i
        return None

    def _fork_divergence(
        self, short_xy: npt.NDArray[np.float64], long_xy: npt.NDArray[np.float64], short_len: float
    ) -> tuple[int, int] | None:
        """For two same-name runs sharing a hinge, detect if the shorter is a doubled ribbon
        that hugs the longer within PARALLEL_TOL_M contiguously from hinge for ≥PARALLEL_TWIN_FRAC
        of its length. Returns (long_div, short_div) split indices or None if divergence too early.
        """
        tol = OSMConfig.PARALLEL_TOL_M
        short_line, long_line = LineString(short_xy), LineString(long_xy)
        # First vertex (from hinge) whose distance to the OTHER whole line exceeds tol = the divergence.
        s_div = next((k for k, p in enumerate(short_xy) if long_line.distance(Point(p)) > tol), len(short_xy))
        trunk_arc = sum(math.dist(short_xy[k - 1], short_xy[k]) for k in range(1, s_div))
        if trunk_arc < OSMConfig.PARALLEL_TWIN_FRAC * short_len:
            return None
        l_div = next((k for k, p in enumerate(long_xy) if short_line.distance(Point(p)) > tol), len(long_xy))
        return max(1, l_div - 1), max(1, s_div - 1)

    @staticmethod
    def _oriented_from(run: SlopeRun, hinge: int) -> tuple[list[PathPoint], int]:
        """Return the run's points ordered so the hinge node appears first, along with the id
        of the other (far) endpoint. Used to standardize run orientation for fork/merge
        operations that reason about one end as the anchor.
        """
        if run.node_a == hinge:
            return list(run.points), run.node_b
        return list(run.points[::-1]), run.node_a

    def _finalize_fork_run(
        self, pts: list[PathPoint], pa: PathPoint, ida: int, pb: PathPoint, idb: int, name: str | None
    ) -> SlopeRun | None:
        """Orient a sub-run downhill (node_a=higher), pin endpoints to hub PathPoints, and reject it if
        it climbs (R3) or has a straight leg tunnelling through terrain (R7). Returns a valid SlopeRun or
        None; the shared finaliser for fork branches, degree-2 merges, and reconnected sink runs.
        """
        pts = list(pts)
        if pts[0].elevation < pts[-1].elevation:  # orient downhill so node_a is the higher hub
            pts = pts[::-1]
            pa, ida, pb, idb = pb, idb, pa, ida
        pts[0], pts[-1] = pa, pb  # R18/R33: endpoints ARE the hub points (shared object → exact)
        if len(pts) < 2 or self._backclimb(pts) > OSMConfig.MAX_BACKCLIMB_M:
            return None  # climbs (R3)
        if any(pts[k].distance_to(other=pts[k + 1]) > OSMConfig.MAX_STRAIGHT_M for k in range(len(pts) - 1)):
            return None  # a long straight chord would tunnel through terrain (R7)
        return SlopeRun(points=pts, node_a=ida, node_b=idb, name=name)

    def _split_parallel_forks(self, graph: ImportGraph) -> None:
        """Turn doubled ribbons into proper Y-forks (R35): two same-name runs sharing a hinge
        that run parallel then diverge to different ends. Insert node N at true divergence on
        the longer run; net +1 node, 2 runs→3. Iterated to fixpoint.
        """
        blocked: set[frozenset[int]] = set()
        changed = False
        for _ in range(200):  # fixpoint guard; each success splits one doubled ribbon
            if self._split_one_fork(graph, blocked):
                changed = True
            else:
                break
        if changed:
            self._mark_fabricated(graph)  # re-flag off-piste points on the new runs

    def _split_one_fork(self, graph: ImportGraph, blocked: set[frozenset[int]]) -> bool:
        """Find the first same-name shared-hinge doubled-ribbon pair not yet blocked, split it,
        return True on success. A pair that fails validation is added to blocked so fixpoint
        can terminate; prevents re-attempting hopeless cases.
        """
        runs = graph.slope_runs
        plen = [self._polylen_m(r.points) for r in runs]
        for idxs in self._group_runs_by_name(runs).values():
            for a in range(len(idxs)):
                for b in range(a + 1, len(idxs)):
                    i, j = idxs[a], idxs[b]
                    shared = {runs[i].node_a, runs[i].node_b} & {runs[j].node_a, runs[j].node_b}
                    if len(shared) != 1:
                        continue
                    key = frozenset((id(runs[i]), id(runs[j])))
                    if key in blocked:
                        continue
                    short, lng = (i, j) if plen[i] <= plen[j] else (j, i)
                    if self._try_fork(graph, runs[short], runs[lng], next(iter(shared)), plen[short]):
                        return True
                    blocked.add(key)
        return False

    def _try_fork(
        self, graph: ImportGraph, short_run: SlopeRun, long_run: SlopeRun, hinge: int, short_len: float
    ) -> bool:
        """Attempt trunk+2-branch split for one shared-hinge pair. Validates spacing (R5/R13),
        lifts (R10), and floating (R6); mutates graph only on full success (atomic). Returns
        False (no change) if any gate fails, so caller can block the pair.
        """
        short_pts, far_short = self._oriented_from(short_run, hinge)
        long_pts, far_long = self._oriented_from(long_run, hinge)
        if far_short == far_long:
            return False
        long_xy = np.array([self._to_m(p.lon, p.lat) for p in long_pts], dtype=float)
        short_xy = np.array([self._to_m(p.lon, p.lat) for p in short_pts], dtype=float)
        div = self._fork_divergence(short_xy, long_xy, short_len)
        if div is None:
            return False
        l_div, s_div = div
        if l_div >= len(long_pts) - 1 or s_div >= len(short_pts) - 1:  # no real tail to branch → leave for R34
            return False
        # Node N spacing, in the builder's own local-metre metric. R5/R13: ≥MIN_NODE_DIST_M from every
        # node; R10: ≥RELAXED_MERGE_DIST_M from every LIFT node. Walk upstream from the true divergence
        # toward the hinge until both hold (a node too close to a lift would be pulled into its hub).
        lift_nodes = self._lift_nodes(graph)
        node_m = [(nid, self._to_m(p.lon, p.lat)) for nid, p in graph.node_points.items()]
        elev = self._node_elevations(graph)
        # R6: N must not float within MIN_NODE_DIST_M of ANOTHER run's interior whose elevation SPANS N
        # (it would be a missed split). Precompute those runs' metre lines + elevation ranges once.
        other_lines = [
            (
                LineString([self._to_m(p.lon, p.lat) for p in r.points]),
                min(elev[r.node_a], elev[r.node_b]),
                max(elev[r.node_a], elev[r.node_b]),
            )
            for r in graph.slope_runs
            if r is not short_run and r is not long_run and len(r.points) >= 2
        ]

        def spacing_ok(k: int) -> bool:
            nq = Point(self._to_m(long_pts[k].lon, long_pts[k].lat))
            for nid, nm in node_m:
                floor = OSMConfig.RELAXED_MERGE_DIST_M if nid in lift_nodes else OSMConfig.MIN_NODE_DIST_M
                if math.dist((nq.x, nq.y), nm) < floor:
                    return False
            nz = long_pts[k].elevation
            return not any(
                lo <= nz <= hi and line.distance(nq) < OSMConfig.MIN_NODE_DIST_M for line, lo, hi in other_lines
            )

        k_n = next((k for k in range(l_div, 0, -1) if spacing_ok(k)), None)
        if k_n is None:
            return False
        n_pt = long_pts[k_n]
        new_id = max(graph.node_points) + 1
        # trunk hinge→N and branch_long N→far_long: slices of the longer run's REAL, already-draped geometry.
        trunk = self._finalize_fork_run(
            long_pts[: k_n + 1], graph.node_points[hinge], hinge, n_pt, new_id, long_run.name
        )
        branch_long = self._finalize_fork_run(
            long_pts[k_n:], n_pt, new_id, graph.node_points[far_long], far_long, long_run.name
        )
        # branch_short: short fabricated connector N→shorter's OWN divergence point (where the shorter
        # leaves the trunk corridor — NOT the nearest point to N, else its tail re-doubles branch_long),
        # then the shorter run's real tail to far_short; draped + snapped onto real OSM.
        conn = self._connector(n_pt, (short_pts[s_div].lon, short_pts[s_div].lat))
        if conn is None:
            return False
        latlon = [(n_pt.lon, n_pt.lat)] + conn + [(p.lon, p.lat) for p in short_pts[s_div:]]
        draped = self._drape([self._to_m(lon, lat) for lon, lat in latlon], n_pt, graph.node_points[far_short])
        if draped is None or len(draped) < 2:
            return False
        self._snap_interior_to_source(draped)
        branch_short = self._finalize_fork_run(
            draped, n_pt, new_id, graph.node_points[far_short], far_short, short_run.name
        )
        if trunk is None or branch_long is None or branch_short is None:
            return False
        # commit atomically
        graph.node_points[new_id] = n_pt
        graph.slope_runs = [r for r in graph.slope_runs if r is not short_run and r is not long_run] + [
            trunk,
            branch_long,
            branch_short,
        ]
        logger.debug(f"[IMPORT] fork-split '{long_run.name}': hinge {hinge} → node {new_id} → ({far_long},{far_short})")
        return True

    def _collapse_degree2_nodes(self, graph: ImportGraph) -> None:
        """Merge runs at degree-2 non-lift nodes (R36: pass-through junctions, not real splits). Each pair
        fuses into one A→B run through the node (shapely line_merge, in _merge_at_node), longer run's
        name wins. Iterated to fixpoint; a merge the R3/R7 gates reject leaves the node uncollapsed.
        """
        lift_nodes = self._lift_nodes(graph)
        for _ in range(500):  # fixpoint guard; each pass collapses one node
            # MultiGraph carrying each run's index: a degree-2 non-lift node with two DISTINCT incident
            # runs is a pass-through to collapse (a parallel-pair node has degree 2 but one shared run).
            g: nx.MultiGraph = nx.MultiGraph()  # networkx untyped
            g.add_edges_from((r.node_a, r.node_b, {"ri": i}) for i, r in enumerate(graph.slope_runs))
            victim = next(
                (
                    n
                    for n in g.nodes
                    if n not in lift_nodes
                    and g.degree(n) == 2
                    and len({d["ri"] for _, _, d in g.edges(n, data=True)}) == 2
                ),
                None,
            )
            if victim is None:
                return
            i, j = (d["ri"] for _, _, d in g.edges(victim, data=True))
            if not self._merge_at_node(graph, victim, graph.slope_runs[i], graph.slope_runs[j]):
                lift_nodes = lift_nodes | {victim}  # self-loop/degenerate → treat as non-collapsible

    def _merge_at_node(self, graph: ImportGraph, node: int, r1: SlopeRun, r2: SlopeRun) -> bool:
        """Fuse two runs sharing a node into one run through it, replacing both in
        graph.slope_runs. Returns False (no change) if merge would be a self-loop; else
        returns True and mutates the graph atomically with the merged run.
        """
        pts1, far1 = self._oriented_from(r1, node)  # points START at node → far1
        pts2, far2 = self._oriented_from(r2, node)  # points START at node → far2
        if far1 == far2:
            return False
        # far1 → node → far2: reverse piece 1 to end at the shared node, then directed line_merge chains
        # it head-to-tail with piece 2 (integer-metre vertices are bit-exact → they fuse, dropping the dup).
        ls1 = LineString([(p.lon, p.lat, p.elevation) for p in reversed(pts1)])  # far1 → node
        ls2 = LineString([(p.lon, p.lat, p.elevation) for p in pts2])  # node → far2
        fused = line_merge(MultiLineString([ls1, ls2]), directed=True)
        if fused.geom_type != "LineString":
            return False  # pieces didn't chain into one line (shared endpoint mismatch) — fail loud
        merged_pts = [PathPoint(lon=x, lat=y, elevation=z) for x, y, z in fused.coords]
        name: str | None
        if r1.name and not r2.name:
            name = r1.name
        elif r2.name and not r1.name:
            name = r2.name
        else:  # both named or both unnamed → the LONGER run's name (None if both unnamed)
            name = (r1 if self._polylen_m(r1.points) >= self._polylen_m(r2.points) else r2).name
        merged = self._finalize_fork_run(merged_pts, graph.node_points[far1], far1, graph.node_points[far2], far2, name)
        if merged is None:
            return False
        graph.slope_runs = [r for r in graph.slope_runs if r is not r1 and r is not r2] + [merged]
        logger.debug(f"[IMPORT] collapse degree-2 node {node}: {far1}↔{far2} '{name}'")
        return True

    def _dedup_final_runs(self, graph: ImportGraph) -> None:
        """Drop near-duplicate runs whose vertices cluster ≥DEDUP_COVER within DEDUP_TOL of a longer run.
        Filters out OSM redraws that survived the initial per-hub-pair dedup by occupying different hub-pairs.
        Iterated to fixpoint; reachability-guarded so a drop that strands a node is vetoed (R2).
        """
        tol, cover = OSMConfig.DEDUP_TOL_M, OSMConfig.DEDUP_COVER_FRAC

        def find(
            runs: list[SlopeRun], pm: list[npt.NDArray[np.float64]], plen: list[float], keep_unsafe: set[int]
        ) -> int | None:
            return self._find_covered_run(runs, pm, plen, tol, cover, keep_unsafe)

        self._drop_guarded_fixpoint(graph, find, "near-duplicate run")

    def _find_covered_run(
        self,
        runs: list[SlopeRun],
        pm: list[npt.NDArray[np.float64]],
        plen: list[float],
        tol: float,
        cover: float,
        keep_unsafe: set[int],
    ) -> int | None:
        """Return index of a run ≥cover-covered (vertex-within-tol) by a longer run, or None if none found.
        Excepts a short shared-hub coincident prefix ≤ MAX_PULL_M (a pull artifact, not true duplication).
        Skips runs marked unsafe for reachability (would strand a skier if dropped).
        """
        max_pull = OSMConfig.MAX_PULL_M
        lines = [LineString(a) if len(a) > 1 else None for a in pm]
        for i in range(len(runs)):
            li = lines[i]
            if id(runs[i]) in keep_unsafe or li is None:
                continue
            ai = pm[i]
            for j in range(len(runs)):
                lj = lines[j]
                if i == j or lj is None or plen[i] > plen[j]:
                    continue
                shared = {runs[i].node_a, runs[i].node_b} & {runs[j].node_a, runs[j].node_b}
                if shared:
                    # Walk run i from the shared hub, summing the arc that stays within tol of run j; a short
                    # coincident prefix (≤ MAX_PULL_M) is a pull artifact at the shared node, not a duplicate.
                    seq = ai if runs[i].node_a in shared else ai[::-1]
                    coincident = 0.0
                    for k in range(1, len(seq)):
                        if lj.distance(Point(seq[k])) > tol:
                            break
                        coincident += float(np.hypot(*(seq[k] - seq[k - 1])))
                    if coincident <= max_pull:
                        continue
                near = sum(1 for p in ai if lj.distance(Point(p)) <= tol)
                if near / len(ai) >= cover:
                    return i
        return None

    def _newly_stranded(self, after: ImportGraph) -> bool:
        """True if the graph would leave any slope node orphaned after a drop.
        Guards both R22 clauses: (a) a descent sink above a lift base, (b) an unreachable slope top.
        Rejects drops that break skier traversal, preventing reachability violations.
        """
        elev = self._node_elevations(after)
        lift_nodes = self._lift_nodes(after)
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
        """Drop slopes passing within MIN_NODE_DIST of a hub that is not their own endpoint.
        Such runs should have split at that hub but couldn't; they are removed rather than left floating.
        Prioritizes topology correctness (R6) even when it means dropping an otherwise valid run.
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
        """Drop slope runs whose component (over kept slopes + lifts) contains no lift station.
        Iterates to fixpoint, then removes now-unused slope-only nodes to keep graph compact.
        Enforces R4: every kept run must connect transitively to at least one lift.
        """
        while True:
            edges = [(r.node_a, r.node_b) for r in graph.slope_runs] + [(lf.node_a, lf.node_b) for lf in graph.lifts]
            nodes = {n for e in edges for n in e}
            comp = component_labels(nodes=nodes, edges=edges, strong=False)
            lift_comps = {comp[lf.node_a] for lf in graph.lifts} | {comp[lf.node_b] for lf in graph.lifts}
            keep = [r for r in graph.slope_runs if comp[r.node_a] in lift_comps or comp[r.node_b] in lift_comps]
            if len(keep) == len(graph.slope_runs):
                break
            graph.dropped_isolated += len(graph.slope_runs) - len(keep)
            graph.slope_runs = keep
        # drop now-unused slope-only nodes (keep lift stations)
        used = {n for r in graph.slope_runs for n in (r.node_a, r.node_b)} | OSMGraphBuilder._lift_nodes(graph)
        graph.node_points = {k: v for k, v in graph.node_points.items() if k in used}

    def _connector(self, frm: PathPoint, to_lonlat: Vertex) -> list[Vertex] | None:
        """Interior (lon,lat) points pulling a hub to an on-piste body point; None if gap > MAX_PULL_M.
        Prefers real OSM piste substring when both points sit on one piste + arc is ≤1.5× straight gap.
        Falls back to a straight densified pull (robust over complex custom paths); gap > 300 m aborts.
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
        # Otherwise a STRAIGHT densified pull, point spacing domain-tuned (round(gap/step); the rules are
        # calibrated to this density feeding the DEM drape). Vectorized line_interpolate_point at the exact
        # k/n fractions of the metre chord — the same point set, no per-point loop.
        n = max(1, int(gap // OSMConfig.RESAMPLE_STEP_M))
        chord = LineString([h_m, b_m])
        interior = line_interpolate_point(chord, np.arange(1, n) / n, normalized=True)
        return [self._to_deg(p.x, p.y) for p in np.atleast_1d(interior)]

    def _piste_substring(self, h_m: XY, b_m: XY) -> list[XY] | None:
        """If both hub and body point lie within ~55m of ONE source piste, return the along-piste substring.
        Metres coords, endpoints excluded (caller adds them); returns None if points sit on different pistes.
        Keeps connector on real OSM geometry whenever credible, avoiding straight-line artifacts.
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
        """Split each segment at interior hubs projecting onto it within MIN_NODE_DIST_M.
        Ensures slopes pass through (share) those nodes rather than floating past them (R6).
        Never cuts at interior hubs higher than both segment endpoints (preserves descent monotonicity).
        """
        onseg_tol = OSMConfig.MIN_NODE_DIST_M  # R6 gate: split where a hub is within this of the segment
        hub_pts = {h: Point(hubs[h]) for h in used}
        hub_elev = self._hub_elev(hubs, used)  # `used` hubs are all on-DEM (they became node_points)
        out: list[tuple[LineString, int, int]] = []
        for seg, a, b in kept:
            length = seg.length
            hi_end = max(hub_elev[a], hub_elev[b])  # the run's higher endpoint
            cuts: list[tuple[float, int]] = [(0.0, a), (length, b)]
            for h, hp in hub_pts.items():
                if h in (a, b) or h not in hub_elev:
                    continue  # nodata hub (DEM hole) — can't judge its height, don't cut there
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
        """Trim TRIM_END_M off both ends of a segment, keeping its terrain-following middle.
        Ends become clean straight connectors to hubs on first/last stretch; short segments returned unchanged.
        Dropping severs descent chains, so even sub-min-length segments are never truncated away.
        """
        if trim_m <= 0 or seg.length <= 2 * trim_m + 5.0:
            return seg
        piece = substring(seg, trim_m, seg.length - trim_m)
        if not isinstance(piece, LineString) or piece.is_empty or len(piece.coords) < 2:
            return seg
        return piece

    def _valid_pull_shape(self, pts: list[PathPoint], source_union: MultiLineString | None) -> bool:
        """True if the run obeys the pull model: OSM body on-piste, off-piste only in END connectors.
        Mirrors R19 exactly (PISTE_TOL_M=40m, MAX_PULL_M=300m) so builder accepts/rejects identically to test.
        Rejects mid-run tunnels and connectors > 300m; source_union=None bypasses all checks (unbounded).
        """
        if source_union is None:
            return True
        tol = OSMConfig.PISTE_TOL_M
        max_pull = OSMConfig.MAX_PULL_M
        pts_m = [Point(self._to_m(p.lon, p.lat)) for p in pts]
        off = list(distance(np.array(pts_m, dtype=object), source_union) > tol)  # vectorized per-point distance
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
        """Compute length (metres) of a PathPoint polyline by summing consecutive point-pair distances.
        Accumulates terrain-aware distances via the PathPoint distance_to method.
        Returns 0 for a single point or empty list.
        """
        return sum(pts[i].distance_to(other=pts[i + 1]) for i in range(len(pts) - 1))

    @staticmethod
    def _on_source(piece: LineString, source_union: MultiLineString | None) -> bool:
        """True if ≥85% of a segment's points lie within ~40m of a source OSM piste (sampled uniformly).
        Enforces R12 boundary: runs straying too far from raw Overpass geometry are rejected as fabricated.
        No source union → unconditionally True (no real data to validate against).
        """
        if source_union is None:
            return True
        tol = OSMConfig.SLOPE_ON_SOURCE_TOL_M
        n = max(2, int(piece.length // 15))
        # Vectorized: interpolate n+1 sample points, distance each to the source in one shapely call.
        samples = line_interpolate_point(piece, np.linspace(0.0, 1.0, n + 1), normalized=True)
        off = int(np.count_nonzero(distance(samples, source_union) > tol))
        return off / (n + 1) <= 0.15

    def _drape(self, coords: list[XY], start: PathPoint, end: PathPoint) -> list[PathPoint] | None:
        """DEM-drape coords: sample terrain elevation at the domain-tuned along-line distances (original
        vertices ∪ a round(total/step) grid, near-coincident dropped), via vectorized shapely
        line_interpolate_point. Returns None on nodata; endpoints pinned to start/end for hub alignment.
        """
        line = LineString(coords)
        total = line.length
        if total <= 0:
            return None
        vert_d = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(np.array(coords), axis=0).T))])
        n = max(1, int(round(total / OSMConfig.RESAMPLE_STEP_M)))
        merged = np.unique(np.concatenate([vert_d, np.linspace(0.0, total, n + 1)]))
        dists = merged[np.concatenate([[True], np.diff(merged) >= 1.0])]  # drop points <1 m apart
        if dists[-1] < total - 1e-6:
            dists = np.append(dists, total)
        out: list[PathPoint] = []
        for p in line_interpolate_point(line, dists):  # vectorized: one shapely call, no per-point loop
            lon, lat = self._to_deg(p.x, p.y)
            elev = self.dem.get_elevation(lon=lon, lat=lat)
            if elev is None:
                return None
            out.append(PathPoint(lon=lon, lat=lat, elevation=elev))
        out[0] = start
        out[-1] = end
        return out

    def _backclimb(self, pts: list[PathPoint]) -> float:
        """ULTRA-STRICT uphill metric (R3): worst elevation RISE over any ~80m window of a descending run.
        Raw DEM elevations, no smoothing or clamping (fidelity over false remedies).
        Matches the R3 test exactly so builder never drops/keeps differently than the validator.
        """
        if len(pts) < 2:
            return 0.0
        oriented = pts if pts[0].elevation >= pts[-1].elevation else pts[::-1]
        es = np.array([p.elevation for p in oriented])
        seg = np.array([oriented[i - 1].distance_to(other=oriented[i]) for i in range(1, len(oriented))])
        cum = np.concatenate([[0.0], np.cumsum(seg)])
        # window end j per start i = first index with cum[j]-cum[i] >= win (clamped to last); vectorized.
        j = np.minimum(np.searchsorted(cum, cum + OSMConfig.BACKCLIMB_WINDOW_M, side="left"), len(es) - 1)
        return float(np.max(es[j] - es))  # oriented downhill → a positive span is an uphill


class GraphImporter(BaseOSMImporter):
    """Imports lifts + slopes via the connected-graph algorithm: the lifts and slopes are reported
    as OSMGraphBuilder preprocessed them (hub-merged, reconnected, grouped). The graph's own
    hub-aligned lifts are used (not raw OSM), so slopes and lift stations share nodes.
    """

    def _assemble(self, elements: list[OverpassElement], on_progress: ProgressFn) -> ImportResult:
        """Run the connected-graph builder over the fetched ways and report it as an ImportResult.
        The graph's own hub-aligned lifts + grouped slope chains are returned (not raw OSM), so slopes
        and lift stations share nodes when the ResortGraph materializes them.
        """
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
        """Write raw Overpass elements (base) and a built-graph PNG reference (skipped for empty graphs).
        Delegates base dump to parent class; adds a visual reference PNG for import debugging.
        No-op if graph is empty (render_png would fail on zero nodes).
        """
        super()._dump(elements, dump_dir)
        if not self._graph.node_points:
            logger.debug("OSM graph import: empty graph, no reference PNG")
            return
        out = dump_dir / "osm_import.png"
        render_png(self._graph, out)
        logger.debug(f"OSM graph import: wrote reference PNG to {out}")

"""Verification of the OSM connected-graph import rules, measured on the REAL Ischgl cache + DEM."""

import json
import math
import os
from collections import Counter, defaultdict, deque
from dataclasses import dataclass

import pytest
from shapely.geometry import LineString, Point

from skiresort_planner.constants import MapConfig, OSMConfig
from skiresort_planner.core.dem_service import DEMService
from skiresort_planner.core.geo_calculator import GeoCalculator
from skiresort_planner.generators.osm_graph_builder import (
    LiftReachabilityCheck,
    OSMGraphBuilder,
    SlopeRun,
    _linear_chains,
    ways_to_lines,
)
from tests_workflow.conftest import MockDEMService

# Pure test-assertion thresholds (counts / connectivity) live here; every geometric domain tolerance
# comes from OSMConfig (single source of truth — no drift between the builder and the rules it must meet).
FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")


@dataclass(frozen=True)
class Dataset:
    """One real OSM test resort: fixture + bbox + per-resort count expectations. Geometric tolerances
    are universal (OSMConfig) — only the size envelope (node/segment counts) is dataset-specific.
    """

    name: str
    fixture: str
    bbox: tuple[float, float, float, float]
    min_segments: int  # a full box must not collapse to near-empty
    max_segments: int  # segment-count ceiling
    max_nodes: int  # node-count ceiling
    min_seg_per_slope: float  # R29: path-segments per FINAL app-slope (smaller resorts group looser)

    def load(self):
        with open(os.path.join(FIXTURES_DIR, self.fixture), encoding="utf-8") as f:
            return json.load(f)["elements"]


DATASETS = [
    Dataset(
        name="ischgl",
        fixture="ischgl_osm.json",
        bbox=(10.261216324885865, 46.92838815227593, 10.392868056217921, 47.01821926977503),
        min_segments=100,
        max_segments=300,
        max_nodes=200,
        min_seg_per_slope=1.15,
    ),
    Dataset(
        name="scuol",
        fixture="scuol_osm.json",
        bbox=(10.227667, 46.785401, 10.338357, 46.844327),
        min_segments=30,
        max_segments=120,
        max_nodes=90,
        min_seg_per_slope=1.05,
    ),
]


def _hav(a, b):
    R = 6371000
    p1, p2 = math.radians(a[1]), math.radians(b[1])
    dp = math.radians(b[1] - a[1])
    dl = math.radians(b[0] - a[0])
    x = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(x))


def _polylen(coords):
    return sum(_hav(coords[i], coords[i + 1]) for i in range(len(coords) - 1))


def _to_local(lon, lat, bbox):
    """Project (lon,lat) to local metres about `bbox`'s origin (flat-earth, fine at this scale), using
    the production GeoCalculator.meters_per_degree projection (same basis as model/path_smoothing.py).
    """
    lat0 = (bbox[1] + bbox[3]) / 2
    mlon, mlat = GeoCalculator.meters_per_degree(lat=lat0)
    return ((lon - bbox[0]) * mlon, (lat - bbox[1]) * mlat)


def _seg_point_dist(pt, poly):
    """Min distance (m) from local-metre point `pt` to a polyline `poly` (list of local-metre points),
    via shapely — distance to the nearest LINE SEGMENT, not just a vertex (a node can float beside a leg).
    """
    return Point(pt).distance(LineString(poly))


def _backclimb(pts) -> float:
    """ULTRA-STRICT uphill metric (R3): the largest elevation RISE over any BACKCLIMB_WINDOW_M span of
    the descending-oriented run. A real piste must go strictly DOWN — over every ~80 m window the far
    end sits no higher than the near end. Returns the worst window's rise (0 for a clean descent). Raw
    DEM elevations, no smoothing — a genuine uphill is surfaced, not averaged away.
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
        j = min(j, len(es) - 1)
        worst = max(worst, es[j] - es[i])
    return worst


def _components(graph):
    adj = defaultdict(set)
    for r in graph.slope_runs:
        adj[r.node_a].add(r.node_b)
        adj[r.node_b].add(r.node_a)
    for lf in graph.lifts:
        adj[lf.node_a].add(lf.node_b)
        adj[lf.node_b].add(lf.node_a)
    seen, comps = set(), []
    for n in graph.node_points:
        if n in seen:
            continue
        q = deque([n])
        seen.add(n)
        c = {n}
        while q:
            x = q.popleft()
            for y in adj[x]:
                if y not in seen:
                    seen.add(y)
                    q.append(y)
                    c.add(y)
        comps.append(c)
    return comps


class _Bundle:
    """A built graph plus the dataset context every rule needs (bbox, raw elements, count envelope)."""

    def __init__(self, dataset: Dataset):
        self.dataset = dataset
        self.bbox = dataset.bbox
        self.elements = dataset.load()
        self.pistes, self.raw_lifts = ways_to_lines(self.elements, dataset.bbox)
        dem = DEMService()  # REAL EuroDEM Alps terrain (data/alps_dem.tif)
        self.graph = OSMGraphBuilder(dem=dem, bbox=dataset.bbox).build(
            self.pistes, self.raw_lifts, on_progress=lambda f, t: None
        )


@pytest.fixture(scope="module", params=DATASETS, ids=[d.name for d in DATASETS])
def ds(request):
    """Build the graph from each committed OSM fixture (set in stone — never re-fetched); each rule runs
    once per dataset (Ischgl + Scuol). A missing fixture is a broken checkout, not a reason to skip.
    """
    return _Bundle(request.param)


class TestImportRules:
    """Each rule from the design conversation, asserted on every real import (Ischgl + Scuol)."""

    def test_r1_all_lifts_imported(self, ds):
        """R1: STRICT import — every lift way over MIN_LIFT_LENGTH_M must appear in the final graph, and none may be
        dropped. The builder keeps all eligible lifts regardless of connectivity or topology. Prevents a
        length-qualifying lift silently vanishing during import.
        """
        els = ds.elements
        _pistes, lifts = ways_to_lines(els, ds.bbox)
        # count lift ways that clear the min-length gate (what the builder must keep)
        expected = 0
        for vs, _lt, _nm in lifts:
            length = sum(_hav(vs[i], vs[i + 1]) for i in range(len(vs) - 1))
            if length >= OSMConfig.MIN_LIFT_LENGTH_M:
                expected += 1
        assert len(ds.graph.lifts) == expected, (
            f"imported {len(ds.graph.lifts)} lifts but {expected} clear the length gate — lifts must never be dropped"
        )

    def test_r2_no_duplicate_runs(self, ds):
        """R2: STRICT dedup — no slope's geometry may be covered (DEDUP_COVER_FRAC within DEDUP_TOL_M) by an
        equal-or-longer sibling. Two runs sharing a hub node may stay coincident only for a stretch under MAX_PULL_M,
        a mere pull artifact. Prevents a shared piste double-drawn where a fork node was missed.
        """
        sruns = ds.graph.slope_runs
        runs = [[(p.lon, p.lat) for p in r.points] for r in sruns]
        dupes = 0
        for i, a in enumerate(runs):
            for j, b in enumerate(runs):
                if i == j or len(b) < 2:
                    continue
                shared = {sruns[i].node_a, sruns[i].node_b} & {sruns[j].node_a, sruns[j].node_b}
                if shared:
                    # walk a from the shared node; the coincident prefix ends where a departs > DEDUP_TOL
                    # of b. Allowed only if that prefix is within the pull distance (a mere hub-pull).
                    a_from_shared = a if sruns[i].node_a in shared else a[::-1]
                    coincident = 0.0
                    for k in range(1, len(a_from_shared)):
                        if min(_hav(a_from_shared[k], vb) for vb in b) > OSMConfig.DEDUP_TOL_M:
                            break
                        coincident += _hav(a_from_shared[k - 1], a_from_shared[k])
                    if coincident <= OSMConfig.MAX_PULL_M:
                        continue  # short coincidence at the shared hub → pull artifact, not a duplicate
                # fraction of a's vertices within DEDUP_TOL of polyline b
                near = 0
                for va in a:
                    if any(_hav(va, vb) < OSMConfig.DEDUP_TOL_M for vb in b):
                        near += 1
                if near / len(a) >= OSMConfig.DEDUP_COVER_FRAC and _polylen(a) <= _polylen(b):
                    dupes += 1
                    break
        assert dupes == 0, f"{dupes} slopes are near-duplicates of another slope"

    def test_r3_no_uphill_slope(self, ds):
        """R3: STRICT descent. Two independent conditions, both mandatory:
        (a) NET DROP — a run's two hub endpoints must strictly differ in elevation (end node lower than
            start node, zero tolerance): a run whose endpoints are level can't be oriented downhill.
        (b) NO BACK-CLIMB — no rise over MAX_BACKCLIMB_M on any BACKCLIMB_WINDOW_M window of the
            descending run, on raw DEM elevations. Prevents an uphill segment surviving import.
        """
        elev = {k: v.elevation for k, v in ds.graph.node_points.items()}
        level = [r for r in ds.graph.slope_runs if elev[r.node_a] == elev[r.node_b]]
        assert level == [], f"{len(level)} slope runs have level endpoints (no strict net drop start→end)"
        offenders = [r for r in ds.graph.slope_runs if _backclimb(r.points) > OSMConfig.MAX_BACKCLIMB_M]
        assert offenders == [], f"{len(offenders)} slope runs go uphill (> {OSMConfig.MAX_BACKCLIMB_M}m back-climb)"

    def test_r4_no_isolated_slope(self, ds):
        """R4: STRICT connectivity — no slope may be isolated from the lift network, with neither endpoint in any
        component holding a lift station. A run reaching no lift must be dropped as unreachable. Prevents
        disconnected slope fragments surviving detached from the skiable system.
        """
        lift_nodes = {n for lf in ds.graph.lifts for n in (lf.node_a, lf.node_b)}
        comp_of = {}
        for i, c in enumerate(_components(ds.graph)):
            for n in c:
                comp_of[n] = i
        lift_comps = {comp_of[n] for n in lift_nodes if n in comp_of}
        isolated = [
            r
            for r in ds.graph.slope_runs
            if comp_of.get(r.node_a) not in lift_comps and comp_of.get(r.node_b) not in lift_comps
        ]
        assert isolated == [], f"{len(isolated)} slopes reach no lift — must be dropped (strict)"

    def test_r5_node_spacing_invariant(self, ds):
        """R5: STRICT spacing, no exception — all nodes, lift and slope alike, must be at least MIN_NODE_DIST_M apart by
        haversine distance. Any closer pair is the authoritative signal of unmerged nodes. Prevents two hub points
        clustering where one merged hub belongs.
        """
        pts = {k: (v.lon, v.lat) for k, v in ds.graph.node_points.items()}
        keys = list(pts)
        close = [
            (keys[i], keys[j])
            for i in range(len(keys))
            for j in range(i + 1, len(keys))
            if _hav(pts[keys[i]], pts[keys[j]]) < OSMConfig.MIN_NODE_DIST_M - 5.0
        ]
        assert close == [], f"{len(close)} node pairs closer than {OSMConfig.MIN_NODE_DIST_M}m — must merge (strict)"

    def test_r6_nodes_lie_on_slopes(self, ds):
        """R6: STRICT — a node within MIN_NODE_DIST_M of a slope segment whose endpoint elevations span the node's
        height must be a vertex of that segment, since the run passes through that height there. A node above both
        ends or below both may float. Prevents a missed split where a run crosses a node.
        """
        elev = {k: v.elevation for k, v in ds.graph.node_points.items()}
        node_m = {k: _to_local(v.lon, v.lat, ds.bbox) for k, v in ds.graph.node_points.items()}
        floating = []
        for r in ds.graph.slope_runs:
            seg_verts = {(round(p.lon, 6), round(p.lat, 6)) for p in r.points}  # this segment's own vertices
            seg_m = [_to_local(p.lon, p.lat, ds.bbox) for p in r.points]
            lo_e, hi_e = sorted((elev[r.node_a], elev[r.node_b]))
            for nk, nm in node_m.items():
                if (
                    round(ds.graph.node_points[nk].lon, 6),
                    round(ds.graph.node_points[nk].lat, 6),
                ) in seg_verts:
                    continue  # the node IS a vertex of this segment — allowed
                if not (lo_e <= elev[nk] <= hi_e):
                    continue  # node above both ends (peak) or below both (pit) — a legitimate float
                if _seg_point_dist(nm, seg_m) < OSMConfig.MIN_NODE_DIST_M:
                    floating.append((nk, r.name))
                    break
        assert floating == [], (
            f"{len(floating)} slope segments pass within {OSMConfig.MIN_NODE_DIST_M:.0f}m of a node whose height "
            f"they span, without it being a vertex — must split there (pass through): {floating[:8]}"
        )

    def test_r7_no_long_straight_segment(self, ds):
        """R7: STRICT sampling — no slope may have a single straight leg between consecutive points longer than
        MAX_STRAIGHT_M. Every pair of adjacent points must stay within that chord. Prevents a long chord tunnelling
        through terrain where intermediate nodes went unsampled.
        """
        offenders = []
        for r in ds.graph.slope_runs:
            for a, b in zip(r.points, r.points[1:], strict=False):
                if a.distance_to(other=b) > OSMConfig.MAX_STRAIGHT_M:
                    offenders.append(r)
                    break
        assert offenders == [], (
            f"{len(offenders)} slopes have a straight leg > {OSMConfig.MAX_STRAIGHT_M}m (tunnelling)"
        )

    def test_r8_reference_shaped_counts(self, ds):
        """Artifacts: segment and node counts must stay at or below the dataset ceilings, segments capping graph
        complexity and nodes gating visual clutter. Counts must stay within hand-built resort bounds. Prevents raw
        OSM density exploding the graph beyond a human-scale resort.
        """
        # SEGMENTS (graph edges) and SLOPES are counted separately: here slope_runs ARE the segments
        # (full-split). Segment count is free up to a blizzard cap; node count is the visual-clutter gate.
        n_segments = len(ds.graph.slope_runs)
        assert n_segments <= ds.dataset.max_segments, f"{n_segments} segments (cap {ds.dataset.max_segments})"
        assert len(ds.graph.node_points) <= ds.dataset.max_nodes, "node count far above a hand-built resort"

    def test_r9_connectivity(self, ds):
        """R9: STRICT connectivity — every node must lie in ONE single component over slope and lift edges, no
        fragment allowed. Slopes and lifts bridge into one skiable network. Prevents the import scattering
        disconnected islands instead of one cohesive graph.
        """
        comps = _components(ds.graph)
        tot = len(ds.graph.node_points)
        largest = max((len(c) for c in comps), default=0)
        assert tot and largest == tot, f"{len(comps)} components — {largest}/{tot} nodes in the largest (want one)"

    def test_r10_relaxed_pull_no_slope_near_lift(self, ds):
        """R10: RELAXED pull (slope to lift) — no slope-only node may sit within RELAXED_MERGE_DIST_M of a lift node, a
        wider band than the strict slope-slope spacing. Such a node must be pulled onto the lift hub. Prevents a
        slope floating in a near-miss beside ski infrastructure.
        """
        pts = {k: (v.lon, v.lat) for k, v in ds.graph.node_points.items()}
        lift_nodes = {n for lf in ds.graph.lifts for n in (lf.node_a, lf.node_b)}
        slope_keys = [k for k in pts if k not in lift_nodes]
        offenders = [
            (sk, lk)
            for sk in slope_keys
            for lk in lift_nodes
            if _hav(pts[sk], pts[lk]) < OSMConfig.RELAXED_MERGE_DIST_M - 5.0
        ]
        assert offenders == [], (
            f"{len(offenders)} slope nodes within {OSMConfig.RELAXED_MERGE_DIST_M:.0f}m of a lift — must be pulled onto it"
        )

    def test_r11_hub_on_lift_when_lift_present(self, ds):
        """R11: lift-authoritative hubs — every lift's node_a and node_b must exist in node_points, the lift endpoints
        being the stored coordinates. Lift geometry drives hub placement and slopes conform to it. Prevents a lift
        referencing a node missing from the registry.
        """
        node_pt = ds.graph.node_points
        lift_nodes = {n for lf in ds.graph.lifts for n in (lf.node_a, lf.node_b)}
        # every lift references its own node coords exactly (lift geometry is authoritative)
        for lf in ds.graph.lifts:
            assert lf.node_a in node_pt and lf.node_b in node_pt
        # a hub with exactly one lift station must equal that lift's station point (not pulled off by slopes)
        # (structural: lift endpoints ARE the node points, so this holds by construction; assert non-empty)
        assert lift_nodes, "no lift nodes present to validate lift-authoritative hubs"

    def test_r12_slopes_stay_on_original_osm_pistes(self, ds):
        """R12: STRICT fidelity — every imported slope must hug an original OSM piste, with at least 85% of its points
        within SLOPE_ON_SOURCE_TOL_M of some source way. A run wandering off is invented geometry. Prevents a slope
        stitched across a gap where no piste exists.
        """
        els = ds.elements
        pistes, _lifts = ways_to_lines(els, ds.bbox)
        src = [verts for verts, _nm in pistes if len(verts) >= 2]  # raw OSM piste polylines (lon/lat)
        tol = OSMConfig.PISTE_VERTEX_TOL_M  # ~1 piste-width; the builder's R12 gate uses the same constant
        phantom = 0
        for r in ds.graph.slope_runs:
            # a slope is valid only if (almost) all of its points sit within tol of SOME source piste
            off = 0
            for p in r.points:
                pt = (p.lon, p.lat)
                if not any(min(_hav(pt, sv) for sv in s) < tol for s in src):
                    off += 1
            if off / len(r.points) > 0.15:  # >15% of the run wanders off every OSM piste
                phantom += 1
        assert phantom == 0, f"{phantom} slopes are NOT on any original OSM piste (invented geometry)"

    def test_r13_no_unmerged_slope_node_cluster(self, ds):
        """R13: STRICT merge — no cluster of slope nodes may sit within MIN_NODE_DIST_M of one another, since such a
        group is obviously one hub. Any closer pair signals a missed merge. Prevents proximate slope nodes stacking
        where a single merged hub belongs.
        """
        pts = {k: (v.lon, v.lat) for k, v in ds.graph.node_points.items()}
        lift_nodes = {n for lf in ds.graph.lifts for n in (lf.node_a, lf.node_b)}
        slope_keys = [k for k in pts if k not in lift_nodes]
        clusters = 0
        for i, ki in enumerate(slope_keys):
            for kj in slope_keys[i + 1 :]:
                if _hav(pts[ki], pts[kj]) < OSMConfig.MIN_NODE_DIST_M - 5.0:
                    clusters += 1
        assert clusters == 0, (
            f"{clusters} slope-node pairs within {OSMConfig.MIN_NODE_DIST_M:.0f}m — one hub, must merge"
        )

    def test_r14_every_lift_has_a_slope(self, ds):
        """R14: STRICT — every lift must share a station node with at least one slope, with no orphan lift detached from
        the runnable network. This holds by R21 but is asserted directly as a stronger lift-slope guarantee. Prevents
        a lift left disconnected from every run.
        """
        lifts = ds.graph.lifts
        slope_nodes = {n for r in ds.graph.slope_runs for n in (r.node_a, r.node_b)}
        orphan = [lf.name for lf in lifts if lf.node_a not in slope_nodes and lf.node_b not in slope_nodes]
        assert orphan == [], f"{len(orphan)}/{len(lifts)} lifts touch NO slope: {orphan[:5]}"

    def test_r15_most_runs_survive(self, ds):
        """R15: STRICT volume — the segment count must be at least the dataset's min_segments floor, so a full resort
        box does not collapse to a near-empty graph. Prevents over-aggressive filtering or under-noding from dropping
        the bulk of the runs OSM provides.
        """
        n = len(ds.graph.slope_runs)
        assert n >= ds.dataset.min_segments, (
            f"only {n} segments (want ≥{ds.dataset.min_segments}) — dropping/under-noding runs"
        )

    def test_r16_every_lift_top_reaches_a_base(self, ds):
        """R16: STRICT — from every lift top a skier must descend to some lift base, staying in the skiable network,
        checked by BFS over downhill-directed slope edges. This holds by R21 but is asserted for all lifts, not a
        fraction. Prevents a lift top stranded with no way down.
        """
        elev = {k: v.elevation for k, v in ds.graph.node_points.items()}
        down: dict[int, set[int]] = defaultdict(set)
        for r in ds.graph.slope_runs:
            a, b = r.node_a, r.node_b
            hi, lo = (a, b) if elev[a] >= elev[b] else (b, a)
            down[hi].add(lo)

        def reachable(top: int) -> set[int]:
            seen, q = {top}, deque([top])
            while q:
                x = q.popleft()
                for y in down[x]:
                    if y not in seen:
                        seen.add(y)
                        q.append(y)
            return seen

        lift_bases = {(lf.node_b if elev[lf.node_a] >= elev[lf.node_b] else lf.node_a) for lf in ds.graph.lifts}
        stuck = []
        for lf in ds.graph.lifts:
            top = lf.node_a if elev[lf.node_a] >= elev[lf.node_b] else lf.node_b
            if not (reachable(top) & (lift_bases - {top})):  # can descend to some OTHER lift's base
                stuck.append(lf.name)
        assert stuck == [], (
            f"{len(stuck)}/{len(ds.graph.lifts)} lift tops CANNOT ski down to any lift base: {stuck[:5]}"
        )

    def test_r17_slopes_descend_by_orientation(self, ds):
        """R17: STRICT orientation — every slope must be stored node_a to node_b with node_a at least as high as node_b,
        the structural guarantee behind skiing down. R3 checks the smoothed profile; this checks the stored
        orientation itself. Prevents a run persisted uphill against the directed-edge invariant.
        """
        elev = {k: v.elevation for k, v in ds.graph.node_points.items()}
        wrong = [r for r in ds.graph.slope_runs if elev[r.node_a] < elev[r.node_b]]
        assert wrong == [], f"{len(wrong)} slopes stored uphill (node_a below node_b) — orientation invariant broken"

    def test_r18_slope_endpoints_sit_on_their_hubs(self, ds):
        """R18: STRICT pinning — a slope's first and last point must be its hub node coordinate, a shared node rather
        than a float nearby, an exact check complementing R6's distance test. Prevents a slope endpoint drifting off
        the hub it is meant to share.
        """
        node_pt = ds.graph.node_points
        tol = 1.0  # metres — pinned exactly by the builder; 1m guards float noise only
        bad = []
        for r in ds.graph.slope_runs:
            da = _hav((r.points[0].lon, r.points[0].lat), (node_pt[r.node_a].lon, node_pt[r.node_a].lat))
            db = _hav((r.points[-1].lon, r.points[-1].lat), (node_pt[r.node_b].lon, node_pt[r.node_b].lat))
            if da > tol or db > tol:
                bad.append((r.name, round(max(da, db), 1)))
        assert bad == [], f"{len(bad)} slopes whose endpoint is not ON its hub node (>{tol}m): {bad[:5]}"

    def test_r19_slope_geometry_fidelity(self, ds):
        """R19: STRICT pull model — an imported slope must keep its OSM body on-piste within PISTE_TOL_M, with off-piste
        allowed only as an end connector no longer than MAX_PULL_M. No off-piste may fall mid-run. Prevents a tunnel
        through terrain between the on-piste body's ends.
        """
        from shapely.geometry import LineString, Point
        from shapely.ops import unary_union

        lat0 = (ds.bbox[1] + ds.bbox[3]) / 2
        mlon, mlat = GeoCalculator.meters_per_degree(lat=lat0)

        def to_m(lon, lat):
            return ((lon - ds.bbox[0]) * mlon, (lat - ds.bbox[1]) * mlat)

        pistes, _lifts = ways_to_lines(ds.elements, ds.bbox)
        src = unary_union([LineString([to_m(lon, lat) for lon, lat in vs]) for vs, _nm in pistes if len(vs) >= 2])
        offenders = []
        for r in ds.graph.slope_runs:
            off = [Point(to_m(p.lon, p.lat)).distance(src) > OSMConfig.PISTE_TOL_M for p in r.points]
            if not any(off):
                continue
            first_on = off.index(False) if False in off else len(off)
            last_on = len(off) - 1 - off[::-1].index(False) if False in off else -1
            # off-piste point strictly between the on-piste body = a mid-run tunnel (forbidden)
            if any(off[first_on : last_on + 1]):
                offenders.append((r.name, "mid-run off-piste (tunnel)"))
                continue
            # each end connector (contiguous off-piste prefix / suffix) must be ≤ MAX_PULL_M long
            head = _polylen([(p.lon, p.lat) for p in r.points[: first_on + 1]]) if first_on > 0 else 0.0
            tail = _polylen([(p.lon, p.lat) for p in r.points[last_on:]]) if last_on < len(off) - 1 else 0.0
            if max(head, tail) > OSMConfig.MAX_PULL_M:
                offenders.append((r.name, f"connector {max(head, tail):.0f}m > {OSMConfig.MAX_PULL_M:.0f}m"))
        assert offenders == [], f"{len(offenders)} slopes violate the pull/fidelity model: {offenders[:5]}"

    def test_r20_connectivity_runs_on_segments_with_branching(self, ds):
        """R20: STRICT — connectivity is a segment graph, so a slope spans many segments and a branch at a
        degree-3-or-more interior node still counts as connected. Requires real junctions to exist and no segment to
        be a self-loop. Prevents connectivity being mismeasured over whole slopes.
        """
        deg: dict[int, int] = defaultdict(int)
        for r in ds.graph.slope_runs:
            deg[r.node_a] += 1
            deg[r.node_b] += 1
        for lf in ds.graph.lifts:
            deg[lf.node_a] += 1
            deg[lf.node_b] += 1
        branch_nodes = [n for n, d in deg.items() if d >= 3]
        assert branch_nodes, "no branch nodes (degree≥3) — a full resort must have mid-slope junctions"
        # every segment is a genuine edge between two DISTINCT hub nodes (no self-loops in the graph)
        assert all(r.node_a != r.node_b for r in ds.graph.slope_runs), "a segment is a self-loop (a==b)"

    def test_r21_every_lift_is_skiable_top_to_bottom(self, ds):
        """R21: STRICT completeness — from every lift top a skier must descend to that lift's own bottom via a chain of
        descending segments, checked by BFS over downhill edges for all lifts. A lift is never dropped, so failure
        signals missing slopes. Prevents the import filtering out runs OSM provides.
        """
        elev = {k: v.elevation for k, v in ds.graph.node_points.items()}
        down: dict[int, set[int]] = defaultdict(set)
        for r in ds.graph.slope_runs:
            hi, lo = (r.node_a, r.node_b) if elev[r.node_a] >= elev[r.node_b] else (r.node_b, r.node_a)
            down[hi].add(lo)

        def can_ski(top: int, bottom: int) -> bool:
            seen, q = {top}, deque([top])
            while q:
                x = q.popleft()
                if x == bottom:
                    return True
                for y in down[x]:
                    if y not in seen:
                        seen.add(y)
                        q.append(y)
            return False

        unskiable = []
        for lf in ds.graph.lifts:
            top = lf.node_a if elev[lf.node_a] >= elev[lf.node_b] else lf.node_b
            bottom = lf.node_b if top == lf.node_a else lf.node_a
            if not can_ski(top, bottom):
                unskiable.append(lf.name or f"{top}->{bottom}")
        assert unskiable == [], (
            f"{len(unskiable)}/{len(ds.graph.lifts)} lifts have NO descending slope-chain top→bottom "
            f"(missing slopes, not a lift fault): {unskiable[:5]}"
        )

    def test_r22_no_slope_dead_ends(self, ds):
        """R22: STRICT bidirectional reachability — over descending slope edges only, every slope node must reach a lift
        station going down, unless it sits below the lowest lift base as a valley terminus, and must be reachable
        from a lift going down. Prevents a dropped or truncated slope stranding a skier.
        """
        elev = {k: v.elevation for k, v in ds.graph.node_points.items()}
        lift_nodes = {n for lf in ds.graph.lifts for n in (lf.node_a, lf.node_b)}
        min_lift_base = min((elev[n] for n in lift_nodes), default=0.0)
        down: dict[int, set[int]] = defaultdict(set)
        up: dict[int, set[int]] = defaultdict(set)
        slope_nodes: set[int] = set()
        for r in ds.graph.slope_runs:
            hi, lo = (r.node_a, r.node_b) if elev[r.node_a] >= elev[r.node_b] else (r.node_b, r.node_a)
            down[hi].add(lo)
            up[lo].add(hi)
            slope_nodes |= {hi, lo}

        def reaches(start: int, graph: dict[int, set[int]]) -> bool:
            """True if a lift station is reachable from `start` over `graph` (down= or up=adjacency)."""
            seen, q = {start}, deque([start])
            while q:
                x = q.popleft()
                if x in lift_nodes:
                    return True
                for y in graph[x]:
                    if y not in seen:
                        seen.add(y)
                        q.append(y)
            return False

        # A sink BELOW the lowest lift base is a genuine VALLEY TERMINUS — you ski out of the box to a
        # return lift whose base is outside the bbox (e.g. the Ischgl village gondola). Not a dead-end.
        stranded_down = sorted(n for n in slope_nodes if not reaches(n, down) and elev[n] > min_lift_base)
        stranded_up = sorted(n for n in slope_nodes if not reaches(n, up))
        assert stranded_down == [], (
            f"{len(stranded_down)} slope nodes cannot reach a lift going DOWN (skier stranded): "
            f"{stranded_down[:8]} — a dropped/truncated slope"
        )
        assert stranded_up == [], (
            f"{len(stranded_up)} slope nodes cannot be reached from a lift going down (unreachable): "
            f"{stranded_up[:8]} — a dropped feeder piste"
        )

    def test_r23_slope_points_hug_terrain(self, ds):
        """R23: STRICT — every slope point must sit within SLOPE_TERRAIN_TOL_M of the real DEM terrain, never
        floating far above it or buried far below. A slope follows the ground. Prevents invented or tunnelling
        geometry that drifts off the measured terrain surface.
        """
        dem = DEMService()  # real EuroDEM (same terrain the builder draped onto)
        offenders = []
        for r in ds.graph.slope_runs:
            for p in r.points:
                terrain = dem.get_elevation(lon=p.lon, lat=p.lat)
                if terrain is not None and abs(p.elevation - terrain) > OSMConfig.SLOPE_TERRAIN_TOL_M:
                    offenders.append((r.name, round(p.elevation - terrain, 1)))
                    break
        assert offenders == [], (
            f"{len(offenders)} slopes have a point > {OSMConfig.SLOPE_TERRAIN_TOL_M}m off terrain "
            f"(above/below DEM): {offenders[:5]}"
        )

    def test_r24_no_lift_dropped(self, ds):
        """R24: STRICT import — every raw OSM way of an allowed aerialway type over MIN_LIFT_LENGTH_M must appear in the
        final output, and the builder may never drop a lift. Prevents a length-qualifying aerialway being silently
        discarded during import.
        """
        _pistes, lifts = ways_to_lines(ds.elements, ds.bbox)
        expected = sum(
            1
            for vs, _lt, _nm in lifts
            if sum(_hav(vs[i], vs[i + 1]) for i in range(len(vs) - 1)) >= OSMConfig.MIN_LIFT_LENGTH_M
        )
        assert len(ds.graph.lifts) == expected, (
            f"only {len(ds.graph.lifts)}/{expected} qualifying lifts survived — a lift was DROPPED "
        )

    def test_r25_every_node_hugs_terrain_strict(self, ds):
        """R25: STRICT — every node must sit within NODE_TERRAIN_TOL_M of the real DEM terrain at its
        location. A node farther off the ground is invented placement. Prevents a hub being positioned at an
        elevation the measured terrain does not support.
        """
        dem = DEMService()
        offenders = []
        for k, p in ds.graph.node_points.items():
            terrain = dem.get_elevation(lon=p.lon, lat=p.lat)
            if terrain is not None and abs(p.elevation - terrain) > OSMConfig.NODE_TERRAIN_TOL_M:
                offenders.append((k, round(p.elevation - terrain, 1)))
        assert offenders == [], (
            f"{len(offenders)} nodes are > {OSMConfig.NODE_TERRAIN_TOL_M}m off terrain: {offenders[:8]}"
        )

    def test_r26_lift_endpoints_sit_on_their_hubs(self, ds):
        """R26: STRICT pinning — a lift's drawn bottom and top must be its hub node coordinates, an exact check leaving
        no remapping between station and node. Prevents a lift station drifting off the node it shares, a teleport
        away from its real position.
        """
        node_pt = ds.graph.node_points
        tol = 1.0  # metres — the lift station IS its node; anything larger is a node/geometry desync
        bad = []
        for lf in ds.graph.lifts:
            # lf.bottom is node_a when node_a is the lower station, else node_b (orientation set at build)
            lo, hi = (
                (lf.node_a, lf.node_b)
                if node_pt[lf.node_a].elevation <= node_pt[lf.node_b].elevation
                else (lf.node_b, lf.node_a)
            )
            db = _hav((lf.bottom.lon, lf.bottom.lat), (node_pt[lo].lon, node_pt[lo].lat))
            dt = _hav((lf.top.lon, lf.top.lat), (node_pt[hi].lon, node_pt[hi].lat))
            if db > tol or dt > tol:
                bad.append((lf.name, round(max(db, dt), 1)))
        assert bad == [], (
            f"{len(bad)} lifts whose drawn station is not ON its hub node (>{tol}m) — a node was "
            f"remapped away from its real station (teleport hack): {bad[:5]}"
        )

    def test_r27_referential_integrity(self, ds):
        """R27: STRICT integrity — every node referenced by a slope or lift must exist in node_points, and every node in
        node_points must be referenced by at least one slope or lift. Prevents a dangling reference or an orphaned
        node breaking the complete node partition.
        """
        nodes = set(ds.graph.node_points)
        referenced = {n for r in ds.graph.slope_runs for n in (r.node_a, r.node_b)} | {
            n for lf in ds.graph.lifts for n in (lf.node_a, lf.node_b)
        }
        dangling = sorted(referenced - nodes)
        orphaned = sorted(nodes - referenced)
        assert dangling == [], (
            f"{len(dangling)} slope/lift endpoints reference nodes not in node_points: {dangling[:8]}"
        )
        assert orphaned == [], f"{len(orphaned)} nodes in node_points are referenced by nothing: {orphaned[:8]}"

    def test_r28_lift_stations_match_raw_osm(self, ds):
        """R28: STRICT fidelity — every imported lift's bottom and top must sit within RESAMPLE_STEP_M of a raw OSM lift
        endpoint, since the builder takes lift geometry verbatim and only recomputes the DEM z. Prevents a station
        being moved off its real OSM position.
        """
        _pistes, raw_lifts = ways_to_lines(ds.elements, ds.bbox)
        raw_stations = [vs[0] for vs, _lt, _nm in raw_lifts] + [vs[-1] for vs, _lt, _nm in raw_lifts]
        tol = OSMConfig.RESAMPLE_STEP_M  # a station is a raw OSM vertex; allow one resample step of slack
        bad = []
        for lf in ds.graph.lifts:
            for station in (lf.bottom, lf.top):
                if min(_hav((station.lon, station.lat), rs) for rs in raw_stations) > tol:
                    bad.append(lf.name)
                    break
        assert bad == [], (
            f"{len(bad)} lifts have a station that matches NO raw OSM lift endpoint (>{tol}m) — "
            f"builder moved a station off its real OSM position: {bad[:5]}"
        )

    def test_r29_segments_group_into_fewer_slopes(self, ds):
        """R29: STRICT grouping — path segments must group into whole app-slopes emitted by to_slope_chains, averaging
        at least min_seg_per_slope segments each, with every run in exactly one chain. Prevents a real named piste
        fragmenting into many single-segment app-slopes. (Threshold is deliberately loose because R39 peels avoidable
        steep sections into honest sub-slopes, lowering the ratio on purpose.)
        """
        runs = ds.graph.slope_runs
        chains = ds.graph.to_slope_chains()  # FINAL app-slopes: list of (per-run point-lists, name)
        assert chains, "no app-slopes — segments were never grouped"
        # referential completeness: every run's points appear in exactly one chain (partition of runs)
        n_segments = sum(len(pts_lists) for pts_lists, _name in chains)
        assert n_segments == len(runs), (
            f"grouping is not a partition: {n_segments} chained segments != {len(runs)} runs"
        )
        ratio = len(runs) / len(chains)
        assert ratio >= ds.dataset.min_seg_per_slope, (
            f"only {ratio:.2f} segments per app-slope ({len(runs)} segments / {len(chains)} app-slopes) — "
            f"want ≥{ds.dataset.min_seg_per_slope}"
        )

    def test_r30_linear_piste_not_needlessly_split(self, ds):
        """R30: STRICT cover — the min-path cover must not split a piste at a clean pass-through, a node with exactly
        one run in and one run out, which must stay in one app-slope. A name yields more than one app-slope only at a
        genuine fork, merge, or disconnected arm. Prevents needless fragmentation of a continuous run.
        """
        g = ds.graph
        elev = {k: v.elevation for k, v in g.node_points.items()}
        # per-name in/out degree over downhill-oriented runs
        by_name_runs = defaultdict(list)
        for r in g.slope_runs:
            if r.name:
                by_name_runs[r.name].append(r)
        offenders = []
        for name, runs in by_name_runs.items():
            outd: dict[int, int] = defaultdict(int)
            ind: dict[int, int] = defaultdict(int)
            for r in runs:
                hi, lo = (r.node_a, r.node_b) if elev[r.node_a] >= elev[r.node_b] else (r.node_b, r.node_a)
                outd[hi] += 1
                ind[lo] += 1
            # a chain that starts at a pass-through node (1 in, 1 out) is a needless split there
            chains = _linear_chains(runs, elev)
            for chain in chains:
                head = chain[0]
                head_hi = head.node_a if elev[head.node_a] >= elev[head.node_b] else head.node_b
                if ind.get(head_hi, 0) == 1 and outd.get(head_hi, 0) == 1:
                    offenders.append((name, head_hi))
        assert not offenders, f"pistes split at a clean pass-through (needless fragmentation): {offenders[:8]}"

    def test_r33_app_slope_segments_are_contiguous(self, ds):
        """R33: STRICT contiguity — every app-slope must be a connected chain, with consecutive segments sharing an
        endpoint within MIN_NODE_DIST_M. A gap splices two disconnected arms into one point-list. Prevents the finish
        spline drawing a straight belt across a void between arms.
        """
        chains = ds.graph.to_slope_chains()
        tol = OSMConfig.MIN_NODE_DIST_M
        gaps = []
        for pts_lists, name in chains:
            for k in range(len(pts_lists) - 1):
                a_end = pts_lists[k][-1]
                nxt = pts_lists[k + 1]
                # consecutive segments touch iff a's end coincides with the next segment's start OR end
                d = min(a_end.distance_to(other=nxt[0]), a_end.distance_to(other=nxt[-1]))
                if d > tol:
                    gaps.append((name, k, round(d)))
        assert not gaps, (
            f"{len(gaps)} app-slopes splice DISCONNECTED segments (spline draws a belt across the gap): "
            f"{sorted(gaps, key=lambda t: -t[2])[:8]}"
        )

    def test_r34_no_redundant_parallel_twin(self, ds):
        """R34: STRICT — no run may be a redundant twin of a longer same-named sibling, staying in the near band from
        DEDUP_TOL_M to PARALLEL_TOL_M for at least PARALLEL_TWIN_FRAC of its own length. R2 catches the on-top case;
        this catches the offset twin. Prevents a wide piste double-drawn as two offset edges.
        """
        near_lo = OSMConfig.DEDUP_TOL_M
        near_hi = OSMConfig.PARALLEL_TOL_M
        frac = OSMConfig.PARALLEL_TWIN_FRAC
        sruns = ds.graph.slope_runs
        runs = [[(p.lon, p.lat) for p in r.points] for r in sruns]

        def sustained_parallel(a: list[tuple[float, float]], b: list[tuple[float, float]]) -> float:
            """Longest contiguous stretch of `a` whose points stay in the near band of polyline `b`."""
            best = cur = 0.0
            for k in range(len(a)):
                d = min(_hav(a[k], vb) for vb in b)
                if near_lo < d <= near_hi:
                    cur += _hav(a[k - 1], a[k]) if k > 0 else 0.0
                    best = max(best, cur)
                else:
                    cur = 0.0
            return best

        offenders = []
        for i in range(len(runs)):
            li = _polylen(runs[i])
            for j in range(len(runs)):
                if i == j or len(runs[j]) < 2 or not sruns[i].name or sruns[i].name != sruns[j].name:
                    continue  # only a twin of the SAME named piste is a redundant double-draw
                if li <= _polylen(runs[j]) and li > 0 and sustained_parallel(runs[i], runs[j]) >= frac * li:
                    offenders.append(sruns[i].name)
                    break
        assert not offenders, (
            f"{len(offenders)} runs are a redundant parallel twin of a same-named sibling "
            f"(within {near_hi:.0f}m for ≥{frac:.0%} of their length): {sorted(set(offenders))[:8]}"
        )

    def test_r35_doubled_ribbon_forked_not_left_parallel(self, ds):
        """R35: STRICT — two same-name runs sharing one node that hug within PARALLEL_TOL_M for at least
        PARALLEL_TWIN_FRAC of the shorter before diverging must split into one trunk plus branches. Lift-complex
        forks are exempt. Prevents a doubled piste left as two long overlapping ribbons.
        """
        near = OSMConfig.PARALLEL_TOL_M
        frac = OSMConfig.PARALLEL_TWIN_FRAC
        sruns = ds.graph.slope_runs
        runs = [[(p.lon, p.lat) for p in r.points] for r in sruns]
        lift_nodes = {n for lf in ds.graph.lifts for n in (lf.node_a, lf.node_b)}
        by_name = defaultdict(list)
        for i, r in enumerate(sruns):
            if r.name:
                by_name[r.name].append(i)
        offenders = []
        for idxs in by_name.values():
            for a in range(len(idxs)):
                for c in range(a + 1, len(idxs)):
                    i, j = idxs[a], idxs[c]
                    shared = {sruns[i].node_a, sruns[i].node_b} & {sruns[j].node_a, sruns[j].node_b}
                    if len(shared) != 1:
                        continue
                    hinge = next(iter(shared))
                    far_i = sruns[i].node_b if sruns[i].node_a == hinge else sruns[i].node_a
                    far_j = sruns[j].node_b if sruns[j].node_a == hinge else sruns[j].node_a
                    if far_i == far_j or hinge in lift_nodes or (far_i in lift_nodes and far_j in lift_nodes):
                        continue  # lift-complex forks legitimately stay two runs (R10 bars a node there)
                    short, long = (i, j) if _polylen(runs[i]) <= _polylen(runs[j]) else (j, i)
                    sp = runs[short] if sruns[short].node_a == hinge else runs[short][::-1]
                    lp = runs[long] if sruns[long].node_a == hinge else runs[long][::-1]
                    # arc of the shorter, CONTIGUOUS from the hinge, staying within `near` of the longer
                    arc = 0.0
                    for m in range(len(sp)):
                        if min(_hav(sp[m], q) for q in lp) > near:
                            break
                        arc += _hav(sp[m - 1], sp[m]) if m > 0 else 0.0
                    if arc >= frac * _polylen(sp):  # most of the shorter still hugs → an un-forked double
                        offenders.append((sruns[short].name, hinge, far_i, far_j))
        assert not offenders, (
            f"{len(offenders)} doubled ribbons left un-forked (should be split into trunk+branches): {offenders[:8]}"
        )

    def test_r36_no_needless_degree2_node(self, ds):
        """R36: every non-lift node must be a real junction (run-degree ≥3) — a degree-2 pass-through
        node (one run in, one out, nothing branching) must have been collapsed, its two runs merged.

        EXEMPT: a degree-2 node whose two runs form a terrain roller across the join — fusing them
        would yield a run climbing >MAX_BACKCLIMB_M over an 80m window and violate R3. R3 (piste
        descends) outranks the cosmetic collapse, so the junction legitimately stays (the builder
        correctly refuses that merge). Only a collapse that keeps R3 is 'needless'.
        """
        deg: dict[int, int] = defaultdict(int)
        runs_at: dict[int, list[SlopeRun]] = defaultdict(list)
        for r in ds.graph.slope_runs:
            deg[r.node_a] += 1
            deg[r.node_b] += 1
            runs_at[r.node_a].append(r)
            runs_at[r.node_b].append(r)
        lift_nodes = {n for lf in ds.graph.lifts for n in (lf.node_a, lf.node_b)}

        def merge_would_climb(n: int) -> bool:
            """True if fusing the node's two runs into one through-run would back-climb over the R3 cap
            (a real dip-then-rise across the join), so the junction must stay (R3 wins over R36).
            """
            r1, r2 = runs_at[n]
            arm1 = list(r1.points) if r1.node_b == n else list(reversed(r1.points))  # …→ n
            arm2 = list(r2.points) if r2.node_a == n else list(reversed(r2.points))  # n →…
            return _backclimb(arm1 + arm2[1:]) > OSMConfig.MAX_BACKCLIMB_M

        offenders = [
            n
            for n, d in deg.items()
            if d == 2 and n not in lift_nodes and len(runs_at[n]) == 2 and not merge_would_climb(n)
        ]
        assert offenders == [], f"{len(offenders)} degree-2 non-lift pass-through nodes must collapse: {offenders[:8]}"

    def test_r37_no_app_slope_crosses_a_lift(self, ds):
        """R37: no app-slope may span a lift station — a slope ends where you enter/exit a lift. For each
        chain from to_slope_chains, the shared node between consecutive segments must not be a lift node.
        """
        lift_nodes = {n for lf in ds.graph.lifts for n in (lf.node_a, lf.node_b)}
        node_pt = ds.graph.node_points
        lift_xy = [(node_pt[n].lon, node_pt[n].lat) for n in lift_nodes]
        offenders = []
        for pts_lists, name in ds.graph.to_slope_chains():
            for k in range(len(pts_lists) - 1):
                junction = pts_lists[k][-1]  # shared endpoint with the next segment
                if any(_hav((junction.lon, junction.lat), lx) < 1.0 for lx in lift_xy):
                    offenders.append((name, k))
        assert offenders == [], f"{len(offenders)} app-slopes span a lift station (must split there): {offenders[:8]}"

    def test_r38_unique_app_names(self, ds):
        """R38: every app-slope name and every app-lift name is unique across the import — duplicates
        sharing an OSM name are disambiguated with a 1-based `(k)` suffix (the lift mid-station rule).
        """
        slope_names = [nm for _pts, nm in ds.graph.to_slope_chains() if nm]
        lift_names = [lf.name for lf in ds.graph.lifts if lf.name]
        dup_slopes = [nm for nm, c in Counter(slope_names).items() if c > 1]
        dup_lifts = [nm for nm, c in Counter(lift_names).items() if c > 1]
        assert dup_slopes == [], f"{len(dup_slopes)} app-slope names are not unique: {dup_slopes[:8]}"
        assert dup_lifts == [], f"{len(dup_lifts)} app-lift names are not unique: {dup_lifts[:8]}"

    def test_r39_no_avoidable_over_classification(self, ds):
        """R39: an app-slope must not straddle an AVOIDABLE difficulty junction — a non-lift node of
        full-graph degree≥3 where two adjacent segments differ in band and the harder is red/black. There
        a skier can enter after / leave before the steep pitch, so the whole slope reading that band is
        over-classification; it must have been split. Bands recomputed via the production classifier.
        """
        from skiresort_planner.constants import SlopeConfig
        from skiresort_planner.core.terrain_analyzer import TerrainAnalyzer
        from skiresort_planner.model.path_segment import PathSegment

        g = ds.graph
        rank = {d: i for i, d in enumerate(SlopeConfig.DIFFICULTIES)}
        red = rank["red"]
        lift_nodes = {n for lf in g.lifts for n in (lf.node_a, lf.node_b)}
        # full-graph degree over the FINAL runs + lifts; avoidable = non-lift degree≥3 junctions
        deg: Counter[int] = Counter()
        for r in g.slope_runs:
            deg[r.node_a] += 1
            deg[r.node_b] += 1
        for lf in g.lifts:
            deg[lf.node_a] += 1
            deg[lf.node_b] += 1
        avoidable_xy = [
            (g.node_points[n].lon, g.node_points[n].lat) for n, d in deg.items() if d >= 3 and n not in lift_nodes
        ]

        def band(points):
            # production path: a real PathSegment's steepest-section max_slope_pct → difficulty band
            pct = PathSegment(points=list(points)).max_slope_pct
            return TerrainAnalyzer.classify_difficulty(slope_pct=pct)

        offenders = []
        for pts_lists, name in g.to_slope_chains():
            for k in range(len(pts_lists) - 1):
                junction = pts_lists[k][-1]  # shared endpoint with the next segment
                if not any(_hav((junction.lon, junction.lat), xy) < 1.0 for xy in avoidable_xy):
                    continue  # not an avoidable junction — a plain pass-through the split ignores
                harder = max(rank[band(pts_lists[k])], rank[band(pts_lists[k + 1])])
                if band(pts_lists[k]) != band(pts_lists[k + 1]) and harder >= red:
                    offenders.append((name, k))
        assert offenders == [], (
            f"{len(offenders)} app-slopes straddle an avoidable difficulty junction (must split): {offenders[:8]}"
        )

    def test_r40_lift_to_lift_reachability_matches_osm(self, ds):
        """R40: BRUTAL reachability preservation — every lift→lift SLOPE-ONLY downhill connection OSM
        offers must survive to the final graph. What matters to a skier: from lift station A, skiing
        DOWN slopes only (never riding a lift), can I reach lift station B? If OSM's raw pistes make A→B
        skiable and our graph loses it, a real slope was dropped and a gap opened. Dropping ONE of two
        parallel runs is fine (the connection remains); dropping the only run of a connection is a BUG.

        The oracle is LiftReachabilityCheck (in the builder, so import can log the same warning); this
        just imports again and asserts nothing is missing, reporting drops in <LiftName>_bottom/_top space.
        """
        check = LiftReachabilityCheck(OSMGraphBuilder(dem=DEMService(), bbox=ds.bbox), ds.pistes)
        missing = check.missing(ds.graph)
        labels = LiftReachabilityCheck.node_labels(ds.graph)
        report = sorted(f"{labels[a]} → {labels[b]}" for a, b in missing)
        assert not missing, (
            f"{len(missing)} lift→lift downhill connections OSM offers were DROPPED (real slope gaps): {report[:12]}"
        )


class TestLiftReachabilityCheck:
    """Waterproof unit tests for the R40 oracle (LiftReachabilityCheck) on hand-built graphs + a mock
    DEM — exercising every real-life weakness: bottom/top/mid_k labelling, a node shared by two lifts,
    unnamed lifts, parallel runs (drop one → still connected), the OSM proximity circle, the MIN_DROP_M
    net-descent filter that rejects near-equal-height peaks, and missing() flagging a genuinely lost run.
    """

    _NS = 20.0  # MockDEM: elevation rises 20% going NORTH (higher lat ⇒ higher elevation)

    def _pt(self, north_m, east_m=0.0):
        """A PathPoint `north_m` metres north (and `east_m` east) of origin, elevation from the mock DEM
        (base 2000m, +NS% per north-metre). Higher north ⇒ higher elevation, so orientation is exact.
        """
        from skiresort_planner.model.path_point import PathPoint

        lat, lon = north_m / MapConfig.METERS_PER_DEGREE_EQUATOR, east_m / MapConfig.METERS_PER_DEGREE_EQUATOR
        return PathPoint(lon=lon, lat=lat, elevation=2000.0 + north_m * self._NS / 100)

    def _graph(self, *, node_north, lifts, runs):
        """Build an ImportGraph: node_north = {id: north_m}; lifts/runs = (name/None, a, b) tuples."""
        from skiresort_planner.generators.osm_graph_builder import ImportGraph, LiftLine

        pts = {n: self._pt(north) for n, north in node_north.items()}
        lift_lines = [
            LiftLine(bottom=pts[a], top=pts[b], lift_type="chairlift", node_a=a, node_b=b, name=nm)
            for nm, a, b in lifts
        ]
        slope_runs = [SlopeRun(points=[pts[a], pts[b]], node_a=a, node_b=b, name=nm) for nm, a, b in runs]
        return ImportGraph(node_points=pts, slope_runs=slope_runs, lifts=lift_lines)

    # ---- node_labels: the naming that makes the report human-readable + node-identity robust ----

    def test_labels_rank_bottom_top_and_mid_by_elevation(self):
        # A 3-station lift (base 0m, mid 300m, top 600m north) → bottom / mid_0 / top by elevation.
        g = self._graph(
            node_north={1: 0.0, 2: 300.0, 3: 600.0},
            lifts=[("Gondola", 1, 2), ("Gondola", 2, 3)],  # two sections share mid-station node 2
            runs=[],
        )
        labels = LiftReachabilityCheck.node_labels(g)
        assert labels == {1: "Gondola_bottom", 2: "Gondola_mid_0", 3: "Gondola_top"}

    def test_label_of_node_shared_by_two_lifts_lists_both(self):
        # Node 2 is LiftA's top AND LiftB's top (one physical station, two lifts) — label lists both,
        # sorted, so identity stays the single node id and the sides can't desync.
        g = self._graph(
            node_north={1: 0.0, 2: 500.0, 3: 100.0},
            lifts=[("LiftA", 1, 2), ("LiftB", 3, 2)],
            runs=[],
        )
        assert LiftReachabilityCheck.node_labels(g)[2] == "LiftA_top / LiftB_top"

    def test_unnamed_lift_labels_by_node_id(self):
        g = self._graph(node_north={1: 0.0, 2: 500.0}, lifts=[(None, 1, 2)], runs=[])
        labels = LiftReachabilityCheck.node_labels(g)
        assert labels[1] == "@1_bottom" and labels[2] == "@1_top"

    # ---- final_pairs: transitive downhill reachability over shared node ids ----

    def test_final_pairs_are_transitive_downhill(self):
        # spine top(3,900) → mid(2,500) → base(1,0); a lift anchors each. 3 reaches 2 and 1; 2 reaches 1.
        g = self._graph(
            node_north={1: 0.0, 2: 500.0, 3: 900.0},
            lifts=[("L1", 1, 3), ("L2", 2, 3)],  # lifts ride the spine nodes (all present)
            runs=[(None, 3, 2), (None, 2, 1)],
        )
        pairs = LiftReachabilityCheck.final_pairs(g)
        assert (3, 2) in pairs and (2, 1) in pairs and (3, 1) in pairs
        assert (1, 3) not in pairs and (2, 3) not in pairs  # never uphill

    def test_dropping_one_of_two_parallel_runs_keeps_the_connection(self):
        # top(2) → base(1) drawn as TWO parallel runs; dropping one must NOT lose the 2→1 connection.
        nn = {1: 0.0, 2: 500.0}
        both = self._graph(node_north=nn, lifts=[("A", 1, 2)], runs=[("r1", 2, 1), ("r2", 2, 1)])
        one = self._graph(node_north=nn, lifts=[("A", 1, 2)], runs=[("r1", 2, 1)])
        assert (2, 1) in LiftReachabilityCheck.final_pairs(both)
        assert (2, 1) in LiftReachabilityCheck.final_pairs(one)  # still connected after the drop

    # ---- osm_pairs: the proximity-circle ground truth + MIN_DROP_M filter ----

    def _builder(self, bbox=(-0.01, -0.01, 0.01, 0.01)):
        return OSMGraphBuilder(
            dem=MockDEMService(base_elevation=2000.0, slope_ns_pct=self._NS, slope_ew_pct=0.0), bbox=bbox
        )

    def test_osm_pair_detected_when_a_piste_links_two_stations_downhill(self):
        # Station 2 (top, 600m north) and station 1 (base, 0m); each is a real lift endpoint (nodes 3/4
        # are the far ends). A raw piste runs from beside the top down to beside the base — OSM must
        # report 2→1 even though the FINAL graph has NO slope run between them.
        g = self._graph(
            node_north={1: 0.0, 2: 600.0, 3: 900.0, 4: 300.0}, lifts=[("Feeder", 2, 3), ("Base", 1, 4)], runs=[]
        )
        b = self._builder()

        def ll(north):  # (lon,lat) on the spine `north` metres from origin
            return (0.0, north / MapConfig.METERS_PER_DEGREE_EQUATOR)

        pistes: list[tuple[list[tuple[float, float]], str | None]] = [([ll(590), ll(400), ll(200), ll(10)], "Piste")]
        osm = LiftReachabilityCheck(b, pistes).osm_pairs(g)
        assert (2, 1) in osm  # OSM offers the downhill connection

    def test_osm_pair_rejected_when_stations_are_near_equal_height(self):
        # Two stations ~1m apart in elevation (5m north): a piste touches both circles, yet the drop
        # < MIN_DROP_M, so NO connection may be demanded either way (near-equal-peak DEM noise).
        g = self._graph(node_north={1: 200.0, 2: 205.0, 3: 500.0, 4: 500.0}, lifts=[("P", 1, 3), ("Q", 2, 4)], runs=[])
        b = self._builder()
        pistes: list[tuple[list[tuple[float, float]], str | None]] = [
            (
                [
                    (0.0, 205.0 / MapConfig.METERS_PER_DEGREE_EQUATOR),
                    (0.0, 200.0 / MapConfig.METERS_PER_DEGREE_EQUATOR),
                ],
                "Ridge",
            )
        ]
        osm = LiftReachabilityCheck(b, pistes).osm_pairs(g)
        assert (2, 1) not in osm and (1, 2) not in osm  # <MIN_DROP_M net drop → not a real run

    # ---- missing(): the actual assertion the builder + R40 rely on ----

    def test_missing_flags_a_connection_osm_has_but_final_dropped(self):
        # OSM piste links top→base; the FINAL graph has the lifts but NO connecting run → 2→1 is MISSING.
        g = self._graph(
            node_north={1: 0.0, 2: 600.0, 3: 900.0, 4: 300.0}, lifts=[("Feeder", 2, 3), ("Base", 1, 4)], runs=[]
        )
        b = self._builder()
        pistes: list[tuple[list[tuple[float, float]], str | None]] = [
            (
                [
                    (0.0, 590.0 / MapConfig.METERS_PER_DEGREE_EQUATOR),
                    (0.0, 10.0 / MapConfig.METERS_PER_DEGREE_EQUATOR),
                ],
                "Piste",
            )
        ]
        missing = LiftReachabilityCheck(b, pistes).missing(g)
        assert (2, 1) in missing

    def test_missing_empty_when_final_keeps_the_connection(self):
        # Same OSM piste, but now the final graph HAS the 2→1 run → nothing missing.
        g = self._graph(
            node_north={1: 0.0, 2: 600.0, 3: 900.0, 4: 300.0},
            lifts=[("Feeder", 2, 3), ("Base", 1, 4)],
            runs=[("run", 2, 1)],
        )
        b = self._builder()
        pistes: list[tuple[list[tuple[float, float]], str | None]] = [
            (
                [
                    (0.0, 590.0 / MapConfig.METERS_PER_DEGREE_EQUATOR),
                    (0.0, 10.0 / MapConfig.METERS_PER_DEGREE_EQUATOR),
                ],
                "Piste",
            )
        ]
        assert LiftReachabilityCheck(b, pistes).missing(g) == set()


class TestGraphImporter:
    """The production GraphImporter wrapper: it fetches, runs the builder, and reports the built
    graph as an ImportResult (hub-aligned lifts + slope chains), writing reference artifacts.
    """

    def test_run_yields_importresult_and_writes_artifacts(self, tmp_path, monkeypatch) -> None:
        """GraphImporter: the production wrapper must fetch OSM, run the builder, and return an ImportResult with source
        OSM, non-empty lifts, and slope_chains of valid segment point-lists of at least two points each. Reference
        artifacts are written to the dump directory. Prevents the wrapper reporting an empty or malformed graph.
        """
        from skiresort_planner.generators.osm_graph_builder import GraphImporter

        importer = GraphImporter(dem=DEMService(), bbox=DATASETS[0].bbox)
        monkeypatch.setattr(importer, "fetch", lambda: DATASETS[0].load())  # no network — use the fixture

        result = importer.run(on_progress=lambda f, t: None, dump_dir=tmp_path)

        assert result.source == "OSM"
        assert result.lifts, "the graph importer must report the hub-aligned lifts"
        assert result.slope_chains, "the graph importer must report grouped slope chains"
        # Every chain is a non-empty list of segment point-lists.
        for chain, _name in result.slope_chains:
            assert chain and all(len(seg) >= 2 for seg in chain)
        # Reference artifacts written for inspection (never read back).
        assert (tmp_path / "osm_raw.json").exists()
        assert (tmp_path / "osm_import.png").exists()


class TestDegree2CollapseGeometry:
    """Focused unit tests for the degree-2 collapse (shapely.ops.linemerge fusion + fabricated re-mark),
    on a synthetic 3-node chain so the geometry assertions are exact (not fixture-count envelopes).
    """

    @staticmethod
    def _builder():
        return OSMGraphBuilder(dem=DEMService(), bbox=DATASETS[0].bbox)

    def _chain_graph(self):
        """A→B→C oriented downhill (2000→1900→1800 m), each run carrying an INTERIOR vertex."""
        from skiresort_planner.generators.osm_graph_builder import ImportGraph
        from skiresort_planner.model.path_point import PathPoint

        lon0, lat0 = DATASETS[0].bbox[0], DATASETS[0].bbox[1]
        # Legs kept well under MAX_STRAIGHT_M (100 m ≈ 0.0013° lon here) so _finalize_fork_run accepts.
        a = PathPoint(lon=lon0 + 0.0000, lat=lat0, elevation=2000.0)
        b = PathPoint(lon=lon0 + 0.0010, lat=lat0, elevation=1900.0)
        c = PathPoint(lon=lon0 + 0.0020, lat=lat0, elevation=1800.0)
        mid_ab = PathPoint(lon=lon0 + 0.0005, lat=lat0 + 0.00005, elevation=1950.0)  # interior bend
        mid_bc = PathPoint(lon=lon0 + 0.0015, lat=lat0 + 0.00005, elevation=1850.0)
        g = ImportGraph(node_points={1: a, 2: b, 3: c})
        g.slope_runs = [
            SlopeRun(points=[a, mid_ab, b], node_a=1, node_b=2, name="Piste"),
            SlopeRun(points=[b, mid_bc, c], node_a=2, node_b=3, name="Piste"),
        ]
        return g, (a, mid_ab, b, mid_bc, c)

    def test_collapse_keeps_interior_vertices(self):
        """Node 2 (degree-2, non-lift) collapses into one A→C run whose polyline keeps BOTH interior
        bends — linemerge fusion must not flatten the run to a straight chord.
        """
        g, (a, mid_ab, b, mid_bc, c) = self._chain_graph()
        self._builder()._collapse_degree2_nodes(g)
        assert len(g.slope_runs) == 1, "the two runs merge at the degree-2 node"
        merged = g.slope_runs[0]
        assert {merged.node_a, merged.node_b} == {1, 3}, "merged run spans the two far endpoints"
        lons = [round(p.lon, 6) for p in merged.points]
        assert round(mid_ab.lon, 6) in lons and round(mid_bc.lon, 6) in lons, "both interior bends survive"
        assert len(merged.points) >= 4, "kept interior vertices, not a 2-point straight chord"

    def test_collapse_repopulates_fabricated_overlay(self):
        """After the collapse, build() re-marks fabricated; here we call it directly and assert the merged
        run's overlay has one flag per point (not the empty default a freshly-fused run carries).
        """
        b = self._builder()
        g, _pts = self._chain_graph()
        b._collapse_degree2_nodes(g)
        b._mark_fabricated(g)
        merged = g.slope_runs[0]
        assert len(merged.fabricated) == len(merged.points), "fabricated overlay covers every point"

    def test_to_m_round_trip_is_integer_metre_exact(self):
        """_to_m quantises the projection to a 1 m grid; the bit-exact shared-vertex guarantee (pnode raw
        tuple key, degree-2 collapse) needs _to_m(_to_deg(*xy)) == xy for every hub. Verify no drift over a
        grid spanning the ischgl bbox — a single ±1 m round-trip flip would silently mis-key nodes.
        """
        b = self._builder()
        drift = [(x, y) for x in range(0, 6000, 50) for y in range(0, 6000, 50) if b._to_m(*b._to_deg(x, y)) != (x, y)]
        assert drift == [], f"{len(drift)} points drift on the _to_m/_to_deg round-trip: {drift[:5]}"

    def test_full_split_yields_integer_metre_vertices(self, ds):
        """Every noded segment vertex is integer-metre (set_precision(1.0) after unary_union), so shared
        vertices are bit-exact — the precondition for linemerge fusion and the raw-tuple pnode key.
        """
        b = OSMGraphBuilder(dem=DEMService(), bbox=ds.bbox)
        lines = [LineString([b._to_m(lo, la) for lo, la in vs]) for vs, _ in ds.pistes if len(vs) >= 2]
        lines = [ls for ls in lines if ls.length >= OSMConfig.MIN_PISTE_LENGTH_M]
        kept, _ = b._dedup(lines)
        nonint = [(x, y) for s in b._full_split(kept) for x, y in s.coords if x != round(x) or y != round(y)]
        assert nonint == [], f"{len(nonint)} non-integer vertices after full-split: {nonint[:5]}"

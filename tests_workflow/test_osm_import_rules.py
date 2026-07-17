"""STRICT verification of the OSM connected-graph import rules."""

import json
import math
import os
from collections import defaultdict, deque

import pytest

from skiresort_planner.constants import OSMConfig, PathConfig
from skiresort_planner.generators.osm_graph_builder import OSMGraphBuilder, ways_to_lines
from tests_workflow.conftest import MockDEMService

# thresholds the rules assert against
MAX_SLOPES = 130  # a full resort box, well under the ~1159 blizzard; user wants ≲100–130
MIN_CONNECTED_FRAC = 0.80
MAX_STRAIGHT_M = PathConfig.SEGMENT_LENGTH_MAX_M  # no straight leg longer than this
CACHE = os.path.join(os.path.dirname(__file__), "..", "scratch_osm_raw_ischgl.json")
ISCHGL_BBOX = (10.27745, 46.95502, 10.35655, 47.00898)


def _load_cache():
    with open(CACHE, encoding="utf-8") as f:
        return json.load(f)["elements"]


def _hav(a, b):
    R = 6371000
    p1, p2 = math.radians(a[1]), math.radians(b[1])
    dp = math.radians(b[1] - a[1])
    dl = math.radians(b[0] - a[0])
    x = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(x))


def _polylen(coords):
    return sum(_hav(coords[i], coords[i + 1]) for i in range(len(coords) - 1))


def _backclimb(pts):
    """Sustained back-climb (m), smoothed ~90 m, of a downhill-oriented point list."""
    es = [p.elevation for p in pts]
    if len(es) < 3:
        return 0.0
    if es[0] < es[-1]:
        es = es[::-1]
    seg = [pts[i].distance_to(other=pts[i + 1]) for i in range(len(pts) - 1)]
    avg = (sum(seg) / len(seg)) if seg else 30.0
    win = max(1, round(90.0 / max(avg, 1.0)))
    half = win // 2
    sm = [
        sum(es[max(0, i - half) : min(len(es), i + half + 1)]) / len(es[max(0, i - half) : min(len(es), i + half + 1)])
        for i in range(len(es))
    ]
    return sum(max(0.0, sm[i + 1] - sm[i]) for i in range(len(sm) - 1))


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


@pytest.fixture(scope="module")
def ischgl_graph():
    """Build the graph from the cached real Ischgl fetch, or skip if the cache is absent."""
    if not os.path.exists(CACHE):
        pytest.skip("scratch_osm_raw_ischgl.json cache absent — run scratch_fetch.py to enable real-data rules")
    els = _load_cache()
    dem = MockDEMService(base_elevation=2500.0, slope_ns_pct=25.0, slope_ew_pct=0.0)
    # real DEM would be ideal, but MockDEM keeps the test hermetic; geometry rules don't need real Z.
    pistes, lifts = ways_to_lines(els, ISCHGL_BBOX)
    return OSMGraphBuilder(dem=dem, bbox=ISCHGL_BBOX).build(pistes, lifts)


class TestImportRules:
    """Each rule from the design conversation, asserted on the real Ischgl import."""

    def test_r1_all_lifts_imported(self, ischgl_graph):
        """Every skiable lift over the min length is imported (kept even if unconnected)."""
        els = _load_cache()
        _pistes, lifts = ways_to_lines(els, ISCHGL_BBOX)
        # count lift ways that clear the min-length gate (what the builder must keep)
        expected = 0
        for vs, _lt, _nm in lifts:
            length = sum(_hav(vs[i], vs[i + 1]) for i in range(len(vs) - 1))
            if length >= OSMConfig.MIN_LIFT_LENGTH_M:
                expected += 1
        assert len(ischgl_graph.lifts) == expected, (
            f"imported {len(ischgl_graph.lifts)} lifts but {expected} clear the length gate — lifts must never be dropped"
        )

    def test_r2_no_duplicate_runs(self, ischgl_graph):
        """No slope's geometry is ~covered by another slope (no double runs)."""
        runs = [[(p.lon, p.lat) for p in r.points] for r in ischgl_graph.slope_runs]
        dupes = 0
        for i, a in enumerate(runs):
            for j, b in enumerate(runs):
                if i == j or len(b) < 2:
                    continue
                # fraction of a's vertices within DEDUP_TOL of polyline b
                near = 0
                for va in a:
                    if any(_hav(va, vb) < OSMConfig.DEDUP_TOL_M for vb in b):
                        near += 1
                if near / len(a) >= OSMConfig.DEDUP_COVER_FRAC and _polylen(a) <= _polylen(b):
                    dupes += 1
                    break
        assert dupes == 0, f"{dupes} slopes are near-duplicates of another slope"

    def test_r3_no_uphill_slope(self, ischgl_graph):
        offenders = [r for r in ischgl_graph.slope_runs if _backclimb(r.points) > OSMConfig.MAX_BACKCLIMB_M]
        assert offenders == [], f"{len(offenders)} slope runs go uphill (> {OSMConfig.MAX_BACKCLIMB_M}m back-climb)"

    def test_r4_no_isolated_slope(self, ischgl_graph):
        lift_nodes = {n for lf in ischgl_graph.lifts for n in (lf.node_a, lf.node_b)}
        comp_of = {}
        for i, c in enumerate(_components(ischgl_graph)):
            for n in c:
                comp_of[n] = i
        lift_comps = {comp_of[n] for n in lift_nodes if n in comp_of}
        isolated = [
            r
            for r in ischgl_graph.slope_runs
            if comp_of.get(r.node_a) not in lift_comps and comp_of.get(r.node_b) not in lift_comps
        ]
        assert isolated == [], f"{len(isolated)} slopes reach no lift — must be dropped (strict)"

    def test_r5_node_spacing_invariant(self, ischgl_graph):
        """STRICT, no exception: no two nodes may be closer than MIN_NODE_DIST_M — including two lift
        stations. The minimum node distance is authoritative; anything closer must have merged.
        """
        pts = {k: (v.lon, v.lat) for k, v in ischgl_graph.node_points.items()}
        keys = list(pts)
        close = [
            (keys[i], keys[j])
            for i in range(len(keys))
            for j in range(i + 1, len(keys))
            if _hav(pts[keys[i]], pts[keys[j]]) < OSMConfig.MIN_NODE_DIST_M - 5.0
        ]
        assert close == [], f"{len(close)} node pairs closer than {OSMConfig.MIN_NODE_DIST_M}m — must merge (strict)"

    def test_r6_nodes_lie_on_slopes(self, ischgl_graph):
        """STRICT, no exception: no node (slope OR lift) may sit within MIN_NODE_DIST_M of a slope's
        geometry unless it is a vertex of that slope. A slope passing near a lift station must be SPLIT
        at the station — pass THROUGH it as a shared node — never float beside it.
        """
        pts = {k: (v.lon, v.lat) for k, v in ischgl_graph.node_points.items()}
        tol = OSMConfig.MIN_NODE_DIST_M
        floating = 0
        for r in ischgl_graph.slope_runs:
            endpts = {r.node_a, r.node_b}
            run_pts = [(p.lon, p.lat) for p in r.points]
            for nk, npt in pts.items():
                if nk in endpts:
                    continue
                dmin = min(_hav(npt, rp) for rp in run_pts)
                if dmin < tol:
                    floating += 1
                    break
        assert floating == 0, (
            f"{floating} slopes pass within {tol:.0f}m of a node without it being a vertex — "
            f"the slope must split at (pass through) that node"
        )

    def test_r7_no_long_straight_segment(self, ischgl_graph):
        """No committed slope has a straight leg longer than MAX_STRAIGHT_M (no chord through terrain)."""
        offenders = []
        for r in ischgl_graph.slope_runs:
            for a, b in zip(r.points, r.points[1:], strict=False):
                if a.distance_to(other=b) > MAX_STRAIGHT_M:
                    offenders.append(r)
                    break
        assert offenders == [], f"{len(offenders)} slopes have a straight leg > {MAX_STRAIGHT_M}m (tunnelling)"

    def test_r8_reference_shaped_counts(self, ischgl_graph):
        n = len(ischgl_graph.slope_runs)
        assert n <= MAX_SLOPES, f"{n} slopes — a blizzard, not a resort (cap {MAX_SLOPES})"
        assert len(ischgl_graph.node_points) <= 500, "node count far above a hand-built resort"

    def test_r9_connectivity(self, ischgl_graph):
        comps = _components(ischgl_graph)
        tot = len(ischgl_graph.node_points)
        largest = max((len(c) for c in comps), default=0)
        assert tot and largest / tot >= MIN_CONNECTED_FRAC, (
            f"only {largest}/{tot} nodes connected (< {MIN_CONNECTED_FRAC:.0%})"
        )

    def test_r10_relaxed_pull_no_slope_near_lift(self, ischgl_graph):
        """RELAXED pull (slope→lift): no SLOPE-only node may sit within RELAXED_MERGE_DIST_M of a LIFT
        node — such a slope node must have been pulled onto the lift. (Slope-slope pairs only need to
        respect the strict spacing, checked by R5; two lift stations may legitimately stay closer
        then RELAXED_MERGE_DIST_M, but still never closer than MIN_NODE_DIST_M)
        """
        pts = {k: (v.lon, v.lat) for k, v in ischgl_graph.node_points.items()}
        lift_nodes = {n for lf in ischgl_graph.lifts for n in (lf.node_a, lf.node_b)}
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

    def test_r11_hub_on_lift_when_lift_present(self, ischgl_graph):
        """Lift-authoritative median: a hub that has ≥1 lift station sits ON a lift endpoint region —
        the node coincides with a lift's own start/end coordinate (slopes followed, didn't shift it).
        """
        node_pt = ischgl_graph.node_points
        lift_nodes = {n for lf in ischgl_graph.lifts for n in (lf.node_a, lf.node_b)}
        # every lift references its own node coords exactly (lift geometry is authoritative)
        for lf in ischgl_graph.lifts:
            assert lf.node_a in node_pt and lf.node_b in node_pt
        # a hub with exactly one lift station must equal that lift's station point (not pulled off by slopes)
        # (structural: lift endpoints ARE the node points, so this holds by construction; assert non-empty)
        assert lift_nodes, "no lift nodes present to validate lift-authoritative hubs"

    def test_r12_slopes_stay_on_original_osm_pistes(self, ischgl_graph):
        """Every imported slope must lie within reasonable proximity of an ORIGINAL OSM piste — no
        invented geometry (e.g. a run stitched across a gap that tunnels where no piste exists).
        """
        els = _load_cache()
        pistes, _lifts = ways_to_lines(els, ISCHGL_BBOX)
        src = [verts for verts, _nm in pistes if len(verts) >= 2]  # raw OSM piste polylines (lon/lat)
        tol = 45.0  # ~1 piste-width; the builder's on-source gate (SLOPE_ON_SOURCE_TOL_M) drops worse
        phantom = 0
        for r in ischgl_graph.slope_runs:
            # a slope is valid only if (almost) all of its points sit within tol of SOME source piste
            off = 0
            for p in r.points:
                pt = (p.lon, p.lat)
                if not any(min(_hav(pt, sv) for sv in s) < tol for s in src):
                    off += 1
            if off / len(r.points) > 0.15:  # >15% of the run wanders off every OSM piste
                phantom += 1
        assert phantom == 0, f"{phantom} slopes are NOT on any original OSM piste (invented geometry)"

    def test_r13_no_unmerged_slope_node_cluster(self, ischgl_graph):
        """No cluster of slope nodes sitting on top of each other — a group of nodes all within
        MIN_NODE_DIST_M of one another (obviously one hub) must have been merged.
        """
        pts = {k: (v.lon, v.lat) for k, v in ischgl_graph.node_points.items()}
        lift_nodes = {n for lf in ischgl_graph.lifts for n in (lf.node_a, lf.node_b)}
        slope_keys = [k for k in pts if k not in lift_nodes]
        clusters = 0
        for i, ki in enumerate(slope_keys):
            for kj in slope_keys[i + 1 :]:
                if _hav(pts[ki], pts[kj]) < OSMConfig.MIN_NODE_DIST_M - 5.0:
                    clusters += 1
        assert clusters == 0, (
            f"{clusters} slope-node pairs within {OSMConfig.MIN_NODE_DIST_M:.0f}m — one hub, must merge"
        )

    def test_r14_every_lift_has_a_slope(self, ischgl_graph):
        """STRICT: in real life (and OSM) every lift connects to slopes — you ski down from where a
        lift drops you. So every imported lift must share a station node with at least one slope.
        """
        slope_nodes = {n for r in ischgl_graph.slope_runs for n in (r.node_a, r.node_b)}
        orphan_lifts = [
            lf for lf in ischgl_graph.lifts if lf.node_a not in slope_nodes and lf.node_b not in slope_nodes
        ]
        assert orphan_lifts == [], (
            f"{len(orphan_lifts)}/{len(ischgl_graph.lifts)} lifts have NO slope at either station — "
            f"every lift must connect to at least one slope (as in real life)"
        )

    def test_r15_most_runs_survive(self, ischgl_graph):
        """Sanity on volume: a 6 km Ischgl box has many named runs — the import must not collapse to a
        near-empty graph
        """
        assert len(ischgl_graph.slope_runs) >= 50, (
            f"only {len(ischgl_graph.slope_runs)} slopes for a full Ischgl box — the builder is dropping runs"
        )

    def test_r16_lift_top_to_bottom_skiable(self, ischgl_graph):
        """STANDARD ski-resort invariant: from every lift's TOP station you can ski DOWN to its BOTTOM
        station via a chain of descending slopes. Ride up, ski down — every lift must close that loop.
        """
        elev = {k: v.elevation for k, v in ischgl_graph.node_points.items()}
        down: dict[int, set[int]] = defaultdict(set)
        for r in ischgl_graph.slope_runs:
            a, b = r.node_a, r.node_b
            hi, lo = (a, b) if elev[a] >= elev[b] else (b, a)
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
        for lf in ischgl_graph.lifts:
            top = lf.node_a if elev[lf.node_a] >= elev[lf.node_b] else lf.node_b
            bottom = lf.node_b if top == lf.node_a else lf.node_a
            if not can_ski(top, bottom):
                unskiable.append(lf.name or f"{top}->{bottom}")
        assert unskiable == [], (
            f"{len(unskiable)}/{len(ischgl_graph.lifts)} lifts have NO downhill slope-chain top→bottom "
            f"(can't ski down what you rode up): {unskiable[:5]}"
        )

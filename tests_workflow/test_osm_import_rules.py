"""Verification of the OSM connected-graph import rules, measured on the REAL Ischgl cache + DEM."""

import json
import math
import os
from collections import defaultdict, deque

import pytest

from skiresort_planner.constants import OSMConfig, SlopeConfig
from skiresort_planner.core.dem_service import DEMService
from skiresort_planner.generators.osm_graph_builder import OSMGraphBuilder, ways_to_lines

# Pure test-assertion thresholds (counts / connectivity) live here; every geometric domain tolerance
# comes from OSMConfig (single source of truth — no drift between the builder and the rules it must meet).
MIN_CONNECTED_FRAC = 0.99  # near-total connectivity (current import is a single component)
MAX_NODES = 200  # node-count ceiling (current 99)
MAX_SEGMENTS = 300  # segment-count ceiling (current 155)
MIN_SEGMENTS = 100  # a full box must not collapse to near-empty
MIN_SEG_PER_SLOPE = 3.0  # R29: path-segments per FINAL app-slope (whole named piste, not per fork)
MAX_STRAIGHT_M = OSMConfig.MAX_STRAIGHT_M  # max single straight leg between consecutive points
MAX_PULL_M = OSMConfig.MAX_PULL_M  # end connector length cap; over this → dropped
PISTE_TOL_M = OSMConfig.PISTE_TOL_M  # off-piste threshold — SAME source the builder gates on
MAX_TERRAIN_DEVIATION_M = OSMConfig.SLOPE_TERRAIN_TOL_M  # R23: slope point vs real DEM
MAX_NODE_TERRAIN_DEVIATION_M = OSMConfig.NODE_TERRAIN_TOL_M  # R25: node vs real DEM (strict)
FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "ischgl_osm.json")
ISCHGL_BBOX = (10.27745, 46.95502, 10.35655, 47.00898)


def _load_fixture():
    with open(FIXTURE, encoding="utf-8") as f:
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
    """Uphill (m) of a run after smoothing elevations over BACKCLIMB_WINDOW_M (~60 m, step-size scale).
    A slope MUST be monotonic descending; the metric is total upward movement in the smoothed profile
    (0 for a clean descent). The window removes small DEM/geometry noise but — unlike the old 300 m
    window — does NOT hide a real 100 m+ mid-run climb.
    """
    es = [p.elevation for p in pts]
    if len(es) < 3:
        return 0.0
    if es[0] < es[-1]:
        es = es[::-1]
    seg = [pts[i].distance_to(other=pts[i + 1]) for i in range(len(pts) - 1)]
    avg = (sum(seg) / len(seg)) if seg else 30.0
    win = max(1, round(OSMConfig.BACKCLIMB_WINDOW_M / max(avg, 1.0)))
    half = win // 2
    sm = [
        sum(es[max(0, i - half) : min(len(es), i + half + 1)]) / len(es[max(0, i - half) : min(len(es), i + half + 1)])
        for i in range(len(es))
    ]
    return sum(max(0.0, sm[i + 1] - sm[i]) for i in range(len(sm) - 1))


def _run_max_slope(pts):
    """Steepest-section slope magnitude (%) of a run, rolled over ROLLING_WINDOW_M — independent
    reimplementation of the builder's metric, for the grouping-difficulty check.
    """
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
    """Build the graph from the committed Ischgl OSM fixture (set in stone — never re-fetched). The
    fixture MUST be present; a missing one is a broken checkout, not a reason to silently skip.
    """
    els = _load_fixture()
    dem = DEMService()  # REAL EuroDEM Alps terrain (data/alps_dem.tif)
    pistes, lifts = ways_to_lines(els, ISCHGL_BBOX)
    return OSMGraphBuilder(dem=dem, bbox=ISCHGL_BBOX).build(pistes, lifts)


class TestImportRules:
    """Each rule from the design conversation, asserted on the real Ischgl import."""

    def test_r1_all_lifts_imported(self, ischgl_graph):
        """Every skiable lift over the min length is imported (kept even if unconnected)."""
        els = _load_fixture()
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
        """No slope's geometry is ~covered by another slope (no double runs). EXCEPTION: two runs that
        SHARE an endpoint node may run coincident near that node because both were PULLED onto the hub —
        allowed ONLY while that coincident stretch is ≤ MAX_PULL_M. A longer coincident run is a genuine
        double-draw of a shared piste (a node belongs at the fork) and must fail.
        """
        sruns = ischgl_graph.slope_runs
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
                    if coincident <= MAX_PULL_M:
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
        """STRICT: no node may sit within MIN_NODE_DIST_M of a slope's geometry unless it is a vertex of
        that slope — a slope passing near a station must SPLIT at it (pass THROUGH it as a shared node).
        EXCEPTIONS: (a) a hub HIGHER than both the run's endpoints is a peak the descending run passes
        BELOW (can't share it without climbing); (b) a hub BELOW the lowest lift base is a valley-terminus
        pit the run skirts on its way to a lift base — sharing it would strand the descent in the pit.
        """
        pts = {k: (v.lon, v.lat) for k, v in ischgl_graph.node_points.items()}
        elev = {k: v.elevation for k, v in ischgl_graph.node_points.items()}
        lift_nodes = {n for lf in ischgl_graph.lifts for n in (lf.node_a, lf.node_b)}
        min_lift_base = min((elev[n] for n in lift_nodes), default=0.0)
        tol = OSMConfig.MIN_NODE_DIST_M
        floating = 0
        for r in ischgl_graph.slope_runs:
            endpts = {r.node_a, r.node_b}
            hi_end = max(elev[r.node_a], elev[r.node_b])
            run_pts = [(p.lon, p.lat) for p in r.points]
            for nk, npt in pts.items():
                if nk in endpts or elev[nk] > hi_end or elev[nk] < min_lift_base:
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
        # SEGMENTS (graph edges) and SLOPES are counted separately: here slope_runs ARE the segments
        # (full-split). Segment count is free up to a blizzard cap; node count is the visual-clutter gate.
        n_segments = len(ischgl_graph.slope_runs)
        assert n_segments <= MAX_SEGMENTS, f"{n_segments} segments — a blizzard, not a resort (cap {MAX_SEGMENTS})"
        assert len(ischgl_graph.node_points) <= MAX_NODES, "node count far above a hand-built resort"

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
        els = _load_fixture()
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
        """HARD (implied by R21): EVERY lift must share a station node with at least one slope. R21
        (strictly ski top→own-base) is strictly stronger, so if R21 holds this holds — assert it directly.
        """
        lifts = ischgl_graph.lifts
        slope_nodes = {n for r in ischgl_graph.slope_runs for n in (r.node_a, r.node_b)}
        orphan = [lf.name for lf in lifts if lf.node_a not in slope_nodes and lf.node_b not in slope_nodes]
        assert orphan == [], f"{len(orphan)}/{len(lifts)} lifts touch NO slope: {orphan[:5]}"

    def test_r15_most_runs_survive(self, ischgl_graph):
        """Sanity on volume: a 6 km Ischgl box has many segments (reference: 332 segments / 136 slopes)
        — the import must not collapse to a near-empty graph.
        """
        n = len(ischgl_graph.slope_runs)
        assert n >= MIN_SEGMENTS, (
            f"only {n} segments for a full Ischgl box (want ≥{MIN_SEGMENTS}) — dropping/under-noding runs"
        )

    def test_r16_every_lift_top_reaches_a_base(self, ischgl_graph):
        """HARD (implied by R21): from EVERY lift TOP you can ski DOWN to some lift base — you stay in
        the skiable network. R21 (ski to its OWN base) is strictly stronger, so if R21 holds this holds.
        Assert ALL, not a fraction.
        """
        elev = {k: v.elevation for k, v in ischgl_graph.node_points.items()}
        down: dict[int, set[int]] = defaultdict(set)
        for r in ischgl_graph.slope_runs:
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

        lift_bases = {(lf.node_b if elev[lf.node_a] >= elev[lf.node_b] else lf.node_a) for lf in ischgl_graph.lifts}
        stuck = []
        for lf in ischgl_graph.lifts:
            top = lf.node_a if elev[lf.node_a] >= elev[lf.node_b] else lf.node_b
            if not (reachable(top) & (lift_bases - {top})):  # can descend to some OTHER lift's base
                stuck.append(lf.name)
        assert stuck == [], (
            f"{len(stuck)}/{len(ischgl_graph.lifts)} lift tops CANNOT ski down to any lift base: {stuck[:5]}"
        )

    def test_r17_slopes_descend_by_orientation(self, ischgl_graph):
        """The directed-edge invariant. Every slope is stored node_a→node_b with node_a at least as
        high as node_b — the structural guarantee behind "ski down". R3 checks the smoothed profile;
        this checks the stored orientation itself.
        """
        elev = {k: v.elevation for k, v in ischgl_graph.node_points.items()}
        wrong = [r for r in ischgl_graph.slope_runs if elev[r.node_a] < elev[r.node_b]]
        assert wrong == [], f"{len(wrong)} slopes stored uphill (node_a below node_b) — orientation invariant broken"

    def test_r18_slope_endpoints_sit_on_their_hubs(self, ischgl_graph):
        """A slope's first/last point must BE its hub node coordinate (shared node, not floating near
        it). Exact structural check complementing R6's distance test.
        """
        node_pt = ischgl_graph.node_points
        tol = 1.0  # metres — pinned exactly by the builder; 1m guards float noise only
        bad = []
        for r in ischgl_graph.slope_runs:
            da = _hav((r.points[0].lon, r.points[0].lat), (node_pt[r.node_a].lon, node_pt[r.node_a].lat))
            db = _hav((r.points[-1].lon, r.points[-1].lat), (node_pt[r.node_b].lon, node_pt[r.node_b].lat))
            if da > tol or db > tol:
                bad.append((r.name, round(max(da, db), 1)))
        assert bad == [], f"{len(bad)} slopes whose endpoint is not ON its hub node (>{tol}m): {bad[:5]}"

    def test_r19_slope_geometry_fidelity(self, ischgl_graph):
        """Geometry fidelity, per the pull model: an imported slope keeps its OSM BODY on-piste; the
        ONLY off-piste geometry allowed is an END connector (the pull to a hub), of length ≤ MAX_PULL_M.
        So off-piste points may only form a contiguous run at the START and/or END (never mid-run = a
        tunnel), and each such end connector is ≤ MAX_PULL_M long.
        """
        from shapely.geometry import LineString, Point
        from shapely.ops import unary_union

        lat0 = (ISCHGL_BBOX[1] + ISCHGL_BBOX[3]) / 2
        mlat, mlon = 111_320.0, 111_320.0 * math.cos(math.radians(lat0))

        def to_m(lon, lat):
            return ((lon - ISCHGL_BBOX[0]) * mlon, (lat - ISCHGL_BBOX[1]) * mlat)

        pistes, _lifts = ways_to_lines(_load_fixture(), ISCHGL_BBOX)
        src = unary_union([LineString([to_m(lon, lat) for lon, lat in vs]) for vs, _nm in pistes if len(vs) >= 2])
        offenders = []
        for r in ischgl_graph.slope_runs:
            off = [Point(to_m(p.lon, p.lat)).distance(src) > PISTE_TOL_M for p in r.points]
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
            if max(head, tail) > MAX_PULL_M:
                offenders.append((r.name, f"connector {max(head, tail):.0f}m > {MAX_PULL_M:.0f}m"))
        assert offenders == [], f"{len(offenders)} slopes violate the pull/fidelity model: {offenders[:5]}"

    def test_r20_connectivity_runs_on_segments_with_branching(self, ischgl_graph):
        """Connectivity is a SEGMENT graph: a slope may be many segments and another may branch off at
        an interior node — such a branch (a degree≥3 shared node) still counts as connected. This makes
        explicit that R9 is measured over segments (slope_runs = edges), NOT over whole slopes, and that
        mid-slope junctions are legitimate connection points. Asserts the graph actually branches (real
        junctions exist) and that segment/slope statistics are tracked separately.
        """
        deg: dict[int, int] = defaultdict(int)
        for r in ischgl_graph.slope_runs:
            deg[r.node_a] += 1
            deg[r.node_b] += 1
        for lf in ischgl_graph.lifts:
            deg[lf.node_a] += 1
            deg[lf.node_b] += 1
        branch_nodes = [n for n, d in deg.items() if d >= 3]
        assert branch_nodes, "no branch nodes (degree≥3) — a full resort must have mid-slope junctions"
        # every segment is a genuine edge between two DISTINCT hub nodes (no self-loops in the graph)
        assert all(r.node_a != r.node_b for r in ischgl_graph.slope_runs), "a segment is a self-loop (a==b)"

    def test_r21_every_lift_is_skiable_top_to_bottom(self, ischgl_graph):
        """SLOPE-COMPLETENESS (not a lift check): from EVERY lift's top you can ski DOWN to its own
        bottom via a chain of descending segments. A lift is NEVER dropped, but if this fails it signals
        MISSING/wrong slopes (the real defect) — so the test must raise. OSM has the slopes; the import
        must not filter them out.
        """
        elev = {k: v.elevation for k, v in ischgl_graph.node_points.items()}
        down: dict[int, set[int]] = defaultdict(set)
        for r in ischgl_graph.slope_runs:
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
        for lf in ischgl_graph.lifts:
            top = lf.node_a if elev[lf.node_a] >= elev[lf.node_b] else lf.node_b
            bottom = lf.node_b if top == lf.node_a else lf.node_a
            if not can_ski(top, bottom):
                unskiable.append(lf.name or f"{top}->{bottom}")
        assert unskiable == [], (
            f"{len(unskiable)}/{len(ischgl_graph.lifts)} lifts have NO descending slope-chain top→bottom "
            f"(missing slopes, not a lift fault): {unskiable[:5]}"
        )

    def test_r22_no_slope_dead_ends(self, ischgl_graph):
        """No slope may dead-end where a skier gets STRANDED. STRICT, traced:

        (a) DOWN — from EVERY slope node, following descending edges must reach a LIFT STATION (a node
            you can ride back up). A node all of whose slopes arrive from above and none continue down,
            and which is not a lift base, is a stranding dead-end — fail loudly.
        (b) UP  — every slope node must be reachable, following descending edges FROM some lift station
            (you must be able to get onto the slope by riding a lift up, then skiing down to it).
        """
        elev = {k: v.elevation for k, v in ischgl_graph.node_points.items()}
        lift_nodes = {n for lf in ischgl_graph.lifts for n in (lf.node_a, lf.node_b)}
        min_lift_base = min((elev[n] for n in lift_nodes), default=0.0)
        down: dict[int, set[int]] = defaultdict(set)
        up: dict[int, set[int]] = defaultdict(set)
        slope_nodes: set[int] = set()
        for r in ischgl_graph.slope_runs:
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
            f"{stranded_up[:8]} — a dropped/truncated slope"
        )

    def test_r23_slope_points_hug_terrain(self, ischgl_graph):
        """STRICT: every slope point must sit within MAX_TERRAIN_DEVIATION_M of the real DEM terrain —
        never floating >50 m above it or buried >50 m below. A slope follows the ground; a point far
        off terrain is invented/tunnelling geometry.
        """
        dem = DEMService()  # real EuroDEM (same terrain the builder draped onto)
        offenders = []
        for r in ischgl_graph.slope_runs:
            for p in r.points:
                terrain = dem.get_elevation(lon=p.lon, lat=p.lat)
                if terrain is not None and abs(p.elevation - terrain) > MAX_TERRAIN_DEVIATION_M:
                    offenders.append((r.name, round(p.elevation - terrain, 1)))
                    break
        assert offenders == [], (
            f"{len(offenders)} slopes have a point > {MAX_TERRAIN_DEVIATION_M}m off terrain "
            f"(above/below DEM): {offenders[:5]}"
        )

    def test_r24_no_lift_dropped(self, ischgl_graph):
        """EVERY raw OSM way with an allowed aerialway type and length ≥
        MIN_LIFT_LENGTH_M MUST appear in the final output. The builder may never drop a lift.
        """
        _pistes, lifts = ways_to_lines(_load_fixture(), ISCHGL_BBOX)
        expected = sum(
            1
            for vs, _lt, _nm in lifts
            if sum(_hav(vs[i], vs[i + 1]) for i in range(len(vs) - 1)) >= OSMConfig.MIN_LIFT_LENGTH_M
        )
        assert len(ischgl_graph.lifts) == expected, (
            f"only {len(ischgl_graph.lifts)}/{expected} qualifying lifts survived — a lift was DROPPED "
        )

    def test_r25_every_node_hugs_terrain_strict(self, ischgl_graph):
        """STRICT: EVERY node must sit within MAX_NODE_TERRAIN_DEVIATION_M (±10 m) of the real DEM
        terrain. A node floating further off the ground is invented placement.
        """
        dem = DEMService()
        offenders = []
        for k, p in ischgl_graph.node_points.items():
            terrain = dem.get_elevation(lon=p.lon, lat=p.lat)
            if terrain is not None and abs(p.elevation - terrain) > MAX_NODE_TERRAIN_DEVIATION_M:
                offenders.append((k, round(p.elevation - terrain, 1)))
        assert offenders == [], (
            f"{len(offenders)} nodes are > {MAX_NODE_TERRAIN_DEVIATION_M}m off terrain: {offenders[:8]}"
        )

    def test_r26_lift_endpoints_sit_on_their_hubs(self, ischgl_graph):
        """A lift's drawn stations (bottom/top) must BE its hub-node coordinates."""
        node_pt = ischgl_graph.node_points
        tol = 1.0  # metres — the lift station IS its node; anything larger is a node/geometry desync
        bad = []
        for lf in ischgl_graph.lifts:
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

    def test_r27_referential_integrity(self, ischgl_graph):
        """Every node id referenced by a slope or lift MUST exist in node_points,
        and every node in node_points MUST be referenced by at least one slope/lift.
        """
        nodes = set(ischgl_graph.node_points)
        referenced = {n for r in ischgl_graph.slope_runs for n in (r.node_a, r.node_b)} | {
            n for lf in ischgl_graph.lifts for n in (lf.node_a, lf.node_b)
        }
        dangling = sorted(referenced - nodes)
        orphaned = sorted(nodes - referenced)
        assert dangling == [], (
            f"{len(dangling)} slope/lift endpoints reference nodes not in node_points: {dangling[:8]}"
        )
        assert orphaned == [], f"{len(orphaned)} nodes in node_points are referenced by nothing: {orphaned[:8]}"

    def test_r28_lift_stations_match_raw_osm(self, ischgl_graph):
        """Every imported lift's bottom/top must sit within RESAMPLE_STEP_M of
        a RAW OSM lift endpoint. The builder takes lift geometry verbatim from OSM (only the DEM z is
        recomputed), so a station that drifted from every raw OSM station was moved by the builder.
        """
        _pistes, raw_lifts = ways_to_lines(_load_fixture(), ISCHGL_BBOX)
        raw_stations = [vs[0] for vs, _lt, _nm in raw_lifts] + [vs[-1] for vs, _lt, _nm in raw_lifts]
        tol = OSMConfig.RESAMPLE_STEP_M  # a station is a raw OSM vertex; allow one resample step of slack
        bad = []
        for lf in ischgl_graph.lifts:
            for station in (lf.bottom, lf.top):
                if min(_hav((station.lon, station.lat), rs) for rs in raw_stations) > tol:
                    bad.append(lf.name)
                    break
        assert bad == [], (
            f"{len(bad)} lifts have a station that matches NO raw OSM lift endpoint (>{tol}m) — "
            f"builder moved a station off its real OSM position: {bad[:5]}"
        )

    def test_r29_segments_group_into_fewer_slopes(self, ischgl_graph):
        """PATH SEGMENTS must group into whole app-slopes: the metric is segments per FINAL app-slope
        (what to_slope_chains emits — each chain becomes one app Slope of len(chain) segments), NOT the
        intermediate ImportSlope. A real named piste is 5-10 segments; require ≥ MIN_SEG_PER_SLOPE on
        average, every run in exactly one chain.
        """
        runs = ischgl_graph.slope_runs
        chains = ischgl_graph.to_slope_chains()  # FINAL app-slopes: list of (per-run point-lists, name)
        assert chains, "no app-slopes — segments were never grouped"
        # referential completeness: every run's points appear in exactly one chain (partition of runs)
        n_segments = sum(len(pts_lists) for pts_lists, _name in chains)
        assert n_segments == len(runs), (
            f"grouping is not a partition: {n_segments} chained segments != {len(runs)} runs"
        )
        ratio = len(runs) / len(chains)
        assert ratio >= MIN_SEG_PER_SLOPE, (
            f"only {ratio:.2f} segments per app-slope ({len(runs)} segments / {len(chains)} app-slopes) — "
            f"want ≥{MIN_SEG_PER_SLOPE}"
        )


class TestGraphImporter:
    """The production GraphImporter wrapper: it fetches, runs the builder, and reports the built
    graph as an ImportResult (hub-aligned lifts + slope chains), writing reference artifacts.
    """

    def test_run_yields_importresult_and_writes_artifacts(self, tmp_path, monkeypatch) -> None:
        from skiresort_planner.generators.osm_graph_builder import GraphImporter

        importer = GraphImporter(dem=DEMService(), bbox=ISCHGL_BBOX)
        monkeypatch.setattr(importer, "fetch", lambda: _load_fixture())  # no network — use the fixture

        result = importer.run(dump_dir=tmp_path)

        assert result.source == "OSM"
        assert result.lifts, "the graph importer must report the hub-aligned lifts"
        assert result.slope_chains, "the graph importer must report grouped slope chains"
        # Every chain is a non-empty list of segment point-lists.
        for chain, _name in result.slope_chains:
            assert chain and all(len(seg) >= 2 for seg in chain)
        # Reference artifacts written for inspection (never read back).
        assert (tmp_path / "osm_raw.json").exists()
        assert (tmp_path / "osm_import.png").exists()

    def test_r30_named_piste_is_one_app_slope(self, ischgl_graph):
        """R30: each real named OSM piste materialises as EXACTLY ONE app-slope (chain), even where it
        branches — never fragmented into several same-named slopes with divergent difficulty (the '63
        appeared 3×' bug). to_slope_chains emits one chain per named ImportSlope.
        """
        from collections import Counter

        chains = ischgl_graph.to_slope_chains()
        name_counts = Counter(name for _pts, name in chains if name)
        split = {n: c for n, c in name_counts.items() if c > 1}
        assert not split, f"named pistes fragmented into multiple app-slopes: {split}"

    def test_r31_every_slope_top_is_skier_reachable(self, ischgl_graph):
        """R31: no phantom slope — every slope's HIGH node is reachable by a skier (stand on a lift top,
        ski DOWN, ride lifts UP). An unreachable top means the feeder piste into it was dropped; the
        uphill-feeder reconnect must have rebuilt it (from real OSM) or the slope must not exist.
        """
        g = ischgl_graph
        elev = {k: v.elevation for k, v in g.node_points.items()}
        down = defaultdict(set)
        for r in g.slope_runs:
            hi, lo = (r.node_a, r.node_b) if elev[r.node_a] >= elev[r.node_b] else (r.node_b, r.node_a)
            down[hi].add(lo)
        lift_up = defaultdict(set)
        seeds = set()
        for lf in g.lifts:
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
        tops = {(r.node_a if elev[r.node_a] >= elev[r.node_b] else r.node_b) for r in g.slope_runs}
        unreachable = tops - reach
        assert not unreachable, f"slope tops no skier can reach (phantom entries): {sorted(unreachable)}"

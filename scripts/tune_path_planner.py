"""Standalone parameter-tuning experiment for the path-planning algorithm.

Loads a saved resort backup, harvests the REAL slope/road segment endpoints (and their
incoming headings) as a corpus of build scenarios, then replays each scenario through
`LeastCostPathPlanner` over the REAL DEM while sweeping the tunable parameters. For every
(parameter, scenario) pair it records objective metrics, aggregates them across the corpus
for statistical evidence, and writes a Markdown report ending in a confirmed good config.

Run:  python scripts/tune_path_planner.py
Output: docs/experiments/path-planner-tuning.md

This is an OFFLINE analysis tool — it imports the production planner but never mutates the
graph or the live constants beyond the value under test (restored after each trial).
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, field
from math import radians, sin
from typing import Callable, Optional

from skiresort_planner.constants import BACKUP_DIR, PROJECT_ROOT, PathConfig, PlannerConfig
from skiresort_planner.core.dem_service import DEMService
from skiresort_planner.core.geo_calculator import GeoCalculator
from skiresort_planner.generators.path_factory import PathFactory
from skiresort_planner.model.proposed_path import ProposedPathSegment

REPORT_PATH = PROJECT_ROOT / "docs" / "experiments" / "path-planner-tuning.md"
ROAD_BAND = (-float(PathConfig.ROAD_MAX_GRADIENT_PCT), float(PathConfig.ROAD_MAX_GRADIENT_PCT))

# Route-distortion guard for pick_best: a parameter value whose median endpoint grid-snap
# rises beyond DISTORTION_FACTOR× the gentlest option's (or GRID_SNAP_SLACK_M above it) is
# bending the whole route, not just cleaning the join — reject it however good its metric.
DISTORTION_FACTOR = 1.5
GRID_SNAP_SLACK_M = 20.0  # absolute slack (~one grid cell) so tiny floors don't over-constrain


# =============================================================================
# Scenario corpus — real build situations harvested from a backup
# =============================================================================


@dataclass(frozen=True)
class Scenario:
    """One 'extend a segment' build situation taken from a real saved entity.

    start/end are real committed segment endpoints; incoming_bearing is the heading the
    prior geometry arrives with (None for the first segment of an entity). is_road picks
    road-mode (gradient band + earthwork) vs slope-mode.
    """

    label: str
    start_lon: float
    start_lat: float
    start_elev: float
    end_lon: float
    end_lat: float
    end_elev: float
    incoming_bearing: Optional[float]
    is_road: bool


def harvest_scenarios(backup: dict, dem: DEMService) -> list[Scenario]:
    """Build a scenario per segment of every slope/road in the backup.

    For each segment we take its first and last point as start/target, and derive the
    incoming bearing from the previous segment's final leg (None for the first segment).
    """
    segments = backup["segments"]
    scenarios: list[Scenario] = []

    def add_entity(entity_id: str, seg_ids: list[str], is_road: bool) -> None:
        prev_last_two: Optional[tuple[dict, dict]] = None
        for idx, sid in enumerate(seg_ids):
            seg = segments.get(sid)
            if seg is None or len(seg["points"]) < 2:
                continue
            pts = seg["points"]
            start, end = pts[0], pts[-1]
            # Only keep scenarios whose endpoints are inside the DEM (offline safety).
            if dem.get_elevation(lon=start["lon"], lat=start["lat"]) is None:
                continue
            if dem.get_elevation(lon=end["lon"], lat=end["lat"]) is None:
                continue
            incoming = None
            if prev_last_two is not None:
                a, b = prev_last_two
                incoming = GeoCalculator.initial_bearing_deg(lon1=a["lon"], lat1=a["lat"], lon2=b["lon"], lat2=b["lat"])
            scenarios.append(
                Scenario(
                    label=f"{entity_id}:{sid}",
                    start_lon=start["lon"],
                    start_lat=start["lat"],
                    start_elev=start["elevation"],
                    end_lon=end["lon"],
                    end_lat=end["lat"],
                    end_elev=end["elevation"],
                    incoming_bearing=incoming,
                    is_road=is_road,
                )
            )
            prev_last_two = (pts[-2], pts[-1])

    for rid, road in backup.get("roads", {}).items():
        add_entity(rid, road["segment_ids"], is_road=True)
    for slid, slope in backup.get("slopes", {}).items():
        add_entity(slid, slope["segment_ids"], is_road=False)
    return scenarios


# =============================================================================
# Metrics — objective quality measures for one generated path
# =============================================================================


@dataclass
class Metrics:
    """Objective measures for a single generated proposal (all lower = better except ok)."""

    ok: bool  # a path was produced at all
    max_slope_pct: float  # steepest rolling-window grade (road cap = 15%)
    join_kink_deg: float  # angle between incoming heading and the path's first leg
    start_jump_m: float  # cross-track offset of the path 30 m from the node (sideways jump)
    length_ratio: float  # path length / straight-line distance (detour; 1.0 = straight)
    max_earthwork_m: float  # peak cut/fill vs ground (0 for slopes)
    endpoint_err_m: float  # how far the path's endpoints sit from the requested nodes


def _turn(a: float, b: float) -> float:
    d = abs(a - b) % 360
    return d if d <= 180 else 360 - d


def measure(path: Optional[ProposedPathSegment], sc: Scenario, dem: DEMService) -> Metrics:
    """Compute objective metrics for one proposal against its scenario."""
    if path is None or len(path.points) < 2:
        return Metrics(False, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    pts = path.points

    # Join kink: incoming heading vs first leg (0 if no incoming bearing).
    first_leg = GeoCalculator.initial_bearing_deg(lon1=pts[0].lon, lat1=pts[0].lat, lon2=pts[3].lon, lat2=pts[3].lat)
    kink = _turn(sc.incoming_bearing, first_leg) if sc.incoming_bearing is not None else 0.0

    # Start jump: cross-track offset ~30 m from the node relative to the incoming line.
    jump = 0.0
    if sc.incoming_bearing is not None:
        for q in pts[1:]:
            d = GeoCalculator.haversine_distance_m(lat1=pts[0].lat, lon1=pts[0].lon, lat2=q.lat, lon2=q.lon)
            if d >= 30.0:
                b = GeoCalculator.initial_bearing_deg(lon1=pts[0].lon, lat1=pts[0].lat, lon2=q.lon, lat2=q.lat)
                jump = d * abs(sin(radians(_turn(b, sc.incoming_bearing))))
                break

    # Length ratio (detour).
    length = sum(pts[i].distance_to(other=pts[i + 1]) for i in range(len(pts) - 1))
    straight = GeoCalculator.haversine_distance_m(
        lat1=sc.start_lat, lon1=sc.start_lon, lat2=sc.end_lat, lon2=sc.end_lon
    )
    length_ratio = length / straight if straight > 0 else 1.0

    # Earthwork peak (deviation of committed elevation from DEM ground).
    earthwork = 0.0
    for pt in pts:
        g = dem.get_elevation(lon=pt.lon, lat=pt.lat)
        if g is not None:
            earthwork = max(earthwork, abs(pt.elevation - g))

    # Endpoint grid-snap: how far the RAW proposal endpoints sit from the requested
    # start/target. This is the grid-cell snap of the Dijkstra search (≈ up to a grid
    # cell + half a cell); production `commit_paths` snaps endpoints exactly onto the
    # node via start_node_id/target_node_id, so the committed value is ~0. We track it
    # only to DETECT route distortion: a value far above the grid-snap floor means
    # strong momentum is bending the whole route, not just the join.
    endpoint_snap = max(
        GeoCalculator.haversine_distance_m(lat1=sc.start_lat, lon1=sc.start_lon, lat2=pts[0].lat, lon2=pts[0].lon),
        GeoCalculator.haversine_distance_m(lat1=sc.end_lat, lon1=sc.end_lon, lat2=pts[-1].lat, lon2=pts[-1].lon),
    )

    return Metrics(True, path.max_slope_pct, kink, jump, length_ratio, earthwork, endpoint_snap)


def run_scenario(factory: PathFactory, sc: Scenario, dem: DEMService) -> Metrics:
    """Generate the (single) best proposal for a scenario and measure it."""
    kwargs = dict(
        start_lon=sc.start_lon,
        start_lat=sc.start_lat,
        start_elevation=sc.start_elev,
        target_lon=sc.end_lon,
        target_lat=sc.end_lat,
        target_elevation=sc.end_elev,
        incoming_bearing=sc.incoming_bearing,
    )
    if sc.is_road:
        kwargs["gradient_band"] = ROAD_BAND
    paths = list(factory.generate_manual_paths(**kwargs))
    # Pick the gentlest proposal (what the user is most likely to commit).
    best = min(paths, key=lambda p: p.max_slope_pct) if paths else None
    return measure(best, sc, dem)


# =============================================================================
# Parameter sweep
# =============================================================================


@dataclass
class Param:
    """One tunable knob and the values to sweep for it."""

    name: str  # PlannerConfig attribute
    values: list[float]
    metric: str  # the Metrics field this knob most directly targets
    goal: str  # "min" — all our targeted metrics are lower-is-better


@dataclass
class Aggregate:
    """Corpus-aggregated metrics for one parameter value."""

    value: float
    n_ok: int
    kink_med: float
    jump_med: float
    slope_med: float
    length_med: float
    earthwork_med: float
    endpoint_med: float  # median grid-snap; a jump above the floor = route distortion
    raw: dict = field(default_factory=dict)


def aggregate(metrics: list[Metrics]) -> Aggregate:
    ok = [m for m in metrics if m.ok]

    def med(vals: list[float]) -> float:
        return statistics.median(vals) if vals else float("nan")

    return Aggregate(
        value=0.0,
        n_ok=len(ok),
        kink_med=med([m.join_kink_deg for m in ok]),
        jump_med=med([m.start_jump_m for m in ok]),
        slope_med=med([m.max_slope_pct for m in ok]),
        length_med=med([m.length_ratio for m in ok]),
        earthwork_med=med([m.max_earthwork_m for m in ok]),
        endpoint_med=med([m.endpoint_err_m for m in ok]),
    )


def sweep_param(
    factory: PathFactory, dem: DEMService, scenarios: list[Scenario], param: Param, applies: Callable[[Scenario], bool]
) -> list[Aggregate]:
    """Sweep one parameter across its values over all applicable scenarios."""
    subset = [s for s in scenarios if applies(s)]
    original = getattr(PlannerConfig, param.name)
    results: list[Aggregate] = []
    try:
        for v in param.values:
            setattr(PlannerConfig, param.name, v)
            metrics = [run_scenario(factory, s, dem) for s in subset]
            agg = aggregate(metrics)
            agg.value = v
            results.append(agg)
    finally:
        setattr(PlannerConfig, param.name, original)  # always restore
    return results


# =============================================================================
# Report
# =============================================================================


def _fmt(x: float) -> str:
    return "n/a" if x != x else f"{x:.2f}"  # NaN check


def pick_best(param: Param, aggs: list[Aggregate]) -> Aggregate:
    """Choose the value that minimises the targeted metric WITHOUT distorting the route.

    A value is only eligible if it doesn't materially bend the whole path: its median
    endpoint grid-snap must stay within DISTORTION_FACTOR× the gentlest option's (strong
    momentum can drag endpoints far past the grid-snap floor — that is route distortion,
    not a cleaner join). Among eligible values, minimise the primary metric, then break
    ties by preferring the lower detour (length_ratio).
    """
    metric_of = {
        "join_kink_deg": lambda a: a.kink_med,
        "start_jump_m": lambda a: a.jump_med,
        "max_slope_pct": lambda a: a.slope_med,
    }[param.metric]

    floor = min(a.endpoint_med for a in aggs)  # gentlest (least-distorting) option
    limit = max(floor * DISTORTION_FACTOR, floor + GRID_SNAP_SLACK_M)
    eligible = [a for a in aggs if a.endpoint_med <= limit] or aggs  # never empty
    return min(eligible, key=lambda a: (round(metric_of(a), 2), round(a.length_med, 3)))


def write_report(
    scenarios: list[Scenario],
    sweeps: dict[str, tuple[Param, list[Aggregate], Aggregate]],
    baseline: dict[str, float],
    date_str: str,
) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    n_road = sum(1 for s in scenarios if s.is_road)
    n_slope = len(scenarios) - n_road
    lines: list[str] = []
    lines.append("# Path-planner parameter tuning — experiment report")
    lines.append("")
    lines.append(f"_Generated by `scripts/tune_path_planner.py` on {date_str}._")
    lines.append("")
    lines.append("## Method")
    lines.append("")
    lines.append(
        f"Loaded the saved resort backup and harvested **{len(scenarios)} real build scenarios** "
        f"({n_road} road, {n_slope} slope) from the committed segment endpoints of every road/slope. "
        "Each scenario replays 'extend this segment' through the production `LeastCostPathPlanner` over "
        "the real DEM, with the segment's real incoming heading. For each tunable parameter we sweep a "
        "range of values across all applicable scenarios and aggregate objective metrics (medians over the "
        "corpus, so a single odd scenario can't skew the pick)."
    )
    lines.append("")
    lines.append("**Objective metrics** (all lower = better):")
    lines.append("")
    lines.append("- `kink` — join angle (deg) between incoming heading and the path's first leg")
    lines.append("- `jump` — sideways cross-track offset (m) ~30 m off the node")
    lines.append("- `slope` — steepest rolling-window grade (%), road validity cap = 15")
    lines.append("- `length` — path length / straight-line (detour ratio; 1.0 = straight)")
    lines.append("- `earth` — peak cut/fill vs ground (m; 0 for slopes)")
    lines.append(
        "- `snap` — raw-proposal endpoint offset from the requested node (m). This is the "
        "Dijkstra grid-snap (≈ up to ~1.5 grid cells); production `commit_paths` snaps endpoints "
        "exactly onto the node, so the committed value is ~0. A `snap` far above the per-sweep "
        "floor flags a value that is bending the whole route (route distortion), and such values "
        "are rejected when picking the best (see distortion guard)."
    )
    lines.append("")

    for param_name, (param, aggs, best) in sweeps.items():
        lines.append(f"## `{param.name}` (targets **{param.metric}**)")
        lines.append("")
        lines.append(f"Applicable scenarios: {aggs[0].n_ok} produced a path.")
        lines.append("")
        lines.append("| value | n_ok | kink | jump | slope | length | earth | snap |")
        lines.append("|------:|-----:|-----:|-----:|------:|-------:|------:|-----:|")
        for a in aggs:
            marker = " ✅" if a.value == best.value else ""
            lines.append(
                f"| {a.value:g}{marker} | {a.n_ok} | {_fmt(a.kink_med)} | {_fmt(a.jump_med)} | "
                f"{_fmt(a.slope_med)} | {_fmt(a.length_med)} | {_fmt(a.earthwork_med)} | {_fmt(a.endpoint_med)} |"
            )
        lines.append("")
        lines.append(
            f"**Best `{param.name}` = {best.value:g}** — minimises {param.metric} "
            f"(median {_fmt(getattr(best, {'join_kink_deg': 'kink_med', 'start_jump_m': 'jump_med', 'max_slope_pct': 'slope_med'}[param.metric]))}) "
            f"among values that don't distort the route (snap {_fmt(best.endpoint_med)}, detour {_fmt(best.length_med)})."
        )
        lines.append("")

    lines.append("## Confirmed good configuration")
    lines.append("")
    lines.append("Per-parameter winners below, each the best on its target metric among values that did")
    lines.append("NOT distort the route (endpoint grid-snap kept near the per-sweep floor). Values that")
    lines.append("scored better on a metric only by dragging the whole path off-line were rejected.")
    lines.append("")
    lines.append("```python")
    lines.append("class PlannerConfig:")
    for param_name, (param, _aggs, best) in sweeps.items():
        cur = baseline[param.name]
        note = "unchanged" if best.value == cur else f"was {cur:g}"
        lines.append(f"    {param.name} = {best.value:g}  # {note}")
    lines.append("```")
    lines.append("")
    lines.append("_Consistency check: `snap` measures the raw grid-snap (production commit pins endpoints")
    lines.append("to the node, → ~0). The chosen values keep `snap` near its floor, so the join metrics")
    lines.append("(kink/jump) reflect a genuinely cleaner departure, not a route dragged off its endpoints._")

    REPORT_PATH.write_text("\n".join(lines) + "\n")


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    backup_path = next(iter(sorted(BACKUP_DIR.glob("*.json"))), None)
    if backup_path is None:
        raise SystemExit(f"No backup found in {BACKUP_DIR}")
    backup = json.loads(backup_path.read_text())

    dem = DEMService()
    factory = PathFactory(dem_service=dem)
    scenarios = harvest_scenarios(backup, dem)
    if not scenarios:
        raise SystemExit("No usable scenarios harvested (backup empty or outside DEM).")

    # Snapshot baseline for the "was X" annotations.
    param_names = [
        "MOMENTUM_TURN_WEIGHT",
        "MOMENTUM_DECAY_M",
        "MOMENTUM_POS_WEIGHT",
        "MOMENTUM_POS_DECAY_M",
        "COST_SIGMA",
    ]
    baseline = {n: getattr(PlannerConfig, n) for n in param_names}

    # Parameters to sweep, each with the metric it primarily targets.
    params = [
        Param("MOMENTUM_TURN_WEIGHT", [0.0, 0.3, 0.6, 1.0, 2.0], metric="join_kink_deg", goal="min"),
        Param("MOMENTUM_DECAY_M", [75.0, 150.0, 300.0, 450.0], metric="join_kink_deg", goal="min"),
        Param("MOMENTUM_POS_WEIGHT", [0.0, 1.0, 2.0, 4.0, 8.0], metric="start_jump_m", goal="min"),
        Param("MOMENTUM_POS_DECAY_M", [30.0, 60.0, 120.0], metric="start_jump_m", goal="min"),
        Param("COST_SIGMA", [4.0, 6.0, 8.0, 12.0], metric="max_slope_pct", goal="min"),
    ]

    # Momentum params only matter where there's an incoming bearing (mid-run extends).
    def has_incoming(s: Scenario) -> bool:
        return s.incoming_bearing is not None

    def any_scenario(_s: Scenario) -> bool:
        return True

    applies_map = {
        "MOMENTUM_TURN_WEIGHT": has_incoming,
        "MOMENTUM_DECAY_M": has_incoming,
        "MOMENTUM_POS_WEIGHT": has_incoming,
        "MOMENTUM_POS_DECAY_M": has_incoming,
        "COST_SIGMA": any_scenario,
    }

    sweeps: dict[str, tuple[Param, list[Aggregate], Aggregate]] = {}
    for param in params:
        aggs = sweep_param(factory, dem, scenarios, param, applies_map[param.name])
        best = pick_best(param, aggs)
        sweeps[param.name] = (param, aggs, best)
        print(f"{param.name}: swept {len(param.values)} values → best {best.value:g}")

    # Date passed in via env to keep the module import-time-pure (no clock in library code).
    import os

    date_str = os.environ.get("TUNE_DATE", "unknown date")
    write_report(scenarios, sweeps, baseline, date_str)
    print(f"Report written to {REPORT_PATH}")


if __name__ == "__main__":
    main()

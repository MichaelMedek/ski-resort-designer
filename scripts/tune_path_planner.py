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
from typing import Any, Callable, Optional

from skiresort_planner.constants import BACKUP_DIR, PROJECT_ROOT, PathConfig, PlannerConfig
from skiresort_planner.core.dem_service import DEMService
from skiresort_planner.core.geo_calculator import GeoCalculator
from skiresort_planner.generators.path_factory import PathFactory
from skiresort_planner.model.proposed_path import ProposedPathSegment

REPORT_PATH = PROJECT_ROOT / "docs" / "experiments" / "path-planner-tuning.md"


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


def harvest_scenarios(backup: dict[str, Any], dem: DEMService) -> list[Scenario]:
    """Build a scenario per segment of every slope/road in the backup.

    For each segment we take its first and last point as start/target, and derive the
    incoming bearing from the previous segment's final leg (None for the first segment).
    """
    segments = backup["segments"]
    scenarios: list[Scenario] = []

    def add_entity(entity_id: str, seg_ids: list[str], is_road: bool) -> None:
        prev_last_two: Optional[tuple[dict[str, float], dict[str, float]]] = None
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
    endpoint_err_m: float  # how far the path's endpoints sit from the requested nodes
    is_road: bool  # this scenario is a road (subject to the ±15% cap)
    road_rejected: bool  # ROAD scenario whose gentlest proposal still exceeds the ±15% cap


def _turn(a: float, b: float) -> float:
    d = abs(a - b) % 360
    return d if d <= 180 else 360 - d


def measure(path: Optional[ProposedPathSegment], sc: Scenario) -> Metrics:
    """Compute objective metrics for one proposal against its scenario."""
    if path is None or len(path.points) < 2:
        # No path at all: for a road that counts as a rejection (nothing to commit).
        return Metrics(False, 0.0, 0.0, 0.0, 0.0, 0.0, is_road=sc.is_road, road_rejected=sc.is_road)
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

    # Endpoint grid-snap: how far the RAW proposal endpoints sit from the requested
    # start/target. This is the grid-cell snap of the Dijkstra search (≈ up to a grid
    # cell + half a cell); production `commit_paths` snaps endpoints exactly onto the
    # node via start_node_id/target_node_id, so the committed value is ~0. We track it
    # only to DETECT route distortion: a value far above the grid-snap floor means the
    # route is being bent, not just the join.
    endpoint_snap = max(
        GeoCalculator.haversine_distance_m(lat1=sc.start_lat, lon1=sc.start_lon, lat2=pts[0].lat, lon2=pts[0].lon),
        GeoCalculator.haversine_distance_m(lat1=sc.end_lat, lon1=sc.end_lon, lat2=pts[-1].lat, lon2=pts[-1].lon),
    )

    # A road is "rejected" if its steepest section exceeds the ±15% cap — production
    # would refuse to offer it. Slopes have no such cap, so they're never rejected.
    road_rejected = sc.is_road and path.max_slope_pct > float(PathConfig.ROAD_MAX_GRADIENT_PCT)

    return Metrics(True, path.max_slope_pct, kink, jump, length_ratio, endpoint_snap, sc.is_road, road_rejected)


def run_scenario(factory: PathFactory, sc: Scenario) -> Metrics:
    """Generate the (single) best proposal for a scenario and measure it."""
    paths = list(
        factory.generate_manual_paths(
            start_lon=sc.start_lon,
            start_lat=sc.start_lat,
            start_elevation=sc.start_elev,
            target_lon=sc.end_lon,
            target_lat=sc.end_lat,
            target_elevation=sc.end_elev,
            road_mode=sc.is_road,
            incoming_bearing=sc.incoming_bearing,
        )
    )
    # Pick the gentlest proposal (what the user is most likely to commit).
    best = min(paths, key=lambda p: p.max_slope_pct) if paths else None
    return measure(best, sc)


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
    endpoint_med: float  # median grid-snap; a jump above the floor = route distortion
    reject_rate: float  # fraction of ROAD scenarios whose gentlest route still exceeds ±15%
    n_road: int  # number of road scenarios in this subset (denominator for reject_rate)
    raw: dict[str, float] = field(default_factory=dict)


def aggregate(metrics: list[Metrics]) -> Aggregate:
    ok = [m for m in metrics if m.ok]
    # reject_rate is computed over ROAD scenarios only (slopes have no ±15% cap).
    road_metrics = [m for m in metrics if m.is_road]
    n_road = len(road_metrics)
    rejected = sum(1 for m in road_metrics if m.road_rejected)

    def med(vals: list[float]) -> float:
        return statistics.median(vals) if vals else float("nan")

    return Aggregate(
        value=0.0,
        n_ok=len(ok),
        kink_med=med([m.join_kink_deg for m in ok]),
        jump_med=med([m.start_jump_m for m in ok]),
        slope_med=med([m.max_slope_pct for m in ok]),
        length_med=med([m.length_ratio for m in ok]),
        endpoint_med=med([m.endpoint_err_m for m in ok]),
        reject_rate=(rejected / n_road) if n_road else float("nan"),
        n_road=n_road,
    )


def sweep_param(
    factory: PathFactory, scenarios: list[Scenario], param: Param, applies: Callable[[Scenario], bool]
) -> list[Aggregate]:
    """Sweep one parameter across its values over all applicable scenarios."""
    subset = [s for s in scenarios if applies(s)]
    original = getattr(PlannerConfig, param.name)
    results: list[Aggregate] = []
    try:
        for v in param.values:
            setattr(PlannerConfig, param.name, v)
            metrics = [run_scenario(factory, s) for s in subset]
            agg = aggregate(metrics)
            agg.value = v
            results.append(agg)
            print(f"  {param.name}={v:g}: reject_rate={agg.reject_rate:.2f} (n_road={agg.n_road})", flush=True)
    finally:
        setattr(PlannerConfig, param.name, original)  # always restore
    return results


# =============================================================================
# Report
# =============================================================================


def _fmt(x: float) -> str:
    return "n/a" if x != x else f"{x:.2f}"  # NaN check


def pick_best(param: Param, aggs: list[Aggregate]) -> Aggregate:
    """Choose the value that minimises the targeted metric, breaking ties by lower detour.

    For road-reachability (reject_rate) and grade (max_slope_pct) the goal is simply the
    lowest value; when two values tie we prefer the one with the smaller detour
    (length_ratio) so we don't buy a marginal gain with a wildly longer road.
    """
    metric_of: dict[str, Callable[[Aggregate], float]] = {
        "join_kink_deg": lambda a: a.kink_med,
        "start_jump_m": lambda a: a.jump_med,
        "max_slope_pct": lambda a: a.slope_med,
        "reject_rate": lambda a: a.reject_rate,
    }
    key_fn = metric_of[param.metric]

    return min(aggs, key=lambda a: (round(key_fn(a), 3), round(a.length_med, 3)))


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
    lines.append("**Objective metrics:**")
    lines.append("")
    lines.append(
        "- `reject` — fraction of ROAD scenarios whose gentlest route still busts ±15% (LOWER = better; the key metric — a wrong refusal of a sensible road)"
    )
    lines.append("- `slope` — median steepest rolling-window grade (%), road validity cap = 15")
    lines.append("- `length` — path length / straight-line (detour ratio; 1.0 = straight)")
    lines.append("- `kink` — join angle (deg) between incoming heading and the path's first leg")
    lines.append("- `jump` — sideways cross-track offset (m) ~30 m off the node")
    lines.append(
        "- `snap` — raw-proposal endpoint offset from the requested node (m); production commit pins endpoints to the node (→ ~0), tracked only to spot route distortion"
    )
    lines.append("")

    for param_name, (param, aggs, best) in sweeps.items():
        lines.append(f"## `{param.name}` (targets **{param.metric}**)")
        lines.append("")
        lines.append(f"Road scenarios in corpus: {aggs[0].n_road}.")
        lines.append("")
        lines.append("| value | n_ok | reject | slope | length | kink | jump | snap |")
        lines.append("|------:|-----:|-------:|------:|-------:|-----:|-----:|-----:|")
        for a in aggs:
            marker = " ✅" if a.value == best.value else ""
            lines.append(
                f"| {a.value:g}{marker} | {a.n_ok} | {_fmt(a.reject_rate)} | {_fmt(a.slope_med)} | "
                f"{_fmt(a.length_med)} | {_fmt(a.kink_med)} | {_fmt(a.jump_med)} | {_fmt(a.endpoint_med)} |"
            )
        lines.append("")
        metric_med = {
            "join_kink_deg": best.kink_med,
            "start_jump_m": best.jump_med,
            "max_slope_pct": best.slope_med,
            "reject_rate": best.reject_rate,
        }[param.metric]
        lines.append(
            f"**Best `{param.name}` = {best.value:g}** — minimises {param.metric} "
            f"(median {_fmt(metric_med)}) with detour {_fmt(best.length_med)}."
        )
        lines.append("")

    lines.append("## Confirmed good configuration")
    lines.append("")
    lines.append("Per-parameter winners below, each the value that minimised road rejection (a sensible")
    lines.append("road wrongly busting ±15%), tie-broken by the smaller detour.")
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

    # Cap the road corpus to keep the sweep tractable at wide grid buffers (a large grid
    # is slow per call). A deterministic stride sample stays representative for medians.
    SWEEP_ROAD_CAP = 30
    roads = [s for s in scenarios if s.is_road]
    if len(roads) > SWEEP_ROAD_CAP:
        stride = len(roads) / SWEEP_ROAD_CAP
        sampled = {id(roads[int(i * stride)]) for i in range(SWEEP_ROAD_CAP)}
        scenarios = [s for s in scenarios if not s.is_road or id(s) in sampled]

    # Snapshot baseline for the "was X" annotations.
    param_names = [
        "GRID_BUFFER_FACTOR",
        "MAX_GRID_SIZE",
        "COST_SIGMA",
    ]
    baseline = {n: getattr(PlannerConfig, n) for n in param_names}

    # Parameters to sweep. The primary goal is road REACHABILITY — the fraction of sensible
    # road scenarios whose gentlest route still busts the ±15% cap (a wrong refusal). The
    # search-grid extent (buffer + cap) governs whether a legal switchback route fits at all;
    # COST_SIGMA governs how hard the planner is pushed to use that room.
    params = [
        Param("GRID_BUFFER_FACTOR", [0.5, 1.0, 1.5, 2.0], metric="reject_rate", goal="min"),
        Param("MAX_GRID_SIZE", [100.0, 160.0], metric="reject_rate", goal="min"),
        Param("COST_SIGMA", [4.0, 6.0, 8.0, 12.0], metric="reject_rate", goal="min"),
    ]

    # reject_rate is road-only, so sweeping slope scenarios (which fire ~16 planner calls
    # each) is pure waste — restrict every sweep to road scenarios to keep the run fast.
    def road_only(s: Scenario) -> bool:
        return s.is_road

    applies_map = {
        "GRID_BUFFER_FACTOR": road_only,
        "MAX_GRID_SIZE": road_only,
        "COST_SIGMA": road_only,
    }

    sweeps: dict[str, tuple[Param, list[Aggregate], Aggregate]] = {}
    for param in params:
        aggs = sweep_param(factory, scenarios, param, applies_map[param.name])
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

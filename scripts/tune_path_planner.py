"""Standalone parameter-tuning experiment for the route-geometry pipeline.

Loads a saved resort backup and harvests two real corpora from it, then sweeps the knobs in
`GeometricTuningConfig` and writes ONE Markdown report ending in a confirmed good config:

  - PLANNER scenarios (one per committed segment): replays `PathFactory.generate_manual_paths`
    over the real DEM. Sweeps GRID_BUFFER_FACTOR / MAX_GRID_SIZE / COST_SIGMA against the
    fraction of sensible ROADS whose gentlest route still busts the ±15% cap (a wrong refusal).
  - SMOOTHING scenarios (one per >=2-segment slope/road): replays `smooth_joined_path`.
    Sweeps PIN_WEIGHT / SMOOTHING_FACTOR / RESAMPLE_STEP_M against node-to-ribbon gap,
    sharpest turn, and steepest-section inflation.

Each sweep mutates ONE knob (others held at their current default) and restores it after.
Knobs are read from the config and passed EXPLICITLY into the call under test, because the
production defaults bind at import time.

Run:  python scripts/tune_path_planner.py
Output: docs/experiments/path-tuning.md

OFFLINE analysis: imports the production planner/smoother and constants but mutates nothing
beyond the value under test (restored after each trial).
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, field
from typing import Any, Optional

from skiresort_planner.constants import BACKUP_DIR, PROJECT_ROOT, GeometricTuningConfig, PathConfig
from skiresort_planner.core.dem_service import DEMService
from skiresort_planner.core.geo_calculator import GeoCalculator
from skiresort_planner.generators.path_factory import PathFactory
from skiresort_planner.model.path_geometry import Path
from skiresort_planner.model.path_point import PathPoint
from skiresort_planner.model.path_smoothing import smooth_joined_path
from skiresort_planner.model.proposed_path import ProposedPathSegment

REPORT_PATH = PROJECT_ROOT / "docs" / "experiments" / "path-tuning.md"


def _pt(raw: dict[str, float]) -> PathPoint:
    return PathPoint(lon=raw["lon"], lat=raw["lat"], elevation=raw["elevation"])


# =============================================================================
# Planner corpus + metrics — one "connect start→target" scenario per segment
# =============================================================================


@dataclass(frozen=True)
class PlannerScenario:
    """One 'connect these endpoints' situation from a real committed segment."""

    label: str
    start_lon: float
    start_lat: float
    start_elev: float
    end_lon: float
    end_lat: float
    end_elev: float
    is_road: bool


def harvest_planner_scenarios(backup: dict[str, Any], dem: DEMService) -> list[PlannerScenario]:
    """One scenario per segment of every slope/road, using its endpoints as start/target."""
    segments = backup["segments"]
    scenarios: list[PlannerScenario] = []

    def add_entity(entity_id: str, seg_ids: list[str], is_road: bool) -> None:
        for sid in seg_ids:
            seg = segments.get(sid)
            if seg is None or len(seg["points"]) < 2:
                continue
            start, end = seg["points"][0], seg["points"][-1]
            if dem.get_elevation(lon=start["lon"], lat=start["lat"]) is None:
                continue
            if dem.get_elevation(lon=end["lon"], lat=end["lat"]) is None:
                continue
            scenarios.append(
                PlannerScenario(
                    label=f"{entity_id}:{sid}",
                    start_lon=start["lon"],
                    start_lat=start["lat"],
                    start_elev=start["elevation"],
                    end_lon=end["lon"],
                    end_lat=end["lat"],
                    end_elev=end["elevation"],
                    is_road=is_road,
                )
            )

    for rid, road in backup.get("roads", {}).items():
        add_entity(rid, road["segment_ids"], is_road=True)
    for slid, slope in backup.get("slopes", {}).items():
        add_entity(slid, slope["segment_ids"], is_road=False)
    return scenarios


@dataclass
class PlannerMetrics:
    """Objective measures for one generated planner proposal (all lower = better except ok)."""

    ok: bool  # a path was produced at all
    max_slope_pct: float  # steepest rolling-window grade (road cap = 15%)
    length_ratio: float  # path length / straight-line distance (detour; 1.0 = straight)
    endpoint_err_m: float  # raw grid-snap of the proposal endpoints from the requested nodes
    is_road: bool  # this scenario is a road (subject to the ±15% cap)
    road_rejected: bool  # ROAD scenario whose gentlest proposal still exceeds the ±15% cap


def measure_planner(path: Optional[ProposedPathSegment], sc: PlannerScenario) -> PlannerMetrics:
    """Compute objective metrics for one planner proposal against its scenario."""
    if path is None or len(path.points) < 2:
        # No path at all: for a road that counts as a rejection (nothing to commit).
        return PlannerMetrics(False, 0.0, 0.0, 0.0, is_road=sc.is_road, road_rejected=sc.is_road)
    pts = path.points

    length = sum(pts[i].distance_to(other=pts[i + 1]) for i in range(len(pts) - 1))
    straight = GeoCalculator.haversine_distance_m(
        lat1=sc.start_lat, lon1=sc.start_lon, lat2=sc.end_lat, lon2=sc.end_lon
    )
    length_ratio = length / straight if straight > 0 else 1.0

    # Raw endpoint grid-snap: production commit pins endpoints onto the node (→ ~0); tracked
    # only to detect route distortion (a value far above the grid-cell floor = bent route).
    endpoint_err = max(
        GeoCalculator.haversine_distance_m(lat1=sc.start_lat, lon1=sc.start_lon, lat2=pts[0].lat, lon2=pts[0].lon),
        GeoCalculator.haversine_distance_m(lat1=sc.end_lat, lon1=sc.end_lon, lat2=pts[-1].lat, lon2=pts[-1].lon),
    )
    road_rejected = sc.is_road and path.max_slope_pct > float(PathConfig.ROAD_MAX_GRADIENT_PCT)
    return PlannerMetrics(True, path.max_slope_pct, length_ratio, endpoint_err, sc.is_road, road_rejected)


def run_planner_scenario(factory: PathFactory, sc: PlannerScenario) -> PlannerMetrics:
    """Generate the gentlest proposal for a scenario and measure it."""
    paths = list(
        factory.generate_manual_paths(
            start_lon=sc.start_lon,
            start_lat=sc.start_lat,
            start_elevation=sc.start_elev,
            target_lon=sc.end_lon,
            target_lat=sc.end_lat,
            target_elevation=sc.end_elev,
            road_mode=sc.is_road,
        )
    )
    best = min(paths, key=lambda p: p.max_slope_pct) if paths else None
    return measure_planner(best, sc)


# =============================================================================
# Smoothing corpus + metrics — one finish scenario per >=2-segment entity
# =============================================================================


@dataclass(frozen=True)
class SmoothingScenario:
    """One finished multi-segment entity: committed per-segment points + boundary node coords."""

    label: str
    segment_points: list[list[PathPoint]]
    node_anchors: list[PathPoint]
    is_road: bool


def harvest_smoothing_scenarios(backup: dict[str, Any]) -> list[SmoothingScenario]:
    """One scenario per multi-segment slope/road in the backup (>=2 segments)."""
    segments = backup["segments"]
    nodes = backup["nodes"]
    scenarios: list[SmoothingScenario] = []

    def add_entity(entity_id: str, seg_ids: list[str], is_road: bool) -> None:
        segs = [segments.get(sid) for sid in seg_ids]
        if len(segs) < 2 or any(s is None or len(s["points"]) < 2 for s in segs):
            return
        boundary_node_ids = [segs[0]["start_node_id"], *(s["end_node_id"] for s in segs)]
        if any(nid not in nodes for nid in boundary_node_ids):
            return
        scenarios.append(
            SmoothingScenario(
                label=entity_id,
                segment_points=[[_pt(p) for p in s["points"]] for s in segs],
                node_anchors=[_pt(nodes[nid]["location"]) for nid in boundary_node_ids],
                is_road=is_road,
            )
        )

    for rid, road in backup.get("roads", {}).items():
        add_entity(rid, road["segment_ids"], is_road=True)
    for slid, slope in backup.get("slopes", {}).items():
        add_entity(slid, slope["segment_ids"], is_road=False)
    return scenarios


@dataclass
class SmoothingMetrics:
    """Objective measures for one finished entity after smoothing (all lower = better)."""

    node_gap_max_m: float  # worst distance from a node marker to the smoothed ribbon
    turn_max_deg: float  # sharpest heading change anywhere along the smoothed ribbon
    slope_before_pct: float  # steepest 300m section before smoothing
    slope_after_pct: float  # steepest 300m section after smoothing
    is_road: bool
    over_cap: bool  # ROAD whose steepest section exceeds ±15% AFTER smoothing


def _turn_deg(a: PathPoint, b: PathPoint, c: PathPoint) -> float:
    """Absolute heading change (deg) at b for the polyline a->b->c."""
    h1 = GeoCalculator.initial_bearing_deg(lon1=a.lon, lat1=a.lat, lon2=b.lon, lat2=b.lat)
    h2 = GeoCalculator.initial_bearing_deg(lon1=b.lon, lat1=b.lat, lon2=c.lon, lat2=c.lat)
    d = abs(h1 - h2) % 360
    return d if d <= 180 else 360 - d


def _steepest_pct(segment_points: list[list[PathPoint]]) -> float:
    """Max steepest-300m-section across the entity's segments (mirrors production)."""
    return max(Path(points=pts).max_slope_pct for pts in segment_points)


def measure_smoothing(scenario: SmoothingScenario, smoothed: list[list[PathPoint]]) -> SmoothingMetrics:
    """Compute objective metrics for one smoothed entity against its node anchors."""
    joined = list(smoothed[0])
    for seg in smoothed[1:]:
        joined.extend(seg[1:])

    node_gap = max(
        min(GeoCalculator.haversine_distance_m(lat1=a.lat, lon1=a.lon, lat2=p.lat, lon2=p.lon) for p in joined)
        for a in scenario.node_anchors
    )
    turn_max = max(
        (_turn_deg(joined[i - 1], joined[i], joined[i + 1]) for i in range(1, len(joined) - 1)),
        default=0.0,
    )
    before = _steepest_pct(scenario.segment_points)
    after = _steepest_pct(smoothed)
    over_cap = scenario.is_road and after > float(PathConfig.ROAD_MAX_GRADIENT_PCT)
    return SmoothingMetrics(node_gap, turn_max, before, after, scenario.is_road, over_cap)


def run_smoothing_scenario(scenario: SmoothingScenario) -> SmoothingMetrics:
    """Smooth one scenario with the CURRENT GeometricTuningConfig values and measure it.

    Reads the class attributes and passes them EXPLICITLY — smooth_joined_path binds its
    defaults at import time, so a swept setattr on the class would otherwise be ignored.
    """
    smoothed = smooth_joined_path(
        segment_point_lists=scenario.segment_points,
        node_anchors=scenario.node_anchors,
        step_m=GeometricTuningConfig.RESAMPLE_STEP_M,
        smoothing_factor=GeometricTuningConfig.SMOOTHING_FACTOR,
        pin_weight=GeometricTuningConfig.PIN_WEIGHT,
    )
    return measure_smoothing(scenario, smoothed)


# =============================================================================
# Sweep engine — shared across both corpora
# =============================================================================


@dataclass
class Param:
    """One tunable GeometricTuningConfig knob and the values to sweep for it."""

    name: str
    values: list[float]
    metric: str  # the aggregate field this knob most directly targets


@dataclass
class Aggregate:
    """Corpus-aggregated metrics for one parameter value (fields depend on the sweep group)."""

    value: float
    stats: dict[str, float] = field(default_factory=dict)


def _med(vals: list[float]) -> float:
    return statistics.median(vals) if vals else float("nan")


def _pctl(vals: list[float], q: float) -> float:
    """q-quantile (0..1) via nearest-rank on the sorted list; NaN if empty."""
    if not vals:
        return float("nan")
    ordered = sorted(vals)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def aggregate_planner(metrics: list[PlannerMetrics]) -> dict[str, float]:
    ok = [m for m in metrics if m.ok]
    road = [m for m in metrics if m.is_road]
    return {
        "reject": (sum(1 for m in road if m.road_rejected) / len(road)) if road else float("nan"),
        "slope": _med([m.max_slope_pct for m in ok]),
        "length": _med([m.length_ratio for m in ok]),
        "snap": _med([m.endpoint_err_m for m in ok]),
    }


def aggregate_smoothing(metrics: list[SmoothingMetrics]) -> dict[str, float]:
    road = [m for m in metrics if m.is_road]
    return {
        "gap_med": _med([m.node_gap_max_m for m in metrics]),
        "gap_p95": _pctl([m.node_gap_max_m for m in metrics], 0.95),
        "turn": _med([m.turn_max_deg for m in metrics]),
        "slope_rise": _med([m.slope_after_pct - m.slope_before_pct for m in metrics]),
        "over_cap": (sum(1 for m in road if m.over_cap) / len(road)) if road else float("nan"),
    }


def sweep(param: Param, evaluate: Any) -> list[Aggregate]:
    """Sweep one GeometricTuningConfig knob; evaluate() returns the aggregate stats dict."""
    original = getattr(GeometricTuningConfig, param.name)
    results: list[Aggregate] = []
    try:
        for v in param.values:
            setattr(GeometricTuningConfig, param.name, v)
            agg = Aggregate(value=v, stats=evaluate())
            results.append(agg)
            summary = " ".join(f"{k}={agg.stats[k]:.2f}" for k in sorted(agg.stats))
            print(f"  {param.name}={v:g}: {summary}", flush=True)
    finally:
        setattr(GeometricTuningConfig, param.name, original)  # always restore
    return results


def pick_best(param: Param, aggs: list[Aggregate]) -> Aggregate:
    """Pick the value minimising this knob's target metric, tie-broken sensibly per group.

    Smoothing turn knobs must keep the node gap tight, so they only choose among values whose
    gap_p95 is within 1m of the best; planner knobs tie-break reject-rate by lower detour.
    """
    m = param.metric
    if m == "reject":
        return min(aggs, key=lambda a: (round(a.stats["reject"], 3), round(a.stats["length"], 3)))
    if m == "gap":
        return min(aggs, key=lambda a: (round(a.stats["gap_p95"], 2), round(a.stats["turn"], 1)))
    # turn-targeting smoothing knobs: gentlest turn among gap-safe values, then least slope-rise.
    best_gap = min(a.stats["gap_p95"] for a in aggs)
    gap_ok = [a for a in aggs if a.stats["gap_p95"] <= best_gap + 1.0]
    return min(gap_ok, key=lambda a: (round(a.stats["turn"], 1), round(a.stats["slope_rise"], 2)))


# =============================================================================
# Report
# =============================================================================


def _fmt(x: float) -> str:
    return "n/a" if x != x else f"{x:.2f}"  # NaN check


def write_report(
    n_planner: int,
    n_planner_road: int,
    n_smoothing: int,
    n_smoothing_road: int,
    groups: list[tuple[str, list[str], list[tuple[Param, list[Aggregate], Aggregate]]]],
    baseline: dict[str, float],
    date_str: str,
) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("# Route-geometry parameter tuning — experiment report")
    lines.append("")
    lines.append(f"_Generated by `scripts/tune_path_planner.py` on {date_str}._")
    lines.append("")
    lines.append("## Method")
    lines.append("")
    lines.append(
        f"Loaded the saved resort backup and harvested two real corpora: **{n_planner} planner "
        f"scenarios** ({n_planner_road} road) — one 'connect these endpoints' per committed segment, "
        f"replayed through `PathFactory.generate_manual_paths` over the real DEM — and **{n_smoothing} "
        f"finish scenarios** ({n_smoothing_road} road) — one per >=2-segment slope/road, replayed "
        "through `smooth_joined_path`. Each parameter is swept alone (others at their current default); "
        "metrics are medians over the corpus. Sweeps are one-at-a-time, so cross-knob interactions are "
        "not explored."
    )
    lines.append("")
    lines.append(
        "**Metrics (all lower = better):** `reject` (roads busting ±15%), `slope` (steepest %), "
        "`length` (detour ratio), `snap` (raw endpoint grid-snap, m), `gap_med`/`gap_p95` "
        "(node→ribbon distance, m), `turn` (sharpest bend, °), `slope_rise` (steepest-section "
        "inflation, pp), `over_cap` (roads over ±15% after smoothing)."
    )
    lines.append("")

    for group_title, cols, sweeps in groups:
        lines.append(f"## {group_title}")
        lines.append("")
        for param, aggs, best in sweeps:
            lines.append(f"### `{param.name}` (targets **{param.metric}**)")
            lines.append("")
            lines.append("| value | " + " | ".join(cols) + " |")
            lines.append("|------:|" + "|".join(["----:"] * len(cols)) + "|")
            for a in aggs:
                marker = " ✅" if a.value == best.value else ""
                cells = " | ".join(_fmt(a.stats[c]) for c in cols)
                lines.append(f"| {a.value:g}{marker} | {cells} |")
            lines.append("")

    lines.append("## Confirmed good configuration")
    lines.append("")
    lines.append("```python")
    lines.append("class GeometricTuningConfig:")
    for _title, _cols, sweeps in groups:
        for param, _aggs, best in sweeps:
            cur = baseline[param.name]
            note = "unchanged" if best.value == cur else f"was {cur:g}"
            lines.append(f"    {param.name} = {best.value:g}  # {note}")
    lines.append("```")
    lines.append("")
    lines.append("_Planner knobs are chosen to minimise wrong road refusals (±15% busts) at the smallest")
    lines.append("detour; smoothing knobs pin every node onto the ribbon (gap→0) then buy the gentlest")
    lines.append("turn that keeps the gap tight and the road over-cap rate in check._")

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

    planner_scenarios = harvest_planner_scenarios(backup, dem)
    smoothing_scenarios = harvest_smoothing_scenarios(backup)
    if not planner_scenarios or not smoothing_scenarios:
        raise SystemExit("No usable scenarios harvested (backup empty or outside DEM).")

    # Cap the planner ROAD corpus for tractability — one generate_manual_paths call is slow at
    # wide grids. A deterministic stride sample stays representative for medians.
    SWEEP_ROAD_CAP = 30
    roads = [s for s in planner_scenarios if s.is_road]
    if len(roads) > SWEEP_ROAD_CAP:
        stride = len(roads) / SWEEP_ROAD_CAP
        keep = {id(roads[int(i * stride)]) for i in range(SWEEP_ROAD_CAP)}
        dropped = len(roads) - SWEEP_ROAD_CAP
        planner_scenarios = [s for s in planner_scenarios if not s.is_road or id(s) in keep]
        print(f"Planner corpus: capped roads {len(roads)}→{SWEEP_ROAD_CAP} (dropped {dropped} for speed).", flush=True)
    # Planner sweeps run over roads only (reject-rate is road-only; slopes have no cap).
    planner_roads = [s for s in planner_scenarios if s.is_road]

    param_names = [
        "GRID_BUFFER_FACTOR",
        "MAX_GRID_SIZE",
        "COST_SIGMA",
        "PIN_WEIGHT",
        "SMOOTHING_FACTOR",
        "RESAMPLE_STEP_M",
    ]
    baseline = {n: getattr(GeometricTuningConfig, n) for n in param_names}

    planner_params = [
        Param("GRID_BUFFER_FACTOR", [0.5, 1.0, 1.5, 2.0], metric="reject"),
        Param("MAX_GRID_SIZE", [100.0, 160.0], metric="reject"),
        Param("COST_SIGMA", [4.0, 6.0, 8.0, 12.0], metric="reject"),
    ]
    smoothing_params = [
        Param("PIN_WEIGHT", [10.0, 100.0, 500.0, 1000.0, 5000.0], metric="gap"),
        Param("SMOOTHING_FACTOR", [1.0, 2.0, 3.0, 5.0, 8.0], metric="turn"),
        Param("RESAMPLE_STEP_M", [4.0, 7.0, 10.0, 15.0], metric="turn"),
    ]

    planner_sweeps: list[tuple[Param, list[Aggregate], Aggregate]] = []
    print("Sweeping PLANNER knobs (roads only)...", flush=True)
    for param in planner_params:
        aggs = sweep(param, lambda: aggregate_planner([run_planner_scenario(factory, s) for s in planner_roads]))
        planner_sweeps.append((param, aggs, pick_best(param, aggs)))

    smoothing_sweeps: list[tuple[Param, list[Aggregate], Aggregate]] = []
    print("Sweeping SMOOTHING knobs...", flush=True)
    for param in smoothing_params:
        aggs = sweep(param, lambda: aggregate_smoothing([run_smoothing_scenario(s) for s in smoothing_scenarios]))
        smoothing_sweeps.append((param, aggs, pick_best(param, aggs)))

    groups = [
        ("Planner (grid-Dijkstra routing)", ["reject", "slope", "length", "snap"], planner_sweeps),
        ("Finish smoothing", ["gap_med", "gap_p95", "turn", "slope_rise", "over_cap"], smoothing_sweeps),
    ]

    import os

    date_str = os.environ.get("TUNE_DATE", "unknown date")
    write_report(
        n_planner=len(planner_scenarios),
        n_planner_road=len(planner_roads),
        n_smoothing=len(smoothing_scenarios),
        n_smoothing_road=sum(1 for s in smoothing_scenarios if s.is_road),
        groups=groups,
        baseline=baseline,
        date_str=date_str,
    )
    print(f"Report written to {REPORT_PATH}")


if __name__ == "__main__":
    main()

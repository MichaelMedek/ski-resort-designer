"""Per-kind behavioural specs — the single source of truth for what differs by SegmentKind.

Every place that used to branch on ``is_road`` instead reads ``KIND_SPECS[kind]``. A slope,
a road, and any future kind (e.g. a nordic trail) differ ONLY in the data captured here:
the build-time grade cap, whether the route may climb, the too-steep refusal message, the
graph-finish method, and the state-machine state ids for its 3-state build flow. Adding a
kind = adding one KindSpec entry; the assert below guarantees every SegmentKind is covered.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from skiresort_planner.constants import MapConfig, PathConfig, SlopeConfig, StyleConfig
from skiresort_planner.model.message import PathTooSteepMessage, ToastMessage
from skiresort_planner.model.path_segment import SegmentKind
from skiresort_planner.model.resort_graph import ResortGraph
from skiresort_planner.model.segment_path import SegmentPath


@dataclass(frozen=True)
class KindSpec:
    """Everything that differs between buildable segment kinds.

    Attributes:
        kind: The SegmentKind this spec describes.
        icon: Kind glyph for UI messages/labels.
        max_grade_pct: Build-time hard cap on the steepest section (magnitude).
        may_climb: Whether a route of this kind may gain elevation (roads yes, slopes no).
        too_steep_message: Factory building the refusal toast when no in-cap route fits.
        finish: Group the given committed segment ids into the finished entity.
        starting_state / building_state / custom_path_state: the 3 state-machine state
            ids for this kind's build flow.

    ``display_noun`` (property): the capitalized UI noun ("Slope"/"Road") derived from ``kind``.
    """

    kind: SegmentKind
    icon: str  # Kind glyph for UI messages/labels (⛷️ slope, 🛣️ road).
    max_grade_pct: float
    may_climb: bool
    has_direct_fallback: bool
    too_steep_message: Callable[[float | None], ToastMessage]
    finish: Callable[[ResortGraph, list[str]], SegmentPath | None]
    starting_state: str
    building_state: str
    custom_path_state: str
    # State-machine transition names for this kind's commit/finish flow. Called by name
    # (getattr) so the action layer never branches on kind.
    fan_commit_event: str  # non-connector commit from a fan state (extend the build)
    custom_continue_event: str  # non-connector commit from the custom-path state
    connector_finish_event: str  # connector auto-finish (target is an existing node)
    finish_event: str  # sidebar Finish button
    cancel_event: str  # cancel the whole build

    @property
    def display_noun(self) -> str:
        """Capitalized UI noun for this kind ("Slope"/"Road") — used in sidebar/button labels."""
        return self.kind.capitalize()


KIND_SPECS: dict[SegmentKind, KindSpec] = {
    SegmentKind.SLOPE: KindSpec(
        kind=SegmentKind.SLOPE,
        icon=StyleConfig.SLOPE_ICON,
        max_grade_pct=float(SlopeConfig.MAX_SKIABLE_PCT),
        may_climb=False,
        has_direct_fallback=False,
        too_steep_message=lambda gentlest: PathTooSteepMessage(
            gentlest_pct=gentlest,
            max_grade_pct=float(SlopeConfig.MAX_SKIABLE_PCT),
            subject="to ski",
            two_sided=False,
        ),
        finish=lambda graph, segment_ids: graph.finish_slope(segment_ids=segment_ids),
        starting_state="slope_starting",
        building_state="slope_building",
        custom_path_state="slope_custom_path",
        fan_commit_event="commit_path",
        custom_continue_event="commit_custom_continue",
        connector_finish_event="commit_custom_finish",
        finish_event="finish_slope",
        cancel_event="cancel_slope",
    ),
    SegmentKind.ROAD: KindSpec(
        kind=SegmentKind.ROAD,
        icon=StyleConfig.ROAD_ICON,
        max_grade_pct=float(PathConfig.ROAD_MAX_GRADIENT_PCT),
        may_climb=True,
        has_direct_fallback=True,
        too_steep_message=lambda gentlest: PathTooSteepMessage(
            gentlest_pct=gentlest,
            max_grade_pct=float(PathConfig.ROAD_MAX_GRADIENT_PCT),
            subject="for a car road",
            two_sided=True,
        ),
        finish=lambda graph, segment_ids: graph.finish_road(segment_ids=segment_ids),
        starting_state="road_starting",
        building_state="road_building",
        custom_path_state="road_custom_path",
        fan_commit_event="commit_road",
        custom_continue_event="commit_road_custom_continue",
        connector_finish_event="commit_road_custom_finish",
        finish_event="finish_road",
        cancel_event="cancel_road",
    ),
}

assert set(KIND_SPECS) == set(SegmentKind), "every SegmentKind must have a KindSpec"
assert {k.value for k in SegmentKind} == set(MapConfig.SEGMENT_FLAT_Z), (
    "MapConfig.SEGMENT_FLAT_Z must have a flat-mode z-offset for every SegmentKind"
)

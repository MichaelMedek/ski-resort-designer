"""State Lifecycle Functions - Entry and exit handlers for state machine states.

One idempotent enter_* per state (called by on_enter_* hooks) plus three exit_* for the states with
real teardown (lift/import/merge). Handlers only touch UI/context side-effects — NO workflow
mutations (those live in before_* transition hooks). See DETAILS_UI.md for the state list.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from skiresort_planner.model.path_segment import SegmentKind

if TYPE_CHECKING:
    from collections.abc import Callable

    from skiresort_planner.ui.context import PlannerContext

logger = logging.getLogger(__name__)


# =============================================================================
# Shared enter bodies — the *_starting / *_building / *_custom_path / idle_viewing
# groups each collapse to one of these, so slope and road (and future kinds) cannot drift.
# =============================================================================


def _enter_fan_state(ctx: PlannerContext, kind: SegmentKind) -> None:
    """Enter a fan state (*_starting or *_building): hide the panel, free the last-clicked node
    marker so it can be re-clicked, and arm the fan for this kind. Same for starting and building —
    building just keeps its already-committed segments in the build context.
    """
    ctx.viewing.hide_panel()
    ctx.click_dedup.clear_marker()
    ctx.pending.fan_generation.add(kind)


def _enter_custom_path(ctx: PlannerContext) -> None:
    """Enter a *_custom_path state: free the last-clicked node so re-clicking retargets, and flag
    deferred custom-connect generation to the stored target. Kind-agnostic.
    """
    ctx.click_dedup.clear_marker()
    ctx.pending.custom_connect = True


def _enter_viewing_panel(ctx: PlannerContext) -> None:
    """Enter an idle_viewing_* state: show the info panel and clear all stale build/placement state.
    Identical for slope/lift/road — the before-hook already recorded which entity to view.
    """
    ctx.viewing.show_panel()
    ctx.clear_proposals()
    ctx.clear_builds()
    ctx.clear_custom_connect()
    ctx.clear_lift()
    ctx.selection.clear()
    ctx.click_dedup.clear_marker()


def _enter_placement_mode(ctx: PlannerContext) -> None:
    """Enter a *_placing state (lift/import/merge): hide the panel and free the last-clicked marker.
    The placed scratch (lift start / import center / merge selection) is preserved by its before-hook.
    """
    ctx.viewing.hide_panel()
    ctx.click_dedup.clear_marker()


# =============================================================================
# 1. IDLE_READY - No panel visible, ready to start building
# =============================================================================


def enter_idle_ready(ctx: PlannerContext) -> None:
    """Enter IDLE_READY: clear all building/placement/viewing state and hide panels.

    Preserves map center/zoom, build mode, and segment-length (user preferences). End: clean slate.
    """
    logger.debug("[LIFECYCLE] ENTER: idle_ready - clearing all building state")
    ctx.clear_proposals()
    ctx.clear_builds()
    ctx.clear_custom_connect()
    ctx.clear_lift()
    ctx.selection.clear()
    ctx.click_dedup.clear_marker()
    ctx.viewing.clear()
    logger.debug(
        f"[LIFECYCLE] idle_ready complete: map_center=({ctx.map.lat:.4f}, {ctx.map.lon:.4f}), zoom={ctx.map.zoom}"
    )


# =============================================================================
# 2. IDLE_VIEWING_SLOPE - Panel showing slope details
# =============================================================================


def enter_idle_viewing_slope(ctx: PlannerContext) -> None:
    """Enter IDLE_VIEWING_SLOPE: show the slope panel and clear stale build state (idempotent).

    The before-hook already recorded which slope to view; this guarantees the panel is visible.
    """
    logger.debug("[LIFECYCLE] ENTER: idle_viewing_slope - showing panel, clearing building state")
    _enter_viewing_panel(ctx)


# =============================================================================
# 3. IDLE_VIEWING_LIFT - Panel showing lift details
# =============================================================================


def enter_idle_viewing_lift(ctx: PlannerContext) -> None:
    """Enter IDLE_VIEWING_LIFT: show the lift panel and clear stale build state (idempotent).

    The before-hook already recorded which lift to view; this guarantees the panel is visible.
    """
    logger.debug("[LIFECYCLE] ENTER: idle_viewing_lift - showing panel, clearing building state")
    _enter_viewing_panel(ctx)


# =============================================================================
# 4. SLOPE_STARTING - 0 segments committed, picking first direction
# =============================================================================


def enter_slope_starting(ctx: PlannerContext) -> None:
    """Enter SLOPE_STARTING: hide panel, clear marker dedup, arm the slope fan (idempotent).

    The before-hook set the start selection/node/name; this delegates to the shared fan-state body.
    """
    logger.debug("[LIFECYCLE] ENTER: slope_starting - hiding panel, clearing marker dedup, arming slope fan")
    _enter_fan_state(ctx, SegmentKind.SLOPE)


# =============================================================================
# 5. SLOPE_BUILDING - 1+ segments committed, continuing slope
# =============================================================================


def enter_slope_building(ctx: PlannerContext) -> None:
    """Enter SLOPE_BUILDING: hide panel, preserve committed segments, arm the slope fan (idempotent).

    Reached from starting (first commit), custom-path (commit/cancel), self-loop, or undo-finish.
    """
    logger.debug("[LIFECYCLE] ENTER: slope_building - hiding panel, preserving building context, arming slope fan")
    _enter_fan_state(ctx, SegmentKind.SLOPE)


# =============================================================================
# 7. SLOPE_CUSTOM_PATH - Showing custom path options
# =============================================================================


def enter_slope_custom_path(ctx: PlannerContext) -> None:
    """Enter SLOPE_CUSTOM_PATH: clear marker, flag deferred custom-connect to the stored target.

    Fires on the retarget self-loop too, so a new target click regenerates proposals.
    """
    logger.debug("[LIFECYCLE] ENTER: slope_custom_path - clearing marker, triggering deferred path generation")
    _enter_custom_path(ctx)


# =============================================================================
# 8. LIFT_PLACING - Start selected, waiting for end station
# =============================================================================


def enter_lift_placing(ctx: PlannerContext) -> None:
    """Enter LIFT_PLACING: hide panel, clear marker dedup, ready for the second-station click.

    The before-hook set lift.first_node_id/first_location.
    """
    logger.debug("[LIFECYCLE] ENTER: lift_placing - hiding panel")
    _enter_placement_mode(ctx)


def exit_lift_placing(ctx: PlannerContext) -> None:
    """Exit LIFT_PLACING: clear the lift context now that placement is done.

    Panel show/hide is handled by before_complete_lift / before_cancel_lift.
    """
    logger.debug("[LIFECYCLE] EXIT: lift_placing - clearing lift context")
    ctx.lift.clear()


def exit_import_placing(ctx: PlannerContext) -> None:
    """Exit IMPORT_PLACING: clear the placed import-box center so no stale box survives.

    Leaves the osm_import_mode fetch flag alone — a confirmed import sets it just before this runs.
    """
    logger.debug("[LIFECYCLE] EXIT: import_placing - clearing placed import-box center")
    ctx.pending.osm_import_center_lon = None
    ctx.pending.osm_import_center_lat = None


def exit_merge_placing(ctx: PlannerContext) -> None:
    """Exit MERGE_PLACING: clear the node-merge selection so none survives the exit."""
    logger.debug("[LIFECYCLE] EXIT: merge_placing - clearing merge selection")
    ctx.merge.clear()


def enter_import_placing(ctx: PlannerContext) -> None:
    """Enter IMPORT_PLACING: hide panel; the placed box center (set by before_start_import) survives.

    Re-entered on each retarget self-loop, so the center must NOT be cleared here.
    """
    logger.debug("[LIFECYCLE] ENTER: import_placing - hiding panel")
    _enter_placement_mode(ctx)


def enter_merge_placing(ctx: PlannerContext) -> None:
    """Enter MERGE_PLACING: hide panel; the merge selection (ctx.merge.node_ids) survives.

    Re-entered on each node-toggle self-loop, so the selection must NOT be cleared here.
    """
    logger.debug("[LIFECYCLE] ENTER: merge_placing - hiding panel")
    _enter_placement_mode(ctx)


# =============================================================================
# 9. IDLE_VIEWING_ROAD - Panel showing road details
# =============================================================================


def enter_idle_viewing_road(ctx: PlannerContext) -> None:
    """Enter IDLE_VIEWING_ROAD: Make road panel visible (Single Point of Truth).

    Mirrors enter_idle_viewing_lift: the road_id was set by a before_* hook;
    this guarantees the panel is visible and clears any stale building state.
    """
    logger.debug("[LIFECYCLE] ENTER: idle_viewing_road - showing panel, clearing building state")
    _enter_viewing_panel(ctx)


# =============================================================================
# 10. ROAD_STARTING / ROAD_BUILDING - segment-by-segment, like a slope
# =============================================================================


def enter_road_starting(ctx: PlannerContext) -> None:
    """Enter ROAD_STARTING: mirror of enter_slope_starting for roads.

    Hide panel, clear marker dedup, arm the road fan from the origin set by before_start_road.
    """
    logger.debug("[LIFECYCLE] ENTER: road_starting - hiding panel, clearing marker dedup, arming road fan")
    _enter_fan_state(ctx, SegmentKind.ROAD)


def enter_road_building(ctx: PlannerContext) -> None:
    """Enter ROAD_BUILDING: mirror of enter_slope_building for roads.

    Preserve committed segments, hide panel, arm the road fan from the new endpoint.
    """
    logger.debug("[LIFECYCLE] ENTER: road_building - hiding panel, preserving road context, arming road fan")
    _enter_fan_state(ctx, SegmentKind.ROAD)


def enter_road_custom_path(ctx: PlannerContext) -> None:
    """Enter ROAD_CUSTOM_PATH: mirror of enter_slope_custom_path for roads.

    Flags shared deferred custom-connect to the stored target; fires on the retarget self-loop too.
    """
    logger.debug("[LIFECYCLE] ENTER: road_custom_path - clearing marker, triggering deferred custom-connect generation")
    _enter_custom_path(ctx)


# The ONLY states with real exit teardown → their exit function.
EXIT_HOOKS: dict[str, Callable[[PlannerContext], None]] = {
    "lift_placing": exit_lift_placing,
    "import_placing": exit_import_placing,
    "merge_placing": exit_merge_placing,
}

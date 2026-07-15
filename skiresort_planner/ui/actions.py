"""UI Actions - All action functions for ski resort planner.

Centralizes all action functions that modify UI state, trigger state
machine transitions, or perform business logic operations.

This module handles:
- Map centering (center_on_slope, center_on_lift)
- Path operations (commit_selected_path, recompute_paths)
- Slope operations (finish_current_slope, cancel_current_slope)
- Undo operations (undo_last_action)
- Custom direction mode (enter/cancel)
- Deferred action handling (process_*_deferred)
"""

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, cast

import streamlit as st

from skiresort_planner.constants import MapConfig, MergeConfig
from skiresort_planner.generators.osm_importer import OSMImporter, bbox_around
from skiresort_planner.generators.path_factory import PathFactory
from skiresort_planner.model.actions import (
    ActionType,
    AddLiftAction,
    AddSegmentsAction,
    DeleteLiftAction,
    DeleteRoadAction,
    DeleteSlopeAction,
    FinishRoadAction,
    FinishSlopeAction,
    ImportOSMAction,
    MergeNodesAction,
    UndoAction,
)
from skiresort_planner.model.lift import Lift
from skiresort_planner.model.message import (
    MergeTooFarMessage,
    OSMImportErrorMessage,
)
from skiresort_planner.model.path_segment import SegmentKind
from skiresort_planner.model.resort_graph import ResortGraph
from skiresort_planner.ui.context import PlannerContext
from skiresort_planner.ui.infra import bump_camera_epoch, bump_dedup_epoch, reload_map, trigger_rerun
from skiresort_planner.ui.kind_spec import KIND_SPECS
from skiresort_planner.ui.state_machine import PlannerStateMachine

if TYPE_CHECKING:
    from skiresort_planner.model.node import Node
    from skiresort_planner.model.proposed_path import ProposedPathSegment
    from skiresort_planner.model.road import Road
    from skiresort_planner.model.segment_path import SegmentPath
    from skiresort_planner.model.slope import Slope
    from skiresort_planner.ui.context import SegmentBuildContext

logger = logging.getLogger(__name__)


# =============================================================================
# MAP CENTERING
# =============================================================================


def center_on_segment_path(
    ctx: PlannerContext,
    graph: ResortGraph,
    path: "SegmentPath",
    zoom: int,
    pitch: float = MapConfig.VIEWING_PITCH,
) -> None:
    """Center map on a segment-group entity's midpoint (shared by slopes and roads)."""
    segments = [graph.segments.get(sid) for sid in path.segment_ids]
    if segments and segments[0] and segments[-1]:
        first_seg, last_seg = segments[0], segments[-1]
        if first_seg.points and last_seg.points:
            start_pt, end_pt = first_seg.points[0], last_seg.points[-1]
            ctx.map.set_center(
                lon=(start_pt.lon + end_pt.lon) / 2,
                lat=(start_pt.lat + end_pt.lat) / 2,
            )
            ctx.map.zoom = zoom
            ctx.map.pitch = pitch


def center_on_slope(
    ctx: PlannerContext,
    graph: ResortGraph,
    slope: "Slope",
    zoom: int,
    pitch: float = MapConfig.VIEWING_PITCH,
) -> None:
    """Center map on slope midpoint with specified zoom and pitch."""
    center_on_segment_path(ctx=ctx, graph=graph, path=slope, zoom=zoom, pitch=pitch)


def center_on_road(
    ctx: PlannerContext,
    graph: ResortGraph,
    road: "Road",
    zoom: int,
    pitch: float = MapConfig.VIEWING_PITCH,
) -> None:
    """Center map on road midpoint with specified zoom and pitch."""
    center_on_segment_path(ctx=ctx, graph=graph, path=road, zoom=zoom, pitch=pitch)


def center_on_lift(
    ctx: PlannerContext,
    graph: ResortGraph,
    lift: Lift,
    zoom: int,
    pitch: float = MapConfig.VIEWING_PITCH,
) -> None:
    """Center map on lift midpoint with specified zoom and pitch."""
    start_node = graph.nodes.get(lift.start_node_id)
    end_node = graph.nodes.get(lift.end_node_id)
    if start_node and end_node:
        ctx.map.set_center(
            lon=(start_node.lon + end_node.lon) / 2,
            lat=(start_node.lat + end_node.lat) / 2,
        )
        ctx.map.zoom = zoom
        ctx.map.pitch = pitch


# =============================================================================
# DEFERRED ACTIONS
# =============================================================================


def process_custom_connect_deferred() -> bool:
    """Process pending custom connect path generation.

    Call this wrapped in st.spinner() from app.py.

    Returns:
        True if processed, False if nothing pending.
    """
    ctx: PlannerContext = st.session_state.context

    if not ctx.deferred.custom_connect:
        return False

    ctx.deferred.custom_connect = False
    _generate_custom_connect_paths()
    # New proposals were generated → bump dedup_epoch so re-clicking the same index registers. This
    # runs inside the current render cycle from app.py, so the natural render shows the new proposals.
    bump_dedup_epoch()
    return True


def confirm_import_action() -> None:
    """Confirm the placed OSM import box: flag the deferred fetch and return to idle.

    Called by both the right-panel "Confirm Import" button and the center-dot click. The box
    center is already stored in ctx.deferred.osm_import_center_lon/lat (set when the box was
    placed/retargeted). The slow network fetch + graph mutation happen in
    process_osm_import_deferred() under a spinner after we return to idle.
    """
    ctx: PlannerContext = st.session_state.context
    sm: PlannerStateMachine = st.session_state.state_machine
    ctx.deferred.osm_import = True
    bump_dedup_epoch()
    sm.complete_import()  # → idle_ready; listener triggers the rerun that runs the deferred fetch


def confirm_merge_action() -> None:
    """Confirm the node-merge selection: collapse the selected nodes to their median, return to idle.

    Validates the span first for a friendly toast — if any pair exceeds MergeConfig.MAX_SPAN_M the
    merge is refused and nothing changes (the state stays in merge_placing so the user can adjust the
    selection). On success the merge is one undoable action and we return to idle.
    """
    ctx: PlannerContext = st.session_state.context
    sm: PlannerStateMachine = st.session_state.state_machine
    graph: ResortGraph = st.session_state.graph
    dem = st.session_state.dem_service

    node_ids = list(ctx.merge.node_ids)
    if len(node_ids) < 2:
        # The Confirm button is disabled below 2, so this is a defensive guard, not a user path.
        raise RuntimeError("confirm_merge_action called with fewer than 2 selected nodes")

    span = graph.max_node_span_m(node_ids)
    if span > MergeConfig.MAX_SPAN_M:
        logger.info(f"Merge refused: span {span:.0f}m > {MergeConfig.MAX_SPAN_M:.0f}m")
        MergeTooFarMessage(span_m=span, max_span_m=MergeConfig.MAX_SPAN_M).display()
        return  # no state change — the user can deselect the far node and retry

    graph.merge_nodes(node_ids=node_ids, dem=dem)
    logger.info(f"Merged {len(node_ids)} nodes into {node_ids[0]}")
    bump_dedup_epoch()
    sm.complete_merge()  # → idle_ready; the before-hook clears the selection


def process_osm_import_deferred() -> bool:
    """Process a pending OSM import: fetch the chosen area, convert, add as one undoable batch.

    The area is the square box the user placed and confirmed (ctx.deferred.osm_import_center_*
    + half-width). Call this wrapped in st.spinner() from app.py. Returns True if it handled a
    pending import. Any network/parse error shows an error toast and imports nothing.
    """
    ctx: PlannerContext = st.session_state.context
    graph: ResortGraph = st.session_state.graph

    if not ctx.deferred.osm_import:
        return False
    ctx.deferred.osm_import = False

    dem = st.session_state.dem_service
    # The box center is always placed before confirm (start_import stores it)
    center_lon = ctx.deferred.osm_import_center_lon
    center_lat = ctx.deferred.osm_import_center_lat
    if center_lon is None or center_lat is None:
        raise RuntimeError("Pending OSM import has no placed center — start_import must set it before confirm.")
    bbox = bbox_around(
        center_lon=center_lon, center_lat=center_lat, half_width_m=ctx.deferred.osm_import_half_width_km * 1000.0
    )
    # Consume the placed center so a later import can't reuse a stale box.
    ctx.deferred.osm_import_center_lon = None
    ctx.deferred.osm_import_center_lat = None

    try:
        importer = OSMImporter(dem=dem)
        summary = importer.convert(bbox=bbox, elements=importer.fetch(bbox=bbox))
    except Exception as exc:  # network / HTTP / parse — report, import nothing
        logger.warning(f"OSM import failed: {exc}")
        OSMImportErrorMessage(error=str(exc)).display()
        return True

    graph.import_osm(
        pistes=[(p.points, p.name) for p in summary.pistes],
        lifts=[(lift.bottom, lift.top, lift.lift_type, lift.name) for lift in summary.lifts],
        dem=dem,
    )
    bump_dedup_epoch()
    return True


def process_path_generation_deferred() -> bool:
    """Process pending path generation.

    Call this wrapped in st.spinner() from app.py.

    Returns:
        True if processed, False if nothing pending.
    """
    sm: PlannerStateMachine = st.session_state.state_machine
    ctx: PlannerContext = st.session_state.context

    if not ctx.deferred.fan_generation:
        return False

    # Regenerate the fan for each pending kind that is actually the active build.
    if sm.active_build_kind in ctx.deferred.fan_generation:
        _generate_fan_for_building_state(kind=sm.active_build_kind)
        # New proposals → bump dedup_epoch (NOT the camera). Runs inside the current render cycle
        # from app.py, so the following natural render shows the new proposals in place.
        bump_dedup_epoch()

    ctx.deferred.fan_generation.clear()
    return True


def _generate_fan_for_building_state(kind: SegmentKind) -> None:
    """Generate the fan of proposals radiating from the active build's endpoint.

    Kind-generic: the factory builds the kind's target set (slopes descend green→black;
    roads fan signed green descend/climb/flat), every proposal is hard-capped at the
    kind's max grade, and if routes existed but all exceed the cap the gentlest grade is
    stashed for the right-panel "too steep" detail. On an empty (no-route) fan nothing is shown.
    """
    ctx: PlannerContext = st.session_state.context
    graph: ResortGraph = st.session_state.graph
    factory: PathFactory = st.session_state.path_factory
    spec = KIND_SPECS[kind]

    lon, lat, elevation, start_node_id = resolve_build_origin(build=ctx.build(kind), graph=graph)

    fan = list(
        factory.generate_fan(kind=kind, lon=lon, lat=lat, elevation=elevation, target_length_m=ctx.segment_length_m)
    )
    kept, gentlest = factory.filter_by_max_grade(paths=fan, cap_pct=spec.max_grade_pct)

    # Extending from an existing node: reuse it exactly on commit (never duplicate it).
    if start_node_id is not None:
        for p in kept:
            p.start_node_id = start_node_id

    ctx.proposals.paths = kept
    # When the fan had routes but all exceed the cap, stash the gentlest grade so the right panel
    # can explain why. `gentlest` is None when the fan was empty for another reason.
    ctx.proposals.too_steep_gentlest_pct = gentlest if (fan and not kept) else None

    # Smart recommendation: pre-select the proposal whose gradient is closest to the last
    # committed segment's gradient. Set by commit_selected_path for EVERY kind (slopes and
    # roads), so both get continuity of grade across committed segments.
    if kept and ctx.deferred.gradient_target is not None:
        best_idx = _find_closest_gradient_path(paths=kept, target_gradient=ctx.deferred.gradient_target)
        ctx.proposals.selected_idx = best_idx
        ctx.deferred.gradient_target = None
    else:
        ctx.proposals.selected_idx = 0 if kept else None
    logger.debug(f"Generated {len(kept)} {kind.value}-fan paths from ({lat:.6f}, {lon:.6f})")


def resolve_build_origin(
    build: "SegmentBuildContext", graph: ResortGraph, *, custom_start_node: str | None = None
) -> tuple[float, float, float, str | None]:
    """Resolve the point a build's next path radiates from — the single origin resolver.

    Priority: the current committed endpoint → the origin node (a custom-connect re-target origin or
    the build's own starting node) → the pending terrain location. Returns (lon, lat, elevation,
    node_id); node_id is the node to reuse on commit, or None when the origin is a location with no
    node yet (commit_paths mints it).

    A committed endpoint must exist (strict). The ORIGIN node id, by contrast, is only a reuse hint:
    it can be cleaned as isolated once the last segment is undone, so start_location is the
    authoritative fallback. Raises only if neither an endpoint, a live origin node, nor a location
    is available.
    """
    if build.endpoints:
        node = graph.nodes[build.endpoints[-1]]  # committed endpoint — must be live
        return node.lon, node.lat, node.elevation, node.id

    origin_node_id = custom_start_node or build.start_node_id
    if origin_node_id is not None and origin_node_id in graph.nodes:
        node = graph.nodes[origin_node_id]
        return node.lon, node.lat, node.elevation, node.id
    if build.start_location is not None:
        # Fresh terrain origin, or an origin whose node was cleaned when its last segment was undone.
        loc = build.start_location
        return loc.lon, loc.lat, loc.elevation, None
    raise ValueError(f"cannot resolve build origin: {build=} has no start node or location")


def _generate_custom_connect_paths() -> None:
    """Generate proposals routing the active build to the clicked custom target (any kind).

    Kind-generic via KIND_SPECS. The grid planner's in-cap serpentine routes are offered, and the
    direct straight line is ALWAYS appended on top (last, not pre-selected) when it is itself within
    the kind's max grade — so the user can choose it even when curvy routes exist. When nothing fits
    (no in-cap serpentine and the straight line over cap), the proposals list is left empty and the
    gentlest over-cap grade is stashed so the right panel can explain why (no toast).
    """
    ctx: PlannerContext = st.session_state.context
    graph: ResortGraph = st.session_state.graph
    factory: PathFactory = st.session_state.path_factory
    sm: PlannerStateMachine = st.session_state.state_machine

    if not ctx.custom_connect.target_location:
        logger.debug("No custom target location set")
        ctx.clear_custom_connect()
        return

    target_lon, target_lat, target_elevation = ctx.custom_connect.target_location
    if target_elevation is None:
        # The click handler only routes a custom target with a validated elevation.
        raise RuntimeError(f"custom target ({target_lat:.6f}, {target_lon:.6f}) has no elevation")
    kind = sm.active_build_kind
    spec = KIND_SPECS[kind]
    build = ctx.build(kind)

    # Resolve the origin (shared resolver): a committed endpoint, the re-target origin, the starting
    # node, or the pending terrain location. start_node_id is None for a fresh terrain origin →
    # commit_paths mints the node from these coords.
    start_lon, start_lat, start_elevation, start_node_id = resolve_build_origin(
        build=build, graph=graph, custom_start_node=ctx.custom_connect.start_node
    )

    candidates = list(
        factory.generate_manual_paths(
            kind=kind,
            start_lon=start_lon,
            start_lat=start_lat,
            start_elevation=start_elevation,
            target_lon=target_lon,
            target_lat=target_lat,
            target_elevation=target_elevation,
        )
    )
    cap = spec.max_grade_pct
    proposals, gentlest = factory.filter_by_max_grade(paths=candidates, cap_pct=cap)

    # Always also offer the straight line ON TOP of the planner proposals (for every kind), so the
    # user can pick it even when curvy routes exist — appended LAST so the planner's gentlest stays
    # the default selection. Only offered when it is itself within the cap.
    straight = factory.straight_line(
        kind=kind,
        start_lon=start_lon,
        start_lat=start_lat,
        start_elevation=start_elevation,
        target_lon=target_lon,
        target_lat=target_lat,
        target_elevation=target_elevation,
    )
    if straight.max_slope_pct <= cap:
        proposals.append(straight)
    elif not proposals:
        # Nothing fits (no in-cap serpentine AND the straight line is over cap): record the gentlest
        # grade seen so the right panel can explain WHY.
        gentlest = straight.max_slope_pct if gentlest is None else min(gentlest, straight.max_slope_pct)

    # Reuse the existing origin node exactly on commit. When there is no node yet (fresh terrain),
    # leave start_node_id unset — commit_paths mints the origin from the path's first point.
    if start_node_id is not None:
        for p in proposals:
            p.start_node_id = start_node_id

    # Target node: prefer the clicked node's identity (drift-proof); fall back to a
    # proximity lookup only for a terrain target that happens to sit on a node.
    target_node: Node | None
    if ctx.custom_connect.target_node and ctx.custom_connect.target_node in graph.nodes:
        target_node = graph.nodes[ctx.custom_connect.target_node]
    else:
        target_node = graph.find_nearest_node(
            lon=target_lon, lat=target_lat, threshold_m=MapConfig.LIFT_END_NODE_THRESHOLD_M
        )
    if target_node is not None:
        for p in proposals:
            p.is_connector = True
            p.target_node_id = target_node.id
            p.sector_name = f"🔗 {p.sector_name}"

    ctx.proposals.paths = proposals
    ctx.proposals.selected_idx = 0 if proposals else None
    ctx.proposals.too_steep_gentlest_pct = None if proposals else gentlest
    # Note: force_mode, target_location, start_node already set by the target before-hook.
    # No cleanup here - before_cancel_* and before_commit_* hooks handle it on exit.
    logger.debug(
        f"Generated {len(proposals)} custom paths from ({start_lat:.6f}, {start_lon:.6f}) "
        f"to ({target_lat:.6f}, {target_lon:.6f})"
    )


def _find_closest_gradient_path(paths: "list[ProposedPathSegment]", target_gradient: float) -> int:
    """Find index of path with gradient closest to target."""
    if not paths:
        return 0
    best_idx = 0
    best_diff = float("inf")
    for i, path in enumerate(paths):
        diff = abs(path.avg_slope_pct - target_gradient)
        if diff < best_diff:
            best_diff = diff
            best_idx = i
    return best_idx


# =============================================================================
# PATH OPERATIONS
# =============================================================================


def _finalize_entity(kind: SegmentKind) -> "SegmentPath":
    """Finalize the active build's committed segments into an entity and recenter.

    Kind-generic: groups the build's segments via the kind's finish method, recenters,
    bumps the map. The caller fires the kind's finish event. Returns the finalized entity.
    """
    ctx: PlannerContext = st.session_state.context
    graph: ResortGraph = st.session_state.graph
    build = ctx.build(kind)

    entity = KIND_SPECS[kind].finish(graph, build.segments)
    logger.info(f"{kind.value.capitalize()} {entity.name} (id={entity.id}) finalized")
    center_on_segment_path(ctx=ctx, graph=graph, path=entity, zoom=MapConfig.VIEWING_ZOOM)
    bump_camera_epoch()  # recenter on the finished entity (remount re-reads the framed view)
    return entity


def _finish_current_entity(kind: SegmentKind) -> None:
    """Finish the current build from committed segments (sidebar Finish buttons), any kind."""
    sm: PlannerStateMachine = st.session_state.state_machine
    ctx: PlannerContext = st.session_state.context

    build = ctx.build(kind)
    if not build.segments:
        raise RuntimeError(f"finish called with no {kind.value} segments")

    entity = _finalize_entity(kind)
    sm.send(KIND_SPECS[kind].finish_event, entity_id=entity.id)


def _finish_connector(*, segment_id: str, kind: SegmentKind) -> None:
    """Auto-finish a connector segment (target is an existing node), any kind.

    Appends the segment to the build, finalizes the entity, fires the kind's
    connector-finish event.
    """
    sm: PlannerStateMachine = st.session_state.state_machine
    ctx: PlannerContext = st.session_state.context

    ctx.build(kind).segments.append(segment_id)
    entity = _finalize_entity(kind)
    sm.send(KIND_SPECS[kind].connector_finish_event, segment_id=segment_id, entity_id=entity.id)


def commit_selected_path(path_idx: int) -> None:
    """Commit the selected proposal, unified across slopes and roads.

    A connector (target is an existing node) auto-finishes the entity via
    _finish_connector; otherwise a road self-loops (commit_road) and a slope
    continues building (commit_path / commit_custom_continue).
    """
    sm: PlannerStateMachine = st.session_state.state_machine
    ctx: PlannerContext = st.session_state.context
    graph: ResortGraph = st.session_state.graph

    if path_idx < 0 or path_idx >= len(ctx.proposals.paths):
        raise RuntimeError(f"Invalid path index {path_idx}, valid range: 0-{len(ctx.proposals.paths) - 1}")

    path = ctx.proposals.paths[path_idx]
    is_connector = bool(path.is_connector and path.target_node_id)
    kind = sm.active_build_kind
    committed_gradient = path.avg_slope_pct

    ctx.custom_connect.clear()
    end_node_ids = graph.commit_paths(paths=[path])
    if not end_node_ids:
        raise RuntimeError(f"graph.commit_paths() returned empty for path {path_idx + 1}")

    segment_id = list(graph.segments.keys())[-1]
    assert segment_id in graph.segments, (
        f"segment_id {segment_id} not found in graph.segments after commit_paths (internal state corruption)"
    )
    endpoint_node_id = end_node_ids[0]
    assert endpoint_node_id in graph.nodes, (
        f"endpoint_node_id {endpoint_node_id} returned by commit_paths not in graph.nodes (internal state bug)"
    )
    logger.info(
        f"Committed path {path_idx + 1} as segment {segment_id}: "
        f"{path.length_m:.0f}m, {path.avg_slope_pct:.1f}%, endpoint={endpoint_node_id}"
    )

    # Connector → auto-finish the entity (any kind).
    if is_connector:
        _finish_connector(segment_id=segment_id, kind=kind)
        return

    # Non-connector: re-arm the fan and fire the kind's commit event (fan-state self-loop, or
    # custom-continue in the custom-path state). Do NOT recenter — the user keeps their current pan;
    # only finish/show-view/reset/3D re-frame the camera.
    ctx.clear_proposals()
    ctx.deferred.fan_generation.add(kind)
    ctx.deferred.gradient_target = committed_gradient
    bump_dedup_epoch()

    sm.commit_active_segment(segment_id=segment_id, endpoint_node_id=endpoint_node_id)


def recompute_paths() -> None:
    """Regenerate proposals from the current position (segment-length slider changed).

    Delegates to the same two generators the click flow uses: the custom-connect
    generator when a target is locked in, else the active kind's fan.
    """
    sm: PlannerStateMachine = st.session_state.state_machine
    ctx: PlannerContext = st.session_state.context

    ctx.click_dedup.pending_recompute = False

    # Custom target mode - regenerate to stored target (force_mode ⟺ target_location is set).
    if ctx.custom_connect.force_mode and ctx.custom_connect.start_node:
        _generate_custom_connect_paths()
    else:
        ctx.clear_custom_connect()
        _generate_fan_for_building_state(kind=sm.active_build_kind)
    reload_map()  # Clear stale click state so proposal 1 can be clicked


# =============================================================================
# SLOPE OPERATIONS
# =============================================================================


def finish_current_build(kind: SegmentKind) -> None:
    """Finish the active build and create the finalized entity (sidebar Finish button), any kind.

    Kind-generic entry the unified path-build sidebar panel calls; the slope/road wrappers below
    delegate here so there is one implementation.
    """
    _finish_current_entity(kind=kind)


def finish_current_slope() -> None:
    """Finish building and create the finalized slope (sidebar Finish button)."""
    finish_current_build(kind=SegmentKind.SLOPE)


def _discard_build(build_ctx: "SegmentBuildContext") -> None:
    """Discard an in-progress slope/road build: strip its undo entries, delete segments, clean up.

    Shared by cancel_current_slope / cancel_current_road (the SM cancel event is fired by each
    caller). Does NOT recenter — cancel keeps the user's current pan.
    """
    graph: ResortGraph = st.session_state.graph

    logger.info(f"Canceling build, discarding {len(build_ctx.segments)} segments")

    # Delete the canceled segments, then drop the undo entries that referenced them — the SAME
    # scrub the graph runs after delete/merge (one implementation, reload-safe, Finish-aware).
    for seg_id in build_ctx.segments:
        if seg_id in graph.segments:
            del graph.segments[seg_id]
    graph.drop_undo_actions_for_removed_segments()

    graph.cleanup_isolated_nodes()  # Remove orphaned nodes from the canceled build
    bump_dedup_epoch()  # canceled segments gone → refresh marker/proposal dedup (no recenter)


def cancel_current_build(kind: SegmentKind) -> None:
    """Cancel the active build and discard its segments (sidebar Cancel button), any kind.

    Kind-generic entry the unified path-build sidebar panel calls; discards the build then fires
    the kind's cancel event from KIND_SPECS. The slope/road wrappers below delegate here.
    """
    sm: PlannerStateMachine = st.session_state.state_machine
    ctx: PlannerContext = st.session_state.context
    _discard_build(build_ctx=ctx.build(kind))
    sm.send(KIND_SPECS[kind].cancel_event)


def cancel_current_slope() -> None:
    """Cancel slope building and discard segments."""
    cancel_current_build(kind=SegmentKind.SLOPE)


def finish_current_road() -> None:
    """Finish building and create the finalized road (sidebar Finish button)."""
    finish_current_build(kind=SegmentKind.ROAD)


def cancel_current_road() -> None:
    """Cancel road building and discard its segments (mirrors cancel_current_slope)."""
    cancel_current_build(kind=SegmentKind.ROAD)


def _undo_add_segments(undone: AddSegmentsAction) -> None:
    """Handle undo of ADD_SEGMENTS action.

    Uses force_idle/force_building instead of state machine transitions.
    This follows the expert recommendation to treat undo as history management,
    not core workflow state transitions.
    """
    sm: PlannerStateMachine = st.session_state.state_machine
    ctx: PlannerContext = st.session_state.context
    graph: ResortGraph = st.session_state.graph

    # Kind-generic: peel the undone segments off the active build, then force the kind's
    # BUILDING state (segments remain) or STARTING state (origin remains, no segments).
    # force_* re-triggers the kind's fan from the restored endpoint.
    kind = sm.active_build_kind
    build = ctx.build(kind)
    remaining = [s for s in build.segments if s not in undone.segment_ids]
    build.segments = remaining
    ctx.clear_proposals()
    if remaining:
        last_seg = graph.segments.get(remaining[-1])
        build.endpoints = [last_seg.end_node_id] if last_seg else []
        logger.debug(f"[ACTION] {kind.value} undo leaves {len(remaining)} segments, forcing building")
        sm.force_building(kind)
    else:
        build.endpoints = []
        logger.debug(f"[ACTION] {kind.value} undo leaves 0 segments, forcing starting")
        sm.force_starting(kind)
    bump_dedup_epoch()
    trigger_rerun()


def _restore_build_context(
    build_ctx: "SegmentBuildContext",
    segment_ids: tuple[str, ...],
    name: str | None,
    start_node_id: str | None,
) -> str:
    """Restore a build context from a finish-undo action; return the last endpoint node id.

    A finish action always references ≥1 committed segment (finish_slope/finish_road
    return None otherwise) and those segments are kept on undo — so the context is always re-enterable.
    """
    graph: ResortGraph = st.session_state.graph
    logger.info(f"Undone finish, restoring {len(segment_ids)} segments")

    build_ctx.segments = list(segment_ids)
    build_ctx.name = name
    build_ctx.start_node_id = start_node_id

    assert build_ctx.segments, "finish-undo must have ≥1 segment (finish_* never records an empty finish)"
    first_seg = graph.segments.get(build_ctx.segments[0])
    last_seg = graph.segments.get(build_ctx.segments[-1])
    assert first_seg and first_seg.points, f"restored segment {build_ctx.segments[0]} must exist with points"
    assert last_seg and last_seg.points, f"restored segment {build_ctx.segments[-1]} must exist with points"

    # Carry the origin as a LOCATION too, not only as start_node_id: undoing the segments one by one
    # eventually cleans the origin node. The origin node still exists now, so snapshot the first segment's start point.
    build_ctx.start_location = first_seg.points[0]
    build_ctx.endpoints = [last_seg.end_node_id]
    return last_seg.end_node_id


def _undo_finish(kind: SegmentKind, segment_ids: tuple[str, ...], name: str, start_node_id: str | None) -> None:
    """Handle undo of a FINISH action (slope or road): restore building + re-arm the fan.

    The graph already ungrouped the entity (segments kept). Restore the kind's build
    context and force its BUILDING state; force_building re-triggers the fan.
    """
    ctx: PlannerContext = st.session_state.context
    sm: PlannerStateMachine = st.session_state.state_machine

    _restore_build_context(
        build_ctx=ctx.build(kind),
        segment_ids=segment_ids,
        name=name,
        start_node_id=start_node_id,
    )
    ctx.clear_proposals()  # force_building re-triggers the kind's fan from the endpoint
    sm.force_building(kind)
    bump_dedup_epoch()
    trigger_rerun()


def _undo_finish_slope(undone: FinishSlopeAction) -> None:
    """Handle undo of FINISH_SLOPE."""
    _undo_finish(
        kind=SegmentKind.SLOPE,
        segment_ids=undone.segment_ids,
        name=undone.slope_name,
        start_node_id=undone.start_node_id,
    )


def _undo_add_lift(undone: AddLiftAction) -> None:
    """Handle undo of ADD_LIFT action."""
    sm: PlannerStateMachine = st.session_state.state_machine

    logger.info(f"Undone lift addition: {undone.lift_id}")

    # If in LiftPlacing state, force to idle (lift placement context is now stale)
    if sm.is_lift_placing:
        sm.force_idle()
        trigger_rerun()
        return

    # If we were viewing the deleted lift, force to idle (exit hooks handle cleanup)
    if sm.is_idle_viewing_lift and st.session_state.context.viewing.lift_id == undone.lift_id:
        sm.force_idle()
        trigger_rerun()
    else:
        reload_map()


def _undo_delete_entity(undone: DeleteSlopeAction | DeleteLiftAction | DeleteRoadAction) -> None:
    """Handle undo of any DELETE_* action: the graph already restored the entity; just redraw."""
    logger.info(f"Restored deleted entity ({type(undone).__name__})")
    reload_map()


def _undo_finish_road(undone: FinishRoadAction) -> None:
    """Handle undo of FINISH_ROAD (mirrors _undo_finish_slope)."""
    _undo_finish(
        kind=SegmentKind.ROAD,
        segment_ids=undone.segment_ids,
        name=undone.road_name,
        start_node_id=undone.start_node_id,
    )


def _undo_import_osm(undone: ImportOSMAction) -> None:
    """Handle undo of IMPORT_OSM: the batch's slopes/lifts are already gone from the graph.

    If the user was viewing one of the removed entities, exit to idle; otherwise just redraw.
    """
    sm: PlannerStateMachine = st.session_state.state_machine

    logger.info(f"Undone OSM import: {len(undone.slope_ids)} slopes, {len(undone.lift_ids)} lifts")

    viewing = sm.viewing_entity
    if viewing is not None and undone.removed_entity(entity_id=viewing[1]):
        sm.force_idle()
        trigger_rerun()
    else:
        reload_map()


def _undo_merge_nodes(undone: MergeNodesAction) -> None:
    """Handle undo of MERGE_NODES: the graph already restored the nodes/endpoints — just redraw."""
    logger.info(f"Undone node merge into {undone.survivor_id}")
    reload_map()


# UI side-effect per action type, keyed by ActionType.name.
# The graph mutation itself is done by ResortGraph.undo_last; these handlers only update UI state
# (rebuild proposals, force SM transitions, redraw). Import-time assert keeps this exhaustive.
_UNDO_SIDE_EFFECTS: dict[str, "Callable[[UndoAction], None]"] = {
    ActionType.ADD_SEGMENTS.name: lambda a: _undo_add_segments(undone=cast(AddSegmentsAction, a)),
    ActionType.FINISH_SLOPE.name: lambda a: _undo_finish_slope(undone=cast(FinishSlopeAction, a)),
    ActionType.ADD_LIFT.name: lambda a: _undo_add_lift(undone=cast(AddLiftAction, a)),
    ActionType.FINISH_ROAD.name: lambda a: _undo_finish_road(undone=cast(FinishRoadAction, a)),
    ActionType.DELETE_SLOPE.name: lambda a: _undo_delete_entity(undone=cast(DeleteSlopeAction, a)),
    ActionType.DELETE_LIFT.name: lambda a: _undo_delete_entity(undone=cast(DeleteLiftAction, a)),
    ActionType.DELETE_ROAD.name: lambda a: _undo_delete_entity(undone=cast(DeleteRoadAction, a)),
    ActionType.IMPORT_OSM.name: lambda a: _undo_import_osm(undone=cast(ImportOSMAction, a)),
    ActionType.MERGE_NODES.name: lambda a: _undo_merge_nodes(undone=cast(MergeNodesAction, a)),
}
_action_names = {t.name for t in ActionType}
assert set(_UNDO_SIDE_EFFECTS) == _action_names, (
    f"_UNDO_SIDE_EFFECTS keys must match ActionType members exactly. "
    f"Missing: {_action_names - set(_UNDO_SIDE_EFFECTS)}; stray: {set(_UNDO_SIDE_EFFECTS) - _action_names}"
)


def undo_last_action() -> None:
    """Undo the most recent action.

    Dispatches to type-specific handlers based on the undone action type.
    Each handler is responsible for:
    - Restoring context state (if needed)
    - Regenerating paths (if needed)
    - Triggering state machine transition or reload

    Note: Undo confirmation is handled by UI dialog before calling this function.
    """
    sm: PlannerStateMachine = st.session_state.state_machine
    ctx: PlannerContext = st.session_state.context
    graph: ResortGraph = st.session_state.graph

    # Guard: nothing to undo (should not happen - UI disables button when empty)
    if not graph.undo_stack:
        return

    logger.debug(f"[ACTION] Undo requested, state={sm.get_state_name()}, undo_stack_size={len(graph.undo_stack)}")

    # Special case: in a build state with no committed segments → cancel that
    # build instead of popping an unrelated undo entry (slopes cancel_slope, roads cancel_road).
    if (sm.is_any_slope_state or sm.is_any_road_state) and not ctx.build(sm.active_build_kind).segments:
        kind = sm.active_build_kind
        logger.info(f"[ACTION] No segments in {kind.value} building state, canceling via undo")
        sm.send(KIND_SPECS[kind].cancel_event)
        return

    undone = graph.undo_last()
    logger.debug(f"[ACTION] Undone: {undone.action_type.name}")

    # Dispatch UI side-effects via the registry keyed by ActionType.name. The bypass (force_*) is
    # legal only inside this scope — undo_running() is what permits it; outside it, force_* raises.
    with sm.undo_running():
        _UNDO_SIDE_EFFECTS[undone.action_type.name](undone)


# =============================================================================
# CUSTOM CONNECT MODE
# =============================================================================


def cancel_custom_path() -> None:
    """Leave custom targeting (from SLOPE_CUSTOM_PATH), back to fan-out proposals.

    Triggers cancel_path_to_starting or cancel_path_to_building.
    Path regeneration is triggered by before_* hooks.
    """
    sm: PlannerStateMachine = st.session_state.state_machine
    logger.debug("[ACTION] Cancel Connection - triggering state transition")
    sm.cancel_custom_connect()


# =============================================================================
# DELETE OPERATIONS
# =============================================================================


def select_lift_type_action(lift_type: str) -> None:
    """Sidebar lift-type button: set the build mode, or re-type the viewed lift.

    When viewing a lift, the four lift buttons change THAT lift's type (Lift.update_type recomputes
    the pylons/catenary); otherwise they just set the build mode for the next lift. Either way the
    global build_mode + ctx.lift.type track the chosen type so a new lift uses it.
    The caller (BuilderOperation.on_select) owns the reload, so this does not reload.
    """
    ctx: PlannerContext = st.session_state.context
    sm: PlannerStateMachine = st.session_state.state_machine
    graph: ResortGraph = st.session_state.graph

    ctx.build_mode.mode = lift_type
    ctx.lift.type = lift_type

    if sm.is_idle_viewing_lift and ctx.viewing.lift_id:
        lift = graph.lifts.get(ctx.viewing.lift_id)
        if lift is not None and lift.lift_type != lift_type:
            start_node = graph.nodes.get(lift.start_node_id)
            end_node = graph.nodes.get(lift.end_node_id)
            assert start_node and end_node, f"lift {lift.id} references missing nodes (data integrity bug)"
            lift.update_type(new_type=lift_type, start_node=start_node, end_node=end_node)
            logger.info(f"UI: Changed viewed lift {lift.id} type to {lift_type}")


def _close_panel_and_refresh(*, deleted: bool, is_viewing_deleted: bool) -> bool:
    """Shared delete tail: close the panel if the deleted entity was being viewed, refresh.

    `deleted` is the graph.delete_* result; returns it unchanged.
    """
    if not deleted:
        return False
    sm: PlannerStateMachine = st.session_state.state_machine
    if is_viewing_deleted:
        sm.send("close_panel")
    bump_dedup_epoch()
    return True


def delete_slope_action(slope_id: str) -> bool:
    """Delete a slope and trigger UI updates."""
    sm: PlannerStateMachine = st.session_state.state_machine
    ctx: PlannerContext = st.session_state.context
    graph: ResortGraph = st.session_state.graph
    deleted = graph.delete_slope(slope_id=slope_id)
    return _close_panel_and_refresh(
        deleted=deleted, is_viewing_deleted=sm.is_idle_viewing_slope and ctx.viewing.slope_id == slope_id
    )


def rename_entity_action(entity_id: str, new_name: str) -> None:
    """Set a custom name on the viewed slope/lift/road; ignore an empty name.

    Kind-agnostic (ids are uniquely prefixed), so one action covers all three. The panel header
    re-renders from the entity's name; bump the map so labels redraw.
    """
    name = new_name.strip()
    if not name:
        return
    graph: ResortGraph = st.session_state.graph
    graph.rename(entity_id=entity_id, new_name=name)
    bump_dedup_epoch()


def delete_lift_action(lift_id: str) -> bool:
    """Delete a lift and trigger UI updates."""
    sm: PlannerStateMachine = st.session_state.state_machine
    ctx: PlannerContext = st.session_state.context
    graph: ResortGraph = st.session_state.graph
    deleted = graph.delete_lift(lift_id=lift_id)
    return _close_panel_and_refresh(
        deleted=deleted, is_viewing_deleted=sm.is_idle_viewing_lift and ctx.viewing.lift_id == lift_id
    )


def delete_road_action(road_id: str) -> bool:
    """Delete a road and trigger UI updates."""
    sm: PlannerStateMachine = st.session_state.state_machine
    ctx: PlannerContext = st.session_state.context
    graph: ResortGraph = st.session_state.graph
    deleted = graph.delete_road(road_id=road_id)
    return _close_panel_and_refresh(
        deleted=deleted, is_viewing_deleted=sm.is_idle_viewing_road and ctx.viewing.road_id == road_id
    )

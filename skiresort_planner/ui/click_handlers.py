"""Click handlers for ski resort planner.

Uses ClickDetector to detect clicks, then dispatches to state-specific handlers.
Each handler processes ClickInfo objects from the unified click detection system.

Design Principles:
- ClickDetector handles ALL detection, logging, and UI display
- One handler per state (no if-else chains)
- Handlers raise exceptions for invalid states (fail-fast)
- STRICT: Unknown/unhandled clicks raise RuntimeError immediately
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import streamlit as st

from skiresort_planner.constants import MapConfig
from skiresort_planner.model.click_info import ClickInfo, MapClickType, MarkerType
from skiresort_planner.model.message import InvalidClickMessage, OutsideTerrainMessage
from skiresort_planner.model.node import Node
from skiresort_planner.model.path_point import PathPoint
from skiresort_planner.model.path_segment import SegmentKind
from skiresort_planner.ui.actions import (
    add_node_on_path_action,
    center_on_lift,
    center_on_road,
    center_on_segment_path,
    center_on_slope,
    commit_selected_path,
    resolve_build_origin,
)
from skiresort_planner.ui.infra import bump_camera_epoch, bump_dedup_epoch, trigger_rerun
from skiresort_planner.ui.kind_spec import KIND_SPECS
from skiresort_planner.ui.validators import (
    validate_custom_target_distance,
    validate_custom_target_downhill,
    validate_lift_different_nodes,
    validate_lift_goes_uphill,
)

if TYPE_CHECKING:
    from skiresort_planner.core.dem_service import DEMService
    from skiresort_planner.model.resort_graph import ResortGraph
    from skiresort_planner.ui.context import PlannerContext
    from skiresort_planner.ui.state_machine import PlannerStateMachine

logger = logging.getLogger(__name__)


# =============================================================================
# STATE-SPECIFIC HANDLERS
# =============================================================================


def _select_or_commit_proposal(ctx: PlannerContext, idx: int) -> None:
    """Body-click behavior shared by road + slope proposals: select, or commit if
    already selected.

    Rule: clicking a proposal you have NOT selected only highlights it; clicking
    the one that is ALREADY selected commits it.
    """
    if not (0 <= idx < len(ctx.proposals.paths)):
        return
    if ctx.proposals.selected_idx == idx:
        commit_selected_path(path_idx=idx)  # re-click the selected one → commit
    else:
        ctx.proposals.selected_idx = idx
        trigger_rerun()  # first click just highlights the proposal — redraw in place, no recenter


def _commit_proposal_endpoint(ctx: PlannerContext, idx: int) -> None:
    """Endpoint-click behavior shared by road + slope: commit immediately (single click).

    The orange endpoint marker is the "go" affordance — clicking it commits that path
    outright, no prior selection needed. Out-of-range indices (stale click state after a
    rerun) are a silent no-op, mirroring the body path's guard.
    """
    if not (0 <= idx < len(ctx.proposals.paths)):
        return
    commit_selected_path(path_idx=idx)


def _start_mode_from_terrain(
    ctx: PlannerContext, sm: PlannerStateMachine, lon: float, lat: float, elevation: float
) -> None:
    """Start the selected build mode from a fresh TERRAIN point (idle first click).

    Does NOT recenter the map — the user keeps their current pan/zoom (only finish / view-panel /
    Reset View / 3D / place-search reframe the camera). Merge is the exception in intent only: it
    starts from a node, so a terrain click is guided, not acted on.
    """
    build_mode = ctx.build_mode.mode
    if ctx.build_mode.is_slope():
        logger.debug(f"[IDLE] Terrain click: starting new slope at ({lat:.6f}, {lon:.6f})")
        # Selection + fan arming happen in the before-hook / enter_slope_starting (Single Point
        # of Truth), like roads — not here.
        sm.start_slope(lon=lon, lat=lat, elevation=elevation, node_id=None)
    elif ctx.build_mode.is_lift():
        logger.debug(f"[IDLE] Terrain click: starting {build_mode} at ({lat:.6f}, {lon:.6f})")
        sm.start_lift(node_id=None, location=PathPoint(lon=lon, lat=lat, elevation=elevation))
    elif ctx.build_mode.is_road():
        logger.debug(f"[IDLE] Terrain click: starting road at ({lat:.6f}, {lon:.6f})")
        # Selection is set in before_start_road (Single Point of Truth), mirroring slopes.
        sm.start_road(node_id=None, location=PathPoint(lon=lon, lat=lat, elevation=elevation))
    elif ctx.build_mode.is_import():
        logger.debug(f"[IDLE] Terrain click: placing import box center at ({lat:.6f}, {lon:.6f})")
        sm.start_import(lon=lon, lat=lat)
    elif ctx.build_mode.is_merge():
        # Merge/delete act on nodes; add-node acts on a path. A bare terrain click hits neither.
        InvalidClickMessage(action="edit nodes", reason="Click a node to select it, or a path to add a node.").display()
    else:
        raise RuntimeError(f"[IDLE] Unknown build_mode '{build_mode}'.")


def _start_mode_from_node(ctx: PlannerContext, sm: PlannerStateMachine, node: Node) -> None:
    """Start the selected build mode from an existing NODE (idle first click).

    Mirrors _start_mode_from_terrain, reusing the node's identity so the flow snaps to the junction.
    Does NOT recenter the map — the user keeps their current view.
    """
    build_mode = ctx.build_mode.mode
    if ctx.build_mode.is_slope():
        logger.debug(f"[IDLE] Node click: starting slope from {node.id}")
        # Selection + fan arming happen in the before-hook / enter_slope_starting (Single Point of
        # Truth), like roads — not here.
        sm.start_slope(lon=node.lon, lat=node.lat, elevation=node.elevation, node_id=node.id)
    elif ctx.build_mode.is_lift():
        logger.debug(f"[IDLE] Node click: starting {build_mode} from {node.id}")
        sm.start_lift(node_id=node.id)
    elif ctx.build_mode.is_road():
        logger.debug(f"[IDLE] Node click: starting road from {node.id}")
        # Selection is set in before_start_road (Single Point of Truth), mirroring slopes.
        sm.start_road(node_id=node.id)
    elif ctx.build_mode.is_import():
        logger.debug(f"[IDLE] Node click: placing import box center at {node.id}")
        sm.start_import(lon=node.lon, lat=node.lat)
    elif ctx.build_mode.is_merge():
        # First node click starts merge. Select the node, then transition (start_merge reruns via
        # listener, so select BEFORE it). No recenter — keep the user's view.
        logger.debug(f"[IDLE] Node click: starting merge from {node.id}")
        ctx.merge.toggle(node.id)
        sm.start_merge()
    else:
        raise RuntimeError(f"[IDLE] Unknown build_mode '{build_mode}'.")


def handle_idle_click(click_info: ClickInfo, elevation: float | None) -> None:
    """Handle click in IDLE state - start building/placing or show info panel.

    Build behavior depends on ctx.build_mode.mode:
        - SLOPE: Start building ski slope
        - CHAIRLIFT/GONDOLA/etc: Start placing lift
        - None: Only view panels, no building

    Valid Click Types:
        NODE → Start building from junction (uses build_mode)
        TERRAIN → Start building at new point (uses build_mode)
        SLOPE → Show slope info panel
        SEGMENT → Show parent slope info panel
        LIFT → Show lift info panel
        PYLON → Show parent lift info panel

    Invalid Click Types:
        PROPOSAL_* → Programming error (no proposals in idle)
    """
    sm: PlannerStateMachine = st.session_state.state_machine
    ctx: PlannerContext = st.session_state.context
    graph: ResortGraph = st.session_state.graph

    # TERRAIN click → start building based on mode
    if click_info.click_type == MapClickType.TERRAIN:
        # ClickInfo validates lat/lon are set for terrain clicks
        assert click_info.lat is not None and click_info.lon is not None
        assert elevation is not None  # We got terrain elevation above
        _start_mode_from_terrain(ctx=ctx, sm=sm, lon=click_info.lon, lat=click_info.lat, elevation=elevation)
        return

    # MARKER clicks
    if click_info.click_type == MapClickType.MARKER:
        marker_type = click_info.marker_type

        # NODE → Start building from junction (uses build_mode)
        if marker_type == MarkerType.NODE:
            assert click_info.node_id is not None  # Validated in ClickInfo
            node = graph.nodes.get(click_info.node_id)
            if not node:
                raise RuntimeError(f"Node {click_info.node_id} not found in graph")
            _start_mode_from_node(ctx=ctx, sm=sm, node=node)
            return

        # SLOPE → Show slope panel (always works regardless of build_mode)
        if marker_type == MarkerType.SLOPE:
            assert click_info.slope_id is not None  # Validated in ClickInfo
            slope = graph.slopes.get(click_info.slope_id)
            if not slope:
                raise RuntimeError(f"Slope {click_info.slope_id} not found in graph")
            logger.debug(f"[IDLE] Slope click: showing panel for {slope.name}")
            center_on_slope(ctx=ctx, graph=graph, slope=slope, zoom=MapConfig.VIEWING_ZOOM)
            sm.view_slope(slope_id=slope.id)  # Triggers st.rerun() via listener
            return

        # SEGMENT → in merge mode a belt click enters merge and adds a node; otherwise it opens the
        # parent's panel. A SEGMENT reaches the panel branch only in the one-frame race before the map
        # re-tags it as its finished entity; an orphan (parent deleted) resolves to None and is ignored.
        if marker_type == MarkerType.SEGMENT:
            assert click_info.segment_id is not None  # Validated in ClickInfo
            if ctx.build_mode.is_merge():
                # A SEGMENT is a positioned marker, so ClickDetector always sets lat/lon (see ClickInfo).
                assert click_info.lon is not None and click_info.lat is not None
                logger.debug(f"[IDLE] Merge mode: adding a node on segment {click_info.segment_id}")
                if add_node_on_path_action(segment_id=click_info.segment_id, lon=click_info.lon, lat=click_info.lat):
                    sm.start_merge()  # enter merge so the user can keep editing (mirrors node-click entry)
                return
            parent = graph.get_entity_by_segment_id(segment_id=click_info.segment_id)
            if not parent:
                logger.debug(f"[IDLE] Segment {click_info.segment_id} click: orphan segment, ignoring")
                return
            logger.debug(f"[IDLE] Segment click: showing panel for {parent.name}")
            # parent is a SegmentPath; branch on its reload-safe .kind (never isinstance).
            center_on_segment_path(ctx=ctx, graph=graph, path=parent, zoom=MapConfig.VIEWING_ZOOM)
            if parent.kind == SegmentKind.SLOPE:
                sm.view_slope(slope_id=parent.id)
            elif parent.kind == SegmentKind.ROAD:
                sm.view_road(road_id=parent.id)
            else:
                raise RuntimeError(f"[IDLE] Segment click: unhandled parent kind {parent.kind}.")
            return

        # LIFT → Show lift panel and sync build mode
        if marker_type == MarkerType.LIFT:
            assert click_info.lift_id is not None  # Validated in ClickInfo
            lift = graph.lifts.get(click_info.lift_id)
            if not lift:
                raise RuntimeError(f"Lift {click_info.lift_id} not found in graph")
            logger.debug(f"[IDLE] Lift click: showing panel for {lift.name}")
            # Sync build mode to the viewed lift's type (single source of truth for selection)
            ctx.build_mode.mode = lift.lift_type
            center_on_lift(ctx=ctx, graph=graph, lift=lift, zoom=MapConfig.VIEWING_ZOOM)
            sm.view_lift(lift_id=lift.id)  # Triggers st.rerun() via listener
            return

        # PYLON → Show parent lift panel and sync build mode
        if marker_type == MarkerType.PYLON:
            assert click_info.lift_id is not None  # Validated in ClickInfo
            lift = graph.lifts.get(click_info.lift_id)
            if not lift:
                raise RuntimeError(f"Lift {click_info.lift_id} not found in graph")
            logger.debug(f"[IDLE] Pylon click: showing panel for {lift.name}")
            # Sync build mode to the viewed lift's type (single source of truth for selection)
            ctx.build_mode.mode = lift.lift_type
            center_on_lift(ctx=ctx, graph=graph, lift=lift, zoom=MapConfig.VIEWING_ZOOM)
            sm.view_lift(lift_id=lift.id)  # Triggers st.rerun() via listener
            return

        # ROAD → Show road panel
        if marker_type == MarkerType.ROAD:
            assert click_info.road_id is not None  # Validated in ClickInfo
            road = graph.roads.get(click_info.road_id)
            if not road:
                raise RuntimeError(f"Road {click_info.road_id} not found in graph")
            logger.debug(f"[IDLE] Road click: showing panel for {road.name}")
            center_on_road(ctx=ctx, graph=graph, road=road, zoom=MapConfig.VIEWING_ZOOM)
            sm.view_road(road_id=road.id)  # Triggers st.rerun() via listener
            return

        # PROPOSAL clicks in idle = programming error
        if marker_type in {MarkerType.PROPOSAL_ENDPOINT, MarkerType.PROPOSAL_BODY}:
            raise RuntimeError(
                "[IDLE] Proposal click detected but no proposals exist in idle state. "
                "This indicates a bug - proposal markers should not be on the map."
            )

        # marker_type must be set for MARKER clicks (validated in ClickInfo)
        assert marker_type is not None
        raise RuntimeError(f"[IDLE] Unhandled marker type {marker_type.value}. Add explicit handling.")

    raise RuntimeError(f"[IDLE] Unknown click_type {click_info.click_type}. Expected MARKER or TERRAIN.")


def handle_path_building_click(click_info: ClickInfo, elevation: float | None) -> None:
    """Handle a click while building ANY path kind (slope or road).

    One handler for slope_starting/building/custom_path AND road_starting/building/custom_path,
    so slopes and roads behave identically by construction. The active kind is used only to
    word the invalid-click messages ("current slope" vs "current road").

    Valid clicks:
        PROPOSAL_ENDPOINT → commit the path immediately (one click)
        PROPOSAL_BODY     → select the variant; commit only on re-clicking the selected one
        TERRAIN / NODE    → route a custom-connect path to that target. An invalid target
                            (uphill for slopes / too far) warns and does NOT change state.

    Invalid clicks (while building):
        SLOPE / SEGMENT / LIFT / PYLON / ROAD → cannot view while building
    """
    sm: PlannerStateMachine = st.session_state.state_machine
    ctx: PlannerContext = st.session_state.context
    noun = sm.active_build_kind.value  # "slope" / "road"

    # TERRAIN click → route a custom-connect path to the clicked point.
    if click_info.click_type == MapClickType.TERRAIN:
        _handle_custom_connect_click(click_info=click_info, elevation=elevation)
        return

    # MARKER clicks
    if click_info.click_type == MapClickType.MARKER:
        marker_type = click_info.marker_type

        # Orange ENDPOINT marker → commit that path immediately (single click).
        if marker_type == MarkerType.PROPOSAL_ENDPOINT:
            assert click_info.proposal_number is not None  # Validated in ClickInfo
            _commit_proposal_endpoint(ctx=ctx, idx=click_info.proposal_number - 1)
            return

        # In-between BODY marker → select the variant; commit only on re-clicking the selected one.
        if marker_type == MarkerType.PROPOSAL_BODY:
            assert click_info.proposal_number is not None  # Validated in ClickInfo
            _select_or_commit_proposal(ctx=ctx, idx=click_info.proposal_number - 1)
            return

        # NODE → route a custom-connect path to that node (snap + connect).
        if marker_type == MarkerType.NODE:
            _handle_custom_connect_click(click_info=click_info, elevation=elevation)
            return

        # SLOPE during building = user error
        if marker_type == MarkerType.SLOPE:
            InvalidClickMessage(
                action="view slope",
                reason=f"Finish or cancel the current {noun} first.",
            ).display()
            return

        # SEGMENT during building = user error (same as SLOPE)
        if marker_type == MarkerType.SEGMENT:
            InvalidClickMessage(
                action="view segment",
                reason=f"Finish or cancel the current {noun} first.",
            ).display()
            return

        # LIFT/PYLON during building = user error
        if marker_type in {MarkerType.LIFT, MarkerType.PYLON}:
            InvalidClickMessage(
                action="view lift",
                reason=f"Finish or cancel the current {noun} first.",
            ).display()
            return

        # ROAD during building = user error
        if marker_type == MarkerType.ROAD:
            InvalidClickMessage(
                action="view road",
                reason=f"Finish or cancel the current {noun} first.",
            ).display()
            return

        # STRICT: Unknown marker type (marker_type must be set for MARKER clicks)
        assert marker_type is not None
        raise RuntimeError(f"[BUILDING] Unhandled marker type {marker_type.value}. Add explicit handling.")

    # STRICT: Unknown click type
    raise RuntimeError(f"[BUILDING] Unknown click_type {click_info.click_type}. Expected MARKER or TERRAIN.")


def _handle_custom_connect_click(click_info: ClickInfo, elevation: float | None) -> None:
    """Route a custom-connect path to a clicked terrain point or node (any build kind).

    Validates range for every kind, and downhill only for kinds that may not climb
    (slopes), BEFORE firing the transition — so an invalid target shows a warning and
    leaves the current state (and its fan proposals) untouched. On success fires
    select_custom_target; the state machine resolves it per the active state.
    """
    sm: PlannerStateMachine = st.session_state.state_machine
    ctx: PlannerContext = st.session_state.context
    graph: ResortGraph = st.session_state.graph

    kind = sm.active_build_kind
    build = ctx.build(kind)

    # Get target coordinates - from terrain click or from node lookup
    target_lon: float
    target_lat: float
    target_elevation: float | None
    target_node_id: str | None = None  # set only when the target IS an existing node

    if click_info.click_type == MapClickType.TERRAIN:
        assert click_info.lon is not None and click_info.lat is not None  # Validated in ClickInfo
        target_lon, target_lat = click_info.lon, click_info.lat
        target_elevation = elevation
        logger.debug(f"Custom connect terrain click at ({target_lat:.6f}, {target_lon:.6f})")
    elif click_info.marker_type == MarkerType.NODE:
        # click_type must be MARKER if not TERRAIN
        assert click_info.node_id is not None  # Validated in ClickInfo
        node = graph.nodes.get(click_info.node_id)
        if not node:
            raise RuntimeError(f"Node {click_info.node_id} not found in graph")
        target_lon, target_lat = node.lon, node.lat
        target_elevation = node.elevation
        target_node_id = node.id  # reuse this exact node on commit (identity, not proximity)
        logger.debug(f"Custom connect snapped to existing node {node.id}")
    else:
        # Other marker types during custom connect = user error
        InvalidClickMessage(
            action="click marker",
            reason="Click on terrain or a node to select custom connect target.",
        ).display()
        return

    if target_elevation is None:
        OutsideTerrainMessage(lat=target_lat, lon=target_lon).display()
        return

    # Resolve the origin coords for validation via the shared resolver (endpoint → re-target origin
    # → starting node → pending terrain location). No node is minted before commit, so a fresh
    # terrain origin is just a location; any node id it returns is guaranteed live.
    start_lon, start_lat, start_elevation, _ = resolve_build_origin(
        build=build, graph=graph, custom_start_node=ctx.custom_connect.start_node
    )

    # Validate range for every kind; downhill only for kinds that may not climb (the validator
    # itself skips the check when may_climb, so there is no per-kind branch here).
    if error := validate_custom_target_downhill(
        start_elevation=start_elevation,
        target_elevation=target_elevation,
        may_climb=KIND_SPECS[kind].may_climb,
    ):
        error.display()
        return

    if error := validate_custom_target_distance(
        start_lat=start_lat, start_lon=start_lon, target_lat=target_lat, target_lon=target_lon
    ):
        logger.warning(
            f"Custom connect distance validation failed from ({start_lat:.6f}, {start_lon:.6f}) "
            f"to ({target_lat:.6f}, {target_lon:.6f}): {error.message}"
        )
        error.display()
        return

    logger.debug(
        f"Custom connect from ({start_lat:.6f}, {start_lon:.6f}, {start_elevation:.0f}m) "
        f"to ({target_lat:.6f}, {target_lon:.6f}, {target_elevation:.0f}m)"
    )

    # Trigger state transition - the target before-hook sets context (start_node,
    # target_location, force_mode); enter_*_custom_path regenerates proposals.
    sm.send(
        "select_custom_target",
        target_location=(target_lon, target_lat, target_elevation),
        target_node=target_node_id,
    )


def handle_lift_placing_click(click_info: ClickInfo, elevation: float | None) -> None:
    """Handle click in LIFT_PLACING state - complete lift placement.

    Pattern: Validate with elevations BEFORE creating nodes.
    - For terrain clicks: use elevation directly, create node only after validation passes
    - For node clicks: use existing node
    This prevents orphan nodes from failed validation attempts.

    Valid Click Types:
        NODE → Complete lift to existing node
        TERRAIN → Create new node and complete lift

    Invalid Click Types (during placement):
        SLOPE → Cannot view while placing
        LIFT → Cannot view while placing
        PYLON → Cannot view while placing
        PROPOSAL_* → No proposals in lift mode
    """
    sm: PlannerStateMachine = st.session_state.state_machine
    ctx: PlannerContext = st.session_state.context
    graph: ResortGraph = st.session_state.graph
    dem: DEMService = st.session_state.dem_service

    # Check for invalid marker clicks first
    if click_info.click_type == MapClickType.MARKER:
        marker_type = click_info.marker_type

        # SLOPE during placement = user error
        if marker_type == MarkerType.SLOPE:
            InvalidClickMessage(
                action="view slope",
                reason="Finish placing the lift first (click uphill for top station).",
            ).display()
            return

        # SEGMENT during placement = user error (same as SLOPE)
        if marker_type == MarkerType.SEGMENT:
            InvalidClickMessage(
                action="view segment",
                reason="Finish placing the lift first (click uphill for top station).",
            ).display()
            return

        # LIFT/PYLON during placement = user error
        if marker_type in {MarkerType.LIFT, MarkerType.PYLON}:
            InvalidClickMessage(
                action="view lift",
                reason="Finish placing the lift first (click uphill for top station).",
            ).display()
            return

        # ROAD during placement = user error
        if marker_type == MarkerType.ROAD:
            InvalidClickMessage(
                action="view road",
                reason="Finish placing the lift first (click uphill for top station).",
            ).display()
            return

        # PROPOSAL during lift placement = programming error (no proposals exist)
        if marker_type in {MarkerType.PROPOSAL_ENDPOINT, MarkerType.PROPOSAL_BODY}:
            raise RuntimeError(
                "[LIFT_PLACING] Proposal click detected but no proposals exist in lift mode. "
                "This indicates a bug - proposal markers should not be on the map."
            )

    # ─────────────────────────────────────────────────────────────────────────
    # Determine START: existing node or pending location
    # ─────────────────────────────────────────────────────────────────────────
    if ctx.lift.start_node_id is not None:
        start_node = graph.nodes.get(ctx.lift.start_node_id)
        if start_node is None:
            raise RuntimeError(f"Start node {ctx.lift.start_node_id} must exist but was not found")
        start_elevation = start_node.elevation
    elif ctx.lift.start_location is not None:
        start_elevation = ctx.lift.start_location.elevation
        start_node = None  # Will create after validation
    else:
        raise RuntimeError("Neither start_node_id nor start_location is set in lift context")

    # ─────────────────────────────────────────────────────────────────────────
    # Determine END: existing node or terrain click
    # ─────────────────────────────────────────────────────────────────────────
    if click_info.click_type == MapClickType.MARKER and click_info.marker_type == MarkerType.NODE:
        assert click_info.node_id is not None
        end_node_existing = graph.nodes.get(click_info.node_id)
        if end_node_existing is None:
            raise RuntimeError(f"End node {click_info.node_id} must exist but was not found")
        end_node_id = end_node_existing.id
        end_elevation = end_node_existing.elevation
        end_lon = end_node_existing.lon
        end_lat = end_node_existing.lat
        logger.debug(f"[LIFT_PLACING] Node click: completing lift to {end_node_id}")
    elif click_info.click_type == MapClickType.TERRAIN:
        assert click_info.lat is not None and click_info.lon is not None
        if elevation is None:
            OutsideTerrainMessage(lat=click_info.lat, lon=click_info.lon).display()
            return
        end_node_id = None  # No existing node for terrain clicks
        end_elevation = elevation
        end_lon = click_info.lon
        end_lat = click_info.lat
    else:
        raise RuntimeError(f"Expected NODE or TERRAIN click but got {click_info.click_type}")

    # ─────────────────────────────────────────────────────────────────────────
    # VALIDATION: Using elevations only - no nodes created yet for terrain clicks
    # ─────────────────────────────────────────────────────────────────────────
    if error := validate_lift_goes_uphill(start_elevation=start_elevation, end_elevation=end_elevation):
        logger.warning(
            f"Lift uphill validation failed: start={start_elevation:.0f}m, end={end_elevation:.0f}m: {error.message}"
        )
        error.display()
        return  # No orphan nodes - nothing was created

    # Same-node check only applies if both are existing nodes
    if (
        ctx.lift.start_node_id is not None
        and end_node_id is not None
        and (error := validate_lift_different_nodes(start_node_id=ctx.lift.start_node_id, end_node_id=end_node_id))
    ):
        logger.info(
            f"Lift same-node validation failed: start_node={ctx.lift.start_node_id}, "
            f"end_node={end_node_id}: {error.message}"
        )
        error.display()
        return

    # ─────────────────────────────────────────────────────────────────────────
    # NODE CREATION: Validation passed - now create nodes if needed
    # ─────────────────────────────────────────────────────────────────────────
    if start_node is None:
        assert ctx.lift.start_location is not None
        start_node, _ = graph.get_or_create_node(
            lon=ctx.lift.start_location.lon,
            lat=ctx.lift.start_location.lat,
            elevation=ctx.lift.start_location.elevation,
        )
        ctx.lift.start_node_id = start_node.id
        ctx.lift.start_location = None
        logger.info(f"Created start node {start_node.id}")

    if end_node_id is not None:
        end_node = graph.nodes[end_node_id]
    else:
        end_node, _ = graph.get_or_create_node(lon=end_lon, lat=end_lat, elevation=end_elevation)
        logger.info(f"Created end node {end_node.id}")

    # ─────────────────────────────────────────────────────────────────────────
    # Create lift
    # ─────────────────────────────────────────────────────────────────────────
    logger.info(
        f"Creating lift from {start_node.id} ({start_node.elevation:.0f}m) to {end_node.id} ({end_node.elevation:.0f}m)"
    )

    lift = graph.add_lift(
        start_node_id=start_node.id,
        end_node_id=end_node.id,
        lift_type=ctx.build_mode.mode,
        dem=dem,
    )

    logger.info(f"Lift {lift.name} created successfully")
    # Frame the new lift + remount, but do NOT rerun here — sm.complete_lift's listener reruns.
    center_on_lift(ctx=ctx, graph=graph, lift=lift, zoom=MapConfig.VIEWING_ZOOM)
    bump_camera_epoch()
    sm.complete_lift(lift_id=lift.id)


def handle_import_placing_click(click_info: ClickInfo, elevation: float | None) -> None:
    """Handle a click while placing an OSM import box (IMPORT_PLACING).

    Clicking terrain re-places the box center. Confirming the import is done ONLY by the right-panel
    buttons ("Import lifts + slopes" / "Import lifts only"). The center-dot marker is inert here.

    Valid clicks:
        TERRAIN → move the box center and redraw
    Anything else (including the center dot) is ignored — confirm via the buttons.
    """
    sm: PlannerStateMachine = st.session_state.state_machine
    ctx: PlannerContext = st.session_state.context

    # Terrain click → re-place the box center (keep placing, redraw the box).
    if click_info.click_type == MapClickType.TERRAIN:
        assert click_info.lat is not None and click_info.lon is not None
        logger.debug(f"[IMPORT] Terrain click: re-placing box center at ({click_info.lat:.6f}, {click_info.lon:.6f})")
        ctx.deferred.osm_import_center_lon = click_info.lon
        ctx.deferred.osm_import_center_lat = click_info.lat
        sm.retarget_import()
        trigger_rerun()  # redraw the box at the clicked point (already on-screen — no recenter)
        return

    logger.debug(f"[IMPORT] Ignoring {click_info.display_name} — click terrain to re-place, or a button to import")


def handle_merge_placing_click(click_info: ClickInfo, elevation: float | None) -> None:
    """Handle a click while selecting nodes to merge (MERGE_PLACING).

    A NODE marker click toggles that node in the selection (re-click removes it) via the
    toggle_merge_node self-loop, then redraws so the selection colour updates. A SEGMENT marker (a
    slope/road belt or center-line) inserts a new node on that path at the click point. Every other
    click (terrain, slope/road icons, lift/proposal markers) is an InvalidClickMessage — this branch
    handles every marker type so the dispatch never crashes.
    """
    sm: PlannerStateMachine = st.session_state.state_machine

    # NODE marker → toggle it in the selection.
    if click_info.click_type == MapClickType.MARKER and click_info.marker_type == MarkerType.NODE:
        assert click_info.node_id is not None  # Validated in ClickInfo
        logger.debug(f"[MERGE] Node click: toggling {click_info.node_id} in the merge selection")
        sm.toggle_merge_node(node_id=click_info.node_id)
        # Refresh dedup so the SAME node can be toggled again (on/off/on);
        # redraw the red selection in place — no recenter.
        bump_dedup_epoch()
        trigger_rerun()
        return

    # SEGMENT marker (a path belt/center-line) → add a node on that path at the clicked point.
    # The belt carries both the segment id and the click coordinate (it is the only positioned marker).
    if click_info.click_type == MapClickType.MARKER and click_info.marker_type == MarkerType.SEGMENT:
        assert click_info.segment_id is not None  # Validated in ClickInfo
        # A SEGMENT is a positioned marker, so ClickDetector always sets lat/lon (see ClickInfo).
        assert click_info.lon is not None and click_info.lat is not None
        logger.debug(f"[MERGE] Path click: adding a node on segment {click_info.segment_id}")
        if add_node_on_path_action(segment_id=click_info.segment_id, lon=click_info.lon, lat=click_info.lat):
            trigger_rerun()  # stay in merge_placing; redraw with the new node in place (no recenter)
        return

    # Anything else (terrain or a non-node marker) is not selectable for merge.
    InvalidClickMessage(
        action="select for merge",
        reason="Click a node to select it, or a path to add a node.",
    ).display()

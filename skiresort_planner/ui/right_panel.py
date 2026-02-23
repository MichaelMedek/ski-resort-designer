"""Right panel components for ski resort planner.

Centralizes all right-side control panel rendering:
- State dispatch to appropriate renderers
- PathSelectionPanel: Path browsing, selection, and commit
- SlopeStatsPanel: Slope statistics in viewing mode
- LiftStatsPanel: Lift statistics in viewing mode

Design Principles:
- One renderer per state (no if-else chains)
- Raise exception for unknown states (fail-fast)
- Clear separation between slope mode and lift mode panels
"""

import logging
from typing import Callable

import streamlit as st

from skiresort_planner.constants import StyleConfig
from skiresort_planner.core.geo_calculator import GeoCalculator
from skiresort_planner.model.message import (
    LiftActionMessage,
    LiftPlacingContextMessage,
    SegmentWarningMessage,
    SlopeActionMessage,
    SlopeBuildingContextMessage,
    SlopeStartingContextMessage,
)
from skiresort_planner.model.resort_graph import ResortGraph
from skiresort_planner.ui.actions import bump_map_version
from skiresort_planner.ui.state_machine import PlannerContext, PlannerStateMachine

logger = logging.getLogger(__name__)


# =============================================================================
# STATE DISPATCH
# =============================================================================


def render_control_panel(
    sm: PlannerStateMachine,
    ctx: PlannerContext,
    graph: ResortGraph,
    on_commit: Callable[[int], None],
    on_custom_direction: Callable[[], None],
    on_cancel_custom: Callable[[], None],
    on_cancel_connection: Callable[[], None],
) -> None:
    """Render the appropriate control panel for the current state.

    Panel visibility is orthogonal to state - the info panel can be shown
    in any idle state without requiring a state transition.

    Raises:
        RuntimeError: If current state has no registered panel renderer
    """
    state_name = sm.get_state_name()

    renderers = {
        "Idle": _render_idle_panel,
        "SlopeBuilding": _render_slope_building_panel,
        "LiftPlacing": _render_lift_placing_panel,
    }

    renderer = renderers.get(state_name)
    if renderer is None:
        raise RuntimeError(
            f"No control panel renderer for state '{state_name}'. Available states: {list(renderers.keys())}"
        )

    renderer(
        sm=sm,
        ctx=ctx,
        graph=graph,
        on_commit=on_commit,
        on_custom_direction=on_custom_direction,
        on_cancel_custom=on_cancel_custom,
        on_cancel_connection=on_cancel_connection,
    )


def _render_idle_panel(
    sm: PlannerStateMachine,
    ctx: PlannerContext,
    graph: ResortGraph,
    on_commit: Callable[[int], None],
    on_custom_direction: Callable[[], None],
    on_cancel_custom: Callable[[], None],
    on_cancel_connection: Callable[[], None],
) -> None:
    """Render control panel for IDLE state.

    If panel is visible, show slope or lift stats depending on what's selected.
    Otherwise show nothing (empty panel).
    """
    if ctx.viewing.panel_visible:
        if ctx.viewing.slope_id:
            _render_slope_info_panel(sm=sm, ctx=ctx, graph=graph)
        elif ctx.viewing.lift_id:
            _render_lift_info_panel(sm=sm, ctx=ctx, graph=graph)


def _render_slope_building_panel(
    sm: PlannerStateMachine,
    ctx: PlannerContext,
    graph: ResortGraph,
    on_commit: Callable[[int], None],
    on_custom_direction: Callable[[], None],
    on_cancel_custom: Callable[[], None],
    on_cancel_connection: Callable[[], None],
) -> None:
    """Render control panel for SLOPE_BUILDING state - progress + path selection."""
    # Render progress context message (blue) first
    _render_slope_progress_message(ctx=ctx, graph=graph)

    # Then render path selection panel (yellow action message + controls)
    PathSelectionPanel(
        context=ctx,
        graph=graph,
        on_commit=on_commit,
        on_custom_direction=on_custom_direction,
        on_cancel_custom=on_cancel_custom,
        on_cancel_connection=on_cancel_connection,
    ).render()


def _render_slope_progress_message(ctx: PlannerContext, graph: ResortGraph) -> None:
    """Render the slope progress context message (blue)."""
    name = ctx.building.name
    segs = len(ctx.building.segments)
    if segs > 0:
        stats = graph.get_segment_stats(segment_ids=ctx.building.segments)
        SlopeBuildingContextMessage(
            slope_name=name,
            num_segments=segs,
            difficulty_emoji=StyleConfig.DIFFICULTY_EMOJIS[stats["difficulty"]],
            total_drop_m=stats["total_drop"],
            total_length_m=stats["total_length"],
            avg_gradient_pct=stats["avg_gradient"],
            max_gradient_pct=stats["max_gradient"],
            start_elevation_m=stats["start_elev"],
            current_elevation_m=stats["current_elev"],
        ).display()
    else:
        # New slope with no segments - show start location
        SlopeStartingContextMessage(
            slope_name=name,
            start_node_id=ctx.building.start_node,
            start_lat=ctx.selection.lat,
            start_lon=ctx.selection.lon,
        ).display()


def _render_lift_placing_panel(
    sm: PlannerStateMachine,
    ctx: PlannerContext,
    graph: ResortGraph,
    on_commit: Callable[[int], None],
    on_custom_direction: Callable[[], None],
    on_cancel_custom: Callable[[], None],
    on_cancel_connection: Callable[[], None],
) -> None:
    """Render control panel for LIFT_PLACING state - progress + action."""
    lift_icon = StyleConfig.LIFT_ICONS[ctx.lift.type]

    # Render progress context message (blue) first
    if ctx.lift.start_node_id:
        node = graph.nodes.get(ctx.lift.start_node_id)
        LiftPlacingContextMessage(
            lift_type=ctx.lift.type,
            lift_icon=lift_icon,
            bottom_node_id=ctx.lift.start_node_id,
            bottom_elevation_m=node.elevation if node else 0.0,
        ).display()
    elif ctx.lift.start_location:
        loc = ctx.lift.start_location
        LiftPlacingContextMessage(
            lift_type=ctx.lift.type,
            lift_icon=lift_icon,
            bottom_lat=loc.lat,
            bottom_lon=loc.lon,
            bottom_elevation_m=loc.elevation,
        ).display()
    else:
        raise RuntimeError("LiftPlacing state requires start_node_id or start_location to be set")

    # Then render action instruction (yellow)
    start_elev = 0.0
    if ctx.lift.start_node_id:
        start_node = graph.nodes.get(ctx.lift.start_node_id)
        if start_node:
            start_elev = start_node.elevation
    elif ctx.lift.start_location:
        start_elev = ctx.lift.start_location.elevation

    LiftActionMessage(is_awaiting_top=True, bottom_elevation_m=start_elev).display()


def _render_slope_info_panel(
    sm: PlannerStateMachine,
    ctx: PlannerContext,
    graph: ResortGraph,
) -> None:
    """Render slope info panel with stats and actions (close/delete)."""
    slope_id = ctx.viewing.slope_id
    if slope_id is None:
        raise ValueError("viewing.slope_id must be set when showing slope panel")

    SlopeStatsPanel(graph=graph).render(slope_id=slope_id)

    col_close, col_delete = st.columns(2)
    with col_close:
        if st.button("✖️ Close", key="close_slope", help="Close this panel to start building new slopes and lifts"):
            logger.info(f"Closing slope panel for {slope_id}")
            bump_map_version()  # Clear stale click state
            sm.hide_info_panel()
            st.rerun()
    with col_delete:
        if st.button("🗑️ Delete", type="secondary", key="delete_slope", help="Permanently remove this slope"):
            slope = graph.slopes.get(slope_id)
            if slope is None:
                raise ValueError(f"Slope {slope_id} must exist when panel shows it")
            if graph.delete_slope(slope_id=slope_id):
                logger.info(f"Deleted slope {slope.name} (id={slope_id})")
                bump_map_version()  # Clear stale click state
                sm.hide_info_panel()
                st.rerun()


def _render_lift_info_panel(
    sm: PlannerStateMachine,
    ctx: PlannerContext,
    graph: ResortGraph,
) -> None:
    """Render lift info panel with stats and actions (close/delete)."""
    lift_id = ctx.viewing.lift_id
    if lift_id is None:
        raise ValueError("viewing.lift_id must be set when showing lift panel")

    LiftStatsPanel(graph=graph).render(lift_id=lift_id)

    col_close, col_delete = st.columns(2)
    with col_close:
        if st.button("✖️ Close", key="close_lift", help="Close this panel to start building new lifts and slopes"):
            logger.info(f"Closing lift panel for {lift_id}")
            bump_map_version()  # Clear stale click state
            sm.hide_info_panel()
            st.rerun()
    with col_delete:
        if st.button("🗑️ Delete", type="secondary", key="delete_lift", help="Permanently remove this lift"):
            lift = graph.lifts.get(lift_id)
            if lift is None:
                raise ValueError(f"Lift {lift_id} must exist when panel shows it")
            if graph.delete_lift(lift_id=lift_id):
                logger.info(f"Deleted lift {lift.name} (id={lift_id})")
                bump_map_version()  # Clear stale click state
                sm.hide_info_panel()
                st.rerun()


# =============================================================================
# PATH SELECTION PANEL
# =============================================================================


class PathSelectionPanel:
    """Renders path selection panel with navigation and statistics."""

    def __init__(
        self,
        context: PlannerContext,
        graph: ResortGraph,
        on_commit: Callable[[int], None],
        on_custom_direction: Callable[[], None],
        on_cancel_custom: Callable[[], None],
        on_cancel_connection: Callable[[], None],
    ) -> None:
        self.ctx = context
        self.graph = graph
        self.on_commit = on_commit
        self.on_custom_direction = on_custom_direction
        self.on_cancel_custom = on_cancel_custom
        self.on_cancel_connection = on_cancel_connection

    def render(self) -> None:
        """Render the path selection panel."""
        if self.ctx.custom_connect.enabled:
            SlopeActionMessage(is_custom_direction=True).display()
            if st.button(
                "❌ Cancel Custom Direction",
                width="stretch",
                help="Return to regular path proposals",
            ):
                logger.info("UI: Cancel Custom Direction clicked")
                self.on_cancel_custom()
            return

        if not self.ctx.proposals.paths:
            SlopeActionMessage().display()
            return

        selected_idx = self.ctx.proposals.selected_idx
        num_paths = len(self.ctx.proposals.paths)

        if selected_idx is None or selected_idx >= num_paths:
            SlopeActionMessage(
                is_selecting_path=True,
                num_paths=num_paths,
                selected_path_idx=0,
                path_difficulty="unknown",
                path_difficulty_emoji="⬜",
            ).display()
            return

        path = self.ctx.proposals.paths[selected_idx]
        emoji = StyleConfig.DIFFICULTY_EMOJIS[path.difficulty]
        is_connector = path.is_connector and path.target_node_id

        SlopeActionMessage(
            is_selecting_path=True,
            is_custom_path=self.ctx.custom_connect.force_mode,
            num_paths=num_paths,
            selected_path_idx=selected_idx,
            path_difficulty=path.difficulty,
            path_difficulty_emoji=emoji,
            actual_gradient_pct=path.avg_slope_pct,
            target_gradient_pct=path.target_slope_pct,
            path_length_m=path.length_m,
            path_drop_m=path.total_drop_m,
            start_elevation_m=path.points[0].elevation if path.points else 0.0,
            end_elevation_m=path.points[-1].elevation if path.points else 0.0,
            is_connector=is_connector,
            target_node_id=path.target_node_id if is_connector else None,
        ).display()

        # Navigation arrows
        col_prev, col_nav_label, col_next = st.columns([1, 2, 1])
        with col_prev:
            if st.button("◀", key="prev_path", width="stretch", help="Previous path variant"):
                self.ctx.proposals.selected_idx = (selected_idx - 1) % num_paths
                st.rerun()
        with col_nav_label:
            st.markdown(f"**◀ ▶ Browse {num_paths} paths**")
        with col_next:
            if st.button("▶", key="next_path", width="stretch", help="Next path variant"):
                self.ctx.proposals.selected_idx = (selected_idx + 1) % num_paths
                st.rerun()

        # Commit button
        if is_connector:
            commit_label = f"🏁 Finish → {path.target_node_id}"
            commit_help = f"Connect to {path.target_node_id} and finish this slope"
        else:
            commit_label = "✅ Commit This Path"
            commit_help = "Add this segment and continue building"

        if st.button(commit_label, type="primary", width="stretch", help=commit_help):
            logger.info(f"UI: Commit button clicked for path {selected_idx}, is_connector={is_connector}")
            self.on_commit(selected_idx)

        # Custom Direction button
        if not self.ctx.custom_connect.enabled and not self.ctx.custom_connect.force_mode:
            if st.button(
                "🎯 Custom Direction",
                width="stretch",
                help="Click anywhere downhill to create a path to that point, or connect to an existing node",
            ):
                logger.info("UI: Custom Direction button clicked")
                self.on_custom_direction()

        # Cancel Connection button
        if self.ctx.custom_connect.force_mode:
            if st.button(
                "❌ Cancel Connection",
                width="stretch",
                help="Return to regular path proposals",
            ):
                logger.info("UI: Cancel Connection clicked")
                self.on_cancel_connection()


# =============================================================================
# STATISTICS PANELS
# =============================================================================


class SlopeStatsPanel:
    """Renders statistics panel for a finalized slope."""

    def __init__(self, graph: ResortGraph) -> None:
        self.graph = graph

    def render(self, slope_id: str) -> None:
        """Render statistics panel for the given slope."""
        slope = self.graph.slopes.get(slope_id)

        if not slope:
            raise RuntimeError(
                f"Slope '{slope_id}' not found in graph.slopes - "
                "state machine transitioned to viewing but slope was deleted"
            )

        st.subheader(f"📊 {slope.name}")

        total_length = slope.get_total_length(segments=self.graph.segments)
        total_drop = slope.get_total_drop(segments=self.graph.segments)
        difficulty = slope.get_difficulty(segments=self.graph.segments)
        avg_gradient = (total_drop / total_length * 100) if total_length > 0 else 0
        max_segment_gradient = slope.get_steepest_segment_slope(segments=self.graph.segments)

        first_seg = self.graph.segments.get(slope.segment_ids[0]) if slope.segment_ids else None
        last_seg = self.graph.segments.get(slope.segment_ids[-1]) if slope.segment_ids else None
        top_elev = first_seg.points[0].elevation if first_seg and first_seg.points else 0.0
        bottom_elev = last_seg.points[-1].elevation if last_seg and last_seg.points else 0.0

        diff_emoji = StyleConfig.DIFFICULTY_EMOJIS[difficulty]

        st.markdown(f"**Difficulty:** {diff_emoji} {difficulty.capitalize()}")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Top Elevation", f"{top_elev:.0f}m")
            st.metric("Length", f"{total_length:.0f}m")
            st.metric("Avg Gradient", f"{avg_gradient:.0f}%")
        with col2:
            st.metric("Bottom Elevation", f"{bottom_elev:.0f}m")
            st.metric("Drop", f"{total_drop:.0f}m")
            st.metric(
                "Max Segment Gradient",
                f"{max_segment_gradient:.0f}%",
                help="Steepest segment average gradient - determines the slope difficulty rating",
            )

        with st.expander("📋 Segment Details", expanded=False):
            for i, seg_id in enumerate(slope.segment_ids, 1):
                seg = self.graph.segments.get(seg_id)
                if not seg:
                    continue

                seg_emoji = StyleConfig.DIFFICULTY_EMOJIS[seg.difficulty]
                seg_line = f"{i}. {seg_emoji} **{seg.difficulty.capitalize()}** — {seg.length_m:.0f}m, {seg.avg_slope_pct:.0f}%"

                if seg.warnings:
                    st.markdown(f"{seg_line}")
                    for warning in seg.warnings:
                        SegmentWarningMessage(warning_text=str(warning)).display()
                else:
                    st.markdown(seg_line)


class LiftStatsPanel:
    """Renders statistics panel for a lift."""

    def __init__(self, graph: ResortGraph) -> None:
        self.graph = graph

    def render(self, lift_id: str) -> None:
        """Render statistics panel for the given lift."""
        lift = self.graph.lifts.get(lift_id)

        if not lift:
            raise RuntimeError(
                f"Lift '{lift_id}' not found in graph.lifts - "
                "state machine transitioned to viewing but lift was deleted"
            )

        lift_icon = StyleConfig.LIFT_ICONS[lift.lift_type]
        lift_type_display = lift.lift_type.replace("_", " ").title()
        st.subheader(f"{lift_icon} {lift.name}")
        st.caption(f"Type: **{lift_type_display}** — *Use sidebar buttons to change*")

        start_node = self.graph.nodes.get(lift.start_node_id)
        end_node = self.graph.nodes.get(lift.end_node_id)

        if start_node and end_node:
            vertical_rise = end_node.elevation - start_node.elevation
            horizontal_length = GeoCalculator.haversine_distance_m(
                lat1=start_node.lat,
                lon1=start_node.lon,
                lat2=end_node.lat,
                lon2=end_node.lon,
            )
            inclined_length = (vertical_rise**2 + horizontal_length**2) ** 0.5
            num_pylons = len(lift.pylons)
            avg_gradient = (vertical_rise / horizontal_length * 100) if horizontal_length > 0 else 0

            max_cable_gradient = 0.0
            if len(lift.pylons) >= 2:
                for i in range(len(lift.pylons) - 1):
                    p1 = lift.pylons[i]
                    p2 = lift.pylons[i + 1]
                    dist = p2.distance_m - p1.distance_m
                    elev_diff = p2.top_elevation_m - p1.top_elevation_m
                    if dist > 0:
                        gradient = abs(elev_diff / dist * 100)
                        max_cable_gradient = max(max_cable_gradient, gradient)

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Bottom Elevation", f"{start_node.elevation:.0f}m")
                st.metric("Horizontal Length", f"{horizontal_length:.0f}m")
                st.metric("Vertical Rise", f"{vertical_rise:.0f}m")
                st.metric("Avg Gradient", f"{avg_gradient:.0f}%")
            with col2:
                st.metric("Top Elevation", f"{end_node.elevation:.0f}m")
                st.metric("Inclined Length", f"{inclined_length:.0f}m")
                st.metric("Pylons", f"{num_pylons}")
                st.metric(
                    "Max Cable Gradient",
                    f"{max_cable_gradient:.0f}%",
                    help="Steepest gradient between any two adjacent pylons",
                )

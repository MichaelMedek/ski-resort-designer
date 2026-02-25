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
from typing import TYPE_CHECKING, Callable

import streamlit as st

from skiresort_planner.constants import MapConfig, SlopeConfig, StyleConfig
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
from skiresort_planner.ui.actions import bump_map_version, reload_map
from skiresort_planner.ui.state_machine import PlannerContext, PlannerStateMachine

if TYPE_CHECKING:
    from skiresort_planner.model import Lift, Slope

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIRMATION DIALOGS
# =============================================================================


@st.dialog("Confirm Delete")
def _confirm_delete_dialog(
    entity_type: str,
    entity_name: str,
    entity_id: str,
    delete_fn: Callable[[str], bool],
    sm: PlannerStateMachine,
) -> None:
    """Show confirmation dialog before deleting a slope or lift."""
    st.write(f"Are you sure you want to delete **{entity_name}**?")
    st.caption("This action can be undone using the Undo button.")

    col_yes, col_no = st.columns(2)
    with col_yes:
        if st.button("🗑️ Yes, Delete", type="primary", use_container_width=True):
            if delete_fn(entity_id):
                logger.info(f"Deleted {entity_type} {entity_name} (id={entity_id})")
                bump_map_version()
                # Uses close_panel event - SM resolves to appropriate transition
                sm.hide_info_panel()
            st.rerun()
    with col_no:
        if st.button("✖️ Cancel", use_container_width=True):
            st.rerun()


# =============================================================================
# SHARED HELPERS FOR INFO PANELS
# =============================================================================


def _render_3d_toggle_button(ctx: PlannerContext, graph: ResortGraph, entity_type: str, entity_id: str) -> None:
    """Render 3D/2D view toggle button. Calls reload_map() if button is clicked."""
    if ctx.viewing.view_3d:
        if st.button("🗺️ Return to 2D View", key=f"{entity_type}_2d_view", use_container_width=True):
            logger.info(f"Switching to 2D view from {entity_type} {entity_id}")
            ctx.viewing.disable_3d()
            # Reset pitch, bearing, and zoom to top-down 2D view
            ctx.map.pitch = MapConfig.DEFAULT_PITCH
            ctx.map.bearing = MapConfig.DEFAULT_BEARING
            ctx.map.zoom = MapConfig.DEFAULT_ZOOM
            # Update map center to entity center so we don't jump to stale position
            if entity_type == "slope" and entity_id in graph.slopes:
                slope = graph.slopes[entity_id]
                center_lat, center_lon = GeoCalculator.compute_center(latlons=[(n.lat, n.lon) for n in slope.nodes])
                ctx.map.lat = center_lat
                ctx.map.lon = center_lon
            elif entity_type == "lift" and entity_id in graph.lifts:
                lift = graph.lifts[entity_id]
                start_node = graph.nodes.get(lift.start_node_id)
                end_node = graph.nodes.get(lift.end_node_id)
                if start_node and end_node:
                    ctx.map.lat = (start_node.lat + end_node.lat) / 2
                    ctx.map.lon = (start_node.lon + end_node.lon) / 2
            reload_map()  # Never returns - raises StopExecution
    else:
        if st.button(
            "🏔️ View in 3D",
            key=f"{entity_type}_3d_view",
            use_container_width=True,
            help=f"View {entity_type} from the side with terrain",
        ):
            logger.info(f"Switching to 3D view for {entity_type} {entity_id}")
            ctx.viewing.enable_3d()
            reload_map()  # Never returns - raises StopExecution


def _render_close_delete_buttons(
    sm: PlannerStateMachine,
    ctx: PlannerContext,
    graph: ResortGraph,
    entity_type: str,
    entity_id: str,
    entity: "Slope | Lift",
    delete_fn: Callable[[str], bool],
) -> None:
    """Render close and delete buttons. Triggers state transition or opens dialog."""
    col_close, col_delete = st.columns(2)
    with col_close:
        if st.button(
            "✖️ Close",
            key=f"close_{entity_type}",
            help="Close this panel to start building new slopes and lifts",
        ):
            logger.info(f"Closing {entity_type} panel for {entity_id}")
            ctx.viewing.disable_3d()
            # Reset pitch and bearing to top-down view (preserve zoom level)
            ctx.map.pitch = MapConfig.DEFAULT_PITCH
            ctx.map.bearing = MapConfig.DEFAULT_BEARING
            bump_map_version()
            # Uses close_panel event - SM resolves to appropriate transition
            # State transition triggers st.rerun() via listener - never returns
            sm.hide_info_panel()
    with col_delete:
        if st.button(
            "🗑️ Delete",
            type="secondary",
            key=f"delete_{entity_type}",
            help=f"Permanently remove this {entity_type}",
        ):
            _confirm_delete_dialog(
                entity_type=entity_type,
                entity_name=entity.name,
                entity_id=entity_id,
                delete_fn=delete_fn,
                sm=sm,
            )


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
    # Use state machine properties instead of string comparison
    if sm.is_idle:
        renderer = _render_idle_panel
    elif sm.is_any_slope_state:
        renderer = _render_slope_building_panel
    elif sm.is_lift_placing:
        renderer = _render_lift_placing_panel
    else:
        raise RuntimeError(
            f"No control panel renderer for state '{sm.get_state_name()}'. "
            f"Expected idle, slope building, or lift placing state."
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
    if sm.is_idle_viewing_slope:
        _render_slope_info_panel(sm=sm, ctx=ctx, graph=graph)
    elif sm.is_idle_viewing_lift:
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
    name = ctx.building.name or "Unnamed Slope"
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
    """Render slope info panel with stats and actions (close/delete/3D view)."""
    slope_id = ctx.viewing.slope_id
    if slope_id is None:
        raise ValueError("viewing.slope_id must be set when showing slope panel")

    slope = graph.slopes.get(slope_id)
    if slope is None:
        raise ValueError(f"Slope {slope_id} must exist when panel shows it")

    SlopeStatsPanel(graph=graph).render(slope_id=slope_id)

    _render_3d_toggle_button(ctx=ctx, graph=graph, entity_type="slope", entity_id=slope_id)

    _render_close_delete_buttons(
        sm=sm,
        ctx=ctx,
        graph=graph,
        entity_type="slope",
        entity_id=slope_id,
        entity=slope,
        delete_fn=graph.delete_slope,
    )


def _render_lift_info_panel(
    sm: PlannerStateMachine,
    ctx: PlannerContext,
    graph: ResortGraph,
) -> None:
    """Render lift info panel with stats and actions (close/delete/3D view)."""
    lift_id = ctx.viewing.lift_id
    if lift_id is None:
        raise ValueError("viewing.lift_id must be set when showing lift panel")

    lift = graph.lifts.get(lift_id)
    if lift is None:
        raise ValueError(f"Lift {lift_id} must exist when panel shows it")

    LiftStatsPanel(graph=graph).render(lift_id=lift_id)

    _render_3d_toggle_button(ctx=ctx, graph=graph, entity_type="lift", entity_id=lift_id)

    _render_close_delete_buttons(
        sm=sm,
        ctx=ctx,
        graph=graph,
        entity_type="lift",
        entity_id=lift_id,
        entity=lift,
        delete_fn=graph.delete_lift,
    )


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
                "✖️ Cancel Custom Direction",
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
        is_connector = bool(path.is_connector and path.target_node_id)

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
                reload_map()  # Refresh map with new selection
        with col_nav_label:
            st.markdown(f"**◀ ▶ Browse {num_paths} paths**")
        with col_next:
            if st.button("▶", key="next_path", width="stretch", help="Next path variant"):
                self.ctx.proposals.selected_idx = (selected_idx + 1) % num_paths
                reload_map()  # Refresh map with new selection

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

        # Custom Direction button (not shown if already in custom connect mode)
        if not self.ctx.custom_connect.enabled and not self.ctx.custom_connect.force_mode:  # type: ignore[redundant-expr]  # noqa: SIM102
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
                "✖️ Cancel Connection",
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
            st.metric("Overall Gradient", f"{avg_gradient:.0f}%")
        with col2:
            st.metric("Bottom Elevation", f"{bottom_elev:.0f}m")
            st.metric("Drop", f"{total_drop:.0f}m")
            st.metric(
                "Steepest Section",
                f"{max_segment_gradient:.0f}%",
                help=f"Steepest {SlopeConfig.ROLLING_WINDOW_M}m section within any segment - determines difficulty rating",
            )

        with st.expander("📋 Segment Details", expanded=False):
            for i, seg_id in enumerate(slope.segment_ids, 1):
                seg = self.graph.segments.get(seg_id)
                if not seg:
                    continue

                seg_emoji = StyleConfig.DIFFICULTY_EMOJIS[seg.difficulty]
                seg_line = f"{i}. {seg_emoji} **{seg.difficulty.capitalize()}** — {seg.length_m:.0f}m, {seg.max_slope_pct:.0f}% steepest, {seg.width_m:.0f}m wide"

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
                st.metric("Overall Gradient", f"{avg_gradient:.0f}%")
            with col2:
                st.metric("Top Elevation", f"{end_node.elevation:.0f}m")
                st.metric("Inclined Length", f"{inclined_length:.0f}m")
                st.metric("Pylons", f"{num_pylons}")
                st.metric(
                    "Steepest Section",
                    f"{max_cable_gradient:.0f}%",
                    help="Steepest gradient between any two adjacent pylons",
                )

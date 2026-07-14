"""Sidebar UI renderer for ski resort planner.

Renders the left sidebar with:
- Mode selector (Slopes/Lifts toggle)
- Building controls during slope construction
- Lift type selector in lift mode
- Resort statistics summary
- Save/Load functionality

All rendering logic is encapsulated to keep the main app.py concise.
"""

import json
import logging
from collections.abc import Callable
from datetime import datetime
from typing import Literal

import streamlit as st

from skiresort_planner.constants import (
    LiftConfig,
    MapConfig,
    OSMConfig,
    PathConfig,
    SlopeConfig,
    StyleConfig,
)
from skiresort_planner.enum_utils import enum_eq
from skiresort_planner.generators.geocoder import geocode
from skiresort_planner.model.actions import UndoAction
from skiresort_planner.model.message import (
    FileLoadErrorMessage,
    PlaceNotFoundMessage,
)
from skiresort_planner.model.resort_graph import ResortGraph
from skiresort_planner.model.undo_handlers import UNDO_HANDLERS
from skiresort_planner.persistence import backup_store
from skiresort_planner.ui.actions import undo_last_action
from skiresort_planner.ui.context import BuildMode, EntityKind, PlannerContext
from skiresort_planner.ui.infra import bump_map_version, reload_map, trigger_rerun
from skiresort_planner.ui.mode_registry import BUILD_STATES, OPERATIONS, BuilderOperation, OperationGroup
from skiresort_planner.ui.state_machine import PlannerStateMachine

logger = logging.getLogger(__name__)


def _describe_undo_action(action: UndoAction, graph: ResortGraph) -> str:
    """Generate human-readable description of what undo will do.

    Delegates to the UNDO_HANDLERS registry (keyed by ActionType.name, reload-safe).
    """
    return UNDO_HANDLERS[action.action_type.name].describe(action=action, graph=graph)


def _request_pending_undo() -> None:
    """Flag the main render loop to execute the undo after the dialog closes."""
    st.session_state._pending_undo = True


@st.dialog("Confirm Undo")
def _confirm_undo_dialog(action: UndoAction, graph: ResortGraph) -> None:
    """Show confirmation dialog before undoing an action."""
    description = _describe_undo_action(action=action, graph=graph)
    st.write("**Action to undo:**")
    st.write(description)

    col_yes, col_no = st.columns(2)
    with col_yes:
        if st.button("↩️ Yes, Undo", type="primary", use_container_width=True):
            _request_pending_undo()
            trigger_rerun()
    with col_no:
        if st.button("✖️ Cancel", use_container_width=True):
            trigger_rerun()


def _perform_reset_resort() -> None:
    """Delete the current resort's backup and prime a fresh empty one.

    Drops all session data so init_session_state rebuilds from scratch.
    """
    current = st.session_state.get("resort_id")
    if current:
        backup_store.delete(resort_id=current)
    st.query_params["resort"] = backup_store.new_resort_id()
    # Drop all session data so init_session_state rebuilds fresh.
    for key in ("resort_id", "graph", "state_machine", "context", "map_renderer", "_saved_token"):
        st.session_state.pop(key, None)


@st.dialog("🗑️ Reset to Empty")
def _confirm_reset_resort_dialog() -> None:
    """Confirm resetting to a fresh empty resort.

    Deletes the current resort's backup and starts a brand-new empty one.
    Needed because the bare link always reloads the biggest existing backup,
    so an empty start must be requested explicitly.
    """
    st.warning("This clears the current resort and starts empty. The current backup is deleted. Cannot be undone.")
    col_yes, col_no = st.columns(2)
    with col_yes:
        if st.button("🆕 Yes, Start Empty", type="primary", use_container_width=True):
            _perform_reset_resort()
            trigger_rerun()
    with col_no:
        if st.button("✖️ Cancel", use_container_width=True):
            trigger_rerun()


class SidebarRenderer:
    """Renders the sidebar UI and returns action flags.

    Encapsulates all sidebar rendering logic including mode selection,
    building controls, lift placement, and resort management.
    """

    def __init__(
        self,
        state_machine: PlannerStateMachine,
        context: PlannerContext,
        graph: ResortGraph,
    ) -> None:
        """Initialize sidebar renderer with required dependencies."""
        self.sm = state_machine
        self.ctx = context
        self.graph = graph

    def _disabled_button_reason(self, mode: str, *, is_building_or_placing: bool) -> str:
        """Why the button for `mode` is greyed out (helper for _get_button_help's disabled branch)."""
        if is_building_or_placing:
            return "Finish or cancel current action first"
        if self.sm.is_idle_viewing_slope and not BuildMode.is_slope(mode):
            return "Close slope panel to switch build mode"
        if self.sm.is_idle_viewing_lift and not BuildMode.is_lift(mode):
            return "Close lift panel to switch build mode"
        if self.sm.is_idle_viewing_road and not BuildMode.is_road(mode):
            return "Close road panel to switch build mode"
        raise ValueError(
            f"Button {mode} is disabled but no known reason (building_or_placing={is_building_or_placing})"
        )

    def _get_button_help(self, *, mode: str, label: str, is_disabled: bool, is_building_or_placing: bool) -> str:
        """Generate contextual help text for a build-mode button (disabled reason or enabled action)."""
        if is_disabled:
            return self._disabled_button_reason(mode, is_building_or_placing=is_building_or_placing)
        if self.sm.is_idle_viewing_lift and BuildMode.is_lift(mode):
            return f"Change viewed lift to {label}"
        if BuildMode.is_slope(mode):
            return "Click on map to start building a ski slope"
        if BuildMode.is_road(mode):
            return "Click two points on the map to connect them with a gentle car road"
        if BuildMode.is_lift(mode):
            return f"Click on map to start placing a {label}"
        if BuildMode.is_import(mode):
            return "Select, then click the map to place an import area — real lifts & pistes inside it are added."
        if BuildMode.is_merge(mode):
            return "Select, then click node markers to merge them into one (median position)."
        raise ValueError(f"Button {mode} has no help text (is_disabled={is_disabled})")

    def render(self) -> dict[str, bool | str]:
        """Render complete sidebar and return action flags.

        Returns:
            Dict with keys: undo, cancel_slope, finish_slope, recompute, lift_type
        """
        # Handle pending undo from confirmation dialog (must be before UI rendering)
        if st.session_state.get("_pending_undo"):
            st.session_state._pending_undo = False
            undo_last_action()
            # undo_last_action() calls st.rerun() internally

        with st.sidebar:
            actions: dict[str, bool | str] = {
                "undo": False,
                "cancel_slope": False,
                "finish_slope": False,
                "recompute": False,
                "finish_road": False,
                "cancel_road": False,
                "lift_type": self.ctx.lift.type,
            }

            self._render_mode_selector()
            st.divider()
            actions.update(self._render_mode_specific_controls())
            st.divider()
            self._render_always_available()
            st.divider()
            self._render_resort_data()

            return actions

    def _render_mode_specific_controls(self) -> dict[str, bool | str]:
        """Render the controls for the current state, dispatched via the BUILD_STATES registry.

        Each state owns its sidebar controls (BuildState.sidebar_controls), so a new state can't be
        forgotten — the registry is bijection-asserted against the SM states at import.
        """
        return BUILD_STATES[self.sm.get_current_state_id()].sidebar_controls(self)

    def _render_resort_data(self) -> None:
        """Render the resort-data group: cumulative stats and save/load controls."""
        self._render_resort_stats()
        self._render_save_load()

    def render_close_panel_button(self) -> None:
        """Render close panel button for viewing states."""
        if st.button(
            "✖️ Close Right Panel",
            width="stretch",
            help="Close the right panel to start building new slopes and lifts",
        ):
            bump_map_version()
            # Uses close_panel event - SM resolves to appropriate transition
            # NOTE: State transition triggers st.rerun() via listener
            self.sm.hide_info_panel()

    def _cancel_button(self, label: str, on_cancel: Callable[[], None], help: str) -> None:
        """Render a full-width cancel button that clears stale click state then transitions.

        Shared by lift placement and import placement (both are single-step "placing" modes whose
        cancel just discards the in-progress placement and returns to idle).
        """
        if st.button(label, width="stretch", help=help):
            bump_map_version()  # clear stale click state before the transition
            on_cancel()  # the state transition triggers st.rerun() via the listener

    def render_lift_cancel_button(self) -> None:
        """Render cancel button during lift placement."""
        self._cancel_button(
            label="✖️ Cancel Lift Placement",
            on_cancel=self.sm.cancel_lift,
            help="Discard start point and return to idle",
        )

    def render_road_building_controls(self) -> dict[str, bool | str]:
        """Render controls during road building (mirrors render_slope_building_controls).

        Returns a dict with finish_road / cancel_road flags for the render loop.
        """
        has_segments = self.ctx.road_build.has_committed_segments()

        finish_road = st.button(
            "🏁 Finish Committed Road",
            type="primary",
            width="stretch",
            disabled=not has_segments,
            help=(
                "Add at least one segment before finishing"
                if not has_segments
                else "Finalize the committed segments (any unconfirmed proposal is discarded)"
            ),
        )
        cancel_road = st.button(
            "✖️ Cancel Road",
            width="stretch",
            help="Discard the current road and return to idle",
        )

        return {"finish_road": finish_road, "cancel_road": cancel_road}

    def _render_search_box(self) -> None:
        """Render a place-search box that recenters the map on the top OSM match.

        A form so Enter submits (enter_to_submit) and the script only reruns on submit.
        In the always-available controls, usable in any mode. Recentering only moves the camera;
        it never touches build state, so it's safe while building or viewing.
        """
        with st.form("place_search_form", clear_on_submit=False, border=False):
            col_input, col_btn = st.columns([4, 1])
            with col_input:
                query = st.text_input(
                    "Search place",
                    key="place_search",
                    placeholder="🔍 Search a resort, town, …",
                    label_visibility="collapsed",
                )
            with col_btn:
                submitted = st.form_submit_button("🔍", width="stretch", help="Search and center the map")

        if not submitted:
            return

        result = geocode(query)
        if result is None:
            PlaceNotFoundMessage(query=query.strip()).display()
            return

        self.ctx.map.set_center(lon=result.lon, lat=result.lat)
        self.ctx.map.zoom = MapConfig.DEFAULT_ZOOM
        logger.info(f"UI: Search centered map on {result.display_name!r} ({result.lat:.4f}, {result.lon:.4f})")
        reload_map()

    def _render_always_available(self) -> None:
        """Render the always-available controls: place search, undo, and reset view."""
        self._render_search_box()
        self._render_undo_button()
        self._render_reset_view_button()

    def _render_undo_button(self) -> None:
        """Render the undo button (opens a confirmation dialog for the last action)."""
        can_undo = bool(self.graph.undo_stack)
        if st.button(
            "↩️ Undo Last Action",
            width="stretch",
            disabled=not can_undo,
            help="Nothing to undo" if not can_undo else "Undo the last action",
        ):
            last_action = self.graph.undo_stack[-1]
            _confirm_undo_dialog(action=last_action, graph=self.graph)

    def _render_reset_view_button(self) -> None:
        """Render the reset-view button (recenters camera to defaults, cleans orphan nodes)."""
        if st.button(
            "📷 Reset View",
            width="stretch",
            help="Reset camera to standard position and orientation",
        ):
            self.ctx.map.reset_view()
            # Manual cleanup fallback for any orphaned nodes
            removed = self.graph.cleanup_isolated_nodes()
            if removed > 0:
                logger.warning(f"Reset View cleaned {removed} orphaned node(s)")
            reload_map()  # Bumps version and triggers rerun

    def render_import_building_controls(self) -> None:
        """Render controls while placing an OSM import box (IMPORT_PLACING).

        Shows the area half-width slider (mirrors slope's Segment Length slider — only visible while
        placing) and a Cancel button. Changing the slider writes the new half-width into the deferred
        state and redraws the box at the new size. Confirming happens from the right panel or by
        re-clicking the box center on the map.
        """
        half_width_km = st.slider(
            "Import area half-width (km)",
            min_value=OSMConfig.HALF_WIDTH_MIN_KM,
            max_value=OSMConfig.HALF_WIDTH_MAX_KM,
            value=self.ctx.deferred.osm_import_half_width_km,
            step=0.5,
            key="import_osm_half_width",
            help="Lifts & pistes fully inside the box (this far from the center in each direction) are imported.",
        )
        if half_width_km != self.ctx.deferred.osm_import_half_width_km:
            self.ctx.deferred.osm_import_half_width_km = half_width_km
            reload_map()  # redraw the box at the new size
        self._cancel_button(
            label="✖️ Cancel Import",
            on_cancel=self.sm.cancel_import,
            help="Discard the placed area and return to idle",
        )

    def render_merge_building_controls(self) -> None:
        """Render controls while selecting nodes to merge (MERGE_PLACING).

        Only a Cancel button — the selection count and what-to-do instructions live on the right
        panel (MergePlacingContextMessage / MergeActionMessage). Confirming happens there too.
        """
        self._cancel_button(
            label="✖️ Cancel Merge",
            on_cancel=self.sm.cancel_merge,
            help="Clear the selection and return to idle",
        )

    def _render_mode_selector(self) -> None:
        """Render unified build type selector with 7 buttons.

        Shows buttons for all build types (slope + road + import + 4 lift types).
        One button is always selected.
        Buttons are disabled when in building or placing states.

        When viewing a lift, the lift type buttons change that lift's type.
        Slope is pre-selected by default.
        """
        # Header + body both derive from this one entity so a viewed kind can't drift.
        viewing = self.sm.viewing_entity  # (EntityKind, id) or None
        viewing_kind = viewing[0] if viewing is not None else None

        # Header + button-disabled state come from the current state's BuildState (registry-driven,
        # bijection-asserted against the SM states), so a new state can't be forgotten here.
        build_state = BUILD_STATES[self.sm.get_current_state_id()]
        head = build_state.header(self.ctx)
        buttons_disabled = build_state.blocks_build_buttons()
        current_mode = self.ctx.build_mode.mode

        if buttons_disabled:
            st.markdown(f"### {head.icon} {head.label}")
            st.caption("⏳ Complete or cancel current build to change type")
        elif viewing_kind is not None:
            st.markdown(f"### {head.icon} {head.label}")
            # Same body for every viewed kind; only lifts add a change-type line.
            # enum_eq is reload-safe: EntityKind survives Streamlit reloads while the class is redefined.
            lines = ["- 🔄 Use lift buttons to change type"] if enum_eq(viewing_kind, EntityKind.LIFT) else []
            lines.append("- ✖️ **Close** the right panel to return")
            lines.append(f"- {StyleConfig.BUILDING_ICON} Click terrain/node → new {viewing_kind.value}")
            st.markdown("\n".join(lines))
        else:
            # Idle: the "Ready to Build" header IS the how-to toggle — expand it for the bullets.
            # Collapsed by default so only that one line shows (saves space on small screens).
            with st.expander(f"{head.icon} {head.label}", expanded=False):
                st.markdown(
                    "- 🔘 Select **Slope**, **Road** or **Lift** type below\n"
                    f"- {StyleConfig.BUILDING_ICON} Click terrain/node → start building\n"
                    f"- {StyleConfig.VIEWING_ICON} Click existing slope/road/lift → view stats\n"
                    "- 🛠️ Or use **Import** / **Node Merge** utilities below"
                )

        def render_op_button(op: BuilderOperation) -> None:
            """Render one registry operation as a full-width button (selected = primary + bold)."""
            enabled = op.enabled(self.sm)
            selected = self._op_selected(op.mode, current_mode)
            icon = BuildMode.icon(op.mode)
            label = BuildMode.display_name(op.mode)
            btn_type: Literal["primary", "secondary"] = "primary" if selected else "secondary"
            btn_label = f"{icon} **{label}**" if selected else f"{icon} {label}"
            help_text = self._get_button_help(
                mode=op.mode,
                label=label,
                is_disabled=not enabled,
                is_building_or_placing=buttons_disabled,
            )
            if st.button(
                btn_label,
                width="stretch",
                type=btn_type,
                key=f"build_btn_{op.mode}",
                disabled=not enabled,
                help=help_text,
            ):
                op.on_select(ctx=self.ctx, sm=self.sm)  # each op owns its own select side effects

        # BUILDER group: Slope + Road full-width, then the 4 lift types in a 2x2 grid.
        builders = [op for op in OPERATIONS.values() if op.group == OperationGroup.BUILDER]
        non_lift_builders = [op for op in builders if not BuildMode.is_lift(op.mode)]
        lift_builders = [op for op in builders if BuildMode.is_lift(op.mode)]
        for op in non_lift_builders:
            render_op_button(op)
        for row_start in range(0, len(lift_builders), 2):
            cols = st.columns(2)
            for col, op in zip(cols, lift_builders[row_start : row_start + 2], strict=False):
                with col:
                    render_op_button(op)

        # UTILITY group (import + node-merge): visually separated by a divider — same category
        # technically, but optically distinct from the real builders.
        utilities = [op for op in OPERATIONS.values() if op.group == OperationGroup.UTILITY]
        if utilities:
            st.divider()
            for op in utilities:
                render_op_button(op)

    def _op_selected(self, mode: str, current_mode: str) -> bool:
        """Whether the button for `mode` shows as selected.

        Normally the selected build mode; while viewing a lift, the lift button matching the viewed
        lift's type is highlighted instead (the lift buttons re-type the viewed lift).
        """
        if self.sm.is_idle_viewing_lift and self.ctx.viewing.lift_id and BuildMode.is_lift(mode):
            viewed_lift = self.graph.lifts.get(self.ctx.viewing.lift_id)
            return viewed_lift is not None and viewed_lift.lift_type == mode
        return current_mode == mode

    def render_slope_building_controls(self) -> dict[str, bool | str]:
        """Render controls for slope building state (mirrors render_road_building_controls).

        Returns dict with finish_slope, cancel_slope, recompute flags.
        Note: Undo button is rendered separately in render() for consistency.
        """
        has_segments = self.ctx.has_committed_segments()

        # Action buttons
        finish_slope = st.button(
            "🏁 Finish Committed Slope",
            type="primary",
            width="stretch",
            disabled=not has_segments,
            help=(
                "Commit at least one segment before finishing"
                if not has_segments
                else "Finalize the committed segments (any unconfirmed proposal is discarded)"
            ),
        )

        # Cancel slope - immediate action (no confirmation)
        cancel_slope = st.button(
            "✖️ Cancel Full Slope",
            width="stretch",
            help="Discard current slope and return to IDLE",
        )
        if cancel_slope:
            logger.info(f"UI: Cancel slope requested for {self.ctx.slope_build.name}")

        # Path settings apply only to fan-out proposals; hide the whole block while
        # routing a custom-connect path to a clicked target (force_mode).
        recompute = False
        if not self.ctx.custom_connect.force_mode:
            st.markdown("**⚙️ Path Settings**")
            segment_length = st.slider(
                "Segment Length (m)",
                min_value=PathConfig.SEGMENT_LENGTH_MIN_M,
                max_value=PathConfig.SEGMENT_LENGTH_MAX_M,
                value=self.ctx.segment_length_m,
                step=50,
                help="Target length for generated path segments",
                key="segment_length_slider",
            )

            if segment_length != self.ctx.segment_length_m:
                logger.info(f"UI: Segment length changed to {segment_length}m")
                self.ctx.segment_length_m = segment_length
                self.ctx.click_dedup.pending_recompute = True

            recompute = st.button(
                "🔄 Recompute Paths",
                width="stretch",
                help="Generate new path variations",
            )

        return {
            "finish_slope": finish_slope,
            "cancel_slope": cancel_slope,
            "recompute": recompute,
        }

    def _render_resort_stats(self) -> None:
        """Render resort summary statistics panel with detailed breakdowns."""
        with st.expander("📊 Resort Summary", expanded=False):
            stats = self.graph.get_stats()
            total_slopes = stats.get("total_slopes", 0)
            total_lifts = stats.get("total_lifts", 0)
            total_roads = stats.get("total_roads", 0)

            # Header with counts
            st.markdown(f"**{total_slopes} Slopes • {total_lifts} Lifts • {total_roads} Roads**")

            # Elevation range across all nodes
            elev_range = self.graph.get_elevation_range()
            if elev_range is not None:
                min_elev, max_elev = elev_range
                st.markdown(f"⛰️ Elevation: {min_elev:.0f}m – {max_elev:.0f}m")
            st.divider()

            # === SLOPES SECTION ===
            st.markdown("**⛷️ Slopes**")
            if total_slopes > 0:
                slope_vertical = sum(
                    slope.get_total_drop(segments=self.graph.segments) for slope in self.graph.slopes.values()
                )
                slope_length = sum(
                    slope.get_total_length(segments=self.graph.segments) for slope in self.graph.slopes.values()
                )
                st.markdown(f"↓ {slope_vertical / 1000:.3f}km drop • {slope_length / 1000:.3f}km length")

                # Difficulty breakdown (km) — loop the single-source difficulty list.
                difficulty_lengths: dict[str, float] = {d: 0.0 for d in SlopeConfig.DIFFICULTIES}
                for slope in self.graph.slopes.values():
                    diff = slope.get_difficulty(segments=self.graph.segments)
                    length = slope.get_total_length(segments=self.graph.segments)
                    difficulty_lengths[diff] += length

                st.markdown(
                    " • ".join(
                        f"{StyleConfig.DIFFICULTY_EMOJIS[d]} {difficulty_lengths[d] / 1000:.3f}km"
                        for d in SlopeConfig.DIFFICULTIES
                    )
                )
            else:
                st.caption("No slopes yet")

            st.divider()

            # === LIFTS SECTION ===
            st.markdown("**🚡 Lifts**")
            if total_lifts > 0:
                lift_vertical = sum(
                    lift.get_vertical_rise(nodes=self.graph.nodes) for lift in self.graph.lifts.values()
                )
                lift_length = sum(lift.get_length_m(nodes=self.graph.nodes) for lift in self.graph.lifts.values())
                st.markdown(f"↑ {lift_vertical / 1000:.3f}km rise • {lift_length / 1000:.3f}km length")

                # Lift type breakdown (count) — loop the single-source lift types.
                lift_counts: dict[str, int] = {t: 0 for t in LiftConfig.TYPES}
                for lift in self.graph.lifts.values():
                    lift_counts[lift.lift_type] += 1

                st.markdown(" • ".join(f"{StyleConfig.LIFT_ICONS[t]} {lift_counts[t]}" for t in LiftConfig.TYPES))
            else:
                st.caption("No lifts yet")

            st.divider()

            # === ROADS SECTION ===
            st.markdown(f"**{StyleConfig.ROAD_ICON} Roads**")
            if total_roads > 0:
                road_length = stats.get("total_road_length_m", 0.0)
                # Elevation change across all roads (mirrors slope drop / lift rise line).
                road_elev_change = sum(
                    abs(road.get_total_drop(segments=self.graph.segments)) for road in self.graph.roads.values()
                )
                st.markdown(f"↕ {road_elev_change / 1000:.3f}km elevation change • {road_length / 1000:.3f}km length")
            else:
                st.caption("No roads yet")

    def _render_save_load(self) -> None:
        """Render save/load resort functionality."""
        with st.expander("💾 Resort Data", expanded=False):
            stats = self.graph.get_stats()
            can_save = stats["total_slopes"] > 0 or stats["total_lifts"] > 0 or stats["total_roads"] > 0

            # Load from File
            uploaded_file = st.file_uploader(
                "📂 Load from File",
                type=["json"],
                help="Load a previously saved resort design",
                label_visibility="collapsed",
                key=f"resort_uploader_{st.session_state.get('_upload_counter', 0)}",
            )

            if uploaded_file is not None:
                try:
                    data = json.load(uploaded_file)
                    loaded_graph = ResortGraph.from_dict(data=data)
                    st.session_state.graph = loaded_graph

                    # Center map on mean lat/lon of all nodes
                    center = loaded_graph.get_center()
                    if center is not None:
                        center_lon, center_lat = center
                        self.ctx.map.set_center(lon=center_lon, lat=center_lat)
                        logger.info(f"Centered map on mean: ({center_lat:.5f}, {center_lon:.5f})")

                    logger.info(f"Loaded resort from file: {uploaded_file.name}")
                    st.session_state._upload_counter = st.session_state.get("_upload_counter", 0) + 1
                    # Persist as the session's working backup so an F5 restores it
                    resort_id = st.session_state.get("resort_id")
                    if resort_id:
                        backup_store.save(graph=loaded_graph, resort_id=resort_id)
                        st.session_state._saved_token = loaded_graph.change_token()
                    reload_map()  # Fresh graph needs map version bump for Pydeck
                except Exception as e:
                    FileLoadErrorMessage(error=str(e)).display()
                    logger.error(f"Failed to load resort file: {e}")

            # Save to File
            if can_save:
                resort_json = json.dumps(self.graph.to_dict(), indent=2)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                json_filename = f"alpin_resort_{timestamp}.json"
                st.download_button(
                    "💾 Save to File",
                    data=resort_json,
                    file_name=json_filename,
                    mime="application/json",
                    width="stretch",
                    help="Download resort design as JSON file",
                )
            else:
                st.button(
                    "💾 Save to File",
                    width="stretch",
                    disabled=True,
                    help="Build some slopes, lifts or roads first",
                )

            # Export GPX - always show, disable if no data
            if can_save:
                gpx = self.graph.to_gpx()
                gpx_filename = f"alpin_resort_{datetime.now().strftime('%Y%m%d_%H%M%S')}.gpx"
                st.download_button(
                    "📥 Export GPX",
                    gpx,
                    gpx_filename,
                    "application/gpx+xml",
                    width="stretch",
                    help="Export for GPS devices and mapping apps",
                )
            else:
                st.button(
                    "📥 Export GPX",
                    width="stretch",
                    disabled=True,
                    help="Build some slopes, lifts or roads first",
                )

            st.divider()

            # Reset to a fresh empty resort. Needed because the bare link always
            # reloads the biggest existing backup, so empty must be requested.
            if st.button(
                "🗑️ Reset to Empty",
                width="stretch",
                help="Clear the current resort and start a new empty one",
                disabled=not can_save,
                key="reset_resort_button",
            ):
                _confirm_reset_resort_dialog()

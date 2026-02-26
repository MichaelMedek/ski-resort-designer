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
from datetime import datetime
from typing import Any, Literal, cast

import streamlit as st

from skiresort_planner.constants import (
    PathConfig,
    StyleConfig,
)
from skiresort_planner.model.message import (
    FileLoadErrorMessage,
)
from skiresort_planner.model.resort_graph import (
    ActionType,
    AddLiftAction,
    AddSegmentsAction,
    DeleteLiftAction,
    DeleteSlopeAction,
    FinishSlopeAction,
    ResortGraph,
    UndoAction,
)
from skiresort_planner.ui.actions import bump_map_version, reload_map, undo_last_action
from skiresort_planner.ui.state_machine import (
    BuildMode,
    PlannerContext,
    PlannerStateMachine,
)

logger = logging.getLogger(__name__)


def _describe_undo_action(action: UndoAction, graph: ResortGraph) -> str:
    """Generate human-readable description of what undo will do.

    IMPORTANT: We compare by .name (string) instead of direct enum equality because
    Streamlit's module reloading creates NEW enum class instances on each rerun.
    Objects in st.session_state.graph.undo_stack hold references to the OLD enum
    values, which fail `==` comparison against the NEW enum class values.
    Using .name ensures stable comparison across module reloads.
    """
    if action.action_type.name == ActionType.ADD_SEGMENTS.name:
        act = cast(AddSegmentsAction, action)
        n_segments = len(act.segment_ids)
        return f"Remove {n_segments} segment(s) from current slope"

    elif action.action_type.name == ActionType.FINISH_SLOPE.name:
        act = cast(FinishSlopeAction, action)
        return f"Restore slope **{act.slope_name}** to building mode"

    elif action.action_type.name == ActionType.ADD_LIFT.name:
        act = cast(AddLiftAction, action)
        lift = graph.lifts.get(act.lift_id)
        name = lift.name if lift else act.lift_id
        return f"Delete lift **{name}**"

    elif action.action_type.name == ActionType.DELETE_SLOPE.name:
        act = cast(DeleteSlopeAction, action)
        return f"Restore deleted slope **{act.deleted_slope.name}**"

    elif action.action_type.name == ActionType.DELETE_LIFT.name:
        act = cast(DeleteLiftAction, action)
        return f"Restore deleted lift **{act.deleted_lift.name}**"

    else:
        raise RuntimeError(f"Unknown action type: {action.action_type}")


@st.dialog("Confirm Undo")
def _confirm_undo_dialog(action: UndoAction, graph: ResortGraph) -> None:
    """Show confirmation dialog before undoing an action."""
    description = _describe_undo_action(action=action, graph=graph)
    st.write("**Action to undo:**")
    st.write(description)

    col_yes, col_no = st.columns(2)
    with col_yes:
        if st.button("↩️ Yes, Undo", type="primary", use_container_width=True):
            # Set flag for main render loop to execute undo after dialog closes
            st.session_state._pending_undo = True
            st.rerun()
    with col_no:
        if st.button("✖️ Cancel", use_container_width=True):
            st.rerun()


logger = logging.getLogger(__name__)


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

    @staticmethod
    def _get_button_help(
        *,
        mode: str,
        label: str,
        is_disabled: bool,
        is_building_or_placing: bool,
        viewing_slope: bool,
        viewing_lift: bool,
    ) -> str:
        """Generate contextual help text for build mode buttons.

        Args:
            mode: The BuildMode (slope or lift type)
            label: Display label for the button
            is_disabled: Whether button is currently disabled
            is_building_or_placing: True if in SlopeBuilding or LiftPlacing state
            viewing_slope: True if viewing a slope panel
            viewing_lift: True if viewing a lift panel

        Returns:
            Help text explaining button action or why it's disabled

        Raises:
            ValueError: If state combination doesn't match any known case
        """
        is_slope_mode = BuildMode.is_slope(mode)
        is_lift_mode = BuildMode.is_lift(mode)

        if is_disabled:
            if is_building_or_placing:
                return "Finish or cancel current action first"
            elif viewing_slope and is_lift_mode:
                return "Close slope panel to switch to lift mode"
            elif viewing_lift and is_slope_mode:
                return "Close lift panel to switch to slope mode"
            else:
                raise ValueError(
                    f"Button {mode} is disabled but no known reason: "
                    f"is_building_or_placing={is_building_or_placing}, "
                    f"viewing_slope={viewing_slope}, viewing_lift={viewing_lift}"
                )
        elif viewing_lift and is_lift_mode:
            return f"Change viewed lift to {label}"
        elif is_slope_mode:
            return "Click on map to start building a ski slope"
        elif is_lift_mode:
            return f"Click on map to start placing a {label}"
        else:
            raise ValueError(
                f"Button {mode} has no help text: is_disabled={is_disabled}, "
                f"viewing_lift={viewing_lift}, is_slope={is_slope_mode}, is_lift={is_lift_mode}"
            )

    def render(self) -> dict[str, Any]:
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
            actions = {
                "undo": False,
                "cancel_slope": False,
                "finish_slope": False,
                "recompute": False,
                "lift_type": self.ctx.lift.type,
            }

            self._render_mode_selector()
            st.divider()

            # Mode-specific controls: close button OR building/placing controls
            if self.sm.is_idle_viewing_slope or self.sm.is_idle_viewing_lift:
                self._render_close_panel_button()
            elif self.sm.is_any_slope_state:
                actions.update(self._render_building_controls())
            elif self.sm.is_lift_placing:
                self._render_lift_cancel_button()

            st.divider()
            self._render_undo_reset_buttons()
            st.divider()
            self._render_resort_stats()
            st.divider()
            self._render_save_load()

            return actions

    def _render_close_panel_button(self) -> None:
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

    def _render_lift_cancel_button(self) -> None:
        """Render cancel button during lift placement."""
        if st.button(
            "✖️ Cancel Lift Placement",
            width="stretch",
            help="Discard start point and return to idle",
        ):
            bump_map_version()  # Clear stale click state
            self.sm.cancel_lift()  # Triggers st.rerun() via listener

    def _render_undo_reset_buttons(self) -> None:
        """Render undo and reset view buttons."""
        can_undo = bool(self.graph.undo_stack)
        if st.button(
            "↩️ Undo Last Action",
            width="stretch",
            disabled=not can_undo,
            help="Nothing to undo" if not can_undo else "Undo the last action",
        ):
            last_action = self.graph.undo_stack[-1]
            _confirm_undo_dialog(action=last_action, graph=self.graph)

        if st.button(
            "🎯 Reset View",
            width="stretch",
            help="Reset camera to standard position and orientation",
        ):
            self.ctx.map.reset_view()
            reload_map()  # Bumps version and triggers rerun

    def _render_mode_selector(self) -> None:
        """Render unified build type selector with 5 buttons.

        Shows buttons for all build types (slope + 4 lift types).
        One button is always selected.
        Buttons are disabled when in building or placing states.

        When viewing a lift, the lift type buttons change that lift's type.
        Slope is pre-selected by default.
        """
        # Use state machine properties for viewing checks
        viewing_slope = self.sm.is_idle_viewing_slope
        viewing_lift = self.sm.is_idle_viewing_lift

        if viewing_slope:
            st.markdown("### 👁️ Viewing Slope")
        elif viewing_lift:
            st.markdown("### 👁️ Viewing Lift")
        elif self.sm.is_any_slope_state:
            st.markdown("### 🏗️ Building Slope...")
        elif self.sm.is_lift_placing:
            st.markdown("### 🏗️ Placing Lift...")
        else:
            # All Idle* states (IdleReady, IdleViewingSlope, IdleViewingLift)
            st.markdown("### ⛷️🚡 Ready to Build")

        # Buttons disabled during building/placing
        buttons_disabled = self.sm.is_any_slope_state or self.sm.is_lift_placing
        current_mode = self.ctx.build_mode.mode

        # Note: viewing_slope and viewing_lift already computed above for header
        if buttons_disabled:
            st.caption("⏳ Complete or cancel current build to change type")
        elif viewing_slope:
            st.markdown("- ✖️ **Close** the right panel to return\n- 🗺️ Click terrain/node → new slope")
        elif viewing_lift:
            st.markdown(
                "- 🔄 Use lift buttons to change type\n"
                "- ✖️ **Close** the right panel to return\n"
                "- 🗺️ Click terrain/node → new lift"
            )
        else:
            # All Idle* states without viewing panel
            st.markdown(
                "- 🔘 Select **Slope** or **Lift** type below\n"
                "- 🗺️ Click terrain/node → start building\n"
                "- 👁️ Click existing slope/lift → view stats"
            )

        # Build type options for lifts (2x2 grid)
        lift_options = [
            (BuildMode.CHAIRLIFT, StyleConfig.LIFT_ICONS["chairlift"], StyleConfig.LIFT_DISPLAY_NAMES["chairlift"]),
            (BuildMode.GONDOLA, StyleConfig.LIFT_ICONS["gondola"], StyleConfig.LIFT_DISPLAY_NAMES["gondola"]),
            (
                BuildMode.SURFACE_LIFT,
                StyleConfig.LIFT_ICONS["surface_lift"],
                StyleConfig.LIFT_DISPLAY_NAMES["surface_lift"],
            ),
            (
                BuildMode.AERIAL_TRAM,
                StyleConfig.LIFT_ICONS["aerial_tram"],
                StyleConfig.LIFT_DISPLAY_NAMES["aerial_tram"],
            ),
        ]

        # === SLOPE button (full width) ===
        slope_disabled = buttons_disabled or viewing_lift
        slope_selected = current_mode == BuildMode.SLOPE
        slope_type: Literal["primary", "secondary"] = "primary" if slope_selected else "secondary"
        slope_label = "⛷️ **Slope**" if slope_selected else "⛷️ Slope"
        slope_help = self._get_button_help(
            mode=BuildMode.SLOPE,
            label="Slope",
            is_disabled=slope_disabled,
            is_building_or_placing=buttons_disabled,
            viewing_slope=viewing_slope,
            viewing_lift=viewing_lift,
        )
        # Build mode changes are context-only → use canonical refresh helper
        if st.button(
            slope_label,
            width="stretch",
            type=slope_type,
            key="build_btn_slope",
            disabled=slope_disabled,
            help=slope_help,
        ):
            self.ctx.build_mode.mode = BuildMode.SLOPE
            logger.info("UI: Build mode set to Slope")
            reload_map()

        # === LIFT buttons (2x2 grid) ===
        # Row 1: Chairlift, Gondola
        col1, col2 = st.columns(2)
        for col, (mode, icon, label) in zip([col1, col2], lift_options[:2]):
            with col:
                self._render_lift_button(
                    mode=mode,
                    icon=icon,
                    label=label,
                    current_mode=current_mode,
                    buttons_disabled=buttons_disabled,
                    viewing_slope=viewing_slope,
                    viewing_lift=viewing_lift,
                )

        # Row 2: Surface Lift, Aerial Tram
        col3, col4 = st.columns(2)
        for col, (mode, icon, label) in zip([col3, col4], lift_options[2:]):
            with col:
                self._render_lift_button(
                    mode=mode,
                    icon=icon,
                    label=label,
                    current_mode=current_mode,
                    buttons_disabled=buttons_disabled,
                    viewing_slope=viewing_slope,
                    viewing_lift=viewing_lift,
                )

    def _render_lift_button(
        self,
        mode: str,
        icon: str,
        label: str,
        current_mode: str,
        buttons_disabled: bool,
        viewing_slope: bool,
        viewing_lift: bool,
    ) -> None:
        """Render a single lift type button."""
        # Lift buttons: disabled when building/placing OR viewing slope
        mode_disabled = buttons_disabled or viewing_slope
        is_selected = current_mode == mode

        # When viewing lift, highlight the viewed lift's type
        if viewing_lift and self.ctx.viewing.lift_id:
            viewed_lift = self.graph.lifts.get(self.ctx.viewing.lift_id)
            is_selected = viewed_lift is not None and viewed_lift.lift_type == mode

        button_type: Literal["primary", "secondary"] = "primary" if is_selected else "secondary"
        button_label = f"{icon} **{label}**" if is_selected else f"{icon} {label}"
        button_help = self._get_button_help(
            mode=mode,
            label=label,
            is_disabled=mode_disabled,
            is_building_or_placing=buttons_disabled,
            viewing_slope=viewing_slope,
            viewing_lift=viewing_lift,
        )

        if st.button(
            button_label,
            width="stretch",
            type=button_type,
            key=f"build_btn_{mode}",
            disabled=mode_disabled,
            help=button_help,
        ):
            # When viewing lift, change the lift's type
            if viewing_lift:
                self._change_viewed_lift_type(new_type=mode)
            else:
                self.ctx.build_mode.mode = mode
                self.ctx.lift.type = mode
                logger.info(f"UI: Build mode set to {BuildMode.display_name(mode)}")
            reload_map()  # Build mode changes are context-only

    def _change_viewed_lift_type(self, new_type: str) -> None:
        """Change the type of the currently viewed lift.

        Uses Lift.update_type() to recalculate all type-dependent fields.
        Also updates global build_mode so new lifts use this type.
        """
        lift_id = self.ctx.viewing.lift_id
        if not lift_id:
            raise RuntimeError("_change_viewed_lift_type called but no lift_id in viewing context")

        lift = self.graph.lifts.get(lift_id)
        if not lift:
            return  # Lift deleted?

        # Always update global build_mode (even if same type - ensures consistency)
        self.ctx.build_mode.mode = new_type
        self.ctx.lift.type = new_type

        if lift.lift_type == new_type:
            return  # No actual type change needed

        # Get nodes for the update
        start_node = self.graph.nodes.get(lift.start_node_id)
        end_node = self.graph.nodes.get(lift.end_node_id)
        if not start_node or not end_node:
            logger.warning(f"Cannot update lift {lift_id}: nodes not found")
            return

        # Use centralized method to update all type-dependent fields
        lift.update_type(new_type=new_type, start_node=start_node, end_node=end_node)

        logger.info(f"UI: Changed lift {lift_id} type to {new_type}")

    def _render_building_controls(self) -> dict[str, Any]:
        """Render controls for slope building state.

        Returns dict with finish_slope, cancel_slope, recompute flags.
        Note: Undo button is rendered separately in render() for consistency.
        """
        has_segments = self.ctx.has_committed_segments()

        # Action buttons
        finish_slope = st.button(
            "🏁 Finish Slope",
            type="primary",
            width="stretch",
            disabled=not has_segments,
            help="Commit at least one segment before finishing" if not has_segments else "Finalize this slope",
        )

        # Cancel slope - immediate action (no confirmation)
        cancel_slope = st.button(
            "✖️ Cancel Full Slope",
            width="stretch",
            help="Discard current slope and return to IDLE",
        )
        if cancel_slope:
            logger.info(f"UI: Cancel slope requested for {self.ctx.building.name}")

        # Path generation settings
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

            # Header with counts
            st.markdown(f"**{total_slopes} Slopes • {total_lifts} Lifts**")
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

                # Difficulty breakdown (km)
                difficulty_lengths: dict[str, float] = {"green": 0, "blue": 0, "red": 0, "black": 0}
                for slope in self.graph.slopes.values():
                    diff = slope.get_difficulty(segments=self.graph.segments)
                    length = slope.get_total_length(segments=self.graph.segments)
                    difficulty_lengths[diff] += length

                st.markdown(
                    f"🟢 {difficulty_lengths['green'] / 1000:.3f}km • "
                    f"🔵 {difficulty_lengths['blue'] / 1000:.3f}km • "
                    f"🔴 {difficulty_lengths['red'] / 1000:.3f}km • "
                    f"⚫ {difficulty_lengths['black'] / 1000:.3f}km"
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

                # Lift type breakdown (count)
                lift_counts: dict[str, int] = {"chairlift": 0, "gondola": 0, "surface_lift": 0, "aerial_tram": 0}
                for lift in self.graph.lifts.values():
                    lift_counts[lift.lift_type] += 1

                st.markdown(
                    f"🪑 {lift_counts['chairlift']} • "
                    f"🚡 {lift_counts['gondola']} • "
                    f"🎿 {lift_counts['surface_lift']} • "
                    f"🚠 {lift_counts['aerial_tram']}"
                )
            else:
                st.caption("No lifts yet")

    def _render_save_load(self) -> None:
        """Render save/load resort functionality."""
        with st.expander("💾 Resort Data", expanded=False):
            stats = self.graph.get_stats()
            can_save = stats["total_slopes"] > 0 or stats["total_lifts"] > 0

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
                    if loaded_graph.nodes:
                        lats = [n.lat for n in loaded_graph.nodes.values()]
                        lons = [n.lon for n in loaded_graph.nodes.values()]
                        mean_lat = sum(lats) / len(lats)
                        mean_lon = sum(lons) / len(lons)
                        self.ctx.map.set_center(lon=mean_lon, lat=mean_lat)
                        logger.info(f"Centered map on mean: ({mean_lat:.5f}, {mean_lon:.5f})")

                    logger.info(f"Loaded resort from file: {uploaded_file.name}")
                    st.session_state._upload_counter = st.session_state.get("_upload_counter", 0) + 1
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
                    help="Build some slopes or lifts first",
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
                    help="Build some slopes or lifts first",
                )

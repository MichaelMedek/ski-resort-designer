"""Sidebar UI renderer for ski resort planner.

Renders the left sidebar with:
- Mode selector (Slopes/Lifts toggle)
- Mode-specific controls, dispatched to the state's SidebarPanel (see ui/sidebar_panels.py)
- Always-available controls (place search, undo, reset view)
- Resort statistics summary and Save/Load

All rendering logic is encapsulated to keep the main app.py concise.
"""

import json
import logging
from datetime import datetime
from typing import Literal

import streamlit as st

from skiresort_planner.constants import (
    LiftConfig,
    MapConfig,
    SlopeConfig,
    StyleConfig,
)
from skiresort_planner.generators.geocoder import geocode
from skiresort_planner.model.actions import UndoAction
from skiresort_planner.model.message import (
    FileLoadErrorMessage,
    PlaceNotFoundMessage,
    UploadBlockedMessage,
)
from skiresort_planner.model.resort_graph import ResortGraph
from skiresort_planner.model.undo_handlers import UNDO_HANDLERS
from skiresort_planner.persistence import backup_store
from skiresort_planner.ui.actions import undo_cancels_current_build, undo_last_action
from skiresort_planner.ui.context import BuildMode, PlannerContext
from skiresort_planner.ui.dialogs import ConfirmDialog
from skiresort_planner.ui.infra import bump_camera_epoch, reload_map, trigger_rerun
from skiresort_planner.ui.mode_registry import BUILD_STATES, OPERATIONS, BuilderOperation, OperationGroup
from skiresort_planner.ui.state_machine import PlannerStateMachine

logger = logging.getLogger(__name__)


def _describe_undo_action(action: UndoAction, graph: ResortGraph) -> str:
    """Human-readable description of a specific undo-stack action (reload-safe registry lookup)."""
    return UNDO_HANDLERS[action.action_type.name].describe(action=action, graph=graph)


def _describe_next_undo(graph: ResortGraph) -> str:
    """Confirmation text for the next undo. Only reached when the undo actually shows a dialog —
    routine steps (segment peel, build cancel) skip confirmation, so they never land here.
    """
    return _describe_undo_action(action=graph.undo_stack[-1], graph=graph)


def _next_undo_skips_confirm(sm: PlannerStateMachine, ctx: PlannerContext, graph: ResortGraph) -> bool:
    """Whether the next undo runs without the confirmation dialog.

    True for a routine builder step — peeling a just-committed path segment mid-build (the
    handler's ``skip_confirm``). Cancelling an in-progress build (no committed segments) also
    counts, since it is the same one-tap "step back" the builder expects.
    """
    if undo_cancels_current_build(sm=sm, ctx=ctx):
        return True
    return UNDO_HANDLERS[graph.undo_stack[-1].action_type.name].skip_confirm


def _request_pending_undo() -> None:
    """Flag the main render loop to execute the undo after the dialog closes."""
    st.session_state._pending_undo = True


class _UndoDialog(ConfirmDialog):
    """Confirm undoing the next action on the stack."""

    def __init__(self, graph: ResortGraph) -> None:
        self.graph = graph

    @property
    def title(self) -> str:
        return "↩️ Confirm Undo"

    def _body(self) -> None:
        st.write("**Undoing will:**")
        st.write(_describe_next_undo(graph=self.graph))

    def _on_confirm(self) -> None:
        _request_pending_undo()


def _perform_reset_resort() -> None:
    """Delete the current resort's backup and prime a fresh empty one.

    Drops all session data so init_session_state rebuilds from scratch.
    """
    current = st.session_state.get("resort_id")
    if current:
        logger.info(f"Resetting resort: deleting backup for resort_id={current}")
        backup_store.delete(resort_id=current)
    st.query_params["resort"] = backup_store.new_resort_id()
    # Drop all session data so init_session_state rebuilds fresh.
    for key in ("resort_id", "graph", "state_machine", "context", "map_renderer", "_saved_token"):
        st.session_state.pop(key, None)


class _ResetResortDialog(ConfirmDialog):
    """Confirm resetting to a fresh empty resort.

    Deletes the current resort's backup and starts a brand-new empty one.
    Needed because the bare link always reloads the biggest existing backup,
    so an empty start must be requested explicitly.
    """

    @property
    def title(self) -> str:
        return "🗑️ Reset to Empty"

    def _body(self) -> None:
        st.write("This clears the current resort and starts empty. The current backup is deleted. Cannot be undone.")

    def _on_confirm(self) -> None:
        _perform_reset_resort()


class SidebarRenderer:
    """Renders the sidebar UI.

    Owns the always-available chrome (mode selector, place search, undo, reset, resort data) and
    dispatches the current state's mode-specific controls to its SidebarPanel. Fire-and-forget:
    panel buttons call their action functions directly, so render() returns nothing.
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

    def _disabled_button_reason(self, *, is_building_or_placing: bool) -> str:
        """Why a greyed-out build button is disabled (helper for _get_button_help's disabled branch)."""
        if is_building_or_placing:
            return "Finish or cancel current action first"
        # The ONLY other way a builder is disabled is a viewing panel being open.
        assert not self.sm.is_idle_ready, "a builder button is only disabled while building/placing or viewing"
        return "Close the open panel to switch build mode"

    def _get_button_help(self, *, mode: str, label: str, is_disabled: bool, is_building_or_placing: bool) -> str:
        """Generate contextual help text for a build-mode button (disabled reason or enabled action)."""
        if is_disabled:
            return self._disabled_button_reason(is_building_or_placing=is_building_or_placing)
        if self.sm.is_idle_viewing_lift and BuildMode.is_lift(mode):
            return f"Change viewed lift to {label}"
        if BuildMode.is_slope(mode):
            return "Select, then click on map to start building a ski slope"
        if BuildMode.is_road(mode):
            return "Select, then click on map to start building a car road"
        if BuildMode.is_lift(mode):
            return f"Select, then click on map to start placing a {label}"
        if BuildMode.is_import(mode):
            return "Select, then click the map to place an OpenStreetMap import area."
        if BuildMode.is_node_edit(mode):
            return "Select, then click node markers to edit them (add, delete or merge)."
        if BuildMode.is_route(mode):
            return "Select, then click a start node and an end node to see the best routes."
        raise ValueError(f"Button {mode} has no help text (is_disabled={is_disabled})")

    def render(self) -> None:
        """Render the complete sidebar.

        Fire-and-forget: mode-specific buttons (Finish/Cancel/Recompute/Confirm) call their action
        functions directly from their SidebarPanel, so nothing is returned to app.py.
        """
        # Handle pending undo from confirmation dialog (must be before UI rendering)
        if st.session_state.get("_pending_undo"):
            st.session_state._pending_undo = False
            undo_last_action()
            # undo_last_action() calls st.rerun() internally

        with st.sidebar:
            self._render_mode_selector()
            st.divider()
            self._render_mode_specific_controls()
            st.divider()
            self._render_always_available()
            st.divider()
            self._render_resort_data()

    def _render_mode_specific_controls(self) -> None:
        """Render the current state's mode-specific controls via the BUILD_STATES registry.

        Each state owns its sidebar panel (BuildState.sidebar_panel), so a new state can't be
        forgotten — the registry is bijection-asserted against the SM states at import.
        In idle, also show the selected mode's one-line first-click hint (abstract on the operation).
        """
        if self.sm.is_idle_ready:
            st.caption(OPERATIONS[self.ctx.build_mode.mode].first_instruction)
        BUILD_STATES[self.sm.get_current_state_id()].sidebar_panel(sm=self.sm, ctx=self.ctx, graph=self.graph).render()

    def _render_resort_data(self) -> None:
        """Render the resort-data group: cumulative stats and save/load controls."""
        self._render_resort_stats()
        self._render_save_load()

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
                    placeholder="🔍 Search place…",
                    label_visibility="collapsed",
                )
            with col_btn:
                submitted = st.form_submit_button("🔍", width="stretch", help="Search and center the map")

        if not submitted:
            return

        result = geocode(query)
        if result is None:
            logger.warning(f"UI: Geocode found no result for query {query.strip()!r}")
            PlaceNotFoundMessage(query=query.strip()).display()
            return

        logger.info(f"UI: Search centered map on {result.display_name!r} ({result.lat:.4f}, {result.lon:.4f})")
        reload_map(center=(result.lon, result.lat), zoom=MapConfig.DEFAULT_ZOOM)

    def _render_always_available(self) -> None:
        """Render the always-available controls: place search, undo, and reset view."""
        self._render_search_box()
        self._render_undo_button()
        self._render_reset_view_button()

    def _render_undo_button(self) -> None:
        """Render the undo button. Routine builder steps (peeling a segment / cancelling a
        just-started build) undo immediately; everything else confirms via a dialog first.
        """
        can_undo = bool(self.graph.undo_stack)
        if st.button(
            "↩️ Undo Last Action",
            width="stretch",
            disabled=not can_undo,
            help="Nothing to undo" if not can_undo else "Undo the last action",
        ):
            if _next_undo_skips_confirm(sm=self.sm, ctx=self.ctx, graph=self.graph):
                _request_pending_undo()
                trigger_rerun()
            else:
                _UndoDialog(graph=self.graph).show()

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
            # Reset zoom/pitch/bearing (reset_view) but keep the user where they are: reframe on the
            # current center at the default zoom.
            reload_map(center=(self.ctx.map.lon, self.ctx.map.lat), zoom=MapConfig.DEFAULT_ZOOM)

    def _render_mode_selector(self) -> None:
        """Render unified build type selector with 7 buttons.

        Shows buttons for all build types (slope + road + import + 4 lift types).
        One button is always selected.
        Buttons are disabled when in building or placing states.

        When viewing a lift, the lift type buttons change that lift's type.
        Slope is pre-selected by default.
        """
        # Info block + button-disabled state come from the current state's BuildState (registry-driven,
        # bijection-asserted against the SM states), so a new state can't be forgotten here.
        build_state = BUILD_STATES[self.sm.get_current_state_id()]
        buttons_disabled = build_state.blocks_build_buttons()
        current_mode = self.ctx.build_mode.mode

        # Every state renders one collapsed expander (header line visible, bullets on expand) — a
        # single shape so viewing/building no longer drift from the idle "Ready to Build" block.
        block = build_state.info_block(self.ctx)
        with st.expander(f"{block.icon} {block.label}", expanded=False):
            st.markdown("\n".join(f"- {b}" for b in block.bullets))

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

        # UTILITY group (OSM importer + node editor): visually separated by a divider — same category
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
            return self.graph.lifts[self.ctx.viewing.lift_id].lift_type == mode
        return current_mode == mode

    def _render_resort_stats(self) -> None:
        """Render resort summary statistics panel with detailed breakdowns."""
        with st.expander("📊 Resort Summary", expanded=False):
            stats = self.graph.get_stats()
            total_slopes = stats["total_slopes"]
            total_lifts = stats["total_lifts"]
            total_roads = stats["total_roads"]

            # Header with counts
            st.markdown(f"**{total_slopes} Slopes • {total_lifts} Lifts • {total_roads} Roads**")

            # Connectivity — one block per category (count in heading + biggest offenders) so
            # they're seen without clicking each entity. One loop so the two lists can't drift apart.
            defects = stats["defects"]
            for attr, heading in (
                ("disconnected", "Disconnected from core area"),
                ("no_return", "One-way (can't loop back)"),
            ):
                offenders = sorted((d for d in defects if getattr(d, attr)), key=lambda d: d.length_m, reverse=True)
                if offenders:
                    st.markdown(f"⚠️ **{len(offenders)} {heading}** (largest {min(3, len(offenders))}):")
                    for d in offenders[:3]:
                        st.markdown(f"&nbsp;&nbsp;• {d.name} — {d.length_m / 1000:.1f}km")

            # Elevation range across all nodes
            elev_range = self.graph.get_elevation_range()
            if elev_range is not None:
                min_elev, max_elev = elev_range
                st.markdown(f"⛰️ Elevation: {min_elev:.0f}m – {max_elev:.0f}m")

            st.divider()

            # === SLOPES SECTION ===
            st.markdown("**⛷️ Slopes**")
            if total_slopes > 0:
                # Resort total
                total_slope_drop_km = stats["total_slope_drop_m"] / 1000
                total_slope_length_km = stats["total_slope_length_m"] / 1000
                st.markdown(
                    f"🏔️ Resort total: {total_slope_drop_km:.1f}km drop • {total_slope_length_km:.1f}km slope length"
                )
                # Greatest continuous descent (chained slopes, no lift) — max vertical drop, the marquee run.
                descent = stats["greatest_descent"]
                st.markdown(
                    f"🏔️ Greatest descent: {descent.drop_m:.0f}m drop • {descent.length_m / 1000:.1f}km "
                    f"({descent.top_elev_m:.0f}m→{descent.bottom_elev_m:.0f}m)"
                )

                # Difficulty breakdown (km) — loop the single-source difficulty list.
                difficulty_lengths: dict[str, float] = {d: 0.0 for d in SlopeConfig.DIFFICULTIES}
                for slope in self.graph.slopes.values():
                    diff = slope.get_difficulty(segments=self.graph.segments)
                    length = slope.get_total_length(segments=self.graph.segments)
                    difficulty_lengths[diff] += length

                st.markdown(
                    " • ".join(
                        f"{StyleConfig.DIFFICULTY_EMOJIS[d]} {difficulty_lengths[d] / 1000:.1f}km"
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
                st.markdown(f"↑ {lift_vertical / 1000:.1f}km rise • {lift_length / 1000:.1f}km lift length")

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
                road_length_km = stats["total_road_length_m"] / 1000
                st.markdown(f"{road_length_km:.1f}km road length")
            else:
                st.caption("No roads yet")

    def _render_save_load(self) -> None:
        """Render save/load resort functionality."""
        with st.expander("💾 Resort Data", expanded=False):
            stats = self.graph.get_stats()
            has_content = stats["total_slopes"] > 0 or stats["total_lifts"] > 0 or stats["total_roads"] > 0

            # Load from File
            uploaded_file = st.file_uploader(
                "📂 Load from File",
                type=["json"],
                help="Load a previously saved resort design (only into an empty resort)",
                label_visibility="collapsed",
                key=f"resort_uploader_{st.session_state.get('_upload_counter', 0)}",
            )

            if uploaded_file is not None:
                if has_content:
                    # Resort still has content — refuse to overwrite it silently.
                    UploadBlockedMessage().display()
                    st.session_state._upload_counter = st.session_state.get("_upload_counter", 0) + 1
                else:
                    try:
                        data = json.load(uploaded_file)
                        loaded_graph = ResortGraph.from_dict(data=data)
                        st.session_state.graph = loaded_graph

                        logger.info(f"Loaded resort from file: {uploaded_file.name}")
                        st.session_state._upload_counter = st.session_state.get("_upload_counter", 0) + 1
                        # Persist as the session's working backup so an F5 restores it
                        resort_id = st.session_state.get("resort_id")
                        if resort_id:
                            backup_store.save(graph=loaded_graph, resort_id=resort_id)
                            st.session_state._saved_token = loaded_graph.change_token()
                        # Frame the loaded resort; empty graph → bare remount.
                        center = loaded_graph.get_center()
                        if center is not None:
                            logger.info(f"Centered map on mean: ({center[1]:.5f}, {center[0]:.5f})")
                            reload_map(center=center, zoom=MapConfig.DEFAULT_ZOOM)
                        else:
                            bump_camera_epoch()
                            trigger_rerun()
                    except Exception as e:
                        FileLoadErrorMessage(error=str(e)).display()
                        logger.error(f"Failed to load resort file: {e}")

            # Save to File + Export GPX: same button, disabled while the resort is empty.
            downloads = (
                ("💾 Save to File", "json", "application/json", "Download resort design as JSON file"),
                ("📥 Export GPX", "gpx", "application/gpx+xml", "Export for GPS devices and mapping apps"),
            )
            for label, ext, mime, help_text in downloads:
                if has_content:
                    payload = json.dumps(self.graph.to_dict(), indent=2) if ext == "json" else self.graph.to_gpx()
                    st.download_button(
                        label,
                        data=payload,
                        file_name=f"alpin_resort_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{ext}",
                        mime=mime,
                        width="stretch",
                        help=help_text,
                    )
                else:
                    st.button(label, width="stretch", disabled=True, help="Build some slopes, lifts or roads first")

            # Reset to a fresh empty resort. Needed because the bare link always
            # reloads the biggest existing backup, so empty must be requested.
            if st.button(
                "🗑️ Reset to Empty",
                width="stretch",
                help="Clear the current resort and start a new empty one",
                disabled=not has_content,
                key="reset_resort_button",
            ):
                _ResetResortDialog().show()

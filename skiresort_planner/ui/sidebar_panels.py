"""Left-sidebar mode-specific control panels (mirror of right_panel.py's ControlPanel).

Each build state owns a SidebarPanel that renders its mode-specific sidebar controls (the buttons
and sliders that appear between the mode selector and the always-available controls). Like the
right panel's ControlPanel, panels are fire-and-forget: a button click calls its action function
directly (finish_current_slope, sm.cancel_lift, …) rather than returning a flag for app.py to act
on. This module sits BELOW the dispatch hub (mode_registry) — exactly like right_panel — so the hub
can construct these panels; it must not import mode_registry or left_panel.

The left panel is simpler than the right (a single control region, no context/action messages), so
SidebarPanel exposes one hook, `controls()`, where ControlPanel has three.
"""

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable

import streamlit as st

from skiresort_planner.constants import OSMConfig, PathConfig
from skiresort_planner.model.path_segment import SegmentKind
from skiresort_planner.model.resort_graph import ResortGraph
from skiresort_planner.ui.actions import (
    cancel_current_build,
    finish_current_build,
    recompute_paths,
)
from skiresort_planner.ui.context import PlannerContext
from skiresort_planner.ui.infra import bump_dedup_epoch, trigger_rerun
from skiresort_planner.ui.kind_spec import KIND_SPECS
from skiresort_planner.ui.state_machine import PlannerStateMachine

logger = logging.getLogger(__name__)


def _cancel_button(label: str, on_cancel: Callable[[], None], help: str) -> None:
    """Render a full-width cancel button that clears stale click state then transitions.

    Shared by the single-step "placing" panels (lift / import / merge) whose cancel just discards
    the in-progress placement and returns to idle. The state transition triggers st.rerun() via the
    SM listener.
    """
    if st.button(label, width="stretch", help=help):
        bump_dedup_epoch()  # canceled build's markers gone → refresh dedup (no recenter)
        on_cancel()


class SidebarPanel(ABC):
    """Base class for a build state's mode-specific sidebar controls.

    Fixed template: render() defers to the single controls() hook. Mirrors right_panel.ControlPanel
    but with one hook instead of three (the left panel has no context/action messages).
    """

    def __init__(self, sm: PlannerStateMachine, ctx: PlannerContext, graph: ResortGraph) -> None:
        self.sm = sm
        self.ctx = ctx
        self.graph = graph

    def render(self) -> None:
        """Fixed template: render the state's controls. Never overridden."""
        self.controls()

    @abstractmethod
    def controls(self) -> None:
        """Render this state's mode-specific sidebar buttons/sliders (fire actions directly)."""

    def _render_close_panel_button(self) -> None:
        """The kind-agnostic 'Close Right Panel' button shared by every viewing state's sidebar.

        Fires the shared close_panel event; the SM resolves it to the right per-state transition.
        """
        if st.button(
            "✖️ Close Right Panel",
            width="stretch",
            help="Close the right panel to start building",
            key="close_panel_btn",
        ):
            bump_dedup_epoch()  # closing the panel keeps the user's pan (no recenter)
            self.sm.close_panel()  # type: ignore[attr-defined]  # dynamic python-statemachine event


class IdleSidebarPanel(SidebarPanel):
    """idle_ready: no mode-specific controls (mirrors EmptyControlPanel)."""

    def controls(self) -> None:
        return None


class ViewingSidebarPanel(SidebarPanel):
    """idle_viewing_{slope,road,lift}: a kind-agnostic Close Right Panel button.

    Unlike the right panel (which needs a per-kind EntityKindSpec for stats/delete), the left close
    button is identical for every viewed kind, so one panel covers all three viewing states.
    """

    def controls(self) -> None:
        self._render_close_panel_button()


class PathBuildSidebarPanel(SidebarPanel):
    """The *_starting / *_building / *_custom_path states for ANY path kind (slope or road):
    Finish + Cancel + the shared Path Settings block (segment-length slider + Recompute).

    One class for every buildable path kind — the per-kind bits (display noun, finish/cancel
    actions) are resolved from the kind, so slope and road cannot drift and a future kind gets
    the full panel for free. Constructed with the active build kind by the dispatch hub.
    """

    def __init__(self, sm: PlannerStateMachine, ctx: PlannerContext, graph: ResortGraph, kind: SegmentKind) -> None:
        super().__init__(sm=sm, ctx=ctx, graph=graph)
        self.kind = kind

    def controls(self) -> None:
        kind = self.kind
        noun = KIND_SPECS[kind].display_noun  # "Slope" / "Road"
        has_segments = self.ctx.build(kind).has_committed_segments()

        if st.button(
            f"🏁 Finish Committed {noun}",
            type="primary",
            width="stretch",
            disabled=not has_segments,
            help=(
                "Commit at least one segment before finishing"
                if not has_segments
                else "Finalize the committed segments (any unconfirmed proposal is discarded)"
            ),
            key=f"finish_{kind.value}_btn",
        ):
            finish_current_build(kind=kind)

        if st.button(
            f"✖️ Cancel Full {noun}",
            width="stretch",
            help=f"Discard current {kind.value} and return to IDLE",
            key=f"cancel_{kind.value}_btn",
        ):
            logger.debug(f"UI: Cancel {kind.value} requested for {self.ctx.build(kind).name}")
            cancel_current_build(kind=kind)

        # Path settings apply only to fan-out proposals; hide the whole block while
        # routing a custom-connect path to a clicked target (force_mode).
        if self.ctx.custom_connect.force_mode:
            return

        st.markdown("**⚙️ Path Settings**")
        segment_length = st.slider(
            "Segment Length (m)",
            min_value=PathConfig.SEGMENT_LENGTH_MIN_M,
            max_value=PathConfig.SEGMENT_LENGTH_MAX_M,
            value=self.ctx.segment_length_m,
            step=50,
            help="Target length for generated path segments",
            key=f"segment_length_slider_{kind.value}",
        )
        if segment_length != self.ctx.segment_length_m:
            logger.debug(f"UI: Segment length changed to {segment_length}m")
            self.ctx.segment_length_m = segment_length
            self.ctx.click_dedup.pending_recompute = True

        recompute = st.button(
            "🔄 Recompute Paths",
            width="stretch",
            help="Generate new path variations",
            key=f"recompute_{kind.value}_btn",
        )
        # Set-and-consume in the same frame/state: the slider above sets pending_recompute and the
        # slider change does not itself rerun, so we honor it here alongside an explicit click.
        if recompute or self.ctx.click_dedup.pending_recompute:
            recompute_paths()


class LiftSidebarPanel(SidebarPanel):
    """lift_placing: a Cancel button (the map click completes the lift)."""

    def controls(self) -> None:
        _cancel_button(
            label="✖️ Cancel Lift Placement",
            on_cancel=self.sm.cancel_lift,
            help="Discard start point and return to idle",
        )


class ImportSidebarPanel(SidebarPanel):
    """import_placing: the area half-width slider + a Cancel button.

    The slider mirrors slope's Segment Length slider (only visible while placing). Changing it writes
    the new half-width into the deferred state and redraws the box. Confirming happens from the right
    panel or by re-clicking the box center on the map.
    """

    def controls(self) -> None:
        half_width_km = st.slider(
            "Import area half-width (km)",
            min_value=OSMConfig.HALF_WIDTH_MIN_KM,
            max_value=OSMConfig.HALF_WIDTH_MAX_KM,
            value=self.ctx.pending.osm_import_half_width_km,
            step=0.5,
            key="import_osm_half_width",
            help="Lifts + slopes fully inside the box (this far from the center in each direction) are imported.",
        )
        if half_width_km != self.ctx.pending.osm_import_half_width_km:
            self.ctx.pending.osm_import_half_width_km = half_width_km
            trigger_rerun()  # redraw the box at the new size (no recenter)
        _cancel_button(
            label="✖️ Cancel Import",
            on_cancel=self.sm.cancel_import,
            help="Discard the placed area and return to idle",
        )


class MergeSidebarPanel(SidebarPanel):
    """merge_placing: a Cancel button (selection count + instructions live on the right panel)."""

    def controls(self) -> None:
        _cancel_button(
            label="✖️ Cancel Merge",
            on_cancel=self.sm.cancel_merge,
            help="Clear the selection and return to idle",
        )


class RoutePlacingSidebarPanel(SidebarPanel):
    """route_placing: a Cancel button (the map clicks pick the start/end nodes)."""

    def controls(self) -> None:
        _cancel_button(
            label="✖️ Cancel Route",
            on_cancel=self.sm.cancel_route_placing,
            help="Discard the route and return to idle",
        )

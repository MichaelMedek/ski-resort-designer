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
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING, Literal

import streamlit as st

from skiresort_planner.constants import MapConfig, SlopeConfig, StyleConfig
from skiresort_planner.core.geo_calculator import GeoCalculator
from skiresort_planner.enum_utils import enum_eq
from skiresort_planner.model.message import (
    ImportActionMessage,
    ImportPlacingContextMessage,
    LiftActionMessage,
    LiftPlacingContextMessage,
    MergeActionMessage,
    MergePlacingContextMessage,
    PathActionMessage,
    PathBuildingContextMessage,
    SegmentWarningMessage,
)
from skiresort_planner.model.path_segment import SegmentKind
from skiresort_planner.model.resort_graph import ResortGraph
from skiresort_planner.ui.actions import (
    confirm_import_action,
    confirm_merge_action,
    rename_entity_action,
)
from skiresort_planner.ui.context import EntityKind, PlannerContext
from skiresort_planner.ui.infra import bump_map_version, reload_map, trigger_rerun
from skiresort_planner.ui.kind_spec import KIND_SPECS
from skiresort_planner.ui.state_machine import PlannerStateMachine

if TYPE_CHECKING:
    from skiresort_planner.model.lift import Lift
    from skiresort_planner.model.message import Message
    from skiresort_planner.model.proposed_path import ProposedPathSegment
    from skiresort_planner.model.road import Road
    from skiresort_planner.model.slope import Slope
    from skiresort_planner.ui.mode_registry import EntityKindSpec

logger = logging.getLogger(__name__)


def _commit_button_label(path: "ProposedPathSegment", *, continue_label: str, continue_help: str) -> tuple[str, str]:
    """(label, help) for the primary commit button, shared by slope + road panels.

    A connector (target is an existing node) finishes the entity; otherwise continue.
    """
    if path.is_connector and path.target_node_id:
        return f"🏁 Finish → {path.target_node_id}", f"Connect to {path.target_node_id} and finish"
    return continue_label, continue_help


# =============================================================================
# CONFIRMATION DIALOGS
# =============================================================================


@st.dialog("Confirm Delete")
def _confirm_delete_dialog(
    kind: EntityKind,
    entity_name: str,
    entity_id: str,
    delete_fn: Callable[[str], bool],
) -> None:
    """Show confirmation dialog before deleting a slope, road, or lift."""
    st.write(f"Are you sure you want to delete **{entity_name}**?")
    st.caption("This action can be undone using the Undo button.")

    col_yes, col_no = st.columns(2)
    with col_yes:
        if st.button("🗑️ Yes, Delete", type="primary", use_container_width=True):
            if delete_fn(entity_id):
                logger.info(f"Deleted {kind.value} {entity_name} (id={entity_id})")
            # Action functions handle state transition and map version bump
            trigger_rerun()
    with col_no:
        if st.button("✖️ Cancel", use_container_width=True):
            trigger_rerun()


@st.dialog("Rename")
def _rename_dialog(entity_id: str, current_name: str) -> None:
    """Prompt for a new name for a slope, road, or lift and apply it on Save."""
    new_name = st.text_input("Name", value=current_name)
    col_save, col_cancel = st.columns(2)
    with col_save:
        if st.button("💾 Save", type="primary", use_container_width=True):
            rename_entity_action(entity_id=entity_id, new_name=new_name)
            trigger_rerun()
    with col_cancel:
        if st.button("✖️ Cancel", use_container_width=True):
            trigger_rerun()


# =============================================================================
# SHARED HELPERS FOR INFO PANELS
# =============================================================================


def _action_button(
    label: str, *, key: str, help: str, type: Literal["primary", "secondary", "tertiary"] = "secondary"
) -> bool:
    """Render one right-panel action button. Returns True when clicked.

    width="stretch" so the button fills its container — full-width when stacked, or the column
    width when placed inside an st.columns cell (the entity-actions 2x2 grid).
    """
    return st.button(label, key=key, width="stretch", help=help, type=type)


def _render_3d_toggle_button(ctx: PlannerContext, graph: ResortGraph, kind: EntityKind, entity_id: str) -> None:
    """Render 3D/2D view toggle button. Calls reload_map() if button is clicked."""
    noun = kind.value
    if ctx.viewing.view_3d:
        if _action_button("🗺️ Return to 2D View", key=f"{noun}_2d_view", help="Return to the top-down 2D map"):
            logger.debug(f"Switching to 2D view from {noun} {entity_id}")
            ctx.viewing.disable_3d()
            # Reset pitch, bearing, and zoom to top-down 2D view
            ctx.map.pitch = MapConfig.DEFAULT_PITCH
            ctx.map.bearing = MapConfig.DEFAULT_BEARING
            ctx.map.zoom = MapConfig.DEFAULT_ZOOM
            # Update map center to entity center so we don't jump to stale position.
            # The viewed entity is guaranteed to exist (caller validated it). enum_eq is
            # reload-safe: EntityKind survives Streamlit reloads while the class is redefined.
            if enum_eq(a=kind, b=EntityKind.SLOPE) or enum_eq(a=kind, b=EntityKind.ROAD):
                # Both are segment groups → center on their segment endpoints.
                owner = graph.slopes[entity_id] if enum_eq(a=kind, b=EntityKind.SLOPE) else graph.roads[entity_id]
                lats, lons = [], []
                for seg_id in owner.segment_ids:
                    seg = graph.segments[seg_id]
                    lats.append(seg.points[0].lat)
                    lons.append(seg.points[0].lon)
                    lats.append(seg.points[-1].lat)
                    lons.append(seg.points[-1].lon)
                ctx.map.lat = sum(lats) / len(lats)
                ctx.map.lon = sum(lons) / len(lons)
            elif enum_eq(a=kind, b=EntityKind.LIFT):
                lift = graph.lifts[entity_id]
                start_node = graph.nodes[lift.start_node_id]
                end_node = graph.nodes[lift.end_node_id]
                ctx.map.lat = (start_node.lat + end_node.lat) / 2
                ctx.map.lon = (start_node.lon + end_node.lon) / 2
            else:
                raise ValueError(f"Unknown {kind=}")
            reload_map()  # Never returns - raises StopExecution
    elif _action_button("🏔️ View in 3D", key=f"{noun}_3d_view", help=f"View {noun} from the side with terrain"):
        logger.debug(f"Switching to 3D view for {noun} {entity_id}")
        ctx.viewing.enable_3d()
        reload_map()  # Never returns - raises StopExecution


def _render_entity_actions(
    sm: PlannerStateMachine,
    ctx: PlannerContext,
    graph: ResortGraph,
    kind: EntityKind,
    entity_id: str,
    entity: "Slope | Lift | Road",
    delete_fn: Callable[[str], bool],
) -> None:
    """Render the viewed-entity action buttons in a 2x2 grid: 3D toggle + Rename on top,
    Close + Delete on the bottom, so every entity panel reads identically.
    """
    noun = kind.value

    top_left, top_right = st.columns(2)
    bottom_left, bottom_right = st.columns(2)

    # Top-left: 3D / 2D toggle (own reload_map side effects live in the helper).
    with top_left:
        _render_3d_toggle_button(ctx=ctx, graph=graph, kind=kind, entity_id=entity_id)

    # Top-right: Rename.
    with top_right:
        if _action_button("✏️ Rename", key=f"rename_{noun}", help=f"Give this {noun} a custom name"):
            _rename_dialog(entity_id=entity_id, current_name=entity.name)

    # Bottom-left: Close.
    with bottom_left:
        if _action_button("✖️ Close", key=f"close_{noun}", help="Close this panel to start building again"):
            logger.debug(f"Closing {noun} panel for {entity_id}")
            ctx.viewing.disable_3d()
            # Reset pitch and bearing to top-down view (preserve zoom level)
            ctx.map.pitch = MapConfig.DEFAULT_PITCH
            ctx.map.bearing = MapConfig.DEFAULT_BEARING
            bump_map_version()
            # Uses close_panel event - SM resolves to appropriate transition
            # State transition triggers st.rerun() via listener - never returns
            sm.hide_info_panel()

    # Bottom-right: Delete.
    with bottom_right:
        if _action_button("🗑️ Delete", key=f"delete_{noun}", help=f"Permanently remove this {noun}"):
            _confirm_delete_dialog(
                kind=kind,
                entity_name=entity.name,
                entity_id=entity_id,
                delete_fn=delete_fn,
            )


# =============================================================================
# CONTROL PANEL BASE + PER-STATE PANELS
# =============================================================================


class ControlPanel(ABC):
    """Base for every right-side control panel — one per BuildState.

    A panel is described by exactly three parts, and `render()` is the fixed template that
    lays them out in order:

        1. `context_message()` — the BLUE "where you are" message (or None for explicitly none)
        2. `action_message()`  — the YELLOW "what to do now" message (or None)
        3. `buttons()`         — the interactive controls (stats, confirm/commit/cancel, entity actions)

    The three methods are abstract, so a new state's panel CANNOT silently forget its blue message,
    its yellow message, or its buttons — the class won't instantiate until all three exist. Returning
    None from a message method is the explicit "this panel has no such message" (not a forgotten one).
    """

    def __init__(
        self,
        sm: PlannerStateMachine,
        ctx: PlannerContext,
        graph: ResortGraph,
        on_commit: Callable[[int], None],
        on_cancel_connection: Callable[[], None],
    ) -> None:
        self.sm = sm
        self.ctx = ctx
        self.graph = graph
        self.on_commit = on_commit
        self.on_cancel_connection = on_cancel_connection

    @abstractmethod
    def context_message(self) -> "Message | None":
        """The blue context message, or None if this panel shows none."""

    @abstractmethod
    def action_message(self) -> "Message | None":
        """The yellow action instruction, or None if this panel shows none."""

    @abstractmethod
    def buttons(self) -> None:
        """Render the panel's interactive controls (stats, confirm/commit/cancel, entity actions)."""

    def render(self) -> None:
        """Fixed template: blue message → yellow message → buttons. Never overridden."""
        context = self.context_message()
        if context is not None:
            context.display()
        action = self.action_message()
        if action is not None:
            action.display()
        self.buttons()


class EmptyControlPanel(ControlPanel):
    """IDLE_READY: nothing to show — the map fills the space, no panel content."""

    def context_message(self) -> "Message | None":
        return None

    def action_message(self) -> "Message | None":
        return None

    def buttons(self) -> None:
        return None


class EntityInfoControlPanel(ControlPanel):
    """A viewed slope/road/lift: a stats block + the four entity actions (3D toggle, Rename, Close,
    Delete), no blue/yellow message. The per-kind pieces come from the injected ``EntityKindSpec``.
    """

    def __init__(
        self,
        sm: PlannerStateMachine,
        ctx: PlannerContext,
        graph: ResortGraph,
        on_commit: Callable[[int], None],
        on_cancel_connection: Callable[[], None],
        spec: "EntityKindSpec",
    ) -> None:
        super().__init__(
            sm=sm,
            ctx=ctx,
            graph=graph,
            on_commit=on_commit,
            on_cancel_connection=on_cancel_connection,
        )
        self.spec = spec

    def context_message(self) -> "Message | None":
        return None

    def action_message(self) -> "Message | None":
        return None

    def buttons(self) -> None:
        kind = self.spec.kind
        entity_id = self.spec.viewed_entity_id(self.ctx)
        if entity_id is None:
            raise ValueError(f"viewing.{kind.value}_id must be set when showing the {kind.value} panel")
        entity = self.spec.get_entity(graph=self.graph, entity_id=entity_id)
        if entity is None:
            raise ValueError(f"{kind.value.capitalize()} {entity_id} must exist when panel shows it")

        self.spec.render_stats(graph=self.graph, entity_id=entity_id)
        _render_entity_actions(
            sm=self.sm,
            ctx=self.ctx,
            graph=self.graph,
            kind=kind,
            entity_id=entity_id,
            entity=entity,
            delete_fn=self.spec.delete_action,
        )


class PathBuildingControlPanel(ControlPanel):
    """The build states for ANY path kind (slope or road): kind's progress/starting context
    message + the shared proposal browse/commit/cancel-custom UI.

    One class for every path kind. `buttons()` delegates to the kind-aware `PathSelectionPanel`
    and `context_message()` builds ONE unified `PathBuildingContextMessage` — both resolve the
    per-kind bits (icon, noun, build context) from the kind, so slope and road cannot drift.
    """

    def __init__(
        self,
        sm: PlannerStateMachine,
        ctx: PlannerContext,
        graph: ResortGraph,
        on_commit: Callable[[int], None],
        on_cancel_connection: Callable[[], None],
        kind: SegmentKind,
    ) -> None:
        super().__init__(sm, ctx, graph, on_commit, on_cancel_connection)
        self.kind = kind

    def context_message(self) -> "Message | None":
        # One unified message for every kind — no per-kind branch. The kind's build context stores
        # origin + segments uniformly; roads pass an empty difficulty_emoji, which drops the
        # ski-difficulty from the stats line.
        spec = KIND_SPECS[self.kind]
        build = self.ctx.build(self.kind)
        name = build.name or f"Unnamed {spec.display_noun}"
        segs = len(build.segments)

        if segs > 0:
            stats = self.graph.get_segment_stats(segment_ids=build.segments)
            emoji = (
                StyleConfig.DIFFICULTY_EMOJIS[stats["difficulty"]] if enum_eq(a=self.kind, b=SegmentKind.SLOPE) else ""
            )
            return PathBuildingContextMessage(
                icon=spec.icon,
                kind=self.kind,
                name=name,
                num_segments=segs,
                difficulty_emoji=emoji,
                total_drop_m=stats["total_drop"],
                total_length_m=stats["total_length"],
                avg_gradient_pct=stats["avg_gradient"],
                max_gradient_pct=stats["max_gradient"],
                start_elevation_m=stats["start_elev"],
                current_elevation_m=stats["current_elev"],
            )

        # Starting (no segments yet): show the origin. Prefer a stored start node; else the pending
        # start_location; else the current selection (fresh terrain click) — same shape for all kinds.
        start_node_id = build.start_node_id
        if start_node_id is None and build.endpoints:
            start_node_id = build.endpoints[-1]
        start_lat, start_lon = self.ctx.selection.lat, self.ctx.selection.lon
        if start_node_id is None and build.start_location is not None:
            start_lat, start_lon = build.start_location.lat, build.start_location.lon
        return PathBuildingContextMessage(
            icon=spec.icon,
            kind=self.kind,
            name=name,
            num_segments=0,
            start_node_id=start_node_id,
            start_lat=start_lat,
            start_lon=start_lon,
        )

    def action_message(self) -> "Message | None":
        # PathSelectionPanel owns the yellow action message (it depends on the selected proposal).
        return None

    def buttons(self) -> None:
        PathSelectionPanel(
            context=self.ctx,
            graph=self.graph,
            kind=self.kind,
            on_commit=self.on_commit,
            on_cancel_connection=self.on_cancel_connection,
        ).render()


class LiftPlacingControlPanel(ControlPanel):
    """LIFT_PLACING: bottom-station context + 'select top station' action (no buttons)."""

    def _start_elevation(self) -> float:
        if self.ctx.lift.start_node_id:
            node = self.graph.nodes.get(self.ctx.lift.start_node_id)
            return node.elevation if node else 0.0
        if self.ctx.lift.start_location:
            return self.ctx.lift.start_location.elevation
        raise RuntimeError("LiftPlacing state requires start_node_id or start_location to be set")

    def context_message(self) -> "Message | None":
        lift_icon = StyleConfig.LIFT_ICONS[self.ctx.lift.type]
        if self.ctx.lift.start_node_id:
            node = self.graph.nodes.get(self.ctx.lift.start_node_id)
            return LiftPlacingContextMessage(
                lift_type=self.ctx.lift.type,
                lift_icon=lift_icon,
                bottom_node_id=self.ctx.lift.start_node_id,
                bottom_elevation_m=node.elevation if node else 0.0,
            )
        if self.ctx.lift.start_location:
            loc = self.ctx.lift.start_location
            return LiftPlacingContextMessage(
                lift_type=self.ctx.lift.type,
                lift_icon=lift_icon,
                bottom_lat=loc.lat,
                bottom_lon=loc.lon,
                bottom_elevation_m=loc.elevation,
            )
        raise RuntimeError("LiftPlacing state requires start_node_id or start_location to be set")

    def action_message(self) -> "Message | None":
        return LiftActionMessage(is_awaiting_top=True, bottom_elevation_m=self._start_elevation())

    def buttons(self) -> None:
        return None


class ImportPlacingControlPanel(ControlPanel):
    """IMPORT_PLACING: box context + confirm-area action + Confirm Import button."""

    def context_message(self) -> "Message | None":
        center_lon = self.ctx.deferred.osm_import_center_lon
        center_lat = self.ctx.deferred.osm_import_center_lat
        if center_lon is None or center_lat is None:
            raise RuntimeError("ImportPlacing state requires a placed box center")
        return ImportPlacingContextMessage(
            center_lat=center_lat,
            center_lon=center_lon,
            half_width_km=self.ctx.deferred.osm_import_half_width_km,
        )

    def action_message(self) -> "Message | None":
        return ImportActionMessage()

    def buttons(self) -> None:
        if st.button("✅ Confirm Import", type="primary", width="stretch", help="Fetch and import this area from OSM"):
            logger.info("UI: Confirm Import clicked")
            confirm_import_action()


class MergePlacingControlPanel(ControlPanel):
    """MERGE_PLACING: selection context + merge action + Confirm Merge button.

    The Confirm button is disabled until at least two nodes are selected. It stays enabled even when
    the span exceeds the limit — confirm_merge_action shows a too-far toast and changes nothing, so
    the user learns why rather than staring at a silently dead button.
    """

    def _span_m(self) -> float:
        return self.graph.max_node_span_m(self.ctx.merge.node_ids)

    def context_message(self) -> "Message | None":
        return MergePlacingContextMessage(
            selected_count=len(self.ctx.merge.node_ids),
            span_m=self._span_m(),
        )

    def action_message(self) -> "Message | None":
        return MergeActionMessage(selected_count=len(self.ctx.merge.node_ids))

    def buttons(self) -> None:
        count = len(self.ctx.merge.node_ids)
        enough = count >= 2
        if st.button(
            "🔗 Confirm Merge",
            type="primary",
            width="stretch",
            disabled=not enough,
            help=("Select at least 2 nodes to merge" if not enough else "Collapse the selected nodes to their median"),
        ):
            logger.info(f"UI: Confirm Merge clicked for {count} nodes")
            confirm_merge_action()


def _render_proposal_browser(ctx: PlannerContext, *, key_prefix: str, noun: str) -> None:
    """Render the ◀ ▶ proposal browser shared by the slope and road panels.

    Cycles ctx.proposals.selected_idx and refreshes the map on each arrow. Only
    drawn when there is more than one proposal; a single proposal needs no browser.
    `noun` is the browsed unit ("paths" / "options"); `key_prefix` scopes the
    Streamlit widget keys so slope and road browsers don't collide.
    """
    num_paths = len(ctx.proposals.paths)
    if num_paths <= 1:
        return
    selected_idx = ctx.proposals.selected_idx if ctx.proposals.selected_idx is not None else 0

    col_prev, col_nav_label, col_next = st.columns([1, 2, 1])
    with col_prev:
        if st.button("◀", key=f"prev_{key_prefix}", width="stretch", help="Previous"):
            ctx.proposals.selected_idx = (selected_idx - 1) % num_paths
            reload_map()
    with col_nav_label:
        st.markdown(f"**◀ ▶ Browse {num_paths} {noun}**")
    with col_next:
        if st.button("▶", key=f"next_{key_prefix}", width="stretch", help="Next"):
            ctx.proposals.selected_idx = (selected_idx + 1) % num_paths
            reload_map()


# =============================================================================
# PATH SELECTION PANEL
# =============================================================================


class PathSelectionPanel:
    """Proposal browse + commit + cancel-custom, shared by EVERY path kind (slope and road).

    Kind-aware via the segment's own `kind`: the difficulty emoji/wording is used for slopes and
    omitted for roads (which carry no ski difficulty), and the commit/finish labels adapt. This is
    the single proposal UI — slope and road build panels both delegate here, so neither can drift.
    """

    def __init__(
        self,
        context: PlannerContext,
        graph: ResortGraph,
        kind: SegmentKind,
        on_commit: Callable[[int], None],
        on_cancel_connection: Callable[[], None],
    ) -> None:
        self.ctx = context
        self.graph = graph
        self.kind = kind
        self.on_commit = on_commit
        self.on_cancel_connection = on_cancel_connection

    def render(self) -> None:
        """Render the path selection panel."""
        noun = KIND_SPECS[self.kind].display_noun  # "Slope" / "Road"
        if not self.ctx.proposals.paths:
            # No proposals (e.g. a too-steep custom target). Show the message, but if we are
            # routing a custom target (force_mode) still offer the escape back to the fan.
            # A connector needs a real path, so an empty result is never a connector → always "Cancel Custom Path".
            # A too-steep result stashes the gentlest grade → surface the exact "why" here
            spec = KIND_SPECS[self.kind]
            PathActionMessage(
                kind=self.kind,
                is_custom_path=self.ctx.custom_connect.force_mode,
                too_steep_gentlest_pct=self.ctx.proposals.too_steep_gentlest_pct,
                too_steep_cap_pct=spec.max_grade_pct,
                too_steep_subject=spec.too_steep_subject,
                too_steep_two_sided=spec.too_steep_two_sided,
            ).display()
            if self.ctx.custom_connect.force_mode:
                self._render_cancel_connection(is_connector=False)
            else:
                # Fan-out that yielded nothing (e.g. all directions too steep): routing to a custom
                # target may still work. Make that discoverable now that there is no button.
                st.caption("🎯 Or click any point or node on the map to route a path there.")
            return

        num_paths = len(self.ctx.proposals.paths)
        # selected_idx is kept in range by generation (reset to 0) and browser nav (% num_paths);
        # None only before the first selection → show the first proposal.
        selected_idx = self.ctx.proposals.selected_idx if self.ctx.proposals.selected_idx is not None else 0

        path = self.ctx.proposals.paths[selected_idx]
        # Roads carry no ski difficulty (empty string) → no difficulty emoji; slopes look theirs up.
        emoji = StyleConfig.DIFFICULTY_EMOJIS[path.difficulty] if path.difficulty else ""
        is_connector = bool(path.is_connector and path.target_node_id)

        PathActionMessage(
            is_selecting_path=True,
            is_custom_path=self.ctx.custom_connect.force_mode,
            kind=self.kind,
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

        # Navigation arrows (keys scoped per kind so slope/road browsers never collide).
        _render_proposal_browser(ctx=self.ctx, key_prefix=f"{self.kind.value}_path", noun="paths")

        # Commit button (shared label logic across kinds)
        commit_label, commit_help = _commit_button_label(
            path,
            continue_label=f"✅ Commit This {noun}",
            continue_help="Add this segment and continue building",
        )
        if st.button(commit_label, type="primary", width="stretch", help=commit_help):
            logger.debug(f"UI: Commit button clicked for path {selected_idx}, is_connector={is_connector}")
            self.on_commit(selected_idx)

        # While showing custom-connect proposals, offer a way back to fan-out.
        if self.ctx.custom_connect.force_mode:
            self._render_cancel_connection(is_connector=is_connector)
        else:
            # Fan-out mode: the panel showed auto-generated proposals, but the user can
            # also aim anywhere. Make that discoverable now that there is no button.
            st.caption("🎯 Or click any point or node on the map to route a path there.")

    def _render_cancel_connection(self, *, is_connector: bool) -> None:
        """The escape back to fan-out during custom-connect. Label adapts: a connector routes to a
        node ("Cancel Connection"), a plain custom target ("Cancel Custom Path").
        """
        cancel_label = "✖️ Cancel Connection" if is_connector else "✖️ Cancel Custom Path"
        if st.button(cancel_label, width="stretch", help="Return to regular fan-out path proposals"):
            logger.debug(f"UI: {cancel_label} clicked")
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
        max_segment_gradient = slope.get_max_gradient(segments=self.graph.segments)

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
                help=f"Steepest {SlopeConfig.ROLLING_WINDOW_M}m section within any single segment - determines difficulty rating",
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


class RoadStatsPanel:
    """Renders statistics panel for a vehicle road."""

    def __init__(self, graph: ResortGraph) -> None:
        self.graph = graph

    def render(self, road_id: str) -> None:
        """Render statistics panel for the given road."""
        road = self.graph.roads.get(road_id)

        if not road:
            raise RuntimeError(
                f"Road '{road_id}' not found in graph.roads - "
                "state machine transitioned to viewing but road was deleted"
            )

        st.subheader(f"{StyleConfig.ROAD_ICON} {road.name}")

        total_length = road.get_total_length(segments=self.graph.segments)
        total_drop = road.get_total_drop(segments=self.graph.segments)
        max_gradient = road.get_max_gradient(segments=self.graph.segments)
        avg_gradient = (abs(total_drop) / total_length * 100) if total_length > 0 else 0

        # Road always has segments (validated on creation); index directly.
        first_seg = self.graph.segments[road.segment_ids[0]]
        last_seg = self.graph.segments[road.segment_ids[-1]]
        start_elev = first_seg.points[0].elevation
        end_elev = last_seg.points[-1].elevation

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Start Elevation", f"{start_elev:.0f}m")
            st.metric("Length", f"{total_length:.0f}m")
            st.metric("Average Gradient", f"{avg_gradient:.0f}%")
        with col2:
            st.metric("End Elevation", f"{end_elev:.0f}m")
            st.metric("Elevation Change", f"{end_elev - start_elev:+.0f}m")
            st.metric(
                "Steepest Section",
                f"{max_gradient:.0f}%",
                help=f"Steepest {SlopeConfig.ROLLING_WINDOW_M}m section within any single segment of the road",
            )

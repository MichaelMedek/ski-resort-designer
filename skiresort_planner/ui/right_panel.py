"""Right panel components for ski resort planner.

Centralizes all right-side control panel rendering:
- State dispatch to appropriate renderers
- PathSelectionPanel: Path browsing, selection, and commit
- PathStatsPanel: Slope/road statistics in viewing mode (kind-parameterized)
- LiftStatsPanel: Lift statistics in viewing mode

Design Principles:
- One renderer per state (no if-else chains)
- Raise exception for unknown states (fail-fast)
- Clear separation between slope mode and lift mode panels
"""

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Literal

import streamlit as st

from skiresort_planner.constants import LiftType, MapConfig, OSMImportMode, RoutePlannerConfig, SlopeConfig, StyleConfig
from skiresort_planner.core.geo_calculator import GeoCalculator
from skiresort_planner.core.terrain_analyzer import TerrainAnalyzer
from skiresort_planner.model.connectivity import CoreMembership
from skiresort_planner.model.message import (
    DisconnectedEntityMessage,
    ImportActionMessage,
    ImportSelectingContextMessage,
    LiftActionMessage,
    LiftPlacingContextMessage,
    NodeEditActionMessage,
    NodeEditContextMessage,
    NoReturnEntityMessage,
    PathActionMessage,
    PathBuildingContextMessage,
    RouteNoResultsMessage,
    RoutePlacingActionMessage,
    RoutePlacingContextMessage,
    RouteResultsContextMessage,
    SegmentWarningMessage,
)
from skiresort_planner.model.path_segment import SegmentKind
from skiresort_planner.model.resort_graph import ResortGraph
from skiresort_planner.ui.actions import (
    apply_lift_retype_action,
    confirm_import_action,
    confirm_merge_action,
    delete_direct_connection_action,
    delete_nodes_action,
    flythrough_keyframe_count,
    recenter_on_selected_route,
    rename_entity_action,
    route_plan_shown_routes,
)
from skiresort_planner.ui.context import EntityKind, PlannerContext
from skiresort_planner.ui.dialogs import ConfirmDialog, InputDialog
from skiresort_planner.ui.infra import bump_camera_epoch, bump_dedup_epoch, reload_map, trigger_rerun
from skiresort_planner.ui.kind_spec import KIND_SPECS
from skiresort_planner.ui.state_machine import PlannerStateMachine

if TYPE_CHECKING:
    from skiresort_planner.model.lift import Lift
    from skiresort_planner.model.message import Message
    from skiresort_planner.model.proposed_path import ProposedPathSegment
    from skiresort_planner.model.road import Road
    from skiresort_planner.model.routing import Route, ViewingGroup
    from skiresort_planner.model.segment_path import SegmentPath
    from skiresort_planner.model.slope import Slope
    from skiresort_planner.ui.mode_registry import EntityKindSpec

logger = logging.getLogger(__name__)


def route_legs(groups: "tuple[ViewingGroup, ...]") -> list[str]:
    """Render each viewing group as a readable leg: a lift shows its name; a folded slope run names up to
    ROUTE_STEP_SLOPE_PREVIEW of its slopes (colour-emoji per difficulty), then "…" if more. Consumes the
    model's `viewing_groups` so the panel legs and the flythrough units are provably the same.
    """
    legs: list[str] = []
    for group in groups:
        if group.is_lift:
            step = group.steps[0]
            legs.append(f"{StyleConfig.LIFT_ICONS[step.detail]} **{step.name}**")
        else:
            preview = group.steps[: RoutePlannerConfig.ROUTE_STEP_SLOPE_PREVIEW]
            named = " · ".join(f"{StyleConfig.DIFFICULTY_EMOJIS[s.detail]} {s.name}" for s in preview)
            tail = " …" if len(group.steps) > len(preview) else ""
            legs.append(f"{StyleConfig.SLOPE_ICON} {named}{tail}")
    return legs


def _render_connectivity_warnings(graph: ResortGraph, start_node_id: str, end_node_id: str, noun: str) -> None:
    """Show the connectivity warnings for a viewed slope/lift, below its stats (0, 1, or 2 of them).

    Single source for both the slope and lift panels; one SCC pass feeds both checks:
    - DISCONNECTED: the entity can't be reached from the core area at all.
    - one-way trip: after taking it, no slopes/lifts bring you back to ride it again.
    Both stay silent until a core exists (anti-false-alarm), so an early/tiny resort warns nothing.
    """
    labels = graph.strongly_connected_labels()
    core = graph.get_core_resort(labels=labels)
    if core is None:
        return  # no core yet → nothing to critique
    if graph.entity_membership(start_node_id=start_node_id, end_node_id=end_node_id, core=core) == (
        CoreMembership.DISCONNECTED
    ):
        DisconnectedEntityMessage(entity_noun=noun, core_lift_name=core.longest_lift_name).display()
    if not graph.can_loop_back(start_node_id=start_node_id, end_node_id=end_node_id, labels=labels):
        NoReturnEntityMessage(entity_noun=noun).display()


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


class _DeleteEntityDialog(ConfirmDialog):
    """Confirm deleting a slope, road, or lift."""

    def __init__(
        self,
        kind: EntityKind,
        entity_name: str,
        entity_id: str,
        delete_fn: Callable[[str], bool],
    ) -> None:
        self.kind = kind
        self.entity_name = entity_name
        self.entity_id = entity_id
        self.delete_fn = delete_fn

    @property
    def title(self) -> str:
        return "🗑️ Confirm Delete"

    def _body(self) -> None:
        st.write(f"Are you sure you want to delete **{self.entity_name}**?")
        st.caption("This action can be undone using the Undo button.")

    def _on_confirm(self) -> None:
        if self.delete_fn(self.entity_id):
            logger.info(f"Deleted {self.kind.value} {self.entity_name} (id={self.entity_id})")


class _RenameDialog(InputDialog):
    """Prompt for a new name for a slope, road, or lift and apply it on Save."""

    def __init__(self, entity_id: str, current_name: str) -> None:
        self.entity_id = entity_id
        self.current_name = current_name

    @property
    def title(self) -> str:
        return "✏️ Rename"

    def _input(self) -> str:
        return st.text_input("Name", value=self.current_name)

    def _on_save(self, value: str) -> None:
        rename_entity_action(entity_id=self.entity_id, new_name=value)


class _ChangeLiftTypeDialog(ConfirmDialog):
    """Confirm re-typing the viewed lift (guards accidental retypes). Confirm → retype, keep viewing;
    Cancel → keep the lift, close the view so the armed new type builds the next lift.
    """

    def __init__(self, lift_id: str, old_type: str, new_type: str, sm: PlannerStateMachine) -> None:
        self.lift_id = lift_id
        self.old_type = old_type
        self.new_type = new_type
        self.sm = sm

    @property
    def title(self) -> str:
        return "🚡 Change Lift Type?"

    def _body(self) -> None:
        old = f"{StyleConfig.LIFT_ICONS[self.old_type]} {StyleConfig.LIFT_DISPLAY_NAMES[self.old_type]}"
        new = f"{StyleConfig.LIFT_ICONS[self.new_type]} {StyleConfig.LIFT_DISPLAY_NAMES[self.new_type]}"
        st.write(f"Change this lift from **{old}** to **{new}**?")
        st.caption(f"Cancel keeps it a {old} and starts building a new {new} instead.")

    def _on_confirm(self) -> None:
        apply_lift_retype_action(lift_id=self.lift_id, lift_type=self.new_type)
        bump_camera_epoch()  # geometry changed → bare remount so the redraw takes

    def _on_cancel(self) -> None:
        # Lift untouched; leave the view so the armed new type (build_mode.mode, already set) drives
        # the next build. close_panel resolves to close_lift_panel from the lift view.
        self.sm.close_panel()  # type: ignore[attr-defined]  # dynamic python-statemachine event


def show_change_lift_type_dialog(lift_id: str, old_type: str, new_type: str, sm: PlannerStateMachine) -> None:
    """Public entrypoint for the confirm-retype dialog — the sidebar op triggers it through this (matching
    how right_panel's other public UI is called cross-module), keeping the dialog class private.
    """
    _ChangeLiftTypeDialog(lift_id=lift_id, old_type=old_type, new_type=new_type, sm=sm).show()


# =============================================================================
# SHARED HELPERS FOR INFO PANELS
# =============================================================================


def _action_button(
    label: str,
    *,
    key: str,
    help: str,
    type: Literal["primary", "secondary", "tertiary"] = "secondary",
    disabled: bool = False,
) -> bool:
    """Render one right-panel action button. Returns True when clicked.

    width="stretch" so the button fills its container — full-width when stacked, or the column
    width when placed inside an st.columns cell (the entity-actions 2x2 grid).
    """
    return st.button(label, key=key, width="stretch", help=help, type=type, disabled=disabled)


def _render_3d_toggle(
    ctx: PlannerContext,
    *,
    key_noun: str,
    noun: str,
    compute_2d_view: Callable[[], tuple[tuple[float, float], float]],
) -> None:
    """The single 3D/2D view toggle, shared by every viewing panel (slope/road/lift AND routes).

    The enable-3D branch and the button rendering are identical everywhere; the ONLY thing that
    differs is where the 2D map reframes when leaving 3D — supplied as `compute_2d_view` (returns the
    (center, adaptive-zoom) for the viewed element). The 3D camera fit itself lives in each state's
    view_state. Never returns when clicked (reload/rerun).
    """
    if ctx.viewing.view_3d:
        if _action_button("🗺️ Return to 2D View", key=f"{key_noun}_2d_view", help="Return to the top-down 2D map"):
            logger.debug(f"Switching to 2D view from {noun}")
            ctx.viewing.disable_3d()
            center, zoom = compute_2d_view()
            reload_map(center=center, zoom=zoom)  # Never returns
    elif _action_button("🏔️ View in 3D", key=f"{key_noun}_3d_view", help=f"View {noun} from the side with terrain"):
        logger.debug(f"Switching to 3D view for {noun}")
        ctx.viewing.enable_3d()
        bump_camera_epoch()  # 3D fit is computed in view_state; bare remount re-reads it
        trigger_rerun()  # Never returns - raises StopExecution


def _render_flythrough_controls(ctx: PlannerContext, noun: str) -> None:
    """Play/Stop flythrough row (call only in 3D) — shown ABOVE the action grid for every 3D element
    (slope/road/lift/route). Play left, Stop right; each disabled when not applicable so the row is stable.
    """
    playing = ctx.viewing.flythrough_active
    can_play = not playing and flythrough_keyframe_count() >= 2
    left, right = st.columns(2)
    with left:
        if _action_button(
            "▶️ Play", key="flythrough_play", help=f"Fly the camera along this {noun}", disabled=not can_play
        ):
            ctx.viewing.start_flythrough()
            trigger_rerun()
    with right:
        if _action_button("⏹️ Stop", key="flythrough_stop", help="Stop the flythrough", disabled=not playing):
            ctx.viewing.stop_flythrough()
            bump_camera_epoch()  # remount → re-read the 3D entry fit (view_3d stays on): reframe there, not 2D
            trigger_rerun()


def _render_entity_3d_toggle(ctx: PlannerContext, graph: ResortGraph, kind: EntityKind, entity_id: str) -> None:
    """Entity 3D toggle: shared toggle with the entity's own centre + adaptive zoom (same length source
    as the finish/click-to-view path) as the 2D reframe.
    """

    def entity_view() -> tuple[tuple[float, float], float]:
        # EntityKind is a StrEnum, so `==` is reload-safe.
        if kind in (EntityKind.SLOPE, EntityKind.ROAD):
            owner = graph.slopes[entity_id] if kind == EntityKind.SLOPE else graph.roads[entity_id]
            return owner.center(segments=graph.segments), MapConfig.zoom_for_span_m(
                span_m=owner.get_total_length(segments=graph.segments)
            )
        if kind == EntityKind.LIFT:
            lift = graph.lifts[entity_id]
            return lift.center(nodes=graph.nodes), MapConfig.zoom_for_span_m(
                span_m=lift.get_length_m(nodes=graph.nodes)
            )
        raise ValueError(f"Unknown {kind=}")

    _render_3d_toggle(ctx, key_noun=kind.value, noun=kind.value, compute_2d_view=entity_view)


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

    # Flythrough Play/Stop row sits ABOVE the 2x2 action grid (3D only).
    if ctx.viewing.view_3d:
        _render_flythrough_controls(ctx, noun)

    top_left, top_right = st.columns(2)
    bottom_left, bottom_right = st.columns(2)

    # Top-left: 3D / 2D toggle (own reload_map side effects live in the helper).
    with top_left:
        _render_entity_3d_toggle(ctx=ctx, graph=graph, kind=kind, entity_id=entity_id)

    # Top-right: Rename.
    with top_right:
        if _action_button("✏️ Rename", key=f"rename_{noun}", help=f"Give this {noun} a custom name"):
            _RenameDialog(entity_id=entity_id, current_name=entity.name).show()

    # Bottom-left: Close.
    with bottom_left:
        if _action_button("✖️ Close", key=f"close_{noun}", help="Close this panel to start building again"):
            logger.debug(f"Closing {noun} panel for {entity_id}")
            ctx.viewing.disable_3d()
            # Reset pitch and bearing to top-down view (preserve zoom level)
            ctx.map.pitch = MapConfig.DEFAULT_PITCH
            ctx.map.bearing = MapConfig.DEFAULT_BEARING
            bump_dedup_epoch()  # keep the user's pan (no recenter); a 3D→2D close re-frames via the view detector
            # close_panel event - SM resolves to the appropriate transition by current state
            # State transition triggers st.rerun() via listener - never returns
            sm.close_panel()  # type: ignore[attr-defined]  # dynamic python-statemachine event

    # Bottom-right: Delete.
    with bottom_right:
        if _action_button("🗑️ Delete", key=f"delete_{noun}", help=f"Permanently remove this {noun}"):
            _DeleteEntityDialog(
                kind=kind,
                entity_name=entity.name,
                entity_id=entity_id,
                delete_fn=delete_fn,
            ).show()


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
            emoji = StyleConfig.DIFFICULTY_EMOJIS[stats["difficulty"]] if self.kind == SegmentKind.SLOPE else ""
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
    """LIFT_PLACING: first-station context + 'select second station' action (no buttons)."""

    def context_message(self) -> "Message | None":
        lift_type = self.ctx.build_mode.mode  # single source of truth for the selected lift type
        if self.ctx.lift.first_node_id:
            node = self.graph.nodes[self.ctx.lift.first_node_id]
            return LiftPlacingContextMessage(
                lift_type=lift_type,
                first_node_id=self.ctx.lift.first_node_id,
                first_elevation_m=node.elevation,
            )
        if self.ctx.lift.first_location:
            loc = self.ctx.lift.first_location
            return LiftPlacingContextMessage(
                lift_type=lift_type,
                first_lat=loc.lat,
                first_lon=loc.lon,
                first_elevation_m=loc.elevation,
            )
        raise RuntimeError("LiftPlacing state requires first_node_id or first_location to be set")

    def action_message(self) -> "Message | None":
        return LiftActionMessage()

    def buttons(self) -> None:
        return None


class ImportSelectingControlPanel(ControlPanel):
    """IMPORT_SELECTING: box context + confirm-area action + Confirm Import button."""

    def context_message(self) -> "Message | None":
        center_lon = self.ctx.pending.osm_import_center_lon
        center_lat = self.ctx.pending.osm_import_center_lat
        if center_lon is None or center_lat is None:
            raise RuntimeError("ImportSelecting state requires a placed box center")
        return ImportSelectingContextMessage(
            center_lat=center_lat,
            center_lon=center_lon,
            half_width_km=self.ctx.pending.osm_import_half_width_km,
        )

    def action_message(self) -> "Message | None":
        return ImportActionMessage()

    def buttons(self) -> None:
        if st.button(
            f"{StyleConfig.LIFT_ICONS[LiftType.GONDOLA]}{StyleConfig.SLOPE_ICON} Import lifts + slopes",
            type="primary",
            width="stretch",
            help="Fetch this area from OSM and build a connected graph of lifts AND slopes",
        ):
            logger.debug("UI: Import lifts + slopes clicked")
            confirm_import_action(OSMImportMode.LIFTS_AND_SLOPES)
        if st.button(
            f"{StyleConfig.LIFT_ICONS[LiftType.GONDOLA]} Import lifts only",
            width="stretch",
            help="Fetch this area from OSM and import ONLY the lifts, exactly as mapped (fast)",
        ):
            logger.debug("UI: Import lifts only clicked")
            confirm_import_action(OSMImportMode.LIFTS_ONLY)


class NodeEditingControlPanel(ControlPanel):
    """NODE_EDITING: selection context + node actions (Confirm Merge / Delete / add on a path).

    The Confirm Merge button is disabled until at least two nodes are selected. It stays enabled even
    when the span exceeds the limit — confirm_merge_action shows a too-far toast and changes nothing, so
    the user learns why rather than staring at a silently dead button.
    """

    def _span_m(self) -> float:
        return self.graph.max_node_span_m(self.ctx.node_edit.node_ids)

    def context_message(self) -> "Message | None":
        return NodeEditContextMessage(
            selected_count=len(self.ctx.node_edit.node_ids),
            span_m=self._span_m(),
        )

    def action_message(self) -> "Message | None":
        return NodeEditActionMessage(selected_count=len(self.ctx.node_edit.node_ids))

    def buttons(self) -> None:
        count = len(self.ctx.node_edit.node_ids)
        enough = count >= 2
        if st.button(
            "🔗 Confirm Merge",
            type="primary",
            width="stretch",
            disabled=not enough,
            help=("Select at least 2 nodes to merge" if not enough else "Collapse the selected nodes to their median"),
        ):
            logger.debug(f"UI: Confirm Merge clicked for {count} nodes")
            confirm_merge_action()
        # Cut the segment between 2 adjacent nodes to SPLIT the path in two; adjacency checked on click.
        exactly_two = count == 2
        if st.button(
            "✂️ Delete Direct Connection",
            type="secondary",
            width="stretch",
            disabled=not exactly_two,
            help=(
                "Select exactly 2 adjacent nodes to delete the connection"
                if not exactly_two
                else "Cut all single segments between the 2 nodes — splits the path in two"
            ),
        ):
            logger.debug(f"UI: Delete Direct Connection clicked for {count} nodes")
            delete_direct_connection_action()
        # Delete is the LAST action (destructive, mirrors other panels); needs only 1+ node, checked on click.
        can_delete = count >= 1
        if st.button(
            "🗑️ Delete Node(s)",
            type="secondary",
            width="stretch",
            disabled=not can_delete,
            help=("Select at least 1 node to delete" if not can_delete else "Delete interior / end nodes of a path"),
        ):
            logger.debug(f"UI: Delete Node(s) clicked for {count} nodes")
            delete_nodes_action()
        # Discoverability hint, mirroring the path builder's "click any point" caption.
        st.caption("🎯 Or click any path on the map to add a node there.")


class RoutePlacingControlPanel(ControlPanel):
    """route_placing: pick the end node. Blue = where the start was placed (node + elevation);
    yellow = 'click the end node'. No buttons — the second node click completes the route and Cancel
    lives in the sidebar (mirrors LiftPlacingControlPanel exactly).
    """

    def context_message(self) -> "Message | None":
        # The start is always set on entry (the first node click sets it before start_route), so index
        # strictly — fail loud otherwise.
        start_id = self.ctx.route_plan.start_node_id
        assert start_id is not None, "route_placing entered without a start node"
        return RoutePlacingContextMessage(
            start_node_id=start_id, start_elevation_m=self.graph.nodes[start_id].elevation
        )

    def action_message(self) -> "Message | None":
        return RoutePlacingActionMessage()  # yellow: click the end node

    def buttons(self) -> None:
        return None  # the second node click completes the route; Cancel is in the sidebar


class RouteViewingControlPanel(ControlPanel):
    """idle_viewing_route: browse the precomputed routes for the selected difficulty cap. Blue =
    results summary (with the cap premise); yellow = no-route guidance. Buttons = the max-difficulty
    selector, the ◀▶ browser, the selected route's stats + colour legend, a 3D toggle, and Close.
    """

    def _shown(self) -> "list[Route]":
        return route_plan_shown_routes()

    def _hardest_cap(self) -> str:
        return SlopeConfig.DIFFICULTIES[-1]

    def context_message(self) -> "Message | None":
        routes = self._shown()
        if not routes:
            return None  # the yellow no-route message carries the whole story
        idx = self.ctx.route_plan.clamped_index(len(routes))
        return RouteResultsContextMessage(
            total=len(routes), selected_index=idx, difficulty_cap=self.ctx.route_plan.selected_cap
        )

    def action_message(self) -> "Message | None":
        if self._shown():
            return None
        # "Cap too strict" (a broader cap DOES have routes) vs "no route exists at all".
        from skiresort_planner.model.routing import routes_for_cap

        broader_has_routes = bool(routes_for_cap(self.ctx.route_plan.routes, max_difficulty=self._hardest_cap()))
        return RouteNoResultsMessage(cap_restrictive=broader_has_routes)

    def buttons(self) -> None:
        # The difficulty selector picks which PRECOMPUTED cap to show (honest — routes were computed
        # per cap, not filtered). Shown whenever any route was computed so the user can widen the cap.
        if self.ctx.route_plan.routes:
            self._render_cap_selector()
        routes = self._shown()
        if routes:
            self._render_route_browser(routes)
            self._render_route_stats(routes)
            # Play/Stop sits ABOVE the 3D toggle (3D only) so the row order matches every entity panel.
            if self.ctx.viewing.view_3d:
                _render_flythrough_controls(self.ctx, "route")
            self._render_3d_toggle()  # side view of the selected route floating above the pistes
        # Idiom: close the panel to leave. To plan again, re-enter Route Planner and pick two nodes.
        if st.button("✖️ Close", width="stretch", help="Close this panel to start building again"):
            self.sm.close_panel()  # type: ignore[attr-defined]  # dynamic python-statemachine event

    def _render_3d_toggle(self) -> None:
        """Route 3D toggle: the shared viewing toggle with the route's start/end midpoint + adaptive zoom
        (from the SELECTED route's length, matching the plan-time frame) as the 2D reframe.
        """

        def route_view() -> tuple[tuple[float, float], float]:
            rp = self.ctx.route_plan
            assert rp.start_node_id is not None and rp.end_node_id is not None, "route view without endpoints"
            a, b = self.graph.nodes[rp.start_node_id], self.graph.nodes[rp.end_node_id]
            routes = route_plan_shown_routes()
            route = routes[rp.clamped_index(len(routes))]
            return ((a.lon + b.lon) / 2, (a.lat + b.lat) / 2), MapConfig.zoom_for_span_m(
                span_m=route.total_slope_length_m
            )

        _render_3d_toggle(self.ctx, key_noun="route", noun="route", compute_2d_view=route_view)

    def _render_cap_selector(self) -> None:
        """Max-difficulty selector — picks which precomputed cap's routes to show (green→black). Each
        cap was computed over ONLY slopes up to that band, so this is an honest premise, not a filter.
        """
        rp = self.ctx.route_plan
        choice = st.select_slider(
            "Max difficulty",
            options=SlopeConfig.DIFFICULTIES,
            value=rp.selected_cap,
            format_func=str.capitalize,
            key="route_difficulty_cap",
            help="Show the best routes computed using only slopes up to this band.",
        )
        if choice != rp.selected_cap:
            rp.selected_cap = choice
            rp.selected_index = 0
            self.ctx.viewing.stop_flythrough()  # changing which route is shown must stop an active flythrough
            recenter_on_selected_route()  # reframe on the newly-shown route (adaptive zoom)
            trigger_rerun()

    def _render_route_browser(self, routes: "list[Route]") -> None:
        """◀ ▶ browser over the shown routes (mirrors _render_proposal_browser)."""
        if len(routes) <= 1:
            return
        idx = self.ctx.route_plan.clamped_index(len(routes))
        col_prev, col_label, col_next = st.columns([1, 2, 1])
        with col_prev:
            if st.button("◀", key="route_prev", width="stretch"):
                self.ctx.route_plan.selected_index = (idx - 1) % len(routes)
                self.ctx.viewing.stop_flythrough()  # browsing to another route must stop an active flythrough
                recenter_on_selected_route()  # reframe on the newly-shown route (adaptive zoom)
                trigger_rerun()
        with col_label:
            st.markdown(f"**Route {idx + 1} / {len(routes)}**")
        with col_next:
            if st.button("▶", key="route_next", width="stretch"):
                self.ctx.route_plan.selected_index = (idx + 1) % len(routes)
                self.ctx.viewing.stop_flythrough()  # browsing to another route must stop an active flythrough
                recenter_on_selected_route()  # reframe on the newly-shown route (adaptive zoom)
                trigger_rerun()

    def _render_route_stats(self, routes: "list[Route]") -> None:
        """Stats for the selected route: a colour-swatch legend naming what it's best for + the premise
        it was computed under, then totals and the ordered slope/lift steps.
        """
        idx = self.ctx.route_plan.clamped_index(len(routes))
        route = routes[idx]
        # Colour swatch matching the map line (keyed by the route's criterion, as drawn).
        rgba = route.color
        swatch = f"<span style='color:rgb({rgba[0]},{rgba[1]},{rgba[2]})'>⬤</span>"
        won = ", ".join(c.value.replace("_", " ") for c in route.criteria)
        # e.g. "⬤ Best for: shortest slope · max red" — the colour ties to the map, the cap is the premise.
        st.markdown(f"{swatch} 🏅 **Best for:** {won} · max **{route.difficulty_cap}**", unsafe_allow_html=True)
        if route.is_scenic:
            # A scenic tour visits every reachable lift under the cap (coverage is exact).
            st.caption(f"🎿 Visits all {route.scenic_lifts_visited} reachable lift(s), back to the start.")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Slope Length", f"{route.total_slope_length_m / 1000:.1f}km")
            st.metric("Slope Drop", f"{route.total_slope_drop_m:.0f}m")
        with col2:
            st.metric("Lifts", f"{route.lift_count}")
            emoji = StyleConfig.DIFFICULTY_EMOJIS[route.max_difficulty]
            st.metric("Max Difficulty", f"{emoji} {route.max_difficulty.capitalize()}")
        with st.expander("📋 Route Steps", expanded=False):
            for i, leg in enumerate(route_legs(route.viewing_groups), 1):
                st.markdown(f"{i}. {leg}")


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
            trigger_rerun()  # browsing only changes the highlight — redraw in place, no recenter
    with col_nav_label:
        st.markdown(f"**◀ ▶ Browse {num_paths} {noun}**")
    with col_next:
        if st.button("▶", key=f"next_{key_prefix}", width="stretch", help="Next"):
            ctx.proposals.selected_idx = (selected_idx + 1) % num_paths
            trigger_rerun()  # browsing only changes the highlight — redraw in place, no recenter


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
        # Committed-segment warnings surface here regardless of proposal state.
        self._render_committed_warnings()
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

    def _render_committed_warnings(self) -> None:
        """Surface any warnings on already-committed segments as ⚠️ messages (not on the plot)."""
        for seg_id in self.ctx.build(self.kind).segments:
            seg = self.graph.segments[seg_id]  # a committed build segment must exist — let it crash if not
            for warning in seg.warnings:
                SegmentWarningMessage(warning_text=str(warning)).display()

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


class StatsPanel(ABC):
    """Base for a viewed-entity statistics panel (slope/road via PathStatsPanel, lift via LiftStatsPanel).

    Mirrors ControlPanel: one abstract `render(entity_id)` the dispatcher calls; the entity lookup +
    "deleted mid-view" guard live in each subclass. Constructed with just the graph.
    """

    def __init__(self, graph: ResortGraph) -> None:
        self.graph = graph

    @abstractmethod
    def render(self, entity_id: str) -> None:
        """Render this kind's stats panel for the given entity id."""


class PathStatsPanel(StatsPanel):
    """Stats panel for a finished SegmentPath entity — one class for slope AND road (kind-parameterized).

    The kinds differ only in wording, all captured on the KindSpec (icon, shows_difficulty).
    Same single-source pattern as PathSelectionPanel, so kinds cannot drift.
    """

    def __init__(self, graph: ResortGraph, kind: SegmentKind) -> None:
        super().__init__(graph=graph)
        self.kind = kind

    def render(self, entity_id: str) -> None:
        spec = KIND_SPECS[self.kind]
        # Kind → its entity dict (data-driven, no if/else); a new kind adds one entry here.
        # Mapping (not dict) so the covariant value type accepts dict[str, Slope] / dict[str, Road].
        by_kind: Mapping[SegmentKind, Mapping[str, SegmentPath]] = {
            SegmentKind.SLOPE: self.graph.slopes,
            SegmentKind.ROAD: self.graph.roads,
        }
        owner = by_kind[self.kind].get(entity_id)
        if owner is None:
            raise RuntimeError(
                f"{self.kind.value} '{entity_id}' not found - state machine transitioned to viewing but it was deleted"
            )

        total_length = owner.get_total_length(segments=self.graph.segments)
        total_drop = owner.get_total_drop(segments=self.graph.segments)
        max_gradient = owner.get_max_gradient(segments=self.graph.segments)
        avg_gradient = (abs(total_drop) / total_length * 100) if total_length > 0 else 0

        # Committed entities always have segments-with-points; index directly (fail loud otherwise).
        first_seg = self.graph.segments[owner.segment_ids[0]]
        last_seg = self.graph.segments[owner.segment_ids[-1]]
        start_elev = first_seg.points[0].elevation
        end_elev = last_seg.points[-1].elevation

        st.subheader(f"{spec.icon} {owner.name}")
        if spec.shows_difficulty:
            difficulty = TerrainAnalyzer.classify_difficulty(slope_pct=max_gradient)
            st.markdown(f"**Difficulty:** {StyleConfig.DIFFICULTY_EMOJIS[difficulty]} {difficulty.capitalize()}")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Start Elevation", f"{start_elev:.0f}m")
            st.metric("Length", f"{total_length:.0f}m")
            st.metric("Average Gradient", f"{avg_gradient:.0f}%")
        with col2:
            st.metric("End Elevation", f"{end_elev:.0f}m")
            st.metric("Elevation Change", f"{abs(end_elev - start_elev):.0f}m")
            st.metric(
                "Steepest Section",
                f"{max_gradient:.0f}%",
                help=f"Steepest {SlopeConfig.ROLLING_WINDOW_M}m section within any single segment",
            )

        # Slopes participate in skiable connectivity; roads don't — warn only for slopes.
        if self.kind == SegmentKind.SLOPE:
            _render_connectivity_warnings(
                graph=self.graph, start_node_id=owner.start_node_id, end_node_id=owner.end_node_id, noun="slope"
            )

        with st.expander("📋 Segment Details", expanded=False):
            for i, seg_id in enumerate(owner.segment_ids, 1):
                seg = self.graph.segments[seg_id]
                if spec.shows_difficulty:
                    emoji = StyleConfig.DIFFICULTY_EMOJIS[seg.difficulty]
                    line = f"{i}. {emoji} **{seg.difficulty.capitalize()}** — {seg.length_m:.0f}m, {seg.max_slope_pct:.0f}% steepest, {seg.width_m:.0f}m wide"
                else:
                    line = f"{i}. {spec.icon} {seg.length_m:.0f}m, {seg.max_slope_pct:.0f}% steepest, {seg.width_m:.0f}m wide"
                st.markdown(line)
                for warning in seg.warnings:
                    SegmentWarningMessage(warning_text=str(warning)).display()


class LiftStatsPanel(StatsPanel):
    """Renders statistics panel for a lift."""

    def render(self, entity_id: str) -> None:
        """Render statistics panel for the given lift."""
        lift = self.graph.lifts.get(entity_id)

        if not lift:
            raise RuntimeError(
                f"Lift '{entity_id}' not found in graph.lifts - "
                "state machine transitioned to viewing but lift was deleted"
            )

        lift_icon = StyleConfig.LIFT_ICONS[lift.lift_type]
        lift_type_display = lift.lift_type.replace("_", " ").title()
        st.subheader(f"{lift_icon} {lift.name}")
        st.caption(f"Type: **{lift_type_display}** — *Use sidebar buttons to change*")

        start_node = self.graph.nodes[lift.start_node_id]
        end_node = self.graph.nodes[lift.end_node_id]

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

        _render_connectivity_warnings(
            graph=self.graph, start_node_id=lift.start_node_id, end_node_id=lift.end_node_id, noun="lift"
        )

"""Central registries for the app's per-state, per-mode, and per-kind behaviour.

The single dispatch hub at the top of the UI layer; exposes `render_control_panel` and
`dispatch_click` over three registries, each keyed to cover its axis exactly (asserted at import):
``BUILD_STATES`` (one BuildState per state-machine state), ``OPERATIONS`` (one BuilderOperation per
BuildMode), and ``ENTITY_KIND_SPECS`` (one EntityKindSpec per EntityKind). Abstract methods + the
bijection asserts make it impossible to add a state/mode/kind without fully registering it.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import plotly.graph_objects as go
import pydeck as pdk
import streamlit as st

from skiresort_planner.constants import StyleConfig
from skiresort_planner.core.dem_service import DEMService
from skiresort_planner.model.click_info import ClickInfo, MapClickType
from skiresort_planner.model.message import OutsideTerrainMessage
from skiresort_planner.model.path_segment import SegmentKind
from skiresort_planner.model.resort_graph import ResortGraph
from skiresort_planner.ui import actions, bottom_chart, click_handlers, right_panel, sidebar_panels
from skiresort_planner.ui.center_map import MapRenderer
from skiresort_planner.ui.context import BuildMode, EntityKind, PlannerContext
from skiresort_planner.ui.infra import bump_camera_epoch, trigger_rerun
from skiresort_planner.ui.kind_spec import KIND_SPECS
from skiresort_planner.ui.state_machine import PlannerStateMachine

if TYPE_CHECKING:
    from skiresort_planner.core.terrain_analyzer import TerrainAnalyzer
    from skiresort_planner.model.lift import Lift
    from skiresort_planner.model.road import Road
    from skiresort_planner.model.slope import Slope

logger = logging.getLogger(__name__)

# The camera framing: (lat, lon, bearing, zoom, pitch).
ViewState = tuple[float, float, float, int, float]

# A map-click handler: (click_info, elevation) -> None.
ClickHandler = Callable[[ClickInfo, float | None], None]


@dataclass(frozen=True)
class ProfileSpec:
    """An elevation-profile figure to render below the map, with its Streamlit chart key."""

    fig: go.Figure
    key: str


@dataclass(frozen=True)
class InfoBlock:
    """The sidebar info block for a state: an icon glyph, a label, and how-to bullets.

    Rendered uniformly as a collapsed expander titled `{icon} {label}`, with `bullets` as a
    markdown list. Each bullet is the text WITHOUT the leading `- `.
    """

    icon: str
    label: str
    bullets: list[str]


# =============================================================================
# BUILD STATE
# =============================================================================


class BuildState(ABC):
    """The complete UI surface the app renders while the state machine sits in one state.

    Subclasses are stateless descriptors keyed by ``state_key`` (== a state-machine state id). Every
    per-state surface is an abstract method, so a state cannot be registered with a part missing.
    """

    #: State-machine state id this BuildState is registered under.
    state_key: str

    @abstractmethod
    def control_panel(
        self,
        sm: PlannerStateMachine,
        ctx: PlannerContext,
        graph: ResortGraph,
        on_commit: Callable[[int], None],
        on_cancel_connection: Callable[[], None],
    ) -> right_panel.ControlPanel:
        """The right-side control panel for this state."""

    @abstractmethod
    def click_handler(self) -> ClickHandler:
        """The map-click handler ``(click_info, elevation) -> None`` for this state."""

    @abstractmethod
    def overlay_layers(
        self,
        ctx: PlannerContext,
        graph: ResortGraph,
        renderer: MapRenderer,
        terrain_analyzer: TerrainAnalyzer,
        dem: DEMService,
        *,
        use_3d: bool,
    ) -> list[pdk.Layer]:
        """Extra pydeck layers drawn on top of the base render (empty when the state has none)."""

    @abstractmethod
    def view_state(self, ctx: PlannerContext, graph: ResortGraph, *, use_3d: bool) -> ViewState:
        """The camera framing for this state: a 3D fit while viewing in 3D, else the stored 2D view."""

    @abstractmethod
    def bottom_profile(self, ctx: PlannerContext, graph: ResortGraph) -> ProfileSpec | None:
        """The elevation profile shown in the right column, or None when the state shows none."""

    @abstractmethod
    def merge_highlight_node_ids(self, ctx: PlannerContext) -> list[str] | None:
        """Node ids to draw as merge candidates, or None when not merging."""

    @abstractmethod
    def renders_custom_path(self, ctx: PlannerContext) -> bool:
        """Whether proposals draw as a single freehand path (roads + slope custom-connect)."""

    @abstractmethod
    def info_block(self, ctx: PlannerContext) -> InfoBlock:
        """The sidebar info block (icon + label + how-to bullets) shown while in this state."""

    @abstractmethod
    def sidebar_panel(
        self, sm: PlannerStateMachine, ctx: PlannerContext, graph: ResortGraph
    ) -> sidebar_panels.SidebarPanel:
        """The left-side sidebar control panel for this state (mirror of control_panel).

        Constructed by the hub and rendered fire-and-forget; its buttons call action functions
        directly, so no flags are returned.
        """

    @abstractmethod
    def blocks_build_buttons(self) -> bool:
        """Whether the build-mode selector buttons are disabled (True while building/placing)."""


def _stored_2d_view(ctx: PlannerContext) -> ViewState:
    """The stored top-down camera — used by every state that does not fit a 3D framing."""
    return (ctx.map.lat, ctx.map.lon, ctx.map.bearing, ctx.map.zoom, ctx.map.pitch)


class _IdleReadyState(BuildState):
    state_key = "idle_ready"

    def control_panel(
        self,
        sm: PlannerStateMachine,
        ctx: PlannerContext,
        graph: ResortGraph,
        on_commit: Callable[[int], None],
        on_cancel_connection: Callable[[], None],
    ) -> right_panel.ControlPanel:
        return right_panel.EmptyControlPanel(
            sm=sm, ctx=ctx, graph=graph, on_commit=on_commit, on_cancel_connection=on_cancel_connection
        )

    def click_handler(self) -> ClickHandler:
        return click_handlers.handle_idle_click

    def overlay_layers(
        self,
        ctx: PlannerContext,
        graph: ResortGraph,
        renderer: MapRenderer,
        terrain_analyzer: TerrainAnalyzer,
        dem: DEMService,
        *,
        use_3d: bool,
    ) -> list[pdk.Layer]:
        return []

    def view_state(self, ctx: PlannerContext, graph: ResortGraph, *, use_3d: bool) -> ViewState:
        return _stored_2d_view(ctx)

    def bottom_profile(self, ctx: PlannerContext, graph: ResortGraph) -> ProfileSpec | None:
        return None

    def merge_highlight_node_ids(self, ctx: PlannerContext) -> list[str] | None:
        return None

    def renders_custom_path(self, ctx: PlannerContext) -> bool:
        return False

    def info_block(self, ctx: PlannerContext) -> InfoBlock:
        # The lift glyph tracks the first lift button so the icon matches whatever lift renders first.
        return InfoBlock(
            icon=f"{StyleConfig.SLOPE_ICON}{StyleConfig.ROAD_ICON}{StyleConfig.LIFT_ICONS[BuildMode.LIFT_TYPES[0]]}",
            label="Ready to Build",
            bullets=[
                "🔘 Select **Slope**, **Road** or **Lift** type below",
                f"{StyleConfig.BUILDING_ICON} Click terrain/node → start building",
                f"{StyleConfig.VIEWING_ICON} Click existing slope/road/lift → view stats",
                "🛠️ Or use **Import** / **Node Merge** utilities below",
            ],
        )

    def sidebar_panel(
        self, sm: PlannerStateMachine, ctx: PlannerContext, graph: ResortGraph
    ) -> sidebar_panels.SidebarPanel:
        return sidebar_panels.IdleSidebarPanel(sm=sm, ctx=ctx, graph=graph)

    def blocks_build_buttons(self) -> bool:
        return False


class _EntityViewingState(BuildState):
    """Shared surface for the three viewing states (slope/road/lift): a 3D fit when viewing in 3D, an
    info control panel, and the entity's finished profile below the map. Subclasses bind the kind.
    """

    kind: EntityKind

    def _fit_3d_view(self, graph: ResortGraph, entity_id: str) -> ViewState:
        """The 3D camera fit for this kind's viewed entity. Bound per-kind by the subclass."""
        raise NotImplementedError

    def control_panel(
        self,
        sm: PlannerStateMachine,
        ctx: PlannerContext,
        graph: ResortGraph,
        on_commit: Callable[[int], None],
        on_cancel_connection: Callable[[], None],
    ) -> right_panel.ControlPanel:
        return right_panel.EntityInfoControlPanel(
            sm=sm,
            ctx=ctx,
            graph=graph,
            on_commit=on_commit,
            on_cancel_connection=on_cancel_connection,
            spec=ENTITY_KIND_SPECS[self.kind],
        )

    def click_handler(self) -> ClickHandler:
        return click_handlers.handle_idle_click

    def overlay_layers(
        self,
        ctx: PlannerContext,
        graph: ResortGraph,
        renderer: MapRenderer,
        terrain_analyzer: TerrainAnalyzer,
        dem: DEMService,
        *,
        use_3d: bool,
    ) -> list[pdk.Layer]:
        return []

    def view_state(self, ctx: PlannerContext, graph: ResortGraph, *, use_3d: bool) -> ViewState:
        entity_id = ENTITY_KIND_SPECS[self.kind].viewed_entity_id(ctx)
        if use_3d and entity_id is not None:
            return self._fit_3d_view(graph=graph, entity_id=entity_id)
        return _stored_2d_view(ctx)

    def bottom_profile(self, ctx: PlannerContext, graph: ResortGraph) -> ProfileSpec | None:
        entity_id = ENTITY_KIND_SPECS[self.kind].viewed_entity_id(ctx)
        if entity_id is None:
            return None
        fig = bottom_chart.render_viewing_profile(kind=self.kind, entity_id=entity_id, graph=graph)
        return ProfileSpec(fig=fig, key="viewing_profile")

    def merge_highlight_node_ids(self, ctx: PlannerContext) -> list[str] | None:
        return None

    def renders_custom_path(self, ctx: PlannerContext) -> bool:
        return False

    def info_block(self, ctx: PlannerContext) -> InfoBlock:
        # Same bullets for every viewed kind; only lifts add the change-type line.
        # EntityKind is a StrEnum, so `==` is reload-safe (survives Streamlit's class redefinition).
        bullets = ["🔄 Use lift buttons to change type"] if self.kind == EntityKind.LIFT else []
        bullets.append("✖️ **Close** the right panel to return")
        bullets.append(f"{StyleConfig.BUILDING_ICON} Click terrain/node → new {self.kind.value}")
        return InfoBlock(
            icon=StyleConfig.VIEWING_ICON, label=f"Viewing {self.kind.value.capitalize()}", bullets=bullets
        )

    def sidebar_panel(
        self, sm: PlannerStateMachine, ctx: PlannerContext, graph: ResortGraph
    ) -> sidebar_panels.SidebarPanel:
        return sidebar_panels.ViewingSidebarPanel(sm=sm, ctx=ctx, graph=graph)

    def blocks_build_buttons(self) -> bool:
        return False


class _IdleViewingSlopeState(_EntityViewingState):
    state_key = "idle_viewing_slope"
    kind = EntityKind.SLOPE

    def _fit_3d_view(self, graph: ResortGraph, entity_id: str) -> ViewState:
        return MapRenderer.calculate_3d_view_for_slope(graph=graph, slope_id=entity_id)


class _IdleViewingRoadState(_EntityViewingState):
    state_key = "idle_viewing_road"
    kind = EntityKind.ROAD

    def _fit_3d_view(self, graph: ResortGraph, entity_id: str) -> ViewState:
        return MapRenderer.calculate_3d_view_for_road(graph=graph, road_id=entity_id)


class _IdleViewingLiftState(_EntityViewingState):
    state_key = "idle_viewing_lift"
    kind = EntityKind.LIFT

    def _fit_3d_view(self, graph: ResortGraph, entity_id: str) -> ViewState:
        return MapRenderer.calculate_3d_view_for_lift(graph=graph, lift_id=entity_id)


class _PathBuildingState(BuildState):
    """Build states for ANY path kind (slope/road): the *_starting / *_building / *_custom_path trio.

    One kind-parameterized class (mirroring the path control/sidebar panels) so slope and road can't
    drift. Draws fall-line arrows at the origin, a downhill arrow while custom-connecting, and the
    in-build elevation profile once ≥1 segment is committed.
    """

    def __init__(self, state_key: str, kind: SegmentKind) -> None:
        self.state_key = state_key
        self.kind = kind

    def control_panel(
        self,
        sm: PlannerStateMachine,
        ctx: PlannerContext,
        graph: ResortGraph,
        on_commit: Callable[[int], None],
        on_cancel_connection: Callable[[], None],
    ) -> right_panel.ControlPanel:
        return right_panel.PathBuildingControlPanel(
            sm=sm,
            ctx=ctx,
            graph=graph,
            on_commit=on_commit,
            on_cancel_connection=on_cancel_connection,
            kind=self.kind,
        )

    def click_handler(self) -> ClickHandler:
        return click_handlers.handle_path_building_click

    def overlay_layers(
        self,
        ctx: PlannerContext,
        graph: ResortGraph,
        renderer: MapRenderer,
        terrain_analyzer: TerrainAnalyzer,
        dem: DEMService,
        *,
        use_3d: bool,
    ) -> list[pdk.Layer]:
        layers: list[pdk.Layer] = []
        sel = ctx.selection
        if sel.lon is not None and sel.lat is not None and sel.elevation is not None:
            orientation = terrain_analyzer.get_orientation(lon=sel.lon, lat=sel.lat)
            if orientation:
                layers.extend(
                    renderer.create_orientation_arrows_layers(
                        lat=sel.lat, lon=sel.lon, elevation=sel.elevation, orientation=orientation, use_3d=use_3d
                    )
                )
        if ctx.custom_connect.force_mode and ctx.custom_connect.start_node:
            start_node = graph.nodes[ctx.custom_connect.start_node]  # live node (never dangling)
            gradient = terrain_analyzer.compute_gradient(lon=start_node.lon, lat=start_node.lat)
            layers.append(
                renderer.create_direction_arrow_layer(
                    start_lat=start_node.lat,
                    start_lon=start_node.lon,
                    bearing_deg=gradient.bearing_deg,
                    direction="downhill",
                    use_3d=use_3d,
                )
            )
        return layers

    def view_state(self, ctx: PlannerContext, graph: ResortGraph, *, use_3d: bool) -> ViewState:
        return _stored_2d_view(ctx)

    def bottom_profile(self, ctx: PlannerContext, graph: ResortGraph) -> ProfileSpec | None:
        build = ctx.build(self.kind)
        if not build.segments:
            return None
        fig = bottom_chart.render_building_profile(
            building_segments=build.segments,
            building_name=build.name,
            graph=graph,
        )
        # Key is scoped per kind so slope and road profile charts never collide.
        return ProfileSpec(fig=fig, key=f"combined_{self.kind.value}_profile")

    def merge_highlight_node_ids(self, ctx: PlannerContext) -> list[str] | None:
        return None

    def renders_custom_path(self, ctx: PlannerContext) -> bool:
        return ctx.custom_connect.force_mode

    def info_block(self, ctx: PlannerContext) -> InfoBlock:
        return InfoBlock(
            icon=StyleConfig.BUILDING_ICON,
            label=f"Building {KIND_SPECS[self.kind].display_noun}…",
            bullets=["⏳ Complete or cancel current build to change type"],
        )

    def sidebar_panel(
        self, sm: PlannerStateMachine, ctx: PlannerContext, graph: ResortGraph
    ) -> sidebar_panels.SidebarPanel:
        return sidebar_panels.PathBuildSidebarPanel(sm=sm, ctx=ctx, graph=graph, kind=self.kind)

    def blocks_build_buttons(self) -> bool:
        return True


class _LiftPlacingState(BuildState):
    state_key = "lift_placing"

    def control_panel(
        self,
        sm: PlannerStateMachine,
        ctx: PlannerContext,
        graph: ResortGraph,
        on_commit: Callable[[int], None],
        on_cancel_connection: Callable[[], None],
    ) -> right_panel.ControlPanel:
        return right_panel.LiftPlacingControlPanel(
            sm=sm, ctx=ctx, graph=graph, on_commit=on_commit, on_cancel_connection=on_cancel_connection
        )

    def click_handler(self) -> ClickHandler:
        return click_handlers.handle_lift_placing_click

    def overlay_layers(
        self,
        ctx: PlannerContext,
        graph: ResortGraph,
        renderer: MapRenderer,
        terrain_analyzer: TerrainAnalyzer,
        dem: DEMService,
        *,
        use_3d: bool,
    ) -> list[pdk.Layer]:
        if ctx.lift.first_node_id:
            node = graph.nodes[ctx.lift.first_node_id]  # live node (never dangling)
            lat, lon, elevation = node.lat, node.lon, node.elevation
        elif ctx.lift.first_location:
            loc = ctx.lift.first_location
            lat, lon, elevation = loc.lat, loc.lon, loc.elevation
        else:
            return []
        gradient = terrain_analyzer.compute_gradient(lon=lon, lat=lat)
        return renderer.create_pending_lift_marker_layers(
            lat=lat, lon=lon, elevation=elevation, fall_line_bearing=gradient.bearing_deg, use_3d=use_3d
        )

    def view_state(self, ctx: PlannerContext, graph: ResortGraph, *, use_3d: bool) -> ViewState:
        return _stored_2d_view(ctx)

    def bottom_profile(self, ctx: PlannerContext, graph: ResortGraph) -> ProfileSpec | None:
        return None

    def merge_highlight_node_ids(self, ctx: PlannerContext) -> list[str] | None:
        return None

    def renders_custom_path(self, ctx: PlannerContext) -> bool:
        return False

    def info_block(self, ctx: PlannerContext) -> InfoBlock:
        return InfoBlock(
            icon=StyleConfig.BUILDING_ICON,
            label="Placing Lift…",
            bullets=["⏳ Complete or cancel current build to change type"],
        )

    def sidebar_panel(
        self, sm: PlannerStateMachine, ctx: PlannerContext, graph: ResortGraph
    ) -> sidebar_panels.SidebarPanel:
        return sidebar_panels.LiftSidebarPanel(sm=sm, ctx=ctx, graph=graph)

    def blocks_build_buttons(self) -> bool:
        return True


class _ImportPlacingState(BuildState):
    state_key = "import_placing"

    def control_panel(
        self,
        sm: PlannerStateMachine,
        ctx: PlannerContext,
        graph: ResortGraph,
        on_commit: Callable[[int], None],
        on_cancel_connection: Callable[[], None],
    ) -> right_panel.ControlPanel:
        return right_panel.ImportPlacingControlPanel(
            sm=sm, ctx=ctx, graph=graph, on_commit=on_commit, on_cancel_connection=on_cancel_connection
        )

    def click_handler(self) -> ClickHandler:
        return click_handlers.handle_import_placing_click

    def overlay_layers(
        self,
        ctx: PlannerContext,
        graph: ResortGraph,
        renderer: MapRenderer,
        terrain_analyzer: TerrainAnalyzer,
        dem: DEMService,
        *,
        use_3d: bool,
    ) -> list[pdk.Layer]:
        center_lon = ctx.pending.osm_import_center_lon
        center_lat = ctx.pending.osm_import_center_lat
        if center_lon is None or center_lat is None:
            return []
        center_elev = dem.get_elevation(lon=center_lon, lat=center_lat) or 0.0
        return renderer.create_import_bbox_layers(
            center_lon=center_lon,
            center_lat=center_lat,
            half_width_m=ctx.pending.osm_import_half_width_km * 1000.0,
            elevation=center_elev,
            use_3d=use_3d,
        )

    def view_state(self, ctx: PlannerContext, graph: ResortGraph, *, use_3d: bool) -> ViewState:
        return _stored_2d_view(ctx)

    def bottom_profile(self, ctx: PlannerContext, graph: ResortGraph) -> ProfileSpec | None:
        return None

    def merge_highlight_node_ids(self, ctx: PlannerContext) -> list[str] | None:
        return None

    def renders_custom_path(self, ctx: PlannerContext) -> bool:
        return False

    def info_block(self, ctx: PlannerContext) -> InfoBlock:
        return InfoBlock(
            icon=StyleConfig.BUILDING_ICON,
            label="Importing Area…",
            bullets=["⏳ Complete or cancel current build to change type"],
        )

    def sidebar_panel(
        self, sm: PlannerStateMachine, ctx: PlannerContext, graph: ResortGraph
    ) -> sidebar_panels.SidebarPanel:
        return sidebar_panels.ImportSidebarPanel(sm=sm, ctx=ctx, graph=graph)

    def blocks_build_buttons(self) -> bool:
        return True


class _MergePlacingState(BuildState):
    state_key = "merge_placing"

    def control_panel(
        self,
        sm: PlannerStateMachine,
        ctx: PlannerContext,
        graph: ResortGraph,
        on_commit: Callable[[int], None],
        on_cancel_connection: Callable[[], None],
    ) -> right_panel.ControlPanel:
        return right_panel.MergePlacingControlPanel(
            sm=sm, ctx=ctx, graph=graph, on_commit=on_commit, on_cancel_connection=on_cancel_connection
        )

    def click_handler(self) -> ClickHandler:
        return click_handlers.handle_merge_placing_click

    def overlay_layers(
        self,
        ctx: PlannerContext,
        graph: ResortGraph,
        renderer: MapRenderer,
        terrain_analyzer: TerrainAnalyzer,
        dem: DEMService,
        *,
        use_3d: bool,
    ) -> list[pdk.Layer]:
        return []

    def view_state(self, ctx: PlannerContext, graph: ResortGraph, *, use_3d: bool) -> ViewState:
        return _stored_2d_view(ctx)

    def bottom_profile(self, ctx: PlannerContext, graph: ResortGraph) -> ProfileSpec | None:
        return None

    def merge_highlight_node_ids(self, ctx: PlannerContext) -> list[str] | None:
        return ctx.merge.node_ids

    def renders_custom_path(self, ctx: PlannerContext) -> bool:
        return False

    def info_block(self, ctx: PlannerContext) -> InfoBlock:
        return InfoBlock(
            icon=StyleConfig.BUILDING_ICON,
            label="Merging Nodes…",
            bullets=["⏳ Complete or cancel current build to change type"],
        )

    def sidebar_panel(
        self, sm: PlannerStateMachine, ctx: PlannerContext, graph: ResortGraph
    ) -> sidebar_panels.SidebarPanel:
        return sidebar_panels.MergeSidebarPanel(sm=sm, ctx=ctx, graph=graph)

    def blocks_build_buttons(self) -> bool:
        return True


class _RoutePlacingState(BuildState):
    """Picking the route's start/end nodes. Highlights the picked start node; no overlay yet."""

    state_key = "route_placing"

    def control_panel(
        self,
        sm: PlannerStateMachine,
        ctx: PlannerContext,
        graph: ResortGraph,
        on_commit: Callable[[int], None],
        on_cancel_connection: Callable[[], None],
    ) -> right_panel.ControlPanel:
        return right_panel.RoutePlacingControlPanel(
            sm=sm, ctx=ctx, graph=graph, on_commit=on_commit, on_cancel_connection=on_cancel_connection
        )

    def click_handler(self) -> ClickHandler:
        return click_handlers.handle_route_placing_click

    def overlay_layers(
        self,
        ctx: PlannerContext,
        graph: ResortGraph,
        renderer: MapRenderer,
        terrain_analyzer: TerrainAnalyzer,
        dem: DEMService,
        *,
        use_3d: bool,
    ) -> list[pdk.Layer]:
        return []

    def view_state(self, ctx: PlannerContext, graph: ResortGraph, *, use_3d: bool) -> ViewState:
        return _stored_2d_view(ctx)

    def bottom_profile(self, ctx: PlannerContext, graph: ResortGraph) -> ProfileSpec | None:
        return None

    def merge_highlight_node_ids(self, ctx: PlannerContext) -> list[str] | None:
        # Reuse the node-highlight channel to show the picked start node while awaiting the end.
        start = ctx.route_plan.start_node_id
        return [start] if start is not None else None

    def renders_custom_path(self, ctx: PlannerContext) -> bool:
        return False

    def info_block(self, ctx: PlannerContext) -> InfoBlock:
        return InfoBlock(
            icon=StyleConfig.BUILDING_ICON,
            label="Planning Route…",
            bullets=["⏳ Complete or cancel current build to change type"],
        )

    def sidebar_panel(
        self, sm: PlannerStateMachine, ctx: PlannerContext, graph: ResortGraph
    ) -> sidebar_panels.SidebarPanel:
        return sidebar_panels.RoutePlacingSidebarPanel(sm=sm, ctx=ctx, graph=graph)

    def blocks_build_buttons(self) -> bool:
        return True


class _IdleViewingRouteState(BuildState):
    """Browsing the computed routes: overlays the filtered routes on the map, filters in the sidebar."""

    state_key = "idle_viewing_route"

    def control_panel(
        self,
        sm: PlannerStateMachine,
        ctx: PlannerContext,
        graph: ResortGraph,
        on_commit: Callable[[int], None],
        on_cancel_connection: Callable[[], None],
    ) -> right_panel.ControlPanel:
        return right_panel.RouteViewingControlPanel(
            sm=sm, ctx=ctx, graph=graph, on_commit=on_commit, on_cancel_connection=on_cancel_connection
        )

    def click_handler(self) -> ClickHandler:
        return click_handlers.handle_idle_click

    def overlay_layers(
        self,
        ctx: PlannerContext,
        graph: ResortGraph,
        renderer: MapRenderer,
        terrain_analyzer: TerrainAnalyzer,
        dem: DEMService,
        *,
        use_3d: bool,
    ) -> list[pdk.Layer]:
        return renderer.create_route_layers(routes=actions.route_plan_filtered_routes(), use_3d=use_3d)

    def view_state(self, ctx: PlannerContext, graph: ResortGraph, *, use_3d: bool) -> ViewState:
        return _stored_2d_view(ctx)

    def bottom_profile(self, ctx: PlannerContext, graph: ResortGraph) -> ProfileSpec | None:
        return None

    def merge_highlight_node_ids(self, ctx: PlannerContext) -> list[str] | None:
        return None

    def renders_custom_path(self, ctx: PlannerContext) -> bool:
        return False

    def info_block(self, ctx: PlannerContext) -> InfoBlock:
        return InfoBlock(
            icon=StyleConfig.VIEWING_ICON,
            label="Viewing Routes…",
            bullets=["⏳ Complete or cancel current build to change type"],
        )

    def sidebar_panel(
        self, sm: PlannerStateMachine, ctx: PlannerContext, graph: ResortGraph
    ) -> sidebar_panels.SidebarPanel:
        return sidebar_panels.RouteViewingSidebarPanel(sm=sm, ctx=ctx, graph=graph)

    def blocks_build_buttons(self) -> bool:
        return False  # a viewing state: keep build buttons live so a click switches mode


_BUILD_STATE_LIST: list[BuildState] = [
    _IdleReadyState(),
    _IdleViewingSlopeState(),
    _IdleViewingRoadState(),
    _IdleViewingLiftState(),
    _IdleViewingRouteState(),
    _PathBuildingState("slope_starting", SegmentKind.SLOPE),
    _PathBuildingState("slope_building", SegmentKind.SLOPE),
    _PathBuildingState("slope_custom_path", SegmentKind.SLOPE),
    _LiftPlacingState(),
    _ImportPlacingState(),
    _MergePlacingState(),
    _RoutePlacingState(),
    _PathBuildingState("road_starting", SegmentKind.ROAD),
    _PathBuildingState("road_building", SegmentKind.ROAD),
    _PathBuildingState("road_custom_path", SegmentKind.ROAD),
]

BUILD_STATES: dict[str, BuildState] = {bs.state_key: bs for bs in _BUILD_STATE_LIST}


# =============================================================================
# BUILDER OPERATION
# =============================================================================


class OperationGroup:
    """Sidebar grouping: the real builders vs the whole-resort utilities (separated by a divider)."""

    BUILDER = "builder"
    UTILITY = "utility"


class BuilderOperation(ABC):
    """One sidebar build-type button: its group, its greyout rule, and its select action."""

    #: BuildMode value this operation is registered under.
    mode: str
    #: OperationGroup.BUILDER or OperationGroup.UTILITY.
    group: str

    def enabled(self, sm: PlannerStateMachine) -> bool:
        """Whether the button is clickable — ONE rule for every button.

        Enabled only while idle (never mid-build/placement), and then only in idle_ready or while
        viewing this button's OWN kind (so a kind-builder can switch straight into rebuilding/
        re-typing that kind). Utilities have no own kind, so they are idle_ready-only.
        """
        if not _idle_not_building(sm):
            return False
        return sm.is_idle_ready or self._enabled_while_viewing_own_kind(sm)

    @abstractmethod
    def _enabled_while_viewing_own_kind(self, sm: PlannerStateMachine) -> bool:
        """Whether the current viewing state is THIS button's own kind.

        The kind-builders name their own viewing state; utilities have no own kind and return False deliberately.
        """

    @property
    @abstractmethod
    def first_instruction(self) -> str:
        """One-line hint shown in idle: what the FIRST map click does in this mode."""

    def on_select(self, ctx: PlannerContext, sm: PlannerStateMachine) -> None:
        """Highlight this mode — the invariant EVERY builder button shares.

        A pure UI pre-selection (state stays idle_ready); mode start happens on the first map click.
        Uses a plain rerun — no camera_epoch bump — to avoid a needless deck.gl remount.
        """
        ctx.build_mode.mode = self.mode
        trigger_rerun()


def _idle_not_building(sm: PlannerStateMachine) -> bool:
    """True while idle and not mid-build/placement (all buttons disable during a build/placement).

    idle_viewing_route (a viewing state) is intentionally NOT excluded — build buttons stay live there so
    a click leaves the route view and starts building, exactly like the slope/lift/road viewers.
    """
    return not (
        sm.is_any_path_state or sm.is_lift_placing or sm.is_import_placing or sm.is_merge_placing or sm.is_route_placing
    )


class _SlopeOperation(BuilderOperation):
    mode = BuildMode.SLOPE
    group = OperationGroup.BUILDER
    first_instruction = "🗺️ Click terrain or a node to start the slope."

    def _enabled_while_viewing_own_kind(self, sm: PlannerStateMachine) -> bool:
        return sm.is_idle_viewing_slope


class _RoadOperation(BuilderOperation):
    mode = BuildMode.ROAD
    group = OperationGroup.BUILDER
    first_instruction = "🗺️ Click terrain or a node to start the road."

    def _enabled_while_viewing_own_kind(self, sm: PlannerStateMachine) -> bool:
        return sm.is_idle_viewing_road


class _LiftOperation(BuilderOperation):
    """The four lift-type buttons: enabled off a slope/road view, and re-typing the viewed lift."""

    group = OperationGroup.BUILDER
    first_instruction = "🗺️ Click terrain or a node to place the first station."

    def __init__(self, mode: str) -> None:
        self.mode = mode

    def _enabled_while_viewing_own_kind(self, sm: PlannerStateMachine) -> bool:
        return sm.is_idle_viewing_lift

    def on_select(self, ctx: PlannerContext, sm: PlannerStateMachine) -> None:
        # Re-typing a viewed lift recomputes its geometry (a REAL map change) → bare remount so the
        # redraw takes, keeping the current framing; otherwise a plain rerun (highlight only).
        retyped_viewed_lift = sm.is_idle_viewing_lift
        actions.select_lift_type_action(self.mode)
        if retyped_viewed_lift:
            bump_camera_epoch()
        trigger_rerun()


class _ImportOperation(BuilderOperation):
    mode = BuildMode.IMPORT
    group = OperationGroup.UTILITY
    first_instruction = "🗺️ Click terrain or a node to place the import area."

    def _enabled_while_viewing_own_kind(self, sm: PlannerStateMachine) -> bool:
        return False  # a utility has no own kind — idle_ready only


class _MergeOperation(BuilderOperation):
    mode = BuildMode.MERGE
    group = OperationGroup.UTILITY
    first_instruction = "🔗 Click a node to start merging."

    def _enabled_while_viewing_own_kind(self, sm: PlannerStateMachine) -> bool:
        return False  # a utility has no own kind — idle_ready only


class _RouteOperation(BuilderOperation):
    mode = BuildMode.ROUTE
    group = OperationGroup.UTILITY
    first_instruction = "🧭 Click a node to set the route start."

    def _enabled_while_viewing_own_kind(self, sm: PlannerStateMachine) -> bool:
        return False  # a utility has no own kind — idle_ready only


_OPERATION_LIST: list[BuilderOperation] = [
    _SlopeOperation(),
    _RoadOperation(),
    _LiftOperation(BuildMode.SURFACE_LIFT),
    _LiftOperation(BuildMode.CHAIRLIFT),
    _LiftOperation(BuildMode.GONDOLA),
    _LiftOperation(BuildMode.AERIAL_TRAM),
    _ImportOperation(),
    _MergeOperation(),
    _RouteOperation(),
]

OPERATIONS: dict[str, BuilderOperation] = {op.mode: op for op in _OPERATION_LIST}


# =============================================================================
# ENTITY KIND SPEC
# =============================================================================


class EntityKindSpec(ABC):
    """The per-kind pieces a viewed slope/road/lift needs, folded into one descriptor."""

    kind: EntityKind

    @abstractmethod
    def viewed_entity_id(self, ctx: PlannerContext) -> str | None:
        """The id of the entity currently being viewed for this kind (from ctx.viewing)."""

    @abstractmethod
    def get_entity(self, graph: ResortGraph, entity_id: str) -> Slope | Road | Lift | None:
        """Look up the entity by id (or None if missing)."""

    @abstractmethod
    def render_stats(self, graph: ResortGraph, entity_id: str) -> None:
        """Render this kind's stats panel for the given entity."""

    @abstractmethod
    def delete_action(self, entity_id: str) -> bool:
        """Delete the entity (returns True if it was deleted)."""


class _SlopeKindSpec(EntityKindSpec):
    kind = EntityKind.SLOPE

    def viewed_entity_id(self, ctx: PlannerContext) -> str | None:
        return ctx.viewing.slope_id

    def get_entity(self, graph: ResortGraph, entity_id: str) -> Slope | Road | Lift | None:
        return graph.slopes.get(entity_id)

    def render_stats(self, graph: ResortGraph, entity_id: str) -> None:
        right_panel.PathStatsPanel(graph=graph, kind=SegmentKind.SLOPE).render(entity_id=entity_id)

    def delete_action(self, entity_id: str) -> bool:
        return actions.delete_slope_action(entity_id)


class _RoadKindSpec(EntityKindSpec):
    kind = EntityKind.ROAD

    def viewed_entity_id(self, ctx: PlannerContext) -> str | None:
        return ctx.viewing.road_id

    def get_entity(self, graph: ResortGraph, entity_id: str) -> Slope | Road | Lift | None:
        return graph.roads.get(entity_id)

    def render_stats(self, graph: ResortGraph, entity_id: str) -> None:
        right_panel.PathStatsPanel(graph=graph, kind=SegmentKind.ROAD).render(entity_id=entity_id)

    def delete_action(self, entity_id: str) -> bool:
        return actions.delete_road_action(entity_id)


class _LiftKindSpec(EntityKindSpec):
    kind = EntityKind.LIFT

    def viewed_entity_id(self, ctx: PlannerContext) -> str | None:
        return ctx.viewing.lift_id

    def get_entity(self, graph: ResortGraph, entity_id: str) -> Slope | Road | Lift | None:
        return graph.lifts.get(entity_id)

    def render_stats(self, graph: ResortGraph, entity_id: str) -> None:
        right_panel.LiftStatsPanel(graph=graph).render(entity_id=entity_id)

    def delete_action(self, entity_id: str) -> bool:
        return actions.delete_lift_action(entity_id)


_ENTITY_KIND_SPEC_LIST: list[EntityKindSpec] = [_SlopeKindSpec(), _RoadKindSpec(), _LiftKindSpec()]

ENTITY_KIND_SPECS: dict[EntityKind, EntityKindSpec] = {spec.kind: spec for spec in _ENTITY_KIND_SPEC_LIST}


# =============================================================================
# DISPATCH ENTRY POINTS
# =============================================================================


def render_control_panel(
    sm: PlannerStateMachine,
    ctx: PlannerContext,
    graph: ResortGraph,
    on_commit: Callable[[int], None],
    on_cancel_connection: Callable[[], None],
) -> None:
    """Render the current state's right-side control panel, then its elevation profile."""
    assert sm.get_current_state_id() in BUILD_STATES, (
        f"State {sm.get_current_state_id()} must be registered in BUILD_STATES"
    )
    build_state = BUILD_STATES[sm.get_current_state_id()]
    panel = build_state.control_panel(
        sm=sm, ctx=ctx, graph=graph, on_commit=on_commit, on_cancel_connection=on_cancel_connection
    )
    panel.render()

    profile = build_state.bottom_profile(ctx=ctx, graph=graph)
    if profile is not None:
        st.plotly_chart(profile.fig, width="stretch", key=profile.key)


def get_click_handler(sm: PlannerStateMachine) -> ClickHandler:
    """The current state's map-click handler."""
    assert sm.get_current_state_id() in BUILD_STATES, (
        f"State {sm.get_current_state_id()} must be registered in BUILD_STATES"
    )
    return BUILD_STATES[sm.get_current_state_id()].click_handler()


def dispatch_click(click_info: ClickInfo) -> None:
    """Look up the terrain elevation for a terrain click, then route to the current state's handler."""
    sm: PlannerStateMachine = st.session_state.state_machine
    dem: DEMService = st.session_state.dem_service

    elevation: float | None = None
    if click_info.click_type == MapClickType.TERRAIN:
        assert click_info.lon is not None and click_info.lat is not None  # validated in ClickInfo
        elevation = dem.get_elevation(lon=click_info.lon, lat=click_info.lat)
        if elevation is None:
            OutsideTerrainMessage(lat=click_info.lat, lon=click_info.lon).display()
            return

    logger.debug(f"Dispatching {click_info.display_name} in state {sm.get_state_name()}")
    get_click_handler(sm=sm)(click_info, elevation)


# =============================================================================
# IMPORT-TIME BIJECTION GUARDS
# =============================================================================
# Each registry must cover its axis with no missing or stray key. A new state, mode, or kind that
# is not registered here fails at import (and in test_mode_registry.py), never silently in the UI.

_sm_ids = {s.id for s in PlannerStateMachine.states}
assert set(BUILD_STATES) == _sm_ids, (
    f"BUILD_STATES keys must equal the state-machine state ids. "
    f"Missing: {_sm_ids - set(BUILD_STATES)}; stray: {set(BUILD_STATES) - _sm_ids}"
)

_modes = set(BuildMode.ALL)
assert set(OPERATIONS) == _modes, (
    f"OPERATIONS keys must equal BuildMode.ALL. Missing: {_modes - set(OPERATIONS)}; stray: {set(OPERATIONS) - _modes}"
)

_kinds = set(EntityKind)
assert set(ENTITY_KIND_SPECS) == _kinds, (
    f"ENTITY_KIND_SPECS keys must equal the EntityKind members. "
    f"Missing: {_kinds - set(ENTITY_KIND_SPECS)}; stray: {set(ENTITY_KIND_SPECS) - _kinds}"
)

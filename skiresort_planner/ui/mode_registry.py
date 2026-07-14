"""Central registries for the app's per-state, per-mode, and per-kind behaviour.

This module is the single dispatch hub at the top of the UI layer: it imports the panels, click
handlers, actions, and renderers, and exposes three registries plus the two dispatch entry points
(`render_control_panel`, `dispatch_click`). Nothing below it imports this module at runtime, so the
dependency flow is one-directional.

Three registries, each keyed to cover its axis exactly (asserted at import):

- ``BUILD_STATES`` — one ``BuildState`` per state-machine state. A ``BuildState`` owns the complete
  per-state UI surface: right panel, click handler, map overlay layers, camera view state, bottom
  elevation profile, merge-highlight ids, and custom-path flag. Keys == the state-machine state ids.
- ``OPERATIONS`` — one ``BuilderOperation`` per build mode (the sidebar build-type buttons): its
  group, its ``enabled(sm)`` greyout rule, and its ``on_select`` action. Keys == ``BuildMode.ALL``.
- ``ENTITY_KIND_SPECS`` — one ``EntityKindSpec`` per ``EntityKind``: the viewed-entity id, entity
  lookup, stats panel, and delete action. Keys == the ``EntityKind`` members.

The abstract methods make it impossible to register a state/mode/kind with a part missing; the
import-time bijection asserts make it impossible to add a state/mode/kind without registering it.
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
from skiresort_planner.model.resort_graph import ResortGraph
from skiresort_planner.ui import actions, bottom_chart, click_handlers, right_panel, sidebar_panels
from skiresort_planner.ui.center_map import MapRenderer
from skiresort_planner.ui.context import BuildMode, EntityKind, PlannerContext
from skiresort_planner.ui.infra import reload_map
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
class StateHeader:
    """The sidebar header for a state: an icon glyph plus a label (rendered as `### {icon} {label}`)."""

    icon: str
    label: str


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
        """The elevation profile below the map, or None when the state shows none."""

    @abstractmethod
    def merge_highlight_node_ids(self, ctx: PlannerContext) -> list[str] | None:
        """Node ids to draw as merge candidates, or None when not merging."""

    @abstractmethod
    def renders_custom_path(self, ctx: PlannerContext) -> bool:
        """Whether proposals draw as a single freehand path (roads + slope custom-connect)."""

    @abstractmethod
    def header(self, ctx: PlannerContext) -> StateHeader:
        """The sidebar header (icon + label) shown while in this state."""

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

    def header(self, ctx: PlannerContext) -> StateHeader:
        # The lift glyph tracks the first lift button so the header matches whatever lift renders first.
        return StateHeader(
            icon=f"{StyleConfig.SLOPE_ICON}{StyleConfig.ROAD_ICON}{StyleConfig.LIFT_ICONS[BuildMode.LIFT_TYPES[0]]}",
            label="Ready to Build",
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

    def header(self, ctx: PlannerContext) -> StateHeader:
        return StateHeader(icon=StyleConfig.VIEWING_ICON, label=f"Viewing {self.kind.value.capitalize()}")

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


class _SlopeBuildingState(BuildState):
    """The three slope states (starting/building/custom_path): orientation arrows at the start point,
    a custom-connect direction arrow while routing, and the in-build slope profile below the map.
    """

    def __init__(self, state_key: str) -> None:
        self.state_key = state_key

    def control_panel(
        self,
        sm: PlannerStateMachine,
        ctx: PlannerContext,
        graph: ResortGraph,
        on_commit: Callable[[int], None],
        on_cancel_connection: Callable[[], None],
    ) -> right_panel.ControlPanel:
        return right_panel.SlopeBuildingControlPanel(
            sm=sm, ctx=ctx, graph=graph, on_commit=on_commit, on_cancel_connection=on_cancel_connection
        )

    def click_handler(self) -> ClickHandler:
        return click_handlers.handle_slope_building_click

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
            start_node = graph.nodes.get(ctx.custom_connect.start_node)
            if start_node:
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
        if not ctx.slope_build.segments:
            return None
        fig = bottom_chart.render_building_profile(
            building_segments=ctx.slope_build.segments, building_name=ctx.slope_build.name, graph=graph
        )
        return ProfileSpec(fig=fig, key="combined_profile")

    def merge_highlight_node_ids(self, ctx: PlannerContext) -> list[str] | None:
        return None

    def renders_custom_path(self, ctx: PlannerContext) -> bool:
        return ctx.custom_connect.force_mode

    def header(self, ctx: PlannerContext) -> StateHeader:
        return StateHeader(icon=StyleConfig.BUILDING_ICON, label="Building Slope...")

    def sidebar_panel(
        self, sm: PlannerStateMachine, ctx: PlannerContext, graph: ResortGraph
    ) -> sidebar_panels.SidebarPanel:
        return sidebar_panels.SlopeSidebarPanel(sm=sm, ctx=ctx, graph=graph)

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
        if ctx.lift.start_node_id:
            node = graph.nodes.get(ctx.lift.start_node_id)
            if node is None:
                raise ValueError(f"Lift start node {ctx.lift.start_node_id} not found in graph")
            lat, lon, elevation = node.lat, node.lon, node.elevation
        elif ctx.lift.start_location:
            loc = ctx.lift.start_location
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

    def header(self, ctx: PlannerContext) -> StateHeader:
        return StateHeader(icon=StyleConfig.BUILDING_ICON, label="Placing Lift...")

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
        center_lon = ctx.deferred.osm_import_center_lon
        center_lat = ctx.deferred.osm_import_center_lat
        if center_lon is None or center_lat is None:
            return []
        center_elev = dem.get_elevation(lon=center_lon, lat=center_lat) or 0.0
        return renderer.create_import_bbox_layers(
            center_lon=center_lon,
            center_lat=center_lat,
            half_width_m=ctx.deferred.osm_import_half_width_km * 1000.0,
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

    def header(self, ctx: PlannerContext) -> StateHeader:
        return StateHeader(icon=StyleConfig.BUILDING_ICON, label="Importing Area...")

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

    def header(self, ctx: PlannerContext) -> StateHeader:
        return StateHeader(icon=StyleConfig.BUILDING_ICON, label="Merging Nodes...")

    def sidebar_panel(
        self, sm: PlannerStateMachine, ctx: PlannerContext, graph: ResortGraph
    ) -> sidebar_panels.SidebarPanel:
        return sidebar_panels.MergeSidebarPanel(sm=sm, ctx=ctx, graph=graph)

    def blocks_build_buttons(self) -> bool:
        return True


class _RoadBuildingState(BuildState):
    """The two road states (starting/building): an origin dot while starting, and the in-build road
    profile below the map. Roads always render as a custom path.
    """

    def __init__(self, state_key: str) -> None:
        self.state_key = state_key

    def control_panel(
        self,
        sm: PlannerStateMachine,
        ctx: PlannerContext,
        graph: ResortGraph,
        on_commit: Callable[[int], None],
        on_cancel_connection: Callable[[], None],
    ) -> right_panel.ControlPanel:
        return right_panel.RoadBuildingControlPanel(
            sm=sm, ctx=ctx, graph=graph, on_commit=on_commit, on_cancel_connection=on_cancel_connection
        )

    def click_handler(self) -> ClickHandler:
        return click_handlers.handle_road_building_click

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
        # Only the origin (starting) shows a dot; once segments exist the road draws itself.
        if self.state_key != "road_starting":
            return []
        if ctx.road_build.start_node_id:
            node = graph.nodes.get(ctx.road_build.start_node_id)
            if node is None:
                raise ValueError(f"Road start node {ctx.road_build.start_node_id} not found in graph")
            lat, lon, elevation = node.lat, node.lon, node.elevation
        elif ctx.road_build.start_location:
            loc = ctx.road_build.start_location
            lat, lon, elevation = loc.lat, loc.lon, loc.elevation
        else:
            return []
        return renderer.create_pending_road_marker_layers(lat=lat, lon=lon, elevation=elevation, use_3d=use_3d)

    def view_state(self, ctx: PlannerContext, graph: ResortGraph, *, use_3d: bool) -> ViewState:
        return _stored_2d_view(ctx)

    def bottom_profile(self, ctx: PlannerContext, graph: ResortGraph) -> ProfileSpec | None:
        if not ctx.road_build.segments:
            return None
        fig = bottom_chart.render_building_profile(
            building_segments=ctx.road_build.segments, building_name=ctx.road_build.name, graph=graph
        )
        return ProfileSpec(fig=fig, key="combined_road_profile")

    def merge_highlight_node_ids(self, ctx: PlannerContext) -> list[str] | None:
        return None

    def renders_custom_path(self, ctx: PlannerContext) -> bool:
        return True

    def header(self, ctx: PlannerContext) -> StateHeader:
        return StateHeader(icon=StyleConfig.BUILDING_ICON, label="Building Road...")

    def sidebar_panel(
        self, sm: PlannerStateMachine, ctx: PlannerContext, graph: ResortGraph
    ) -> sidebar_panels.SidebarPanel:
        return sidebar_panels.RoadSidebarPanel(sm=sm, ctx=ctx, graph=graph)

    def blocks_build_buttons(self) -> bool:
        return True


_BUILD_STATE_LIST: list[BuildState] = [
    _IdleReadyState(),
    _IdleViewingSlopeState(),
    _IdleViewingRoadState(),
    _IdleViewingLiftState(),
    _SlopeBuildingState("slope_starting"),
    _SlopeBuildingState("slope_building"),
    _SlopeBuildingState("slope_custom_path"),
    _LiftPlacingState(),
    _ImportPlacingState(),
    _MergePlacingState(),
    _RoadBuildingState("road_starting"),
    _RoadBuildingState("road_building"),
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

    @abstractmethod
    def enabled(self, sm: PlannerStateMachine) -> bool:
        """Whether the button is clickable in the current state."""

    @property
    @abstractmethod
    def first_instruction(self) -> str:
        """One-line hint shown in idle: what the FIRST map click does in this mode."""

    def on_select(self, ctx: PlannerContext, sm: PlannerStateMachine) -> None:
        """Highlight this mode and reload — the invariant EVERY builder button shares.

        A button only highlights; the mode start (state entry) always happens later on the first
        map click. Ops needing extra setup override this, do their work first, then call super().
        """
        ctx.build_mode.mode = self.mode
        reload_map()


def _idle_not_building(sm: PlannerStateMachine) -> bool:
    """True while idle and not mid-build/placement (all buttons disable during a build/placement)."""
    return not (
        sm.is_any_slope_state
        or sm.is_lift_placing
        or sm.is_any_road_state
        or sm.is_import_placing
        or sm.is_merge_placing
    )


class _SlopeOperation(BuilderOperation):
    mode = BuildMode.SLOPE
    group = OperationGroup.BUILDER
    first_instruction = "🗺️ Click terrain or a node to start the slope."

    def enabled(self, sm: PlannerStateMachine) -> bool:
        return _idle_not_building(sm) and not (sm.is_idle_viewing_lift or sm.is_idle_viewing_road)


class _RoadOperation(BuilderOperation):
    mode = BuildMode.ROAD
    group = OperationGroup.BUILDER
    first_instruction = "🗺️ Click terrain or a node to start the road."

    def enabled(self, sm: PlannerStateMachine) -> bool:
        return _idle_not_building(sm) and not (sm.is_idle_viewing_slope or sm.is_idle_viewing_lift)


class _LiftOperation(BuilderOperation):
    """The four lift-type buttons: enabled off a slope/road view, and re-typing the viewed lift."""

    group = OperationGroup.BUILDER
    first_instruction = "🗺️ Click terrain or a node to place the bottom station."

    def __init__(self, mode: str) -> None:
        self.mode = mode

    def enabled(self, sm: PlannerStateMachine) -> bool:
        return _idle_not_building(sm) and not (sm.is_idle_viewing_slope or sm.is_idle_viewing_road)

    def on_select(self, ctx: PlannerContext, sm: PlannerStateMachine) -> None:
        # Extra work: track the chosen type and, when viewing a lift, re-type it. Then the shared
        # highlight + reload via super().
        actions.select_lift_type_action(self.mode)
        super().on_select(ctx=ctx, sm=sm)


class _ImportOperation(BuilderOperation):
    mode = BuildMode.IMPORT
    group = OperationGroup.UTILITY
    first_instruction = "🗺️ Click the map to place the import area."

    def enabled(self, sm: PlannerStateMachine) -> bool:
        return _idle_not_building(sm)


class _MergeOperation(BuilderOperation):
    mode = BuildMode.MERGE
    group = OperationGroup.UTILITY
    first_instruction = "🔗 Click a node to start merging."

    def enabled(self, sm: PlannerStateMachine) -> bool:
        return _idle_not_building(sm)


_OPERATION_LIST: list[BuilderOperation] = [
    _SlopeOperation(),
    _RoadOperation(),
    _LiftOperation(BuildMode.SURFACE_LIFT),
    _LiftOperation(BuildMode.CHAIRLIFT),
    _LiftOperation(BuildMode.GONDOLA),
    _LiftOperation(BuildMode.AERIAL_TRAM),
    _ImportOperation(),
    _MergeOperation(),
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
        right_panel.SlopeStatsPanel(graph=graph).render(slope_id=entity_id)

    def delete_action(self, entity_id: str) -> bool:
        return actions.delete_slope_action(entity_id)


class _RoadKindSpec(EntityKindSpec):
    kind = EntityKind.ROAD

    def viewed_entity_id(self, ctx: PlannerContext) -> str | None:
        return ctx.viewing.road_id

    def get_entity(self, graph: ResortGraph, entity_id: str) -> Slope | Road | Lift | None:
        return graph.roads.get(entity_id)

    def render_stats(self, graph: ResortGraph, entity_id: str) -> None:
        right_panel.RoadStatsPanel(graph=graph).render(road_id=entity_id)

    def delete_action(self, entity_id: str) -> bool:
        return actions.delete_road_action(entity_id)


class _LiftKindSpec(EntityKindSpec):
    kind = EntityKind.LIFT

    def viewed_entity_id(self, ctx: PlannerContext) -> str | None:
        return ctx.viewing.lift_id

    def get_entity(self, graph: ResortGraph, entity_id: str) -> Slope | Road | Lift | None:
        return graph.lifts.get(entity_id)

    def render_stats(self, graph: ResortGraph, entity_id: str) -> None:
        right_panel.LiftStatsPanel(graph=graph).render(lift_id=entity_id)

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
    """Render the current state's right-side control panel."""
    panel = BUILD_STATES[sm.get_current_state_id()].control_panel(
        sm=sm, ctx=ctx, graph=graph, on_commit=on_commit, on_cancel_connection=on_cancel_connection
    )
    panel.render()


def get_click_handler(sm: PlannerStateMachine) -> ClickHandler:
    """The current state's map-click handler."""
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

    logger.info(f"Dispatching {click_info.display_name} in state {sm.get_state_name()}")
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

"""MapRenderer - Pydeck map rendering for ski resort planner.

Renders all resort elements on an interactive 3D map using GPU-accelerated deck.gl:
- Terrain base map with 3D pitch
- Committed slopes with difficulty-colored polygons (PolygonLayer)
- Lifts as straight lines with pylon markers (PathLayer + ScatterplotLayer)
- Proposed paths as dashed lines (PathLayer)
- Nodes as clickable markers (ScatterplotLayer)
- Terrain orientation arrows (PathLayer)
- Direction arrows for custom connect and lift placement modes

Key differences from Folium:
- Uses [lon, lat] coordinate order (GeoJSON standard)
- Colors as RGBA lists [R, G, B, A] (0-255)
- Data prepared as list[dict] for GPU streaming
- pickable=True enables click detection

Reference: DETAILS_UI.md for interaction patterns
"""

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, cast

import pydeck as pdk

from skiresort_planner.constants import (
    ClickConfig,
    MapConfig,
    MarkerConfig,
    RoutePlannerConfig,
    StyleConfig,
)
from skiresort_planner.core.geo_calculator import GeoCalculator
from skiresort_planner.generators.osm_importer import bbox_around
from skiresort_planner.model.path_point import PathPoint
from skiresort_planner.model.path_segment import SegmentKind
from skiresort_planner.model.proposed_path import ProposedPathSegment
from skiresort_planner.model.resort_graph import ResortGraph

if TYPE_CHECKING:
    from skiresort_planner.core.terrain_analyzer import TerrainOrientation
    from skiresort_planner.model.lift import Lift
    from skiresort_planner.model.road import Road
    from skiresort_planner.model.routing import Route, ViewingGroup
    from skiresort_planner.model.slope import Slope

logger = logging.getLogger(__name__)


@dataclass
class LayerCollection:
    """Manages Pydeck layers with correct z-ordering.

    Z-order (back to front): terrain → slopes → roads → lifts → pylons → nodes → proposals → markers

    Slopes and roads render in separate buckets (built by one shared segment
    loop, kept distinct for z-order and brown-vs-difficulty styling). Pylons render
    AFTER their lift cables so a pylon marker wins the hover/pick over the cable
    beneath it (matching Z_OFFSET_2D_PYLONS > Z_OFFSET_2D_LIFTS). Nodes are
    placed AFTER slopes/lifts/pylons so they:
    1. Render visually on top of lines (correct for junction display)
    2. Get click priority over slopes/lifts (nodes are small, need priority)

    Parking places are not a separate layer — a parking node renders as its node
    marker (blue + bigger) inside the nodes layer (see _create_node_layer).

    Markers layer includes crosshair for terrain node placement when active.
    """

    terrain: list[pdk.Layer] = field(default_factory=list)
    pylons: list[pdk.Layer] = field(default_factory=list)
    slopes: list[pdk.Layer] = field(default_factory=list)
    roads: list[pdk.Layer] = field(default_factory=list)
    lifts: list[pdk.Layer] = field(default_factory=list)
    nodes: list[pdk.Layer] = field(default_factory=list)
    proposals: list[pdk.Layer] = field(default_factory=list)
    markers: list[pdk.Layer] = field(default_factory=list)

    def get_ordered_layers(self) -> list[pdk.Layer]:
        """Return all layers in correct z-order (back to front)."""
        return (
            self.terrain
            + self.slopes
            + self.roads
            + self.lifts
            + self.pylons
            + self.nodes
            + self.proposals
            + self.markers
        )


class MapRenderer:
    """Renders ski resort graph on a Pydeck map.

    Example:
        renderer = MapRenderer(graph=graph)
        deck = renderer.render()
        st.pydeck_chart(deck)
    """

    def __init__(
        self,
        graph: ResortGraph | None = None,
        center_lat: float = MapConfig.START_CENTER_LAT,
        center_lon: float = MapConfig.START_CENTER_LON,
        zoom: float = MapConfig.VIEWING_ZOOM,
        pitch: float = MapConfig.DEFAULT_PITCH,
        bearing: float = MapConfig.DEFAULT_BEARING,
    ) -> None:
        """Initialize map renderer.

        Args:
            graph: Resort graph to render (can set later)
            center_lat: Initial map center latitude
            center_lon: Initial map center longitude
            zoom: Initial zoom level
            pitch: 3D tilt angle (0=top-down, 60=angled)
            bearing: Map rotation (0=north up)
        """
        self.graph = graph
        self.center_lat = center_lat
        self.center_lon = center_lon
        self.zoom = zoom
        self.pitch = pitch
        self.bearing = bearing
        # Set by the render loop while a flythrough plays (None = normal viewing) so deck.gl eases the
        # camera between frames instead of jumping.
        self.transition_duration: int | None = None

    def get_view_state(self) -> pdk.ViewState:
        """Create Pydeck ViewState from current settings.

        While a flythrough plays, emit transition_duration + a LinearInterpolator (via the @@type JSON the
        8.8.9 converter resolves) so deck.gl GLIDES the camera between sparse keyframes at 60fps client-side —
        pydeck passes unknown kwargs straight into the deck.gl JSON. Absent otherwise (normal viewing).
        """
        extra: dict[str, object] = {}
        if self.transition_duration is not None:
            extra["transition_duration"] = self.transition_duration
            extra["transition_interpolator"] = {
                "@@type": "LinearInterpolator",
                "transitionProps": ["longitude", "latitude", "zoom", "pitch", "bearing"],
            }
        return pdk.ViewState(
            latitude=self.center_lat,
            longitude=self.center_lon,
            zoom=self.zoom,
            pitch=self.pitch,
            bearing=self.bearing,
            **extra,
        )

    def update_view(
        self,
        lat: float | None = None,
        lon: float | None = None,
        zoom: float | None = None,
        pitch: float | None = None,
        bearing: float | None = None,
    ) -> None:
        """Update view state parameters."""
        if lat is not None:
            self.center_lat = lat
        if lon is not None:
            self.center_lon = lon
        if zoom is not None:
            self.zoom = zoom
        if pitch is not None:
            self.pitch = pitch
        if bearing is not None:
            self.bearing = bearing

    def set_flythrough_easing(self, *, flying: bool) -> None:
        """While flying, emit a deck.gl transition so the camera GLIDES between keyframes; else none."""
        self.transition_duration = MapConfig.FLYTHROUGH_TRANSITION_MS if flying else None

    def render(
        self,
        proposals: list[ProposedPathSegment] | None = None,
        selected_proposal_idx: int | None = None,
        *,
        show_nodes: bool = True,
        show_segments: bool = True,
        show_lifts: bool = True,
        highlight_segment_ids: list[str] | None = None,
        is_custom_path: bool = False,
        extra_layers: list[pdk.Layer] | None = None,
        terrain_layer: pdk.Layer | None = None,
        use_3d: bool = False,
        selected_node_ids: list[str] | None = None,
    ) -> pdk.Deck:
        """Render complete map with all layers.

        Args:
            proposals: Proposed paths to display
            selected_proposal_idx: Index of highlighted proposal
            show_nodes: Whether to show node markers
            show_segments: Whether to show segment polygons
            show_lifts: Whether to show lift lines
            highlight_segment_ids: Segment IDs to highlight (active slope)
            is_custom_path: Whether showing custom connect paths
            extra_layers: Additional layers to include (markers always on top)
            terrain_layer: Pre-generated terrain elevation layer (BitmapLayer)
            use_3d: If True, render with 3D terrain elevations. If False, flat 2D at z=0.
            selected_node_ids: Node ids currently selected (merge/delete/route start) — drawn RED.

        Returns:
            pdk.Deck object ready for display.

        Z-order (back to front): terrain → pylons → slopes → lifts → nodes → proposals → markers
        """
        layer_collection = LayerCollection()

        # Basemap layer (2D TileLayer or 3D TerrainLayer) always at bottom
        if terrain_layer:
            layer_collection.terrain.append(terrain_layer)

        if self.graph:
            defect_ids = self._defect_entity_ids()  # slopes/lifts to gray out (empty when no core yet)
            if show_lifts:
                lift_layers = self._create_lift_layers(use_3d=use_3d, defect_ids=defect_ids)
                layer_collection.pylons.extend(lift_layers["pylons"])
                layer_collection.lifts.extend(lift_layers["cables_icons"])

            if show_nodes:
                layer_collection.nodes.append(
                    self._create_node_layer(use_3d=use_3d, selected_node_ids=selected_node_ids)
                )

            if show_segments:
                # One shared loop builds slope + road layers, returned in
                # separate buckets so each keeps its z-order and styling.
                segment_layers = self._create_segment_layers(
                    highlight_ids=highlight_segment_ids, use_3d=use_3d, defect_ids=defect_ids
                )
                layer_collection.slopes.extend(segment_layers["slopes"])
                layer_collection.roads.extend(segment_layers["roads"])

        if proposals:
            layer_collection.proposals.extend(
                self._create_proposal_layers(
                    proposals=proposals,
                    selected_idx=selected_proposal_idx,
                    is_custom_path=is_custom_path,
                    use_3d=use_3d,
                )
            )

        # Extra layers (commit/select markers) always on top
        if extra_layers:
            layer_collection.markers.extend(extra_layers)

        # 3D mode: terrain_layer (TerrainLayer) provides basemap, set map_style=None
        # 2D mode: Use OPENTOPOMAP_STYLE dict - the proper way to render XYZ raster tiles
        #          (TileLayer doesn't work because pydeck doesn't expose renderSubLayers)
        if terrain_layer is not None:
            map_style = None  # TerrainLayer provides the basemap in 3D mode
            map_provider = None
        else:
            from skiresort_planner.ui.terrain_layer import OPENTOPOMAP_STYLE

            map_style = OPENTOPOMAP_STYLE  # Custom raster basemap for 2D mode
            map_provider = "mapbox"  # Required when map_style is a dict

        return pdk.Deck(
            map_style=map_style,
            map_provider=map_provider,
            initial_view_state=self.get_view_state(),
            layers=layer_collection.get_ordered_layers(),
            tooltip=self._create_tooltip_config(),
            parameters={"pickingRadius": ClickConfig.PICKING_RADIUS_PX},
        )

    # =========================================================================
    # Z-COORDINATE HELPERS
    # =========================================================================

    @staticmethod
    def _get_z(elevation: float, z_offset: float, *, use_3d: bool, flat_z: float = 0.0) -> float:
        """Get z-coordinate based on view mode.

        Args:
            elevation: Real elevation in meters
            z_offset: Offset above terrain for visibility
            use_3d: If True, use real elevation. If False, use flat z.
            flat_z: Z-value for 2D mode (default 0, can use layer ordering offsets)

        Returns:
            Z-coordinate for rendering
        """
        if use_3d:
            return elevation + z_offset
        return flat_z

    @staticmethod
    def _calculate_3d_view_for_endpoints(
        start_lat: float,
        start_lon: float,
        start_elev: float,
        end_lat: float,
        end_lon: float,
        end_elev: float,
        camera_bearing_offset: float,
    ) -> tuple[float, float, float, float, float]:
        """Calculate optimal camera position to view a feature between two endpoints.

        Unified helper for both slope and lift 3D view calculations.
        Positions camera perpendicular to the feature direction.
        Result shows start point on LEFT, end point on RIGHT.

        Args:
            start_lat, start_lon, start_elev: Start endpoint coordinates (appears on left)
            end_lat, end_lon, end_elev: End endpoint coordinates (appears on right)
            camera_bearing_offset: Offset from feature bearing (-90 for start-left/end-right)

        Returns:
            Tuple (lat, lon, bearing, zoom, pitch) for camera settings.
        """
        # Calculate feature direction (from start to end)
        feature_bearing = GeoCalculator.initial_bearing_deg(
            lon1=start_lon,
            lat1=start_lat,
            lon2=end_lon,
            lat2=end_lat,
        )

        # Camera bearing: perpendicular to feature
        camera_bearing = (feature_bearing + camera_bearing_offset) % 360

        # Center on midpoint
        center_lat = (start_lat + end_lat) / 2
        center_lon = (start_lon + end_lon) / 2

        return (center_lat, center_lon, camera_bearing, MapConfig.VIEW_3D_ZOOM, MapConfig.VIEW_3D_PITCH)

    @staticmethod
    def _calculate_3d_view_for_entity(
        graph: ResortGraph,
        entity: "Slope | Road | Lift",
    ) -> tuple[float, float, float, float, float]:
        """Side-view camera for any start/end-node entity (slope, road, or lift).

        All three expose start_node_id/end_node_id; the -90 camera offset puts
        start_node on the LEFT and end_node on the RIGHT.

        Args:
            graph: Resort graph (for node lookup).
            entity: A Slope/Road/Lift with start_node_id/end_node_id.

        Returns:
            Tuple (lat, lon, bearing, zoom, pitch) for camera settings.
        """
        start_node = graph.nodes[entity.start_node_id]
        end_node = graph.nodes[entity.end_node_id]

        return MapRenderer._calculate_3d_view_for_endpoints(
            start_lat=start_node.lat,
            start_lon=start_node.lon,
            start_elev=start_node.elevation,
            end_lat=end_node.lat,
            end_lon=end_node.lon,
            end_elev=end_node.elevation,
            camera_bearing_offset=-90,
        )

    @staticmethod
    def calculate_3d_view_for_slope(
        graph: ResortGraph,
        slope_id: str,
    ) -> tuple[float, float, float, float, float]:
        """Calculate optimal side-view camera to view a slope in 3D."""
        slope = graph.slopes.get(slope_id)
        if not slope:
            raise ValueError(f"Slope {slope_id} not found")
        return MapRenderer._calculate_3d_view_for_entity(graph=graph, entity=slope)

    @staticmethod
    def calculate_3d_view_for_road(
        graph: ResortGraph,
        road_id: str,
    ) -> tuple[float, float, float, float, float]:
        """Calculate optimal side-view camera to view a road in 3D."""
        road = graph.roads.get(road_id)
        if not road:
            raise ValueError(f"Road {road_id} not found")
        return MapRenderer._calculate_3d_view_for_entity(graph=graph, entity=road)

    @staticmethod
    def calculate_3d_view_for_lift(
        graph: ResortGraph,
        lift_id: str,
    ) -> tuple[float, float, float, float, float]:
        """Calculate optimal side-view camera to view a lift in 3D."""
        lift = graph.lifts.get(lift_id)
        if not lift:
            raise ValueError(f"Lift {lift_id} not found")
        return MapRenderer._calculate_3d_view_for_entity(graph=graph, entity=lift)

    @staticmethod
    def calculate_3d_view_for_route(
        graph: ResortGraph,
        start_node_id: str,
        end_node_id: str,
    ) -> tuple[float, float, float, float, float]:
        """Side-view camera framing a route between its start and end nodes (same helper as entities),
        but one flat VIEW_3D_ROUTE_ZOOM_OUT step further out so a whole route fits (not per-size).
        """
        start, end = graph.nodes[start_node_id], graph.nodes[end_node_id]
        lat, lon, bearing, zoom, pitch = MapRenderer._calculate_3d_view_for_endpoints(
            start_lat=start.lat,
            start_lon=start.lon,
            start_elev=start.elevation,
            end_lat=end.lat,
            end_lon=end.lon,
            end_elev=end.elevation,
            camera_bearing_offset=-90,
        )
        return (lat, lon, bearing, zoom - MapConfig.VIEW_3D_ROUTE_ZOOM_OUT, pitch)

    @staticmethod
    def flythrough_keyframes(groups: "Sequence[ViewingGroup]") -> list[tuple[float, float, float]]:
        """Camera keyframes (lat, lon, bearing) for a flythrough, glided between by deck.gl client-side.

        A single group → 2 keyframes (start, end): deck.gl glides the camera its whole length in one smooth
        move. A multi-group route → one keyframe per group at FLYTHROUGH_ANCHOR_FRACTION + a final end
        keyframe. Bearing is the group's gross straight-line heading (start→end), so a folded slope run is
        one steady lift→lift sightline, not a curve-tracking wobble.
        """
        if not groups:  # nothing viewed / not flying — a real empty state, not a bad group
            return []

        def keyframe_at(group: "ViewingGroup", fraction: float) -> tuple[float, float, float]:
            # A viewing group is always real committed geometry (slope/cable/route element) — never <2 pts.
            assert len(group.actual_polyline) >= 2, f"viewing group has <2 points: {group.actual_polyline}"
            points = [PathPoint(lon=lon, lat=lat, elevation=elev) for lon, lat, elev in group.actual_polyline]
            here = PathPoint.interpolate_at_fraction(points=points, fraction=fraction)
            (s_lon, s_lat, _), (e_lon, e_lat, _) = group.straight_line
            bearing = GeoCalculator.initial_bearing_deg(lon1=s_lon, lat1=s_lat, lon2=e_lon, lat2=e_lat)
            # Nav-style: center the map AHEAD of the real position along the bearing, so "here" sits below
            # screen centre and the run ahead is in view.
            ahead_lon, ahead_lat = GeoCalculator.destination(
                lon=here.lon, lat=here.lat, bearing_deg=bearing, distance_m=MapConfig.FLYTHROUGH_LOOK_AHEAD_M
            )
            return (ahead_lat, ahead_lon, bearing)

        if len(groups) == 1:
            return [keyframe_at(groups[0], 0.0), keyframe_at(groups[0], 1.0)]
        keyframes = [keyframe_at(g, MapConfig.FLYTHROUGH_ANCHOR_FRACTION) for g in groups]
        keyframes.append(keyframe_at(groups[-1], 1.0))
        return keyframes

    @staticmethod
    def flythrough_view_state(
        keyframes: "Sequence[tuple[float, float, float]]", index: int
    ) -> tuple[float, float, float, float, float]:
        """The camera pose at keyframe `index`: (lat, lon, bearing, zoom, pitch). Caller supplies a valid
        index (via ViewingContext.flythrough_index). Zoom+pitch are the unified 3D-view constants so the
        flythrough frames exactly like the standard 3D view.
        """
        if len(keyframes) < 2:
            raise ValueError("flythrough needs at least 2 keyframes")
        lat, lon, bearing = keyframes[index]
        return (lat, lon, bearing, MapConfig.VIEW_3D_ZOOM, MapConfig.VIEW_3D_PITCH)

    # =========================================================================
    # SEGMENT LAYERS
    # =========================================================================

    def _defect_entity_ids(self) -> set[str]:
        """Slope/lift ids with a connectivity defect (disconnected or one-way) — grayed out on the map.

        Derived from the same connectivity_defects classifier the panel counts/lists use, so what the
        map dims and what the summary reports can't disagree. Empty when no core exists yet.
        """
        if not self.graph:
            return set()
        labels = self.graph.strongly_connected_labels()
        core = self.graph.get_core_resort(labels=labels)
        return {d.entity_id for d in self.graph.connectivity_defects(labels=labels, core=core)}

    def _create_segment_layers(
        self, highlight_ids: list[str] | None = None, *, use_3d: bool = False, defect_ids: set[str] | None = None
    ) -> dict[str, list[pdk.Layer]]:
        """Create belt/center-line/icon layers for slopes AND roads in one pass.

        Each committed segment carries its own kind (SegmentKind.SLOPE/ROAD), so
        road-vs-slope is read straight off the segment — never reconstructed from
        which entity happens to own it (a Road entity doesn't exist yet while its
        segments are being built). road_of/slope_of are used ONLY to find the
        owning entity for panel-click routing, not to classify color. Road data
        goes to a SEPARATE bucket so roads keep their own z-order and brown
        styling, distinct from difficulty slopes.

        Returns:
            Dict with 'slopes' and 'roads' keys, each a list of pdk layers.
        """
        if not self.graph:
            return {"slopes": [], "roads": []}

        if highlight_ids is None:
            highlight_ids = []
        if defect_ids is None:
            defect_ids = set()  # entity ids to gray out; here matched against slope ids
        # Segment → owner maps, built once — used only for click/panel routing.
        road_of = self.graph.segment_owner_map(SegmentKind.ROAD)
        slope_of = self.graph.segment_owner_map(SegmentKind.SLOPE)

        # One record per segment, sorted into its owner's bucket by segment.kind.
        # Roads are flat brown; slopes are difficulty-colored. In-build segments
        # (no finished Slope/Road yet) stay segment-typed for clicks.
        slope_records: list[dict[str, object]] = []
        road_records: list[dict[str, object]] = []

        for seg_id, segment in self.graph.segments.items():
            polygon_coords = segment.get_belt_polygon()
            if not polygon_coords:
                # A committed segment always has ≥2 points, so its belt polygon is never empty.
                raise RuntimeError(f"Segment {seg_id} produced an empty belt polygon")

            # Flat-mode z-offset per kind (keyed by the StrEnum value → reload-safe).
            flat_z = MapConfig.SEGMENT_FLAT_Z[segment.kind.value]
            center_line = [
                [
                    p.lon,
                    p.lat,
                    self._get_z(
                        elevation=p.elevation, z_offset=MarkerConfig.PATH_Z_OFFSET_M, use_3d=use_3d, flat_z=flat_z
                    ),
                ]
                for p in segment.points
            ]
            mid_pt = segment.points[len(segment.points) // 2]
            icon_z = self._get_z(
                elevation=mid_pt.elevation,
                z_offset=MarkerConfig.MARKER_Z_OFFSET_M,
                use_3d=use_3d,
                flat_z=MapConfig.Z_OFFSET_2D_ICONS,
            )
            icon_position = [mid_pt.lon, mid_pt.lat, icon_z]

            if segment.kind == SegmentKind.ROAD:
                # Road segment: flat brown. A finished road opens its panel on click;
                # an in-build road segment (no Road entity yet) stays segment-typed.
                road = road_of.get(seg_id)
                road_records.append(
                    {
                        "type": ClickConfig.TYPE_ROAD if road is not None else ClickConfig.TYPE_SEGMENT,
                        "id": road.id if road is not None else seg_id,
                        "polygon": list(polygon_coords),
                        "center_line": center_line,
                        "width": segment.width_m,  # real belt width → 3D renders a terrain-draped ribbon
                        "color": list(StyleConfig.ROAD_COLOR_RGBA),
                        "name": f"{StyleConfig.ROAD_ICON} {road.name}"
                        if road is not None
                        else f"Building road: {seg_id}",
                        "icon_type": ClickConfig.TYPE_ROAD if road is not None else ClickConfig.TYPE_SEGMENT,
                        "icon_id": road.id if road is not None else seg_id,
                        "icon_position": icon_position,
                        "icon_color": list(StyleConfig.ROAD_COLOR_RGBA),
                        "icon_name": f"{StyleConfig.ROAD_ICON} {road.name}"
                        if road is not None
                        else f"Building road: {seg_id}",
                    }
                )
                continue

            # Slope segment: difficulty-colored. (Any other kind must be handled above.)
            if segment.kind != SegmentKind.SLOPE:
                raise ValueError(f"segment {seg_id} has unhandled kind for rendering: {segment.kind!r}")
            slope = slope_of.get(seg_id)
            if slope is not None:
                difficulty = cast("Slope", slope).get_difficulty(segments=self.graph.segments)
                slope_id: str | None = slope.id
            else:
                difficulty = segment.difficulty
                slope_id = None

            color = list(StyleConfig.SLOPE_COLORS_RGBA[difficulty])
            # Connectivity-defect slope → mute only the belt/centerline toward gray ("half-dead").
            # The center-circle icon keeps its full hue so it stays a clear clickable marker.
            if slope_id in defect_ids:
                color = StyleConfig.gray_out(color)

            # Adjust opacity for highlight
            if seg_id in highlight_ids:
                color[3] = 180  # More opaque
            else:
                color[3] = 100  # Semi-transparent

            # Finished slope → icon opens slope panel; orphan → segment-typed (in build).
            slope_records.append(
                {
                    "type": ClickConfig.TYPE_SEGMENT,
                    "id": seg_id,
                    "polygon": list(polygon_coords),
                    "center_line": center_line,
                    "width": segment.width_m,  # real belt width → 3D renders a terrain-draped ribbon
                    "color": color,
                    "name": f"{StyleConfig.SLOPE_ICON} {slope.name}" if slope is not None else f"Segment {seg_id}",
                    "icon_type": ClickConfig.TYPE_SLOPE if slope is not None else ClickConfig.TYPE_SEGMENT,
                    "icon_id": slope_id if slope is not None else seg_id,
                    "icon_position": icon_position,
                    "icon_color": StyleConfig.SLOPE_COLORS_RGBA[difficulty],
                    "icon_name": f"{StyleConfig.SLOPE_ICON} {slope.name}"
                    if slope is not None
                    else f"Building: {seg_id}",
                }
            )

        return {
            "slopes": self._build_path_layers(slope_records, id_prefix="segments", use_3d=use_3d),
            "roads": self._build_path_layers(road_records, id_prefix="roads", use_3d=use_3d),
        }

    def _build_path_layers(self, records: list[dict[str, object]], id_prefix: str, *, use_3d: bool) -> list[pdk.Layer]:
        """Build belt/center-line/icon layers from segment records.

        Shared by slopes and roads so both render identically (belt polygon in
        2D, center line in 2D+3D, icon marker); only the record data (color,
        click type/id) differs. `id_prefix` namespaces the pydeck layer ids.

        Each record's `type`/`id` drive belt+center-line clicks; the `icon_*`
        fields drive the midpoint marker (which may route elsewhere, e.g. a
        slope icon opens the slope panel while its belt is segment-typed).
        """
        if not records:
            return []

        layers: list[pdk.Layer] = []

        # Belt polygons - ONLY in 2D mode (PolygonLayer doesn't support z-coords)
        if not use_3d:
            layers.append(
                pdk.Layer(
                    "PolygonLayer",
                    records,
                    get_polygon="polygon",
                    get_fill_color="color",
                    get_line_color=[255, 255, 255, 100],
                    line_width_min_pixels=1,
                    pickable=True,
                    auto_highlight=True,
                    highlight_color=[255, 255, 255, 80],
                    id=f"{id_prefix}_belt",
                )
            )

        # Center line: 2D is a thin line over the belt polygon; 3D has no belt polygon (PolygonLayer
        # is flat), so the line IS the belt — rendered at each segment's real width_m so it drapes over
        # terrain as a ribbon. Widths are in metres (deck.gl PathLayer default; same as route/proposal
        # layers) — do NOT pass width_units, pydeck mangles string props into "@@=" accessors.
        layers.append(
            pdk.Layer(
                "PathLayer",
                records,
                get_path="center_line",
                get_color="color",
                get_width="width" if use_3d else 4,
                width_min_pixels=2,
                cap_rounded=True,
                joint_rounded=True,
                pickable=True,
                id=f"{id_prefix}_centerline",
            )
        )

        # Icons at segment midpoints (separate records → own click type/id/color)
        icon_records = [
            {
                "type": r["icon_type"],
                "id": r["icon_id"],
                "position": r["icon_position"],
                "color": r["icon_color"],
                "name": r["icon_name"],
            }
            for r in records
        ]
        layers.append(
            pdk.Layer(
                "ScatterplotLayer",
                icon_records,
                get_position="position",
                get_radius=ClickConfig.SLOPE_ICON_MARKER_RADIUS,
                get_fill_color="color",
                pickable=True,
                auto_highlight=True,
                id=f"{id_prefix}_icons",
            )
        )

        return layers

    # =========================================================================
    # LIFT LAYERS
    # =========================================================================

    def _create_lift_layers(
        self, *, use_3d: bool = False, defect_ids: set[str] | None = None
    ) -> dict[str, list[pdk.Layer]]:
        """Create layers for lift cables, pylons, and icons.

        Args:
            use_3d: If True, use real elevations. If False, use flat z offsets.
            defect_ids: Lift ids with a connectivity defect — grayed out ("half-dead").

        Returns:
            Dict with 'pylons' and 'cables_icons' keys for separate z-ordering.
        """
        if not self.graph:
            return {"pylons": [], "cables_icons": []}

        defect_ids = defect_ids or set()
        cable_data = []
        pylon_data = []
        icon_data = []

        for lift_id, lift in self.graph.lifts.items():
            color = list(StyleConfig.LIFT_COLORS_RGBA[lift.lift_type])
            # Connectivity-defect lift → mute only the cable (the "line") toward gray ("half-dead").
            # The center icon keeps its full hue so it stays a clear clickable marker.
            cable_color = StyleConfig.gray_out(color) if lift_id in defect_ids else color

            # Use pre-computed cable points with sag (from Lift.calculate_cable_points)
            cable_path = [
                [
                    pt.lon,
                    pt.lat,
                    self._get_z(
                        elevation=pt.elevation,
                        z_offset=MarkerConfig.PATH_Z_OFFSET_M,
                        use_3d=use_3d,
                        flat_z=MapConfig.Z_OFFSET_2D_LIFTS,
                    ),
                ]
                for pt in lift.cable_points
            ]

            cable_data.append(
                {
                    "type": ClickConfig.TYPE_LIFT,
                    "id": lift_id,
                    "path": cable_path,
                    "color": cable_color,
                    "name": f"{StyleConfig.LIFT_ICONS[lift.lift_type]} {lift.name}",
                    "lift_type": lift.lift_type,
                }
            )

            # Pylon markers at top of each pylon
            for i, pylon in enumerate(lift.pylons):
                pylon_z = self._get_z(
                    elevation=pylon.top_elevation_m,
                    z_offset=MarkerConfig.MARKER_Z_OFFSET_M,
                    use_3d=use_3d,
                    flat_z=MapConfig.Z_OFFSET_2D_PYLONS,
                )
                pylon_data.append(
                    {
                        "type": ClickConfig.TYPE_PYLON,
                        "lift_id": lift_id,
                        "pylon_index": i,  # 0-indexed
                        "position": [pylon.lon, pylon.lat, pylon_z],
                        "color": MarkerConfig.PYLON_MARKER_COLOR,
                        "name": f"Pylon {i + 1} on {StyleConfig.LIFT_ICONS[lift.lift_type]} {lift.name}",
                    }
                )

            # Lift icon at the ACTUAL cable midpoint. MARKER_Z_OFFSET_M then sits it a fixed height over the cable.
            cable_mid = lift.cable_points[len(lift.cable_points) // 2]
            mid_lon, mid_lat = cable_mid.lon, cable_mid.lat
            icon_z = self._get_z(
                elevation=cable_mid.elevation,
                z_offset=MarkerConfig.MARKER_Z_OFFSET_M,
                use_3d=use_3d,
                flat_z=MapConfig.Z_OFFSET_2D_ICONS,
            )
            icon_data.append(
                {
                    "type": ClickConfig.TYPE_LIFT,
                    "id": lift_id,
                    "position": [mid_lon, mid_lat, icon_z],
                    "color": color,
                    "name": f"{StyleConfig.LIFT_ICONS[lift.lift_type]} {lift.name}",
                    "lift_type": lift.lift_type,
                }
            )

        pylon_layers = []
        cable_icon_layers = []

        # Pylon markers (separate for z-ordering - very back)
        if pylon_data:
            pylon_layers.append(
                pdk.Layer(
                    "ScatterplotLayer",
                    pylon_data,
                    get_position="position",
                    get_radius=ClickConfig.PYLON_MARKER_RADIUS,
                    get_fill_color="color",
                    get_line_color=MarkerConfig.PYLON_BORDER_COLOR,
                    stroked=True,
                    line_width_min_pixels=2,
                    pickable=True,
                    id="lift_pylons",
                )
            )

        # Cable lines — CABLE_WIDTH is in metres (deck.gl PathLayer default), so the lift is a 10m-wide
        # ribbon that drapes over terrain in 3D. Do NOT pass width_units (pydeck mangles string props).
        if cable_data:
            cable_icon_layers.append(
                pdk.Layer(
                    "PathLayer",
                    cable_data,
                    get_path="path",
                    get_color="color",
                    get_width=MarkerConfig.CABLE_WIDTH,
                    width_min_pixels=MarkerConfig.CABLE_MIN_PIXELS,  # Minimum visible width when zoomed out
                    pickable=True,
                    id="lift_cables",
                )
            )

        # Lift icons
        if icon_data:
            cable_icon_layers.append(
                pdk.Layer(
                    "ScatterplotLayer",
                    icon_data,
                    get_position="position",
                    get_radius=ClickConfig.LIFT_ICON_MARKER_RADIUS,
                    get_fill_color="color",
                    pickable=True,
                    auto_highlight=True,
                    id="lift_icons",
                )
            )

        return {"pylons": pylon_layers, "cables_icons": cable_icon_layers}

    # =========================================================================
    # NODE LAYER
    # =========================================================================

    def _create_node_layer(self, *, use_3d: bool = False, selected_node_ids: list[str] | None = None) -> pdk.Layer:
        """Create layer for junction nodes.

        A node that is also a **parking node** (a road junction shared with a
        slope or lift, per graph.get_parking_nodes()) renders bigger and blue
        with a "Parking place" tooltip — the parking marker IS the node marker,
        so it's always visible and hoverable (no separate under-layer).

        A node in `selected_node_ids` (highlighted by merge/delete OR the route planner as start) renders
        RED and bigger, so the user sees exactly which nodes are selected. Selection takes priority over
        the parking style (a selected parking node still shows red while selected).

        Args:
            use_3d: If True, use terrain elevation. If False, use z-offset.
            selected_node_ids: Node ids currently selected (merge/delete/route start) — drawn red.

        Returns:
            ScatterplotLayer with nodes; per-point color/radius/name.
        """
        if not self.graph:
            return pdk.Layer("ScatterplotLayer", [], id="nodes")

        parking_ids = {n.id for n in self.graph.get_parking_nodes()}
        selected_ids = set(selected_node_ids) if selected_node_ids is not None else set()

        node_data = []
        for node_id, node in self.graph.nodes.items():
            is_parking = node_id in parking_ids
            is_selected = node_id in selected_ids
            is_big = is_selected or is_parking
            if is_selected:
                color = list(StyleConfig.SELECTED_NODE_RGBA)
                name = f"✅ Selected — {node_id}"
            elif is_parking:
                color = list(StyleConfig.PARKING_COLOR_RGBA)
                name = f"{StyleConfig.PARKING_ICON} Parking place — {node_id}"
            else:
                color = list(MarkerConfig.NODE_MARKER_COLOR)
                name = f"Node {node_id}"
            # A big node (merge-red or parking-blue) sits one step below plain nodes so the smaller
            # plain nodes stay on top and clickable where markers overlap in a cluster.
            radius = ClickConfig.NODE_MARKER_RADIUS_BIG if is_big else ClickConfig.NODE_MARKER_RADIUS
            flat_z = MapConfig.Z_OFFSET_2D_NODE_BIG if is_big else MapConfig.Z_OFFSET_2D_NODES
            node_data.append(
                {
                    "type": ClickConfig.TYPE_NODE,
                    "id": node_id,
                    "position": [
                        node.lon,
                        node.lat,
                        self._get_z(
                            elevation=node.elevation,
                            z_offset=MarkerConfig.MARKER_Z_OFFSET_M,
                            use_3d=use_3d,
                            flat_z=flat_z,
                        ),
                    ],
                    "elevation": node.elevation,
                    "color": color,
                    "radius": radius,
                    "name": name,
                }
            )

        return pdk.Layer(
            "ScatterplotLayer",
            node_data,
            get_position="position",
            get_radius="radius",
            get_fill_color="color",
            get_line_color=MarkerConfig.NODE_MARKER_BORDER,
            stroked=True,
            line_width_min_pixels=2,
            pickable=True,
            auto_highlight=True,
            highlight_color=[255, 255, 0, 180],
            id="nodes",
        )

    # =========================================================================
    # PROPOSAL LAYERS
    # =========================================================================

    def _create_proposal_layers(
        self,
        proposals: list[ProposedPathSegment],
        selected_idx: int | None,
        *,
        is_custom_path: bool = False,
        use_3d: bool = False,
    ) -> list[pdk.Layer]:
        """Create layers for proposed paths with selection markers.

        Args:
            proposals: List of proposed slope segments.
            selected_idx: Index of selected proposal, or None.
            is_custom_path: Whether this is a custom path (no endpoint markers).
            use_3d: If True, use real elevations. If False, use flat z offsets.
        """
        path_data: list[dict[str, object]] = []
        endpoint_data: list[dict[str, object]] = []
        body_data: list[dict[str, object]] = []

        # Proposals use marker z-offset for 2D mode
        z_offset_2d = MapConfig.Z_OFFSET_2D_MARKERS

        for i, proposal in enumerate(proposals):
            if not proposal.points:
                continue

            is_selected = selected_idx is not None and i == selected_idx
            # Road proposals are brown (translucent → solid when selected); slope
            # proposals are difficulty-colored. SegmentKind is a StrEnum → `==` is reload-safe.
            if proposal.kind == SegmentKind.ROAD:
                color = list(StyleConfig.ROAD_PROPOSAL_COLOR_RGBA)
            elif proposal.kind == SegmentKind.SLOPE:
                color = list(StyleConfig.SLOPE_COLORS_RGBA[proposal.difficulty])
            else:
                raise ValueError(f"Unexpected {proposal.kind=}")

            # Adjust for selection state
            if is_selected:
                color[3] = 255  # Full opacity
                width = 6
            else:
                color[3] = 150  # Semi-transparent
                width = 3

            path_data.append(
                {
                    "type": ClickConfig.TYPE_PROPOSAL_BODY,  # Clicking path = select
                    "id": f"path_{i}",  # Unique ID for click deduplication
                    "proposal_index": i,
                    "path": [
                        [
                            p.lon,
                            p.lat,
                            self._get_z(
                                elevation=p.elevation,
                                z_offset=MarkerConfig.PATH_Z_OFFSET_M,
                                use_3d=use_3d,
                                flat_z=z_offset_2d,
                            ),
                        ]
                        for p in proposal.points
                    ],
                    "color": color,
                    "width": width,
                    "name": f"Proposal {i + 1}",
                    "difficulty": proposal.difficulty,
                    "slope_pct": proposal.avg_slope_pct,
                    "length_m": proposal.length_m,
                }
            )

            # Start marker
            start_pt = proposal.points[0]
            body_data.append(
                {
                    "type": ClickConfig.TYPE_START_MARKER,
                    "position": [
                        start_pt.lon,
                        start_pt.lat,
                        self._get_z(
                            elevation=start_pt.elevation,
                            z_offset=MarkerConfig.MARKER_Z_OFFSET_M,
                            use_3d=use_3d,
                            flat_z=z_offset_2d,
                        ),
                    ],
                    "color": [255, 255, 255, 200],
                    "elevation": start_pt.elevation,
                    "name": f"Start: {start_pt.elevation:.0f}m",
                }
            )

            # Body marker at midpoint (for selection)
            mid_idx = len(proposal.points) // 2
            mid_pt = proposal.points[mid_idx]
            body_data.append(
                {
                    "type": ClickConfig.TYPE_PROPOSAL_BODY,
                    "id": f"body_{i}",  # Unique ID for click deduplication
                    "proposal_index": i,
                    "position": [
                        mid_pt.lon,
                        mid_pt.lat,
                        self._get_z(
                            elevation=mid_pt.elevation,
                            z_offset=MarkerConfig.MARKER_Z_OFFSET_M,
                            use_3d=use_3d,
                            flat_z=z_offset_2d,
                        ),
                    ],
                    "color": color,
                    "name": f"Select Proposal {i + 1}",
                    "difficulty": proposal.difficulty,
                    "slope_pct": proposal.avg_slope_pct,
                }
            )

            # Endpoint marker (for commit) - skip for custom connect
            if not is_custom_path:
                end_pt = proposal.points[-1]
                endpoint_data.append(
                    {
                        "type": ClickConfig.TYPE_PROPOSAL_ENDPOINT,
                        "id": f"endpoint_{i}",  # Unique ID for click deduplication
                        "proposal_index": i,
                        "position": [
                            end_pt.lon,
                            end_pt.lat,
                            self._get_z(
                                elevation=end_pt.elevation,
                                z_offset=MarkerConfig.MARKER_Z_OFFSET_M,
                                use_3d=use_3d,
                                flat_z=z_offset_2d,
                            ),
                        ],
                        "color": ClickConfig.PROPOSAL_ENDPOINT_COLOR,
                        "elevation": end_pt.elevation,
                        "name": f"Commit Proposal {i + 1}",
                    }
                )

        layers = []

        # Proposal paths (NOT pickable - use markers for selection/commit)
        if path_data:
            logger.debug(
                f"[RENDER] proposal layer: {len(path_data)} path(s), is_custom_path={is_custom_path}, "
                f"first_color={path_data[0]['color']}"
            )
            layers.append(
                pdk.Layer(
                    "PathLayer",
                    path_data,
                    get_path="path",
                    get_color="color",
                    get_width="width",
                    width_min_pixels=2,
                    cap_rounded=True,
                    joint_rounded=True,
                    pickable=True,
                    id="proposal_paths",
                )
            )

        # Body markers (selection)
        if body_data:
            layers.append(
                pdk.Layer(
                    "ScatterplotLayer",
                    body_data,
                    get_position="position",
                    get_radius=ClickConfig.PROPOSAL_BODY_RADIUS,
                    get_fill_color="color",
                    pickable=True,
                    auto_highlight=True,
                    id="proposal_bodies",
                )
            )

        # Endpoint markers (commit)
        if endpoint_data:
            layers.append(
                pdk.Layer(
                    "ScatterplotLayer",
                    endpoint_data,
                    get_position="position",
                    get_radius=ClickConfig.PROPOSAL_ENDPOINT_RADIUS,
                    get_fill_color="color",
                    get_line_color=[255, 255, 255, 255],
                    stroked=True,
                    line_width_min_pixels=2,
                    pickable=True,
                    auto_highlight=True,
                    highlight_color=[255, 200, 0, 255],
                    id="proposal_endpoints",
                )
            )

        return layers

    # =========================================================================
    # ORIENTATION ARROWS
    # =========================================================================

    def create_orientation_arrows_layers(
        self,
        lat: float,
        lon: float,
        elevation: float,
        orientation: "TerrainOrientation",
        *,
        use_3d: bool = False,
    ) -> list[pdk.Layer]:
        """Create arrow layers showing fall line and contours at selection point.

        Args:
            lat, lon: Center position
            elevation: Terrain elevation at center
            orientation: Terrain orientation data
            use_3d: If True, render at terrain elevation. If False, render flat.
        """
        arrow_data = []
        arrow_z = self._get_z(
            elevation=elevation,
            z_offset=MarkerConfig.MARKER_Z_OFFSET_M,
            use_3d=use_3d,
            flat_z=MapConfig.Z_OFFSET_2D_MARKERS,
        )

        def _append_arrow(bearing_deg: float, arrow_color: list[int], name: str) -> None:
            end_lon, end_lat = GeoCalculator.destination(
                lon=lon,
                lat=lat,
                bearing_deg=bearing_deg,
                distance_m=MarkerConfig.ORIENTATION_ARROW_LENGTH_M,
            )
            arrow_data.append(
                {
                    "path": [[lon, lat, arrow_z], [end_lon, end_lat, arrow_z]],
                    "color": arrow_color,
                    "name": name,
                }
            )

        # Fall line arrow (difficulty colored)
        if orientation.fall_line is not None:
            # difficulty_color is a hex string; map it back to the matching RGBA, defaulting to green.
            color = next(
                (
                    list(rgba)
                    for name, rgba in StyleConfig.SLOPE_COLORS_RGBA.items()
                    if StyleConfig.SLOPE_COLORS[name] == orientation.difficulty_color
                ),
                list(StyleConfig.SLOPE_COLORS_RGBA["green"]),
            )
            _append_arrow(orientation.fall_line, color, "Fall line")

        # Contour arrows (gray)
        for bearing in [orientation.contour_left, orientation.contour_right]:
            _append_arrow(bearing, MarkerConfig.ORIENTATION_CONTOUR_COLOR, "Contour")

        layers = []

        if arrow_data:
            layers.append(
                pdk.Layer(
                    "PathLayer",
                    arrow_data,
                    get_path="path",
                    get_color="color",
                    get_width=MarkerConfig.DIRECTION_ARROW_WIDTH,
                    width_min_pixels=3,
                    cap_rounded=True,
                    id="orientation_arrows",
                )
            )

            # Center marker
            layers.append(
                pdk.Layer(
                    "ScatterplotLayer",
                    [{"position": [lon, lat, arrow_z], "name": "Selection point"}],
                    get_position="position",
                    get_radius=MarkerConfig.DIRECTION_CENTER_MARKER_RADIUS,
                    get_fill_color=arrow_data[0]["color"],
                    get_line_color=[255, 255, 255, 255],
                    stroked=True,
                    line_width_min_pixels=3,
                    id="orientation_center",
                )
            )

        return layers

    # =========================================================================
    # DIRECTION ARROWS
    # =========================================================================

    def create_direction_arrow_layer(
        self,
        start_lat: float,
        start_lon: float,
        bearing_deg: float,
        direction: str = "downhill",
        *,
        use_3d: bool = False,
    ) -> pdk.Layer:
        """Create directional arrow from a point.

        Args:
            start_lat, start_lon: Starting point
            bearing_deg: Direction in degrees
            direction: "downhill" (green) or "uphill" (purple)
            use_3d: If True, render at terrain elevation. If False, render flat.
        """
        if direction == "uphill":
            color = MarkerConfig.DIRECTION_ARROW_COLOR_UPHILL
        else:
            color = MarkerConfig.DIRECTION_ARROW_COLOR_DOWNHILL

        end_lon, end_lat = GeoCalculator.destination(
            lon=start_lon,
            lat=start_lat,
            bearing_deg=bearing_deg,
            distance_m=MarkerConfig.DIRECTION_ARROW_LENGTH_M,
        )

        arrow_z = MapConfig.Z_OFFSET_2D_MARKERS if not use_3d else 0

        arrow_data = [
            {
                "path": [[start_lon, start_lat, arrow_z], [end_lon, end_lat, arrow_z]],
                "color": color,
                "name": f"{'Uphill' if direction == 'uphill' else 'Downhill'} direction",
            }
        ]

        return pdk.Layer(
            "PathLayer",
            arrow_data,
            get_path="path",
            get_color="color",
            get_width=MarkerConfig.DIRECTION_ARROW_WIDTH,
            width_min_pixels=3,
            cap_rounded=True,
            id=f"direction_arrow_{direction}",
        )

    # =========================================================================
    # LIFT PLACEMENT MARKER
    # =========================================================================

    def create_pending_lift_marker_layers(
        self,
        lat: float,
        lon: float,
        elevation: float,
        fall_line_bearing: float,
        *,
        use_3d: bool = False,
    ) -> list[pdk.Layer]:
        """Create marker for pending lift placement with terrain fall-line arrow.

        Args:
            lat, lon: First station location
            elevation: Ground elevation
            fall_line_bearing: Downhill direction (arrow shows the uphill terrain, opposite)
            use_3d: If True, render at terrain elevation. If False, render flat.
        """
        layers = []
        marker_z = self._get_z(
            elevation=elevation,
            z_offset=MarkerConfig.MARKER_Z_OFFSET_M,
            use_3d=use_3d,
            flat_z=MapConfig.Z_OFFSET_2D_MARKERS,
        )

        # Station marker
        station_data = [
            {
                "position": [lon, lat, marker_z],
                "elevation": elevation,
                "name": f"First Station ({elevation:.0f}m)",
            }
        ]

        layers.append(
            pdk.Layer(
                "ScatterplotLayer",
                station_data,
                get_position="position",
                get_radius=MarkerConfig.LIFT_STATION_RADIUS,
                get_fill_color=MarkerConfig.LIFT_STATION_COLOR,
                get_line_color=[255, 255, 255, 255],
                stroked=True,
                line_width_min_pixels=3,
                id="pending_lift_station",
            )
        )

        # Uphill direction arrow
        uphill_bearing = (fall_line_bearing + 180) % 360
        layers.append(
            self.create_direction_arrow_layer(
                start_lat=lat,
                start_lon=lon,
                bearing_deg=uphill_bearing,
                direction="uphill",
                use_3d=use_3d,
            )
        )

        return layers

    def _path_layer(
        self,
        polyline: "Sequence[tuple[float, float, float]]",
        *,
        color: list[int],
        width_m: float,
        float_above_m: float,
        use_3d: bool,
        layer_id: str,
        name: str = "",
    ) -> list[pdk.Layer]:
        """One floated PathLayer over a (lon,lat,elev) polyline — the shared body for the route overlay and
        the flythrough highlight. In 3D it hovers `float_above_m` above terrain; flat in 2D.
        """
        assert len(polyline) >= 2, f"a path layer needs ≥2 points, got {len(polyline)}"
        z_offset = MarkerConfig.PATH_Z_OFFSET_M + (float_above_m if use_3d else 0)
        path = [
            [lon, lat, self._get_z(elevation=elev, z_offset=z_offset, use_3d=use_3d)] for lon, lat, elev in polyline
        ]
        return [
            pdk.Layer(
                "PathLayer",
                [{"path": path, "color": color, "name": name}],
                get_path="path",
                get_color="color",
                get_width=width_m,
                width_min_pixels=6,
                cap_rounded=True,
                id=layer_id,
            )
        ]

    def create_route_layers(self, route: "Route", *, use_3d: bool) -> list[pdk.Layer]:
        """One thick, semi-transparent polyline for the selected route, tracing the actual slope geometry.
        Wider than any slope belt (ROUTE_WIDTH_M) so it reads as an overlay; in 3D it floats
        ROUTE_FLOAT_ABOVE_M above the pistes/lifts it traces. Colour is keyed to its criterion.
        """
        return self._path_layer(
            route.path_points,
            color=route.color,
            width_m=RoutePlannerConfig.ROUTE_WIDTH_M,
            float_above_m=RoutePlannerConfig.ROUTE_FLOAT_ABOVE_M,
            use_3d=use_3d,
            layer_id="route_selected",
            name=", ".join(c.value for c in route.criteria),
        )

    def create_highlight_ribbon(
        self, polyline: "Sequence[tuple[float, float, float]]", *, use_3d: bool
    ) -> list[pdk.Layer]:
        """The hot-orange signal ribbon over the element currently being flown, floated
        FLYTHROUGH_HIGHLIGHT_FLOAT_ABOVE_M (just above the route overlay). Follows the element's real path.
        """
        return self._path_layer(
            polyline,
            color=MapConfig.FLYTHROUGH_HIGHLIGHT_COLOR,
            width_m=RoutePlannerConfig.ROUTE_WIDTH_M,
            float_above_m=MapConfig.FLYTHROUGH_HIGHLIGHT_FLOAT_ABOVE_M,
            use_3d=use_3d,
            layer_id="flythrough_highlight",
        )

    def create_import_bbox_layers(
        self,
        center_lon: float,
        center_lat: float,
        half_width_m: float,
        elevation: float,
        *,
        use_3d: bool = False,
    ) -> list[pdk.Layer]:
        """Draw the OSM import box: a translucent square + a pickable center dot (re-click = confirm).

        The rectangle is the region that will be fetched (corners from bbox_around, same maths the
        import uses). The center dot carries ClickConfig.TYPE_IMPORT_CENTER so a click on it is
        classified as MarkerType.IMPORT_CENTER and routed to confirm. PolygonLayer is 2D-only, which
        is fine — import is a top-down action.

        Args:
            center_lon, center_lat: Placed box center.
            half_width_m: Half the box side length in metres (the slider value × 1000).
            elevation: Ground elevation at the center (for the dot's z in 3D).
            use_3d: If True, place the dot at terrain elevation.
        """
        min_lon, min_lat, max_lon, max_lat = bbox_around(
            center_lon=center_lon, center_lat=center_lat, half_width_m=half_width_m
        )
        ring = [
            [min_lon, min_lat],
            [max_lon, min_lat],
            [max_lon, max_lat],
            [min_lon, max_lat],
            [min_lon, min_lat],
        ]
        marker_z = self._get_z(
            elevation=elevation,
            z_offset=MarkerConfig.MARKER_Z_OFFSET_M,
            use_3d=use_3d,
            flat_z=MapConfig.Z_OFFSET_2D_MARKERS,
        )
        return [
            pdk.Layer(
                "PolygonLayer",
                [{"polygon": ring}],
                get_polygon="polygon",
                get_fill_color=list(StyleConfig.IMPORT_BOX_RGBA),
                get_line_color=list(StyleConfig.IMPORT_BOX_RGBA),
                line_width_min_pixels=2,
                id="import_bbox",
            ),
            pdk.Layer(
                "ScatterplotLayer",
                [
                    {
                        "type": ClickConfig.TYPE_IMPORT_CENTER,
                        "position": [center_lon, center_lat, marker_z],
                        "name": f"{StyleConfig.IMPORT_ICON} Import center — click to confirm",
                    }
                ],
                get_position="position",
                get_radius=MarkerConfig.LIFT_STATION_RADIUS,
                get_fill_color=list(StyleConfig.IMPORT_CENTER_RGBA),
                pickable=True,
                auto_highlight=True,
                id="import_center",
            ),
        ]

    # =========================================================================
    # TOOLTIP CONFIGURATION
    # =========================================================================

    def _create_tooltip_config(self) -> dict[str, str | dict[str, str]]:
        """Create Pydeck tooltip configuration - name only, details in side panel."""
        return {
            "html": "<b>{name}</b>",
            "style": {
                "backgroundColor": "rgba(255, 255, 255, 0.95)",
                "color": "#333",
                "padding": "6px 10px",
                "borderRadius": "4px",
            },
        }

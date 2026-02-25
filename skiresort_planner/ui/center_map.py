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
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pydeck as pdk

from skiresort_planner.constants import (
    ClickConfig,
    MapConfig,
    MarkerConfig,
    StyleConfig,
)
from skiresort_planner.core.geo_calculator import GeoCalculator
from skiresort_planner.model.proposed_path import ProposedSlopeSegment
from skiresort_planner.model.resort_graph import ResortGraph

if TYPE_CHECKING:
    from skiresort_planner.core.terrain_analyzer import TerrainOrientation

logger = logging.getLogger(__name__)


@dataclass
class LayerCollection:
    """Manages Pydeck layers with correct z-ordering.

    Z-order (back to front): terrain → pylons → slopes → lifts → nodes → proposals → markers

    Nodes are placed AFTER slopes/lifts so they:
    1. Render visually on top of lines (correct for junction display)
    2. Get click priority over slopes/lifts (nodes are small, need priority)

    Markers layer includes crosshair for terrain node placement when active.
    """

    terrain: list[pdk.Layer] = field(default_factory=list)
    pylons: list[pdk.Layer] = field(default_factory=list)
    slopes: list[pdk.Layer] = field(default_factory=list)
    lifts: list[pdk.Layer] = field(default_factory=list)
    nodes: list[pdk.Layer] = field(default_factory=list)
    proposals: list[pdk.Layer] = field(default_factory=list)
    markers: list[pdk.Layer] = field(default_factory=list)

    def get_ordered_layers(self) -> list[pdk.Layer]:
        """Return all layers in correct z-order (back to front)."""
        return self.terrain + self.pylons + self.slopes + self.lifts + self.nodes + self.proposals + self.markers


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
        zoom: int = MapConfig.DEFAULT_ZOOM,
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

    def get_view_state(self) -> pdk.ViewState:
        """Create Pydeck ViewState from current settings."""
        return pdk.ViewState(
            latitude=self.center_lat,
            longitude=self.center_lon,
            zoom=self.zoom,
            pitch=self.pitch,
            bearing=self.bearing,
        )

    def update_view(
        self,
        lat: float | None = None,
        lon: float | None = None,
        zoom: int | None = None,
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

    def render(
        self,
        proposals: list[ProposedSlopeSegment] | None = None,
        selected_proposal_idx: int | None = None,
        show_nodes: bool = True,
        show_segments: bool = True,
        show_lifts: bool = True,
        highlight_segment_ids: list[str] | None = None,
        is_custom_path: bool = False,
        extra_layers: list[pdk.Layer] | None = None,
        terrain_layer: pdk.Layer | None = None,
        use_3d: bool = False,
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

        Returns:
            pdk.Deck object ready for display.

        Z-order (back to front): terrain → pylons → slopes → lifts → nodes → proposals → markers
        """
        layer_collection = LayerCollection()

        # Basemap layer (2D TileLayer or 3D TerrainLayer) always at bottom
        if terrain_layer:
            layer_collection.terrain.append(terrain_layer)

        if self.graph:
            if show_lifts:
                lift_layers = self._create_lift_layers(use_3d=use_3d)
                layer_collection.pylons.extend(lift_layers["pylons"])
                layer_collection.lifts.extend(lift_layers["cables_icons"])

            if show_nodes:
                layer_collection.nodes.append(self._create_node_layer(use_3d=use_3d))

            if show_segments:
                layer_collection.slopes.extend(
                    self._create_segment_layers(highlight_ids=highlight_segment_ids, use_3d=use_3d)
                )

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
    def _get_z(elevation: float, z_offset: float, use_3d: bool, flat_z: float = 0.0) -> float:
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
    ) -> tuple[float, float, float, int, float]:
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

        # Calculate average elevation and adjust zoom accordingly
        # Higher elevation = slightly more zoomed out to keep camera above terrain
        avg_elevation = (start_elev + end_elev) / 2
        # Gentle adjustment: every 1000m elevation, zoom out by 0.5 level
        elevation_zoom_adjustment = avg_elevation / 2000.0
        adjusted_zoom = max(MapConfig.VIEW_3D_MIN_ZOOM, MapConfig.VIEW_3D_ZOOM - elevation_zoom_adjustment)

        return (center_lat, center_lon, camera_bearing, int(adjusted_zoom), MapConfig.VIEW_3D_PITCH)

    @staticmethod
    def calculate_3d_view_for_slope(
        graph: ResortGraph,
        slope_id: str,
    ) -> tuple[float, float, float, int, float]:
        """Calculate optimal camera position to view a slope in 3D.

        Positions camera perpendicular to slope direction so it's viewed from the side.
        Slope appears from left-up to right-down (start_node on left, end_node on right).

        Args:
            graph: Resort graph containing the slope.
            slope_id: ID of the slope to view.

        Returns:
            Tuple (lat, lon, bearing, zoom, pitch) for camera settings.
        """
        slope = graph.slopes.get(slope_id)
        if not slope:
            raise ValueError(f"Slope {slope_id} not found")

        start_node = graph.nodes.get(slope.start_node_id)
        end_node = graph.nodes.get(slope.end_node_id)
        if not start_node or not end_node:
            raise ValueError(f"Slope {slope_id} has missing nodes")

        # Camera offset -90: positions start_node on LEFT, end_node on RIGHT
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
    def calculate_3d_view_for_lift(
        graph: ResortGraph,
        lift_id: str,
    ) -> tuple[float, float, float, int, float]:
        """Calculate optimal camera position to view a lift in 3D.

        Positions camera perpendicular to lift direction so it's viewed from the side.
        Lift appears from left-down to right-up (start_node on left, end_node on right).

        Args:
            graph: Resort graph containing the lift.
            lift_id: ID of the lift to view.

        Returns:
            Tuple (lat, lon, bearing, zoom, pitch) for camera settings.
        """
        lift = graph.lifts.get(lift_id)
        if not lift:
            raise ValueError(f"Lift {lift_id} not found")

        start_node = graph.nodes.get(lift.start_node_id)
        end_node = graph.nodes.get(lift.end_node_id)
        if not start_node or not end_node:
            raise ValueError(f"Lift {lift_id} has missing nodes")

        # Camera offset -90: positions start_node on LEFT, end_node on RIGHT
        return MapRenderer._calculate_3d_view_for_endpoints(
            start_lat=start_node.lat,
            start_lon=start_node.lon,
            start_elev=start_node.elevation,
            end_lat=end_node.lat,
            end_lon=end_node.lon,
            end_elev=end_node.elevation,
            camera_bearing_offset=-90,
        )

    # =========================================================================
    # SEGMENT LAYERS
    # =========================================================================

    def _create_segment_layers(self, highlight_ids: list[str] | None = None, use_3d: bool = False) -> list[pdk.Layer]:
        """Create layers for segment belt polygons and center lines."""
        if not self.graph:
            return []

        highlight_ids = highlight_ids or []
        segment_data = []
        icon_data = []

        for seg_id, segment in self.graph.segments.items():
            polygon_coords = segment.get_belt_polygon()
            if not polygon_coords:
                continue

            # Get parent slope for consistent coloring
            parent_slope = self.graph.get_slope_by_segment_id(segment_id=seg_id)
            if parent_slope:
                difficulty = parent_slope.get_difficulty(segments=self.graph.segments)
                slope_id = parent_slope.id
            else:
                # Orphan segment (being built) - no parent slope yet
                difficulty = segment.difficulty
                slope_id = None

            color = list(StyleConfig.SLOPE_COLORS_RGBA[difficulty])
            is_highlighted = seg_id in highlight_ids

            # Adjust opacity for highlight
            if is_highlighted:
                color[3] = 180  # More opaque
            else:
                color[3] = 100  # Semi-transparent

            # Belt polygon data - 2D polygons stay at z=0 (no elevation)
            segment_data.append(
                {
                    "type": ClickConfig.TYPE_SEGMENT,
                    "id": seg_id,
                    "slope_id": slope_id if parent_slope else None,
                    "polygon": list(polygon_coords),  # [[lon, lat], ...] - 2D for PolygonLayer
                    "center_line": [
                        [
                            p.lon,
                            p.lat,
                            self._get_z(
                                p.elevation, MarkerConfig.PATH_Z_OFFSET_M, use_3d, MapConfig.Z_OFFSET_2D_SLOPES
                            ),
                        ]
                        for p in segment.points
                    ],
                    "color": color,
                    "highlighted": is_highlighted,
                    "difficulty": difficulty,
                    "name": f"Segment {seg_id}",
                }
            )

            # Slope icon at segment midpoint - only for finished slopes (not orphan segments)
            if parent_slope:
                center_line = segment.points
                if center_line:
                    mid_idx = len(center_line) // 2
                    mid_pt = center_line[mid_idx]
                    icon_z = self._get_z(
                        mid_pt.elevation, MarkerConfig.MARKER_Z_OFFSET_M, use_3d, MapConfig.Z_OFFSET_2D_ICONS
                    )
                    icon_data.append(
                        {
                            "type": ClickConfig.TYPE_SLOPE,
                            "id": slope_id,
                            "position": [mid_pt.lon, mid_pt.lat, icon_z],
                            "color": StyleConfig.SLOPE_COLORS_RGBA[difficulty],
                            "name": f"Slope {slope_id}",
                            "difficulty": difficulty,
                        }
                    )

        layers = []

        # Belt polygons - ONLY in 2D mode (PolygonLayer doesn't support z-coords)
        # In 3D mode, polygons render at z=0 which looks wrong
        if segment_data and not use_3d:
            layers.append(
                pdk.Layer(
                    "PolygonLayer",
                    segment_data,
                    get_polygon="polygon",
                    get_fill_color="color",
                    get_line_color=[255, 255, 255, 100],
                    line_width_min_pixels=1,
                    pickable=True,
                    auto_highlight=True,
                    highlight_color=[255, 255, 255, 80],
                    id="segments_belt",
                )
            )

        # Center lines (colored by difficulty) - render in both 2D and 3D
        if segment_data:
            layers.append(
                pdk.Layer(
                    "PathLayer",
                    segment_data,
                    get_path="center_line",
                    get_color="color",
                    get_width=6 if use_3d else 4,  # Thicker in 3D since no belt polygon
                    width_min_pixels=2,
                    pickable=True,
                    id="segments_centerline",
                )
            )

        # Slope icons
        if icon_data:
            layers.append(
                pdk.Layer(
                    "ScatterplotLayer",
                    icon_data,
                    get_position="position",
                    get_radius=ClickConfig.SLOPE_ICON_MARKER_RADIUS,
                    get_fill_color="color",
                    pickable=True,
                    auto_highlight=True,
                    id="slope_icons",
                )
            )

        return layers

    # =========================================================================
    # LIFT LAYERS
    # =========================================================================

    def _create_lift_layers(self, use_3d: bool = False) -> dict[str, list[pdk.Layer]]:
        """Create layers for lift cables, pylons, and icons.

        Args:
            use_3d: If True, use real elevations. If False, use flat z offsets.

        Returns:
            Dict with 'pylons' and 'cables_icons' keys for separate z-ordering.
        """
        if not self.graph:
            return {"pylons": [], "cables_icons": []}

        cable_data = []
        pylon_data = []
        icon_data = []

        for lift_id, lift in self.graph.lifts.items():
            start_node = self.graph.nodes.get(lift.start_node_id)
            end_node = self.graph.nodes.get(lift.end_node_id)

            if start_node is None or end_node is None:
                logger.warning(f"Lift {lift_id} has missing nodes, skipping")
                continue

            color = list(StyleConfig.LIFT_COLORS_RGBA[lift.lift_type])

            # Use pre-computed cable points with sag (from Lift.calculate_cable_points)
            cable_path = [
                [
                    pt.lon,
                    pt.lat,
                    self._get_z(pt.elevation, MarkerConfig.PATH_Z_OFFSET_M, use_3d, MapConfig.Z_OFFSET_2D_LIFTS),
                ]
                for pt in lift.cable_points
            ]

            cable_data.append(
                {
                    "type": ClickConfig.TYPE_LIFT,
                    "id": lift_id,
                    "path": cable_path,
                    "color": color,
                    "name": lift.name,
                    "lift_type": lift.lift_type,
                }
            )

            # Pylon markers at top of each pylon
            for i, pylon in enumerate(lift.pylons):
                pylon_z = self._get_z(
                    pylon.top_elevation_m, MarkerConfig.MARKER_Z_OFFSET_M, use_3d, MapConfig.Z_OFFSET_2D_PYLONS
                )
                pylon_data.append(
                    {
                        "type": ClickConfig.TYPE_PYLON,
                        "lift_id": lift_id,
                        "pylon_index": i,  # 0-indexed
                        "position": [pylon.lon, pylon.lat, pylon_z],
                        "color": MarkerConfig.PYLON_MARKER_COLOR,
                        "name": f"Pylon {i + 1} on {lift_id}",
                    }
                )

            # Lift icon at midpoint (average elevation)
            mid_lat = (start_node.lat + end_node.lat) / 2
            mid_lon = (start_node.lon + end_node.lon) / 2
            mid_elev = (start_node.elevation + end_node.elevation) / 2
            icon_z = self._get_z(mid_elev, MarkerConfig.MARKER_Z_OFFSET_M, use_3d, MapConfig.Z_OFFSET_2D_ICONS)
            icon_data.append(
                {
                    "type": ClickConfig.TYPE_LIFT,
                    "id": lift_id,
                    "position": [mid_lon, mid_lat, icon_z],
                    "color": color,
                    "name": lift.name,
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

        # Cable lines
        if cable_data:
            cable_icon_layers.append(
                pdk.Layer(
                    "PathLayer",
                    cable_data,
                    get_path="path",
                    get_color="color",
                    get_width=MarkerConfig.CABLE_WIDTH,
                    width_min_pixels=2,
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
                    get_radius=25,
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

    def _create_node_layer(self, use_3d: bool = False) -> pdk.Layer:
        """Create layer for junction nodes.

        Args:
            use_3d: If True, use terrain elevation. If False, use z-offset.

        Returns:
            ScatterplotLayer with nodes.
        """
        if not self.graph:
            return pdk.Layer("ScatterplotLayer", [], id="nodes")

        node_data = [
            {
                "type": ClickConfig.TYPE_NODE,
                "id": node_id,
                "position": [
                    node.lon,
                    node.lat,
                    self._get_z(node.elevation, MarkerConfig.MARKER_Z_OFFSET_M, use_3d, MapConfig.Z_OFFSET_2D_NODES),
                ],
                "elevation": node.elevation,
                "name": f"Node {node_id}",
            }
            for node_id, node in self.graph.nodes.items()
        ]

        return pdk.Layer(
            "ScatterplotLayer",
            node_data,
            get_position="position",
            get_radius=ClickConfig.NODE_MARKER_RADIUS,
            get_fill_color=MarkerConfig.NODE_MARKER_COLOR,
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
        proposals: list[ProposedSlopeSegment],
        selected_idx: int | None,
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
        path_data = []
        endpoint_data = []
        body_data = []

        # Proposals use marker z-offset for 2D mode
        z_offset_2d = MapConfig.Z_OFFSET_2D_MARKERS

        for i, proposal in enumerate(proposals):
            if not proposal.points:
                continue

            is_selected = selected_idx is not None and i == selected_idx
            color = list(StyleConfig.SLOPE_COLORS_RGBA[proposal.difficulty])

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
                        [p.lon, p.lat, self._get_z(p.elevation, MarkerConfig.PATH_Z_OFFSET_M, use_3d, z_offset_2d)]
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
                    "type": "start_marker",
                    "position": [
                        start_pt.lon,
                        start_pt.lat,
                        self._get_z(start_pt.elevation, MarkerConfig.MARKER_Z_OFFSET_M, use_3d, z_offset_2d),
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
                        self._get_z(mid_pt.elevation, MarkerConfig.MARKER_Z_OFFSET_M, use_3d, z_offset_2d),
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
                            self._get_z(end_pt.elevation, MarkerConfig.MARKER_Z_OFFSET_M, use_3d, z_offset_2d),
                        ],
                        "color": ClickConfig.PROPOSAL_ENDPOINT_COLOR,
                        "elevation": end_pt.elevation,
                        "name": f"Commit Proposal {i + 1}",
                    }
                )

        layers = []

        # Proposal paths (NOT pickable - use markers for selection/commit)
        if path_data:
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
        arrow_z = self._get_z(elevation, MarkerConfig.MARKER_Z_OFFSET_M, use_3d, MapConfig.Z_OFFSET_2D_MARKERS)

        # Fall line arrow (difficulty colored)
        if orientation.fall_line is not None:
            end_lon, end_lat = GeoCalculator.destination(
                lon=lon,
                lat=lat,
                bearing_deg=orientation.fall_line,
                distance_m=MarkerConfig.ORIENTATION_ARROW_LENGTH_M,
            )
            color = StyleConfig.SLOPE_COLORS_RGBA.get(
                orientation.difficulty_color.lower() if orientation.difficulty_color else "green",
                [34, 197, 94, 230],  # Default green if not found
            )
            arrow_data.append(
                {
                    "path": [[lon, lat, arrow_z], [end_lon, end_lat, arrow_z]],
                    "color": color,
                    "name": "Fall line",
                }
            )

        # Contour arrows (gray)
        for bearing in [orientation.contour_left, orientation.contour_right]:
            end_lon, end_lat = GeoCalculator.destination(
                lon=lon,
                lat=lat,
                bearing_deg=bearing,
                distance_m=MarkerConfig.ORIENTATION_ARROW_LENGTH_M,
            )
            arrow_data.append(
                {
                    "path": [[lon, lat, arrow_z], [end_lon, end_lat, arrow_z]],
                    "color": MarkerConfig.ORIENTATION_CONTOUR_COLOR,
                    "name": "Contour",
                }
            )

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
                    get_radius=12,
                    get_fill_color=arrow_data[0]["color"] if arrow_data else [255, 255, 255, 200],
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
        use_3d: bool = False,
    ) -> list[pdk.Layer]:
        """Create marker for pending lift placement with uphill arrow.

        Args:
            lat, lon: Bottom station location
            elevation: Ground elevation
            fall_line_bearing: Downhill direction (arrow shows opposite)
            use_3d: If True, render at terrain elevation. If False, render flat.
        """
        layers = []
        marker_z = self._get_z(elevation, MarkerConfig.MARKER_Z_OFFSET_M, use_3d, MapConfig.Z_OFFSET_2D_MARKERS)

        # Station marker
        station_data = [
            {
                "position": [lon, lat, marker_z],
                "elevation": elevation,
                "name": f"Bottom Station ({elevation:.0f}m)",
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

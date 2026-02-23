"""MapRenderer - Folium map rendering for ski resort planner.

Renders all resort elements on an interactive map:
- Base terrain tiles
- Committed slopes with difficulty-colored polygons
- Lifts as straight lines
- Proposed paths (dashed lines)
- Nodes as markers
- Click capture for interactions
- Terrain orientation arrows (fall line compass)
- Direction arrows for custom connect and lift placement modes
- Target markers for calculation feedback

Reference: DETAILS_UI.md for interaction patterns
"""

import logging
from typing import TYPE_CHECKING, Optional

import folium
from folium.plugins import Draw

from skiresort_planner.constants import ClickConfig, MapConfig, MarkerConfig, StyleConfig
from skiresort_planner.core.geo_calculator import GeoCalculator
from skiresort_planner.model.path_point import PathPoint
from skiresort_planner.model.resort_graph import (
    ProposedSlopeSegment,
    ResortGraph,
)

if TYPE_CHECKING:
    from skiresort_planner.core.terrain_analyzer import TerrainOrientation

logger = logging.getLogger(__name__)


class MapRenderer:
    """Renders ski resort graph on a Folium map.

    Example:
        renderer = MapRenderer(graph=graph)
        m = renderer.render()
        st_folium(m)
    """

    def __init__(
        self,
        graph: Optional[ResortGraph] = None,
        center_lat: float = MapConfig.START_CENTER_LAT,
        center_lon: float = MapConfig.START_CENTER_LON,
        zoom: int = MapConfig.DEFAULT_ZOOM,
    ) -> None:
        """Initialize map renderer.

        Args:
            graph: Resort graph to render (can set later)
            center_lat: Initial map center latitude
            center_lon: Initial map center longitude
            zoom: Initial zoom level
        """
        self.graph = graph
        self.center_lat = center_lat
        self.center_lon = center_lon
        self.zoom = zoom

    def render(
        self,
        proposals: Optional[list[ProposedSlopeSegment]] = None,
        selected_proposal_idx: Optional[int] = None,
        show_nodes: bool = True,
        show_segments: bool = True,
        show_slopes: bool = True,
        show_lifts: bool = True,
        highlight_segment_ids: Optional[list[str]] = None,
        is_custom_path: bool = False,
    ) -> folium.Map:
        """Render complete map with all layers.

        Args:
            proposals: Proposed paths to display (dashed lines)
            selected_proposal_idx: Index of highlighted proposal
            show_nodes: Whether to show node markers
            show_segments: Whether to show segment polygons
            show_slopes: Whether to show finished slope names
            show_lifts: Whether to show lift lines
            highlight_segment_ids: Segment IDs to highlight (active slope)

        Returns:
            Folium Map object ready for display.
        """
        m = folium.Map(
            location=[self.center_lat, self.center_lon],
            zoom_start=self.zoom,
            tiles="OpenStreetMap",
        )

        # Add terrain layer
        folium.TileLayer(
            tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            attr="Esri World Imagery",
            name="Satellite",
            overlay=False,
        ).add_to(m)

        # Add OpenTopoMap as option
        folium.TileLayer(
            tiles="https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png",
            attr="OpenTopoMap",
            name="Topo",
            overlay=False,
        ).add_to(m)

        folium.LayerControl().add_to(m)

        if self.graph:
            if show_segments:
                self._render_segments(m=m, highlight_ids=highlight_segment_ids)

            if show_lifts:
                self._render_lifts(m=m)

            if show_nodes:
                self._render_nodes(m=m)

        if proposals:
            self._render_proposals(
                m=m,
                proposals=proposals,
                selected_idx=selected_proposal_idx,
                is_custom_path=is_custom_path,
            )

        # Add draw control for click capture
        Draw(
            draw_options={
                "polyline": False,
                "polygon": False,
                "circle": False,
                "circlemarker": False,
                "rectangle": False,
                "marker": False,
            },
            edit_options={"edit": False, "remove": False},
        ).add_to(m)

        return m

    def _render_segments(
        self,
        m: folium.Map,
        highlight_ids: Optional[list[str]] = None,
    ) -> None:
        """Render segment belt polygons with parent slope's difficulty color."""
        if not self.graph:
            return

        highlight_ids = highlight_ids or []

        for seg_id, segment in self.graph.segments.items():
            polygon_coords = segment.get_belt_polygon()
            if not polygon_coords:
                continue

            # Use parent slope's difficulty color for consistent slope coloring
            parent_slope = self.graph.get_slope_by_segment_id(segment_id=seg_id)
            if parent_slope:
                color = StyleConfig.SLOPE_COLORS[parent_slope.get_difficulty(segments=self.graph.segments)]
            else:
                # Fallback to segment color if not part of a slope (building mode)
                color = StyleConfig.SLOPE_COLORS[segment.difficulty]

            # Style based on highlight status
            is_highlighted = seg_id in highlight_ids
            fill_opacity = 0.5 if is_highlighted else 0.3
            weight = 3 if is_highlighted else 1

            # Convert (lon, lat) to (lat, lon) for folium
            latlons = [(lat, lon) for lon, lat in polygon_coords]

            # No tooltip on polygon - info available on icon marker
            folium.Polygon(
                locations=latlons,
                color=color,
                weight=weight,
                fill=True,
                fill_color=color,
                fill_opacity=fill_opacity,
            ).add_to(m)

            # Render center line for clarity
            center_line = [(p.lat, p.lon) for p in segment.points]

            folium.PolyLine(
                locations=center_line,
                color="white",
                weight=2 if is_highlighted else 1,
                opacity=0.7,
            ).add_to(m)

            # Add difficulty icon at segment midpoint
            mid_idx = len(center_line) // 2
            if mid_idx < len(center_line):
                mid_lat, mid_lon = center_line[mid_idx]

                # Use parent slope ID if available, otherwise segment ID for building mode
                # Clicking during building will show InvalidClickMessage from handler
                if parent_slope is not None:
                    slope_id = parent_slope.id
                else:
                    slope_id = seg_id  # Use segment ID for building mode

                tooltip = f"{ClickConfig.TOOLTIP_PREFIX_SLOPE_ICON} {slope_id}"
                folium.CircleMarker(
                    location=(mid_lat, mid_lon),
                    radius=ClickConfig.SLOPE_ICON_MARKER_RADIUS,
                    color=color,
                    fill=True,
                    fill_color=color,
                    fill_opacity=0.9,
                    tooltip=tooltip,
                ).add_to(m)

    def _render_lifts(self, m: folium.Map) -> None:
        """Render lift lines with pylons."""
        if not self.graph:
            return

        for lift_id, lift in self.graph.lifts.items():
            start_node = self.graph.nodes.get(lift.start_node_id)
            end_node = self.graph.nodes.get(lift.end_node_id)

            if start_node is None:
                raise ValueError(f"Lift {lift_id} references non-existent start node {lift.start_node_id}")
            if end_node is None:
                raise ValueError(f"Lift {lift_id} references non-existent end node {lift.end_node_id}")

            lift_color = StyleConfig.LIFT_COLORS[lift.lift_type]

            # Build cable path: start station -> pylons -> end station
            cable_points = [(start_node.lat, start_node.lon)]
            for pylon in lift.pylons:
                cable_points.append((pylon.lat, pylon.lon))
            cable_points.append((end_node.lat, end_node.lon))

            # No tooltip on cable line - info available on lift icon marker
            folium.PolyLine(
                locations=cable_points,
                color=lift_color,
                weight=4,
            ).add_to(m)

            # Add pylon markers with user-friendly tooltips (1-indexed for display)
            for i, pylon in enumerate(lift.pylons, start=1):
                # User-friendly tooltip: "View Pylon 3 on L1" (1-indexed)
                pylon_tooltip = f"{ClickConfig.TOOLTIP_PREFIX_PYLON} {i}{ClickConfig.TOOLTIP_SEPARATOR_ON}{lift_id}"

                folium.CircleMarker(
                    location=(pylon.lat, pylon.lon),
                    radius=ClickConfig.PYLON_MARKER_RADIUS,
                    color=MarkerConfig.PYLON_BORDER_COLOR,
                    fill=True,
                    fill_color=MarkerConfig.PYLON_MARKER_COLOR,
                    fill_opacity=0.9,
                    weight=2,
                    tooltip=pylon_tooltip,
                ).add_to(m)

            # Add lift icon at midpoint with lift type emoji
            mid_lat = (start_node.lat + end_node.lat) / 2
            mid_lon = (start_node.lon + end_node.lon) / 2

            # User-friendly tooltip: "View Lift L1"
            lift_tooltip = f"{ClickConfig.TOOLTIP_PREFIX_LIFT_ICON} {lift_id}"
            lift_icon = StyleConfig.LIFT_ICONS[lift.lift_type]

            # Single emoji marker (no background circle)
            folium.Marker(
                location=(mid_lat, mid_lon),
                icon=folium.DivIcon(
                    html=f'<div style="font-size: 20px; text-align: center; margin-left: -10px; margin-top: -10px;">{lift_icon}</div>',
                    icon_size=(20, 20),
                    icon_anchor=(10, 10),
                ),
                tooltip=lift_tooltip,
            ).add_to(m)

    def _render_nodes(self, m: folium.Map) -> None:
        """Render junction nodes with clickable markers."""
        if not self.graph:
            return

        for node_id, node in self.graph.nodes.items():
            # User-friendly tooltip: "Node N1"
            tooltip = f"{ClickConfig.TOOLTIP_PREFIX_NODE} {node_id}"

            folium.CircleMarker(
                location=(node.lat, node.lon),
                radius=ClickConfig.NODE_MARKER_RADIUS,
                color=MarkerConfig.NODE_MARKER_COLOR,
                fill=True,
                fill_color=MarkerConfig.NODE_MARKER_COLOR,
                fill_opacity=0.8,
                tooltip=tooltip,
            ).add_to(m)

    def _render_proposals(
        self,
        m: folium.Map,
        proposals: list[ProposedSlopeSegment],
        selected_idx: Optional[int] = None,
        is_custom_path: bool = False,
    ) -> None:
        """Render proposed paths as dashed lines with clickable markers."""
        for i, proposal in enumerate(proposals):
            is_selected = selected_idx is not None and i == selected_idx

            # Use actual (computed) difficulty color, not target
            color = StyleConfig.SLOPE_COLORS[proposal.difficulty]
            weight = 4 if is_selected else 2
            opacity = 1.0 if is_selected else 0.6

            # Use points for rendering
            points = proposal.points
            line = [(p.lat, p.lon) for p in points]

            # Line tooltip shows basic info (no target difficulty - that's in the stats panel)
            tooltip = (
                f"<b>Proposal {i + 1}</b><br>"
                f"Difficulty: {proposal.difficulty}<br>"
                f"Slope: {proposal.avg_slope_pct:.0f}%<br>"
                f"Length: {proposal.length_m:.0f}m"
            )

            folium.PolyLine(
                locations=line,
                color=color,
                weight=weight,
                opacity=opacity,
                dash_array="8, 4" if not is_selected else None,
                tooltip=tooltip,
            ).add_to(m)

            # Add markers
            if points:
                # Start marker (not clickable, just visual)
                folium.CircleMarker(
                    location=(points[0].lat, points[0].lon),
                    radius=5,
                    color=color,
                    fill=True,
                    fill_color="white",
                    tooltip=f"Start: {points[0].elevation:.0f}m",
                ).add_to(m)

                # Body marker at midpoint for selection (clickable, difficulty colored)
                mid_idx = len(points) // 2
                if mid_idx > 0:
                    # User-friendly tooltip (also used for click detection)
                    body_tooltip = f"{ClickConfig.TOOLTIP_PREFIX_PROPOSAL_BODY} {i + 1}"

                    folium.CircleMarker(
                        location=(points[mid_idx].lat, points[mid_idx].lon),
                        radius=ClickConfig.PROPOSAL_BODY_RADIUS,
                        color=color,
                        fill=True,
                        fill_color=color,
                        fill_opacity=0.7,
                        tooltip=body_tooltip,
                    ).add_to(m)

                # Endpoint marker for committing (clickable, always orange)
                # Skip for custom connect paths - all endpoints go to same target,
                # so commit via body markers only (commit button in sidebar)
                if not is_custom_path:
                    end_tooltip = f"{ClickConfig.TOOLTIP_PREFIX_PROPOSAL_END} {i + 1}"

                    # Orange circle marker for commit action
                    folium.CircleMarker(
                        location=(points[-1].lat, points[-1].lon),
                        radius=ClickConfig.PROPOSAL_ENDPOINT_RADIUS,
                        color=ClickConfig.PROPOSAL_ENDPOINT_COLOR,
                        fill=True,
                        fill_color=ClickConfig.PROPOSAL_ENDPOINT_COLOR,
                        fill_opacity=0.9,
                        tooltip=end_tooltip,
                    ).add_to(m)

    def fit_bounds(self, m: folium.Map) -> None:
        """Fit map bounds to graph contents."""
        if not self.graph:
            return

        all_points = []

        for node in self.graph.nodes.values():
            all_points.append((node.lat, node.lon))

        for segment in self.graph.segments.values():
            for pt in segment.points:
                all_points.append((pt.lat, pt.lon))

        if all_points:
            m.fit_bounds(all_points)

    def update_center(self, lat: float, lon: float) -> None:
        """Update map center for next render."""
        self.center_lat = lat
        self.center_lon = lon

    def add_orientation_arrows(
        self,
        m: folium.Map,
        lat: float,
        lon: float,
        orientation: "TerrainOrientation",
    ) -> None:
        """Add orientation arrows showing fall line and contours at selection point."""
        difficulty_color = orientation.difficulty_color
        arrow_length_m = MarkerConfig.ORIENTATION_ARROW_LENGTH_M

        # Fall line (difficulty colored)
        if orientation.fall_line is not None:
            end_lon, end_lat = GeoCalculator.destination(
                lon=lon, lat=lat, bearing_deg=orientation.fall_line, distance_m=arrow_length_m
            )
            folium.PolyLine([(lat, lon), (end_lat, end_lon)], color=difficulty_color, weight=4, opacity=0.9).add_to(m)

        # Contour lines (gray)
        for bearing in [orientation.contour_left, orientation.contour_right]:
            if bearing is None:
                continue
            end_lon, end_lat = GeoCalculator.destination(
                lon=lon, lat=lat, bearing_deg=bearing, distance_m=arrow_length_m
            )
            folium.PolyLine(
                [(lat, lon), (end_lat, end_lon)],
                color=MarkerConfig.ORIENTATION_CONTOUR_COLOR,
                weight=3,
                opacity=0.8,
            ).add_to(m)

        # Center marker
        folium.CircleMarker(
            (lat, lon), radius=8, color="#FFFFFF", fill=True, fill_color=difficulty_color, fill_opacity=0.9
        ).add_to(m)

    # =========================================================================
    # STATIC MARKER UTILITIES (UI feedback components without animation)
    # =========================================================================

    def add_direction_arrow(
        self,
        m: folium.Map,
        start_lat: float,
        start_lon: float,
        bearing_deg: float,
        direction: str = "downhill",
        tooltip: str = "",
    ) -> None:
        """Add static directional arrow from a point.

        Shows an arrow indicating the direction of travel.
        Used for custom connect mode (green downhill) and lift placement (purple uphill).

        Args:
            m: Folium map to add indicator to
            start_lat, start_lon: Starting point coordinates
            bearing_deg: Direction in degrees (0=N, 90=E, 180=S, 270=W)
            direction: "downhill" (green, slope mode) or "uphill" (purple, lift mode)
            tooltip: Tooltip text for the indicator
        """
        # Select color based on direction
        if direction == "uphill":
            color = MarkerConfig.DIRECTION_ARROW_COLOR_UPHILL
        else:
            color = MarkerConfig.DIRECTION_ARROW_COLOR_DOWNHILL

        # Calculate end point
        end_lon, end_lat = GeoCalculator.destination(
            lon=start_lon,
            lat=start_lat,
            bearing_deg=bearing_deg,
            distance_m=MarkerConfig.DIRECTION_ARROW_LENGTH_M,
        )

        # Draw arrow line
        folium.PolyLine(
            locations=[(start_lat, start_lon), (end_lat, end_lon)],
            color=color,
            weight=MarkerConfig.DIRECTION_ARROW_WIDTH,
            opacity=0.9,
            tooltip=tooltip or f"{'↑ Uphill' if direction == 'uphill' else '↓ Downhill'} direction",
        ).add_to(m)

        # Arrowhead at end
        folium.CircleMarker(
            location=(end_lat, end_lon),
            radius=6,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.9,
        ).add_to(m)

    def add_pending_lift_marker(
        self,
        m: folium.Map,
        fall_line_bearing: float,
        node_id: Optional[str] = None,
        location: Optional[PathPoint] = None,
    ) -> None:
        """Add marker for pending lift placement with uphill direction indicator.

        Shows a station marker at the bottom station when lift placement
        is in progress, plus a static arrow pointing uphill (opposite fall line).

        Args:
            m: Folium map to add marker to
            fall_line_bearing: Fall line direction for uphill arrow
            node_id: Node ID where lift placement started (existing node)
            location: PathPoint for new location (no existing node)
        """
        # Get location from node or from provided location
        if node_id and self.graph:
            node = self.graph.nodes.get(node_id)
            if not node:
                return
            lat, lon, elev = node.lat, node.lon, node.elevation
            tooltip_id = node_id
        elif location:
            lat, lon, elev = location.lat, location.lon, location.elevation
            tooltip_id = f"({lat:.4f}, {lon:.4f})"
        else:
            return

        # Station marker (static)
        folium.CircleMarker(
            (lat, lon),
            radius=MarkerConfig.LIFT_STATION_RADIUS,
            color="#FFFFFF",
            fill=True,
            fill_color=MarkerConfig.LIFT_STATION_COLOR,
            fill_opacity=0.9,
            weight=3,
            tooltip=f"🚡 BOTTOM: {tooltip_id} ({elev:.0f}m) - Click HIGHER point for top!",
        ).add_to(m)

        # Simple station icon
        folium.Marker(
            (lat, lon),
            icon=folium.DivIcon(
                html='<div style="font-size:20px;text-align:center;">🚉</div>',
                icon_size=(24, 24),
                icon_anchor=(12, 12),
            ),
        ).add_to(m)

        # Add uphill direction arrow (opposite of fall line - 180° rotation)
        uphill_bearing = (fall_line_bearing + 180) % 360
        self.add_direction_arrow(
            m=m,
            start_lat=lat,
            start_lon=lon,
            bearing_deg=uphill_bearing,
            direction="uphill",
            tooltip="🚡 Lift goes UPHILL - click a higher point",
        )

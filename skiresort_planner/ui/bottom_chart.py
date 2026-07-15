"""ProfileChart - Plotly elevation profile rendering.

Renders elevation profiles showing:
- Terrain elevation along path
- Slope gradient coloring
- Distance markers
- Warning annotations
- Lift profiles with pylons and cable

Reference: DETAILS_UI.md for profile display
"""

import plotly.graph_objects as go

from skiresort_planner.constants import ChartConfig, LiftConfig, LiftType, StyleConfig
from skiresort_planner.core.geo_calculator import GeoCalculator
from skiresort_planner.core.terrain_analyzer import TerrainAnalyzer
from skiresort_planner.enum_utils import enum_eq
from skiresort_planner.model.lift import Lift
from skiresort_planner.model.path_segment import PathSegment, SegmentKind
from skiresort_planner.model.resort_graph import ResortGraph
from skiresort_planner.model.road import Road
from skiresort_planner.model.segment_path import SegmentPath, steepest_section_pct
from skiresort_planner.model.slope import Slope
from skiresort_planner.ui.context import EntityKind


class ProfileChart:
    """Renders elevation profiles using Plotly.

    Example:
        chart = ProfileChart(height=260)
        fig = chart.render_slope(slope=slope, graph=graph)
        st.plotly_chart(fig, width="stretch")
    """

    def __init__(self, height: int) -> None:
        """Initialize profile chart renderer with a pixel height."""
        self.height = height

    def render_segment(
        self,
        segment: PathSegment,
        difficulty: str,
        title: str | None = None,
    ) -> go.Figure:
        """Render elevation profile for a committed segment.

        Args:
            segment: Slope segment to visualize
            difficulty: Difficulty level for coloring (caller determines the correct value)
            title: Optional chart title

        Returns:
            Plotly Figure object.
        """
        points = segment.points
        if not points:
            raise ValueError("Segment must have points to render")

        distances = [0.0]
        for i in range(1, len(points)):
            dist = GeoCalculator.haversine_distance_m(
                lat1=points[i - 1].lat,
                lon1=points[i - 1].lon,
                lat2=points[i].lat,
                lon2=points[i].lon,
            )
            distances.append(distances[-1] + dist)

        elevations = [p.elevation for p in points]

        # Calculate Y-axis range (not starting from 0)
        min_elev = min(elevations)
        max_elev = max(elevations)
        padding = max(
            (max_elev - min_elev) * ChartConfig.ELEVATION_PADDING_FACTOR,
            ChartConfig.ELEVATION_PADDING_MIN_M,
        )

        color = StyleConfig.SLOPE_COLORS[difficulty]

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=distances,
                y=elevations,
                fill="tozeroy",
                fillcolor=f"rgba{self._hex_to_rgba(hex_color=color, alpha=0.3)}",
                line=dict(color=color, width=2),
                name=segment.name,
                hovertemplate="Distance: %{x:.0f}m<br>Elevation: %{y:.0f}m<extra></extra>",
            )
        )

        title = title or segment.name
        fig.update_layout(
            title=dict(text=title, x=0.5),
            xaxis=dict(
                title="Distance (m)",
                showgrid=True,
                gridcolor="rgba(200, 200, 200, 0.3)",
            ),
            yaxis=dict(
                title="Elevation (m)",
                showgrid=True,
                gridcolor="rgba(200, 200, 200, 0.3)",
                range=[min_elev - padding, max_elev + padding],
            ),
            showlegend=False,
            height=self.height,
            margin=ChartConfig.PROFILE_MARGIN,
            plot_bgcolor="white",
        )

        # Add warnings if present
        if segment.warnings:
            fig.add_annotation(
                xref="paper",
                yref="paper",
                x=0.5,
                y=1.05,
                text=" | ".join(str(w) for w in segment.warnings),
                showarrow=False,
                font=dict(size=10, color="red"),
            )

        return fig

    def render_slope(
        self,
        slope: Slope,
        graph: ResortGraph,
        title: str | None = None,
    ) -> go.Figure:
        """Render elevation profile for a complete slope (colored by difficulty)."""
        difficulty = slope.get_difficulty(segments=graph.segments)
        total_length = slope.get_total_length(segments=graph.segments)
        total_drop = slope.get_total_drop(segments=graph.segments)
        stats_text = f"Length: {total_length:.0f}m | Drop: {total_drop:.0f}m | Segments: {len(slope.segment_ids)}"
        return self._render_path_profile(
            path=slope,
            graph=graph,
            fill_color=StyleConfig.SLOPE_COLORS[difficulty],
            title=title or slope.name,
            stats_text=stats_text,
        )

    def render_road(
        self,
        road: Road,
        graph: ResortGraph,
        title: str | None = None,
    ) -> go.Figure:
        """Render elevation profile for a road (brown, shows climb/descent)."""
        total_length = road.get_total_length(segments=graph.segments)
        total_drop = road.get_total_drop(segments=graph.segments)
        max_gradient = road.get_max_gradient(segments=graph.segments)
        stats_text = (
            f"Length: {total_length:.0f}m | Elevation change: {abs(total_drop):.0f}m | Steepest: {max_gradient:.0f}%"
        )
        return self._render_path_profile(
            path=road,
            graph=graph,
            fill_color=StyleConfig.ROAD_COLOR,
            title=title or road.name,
            stats_text=stats_text,
        )

    def _render_path_profile(
        self,
        path: SegmentPath,
        graph: ResortGraph,
        fill_color: str,
        title: str,
        stats_text: str,
    ) -> go.Figure:
        """Shared elevation-profile figure for any segment group (slope or road).

        Plots elevation vs. cumulative distance with segment-boundary markers.
        Caller supplies the fill color and the stats caption so slope-specific
        (difficulty) and road-specific (climb) framing stay with their owners.
        """
        all_points = path.get_all_points(segments=graph.segments)
        if not all_points:
            raise ValueError(f"{path.id} must have points to render")

        distances = [0.0]
        for i in range(1, len(all_points)):
            dist = GeoCalculator.haversine_distance_m(
                lat1=all_points[i - 1].lat,
                lon1=all_points[i - 1].lon,
                lat2=all_points[i].lat,
                lon2=all_points[i].lon,
            )
            distances.append(distances[-1] + dist)

        elevations = [p.elevation for p in all_points]

        # Calculate Y-axis range (not starting from 0)
        min_elev = min(elevations)
        max_elev = max(elevations)
        padding = max(
            (max_elev - min_elev) * ChartConfig.ELEVATION_PADDING_FACTOR,
            ChartConfig.ELEVATION_PADDING_MIN_M,
        )

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=distances,
                y=elevations,
                fill="tozeroy",
                fillcolor=f"rgba{self._hex_to_rgba(hex_color=fill_color, alpha=0.3)}",
                line=dict(color=fill_color, width=2),
                name=title,
                hovertemplate="Distance: %{x:.0f}m<br>Elevation: %{y:.0f}m<extra></extra>",
            )
        )

        # Add segment boundaries
        cum_dist = 0.0
        for seg_id in path.segment_ids:
            seg = graph.segments.get(seg_id)
            if seg is None:
                raise ValueError(f"{path.id} references non-existent segment {seg_id}")

            cum_dist += seg.length_m
            fig.add_vline(
                x=cum_dist,
                line_dash="dot",
                line_color=fill_color,
                opacity=0.5,
            )

        fig.update_layout(
            title=dict(text=title, x=0.5),
            xaxis=dict(
                title="Distance (m)",
                showgrid=True,
                gridcolor="rgba(200, 200, 200, 0.3)",
            ),
            yaxis=dict(
                title="Elevation (m)",
                showgrid=True,
                gridcolor="rgba(200, 200, 200, 0.3)",
                range=[min_elev - padding, max_elev + padding],
            ),
            showlegend=False,
            height=self.height,
            margin=ChartConfig.PROFILE_MARGIN,
            plot_bgcolor="white",
        )

        fig.add_annotation(
            xref="paper",
            yref="paper",
            x=0.5,
            y=-0.15,
            text=stats_text,
            showarrow=False,
            font=dict(size=11),
        )

        return fig

    def _empty_figure(self, message: str) -> go.Figure:
        """Create empty figure with message."""
        fig = go.Figure()
        fig.add_annotation(
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            text=message,
            showarrow=False,
            font=dict(size=14, color="gray"),
        )
        fig.update_layout(
            height=self.height,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            plot_bgcolor="white",
        )
        return fig

    def _hex_to_rgba(self, hex_color: str, alpha: float) -> tuple[int, int, int, float]:
        """Convert hex color to RGBA tuple."""
        hex_color = hex_color.lstrip("#")
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        return (r, g, b, alpha)

    def render_lift(
        self,
        lift: Lift,
        graph: ResortGraph,
    ) -> go.Figure:
        """Render elevation profile for a lift with terrain, pylons, and cable.

        Shows:
        - Ground terrain profile (filled area)
        - Pylons as vertical bars from ground to cable height
        - Cable line connecting stations via pylon tops

        Args:
            lift: The Lift object to visualize
            graph: ResortGraph containing the lift's nodes

        Returns:
            Plotly Figure with lift profile
        """
        start_node = graph.nodes[lift.start_node_id]
        end_node = graph.nodes[lift.end_node_id]

        # Get station height from config
        config = LiftConfig.PYLON_CONFIG[LiftType(lift.lift_type)]
        station_height: float = config["station_height_m"]

        # Calculate lift metrics
        length_m = GeoCalculator.haversine_distance_m(
            lat1=start_node.lat,
            lon1=start_node.lon,
            lat2=end_node.lat,
            lon2=end_node.lon,
        )
        vertical_rise = end_node.elevation - start_node.elevation

        # Use stored terrain_points, pylons, and cable_points
        n_terrain = len(lift.terrain_points)
        terrain_distances = [length_m * i / (n_terrain - 1) for i in range(n_terrain)] if n_terrain > 1 else [0]
        terrain_elevs = [p.elevation for p in lift.terrain_points]

        # Use stored cable_points - compute distances from start
        cable_x = []
        cable_y = []
        cable_ground_elevs = []

        for cable_pt in lift.cable_points:
            # Distance from start
            dist = GeoCalculator.haversine_distance_m(
                lat1=start_node.lat,
                lon1=start_node.lon,
                lat2=cable_pt.lat,
                lon2=cable_pt.lon,
            )
            cable_x.append(dist)
            cable_y.append(cable_pt.elevation)

            # Interpolate ground elevation from terrain points
            terrain_frac = dist / length_m if length_m > 0 else 0
            terrain_idx = terrain_frac * (n_terrain - 1)
            idx_low = int(terrain_idx)
            idx_high = min(idx_low + 1, n_terrain - 1)
            interp_frac = terrain_idx - idx_low
            ground_elev = terrain_elevs[idx_low] * (1 - interp_frac) + terrain_elevs[idx_high] * interp_frac
            cable_ground_elevs.append(ground_elev)

        color = StyleConfig.LIFT_COLORS[lift.lift_type]
        fig = go.Figure()

        # 1. Terrain profile (filled area)
        terrain_percentages = [(d / length_m * 100) if length_m > 0 else 0 for d in terrain_distances]

        fig.add_trace(
            go.Scatter(
                x=terrain_distances,
                y=terrain_elevs,
                mode="lines",
                fill="tozeroy",
                fillcolor="rgba(139, 90, 43, 0.4)",
                line=dict(color="rgb(101, 67, 33)", width=2),
                name="Terrain",
                customdata=terrain_percentages,
                hovertemplate=(
                    "<b>Distance:</b> %{x:.0f}m (%{customdata:.0f}%)<br><b>Ground:</b> %{y:.0f}m<extra></extra>"
                ),
            )
        )

        # 2. Cable line
        if cable_x:
            cable_heights = [cable_y[i] - cable_ground_elevs[i] for i in range(len(cable_x))]
            cable_percentages = [(x / length_m * 100) if length_m > 0 else 0 for x in cable_x]

            fig.add_trace(
                go.Scatter(
                    x=cable_x,
                    y=cable_y,
                    mode="lines",
                    line=dict(color="#333333", width=3),
                    name="Cable",
                    customdata=list(zip(cable_ground_elevs, cable_heights, cable_percentages, strict=False)),
                    hovertemplate=(
                        "<b>Distance:</b> %{x:.0f}m (%{customdata[2]:.0f}%)<br>"
                        "<b>Ground:</b> %{customdata[0]:.0f}m<br>"
                        "<b>Cable height:</b> %{customdata[1]:.0f}m<extra></extra>"
                    ),
                )
            )

        # 3. Pylons as vertical bars
        for pylon_num, pylon in enumerate(lift.pylons, 1):
            pylon_x = pylon.distance_m
            pylon_base = pylon.ground_elevation_m
            pylon_top = pylon.top_elevation_m
            pylon_pct = (pylon_x / length_m * 100) if length_m > 0 else 0

            # Pylon bar
            fig.add_trace(
                go.Scatter(
                    x=[pylon_x, pylon_x],
                    y=[pylon_base, pylon_top],
                    mode="lines",
                    line=dict(color=color, width=6),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
            # Pylon cap
            fig.add_trace(
                go.Scatter(
                    x=[pylon_x],
                    y=[pylon_top],
                    mode="markers",
                    marker=dict(size=8, color=color, symbol="square"),
                    showlegend=False,
                    hovertemplate=(
                        f"Pylon {pylon_num}<br>"
                        f"Distance: {pylon_x:.0f}m ({pylon_pct:.0f}%)<br>"
                        f"Ground: {pylon_base:.0f}m<br>"
                        f"Height: {pylon.height_m:.0f}m<extra></extra>"
                    ),
                )
            )

        # 4. Station markers
        fig.add_trace(
            go.Scatter(
                x=[0, length_m],
                y=[start_node.elevation + station_height, end_node.elevation + station_height],
                mode="markers",
                marker=dict(size=14, color=color, symbol="square"),
                name="Stations",
                hovertemplate=[
                    f"Bottom Station<br>Ground: {start_node.elevation:.0f}m<br>Height: {station_height}m<extra></extra>",
                    f"Top Station<br>Ground: {end_node.elevation:.0f}m<br>Height: {station_height}m<extra></extra>",
                ],
            )
        )

        # 5. Station buildings (vertical bars)
        for x, ground_elev, station_elev in [
            (0, start_node.elevation, start_node.elevation + station_height),
            (length_m, end_node.elevation, end_node.elevation + station_height),
        ]:
            fig.add_trace(
                go.Scatter(
                    x=[x, x],
                    y=[ground_elev, station_elev],
                    mode="lines",
                    line=dict(color=color, width=8),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

        # Layout
        all_elevs = terrain_elevs + (cable_y if cable_y else [])
        min_elev = min(all_elevs)
        max_elev = max(all_elevs)
        padding = max(
            (max_elev - min_elev) * ChartConfig.LIFT_ELEVATION_PADDING_FACTOR,
            ChartConfig.LIFT_ELEVATION_PADDING_MIN_M,
        )

        fig.update_layout(
            height=self.height,
            margin=ChartConfig.PROFILE_MARGIN,
            xaxis_title="Distance (m)",
            yaxis_title="Elevation (m)",
            yaxis=dict(range=[min_elev - padding, max_elev + padding]),
            xaxis=dict(range=[-length_m * 0.02, length_m * 1.02]),
            showlegend=False,
            title=dict(
                text=f"🚡 {lift.name}: {vertical_rise:.0f}m rise | {length_m:.0f}m | {len(lift.pylons)} pylons",
                font=dict(size=12),
            ),
            plot_bgcolor="rgba(240, 248, 255, 0.5)",
        )

        return fig


# =============================================================================
# CONVENIENCE FUNCTIONS FOR APP
# =============================================================================


def render_building_profile(
    building_segments: list[str],
    building_name: str | None,
    graph: ResortGraph,
) -> go.Figure:
    """Render the in-build elevation profile for a slope OR road.

    One function for both: the committed segments carry their SegmentKind, so a
    road renders brown (via render_road) and a slope renders difficulty-colored
    (via render_segment). Uses max difficulty from the individual segments for a
    slope (steepest segment determines color, matching the sidebar display).

    Args:
        building_segments: Segment IDs in the current slope/road (must not be empty)
        building_name: Name of the entity being built
        graph: Resort graph containing segments

    Returns:
        Plotly figure.

    Raises:
        ValueError: If building_segments is empty or segments have no points.
    """
    if not building_segments:
        raise ValueError("building_segments must not be empty - check before calling")

    segs = [graph.segments[sid] for sid in building_segments if sid in graph.segments and graph.segments[sid].points]
    all_points = [p for seg in segs for p in seg.points]

    if not all_points:
        raise ValueError(f"Segments {building_segments} have no points - data integrity issue")

    chart = ProfileChart(height=ChartConfig.PROFILE_HEIGHT_PX)

    first_seg = graph.segments[building_segments[0]]
    # enum_eq (reload-safe str compare): Streamlit reloads create a fresh SegmentKind class,
    # so `is` fails on segments built under the old class.
    if enum_eq(a=first_seg.kind, b=SegmentKind.ROAD):
        combined_road = Road(
            id="combined",
            name=building_name or "Current Road",
            segment_ids=list(building_segments),
            start_node_id="",
            end_node_id="",
        )
        return chart.render_road(road=combined_road, graph=graph, title="Current committed Road Progress")
    elif enum_eq(a=first_seg.kind, b=SegmentKind.SLOPE):
        # Color by the steepest section — the SAME metric as the map marker and the
        # finished slope, so the progress color never disagrees with them.
        max_difficulty = TerrainAnalyzer.classify_difficulty(slope_pct=steepest_section_pct(segments=segs))
        combined = PathSegment(id="combined", name=building_name or "Current Slope", points=all_points)
        return chart.render_segment(
            segment=combined, difficulty=max_difficulty, title="Current committed Slope Progress"
        )
    else:
        raise RuntimeError(f"Unknown segment kind {first_seg.kind} for {first_seg.id}")


def render_viewing_profile(kind: EntityKind, entity_id: str, graph: ResortGraph) -> go.Figure:
    """Render the elevation profile of a finished slope / road / lift being viewed."""
    chart = ProfileChart(height=ChartConfig.PROFILE_HEIGHT_PX)
    if enum_eq(a=kind, b=EntityKind.SLOPE):
        slope = graph.slopes.get(entity_id)
        if slope is None:
            raise ValueError(f"Slope {entity_id} must exist when panel shows slope")
        return chart.render_slope(slope=slope, graph=graph)
    if enum_eq(a=kind, b=EntityKind.ROAD):
        road = graph.roads.get(entity_id)
        if road is None:
            raise ValueError(f"Road {entity_id} must exist when panel shows road")
        return chart.render_road(road=road, graph=graph)
    if enum_eq(a=kind, b=EntityKind.LIFT):
        lift = graph.lifts.get(entity_id)
        if lift is None:
            raise ValueError(f"Lift {entity_id} must exist when panel shows lift")
        return chart.render_lift(lift=lift, graph=graph)
    raise RuntimeError(f"Unknown viewing kind {kind!r}")

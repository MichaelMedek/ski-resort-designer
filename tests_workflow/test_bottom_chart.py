"""Unit tests for ProfileChart elevation-profile rendering (ui/bottom_chart.py)."""

from skiresort_planner.constants import MapConfig


class TestProfileChartRendering:
    """Tests for elevation profile chart rendering."""

    def test_building_profile_uses_steepest_section_metric(self, empty_graph) -> None:
        """Regression: the in-build profile colors by the SAME steepest-section metric
        as the finished slope / map marker — not by average slope (which read gentler
        and showed the wrong color). steepest_section_pct is the single source.
        """
        from skiresort_planner.constants import StyleConfig
        from skiresort_planner.core.terrain_analyzer import TerrainAnalyzer
        from skiresort_planner.model.path_point import PathPoint
        from skiresort_planner.model.proposed_path import ProposedPathSegment
        from skiresort_planner.model.segment_path import steepest_section_pct
        from skiresort_planner.ui.bottom_chart import render_building_profile

        # A long segment whose steepest 300m section is markedly steeper than its average.
        pts = [
            PathPoint(lon=0.0, lat=-d / MapConfig.METERS_PER_DEGREE_EQUATOR, elevation=2000.0 - e)
            for d, e in [(0, 0), (300, 120), (700, 150)]
        ]
        empty_graph.commit_paths(paths=[ProposedPathSegment(points=pts, target_difficulty="red")])
        seg_ids = list(empty_graph.segments.keys())
        segs = [empty_graph.segments[s] for s in seg_ids]

        expected = TerrainAnalyzer.classify_difficulty(slope_pct=steepest_section_pct(segments=segs))
        expected_color = StyleConfig.SLOPE_COLORS[expected]

        fig = render_building_profile(building_segments=seg_ids, building_name="WIP", graph=empty_graph)
        line_colors = [tr.line.color for tr in fig.data if tr.line and tr.line.color]
        assert expected_color in line_colors, f"building profile must color by steepest section ({expected})"

    def test_segment_chart_renders(self, empty_graph, path_points_blue) -> None:
        """ProfileChart can render a committed segment."""
        from skiresort_planner.model.proposed_path import ProposedPathSegment
        from skiresort_planner.ui.bottom_chart import ProfileChart

        graph = empty_graph
        proposal = ProposedPathSegment(
            points=path_points_blue,
            target_slope_pct=20.0,
            target_difficulty="blue",
            sector_name="Test",
        )
        graph.commit_paths(paths=[proposal])
        segment = list(graph.segments.values())[0]

        chart = ProfileChart(height=300)
        fig = chart.render_segment(segment=segment, difficulty="blue")

        assert len(fig.data) > 0, "Figure should have data traces"

    def test_slope_chart_renders_with_segment_boundaries(self, empty_graph, path_points_blue) -> None:
        """render_slope produces a figure with a vline per segment boundary."""
        from skiresort_planner.model.proposed_path import ProposedPathSegment
        from skiresort_planner.ui.bottom_chart import ProfileChart

        empty_graph.commit_paths(paths=[ProposedPathSegment(points=path_points_blue, target_difficulty="blue")])
        slope = empty_graph.finish_slope(segment_ids=list(empty_graph.segments.keys()))

        fig = ProfileChart(height=300).render_slope(slope=slope, graph=empty_graph)
        assert len(fig.data) > 0
        # One dotted vline per segment boundary lives in layout.shapes.
        assert len(fig.layout.shapes) == len(slope.segment_ids)
        # Fill trace is colored by the slope's derived difficulty (steepest-segment metric).
        from skiresort_planner.constants import StyleConfig

        difficulty = slope.get_difficulty(segments=empty_graph.segments)
        line_colors = [tr.line.color for tr in fig.data if tr.line and tr.line.color]
        assert StyleConfig.SLOPE_COLORS[difficulty] in line_colors
        # No title/stats caption — that info lives in the right-side panel, not the plot.
        assert not fig.layout.title.text
        assert " ".join(a.text for a in fig.layout.annotations) == ""

    def test_road_chart_renders_brown_without_redundant_caption(self, empty_graph) -> None:
        """render_road produces a brown figure with no title/stats caption (panel shows those)."""
        from skiresort_planner.model.path_point import PathPoint
        from skiresort_planner.model.path_segment import SegmentKind
        from skiresort_planner.model.proposed_path import ProposedPathSegment
        from skiresort_planner.ui.bottom_chart import ProfileChart

        pts = [
            PathPoint(lon=0.0, lat=0.0, elevation=2000.0),
            PathPoint(lon=300 / MapConfig.METERS_PER_DEGREE_EQUATOR, lat=0.0, elevation=1990.0),
        ]
        empty_graph.commit_paths(
            paths=[ProposedPathSegment(points=pts, is_connector=True, kind=SegmentKind.ROAD)], record_undo=False
        )
        road = empty_graph.finish_road(segment_ids=[list(empty_graph.segments.keys())[-1]])

        fig = ProfileChart(height=300).render_road(road=road, graph=empty_graph)
        assert len(fig.data) > 0
        from skiresort_planner.constants import StyleConfig

        line_colors = [tr.line.color for tr in fig.data if tr.line and tr.line.color]
        assert StyleConfig.ROAD_COLOR in line_colors
        # No title/stats caption — that info lives in the right-side panel, not the plot.
        assert not fig.layout.title.text
        assert " ".join(a.text for a in fig.layout.annotations) == ""

    def test_lift_chart_renders_terrain_cable_pylons(self, empty_graph, mock_dem_blue_slope) -> None:
        """render_lift produces a figure with terrain + cable + pylon traces."""
        from skiresort_planner.ui.bottom_chart import ProfileChart

        dem = mock_dem_blue_slope
        bottom, _ = empty_graph.get_or_create_node(
            lon=0.0,
            lat=-1000 / MapConfig.METERS_PER_DEGREE_EQUATOR,
            elevation=dem.get_elevation_or_raise(lon=0.0, lat=-1000 / MapConfig.METERS_PER_DEGREE_EQUATOR),
        )
        top, _ = empty_graph.get_or_create_node(
            lon=0.0, lat=0.0, elevation=dem.get_elevation_or_raise(lon=0.0, lat=0.0)
        )
        lift = empty_graph.add_lift(start_node_id=bottom.id, end_node_id=top.id, lift_type="chairlift", dem=dem)

        fig = ProfileChart(height=300).render_lift(lift=lift, graph=empty_graph)
        assert len(fig.data) > 0
        from skiresort_planner.constants import StyleConfig

        trace_names = {tr.name for tr in fig.data if tr.name}
        assert "Terrain" in trace_names and "Cable" in trace_names
        # Each pylon is drawn as a width-6 line bar (stations use width 8, cable 3, terrain 2).
        pylon_bars = [tr for tr in fig.data if tr.line and tr.line.width == 6]
        assert len(pylon_bars) == len(lift.pylons)
        # No title/stats caption — the right-side panel shows name + stats; pylon bars use the chairlift color.
        assert not fig.layout.title.text
        assert all(bar.line.color == StyleConfig.LIFT_COLORS["chairlift"] for bar in pylon_bars)

    def test_building_profile_slope(self, empty_graph, path_points_blue) -> None:
        """render_building_profile builds a combined in-progress SLOPE figure (kind-driven)."""
        from skiresort_planner.model.proposed_path import ProposedPathSegment
        from skiresort_planner.ui.bottom_chart import render_building_profile

        empty_graph.commit_paths(paths=[ProposedPathSegment(points=path_points_blue, target_difficulty="blue")])
        seg_ids = list(empty_graph.segments.keys())

        fig = render_building_profile(building_segments=seg_ids, building_name="WIP", graph=empty_graph)
        assert len(fig.data) > 0

    def test_building_profile_road_is_brown(self, empty_graph, path_points_blue) -> None:
        """render_building_profile renders a brown ROAD figure when segments are road-kind."""
        from skiresort_planner.constants import StyleConfig
        from skiresort_planner.model.path_segment import SegmentKind
        from skiresort_planner.model.proposed_path import ProposedPathSegment
        from skiresort_planner.ui.bottom_chart import render_building_profile

        empty_graph.commit_paths(
            paths=[ProposedPathSegment(points=path_points_blue, is_connector=True, kind=SegmentKind.ROAD)]
        )
        seg_ids = list(empty_graph.segments.keys())

        fig = render_building_profile(building_segments=seg_ids, building_name="WIP Road", graph=empty_graph)
        assert len(fig.data) > 0
        # Brown road fill (same road color the finished-road profile uses).
        line_colors = [tr.line.color for tr in fig.data if tr.line and tr.line.color]
        assert StyleConfig.ROAD_COLOR in line_colors

    def test_building_profile_kind_compared_by_value_not_identity(self, empty_graph, path_points_blue) -> None:
        """A segment whose kind is a DISTINCT SegmentKind class (Streamlit module reload)
        must still be recognized as a road — regression for the `is` vs `==` crash.
        """
        from enum import Enum

        from skiresort_planner.model.path_segment import SegmentKind
        from skiresort_planner.model.proposed_path import ProposedPathSegment
        from skiresort_planner.ui.bottom_chart import render_building_profile

        empty_graph.commit_paths(
            paths=[ProposedPathSegment(points=path_points_blue, is_connector=True, kind=SegmentKind.ROAD)]
        )
        seg_id = list(empty_graph.segments.keys())[-1]

        # Simulate a module reload: a fresh SegmentKind class with the same values.
        reloaded_kind = Enum("SegmentKind", {"SLOPE": "slope", "ROAD": "road"}, type=str)  # type: ignore[misc]  # functional enum name intentionally matches the reloaded class, not the variable
        empty_graph.segments[seg_id].kind = reloaded_kind.ROAD
        assert empty_graph.segments[seg_id].kind is not SegmentKind.ROAD, "must be a different class instance"
        assert empty_graph.segments[seg_id].kind == SegmentKind.ROAD, (
            "StrEnum == must still match ROAD by value across a reload"
        )

        # Must not raise 'Unknown segment kind' and must still render the road.
        fig = render_building_profile(building_segments=[seg_id], building_name="Reloaded Road", graph=empty_graph)
        assert len(fig.data) > 0

    def test_viewing_profile_slope(self, empty_graph, path_points_blue) -> None:
        """render_viewing_profile renders a finished slope's profile (kind-driven)."""
        from skiresort_planner.model.proposed_path import ProposedPathSegment
        from skiresort_planner.ui.bottom_chart import render_viewing_profile
        from skiresort_planner.ui.context import EntityKind

        empty_graph.commit_paths(paths=[ProposedPathSegment(points=path_points_blue, target_difficulty="blue")])
        slope = empty_graph.finish_slope(segment_ids=list(empty_graph.segments.keys()))
        fig = render_viewing_profile(kind=EntityKind.SLOPE, entity_id=slope.id, graph=empty_graph)
        assert len(fig.data) > 0

    def test_viewing_profile_road(self, empty_graph, path_points_blue) -> None:
        """render_viewing_profile renders a finished road's profile."""
        from skiresort_planner.model.path_segment import SegmentKind
        from skiresort_planner.model.proposed_path import ProposedPathSegment
        from skiresort_planner.ui.bottom_chart import render_viewing_profile
        from skiresort_planner.ui.context import EntityKind

        empty_graph.commit_paths(
            paths=[ProposedPathSegment(points=path_points_blue, is_connector=True, kind=SegmentKind.ROAD)],
            record_undo=False,
        )
        road = empty_graph.finish_road(segment_ids=[list(empty_graph.segments.keys())[-1]])
        fig = render_viewing_profile(kind=EntityKind.ROAD, entity_id=road.id, graph=empty_graph)
        assert len(fig.data) > 0

    def test_viewing_profile_kind_compared_by_value_not_identity(self, empty_graph, path_points_blue) -> None:
        """Viewing a finished ROAD with a reloaded EntityKind class must still render —
        regression for the `RuntimeError: Unknown viewing kind ROAD` crash after a reload.
        """
        from enum import Enum

        from skiresort_planner.model.path_segment import SegmentKind
        from skiresort_planner.model.proposed_path import ProposedPathSegment
        from skiresort_planner.ui.bottom_chart import render_viewing_profile

        empty_graph.commit_paths(
            paths=[ProposedPathSegment(points=path_points_blue, is_connector=True, kind=SegmentKind.ROAD)],
            record_undo=False,
        )
        road = empty_graph.finish_road(segment_ids=[list(empty_graph.segments.keys())[-1]])

        # Simulate a Streamlit module reload: a fresh EntityKind class with the same values.
        reloaded_kind = Enum("EntityKind", {"SLOPE": "slope", "ROAD": "road", "LIFT": "lift"}, type=str)  # type: ignore[misc]  # functional enum name intentionally matches the reloaded class, not the variable
        fig = render_viewing_profile(kind=reloaded_kind.ROAD, entity_id=road.id, graph=empty_graph)  # type: ignore[arg-type]
        assert len(fig.data) > 0

    def test_viewing_profile_lift(self, empty_graph, mock_dem_blue_slope) -> None:
        """render_viewing_profile renders a lift's profile."""
        from skiresort_planner.ui.bottom_chart import render_viewing_profile
        from skiresort_planner.ui.context import EntityKind

        dem = mock_dem_blue_slope
        bottom, _ = empty_graph.get_or_create_node(
            lon=0.0,
            lat=-1000 / MapConfig.METERS_PER_DEGREE_EQUATOR,
            elevation=dem.get_elevation_or_raise(lon=0.0, lat=-1000 / MapConfig.METERS_PER_DEGREE_EQUATOR),
        )
        top, _ = empty_graph.get_or_create_node(
            lon=0.0, lat=0.0, elevation=dem.get_elevation_or_raise(lon=0.0, lat=0.0)
        )
        lift = empty_graph.add_lift(start_node_id=bottom.id, end_node_id=top.id, lift_type="chairlift", dem=dem)
        fig = render_viewing_profile(kind=EntityKind.LIFT, entity_id=lift.id, graph=empty_graph)
        assert len(fig.data) > 0

"""Unit tests for ProfileChart elevation-profile rendering (ui/bottom_chart.py)."""

M = 111320.0


class TestProfileChartRendering:
    """Tests for elevation profile chart rendering."""

    def test_proposal_chart_renders(self, path_points_blue) -> None:
        """ProfileChart can render a proposal."""
        from skiresort_planner.model.proposed_path import ProposedPathSegment
        from skiresort_planner.ui.bottom_chart import ProfileChart

        proposal = ProposedPathSegment(
            points=path_points_blue,
            target_slope_pct=20.0,
            target_difficulty="blue",
            sector_name="Test",
        )

        chart = ProfileChart(width=800, height=300)
        fig = chart.render_proposal(proposal=proposal, proposed_segment_title="Test Segment")

        assert fig is not None, "Should produce a figure"
        assert len(fig.data) > 0, "Figure should have data traces"

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

        chart = ProfileChart(width=800, height=300)
        fig = chart.render_segment(segment=segment, difficulty="blue", title="Test Segment")

        assert fig is not None, "Should produce a figure"
        assert len(fig.data) > 0, "Figure should have data traces"

    def test_slope_chart_renders_with_segment_boundaries(self, empty_graph, path_points_blue) -> None:
        """render_slope produces a figure with a vline per segment boundary."""
        from skiresort_planner.model.proposed_path import ProposedPathSegment
        from skiresort_planner.ui.bottom_chart import ProfileChart

        empty_graph.commit_paths(paths=[ProposedPathSegment(points=path_points_blue, target_difficulty="blue")])
        slope = empty_graph.finish_slope(segment_ids=list(empty_graph.segments.keys()))

        fig = ProfileChart(width=800, height=300).render_slope(slope=slope, graph=empty_graph)
        assert len(fig.data) > 0

    def test_road_chart_renders_brown_with_climb_stats(self, empty_graph) -> None:
        """render_road produces a figure (brown, elevation-change caption)."""
        from skiresort_planner.model.path_point import PathPoint
        from skiresort_planner.model.path_segment import SegmentKind
        from skiresort_planner.model.proposed_path import ProposedPathSegment
        from skiresort_planner.ui.bottom_chart import ProfileChart

        pts = [PathPoint(lon=0.0, lat=0.0, elevation=2000.0), PathPoint(lon=300 / M, lat=0.0, elevation=1990.0)]
        empty_graph.commit_paths(
            paths=[ProposedPathSegment(points=pts, is_connector=True, kind=SegmentKind.ROAD)], record_undo=False
        )
        road = empty_graph.finish_road(segment_ids=[list(empty_graph.segments.keys())[-1]])

        fig = ProfileChart(width=800, height=300).render_road(road=road, graph=empty_graph)
        assert len(fig.data) > 0

    def test_comparison_chart_overlays_all_proposals(self, path_points_blue) -> None:
        """render_comparison overlays one trace per proposal."""
        from skiresort_planner.model.proposed_path import ProposedPathSegment
        from skiresort_planner.ui.bottom_chart import ProfileChart

        proposals = [
            ProposedPathSegment(points=path_points_blue, target_difficulty="blue"),
            ProposedPathSegment(points=path_points_blue, target_difficulty="blue"),
        ]
        fig = ProfileChart(width=800, height=300).render_comparison(proposals=proposals)
        assert len(fig.data) == len(proposals)

    def test_lift_chart_renders_terrain_cable_pylons(self, empty_graph, mock_dem_blue_slope) -> None:
        """render_lift produces a figure with terrain + cable + pylon traces."""
        from skiresort_planner.ui.bottom_chart import ProfileChart

        dem = mock_dem_blue_slope
        bottom, _ = empty_graph.get_or_create_node(
            lon=0.0, lat=-1000 / M, elevation=dem.get_elevation_or_raise(lon=0.0, lat=-1000 / M)
        )
        top, _ = empty_graph.get_or_create_node(
            lon=0.0, lat=0.0, elevation=dem.get_elevation_or_raise(lon=0.0, lat=0.0)
        )
        lift = empty_graph.add_lift(start_node_id=bottom.id, end_node_id=top.id, lift_type="chairlift", dem=dem)

        fig = ProfileChart(width=800, height=300).render_lift(lift=lift, graph=empty_graph)
        assert len(fig.data) > 0

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
        reloaded_kind = Enum("SegmentKind", {"SLOPE": "slope", "ROAD": "road"}, type=str)
        empty_graph.segments[seg_id].kind = reloaded_kind.ROAD  # type: ignore[assignment]
        assert empty_graph.segments[seg_id].kind is not SegmentKind.ROAD, "must be a different class instance"

        # Must not raise 'Unknown segment kind' and must still render the road.
        fig = render_building_profile(building_segments=[seg_id], building_name="Reloaded Road", graph=empty_graph)
        assert len(fig.data) > 0

    def test_proposal_preview_convenience(self, path_points_blue) -> None:
        """render_proposal_preview renders the selected proposal."""
        from skiresort_planner.model.proposed_path import ProposedPathSegment
        from skiresort_planner.ui.bottom_chart import render_proposal_preview

        proposals = [ProposedPathSegment(points=path_points_blue, target_difficulty="blue")]
        fig = render_proposal_preview(proposals=proposals, selected_idx=0)
        assert len(fig.data) > 0

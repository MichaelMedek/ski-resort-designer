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
        from skiresort_planner.model.proposed_path import ProposedPathSegment
        from skiresort_planner.ui.bottom_chart import ProfileChart

        pts = [PathPoint(lon=0.0, lat=0.0, elevation=2000.0), PathPoint(lon=300 / M, lat=0.0, elevation=1990.0)]
        empty_graph.commit_paths(paths=[ProposedPathSegment(points=pts, is_connector=True)], record_undo=False)
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

    def test_building_profiles_convenience(self, empty_graph, path_points_blue) -> None:
        """render_building_profiles builds a combined in-progress slope figure."""
        from skiresort_planner.model.proposed_path import ProposedPathSegment
        from skiresort_planner.ui.bottom_chart import render_building_profiles

        empty_graph.commit_paths(paths=[ProposedPathSegment(points=path_points_blue, target_difficulty="blue")])
        seg_ids = list(empty_graph.segments.keys())

        fig = render_building_profiles(building_segments=seg_ids, building_name="WIP", graph=empty_graph)
        assert len(fig.data) > 0

    def test_proposal_preview_convenience(self, path_points_blue) -> None:
        """render_proposal_preview renders the selected proposal."""
        from skiresort_planner.model.proposed_path import ProposedPathSegment
        from skiresort_planner.ui.bottom_chart import render_proposal_preview

        proposals = [ProposedPathSegment(points=path_points_blue, target_difficulty="blue")]
        fig = render_proposal_preview(proposals=proposals, selected_idx=0)
        assert len(fig.data) > 0

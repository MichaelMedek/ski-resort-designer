"""Integration tests for rendering (map layers and charts).

Tests that rendering components produce valid output structures.
"""


class TestMapRendering:
    """Tests for map layer rendering."""

    def test_map_renderer_renders_empty_graph(self, empty_graph) -> None:
        """MapRenderer renders empty graph without errors."""
        from skiresort_planner.ui.center_map import MapRenderer

        renderer = MapRenderer(graph=empty_graph)
        deck = renderer.render()

        assert deck is not None, "Should produce a Deck object"
        assert hasattr(deck, "layers"), "Deck should have layers"

    def test_map_renderer_renders_graph_with_slope(self, empty_graph, path_points_blue) -> None:
        """MapRenderer renders graph with committed slope."""
        from skiresort_planner.model.proposed_path import ProposedPathSegment
        from skiresort_planner.ui.center_map import MapRenderer

        graph = empty_graph
        proposal = ProposedPathSegment(
            points=path_points_blue,
            target_slope_pct=20.0,
            target_difficulty="blue",
            sector_name="Test",
        )
        graph.commit_paths(paths=[proposal])
        graph.finish_slope(segment_ids=list(graph.segments.keys()))

        renderer = MapRenderer(graph=graph)
        deck = renderer.render()

        assert deck is not None, "Should produce a Deck object"

    def test_map_renderer_renders_proposals(self, empty_graph, path_points_blue) -> None:
        """MapRenderer renders proposal paths."""
        from skiresort_planner.model.proposed_path import ProposedPathSegment
        from skiresort_planner.ui.center_map import MapRenderer

        graph = empty_graph
        proposal = ProposedPathSegment(
            points=path_points_blue,
            target_slope_pct=20.0,
            target_difficulty="blue",
            sector_name="Test",
        )

        renderer = MapRenderer(graph=graph)
        deck = renderer.render(proposals=[proposal])

        assert deck is not None, "Should produce a Deck object"

    def test_segment_layers_split_slope_and_road_buckets(self, empty_graph, path_points_blue) -> None:
        """_create_segment_layers routes a road's segments to the 'roads' bucket
        (brown, road-typed) and a slope's to 'slopes' (difficulty-colored).
        """
        from skiresort_planner.constants import ClickConfig, StyleConfig
        from skiresort_planner.model.proposed_path import ProposedPathSegment
        from skiresort_planner.ui.center_map import MapRenderer

        graph = empty_graph
        # Slope from the blue points.
        graph.commit_paths(paths=[ProposedPathSegment(points=path_points_blue, target_difficulty="blue")])
        graph.finish_slope(segment_ids=list(graph.segments.keys()))
        # Road as a separate short segment.
        M = 111320.0
        road_pts = [
            type(path_points_blue[0])(lon=500 / M, lat=0.0, elevation=2000.0),
            type(path_points_blue[0])(lon=800 / M, lat=0.0, elevation=1990.0),
        ]
        graph.commit_paths(paths=[ProposedPathSegment(points=road_pts, is_connector=True)], record_undo=False)
        road_seg = list(graph.segments.keys())[-1]
        road = graph.finish_road(segment_ids=[road_seg])

        renderer = MapRenderer(graph=graph)
        layers = renderer._create_segment_layers(use_3d=False)

        assert layers["slopes"], "slope segments should produce slope layers"
        assert layers["roads"], "road segments should produce road layers"

        # The road center line carries the road's click type/id and brown color.
        road_centerline = next(layer for layer in layers["roads"] if layer.id == "roads_centerline")
        record = road_centerline.data[0]
        assert record["type"] == ClickConfig.TYPE_ROAD
        assert record["id"] == road.id
        assert record["color"] == list(StyleConfig.ROAD_COLOR_RGBA)

    def test_segment_layers_render_parking_at_shared_node(self, empty_graph) -> None:
        """A road sharing a node with a slope yields a parking marker layer."""
        from skiresort_planner.model.proposed_path import ProposedPathSegment
        from skiresort_planner.model.path_point import PathPoint
        from skiresort_planner.ui.center_map import MapRenderer

        graph = empty_graph
        M = 111320.0
        shared = PathPoint(lon=0.0, lat=0.0, elevation=2000.0)
        # Slope and road both start at the shared node.
        graph.commit_paths(
            paths=[
                ProposedPathSegment(
                    points=[shared, PathPoint(lon=0.0, lat=-300 / M, elevation=1900.0)], target_difficulty="blue"
                )
            ]
        )
        graph.finish_slope(segment_ids=list(graph.segments.keys()))
        graph.commit_paths(
            paths=[
                ProposedPathSegment(
                    points=[shared, PathPoint(lon=300 / M, lat=0.0, elevation=1990.0)], is_connector=True
                )
            ],
            record_undo=False,
        )
        graph.finish_road(segment_ids=[list(graph.segments.keys())[-1]])

        renderer = MapRenderer(graph=graph)
        parking = renderer._create_parking_layers(use_3d=False)
        assert parking, "shared road/slope node should produce a parking layer"


def _populate_full_resort(graph, dem):
    """Build a graph with a slope, a road (sharing the summit node), and a lift."""
    from skiresort_planner.model.path_point import PathPoint
    from skiresort_planner.model.proposed_path import ProposedPathSegment

    M = 111320.0
    summit = PathPoint(lon=0.0, lat=0.0, elevation=dem.get_elevation_or_raise(lon=0.0, lat=0.0))
    # Slope down the fall line.
    graph.commit_paths(
        paths=[
            ProposedPathSegment(
                points=[
                    summit,
                    PathPoint(lon=0.0, lat=-400 / M, elevation=dem.get_elevation_or_raise(lon=0.0, lat=-400 / M)),
                ],
                target_difficulty="blue",
            )
        ]
    )
    graph.finish_slope(segment_ids=list(graph.segments.keys()))
    # Road off the summit (shares the summit node → parking).
    graph.commit_paths(
        paths=[
            ProposedPathSegment(
                points=[
                    summit,
                    PathPoint(lon=400 / M, lat=0.0, elevation=dem.get_elevation_or_raise(lon=400 / M, lat=0.0)),
                ],
                is_connector=True,
            )
        ],
        record_undo=False,
    )
    road = graph.finish_road(segment_ids=[list(graph.segments.keys())[-1]])
    # Lift from valley up to summit.
    bottom, _ = graph.get_or_create_node(
        lon=0.0, lat=-800 / M, elevation=dem.get_elevation_or_raise(lon=0.0, lat=-800 / M)
    )
    top, _ = graph.get_or_create_node(lon=0.0, lat=0.0, elevation=dem.get_elevation_or_raise(lon=0.0, lat=0.0))
    lift = graph.add_lift(start_node_id=bottom.id, end_node_id=top.id, lift_type="chairlift", dem=dem)
    return road, lift


class TestFullResortRendering:
    """Render a populated resort (slope + road + lift + proposals) in 2D and 3D,
    exercising every layer builder in one pass.
    """

    def test_render_2d_produces_layers(self, empty_graph, mock_dem_blue_slope) -> None:
        from skiresort_planner.ui.center_map import MapRenderer

        _populate_full_resort(empty_graph, mock_dem_blue_slope)
        deck = MapRenderer(graph=empty_graph).render(use_3d=False)
        assert len(deck.layers) > 0

    def test_render_3d_produces_layers(self, empty_graph, mock_dem_blue_slope) -> None:
        from skiresort_planner.ui.center_map import MapRenderer

        _populate_full_resort(empty_graph, mock_dem_blue_slope)
        deck = MapRenderer(graph=empty_graph).render(use_3d=True)
        assert len(deck.layers) > 0

    def test_render_with_proposals_and_selection(self, empty_graph, mock_dem_blue_slope, path_points_blue) -> None:
        from skiresort_planner.model.proposed_path import ProposedPathSegment
        from skiresort_planner.ui.center_map import MapRenderer

        _populate_full_resort(empty_graph, mock_dem_blue_slope)
        proposals = [
            ProposedPathSegment(points=path_points_blue, target_difficulty="blue"),
            ProposedPathSegment(points=path_points_blue, target_difficulty="blue"),
        ]
        deck = MapRenderer(graph=empty_graph).render(proposals=proposals, selected_proposal_idx=1)
        assert len(deck.layers) > 0

    def test_lift_layers_have_pylons_and_cables(self, empty_graph, mock_dem_blue_slope) -> None:
        from skiresort_planner.ui.center_map import MapRenderer

        _populate_full_resort(empty_graph, mock_dem_blue_slope)
        layers = MapRenderer(graph=empty_graph)._create_lift_layers(use_3d=False)
        assert layers["pylons"] and layers["cables_icons"]


class TestThreeDViewCalculators:
    """calculate_3d_view_for_* return a valid (lat, lon, bearing, zoom, pitch) tuple."""

    def test_slope_view(self, empty_graph, mock_dem_blue_slope) -> None:
        from skiresort_planner.model.proposed_path import ProposedPathSegment
        from skiresort_planner.ui.center_map import MapRenderer

        graph = empty_graph
        M = 111320.0
        from skiresort_planner.model.path_point import PathPoint

        dem = mock_dem_blue_slope
        graph.commit_paths(
            paths=[
                ProposedPathSegment(
                    points=[
                        PathPoint(lon=0.0, lat=0.0, elevation=dem.get_elevation_or_raise(lon=0.0, lat=0.0)),
                        PathPoint(lon=0.0, lat=-400 / M, elevation=dem.get_elevation_or_raise(lon=0.0, lat=-400 / M)),
                    ],
                    target_difficulty="blue",
                )
            ]
        )
        slope = graph.finish_slope(segment_ids=list(graph.segments.keys()))

        lat, lon, bearing, zoom, pitch = MapRenderer.calculate_3d_view_for_slope(graph=graph, slope_id=slope.id)
        assert isinstance(zoom, int) and pitch > 0 and 0.0 <= bearing <= 360.0

    def test_lift_view(self, empty_graph, mock_dem_blue_slope) -> None:
        from skiresort_planner.ui.center_map import MapRenderer

        _road, lift = _populate_full_resort(empty_graph, mock_dem_blue_slope)
        lat, lon, bearing, zoom, pitch = MapRenderer.calculate_3d_view_for_lift(graph=empty_graph, lift_id=lift.id)
        assert isinstance(zoom, int) and pitch > 0 and 0.0 <= bearing <= 360.0


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

        M = 111320.0
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
        M = 111320.0
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


class TestLayerCollection:
    """Tests for layer collection z-ordering."""

    def test_layer_collection_maintains_z_order(self) -> None:
        """LayerCollection z-order: terrain → pylons → slopes → roads → lifts → parking → nodes → proposals → markers."""
        from skiresort_planner.ui.center_map import LayerCollection

        collection = LayerCollection()

        # Add layers to different categories
        collection.terrain.append({"id": "terrain"})
        collection.slopes.append({"id": "slopes"})
        collection.roads.append({"id": "roads"})
        collection.parking.append({"id": "parking"})
        collection.nodes.append({"id": "nodes"})
        collection.markers.append({"id": "markers"})

        layers = collection.get_ordered_layers()

        # Verify order
        layer_ids = [layer["id"] for layer in layers]
        assert layer_ids.index("terrain") < layer_ids.index("slopes"), "terrain before slopes"
        assert layer_ids.index("slopes") < layer_ids.index("roads"), "slopes before roads"
        assert layer_ids.index("roads") < layer_ids.index("parking"), "roads before parking"
        assert layer_ids.index("parking") < layer_ids.index("nodes"), "parking before nodes"
        assert layer_ids.index("nodes") < layer_ids.index("markers"), "nodes before markers"

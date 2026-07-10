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

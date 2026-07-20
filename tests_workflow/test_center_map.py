"""Integration tests for the Pydeck map renderer (ui/center_map.py).

Covers MapRenderer layer building (slopes/roads/parking/lifts/proposals),
the 2D/3D full-resort render, LayerCollection z-ordering, and the
calculate_3d_view_for_* camera calculators.
"""

from skiresort_planner.constants import MapConfig
from skiresort_planner.model.path_segment import SegmentKind
from skiresort_planner.ui.click_detector import ClickDetector
from skiresort_planner.ui.context import ClickDeduplicationContext


def _populate_full_resort(graph, dem):
    """Build a graph with a slope, a road (sharing the summit node), and a lift."""
    from skiresort_planner.model.path_point import PathPoint
    from skiresort_planner.model.proposed_path import ProposedPathSegment

    summit = PathPoint(lon=0.0, lat=0.0, elevation=dem.get_elevation_or_raise(lon=0.0, lat=0.0))
    # Slope down the fall line.
    graph.commit_paths(
        paths=[
            ProposedPathSegment(
                points=[
                    summit,
                    PathPoint(
                        lon=0.0,
                        lat=-400 / MapConfig.METERS_PER_DEGREE_EQUATOR,
                        elevation=dem.get_elevation_or_raise(lon=0.0, lat=-400 / MapConfig.METERS_PER_DEGREE_EQUATOR),
                    ),
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
                    PathPoint(
                        lon=400 / MapConfig.METERS_PER_DEGREE_EQUATOR,
                        lat=0.0,
                        elevation=dem.get_elevation_or_raise(lon=400 / MapConfig.METERS_PER_DEGREE_EQUATOR, lat=0.0),
                    ),
                ],
                is_connector=True,
                kind=SegmentKind.ROAD,
            )
        ],
        record_undo=False,
    )
    road = graph.finish_road(segment_ids=[list(graph.segments.keys())[-1]])
    # Lift from valley up to summit.
    bottom, _ = graph.get_or_create_node(
        lon=0.0,
        lat=-800 / MapConfig.METERS_PER_DEGREE_EQUATOR,
        elevation=dem.get_elevation_or_raise(lon=0.0, lat=-800 / MapConfig.METERS_PER_DEGREE_EQUATOR),
    )
    top, _ = graph.get_or_create_node(lon=0.0, lat=0.0, elevation=dem.get_elevation_or_raise(lon=0.0, lat=0.0))
    lift = graph.add_lift(start_node_id=bottom.id, end_node_id=top.id, lift_type="chairlift", dem=dem)
    return road, lift


class TestGrayOut:
    """StyleConfig.gray_out — the pure color-muting helper behind the connectivity-defect dimming."""

    def test_blends_each_channel_strongly_to_128_and_keeps_alpha(self) -> None:
        from skiresort_planner.constants import StyleConfig

        # chairlift purple [168,85,247,200] pulled 75% toward 128; alpha untouched.
        assert StyleConfig.gray_out([168, 85, 247, 200]) == [138, 117, 158, 200]
        # A fully opaque black stays opaque; 31→round(31*.25+128*.75)=104 etc.
        assert StyleConfig.gray_out([31, 41, 55, 255]) == [104, 106, 110, 255]

    def test_gray_is_a_fixed_point(self) -> None:
        """Mid-gray blended toward mid-gray is unchanged — the muting converges there."""
        from skiresort_planner.constants import StyleConfig

        assert StyleConfig.gray_out([128, 128, 128, 200]) == [128, 128, 128, 200]

    def test_moves_color_toward_gray_not_away(self) -> None:
        """Every muted channel sits strictly between the original and 128 (never overshoots)."""
        from skiresort_planner.constants import StyleConfig

        for original in ([34, 197, 94, 200], [239, 68, 68, 200], [216, 180, 254, 200]):
            muted = StyleConfig.gray_out(original)
            for orig_c, muted_c in zip(original[:3], muted[:3], strict=True):
                lo, hi = sorted((orig_c, 128))
                assert lo <= muted_c <= hi


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

        from skiresort_planner.constants import StyleConfig

        renderer = MapRenderer(graph=graph)
        deck = renderer.render()

        assert deck is not None, "Should produce a Deck object"
        # 2D render: the slope's belt polygon layer exists and is blue-difficulty colored
        # (alpha dropped to 100 for the non-highlighted, semi-transparent belt).
        belt = next(layer for layer in deck.layers if layer.id == "segments_belt")
        expected_color = list(StyleConfig.SLOPE_COLORS_RGBA["blue"])
        expected_color[3] = 100
        assert belt.data[0]["color"] == expected_color

    def test_defective_slope_grays_belt_but_keeps_icon_hue(self, empty_graph, path_points_blue) -> None:
        """A slope in defect_ids mutes its belt/centerline toward gray, but the center-circle icon
        keeps its full difficulty hue so it stays a clear clickable marker. A non-defect slope keeps
        the belt hue too.
        """
        from skiresort_planner.constants import StyleConfig
        from skiresort_planner.model.proposed_path import ProposedPathSegment
        from skiresort_planner.ui.center_map import MapRenderer

        graph = empty_graph
        graph.commit_paths(paths=[ProposedPathSegment(points=path_points_blue, target_difficulty="blue")])
        slope = graph.finish_slope(segment_ids=list(graph.segments.keys()))
        renderer = MapRenderer(graph=graph)

        base = list(StyleConfig.SLOPE_COLORS_RGBA["blue"])
        # Non-defect control: belt keeps base color (alpha dropped to 100 for the belt).
        plain = renderer._create_segment_layers(use_3d=False)
        plain_belt = next(layer for layer in plain["slopes"] if layer.id == "segments_belt")
        assert plain_belt.data[0]["color"][:3] == base[:3]

        # Defect: belt grays, but the icon marker stays the full difficulty color.
        grayed = renderer._create_segment_layers(use_3d=False, defect_ids={slope.id})
        grayed_belt = next(layer for layer in grayed["slopes"] if layer.id == "segments_belt")
        grayed_icons = next(layer for layer in grayed["slopes"] if layer.id == "segments_icons")
        assert grayed_belt.data[0]["color"][:3] == StyleConfig.gray_out(base)[:3]
        assert grayed_icons.data[0]["color"] == list(StyleConfig.SLOPE_COLORS_RGBA["blue"])

    def test_defective_lift_grays_cable_but_keeps_icon_hue(self, empty_graph, mock_dem_blue_slope) -> None:
        """A lift in defect_ids mutes its cable toward gray, but the center icon keeps its per-type
        purple so it stays a clear clickable marker; the control cable keeps its purple too.
        """
        from skiresort_planner.constants import StyleConfig
        from skiresort_planner.ui.center_map import MapRenderer

        graph = empty_graph
        dem = mock_dem_blue_slope
        bottom, _ = graph.get_or_create_node(
            lon=0.0,
            lat=-1000 / MapConfig.METERS_PER_DEGREE_EQUATOR,
            elevation=dem.get_elevation_or_raise(lon=0.0, lat=-1000 / MapConfig.METERS_PER_DEGREE_EQUATOR),
        )
        top, _ = graph.get_or_create_node(lon=0.0, lat=0.0, elevation=dem.get_elevation_or_raise(lon=0.0, lat=0.0))
        lift = graph.add_lift(start_node_id=bottom.id, end_node_id=top.id, lift_type="chairlift", dem=dem)
        renderer = MapRenderer(graph=graph)
        base = list(StyleConfig.LIFT_COLORS_RGBA["chairlift"])

        plain = renderer._create_lift_layers(use_3d=False)
        plain_cable = next(layer for layer in plain["cables_icons"] if layer.id == "lift_cables")
        assert plain_cable.data[0]["color"][:3] == base[:3]

        grayed = renderer._create_lift_layers(use_3d=False, defect_ids={lift.id})
        grayed_cable = next(layer for layer in grayed["cables_icons"] if layer.id == "lift_cables")
        grayed_icons = next(layer for layer in grayed["cables_icons"] if layer.id == "lift_icons")
        assert grayed_cable.data[0]["color"] == StyleConfig.gray_out(base)
        assert grayed_icons.data[0]["color"] == base

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
        assert any(layer.id == "proposal_paths" for layer in deck.layers), "proposal path layer must render"

    def test_proposal_color_matches_kind(self, empty_graph, path_points_blue) -> None:
        """A ROAD proposal renders translucent brown; a slope proposal difficulty-colored."""
        from skiresort_planner.constants import StyleConfig
        from skiresort_planner.model.proposed_path import ProposedPathSegment
        from skiresort_planner.ui.center_map import MapRenderer

        renderer = MapRenderer(graph=empty_graph)

        road_proposal = ProposedPathSegment(points=path_points_blue, is_connector=True, kind=SegmentKind.ROAD)
        road_layers = renderer._create_proposal_layers(proposals=[road_proposal], selected_idx=None, use_3d=False)
        road_path_layer = next(layer for layer in road_layers if layer.id == "proposal_paths")
        assert road_path_layer.data[0]["color"][:3] == StyleConfig.ROAD_PROPOSAL_COLOR_RGBA[:3]

        slope_proposal = ProposedPathSegment(points=path_points_blue, target_difficulty="blue")
        slope_layers = renderer._create_proposal_layers(proposals=[slope_proposal], selected_idx=None, use_3d=False)
        slope_path_layer = next(layer for layer in slope_layers if layer.id == "proposal_paths")
        assert slope_path_layer.data[0]["color"][:3] == list(StyleConfig.SLOPE_COLORS_RGBA["blue"])[:3]

    def test_custom_path_proposals_have_no_endpoint_markers(self, empty_graph, path_points_blue) -> None:
        """is_custom_path=True (custom-connect AND roads) suppresses proposal endpoint markers.

        Their proposals all run to the same clicked target, so overlapping endpoint
        dots would be ambiguous — commit is button-only. Fan-out (is_custom_path=False)
        keeps distinct endpoint markers.
        """
        from skiresort_planner.model.proposed_path import ProposedPathSegment
        from skiresort_planner.ui.center_map import MapRenderer

        renderer = MapRenderer(graph=empty_graph)
        proposal = ProposedPathSegment(points=path_points_blue, is_connector=True, kind=SegmentKind.ROAD)

        custom = renderer._create_proposal_layers(
            proposals=[proposal], selected_idx=0, is_custom_path=True, use_3d=False
        )
        assert not any(layer.id == "proposal_endpoints" for layer in custom), (
            "custom/road proposals must NOT render endpoint markers"
        )

        fan = renderer._create_proposal_layers(proposals=[proposal], selected_idx=0, is_custom_path=False, use_3d=False)
        assert any(layer.id == "proposal_endpoints" for layer in fan), "fan-out proposals keep endpoint markers"

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
        road_pts = [
            type(path_points_blue[0])(lon=500 / MapConfig.METERS_PER_DEGREE_EQUATOR, lat=0.0, elevation=2000.0),
            type(path_points_blue[0])(lon=800 / MapConfig.METERS_PER_DEGREE_EQUATOR, lat=0.0, elevation=1990.0),
        ]
        graph.commit_paths(
            paths=[ProposedPathSegment(points=road_pts, is_connector=True, kind=SegmentKind.ROAD)], record_undo=False
        )
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

        # The road icon tooltip carries the road icon + name (hover feedback for a finished road).
        road_icons = next(layer for layer in layers["roads"] if layer.id == "roads_icons")
        assert road_icons.data[0]["name"] == f"{StyleConfig.ROAD_ICON} {road.name}"

        # The slope icon tooltip carries the slope icon + name.
        slope = next(iter(graph.slopes.values()))
        slope_icons = next(layer for layer in layers["slopes"] if layer.id == "segments_icons")
        assert slope_icons.data[0]["name"] == f"{StyleConfig.SLOPE_ICON} {slope.name}"

    def test_lift_icon_tooltip_carries_lift_icon_and_name(self, empty_graph, mock_dem_blue_slope) -> None:
        """A finished lift's cable/icon tooltips show the per-type lift icon + name."""
        from skiresort_planner.constants import StyleConfig
        from skiresort_planner.ui.center_map import MapRenderer

        graph = empty_graph
        dem = mock_dem_blue_slope
        bottom, _ = graph.get_or_create_node(
            lon=0.0,
            lat=-1000 / MapConfig.METERS_PER_DEGREE_EQUATOR,
            elevation=dem.get_elevation_or_raise(lon=0.0, lat=-1000 / MapConfig.METERS_PER_DEGREE_EQUATOR),
        )
        top, _ = graph.get_or_create_node(lon=0.0, lat=0.0, elevation=dem.get_elevation_or_raise(lon=0.0, lat=0.0))
        lift = graph.add_lift(start_node_id=bottom.id, end_node_id=top.id, lift_type="chairlift", dem=dem)

        layers = MapRenderer(graph=graph)._create_lift_layers(use_3d=False)
        expected = f"{StyleConfig.LIFT_ICONS[lift.lift_type]} {lift.name}"
        names = [d["name"] for layer in layers["cables_icons"] for d in layer.data if "name" in d and "lift_type" in d]
        assert names, "lift cable/icon records should exist"
        assert all(name == expected for name in names), names

    def test_pylon_tooltip_uses_lift_display_name_not_id(self, empty_graph, mock_dem_blue_slope) -> None:
        """A pylon's hover tooltip names its lift by display name (icon + name), not the raw "L5" id."""
        from skiresort_planner.constants import StyleConfig
        from skiresort_planner.ui.center_map import MapRenderer

        graph = empty_graph
        dem = mock_dem_blue_slope
        bottom, _ = graph.get_or_create_node(
            lon=0.0,
            lat=-1000 / MapConfig.METERS_PER_DEGREE_EQUATOR,
            elevation=dem.get_elevation_or_raise(lon=0.0, lat=-1000 / MapConfig.METERS_PER_DEGREE_EQUATOR),
        )
        top, _ = graph.get_or_create_node(lon=0.0, lat=0.0, elevation=dem.get_elevation_or_raise(lon=0.0, lat=0.0))
        lift = graph.add_lift(start_node_id=bottom.id, end_node_id=top.id, lift_type="chairlift", dem=dem)

        layers = MapRenderer(graph=graph)._create_lift_layers(use_3d=False)
        pylon_layer = next(layer for layer in layers["pylons"] if layer.id == "lift_pylons")
        assert pylon_layer.data, "a lift should have pylons"
        expected_suffix = f"{StyleConfig.LIFT_ICONS[lift.lift_type]} {lift.name}"
        for record in pylon_layer.data:
            assert record["name"].endswith(expected_suffix), record["name"]
            assert lift.id not in record["name"], "tooltip must not fall back to the raw lift id"

    def test_segment_layers_render_parking_at_shared_node(self, empty_graph) -> None:
        """A road sharing a node with a slope renders that node as a parking marker.

        Parking is not a separate layer — the shared node in the *nodes* layer gets
        the parking color, larger radius, and a "Parking place" tooltip.
        """
        from skiresort_planner.constants import ClickConfig, MarkerConfig, StyleConfig
        from skiresort_planner.model.path_point import PathPoint
        from skiresort_planner.model.proposed_path import ProposedPathSegment
        from skiresort_planner.ui.center_map import MapRenderer

        graph = empty_graph
        shared = PathPoint(lon=0.0, lat=0.0, elevation=2000.0)
        # Slope and road both start at the shared node.
        graph.commit_paths(
            paths=[
                ProposedPathSegment(
                    points=[
                        shared,
                        PathPoint(lon=0.0, lat=-300 / MapConfig.METERS_PER_DEGREE_EQUATOR, elevation=1900.0),
                    ],
                    target_difficulty="blue",
                )
            ]
        )
        graph.finish_slope(segment_ids=list(graph.segments.keys()))
        graph.commit_paths(
            paths=[
                ProposedPathSegment(
                    points=[
                        shared,
                        PathPoint(lon=300 / MapConfig.METERS_PER_DEGREE_EQUATOR, lat=0.0, elevation=1990.0),
                    ],
                    is_connector=True,
                    kind=SegmentKind.ROAD,
                )
            ],
        )
        graph.finish_road(segment_ids=[list(graph.segments.keys())[-1]])

        parking_ids = {n.id for n in graph.get_parking_nodes()}
        assert parking_ids, "shared road/slope node should be a parking node"

        renderer = MapRenderer(graph=graph)
        node_layer = renderer._create_node_layer(use_3d=False)
        parking_points = [d for d in node_layer.data if d["id"] in parking_ids]
        assert parking_points, "parking node should appear in the nodes layer"
        for point in parking_points:
            assert point["color"] == list(StyleConfig.PARKING_COLOR_RGBA)
            assert point["radius"] == ClickConfig.NODE_MARKER_RADIUS_BIG
            assert point["radius"] > ClickConfig.NODE_MARKER_RADIUS, "parking marker must be bigger than a plain node"
            assert StyleConfig.PARKING_ICON in point["name"] and "Parking place" in point["name"]

        # Non-parking nodes keep the normal white node styling.
        plain_points = [d for d in node_layer.data if d["id"] not in parking_ids]
        for point in plain_points:
            assert point["color"] == list(MarkerConfig.NODE_MARKER_COLOR)
            assert point["radius"] == ClickConfig.NODE_MARKER_RADIUS

    def test_selected_node_overrides_parking_style(self, empty_graph) -> None:
        """A node in selected_node_ids renders RED + big with a neutral "Selected" tooltip; selection
        styling wins over the parking (blue) style even for a shared road/slope parking node.
        """
        from skiresort_planner.constants import ClickConfig, StyleConfig
        from skiresort_planner.model.path_point import PathPoint
        from skiresort_planner.model.proposed_path import ProposedPathSegment
        from skiresort_planner.ui.center_map import MapRenderer

        graph = empty_graph
        shared = PathPoint(lon=0.0, lat=0.0, elevation=2000.0)
        graph.commit_paths(
            paths=[
                ProposedPathSegment(
                    points=[
                        shared,
                        PathPoint(lon=0.0, lat=-300 / MapConfig.METERS_PER_DEGREE_EQUATOR, elevation=1900.0),
                    ],
                    target_difficulty="blue",
                )
            ]
        )
        graph.finish_slope(segment_ids=list(graph.segments.keys()))
        graph.commit_paths(
            paths=[
                ProposedPathSegment(
                    points=[
                        shared,
                        PathPoint(lon=300 / MapConfig.METERS_PER_DEGREE_EQUATOR, lat=0.0, elevation=1990.0),
                    ],
                    is_connector=True,
                    kind=SegmentKind.ROAD,
                )
            ],
        )
        graph.finish_road(segment_ids=[list(graph.segments.keys())[-1]])

        parking_ids = {n.id for n in graph.get_parking_nodes()}
        assert parking_ids, "shared road/slope node should be a parking node"
        selected_id = next(iter(parking_ids))  # select the parking node itself

        node_layer = MapRenderer(graph=graph)._create_node_layer(use_3d=False, selected_node_ids=[selected_id])
        record = next(d for d in node_layer.data if d["id"] == selected_id)
        # Selection wins over parking: red (not parking blue), big radius, neutral "Selected" tooltip.
        assert record["color"] == list(StyleConfig.SELECTED_NODE_RGBA)
        assert record["color"] != list(StyleConfig.PARKING_COLOR_RGBA)
        assert record["radius"] == ClickConfig.NODE_MARKER_RADIUS_BIG
        assert "Selected" in record["name"] and "merge" not in record["name"].lower()


class TestFullResortRendering:
    """Render a populated resort (slope + road + lift + proposals) in 2D and 3D,
    exercising every layer builder in one pass.
    """

    def test_render_2d_produces_layers(self, empty_graph, mock_dem_blue_slope) -> None:
        from skiresort_planner.ui.center_map import MapRenderer

        _populate_full_resort(empty_graph, mock_dem_blue_slope)
        deck = MapRenderer(graph=empty_graph).render(use_3d=False)
        assert len(deck.layers) > 0
        # 2D: slopes render a belt polygon, and center lines sit at the flat 2D slope z-offset.
        assert any(layer.id == "segments_belt" for layer in deck.layers), "2D must build a belt polygon"
        centerline = next(layer for layer in deck.layers if layer.id == "segments_centerline")
        assert centerline.data[0]["center_line"][0][2] == MapConfig.Z_OFFSET_2D_SLOPES

    def test_render_3d_produces_layers(self, empty_graph, mock_dem_blue_slope) -> None:
        from skiresort_planner.constants import MarkerConfig
        from skiresort_planner.ui.center_map import MapRenderer

        _populate_full_resort(empty_graph, mock_dem_blue_slope)
        deck = MapRenderer(graph=empty_graph).render(use_3d=True)
        assert len(deck.layers) > 0
        # 3D: PolygonLayer belts are dropped (no z support); center lines carry real terrain
        # elevation lifted by PATH_Z_OFFSET_M for visibility.
        assert not any(layer.id.endswith("_belt") for layer in deck.layers), "3D must not build belt polygons"
        centerline = next(layer for layer in deck.layers if layer.id == "segments_centerline")
        first = centerline.data[0]["center_line"][0]
        assert (
            first[2]
            == mock_dem_blue_slope.get_elevation_or_raise(lon=first[0], lat=first[1]) + MarkerConfig.PATH_Z_OFFSET_M
        )

    def test_3d_slope_renders_at_real_belt_width(self, empty_graph, path_points_blue) -> None:
        """In 3D the center-line PathLayer IS the belt: rendered at each segment's real width_m (a
        terrain-draped ribbon; deck.gl widths are metres by default). 2D keeps the thin belt+line.
        """
        from skiresort_planner.model.proposed_path import ProposedPathSegment
        from skiresort_planner.ui.center_map import MapRenderer

        graph = empty_graph
        graph.commit_paths(paths=[ProposedPathSegment(points=path_points_blue, target_difficulty="blue")])
        graph.finish_slope(segment_ids=list(graph.segments.keys()))
        seg_width = next(iter(graph.segments.values())).width_m
        renderer = MapRenderer(graph=graph)

        layers_3d = renderer._create_segment_layers(use_3d=True)
        centerline_3d = next(layer for layer in layers_3d["slopes"] if layer.id == "segments_centerline")
        # width_units is NOT set (pydeck would mangle the string into a "@@=" accessor and break the
        # layer); metres is the deck.gl default. get_width="width" reads the per-record width_m.
        assert getattr(centerline_3d, "width_units", None) is None
        assert centerline_3d.get_width == "@@=width"
        assert centerline_3d.data[0]["width"] == seg_width
        assert not any(layer.id.endswith("_belt") for layer in layers_3d["slopes"]), "3D has no flat belt"

        # 2D: fixed thin line (numeric literal, unmangled) over the belt polygon.
        layers_2d = renderer._create_segment_layers(use_3d=False)
        centerline_2d = next(layer for layer in layers_2d["slopes"] if layer.id == "segments_centerline")
        assert centerline_2d.get_width == 4
        assert any(layer.id == "segments_belt" for layer in layers_2d["slopes"]), "2D has a belt polygon"

    def test_3d_lift_cable_renders_at_ten_meter_width(self, empty_graph, mock_dem_blue_slope) -> None:
        """The lift cable is a 10m-wide ribbon (CABLE_WIDTH metres) so it drapes over terrain in 3D."""
        from skiresort_planner.constants import MarkerConfig
        from skiresort_planner.ui.center_map import MapRenderer

        graph = empty_graph
        dem = mock_dem_blue_slope
        bottom, _ = graph.get_or_create_node(
            lon=0.0,
            lat=-1000 / MapConfig.METERS_PER_DEGREE_EQUATOR,
            elevation=dem.get_elevation_or_raise(lon=0.0, lat=-1000 / MapConfig.METERS_PER_DEGREE_EQUATOR),
        )
        top, _ = graph.get_or_create_node(lon=0.0, lat=0.0, elevation=dem.get_elevation_or_raise(lon=0.0, lat=0.0))
        graph.add_lift(start_node_id=bottom.id, end_node_id=top.id, lift_type="chairlift", dem=dem)

        layers = MapRenderer(graph=graph)._create_lift_layers(use_3d=True)
        cable = next(layer for layer in layers["cables_icons"] if layer.id == "lift_cables")
        assert (
            getattr(cable, "width_units", None) is None
        )  # unset → deck.gl default metres (setting it breaks the layer)
        assert cable.get_width == MarkerConfig.CABLE_WIDTH == 10  # numeric literal passes through unprefixed

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
        from skiresort_planner.model.path_point import PathPoint
        from skiresort_planner.model.proposed_path import ProposedPathSegment
        from skiresort_planner.ui.center_map import MapRenderer

        graph = empty_graph
        dem = mock_dem_blue_slope
        graph.commit_paths(
            paths=[
                ProposedPathSegment(
                    points=[
                        PathPoint(lon=0.0, lat=0.0, elevation=dem.get_elevation_or_raise(lon=0.0, lat=0.0)),
                        PathPoint(
                            lon=0.0,
                            lat=-400 / MapConfig.METERS_PER_DEGREE_EQUATOR,
                            elevation=dem.get_elevation_or_raise(
                                lon=0.0, lat=-400 / MapConfig.METERS_PER_DEGREE_EQUATOR
                            ),
                        ),
                    ],
                    target_difficulty="blue",
                )
            ]
        )
        slope = graph.finish_slope(segment_ids=list(graph.segments.keys()))

        from skiresort_planner.core.geo_calculator import GeoCalculator

        start = graph.nodes[slope.start_node_id]
        end = graph.nodes[slope.end_node_id]
        lat, lon, bearing, zoom, pitch = MapRenderer.calculate_3d_view_for_slope(graph=graph, slope_id=slope.id)
        assert lat == (start.lat + end.lat) / 2
        assert lon == (start.lon + end.lon) / 2
        assert pitch == MapConfig.VIEW_3D_PITCH
        feature_bearing = GeoCalculator.initial_bearing_deg(lon1=start.lon, lat1=start.lat, lon2=end.lon, lat2=end.lat)
        assert bearing == (feature_bearing - 90) % 360
        # High-elevation slope zooms out from the base 3D zoom, but never below the floor.
        assert MapConfig.VIEW_3D_MIN_ZOOM <= zoom < MapConfig.VIEW_3D_ZOOM

    def test_lift_view(self, empty_graph, mock_dem_blue_slope) -> None:
        from skiresort_planner.core.geo_calculator import GeoCalculator
        from skiresort_planner.ui.center_map import MapRenderer

        _road, lift = _populate_full_resort(empty_graph, mock_dem_blue_slope)
        start = empty_graph.nodes[lift.start_node_id]
        end = empty_graph.nodes[lift.end_node_id]
        lat, lon, bearing, zoom, pitch = MapRenderer.calculate_3d_view_for_lift(graph=empty_graph, lift_id=lift.id)
        assert lat == (start.lat + end.lat) / 2
        assert lon == (start.lon + end.lon) / 2
        assert pitch == MapConfig.VIEW_3D_PITCH
        feature_bearing = GeoCalculator.initial_bearing_deg(lon1=start.lon, lat1=start.lat, lon2=end.lon, lat2=end.lat)
        assert bearing == (feature_bearing - 90) % 360
        assert MapConfig.VIEW_3D_MIN_ZOOM <= zoom < MapConfig.VIEW_3D_ZOOM

    def test_road_view(self, empty_graph, path_points_blue) -> None:
        from skiresort_planner.model.proposed_path import ProposedPathSegment
        from skiresort_planner.ui.center_map import MapRenderer

        proposal = ProposedPathSegment(points=path_points_blue, is_connector=True, kind=SegmentKind.ROAD)
        empty_graph.commit_paths(paths=[proposal], record_undo=False)
        road = empty_graph.finish_road(segment_ids=[list(empty_graph.segments.keys())[-1]])

        from skiresort_planner.core.geo_calculator import GeoCalculator

        start = empty_graph.nodes[road.start_node_id]
        end = empty_graph.nodes[road.end_node_id]
        lat, lon, bearing, zoom, pitch = MapRenderer.calculate_3d_view_for_road(graph=empty_graph, road_id=road.id)
        assert lat == (start.lat + end.lat) / 2
        assert lon == (start.lon + end.lon) / 2
        assert pitch == MapConfig.VIEW_3D_PITCH
        feature_bearing = GeoCalculator.initial_bearing_deg(lon1=start.lon, lat1=start.lat, lon2=end.lon, lat2=end.lat)
        assert bearing == (feature_bearing - 90) % 360
        assert MapConfig.VIEW_3D_MIN_ZOOM <= zoom < MapConfig.VIEW_3D_ZOOM


class TestLayerCollection:
    """Tests for layer collection z-ordering."""

    def test_layer_collection_maintains_z_order(self) -> None:
        """LayerCollection z-order: terrain → pylons → slopes → roads → lifts → nodes → proposals → markers.

        Parking is not its own bucket — a parking node renders inside the nodes layer.
        """
        from skiresort_planner.ui.center_map import LayerCollection

        collection = LayerCollection()

        # Add layers to different categories
        collection.terrain.append({"id": "terrain"})
        collection.slopes.append({"id": "slopes"})
        collection.roads.append({"id": "roads"})
        collection.nodes.append({"id": "nodes"})
        collection.markers.append({"id": "markers"})

        layers = collection.get_ordered_layers()

        # Verify order
        layer_ids = [layer["id"] for layer in layers]
        assert layer_ids.index("terrain") < layer_ids.index("slopes"), "terrain before slopes"
        assert layer_ids.index("slopes") < layer_ids.index("roads"), "slopes before roads"
        assert layer_ids.index("roads") < layer_ids.index("nodes"), "roads before nodes"
        assert layer_ids.index("nodes") < layer_ids.index("markers"), "nodes before markers"


class TestImportBoxLayers:
    """create_import_bbox_layers draws a square + a PICKABLE center dot tagged so a click on it
    is classified as an IMPORT_CENTER confirm. Guards the confirm loop end-to-end with the detector.
    """

    def test_box_and_pickable_center_with_confirm_tag(self, empty_graph) -> None:
        from skiresort_planner.constants import ClickConfig, StyleConfig
        from skiresort_planner.model.click_info import MapClickType, MarkerType
        from skiresort_planner.ui.center_map import MapRenderer

        renderer = MapRenderer(graph=empty_graph)
        layers = renderer.create_import_bbox_layers(
            center_lon=10.3, center_lat=47.0, half_width_m=2000.0, elevation=2000.0, use_3d=False
        )
        by_id = {layer.id: layer for layer in layers}

        # The square: a closed 5-point ring in blue.
        box = by_id["import_bbox"]
        ring = box.data[0]["polygon"]
        assert len(ring) == 5 and ring[0] == ring[-1], "closed rectangle ring"
        assert box.get_fill_color == list(StyleConfig.IMPORT_BOX_RGBA)

        # The center dot: pickable + tagged so the detector routes a click to confirm.
        center = by_id["import_center"]
        assert center.pickable, "center dot must be clickable to confirm"
        marker = center.data[0]
        assert marker["type"] == ClickConfig.TYPE_IMPORT_CENTER

        # End-to-end: feed the dot's data through the REAL detector → IMPORT_CENTER marker.
        class _Dedup(ClickDeduplicationContext):
            def is_new_click(self, coord: tuple[float, ...] | None, obj_id: str | None) -> bool:
                return True

        info = ClickDetector(dedup=_Dedup()).detect(clicked_object=marker, clicked_coordinate=None)
        assert info is not None
        assert info.click_type == MapClickType.MARKER and info.marker_type == MarkerType.IMPORT_CENTER

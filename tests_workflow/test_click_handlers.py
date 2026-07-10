"""Unit tests for the four map click handlers, driven via the fake `st` session.

The click handlers read st.session_state.{state_machine, context, graph,
path_factory, dem_service}; with the fake `st` installed we seed those and call
the handler directly, asserting the real routing/commit logic without a browser.

One class per handler, symmetric across slope / lift / road so every build
mode's click flow is exercised the same way:
    handle_idle_click            → TestIdleClickRouting
    handle_slope_building_click  → TestSlopeBuildingClick
    handle_lift_placing_click    → TestLiftPlacingClick
    handle_road_building_click  → TestRoadBuildingClick
"""

from skiresort_planner.constants import PathConfig
from skiresort_planner.model.click_info import ClickInfo, MapClickType, MarkerType
from skiresort_planner.model.path_point import PathPoint
from skiresort_planner.model.path_segment import SegmentKind
from skiresort_planner.model.proposed_path import ProposedPathSegment
from skiresort_planner.model.resort_graph import ResortGraph
from skiresort_planner.ui.context import BuildMode
from skiresort_planner.ui.state_machine import PlannerStateMachine

M = 111320.0  # metres per degree near the equator


def _session(fake_st, graph, factory, dem):
    """Seed fake st.session_state with the objects the click handlers read."""
    sm, ctx = PlannerStateMachine.create(graph=graph, add_ui_listener=False)
    fake_st.session_state["state_machine"] = sm
    fake_st.session_state["context"] = ctx
    fake_st.session_state["graph"] = graph
    fake_st.session_state["path_factory"] = factory
    fake_st.session_state["dem_service"] = dem
    fake_st.session_state["map_version"] = 0
    return sm, ctx


def _commit_road(graph: ResortGraph):
    """Build a finished road on the graph and return it."""
    pts = [PathPoint(lon=0.0, lat=0.0, elevation=2000.0), PathPoint(lon=300 / M, lat=0.0, elevation=1990.0)]
    graph.commit_paths(
        paths=[ProposedPathSegment(points=pts, is_connector=True, kind=SegmentKind.ROAD)], record_undo=False
    )
    return graph.finish_road(segment_ids=[list(graph.segments.keys())[-1]])


# =============================================================================
# handle_idle_click — dispatch to the right build flow / open an entity panel
# =============================================================================


class TestIdleClickRouting:
    def test_slope_mode_terrain_click_starts_building(self, fake_st, path_factory, mock_dem_red_slope_diagonal) -> None:
        from skiresort_planner.ui.click_handlers import handle_idle_click

        dem = mock_dem_red_slope_diagonal
        sm, ctx = _session(fake_st, ResortGraph(), path_factory, dem)
        ctx.build_mode.mode = BuildMode.SLOPE

        handle_idle_click(
            ClickInfo(click_type=MapClickType.TERRAIN, lat=0.0, lon=0.0),
            elevation=dem.get_elevation_or_raise(lon=0.0, lat=0.0),
        )
        assert sm.is_slope_starting

    def test_lift_mode_terrain_click_starts_placing(self, fake_st, path_factory, mock_dem_red_slope_diagonal) -> None:
        from skiresort_planner.ui.click_handlers import handle_idle_click

        dem = mock_dem_red_slope_diagonal
        sm, ctx = _session(fake_st, ResortGraph(), path_factory, dem)
        ctx.build_mode.mode = BuildMode.CHAIRLIFT

        handle_idle_click(
            ClickInfo(click_type=MapClickType.TERRAIN, lat=0.0, lon=0.0),
            elevation=dem.get_elevation_or_raise(lon=0.0, lat=0.0),
        )
        assert sm.is_lift_placing

    def test_road_mode_node_click_starts_building(self, fake_st, path_factory, mock_dem_red_slope_diagonal) -> None:
        from skiresort_planner.ui.click_handlers import handle_idle_click

        graph = ResortGraph()
        node, _ = graph.get_or_create_node(lon=0.0, lat=0.0, elevation=2000.0)
        sm, ctx = _session(fake_st, graph, path_factory, mock_dem_red_slope_diagonal)
        ctx.build_mode.mode = BuildMode.ROAD

        handle_idle_click(
            ClickInfo(click_type=MapClickType.MARKER, marker_type=MarkerType.NODE, node_id=node.id), elevation=None
        )
        assert sm.is_road_starting
        assert ctx.road_build.start_node_id == node.id

    def test_slope_mode_node_click_starts_building(self, fake_st, path_factory, mock_dem_red_slope_diagonal) -> None:
        from skiresort_planner.ui.click_handlers import handle_idle_click

        graph = ResortGraph()
        node, _ = graph.get_or_create_node(lon=0.0, lat=0.0, elevation=2000.0)
        sm, ctx = _session(fake_st, graph, path_factory, mock_dem_red_slope_diagonal)
        ctx.build_mode.mode = BuildMode.SLOPE

        handle_idle_click(
            ClickInfo(click_type=MapClickType.MARKER, marker_type=MarkerType.NODE, node_id=node.id), elevation=None
        )
        assert sm.is_slope_starting

    def test_lift_mode_node_click_starts_placing(self, fake_st, path_factory, mock_dem_red_slope_diagonal) -> None:
        from skiresort_planner.ui.click_handlers import handle_idle_click

        graph = ResortGraph()
        node, _ = graph.get_or_create_node(lon=0.0, lat=0.0, elevation=2000.0)
        sm, ctx = _session(fake_st, graph, path_factory, mock_dem_red_slope_diagonal)
        ctx.build_mode.mode = BuildMode.CHAIRLIFT

        handle_idle_click(
            ClickInfo(click_type=MapClickType.MARKER, marker_type=MarkerType.NODE, node_id=node.id), elevation=None
        )
        assert sm.is_lift_placing
        assert ctx.lift.start_node_id == node.id

    def test_click_pylon_opens_parent_lift_panel(self, fake_st, path_factory, mock_dem_blue_slope) -> None:
        from skiresort_planner.ui.click_handlers import handle_idle_click

        dem = mock_dem_blue_slope
        graph = ResortGraph()
        bottom, _ = graph.get_or_create_node(
            lon=0.0, lat=-1000 / M, elevation=dem.get_elevation_or_raise(lon=0.0, lat=-1000 / M)
        )
        top, _ = graph.get_or_create_node(lon=0.0, lat=0.0, elevation=dem.get_elevation_or_raise(lon=0.0, lat=0.0))
        lift = graph.add_lift(start_node_id=bottom.id, end_node_id=top.id, lift_type="chairlift", dem=dem)
        sm, ctx = _session(fake_st, graph, path_factory, dem)

        handle_idle_click(
            ClickInfo(click_type=MapClickType.MARKER, marker_type=MarkerType.PYLON, lift_id=lift.id, pylon_index=0),
            elevation=None,
        )
        assert sm.is_idle_viewing_lift
        assert ctx.viewing.lift_id == lift.id

    def test_click_existing_slope_opens_panel(
        self, fake_st, path_factory, mock_dem_red_slope_diagonal, path_points_blue
    ) -> None:
        from skiresort_planner.ui.click_handlers import handle_idle_click

        graph = ResortGraph()
        graph.commit_paths(paths=[ProposedPathSegment(points=path_points_blue, target_difficulty="blue")])
        slope = graph.finish_slope(segment_ids=list(graph.segments.keys()))
        sm, ctx = _session(fake_st, graph, path_factory, mock_dem_red_slope_diagonal)

        handle_idle_click(
            ClickInfo(click_type=MapClickType.MARKER, marker_type=MarkerType.SLOPE, slope_id=slope.id), elevation=None
        )
        assert sm.is_idle_viewing_slope
        assert ctx.viewing.slope_id == slope.id

    def test_click_existing_road_opens_panel(self, fake_st, path_factory, mock_dem_red_slope_diagonal) -> None:
        from skiresort_planner.ui.click_handlers import handle_idle_click

        graph = ResortGraph()
        road = _commit_road(graph)
        sm, ctx = _session(fake_st, graph, path_factory, mock_dem_red_slope_diagonal)

        handle_idle_click(
            ClickInfo(click_type=MapClickType.MARKER, marker_type=MarkerType.ROAD, road_id=road.id), elevation=None
        )
        assert sm.is_idle_viewing_road
        assert ctx.viewing.road_id == road.id

    def test_click_segment_opens_parent_slope_panel(
        self, fake_st, path_factory, mock_dem_red_slope_diagonal, path_points_blue
    ) -> None:
        from skiresort_planner.ui.click_handlers import handle_idle_click

        graph = ResortGraph()
        graph.commit_paths(paths=[ProposedPathSegment(points=path_points_blue, target_difficulty="blue")])
        slope = graph.finish_slope(segment_ids=list(graph.segments.keys()))
        seg_id = slope.segment_ids[0]
        sm, ctx = _session(fake_st, graph, path_factory, mock_dem_red_slope_diagonal)

        handle_idle_click(
            ClickInfo(click_type=MapClickType.MARKER, marker_type=MarkerType.SEGMENT, segment_id=seg_id), elevation=None
        )
        assert sm.is_idle_viewing_slope
        assert ctx.viewing.slope_id == slope.id

    def test_click_lift_opens_panel_and_syncs_mode(self, fake_st, path_factory, mock_dem_blue_slope) -> None:
        from skiresort_planner.ui.click_handlers import handle_idle_click

        dem = mock_dem_blue_slope
        graph = ResortGraph()
        bottom, _ = graph.get_or_create_node(
            lon=0.0, lat=-1000 / M, elevation=dem.get_elevation_or_raise(lon=0.0, lat=-1000 / M)
        )
        top, _ = graph.get_or_create_node(lon=0.0, lat=0.0, elevation=dem.get_elevation_or_raise(lon=0.0, lat=0.0))
        lift = graph.add_lift(start_node_id=bottom.id, end_node_id=top.id, lift_type="gondola", dem=dem)
        sm, ctx = _session(fake_st, graph, path_factory, dem)

        handle_idle_click(
            ClickInfo(click_type=MapClickType.MARKER, marker_type=MarkerType.LIFT, lift_id=lift.id), elevation=None
        )
        assert sm.is_idle_viewing_lift
        assert ctx.viewing.lift_id == lift.id
        assert ctx.build_mode.mode == "gondola"  # build mode synced to the viewed lift


class TestCustomConnectClick:
    """A custom-connect target click (while building) transitions to custom picking."""

    def test_downhill_terrain_target_transitions_to_custom_path(
        self, fake_st, path_factory, mock_dem_blue_slope
    ) -> None:
        from skiresort_planner.ui.click_handlers import handle_slope_building_click

        dem = mock_dem_blue_slope  # drops going south
        graph = ResortGraph()
        sm, ctx = _session(fake_st, graph, path_factory, dem)
        sm.start_building(lon=0.0, lat=0.0, elevation=dem.get_elevation_or_raise(lon=0.0, lat=0.0))
        sm.enable_custom_connect()
        assert ctx.custom_connect.enabled

        # Click downhill terrain 400m south → valid custom target.
        handle_slope_building_click(
            ClickInfo(click_type=MapClickType.TERRAIN, lat=-400 / M, lon=0.0),
            elevation=dem.get_elevation_or_raise(lon=0.0, lat=-400 / M),
        )
        assert sm.is_slope_custom_path, "valid target transitions to custom-path state"

    def test_uphill_target_is_rejected(self, fake_st, path_factory, mock_dem_blue_slope) -> None:
        from skiresort_planner.ui.click_handlers import handle_slope_building_click

        dem = mock_dem_blue_slope
        graph = ResortGraph()
        sm, ctx = _session(fake_st, graph, path_factory, dem)
        # Start low so an uphill target is invalid.
        sm.start_building(lon=0.0, lat=-400 / M, elevation=dem.get_elevation_or_raise(lon=0.0, lat=-400 / M))
        sm.enable_custom_connect()

        handle_slope_building_click(
            ClickInfo(click_type=MapClickType.TERRAIN, lat=0.0, lon=0.0),  # uphill (summit)
            elevation=dem.get_elevation_or_raise(lon=0.0, lat=0.0),
        )
        assert not sm.is_slope_custom_path  # rejected → stayed in custom picking


# =============================================================================
# handle_slope_building_click — commit / select proposals, reject stray clicks
# =============================================================================


class TestSlopeBuildingClick:
    def _building(self, fake_st, dem, factory):
        graph = ResortGraph()
        sm, ctx = _session(fake_st, graph, factory, dem)
        sm.start_building(lon=0.0, lat=0.0, elevation=dem.get_elevation_or_raise(lon=0.0, lat=0.0))
        return sm, ctx, graph

    def test_proposal_body_click_selects_variant(
        self, fake_st, path_factory, mock_dem_red_slope_diagonal, path_points_blue
    ) -> None:
        from skiresort_planner.ui.click_handlers import handle_slope_building_click

        _sm, ctx, _graph = self._building(fake_st, mock_dem_red_slope_diagonal, path_factory)
        ctx.proposals.paths = [
            ProposedPathSegment(points=path_points_blue, target_difficulty="blue"),
            ProposedPathSegment(points=path_points_blue, target_difficulty="blue"),
        ]
        ctx.proposals.selected_idx = 0

        handle_slope_building_click(
            ClickInfo(click_type=MapClickType.MARKER, marker_type=MarkerType.PROPOSAL_BODY, proposal_index=1),
            elevation=None,
        )
        assert ctx.proposals.selected_idx == 1

    def test_proposal_endpoint_click_commits_path(
        self, fake_st, path_factory, mock_dem_red_slope_diagonal, path_points_blue
    ) -> None:
        from skiresort_planner.ui.click_handlers import handle_slope_building_click

        _sm, ctx, graph = self._building(fake_st, mock_dem_red_slope_diagonal, path_factory)
        # A non-connector proposal committed via its endpoint marker becomes a segment.
        ctx.proposals.paths = [ProposedPathSegment(points=path_points_blue, target_difficulty="blue")]
        ctx.proposals.selected_idx = 0

        handle_slope_building_click(
            ClickInfo(click_type=MapClickType.MARKER, marker_type=MarkerType.PROPOSAL_ENDPOINT, proposal_index=0),
            elevation=None,
        )
        assert len(graph.segments) == 1, "endpoint click on a non-connector commits the path"

    def test_node_click_without_custom_connect_is_rejected(
        self, fake_st, path_factory, mock_dem_red_slope_diagonal
    ) -> None:
        from skiresort_planner.ui.click_handlers import handle_slope_building_click

        sm, _ctx, graph = self._building(fake_st, mock_dem_red_slope_diagonal, path_factory)
        node, _ = graph.get_or_create_node(lon=0.0, lat=-0.001, elevation=1990.0)
        # A node click while building without custom-connect is a user error: no state change.
        handle_slope_building_click(
            ClickInfo(click_type=MapClickType.MARKER, marker_type=MarkerType.NODE, node_id=node.id), elevation=None
        )
        assert sm.is_any_slope_state

    def test_terrain_click_without_custom_connect_is_rejected(
        self, fake_st, path_factory, mock_dem_red_slope_diagonal
    ) -> None:
        from skiresort_planner.ui.click_handlers import handle_slope_building_click

        sm, _ctx, _graph = self._building(fake_st, mock_dem_red_slope_diagonal, path_factory)
        # A bare terrain click while building is a user error: no state change.
        handle_slope_building_click(ClickInfo(click_type=MapClickType.TERRAIN, lat=0.0, lon=0.0), elevation=1000.0)
        assert sm.is_slope_starting or sm.is_slope_building

    def test_lift_marker_click_while_building_is_rejected(
        self, fake_st, path_factory, mock_dem_red_slope_diagonal
    ) -> None:
        from skiresort_planner.ui.click_handlers import handle_slope_building_click

        sm, _ctx, _graph = self._building(fake_st, mock_dem_red_slope_diagonal, path_factory)
        handle_slope_building_click(
            ClickInfo(click_type=MapClickType.MARKER, marker_type=MarkerType.LIFT, lift_id="L1"), elevation=None
        )
        # Still in a slope state — the stray click did not navigate away.
        assert sm.is_any_slope_state


# =============================================================================
# handle_lift_placing_click — complete lift, reject stray clicks / bad geometry
# =============================================================================


class TestLiftPlacingClick:
    def _placing(self, fake_st, dem, factory):
        graph = ResortGraph()
        sm, ctx = _session(fake_st, graph, factory, dem)
        loc = PathPoint(lon=0.0, lat=-1000 / M, elevation=dem.get_elevation_or_raise(lon=0.0, lat=-1000 / M))
        sm.start_lift(node_id=None, location=loc)
        return sm, ctx, graph

    def test_terrain_uphill_end_completes_lift(self, fake_st, path_factory, mock_dem_blue_slope) -> None:
        from skiresort_planner.ui.click_handlers import handle_lift_placing_click

        dem = mock_dem_blue_slope  # drops going south → lat=0 is uphill of lat=-1000
        sm, _ctx, graph = self._placing(fake_st, dem, path_factory)

        handle_lift_placing_click(
            ClickInfo(click_type=MapClickType.TERRAIN, lat=0.0, lon=0.0),
            elevation=dem.get_elevation_or_raise(lon=0.0, lat=0.0),
        )
        assert len(graph.lifts) == 1
        assert sm.is_idle_viewing_lift

    def test_downhill_end_is_rejected(self, fake_st, path_factory, mock_dem_blue_slope) -> None:
        from skiresort_planner.ui.click_handlers import handle_lift_placing_click

        dem = mock_dem_blue_slope
        sm, _ctx, graph = self._placing(fake_st, dem, path_factory)

        # lat=-2000 is downhill of the lat=-1000 bottom → lift must go uphill → rejected.
        handle_lift_placing_click(
            ClickInfo(click_type=MapClickType.TERRAIN, lat=-2000 / M, lon=0.0),
            elevation=dem.get_elevation_or_raise(lon=0.0, lat=-2000 / M),
        )
        assert len(graph.lifts) == 0
        assert sm.is_lift_placing

    def test_slope_marker_click_while_placing_is_rejected(self, fake_st, path_factory, mock_dem_blue_slope) -> None:
        from skiresort_planner.ui.click_handlers import handle_lift_placing_click

        sm, _ctx, graph = self._placing(fake_st, mock_dem_blue_slope, path_factory)
        handle_lift_placing_click(
            ClickInfo(click_type=MapClickType.MARKER, marker_type=MarkerType.SLOPE, slope_id="SL1"), elevation=None
        )
        assert len(graph.lifts) == 0
        assert sm.is_lift_placing

    def test_node_end_completes_lift(self, fake_st, path_factory, mock_dem_blue_slope) -> None:
        from skiresort_planner.ui.click_handlers import handle_lift_placing_click

        dem = mock_dem_blue_slope
        sm, _ctx, graph = self._placing(fake_st, dem, path_factory)
        # Existing uphill node as the end station.
        top, _ = graph.get_or_create_node(lon=0.0, lat=0.0, elevation=dem.get_elevation_or_raise(lon=0.0, lat=0.0))

        handle_lift_placing_click(
            ClickInfo(click_type=MapClickType.MARKER, marker_type=MarkerType.NODE, node_id=top.id), elevation=None
        )
        assert len(graph.lifts) == 1
        assert sm.is_idle_viewing_lift


# =============================================================================
# handle_road_building_click — propose gentle (±15%) segments per click; extend/refuse
# =============================================================================


class TestRoadBuildingClick:
    def _building(self, fake_st, factory, dem):
        """Start a road at the origin so the handler is in road_starting."""
        sm, ctx = _session(fake_st, ResortGraph(), factory, dem)
        ctx.build_mode.mode = BuildMode.ROAD
        from skiresort_planner.ui.click_handlers import handle_idle_click

        handle_idle_click(
            ClickInfo(click_type=MapClickType.TERRAIN, lat=0.0, lon=0.0),
            elevation=dem.get_elevation_or_raise(lon=0.0, lat=0.0),
        )
        assert sm.is_road_starting
        return sm, ctx, fake_st.session_state["graph"]

    def _commit_proposal(self, proposal_index: int = 0) -> None:
        """Commit the selected road proposal via the button path (commit_selected_path).

        Roads commit exactly like slope custom-connect: NOT by clicking a proposal
        marker (those only select) but via the "✅ Commit Road Segment" button,
        which calls commit_selected_path.
        """
        from skiresort_planner.ui.actions import commit_selected_path

        commit_selected_path(path_idx=proposal_index)

    def test_terrain_click_generates_proposals_without_committing(
        self, fake_st, path_factory, mock_dem_red_slope_diagonal
    ) -> None:
        from skiresort_planner.ui.click_handlers import handle_road_building_click

        dem = mock_dem_red_slope_diagonal
        sm, ctx, graph = self._building(fake_st, path_factory, dem)
        version_before = fake_st.session_state["map_version"]

        # A target click proposes route(s) to browse — like slope custom-connect,
        # minus the fan-out. It commits NOTHING until a proposal is clicked.
        # (Left/right variants are traced, then deduped: on smooth terrain the two
        # collapse to one identical route, same shared dedup slopes use.)
        handle_road_building_click(
            ClickInfo(click_type=MapClickType.TERRAIN, lat=0.0, lon=300 / M),
            elevation=dem.get_elevation_or_raise(lon=300 / M, lat=0.0),
        )
        assert len(ctx.proposals.paths) >= 1, "a reachable target proposes at least one gentle route"
        assert all(p.max_slope_pct <= float(PathConfig.ROAD_MAX_GRADIENT_PCT) for p in ctx.proposals.paths)
        assert ctx.proposals.selected_idx == 0
        assert ctx.road_build.segments == [], "a target click proposes, it does not commit"
        assert sm.is_road_starting, "still starting until a proposal is committed"
        # The handler MUST bump the map version so the fragment reruns and redraws
        # WITH the new proposals — the deck was already built before this click
        # dispatched. Regression for invisible road proposals.
        assert fake_st.session_state["map_version"] > version_before, (
            "generating road proposals must bump map_version to force a redraw"
        )

    def test_proposal_marker_click_selects_but_does_not_commit(
        self, fake_st, path_factory, mock_dem_red_slope_diagonal
    ) -> None:
        """A PROPOSAL_ENDPOINT/BODY click only SELECTS (roads = custom-connect); commit is button-only."""
        from skiresort_planner.ui.click_handlers import handle_road_building_click

        dem = mock_dem_red_slope_diagonal
        sm, ctx, graph = self._building(fake_st, path_factory, dem)
        handle_road_building_click(
            ClickInfo(click_type=MapClickType.TERRAIN, lat=0.0, lon=300 / M),
            elevation=dem.get_elevation_or_raise(lon=300 / M, lat=0.0),
        )

        # Endpoint-marker click → select only, NO commit.
        handle_road_building_click(
            ClickInfo(click_type=MapClickType.MARKER, marker_type=MarkerType.PROPOSAL_ENDPOINT, proposal_index=0),
            elevation=None,
        )
        assert ctx.road_build.segments == [], "endpoint click must NOT commit a road segment"
        assert sm.is_road_starting, "still starting — no commit happened"
        assert graph.segments == {}, "no segment committed on a select-only marker click"
        # Body-marker click → also select only.
        handle_road_building_click(
            ClickInfo(click_type=MapClickType.MARKER, marker_type=MarkerType.PROPOSAL_BODY, proposal_index=0),
            elevation=None,
        )
        assert ctx.road_build.segments == [], "body click must NOT commit either"

    def test_proposal_commit_via_button_stays_building(
        self, fake_st, path_factory, mock_dem_red_slope_diagonal
    ) -> None:
        from skiresort_planner.ui.click_handlers import handle_road_building_click

        dem = mock_dem_red_slope_diagonal
        sm, ctx, graph = self._building(fake_st, path_factory, dem)

        handle_road_building_click(
            ClickInfo(click_type=MapClickType.TERRAIN, lat=0.0, lon=300 / M),
            elevation=dem.get_elevation_or_raise(lon=300 / M, lat=0.0),
        )
        self._commit_proposal()  # button path → commit_selected_path
        assert sm.is_road_building_only, "a committed segment keeps building (no auto-finish)"
        assert len(ctx.road_build.segments) == 1
        assert len(graph.roads) == 0, "no Road entity until Finish Road"
        # The committed segment's kind IS road — identity lives on the segment, not a UI list.
        assert graph.segments[ctx.road_build.segments[-1]].kind is SegmentKind.ROAD
        # Per-segment undo: the commit pushed an AddSegmentsAction.
        assert graph.undo_stack, "committing a road segment records an undo entry"
        assert graph.undo_stack[-1].action_type.name == "ADD_SEGMENTS"

    def test_second_click_extends_the_road(self, fake_st, path_factory, mock_dem_red_slope_diagonal) -> None:
        from skiresort_planner.ui.click_handlers import handle_road_building_click

        dem = mock_dem_red_slope_diagonal
        sm, ctx, _graph = self._building(fake_st, path_factory, dem)

        handle_road_building_click(
            ClickInfo(click_type=MapClickType.TERRAIN, lat=0.0, lon=300 / M),
            elevation=dem.get_elevation_or_raise(lon=300 / M, lat=0.0),
        )
        self._commit_proposal()
        handle_road_building_click(
            ClickInfo(click_type=MapClickType.TERRAIN, lat=0.0, lon=600 / M),
            elevation=dem.get_elevation_or_raise(lon=600 / M, lat=0.0),
        )
        self._commit_proposal()
        assert sm.is_road_building_only
        assert len(ctx.road_build.segments) == 2, "each committed proposal adds one segment"

    def test_too_steep_target_is_refused(self, fake_st, path_factory, mock_dem_black_slope) -> None:
        from skiresort_planner.ui.click_handlers import handle_road_building_click

        # 45% south DEM: a target straight downhill can't be reached within ±15%.
        dem = mock_dem_black_slope
        sm, ctx, graph = self._building(fake_st, path_factory, dem)

        handle_road_building_click(
            ClickInfo(click_type=MapClickType.TERRAIN, lat=-300 / M, lon=0.0),
            elevation=dem.get_elevation_or_raise(lon=0.0, lat=-300 / M),
        )
        assert ctx.proposals.paths == [], "steep target proposes nothing"
        assert ctx.road_build.segments == [], "steep target commits nothing"
        assert sm.is_road_starting, "stays in building flow, no segment added"
        assert len(graph.segments) == 0

    def test_stray_marker_click_is_rejected(self, fake_st, path_factory, mock_dem_red_slope_diagonal) -> None:
        from skiresort_planner.ui.click_handlers import handle_road_building_click

        sm, ctx, graph = self._building(fake_st, path_factory, mock_dem_red_slope_diagonal)
        handle_road_building_click(
            ClickInfo(click_type=MapClickType.MARKER, marker_type=MarkerType.SLOPE, slope_id="SL1"), elevation=None
        )
        assert ctx.road_build.segments == []
        assert len(graph.segments) == 0

    def test_node_target_extends_and_joins(self, fake_st, path_factory, mock_dem_red_slope_diagonal) -> None:
        from skiresort_planner.ui.click_handlers import handle_road_building_click

        dem = mock_dem_red_slope_diagonal
        sm, ctx, graph = self._building(fake_st, path_factory, dem)
        end, _ = graph.get_or_create_node(
            lon=300 / M, lat=0.0, elevation=dem.get_elevation_or_raise(lon=300 / M, lat=0.0)
        )

        handle_road_building_click(
            ClickInfo(click_type=MapClickType.MARKER, marker_type=MarkerType.NODE, node_id=end.id), elevation=None
        )
        assert len(ctx.proposals.paths) >= 1, "a node target also proposes a route to browse"
        assert all(p.target_node_id == end.id for p in ctx.proposals.paths), "proposals snap onto the target node"
        self._commit_proposal()
        assert sm.is_road_building_only
        assert len(ctx.road_build.segments) == 1


# =============================================================================
# dispatch_click — entry point: elevation lookup + routing to state handler
# =============================================================================


class TestDispatchClick:
    def test_terrain_click_routes_to_idle_handler(self, fake_st, path_factory, mock_dem_red_slope_diagonal) -> None:
        from skiresort_planner.ui.click_handlers import dispatch_click

        dem = mock_dem_red_slope_diagonal
        sm, ctx = _session(fake_st, ResortGraph(), path_factory, dem)
        ctx.build_mode.mode = BuildMode.SLOPE

        # dispatch_click looks up elevation for terrain clicks, then routes to the idle handler.
        dispatch_click(ClickInfo(click_type=MapClickType.TERRAIN, lat=0.0, lon=0.0))
        assert sm.is_slope_starting

    def test_marker_click_routes_without_elevation_lookup(
        self, fake_st, path_factory, mock_dem_red_slope_diagonal, path_points_blue
    ) -> None:
        from skiresort_planner.ui.click_handlers import dispatch_click

        graph = ResortGraph()
        graph.commit_paths(paths=[ProposedPathSegment(points=path_points_blue, target_difficulty="blue")])
        slope = graph.finish_slope(segment_ids=list(graph.segments.keys()))
        sm, ctx = _session(fake_st, graph, path_factory, mock_dem_red_slope_diagonal)

        # A marker click carries no lat/lon; dispatch must route it straight to the handler.
        dispatch_click(ClickInfo(click_type=MapClickType.MARKER, marker_type=MarkerType.SLOPE, slope_id=slope.id))
        assert sm.is_idle_viewing_slope and ctx.viewing.slope_id == slope.id

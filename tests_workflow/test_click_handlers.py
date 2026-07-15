"""Unit tests for the four map click handlers, driven via the fake `st` session.

The click handlers read st.session_state.{state_machine, context, graph,
path_factory, dem_service}; with the fake `st` installed we seed those and call
the handler directly, asserting the real routing/commit logic without a browser.

One class per handler, symmetric across slope / lift / road so every build
mode's click flow is exercised the same way:
    handle_idle_click            → TestIdleClickRouting
    handle_path_building_click  → TestSlopeBuildingClick
    handle_lift_placing_click    → TestLiftPlacingClick
    handle_path_building_click  → TestRoadBuildingClick
"""

import pytest

from skiresort_planner.constants import PathConfig, SlopeConfig
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
    fake_st.session_state["camera_epoch"] = 0
    fake_st.session_state["dedup_epoch"] = 0
    return sm, ctx


def _capture_toasts(monkeypatch) -> list[str]:
    """Capture the toast strings the handlers raise via ToastMessage.display().

    ToastMessage.display() does a function-local `import streamlit as st; st.toast(...)`, so it
    resolves the REAL streamlit module (not the fake `st` patched into the ui package). We patch
    `streamlit.toast` itself to record WHICH rejection/validation message fired — a reject branch
    that silently did the wrong thing (or navigated away) is exactly what these tests must catch.
    Returns the list; it fills as handlers run.
    """
    import streamlit

    toasts: list[str] = []
    monkeypatch.setattr(streamlit, "toast", lambda text, *a, **k: toasts.append(text))
    return toasts


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
        assert ctx.build(SegmentKind.ROAD).start_node_id == node.id

    def test_import_mode_terrain_click_places_box(self, fake_st, path_factory, mock_dem_red_slope_diagonal) -> None:
        from skiresort_planner.ui.click_handlers import handle_idle_click

        dem = mock_dem_red_slope_diagonal
        sm, ctx = _session(fake_st, ResortGraph(), path_factory, dem)
        ctx.build_mode.mode = BuildMode.IMPORT

        handle_idle_click(
            ClickInfo(click_type=MapClickType.TERRAIN, lat=0.01, lon=0.02),
            elevation=dem.get_elevation_or_raise(lon=0.02, lat=0.01),
        )
        assert sm.is_import_placing
        assert ctx.deferred.osm_import_center_lon == 0.02 and ctx.deferred.osm_import_center_lat == 0.01

    def test_import_mode_node_click_places_box_at_node(
        self, fake_st, path_factory, mock_dem_red_slope_diagonal
    ) -> None:
        from skiresort_planner.ui.click_handlers import handle_idle_click

        graph = ResortGraph()
        node, _ = graph.get_or_create_node(lon=0.03, lat=0.04, elevation=2000.0)
        sm, ctx = _session(fake_st, graph, path_factory, mock_dem_red_slope_diagonal)
        ctx.build_mode.mode = BuildMode.IMPORT

        handle_idle_click(
            ClickInfo(click_type=MapClickType.MARKER, marker_type=MarkerType.NODE, node_id=node.id), elevation=None
        )
        assert sm.is_import_placing
        assert ctx.deferred.osm_import_center_lon == 0.03 and ctx.deferred.osm_import_center_lat == 0.04

    def test_merge_mode_node_click_starts_merge_and_selects(
        self, fake_st, path_factory, mock_dem_red_slope_diagonal
    ) -> None:
        from skiresort_planner.ui.click_handlers import handle_idle_click

        graph = ResortGraph()
        node, _ = graph.get_or_create_node(lon=0.0, lat=0.0, elevation=2000.0)
        sm, ctx = _session(fake_st, graph, path_factory, mock_dem_red_slope_diagonal)
        ctx.build_mode.mode = BuildMode.MERGE

        handle_idle_click(
            ClickInfo(click_type=MapClickType.MARKER, marker_type=MarkerType.NODE, node_id=node.id), elevation=None
        )
        assert sm.is_merge_placing, "first node click starts merge (like other modes' first click)"
        assert ctx.merge.node_ids == [node.id], "the clicked node is the first merge selection"

    def test_merge_mode_terrain_click_is_invalid_and_stays_idle(
        self, fake_st, path_factory, mock_dem_red_slope_diagonal
    ) -> None:
        from skiresort_planner.ui.click_handlers import handle_idle_click

        dem = mock_dem_red_slope_diagonal
        sm, ctx = _session(fake_st, ResortGraph(), path_factory, dem)
        ctx.build_mode.mode = BuildMode.MERGE

        handle_idle_click(
            ClickInfo(click_type=MapClickType.TERRAIN, lat=0.01, lon=0.02),
            elevation=dem.get_elevation_or_raise(lon=0.02, lat=0.01),
        )
        assert sm.is_idle_ready, "terrain is not a merge target — stay idle"
        assert ctx.merge.node_ids == [], "no selection from a terrain click"

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
        assert slope is not None
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
        assert slope is not None
        seg_id = slope.segment_ids[0]
        sm, ctx = _session(fake_st, graph, path_factory, mock_dem_red_slope_diagonal)

        handle_idle_click(
            ClickInfo(click_type=MapClickType.MARKER, marker_type=MarkerType.SEGMENT, segment_id=seg_id), elevation=None
        )
        assert sm.is_idle_viewing_slope
        assert ctx.viewing.slope_id == slope.id

    def test_click_segment_opens_parent_road_panel(self, fake_st, path_factory, mock_dem_red_slope_diagonal) -> None:
        # Parity with the slope case: a SEGMENT marker whose parent is a ROAD (one-frame race before
        # the map re-tags it) must open the ROAD panel, not fall through to a slope panel. Resolved by
        # get_entity_by_segment_id + reload-safe `.kind` dispatch (never isinstance).
        from skiresort_planner.ui.click_handlers import handle_idle_click

        graph = ResortGraph()
        road = _commit_road(graph)
        assert road is not None
        seg_id = road.segment_ids[0]
        sm, ctx = _session(fake_st, graph, path_factory, mock_dem_red_slope_diagonal)

        handle_idle_click(
            ClickInfo(click_type=MapClickType.MARKER, marker_type=MarkerType.SEGMENT, segment_id=seg_id), elevation=None
        )
        assert sm.is_idle_viewing_road, "a road segment resolves to the ROAD panel (not slope)"
        assert ctx.viewing.road_id == road.id

    def test_click_segment_on_reloaded_slope_class_does_not_crash(
        self, fake_st, path_factory, mock_dem_red_slope_diagonal, path_points_blue
    ) -> None:
        """Regression for the `unhandled parent entity Slope` crash.

        A Streamlit module reload redefines both the Slope class AND the SegmentKind enum, so the old
        `isinstance(parent, Slope)` dispatch failed for a slope built under the pre-reload class and
        raised. The fix branches on the reload-safe `.kind` StrEnum (value-compared). Simulate the
        reload with a fresh SegmentKind class (same values) as the entity's kind; dispatch must still
        open the slope panel — identity differs, value matches.
        """
        from enum import StrEnum
        from typing import Any

        from skiresort_planner.ui.click_handlers import handle_idle_click

        graph = ResortGraph()
        graph.commit_paths(paths=[ProposedPathSegment(points=path_points_blue, target_difficulty="blue")])
        slope = graph.finish_slope(segment_ids=list(graph.segments.keys()))
        assert slope is not None
        seg_id = slope.segment_ids[0]

        # A distinct SegmentKind class (as after a reload): a different object, equal by value. Go
        # through Any so mypy doesn't fold slope.kind to its ClassVar literal (this is a runtime reload
        # simulation, not a statically-knowable value).
        reloaded_kind: Any = StrEnum("SegmentKind", {"SLOPE": "slope", "ROAD": "road"})  # type: ignore[misc]  # functional enum name mirrors the reloaded class
        reloaded_slope_kind: Any = reloaded_kind.SLOPE
        object.__setattr__(slope, "kind", reloaded_slope_kind)
        assert reloaded_slope_kind is not SegmentKind.SLOPE, "must be a different class instance (reload)"
        assert reloaded_slope_kind == SegmentKind.SLOPE, "StrEnum compares equal by value across the reload"

        sm, ctx = _session(fake_st, graph, path_factory, mock_dem_red_slope_diagonal)
        handle_idle_click(
            ClickInfo(click_type=MapClickType.MARKER, marker_type=MarkerType.SEGMENT, segment_id=seg_id), elevation=None
        )
        assert sm.is_idle_viewing_slope, "reload-safe .kind dispatch opens the slope panel without crashing"
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


class TestIdleClickEdgeCases:
    """Fail-fast + no-op branches of handle_idle_click that the routing tests don't reach.

    The idle handler is STRICT: a marker whose entity is missing from the graph is a
    map/graph-desync bug and must raise (never silently open a blank panel), while an orphan
    SEGMENT (parent slope already deleted) is an expected transient and must be a silent no-op.
    """

    def test_orphan_segment_click_is_silent_noop(
        self, fake_st, path_factory, mock_dem_red_slope_diagonal, path_points_blue
    ) -> None:
        # A SEGMENT marker can outlive its parent slope for one frame (slope deleted, map not yet
        # redrawn). Clicking it must NOT open a panel and must NOT crash — just ignore it.
        from skiresort_planner.ui.click_handlers import handle_idle_click

        graph = ResortGraph()
        graph.commit_paths(paths=[ProposedPathSegment(points=path_points_blue, target_difficulty="blue")])
        slope = graph.finish_slope(segment_ids=list(graph.segments.keys()))
        assert slope is not None
        orphan_seg_id = slope.segment_ids[0]
        assert graph.delete_slope(slope.id), "parent slope removed, segment id now orphaned"
        sm, ctx = _session(fake_st, graph, path_factory, mock_dem_red_slope_diagonal)

        handle_idle_click(
            ClickInfo(click_type=MapClickType.MARKER, marker_type=MarkerType.SEGMENT, segment_id=orphan_seg_id),
            elevation=None,
        )
        assert sm.is_idle_ready, "orphan segment click stays idle"
        assert ctx.viewing.slope_id is None, "no panel opened for an orphan segment"

    def test_missing_node_raises(self, fake_st, path_factory, mock_dem_red_slope_diagonal) -> None:
        from skiresort_planner.ui.click_handlers import handle_idle_click

        sm, _ctx = _session(fake_st, ResortGraph(), path_factory, mock_dem_red_slope_diagonal)
        with pytest.raises(RuntimeError, match="Node GHOST not found in graph"):
            handle_idle_click(
                ClickInfo(click_type=MapClickType.MARKER, marker_type=MarkerType.NODE, node_id="GHOST"), elevation=None
            )
        assert sm.is_idle_ready, "a graph-desync error must not have moved state before raising"

    def test_missing_slope_raises(self, fake_st, path_factory, mock_dem_red_slope_diagonal) -> None:
        from skiresort_planner.ui.click_handlers import handle_idle_click

        sm, _ctx = _session(fake_st, ResortGraph(), path_factory, mock_dem_red_slope_diagonal)
        with pytest.raises(RuntimeError, match="Slope GHOST not found in graph"):
            handle_idle_click(
                ClickInfo(click_type=MapClickType.MARKER, marker_type=MarkerType.SLOPE, slope_id="GHOST"),
                elevation=None,
            )
        assert sm.is_idle_ready, "a graph-desync error must not open a panel before raising"

    def test_missing_lift_raises_before_syncing_mode(self, fake_st, path_factory, mock_dem_red_slope_diagonal) -> None:
        from skiresort_planner.ui.click_handlers import handle_idle_click

        _sm, ctx = _session(fake_st, ResortGraph(), path_factory, mock_dem_red_slope_diagonal)
        mode_before = ctx.build_mode.mode
        with pytest.raises(RuntimeError, match="Lift GHOST not found in graph"):
            handle_idle_click(
                ClickInfo(click_type=MapClickType.MARKER, marker_type=MarkerType.LIFT, lift_id="GHOST"), elevation=None
            )
        assert ctx.build_mode.mode == mode_before, "must raise before syncing build_mode to a missing lift"

    def test_missing_pylon_parent_lift_raises(self, fake_st, path_factory, mock_dem_red_slope_diagonal) -> None:
        from skiresort_planner.ui.click_handlers import handle_idle_click

        _sm, _ctx = _session(fake_st, ResortGraph(), path_factory, mock_dem_red_slope_diagonal)
        with pytest.raises(RuntimeError, match="Lift GHOST not found in graph"):
            handle_idle_click(
                ClickInfo(click_type=MapClickType.MARKER, marker_type=MarkerType.PYLON, lift_id="GHOST", pylon_index=0),
                elevation=None,
            )

    def test_missing_road_raises(self, fake_st, path_factory, mock_dem_red_slope_diagonal) -> None:
        from skiresort_planner.ui.click_handlers import handle_idle_click

        sm, _ctx = _session(fake_st, ResortGraph(), path_factory, mock_dem_red_slope_diagonal)
        with pytest.raises(RuntimeError, match="Road GHOST not found in graph"):
            handle_idle_click(
                ClickInfo(click_type=MapClickType.MARKER, marker_type=MarkerType.ROAD, road_id="GHOST"), elevation=None
            )
        assert sm.is_idle_ready, "a graph-desync error must not open a panel before raising"

    def test_import_center_marker_in_idle_raises_unhandled(
        self, fake_st, path_factory, mock_dem_red_slope_diagonal
    ) -> None:
        # IMPORT_CENTER is a valid MarkerType but has no meaning in idle — it must hit the strict
        # catch-all so a newly-added marker type can never be silently swallowed here.
        from skiresort_planner.ui.click_handlers import handle_idle_click

        sm, _ctx = _session(fake_st, ResortGraph(), path_factory, mock_dem_red_slope_diagonal)
        with pytest.raises(RuntimeError, match="Unhandled marker type import_center"):
            handle_idle_click(
                ClickInfo(click_type=MapClickType.MARKER, marker_type=MarkerType.IMPORT_CENTER), elevation=None
            )
        assert sm.is_idle_ready

    def test_proposal_marker_in_idle_raises(self, fake_st, path_factory, mock_dem_red_slope_diagonal) -> None:
        # Proposal markers only exist while building; one in idle means a stale-overlay bug → raise.
        from skiresort_planner.ui.click_handlers import handle_idle_click

        _sm, _ctx = _session(fake_st, ResortGraph(), path_factory, mock_dem_red_slope_diagonal)
        with pytest.raises(RuntimeError, match=r"\[IDLE\] Proposal click detected"):
            handle_idle_click(
                ClickInfo(click_type=MapClickType.MARKER, marker_type=MarkerType.PROPOSAL_BODY, proposal_index=0),
                elevation=None,
            )

    def test_viewing_slope_then_lift_marker_switches_view_and_syncs_mode(
        self, fake_st, path_factory, mock_dem_blue_slope, path_points_blue
    ) -> None:
        # The viewing sub-states reuse handle_idle_click, so clicking a DIFFERENT entity while a
        # panel is open must switch the panel — and a lift click must re-sync build_mode to it.
        from skiresort_planner.ui.click_handlers import handle_idle_click

        dem = mock_dem_blue_slope
        graph = ResortGraph()
        graph.commit_paths(paths=[ProposedPathSegment(points=path_points_blue, target_difficulty="blue")])
        slope = graph.finish_slope(segment_ids=list(graph.segments.keys()))
        assert slope is not None
        bottom, _ = graph.get_or_create_node(
            lon=0.0, lat=-1000 / M, elevation=dem.get_elevation_or_raise(lon=0.0, lat=-1000 / M)
        )
        top, _ = graph.get_or_create_node(lon=0.0, lat=0.0, elevation=dem.get_elevation_or_raise(lon=0.0, lat=0.0))
        lift = graph.add_lift(start_node_id=bottom.id, end_node_id=top.id, lift_type="gondola", dem=dem)
        sm, ctx = _session(fake_st, graph, path_factory, dem)

        # Open the slope panel first, then click the lift marker while viewing it.
        sm.show_slope_info_panel(slope_id=slope.id)
        assert sm.is_idle_viewing_slope
        handle_idle_click(
            ClickInfo(click_type=MapClickType.MARKER, marker_type=MarkerType.LIFT, lift_id=lift.id), elevation=None
        )
        assert sm.is_idle_viewing_lift, "clicking a lift while viewing a slope switches the panel"
        assert ctx.viewing.lift_id == lift.id
        assert ctx.build_mode.mode == "gondola", "the switch re-syncs build_mode to the newly viewed lift"


class TestCustomConnectClick:
    """A terrain/node click while building auto-routes a custom-connect path (no button)."""

    def test_downhill_terrain_target_transitions_to_custom_path(
        self, fake_st, path_factory, mock_dem_blue_slope
    ) -> None:
        from skiresort_planner.ui.click_handlers import handle_path_building_click

        dem = mock_dem_blue_slope  # drops going south
        graph = ResortGraph()
        sm, ctx = _session(fake_st, graph, path_factory, dem)
        sm.start_building(lon=0.0, lat=0.0, elevation=dem.get_elevation_or_raise(lon=0.0, lat=0.0))

        # Click downhill terrain 400m south → valid custom target, auto-enters custom path.
        handle_path_building_click(
            ClickInfo(click_type=MapClickType.TERRAIN, lat=-400 / M, lon=0.0),
            elevation=dem.get_elevation_or_raise(lon=0.0, lat=-400 / M),
        )
        assert sm.is_slope_custom_path, "valid target auto-transitions to custom-path state"
        assert ctx.custom_connect.force_mode, "force_mode is set when showing custom proposals"

    def test_uphill_target_is_rejected(self, fake_st, path_factory, mock_dem_blue_slope) -> None:
        from skiresort_planner.ui.click_handlers import handle_path_building_click

        dem = mock_dem_blue_slope
        graph = ResortGraph()
        sm, ctx = _session(fake_st, graph, path_factory, dem)
        # Start low so an uphill target is invalid.
        sm.start_building(lon=0.0, lat=-400 / M, elevation=dem.get_elevation_or_raise(lon=0.0, lat=-400 / M))

        handle_path_building_click(
            ClickInfo(click_type=MapClickType.TERRAIN, lat=0.0, lon=0.0),  # uphill (summit)
            elevation=dem.get_elevation_or_raise(lon=0.0, lat=0.0),
        )
        assert not sm.is_slope_custom_path, "invalid (uphill) target does NOT enter custom mode"
        assert not ctx.custom_connect.force_mode


# =============================================================================
# handle_path_building_click — commit / select proposals, reject stray clicks
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
        from skiresort_planner.ui.click_handlers import handle_path_building_click

        _sm, ctx, _graph = self._building(fake_st, mock_dem_red_slope_diagonal, path_factory)
        ctx.proposals.paths = [
            ProposedPathSegment(points=path_points_blue, target_difficulty="blue"),
            ProposedPathSegment(points=path_points_blue, target_difficulty="blue"),
        ]
        ctx.proposals.selected_idx = 0

        handle_path_building_click(
            ClickInfo(click_type=MapClickType.MARKER, marker_type=MarkerType.PROPOSAL_BODY, proposal_index=1),
            elevation=None,
        )
        assert ctx.proposals.selected_idx == 1

    def test_body_click_on_already_selected_slope_proposal_commits(
        self, fake_st, path_factory, mock_dem_red_slope_diagonal, path_points_blue
    ) -> None:
        """Re-clicking the already-selected slope proposal body commits it (panel-free)."""
        from skiresort_planner.ui.click_handlers import handle_path_building_click

        _sm, ctx, graph = self._building(fake_st, mock_dem_red_slope_diagonal, path_factory)
        ctx.proposals.paths = [ProposedPathSegment(points=path_points_blue, target_difficulty="blue")]
        ctx.proposals.selected_idx = 0  # already selected

        handle_path_building_click(
            ClickInfo(click_type=MapClickType.MARKER, marker_type=MarkerType.PROPOSAL_BODY, proposal_index=0),
            elevation=None,
        )
        assert len(graph.segments) == 1, "body click on the selected proposal commits it"

    def test_proposal_endpoint_click_commits_path(
        self, fake_st, path_factory, mock_dem_red_slope_diagonal, path_points_blue
    ) -> None:
        from skiresort_planner.ui.click_handlers import handle_path_building_click

        _sm, ctx, graph = self._building(fake_st, mock_dem_red_slope_diagonal, path_factory)
        # A non-connector proposal committed via its endpoint marker becomes a segment.
        ctx.proposals.paths = [ProposedPathSegment(points=path_points_blue, target_difficulty="blue")]
        ctx.proposals.selected_idx = 0

        handle_path_building_click(
            ClickInfo(click_type=MapClickType.MARKER, marker_type=MarkerType.PROPOSAL_ENDPOINT, proposal_index=0),
            elevation=None,
        )
        assert len(graph.segments) == 1, "endpoint click on a non-connector commits the path"

    def test_node_click_routes_custom_connect(self, fake_st, path_factory, mock_dem_red_slope_diagonal) -> None:
        from skiresort_planner.ui.click_handlers import handle_path_building_click

        sm, _ctx, graph = self._building(fake_st, mock_dem_red_slope_diagonal, path_factory)
        # A downhill node south of the origin → clicking it auto-routes a connector.
        node, _ = graph.get_or_create_node(lon=0.0, lat=-0.001, elevation=1990.0)
        handle_path_building_click(
            ClickInfo(click_type=MapClickType.MARKER, marker_type=MarkerType.NODE, node_id=node.id), elevation=None
        )
        assert sm.is_slope_custom_path, "node click auto-enters custom path"

    def test_terrain_click_routes_custom_connect(self, fake_st, path_factory, mock_dem_red_slope_diagonal) -> None:
        from skiresort_planner.ui.click_handlers import handle_path_building_click

        sm, _ctx, _graph = self._building(fake_st, mock_dem_red_slope_diagonal, path_factory)
        dem = mock_dem_red_slope_diagonal
        # A downhill terrain point (south of origin) auto-routes a custom-connect path.
        handle_path_building_click(
            ClickInfo(click_type=MapClickType.TERRAIN, lat=-300 / M, lon=0.0),
            elevation=dem.get_elevation_or_raise(lon=0.0, lat=-300 / M),
        )
        assert sm.is_slope_custom_path, "terrain click auto-enters custom path"

    def test_lift_marker_click_while_building_is_rejected(
        self, fake_st, path_factory, mock_dem_red_slope_diagonal
    ) -> None:
        from skiresort_planner.ui.click_handlers import handle_path_building_click

        sm, _ctx, _graph = self._building(fake_st, mock_dem_red_slope_diagonal, path_factory)
        handle_path_building_click(
            ClickInfo(click_type=MapClickType.MARKER, marker_type=MarkerType.LIFT, lift_id="L1"), elevation=None
        )
        # Still in a slope state — the stray click did not navigate away.
        assert sm.is_any_slope_state


class TestSlopeBuildingEdgeCases:
    """Guard, reject, and re-target branches of handle_path_building_click.

    Covers the connector-endpoint no-op, out-of-range proposal indices, the too-far distance guard
    reached THROUGH the handler (not just the isolated validator), the custom-connect re-target that
    must reuse the original origin, and the strict catch-all for an unhandled marker type.
    """

    def _building(self, fake_st, dem, factory):
        graph = ResortGraph()
        sm, ctx = _session(fake_st, graph, factory, dem)
        sm.start_building(lon=0.0, lat=0.0, elevation=dem.get_elevation_or_raise(lon=0.0, lat=0.0))
        return sm, ctx, graph

    def test_connector_proposal_click_commits_and_auto_finishes(
        self, fake_st, path_factory, mock_dem_red_slope_diagonal
    ) -> None:
        # Drive the real connector flow: click a downhill NODE target → custom-path →
        # generate proposals → commit the (connector) proposal → it auto-finishes the slope.
        from skiresort_planner.ui.actions import commit_selected_path, process_custom_connect_deferred
        from skiresort_planner.ui.click_handlers import handle_path_building_click

        sm, ctx, graph = self._building(fake_st, mock_dem_red_slope_diagonal, path_factory)
        node, _ = graph.get_or_create_node(lon=0.0, lat=-300 / M, elevation=2400.0)
        handle_path_building_click(
            ClickInfo(click_type=MapClickType.MARKER, marker_type=MarkerType.NODE, node_id=node.id), elevation=None
        )
        assert sm.is_slope_custom_path, "node click auto-enters custom path"
        process_custom_connect_deferred()  # generate the connector proposal(s)
        assert ctx.proposals.paths, "a downhill node target yields a connector proposal"
        assert all(p.is_connector for p in ctx.proposals.paths), "targeting a node makes the proposal a connector"

        commit_selected_path(path_idx=0)
        assert len(graph.segments) == 1, "the connector proposal commits"
        assert sm.is_idle_viewing_slope, "a committed connector auto-finishes the slope"

    def test_straight_line_appended_on_top_of_planner_proposals(
        self, fake_st, path_factory, mock_dem_red_slope_diagonal
    ) -> None:
        """A SLOPE custom-connect offers the straight line on TOP of the planner routes (last,
        not pre-selected), when both the planner routes and the straight line are within cap.
        Slopes previously had no straight-line option at all — this is the new capability.
        """
        from skiresort_planner.ui.actions import process_custom_connect_deferred
        from skiresort_planner.ui.click_handlers import handle_path_building_click

        dem = mock_dem_red_slope_diagonal
        sm, ctx, _graph = self._building(fake_st, dem, path_factory)
        # A gentle downhill target the planner can reach AND a straight line can reach in-cap.
        handle_path_building_click(
            ClickInfo(click_type=MapClickType.TERRAIN, lat=-300 / M, lon=0.0),
            elevation=dem.get_elevation_or_raise(lon=0.0, lat=-300 / M),
        )
        process_custom_connect_deferred()

        assert sm.is_slope_custom_path
        assert len(ctx.proposals.paths) >= 2, "planner route(s) PLUS the straight line"
        assert "straight line" in ctx.proposals.paths[-1].sector_name.lower(), "straight line is appended LAST"
        assert not any("straight line" in p.sector_name.lower() for p in ctx.proposals.paths[:-1]), (
            "only the last proposal is the straight line"
        )
        assert ctx.proposals.selected_idx == 0, "the planner's gentlest stays pre-selected, not the straight line"

    def test_straight_line_omitted_when_over_cap_but_planner_succeeds(
        self, fake_st, path_factory, mock_dem_red_slope_diagonal
    ) -> None:
        """If the straight line is over cap but planner routes fit, only planner routes show."""
        from skiresort_planner.ui.actions import process_custom_connect_deferred
        from skiresort_planner.ui.click_handlers import handle_path_building_click

        dem = mock_dem_red_slope_diagonal
        sm, ctx, graph = self._building(fake_st, dem, path_factory)
        # A reachable downhill node target so we enter custom-path. Force the planner to return one
        # in-cap route and the straight line to be over the slope cap → only the planner route shows.
        node, _ = graph.get_or_create_node(lon=0.0, lat=-300 / M, elevation=2400.0)
        in_cap = ProposedPathSegment(
            points=[PathPoint(lon=0.0, lat=0.0, elevation=2500.0), PathPoint(lon=0.0, lat=-300 / M, elevation=2400.0)],
            target_difficulty="blue",
        )
        over_cap_straight = ProposedPathSegment(
            points=[PathPoint(lon=0.0, lat=0.0, elevation=2500.0), PathPoint(lon=0.0, lat=-1 / M, elevation=2400.0)],
            is_connector=True,
        )
        assert over_cap_straight.max_slope_pct > float(SlopeConfig.MAX_SKIABLE_PCT), "straight line must be over cap"
        mp = pytest.MonkeyPatch()
        mp.setattr(path_factory, "generate_manual_paths", lambda **kwargs: iter([in_cap]))
        mp.setattr(path_factory, "straight_line", lambda **kwargs: over_cap_straight)
        handle_path_building_click(
            ClickInfo(click_type=MapClickType.MARKER, marker_type=MarkerType.NODE, node_id=node.id), elevation=None
        )
        process_custom_connect_deferred()
        mp.undo()

        assert sm.is_slope_custom_path
        assert len(ctx.proposals.paths) == 1, "only the in-cap planner route (straight line over cap dropped)"
        assert "straight line" not in ctx.proposals.paths[0].sector_name.lower()

    def test_proposal_body_index_out_of_range_is_noop(
        self, fake_st, path_factory, mock_dem_red_slope_diagonal, path_points_blue
    ) -> None:
        # A proposal_index past the end (stale overlay clicked after proposals shrank) must be a
        # silent no-op through the shared _select_or_commit_proposal bounds guard, never an IndexError.
        from skiresort_planner.ui.click_handlers import handle_path_building_click

        _sm, ctx, graph = self._building(fake_st, mock_dem_red_slope_diagonal, path_factory)
        ctx.proposals.paths = [ProposedPathSegment(points=path_points_blue, target_difficulty="blue")]
        ctx.proposals.selected_idx = 0

        handle_path_building_click(
            ClickInfo(click_type=MapClickType.MARKER, marker_type=MarkerType.PROPOSAL_BODY, proposal_index=99),
            elevation=None,
        )
        assert len(graph.segments) == 0, "out-of-range proposal commits nothing"
        assert ctx.proposals.selected_idx == 0, "out-of-range proposal changes no selection"

    def test_proposal_endpoint_index_out_of_range_is_noop(
        self, fake_st, path_factory, mock_dem_red_slope_diagonal, path_points_blue
    ) -> None:
        # The ENDPOINT path has its OWN inline bounds guard (separate from the body path); an
        # out-of-range endpoint click must also be a silent no-op, not an IndexError.
        from skiresort_planner.ui.click_handlers import handle_path_building_click

        _sm, ctx, graph = self._building(fake_st, mock_dem_red_slope_diagonal, path_factory)
        ctx.proposals.paths = [ProposedPathSegment(points=path_points_blue, target_difficulty="blue")]
        ctx.proposals.selected_idx = 0

        handle_path_building_click(
            ClickInfo(click_type=MapClickType.MARKER, marker_type=MarkerType.PROPOSAL_ENDPOINT, proposal_index=99),
            elevation=None,
        )
        assert len(graph.segments) == 0, "out-of-range endpoint commits nothing"
        assert ctx.proposals.selected_idx == 0, "out-of-range endpoint changes no selection"

    def test_terrain_target_too_far_is_refused_via_handler(self, fake_st, path_factory, mock_dem_blue_slope) -> None:
        # The distance cap has an isolated validator test; this drives it THROUGH the handler to
        # prove a >1000 m downhill target short-circuits before any transition or proposal.
        from skiresort_planner.ui.click_handlers import handle_path_building_click

        dem = mock_dem_blue_slope  # drops going south, so a far-south target is downhill AND too far
        sm, ctx, _graph = self._building(fake_st, dem, path_factory)
        far_lat = -(PathConfig.SEGMENT_LENGTH_MAX_M + 500) / M  # 1500 m south of the origin

        handle_path_building_click(
            ClickInfo(click_type=MapClickType.TERRAIN, lat=far_lat, lon=0.0),
            elevation=dem.get_elevation_or_raise(lon=0.0, lat=far_lat),
        )
        assert not sm.is_slope_custom_path, "a too-far target does not enter custom path"
        assert ctx.proposals.paths == [], "a too-far target proposes nothing"

    def test_custom_path_retarget_reuses_original_origin(self, fake_st, path_factory, mock_dem_blue_slope) -> None:
        # Once in slope_custom_path, a second downhill click must RE-target while keeping the
        # original start node (custom_connect.start_node), not drift to the new click.
        from skiresort_planner.ui.click_handlers import handle_path_building_click

        dem = mock_dem_blue_slope
        sm, ctx, _graph = self._building(fake_st, dem, path_factory)
        # First downhill click enters custom-path and records the origin.
        handle_path_building_click(
            ClickInfo(click_type=MapClickType.TERRAIN, lat=-300 / M, lon=0.0),
            elevation=dem.get_elevation_or_raise(lon=0.0, lat=-300 / M),
        )
        assert sm.is_slope_custom_path
        origin_before = ctx.custom_connect.start_node
        assert origin_before is None, "a fresh terrain origin is a pending location, not a node id"
        loc_before = ctx.build(SegmentKind.SLOPE).start_location
        assert loc_before is not None, "the routing origin is carried as a location"

        # Second downhill click (further south) re-targets; the origin must be preserved.
        handle_path_building_click(
            ClickInfo(click_type=MapClickType.TERRAIN, lat=-600 / M, lon=0.0),
            elevation=dem.get_elevation_or_raise(lon=0.0, lat=-600 / M),
        )
        assert sm.is_slope_custom_path, "re-target stays in custom path"
        assert ctx.custom_connect.start_node == origin_before, "re-target keeps the original origin (still None)"
        assert ctx.build(SegmentKind.SLOPE).start_location == loc_before, "re-target keeps the origin location"

    def test_import_center_marker_while_building_raises_unhandled(
        self, fake_st, path_factory, mock_dem_red_slope_diagonal
    ) -> None:
        # The building handler is strict: an unmapped marker type hits the catch-all so a new
        # MarkerType can never be silently ignored mid-build.
        from skiresort_planner.ui.click_handlers import handle_path_building_click

        _sm, _ctx, _graph = self._building(fake_st, mock_dem_red_slope_diagonal, path_factory)
        with pytest.raises(RuntimeError, match="Unhandled marker type import_center"):
            handle_path_building_click(
                ClickInfo(click_type=MapClickType.MARKER, marker_type=MarkerType.IMPORT_CENTER), elevation=None
            )


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


class TestLiftPlacingEdgeCases:
    """Reject, validation, and node-reuse branches of handle_lift_placing_click.

    A lift needs two DISTINCT stations and must go uphill; a start from an existing node must be
    reused (never duplicated) on completion, and stray entity markers must be politely rejected
    (correct verb) without touching the in-progress placement.
    """

    def _placing_from_node(self, fake_st, dem, factory):
        """Enter lift_placing with the bottom station being an EXISTING node (start_node_id set)."""
        graph = ResortGraph()
        sm, ctx = _session(fake_st, graph, factory, dem)
        bottom, _ = graph.get_or_create_node(
            lon=0.0, lat=-1000 / M, elevation=dem.get_elevation_or_raise(lon=0.0, lat=-1000 / M)
        )
        sm.start_lift(node_id=bottom.id, location=None)
        return sm, ctx, graph, bottom

    def test_same_node_start_and_end_is_rejected(self, fake_st, monkeypatch, path_factory, mock_dem_blue_slope) -> None:
        # Clicking the SAME node as both stations is refused. NOTE: the handler's uphill check runs
        # BEFORE its same-node check, and one node has equal start==end elevation, so the refusal
        # that actually fires is "Lift Must Go Uphill" (the dedicated same-node guard is unreachable
        # via this path — asserting SameNodeLift here would be a lie). Either way: no lift, still placing.
        from skiresort_planner.ui.click_handlers import handle_lift_placing_click

        dem = mock_dem_blue_slope
        sm, _ctx, graph, bottom = self._placing_from_node(fake_st, dem, path_factory)
        toasts = _capture_toasts(monkeypatch)

        handle_lift_placing_click(
            ClickInfo(click_type=MapClickType.MARKER, marker_type=MarkerType.NODE, node_id=bottom.id), elevation=None
        )
        assert len(graph.lifts) == 0, "a same-node lift is not built"
        assert sm.is_lift_placing, "rejection keeps us placing"
        assert any("Uphill" in t for t in toasts), "a zero-rise (same-node) lift is refused as not uphill"

    def test_node_end_from_existing_node_start_reuses_both_nodes(
        self, fake_st, path_factory, mock_dem_blue_slope
    ) -> None:
        # Both stations are pre-existing nodes: completing the lift must reuse them (no new nodes),
        # unlike the terrain path which creates nodes.
        from skiresort_planner.ui.click_handlers import handle_lift_placing_click

        dem = mock_dem_blue_slope
        sm, ctx, graph, bottom = self._placing_from_node(fake_st, dem, path_factory)
        top, _ = graph.get_or_create_node(lon=0.0, lat=0.0, elevation=dem.get_elevation_or_raise(lon=0.0, lat=0.0))
        nodes_before = len(graph.nodes)

        handle_lift_placing_click(
            ClickInfo(click_type=MapClickType.MARKER, marker_type=MarkerType.NODE, node_id=top.id), elevation=None
        )
        assert len(graph.lifts) == 1
        assert len(graph.nodes) == nodes_before, "both stations already existed — no node is duplicated"
        lift = next(iter(graph.lifts.values()))
        assert {lift.start_node_id, lift.end_node_id} == {bottom.id, top.id}, "lift joins the two existing nodes"
        assert sm.is_idle_viewing_lift and ctx.viewing.lift_id == lift.id

    def test_terrain_end_from_pending_start_location_materialises_start_node(
        self, fake_st, path_factory, mock_dem_blue_slope
    ) -> None:
        # Bottom is a PENDING location (start_node_id None): completing via terrain must create BOTH
        # nodes and clear start_location — the "fresh point" origin path, distinct from node-reuse.
        from skiresort_planner.ui.click_handlers import handle_lift_placing_click

        dem = mock_dem_blue_slope
        graph = ResortGraph()
        sm, ctx = _session(fake_st, graph, path_factory, dem)
        loc = PathPoint(lon=0.0, lat=-1000 / M, elevation=dem.get_elevation_or_raise(lon=0.0, lat=-1000 / M))
        sm.start_lift(node_id=None, location=loc)  # pending-location start: no start node materialised yet

        handle_lift_placing_click(
            ClickInfo(click_type=MapClickType.TERRAIN, lat=0.0, lon=0.0),
            elevation=dem.get_elevation_or_raise(lon=0.0, lat=0.0),
        )
        assert len(graph.lifts) == 1
        assert len(graph.nodes) == 2, "a pending-location start + terrain end create exactly two nodes"
        # The bottom station was materialised from the pending location: the built lift references a
        # real graph node for its start (ctx.lift itself is reset on the transition to viewing).
        lift = next(iter(graph.lifts.values()))
        assert lift.start_node_id in graph.nodes, "the pending start location became a real graph node"
        assert ctx.lift.start_location is None, "the pending start location is consumed, not left dangling"
        assert sm.is_idle_viewing_lift

    def test_road_marker_click_while_placing_is_rejected_with_view_road_verb(
        self, fake_st, monkeypatch, path_factory, mock_dem_blue_slope
    ) -> None:
        # A ROAD marker mid-lift is rejected with the ROAD-specific verb (regression for the
        # marker-dispatch-completeness bug where a ROAD click in lift mode used to crash).
        from skiresort_planner.ui.click_handlers import handle_lift_placing_click

        dem = mock_dem_blue_slope
        sm, _ctx, graph, _bottom = self._placing_from_node(fake_st, dem, path_factory)
        toasts = _capture_toasts(monkeypatch)

        handle_lift_placing_click(
            ClickInfo(click_type=MapClickType.MARKER, marker_type=MarkerType.ROAD, road_id="R1"), elevation=None
        )
        assert len(graph.lifts) == 0
        assert sm.is_lift_placing, "a stray road marker does not abandon the lift"
        assert any("view road" in t for t in toasts), "rejection names the road-specific action"

    def test_stale_end_node_id_raises(self, fake_st, path_factory, mock_dem_blue_slope) -> None:
        # Clicking a NODE marker whose id is not in the graph is a map/graph desync → fail-fast.
        from skiresort_planner.ui.click_handlers import handle_lift_placing_click

        dem = mock_dem_blue_slope
        _sm, _ctx, _graph, _bottom = self._placing_from_node(fake_st, dem, path_factory)
        with pytest.raises(RuntimeError, match="End node GHOST must exist but was not found"):
            handle_lift_placing_click(
                ClickInfo(click_type=MapClickType.MARKER, marker_type=MarkerType.NODE, node_id="GHOST"), elevation=None
            )


# =============================================================================
# handle_path_building_click — propose gentle (±15%) segments per click; extend/refuse
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

    def _target(self, click_info: ClickInfo, elevation: float | None = None) -> None:
        """Click a road target, then run the deferred custom-connect generation.

        A target click now routes into ROAD_CUSTOM_PATH and ARMS deferred generation
        (mirrors slope custom-connect); the proposals appear when the deferred pass runs.
        This helper does both so tests can assert on the resulting proposals.
        """
        from skiresort_planner.ui.actions import process_custom_connect_deferred
        from skiresort_planner.ui.click_handlers import handle_path_building_click

        handle_path_building_click(click_info, elevation=elevation)
        process_custom_connect_deferred()

    def test_terrain_click_generates_proposals_without_committing(
        self, fake_st, path_factory, mock_dem_red_slope_diagonal
    ) -> None:
        dem = mock_dem_red_slope_diagonal
        sm, ctx, graph = self._building(fake_st, path_factory, dem)
        dedup_before = fake_st.session_state["dedup_epoch"]
        camera_before = fake_st.session_state["camera_epoch"]

        # A target click enters ROAD_CUSTOM_PATH and arms deferred generation; the
        # deferred pass produces the proposal(s) to browse. It commits NOTHING until a
        # proposal is clicked/committed.
        self._target(
            ClickInfo(click_type=MapClickType.TERRAIN, lat=0.0, lon=300 / M),
            elevation=dem.get_elevation_or_raise(lon=300 / M, lat=0.0),
        )
        assert len(ctx.proposals.paths) >= 1, "a reachable target proposes at least one gentle route"
        assert all(p.max_slope_pct <= float(PathConfig.ROAD_MAX_GRADIENT_PCT) for p in ctx.proposals.paths)
        assert ctx.proposals.selected_idx == 0
        assert ctx.build(SegmentKind.ROAD).segments == [], "a target click proposes, it does not commit"
        assert sm.is_road_custom_path, "still targeting until a proposal is committed"
        # The deferred pass bumps dedup_epoch so the new proposals are clickable — but must NOT
        # recenter (camera_epoch unchanged). Regression for invisible road proposals + no-jump.
        assert fake_st.session_state["dedup_epoch"] > dedup_before, (
            "generating road proposals must bump dedup_epoch so they are clickable"
        )
        assert fake_st.session_state["camera_epoch"] == camera_before, "generating proposals must NOT recenter the map"

    def test_clicking_already_selected_proposal_commits(
        self, fake_st, path_factory, mock_dem_red_slope_diagonal
    ) -> None:
        """Clicking the ALREADY-selected proposal commits it (panel button not needed).

        Road generates a single proposal auto-selected at idx 0, so the first
        marker click lands on the selected one → commit. Mirrors the user rule:
        "IF proposal IS selected AND is clicked → commit."
        """
        from skiresort_planner.ui.click_handlers import handle_path_building_click

        dem = mock_dem_red_slope_diagonal
        sm, ctx, _graph = self._building(fake_st, path_factory, dem)
        self._target(
            ClickInfo(click_type=MapClickType.TERRAIN, lat=0.0, lon=300 / M),
            elevation=dem.get_elevation_or_raise(lon=300 / M, lat=0.0),
        )
        assert ctx.proposals.selected_idx == 0, "the sole proposal is auto-selected"

        # Click the already-selected proposal body → commits (no panel button).
        handle_path_building_click(
            ClickInfo(click_type=MapClickType.MARKER, marker_type=MarkerType.PROPOSAL_BODY, proposal_index=0),
            elevation=None,
        )
        assert len(ctx.build(SegmentKind.ROAD).segments) == 1, "clicking the selected proposal commits it"
        assert sm.is_road_building_only, "committed segment keeps building"

    def test_clicking_unselected_proposal_only_selects(
        self, fake_st, path_factory, mock_dem_red_slope_diagonal
    ) -> None:
        """Clicking a NOT-yet-selected proposal only highlights it — no commit.

        Force a two-proposal state and select idx 1, then click idx 0: it must
        select (not commit), because idx 0 was not the selected one.
        """
        from skiresort_planner.ui.click_handlers import handle_path_building_click

        dem = mock_dem_red_slope_diagonal
        sm, ctx, _graph = self._building(fake_st, path_factory, dem)
        self._target(
            ClickInfo(click_type=MapClickType.TERRAIN, lat=0.0, lon=300 / M),
            elevation=dem.get_elevation_or_raise(lon=300 / M, lat=0.0),
        )
        # Simulate a multi-proposal browse state with a different one selected.
        ctx.proposals.paths = ctx.proposals.paths + ctx.proposals.paths[:1]  # 2 entries
        ctx.proposals.selected_idx = 1

        handle_path_building_click(
            ClickInfo(click_type=MapClickType.MARKER, marker_type=MarkerType.PROPOSAL_BODY, proposal_index=0),
            elevation=None,
        )
        assert ctx.proposals.selected_idx == 0, "clicking an unselected proposal selects it"
        assert ctx.build(SegmentKind.ROAD).segments == [], "selecting an unselected proposal must NOT commit"
        assert sm.is_road_custom_path

    def test_proposal_commit_via_button_stays_building(
        self, fake_st, path_factory, mock_dem_red_slope_diagonal
    ) -> None:
        dem = mock_dem_red_slope_diagonal
        sm, ctx, graph = self._building(fake_st, path_factory, dem)

        self._target(
            ClickInfo(click_type=MapClickType.TERRAIN, lat=0.0, lon=300 / M),
            elevation=dem.get_elevation_or_raise(lon=300 / M, lat=0.0),
        )
        self._commit_proposal()  # button path → commit_selected_path
        assert sm.is_road_building_only, "a committed segment keeps building (no auto-finish)"
        assert len(ctx.build(SegmentKind.ROAD).segments) == 1
        assert len(graph.roads) == 0, "no Road entity until Finish Road"
        # The committed segment's kind IS road — identity lives on the segment, not a UI list.
        assert graph.segments[ctx.build(SegmentKind.ROAD).segments[-1]].kind == SegmentKind.ROAD
        # Per-segment undo: the commit pushed an AddSegmentsAction.
        assert graph.undo_stack, "committing a road segment records an undo entry"
        assert graph.undo_stack[-1].action_type.name == "ADD_SEGMENTS"

    def test_second_click_extends_the_road(self, fake_st, path_factory, mock_dem_red_slope_diagonal) -> None:
        dem = mock_dem_red_slope_diagonal
        sm, ctx, _graph = self._building(fake_st, path_factory, dem)

        self._target(
            ClickInfo(click_type=MapClickType.TERRAIN, lat=0.0, lon=300 / M),
            elevation=dem.get_elevation_or_raise(lon=300 / M, lat=0.0),
        )
        self._commit_proposal()
        self._target(
            ClickInfo(click_type=MapClickType.TERRAIN, lat=0.0, lon=600 / M),
            elevation=dem.get_elevation_or_raise(lon=600 / M, lat=0.0),
        )
        self._commit_proposal()
        assert sm.is_road_building_only
        assert len(ctx.build(SegmentKind.ROAD).segments) == 2, "each committed proposal adds one segment"

    def test_too_steep_target_is_refused(self, fake_st, monkeypatch, mock_dem_black_slope) -> None:
        from skiresort_planner.core.path_tracer import PathTracer
        from skiresort_planner.core.terrain_analyzer import TerrainAnalyzer
        from skiresort_planner.generators.path_factory import PathFactory

        # 45% south DEM: a target straight downhill can't be reached within ±15%, even
        # with earthwork. Build the factory on THIS DEM (the shared path_factory fixture
        # is bound to a gentler diagonal DEM, which wouldn't exercise the refusal).
        dem = mock_dem_black_slope
        analyzer = TerrainAnalyzer(dem=dem)
        factory = PathFactory(
            dem_service=dem, path_tracer=PathTracer(dem=dem, analyzer=analyzer), terrain_analyzer=analyzer
        )
        sm, ctx, graph = self._building(fake_st, factory, dem)
        toasts = _capture_toasts(monkeypatch)

        self._target(
            ClickInfo(click_type=MapClickType.TERRAIN, lat=-300 / M, lon=0.0),
            elevation=dem.get_elevation_or_raise(lon=0.0, lat=-300 / M),
        )
        assert ctx.proposals.paths == [], "steep target proposes nothing (even the straight line is over cap)"
        assert ctx.build(SegmentKind.ROAD).segments == [], "steep target commits nothing"
        # The target click transitions into ROAD_CUSTOM_PATH; the deferred pass then finds no
        # in-band route and refuses (the user cancels or retargets from there).
        assert sm.is_road_custom_path, "stays in the custom-path flow, no segment added"
        assert len(graph.segments) == 0
        # No transient toast any more — the "too steep" detail is consolidated into the persistent
        # right-panel block, flagged on the proposals context.
        assert toasts == [], "no transient too-steep toast for roads"
        assert ctx.proposals.too_steep_gentlest_pct is not None, (
            "the too-steep reason is stashed for the right-panel detail"
        )

    def test_stray_marker_click_is_rejected(self, fake_st, path_factory, mock_dem_red_slope_diagonal) -> None:
        from skiresort_planner.ui.click_handlers import handle_path_building_click

        sm, ctx, graph = self._building(fake_st, path_factory, mock_dem_red_slope_diagonal)
        handle_path_building_click(
            ClickInfo(click_type=MapClickType.MARKER, marker_type=MarkerType.SLOPE, slope_id="SL1"), elevation=None
        )
        assert ctx.build(SegmentKind.ROAD).segments == []
        assert len(graph.segments) == 0

    def test_node_target_is_connector_and_auto_finishes(
        self, fake_st, path_factory, mock_dem_red_slope_diagonal
    ) -> None:
        dem = mock_dem_red_slope_diagonal
        sm, ctx, graph = self._building(fake_st, path_factory, dem)
        end, _ = graph.get_or_create_node(
            lon=300 / M, lat=0.0, elevation=dem.get_elevation_or_raise(lon=300 / M, lat=0.0)
        )

        self._target(
            ClickInfo(click_type=MapClickType.MARKER, marker_type=MarkerType.NODE, node_id=end.id), elevation=None
        )
        assert len(ctx.proposals.paths) >= 1, "a node target proposes a route to browse"
        assert all(p.is_connector and p.target_node_id == end.id for p in ctx.proposals.paths), (
            "a node target is a connector that snaps onto the node"
        )
        # Committing a connector auto-finishes the road (parity with slope custom-connect).
        self._commit_proposal()
        assert sm.is_idle_viewing_road, "connector commit auto-finishes the road"
        assert len(graph.roads) == 1


class TestRoadBuildingEdgeCases:
    """Distance guard, proposal-endpoint parity, and node-reuse branches of the road handler."""

    def _building(self, fake_st, factory, dem):
        sm, ctx = _session(fake_st, ResortGraph(), factory, dem)
        ctx.build_mode.mode = BuildMode.ROAD
        from skiresort_planner.ui.click_handlers import handle_idle_click

        handle_idle_click(
            ClickInfo(click_type=MapClickType.TERRAIN, lat=0.0, lon=0.0),
            elevation=dem.get_elevation_or_raise(lon=0.0, lat=0.0),
        )
        assert sm.is_road_starting
        return sm, ctx, fake_st.session_state["graph"]

    def _target(self, click_info: ClickInfo, elevation: float | None = None) -> None:
        """Click a road target, then run the deferred custom-connect generation (see sibling class)."""
        from skiresort_planner.ui.actions import process_custom_connect_deferred
        from skiresort_planner.ui.click_handlers import handle_path_building_click

        handle_path_building_click(click_info, elevation=elevation)
        process_custom_connect_deferred()

    def test_target_too_far_is_refused_via_handler(self, fake_st, path_factory, mock_dem_red_slope_diagonal) -> None:
        # The 1000 m click-distance cap has an isolated validator test; drive it THROUGH the handler
        # to prove a far target short-circuits before proposals/segments (a distinct guard from the
        # too-steep refusal, which the existing suite covers).
        from skiresort_planner.ui.click_handlers import handle_path_building_click

        dem = mock_dem_red_slope_diagonal
        sm, ctx, graph = self._building(fake_st, path_factory, dem)
        far_lon = (PathConfig.SEGMENT_LENGTH_MAX_M + 500) / M  # 1500 m east of the origin

        handle_path_building_click(
            ClickInfo(click_type=MapClickType.TERRAIN, lat=0.0, lon=far_lon),
            elevation=dem.get_elevation_or_raise(lon=far_lon, lat=0.0),
        )
        assert ctx.proposals.paths == [], "a too-far target proposes nothing"
        assert ctx.build(SegmentKind.ROAD).segments == [] and len(graph.segments) == 0, (
            "a too-far target commits nothing"
        )
        assert sm.is_road_starting, "a too-far target does not leave the building flow"

    def test_proposal_endpoint_click_commits_immediately(
        self, fake_st, path_factory, mock_dem_red_slope_diagonal
    ) -> None:
        # For roads (as for slopes), the orange ENDPOINT marker is an instant-commit affordance:
        # one click commits that path outright, even when a DIFFERENT proposal is selected. Only the
        # in-between BODY markers use the select-then-commit rule.
        from skiresort_planner.ui.click_handlers import handle_path_building_click

        dem = mock_dem_red_slope_diagonal
        sm, ctx, _graph = self._building(fake_st, path_factory, dem)
        self._target(
            ClickInfo(click_type=MapClickType.TERRAIN, lat=0.0, lon=300 / M),
            elevation=dem.get_elevation_or_raise(lon=300 / M, lat=0.0),
        )
        # Two-proposal browse state with a DIFFERENT one selected (idx 1); an endpoint click on
        # idx 0 must commit idx 0 straight away — no prior selection of it required.
        ctx.proposals.paths = ctx.proposals.paths + ctx.proposals.paths[:1]
        ctx.proposals.selected_idx = 1
        handle_path_building_click(
            ClickInfo(click_type=MapClickType.MARKER, marker_type=MarkerType.PROPOSAL_ENDPOINT, proposal_index=0),
            elevation=None,
        )
        assert len(ctx.build(SegmentKind.ROAD).segments) == 1, "an endpoint click commits immediately"
        assert sm.is_road_building_only

    def test_brand_new_terrain_start_proposals_have_no_node_ids(
        self, fake_st, path_factory, mock_dem_red_slope_diagonal
    ) -> None:
        # A first segment from a fresh terrain origin: NO origin node is materialised while routing
        # (the origin is a pending location, minted only at commit). So proposals carry no start node
        # id yet, and committing mints BOTH the origin and the endpoint node.
        from skiresort_planner.ui.actions import commit_selected_path

        dem = mock_dem_red_slope_diagonal
        _sm, ctx, graph = self._building(fake_st, path_factory, dem)
        nodes_at_start = len(graph.nodes)
        self._target(
            ClickInfo(click_type=MapClickType.TERRAIN, lat=0.0, lon=300 / M),
            elevation=dem.get_elevation_or_raise(lon=300 / M, lat=0.0),
        )
        assert ctx.proposals.paths, "a reachable target proposes at least one route"
        assert all(not p.start_node_id for p in ctx.proposals.paths), "no origin node before commit"
        assert all(not p.target_node_id for p in ctx.proposals.paths), "a terrain target is not a node"
        assert len(graph.nodes) == nodes_at_start, "routing a fresh-terrain target materialises no node"
        # Committing mints the origin AND the endpoint (neither existed before) — one origin, no dup.
        nodes_before = len(graph.nodes)
        commit_selected_path(path_idx=0)
        assert len(graph.nodes) == nodes_before + 2, "commit adds the origin + endpoint nodes"

    def test_extension_proposals_anchor_on_the_last_endpoint(
        self, fake_st, path_factory, mock_dem_red_slope_diagonal
    ) -> None:
        # After one committed segment, the NEXT click extends from the last endpoint node — its
        # proposals must reuse that exact node id as their start (never duplicate the junction).
        from skiresort_planner.ui.actions import commit_selected_path

        dem = mock_dem_red_slope_diagonal
        _sm, ctx, graph = self._building(fake_st, path_factory, dem)
        self._target(
            ClickInfo(click_type=MapClickType.TERRAIN, lat=0.0, lon=300 / M),
            elevation=dem.get_elevation_or_raise(lon=300 / M, lat=0.0),
        )
        commit_selected_path(path_idx=0)
        assert ctx.build(SegmentKind.ROAD).endpoints, "a committed segment records an endpoint"
        last_endpoint_id = ctx.build(SegmentKind.ROAD).endpoints[-1]

        self._target(
            ClickInfo(click_type=MapClickType.TERRAIN, lat=0.0, lon=600 / M),
            elevation=dem.get_elevation_or_raise(lon=600 / M, lat=0.0),
        )
        assert ctx.proposals.paths, "extending proposes at least one route"
        assert all(p.start_node_id == last_endpoint_id for p in ctx.proposals.paths), (
            "extension proposals reuse the last endpoint node as their start"
        )
        assert last_endpoint_id in graph.nodes, "the reused endpoint is a real graph node, not a duplicate"

    def test_straight_line_offered_when_grid_returns_only_over_cap(
        self, fake_st, path_factory, mock_dem_red_slope_diagonal
    ) -> None:
        """When the grid planner returns only over-cap routes, the in-cap straight line is offered.

        The grid planner is forced to yield only an over-cap serpentine; the endpoint-to-endpoint
        grade is gentle, so the straight line (≤15%) is the sole surviving proposal.
        """
        from skiresort_planner.ui.actions import process_custom_connect_deferred
        from skiresort_planner.ui.click_handlers import handle_path_building_click

        dem = mock_dem_red_slope_diagonal
        sm, ctx, _graph = self._building(fake_st, path_factory, dem)
        toasts = _capture_toasts(pytest.MonkeyPatch())

        # Force the grid planner to emit only an OVER-cap candidate so only the straight line
        # (to the gently reachable target) survives the cap.
        over_cap = ProposedPathSegment(
            points=[PathPoint(lon=0.0, lat=0.0, elevation=2500.0), PathPoint(lon=150 / M, lat=0.0, elevation=2400.0)],
            kind=SegmentKind.ROAD,
        )
        assert over_cap.max_slope_pct > float(PathConfig.ROAD_MAX_GRADIENT_PCT), "fixture must be over the cap"
        mp = pytest.MonkeyPatch()
        mp.setattr(path_factory, "generate_manual_paths", lambda **kwargs: iter([over_cap]))

        # Target ~15m east, ~1m drop → a straight line well under 15%. Generation lives in
        # the deferred pass, so keep the mock active across it (do NOT use self._target here).
        handle_path_building_click(
            ClickInfo(click_type=MapClickType.TERRAIN, lat=0.0, lon=15 / M),
            elevation=dem.get_elevation_or_raise(lon=15 / M, lat=0.0),
        )
        process_custom_connect_deferred()
        mp.undo()

        assert len(ctx.proposals.paths) == 1, "only the in-cap straight line survives"
        straight = ctx.proposals.paths[0]
        assert straight.kind == SegmentKind.ROAD
        assert straight.max_slope_pct <= float(PathConfig.ROAD_MAX_GRADIENT_PCT)
        assert "straight line" in straight.sector_name.lower()
        assert not toasts, "an in-band straight line must NOT raise a too-steep toast"
        assert sm.is_road_custom_path

    def test_straight_line_refused_when_direct_also_too_steep(self, fake_st, mock_dem_black_slope) -> None:
        """A car road is refused when even the straight line exceeds ±15%.

        Straight down the 45% DEM the direct grade is ~45% — over the cap — so neither a
        serpentine nor the straight line fits, and the too-steep state is flagged (no toast).
        """
        from skiresort_planner.core.path_tracer import PathTracer
        from skiresort_planner.core.terrain_analyzer import TerrainAnalyzer
        from skiresort_planner.generators.path_factory import PathFactory

        dem = mock_dem_black_slope
        analyzer = TerrainAnalyzer(dem=dem)
        factory = PathFactory(
            dem_service=dem, path_tracer=PathTracer(dem=dem, analyzer=analyzer), terrain_analyzer=analyzer
        )
        sm, ctx, graph = self._building(fake_st, factory, dem)
        toasts = _capture_toasts(pytest.MonkeyPatch())

        self._target(
            ClickInfo(click_type=MapClickType.TERRAIN, lat=-300 / M, lon=0.0),
            elevation=dem.get_elevation_or_raise(lon=0.0, lat=-300 / M),
        )
        assert ctx.proposals.paths == [], "no serpentine and no in-band straight line → nothing proposed"
        assert ctx.build(SegmentKind.ROAD).segments == [], "nothing committed"
        assert sm.is_road_custom_path
        assert len(graph.segments) == 0
        assert toasts == [], "no transient too-steep toast for roads"
        assert ctx.proposals.too_steep_gentlest_pct is not None, (
            "the too-steep reason is stashed for the right-panel detail"
        )

    def test_straight_line_fallback_carries_connector_and_start_node_ids(
        self, fake_st, path_factory, mock_dem_red_slope_diagonal
    ) -> None:
        """The direct-line fallback is tagged like any proposal: connector + start-node reuse.

        Guards the 'fall through, don't early-return' design: after building the fallback,
        the connector loop and start-node loop must still run on it.
        """
        from skiresort_planner.ui.actions import process_custom_connect_deferred
        from skiresort_planner.ui.click_handlers import handle_path_building_click

        dem = mock_dem_red_slope_diagonal
        # Start the road AT an existing node so start_node_id must propagate.
        graph = ResortGraph()
        sm, ctx = _session(fake_st, graph, path_factory, dem)
        from skiresort_planner.model.node import Node

        start = Node(id="N_start", location=PathPoint(lon=0.0, lat=0.0, elevation=2500.0))
        target = Node(id="N_target", location=PathPoint(lon=15 / M, lat=0.0, elevation=2499.0))
        graph.nodes[start.id] = start
        graph.nodes[target.id] = target
        ctx.build_mode.mode = BuildMode.ROAD
        sm.select_road_start(node_id=start.id, location=None)
        assert sm.is_road_starting

        over_cap = ProposedPathSegment(
            points=[PathPoint(lon=0.0, lat=0.0, elevation=2500.0), PathPoint(lon=150 / M, lat=0.0, elevation=2400.0)],
            kind=SegmentKind.ROAD,
        )
        mp = pytest.MonkeyPatch()
        mp.setattr(path_factory, "generate_manual_paths", lambda **kwargs: iter([over_cap]))
        # The fallback runs in the deferred pass, so keep the mock active across it.
        handle_path_building_click(
            ClickInfo(click_type=MapClickType.MARKER, marker_type=MarkerType.NODE, node_id=target.id),
            elevation=None,
        )
        process_custom_connect_deferred()
        mp.undo()

        assert len(ctx.proposals.paths) == 1
        straight = ctx.proposals.paths[0]
        assert straight.is_connector, "a node target makes the fallback a connector"
        assert straight.target_node_id == target.id, "connector carries the target node id"
        assert straight.start_node_id == start.id, "fallback reuses the existing start node"

    def test_road_fan_appears_on_enter_and_is_band_filtered(
        self, fake_st, path_factory, mock_dem_red_slope_diagonal
    ) -> None:
        """Entering road build triggers the road fan, hard-capped at ±15%.

        The enter hook sets the deferred flag; process_path_generation_deferred fills
        ctx.proposals with a fan whose every member is within the band.
        """
        from skiresort_planner.ui.actions import process_path_generation_deferred

        dem = mock_dem_red_slope_diagonal
        sm, ctx, _graph = self._building(fake_st, path_factory, dem)
        assert SegmentKind.ROAD in ctx.deferred.fan_generation, "entering road build queues the road fan"

        process_path_generation_deferred()
        assert ctx.proposals.paths, "the road fan proposes routes from the origin on gentle-enough terrain"
        assert all(p.max_slope_pct <= float(PathConfig.ROAD_MAX_GRADIENT_PCT) for p in ctx.proposals.paths), (
            "every road-fan proposal is within the ±15% band"
        )
        assert all(p.kind == SegmentKind.ROAD for p in ctx.proposals.paths)

    def test_road_fan_filter_drops_over_cap_routes_on_steep_terrain(self, fake_st, mock_dem_black_slope) -> None:
        """On 45% terrain the ±15% filter drops the steep-green routes but keeps the gentle ones.

        This is the fan's whole value on steep ground: a 7% traverse holds ~11% (in-band)
        while a 12% traverse spills to ~17% (dropped). Proves the filter actually bites —
        the surviving set is a strict, non-empty subset of the raw fan.
        """
        from skiresort_planner.core.path_tracer import PathTracer
        from skiresort_planner.core.terrain_analyzer import TerrainAnalyzer
        from skiresort_planner.generators.path_factory import PathFactory
        from skiresort_planner.ui.actions import process_path_generation_deferred

        dem = mock_dem_black_slope
        analyzer = TerrainAnalyzer(dem=dem)
        factory = PathFactory(
            dem_service=dem, path_tracer=PathTracer(dem=dem, analyzer=analyzer), terrain_analyzer=analyzer
        )
        sm, ctx, _graph = self._building(fake_st, factory, dem)

        raw = list(
            factory.generate_fan(
                kind=SegmentKind.ROAD, lon=0.0, lat=0.0, elevation=2500.0, target_length_m=ctx.segment_length_m
            )
        )
        over_cap = [p for p in raw if p.max_slope_pct > float(PathConfig.ROAD_MAX_GRADIENT_PCT)]
        assert over_cap, "on 45% terrain the steep-green routes exceed the cap (filter must have something to drop)"

        process_path_generation_deferred()
        assert ctx.proposals.paths, "gentle green traverses still hold an in-band grade on steep ground"
        assert len(ctx.proposals.paths) < len(raw), "the ±15% filter drops the over-cap routes"
        assert all(p.max_slope_pct <= float(PathConfig.ROAD_MAX_GRADIENT_PCT) for p in ctx.proposals.paths)


# =============================================================================
# dispatch_click — entry point: elevation lookup + routing to state handler
# =============================================================================


class TestDispatchClick:
    def test_terrain_click_routes_to_idle_handler(self, fake_st, path_factory, mock_dem_red_slope_diagonal) -> None:
        from skiresort_planner.ui.mode_registry import dispatch_click

        dem = mock_dem_red_slope_diagonal
        sm, ctx = _session(fake_st, ResortGraph(), path_factory, dem)
        ctx.build_mode.mode = BuildMode.SLOPE

        # dispatch_click looks up elevation for terrain clicks, then routes to the idle handler.
        dispatch_click(ClickInfo(click_type=MapClickType.TERRAIN, lat=0.0, lon=0.0))
        assert sm.is_slope_starting

    def test_marker_click_routes_without_elevation_lookup(
        self, fake_st, path_factory, mock_dem_red_slope_diagonal, path_points_blue
    ) -> None:
        from skiresort_planner.ui.mode_registry import dispatch_click

        graph = ResortGraph()
        graph.commit_paths(paths=[ProposedPathSegment(points=path_points_blue, target_difficulty="blue")])
        slope = graph.finish_slope(segment_ids=list(graph.segments.keys()))
        assert slope is not None
        sm, ctx = _session(fake_st, graph, path_factory, mock_dem_red_slope_diagonal)

        # A marker click carries no lat/lon; dispatch must route it straight to the handler.
        dispatch_click(ClickInfo(click_type=MapClickType.MARKER, marker_type=MarkerType.SLOPE, slope_id=slope.id))
        assert sm.is_idle_viewing_slope and ctx.viewing.slope_id == slope.id


# =============================================================================
# no build-state handler crashes on ANY marker the map can emit.
# =============================================================================


class TestBuildStateMarkerCompleteness:
    """The map renders finished slopes/roads/lifts (and their segments/pylons) as pickable in every
    state, so a user can click any of them mid-build. Each build-state handler must handle those
    entity markers WITHOUT raising — it should politely reject them (InvalidClickMessage), never crash.

    This guards the class of bug where a handler forgot an entity-marker type: a ROAD marker clicked
    while building a slope or placing a lift used to hit `raise RuntimeError`. (NODE and PROPOSAL_*
    are functional interaction paths needing live state — covered by the routing tests above.)
    """

    # Entity markers the map emits for FINISHED entities — all must be politely rejected mid-build.
    _ENTITY_MARKERS = {
        MarkerType.SLOPE: {"slope_id": "SL1"},
        MarkerType.SEGMENT: {"segment_id": "S1"},
        MarkerType.LIFT: {"lift_id": "L1"},
        MarkerType.ROAD: {"road_id": "R1"},
        MarkerType.PYLON: {"lift_id": "L1", "pylon_index": 0},
    }

    def _entity_marker_clicks(self):
        for marker_type, kwargs in self._ENTITY_MARKERS.items():
            yield marker_type, ClickInfo(click_type=MapClickType.MARKER, marker_type=marker_type, **kwargs)  # type: ignore[arg-type]  # dynamic parametrize kwargs

    def test_slope_building_rejects_every_entity_marker(
        self, fake_st, path_factory, mock_dem_red_slope_diagonal
    ) -> None:
        from skiresort_planner.ui.click_handlers import handle_path_building_click

        for _marker_type, ci in self._entity_marker_clicks():
            sm, ctx = _session(fake_st, ResortGraph(), path_factory, mock_dem_red_slope_diagonal)
            sm.start_slope(lon=0.0, lat=0.0, elevation=2500.0, node_id=None)
            # Must not raise for any entity marker (shows an InvalidClickMessage instead).
            handle_path_building_click(click_info=ci, elevation=2000.0)
            assert sm.is_slope_starting, "a stray entity marker must not advance or leave the slope build"

    def test_lift_placing_rejects_every_entity_marker(self, fake_st, path_factory, mock_dem_red_slope_diagonal) -> None:
        from skiresort_planner.ui.click_handlers import handle_lift_placing_click

        for marker_type, ci in self._entity_marker_clicks():
            graph = ResortGraph()
            sm, ctx = _session(fake_st, graph, path_factory, mock_dem_red_slope_diagonal)
            ctx.lift.start_location = PathPoint(lon=0.0, lat=-0.01, elevation=2400.0)
            sm.select_lift_start(location=ctx.lift.start_location)
            handle_lift_placing_click(click_info=ci, elevation=2000.0)
            # Politely rejected: no lift built, still placing, and the pending start is preserved.
            assert len(graph.lifts) == 0, f"{marker_type.name} must not build a lift"
            assert sm.is_lift_placing, f"{marker_type.name} must not abandon lift placement"
            assert ctx.lift.start_location is not None, f"{marker_type.name} must not clear the pending start"

    def test_road_building_rejects_every_entity_marker(
        self, fake_st, path_factory, mock_dem_red_slope_diagonal
    ) -> None:
        from skiresort_planner.ui.click_handlers import handle_path_building_click

        for marker_type, ci in self._entity_marker_clicks():
            sm, ctx = _session(fake_st, ResortGraph(), path_factory, mock_dem_red_slope_diagonal)
            ctx.build_mode.mode = BuildMode.ROAD
            sm.select_road_start(location=PathPoint(lon=0.0, lat=0.0, elevation=2500.0))
            graph = fake_st.session_state["graph"]
            handle_path_building_click(click_info=ci, elevation=2000.0)
            # Politely rejected: no segment committed, no proposal generated, still starting the road.
            assert ctx.build(SegmentKind.ROAD).segments == [], f"{marker_type.name} must not commit a segment"
            assert ctx.proposals.paths == [], f"{marker_type.name} must not generate proposals"
            assert len(graph.segments) == 0, f"{marker_type.name} must not add graph segments"
            assert sm.is_road_starting, f"{marker_type.name} must not leave road building"

    def test_merge_placing_rejects_every_entity_marker(
        self, fake_st, path_factory, mock_dem_red_slope_diagonal
    ) -> None:
        from skiresort_planner.ui.click_handlers import handle_merge_placing_click

        for _marker_type, ci in self._entity_marker_clicks():
            sm, ctx = _session(fake_st, ResortGraph(), path_factory, mock_dem_red_slope_diagonal)
            sm.start_merge()
            handle_merge_placing_click(click_info=ci, elevation=None)
            assert ctx.merge.node_ids == [], "no entity marker adds to the merge selection"
            assert sm.is_merge_placing, "an entity marker must not navigate away from merge placing"

    def test_import_placing_rejects_every_entity_marker(
        self, fake_st, path_factory, mock_dem_red_slope_diagonal
    ) -> None:
        # The import handler was the one build-state handler missing from this completeness suite.
        # An entity marker clicked while placing an import box must be inert: no confirm, no move,
        # no crash, still placing.
        from skiresort_planner.ui.click_handlers import handle_import_placing_click

        for marker_type, ci in self._entity_marker_clicks():
            sm, ctx = _session(fake_st, ResortGraph(), path_factory, mock_dem_red_slope_diagonal)
            ctx.build_mode.mode = BuildMode.IMPORT
            sm.start_import(lon=0.02, lat=0.03)  # a distinctive placed center, so a stray re-place shows
            handle_import_placing_click(click_info=ci, elevation=None)
            assert sm.is_import_placing, f"{marker_type.name} must not leave import placing"
            assert ctx.deferred.osm_import is False, f"{marker_type.name} must not confirm the import"
            assert (ctx.deferred.osm_import_center_lon, ctx.deferred.osm_import_center_lat) == (0.02, 0.03), (
                f"{marker_type.name} must not move the placed box center"
            )


# =============================================================================
# handle_import_placing_click — center-dot re-click confirms; terrain re-places
# =============================================================================


class TestImportPlacingClick:
    def _placing(self, fake_st, factory, dem):
        """Enter import_placing with a box center already placed."""
        sm, ctx = _session(fake_st, ResortGraph(), factory, dem)
        ctx.build_mode.mode = BuildMode.IMPORT
        sm.start_import(lon=0.0, lat=0.0)
        return sm, ctx

    def test_center_dot_click_confirms_import(self, fake_st, path_factory, mock_dem_red_slope_diagonal) -> None:
        from skiresort_planner.ui.click_handlers import handle_import_placing_click

        sm, ctx = self._placing(fake_st, path_factory, mock_dem_red_slope_diagonal)
        handle_import_placing_click(
            ClickInfo(click_type=MapClickType.MARKER, marker_type=MarkerType.IMPORT_CENTER), elevation=None
        )
        # confirm_import_action flags the deferred fetch and returns to idle
        assert ctx.deferred.osm_import is True
        assert sm.is_idle_ready

    def test_terrain_click_replaces_center_and_keeps_placing(
        self, fake_st, path_factory, mock_dem_red_slope_diagonal
    ) -> None:
        from skiresort_planner.ui.click_handlers import handle_import_placing_click

        sm, ctx = self._placing(fake_st, path_factory, mock_dem_red_slope_diagonal)
        handle_import_placing_click(ClickInfo(click_type=MapClickType.TERRAIN, lat=0.05, lon=0.06), elevation=2000.0)
        assert sm.is_import_placing, "re-placing keeps us in import mode"
        assert ctx.deferred.osm_import_center_lon == 0.06 and ctx.deferred.osm_import_center_lat == 0.05
        assert ctx.deferred.osm_import is False, "re-placing does not confirm"

    def test_replace_then_confirm_targets_the_replaced_center(
        self, fake_st, path_factory, mock_dem_red_slope_diagonal
    ) -> None:
        # The two-click flow: re-place the box, THEN confirm. The confirmed fetch must target the
        # LAST-placed center (0.06, 0.05), not the original (0, 0) — otherwise a user who nudges the
        # box would silently import the wrong area.
        from skiresort_planner.ui.click_handlers import handle_import_placing_click

        sm, ctx = self._placing(fake_st, path_factory, mock_dem_red_slope_diagonal)
        handle_import_placing_click(ClickInfo(click_type=MapClickType.TERRAIN, lat=0.05, lon=0.06), elevation=2000.0)
        handle_import_placing_click(
            ClickInfo(click_type=MapClickType.MARKER, marker_type=MarkerType.IMPORT_CENTER), elevation=None
        )
        assert ctx.deferred.osm_import is True, "the center dot confirms the fetch"
        assert sm.is_idle_ready
        assert ctx.deferred.osm_import_center_lon == 0.06 and ctx.deferred.osm_import_center_lat == 0.05, (
            "confirm imports the re-placed center, not the original"
        )


# =============================================================================
# handle_merge_placing_click — toggle node selection, reject non-node clicks
# =============================================================================


class TestMergePlacingClick:
    def _merge_session(self, fake_st, graph, factory, dem):
        sm, ctx = _session(fake_st, graph, factory, dem)
        sm.start_merge()
        return sm, ctx

    def test_node_click_toggles_selection(self, fake_st, path_factory, mock_dem_red_slope_diagonal) -> None:
        from skiresort_planner.ui.click_handlers import handle_merge_placing_click

        graph = ResortGraph()
        node, _ = graph.get_or_create_node(lon=0.0, lat=0.0, elevation=2000.0)
        sm, ctx = self._merge_session(fake_st, graph, path_factory, mock_dem_red_slope_diagonal)

        handle_merge_placing_click(
            ClickInfo(click_type=MapClickType.MARKER, marker_type=MarkerType.NODE, node_id=node.id), elevation=None
        )
        assert ctx.merge.node_ids == [node.id]
        assert sm.is_merge_placing, "selecting a node keeps us in merge placing"

    def test_reclick_node_removes_it(self, fake_st, path_factory, mock_dem_red_slope_diagonal) -> None:
        from skiresort_planner.ui.click_handlers import handle_merge_placing_click

        graph = ResortGraph()
        node, _ = graph.get_or_create_node(lon=0.0, lat=0.0, elevation=2000.0)
        sm, ctx = self._merge_session(fake_st, graph, path_factory, mock_dem_red_slope_diagonal)
        click = ClickInfo(click_type=MapClickType.MARKER, marker_type=MarkerType.NODE, node_id=node.id)

        handle_merge_placing_click(click, elevation=None)
        handle_merge_placing_click(click, elevation=None)
        assert ctx.merge.node_ids == [], "re-clicking a selected node deselects it"

    def test_terrain_click_is_rejected_without_crashing(
        self, fake_st, path_factory, mock_dem_red_slope_diagonal
    ) -> None:
        from skiresort_planner.ui.click_handlers import handle_merge_placing_click

        graph = ResortGraph()
        _sm, ctx = self._merge_session(fake_st, graph, path_factory, mock_dem_red_slope_diagonal)

        handle_merge_placing_click(
            ClickInfo(click_type=MapClickType.TERRAIN, lat=0.0, lon=0.0),
            elevation=mock_dem_red_slope_diagonal.get_elevation_or_raise(lon=0.0, lat=0.0),
        )
        assert ctx.merge.node_ids == [], "terrain clicks never add to the merge selection"

    def test_slope_marker_click_is_rejected(self, fake_st, path_factory, mock_dem_blue_slope, path_points_blue) -> None:
        from skiresort_planner.ui.click_handlers import handle_merge_placing_click

        graph = ResortGraph()
        graph.commit_paths(paths=[ProposedPathSegment(points=path_points_blue, target_difficulty="blue")])
        slope = graph.finish_slope(segment_ids=list(graph.segments.keys()))
        assert slope is not None
        _sm, ctx = self._merge_session(fake_st, graph, path_factory, mock_dem_blue_slope)

        # A non-node marker must be rejected (InvalidClickMessage), never crash, never select.
        handle_merge_placing_click(
            ClickInfo(click_type=MapClickType.MARKER, marker_type=MarkerType.SLOPE, slope_id=slope.id), elevation=None
        )
        assert ctx.merge.node_ids == []

    def test_second_distinct_node_grows_the_selection(self, fake_st, path_factory, mock_dem_red_slope_diagonal) -> None:
        # Existing tests select one node or re-click the same one; this exercises the multi-select
        # path — a second DISTINCT node must be appended (the state a real merge always needs).
        from skiresort_planner.ui.click_handlers import handle_merge_placing_click

        graph = ResortGraph()
        first, _ = graph.get_or_create_node(lon=0.02, lat=0.03, elevation=2000.0)
        second, _ = graph.get_or_create_node(lon=0.05, lat=0.07, elevation=1950.0)
        sm, ctx = self._merge_session(fake_st, graph, path_factory, mock_dem_red_slope_diagonal)

        handle_merge_placing_click(
            ClickInfo(click_type=MapClickType.MARKER, marker_type=MarkerType.NODE, node_id=first.id), elevation=None
        )
        handle_merge_placing_click(
            ClickInfo(click_type=MapClickType.MARKER, marker_type=MarkerType.NODE, node_id=second.id), elevation=None
        )
        assert ctx.merge.node_ids == [first.id, second.id], "a second distinct node joins the selection"
        assert sm.is_merge_placing, "selecting nodes keeps us in merge placing"

    def test_stale_node_id_still_toggles_without_crashing(
        self, fake_st, path_factory, mock_dem_red_slope_diagonal
    ) -> None:
        # A NODE marker id missing from the graph (deleted-but-still-drawn) must be recorded by the
        # toggle without raising — the selection is a pure id set, robust to a one-frame desync.
        from skiresort_planner.ui.click_handlers import handle_merge_placing_click

        sm, ctx = self._merge_session(fake_st, ResortGraph(), path_factory, mock_dem_red_slope_diagonal)

        handle_merge_placing_click(
            ClickInfo(click_type=MapClickType.MARKER, marker_type=MarkerType.NODE, node_id="GHOST"), elevation=None
        )
        assert ctx.merge.node_ids == ["GHOST"], "the toggle records the id even when the node is gone"
        assert sm.is_merge_placing, "a stale node id does not crash or leave merge placing"

    def test_terrain_reject_stays_in_merge_placing(self, fake_st, path_factory, mock_dem_red_slope_diagonal) -> None:
        # Strengthen the reject path: a rejected terrain click must not just leave the selection
        # empty, it must also NOT navigate away (a regression that fired a cancel/finish would slip
        # past an assertion that only checks node_ids).
        from skiresort_planner.ui.click_handlers import handle_merge_placing_click

        sm, ctx = self._merge_session(fake_st, ResortGraph(), path_factory, mock_dem_red_slope_diagonal)
        handle_merge_placing_click(
            ClickInfo(click_type=MapClickType.TERRAIN, lat=0.0, lon=0.0),
            elevation=mock_dem_red_slope_diagonal.get_elevation_or_raise(lon=0.0, lat=0.0),
        )
        assert ctx.merge.node_ids == []
        assert sm.is_merge_placing, "a rejected terrain click keeps us in merge placing"

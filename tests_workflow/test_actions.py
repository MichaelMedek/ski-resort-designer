"""Core-logic tests for the UI action functions (actions.py).

These functions read st.session_state.{state_machine, context, graph}; with the
fake `st` installed we seed those and call the action directly, asserting the
real effect (entity removed, panel closed when it was being viewed). Covers the
delete actions for slope/lift/road uniformly.
"""

import pytest

from skiresort_planner.constants import MapConfig
from skiresort_planner.model.actions import ActionType
from skiresort_planner.model.node import Node
from skiresort_planner.model.path_point import PathPoint
from skiresort_planner.model.path_segment import SegmentKind
from skiresort_planner.model.proposed_path import ProposedPathSegment
from skiresort_planner.model.resort_graph import ResortGraph
from skiresort_planner.ui.state_machine import PlannerStateMachine
from tests_workflow.conftest import MockDEMService


def _node_at(dem: MockDEMService, node_id: str, lon: float, lat: float) -> Node:
    """A Node at (lon, lat) with DEM elevation — for seeding lift stations in delete tests."""
    return Node(
        id=node_id, location=PathPoint(lon=lon, lat=lat, elevation=dem.get_elevation_or_raise(lon=lon, lat=lat))
    )


def _fake_import_result(dem: MockDEMService):
    """An ImportResult with one slope chain + one lift, all DEM-sampled inside MockDEM bounds — the
    payload the mocked GraphImporter returns so process_osm_import_pending can materialise it.
    """
    from skiresort_planner.generators.osm_importer import ImportResult

    slope_points = [
        PathPoint(lon=0.0, lat=0.0, elevation=dem.get_elevation_or_raise(lon=0.0, lat=0.0)),
        PathPoint(
            lon=0.0,
            lat=-500 / MapConfig.METERS_PER_DEGREE_EQUATOR,
            elevation=dem.get_elevation_or_raise(lon=0.0, lat=-500 / MapConfig.METERS_PER_DEGREE_EQUATOR),
        ),
    ]
    lift = (
        PathPoint(
            lon=0.02,
            lat=-500 / MapConfig.METERS_PER_DEGREE_EQUATOR,
            elevation=dem.get_elevation_or_raise(lon=0.02, lat=-500 / MapConfig.METERS_PER_DEGREE_EQUATOR),
        ),
        PathPoint(lon=0.02, lat=0.0, elevation=dem.get_elevation_or_raise(lon=0.02, lat=0.0)),
        "chairlift",
        "Gipfelbahn",
    )
    return ImportResult(lifts=[lift], slope_chains=[([slope_points], "Imported Run")])


def _noop_report(frac: float, text: str) -> None:
    """A no-op ProgressFn for tests that call process_osm_import_pending directly (no progress bar)."""


def _session(fake_st, graph, factory=None, dem=None):
    """Seed fake st.session_state with the objects action functions read."""
    sm, ctx = PlannerStateMachine.create(graph=graph, add_ui_listener=False)
    fake_st.session_state["state_machine"] = sm
    fake_st.session_state["context"] = ctx
    fake_st.session_state["graph"] = graph
    fake_st.session_state["camera_epoch"] = 0
    fake_st.session_state["dedup_epoch"] = 0
    if factory is not None:
        fake_st.session_state["path_factory"] = factory
    if dem is not None:
        fake_st.session_state["dem_service"] = dem
    return sm, ctx


def _make_slope(graph, path_points):
    graph.commit_paths(paths=[ProposedPathSegment(points=path_points, target_difficulty="blue")])
    return graph.finish_slope(segment_ids=list(graph.segments.keys()))


def _make_road(graph):
    pts = [
        PathPoint(lon=0.0, lat=0.0, elevation=2000.0),
        PathPoint(lon=300 / MapConfig.METERS_PER_DEGREE_EQUATOR, lat=0.0, elevation=1990.0),
    ]
    graph.commit_paths(
        paths=[ProposedPathSegment(points=pts, is_connector=True, kind=SegmentKind.ROAD)], record_undo=False
    )
    return graph.finish_road(segment_ids=[list(graph.segments.keys())[-1]])


def _straight_points(dem: MockDEMService) -> list[PathPoint]:
    """A 2-point straight south slope polyline sampled from the DEM (for _make_slope in the matrix)."""
    return [
        PathPoint(lon=0.0, lat=0.0, elevation=dem.get_elevation_or_raise(lon=0.0, lat=0.0)),
        PathPoint(
            lon=0.0,
            lat=-300 / MapConfig.METERS_PER_DEGREE_EQUATOR,
            elevation=dem.get_elevation_or_raise(lon=0.0, lat=-300 / MapConfig.METERS_PER_DEGREE_EQUATOR),
        ),
    ]


def _build_finish_slope(fake_st, factory, dem):
    """Build+finish a 1-segment slope on a fresh session; leave the SM in idle_viewing_slope.

    Returns (sm, ctx, graph, slope, seg_id).
    """
    graph = ResortGraph()
    sm, ctx = _session(fake_st=fake_st, graph=graph, factory=factory, dem=dem)
    pts = [
        PathPoint(lon=0.0, lat=0.0, elevation=2000.0),
        PathPoint(lon=300 / MapConfig.METERS_PER_DEGREE_EQUATOR, lat=0.0, elevation=1990.0),
    ]
    endpoint_ids = graph.commit_paths(paths=[ProposedPathSegment(points=pts, target_difficulty="blue")])
    seg_id = list(graph.segments.keys())[-1]
    sm.start_slope(lon=0.0, lat=0.0, elevation=2000.0, node_id=None)
    sm.commit_path(segment_id=seg_id, endpoint_node_id=endpoint_ids[0])
    slope = graph.finish_slope(segment_ids=ctx.build(kind=SegmentKind.SLOPE).segments)
    sm.finish_slope(entity_id=slope.id)
    return sm, ctx, graph, slope, seg_id


def _commit_straight_slope_len(graph, dem, n_segments: int, start_lat: float = 0.0):
    """Commit n straight ~480m south segments into one finished slope; longer n = longer slope."""
    before = set(graph.segments)
    lat = start_lat
    for _ in range(n_segments):
        pts = []
        for i in range(25):
            plat = lat - 20.0 * i / MapConfig.METERS_PER_DEGREE_EQUATOR
            pts.append(PathPoint(lon=0.0, lat=plat, elevation=dem.get_elevation_or_raise(lon=0.0, lat=plat)))
        graph.commit_paths(paths=[ProposedPathSegment(points=pts, target_difficulty="blue")])
        lat = pts[-1].lat
    return graph.finish_slope(segment_ids=[sid for sid in graph.segments if sid not in before])


class TestDeleteSlopeAction:
    def test_removes_slope(self, fake_st, empty_graph, path_points_blue) -> None:
        from skiresort_planner.ui.actions import delete_slope_action

        slope = _make_slope(graph=empty_graph, path_points=path_points_blue)
        _session(fake_st=fake_st, graph=empty_graph)

        delete_slope_action(slope_id=slope.id)
        assert slope.id not in empty_graph.slopes

    def test_missing_slope_raises(self, fake_st, empty_graph) -> None:
        # Delete only reaches the graph with a live viewed id (the panel asserts liveness first), so a
        # missing id is an internal-invariant violation and fails loud rather than silently no-op'ing.
        from skiresort_planner.ui.actions import delete_slope_action

        _session(fake_st=fake_st, graph=empty_graph)
        with pytest.raises(KeyError):
            delete_slope_action(slope_id="SL999")

    def test_closes_panel_when_viewing_deleted_slope(self, fake_st, empty_graph, path_points_blue) -> None:
        from skiresort_planner.ui.actions import delete_slope_action

        slope = _make_slope(graph=empty_graph, path_points=path_points_blue)
        sm, _ctx = _session(fake_st=fake_st, graph=empty_graph)
        sm.view_slope(slope_id=slope.id)

        delete_slope_action(slope_id=slope.id)
        assert not sm.is_idle_viewing_slope, "deleting the viewed slope must close its panel"


class TestRenameEntityAction:
    def test_sets_name_and_bumps_map(self, fake_st, empty_graph, path_points_blue) -> None:
        from skiresort_planner.ui.actions import rename_entity_action

        slope = _make_slope(graph=empty_graph, path_points=path_points_blue)
        _session(fake_st=fake_st, graph=empty_graph)
        epoch_before = fake_st.session_state["dedup_epoch"]
        camera_before = fake_st.session_state["camera_epoch"]

        rename_entity_action(entity_id=slope.id, new_name="  Renamed  ")

        assert empty_graph.slopes[slope.id].name == "Renamed", "name is trimmed and applied"
        assert fake_st.session_state["dedup_epoch"] > epoch_before, "rename refreshes the label redraw"
        assert fake_st.session_state["camera_epoch"] == camera_before, "rename must NOT recenter"

    def test_empty_name_is_noop(self, fake_st, empty_graph, path_points_blue) -> None:
        from skiresort_planner.ui.actions import rename_entity_action

        slope = _make_slope(graph=empty_graph, path_points=path_points_blue)
        original = slope.name
        _session(fake_st=fake_st, graph=empty_graph)

        rename_entity_action(entity_id=slope.id, new_name="   ")

        assert empty_graph.slopes[slope.id].name == original, "blank name must not overwrite"


class TestDeleteRoadAction:
    def test_removes_road(self, fake_st, empty_graph) -> None:
        from skiresort_planner.ui.actions import delete_road_action

        road = _make_road(graph=empty_graph)
        _session(fake_st=fake_st, graph=empty_graph)

        delete_road_action(road_id=road.id)
        assert road.id not in empty_graph.roads

    def test_missing_road_raises(self, fake_st, empty_graph) -> None:
        # A missing id is an internal-invariant violation (see test_missing_slope_raises) — fail loud.
        from skiresort_planner.ui.actions import delete_road_action

        _session(fake_st=fake_st, graph=empty_graph)
        with pytest.raises(KeyError):
            delete_road_action(road_id="R999")

    def test_closes_panel_when_viewing_deleted_road(self, fake_st, empty_graph) -> None:
        from skiresort_planner.ui.actions import delete_road_action

        road = _make_road(graph=empty_graph)
        sm, _ctx = _session(fake_st=fake_st, graph=empty_graph)
        sm.view_road(road_id=road.id)

        delete_road_action(road_id=road.id)
        assert not sm.is_idle_viewing_road, "deleting the viewed road must close its panel"


class TestDeleteLiftAction:
    def test_removes_lift(self, fake_st, empty_graph, mock_dem_blue_slope) -> None:
        from skiresort_planner.ui.actions import delete_lift_action

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
        _session(fake_st=fake_st, graph=empty_graph)

        delete_lift_action(lift_id=lift.id)
        assert lift.id not in empty_graph.lifts


def _two_segment_slope(graph: ResortGraph, dem: MockDEMService) -> ResortGraph:
    """Commit two contiguous 300m slope segments so the graph has 3 nodes with a junction.

    Nodes sit at lat 0, -300m, -600m (all lon 0). Adjacent nodes are 300m apart (< 500m,
    mergeable); the endpoints are 600m apart (> MergeConfig.MAX_SPAN_M, not mergeable).
    """
    mid = -300 / MapConfig.METERS_PER_DEGREE_EQUATOR
    bot = -600 / MapConfig.METERS_PER_DEGREE_EQUATOR
    seg_a = [
        PathPoint(lon=0.0, lat=0.0, elevation=dem.get_elevation_or_raise(lon=0.0, lat=0.0)),
        PathPoint(lon=0.0, lat=mid, elevation=dem.get_elevation_or_raise(lon=0.0, lat=mid)),
    ]
    seg_b = [
        PathPoint(lon=0.0, lat=mid, elevation=dem.get_elevation_or_raise(lon=0.0, lat=mid)),
        PathPoint(lon=0.0, lat=bot, elevation=dem.get_elevation_or_raise(lon=0.0, lat=bot)),
    ]
    graph.commit_paths(paths=[ProposedPathSegment(points=seg_a, target_difficulty="blue")])
    graph.commit_paths(paths=[ProposedPathSegment(points=seg_b, target_difficulty="blue")])
    return graph


class TestConfirmMergeAction:
    """confirm_merge_action validates the span, merges as one undoable action, returns to idle."""

    def test_close_nodes_merge_and_return_to_idle(self, fake_st, empty_graph, mock_dem_blue_slope) -> None:
        from skiresort_planner.ui.actions import confirm_merge_action

        dem = mock_dem_blue_slope
        _two_segment_slope(graph=empty_graph, dem=dem)
        by_lat = sorted(empty_graph.nodes.values(), key=lambda n: n.lat, reverse=True)
        top, mid = by_lat[0], by_lat[1]  # 300m apart, within MergeConfig.MAX_SPAN_M
        sm, ctx = _session(fake_st=fake_st, graph=empty_graph, dem=dem)
        count_before = len(empty_graph.nodes)
        sm.start_node_edit()
        sm.toggle_node_edit_node(node_id=top.id)
        sm.toggle_node_edit_node(node_id=mid.id)

        confirm_merge_action()

        assert len(empty_graph.nodes) == count_before - 1, "two close nodes collapsed into one"
        assert empty_graph.undo_stack[-1].action_type.name == "MERGE_NODES", "one undoable merge action"
        assert sm.is_idle_ready, "confirm returns to idle"
        assert ctx.node_edit.node_ids == [], "selection cleared by the before-hook"

    def test_far_nodes_refused_no_change(self, fake_st, empty_graph, mock_dem_blue_slope, monkeypatch) -> None:
        from skiresort_planner.ui.actions import confirm_merge_action

        dem = mock_dem_blue_slope
        _two_segment_slope(graph=empty_graph, dem=dem)
        by_lat = sorted(empty_graph.nodes.values(), key=lambda n: n.lat, reverse=True)
        top, bottom = by_lat[0], by_lat[-1]  # 600m apart, exceeds MergeConfig.MAX_SPAN_M
        sm, ctx = _session(fake_st=fake_st, graph=empty_graph, dem=dem)
        count_before = len(empty_graph.nodes)
        stack_before = len(empty_graph.undo_stack)
        sm.start_node_edit()
        sm.toggle_node_edit_node(node_id=top.id)
        sm.toggle_node_edit_node(node_id=bottom.id)

        # MergeTooFarMessage.display() does a function-local `import streamlit as st; st.toast(...)`,
        # so it hits the REAL streamlit module (not the fake `st`); capture it to prove the user is told.
        import streamlit

        toasts: list[str] = []
        monkeypatch.setattr(streamlit, "toast", lambda text, *a, **k: toasts.append(text))

        confirm_merge_action()

        assert len(empty_graph.nodes) == count_before, "nothing merged when the span is too large"
        assert len(empty_graph.undo_stack) == stack_before, "no undo action recorded on refusal"
        assert sm.is_node_edit_selecting, "stays in node edit so the user can adjust the selection"
        assert ctx.node_edit.node_ids == [top.id, bottom.id], "selection preserved for retry"
        assert any("too far" in t.lower() for t in toasts), "the user is told why the merge was refused"

    def test_fewer_than_two_nodes_raises(self, fake_st, empty_graph, mock_dem_blue_slope) -> None:
        from skiresort_planner.ui.actions import confirm_merge_action

        _session(fake_st=fake_st, graph=empty_graph, dem=mock_dem_blue_slope)
        with pytest.raises(RuntimeError, match="fewer than 2"):
            confirm_merge_action()

    def test_missing_lift_raises(self, fake_st, empty_graph) -> None:
        # A missing id is an internal-invariant violation (see test_missing_slope_raises) — fail loud.
        from skiresort_planner.ui.actions import delete_lift_action

        _session(fake_st=fake_st, graph=empty_graph)
        with pytest.raises(KeyError):
            delete_lift_action(lift_id="L999")


class TestDeleteNodesAction:
    """delete_nodes_action deletes deletable nodes (return to idle) or refuses with a toast."""

    def test_interior_node_deletes_and_returns_to_idle(self, fake_st, empty_graph, mock_dem_blue_slope) -> None:
        from skiresort_planner.ui.actions import delete_nodes_action

        dem = mock_dem_blue_slope
        _two_segment_slope(graph=empty_graph, dem=dem)
        slope = empty_graph.finish_slope(segment_ids=list(empty_graph.segments.keys()))
        interior = empty_graph.segments[slope.segment_ids[0]].end_node_id
        sm, ctx = _session(fake_st=fake_st, graph=empty_graph, dem=dem)
        sm.start_node_edit()
        sm.toggle_node_edit_node(node_id=interior)

        delete_nodes_action()

        assert interior not in empty_graph.nodes, "the interior node was deleted"
        assert empty_graph.undo_stack[-1].action_type.name == "DELETE_NODES", "one DELETE_NODES undo action"
        assert sm.is_idle_ready, "delete returns to idle"
        assert ctx.node_edit.node_ids == [], "selection cleared by the before-hook"

    def test_lift_station_refused_no_change(self, fake_st, empty_graph, mock_dem_blue_slope, monkeypatch) -> None:
        from skiresort_planner.ui.actions import delete_nodes_action

        dem = mock_dem_blue_slope
        empty_graph.nodes["A"] = _node_at(dem=dem, node_id="A", lon=0.0, lat=0.0)
        empty_graph.nodes["T"] = _node_at(
            dem=dem, node_id="T", lon=0.0, lat=-1000 / MapConfig.METERS_PER_DEGREE_EQUATOR
        )
        empty_graph.add_lift(start_node_id="A", end_node_id="T", lift_type="chairlift", dem=dem)
        sm, ctx = _session(fake_st=fake_st, graph=empty_graph, dem=dem)
        stack_before = len(empty_graph.undo_stack)
        sm.start_node_edit()
        sm.toggle_node_edit_node(node_id="A")

        import streamlit

        toasts: list[str] = []
        monkeypatch.setattr(streamlit, "toast", lambda text, *a, **k: toasts.append(text))

        delete_nodes_action()

        assert "A" in empty_graph.nodes, "a lift station is never deleted"
        assert len(empty_graph.undo_stack) == stack_before, "no undo action recorded on refusal"
        assert sm.is_node_edit_selecting, "stays in node edit so the user can adjust the selection"
        assert any("lift" in t.lower() for t in toasts), "the user is told why the delete was refused"

    def test_no_nodes_raises(self, fake_st, empty_graph, mock_dem_blue_slope) -> None:
        from skiresort_planner.ui.actions import delete_nodes_action

        _session(fake_st=fake_st, graph=empty_graph, dem=mock_dem_blue_slope)
        with pytest.raises(RuntimeError, match="no selected nodes"):
            delete_nodes_action()

    def test_deleting_whole_path_is_refused(self, fake_st, empty_graph, mock_dem_blue_slope, monkeypatch) -> None:
        """An end node + the sole interior node of a 2-segment slope are each individually deletable,
        but together they'd empty the path — refuse with a message and change nothing.
        """
        from skiresort_planner.ui.actions import delete_nodes_action

        dem = mock_dem_blue_slope
        _two_segment_slope(graph=empty_graph, dem=dem)
        slope = empty_graph.finish_slope(segment_ids=list(empty_graph.segments.keys()))
        end = slope.start_node_id
        interior = empty_graph.segments[slope.segment_ids[0]].end_node_id
        sm, ctx = _session(fake_st=fake_st, graph=empty_graph, dem=dem)
        stack_before = len(empty_graph.undo_stack)
        sm.start_node_edit()
        sm.toggle_node_edit_node(node_id=end)
        sm.toggle_node_edit_node(node_id=interior)

        import streamlit

        toasts: list[str] = []
        monkeypatch.setattr(streamlit, "toast", lambda text, *a, **k: toasts.append(text))

        delete_nodes_action()

        assert slope.id in empty_graph.slopes, "the path is not emptied"
        assert len(empty_graph.undo_stack) == stack_before, "no undo action recorded on refusal"
        assert sm.is_node_edit_selecting, "stays in node edit so the user can adjust the selection"
        assert any("whole path" in t.lower() for t in toasts), "the user is told the delete was refused"

    def test_degree2_junction_of_two_slopes_fuses(self, fake_st, empty_graph, mock_dem_blue_slope, monkeypatch) -> None:
        """A degree-2 node where two slopes meet end-to-end fuses them into one on delete: the node is
        removed and one slope absorbs the other.
        """
        from skiresort_planner.ui.actions import delete_nodes_action

        dem = mock_dem_blue_slope
        # Slope 1 south (2 points) to a junction; slope 2 continues south-east from that same node.
        leg1 = [
            PathPoint(lon=0.0, lat=0.0, elevation=dem.get_elevation_or_raise(lon=0.0, lat=0.0)),
            PathPoint(
                lon=0.0,
                lat=-400 / MapConfig.METERS_PER_DEGREE_EQUATOR,
                elevation=dem.get_elevation_or_raise(lon=0.0, lat=-400 / MapConfig.METERS_PER_DEGREE_EQUATOR),
            ),
        ]
        empty_graph.commit_paths(paths=[ProposedPathSegment(points=leg1, target_difficulty="blue")])
        slope1 = empty_graph.finish_slope(segment_ids=list(empty_graph.segments.keys()))
        junction = slope1.end_node_id
        j = empty_graph.nodes[junction]
        leg2 = [
            PathPoint(lon=j.lon, lat=j.lat, elevation=j.elevation),
            PathPoint(
                lon=400 / MapConfig.METERS_PER_DEGREE_EQUATOR,
                lat=j.lat - 400 / MapConfig.METERS_PER_DEGREE_EQUATOR,
                elevation=dem.get_elevation_or_raise(
                    lon=400 / MapConfig.METERS_PER_DEGREE_EQUATOR, lat=j.lat - 400 / MapConfig.METERS_PER_DEGREE_EQUATOR
                ),
            ),
        ]
        before = set(empty_graph.segments)
        empty_graph.commit_paths(paths=[ProposedPathSegment(points=leg2, target_difficulty="blue")])
        empty_graph.finish_slope(segment_ids=list(set(empty_graph.segments) - before))
        sm, ctx = _session(fake_st=fake_st, graph=empty_graph, dem=dem)
        sm.start_node_edit()
        sm.toggle_node_edit_node(node_id=junction)

        delete_nodes_action()

        assert junction not in empty_graph.nodes, "the degree-2 junction is deleted (slopes fuse)"
        assert len(empty_graph.slopes) == 1, "the two slopes fused into one"

    def test_three_way_branch_refused_no_change(self, fake_st, empty_graph, mock_dem_blue_slope, monkeypatch) -> None:
        """A node where THREE slopes meet is a real branch — deleting it is refused (delete a path
        first), nothing changes.
        """
        from skiresort_planner.ui.actions import delete_nodes_action

        dem = mock_dem_blue_slope
        leg1 = [
            PathPoint(lon=0.0, lat=0.0, elevation=dem.get_elevation_or_raise(lon=0.0, lat=0.0)),
            PathPoint(
                lon=0.0,
                lat=-400 / MapConfig.METERS_PER_DEGREE_EQUATOR,
                elevation=dem.get_elevation_or_raise(lon=0.0, lat=-400 / MapConfig.METERS_PER_DEGREE_EQUATOR),
            ),
        ]
        empty_graph.commit_paths(paths=[ProposedPathSegment(points=leg1, target_difficulty="blue")])
        slope1 = empty_graph.finish_slope(segment_ids=list(empty_graph.segments.keys()))
        junction = slope1.end_node_id
        j = empty_graph.nodes[junction]
        # Two more slopes both start at the junction → 3 segments meet there.
        for d_lon in (400, -400):
            leg = [
                PathPoint(lon=j.lon, lat=j.lat, elevation=j.elevation),
                PathPoint(
                    lon=d_lon / MapConfig.METERS_PER_DEGREE_EQUATOR,
                    lat=j.lat - 400 / MapConfig.METERS_PER_DEGREE_EQUATOR,
                    elevation=dem.get_elevation_or_raise(
                        lon=d_lon / MapConfig.METERS_PER_DEGREE_EQUATOR,
                        lat=j.lat - 400 / MapConfig.METERS_PER_DEGREE_EQUATOR,
                    ),
                ),
            ]
            before = set(empty_graph.segments)
            empty_graph.commit_paths(paths=[ProposedPathSegment(points=leg, target_difficulty="blue")])
            empty_graph.finish_slope(segment_ids=list(set(empty_graph.segments) - before))
        sm, ctx = _session(fake_st=fake_st, graph=empty_graph, dem=dem)
        stack_before = len(empty_graph.undo_stack)
        sm.start_node_edit()
        sm.toggle_node_edit_node(node_id=junction)

        import streamlit

        toasts: list[str] = []
        monkeypatch.setattr(streamlit, "toast", lambda text, *a, **k: toasts.append(text))

        delete_nodes_action()

        assert junction in empty_graph.nodes, "a 3-way branch junction is not deleted"
        assert len(empty_graph.undo_stack) == stack_before, "no undo action recorded on refusal"
        assert sm.is_node_edit_selecting, "stays in node edit so the user can adjust the selection"
        assert any("delete a path" in t.lower() for t in toasts), "the user is told to delete a path first"


class TestDeleteDirectConnectionAction:
    """delete_direct_connection_action cuts every segment joining the 2 selected ADJACENT nodes,
    splitting each owner in two, and toasts NotAdjacentNodesMessage when the pair isn't adjacent.
    """

    def _one_seg_slope(self, graph, dem):
        pts = [
            PathPoint(lon=0.0, lat=0.0, elevation=dem.get_elevation_or_raise(lon=0.0, lat=0.0)),
            PathPoint(
                lon=0.0,
                lat=-400 / MapConfig.METERS_PER_DEGREE_EQUATOR,
                elevation=dem.get_elevation_or_raise(lon=0.0, lat=-400 / MapConfig.METERS_PER_DEGREE_EQUATOR),
            ),
        ]
        graph.commit_paths(paths=[ProposedPathSegment(points=pts, target_difficulty="blue")])
        return graph.finish_slope(segment_ids=list(graph.segments.keys()))

    def _three_seg_slope(self, graph, dem):
        lat = 0.0
        for _ in range(3):
            step = -400 / MapConfig.METERS_PER_DEGREE_EQUATOR
            pts = [
                PathPoint(lon=0.0, lat=lat, elevation=dem.get_elevation_or_raise(lon=0.0, lat=lat)),
                PathPoint(lon=0.0, lat=lat + step, elevation=dem.get_elevation_or_raise(lon=0.0, lat=lat + step)),
            ]
            graph.commit_paths(paths=[ProposedPathSegment(points=pts, target_difficulty="blue")])
            lat += step
        return graph.finish_slope(segment_ids=list(graph.segments.keys()))

    def test_cutting_sole_segment_deletes_it_and_returns_to_idle(
        self, fake_st, empty_graph, mock_dem_blue_slope
    ) -> None:
        from skiresort_planner.ui.actions import delete_direct_connection_action

        dem = mock_dem_blue_slope
        slope = self._one_seg_slope(graph=empty_graph, dem=dem)
        sm, ctx = _session(fake_st=fake_st, graph=empty_graph, dem=dem)
        sm.start_node_edit()
        sm.toggle_node_edit_node(node_id=slope.start_node_id)
        sm.toggle_node_edit_node(node_id=slope.end_node_id)

        delete_direct_connection_action()

        assert slope.id not in empty_graph.slopes, "the sole-segment slope was deleted"
        assert sm.is_idle_ready, "returns to idle after a successful cut"
        assert ctx.node_edit.node_ids == [], "the selection is cleared by the before-hook"

    def test_interior_cut_splits_slope_into_two_and_returns_to_idle(
        self, fake_st, empty_graph, mock_dem_blue_slope
    ) -> None:
        from skiresort_planner.ui.actions import delete_direct_connection_action

        dem = mock_dem_blue_slope
        slope = self._three_seg_slope(graph=empty_graph, dem=dem)
        # The two ADJACENT interior nodes flanking the middle segment.
        mid_seg = slope.segment_ids[1]
        a = empty_graph.segments[slope.segment_ids[0]].end_node_id
        b = empty_graph.segments[mid_seg].end_node_id
        sm, ctx = _session(fake_st=fake_st, graph=empty_graph, dem=dem)
        sm.start_node_edit()
        sm.toggle_node_edit_node(node_id=a)
        sm.toggle_node_edit_node(node_id=b)

        delete_direct_connection_action()

        assert len(empty_graph.slopes) == 2, "the interior cut split the slope into two"
        assert mid_seg not in empty_graph.segments, "the middle segment was cut"
        assert sm.is_idle_ready, "returns to idle after a successful cut"
        assert ctx.node_edit.node_ids == [], "the selection is cleared by the before-hook"

    def test_non_adjacent_pair_toasts_and_no_change(
        self, fake_st, empty_graph, mock_dem_blue_slope, monkeypatch
    ) -> None:
        from skiresort_planner.ui.actions import delete_direct_connection_action

        dem = mock_dem_blue_slope
        # The two ENDPOINTS of a 3-seg slope span a multi-segment gap → not adjacent.
        slope = self._three_seg_slope(graph=empty_graph, dem=dem)
        sm, ctx = _session(fake_st=fake_st, graph=empty_graph, dem=dem)
        sm.start_node_edit()
        sm.toggle_node_edit_node(node_id=slope.start_node_id)
        sm.toggle_node_edit_node(node_id=slope.end_node_id)

        import streamlit

        toasts: list[str] = []
        monkeypatch.setattr(streamlit, "toast", lambda text, *a, **k: toasts.append(text))

        delete_direct_connection_action()

        assert slope.id in empty_graph.slopes and len(empty_graph.slopes) == 1, "a non-adjacent pair is left alone"
        assert sm.is_node_edit_selecting, "stays in node edit"
        assert any("not adjacent" in t.lower() for t in toasts), "toast explains the pair isn't adjacent"

    def test_unconnected_pair_toasts_and_no_change(
        self, fake_st, empty_graph, mock_dem_blue_slope, monkeypatch
    ) -> None:
        from skiresort_planner.ui.actions import delete_direct_connection_action

        dem = mock_dem_blue_slope
        slope = self._one_seg_slope(graph=empty_graph, dem=dem)
        empty_graph.nodes["N_FAR"] = Node(id="N_FAR", location=PathPoint(lon=5.0, lat=5.0, elevation=2000.0))
        sm, ctx = _session(fake_st=fake_st, graph=empty_graph, dem=dem)
        sm.start_node_edit()
        sm.toggle_node_edit_node(node_id=slope.start_node_id)
        sm.toggle_node_edit_node(node_id="N_FAR")

        import streamlit

        toasts: list[str] = []
        monkeypatch.setattr(streamlit, "toast", lambda text, *a, **k: toasts.append(text))

        delete_direct_connection_action()

        assert slope.id in empty_graph.slopes, "nothing cut when the pair has no joining segment"
        assert sm.is_node_edit_selecting, "stays in node edit"
        assert any("not adjacent" in t.lower() for t in toasts)


class TestAddNodeOnPathAction:
    """add_node_on_path_action returns True (inserted) / False (rejected) — the bool the click
    handlers gate their state transition on — and shows an InvalidClickMessage on rejection.
    """

    def test_success_returns_true_inserts_node_and_bumps_epoch(
        self, fake_st, empty_graph, mock_dem_blue_slope, path_points_blue
    ) -> None:
        from skiresort_planner.ui.actions import add_node_on_path_action

        slope = _make_slope(graph=empty_graph, path_points=path_points_blue)
        seg_id = slope.segment_ids[0]
        # Click the geometric midpoint of the segment (interior). After finish-simplification a straight
        # run is just its two endpoints, so this projects onto the single leg — not onto an existing vertex.
        pts = empty_graph.segments[seg_id].points
        mid_lon = (pts[0].lon + pts[-1].lon) / 2
        mid_lat = (pts[0].lat + pts[-1].lat) / 2
        _session(fake_st=fake_st, graph=empty_graph, dem=mock_dem_blue_slope)
        nodes_before = len(empty_graph.nodes)
        epoch_before = fake_st.session_state["dedup_epoch"]

        result = add_node_on_path_action(segment_id=seg_id, lon=mid_lon, lat=mid_lat)

        assert result is True, "a successful insert returns True (callers gate the transition on it)"
        assert len(empty_graph.nodes) == nodes_before + 1, "one node inserted"
        assert seg_id not in empty_graph.segments, "the clicked segment was split"
        assert fake_st.session_state["dedup_epoch"] > epoch_before, "insert refreshes the map"

    def test_rejected_returns_false_changes_nothing_and_toasts(
        self, fake_st, empty_graph, mock_dem_blue_slope, path_points_blue, monkeypatch
    ) -> None:
        from skiresort_planner.ui.actions import add_node_on_path_action

        slope = _make_slope(graph=empty_graph, path_points=path_points_blue)
        seg_id = slope.segment_ids[0]
        near_end = empty_graph.segments[seg_id].points[0]  # within STEP_SIZE_M of the endpoint node
        _session(fake_st=fake_st, graph=empty_graph, dem=mock_dem_blue_slope)
        nodes_before = len(empty_graph.nodes)

        import streamlit

        toasts: list[str] = []
        monkeypatch.setattr(streamlit, "toast", lambda text, *a, **k: toasts.append(text))

        result = add_node_on_path_action(segment_id=seg_id, lon=near_end.lon, lat=near_end.lat)

        assert result is False, "a rejected insert returns False (callers must NOT transition)"
        assert len(empty_graph.nodes) == nodes_before, "nothing inserted"
        assert seg_id in empty_graph.segments, "the segment is untouched"
        assert any("add a node" in t.lower() for t in toasts), "the user is told why"


class TestUndoLastActionDispatch:
    """undo_last_action ROUTING only — the per-entity graph undo end-state is
    owned by test_resort_graph. Here we assert the action layer pops the stack,
    dispatches to a handler, and honors the empty-stack + slope-cancel guards.
    """

    def test_dispatch_pops_the_stack(self, fake_st, empty_graph) -> None:
        """A committed road leaves an undo entry; undo_last_action consumes it."""
        from skiresort_planner.ui.actions import undo_last_action

        _make_road(graph=empty_graph)
        _session(fake_st=fake_st, graph=empty_graph)
        assert len(empty_graph.undo_stack) == 1

        undo_last_action()
        assert empty_graph.undo_stack == [], "dispatch must pop the undone action"

    def test_dispatch_routes_finish_slope_needs_factory(
        self, fake_st, path_factory, mock_dem_red_slope_diagonal, empty_graph, path_points_blue
    ) -> None:
        """FINISH_SLOPE routes to the slope handler (which reads path_factory)."""
        from skiresort_planner.ui.actions import undo_last_action

        slope = _make_slope(graph=empty_graph, path_points=path_points_blue)  # ADD_SEGMENTS + FINISH_SLOPE
        _session(fake_st=fake_st, graph=empty_graph, factory=path_factory, dem=mock_dem_red_slope_diagonal)

        undo_last_action()  # undo FINISH_SLOPE via the dispatch
        assert slope.id not in empty_graph.slopes, "routed to the finish-slope undo handler"

    def test_empty_stack_is_noop(self, fake_st, empty_graph) -> None:
        """Guard: undo_last_action on an empty stack does nothing and never raises."""
        from skiresort_planner.ui.actions import undo_last_action

        _session(fake_st=fake_st, graph=empty_graph)
        undo_last_action()
        assert empty_graph.undo_stack == []

    def test_undo_in_slope_starting_cancels_slope_not_stack(self, fake_st, empty_graph, path_points_blue) -> None:
        """In slope_starting (0 segments) Undo cancels the slope, NOT an unrelated stack entry."""
        from skiresort_planner.ui.actions import undo_last_action

        _make_road(graph=empty_graph)  # an unrelated FINISH_ROAD entry sits on the stack
        sm, _ctx = _session(fake_st=fake_st, graph=empty_graph)
        sm.start_slope(lon=0.0, lat=0.0, elevation=2500.0, node_id=None)
        assert sm.is_slope_starting

        undo_last_action()
        assert sm.is_idle, "undo in slope_starting cancels the slope"
        assert len(empty_graph.undo_stack) == 1, "the unrelated road entry must NOT be consumed"

    def test_undo_in_road_starting_cancels_road_not_stack(self, fake_st, empty_graph, path_points_blue) -> None:
        """In road_starting (0 segments) Undo cancels the road, NOT an unrelated stack entry.

        Mirror of the slope guard — regression for the missing road short-circuit.
        """
        from skiresort_planner.ui.actions import undo_last_action

        _make_slope(graph=empty_graph, path_points=path_points_blue)  # unrelated ADD_SEGMENTS + FINISH_SLOPE
        stack_before = len(empty_graph.undo_stack)
        sm, _ctx = _session(fake_st=fake_st, graph=empty_graph)
        sm.start_road(node_id=None, location=path_points_blue[0])
        assert sm.is_road_starting

        undo_last_action()
        assert sm.is_idle, "undo in road_starting cancels the road"
        assert len(empty_graph.undo_stack) == stack_before, "the unrelated slope entries must NOT be consumed"


class TestCenterHelpers:
    """center_on_* set the map to the entity midpoint at a zoom that fits the entity's length."""

    def test_center_on_segment_path_slope_sets_map(self, empty_graph, path_points_blue) -> None:
        from skiresort_planner.ui.actions import center_on_segment_path

        slope = _make_slope(graph=empty_graph, path_points=path_points_blue)
        _sm, ctx = PlannerStateMachine.create(graph=empty_graph, add_ui_listener=False)
        center_on_segment_path(ctx=ctx, graph=empty_graph, path=slope)

        start_pt = empty_graph.segments[slope.segment_ids[0]].points[0]
        end_pt = empty_graph.segments[slope.segment_ids[-1]].points[-1]
        assert ctx.map.zoom == MapConfig.zoom_for_span_m(span_m=slope.get_total_length(segments=empty_graph.segments))
        assert ctx.map.lon == (start_pt.lon + end_pt.lon) / 2, "centered on the path midpoint"
        assert ctx.map.lat == (start_pt.lat + end_pt.lat) / 2
        assert ctx.map.pitch == MapConfig.VIEWING_PITCH

    def test_center_on_segment_path_road_sets_map(self, empty_graph) -> None:
        from skiresort_planner.ui.actions import center_on_segment_path

        road = _make_road(graph=empty_graph)
        _sm, ctx = PlannerStateMachine.create(graph=empty_graph, add_ui_listener=False)
        center_on_segment_path(ctx=ctx, graph=empty_graph, path=road)

        start_pt = empty_graph.segments[road.segment_ids[0]].points[0]
        end_pt = empty_graph.segments[road.segment_ids[-1]].points[-1]
        assert ctx.map.zoom == MapConfig.zoom_for_span_m(span_m=road.get_total_length(segments=empty_graph.segments))
        assert ctx.map.lon == (start_pt.lon + end_pt.lon) / 2, "centered on the road midpoint"
        assert ctx.map.lat == (start_pt.lat + end_pt.lat) / 2
        assert ctx.map.pitch == MapConfig.VIEWING_PITCH

    def test_shorter_slope_zooms_in_more_than_a_longer_one(self, empty_graph, mock_dem_blue_slope) -> None:
        # The adaptive law: a shorter build frames at a HIGHER (more-in) zoom than a longer one.
        short = _commit_straight_slope_len(graph=empty_graph, dem=mock_dem_blue_slope, n_segments=2)
        long = _commit_straight_slope_len(graph=empty_graph, dem=mock_dem_blue_slope, n_segments=8, start_lat=-1.0)
        _sm, ctx = PlannerStateMachine.create(graph=empty_graph, add_ui_listener=False)
        from skiresort_planner.ui.actions import center_on_segment_path

        center_on_segment_path(ctx=ctx, graph=empty_graph, path=short)
        short_zoom = ctx.map.zoom
        center_on_segment_path(ctx=ctx, graph=empty_graph, path=long)
        long_zoom = ctx.map.zoom
        assert short_zoom > long_zoom, "the shorter slope frames tighter (higher zoom)"

    def test_center_on_lift_sets_map(self, empty_graph, mock_dem_blue_slope) -> None:
        from skiresort_planner.ui.actions import center_on_lift

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
        _sm, ctx = PlannerStateMachine.create(graph=empty_graph, add_ui_listener=False)

        center_on_lift(ctx=ctx, graph=empty_graph, lift=lift)
        assert ctx.map.zoom == MapConfig.zoom_for_span_m(span_m=lift.get_length_m(nodes=empty_graph.nodes))
        assert ctx.map.lon == (bottom.lon + top.lon) / 2, "centered on the lift-station midpoint"
        assert ctx.map.lat == (bottom.lat + top.lat) / 2
        assert ctx.map.pitch == MapConfig.VIEWING_PITCH


class TestSelectLiftTypeAction:
    """The sidebar lift-type buttons only arm the build mode; retyping a viewed lift is a separate,
    confirm-gated action (apply_lift_retype_action), so the button never silently mutates a lift.
    """

    def test_sets_build_mode_when_not_viewing_a_lift(self, fake_st, empty_graph) -> None:
        from skiresort_planner.ui.actions import select_lift_type_action

        _sm, ctx = _session(fake_st=fake_st, graph=empty_graph)
        select_lift_type_action(lift_type="gondola")

        assert ctx.build_mode.mode == "gondola", "the next lift will be built as a gondola"

    def test_does_not_retype_a_viewed_lift(self, fake_st, empty_graph, mock_dem_blue_slope) -> None:
        from skiresort_planner.ui.actions import select_lift_type_action

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
        sm, ctx = _session(fake_st=fake_st, graph=empty_graph, dem=dem)
        sm.view_lift(lift_id=lift.id)

        select_lift_type_action(lift_type="gondola")

        assert empty_graph.lifts[lift.id].lift_type == "chairlift", "the button must NOT retype the viewed lift"
        assert ctx.build_mode.mode == "gondola", "but it arms the new type for the next build"


class TestApplyLiftRetypeAction:
    """apply_lift_retype_action is the confirm-gated retype: Lift.update_type recomputes pylons/cable."""

    def _viewed_lift(self, fake_st, empty_graph, dem):
        bottom, _ = empty_graph.get_or_create_node(
            lon=0.0,
            lat=-1000 / MapConfig.METERS_PER_DEGREE_EQUATOR,
            elevation=dem.get_elevation_or_raise(lon=0.0, lat=-1000 / MapConfig.METERS_PER_DEGREE_EQUATOR),
        )
        top, _ = empty_graph.get_or_create_node(
            lon=0.0, lat=0.0, elevation=dem.get_elevation_or_raise(lon=0.0, lat=0.0)
        )
        lift = empty_graph.add_lift(start_node_id=bottom.id, end_node_id=top.id, lift_type="chairlift", dem=dem)
        _session(fake_st=fake_st, graph=empty_graph, dem=dem)
        return lift

    def test_retypes_the_lift(self, fake_st, empty_graph, mock_dem_blue_slope) -> None:
        from skiresort_planner.ui.actions import apply_lift_retype_action

        lift = self._viewed_lift(fake_st=fake_st, empty_graph=empty_graph, dem=mock_dem_blue_slope)
        apply_lift_retype_action(lift_id=lift.id, lift_type="gondola")

        assert empty_graph.lifts[lift.id].lift_type == "gondola", "update_type re-typed the lift"

    def test_same_type_is_a_noop(self, fake_st, empty_graph, mock_dem_blue_slope) -> None:
        from skiresort_planner.ui.actions import apply_lift_retype_action

        lift = self._viewed_lift(fake_st=fake_st, empty_graph=empty_graph, dem=mock_dem_blue_slope)
        pylons_before = lift.pylons
        apply_lift_retype_action(lift_id=lift.id, lift_type="chairlift")

        assert lift.pylons is pylons_before, "re-typing to the same type must not recompute geometry"


class TestSlopeBuildingActionFlow:
    """The slope-building action entry points, driven via the fake session.

    Uses a real PathFactory + DEM so commit/recompute/finish exercise the true
    generate → commit → finish path, not hand-built stubs.
    """

    def _start_building(self, fake_st, factory, dem):
        graph = ResortGraph()
        sm, ctx = _session(fake_st=fake_st, graph=graph, factory=factory, dem=dem)
        start_elev = dem.get_elevation_or_raise(lon=0.0, lat=0.0)
        sm.start_slope(lon=0.0, lat=0.0, elevation=start_elev, node_id=None)
        ctx.selection.set(lon=0.0, lat=0.0, elevation=start_elev)
        return sm, ctx, graph

    def test_recompute_then_commit_then_finish(self, fake_st, path_factory, mock_dem_red_slope_diagonal) -> None:
        from skiresort_planner.ui.actions import commit_selected_path, finish_current_slope, recompute_paths

        dem = mock_dem_red_slope_diagonal
        sm, ctx, graph = self._start_building(fake_st=fake_st, factory=path_factory, dem=dem)

        recompute_paths()
        assert ctx.proposals.paths, "recompute must generate fan proposals"

        commit_selected_path(path_idx=0)
        assert ctx.build(kind=SegmentKind.SLOPE).segments, "commit must add a segment to the building context"

        finish_current_slope()
        assert sm.is_idle_viewing_slope
        assert len(graph.slopes) == 1

    def test_cancel_current_slope_discards(self, fake_st, path_factory, mock_dem_red_slope_diagonal) -> None:
        from skiresort_planner.ui.actions import cancel_current_slope, commit_selected_path, recompute_paths

        dem = mock_dem_red_slope_diagonal
        sm, ctx, graph = self._start_building(fake_st=fake_st, factory=path_factory, dem=dem)
        recompute_paths()
        commit_selected_path(path_idx=0)

        cancel_current_slope()
        assert sm.is_idle
        assert len(graph.slopes) == 0, "canceling discards the in-progress slope"

    def test_finish_then_undo_deletes_slope_to_idle(self, fake_st, path_factory, mock_dem_red_slope_diagonal) -> None:
        # Undoing a finish DELETES the whole slope and returns to idle_ready, so build_mode always
        # matches the idle state.
        from skiresort_planner.ui.actions import (
            commit_selected_path,
            finish_current_slope,
            recompute_paths,
            undo_last_action,
        )

        dem = mock_dem_red_slope_diagonal
        sm, ctx, graph = self._start_building(fake_st=fake_st, factory=path_factory, dem=dem)
        recompute_paths()
        commit_selected_path(path_idx=0)
        finish_current_slope()
        assert sm.is_idle_viewing_slope
        assert len(graph.slopes) == 1

        undo_last_action()  # undo FINISH_SLOPE
        assert sm.is_idle_ready, "undo of finish returns to idle_ready"
        assert len(graph.slopes) == 0, "the whole slope is deleted"
        assert len(ctx.build(kind=SegmentKind.SLOPE).segments) == 0, "build context is cleared"
        assert not graph.undo_stack, "the per-segment ADD_SEGMENTS entry is scrubbed with the segments"


class TestRoadBuildingActionFlow:
    """Roads commit through the SAME commit_selected_path as slopes (no fan, no
    connector auto-finish): a road-state commit fires the commit_road event and
    stays in road_building with a road-kind segment + per-segment undo.
    """

    def test_commit_selected_path_in_road_state_commits_segment(
        self, fake_st, path_factory, mock_dem_red_slope_diagonal, path_points_blue
    ) -> None:
        from skiresort_planner.model.path_segment import SegmentKind
        from skiresort_planner.ui.actions import commit_selected_path

        dem = mock_dem_red_slope_diagonal
        graph = ResortGraph()
        sm, ctx = _session(fake_st=fake_st, graph=graph, factory=path_factory, dem=dem)
        sm.start_road(node_id=None, location=path_points_blue[0])

        # Seed a road proposal (as handle_path_building_click would) and commit it.
        ctx.proposals.paths = [ProposedPathSegment(points=path_points_blue, is_connector=True, kind=SegmentKind.ROAD)]
        ctx.proposals.selected_idx = 0
        commit_selected_path(path_idx=0)

        assert sm.is_road_building_only, "road commit stays in road_building"
        assert len(ctx.build(kind=SegmentKind.ROAD).segments) == 1
        assert len(graph.roads) == 0, "no Road entity until Finish Road"
        assert graph.segments[ctx.build(kind=SegmentKind.ROAD).segments[-1]].kind == SegmentKind.ROAD
        assert graph.undo_stack[-1].action_type.name == "ADD_SEGMENTS", "per-segment undo recorded"

    def test_finish_then_undo_deletes_road_to_idle(
        self, fake_st, path_factory, mock_dem_red_slope_diagonal, path_points_blue
    ) -> None:
        # Undoing a road finish DELETES the whole road and returns to idle_ready (mirrors slope).
        from skiresort_planner.model.path_segment import SegmentKind
        from skiresort_planner.ui.actions import commit_selected_path, finish_current_road, undo_last_action

        dem = mock_dem_red_slope_diagonal
        graph = ResortGraph()
        sm, ctx = _session(fake_st=fake_st, graph=graph, factory=path_factory, dem=dem)
        sm.start_road(node_id=None, location=path_points_blue[0])
        ctx.proposals.paths = [ProposedPathSegment(points=path_points_blue, is_connector=True, kind=SegmentKind.ROAD)]
        ctx.proposals.selected_idx = 0
        commit_selected_path(path_idx=0)
        finish_current_road()
        assert sm.is_idle_viewing_road
        assert len(graph.roads) == 1

        undo_last_action()  # undo FINISH_ROAD
        assert sm.is_idle_ready, "undo of finish returns to idle_ready"
        assert len(graph.roads) == 0, "the whole road is deleted"
        assert len(ctx.build(kind=SegmentKind.ROAD).segments) == 0, "build context is cleared"

    def test_connector_proposal_auto_finishes_to_viewing(
        self, fake_st, path_factory, mock_dem_red_slope_diagonal, path_points_blue
    ) -> None:
        # A proposal that IS a connector (is_connector AND target_node_id set), committed from the
        # custom-path state, must NOT stay in building — commit_selected_path routes it through
        # _finish_connector → the Road entity is created and the machine lands in idle_viewing_road.
        # This is the branch the other road tests miss (they leave target_node_id empty → continue).
        from skiresort_planner.model.path_segment import SegmentKind
        from skiresort_planner.ui.actions import commit_selected_path

        dem = mock_dem_red_slope_diagonal
        graph = ResortGraph()
        sm, ctx = _session(fake_st=fake_st, graph=graph, factory=path_factory, dem=dem)
        sm.start_road(node_id=None, location=path_points_blue[0])

        # Commit one real fan segment so we're in road_building with a target node to connect to.
        first = ProposedPathSegment(points=path_points_blue, kind=SegmentKind.ROAD)
        end_ids = graph.commit_paths(paths=[first])
        seg0 = list(graph.segments.keys())[-1]
        sm.commit_road(segment_id=seg0, endpoint_node_id=end_ids[0])
        assert sm.is_road_building_only

        # Route to a custom target → road_custom_path, then commit a CONNECTOR proposal onto an
        # existing node (target_node_id set) → auto-finish.
        target_node_id = end_ids[0]
        target = graph.nodes[target_node_id]
        sm.select_custom_target(target_location=(target.lon, target.lat, target.elevation))
        assert sm.is_road_custom_path

        connector = ProposedPathSegment(
            points=path_points_blue, is_connector=True, target_node_id=target_node_id, kind=SegmentKind.ROAD
        )
        ctx.proposals.paths = [connector]
        ctx.proposals.selected_idx = 0
        commit_selected_path(path_idx=0)

        assert sm.is_idle_viewing_road, "a real connector auto-finishes to the viewing state"
        assert len(graph.roads) == 1, "the Road entity was created by the connector auto-finish"
        assert ctx.viewing.road_id is not None, "the finished road is being viewed"


class TestDeferredProcessing:
    """Deferred-action processors read/clear ctx.pending flags and act on them."""

    def test_process_path_generation_noop_when_not_pending(
        self, fake_st, path_factory, mock_dem_red_slope_diagonal
    ) -> None:
        from skiresort_planner.ui.actions import process_path_generation_pending

        _sm, ctx = _session(fake_st=fake_st, graph=ResortGraph(), factory=path_factory, dem=mock_dem_red_slope_diagonal)
        ctx.pending.fan_generation.discard(SegmentKind.SLOPE)
        assert process_path_generation_pending() is False

    def test_process_path_generation_builds_fan_when_pending(
        self, fake_st, path_factory, mock_dem_red_slope_diagonal
    ) -> None:
        from skiresort_planner.ui.actions import process_path_generation_pending

        dem = mock_dem_red_slope_diagonal
        graph = ResortGraph()
        sm, ctx = _session(fake_st=fake_st, graph=graph, factory=path_factory, dem=dem)
        start_elev = dem.get_elevation_or_raise(lon=0.0, lat=0.0)
        sm.start_slope(lon=0.0, lat=0.0, elevation=start_elev, node_id=None)
        ctx.selection.set(lon=0.0, lat=0.0, elevation=start_elev)
        ctx.pending.fan_generation.add(SegmentKind.SLOPE)

        assert process_path_generation_pending() is True
        assert SegmentKind.SLOPE not in ctx.pending.fan_generation, "flag cleared after processing"
        assert ctx.proposals.paths, "fan proposals generated for the building state"

    def test_process_custom_connect_noop_when_not_pending(
        self, fake_st, path_factory, mock_dem_red_slope_diagonal
    ) -> None:
        from skiresort_planner.ui.actions import process_custom_connect_pending

        _sm, ctx = _session(fake_st=fake_st, graph=ResortGraph(), factory=path_factory, dem=mock_dem_red_slope_diagonal)
        ctx.pending.custom_connect = False
        assert process_custom_connect_pending() is False

    def test_custom_connect_orders_shortest_first_straight_last(
        self, fake_st, monkeypatch, path_factory, mock_dem_red_slope_diagonal
    ) -> None:
        # Custom-connect sorts serpentine proposals SHORTEST→longest, appends the straight line LAST,
        # and pre-selects the shortest (index 0) — NOT the gradient-closest (that's the fan's rule).
        from skiresort_planner.ui import actions

        dem = mock_dem_red_slope_diagonal
        sm, ctx = _session(fake_st=fake_st, graph=ResortGraph(), factory=path_factory, dem=dem)
        start_elev = dem.get_elevation_or_raise(lon=0.0, lat=0.0)
        sm.start_slope(lon=0.0, lat=0.0, elevation=start_elev, node_id=None)
        ctx.selection.set(lon=0.0, lat=0.0, elevation=start_elev)

        # A run of N points stepping south by `step` metres each — length scales with (N-1)*step.
        def _seg(n: int, step: float) -> ProposedPathSegment:
            pts = [
                PathPoint(
                    lon=0.0,
                    lat=-(i * step) / MapConfig.METERS_PER_DEGREE_EQUATOR,
                    elevation=start_elev - i * step * 0.1,
                )
                for i in range(n)
            ]
            return ProposedPathSegment(points=pts, is_connector=False)

        long_route, short_route = _seg(6, 100.0), _seg(3, 100.0)  # ~500m vs ~200m, out of order
        straight = _seg(2, 150.0)  # the straight line, appended last regardless of length
        monkeypatch.setattr(path_factory, "generate_manual_paths", lambda **_: [long_route, short_route])
        monkeypatch.setattr(path_factory, "straight_line", lambda **_: straight)

        ctx.custom_connect.target_location = (
            0.0,
            -500 / MapConfig.METERS_PER_DEGREE_EQUATOR,
            dem.get_elevation_or_raise(lon=0.0, lat=-500 / MapConfig.METERS_PER_DEGREE_EQUATOR),
        )
        ctx.pending.gradient_target = 99.0  # a stale fan target must be IGNORED by custom-connect
        ctx.pending.custom_connect = True
        assert actions.process_custom_connect_pending() is True

        paths = ctx.proposals.paths
        lengths = [p.length_m for p in paths]
        assert lengths[:2] == sorted(lengths[:2]), "serpentine proposals ordered shortest-first"
        assert paths[0] is short_route, "shortest route is first"
        assert paths[-1] is straight, "straight line appended last"
        assert ctx.proposals.selected_idx == 0, "shortest route pre-selected (not gradient-closest)"
        assert ctx.pending.gradient_target is None, "stale fan gradient target consumed/ignored"


class TestGradientPreselection:
    """_preselect_by_rule — the fan passes the closest-gradient rule (grade continuity across
    committed segments); custom-connect passes a shortest-first (index 0) rule. Both always
    consume the one-shot gradient_target.
    """

    def _paths(self, *slopes: float) -> "list[ProposedPathSegment]":
        # Two-point segments whose avg_slope_pct is the given grade (100m run).
        out = []
        for s in slopes:
            pts = [PathPoint(lon=0.0, lat=0.0, elevation=1000.0), PathPoint(lon=0.001, lat=0.0, elevation=1000.0 - s)]
            out.append(ProposedPathSegment(points=pts, kind=SegmentKind.SLOPE))
        return out

    def test_gradient_rule_preselects_closest_and_consumes_target(self, fake_st, mock_dem_blue_slope) -> None:
        from skiresort_planner.ui.actions import _closest_gradient_rule, _preselect_by_rule

        _sm, ctx = _session(fake_st=fake_st, graph=ResortGraph(), dem=mock_dem_blue_slope)
        paths = self._paths(5.0, 18.0, 30.0)
        ctx.pending.gradient_target = 17.0  # closest to the 18% path (index 1)
        _preselect_by_rule(ctx=ctx, paths=paths, rule=_closest_gradient_rule(ctx=ctx))
        assert ctx.proposals.selected_idx == 1, "pre-selects the proposal nearest the last committed grade"
        assert ctx.pending.gradient_target is None, "one-shot: the target is consumed"

    def test_gradient_rule_defaults_to_first_without_target(self, fake_st, mock_dem_blue_slope) -> None:
        from skiresort_planner.ui.actions import _closest_gradient_rule, _preselect_by_rule

        _sm, ctx = _session(fake_st=fake_st, graph=ResortGraph(), dem=mock_dem_blue_slope)
        ctx.pending.gradient_target = None
        _preselect_by_rule(ctx=ctx, paths=self._paths(5.0, 18.0), rule=_closest_gradient_rule(ctx=ctx))
        assert ctx.proposals.selected_idx == 0, "no target → first proposal"

    def test_shortest_rule_selects_index_zero(self, fake_st, mock_dem_blue_slope) -> None:
        from skiresort_planner.ui.actions import _preselect_by_rule, _shortest_rule

        _sm, ctx = _session(fake_st=fake_st, graph=ResortGraph(), dem=mock_dem_blue_slope)
        ctx.pending.gradient_target = 17.0  # a stale fan target must still be consumed
        _preselect_by_rule(ctx=ctx, paths=self._paths(5.0, 18.0), rule=_shortest_rule)
        assert ctx.proposals.selected_idx == 0, "custom-connect shortest-first → index 0"
        assert ctx.pending.gradient_target is None, "stale fan target consumed even for the shortest rule"

    def test_none_when_no_paths(self, fake_st, mock_dem_blue_slope) -> None:
        from skiresort_planner.ui.actions import _closest_gradient_rule, _preselect_by_rule

        _sm, ctx = _session(fake_st=fake_st, graph=ResortGraph(), dem=mock_dem_blue_slope)
        ctx.pending.gradient_target = 17.0
        _preselect_by_rule(ctx=ctx, paths=[], rule=_closest_gradient_rule(ctx=ctx))
        assert ctx.proposals.selected_idx is None, "empty proposals → no selection"


class TestOSMImport:
    """Click-to-place import: start_import stores the box center; confirm_import_action flags the
    deferred fetch + returns to idle; process_osm_import_pending runs it (mocked network) as one
    undoable batch centered on the placed box; undo removes the batch; re-import dedups.
    """

    def test_start_import_stores_center_and_confirm_flags_deferred(self, fake_st, mock_dem_blue_slope) -> None:
        from skiresort_planner.constants import OSMImportMode
        from skiresort_planner.ui.actions import confirm_import_action

        sm, ctx = _session(fake_st=fake_st, graph=ResortGraph(), dem=mock_dem_blue_slope)
        sm.start_import(lon=0.1, lat=0.3)  # first map click places the box center
        assert sm.is_import_selecting
        assert ctx.pending.osm_import_center_lon == 0.1 and ctx.pending.osm_import_center_lat == 0.3

        confirm_import_action(mode=OSMImportMode.LIFTS_AND_SLOPES)  # center-dot click / import button
        assert ctx.pending.osm_import_mode == OSMImportMode.LIFTS_AND_SLOPES
        assert sm.is_idle_ready, "confirm returns to idle so the deferred fetch runs under the spinner"

    def test_placed_center_reaches_fetch_as_bbox(self, fake_st, mock_dem_blue_slope, monkeypatch) -> None:
        """The placed box center + half-width must arrive at the importer as a square bbox around it."""
        from skiresort_planner.constants import OSMImportMode
        from skiresort_planner.generators.osm_importer import ImportResult
        from skiresort_planner.generators.osm_lift_importer import LiftOnlyImporter
        from skiresort_planner.ui import actions
        from skiresort_planner.ui.actions import confirm_import_action

        sm, ctx = _session(fake_st=fake_st, graph=ResortGraph(), dem=mock_dem_blue_slope)
        ctx.map.lat, ctx.map.lon = 0.9, 0.9  # deliberately NOT the placed center — must be ignored

        seen: dict[str, tuple[float, float, float, float]] = {}

        # The base __init__ stores self.bbox; capture it in _assemble (no network, no __init__ patch).
        def _record_assemble(self: LiftOnlyImporter, elements: list[object], on_progress) -> ImportResult:
            seen["bbox"] = self.bbox
            return ImportResult()

        monkeypatch.setattr("skiresort_planner.generators.osm_importer.BaseOSMImporter.fetch", lambda self: [])
        monkeypatch.setattr(
            "skiresort_planner.generators.osm_lift_importer.LiftOnlyImporter._assemble", _record_assemble
        )

        sm.start_import(lon=0.1, lat=0.3)  # placed center
        ctx.pending.osm_import_half_width_km = 3.5
        confirm_import_action(mode=OSMImportMode.LIFTS_ONLY)
        actions.process_osm_import_pending(report=_noop_report)

        min_lon, min_lat, max_lon, max_lat = seen["bbox"]
        assert (min_lon + max_lon) / 2 == 0.1 and (min_lat + max_lat) / 2 == 0.3, "box centered on the PLACED center"
        assert max_lat - min_lat > 0 and max_lon - min_lon > 0, "3.5 km half-width → a real square box"

    def test_process_without_placed_center_raises(self, fake_st, mock_dem_blue_slope) -> None:
        """A pending import with no placed center is a bug — no silent map-center fallback."""
        from skiresort_planner.constants import OSMImportMode
        from skiresort_planner.ui import actions

        _sm, ctx = _session(fake_st=fake_st, graph=ResortGraph(), dem=mock_dem_blue_slope)
        ctx.pending.osm_import_mode = OSMImportMode.LIFTS_AND_SLOPES  # flagged, but no center placed
        with pytest.raises(RuntimeError, match="no placed center"):
            actions.process_osm_import_pending(report=_noop_report)

    def test_process_import_adds_entities_and_bumps_map(self, fake_st, mock_dem_blue_slope, monkeypatch) -> None:
        from skiresort_planner.constants import OSMImportMode
        from skiresort_planner.ui import actions

        dem = mock_dem_blue_slope
        graph = ResortGraph()
        _sm, ctx = _session(fake_st=fake_st, graph=graph, dem=dem)
        ctx.pending.osm_import_mode = OSMImportMode.LIFTS_AND_SLOPES
        ctx.pending.osm_import_center_lon = 0.0  # inside MockDEM bounds (-1..1)
        ctx.pending.osm_import_center_lat = 0.0

        result = _fake_import_result(dem)

        # Mock the importer so no network happens: fetch returns nothing, _assemble returns our result.
        # Mock run() so no network/plot happens: the importer just returns our prepared result.
        monkeypatch.setattr(
            "skiresort_planner.generators.osm_graph_builder.GraphImporter.run",
            lambda self, *, on_progress, dump_dir=None: result,
        )
        epoch_before = fake_st.session_state["dedup_epoch"]

        handled = actions.process_osm_import_pending(report=_noop_report)

        assert handled is True
        assert len(graph.slopes) == 1 and len(graph.lifts) == 1
        assert len(graph.undo_stack) == 1, "import is one undoable batch"
        assert fake_st.session_state["dedup_epoch"] > epoch_before, "import redraws new geometry (no recenter)"
        assert ctx.pending.osm_import_mode is None, "mode consumed"
        assert ctx.pending.osm_import_center_lon is None, "placed center consumed"

    def test_process_import_network_error_propagates_and_imports_nothing(
        self, fake_st, mock_dem_blue_slope, monkeypatch
    ) -> None:
        # The processor lets failures propagate to run_pending_load, which shows its warning toast.
        # Here we assert it raises and imports nothing.
        import pytest

        from skiresort_planner.constants import OSMImportMode
        from skiresort_planner.ui import actions

        graph = ResortGraph()
        _sm, ctx = _session(fake_st=fake_st, graph=graph, dem=mock_dem_blue_slope)
        ctx.pending.osm_import_mode = OSMImportMode.LIFTS_AND_SLOPES
        ctx.pending.osm_import_center_lon = 0.0
        ctx.pending.osm_import_center_lat = 0.0

        def boom(self):
            raise RuntimeError("overpass down")

        monkeypatch.setattr("skiresort_planner.generators.osm_importer.BaseOSMImporter.fetch", boom)

        with pytest.raises(RuntimeError, match="overpass down"):
            actions.process_osm_import_pending(report=_noop_report)

        assert len(graph.slopes) == 0 and len(graph.lifts) == 0
        assert len(graph.undo_stack) == 0, "a network error imports nothing"

    def test_undo_last_action_reverts_whole_import(self, fake_st, mock_dem_blue_slope, monkeypatch) -> None:
        """The headline promise: one Undo (via the UI dispatcher) removes the entire import.

        Exercises undo_last_action() — the dispatcher a direct graph.undo_last() call bypasses.
        """
        from skiresort_planner.constants import OSMImportMode
        from skiresort_planner.ui import actions
        from skiresort_planner.ui.actions import undo_last_action

        dem = mock_dem_blue_slope
        graph = ResortGraph()
        _sm, ctx = _session(fake_st=fake_st, graph=graph, dem=dem)
        ctx.pending.osm_import_mode = OSMImportMode.LIFTS_AND_SLOPES
        ctx.pending.osm_import_center_lon = 0.0
        ctx.pending.osm_import_center_lat = 0.0

        result = _fake_import_result(dem)
        # Mock run() so no network/plot happens: the importer just returns our prepared result.
        monkeypatch.setattr(
            "skiresort_planner.generators.osm_graph_builder.GraphImporter.run",
            lambda self, *, on_progress, dump_dir=None: result,
        )

        actions.process_osm_import_pending(report=_noop_report)
        assert len(graph.slopes) == 1 and len(graph.lifts) == 1

        undo_last_action()  # dispatch IMPORT_OSM — must not raise, must wipe the batch

        assert len(graph.slopes) == 0 and len(graph.lifts) == 0
        assert len(graph.segments) == 0 and len(graph.nodes) == 0
        assert len(graph.undo_stack) == 0

    def test_reimport_same_area_adds_nothing(self, fake_st, mock_dem_blue_slope, monkeypatch) -> None:
        """Importing the same area twice adds entities once, then dedups the rest."""
        from skiresort_planner.constants import OSMImportMode
        from skiresort_planner.ui import actions

        dem = mock_dem_blue_slope
        graph = ResortGraph()
        _sm, ctx = _session(fake_st=fake_st, graph=graph, dem=dem)

        result = _fake_import_result(dem)
        # Mock run() so no network/plot happens: the importer just returns our prepared result.
        monkeypatch.setattr(
            "skiresort_planner.generators.osm_graph_builder.GraphImporter.run",
            lambda self, *, on_progress, dump_dir=None: result,
        )

        def _flag_import() -> None:
            ctx.pending.osm_import_mode = OSMImportMode.LIFTS_AND_SLOPES
            ctx.pending.osm_import_center_lon = 0.0
            ctx.pending.osm_import_center_lat = 0.0

        _flag_import()
        actions.process_osm_import_pending(report=_noop_report)
        assert len(graph.slopes) == 1 and len(graph.lifts) == 1

        _flag_import()
        actions.process_osm_import_pending(report=_noop_report)  # same area again
        assert len(graph.slopes) == 1 and len(graph.lifts) == 1, "no duplicates on re-import"


class TestSegmentOrigin:
    """_segment_origin resolves the point a fan radiates from.

    No origin node is materialised before commit, so start_node_id is either a LIVE node (existing
    junction / committed endpoint) or None (fresh terrain origin, carried as start_location). A
    non-None id must therefore resolve strictly — a dangling id is a bug and raises (fail-fast).
    """

    def test_falls_back_to_start_location_when_no_origin_node(self, empty_graph) -> None:
        from skiresort_planner.ui.actions import resolve_build_origin
        from skiresort_planner.ui.context import SegmentBuildContext

        # Fresh terrain origin: no node yet, carried as start_location.
        build = SegmentBuildContext(start_location=PathPoint(lon=8.019, lat=46.584, elevation=3065.0))
        lon, lat, elevation, start_node_id = resolve_build_origin(build=build, graph=empty_graph)

        assert (lon, lat, elevation) == (8.019, 46.584, 3065.0), "routes from the pending origin location"
        assert start_node_id is None, "no node yet — commit_paths mints it"

    def test_stale_origin_node_falls_back_to_start_location(self, empty_graph) -> None:
        from skiresort_planner.ui.actions import resolve_build_origin
        from skiresort_planner.ui.context import SegmentBuildContext

        # The origin node was cleaned when the last segment was undone, but start_location survives.
        # The dangling id is ignored; the location is used.
        build = SegmentBuildContext(
            start_node_id="N999",  # cleaned as isolated
            start_location=PathPoint(lon=8.019, lat=46.584, elevation=3065.0),
        )
        lon, lat, elevation, start_node_id = resolve_build_origin(build=build, graph=empty_graph)
        assert (lon, lat, elevation) == (8.019, 46.584, 3065.0), "falls back to the origin location"
        assert start_node_id is None, "the stale id is not reused"

    def test_raises_when_no_origin_at_all(self, empty_graph) -> None:
        import pytest

        from skiresort_planner.ui.actions import resolve_build_origin
        from skiresort_planner.ui.context import SegmentBuildContext

        # No endpoint, a dangling origin id, and NO location fallback → genuine programming error.
        build = SegmentBuildContext(start_node_id="N999")
        with pytest.raises(ValueError, match="no start node or location"):
            resolve_build_origin(build=build, graph=empty_graph)

    def test_endpoint_must_be_live(self, empty_graph) -> None:
        import pytest

        from skiresort_planner.ui.actions import resolve_build_origin
        from skiresort_planner.ui.context import SegmentBuildContext

        # A committed endpoint must exist — a missing one is an invariant violation (strict []).
        build = SegmentBuildContext(endpoints=["N999"])
        with pytest.raises(KeyError):
            resolve_build_origin(build=build, graph=empty_graph)

    def test_uses_node_when_present(self, empty_graph, path_points_blue) -> None:
        from skiresort_planner.ui.actions import resolve_build_origin
        from skiresort_planner.ui.context import SegmentBuildContext

        node, _ = empty_graph.get_or_create_node(lon=8.02, lat=46.58, elevation=3000.0)
        build = SegmentBuildContext(start_node_id=node.id)
        lon, lat, elevation, start_node_id = resolve_build_origin(build=build, graph=empty_graph)

        assert (lon, lat, elevation) == (node.lon, node.lat, node.elevation)
        assert start_node_id == node.id, "an existing origin node is returned for reuse on commit"


class TestUndoToZeroAfterFinish:
    """Undo semantics: undoing a finish deletes the whole entity in one step; undoing segments during
    a build stays in place (re-arming) until the last one, which cancels the build to idle_ready.
    """

    def test_undo_finish_deletes_everything_cleanly(
        self, fake_st, empty_graph, path_factory, mock_dem_red_slope_diagonal
    ) -> None:
        # Build a 2-segment road, finish, undo the finish → the whole road + its per-segment undo
        # entries are gone in one step, landing idle_ready.
        from skiresort_planner.ui.actions import undo_last_action

        dem = mock_dem_red_slope_diagonal
        sm, ctx = _session(fake_st=fake_st, graph=empty_graph, factory=path_factory, dem=dem)
        ctx.build_mode.mode = SegmentKind.ROAD.value

        # Start a road from fresh terrain and commit two segments (undo actions recorded).
        sm.start_road(node_id=None, location=PathPoint(lon=0.0, lat=0.0, elevation=2000.0))
        for i in range(1, 3):
            pts = [
                PathPoint(
                    lon=(i - 1) * 300 / MapConfig.METERS_PER_DEGREE_EQUATOR, lat=0.0, elevation=2000.0 - (i - 1) * 10
                ),
                PathPoint(lon=i * 300 / MapConfig.METERS_PER_DEGREE_EQUATOR, lat=0.0, elevation=2000.0 - i * 10),
            ]
            endpoint_ids = empty_graph.commit_paths(paths=[ProposedPathSegment(points=pts, kind=SegmentKind.ROAD)])
            seg = list(empty_graph.segments.keys())[-1]
            sm.commit_road(segment_id=seg, endpoint_node_id=endpoint_ids[0])

        road = empty_graph.finish_road(segment_ids=ctx.build(kind=SegmentKind.ROAD).segments)
        sm.finish_road(entity_id=road.id)
        assert sm.is_idle_viewing_road

        undo_last_action()  # undo FINISH_ROAD → deletes the whole road
        assert sm.is_idle_ready
        assert len(empty_graph.roads) == 0
        assert len(ctx.build(kind=SegmentKind.ROAD).segments) == 0
        assert not empty_graph.undo_stack, "the finish-undo scrubs the per-segment ADD_SEGMENTS entries too"

    def test_undo_segments_stays_building_then_cancels_to_idle(
        self, fake_st, empty_graph, path_factory, mock_dem_red_slope_diagonal
    ) -> None:
        # Undoing a committed segment (no finish) stays in slope_building and re-arms the fan; undoing
        # the last remaining segment cancels the build to idle_ready.
        from skiresort_planner.ui.actions import process_path_generation_pending, undo_last_action

        dem = mock_dem_red_slope_diagonal
        sm, ctx = _session(fake_st=fake_st, graph=empty_graph, factory=path_factory, dem=dem)
        ctx.build_mode.mode = SegmentKind.SLOPE.value

        # Build a slope from fresh terrain, committing two segments (ADD_SEGMENTS undo actions).
        sm.start_slope(lon=0.0, lat=0.0, elevation=2000.0, node_id=None)
        for i in range(1, 3):
            pts = [
                PathPoint(
                    lon=(i - 1) * 300 / MapConfig.METERS_PER_DEGREE_EQUATOR, lat=0.0, elevation=2000.0 - (i - 1) * 10
                ),
                PathPoint(lon=i * 300 / MapConfig.METERS_PER_DEGREE_EQUATOR, lat=0.0, elevation=2000.0 - i * 10),
            ]
            endpoint_ids = empty_graph.commit_paths(paths=[ProposedPathSegment(points=pts, target_difficulty="blue")])
            seg = list(empty_graph.segments.keys())[-1]
            sm.commit_path(segment_id=seg, endpoint_node_id=endpoint_ids[0])
        assert sm.is_slope_building_only and len(ctx.build(kind=SegmentKind.SLOPE).segments) == 2

        undo_last_action()  # peel one segment → stay building, fan re-armed
        assert sm.is_slope_building_only, "one segment remains → stay in slope_building"
        assert len(ctx.build(kind=SegmentKind.SLOPE).segments) == 1
        assert SegmentKind.SLOPE in ctx.pending.fan_generation, "the fan is re-armed from the moved endpoint"
        process_path_generation_pending()  # the deferred fan pass must not raise
        assert ctx.proposals.paths, "the fan regenerates from the new endpoint"

        undo_last_action()  # peel the last segment → cancel to idle
        assert sm.is_idle_ready, "undoing the last segment cancels the build to idle_ready"
        assert len(ctx.build(kind=SegmentKind.SLOPE).segments) == 0


class TestMapEpochs:
    """camera_epoch (remount → recenter) moves ONLY on finish; dedup_epoch (click-id) moves on
    proposal regeneration. Neither commit nor cancel nor start recenters (keeps the user's pan).
    """

    def _road_building(self, fake_st, empty_graph, path_factory, dem):
        sm, ctx = _session(fake_st=fake_st, graph=empty_graph, factory=path_factory, dem=dem)
        ctx.build_mode.mode = SegmentKind.ROAD.value
        sm.start_road(node_id=None, location=PathPoint(lon=0.0, lat=0.0, elevation=2000.0))
        return sm, ctx

    def test_commit_does_not_recenter(self, fake_st, empty_graph, path_factory, mock_dem_red_slope_diagonal) -> None:
        from skiresort_planner.ui.actions import commit_selected_path

        dem = mock_dem_red_slope_diagonal
        sm, ctx = self._road_building(fake_st, empty_graph, path_factory, dem)
        pts = [
            PathPoint(lon=0.0, lat=0.0, elevation=2000.0),
            PathPoint(lon=300 / MapConfig.METERS_PER_DEGREE_EQUATOR, lat=0.0, elevation=1990.0),
        ]
        ctx.proposals.paths = [ProposedPathSegment(points=pts, kind=SegmentKind.ROAD)]
        ctx.proposals.selected_idx = 0
        camera_before = fake_st.session_state["camera_epoch"]

        commit_selected_path(path_idx=0)

        assert fake_st.session_state["camera_epoch"] == camera_before, "commit must NOT recenter"

    def test_finish_recenters(self, fake_st, empty_graph, path_factory, mock_dem_red_slope_diagonal) -> None:
        from skiresort_planner.ui.actions import finish_current_build

        dem = mock_dem_red_slope_diagonal
        sm, ctx = self._road_building(fake_st, empty_graph, path_factory, dem)
        pts = [
            PathPoint(lon=0.0, lat=0.0, elevation=2000.0),
            PathPoint(lon=300 / MapConfig.METERS_PER_DEGREE_EQUATOR, lat=0.0, elevation=1990.0),
        ]
        endpoint_ids = empty_graph.commit_paths(paths=[ProposedPathSegment(points=pts, kind=SegmentKind.ROAD)])
        sm.commit_road(segment_id=list(empty_graph.segments.keys())[-1], endpoint_node_id=endpoint_ids[0])
        camera_before = fake_st.session_state["camera_epoch"]
        map_before = (ctx.map.lon, ctx.map.lat)

        finish_current_build(kind=SegmentKind.ROAD)

        assert sm.is_idle_viewing_road
        # Reframes on the finished entity IN PLACE: ctx.map moves, but camera_epoch is NOT bumped
        # (bumping remounts the deck.gl iframe → the ~0.5s gray-out). See tests_workflow/test_map_reframe.py.
        assert (ctx.map.lon, ctx.map.lat) != map_before, "finish recenters on the entity (view moved)"
        assert fake_st.session_state["camera_epoch"] == camera_before, "in-place reframe: no remount bump"

    def test_cancel_does_not_recenter(self, fake_st, empty_graph, path_factory, mock_dem_red_slope_diagonal) -> None:
        from skiresort_planner.ui.actions import cancel_current_build

        dem = mock_dem_red_slope_diagonal
        sm, ctx = self._road_building(fake_st, empty_graph, path_factory, dem)
        pts = [
            PathPoint(lon=0.0, lat=0.0, elevation=2000.0),
            PathPoint(lon=300 / MapConfig.METERS_PER_DEGREE_EQUATOR, lat=0.0, elevation=1990.0),
        ]
        endpoint_ids = empty_graph.commit_paths(paths=[ProposedPathSegment(points=pts, kind=SegmentKind.ROAD)])
        sm.commit_road(segment_id=list(empty_graph.segments.keys())[-1], endpoint_node_id=endpoint_ids[0])
        camera_before = fake_st.session_state["camera_epoch"]

        cancel_current_build(kind=SegmentKind.ROAD)

        assert fake_st.session_state["camera_epoch"] == camera_before, "cancel must NOT recenter"


class TestUndoDispatchMatrix:
    """Exhaustive undo dispatch: every ActionType, driven through the real undo_last_action() (which
    runs graph.undo_last() plus the _UNDO_SIDE_EFFECTS state-machine side-effect inside
    `with sm.undo_running()`), asserting the resulting SM state and graph effect — never a
    TransitionNotAllowed. A completeness guard asserts every ActionType has a scenario, so a new
    action type fails this suite until it is covered here.
    """

    def _lift_id(self, graph: ResortGraph, dem: MockDEMService) -> str:
        """Add a bottom→top lift on the graph; return its id (records ADD_LIFT)."""
        bottom = _node_at(dem=dem, node_id="LB", lon=0.0, lat=-1000 / MapConfig.METERS_PER_DEGREE_EQUATOR)
        top = _node_at(dem=dem, node_id="LT", lon=0.0, lat=-500 / MapConfig.METERS_PER_DEGREE_EQUATOR)
        graph.nodes[bottom.id] = bottom
        graph.nodes[top.id] = top
        return graph.add_lift(start_node_id=bottom.id, end_node_id=top.id, lift_type="chairlift", dem=dem).id

    def _two_seg_building(self, fake_st, factory, dem):
        """Drive the SM to slope_building with 2 committed segments; return (sm, ctx, graph)."""
        graph = ResortGraph()
        sm, ctx = _session(fake_st=fake_st, graph=graph, factory=factory, dem=dem)
        ctx.build_mode.mode = SegmentKind.SLOPE.value
        sm.start_slope(lon=0.0, lat=0.0, elevation=2000.0, node_id=None)
        for i in range(1, 3):
            pts = [
                PathPoint(
                    lon=(i - 1) * 300 / MapConfig.METERS_PER_DEGREE_EQUATOR, lat=0.0, elevation=2000.0 - (i - 1) * 10
                ),
                PathPoint(lon=i * 300 / MapConfig.METERS_PER_DEGREE_EQUATOR, lat=0.0, elevation=2000.0 - i * 10),
            ]
            endpoint_ids = graph.commit_paths(paths=[ProposedPathSegment(points=pts, target_difficulty="blue")])
            sm.commit_path(segment_id=list(graph.segments.keys())[-1], endpoint_node_id=endpoint_ids[0])
        return sm, ctx, graph

    # One scenario builder per case; each seeds pre-undo state and returns (sm, ctx, graph, check).

    def _scn_add_segments_two_remain(self, fake_st, factory, dem):
        sm, ctx, graph = self._two_seg_building(fake_st=fake_st, factory=factory, dem=dem)

        def check() -> None:
            assert sm.is_slope_building_only, "one segment remains → stay in slope_building"
            assert len(ctx.build(kind=SegmentKind.SLOPE).segments) == 1
            assert SegmentKind.SLOPE in ctx.pending.fan_generation, "fan re-armed from the moved endpoint"

        return sm, ctx, graph, check

    def _scn_add_segments_last_seg(self, fake_st, factory, dem):
        sm, ctx, graph = self._two_seg_building(fake_st=fake_st, factory=factory, dem=dem)
        from skiresort_planner.ui.actions import undo_last_action

        undo_last_action()  # peel one → still building with 1 segment

        def check() -> None:
            assert sm.is_idle_ready, "undoing the last segment cancels the build to idle_ready"
            assert len(ctx.build(kind=SegmentKind.SLOPE).segments) == 0

        return sm, ctx, graph, check

    def _scn_add_segments_in_custom_path(self, fake_st, factory, dem):
        sm, ctx, graph = self._two_seg_building(fake_st=fake_st, factory=factory, dem=dem)
        target = (300 / MapConfig.METERS_PER_DEGREE_EQUATOR, -300 / MapConfig.METERS_PER_DEGREE_EQUATOR, 1980.0)
        sm.select_custom_target(target_location=target)
        assert sm.is_slope_custom_path

        def check() -> None:
            assert sm.is_slope_custom_path, "undo of a committed segment while targeting stays in custom-path"
            assert ctx.pending.custom_connect is True, "custom routes regenerate from the re-anchored origin"

        return sm, ctx, graph, check

    def _scn_finish_slope(self, fake_st, factory, dem):
        sm, ctx, graph, slope, _seg = _build_finish_slope(fake_st=fake_st, factory=factory, dem=dem)

        def check() -> None:
            assert sm.is_idle_ready, "undo of finish → idle_ready"
            assert slope.id not in graph.slopes, "the whole slope is deleted"
            assert graph.undo_stack == [], "the per-segment ADD_SEGMENTS entry is scrubbed with the segments"

        return sm, ctx, graph, check

    def _scn_finish_road(self, fake_st, factory, dem):
        graph = ResortGraph()
        sm, ctx = _session(fake_st=fake_st, graph=graph, factory=factory, dem=dem)
        sm.start_road(node_id=None, location=PathPoint(lon=0.0, lat=0.0, elevation=2000.0))
        pts = [
            PathPoint(lon=0.0, lat=0.0, elevation=2000.0),
            PathPoint(lon=300 / MapConfig.METERS_PER_DEGREE_EQUATOR, lat=0.0, elevation=1990.0),
        ]
        endpoint_ids = graph.commit_paths(paths=[ProposedPathSegment(points=pts, kind=SegmentKind.ROAD)])
        sm.commit_road(segment_id=list(graph.segments.keys())[-1], endpoint_node_id=endpoint_ids[0])
        road = graph.finish_road(segment_ids=ctx.build(kind=SegmentKind.ROAD).segments)
        sm.finish_road(entity_id=road.id)

        def check() -> None:
            assert sm.is_idle_ready, "undo of finish → idle_ready"
            assert road.id not in graph.roads, "the whole road is deleted"

        return sm, ctx, graph, check

    def _scn_add_lift_while_placing(self, fake_st, factory, dem):
        graph = ResortGraph()
        sm, ctx = _session(fake_st=fake_st, graph=graph, factory=factory, dem=dem)
        lift_id = self._lift_id(graph=graph, dem=dem)
        sm.start_lift(node_id=None, location=None)  # in lift_placing when the lift-add is undone

        def check() -> None:
            assert sm.is_idle_ready, "undoing the lift while placing forces idle"
            assert lift_id not in graph.lifts

        return sm, ctx, graph, check

    def _scn_add_lift_while_viewing(self, fake_st, factory, dem):
        graph = ResortGraph()
        sm, ctx = _session(fake_st=fake_st, graph=graph, factory=factory, dem=dem)
        lift_id = self._lift_id(graph=graph, dem=dem)
        sm.view_lift(lift_id=lift_id)

        def check() -> None:
            assert sm.is_idle_ready, "undoing the viewed lift forces idle"
            assert lift_id not in graph.lifts

        return sm, ctx, graph, check

    def _scn_add_lift_elsewhere(self, fake_st, factory, dem):
        graph = ResortGraph()
        sm, ctx = _session(fake_st=fake_st, graph=graph, factory=factory, dem=dem)
        lift_id = self._lift_id(graph=graph, dem=dem)  # stay in idle_ready

        def check() -> None:
            assert sm.is_idle_ready, "undo from idle stays idle (redraw only)"
            assert lift_id not in graph.lifts

        return sm, ctx, graph, check

    def _scn_delete_slope(self, fake_st, factory, dem):
        graph = ResortGraph()
        sm, ctx = _session(fake_st=fake_st, graph=graph, factory=factory, dem=dem)
        slope = _make_slope(graph=graph, path_points=_straight_points(dem=dem))
        graph.delete_slope(slope_id=slope.id)  # top of stack = DELETE_SLOPE

        def check() -> None:
            assert sm.is_idle_ready
            assert slope.id in graph.slopes, "undo of delete restores the slope"

        return sm, ctx, graph, check

    def _scn_delete_road(self, fake_st, factory, dem):
        graph = ResortGraph()
        sm, ctx = _session(fake_st=fake_st, graph=graph, factory=factory, dem=dem)
        road = _make_road(graph=graph)
        graph.delete_road(road_id=road.id)

        def check() -> None:
            assert sm.is_idle_ready
            assert road.id in graph.roads, "undo of delete restores the road"

        return sm, ctx, graph, check

    def _scn_delete_lift(self, fake_st, factory, dem):
        graph = ResortGraph()
        sm, ctx = _session(fake_st=fake_st, graph=graph, factory=factory, dem=dem)
        lift_id = self._lift_id(graph=graph, dem=dem)
        graph.undo_stack.clear()  # drop the ADD_LIFT so DELETE_LIFT is the top of the stack
        graph.delete_lift(lift_id=lift_id)

        def check() -> None:
            assert sm.is_idle_ready
            assert lift_id in graph.lifts, "undo of delete restores the lift"

        return sm, ctx, graph, check

    def _scn_import_osm_viewing_removed(self, fake_st, factory, dem):
        graph = ResortGraph()
        sm, ctx = _session(fake_st=fake_st, graph=graph, factory=factory, dem=dem)
        graph.import_osm(result=_fake_import_result(dem=dem), dem=dem)
        slope_id = next(iter(graph.slopes))  # the batch created exactly one slope
        sm.view_slope(slope_id=slope_id)  # viewing an imported entity when the batch is undone

        def check() -> None:
            assert sm.is_idle_ready, "undoing the import while viewing a removed entity forces idle"
            assert slope_id not in graph.slopes

        return sm, ctx, graph, check

    def _scn_import_osm_elsewhere(self, fake_st, factory, dem):
        graph = ResortGraph()
        sm, ctx = _session(fake_st=fake_st, graph=graph, factory=factory, dem=dem)
        graph.import_osm(result=_fake_import_result(dem=dem), dem=dem)
        slope_id = next(iter(graph.slopes))  # stay in idle_ready

        def check() -> None:
            assert sm.is_idle_ready, "undo from idle stays idle (redraw only)"
            assert slope_id not in graph.slopes

        return sm, ctx, graph, check

    def _scn_merge_nodes(self, fake_st, factory, dem):
        graph = ResortGraph()
        sm, ctx = _session(fake_st=fake_st, graph=graph, factory=factory, dem=dem)
        slope = _commit_straight_slope_len(graph=graph, dem=dem, n_segments=2)
        junction_id = graph.segments[slope.segment_ids[0]].end_node_id  # shared node between the 2 segments
        jn = graph.nodes[junction_id]
        near_lat = jn.lat - 8 / MapConfig.METERS_PER_DEGREE_EQUATOR
        graph.nodes["X"] = _node_at(dem=dem, node_id="X", lon=jn.lon, lat=near_lat)
        graph.merge_nodes(node_ids=[junction_id, "X"], dem=dem)

        def check() -> None:
            assert sm.is_idle_ready, "node-edit undo is redraw-only (no state change)"
            assert "X" in graph.nodes, "undo of merge restores the merged-away node"

        return sm, ctx, graph, check

    def _scn_delete_nodes(self, fake_st, factory, dem):
        graph = ResortGraph()
        sm, ctx = _session(fake_st=fake_st, graph=graph, factory=factory, dem=dem)
        slope = _commit_straight_slope_len(graph=graph, dem=dem, n_segments=2)
        junction_id = graph.segments[slope.segment_ids[0]].end_node_id  # interior node: deletable
        graph.delete_nodes(node_ids=[junction_id])

        def check() -> None:
            assert sm.is_idle_ready, "node-edit undo is redraw-only (no state change)"
            assert junction_id in graph.nodes, "undo of delete-nodes restores the node"

        return sm, ctx, graph, check

    def _scn_insert_node(self, fake_st, factory, dem):
        graph = ResortGraph()
        sm, ctx = _session(fake_st=fake_st, graph=graph, factory=factory, dem=dem)
        slope = _commit_straight_slope_len(graph=graph, dem=dem, n_segments=1)
        seg_id = slope.segment_ids[0]
        seg = graph.segments[seg_id]
        # A finished straight run simplifies to its two endpoints, so use the geometric midpoint of
        # the leg (far from both vertices) rather than a stored interior point.
        mid_lon = (seg.points[0].lon + seg.points[-1].lon) / 2
        mid_lat = (seg.points[0].lat + seg.points[-1].lat) / 2
        graph.insert_node_on_path(segment_id=seg_id, lon=mid_lon, lat=mid_lat, dem=dem)

        def check() -> None:
            assert sm.is_idle_ready, "node-edit undo is redraw-only (no state change)"
            assert seg_id in graph.segments, "undo of insert restores the original segment"

        return sm, ctx, graph, check

    def _scn_cut_segment(self, fake_st, factory, dem):
        graph = ResortGraph()
        sm, ctx = _session(fake_st=fake_st, graph=graph, factory=factory, dem=dem)
        slope = _commit_straight_slope_len(graph=graph, dem=dem, n_segments=1)
        seg = slope.segment_ids[0]
        a = graph.segments[seg].start_node_id
        b = graph.segments[seg].end_node_id
        graph.cut_segments_between(node_a_id=a, node_b_id=b)

        def check() -> None:
            assert sm.is_idle_ready, "node-edit undo is redraw-only (no state change)"
            assert seg in graph.segments, "undo of cut restores the segment"

        return sm, ctx, graph, check

    def _scenarios(self):
        """(ActionType exercised, scenario builder) rows. Several ActionTypes have more than one row
        (e.g. ADD_SEGMENTS: segments-remain / last-segment / in-custom-path); each row is tagged with
        the ActionType it exercises, and the guard below unions the tags to prove full coverage.
        """
        A = ActionType
        return [
            (A.ADD_SEGMENTS, self._scn_add_segments_two_remain),
            (A.ADD_SEGMENTS, self._scn_add_segments_last_seg),
            (A.ADD_SEGMENTS, self._scn_add_segments_in_custom_path),
            (A.FINISH_SLOPE, self._scn_finish_slope),
            (A.FINISH_ROAD, self._scn_finish_road),
            (A.ADD_LIFT, self._scn_add_lift_while_placing),
            (A.ADD_LIFT, self._scn_add_lift_while_viewing),
            (A.ADD_LIFT, self._scn_add_lift_elsewhere),
            (A.DELETE_SLOPE, self._scn_delete_slope),
            (A.DELETE_ROAD, self._scn_delete_road),
            (A.DELETE_LIFT, self._scn_delete_lift),
            (A.IMPORT_OSM, self._scn_import_osm_viewing_removed),
            (A.IMPORT_OSM, self._scn_import_osm_elsewhere),
            (A.MERGE_NODES, self._scn_merge_nodes),
            (A.DELETE_NODES, self._scn_delete_nodes),
            (A.INSERT_NODE, self._scn_insert_node),
            (A.CUT_SEGMENT, self._scn_cut_segment),
        ]

    def test_every_action_type_has_a_scenario(self) -> None:
        covered = {action_type for action_type, _ in self._scenarios()}
        assert covered == set(ActionType), (
            f"undo dispatch matrix must cover every ActionType. "
            f"Missing: {set(ActionType) - covered}; extra: {covered - set(ActionType)}"
        )

    @pytest.mark.parametrize("idx", range(17))
    def test_undo_dispatch(self, fake_st, path_factory, mock_dem_blue_slope, idx) -> None:
        from skiresort_planner.ui.actions import undo_last_action

        action_type, builder = self._scenarios()[idx]
        _sm, _ctx, _graph, check = builder(fake_st=fake_st, factory=path_factory, dem=mock_dem_blue_slope)
        undo_last_action()  # must never raise TransitionNotAllowed
        assert action_type in set(ActionType)  # the row's tag is a real ActionType
        check()

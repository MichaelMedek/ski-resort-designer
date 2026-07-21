"""Tests for the Play flythrough: group-anchored camera keyframes + the viewing-group resolver.

`flythrough_keyframes` / `flythrough_view_state` are pure (no Streamlit) — tested directly. The camera
glides between a FEW keyframes (deck.gl client-side): a single group gets start+end, a route gets one
keyframe per group (between-lift unit) at FLYTHROUGH_ANCHOR_FRACTION. `flythrough_viewing_groups`
dispatches on the viewed element via the fake session state.
"""

import pytest

from skiresort_planner.constants import MapConfig
from skiresort_planner.core.geo_calculator import GeoCalculator
from skiresort_planner.model.proposed_path import ProposedPathSegment
from skiresort_planner.model.routing import ViewingGroup
from skiresort_planner.ui.actions import (
    flythrough_keyframe_count,
    flythrough_viewing_groups,
    process_route_plan_pending,
)
from skiresort_planner.ui.center_map import MapRenderer
from skiresort_planner.ui.state_machine import PlannerStateMachine
from tests_workflow.conftest import MockDEMService, add_node, add_slope

# A group heading due SOUTH, and one heading due EAST — distinct overall bearings (~180 and ~90).
_SOUTH = ViewingGroup(is_lift=False, actual_polyline=((0.0, 0.010, 3000.0), (0.0, 0.0, 2800.0)))
_EAST = ViewingGroup(is_lift=False, actual_polyline=((0.0, 0.0, 2800.0), (0.010, 0.0, 2600.0)))
_SOUTH_START = _SOUTH.actual_polyline[0]  # (lon, lat, elev)


class TestFlythroughKeyframes:
    def test_single_group_is_start_and_end(self) -> None:
        # One group → exactly start + end; deck.gl GLIDES the camera its whole length in one smooth move.
        kfs = MapRenderer.flythrough_keyframes([_SOUTH])
        assert len(kfs) == 2, "a single group gives start+end keyframes (one glide, not many hops)"
        # Nav-style: each center is pushed LOOK_AHEAD_M ahead along the (south) bearing, so lat < true point.
        assert kfs[0][0] < _SOUTH.actual_polyline[0][1], "centre is ahead of the real position"

    def test_center_is_look_ahead_of_true_position(self) -> None:
        # The keyframe centre sits FLYTHROUGH_LOOK_AHEAD_M ahead of the anchor point along the bearing.
        kfs = MapRenderer.flythrough_keyframes([_SOUTH])
        s_lon, s_lat, _ = _SOUTH_START
        d = GeoCalculator.haversine_distance_m(lat1=s_lat, lon1=s_lon, lat2=kfs[0][0], lon2=kfs[0][1])
        assert abs(d - MapConfig.FLYTHROUGH_LOOK_AHEAD_M) < 1.0, f"centre must be ~LOOK_AHEAD_M ahead, got {d:.0f}m"

    def test_bearing_is_group_straight_line_heading(self) -> None:
        # Every keyframe faces the group's gross start→end heading (south ~180), not a local tangent.
        kfs = MapRenderer.flythrough_keyframes([_SOUTH])
        assert all(abs(b - 180.0) < 1.0 for _, _, b in kfs), f"south group faces ~180, got {[k[2] for k in kfs]}"

    def test_route_keyframes_capped(self) -> None:
        # A route with more groups than the cap can't emit unbounded keyframes (bounds the rerun count).
        many = [_SOUTH, _EAST] * MapConfig.FLYTHROUGH_MAX_KEYFRAMES
        assert len(MapRenderer.flythrough_keyframes(many)) == MapConfig.FLYTHROUGH_MAX_KEYFRAMES

    def test_route_one_keyframe_per_group(self) -> None:
        # Two groups (south then east) → one keyframe each + a final end keyframe, each facing its group.
        kfs = MapRenderer.flythrough_keyframes([_SOUTH, _EAST])
        assert len(kfs) == 3, "K groups → K anchor keyframes + 1 end keyframe"
        assert abs(kfs[0][2] - 180.0) < 1.0, "first group faces south (~180)"
        assert abs(kfs[1][2] - 90.0) < 1.0, "second group faces east (~90)"

    def test_no_groups_is_empty_but_degenerate_group_fails_fast(self) -> None:
        assert MapRenderer.flythrough_keyframes([]) == [], "nothing viewed → no keyframes"
        # A real viewing group is always ≥2 points; a 1-point group is a broken invariant → fail loud.
        one_point = ViewingGroup(is_lift=False, actual_polyline=((0.0, 0.0, 100.0),))
        with pytest.raises(AssertionError):
            MapRenderer.flythrough_keyframes([one_point])


class TestFlythroughViewState:
    def test_index_selects_keyframe(self) -> None:
        kfs = MapRenderer.flythrough_keyframes([_SOUTH, _EAST])
        lat0, lon0, bearing0, zoom0, pitch0 = MapRenderer.flythrough_view_state(kfs, 0)
        assert (lat0, lon0, bearing0) == kfs[0], "index 0 → first keyframe"
        assert zoom0 == MapConfig.VIEW_3D_ZOOM and pitch0 == MapConfig.VIEW_3D_PITCH, "unified 3D zoom/pitch"

    def test_needs_two_keyframes(self) -> None:
        with pytest.raises(ValueError):
            MapRenderer.flythrough_view_state([(0.0, 0.0, 0.0)], 0)


class TestFlythroughEasing:
    """While flying, get_view_state must emit transitionDuration + a LinearInterpolator so deck.gl glides
    the camera between sparse keyframes CLIENT-SIDE (the fix for per-frame-rerun slowness).
    """

    def test_transition_emitted_only_while_flying(self) -> None:
        import json

        r = MapRenderer(graph=None)
        assert "transitionDuration" not in json.loads(r.get_view_state().to_json()), "no transition when idle"
        r.transition_duration = MapConfig.FLYTHROUGH_TRANSITION_MS
        vs = json.loads(r.get_view_state().to_json())
        assert vs["transitionDuration"] == MapConfig.FLYTHROUGH_TRANSITION_MS
        assert vs["transitionInterpolator"]["@@type"] == "LinearInterpolator"
        assert "bearing" in vs["transitionInterpolator"]["transitionProps"]


class TestViewingGroupsForView:
    def _seed(self, fake_st, graph, sm, ctx) -> None:
        fake_st.session_state["graph"] = graph
        fake_st.session_state["state_machine"] = sm
        fake_st.session_state["context"] = ctx

    def test_slope_is_one_group(self, fake_st, empty_graph, path_points_blue) -> None:
        empty_graph.commit_paths(paths=[ProposedPathSegment(points=path_points_blue, target_difficulty="blue")])
        slope = empty_graph.finish_slope(segment_ids=list(empty_graph.segments.keys()))
        assert slope is not None
        sm, ctx = PlannerStateMachine.create(graph=empty_graph, add_ui_listener=False)
        ctx.viewing.set_slope_id(slope.id)
        self._seed(fake_st, empty_graph, sm, ctx)

        groups = flythrough_viewing_groups()
        assert len(groups) == 1 and not groups[0].is_lift, "a single slope → one non-lift group"
        assert len(groups[0].actual_polyline) >= 2
        assert flythrough_keyframe_count() == 2, "a single group gives start+end keyframes"

    def test_nothing_viewed_is_empty(self, fake_st, empty_graph) -> None:
        # No slope/road/lift viewed and no routes → empty (Play is a no-op).
        sm, ctx = PlannerStateMachine.create(graph=empty_graph, add_ui_listener=False)
        self._seed(fake_st, empty_graph, sm, ctx)
        assert flythrough_viewing_groups() == []
        assert flythrough_keyframe_count() == 0


class TestReplanStopsFlythrough:
    """Regression: playing a flythrough then planning a NEW route must not keep riding the old route.
    Computing a fresh plan stops playback (single source: groups resolve live, so a fresh plan with
    playback cleared can't drift onto the previous route).
    """

    def test_new_route_plan_stops_active_flythrough(self, fake_st, empty_graph) -> None:
        dem = MockDEMService(base_elevation=2500.0, slope_ns_pct=20.0, slope_ew_pct=0.0)
        add_node(empty_graph, "A", 0.0, 0.0, 2000.0)
        add_node(empty_graph, "B", 0.0, -0.001, 1900.0)
        add_slope(empty_graph, "SL1", top="A", bottom="B")
        sm, ctx = PlannerStateMachine.create(graph=empty_graph, add_ui_listener=False)
        fake_st.session_state["graph"] = empty_graph
        fake_st.session_state["state_machine"] = sm
        fake_st.session_state["context"] = ctx
        fake_st.session_state["dem_service"] = dem

        ctx.viewing.enable_3d()
        ctx.viewing.start_flythrough()
        # Arm a fresh point-to-point plan (mirrors the route-placing completion click).
        ctx.route_plan.start_node_id = "A"
        ctx.route_plan.end_node_id = "B"
        ctx.pending.route_plan_generation = True

        process_route_plan_pending()

        assert not ctx.viewing.flythrough_active, "a fresh route plan must stop the old flythrough"
        assert ctx.viewing.flythrough_frame == 0

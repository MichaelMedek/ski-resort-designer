"""Tests for the Play flythrough camera math + the current-view polyline resolver.

`flythrough_view_state` is a pure function (no Streamlit) — tested directly for constant-speed / endpoints /
bearing. `flythrough_points_for_view` dispatches on the viewed element via the fake session state.
"""

import pytest

from skiresort_planner.constants import MapConfig
from skiresort_planner.core.geo_calculator import GeoCalculator
from skiresort_planner.model.path_point import PathPoint
from skiresort_planner.model.proposed_path import ProposedPathSegment
from skiresort_planner.ui.actions import flythrough_points_for_view
from skiresort_planner.ui.center_map import MapRenderer
from skiresort_planner.ui.state_machine import PlannerStateMachine

# A descending L-shaped route (goes south, then east) so bearing must change along it.
_L_ROUTE = [
    PathPoint(lon=0.0, lat=0.010, elevation=3000.0),
    PathPoint(lon=0.0, lat=0.000, elevation=2800.0),
    PathPoint(lon=0.010, lat=0.000, elevation=2600.0),
]


class TestFlythroughViewState:
    def test_endpoints(self) -> None:
        lat0, lon0, _, zoom0, pitch0 = MapRenderer.flythrough_view_state(_L_ROUTE, progress=0.0)
        lat1, lon1, *_ = MapRenderer.flythrough_view_state(_L_ROUTE, progress=1.0)
        assert (lat0, lon0) == (_L_ROUTE[0].lat, _L_ROUTE[0].lon), "progress 0 frames the start"
        assert (round(lat1, 9), round(lon1, 9)) == (_L_ROUTE[-1].lat, _L_ROUTE[-1].lon), "progress 1 frames the end"
        assert zoom0 == MapConfig.VIEW_3D_ZOOM and pitch0 == MapConfig.VIEW_3D_PITCH

    def test_bearing_tracks_local_heading(self) -> None:
        # Early on we head SOUTH (bearing ~180); late we head EAST (bearing ~90).
        _, _, bearing_early, *_ = MapRenderer.flythrough_view_state(_L_ROUTE, progress=0.1)
        _, _, bearing_late, *_ = MapRenderer.flythrough_view_state(_L_ROUTE, progress=0.9)
        assert abs(bearing_early - 180.0) < 20.0, f"early leg heads south, got {bearing_early}"
        assert abs(bearing_late - 90.0) < 20.0, f"late leg heads east, got {bearing_late}"

    def test_constant_arc_length_speed(self) -> None:
        # Equal progress steps must move equal ground distance (arc-length parameterisation), NOT
        # vertex-stepping. Sample 20 steps and assert the per-step distance spread is tight.
        n = 20
        prev = MapRenderer.flythrough_view_state(_L_ROUTE, progress=0.0)
        dists = []
        for i in range(1, n + 1):
            cur = MapRenderer.flythrough_view_state(_L_ROUTE, progress=i / n)
            dists.append(GeoCalculator.haversine_distance_m(lat1=prev[0], lon1=prev[1], lat2=cur[0], lon2=cur[1]))
            prev = cur
        assert min(dists) > 0
        assert max(dists) / min(dists) < 1.6, f"steps should be near-equal distance, spread={max(dists) / min(dists)}"

    def test_progress_clamped(self) -> None:
        below = MapRenderer.flythrough_view_state(_L_ROUTE, progress=-5.0)
        above = MapRenderer.flythrough_view_state(_L_ROUTE, progress=5.0)
        assert (below[0], below[1]) == (_L_ROUTE[0].lat, _L_ROUTE[0].lon)
        assert (round(above[0], 9), round(above[1], 9)) == (_L_ROUTE[-1].lat, _L_ROUTE[-1].lon)

    def test_needs_two_points(self) -> None:
        with pytest.raises(ValueError):
            MapRenderer.flythrough_view_state([PathPoint(lon=0.0, lat=0.0, elevation=100.0)], progress=0.0)


class TestPointsForView:
    def _seed(self, fake_st, graph, sm, ctx) -> None:
        fake_st.session_state["graph"] = graph
        fake_st.session_state["state_machine"] = sm
        fake_st.session_state["context"] = ctx

    def test_slope_polyline(self, fake_st, empty_graph, path_points_blue) -> None:
        empty_graph.commit_paths(paths=[ProposedPathSegment(points=path_points_blue, target_difficulty="blue")])
        slope = empty_graph.finish_slope(segment_ids=list(empty_graph.segments.keys()))
        assert slope is not None
        sm, ctx = PlannerStateMachine.create(graph=empty_graph, add_ui_listener=False)
        ctx.viewing.set_slope_id(slope.id)
        self._seed(fake_st, empty_graph, sm, ctx)

        pts = flythrough_points_for_view()
        assert len(pts) >= 2 and all(isinstance(p, PathPoint) for p in pts), "slope resolves to a PathPoint polyline"

    def test_route_polyline(self, fake_st, empty_graph) -> None:
        # No slope/road/lift viewed and no routes → empty (Play is a no-op).
        sm, ctx = PlannerStateMachine.create(graph=empty_graph, add_ui_listener=False)
        self._seed(fake_st, empty_graph, sm, ctx)
        assert flythrough_points_for_view() == []

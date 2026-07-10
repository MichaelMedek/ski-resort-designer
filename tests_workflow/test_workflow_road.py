"""Tests for the Road feature: model, planner gradient band, graph operations,
serialization, computed parking, and the SegmentPath refactor.
"""

import math

import pytest

from skiresort_planner.constants import PathConfig
from skiresort_planner.generators.connection_planners import LeastCostPathPlanner
from skiresort_planner.model.node import Node
from skiresort_planner.model.path_point import PathPoint
from skiresort_planner.model.proposed_path import ProposedPathSegment
from skiresort_planner.model.resort_graph import (
    AddRoadAction,
    DeleteRoadAction,
    ResortGraph,
)
from skiresort_planner.model.road import Road
from skiresort_planner.model.segment_path import SegmentPath
from skiresort_planner.model.slope import Slope


def _commit_road(graph: ResortGraph, path_points: list[PathPoint]) -> Road:
    """Commit a path as a road (record_undo=False + finish_road), return the Road."""
    proposal = ProposedPathSegment(points=path_points, is_connector=True)
    graph.commit_paths(paths=[proposal], record_undo=False)
    seg_id = list(graph.segments.keys())[-1]
    road = graph.finish_road(segment_ids=[seg_id])
    assert road is not None
    return road


# =============================================================================
# SegmentPath base / Road & Slope subclasses
# =============================================================================


class TestSegmentPathHierarchy:
    def test_slope_and_road_are_segment_paths(self) -> None:
        assert issubclass(Slope, SegmentPath)
        assert issubclass(Road, SegmentPath)

    def test_number_from_id_uses_subclass_prefix(self) -> None:
        assert Slope.number_from_id("SL7") == 7
        assert Road.number_from_id("R3") == 3

    def test_road_name_is_compass_based(self) -> None:
        # bearing 90 → East
        assert Road.generate_name(road_id="R1", avg_bearing=90.0) == "1 (E Access)"


class TestSegmentPathBaseMethods:
    """Shared SegmentPath geometry, exercised through a committed Road."""

    def test_number_property_derives_from_id(self, empty_graph, path_points_blue) -> None:
        road = _commit_road(empty_graph, path_points_blue)
        assert road.number == Road.number_from_id(road.id)

    def test_total_length_and_drop_match_segment(self, empty_graph, path_points_blue) -> None:
        road = _commit_road(empty_graph, path_points_blue)
        seg = empty_graph.segments[road.segment_ids[0]]
        assert road.get_total_length(segments=empty_graph.segments) == pytest.approx(seg.length_m)
        assert road.get_total_drop(segments=empty_graph.segments) == pytest.approx(seg.total_drop_m)

    def test_get_all_points_returns_segment_points(self, empty_graph, path_points_blue) -> None:
        road = _commit_road(empty_graph, path_points_blue)
        points = road.get_all_points(segments=empty_graph.segments)
        assert len(points) == len(empty_graph.segments[road.segment_ids[0]].points)

    def test_get_all_points_raises_when_empty(self) -> None:
        # A road referencing no existing segments has no points → error.
        orphan = Road(id="R9", name="x", segment_ids=[], start_node_id="N1", end_node_id="N2")
        with pytest.raises(ValueError, match="at least one point"):
            orphan.get_all_points(segments={})

    def test_has_warnings_reflects_segments(self, empty_graph, path_points_blue) -> None:
        road = _commit_road(empty_graph, path_points_blue)
        seg = empty_graph.segments[road.segment_ids[0]]
        assert road.has_warnings(segments=empty_graph.segments) == bool(seg.warnings)


# =============================================================================
# Road model metrics
# =============================================================================


class TestRoadModel:
    def test_max_gradient_positive_for_descending_road(self, empty_graph, path_points_blue) -> None:
        # path_points_blue descends; max gradient is a positive magnitude.
        road = _commit_road(empty_graph, path_points_blue)
        assert road.get_max_gradient(segments=empty_graph.segments) > 0.0

    def test_max_gradient_is_magnitude_for_short_climb(self, empty_graph) -> None:
        """A short steep CLIMB must report a positive steepness, not a negative avg.

        A climbing segment's max_slope_pct is negative (its seed is the signed
        avg). get_max_gradient takes the magnitude, so the ±12% road badge
        correctly catches a steep climb.
        """
        M = 111320.0
        # ~20% climb over 100m (well under the 300m rolling window).
        steep_climb = [
            PathPoint(lon=0.0, lat=0.0, elevation=2000.0),
            PathPoint(lon=0.0, lat=100 / M, elevation=2020.0),
        ]
        road = _commit_road(empty_graph, steep_climb)
        assert road.get_max_gradient(segments=empty_graph.segments) == pytest.approx(20.0, abs=1.0)


# =============================================================================
# Planner gradient band (road mode) — the one real algorithm change
# =============================================================================


class TestPlannerGradientBand:
    def _cost(self, planner, from_elev, to_elev, band):
        # 30m east step; deterministic horizontal distance.
        return planner._calc_edge_cost(
            from_elev=from_elev,
            to_elev=to_elev,
            from_lon=0.0,
            from_lat=0.0,
            to_lon=30.0 / 111320.0,
            to_lat=0.0,
            target_slope_pct=0.0,
            side="left",
            target_lon=1.0,
            target_lat=0.0,
            gradient_band=band,
        )

    def test_in_band_flat_has_no_extra_penalty(self, mock_dem_blue_slope) -> None:
        from skiresort_planner.core.terrain_analyzer import TerrainAnalyzer

        planner = LeastCostPathPlanner(
            dem_service=mock_dem_blue_slope, terrain_analyzer=TerrainAnalyzer(dem=mock_dem_blue_slope)
        )
        band = (-12.0, 12.0)
        horiz = 30.0
        # Flat edge inside band → cost == horizontal distance (exp(0) == 1).
        flat_cost = self._cost(planner, 2000.0, 2000.0, band)
        assert flat_cost == pytest.approx(horiz, rel=0.05)

    def test_uphill_in_band_not_penalized_in_road_mode(self, mock_dem_blue_slope) -> None:
        from skiresort_planner.core.terrain_analyzer import TerrainAnalyzer

        planner = LeastCostPathPlanner(
            dem_service=mock_dem_blue_slope, terrain_analyzer=TerrainAnalyzer(dem=mock_dem_blue_slope)
        )
        band = (-12.0, 12.0)
        # +10% uphill (climb 3m over 30m) is within band → same as flat.
        uphill = self._cost(planner, 2000.0, 2003.0, band)
        flat = self._cost(planner, 2000.0, 2000.0, band)
        assert uphill == pytest.approx(flat, rel=0.05)

    def test_out_of_band_costs_more(self, mock_dem_blue_slope) -> None:
        from skiresort_planner.core.terrain_analyzer import TerrainAnalyzer

        planner = LeastCostPathPlanner(
            dem_service=mock_dem_blue_slope, terrain_analyzer=TerrainAnalyzer(dem=mock_dem_blue_slope)
        )
        band = (-12.0, 12.0)
        in_band = self._cost(planner, 2000.0, 2000.0, band)
        # 40% descent (12m over 30m) is well outside the band → strictly costlier.
        out_of_band = self._cost(planner, 2000.0, 1988.0, band)
        assert out_of_band > in_band

    def test_slope_mode_unchanged_by_band_param(self, mock_dem_blue_slope) -> None:
        """gradient_band=None must reproduce the original slope cost exactly."""
        from math import exp

        from skiresort_planner.constants import PlannerConfig
        from skiresort_planner.core.geo_calculator import GeoCalculator
        from skiresort_planner.core.terrain_analyzer import TerrainAnalyzer

        planner = LeastCostPathPlanner(
            dem_service=mock_dem_blue_slope, terrain_analyzer=TerrainAnalyzer(dem=mock_dem_blue_slope)
        )
        cost = self._cost(planner, 2000.0, 1994.0, None)  # 6m drop over one east step, target 0
        # Recompute expected via the documented slope formula, using the exact
        # horizontal distance the cost function itself would measure.
        horiz = GeoCalculator.haversine_distance_m(lat1=0.0, lon1=0.0, lat2=0.0, lon2=30.0 / 111320.0)
        actual_slope = (6.0 / horiz) * 100
        expected = horiz * exp(abs(actual_slope - 0.0) / PlannerConfig.COST_SIGMA)
        assert cost == pytest.approx(expected, rel=1e-6)


# =============================================================================
# Graph: finish_road, delete_road, undo
# =============================================================================


class TestRoadGraphOps:
    def test_finish_road_creates_road_with_atomic_undo(self, empty_graph, path_points_blue) -> None:
        road = _commit_road(empty_graph, path_points_blue)
        assert road.id in empty_graph.roads
        assert len(empty_graph.roads) == 1
        # commit used record_undo=False, so the only undo entry is the atomic AddRoadAction.
        assert len(empty_graph.undo_stack) == 1
        assert isinstance(empty_graph.undo_stack[0], AddRoadAction)

    def test_undo_add_road_removes_everything(self, empty_graph, path_points_blue) -> None:
        _commit_road(empty_graph, path_points_blue)
        empty_graph.undo_last()
        assert len(empty_graph.roads) == 0
        assert len(empty_graph.segments) == 0
        assert len(empty_graph.nodes) == 0  # orphaned nodes cleaned up

    def test_delete_road_removes_and_records_undo(self, empty_graph, path_points_blue) -> None:
        road = _commit_road(empty_graph, path_points_blue)
        assert empty_graph.delete_road(road_id=road.id) is True
        assert road.id not in empty_graph.roads
        assert len(empty_graph.segments) == 0
        assert isinstance(empty_graph.undo_stack[-1], DeleteRoadAction)

    def test_undo_delete_road_restores(self, empty_graph, path_points_blue) -> None:
        road = _commit_road(empty_graph, path_points_blue)
        seg_count = len(empty_graph.segments)
        empty_graph.delete_road(road_id=road.id)
        empty_graph.undo_last()
        assert road.id in empty_graph.roads
        assert len(empty_graph.segments) == seg_count

    def test_delete_missing_road_is_false(self, empty_graph) -> None:
        assert empty_graph.delete_road(road_id="R99") is False


# =============================================================================
# Serialization (round-trip + back-compat)
# =============================================================================


class TestRoadSerialization:
    def test_roundtrip_preserves_roads(self, empty_graph, path_points_blue) -> None:
        road = _commit_road(empty_graph, path_points_blue)
        data = empty_graph.to_dict()
        restored = ResortGraph.from_dict(data=data)
        assert road.id in restored.roads
        assert restored.roads[road.id].name == road.name
        assert restored._road_counter == empty_graph._road_counter


# =============================================================================
# Computed parking places
# =============================================================================


class TestParkingNodes:
    def test_no_parking_without_roads(self, empty_graph, path_points_blue) -> None:
        proposal = ProposedPathSegment(points=path_points_blue, target_difficulty="blue")
        empty_graph.commit_paths(paths=[proposal])
        empty_graph.finish_slope(segment_ids=list(empty_graph.segments.keys()))
        assert empty_graph.get_parking_nodes() == []

    def test_parking_appears_where_road_shares_slope_node(self, empty_graph, path_points_blue) -> None:
        # Slope along the blue points.
        slope_proposal = ProposedPathSegment(points=path_points_blue, target_difficulty="blue")
        empty_graph.commit_paths(paths=[slope_proposal])
        empty_graph.finish_slope(segment_ids=list(empty_graph.segments.keys()))

        shared_start = path_points_blue[0]  # slope's top node
        # Road starting at the slope's top node, heading east (new distinct end).
        M = 111320.0
        road_points = [
            PathPoint(lon=shared_start.lon, lat=shared_start.lat, elevation=shared_start.elevation),
            PathPoint(lon=300 / M, lat=shared_start.lat, elevation=shared_start.elevation),
        ]
        _commit_road(empty_graph, road_points)

        parking = empty_graph.get_parking_nodes()
        assert len(parking) == 1
        assert isinstance(parking[0], Node)
        # The parking node is the shared start node.
        assert parking[0].distance_to(lon=shared_start.lon, lat=shared_start.lat) < 1.0

    def test_road_touching_nothing_yields_no_parking(self, empty_graph) -> None:
        M = 111320.0
        road_points = [
            PathPoint(lon=0.0, lat=0.0, elevation=2000.0),
            PathPoint(lon=300 / M, lat=0.0, elevation=2000.0),
        ]
        _commit_road(empty_graph, road_points)
        assert empty_graph.get_parking_nodes() == []


# =============================================================================
# get_stats includes roads
# =============================================================================


class TestStatsWithRoads:
    def test_stats_reports_road_count_and_length(self, empty_graph, path_points_blue) -> None:
        _commit_road(empty_graph, path_points_blue)
        stats = empty_graph.get_stats()
        assert stats["total_roads"] == 1
        assert stats["total_road_length_m"] > 0


# =============================================================================
# State-machine workflow (place → view → cancel)
# =============================================================================


class TestRoadPlacementWorkflow:
    """Road placement state transitions (no Streamlit)."""

    def _sm(self, graph: ResortGraph):
        from skiresort_planner.ui.state_machine import PlannerStateMachine

        return PlannerStateMachine.create(graph=graph, add_ui_listener=False)

    def test_start_then_complete_road(self, empty_graph, path_points_blue) -> None:
        sm, ctx = self._sm(empty_graph)
        assert sm.current_state_value == "idle_ready"

        sm.start_road(node_id=None, location=path_points_blue[0])
        assert sm.current_state_value == "road_placing"
        assert ctx.road.start_location is path_points_blue[0]

        road = _commit_road(empty_graph, path_points_blue)
        sm.complete_road(road_id=road.id)
        assert sm.current_state_value == "idle_viewing_road"
        assert ctx.viewing.road_id == road.id
        assert ctx.viewing.panel_visible is True

    def test_cancel_road_returns_to_idle(self, empty_graph, path_points_blue) -> None:
        sm, ctx = self._sm(empty_graph)
        sm.start_road(node_id=None, location=path_points_blue[0])
        assert sm.current_state_value == "road_placing"

        sm.cancel_road()
        assert sm.current_state_value == "idle_ready"
        assert ctx.road.start_location is None  # cleared on exit


# =============================================================================
# 3D side-view camera for a road
# =============================================================================


class TestRoad3DView:
    def test_calculate_3d_view_returns_camera_tuple(self, empty_graph, path_points_blue) -> None:
        from skiresort_planner.ui.center_map import MapRenderer

        road = _commit_road(empty_graph, path_points_blue)
        lat, lon, bearing, zoom, pitch = MapRenderer.calculate_3d_view_for_road(graph=empty_graph, road_id=road.id)
        assert isinstance(zoom, int)
        assert pitch > 0  # angled side view
        assert 0.0 <= bearing <= 360.0

"""Unit tests for the Road model (model/road.py).

Road is a SegmentPath subclass for vehicle roads. Covers its compass-based name
and the max-gradient magnitude used by the ±12% car-road badge.
"""

import pytest

from skiresort_planner.model.path_point import PathPoint
from skiresort_planner.model.proposed_path import ProposedPathSegment
from skiresort_planner.model.resort_graph import ResortGraph
from skiresort_planner.model.road import Road

M = 111320.0


def _commit_road(graph: ResortGraph, path_points: list[PathPoint]) -> Road:
    graph.commit_paths(paths=[ProposedPathSegment(points=path_points, is_connector=True)], record_undo=False)
    road = graph.finish_road(segment_ids=[list(graph.segments.keys())[-1]])
    assert road is not None
    return road


class TestRoadName:
    def test_road_name_is_compass_based(self) -> None:
        # bearing 90 → East
        assert Road.generate_name(road_id="R1", avg_bearing=90.0) == "1 (E Access)"


class TestRoadMaxGradient:
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
        # ~20% climb over 100m (well under the 300m rolling window).
        steep_climb = [
            PathPoint(lon=0.0, lat=0.0, elevation=2000.0),
            PathPoint(lon=0.0, lat=100 / M, elevation=2020.0),
        ]
        road = _commit_road(empty_graph, steep_climb)
        assert road.get_max_gradient(segments=empty_graph.segments) == pytest.approx(20.0, abs=1.0)

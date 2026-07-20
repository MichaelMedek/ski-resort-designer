"""Unit tests for the Road model (model/road.py).

Road is a SegmentPath subclass for vehicle roads. Covers its compass-based name
and the max-gradient magnitude used by the ±15% (ROAD_MAX_GRADIENT_PCT) car-road badge.
"""

import pytest

from skiresort_planner.constants import MapConfig
from skiresort_planner.model.path_point import PathPoint
from skiresort_planner.model.path_segment import SegmentKind
from skiresort_planner.model.proposed_path import ProposedPathSegment
from skiresort_planner.model.resort_graph import ResortGraph
from skiresort_planner.model.road import Road


def _commit_road(graph: ResortGraph, path_points: list[PathPoint]) -> Road:
    graph.commit_paths(
        paths=[ProposedPathSegment(points=path_points, is_connector=True, kind=SegmentKind.ROAD)], record_undo=False
    )
    road = graph.finish_road(segment_ids=[list(graph.segments.keys())[-1]])
    assert road is not None
    return road


class TestRoadName:
    def test_road_name_is_creative_and_compass_based(self) -> None:
        # bearing 90 → Ost. Format: "{n} ({direction} {Prefix} {Suffix})".
        from skiresort_planner.constants import NameConfig

        name = Road.generate_name(road_id="R1", avg_bearing=90.0)
        assert name.startswith("1 (Ost "), f"expected number + compass direction, got {name!r}"
        assert name.endswith(")")
        inner = name[len("1 (Ost ") : -1]  # "{Prefix} {Suffix}"
        prefix, suffix = inner.split(" ")
        assert prefix in NameConfig.ROAD_PREFIXES
        assert suffix in NameConfig.ROAD_SUFFIXES


class TestRoadMaxGradient:
    def test_max_gradient_positive_for_descending_road(self, empty_graph, path_points_blue) -> None:
        # path_points_blue descends; max gradient is a positive magnitude.
        road = _commit_road(empty_graph, path_points_blue)
        assert road.get_max_gradient(segments=empty_graph.segments) > 0.0

    def test_max_gradient_is_magnitude_for_short_climb(self, empty_graph) -> None:
        """A short steep CLIMB must report a positive steepness, not a negative avg.

        max_slope_pct is a MAGNITUDE (steepest-section steepness, direction-agnostic),
        so a climbing road's steepness is positive and the ±15% road badge / commit
        cap correctly catches a steep climb the same as a steep descent.
        """
        # ~20% climb over 100m (well under the 300m rolling window).
        steep_climb = [
            PathPoint(lon=0.0, lat=0.0, elevation=2000.0),
            PathPoint(lon=0.0, lat=100 / MapConfig.METERS_PER_DEGREE_EQUATOR, elevation=2020.0),
        ]
        road = _commit_road(empty_graph, steep_climb)
        assert road.get_max_gradient(segments=empty_graph.segments) == pytest.approx(20.0, abs=1.0)

    def test_max_slope_pct_is_magnitude_for_climb(self) -> None:
        """max_slope_pct is a magnitude: a climbing path reports POSITIVE steepness.

        Regression for the signed-value bug where a steep climb (negative slope)
        slipped under the road `<= +15%` cap because max_slope_pct returned the
        signed average instead of its magnitude.
        """
        climb = ProposedPathSegment(
            points=[
                PathPoint(lon=0.0, lat=0.0, elevation=2000.0),
                PathPoint(
                    lon=0.0, lat=100 / MapConfig.METERS_PER_DEGREE_EQUATOR, elevation=2020.0
                ),  # +20m over 100m → 20% climb
            ]
        )
        assert climb.avg_slope_pct < 0.0, "climb has a negative signed average"
        assert climb.max_slope_pct == pytest.approx(20.0, abs=1.0), "steepest section is a positive magnitude"

    def test_max_gradient_is_per_segment_not_across_joins(self, empty_graph) -> None:
        """Steepest section is the max of each segment's own window, NOT a window rolled
        across a junction.

        Two continuous ~350m segments each ~13% form a sustained stretch across their
        join, but because the junction is continuous (no jump) and each segment is
        individually valid, we report the per-segment steepest (~13%), not an inflated
        cross-join figure. Keeps the metric simple and consistent with commit validation.
        """
        # Each segment ≥ ROLLING_WINDOW_M (300m) so it has a real window and is counted.
        a = [
            PathPoint(lon=0.0, lat=0.0, elevation=2000.0),
            PathPoint(lon=0.0, lat=350 / MapConfig.METERS_PER_DEGREE_EQUATOR, elevation=2045.5),  # +45.5m/350m = 13%
        ]
        b = [
            PathPoint(lon=0.0, lat=350 / MapConfig.METERS_PER_DEGREE_EQUATOR, elevation=2045.5),
            PathPoint(lon=0.0, lat=700 / MapConfig.METERS_PER_DEGREE_EQUATOR, elevation=2091.0),  # +45.5m/350m = 13%
        ]
        empty_graph.commit_paths(
            paths=[ProposedPathSegment(points=a, is_connector=True, kind=SegmentKind.ROAD)], record_undo=False
        )
        empty_graph.commit_paths(
            paths=[ProposedPathSegment(points=b, is_connector=True, kind=SegmentKind.ROAD)], record_undo=False
        )
        seg_ids = list(empty_graph.segments.keys())[-2:]
        road = empty_graph.finish_road(segment_ids=seg_ids)
        # Reports each segment's own ~13%, not a cross-join figure > 13%.
        assert road.get_max_gradient(segments=empty_graph.segments) == pytest.approx(13.0, abs=1.5)

    def test_max_gradient_ignores_sub_window_segments(self, empty_graph) -> None:
        """A short (< ROLLING_WINDOW_M) steep segment is NOT counted when a full-window
        segment exists — its max_slope_pct degrades to its average and would mislead.
        """
        # Long gentle segment (counts) + short steep segment (ignored while a long one exists).
        long_gentle = [
            PathPoint(lon=0.0, lat=0.0, elevation=2000.0),
            PathPoint(lon=0.0, lat=400 / MapConfig.METERS_PER_DEGREE_EQUATOR, elevation=2020.0),  # 400m @ 5%
        ]
        short_steep = [
            PathPoint(lon=0.0, lat=400 / MapConfig.METERS_PER_DEGREE_EQUATOR, elevation=2020.0),
            PathPoint(
                lon=0.0, lat=450 / MapConfig.METERS_PER_DEGREE_EQUATOR, elevation=2027.0
            ),  # 50m @ 14% (sub-window)
        ]
        empty_graph.commit_paths(
            paths=[ProposedPathSegment(points=long_gentle, is_connector=True, kind=SegmentKind.ROAD)], record_undo=False
        )
        empty_graph.commit_paths(
            paths=[ProposedPathSegment(points=short_steep, is_connector=True, kind=SegmentKind.ROAD)], record_undo=False
        )
        road = empty_graph.finish_road(segment_ids=list(empty_graph.segments.keys())[-2:])
        # Only the long 5% segment counts; the 50m spur is below the window and excluded.
        assert road.get_max_gradient(segments=empty_graph.segments) == pytest.approx(5.0, abs=1.5)

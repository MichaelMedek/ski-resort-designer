"""Unit tests for the PathSegment model (model/path_segment.py)."""

from skiresort_planner.model.path_point import PathPoint
from skiresort_planner.model.path_segment import PathSegment, SegmentKind
from skiresort_planner.model.proposed_path import ProposedPathSegment


class TestSegmentKind:
    """A committed segment's kind (SLOPE/ROAD) is intrinsic — set at commit,
    persisted, and read straight off the segment (not reconstructed from owners).
    """

    def test_slope_commit_defaults_to_slope_kind(self, empty_graph, path_points_blue) -> None:
        empty_graph.commit_paths(paths=[ProposedPathSegment(points=path_points_blue, target_difficulty="blue")])
        seg = list(empty_graph.segments.values())[-1]
        assert seg.kind == SegmentKind.SLOPE

    def test_road_commit_carries_road_kind(self, empty_graph, path_points_blue) -> None:
        empty_graph.commit_paths(
            paths=[ProposedPathSegment(points=path_points_blue, is_connector=True, kind=SegmentKind.ROAD)]
        )
        seg = list(empty_graph.segments.values())[-1]
        assert seg.kind == SegmentKind.ROAD

    def test_from_dict_defaults_to_slope_when_kind_absent(self) -> None:
        # Pre-enum saves have no "kind" key → SLOPE (backward compatible).
        data: dict[str, object] = {
            "id": "S1",
            "name": "Segment 1",
            "points": [
                {"lon": 0.0, "lat": 0.0, "elevation": 2000.0},
                {"lon": 0.0, "lat": -0.001, "elevation": 1990.0},
            ],
            "start_node_id": "N1",
            "end_node_id": "N2",
        }
        assert PathSegment.from_dict(data=data).kind == SegmentKind.SLOPE

    def test_from_dict_reads_road_kind(self) -> None:
        data: dict[str, object] = {
            "id": "S1",
            "name": "Segment 1",
            "points": [
                {"lon": 0.0, "lat": 0.0, "elevation": 2000.0},
                {"lon": 0.0, "lat": -0.001, "elevation": 1990.0},
            ],
            "start_node_id": "N1",
            "end_node_id": "N2",
            "kind": "road",
        }
        assert PathSegment.from_dict(data=data).kind == SegmentKind.ROAD


class TestBeltWidth:
    """PathSegment.width_m: constant for roads, side-slope-adaptive for slopes."""

    @staticmethod
    def _segment(kind: SegmentKind, side_slope_pct: float) -> "PathSegment":
        return PathSegment(
            points=[
                PathPoint(lon=0.0, lat=0.0, elevation=2000.0),
                PathPoint(lon=0.0, lat=-0.005, elevation=1900.0),
            ],
            id="S1",
            kind=kind,
            side_slope_pct=side_slope_pct,
        )

    def test_road_width_is_constant_regardless_of_side_slope(self) -> None:
        from skiresort_planner.constants import EarthworkConfig

        gentle = self._segment(kind=SegmentKind.ROAD, side_slope_pct=2.0)
        steep = self._segment(kind=SegmentKind.ROAD, side_slope_pct=45.0)
        assert gentle.width_m == float(EarthworkConfig.ROAD_WIDTH_M)
        assert steep.width_m == gentle.width_m  # roads never vary with terrain

    def test_slope_width_narrows_on_steeper_side_slope(self) -> None:
        # Slopes stay adaptive: steeper cross-slope → narrower belt (excavation limit).
        gentle = self._segment(kind=SegmentKind.SLOPE, side_slope_pct=2.0)
        steep = self._segment(kind=SegmentKind.SLOPE, side_slope_pct=45.0)
        assert steep.width_m < gentle.width_m

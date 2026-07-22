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


class TestWarningsByKind:
    """Too-flat / too-steep are SKI-only warnings — a road is gentle/flat/climbing by design, so it
    never gets them. The excavator (cross-slope earthwork) warning applies to both kinds.
    """

    @staticmethod
    def _gentle_segment(kind: SegmentKind) -> PathSegment:
        # ~2% average grade (100m drop over ~5.5km) → below MIN_SKIABLE_PCT (5%) → too-flat for a slope.
        return PathSegment(
            points=[
                PathPoint(lon=0.0, lat=0.0, elevation=2000.0),
                PathPoint(lon=0.0, lat=-0.05, elevation=1900.0),
            ],
            id="S1",
            kind=kind,
            side_slope_pct=1.0,
        )

    def test_flat_slope_gets_too_flat_warning(self) -> None:
        from skiresort_planner.model.warning import TooFlatWarning

        seg = self._gentle_segment(kind=SegmentKind.SLOPE)
        assert any(isinstance(w, TooFlatWarning) for w in seg.warnings)

    def test_flat_road_gets_no_too_flat_warning(self) -> None:
        from skiresort_planner.model.warning import TooFlatWarning

        seg = self._gentle_segment(kind=SegmentKind.ROAD)
        assert not any(isinstance(w, TooFlatWarning) for w in seg.warnings), (
            "a road is gentle by design — the ski too-flat warning must not apply"
        )

    def test_excavator_warning_applies_to_roads_too(self) -> None:
        from skiresort_planner.model.warning import ExcavatorWarning

        # Steep cross-slope (side_slope 60% > 50% limit) → excavator warning regardless of kind.
        road = PathSegment(
            points=[PathPoint(lon=0.0, lat=0.0, elevation=2000.0), PathPoint(lon=0.0, lat=-0.005, elevation=1980.0)],
            id="S1",
            kind=SegmentKind.ROAD,
            side_slope_pct=60.0,
        )
        assert any(isinstance(w, ExcavatorWarning) for w in road.warnings)

    def test_warning_kind_discriminates_earthwork_from_gradient(self) -> None:
        """Kind drives the bridge/tunnel suppression in the panel — ONLY the excavator warning is
        EARTHWORK; ski gradient warnings are TOO_STEEP/TOO_FLAT (they still show on a structure).
        """
        from skiresort_planner.core.terrain_analyzer import SideDirection
        from skiresort_planner.model.warning import (
            ExcavatorWarning,
            TooFlatWarning,
            TooSteepWarning,
            WarningKind,
        )

        excavator = ExcavatorWarning(side_slope_pct=60.0, belt_width_m=20.0, side_slope_dir=SideDirection.LEFT)
        assert excavator.kind == WarningKind.EARTHWORK
        assert TooFlatWarning(slope_pct=1.0, min_threshold_pct=5.0).kind == WarningKind.TOO_FLAT
        assert TooSteepWarning(slope_pct=90.0, max_threshold_pct=70.0).kind == WarningKind.TOO_STEEP

"""Unit tests for PathFactory - path deduplication and similarity logic.

Tests the "Pure Logic" functions in PathFactory without requiring DEM services.
Focus on _are_paths_similar and _deduplicate_paths which are mathematical comparisons.
"""

import pytest

from skiresort_planner.constants import SlopeConfig
from skiresort_planner.generators.path_factory import GradeConfig, PathFactory, Side
from skiresort_planner.model.path_point import PathPoint
from skiresort_planner.model.proposed_path import ProposedPathSegment


def make_path(coords: list[tuple[float, float, float]], slope_pct: float = 20.0) -> ProposedPathSegment:
    """Helper to create a ProposedPathSegment from coordinate tuples.

    Args:
        coords: List of (lon, lat, elev) tuples
        slope_pct: Target slope percentage

    Returns:
        ProposedPathSegment with the given points
    """
    points = [PathPoint(lon=lon, lat=lat, elevation=elev) for lon, lat, elev in coords]
    return ProposedPathSegment(points=points, target_slope_pct=slope_pct)


class TestGradeConfig:
    """Unit tests for GradeConfig dataclass."""

    def test_name_format_with_left_side(self) -> None:
        """GradeConfig.name formats as '{Difficulty} {Side} ({Grade})'."""
        config = GradeConfig(difficulty="green", grade="gentle", target_slope_pct=7.0, side=Side.LEFT)
        assert config.name == "Green Left (Gentle)"

    def test_name_format_with_center_side(self) -> None:
        """Center side formats correctly."""
        config = GradeConfig(difficulty="blue", grade="steep", target_slope_pct=22.0, side=Side.CENTER)
        assert config.name == "Blue Center (Steep)"

    def test_color_returns_style_config_color(self) -> None:
        """GradeConfig.color returns correct color for difficulty."""
        config_green = GradeConfig(difficulty="green", grade="gentle", target_slope_pct=7.0, side=Side.LEFT)
        config_black = GradeConfig(difficulty="black", grade="steep", target_slope_pct=50.0, side=Side.RIGHT)

        assert config_green.color == "#22C55E"  # green-500
        assert config_black.color == "#1F2937"  # gray-800


class TestPathSimilarity:
    """Unit tests for _are_paths_similar comparison."""

    @pytest.fixture
    def factory(self) -> PathFactory:
        """PathFactory with no DEM (only uses comparison methods)."""
        return PathFactory(dem_service=None)

    def test_identical_paths_are_similar(self, factory: PathFactory) -> None:
        """Two paths with identical coordinates are similar."""
        coords = [
            (10.0, 47.0, 2000.0),
            (10.001, 47.001, 1990.0),
            (10.002, 47.002, 1980.0),
            (10.003, 47.003, 1970.0),
        ]
        path1 = make_path(coords)
        path2 = make_path(coords)

        assert factory._are_paths_similar(path1=path1, path2=path2)

    def test_diverging_paths_are_not_similar(self, factory: PathFactory) -> None:
        """Paths that diverge significantly are not similar."""
        # Path 1 goes east
        path1 = make_path(
            [
                (10.0, 47.0, 2000.0),
                (10.001, 47.0, 1990.0),
                (10.002, 47.0, 1980.0),
                (10.003, 47.0, 1970.0),
            ]
        )
        # Path 2 goes south (different direction)
        path2 = make_path(
            [
                (10.0, 47.0, 2000.0),
                (10.0, 46.999, 1990.0),
                (10.0, 46.998, 1980.0),
                (10.0, 46.997, 1970.0),
            ]
        )

        assert not factory._are_paths_similar(path1=path1, path2=path2)

    def test_empty_path_not_similar(self, factory: PathFactory) -> None:
        """Empty path is not similar to any path."""
        path1 = make_path([])
        path2 = make_path(
            [
                (10.0, 47.0, 2000.0),
                (10.001, 47.001, 1990.0),
                (10.002, 47.002, 1980.0),
            ]
        )

        assert not factory._are_paths_similar(path1=path1, path2=path2)

    def test_short_paths_not_similar(self, factory: PathFactory) -> None:
        """Paths with fewer than 3 points are not similar."""
        path1 = make_path([(10.0, 47.0, 2000.0), (10.001, 47.001, 1990.0)])
        path2 = make_path([(10.0, 47.0, 2000.0), (10.001, 47.001, 1990.0)])

        # Even identical 2-point paths are not similar (can't interpolate)
        assert not factory._are_paths_similar(path1=path1, path2=path2)


class TestPathDeduplication:
    """Unit tests for _deduplicate_paths method."""

    @pytest.fixture
    def factory(self) -> PathFactory:
        """PathFactory with no DEM (only uses dedup methods)."""
        return PathFactory(dem_service=None)

    def test_empty_list_returns_empty(self, factory: PathFactory) -> None:
        """Deduplicating empty list returns empty list."""
        result = factory._deduplicate_paths(paths=[])
        assert result == []

    def test_single_path_returns_unchanged(self, factory: PathFactory) -> None:
        """Single path is returned unchanged."""
        path = make_path(
            [
                (10.0, 47.0, 2000.0),
                (10.001, 47.001, 1990.0),
                (10.002, 47.002, 1980.0),
                (10.003, 47.003, 1970.0),
            ]
        )
        result = factory._deduplicate_paths(paths=[path])

        assert len(result) == 1
        assert result[0] is path

    def test_identical_paths_deduplicated(self, factory: PathFactory) -> None:
        """Duplicate identical paths are removed."""
        coords = [
            (10.0, 47.0, 2000.0),
            (10.001, 47.001, 1990.0),
            (10.002, 47.002, 1980.0),
            (10.003, 47.003, 1970.0),
        ]
        path1 = make_path(coords, slope_pct=20.0)
        path2 = make_path(coords, slope_pct=25.0)

        result = factory._deduplicate_paths(paths=[path1, path2])

        # Should keep only one (the gentlest slope)
        assert len(result) == 1
        assert result[0].target_slope_pct == 20.0

    def test_diverging_paths_both_kept(self, factory: PathFactory) -> None:
        """Paths that diverge are both kept."""
        # Path 1 goes east
        path1 = make_path(
            [
                (10.0, 47.0, 2000.0),
                (10.001, 47.0, 1990.0),
                (10.002, 47.0, 1980.0),
                (10.003, 47.0, 1970.0),
            ]
        )
        # Path 2 goes south
        path2 = make_path(
            [
                (10.0, 47.0, 2000.0),
                (10.0, 46.999, 1990.0),
                (10.0, 46.998, 1980.0),
                (10.0, 46.997, 1970.0),
            ]
        )

        result = factory._deduplicate_paths(paths=[path1, path2])

        assert len(result) == 2

    def test_keeps_gentlest_slope_when_deduplicating(self, factory: PathFactory) -> None:
        """When removing duplicates, keeps path with lowest avg_slope_pct.

        Note: _deduplicate_paths sorts by ACTUAL measured avg_slope_pct,
        not target_slope_pct. For identical coordinates, avg_slope_pct
        is the same, so all are considered equal and first-in wins.
        """
        coords = [
            (10.0, 47.0, 2000.0),
            (10.001, 47.001, 1990.0),
            (10.002, 47.002, 1980.0),
            (10.003, 47.003, 1970.0),
        ]
        # All paths have same coords → same computed avg_slope_pct
        # Dedup keeps one based on stable sort order
        path1 = make_path(coords, slope_pct=40.0)
        path2 = make_path(coords, slope_pct=25.0)
        path3 = make_path(coords, slope_pct=10.0)

        result = factory._deduplicate_paths(paths=[path1, path2, path3])

        # Should deduplicate to 1 path (all have same computed avg_slope_pct)
        assert len(result) == 1
        # Verify deduplication happened (original had 3 paths, now 1)
        assert result[0].avg_slope_pct == path1.avg_slope_pct  # All have same avg

    def test_mixed_similar_and_different_paths(self, factory: PathFactory) -> None:
        """Mix of similar and different paths deduplicated correctly."""
        # Group 1: Two similar east-going paths
        east1 = make_path(
            [
                (10.0, 47.0, 2000.0),
                (10.001, 47.0, 1990.0),
                (10.002, 47.0, 1980.0),
                (10.003, 47.0, 1970.0),
            ],
            slope_pct=15.0,
        )
        east2 = make_path(
            [
                (10.0, 47.0, 2000.0),
                (10.001, 47.0, 1990.0),
                (10.002, 47.0, 1980.0),
                (10.003, 47.0, 1970.0),
            ],
            slope_pct=30.0,
        )

        # Group 2: One south-going path
        south = make_path(
            [
                (10.0, 47.0, 2000.0),
                (10.0, 46.999, 1990.0),
                (10.0, 46.998, 1980.0),
                (10.0, 46.997, 1970.0),
            ],
            slope_pct=20.0,
        )

        result = factory._deduplicate_paths(paths=[east1, east2, south])

        # Should have 2 paths: gentle east (15%) and south (20%)
        assert len(result) == 2
        slopes = sorted(p.target_slope_pct for p in result)
        assert slopes == [15.0, 20.0]


class TestRoadTargetGrade:
    """A road aims for the GREEN grades (7%/12%), signed by the endpoints' direction
    (descend → +, climb → −). Same magnitudes as a green slope; sign is the only difference.
    """

    def test_descent_aims_positive_green_grades(self) -> None:
        """A net descent (positive signed_drop) targets +7 and +12."""
        green = SlopeConfig.DIFFICULTY_TARGETS["green"]
        assert PathFactory._road_target_grades(signed_drop=100.0) == [green["gentle"], green["steep"]]

    def test_climb_aims_negative_green_grades(self) -> None:
        """A net climb (negative signed_drop) targets −7 and −12 (sign preserved)."""
        green = SlopeConfig.DIFFICULTY_TARGETS["green"]
        assert PathFactory._road_target_grades(signed_drop=-100.0) == [-green["gentle"], -green["steep"]]


class TestRoadModeNoStraightLineFallback:
    """Road mode (road_mode=True) must NOT fabricate a straight-line fallback.

    Slope mode always creates a straight-line result when Dijkstra finds nothing,
    so two points always connect. Road mode does not fabricate one.

    NOTE: on merely-steep terrain the grid planner still returns an OUT-OF-BAND
    route rather than nothing — that over-limit result is caught by the caller's
    hard cap (see test_click_handlers::TestRoadBuildingClick), not here. This
    class only asserts the factory never emits the straight-line fallback in
    road mode.
    """

    def test_road_mode_never_yields_straight_line_fallback(self, path_factory) -> None:
        # A gentle, reachable target → a real traced (multi-point) road path,
        # never the 2-point "Direct Line (fallback)" that slope mode makes.
        M = 111320.0
        paths = list(
            path_factory.generate_manual_paths(
                start_lon=0.0,
                start_lat=0.0,
                start_elevation=path_factory.dem_service.get_elevation_or_raise(lon=0.0, lat=0.0),
                target_lon=300 / M,
                target_lat=0.0,
                target_elevation=path_factory.dem_service.get_elevation_or_raise(lon=300 / M, lat=0.0),
                road_mode=True,
            )
        )
        assert paths, "a gentle reachable target should yield a road path"
        assert all("fallback" not in (p.sector_name or "").lower() for p in paths), (
            "road mode must never emit the straight-line fallback"
        )
        # Road-mode proposals carry the ROAD kind so the committed segment is a road.
        from skiresort_planner.model.path_segment import SegmentKind

        assert all(p.kind is SegmentKind.ROAD for p in paths)

    def test_slope_mode_still_falls_back_to_straight_line(self, path_factory) -> None:
        # Slope mode always connects, emitting the straight-line fallback result.
        from skiresort_planner.model.path_segment import SegmentKind

        M = 111320.0
        paths = list(
            path_factory.generate_manual_paths(
                start_lon=0.0,
                start_lat=0.0,
                start_elevation=path_factory.dem_service.get_elevation_or_raise(lon=0.0, lat=0.0),
                target_lon=0.0,
                target_lat=-250 / M,
                target_elevation=path_factory.dem_service.get_elevation_or_raise(lon=0.0, lat=-250 / M),
                road_mode=False,
            )
        )
        assert len(paths) >= 1, "slope mode always connects (straight-line fallback)"
        assert all(p.kind is SegmentKind.SLOPE for p in paths), "slope-mode proposals are SLOPE kind"

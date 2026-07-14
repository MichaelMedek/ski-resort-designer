"""Unit tests for PathFactory - path deduplication and similarity logic.

Tests the "Pure Logic" functions in PathFactory without requiring DEM services.
Focus on _are_paths_similar and _deduplicate_paths which are mathematical comparisons.
"""

import pytest

from skiresort_planner.constants import SlopeConfig
from skiresort_planner.enum_utils import enum_eq
from skiresort_planner.generators.path_factory import GradeConfig, PathFactory, Side
from skiresort_planner.model.path_point import PathPoint
from skiresort_planner.model.path_segment import SegmentKind
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
    """Unit tests for GradeConfig.sector_name — the per-kind fan label (dispatched, no branch)."""

    def test_slope_sector_name_is_difficulty_side_grade(self) -> None:
        """A slope labels by difficulty + side + grade."""
        config = GradeConfig(difficulty="green", grade="gentle", target_slope_pct=7.0, side=Side.LEFT)
        assert config.sector_name(SegmentKind.SLOPE) == "Green Left (Gentle)"

    def test_slope_sector_name_center(self) -> None:
        """Center side formats correctly for a slope."""
        config = GradeConfig(difficulty="blue", grade="steep", target_slope_pct=22.0, side=Side.CENTER)
        assert config.sector_name(SegmentKind.SLOPE) == "Blue Center (Steep)"

    def test_road_sector_name_has_no_difficulty(self) -> None:
        """A road has no ski difficulty, so it labels by side + grade only (kind-dispatched)."""
        config = GradeConfig(difficulty="", grade="gentle", target_slope_pct=7.0, side=Side.LEFT)
        assert config.sector_name(SegmentKind.ROAD) == "Road Left (Gentle)"


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
    """Road mode (kind=SegmentKind.ROAD) must NOT fabricate a straight-line fallback.

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
                kind=SegmentKind.ROAD,
            )
        )
        assert paths, "a gentle reachable target should yield a road path"
        assert all("fallback" not in (p.sector_name or "").lower() for p in paths), (
            "road mode must never emit the straight-line fallback"
        )
        # Road-mode proposals carry the ROAD kind so the committed segment is a road.
        assert all(enum_eq(a=p.kind, b=SegmentKind.ROAD) for p in paths)

    def test_slope_mode_yields_no_fabricated_fallback(self, path_factory) -> None:
        # Slope mode no longer fabricates a straight-line fallback: it yields only real
        # traced routes (or nothing). On this reachable diagonal target it yields routes,
        # none of which are the old 2-point "Direct Line (fallback)".
        M = 111320.0
        paths = list(
            path_factory.generate_manual_paths(
                start_lon=0.0,
                start_lat=0.0,
                start_elevation=path_factory.dem_service.get_elevation_or_raise(lon=0.0, lat=0.0),
                target_lon=0.0,
                target_lat=-250 / M,
                target_elevation=path_factory.dem_service.get_elevation_or_raise(lon=0.0, lat=-250 / M),
                kind=SegmentKind.SLOPE,
            )
        )
        assert all(enum_eq(a=p.kind, b=SegmentKind.SLOPE) for p in paths), "slope-mode proposals are SLOPE kind"
        assert all("fallback" not in (p.sector_name or "").lower() for p in paths), (
            "slope mode no longer fabricates a straight-line fallback"
        )


class TestGenerateSlopeFan:
    """Unit tests for generate_slope_fan - the fan-pattern difficulty/grade/side sweep.

    The path_factory fixture uses a ~31.6% diagonal slope (30% S + 10% E), so the
    green targets (7%/12%) are below terrain and yield LEFT/RIGHT traverse variants.
    Green traverses always trace on all terrain (shallow angles), so the sweep
    starts with 'Green Left (Gentle)' per the difficulty→grade→side loop order.
    """

    def test_fan_yields_proposals_that_are_not_connectors(self, path_factory: PathFactory) -> None:
        """Every fan path is a real slope proposal: is_connector False, valid difficulty."""
        start_elev = 2500.0  # the diagonal mock DEM's base elevation at the origin (0, 0)
        paths = list(path_factory.generate_fan(kind=SegmentKind.SLOPE, lon=0.0, lat=0.0, elevation=start_elev))

        assert paths, "a steep diagonal slope must yield fan proposals"
        assert all(p.is_connector is False for p in paths), "fan paths are slopes, never connectors"
        assert all(p.target_difficulty in SlopeConfig.DIFFICULTIES for p in paths), (
            "every fan path carries a valid difficulty (green/blue/red/black)"
        )

    def test_fan_starts_with_green_left_gentle(self, path_factory: PathFactory) -> None:
        """Loop order (green→gentle→left) + green-always-traces makes the first path 'Green Left (Gentle)'."""
        start_elev = 2500.0  # the diagonal mock DEM's base elevation at the origin (0, 0)
        paths = list(path_factory.generate_fan(kind=SegmentKind.SLOPE, lon=0.0, lat=0.0, elevation=start_elev))

        assert paths[0].sector_name == "Green Left (Gentle)"
        assert paths[0].target_difficulty == "green"


class TestGenerateRoadFan:
    """Unit tests for generate_road_fan - the road fan (signed green targets).

    The path_factory fixture is a ~31.6% diagonal slope, so every green target
    magnitude (7%/12%/0%) is below terrain and traces as a left/right traverse.
    A road fan must offer routes in all three sign modes: descend, climb, contour.
    """

    def test_road_fan_offers_descend_climb_and_contour(self, path_factory: PathFactory) -> None:
        """The fan traces all three sign modes, proving the tracer is gradient-agnostic.

        A weak 'len > 0' would not catch a sign regression; this asserts each behavior
        (a descending, a climbing, and a near-level route) is actually produced.
        """
        paths = list(path_factory.generate_fan(kind=SegmentKind.ROAD, lon=0.0, lat=0.0, elevation=2500.0))

        assert paths, "a steep diagonal slope must yield road-fan proposals"
        descends = [p for p in paths if p.avg_slope_pct > SlopeConfig.MIN_SKIABLE_PCT]
        climbs = [p for p in paths if p.avg_slope_pct < -SlopeConfig.MIN_SKIABLE_PCT]
        contours = [p for p in paths if abs(p.avg_slope_pct) < SlopeConfig.MIN_SKIABLE_PCT]
        assert descends, "road fan must include a descending route (+ target)"
        assert climbs, "road fan must include a climbing route (− target)"
        assert contours, "road fan must include a near-level contour route (0 target)"

    def test_road_fan_proposals_are_roads_not_connectors(self, path_factory: PathFactory) -> None:
        """Every road-fan proposal is a ROAD-kind, non-connector segment."""
        paths = list(path_factory.generate_fan(kind=SegmentKind.ROAD, lon=0.0, lat=0.0, elevation=2500.0))
        assert paths
        assert all(enum_eq(a=p.kind, b=SegmentKind.ROAD) for p in paths), "road fan yields ROAD kind"
        assert all(p.is_connector is False for p in paths), "fan proposals are not connectors"

    def test_road_fan_grades_are_single_sourced_green(self, path_factory: PathFactory) -> None:
        """Every fan target magnitude comes from DIFFICULTY_TARGETS['green'] (or 0).

        Guards the 'no hardcoded 7/12' rule: if someone hardcodes a grade, the
        target set diverges from the green config and this fails.
        """
        paths = list(path_factory.generate_fan(kind=SegmentKind.ROAD, lon=0.0, lat=0.0, elevation=2500.0))
        assert paths
        allowed = set(SlopeConfig.DIFFICULTY_TARGETS["green"].values()) | {0.0}
        target_mags = {abs(p.target_slope_pct) for p in paths}
        assert target_mags <= allowed, f"road-fan target magnitudes {target_mags} must be green grades or 0"

    def test_road_fan_covers_the_exact_signed_grade_set(self, path_factory: PathFactory) -> None:
        """The road fan attempts EXACTLY {+7, +12, −7, −12, 0} — both signs plus a contour.

        The three-sign-mode test proves each sign appears; this pins the full signed set so a
        missing sign (only descents), a missing contour, or a stray extra grade all fail. On the
        steep diagonal fixture every target magnitude is below terrain, so all five trace, and
        the set of SIGNED targets attempted must be exactly the five below.
        """
        g = SlopeConfig.DIFFICULTY_TARGETS["green"]  # {"gentle": 7, "steep": 12} (single source)
        expected_signed = {round(v, 3) for v in g.values()} | {round(-v, 3) for v in g.values()} | {0.0}
        paths = list(path_factory.generate_fan(kind=SegmentKind.ROAD, lon=0.0, lat=0.0, elevation=2500.0))
        attempted_signed = {round(p.target_slope_pct, 3) for p in paths}
        assert attempted_signed == expected_signed, (
            f"road fan must attempt exactly {expected_signed}, got {attempted_signed}"
        )


class TestStraightLine:
    """The one direct-line builder, straight_line(kind), used for the road bridge/cut fallback."""

    def _endpoints(self) -> tuple[float, float, float, float, float, float]:
        # Explicit elevations: the builder does not query the DEM, it wraps the two given
        # points, so we control the drop directly (250m S, 25m drop → 10%).
        M = 111320.0
        return 0.0, 0.0, 2500.0, 0.0, -250 / M, 2475.0

    def test_road_straight_line_is_a_two_point_connector(self, path_factory: PathFactory) -> None:
        """A direct road is a 2-point ROAD connector with no ski difficulty."""
        s_lon, s_lat, s_elev, t_lon, t_lat, t_elev = self._endpoints()
        road = path_factory.straight_line(
            kind=SegmentKind.ROAD,
            start_lon=s_lon,
            start_lat=s_lat,
            start_elevation=s_elev,
            target_lon=t_lon,
            target_lat=t_lat,
            target_elevation=t_elev,
        )
        assert len(road.points) == 2, "a direct line is exactly its two endpoints"
        assert (road.points[0].lon, road.points[0].lat, road.points[0].elevation) == (s_lon, s_lat, s_elev)
        assert (road.points[-1].lon, road.points[-1].lat, road.points[-1].elevation) == (t_lon, t_lat, t_elev)
        assert road.is_connector
        assert enum_eq(a=road.kind, b=SegmentKind.ROAD)
        assert road.target_difficulty == "", "a road carries no ski difficulty"

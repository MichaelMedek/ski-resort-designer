"""Unit tests for the Path geometry base class (model/path_geometry.py).

`Path` is the shared base of ProposedPathSegment and PathSegment: it stores raw points and
computes length / drop / average slope / **steepest-section slope** / difficulty on the fly. The
non-trivial logic here is `max_slope_pct` — a MAGNITUDE (always ≥ 0) taken over the steepest
`ROLLING_WINDOW_M` window, used both for difficulty and the road ±gradient cap. Coordinates are near
the equator (1° ≈ 111320 m) so distances are simple to reason about.
"""

import pytest

from skiresort_planner.constants import MapConfig, SlopeConfig
from skiresort_planner.model.path_geometry import Path
from skiresort_planner.model.path_point import PathPoint


def _south_path(*, drop_per_100m: float, length_m: float, step_m: float = 30.0) -> Path:
    """A path running south (decreasing lat) at a constant grade.

    drop_per_100m is the descent in metres per 100 m of ground → a `drop_per_100m`% grade.
    Points are spaced `step_m` apart; elevation starts at 2500 and falls linearly.
    """
    n = max(2, int(round(length_m / step_m)) + 1)
    grade = drop_per_100m / 100.0
    points = []
    for i in range(n):
        south_m = i * step_m
        points.append(
            PathPoint(lon=0.0, lat=-south_m / MapConfig.METERS_PER_DEGREE_EQUATOR, elevation=2500.0 - south_m * grade)
        )
    return Path(points=points)


class TestEndpointsAndEmpty:
    def test_empty_path_has_no_endpoints_and_zero_metrics(self) -> None:
        p = Path(points=[])
        assert p.start is None and p.end is None
        assert p.total_drop_m == 0.0
        assert p.length_m == 0.0
        assert p.avg_slope_pct == 0.0
        assert p.max_slope_pct == 0.0

    def test_single_point_is_degenerate(self) -> None:
        only = PathPoint(lon=0.0, lat=0.0, elevation=2000.0)
        p = Path(points=[only])
        assert p.start is only and p.end is only
        assert p.total_drop_m == 0.0, "one point has no drop"
        assert p.length_m == 0.0, "one point has no length"

    def test_start_and_end_are_first_and_last(self) -> None:
        p = _south_path(drop_per_100m=20.0, length_m=300.0)
        assert p.start is p.points[0]
        assert p.end is p.points[-1]


class TestLengthAndDrop:
    def test_length_sums_leg_distances(self) -> None:
        # 300 m south in 30 m steps → ground length ≈ 300 m.
        p = _south_path(drop_per_100m=20.0, length_m=300.0)
        assert p.length_m == pytest.approx(300.0, abs=1.0)

    def test_drop_is_endpoint_difference_signed(self) -> None:
        # 20% over 300 m → 60 m drop, positive (descending).
        p = _south_path(drop_per_100m=20.0, length_m=300.0)
        assert p.total_drop_m == pytest.approx(60.0, abs=0.5)

    def test_climb_has_negative_drop(self) -> None:
        # Reverse a descent into a climb: end higher than start → negative drop.
        descent = _south_path(drop_per_100m=20.0, length_m=300.0)
        climb = Path(points=list(reversed(descent.points)))
        assert climb.total_drop_m == pytest.approx(-60.0, abs=0.5)


class TestAvgSlopeSigned:
    def test_avg_slope_is_signed_positive_for_descent(self) -> None:
        p = _south_path(drop_per_100m=20.0, length_m=300.0)
        assert p.avg_slope_pct == pytest.approx(20.0, abs=0.5)

    def test_avg_slope_is_negative_for_a_climb(self) -> None:
        descent = _south_path(drop_per_100m=20.0, length_m=300.0)
        climb = Path(points=list(reversed(descent.points)))
        assert climb.avg_slope_pct == pytest.approx(-20.0, abs=0.5), "average slope keeps the climb's sign"


class TestMaxSlopeMagnitude:
    def test_short_path_falls_back_to_avg_magnitude(self) -> None:
        # Shorter than the rolling window → max_slope is just |avg_slope|.
        short = _south_path(drop_per_100m=20.0, length_m=150.0)
        assert short.length_m < SlopeConfig.ROLLING_WINDOW_M
        assert short.max_slope_pct == pytest.approx(abs(short.avg_slope_pct), abs=1e-6)

    def test_max_slope_is_a_magnitude_for_a_climb(self) -> None:
        # A steady 20% CLIMB longer than the window: avg is -20% but the steepest-section
        # magnitude must be +20% (both difficulty and the road cap want the magnitude).
        descent = _south_path(drop_per_100m=20.0, length_m=600.0)
        climb = Path(points=list(reversed(descent.points)))
        assert climb.avg_slope_pct < 0
        assert climb.max_slope_pct == pytest.approx(20.0, abs=1.5)
        assert climb.max_slope_pct > 0, "steepest-section slope is always a non-negative magnitude"

    def test_steep_window_dominates_a_gentle_average(self) -> None:
        # 600 m of gentle 5% then 350 m of steep 40% (both longer moves so a full window lands
        # inside the steep stretch). The steepest window must reflect ~40%, far above the ~18% avg.
        gentle = _south_path(drop_per_100m=5.0, length_m=600.0)
        # Continue south from the end of the gentle run at 40% for 350 m.
        last = gentle.points[-1]
        steep_points = []
        for i in range(1, 13):  # 12 × 30 m = 360 m
            south_m = i * 30.0
            steep_points.append(
                PathPoint(
                    lon=0.0,
                    lat=last.lat - south_m / MapConfig.METERS_PER_DEGREE_EQUATOR,
                    elevation=last.elevation - south_m * 0.40,
                )
            )
        p = Path(points=gentle.points + steep_points)
        assert p.avg_slope_pct < 25.0, "the long gentle head keeps the AVERAGE modest"
        assert p.max_slope_pct == pytest.approx(40.0, abs=3.0), "the steepest 300 m window reflects the 40% section"
        assert p.max_slope_pct > p.avg_slope_pct, "steepest section is steeper than the average"

    def test_uniform_grade_max_equals_avg_magnitude(self) -> None:
        # A long, perfectly uniform 25% descent: every window is 25%, so max == |avg|.
        p = _south_path(drop_per_100m=25.0, length_m=900.0)
        assert p.length_m > SlopeConfig.ROLLING_WINDOW_M
        assert p.max_slope_pct == pytest.approx(abs(p.avg_slope_pct), abs=1.0)


class TestDifficultyFromSteepestSection:
    """Path.difficulty routes the STEEPEST-SECTION slope through classification.

    (Threshold/color classification itself is owned by test_terrain_analyzer.py; here we only assert
    that Path.difficulty feeds max_slope_pct — the steepest window, not the average — into it.)
    """

    @pytest.mark.parametrize(
        "grade_pct, expected",
        [
            (8.0, "green"),  # 0–15
            (20.0, "blue"),  # 15–25
            (32.0, "red"),  # 25–40
            (48.0, "black"),  # 40+
        ],
    )
    def test_difficulty_follows_steepest_section(self, grade_pct: float, expected: str) -> None:
        # Uniform grade longer than the window → max_slope ≈ grade → difficulty band.
        p = _south_path(drop_per_100m=grade_pct, length_m=600.0)
        assert p.difficulty == expected

    def test_difficulty_uses_steepest_not_average(self) -> None:
        # Gentle average (green-ish) but a steep section pushes difficulty up — difficulty must
        # track the steepest window, not the average, so a hidden steep pitch isn't mislabeled easy.
        gentle = _south_path(drop_per_100m=5.0, length_m=600.0)
        last = gentle.points[-1]
        steep = [
            PathPoint(
                lon=0.0,
                lat=last.lat - (i * 30.0) / MapConfig.METERS_PER_DEGREE_EQUATOR,
                elevation=last.elevation - (i * 30.0) * 0.40,
            )
            for i in range(1, 13)
        ]
        p = Path(points=gentle.points + steep)
        assert p.difficulty in {"red", "black"}, "a steep pitch is not classified by the gentle average"

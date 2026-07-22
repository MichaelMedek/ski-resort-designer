"""Acceptance tests for the custom-path planner's elevation-budget anchor.

For each FLAT conftest terrain and two target setups, we construct the IDEAL grade-g path, ask the
planner for the REAL path to the same target, and push BOTH through one geometry framework:

    _assert_path_geometry(path, expected_len, tol)

which requires the path to
  1. have horizontal length ≈ 100·D/g (the length that sheds drop D at grade g), and
  2. sit on the straight endpoint→endpoint elevation line at 25/50/75% along its length.

The ideal path is checked at a tight 1% (it descends uniformly by construction, so this proves the
framework asserts correctly); the real planner path is checked at 10% (the actual acceptance test).

Setups (target placed relative to the terrain fall line, steepness S, target grade g, drop D):
  - "TRAVERSE": bearing offset arccos(g/S), distance 100·D/g — a STRAIGHT chord already holds g. Ideal = line.
  - "FALLLINE": straight down the fall line, drop D at grade S — too steep to go straight at g, so the ideal
    is a 2-leg right-then-left serpentine of length 100·D/g reaching that fall-line target.
  - "MATCHED": g == S — the fall line itself holds the grade, so the direct straight line is optimal.

Every case can also be rendered as a 3D plot (hill + ideal + real path).
"""

import itertools
import math

import numpy as np
import pytest

from skiresort_planner.constants import PACKAGE_DIR, GeometricTuningConfig
from skiresort_planner.core.geo_calculator import GeoCalculator
from skiresort_planner.core.terrain_analyzer import TerrainAnalyzer
from skiresort_planner.generators.connection_planners import GradientMode, LeastCostPathPlanner
from skiresort_planner.model.path_point import PathPoint

# Arbitrary start well inside every mock-DEM bound (±1–2°) and off the cone's singular summit at (0,0).
_START_LON = 0.02
_START_LAT = 0.02
_PLOT_DIR = PACKAGE_DIR.parent / "output" / "path_geometry_plots"

# Three FLAT conftest fixtures only. (fixture name, fall-line steepness %).
_PLANAR_TERRAINS = [
    ("mock_dem_blue_slope", 20.0),
    ("mock_dem_black_slope", 45.0),
    ("mock_dem_red_slope_diagonal", math.hypot(30.0, 10.0)),  # ≈31.6% diagonal fall line
]

# Target grades (%): every 5% step. FALLLINE/TRAVERSE use grades strictly below the terrain steepness S
# (you can only traverse to go GENTLER than the fall line); MATCHED uses g == S (fall line already holds
# it → straight line is optimal). Drops (m): 50 / 200 / 500. ALL combinations parametrized.
_GRADES = list(range(5, 55, 5))  # 5, 10, 15, …, 50
_DROPS_M = [50.0, 200.0, 500.0]

_CASES = [
    pytest.param(fname, steep, float(g), drop, setup, id=f"{fname}-{setup}-g{g:g}-D{drop:g}")
    for (fname, steep), setup, g, drop in itertools.product(
        _PLANAR_TERRAINS, ("FALLLINE", "TRAVERSE"), _GRADES, _DROPS_M
    )
    if g < steep
] + [
    pytest.param(fname, steep, steep, drop, "MATCHED", id=f"{fname}-MATCHED-g{steep:g}-D{drop:g}")
    for (fname, steep), drop in itertools.product(_PLANAR_TERRAINS, _DROPS_M)
]

# ----------------------------------------------------------------------------- geometry framework


def _assert_path_geometry(points: list[PathPoint], expected_len_m: float, grade: float, tol_frac: float) -> None:
    """Require: length ≈ 100·D/g; and at 25/50/75% along the path the elevation lost ≈ grade × distance
    (i.e. the point sits on the straight descent line). Tolerances are `tol_frac` of the expected value.
    """
    cum = np.asarray(PathPoint.cumulative_distances(points), dtype=float)
    total = float(cum[-1])
    start_e = points[0].elevation

    assert total == pytest.approx(expected_len_m, rel=tol_frac), (
        f"length {total:.1f}m vs expected {expected_len_m:.1f}m (tol {tol_frac:.0%})"
    )
    for frac in (0.25, 0.50, 0.75):
        d = frac * total
        i = int(np.argmin(np.abs(cum - d)))  # nearest dense point (~7m spacing) to the checkpoint
        lost_actual = start_e - points[i].elevation
        lost_ideal = grade * cum[i] / 100.0  # grade % over the distance actually travelled to that point
        assert lost_actual == pytest.approx(lost_ideal, rel=tol_frac), (
            f"at {frac:.0%} along ({cum[i]:.0f}m): lost {lost_actual:.1f}m vs ideal {lost_ideal:.1f}m (tol {tol_frac:.0%})"
        )


# ----------------------------------------------------------------------------- setup + path builders


def _fall_bearing(dem) -> float:
    return TerrainAnalyzer(dem=dem).compute_gradient(lon=_START_LON, lat=_START_LAT).bearing_deg


def _setup_target(dem, steep: float, grade: float, drop: float, setup: str) -> tuple[float, float]:
    """Target (lon, lat) per setup:
    - "TRAVERSE": offset chord holding g, at 100·D/g (offset arccos(g/S) from the fall line).
    - "MATCHED": grade g==S → offset arccos(1)=0 → straight down the fall line at 100·D/g (=D/S).
    - "FALLLINE": straight down the fall line, close (drop D at the terrain grade S).
    """
    if setup in ("TRAVERSE", "MATCHED"):
        bearing = _fall_bearing(dem) + math.degrees(math.acos(grade / steep))
        dist = 100.0 * drop / grade
    else:
        assert steep == grade
        bearing = _fall_bearing(dem)
        dist = 100.0 * drop / steep
    return GeoCalculator.destination(lon=_START_LON, lat=_START_LAT, bearing_deg=bearing, distance_m=dist)


def _build_ideal_path(
    dem, start_elev: float, target_lon: float, target_lat: float, grade: float, setup: str
) -> list[PathPoint]:
    """The IDEAL grade-g path, descending uniformly: TRAVERSE/MATCHED = straight line to the target;
    FALLLINE = a 2-leg right-then-left serpentine of length 100·D/g reaching the fall-line target.
    """
    target_elev = dem.get_elevation(lon=target_lon, lat=target_lat)
    drop = start_elev - target_elev
    total_len = 100.0 * drop / grade  # length that sheds `drop` at grade g

    def _pt(along: float, lon: float, lat: float) -> PathPoint:
        return PathPoint(lon=lon, lat=lat, elevation=start_elev - (along / total_len) * drop)

    # straight line to the target (MATCHED: fall-line chord holds g==S)
    if setup in ("TRAVERSE", "MATCHED"):
        n = 40
        points = [
            _pt(
                (i / n) * total_len,
                _START_LON + (target_lon - _START_LON) * (i / n),
                _START_LAT + (target_lat - _START_LAT) * (i / n),
            )
            for i in range(n + 1)
        ]
    # serpentine: leg 1 out to `across` while descending half; leg 2 back to the fall-line target.
    elif setup in ("FALLLINE"):
        bearing = _fall_bearing(dem)
        direct = GeoCalculator.haversine_distance_m(lat1=_START_LAT, lon1=_START_LON, lat2=target_lat, lon2=target_lon)
        half_len, half_fall = total_len / 2, direct / 2
        across = math.sqrt(max(half_len**2 - half_fall**2, 0.0))
        n_leg = 20
        points: list[PathPoint] = []
        for i in range(2 * n_leg + 1):
            along = total_len * i / (2 * n_leg)
            if along <= half_len:
                u = along / half_len
                fall, side = half_fall * u, across * u
            else:
                u = (along - half_len) / half_len
                fall, side = half_fall * (1 + u), across * (1 - u)
            pf = GeoCalculator.destination(lon=_START_LON, lat=_START_LAT, bearing_deg=bearing, distance_m=fall)
            lon, lat = GeoCalculator.destination(lon=pf[0], lat=pf[1], bearing_deg=bearing + 90.0, distance_m=side)
            points.append(_pt(along, lon, lat))
    # No fallbacks
    else:
        raise ValueError
    return points


def _real_path(dem, start_elev: float, target_lon: float, target_lat: float, grade: float) -> list[PathPoint] | None:
    path = LeastCostPathPlanner(dem_service=dem, terrain_analyzer=TerrainAnalyzer(dem=dem)).plan(
        start_lon=_START_LON,
        start_lat=_START_LAT,
        start_elevation=start_elev,
        target_lon=target_lon,
        target_lat=target_lat,
        target_elevation=dem.get_elevation(lon=target_lon, lat=target_lat),
        target_grade_pct=grade,
        smoothing_factor=GeometricTuningConfig.SLOPE_SMOOTHING_FACTOR,
        gradient_mode=GradientMode.DOWNHILL,
    )
    return path.points if path is not None else None


def _plot_3d(dem, ideal: list[PathPoint], real: list[PathPoint] | None, title: str, out_path) -> None:
    """3D hill surface around the paths + the ideal (green) and real (red) polylines; saved to `out_path`."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lons = [p.lon for p in ideal] + ([p.lon for p in real] if real else [])
    lats = [p.lat for p in ideal] + ([p.lat for p in real] if real else [])
    pad = 0.0015
    mlon, mlat = np.meshgrid(
        np.linspace(min(lons) - pad, max(lons) + pad, 40),
        np.linspace(min(lats) - pad, max(lats) + pad, 40),
    )
    z = dem.get_elevations(mlon.ravel(), mlat.ravel()).reshape(mlon.shape)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(mlon, mlat, z, alpha=0.3, cmap="terrain", linewidth=0)
    ax.plot(
        [p.lon for p in ideal],
        [p.lat for p in ideal],
        [p.elevation for p in ideal],
        color="green",
        linewidth=3,
        label="ideal",
    )
    if real:
        ax.plot(
            [p.lon for p in real],
            [p.lat for p in real],
            [p.elevation for p in real],
            color="red",
            linewidth=3,
            label="real (planner)",
        )
    ax.set_title(title)
    ax.legend()
    fig.savefig(out_path, dpi=90)
    plt.close(fig)


# ----------------------------------------------------------------------------- the test


class TestPathGeometry:
    """Ideal path (1%) proves the framework; the real planner path (10%) is the acceptance assertion."""

    @pytest.mark.parametrize(("fname", "steep", "grade", "drop", "setup"), _CASES)
    def test_ideal_and_real_paths(
        self, request, fname: str, steep: float, grade: float, drop: float, setup: str
    ) -> None:
        dem = request.getfixturevalue(fname)
        start_elev = dem.get_elevation(lon=_START_LON, lat=_START_LAT)
        target_lon, target_lat = _setup_target(dem, steep, grade, drop, setup)
        expected_len = 100.0 * (start_elev - dem.get_elevation(lon=target_lon, lat=target_lat)) / grade

        ideal = _build_ideal_path(dem, start_elev, target_lon, target_lat, grade, setup)
        assert ideal is not None
        real = _real_path(dem, start_elev, target_lon, target_lat, grade)
        assert real is not None, f"planner found no grade-{grade}% path on {fname} (setup {setup})"

        if _PLOT_DIR is not None:
            _PLOT_DIR.mkdir(parents=True, exist_ok=True)
            _plot_3d(
                dem,
                ideal,
                real,
                title=f"{fname} setup {setup} g={grade:g}% D={drop:g}m",
                out_path=_PLOT_DIR / f"{fname}_{setup}_g{grade:g}_D{drop:g}.png",
            )

        _assert_path_geometry(ideal, expected_len_m=expected_len, grade=grade, tol_frac=0.01)
        _assert_path_geometry(real, expected_len_m=expected_len, grade=grade, tol_frac=0.10)

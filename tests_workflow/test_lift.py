"""Unit tests for the Lift model (model/lift.py)."""

import pytest

from skiresort_planner.constants import MapConfig
from skiresort_planner.model.lift import Lift
from skiresort_planner.model.node import Node
from skiresort_planner.model.path_point import PathPoint

_PTS = [
    PathPoint(lon=10.0, lat=46.0, elevation=2000.0),
    PathPoint(lon=10.001, lat=46.001, elevation=2100.0),
]


def _valid_lift() -> Lift:
    """A minimal structurally-valid lift (2 terrain + 2 cable points, valid type)."""
    return Lift(
        id="L1",
        name="Test Lift",
        start_node_id="N1",
        end_node_id="N2",
        lift_type="chairlift",
        terrain_points=list(_PTS),
        pylons=[],
        cable_points=list(_PTS),
    )


class TestLiftPostInitValidation:
    def test_valid_lift_constructs(self) -> None:
        assert _valid_lift().lift_type == "chairlift"

    def test_invalid_lift_type_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid lift_type"):
            Lift(
                id="L1",
                name="x",
                start_node_id="N1",
                end_node_id="N2",
                lift_type="rocket",
                terrain_points=list(_PTS),
                pylons=[],
                cable_points=list(_PTS),
            )

    def test_too_few_terrain_points_raises(self) -> None:
        with pytest.raises(ValueError, match="at least 2 terrain_points"):
            Lift(
                id="L1",
                name="x",
                start_node_id="N1",
                end_node_id="N2",
                lift_type="chairlift",
                terrain_points=_PTS[:1],
                pylons=[],
                cable_points=list(_PTS),
            )

    def test_too_few_cable_points_raises(self) -> None:
        with pytest.raises(ValueError, match="at least 2 cable_points"):
            Lift(
                id="L1",
                name="x",
                start_node_id="N1",
                end_node_id="N2",
                lift_type="chairlift",
                terrain_points=list(_PTS),
                pylons=[],
                cable_points=_PTS[:1],
            )


class TestLiftNodeLookups:
    def test_get_vertical_rise_missing_node_raises(self) -> None:
        # Missing station node is an internal invariant violation → fail loud with KeyError.
        with pytest.raises(KeyError):
            _valid_lift().get_vertical_rise(nodes={})

    def test_get_length_m_missing_node_raises(self) -> None:
        with pytest.raises(KeyError):
            _valid_lift().get_length_m(nodes={})

    def test_get_vertical_rise_computes_when_nodes_present(self) -> None:
        nodes = {
            "N1": Node(id="N1", location=PathPoint(lon=10.0, lat=46.0, elevation=2000.0)),
            "N2": Node(id="N2", location=PathPoint(lon=10.001, lat=46.001, elevation=2100.0)),
        }
        assert _valid_lift().get_vertical_rise(nodes=nodes) == pytest.approx(100.0)


class TestUpdateTypeValidation:
    def test_update_type_invalid_raises(self) -> None:
        node_a = Node(id="N1", location=PathPoint(lon=10.0, lat=46.0, elevation=2000.0))
        node_b = Node(id="N2", location=PathPoint(lon=10.001, lat=46.001, elevation=2100.0))
        with pytest.raises(ValueError, match="Invalid lift_type"):
            _valid_lift().update_type(new_type="rocket", start_node=node_a, end_node=node_b)


class TestCalculateStaticmethodValidation:
    def test_calculate_pylons_too_few_points_raises(self) -> None:
        with pytest.raises(ValueError, match="at least 2 points"):
            Lift.calculate_pylons(terrain_points=_PTS[:1], lift_type="chairlift", total_distance_m=500.0)

    def test_calculate_pylons_nonpositive_distance_raises(self) -> None:
        with pytest.raises(ValueError, match="total_distance_m must be positive"):
            Lift.calculate_pylons(terrain_points=list(_PTS), lift_type="chairlift", total_distance_m=0.0)

    def test_calculate_cable_points_too_few_points_raises(self) -> None:
        with pytest.raises(ValueError, match="at least 2 points"):
            Lift.calculate_cable_points(
                terrain_points=_PTS[:1],
                pylons=[],
                start_elevation=2000.0,
                end_elevation=2100.0,
                lift_type="chairlift",
                total_distance_m=500.0,
            )

    def test_calculate_cable_points_nonpositive_distance_raises(self) -> None:
        with pytest.raises(ValueError, match="total_distance_m must be positive"):
            Lift.calculate_cable_points(
                terrain_points=list(_PTS),
                pylons=[],
                start_elevation=2000.0,
                end_elevation=2100.0,
                lift_type="chairlift",
                total_distance_m=-5.0,
            )


class TestLiftNaming:
    """generate_name / number_from_id. RNG seeded by the autouse _deterministic_rng fixture."""

    def test_summit_name_for_large_rise(self) -> None:
        # Vertical rise above SUMMIT_RISE_M (500m) → "Gipfel" lift name.
        name = Lift.generate_name(
            lift_type="chairlift", lift_id="L1", length_m=800.0, vertical_rise_m=600.0, avg_bearing=0.0
        )
        assert "Gipfel" in name

    def test_name_embeds_lift_number(self) -> None:
        name = Lift.generate_name(
            lift_type="gondola", lift_id="L7", length_m=800.0, vertical_rise_m=100.0, avg_bearing=0.0
        )
        assert name.startswith("7 (")

    @pytest.mark.parametrize(
        "id_str,expected",
        [("L1", 1), ("L7", 7), ("L99", 99)],
    )
    def test_number_from_id(self, id_str: str, expected: int) -> None:
        assert Lift.number_from_id(lift_id=id_str) == expected


def _terrain(n: int, *, span_m: float = 1500.0, bumpy: bool = False) -> list[PathPoint]:
    """A straight-in-plan terrain of n points over span_m; elevation ramps, optionally with a bump."""
    step_deg = (span_m / (n - 1)) / MapConfig.METERS_PER_DEGREE_EQUATOR if n > 1 else 0.0
    pts = []
    for i in range(n):
        elev = 2000.0 + i * 10.0
        if bumpy and i == n // 2:
            elev += 200.0  # a sharp terrain bump the clearance sim must react to
        pts.append(PathPoint(lon=10.0, lat=46.0 + step_deg * i, elevation=elev))
    return pts


class TestFinalizeGeometry:
    """finalize_geometry is the pure (no DEM/no Node) single source of truth for lift geometry."""

    def test_pure_no_dem_no_nodes(self) -> None:
        terrain = _terrain(60, bumpy=True)
        thinned, pylons, cable = Lift.finalize_geometry(terrain_points=terrain, lift_type="chairlift")
        assert len(thinned) >= 2 and len(cable) >= 2
        assert thinned[0] is terrain[0] and thinned[-1] is terrain[-1], "endpoints (stations) preserved"

    def test_thins_a_dense_uniform_terrain(self) -> None:
        # A dense, mostly-constant-grade terrain (one bump) should shed most terrain points.
        terrain = _terrain(120, bumpy=True)
        thinned, _pylons, _cable = Lift.finalize_geometry(terrain_points=terrain, lift_type="chairlift")
        assert len(thinned) < len(terrain), "vertical-DP drops the straight/near-linear profile runs"

    def test_idempotent(self) -> None:
        terrain = _terrain(120, bumpy=True)
        thinned1, pylons1, cable1 = Lift.finalize_geometry(terrain_points=terrain, lift_type="chairlift")
        thinned2, pylons2, cable2 = Lift.finalize_geometry(terrain_points=thinned1, lift_type="chairlift")
        assert len(thinned1) == len(thinned2)
        assert len(pylons1) == len(pylons2)
        assert len(cable1) == len(cable2), "re-finalizing already-finalized geometry is a no-op in count"


class TestCableCurvatureSampling:
    """Cable is sampled curvature-adaptively: points track the parabola's sag, not a fixed step/floor."""

    def test_short_low_sag_spans_get_few_points(self) -> None:
        # Two pylons close together → a short span → tiny sag → few interior points (chord error is small).
        terrain = _terrain(60)
        _thinned, pylons, cable = Lift.finalize_geometry(terrain_points=terrain, lift_type="chairlift")
        assert len(pylons) >= 2
        # Points-per-span should be modest; a fixed floor of 4 would inflate this on many short spans.
        pts_per_span = len(cable) / (len(pylons) + 1)
        assert pts_per_span < 4.0, "curvature sampling keeps short spans lean, unlike a fixed floor"

    def test_cable_sampling_scales_with_sag(self) -> None:
        # Within a span, n = ceil(sqrt(sag/tol)) grows with span length, so points-per-span is higher for
        # a lift with longer spans. Compare a short lift vs a long one of the same type.
        short = _terrain(20, span_m=400.0)
        long = _terrain(20, span_m=3000.0)
        _t1, p_short, cable_short = Lift.finalize_geometry(terrain_points=short, lift_type="aerial_tram")
        _t2, p_long, cable_long = Lift.finalize_geometry(terrain_points=long, lift_type="aerial_tram")
        pps_short = len(cable_short) / (len(p_short) + 1)
        pps_long = len(cable_long) / (len(p_long) + 1)
        assert pps_long > pps_short, "longer/deeper-sag spans earn more cable points per span"


class TestPylonsDistanceSpace:
    """Pylons are placed in distance-space, so the sim is correct on non-uniformly spaced terrain."""

    def test_nonuniform_terrain_respects_max_spacing_where_possible(self) -> None:
        # NON-uniform terrain: dense first half, sparser second half, but everywhere fine enough that
        # max-spacing can host midpoint pylons. Build two straight legs over 46.0..46.0135 (~1500m).
        dense = [
            PathPoint(
                lon=10.0, lat=46.0 + (750.0 / MapConfig.METERS_PER_DEGREE_EQUATOR) * (i / 30), elevation=2000.0 + i
            )
            for i in range(31)
        ]
        sparse = [
            PathPoint(
                lon=10.0,
                lat=dense[-1].lat + (750.0 / MapConfig.METERS_PER_DEGREE_EQUATOR) * (j / 8),
                elevation=2030.0 + j * 4,
            )
            for j in range(1, 9)
        ]
        terrain = dense + sparse
        total_m = PathPoint.total_length_m(terrain)
        pylons = Lift.calculate_pylons(terrain_points=terrain, lift_type="chairlift", total_distance_m=total_m)
        assert pylons, "a 1500m chairlift must place pylons"
        assert all(0.0 < p.distance_m < total_m for p in pylons), "pylon distances lie within the lift"
        assert pylons == sorted(pylons, key=lambda p: p.distance_m), "pylons ordered by distance"
        anchors_m = [0.0] + [p.distance_m for p in pylons] + [total_m]
        gaps = [anchors_m[i + 1] - anchors_m[i] for i in range(len(anchors_m) - 1)]
        assert max(gaps) <= 200.0 + 1e-6, "no span exceeds chairlift max_spacing (200m) in metres"


class TestResampleUniform:
    """_resample_uniform: transient fine grid the pylon physics runs on, so its resolution is independent
    of how coarse the stored terrain is. Endpoints preserved; spacing ~= step_m.
    """

    def test_uniform_spacing_and_endpoints(self) -> None:
        # Coarse 2-point terrain over ~1500m → resample at 30m → ~51 points, evenly spaced.
        coarse = [
            PathPoint(lon=10.0, lat=46.0, elevation=2000.0),
            PathPoint(lon=10.0, lat=46.0 + 1500.0 / MapConfig.METERS_PER_DEGREE_EQUATOR, elevation=2300.0),
        ]
        grid = Lift._resample_uniform(terrain_points=coarse, step_m=30.0)
        assert grid[0].elevation == 2000.0 and grid[-1].elevation == pytest.approx(2300.0)
        d = PathPoint.cumulative_distances(grid)
        steps = [d[i + 1] - d[i] for i in range(len(d) - 1)]
        assert max(steps) - min(steps) < 1e-6, "grid is uniformly spaced"
        assert 28.0 < steps[0] < 32.0, "spacing is ~step_m"

    def test_too_few_points_raises(self) -> None:
        with pytest.raises(AssertionError, match=">=2 terrain points"):
            Lift._resample_uniform(terrain_points=[PathPoint(lon=10.0, lat=46.0, elevation=2000.0)], step_m=30.0)

    def test_nonpositive_step_raises(self) -> None:
        pts = [PathPoint(lon=10.0, lat=46.0, elevation=2000.0), PathPoint(lon=10.0, lat=46.01, elevation=2100.0)]
        with pytest.raises(AssertionError, match="step_m must be positive"):
            Lift._resample_uniform(terrain_points=pts, step_m=0.0)


class TestNearestIndexToDistance:
    """_nearest_index_to_distance: the midpoint-pylon helper — nearest interior vertex to a target, or -1
    when the open interval has no interior index (coarse terrain).
    """

    def test_picks_nearest_interior(self) -> None:
        dists = [0.0, 10.0, 20.0, 30.0, 40.0]
        assert Lift._nearest_index_to_distance(dists=dists, target_m=22.0, lo=0, hi=4) == 2

    def test_respects_open_interval_bounds(self) -> None:
        dists = [0.0, 10.0, 20.0, 30.0, 40.0]
        # Only indices strictly between lo=1 and hi=3 are eligible → index 2.
        assert Lift._nearest_index_to_distance(dists=dists, target_m=100.0, lo=1, hi=3) == 2

    def test_no_interior_index_returns_minus_one(self) -> None:
        dists = [0.0, 10.0]
        assert Lift._nearest_index_to_distance(dists=dists, target_m=5.0, lo=0, hi=1) == -1

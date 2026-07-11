"""Unit tests for LeastCostPathPlanner - grid-based Dijkstra algorithm.

Tests the core path planning algorithm in isolation using mock terrain data.
These are "Pure Logic" tests that verify the mathematical correctness without
requiring the full terrain services.

Test Categories:
    1. Edge Cost Function: Slope deviation penalties, uphill penalties
    2. Grid Building: Node finding, grid coordinate calculation
    3. Path Reconstruction: Dijkstra output to PathPoints
"""

import math
from typing import Optional

import pytest

from skiresort_planner.core.terrain_analyzer import TerrainAnalyzer
from skiresort_planner.generators.connection_planners import (
    GridNode,
    LeastCostPathPlanner,
)
from skiresort_planner.model.path_point import PathPoint


class MockDEMForPlanner:
    """Minimal mock DEM for testing planner algorithms.

    Provides a simple elevation model where elevation = 2000 - (lat * 1000).
    This creates a terrain sloping from north to south.
    """

    def __init__(self, base_elevation: float = 2000.0, slope_per_degree: float = 1000.0) -> None:
        self.base_elevation = base_elevation
        self.slope_per_degree = slope_per_degree
        self._call_count = 0

    def get_elevation(self, lon: float, lat: float) -> Optional[float]:
        """Return elevation based on latitude (higher north, lower south)."""
        self._call_count += 1
        return self.base_elevation - (lat * self.slope_per_degree)


@pytest.fixture
def mock_dem() -> MockDEMForPlanner:
    """Simple mock DEM with north-south slope."""
    return MockDEMForPlanner(base_elevation=2000.0, slope_per_degree=1000.0)


@pytest.fixture
def planner(mock_dem: MockDEMForPlanner) -> LeastCostPathPlanner:
    """LeastCostPathPlanner with mock DEM."""
    terrain = TerrainAnalyzer(dem=mock_dem)
    return LeastCostPathPlanner(dem_service=mock_dem, terrain_analyzer=terrain)


class TestEdgeCostFunction:
    """Unit tests for the edge cost function (_calc_edge_cost).

    The cost function implements:
        cost = distance × exp(|actual_slope - target_slope| / σ) × uphill_penalty

    These tests verify the mathematical behavior in isolation.
    """

    def test_cost_is_minimal_when_slope_matches_target(self, planner: LeastCostPathPlanner) -> None:
        """Edge cost is minimized when actual slope matches target slope.

        Given an edge with exactly 20% downhill slope and target of 20%,
        the slope deviation penalty should be 1.0 (exp(0)).
        """
        # 20m drop over 100m horizontal = 20% slope
        cost = planner._calc_edge_cost(
            from_elev=2100.0,
            to_elev=2080.0,  # 20m drop
            from_lon=10.0,
            from_lat=47.0,
            to_lon=10.0009,  # ~100m east at 47°N
            to_lat=47.0,
            target_grade_pct=20.0,
            side="left",
            target_lon=10.001,
            target_lat=46.999,
        )

        # Cost should be approximately distance (since exp(0) = 1.0)
        # Note: Some overhead due to coordinate conversion and precision
        assert cost > 0, "Cost should be positive"
        assert cost < 300, "Cost should be in same order of magnitude as distance"

    def test_cost_increases_with_slope_deviation(self, planner: LeastCostPathPlanner) -> None:
        """Edge cost increases exponentially with slope deviation from target.

        Cost = distance × exp(|actual - target| / σ)
        If σ = 15 and deviation = 30%, penalty = exp(30/15) = exp(2) ≈ 7.4
        """
        # Same distance, but 50% slope vs 20% target = 30% deviation
        cost_matching = planner._calc_edge_cost(
            from_elev=2100.0,
            to_elev=2080.0,  # 20% slope
            from_lon=10.0,
            from_lat=47.0,
            to_lon=10.0009,
            to_lat=47.0,
            target_grade_pct=20.0,
            side="left",
            target_lon=10.001,
            target_lat=46.999,
        )

        cost_deviating = planner._calc_edge_cost(
            from_elev=2100.0,
            to_elev=2050.0,  # 50% slope (30% deviation from target)
            from_lon=10.0,
            from_lat=47.0,
            to_lon=10.0009,
            to_lat=47.0,
            target_grade_pct=20.0,
            side="left",
            target_lon=10.001,
            target_lat=46.999,
        )

        assert cost_deviating > cost_matching, "Higher deviation should increase cost"
        # Exponential penalty should make cost significantly higher
        assert cost_deviating > cost_matching * 2, "Penalty should be substantial"

    def test_uphill_edge_has_penalty(self, planner: LeastCostPathPlanner) -> None:
        """Uphill edges receive additional penalty.

        For ski paths, going uphill is physically wrong and should be penalized.
        """
        # Downhill edge (20m drop)
        cost_downhill = planner._calc_edge_cost(
            from_elev=2100.0,
            to_elev=2080.0,
            from_lon=10.0,
            from_lat=47.0,
            to_lon=10.0009,
            to_lat=47.0,
            target_grade_pct=20.0,
            side="left",
            target_lon=10.001,
            target_lat=46.999,
        )

        # Uphill edge (20m climb)
        cost_uphill = planner._calc_edge_cost(
            from_elev=2080.0,
            to_elev=2100.0,
            from_lon=10.0,
            from_lat=47.0,
            to_lon=10.0009,
            to_lat=47.0,
            target_grade_pct=20.0,
            side="left",
            target_lon=10.001,
            target_lat=46.999,
        )

        assert cost_uphill > cost_downhill, "Uphill should be more expensive"
        # Uphill penalty is exponential
        assert cost_uphill > cost_downhill * 5, "Uphill penalty should be significant"

    def test_zero_distance_returns_infinity(self, planner: LeastCostPathPlanner) -> None:
        """Edge with zero horizontal distance returns infinite cost.

        Prevents degenerate paths with no horizontal movement.
        """
        cost = planner._calc_edge_cost(
            from_elev=2100.0,
            to_elev=2080.0,
            from_lon=10.0,
            from_lat=47.0,
            to_lon=10.0,  # Same position
            to_lat=47.0,
            target_grade_pct=20.0,
            side="left",
            target_lon=10.001,
            target_lat=46.999,
        )

        assert math.isinf(cost), "Zero distance should return infinity"


class TestMomentumTurnPenalty:
    """Momentum: a light, distance-decaying turn penalty biases edges near the start
    node to continue in-line with the incoming heading, then fades to nothing so
    mid-segment routing is untouched. incoming_bearing=None reproduces the old cost.
    """

    def _cost(self, planner, from_lon, from_lat, to_lon, to_lat, **momentum):
        return planner._calc_edge_cost(
            from_elev=2100.0,
            to_elev=2080.0,  # 20% down over ~100m — same slope term for all calls here
            from_lon=from_lon,
            from_lat=from_lat,
            to_lon=to_lon,
            to_lat=to_lat,
            target_grade_pct=20.0,
            side="left",
            target_lon=10.01,
            target_lat=47.0,
            **momentum,
        )

    def test_none_incoming_bearing_reproduces_base_cost_exactly(self, planner: LeastCostPathPlanner) -> None:
        """incoming_bearing=None (slope fan / first segment) → cost unchanged (multiplier 1.0)."""
        base = self._cost(planner, 10.0, 47.0, 10.0009, 47.0)
        with_none = self._cost(
            planner, 10.0, 47.0, 10.0009, 47.0, incoming_bearing=None, start_lon=10.0, start_lat=47.0
        )
        assert with_none == pytest.approx(base, rel=1e-12)

    def test_straight_continuation_cheaper_than_sharp_turn_near_start(self, planner: LeastCostPathPlanner) -> None:
        """At the start node, the momentum multiplier is 1.0 for an edge continuing the
        incoming heading and > 1.0 for one turning away — so straight is cheaper.
        """
        # from == start node (full weight), incoming heading due east (90°).
        straight = planner._momentum_multiplier(
            from_lon=10.0,
            from_lat=47.0,
            to_lon=10.0009,
            to_lat=47.0,  # heads ~east
            incoming_bearing=90.0,
            start_lon=10.0,
            start_lat=47.0,
        )
        turned = planner._momentum_multiplier(
            from_lon=10.0,
            from_lat=47.0,
            to_lon=10.0,
            to_lat=47.0009,  # heads north (90° off)
            incoming_bearing=90.0,
            start_lon=10.0,
            start_lat=47.0,
        )
        assert straight == pytest.approx(1.0, abs=1e-4), "continuing the heading has ~no penalty"
        assert turned > straight, "a 90°-off edge at the start node is penalized"

    def test_penalty_decays_to_nothing_beyond_decay_distance(self, planner: LeastCostPathPlanner) -> None:
        """An edge farther than MOMENTUM_DECAY_M from the start is unaffected (multiplier 1.0)."""
        from skiresort_planner.constants import PlannerConfig

        # Place the edge's from-point well beyond the decay radius NORTH of the start.
        # Offset in latitude (1° ≈ 111320 m holds; longitude would shrink by cos(lat)).
        far_deg = (PlannerConfig.MOMENTUM_DECAY_M + 50.0) / 111320.0
        base = self._cost(planner, 10.0, 47.0 + far_deg, 10.0009, 47.0 + far_deg)  # east edge, no momentum
        with_momentum = self._cost(
            planner,
            10.0,
            47.0 + far_deg,
            10.0009,
            47.0 + far_deg,
            incoming_bearing=0.0,  # 90° off the east edge — would be penalized if in range
            start_lon=10.0,
            start_lat=47.0,
        )
        assert with_momentum == pytest.approx(base, rel=1e-12), "beyond MOMENTUM_DECAY_M momentum has no effect"

    def test_position_pin_penalizes_sideways_jump_at_node(self, planner: LeastCostPathPlanner) -> None:
        """The position term makes an edge that jumps SIDEWAYS off the node far costlier
        than one staying on the incoming line — so the path leaves from where it should,
        not just at the right heading. Regression for the 'jumps left off the node' bug.
        """
        M = 111320.0
        # Incoming due east (90°). On-line endpoint (east) vs sideways endpoint (north), same 20m from node.
        on_line = planner._momentum_multiplier(
            from_lon=0.0,
            from_lat=0.0,
            to_lon=20 / M,
            to_lat=0.0,
            incoming_bearing=90.0,
            start_lon=0.0,
            start_lat=0.0,
        )
        sideways = planner._momentum_multiplier(
            from_lon=0.0,
            from_lat=0.0,
            to_lon=0.0,
            to_lat=20 / M,
            incoming_bearing=90.0,
            start_lon=0.0,
            start_lat=0.0,
        )
        assert on_line == pytest.approx(1.0, abs=1e-3), "staying on the incoming line has ~no penalty"
        assert sideways > on_line * 3, "a sideways jump off the node is strongly penalized"

    def test_position_pin_fades_faster_than_turn(self, planner: LeastCostPathPlanner) -> None:
        """The position pin fades over MOMENTUM_POS_DECAY_M (≪ MOMENTUM_DECAY_M): a
        lateral offset just past the position decay is unpenalized by position, while
        the turn term can still act within its longer range.
        """
        from skiresort_planner.constants import PlannerConfig

        assert PlannerConfig.MOMENTUM_POS_DECAY_M < PlannerConfig.MOMENTUM_DECAY_M, "position pin must fade faster"
        M = 111320.0
        # An endpoint laterally offset but beyond the POS decay radius → no position term;
        # place both from and to past POS decay so only the (still-active) turn term remains.
        past_pos = (PlannerConfig.MOMENTUM_POS_DECAY_M + 20.0) / M  # north of start, beyond pos range
        mult = planner._momentum_multiplier(
            from_lon=0.0,
            from_lat=past_pos,
            to_lon=20 / M,
            to_lat=past_pos,  # on-heading (east) edge
            incoming_bearing=90.0,
            start_lon=0.0,
            start_lat=0.0,
        )
        # On-heading + no position penalty in range → ~1.0 (turn term is 0 for a matching heading).
        assert mult == pytest.approx(1.0, abs=1e-3)


class TestGridNode:
    """Unit tests for GridNode dataclass."""

    def test_grid_node_ordering(self) -> None:
        """GridNode comparison for sorting.

        Nodes are compared by (row, col) tuple ordering.
        """
        n1 = GridNode(row=1, col=2)
        n2 = GridNode(row=1, col=3)
        n3 = GridNode(row=2, col=1)

        assert n1 < n2, "Same row, lower col should be smaller"
        assert n1 < n3, "Lower row should be smaller"
        assert n2 < n3, "Row takes precedence over col"

    def test_grid_node_equality(self) -> None:
        """GridNode equality based on row/col."""
        n1 = GridNode(row=1, col=2)
        n2 = GridNode(row=1, col=2)
        n3 = GridNode(row=1, col=3)

        assert n1 == n2, "Same row/col should be equal"
        assert n1 != n3, "Different col should not be equal"


class TestFindNearestNode:
    """Unit tests for _find_nearest_node method."""

    def test_finds_exact_match(self, planner: LeastCostPathPlanner) -> None:
        """Find node that exactly matches target coordinates."""
        # Create a simple 3x3 grid
        lons = [[10.0, 10.1, 10.2], [10.0, 10.1, 10.2], [10.0, 10.1, 10.2]]
        lats = [[47.0, 47.0, 47.0], [47.1, 47.1, 47.1], [47.2, 47.2, 47.2]]

        node = planner._find_nearest_node(target_lon=10.1, target_lat=47.1, lons=lons, lats=lats)

        assert node is not None
        assert node.row == 1
        assert node.col == 1

    def test_finds_nearest_when_no_exact_match(self, planner: LeastCostPathPlanner) -> None:
        """Find closest node when target is between grid points."""
        lons = [[10.0, 10.1, 10.2], [10.0, 10.1, 10.2], [10.0, 10.1, 10.2]]
        lats = [[47.0, 47.0, 47.0], [47.1, 47.1, 47.1], [47.2, 47.2, 47.2]]

        # Target closer to (1, 1) than any other node
        node = planner._find_nearest_node(target_lon=10.09, target_lat=47.09, lons=lons, lats=lats)

        assert node is not None
        assert node.row == 1
        assert node.col == 1


class TestPathToPoints:
    """Unit tests for _path_to_points conversion."""

    def test_converts_grid_path_to_pathpoints(self, planner: LeastCostPathPlanner) -> None:
        """Convert list of GridNodes to list of PathPoints."""
        elevations = [[2000.0, 1990.0], [1980.0, 1970.0]]
        lons = [[10.0, 10.1], [10.0, 10.1]]
        lats = [[47.0, 47.0], [47.1, 47.1]]

        path_nodes = [GridNode(row=0, col=0), GridNode(row=0, col=1), GridNode(row=1, col=1)]

        points = planner._path_to_points(
            path_nodes=path_nodes,
            elevations=elevations,
            lons=lons,
            lats=lats,
        )

        assert len(points) == 3
        assert isinstance(points[0], PathPoint)
        assert points[0].lon == 10.0
        assert points[0].lat == 47.0
        assert points[0].elevation == 2000.0
        assert points[2].elevation == 1970.0


class TestPlannerIntegration:
    """Integration tests for the full plan() method."""

    def test_plan_returns_none_for_uphill_target(self, planner: LeastCostPathPlanner) -> None:
        """Planner returns None when target is higher than start.

        Ski paths must go downhill.
        """
        result = planner.plan(
            start_lon=10.0,
            start_lat=47.0,
            start_elevation=2000.0,
            target_lon=10.0,
            target_lat=47.001,
            target_elevation=2100.0,  # Higher than start
            target_grade_pct=20.0,
            side="left",
        )

        assert result is None, "Uphill path should return None"

    def test_plan_returns_none_for_zero_distance(self, planner: LeastCostPathPlanner) -> None:
        """Planner returns None when start and target are the same."""
        result = planner.plan(
            start_lon=10.0,
            start_lat=47.0,
            start_elevation=2000.0,
            target_lon=10.0,
            target_lat=47.0,
            target_elevation=1900.0,
            target_grade_pct=20.0,
            side="left",
        )

        assert result is None, "Zero distance path should return None"


class TestEdgeCostGradeAttractor:
    """The edge cost pulls the path toward target_grade_pct via one exponential
    attractor. allow_uphill toggles whether climbing is penalized (False = descent-only
    pistes, True = vehicle roads). The planner is domain-agnostic — no road/slope modes.
    """

    def _planner(self, mock_dem_blue_slope) -> LeastCostPathPlanner:
        return LeastCostPathPlanner(
            dem_service=mock_dem_blue_slope, terrain_analyzer=TerrainAnalyzer(dem=mock_dem_blue_slope)
        )

    def _cost(self, planner, from_elev, to_elev, target_grade_pct, allow_uphill):
        # 30m east step; deterministic horizontal distance.
        return planner._calc_edge_cost(
            from_elev=from_elev,
            to_elev=to_elev,
            from_lon=0.0,
            from_lat=0.0,
            to_lon=30.0 / 111320.0,
            to_lat=0.0,
            target_grade_pct=target_grade_pct,
            side="left",
            target_lon=1.0,
            target_lat=0.0,
            allow_uphill=allow_uphill,
        )

    def test_cost_minimal_when_edge_matches_target_grade(self, mock_dem_blue_slope) -> None:
        """An edge whose grade equals the target grade has ~no penalty → cost ≈ distance."""
        planner = self._planner(mock_dem_blue_slope)
        # Target 10% descent; a 10% edge (3m drop over 30m) deviates 0.
        cost = self._cost(planner, 2000.0, 1997.0, target_grade_pct=10.0, allow_uphill=True)
        assert cost == pytest.approx(30.0, rel=0.05)

    def test_cost_grows_with_deviation_from_target(self, mock_dem_blue_slope) -> None:
        """Deviating from the target grade costs strictly more (the attractor)."""
        planner = self._planner(mock_dem_blue_slope)
        on_target = self._cost(planner, 2000.0, 1997.0, target_grade_pct=10.0, allow_uphill=True)  # 10% edge
        off_target = self._cost(planner, 2000.0, 1988.0, target_grade_pct=10.0, allow_uphill=True)  # 40% edge
        assert off_target > on_target * 2, "large deviation is exponentially penalized"

    def test_allow_uphill_drops_the_climb_penalty(self, mock_dem_blue_slope) -> None:
        """With allow_uphill=True, for the same |deviation| a climbing edge costs the
        SAME as a descending one — climbing is free (vehicle roads).
        """
        planner = self._planner(mock_dem_blue_slope)
        # Target 0%. A +10% edge and a −10% edge both deviate 10% → equal cost.
        descend = self._cost(planner, 2000.0, 1997.0, target_grade_pct=0.0, allow_uphill=True)  # 10% down
        climb = self._cost(planner, 2000.0, 2003.0, target_grade_pct=0.0, allow_uphill=True)  # 10% up
        assert climb == pytest.approx(descend, rel=1e-9), "no climb penalty when allow_uphill"

    def test_descent_only_reproduces_documented_formula(self, mock_dem_blue_slope) -> None:
        """allow_uphill=False, downhill edge: cost = dist × exp(|actual−target| / σ)."""
        from math import exp

        from skiresort_planner.constants import PlannerConfig
        from skiresort_planner.core.geo_calculator import GeoCalculator

        planner = self._planner(mock_dem_blue_slope)
        cost = self._cost(planner, 2000.0, 1994.0, target_grade_pct=0.0, allow_uphill=False)  # 6m drop, target 0
        horiz = GeoCalculator.haversine_distance_m(lat1=0.0, lon1=0.0, lat2=0.0, lon2=30.0 / 111320.0)
        actual_grade = (6.0 / horiz) * 100
        expected = horiz * exp(abs(actual_grade - 0.0) / PlannerConfig.COST_SIGMA)
        assert cost == pytest.approx(expected, rel=1e-6)

    def test_descent_only_penalizes_uphill(self, mock_dem_blue_slope) -> None:
        """allow_uphill=False keeps the climb penalty (skiers descend)."""
        planner = self._planner(mock_dem_blue_slope)
        # Target 20%. A 20% descent matches; a 20% climb both deviates AND is uphill.
        descend = self._cost(planner, 2000.0, 1994.0, target_grade_pct=20.0, allow_uphill=False)  # 20% down
        climb = self._cost(planner, 2000.0, 2006.0, target_grade_pct=20.0, allow_uphill=False)  # 20% up
        assert climb > descend * 5, "descent-only mode still penalizes climbing steeply"

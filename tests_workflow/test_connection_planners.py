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
from unittest.mock import MagicMock

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
            target_slope_pct=20.0,
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
            target_slope_pct=20.0,
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
            target_slope_pct=20.0,
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
            target_slope_pct=20.0,
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
            target_slope_pct=20.0,
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
            target_slope_pct=20.0,
            side="left",
            target_lon=10.001,
            target_lat=46.999,
        )

        assert math.isinf(cost), "Zero distance should return infinity"


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
            target_slope_pct=20.0,
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
            target_slope_pct=20.0,
            side="left",
        )

        assert result is None, "Zero distance path should return None"


class TestPlannerGradientBand:
    """Road-mode cost: soft penalty ramps from the COMFORT knee (ROAD_SOFT_GRADIENT_PCT,
    12%), NOT the hard cap — no uphill penalty. Passing gradient_band selects road mode.
    """

    def _planner(self, mock_dem_blue_slope) -> LeastCostPathPlanner:
        return LeastCostPathPlanner(
            dem_service=mock_dem_blue_slope, terrain_analyzer=TerrainAnalyzer(dem=mock_dem_blue_slope)
        )

    def _cost(self, planner, from_elev, to_elev, band):
        # 30m east step; deterministic horizontal distance.
        return planner._calc_edge_cost(
            from_elev=from_elev,
            to_elev=to_elev,
            from_lon=0.0,
            from_lat=0.0,
            to_lon=30.0 / 111320.0,
            to_lat=0.0,
            target_slope_pct=0.0,
            side="left",
            target_lon=1.0,
            target_lat=0.0,
            gradient_band=band,
        )

    def test_in_band_flat_has_no_extra_penalty(self, mock_dem_blue_slope) -> None:
        planner = self._planner(mock_dem_blue_slope)
        # Flat edge inside band → cost == horizontal distance (exp(0) == 1).
        flat_cost = self._cost(planner, 2000.0, 2000.0, (-12.0, 12.0))
        assert flat_cost == pytest.approx(30.0, rel=0.05)

    def test_uphill_in_band_not_penalized_in_road_mode(self, mock_dem_blue_slope) -> None:
        planner = self._planner(mock_dem_blue_slope)
        band = (-12.0, 12.0)
        # +10% uphill (climb 3m over 30m) is within band → same as flat.
        uphill = self._cost(planner, 2000.0, 2003.0, band)
        flat = self._cost(planner, 2000.0, 2000.0, band)
        assert uphill == pytest.approx(flat, rel=0.05)

    def test_out_of_band_costs_more(self, mock_dem_blue_slope) -> None:
        planner = self._planner(mock_dem_blue_slope)
        band = (-12.0, 12.0)
        in_band = self._cost(planner, 2000.0, 2000.0, band)
        # 40% descent (12m over 30m) is well outside the band → strictly costlier.
        out_of_band = self._cost(planner, 2000.0, 1988.0, band)
        assert out_of_band > in_band

    def test_soft_penalty_ramps_from_comfort_knee_below_hard_cap(self, mock_dem_blue_slope) -> None:
        """A ~13% grade (below the 15% hard cap, above the 12% soft knee) is already penalized.

        Regression for the soft-band tightening: cost must NOT stay flat up to 15%.
        """
        planner = self._planner(mock_dem_blue_slope)
        band = (-15.0, 15.0)  # road mode; the knee comes from ROAD_SOFT_GRADIENT_PCT
        # ≤12% comfort → flat (x1.0).
        comfort = self._cost(planner, 2000.0, 2000.0 - 0.12 * 30.0, band)  # exactly 12%
        # ~13% (below 15% hard cap) → already ramped above flat.
        near_limit = self._cost(planner, 2000.0, 2000.0 - 0.13 * 30.0, band)  # ~13%
        assert near_limit > comfort, "soft penalty must start below the 15% hard cap (at the 12% knee)"

    def test_slope_mode_unchanged_by_band_param(self, mock_dem_blue_slope) -> None:
        """gradient_band=None must reproduce the original slope cost exactly."""
        from math import exp

        from skiresort_planner.constants import PlannerConfig
        from skiresort_planner.core.geo_calculator import GeoCalculator

        planner = self._planner(mock_dem_blue_slope)
        cost = self._cost(planner, 2000.0, 1994.0, None)  # 6m drop over one east step, target 0
        # Recompute expected via the documented slope formula, using the exact
        # horizontal distance the cost function itself would measure.
        horiz = GeoCalculator.haversine_distance_m(lat1=0.0, lon1=0.0, lat2=0.0, lon2=30.0 / 111320.0)
        actual_slope = (6.0 / horiz) * 100
        expected = horiz * exp(abs(actual_slope - 0.0) / PlannerConfig.COST_SIGMA)
        assert cost == pytest.approx(expected, rel=1e-6)

"""Shared pytest fixtures for skiresort_planner tests.

Provides MockDEMService and reusable test data for all skiresort_planner tests.
All fixtures use explicit values with documented rationale.

COORDINATE SYSTEM:
    Tests use coordinates near the equator (lat~0) and prime meridian (lon~0)
    where the math is simple: 1 degree ≈ 111,320 meters in both directions.
    This avoids needing GeoCalculator in the mock (which would test with tested code).
"""

import pytest

from skiresort_planner.constants import DEMConfig, MapConfig
from skiresort_planner.core.dem_service import DEMService
from skiresort_planner.core.path_tracer import PathTracer
from skiresort_planner.core.terrain_analyzer import TerrainAnalyzer
from skiresort_planner.generators.path_factory import PathFactory
from skiresort_planner.model.node import Node
from skiresort_planner.model.path_point import PathPoint
from skiresort_planner.model.proposed_path import ProposedSlopeSegment
from skiresort_planner.model.resort_graph import ResortGraph
from skiresort_planner.ui.state_machine import (
    PlannerContext,
    PlannerStateMachine,
)


# =============================================================================
# MOCK DEM SERVICE
# =============================================================================


class MockDEMService:
    """Mock DEM returning synthetic elevation based on simple linear formula.

    Uses coordinates near the equator where 1° ≈ 111,320m, allowing simple math
    without needing GeoCalculator (which would be testing with tested code).

    Elevation formula:
        elevation = base_elev + (lat * METERS_PER_DEGREE * slope_ns_pct / 100)
                              - (lon * METERS_PER_DEGREE * slope_ew_pct / 100)

    At lat=0, lon=0: elevation = base_elev (summit)
    Going south (negative lat): elevation drops if slope_ns > 0
    Going east (positive lon): elevation drops if slope_ew > 0

    Example with base=2500m, slope_ns=20%, slope_ew=0%:
        - lat=0.000: 2500m (summit)
        - lat=-0.009 (1000m south): 2500 - 200 = 2300m (20% drop)
    """

    def __init__(
        self,
        base_elevation: float,
        slope_ns_pct: float,
        slope_ew_pct: float,
    ) -> None:
        """Initialize mock DEM.

        Test region centered at equator/prime meridian intersection.
        All test coordinates should be within this region for consistent math.

        Args:
            base_elevation: Elevation at origin (lat=0, lon=0)
            slope_ns_pct: North-south slope percentage. Positive = drops going south.
            slope_ew_pct: East-west slope percentage. Positive = drops going east.
        """
        self.base_elevation = base_elevation
        self.slope_ns_pct = slope_ns_pct
        self.slope_ew_pct = slope_ew_pct
        self._bounds = (-1.0, -1.0, 1.0, 1.0)  # (lon_min, lat_min, lon_max, lat_max)

    @property
    def is_loaded(self) -> bool:
        return True

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        return self._bounds

    def get_elevation(self, lon: float, lat: float) -> float:
        """Return elevation using simple linear formula.

        Drops going south (negative lat) and east (positive lon).
        """
        M = MapConfig.METERS_PER_DEGREE_EQUATOR
        return self.base_elevation + lat * M * (self.slope_ns_pct / 100) - lon * M * (self.slope_ew_pct / 100)


# =============================================================================
# MOCK DEM FIXTURES
# =============================================================================


@pytest.fixture
def mock_dem_blue_slope_south() -> MockDEMService:
    """Mock DEM: 20% slope going south (blue difficulty), flat east-west.

    At origin (0,0): 2500m elevation.
    Moving 1000m south (lat=-0.009): drops to 2300m (20% grade).
    This simulates blue difficulty terrain (15-25% range).
    """
    return MockDEMService(
        base_elevation=2500.0,
        slope_ns_pct=20.0,
        slope_ew_pct=0.0,
    )


@pytest.fixture
def mock_dem_black_slope_south() -> MockDEMService:
    """Mock DEM: 40% slope going south (black difficulty), slight east slope.

    At origin (0,0): 2500m elevation.
    Moving 1000m south: drops to 2100m (40% grade).
    Moving 1000m east: drops 50m (5% grade).
    This simulates black diamond terrain.
    """
    return MockDEMService(
        base_elevation=2500.0,
        slope_ns_pct=40.0,
        slope_ew_pct=5.0,
    )


@pytest.fixture
def mock_dem_red_slope_southeast() -> MockDEMService:
    """Mock DEM: 25% slope going south (red difficulty), 5% slope east.

    At origin (0,0): 2500m elevation.
    Fall line goes roughly southeast due to combined slopes.
    This simulates red difficulty terrain with diagonal fall line.
    """
    return MockDEMService(
        base_elevation=2500.0,
        slope_ns_pct=25.0,
        slope_ew_pct=5.0,
    )


# =============================================================================
# PATH POINT FIXTURES
# =============================================================================


@pytest.fixture
def path_points_800m_south_20pct_drop(mock_dem_blue_slope_south: MockDEMService) -> list[PathPoint]:
    """Path going 800m south with 5 points, 200m steps, 160m total drop.

    Geometry: 5 points at lon=0, each 200m south.
    Uses blue slope DEM (20%) so total drop = 800m * 20% = 160m.
    Classification: blue difficulty.
    """
    dem = mock_dem_blue_slope_south
    M = MapConfig.METERS_PER_DEGREE_EQUATOR
    return [
        PathPoint(lon=0.0, lat=-0 / M, elevation=dem.get_elevation(lon=0.0, lat=-0 / M)),
        PathPoint(lon=0.0, lat=-200 / M, elevation=dem.get_elevation(lon=0.0, lat=-200 / M)),
        PathPoint(lon=0.0, lat=-400 / M, elevation=dem.get_elevation(lon=0.0, lat=-400 / M)),
        PathPoint(lon=0.0, lat=-600 / M, elevation=dem.get_elevation(lon=0.0, lat=-600 / M)),
        PathPoint(lon=0.0, lat=-800 / M, elevation=dem.get_elevation(lon=0.0, lat=-800 / M)),
    ]


@pytest.fixture
def path_points_600m_south_3pts(mock_dem_blue_slope_south: MockDEMService) -> list[PathPoint]:
    """Shorter path: 600m south with 3 points, 300m steps.

    Uses blue slope DEM (20%) so total drop = 600m * 20% = 120m.
    Classification: blue difficulty.
    """
    dem = mock_dem_blue_slope_south
    M = MapConfig.METERS_PER_DEGREE_EQUATOR
    return [
        PathPoint(lon=0.0, lat=-0 / M, elevation=dem.get_elevation(lon=0.0, lat=-0 / M)),
        PathPoint(lon=0.0, lat=-300 / M, elevation=dem.get_elevation(lon=0.0, lat=-300 / M)),
        PathPoint(lon=0.0, lat=-600 / M, elevation=dem.get_elevation(lon=0.0, lat=-600 / M)),
    ]


# =============================================================================
# NODE FIXTURES
# =============================================================================


@pytest.fixture
def node_at_origin_summit(mock_dem_blue_slope_south: MockDEMService) -> Node:
    """Single node at origin (0, 0) representing mountain summit at 2500m."""
    dem = mock_dem_blue_slope_south
    return Node(
        id="N1",
        location=PathPoint(lon=0.0, lat=0.0, elevation=dem.get_elevation(lon=0.0, lat=0.0)),
    )


# =============================================================================
# RESORT GRAPH FIXTURES
# =============================================================================


@pytest.fixture
def empty_resort_graph() -> ResortGraph:
    """Empty resort graph for testing graph operations from scratch."""
    return ResortGraph()


@pytest.fixture
def resort_graph_with_3_nodes_vertical(
    empty_resort_graph: ResortGraph,
    mock_dem_blue_slope_south: MockDEMService,
) -> ResortGraph:
    """Graph with 3 nodes arranged vertically: summit → mid → valley, 1000m apart.

    Uses blue slope DEM (20%):
    - N1 (summit): lat=0, elevation=2500m
    - N2 (mid): lat=-1000m, elevation=2300m
    - N3 (valley): lat=-2000m, elevation=2100m
    """
    dem = mock_dem_blue_slope_south
    M = MapConfig.METERS_PER_DEGREE_EQUATOR
    empty_resort_graph.nodes["N1"] = Node(
        id="N1",
        location=PathPoint(lon=0.0, lat=-0 / M, elevation=dem.get_elevation(lon=0.0, lat=-0 / M)),
    )
    empty_resort_graph.nodes["N2"] = Node(
        id="N2",
        location=PathPoint(lon=0.0, lat=-1000 / M, elevation=dem.get_elevation(lon=0.0, lat=-1000 / M)),
    )
    empty_resort_graph.nodes["N3"] = Node(
        id="N3",
        location=PathPoint(lon=0.0, lat=-2000 / M, elevation=dem.get_elevation(lon=0.0, lat=-2000 / M)),
    )
    return empty_resort_graph


# =============================================================================
# PROPOSED SEGMENT FIXTURES
# =============================================================================


@pytest.fixture
def proposed_segment_blue_800m(path_points_800m_south_20pct_drop: list[PathPoint]) -> ProposedSlopeSegment:
    """Proposed segment: 800m long, 20% slope, blue difficulty.

    Uses path_points_800m_south_20pct_drop fixture.
    """
    return ProposedSlopeSegment(
        points=path_points_800m_south_20pct_drop,
        target_slope_pct=20.0,
        target_difficulty="blue",
        sector_name="Blue-Steep Left",
    )


# =============================================================================
# STATE MACHINE FIXTURES
# =============================================================================


@pytest.fixture
def state_machine_and_context(
    empty_resort_graph: ResortGraph,
) -> tuple[PlannerStateMachine, PlannerContext]:
    """Fresh state machine and context pair, starting in IDLE state."""
    return PlannerStateMachine.create(graph=empty_resort_graph)


# =============================================================================
# WORKFLOW INTEGRATION FIXTURES
# (for full end-to-end testing with PathFactory)
# =============================================================================


@pytest.fixture
def workflow_path_factory(mock_dem_red_slope_southeast: MockDEMService) -> PathFactory:
    """PathFactory configured with mock DEM for deterministic path generation.

    Uses red slope DEM (25% south, 5% east) for realistic multi-direction paths.
    """
    dem = mock_dem_red_slope_southeast
    analyzer = TerrainAnalyzer(dem=dem)
    tracer = PathTracer(dem=dem, analyzer=analyzer)
    return PathFactory(dem_service=dem, path_tracer=tracer, terrain_analyzer=analyzer)


@pytest.fixture
def workflow_complete_setup(
    mock_dem_red_slope_southeast: MockDEMService,
    workflow_path_factory: PathFactory,
    empty_resort_graph: ResortGraph,
) -> tuple[PlannerStateMachine, PlannerContext, ResortGraph, PathFactory, MockDEMService]:
    """Complete workflow setup for end-to-end testing.

    Returns tuple of:
    - PlannerStateMachine (in IDLE state)
    - PlannerContext
    - Empty ResortGraph
    - PathFactory with mock DEM
    - MockDEMService (red slope)

    Use this fixture when testing full planning workflows.
    """
    sm, ctx = PlannerStateMachine.create(graph=empty_resort_graph)
    return sm, ctx, empty_resort_graph, workflow_path_factory, mock_dem_red_slope_southeast


# =============================================================================
# REAL DEM FIXTURES (skipped if file unavailable)
# =============================================================================


@pytest.fixture
def real_dem_eurodem() -> DEMService:
    """Real EuroDEM service for integration tests.

    Automatically skips test if eurodem.tif file is not available.
    """
    if not DEMConfig.EURODEM_PATH.exists():
        pytest.skip("EuroDEM file not available")
    return DEMService(dem_path=DEMConfig.EURODEM_PATH)

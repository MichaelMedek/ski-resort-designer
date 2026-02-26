"""Shared pytest fixtures for skiresort_planner workflow tests.

Provides MockDEMService and reusable test data for all tests.
Minimal fixtures following the design document principle: keep conftest.py minimal.

COORDINATE SYSTEM:
    Tests use coordinates near the equator (lat~0) and prime meridian (lon~0)
    where the math is simple: 1 degree ≈ 111,320 meters in both directions.
"""

from typing import TYPE_CHECKING

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
from skiresort_planner.ui.state_machine import PlannerStateMachine

if TYPE_CHECKING:
    from skiresort_planner.ui.context import UIContext

# Type alias for workflow_setup fixture return value
WorkflowSetup = tuple[PlannerStateMachine, "UIContext", ResortGraph, PathFactory, "MockDEMService"]
SMAndCtx = tuple[PlannerStateMachine, "UIContext"]


class MockDEMService(DEMService):
    """Mock DEM returning synthetic elevation based on simple linear formula.

    Uses coordinates near the equator where 1° ≈ 111,320m, allowing simple math.

    Elevation formula:
        elevation = base_elev + (lat * METERS_PER_DEGREE * slope_ns_pct / 100)
                              - (lon * METERS_PER_DEGREE * slope_ew_pct / 100)

    At lat=0, lon=0: elevation = base_elev (summit)
    Going south (negative lat): elevation drops if slope_ns > 0
    Going east (positive lon): elevation drops if slope_ew > 0
    """

    _instance = None  # Override singleton

    def __new__(cls, *args: object, **kwargs: object) -> "MockDEMService":
        """Create new instance (bypass singleton for tests)."""
        return object.__new__(cls)

    def __init__(
        self,
        base_elevation: float,
        slope_ns_pct: float,
        slope_ew_pct: float,
    ) -> None:
        """Initialize mock DEM.

        Args:
            base_elevation: Elevation at origin (lat=0, lon=0)
            slope_ns_pct: North-south slope percentage. Positive = drops going south.
            slope_ew_pct: East-west slope percentage. Positive = drops going east.
        """
        self.base_elevation = base_elevation
        self.slope_ns_pct = slope_ns_pct
        self.slope_ew_pct = slope_ew_pct
        self._bounds = (-1.0, -1.0, 1.0, 1.0)

    @property
    def is_loaded(self) -> bool:
        return True

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        return self._bounds

    def get_elevation(self, lon: float, lat: float) -> float | None:
        """Return elevation using simple linear formula."""
        M = MapConfig.METERS_PER_DEGREE_EQUATOR
        return self.base_elevation + lat * M * (self.slope_ns_pct / 100) - lon * M * (self.slope_ew_pct / 100)

    def get_elevation_or_raise(self, lon: float, lat: float) -> float:
        """Return elevation, raising if None (never happens for mock)."""
        elev = self.get_elevation(lon=lon, lat=lat)
        assert elev is not None, "MockDEMService always returns elevation"
        return elev


# =============================================================================
# MOCK DEM FIXTURES
# =============================================================================


@pytest.fixture
def mock_dem_blue_slope() -> MockDEMService:
    """Mock DEM: 20% slope going south (blue difficulty), flat east-west."""
    return MockDEMService(base_elevation=2500.0, slope_ns_pct=20.0, slope_ew_pct=0.0)


@pytest.fixture
def mock_dem_black_slope() -> MockDEMService:
    """Mock DEM: 45% slope going south (black difficulty)."""
    return MockDEMService(base_elevation=2500.0, slope_ns_pct=45.0, slope_ew_pct=0.0)


@pytest.fixture
def mock_dem_red_slope_diagonal() -> MockDEMService:
    """Mock DEM: 30% south slope + 10% east slope (diagonal fall line)."""
    return MockDEMService(base_elevation=2500.0, slope_ns_pct=30.0, slope_ew_pct=10.0)


# =============================================================================
# RESORT GRAPH FIXTURES
# =============================================================================


@pytest.fixture
def empty_graph() -> ResortGraph:
    """Empty resort graph for testing graph operations from scratch."""
    return ResortGraph()


@pytest.fixture
def graph_with_nodes(mock_dem_blue_slope: MockDEMService) -> ResortGraph:
    """Graph with 3 nodes arranged vertically: summit → mid → valley."""
    graph = ResortGraph()
    dem = mock_dem_blue_slope
    M = MapConfig.METERS_PER_DEGREE_EQUATOR

    graph.nodes["N1"] = Node(
        id="N1",
        location=PathPoint(
            lon=0.0,
            lat=0.0,
            elevation=dem.get_elevation_or_raise(lon=0.0, lat=0.0),
        ),
    )
    graph.nodes["N2"] = Node(
        id="N2",
        location=PathPoint(
            lon=0.0,
            lat=-1000 / M,
            elevation=dem.get_elevation_or_raise(lon=0.0, lat=-1000 / M),
        ),
    )
    graph.nodes["N3"] = Node(
        id="N3",
        location=PathPoint(
            lon=0.0,
            lat=-2000 / M,
            elevation=dem.get_elevation_or_raise(lon=0.0, lat=-2000 / M),
        ),
    )
    return graph


# =============================================================================
# STATE MACHINE FIXTURES
# =============================================================================


@pytest.fixture
def sm_and_ctx(empty_graph: ResortGraph) -> SMAndCtx:
    """Fresh state machine and context pair, starting in IdleReady state."""
    return PlannerStateMachine.create(graph=empty_graph)


# =============================================================================
# PATH FIXTURES
# =============================================================================


@pytest.fixture
def path_points_blue(mock_dem_blue_slope: MockDEMService) -> list[PathPoint]:
    """Path going 800m south with 5 points on blue slope terrain."""
    dem = mock_dem_blue_slope
    M = MapConfig.METERS_PER_DEGREE_EQUATOR
    return [
        PathPoint(lon=0.0, lat=-0 / M, elevation=dem.get_elevation_or_raise(lon=0.0, lat=-0 / M)),
        PathPoint(lon=0.0, lat=-200 / M, elevation=dem.get_elevation_or_raise(lon=0.0, lat=-200 / M)),
        PathPoint(lon=0.0, lat=-400 / M, elevation=dem.get_elevation_or_raise(lon=0.0, lat=-400 / M)),
        PathPoint(lon=0.0, lat=-600 / M, elevation=dem.get_elevation_or_raise(lon=0.0, lat=-600 / M)),
        PathPoint(lon=0.0, lat=-800 / M, elevation=dem.get_elevation_or_raise(lon=0.0, lat=-800 / M)),
    ]


@pytest.fixture
def proposed_segment_blue(path_points_blue: list[PathPoint]) -> ProposedSlopeSegment:
    """Proposed segment: 800m long, 20% slope, blue difficulty."""
    return ProposedSlopeSegment(
        points=path_points_blue,
        target_slope_pct=20.0,
        target_difficulty="blue",
        sector_name="Blue-Steep Left",
    )


# =============================================================================
# WORKFLOW INTEGRATION FIXTURES
# =============================================================================


@pytest.fixture
def path_factory(mock_dem_red_slope_diagonal: MockDEMService) -> PathFactory:
    """PathFactory configured with mock DEM for deterministic path generation."""
    dem = mock_dem_red_slope_diagonal
    analyzer = TerrainAnalyzer(dem=dem)
    tracer = PathTracer(dem=dem, analyzer=analyzer)
    return PathFactory(dem_service=dem, path_tracer=tracer, terrain_analyzer=analyzer)


@pytest.fixture
def workflow_setup(
    mock_dem_red_slope_diagonal: MockDEMService,
    path_factory: PathFactory,
    empty_graph: ResortGraph,
) -> WorkflowSetup:
    """Complete workflow setup for end-to-end testing."""
    sm, ctx = PlannerStateMachine.create(graph=empty_graph)
    return sm, ctx, empty_graph, path_factory, mock_dem_red_slope_diagonal


# =============================================================================
# REAL DEM FIXTURES
# =============================================================================


@pytest.fixture
def real_dem() -> DEMService:
    """Real EuroDEM service for integration tests. Skips if unavailable."""
    if not DEMConfig.EURODEM_PATH.exists():
        pytest.skip("EuroDEM file not available")
    return DEMService(dem_path=DEMConfig.EURODEM_PATH)

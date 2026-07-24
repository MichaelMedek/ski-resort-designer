"""Shared pytest fixtures for skiresort_planner workflow tests.

Provides MockDEMService and reusable test data for all tests.
Minimal fixtures following the design document principle: keep conftest.py minimal.

COORDINATE SYSTEM:
    Tests use coordinates near the equator (lat~0) and prime meridian (lon~0)
    where the math is simple: 1 degree ≈ 111,320 meters in both directions.
"""

import random
from collections.abc import Callable
from typing import TYPE_CHECKING, Literal
from unittest.mock import MagicMock

import numpy as np
import pytest

from skiresort_planner.constants import DEMConfig, LiftType, MapConfig
from skiresort_planner.core.dem_service import DEMService
from skiresort_planner.core.path_tracer import PathTracer
from skiresort_planner.core.terrain_analyzer import TerrainAnalyzer
from skiresort_planner.generators.path_factory import PathFactory
from skiresort_planner.model.node import Node
from skiresort_planner.model.path_point import PathPoint
from skiresort_planner.model.path_segment import PathSegment, SegmentKind
from skiresort_planner.model.proposed_path import ProposedPathSegment
from skiresort_planner.model.resort_graph import ResortGraph
from skiresort_planner.model.slope import Slope
from skiresort_planner.ui.state_machine import PlannerStateMachine

if TYPE_CHECKING:
    from skiresort_planner.ui.context import PlannerContext

# Type alias for workflow_setup fixture return value
WorkflowSetup = tuple[PlannerStateMachine, "PlannerContext", ResortGraph, PathFactory, "MockDEMService"]
SMAndCtx = tuple[PlannerStateMachine, "PlannerContext"]


# =============================================================================
# INFRA MOCKING - Mock Streamlit infrastructure for unit tests
# =============================================================================


@pytest.fixture(autouse=True)
def mock_infra_rerun(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> MagicMock | None:
    """Auto-mock trigger_rerun() for unit tests to prevent Streamlit reruns.

    Enabled automatically for all tests EXCEPT those marked with @pytest.mark.apptest.
    AppTests (E2E) should handle reruns via their own mechanisms.

    Returns:
        MagicMock for unit tests, None for AppTests (no mocking).
    """
    # Skip mocking for AppTests - they handle reruns via add_ui_listener=False or real execution
    if request.node.get_closest_marker("apptest"):
        return None

    mock = MagicMock()
    monkeypatch.setattr("skiresort_planner.ui.infra.trigger_rerun", mock)
    return mock


@pytest.fixture(autouse=True)
def _deterministic_rng() -> None:
    """Seed the global RNG before every test so generated geometry is reproducible.

    Production path tracing uses ``random.gauss`` (the global module) on purpose — real users
    get fresh, varied routes each time. That global state makes the suite order-dependent.
    """
    random.seed(3)


# =============================================================================
# FAKE STREAMLIT (for exercising panel render() logic without a browser)
# =============================================================================


class _FakeSessionState(dict[str, object]):
    """Supports both st.session_state.foo and st.session_state['foo']."""

    def __getattr__(self, name: str) -> object:
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(name) from e

    def __setattr__(self, name: str, value: object) -> None:
        self[name] = value

    def __delattr__(self, name: str) -> None:
        del self[name]


class _Ctx:
    """No-op context manager usable as a column/expander/sidebar/spinner/progress-bar target."""

    def __enter__(self) -> "_Ctx":
        return self

    def __exit__(self, *exc: object) -> Literal[False]:
        return False

    def __getattr__(self, name: str) -> Callable[..., None]:
        # Any method call on the handle (e.g. progress-bar .progress(v, text=...)) is a no-op.
        def _noop(*args: object, **kwargs: object) -> None:
            return None

        return _noop


class FakeStreamlit:
    """Lightweight fake `st` so panel render() runs end-to-end without a browser.

    Every display call is a no-op, every interactive widget returns a falsy
    default (so click/if branches don't fire), and every container is a context
    manager. `st.dialog` is an identity decorator, so a decorated function stays
    directly callable in a test.
    """

    def __init__(self) -> None:
        self.session_state = _FakeSessionState()
        self.sidebar = _Ctx()
        self.query_params: dict[str, str] = {}
        # Keys of buttons that should register as "clicked" this render, so the
        # branch body under `if st.button(..., key=k):` actually executes.
        self.clicked_keys: set[str] = set()
        # Every explicit widget key seen this run — real Streamlit raises on a duplicate, so we do
        # too (catches copy-paste key collisions across shared render helpers). Reset via new_run().
        self._seen_keys: set[object] = set()

    def new_run(self) -> None:
        """Reset per-run widget-key tracking (Streamlit clears element ids at each script run). Call
        between successive render passes in one test that legitimately re-renders the same widgets.
        """
        self._seen_keys.clear()

    def _register_key(self, key: object) -> None:
        """Record an explicit widget key, raising on a duplicate like StreamlitDuplicateElementKey."""
        if key is None:
            return
        if key in self._seen_keys:
            raise RuntimeError(f"StreamlitDuplicateElementKey: duplicate widget key {key!r} in one run")
        self._seen_keys.add(key)

    # --- containers (context managers) ---
    def columns(self, spec: "int | list[float]", **kwargs: object) -> list[_Ctx]:
        n = spec if isinstance(spec, int) else len(spec)
        return [_Ctx() for _ in range(n)]

    def tabs(self, labels: "list[str]", **kwargs: object) -> list[_Ctx]:
        return [_Ctx() for _ in labels]

    def expander(self, *args: object, **kwargs: object) -> _Ctx:
        return _Ctx()

    def container(self, *args: object, **kwargs: object) -> _Ctx:
        return _Ctx()

    def spinner(self, *args: object, **kwargs: object) -> _Ctx:
        return _Ctx()

    def progress(self, *args: object, **kwargs: object) -> "_Ctx":
        # st.progress(v) returns a bar handle whose .progress(v, text=...) updates it; _Ctx.progress
        # is itself a no-op via __getattr__, so returning a _Ctx makes both calls safe in tests.
        return _Ctx()

    def form(self, *args: object, **kwargs: object) -> _Ctx:
        return _Ctx()

    def form_submit_button(self, *args: object, **kwargs: object) -> bool:
        # Like button(): fires only if its key was pre-registered. Forms often omit an explicit
        # key, so also honour the label as a fallback key.
        key = kwargs.get("key") or (args[0] if args else None)
        self._register_key(kwargs.get("key"))
        return key in self.clicked_keys

    # --- dialog: decorator factory -> identity decorator ---
    def dialog(self, *dargs: object, **dkwargs: object) -> Callable[[Callable[..., object]], Callable[..., object]]:
        def decorator(func: Callable[..., object]) -> Callable[..., object]:
            return func

        return decorator

    # --- interactive widgets: falsy defaults so branches don't fire ---
    def button(self, *args: object, **kwargs: object) -> bool:
        # A button "fires" only if its key was pre-registered in clicked_keys,
        # so a test can drive a specific button-click branch deterministically.
        self._register_key(kwargs.get("key"))
        return kwargs.get("key") in self.clicked_keys

    def download_button(self, *args: object, **kwargs: object) -> bool:
        self._register_key(kwargs.get("key"))
        return False

    def checkbox(self, *args: object, **kwargs: object) -> bool:
        return False

    def file_uploader(self, *args: object, **kwargs: object) -> None:
        return None

    def slider(self, label: str, min_value: object = 0, value: object = None, **kwargs: object) -> object:
        return value if value is not None else min_value

    def selectbox(self, label: str, options: "tuple[object, ...]" = (), index: int = 0, **kwargs: object) -> object:
        opts = list(options)
        return opts[index] if opts else None

    # --- everything else (metric/markdown/write/success/warning/caption/... ) is a no-op ---
    def __getattr__(self, name: str) -> Callable[..., None]:
        def _noop(*args: object, **kwargs: object) -> None:
            return None

        return _noop


@pytest.fixture
def fake_st(monkeypatch: pytest.MonkeyPatch) -> FakeStreamlit:
    """Install a fake `st` into every skiresort_planner module that imports it.

    Modules do `import streamlit as st`, so we patch each module's `st`
    attribute. Panel render() / app functions then exercise all widget code
    paths without a browser. Covers the whole `skiresort_planner.ui` package
    plus the top-level `app` module (which drives the full render loop).
    """
    import importlib
    import pkgutil

    import skiresort_planner.ui as ui_pkg

    fake = FakeStreamlit()
    modules = [importlib.import_module(mi.name) for mi in pkgutil.iter_modules(ui_pkg.__path__, ui_pkg.__name__ + ".")]
    modules.append(importlib.import_module("skiresort_planner.app"))
    for module in modules:
        if hasattr(module, "st"):
            monkeypatch.setattr(module, "st", fake, raising=False)
    return fake


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
        return (
            self.base_elevation
            + lat * MapConfig.METERS_PER_DEGREE_EQUATOR * (self.slope_ns_pct / 100)
            - lon * MapConfig.METERS_PER_DEGREE_EQUATOR * (self.slope_ew_pct / 100)
        )

    def get_elevation_or_raise(self, lon: float, lat: float) -> float:
        """Return elevation, raising if None (never happens for mock)."""
        elev = self.get_elevation(lon=lon, lat=lat)
        assert elev is not None, "MockDEMService always returns elevation"
        return elev

    def get_elevations(self, lons, lats):  # noqa: ANN001, ANN201
        """Batch lookup mapping the formula `get_elevation` — mirrors DEMService's batch API for mocks."""
        return np.array([self.get_elevation(lon=lo, lat=la) for lo, la in zip(lons, lats, strict=True)], dtype=float)


class ConeDEMService(MockDEMService):
    """Radial cone: elevation = summit − radius·grade. Contours are circles, so the fall
    line ROTATES along any contour — the curved terrain that exposes contour drift that
    the planar MockDEMService (constant fall line) cannot.
    """

    def __init__(self, summit: float, grade_pct: float) -> None:
        """Args: summit elevation at origin; grade_pct radial slope (rise/run %)."""
        self.summit = summit
        self.grade_pct = grade_pct
        self._bounds = (-2.0, -2.0, 2.0, 2.0)

    def get_elevation(self, lon: float, lat: float) -> float | None:
        return (
            self.summit
            - float(
                ((lon * MapConfig.METERS_PER_DEGREE_EQUATOR) ** 2 + (lat * MapConfig.METERS_PER_DEGREE_EQUATOR) ** 2)
                ** 0.5
            )
            * self.grade_pct
            / 100.0
        )


class RoughDEMService(MockDEMService):
    """A steady south descent plus a sinusoidal bump train (rolling knolls) along the fall line.

    The bumps make the steepest-300m window sensitive to smoothing: rounding the corners
    shifts which sub-section is steepest, so a RAW fan proposal and its finish-smoothed self
    can straddle a difficulty band — the terrain shape that reproduces the blue→red-on-finish
    bug that planar/cone mocks are too smooth to show.
    """

    def __init__(self, slope_ns_pct: float = 20.0, bump_amp_m: float = 12.0, bump_wavelength_m: float = 180.0) -> None:
        """Args: mean N-S grade %, bump amplitude (m), bump wavelength (m) along the descent."""
        self.slope_ns_pct = slope_ns_pct
        self.bump_amp_m = bump_amp_m
        self.bump_wavelength_m = bump_wavelength_m
        self._bounds = (-2.0, -2.0, 2.0, 2.0)

    def get_elevation(self, lon: float, lat: float) -> float | None:
        import math

        y = lat * MapConfig.METERS_PER_DEGREE_EQUATOR
        return (
            3000.0
            + y * self.slope_ns_pct / 100.0
            + self.bump_amp_m * math.sin(2 * math.pi * y / self.bump_wavelength_m)
        )


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


@pytest.fixture
def cone_dem_steep() -> ConeDEMService:
    """Steep radial cone (25% terrain) — curved terrain for contour-drift tests."""
    return ConeDEMService(summit=4000.0, grade_pct=50.0)


@pytest.fixture
def rough_dem_bumpy() -> RoughDEMService:
    """20% descent with 12m knolls — bumpy terrain where finish-smoothing can shift difficulty."""
    return RoughDEMService(slope_ns_pct=20.0, bump_amp_m=12.0)


# =============================================================================
# RESORT GRAPH FIXTURES
# =============================================================================


@pytest.fixture
def empty_graph() -> ResortGraph:
    """Empty resort graph for testing graph operations from scratch."""
    return ResortGraph()


# =============================================================================
# EXACT-TOPOLOGY BUILDERS — shared by test_connectivity + test_route_planner
# =============================================================================
# These build precise ski graphs directly (real Node/PathSegment/Slope + add_lift) so the graph
# topology under test is exact. Slopes must carry a real segment chain — interior junction nodes are
# what stitch the resort together and what the ski graph walks per-segment.

BUILDER_BASE_ELEV = 1000.0
BUILDER_PEAK_ELEV = 2000.0


def add_node(graph: ResortGraph, nid: str, lon: float, lat: float, elev: float) -> None:
    """Materialise one node at (lon, lat, elev)."""
    graph.nodes[nid] = Node(id=nid, location=PathPoint(lon=lon, lat=lat, elevation=elev))


def add_slope_segment(graph: ResortGraph, sid: str, start: str, end: str) -> str:
    """A one-hop slope segment start->end (2-point geometry from the node locations)."""
    a, b = graph.nodes[start], graph.nodes[end]
    graph.segments[sid] = PathSegment(
        id=sid, name=sid, start_node_id=start, end_node_id=end, kind=SegmentKind.SLOPE, points=[a.location, b.location]
    )
    return sid


def add_slope(graph: ResortGraph, slid: str, top: str, bottom: str, *, via: list[str] | None = None) -> None:
    """A slope descending top -> bottom, optionally through interior junction nodes `via`, as a real
    chain of one-hop segments (interior nodes are what stitch the resort together).
    """
    chain = [top, *(via or []), bottom]
    seg_ids = [add_slope_segment(graph, f"{slid}_S{i}", chain[i], chain[i + 1]) for i in range(len(chain) - 1)]
    graph.slopes[slid] = Slope(id=slid, name=slid, segment_ids=seg_ids, start_node_id=top, end_node_id=bottom)


def build_ladder_core(graph: ResortGraph, dem: "MockDEMService", *, n_lifts: int, base: str = "B") -> None:
    """A base hub + n peaks, each reached by an uphill chairlift and returned by a downhill slope.

    Makes {base, peaks...} one strongly-connected component holding n_lifts lifts: from the base you
    can lift to any peak and ski back, and hop peak->peak via the base.
    """
    add_node(graph, base, lon=0.0, lat=0.0, elev=BUILDER_BASE_ELEV)
    for i in range(1, n_lifts + 1):
        peak = f"P{i}"
        # Farther peaks = longer lifts, so the longest in-core lift is deterministic (the last one).
        add_node(graph, peak, lon=0.0, lat=0.001 * i, elev=BUILDER_PEAK_ELEV)
        graph.add_lift(start_node_id=base, end_node_id=peak, lift_type=LiftType.CHAIRLIFT, dem=dem, name=f"Lift {i}")
        add_slope(graph, f"SL{i}", top=peak, bottom=base)


@pytest.fixture
def graph_with_nodes(mock_dem_blue_slope: MockDEMService) -> ResortGraph:
    """Graph with 3 nodes arranged vertically: summit → mid → valley."""
    graph = ResortGraph()
    dem = mock_dem_blue_slope

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
            lat=-1000 / MapConfig.METERS_PER_DEGREE_EQUATOR,
            elevation=dem.get_elevation_or_raise(lon=0.0, lat=-1000 / MapConfig.METERS_PER_DEGREE_EQUATOR),
        ),
    )
    graph.nodes["N3"] = Node(
        id="N3",
        location=PathPoint(
            lon=0.0,
            lat=-2000 / MapConfig.METERS_PER_DEGREE_EQUATOR,
            elevation=dem.get_elevation_or_raise(lon=0.0, lat=-2000 / MapConfig.METERS_PER_DEGREE_EQUATOR),
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
    return [
        PathPoint(
            lon=0.0,
            lat=-0 / MapConfig.METERS_PER_DEGREE_EQUATOR,
            elevation=dem.get_elevation_or_raise(lon=0.0, lat=-0 / MapConfig.METERS_PER_DEGREE_EQUATOR),
        ),
        PathPoint(
            lon=0.0,
            lat=-200 / MapConfig.METERS_PER_DEGREE_EQUATOR,
            elevation=dem.get_elevation_or_raise(lon=0.0, lat=-200 / MapConfig.METERS_PER_DEGREE_EQUATOR),
        ),
        PathPoint(
            lon=0.0,
            lat=-400 / MapConfig.METERS_PER_DEGREE_EQUATOR,
            elevation=dem.get_elevation_or_raise(lon=0.0, lat=-400 / MapConfig.METERS_PER_DEGREE_EQUATOR),
        ),
        PathPoint(
            lon=0.0,
            lat=-600 / MapConfig.METERS_PER_DEGREE_EQUATOR,
            elevation=dem.get_elevation_or_raise(lon=0.0, lat=-600 / MapConfig.METERS_PER_DEGREE_EQUATOR),
        ),
        PathPoint(
            lon=0.0,
            lat=-800 / MapConfig.METERS_PER_DEGREE_EQUATOR,
            elevation=dem.get_elevation_or_raise(lon=0.0, lat=-800 / MapConfig.METERS_PER_DEGREE_EQUATOR),
        ),
    ]


@pytest.fixture
def proposed_segment_blue(path_points_blue: list[PathPoint]) -> ProposedPathSegment:
    """Proposed segment: 800m long, 20% slope, blue difficulty."""
    return ProposedPathSegment(
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

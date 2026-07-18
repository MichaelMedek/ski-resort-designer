"""Tests for core-resort connectivity: the shared scipy primitive, the directed SCC core
detection, per-entity membership, and the disconnected-count summary.

Graphs are built directly (Node + Slope objects + add_lift) so each topology is exact — the
connectivity code reads only node-id endpoints, never geometry, so no path tracing is needed.
"""

import pytest

from skiresort_planner.constants import ConnectivityConfig, LiftType
from skiresort_planner.model.connectivity import component_labels
from skiresort_planner.model.node import Node
from skiresort_planner.model.path_point import PathPoint
from skiresort_planner.model.resort_graph import CoreMembership, ResortGraph
from skiresort_planner.model.road import Road
from skiresort_planner.model.slope import Slope
from tests_workflow.conftest import MockDEMService

BASE_ELEV = 1000.0
PEAK_ELEV = 2000.0


def _node(graph: ResortGraph, nid: str, lon: float, lat: float, elev: float) -> None:
    graph.nodes[nid] = Node(id=nid, location=PathPoint(lon=lon, lat=lat, elevation=elev))


def _slope(graph: ResortGraph, slid: str, top: str, bottom: str) -> None:
    """A slope descends top -> bottom. Geometry is irrelevant to connectivity, so segment_ids is []."""
    graph.slopes[slid] = Slope(id=slid, name=slid, segment_ids=[], start_node_id=top, end_node_id=bottom)


def _ladder_core(graph: ResortGraph, dem: MockDEMService, *, n_lifts: int, base: str = "B") -> None:
    """A base hub + n peaks, each reached by an uphill chairlift and returned by a downhill slope.

    That makes {base, peaks...} one strongly-connected component holding n_lifts lifts: from the
    base you can lift to any peak and ski back, and hop peak->peak via the base.
    """
    _node(graph, base, lon=0.0, lat=0.0, elev=BASE_ELEV)
    for i in range(1, n_lifts + 1):
        peak = f"P{i}"
        # Farther peaks = longer lifts, so the longest in-core lift is deterministic (the last one).
        _node(graph, peak, lon=0.0, lat=0.001 * i, elev=PEAK_ELEV)
        graph.add_lift(start_node_id=base, end_node_id=peak, lift_type=LiftType.CHAIRLIFT, dem=dem, name=f"Lift {i}")
        _slope(graph, f"SL{i}", top=peak, bottom=base)


@pytest.fixture
def dem() -> MockDEMService:
    return MockDEMService(base_elevation=2500.0, slope_ns_pct=20.0, slope_ew_pct=0.0)


# =============================================================================
# 1. Shared undirected primitive — parity + isolated-node labelling
# =============================================================================


class TestComponentLabels:
    def test_isolated_node_gets_own_component(self) -> None:
        """A node with no incident edge must still be labelled (its own component)."""
        labels = component_labels(["a", "b", "c"], [("a", "b")], strong=False)
        assert labels["a"] == labels["b"]
        assert labels["c"] != labels["a"], "isolated 'c' must be its own component"

    def test_undirected_grouping(self) -> None:
        """Two disjoint edge groups → two components; membership is by equivalence class."""
        labels = component_labels([0, 1, 2, 3], [(0, 1), (2, 3)], strong=False)
        assert labels[0] == labels[1]
        assert labels[2] == labels[3]
        assert labels[0] != labels[2]

    def test_strong_vs_weak_on_one_way_edge(self) -> None:
        """A single directed edge: weakly connected, but NOT strongly connected."""
        weak = component_labels(["x", "y"], [("x", "y")], strong=False)
        strong = component_labels(["x", "y"], [("x", "y")], strong=True)
        assert weak["x"] == weak["y"]
        assert strong["x"] != strong["y"], "one-way edge is not strongly connected"


# =============================================================================
# 2. <5-lift gate — no false alarms early
# =============================================================================


class TestCoreLiftGate:
    def test_four_lift_cycle_has_no_core(self, empty_graph: ResortGraph, dem: MockDEMService) -> None:
        """Below MIN_CORE_LIFTS the largest SCC does NOT count as a core → get_core_resort is None."""
        assert ConnectivityConfig.MIN_CORE_LIFTS == 5
        _ladder_core(empty_graph, dem, n_lifts=4)
        assert empty_graph.get_core_resort() is None
        assert empty_graph.count_disconnected() == 0

    def test_membership_is_no_core_yet_when_no_core(self, empty_graph: ResortGraph, dem: MockDEMService) -> None:
        _ladder_core(empty_graph, dem, n_lifts=4)
        assert (
            empty_graph.entity_membership(start_node_id="P1", end_node_id="B", core=None) == CoreMembership.NO_CORE_YET
        )


# =============================================================================
# 3. >=5-lift core — everything connected is in-core
# =============================================================================


class TestCoreResort:
    def test_five_lift_ladder_forms_core(self, empty_graph: ResortGraph, dem: MockDEMService) -> None:
        _ladder_core(empty_graph, dem, n_lifts=5)
        core = empty_graph.get_core_resort()
        assert core is not None
        assert core.lift_count == 5
        assert core.node_ids == {"B", "P1", "P2", "P3", "P4", "P5"}
        assert empty_graph.count_disconnected() == 0

    def test_longest_core_lift_is_named(self, empty_graph: ResortGraph, dem: MockDEMService) -> None:
        """The warning names the longest in-core lift — the farthest peak's lift here."""
        _ladder_core(empty_graph, dem, n_lifts=5)
        core = empty_graph.get_core_resort()
        assert core is not None
        assert core.longest_lift_name == "Lift 5"

    def test_all_core_entities_in_core(self, empty_graph: ResortGraph, dem: MockDEMService) -> None:
        _ladder_core(empty_graph, dem, n_lifts=5)
        core = empty_graph.get_core_resort()
        for slope in empty_graph.slopes.values():
            assert (
                empty_graph.entity_membership(
                    start_node_id=slope.start_node_id, end_node_id=slope.end_node_id, core=core
                )
                == CoreMembership.IN_CORE
            )


# =============================================================================
# 4 & 5. Disconnected slope (dead-end valley) and disconnected lift
# =============================================================================


class TestDisconnected:
    def test_dead_end_valley_slope_is_disconnected(self, empty_graph: ResortGraph, dem: MockDEMService) -> None:
        """A run from a core peak down to a valley node with no lift back → DISCONNECTED."""
        _ladder_core(empty_graph, dem, n_lifts=5)
        _node(empty_graph, "V", lon=0.0, lat=-0.01, elev=500.0)  # below the base, no return lift
        _slope(empty_graph, "SL_dead", top="P1", bottom="V")
        core = empty_graph.get_core_resort()
        assert (
            empty_graph.entity_membership(start_node_id="P1", end_node_id="V", core=core) == CoreMembership.DISCONNECTED
        )
        assert empty_graph.count_disconnected() == 1

    def test_isolated_lift_is_disconnected(self, empty_graph: ResortGraph, dem: MockDEMService) -> None:
        """A lift into an isolated pocket (neither end in the core) → DISCONNECTED."""
        _ladder_core(empty_graph, dem, n_lifts=5)
        _node(empty_graph, "X", lon=0.02, lat=0.0, elev=BASE_ELEV)
        _node(empty_graph, "Y", lon=0.02, lat=0.01, elev=PEAK_ELEV)
        empty_graph.add_lift(start_node_id="X", end_node_id="Y", lift_type=LiftType.CHAIRLIFT, dem=dem, name="Orphan")
        core = empty_graph.get_core_resort()
        assert (
            empty_graph.entity_membership(start_node_id="X", end_node_id="Y", core=core) == CoreMembership.DISCONNECTED
        )
        assert empty_graph.count_disconnected() == 1


# =============================================================================
# 6. Roads are excluded from connectivity
# =============================================================================


class TestRoadExcluded:
    def test_road_does_not_bridge_components(self, empty_graph: ResortGraph, dem: MockDEMService) -> None:
        """A road between the core and an isolated pocket must NOT make the pocket in-core."""
        _ladder_core(empty_graph, dem, n_lifts=5)
        _node(empty_graph, "X", lon=0.02, lat=0.0, elev=BASE_ELEV)
        _node(empty_graph, "Y", lon=0.02, lat=0.01, elev=PEAK_ELEV)
        empty_graph.add_lift(start_node_id="X", end_node_id="Y", lift_type=LiftType.CHAIRLIFT, dem=dem, name="Orphan")
        # A road from the core base to the isolated pocket — connectivity must ignore it.
        empty_graph.roads["R1"] = Road(id="R1", name="Access", segment_ids=[], start_node_id="B", end_node_id="X")
        core = empty_graph.get_core_resort()
        assert core is not None
        assert "X" not in core.node_ids and "Y" not in core.node_ids


# =============================================================================
# 7. Directionality driven by LiftConfig.UPHILL_ONLY (not hardcoded types)
# =============================================================================


class TestDirectionality:
    def test_bidirectional_gondola_joins_core(self, empty_graph: ResortGraph, dem: MockDEMService) -> None:
        """A gondola (both ways) from a core node to a new node makes that node strongly connected."""
        _ladder_core(empty_graph, dem, n_lifts=5)
        _node(empty_graph, "G", lon=-0.01, lat=0.005, elev=PEAK_ELEV)
        empty_graph.add_lift(start_node_id="B", end_node_id="G", lift_type=LiftType.GONDOLA, dem=dem, name="Gondola")
        core = empty_graph.get_core_resort()
        assert core is not None
        assert "G" in core.node_ids, "bidirectional gondola should pull G into the core SCC"
        assert empty_graph.entity_membership(start_node_id="B", end_node_id="G", core=core) == CoreMembership.IN_CORE

    def test_one_way_chairlift_deadend_is_disconnected(self, empty_graph: ResortGraph, dem: MockDEMService) -> None:
        """A one-way chairlift to a node with no ski-back route leaves that node out of the core."""
        _ladder_core(empty_graph, dem, n_lifts=5)
        _node(empty_graph, "C", lon=-0.01, lat=0.005, elev=PEAK_ELEV)
        empty_graph.add_lift(start_node_id="B", end_node_id="C", lift_type=LiftType.CHAIRLIFT, dem=dem, name="Dead")
        core = empty_graph.get_core_resort()
        assert core is not None
        assert "C" not in core.node_ids
        assert (
            empty_graph.entity_membership(start_node_id="B", end_node_id="C", core=core) == CoreMembership.DISCONNECTED
        )

"""Tests for the route planner: per-difficulty-cap best A→B routes over the ski graph.

Difficulty is an HONEST computation input: best_routes precomputes, for every cap (green→black),
the best routes using ONLY slopes up to that band (harder slopes pruned from the graph). Under each
cap two criteria are computed — FEWEST_LIFTS and SHORTEST_SLOPE (distance + light drop weight).

Topologies are built with the shared conftest builders so each route is exact. Tests assert concrete
return values — the chosen node path, the mapped entities, the premise (difficulty_cap), and totals.
"""

from dataclasses import replace

import pytest

from skiresort_planner.constants import LiftType, MapConfig, RoutePlannerConfig, SlopeConfig
from skiresort_planner.model.path_point import PathPoint
from skiresort_planner.model.path_segment import PathSegment, SegmentKind
from skiresort_planner.model.resort_graph import ResortGraph
from skiresort_planner.model.routing import Route, RouteCriterion, RoutePlanner, RouteStep, routes_for_cap
from skiresort_planner.model.slope import Slope
from tests_workflow.conftest import MockDEMService, add_node, add_slope

M = MapConfig.METERS_PER_DEGREE_EQUATOR
BLACK = SlopeConfig.DIFFICULTIES[-1]  # the broadest cap — every slope allowed


@pytest.fixture
def dem() -> MockDEMService:
    return MockDEMService(base_elevation=2500.0, slope_ns_pct=20.0, slope_ew_pct=0.0)


def _at_cap(routes: list[Route], cap: str) -> list[Route]:
    """The precomputed routes whose premise is exactly `cap`."""
    return routes_for_cap(routes, max_difficulty=cap)


def _route_for(routes: list[Route], criterion: RouteCriterion, *, cap: str = BLACK) -> Route:
    """The route winning `criterion` under `cap` (each criterion maps to one deduped route per cap)."""
    for r in _at_cap(routes, cap):
        if criterion in r.criteria:
            return r
    raise AssertionError(f"no route won {criterion} at cap {cap}")


class TestFeasibility:
    def test_unknown_nodes_yield_no_routes(self, empty_graph: ResortGraph, dem: MockDEMService) -> None:
        add_node(empty_graph, "A", 0.0, 0.0, 2000.0)
        add_node(empty_graph, "B", 0.0, -0.001, 1900.0)
        add_slope(empty_graph, "SL1", top="A", bottom="B")
        planner = RoutePlanner(empty_graph)
        assert planner.best_routes("A", "NOPE") == []
        assert planner.best_routes("GHOST", "B") == []

    def test_unreachable_target_yields_no_routes(self, empty_graph: ResortGraph, dem: MockDEMService) -> None:
        """B sits in a separate pocket with no directed path from A → no routes at any cap (not a crash)."""
        add_node(empty_graph, "A", 0.0, 0.0, 2000.0)
        add_node(empty_graph, "B", 0.0, -0.001, 1900.0)
        add_slope(empty_graph, "SL1", top="A", bottom="B")
        add_node(empty_graph, "C", 0.01, 0.0, 2000.0)
        add_node(empty_graph, "D", 0.01, -0.001, 1900.0)
        add_slope(empty_graph, "SL2", top="C", bottom="D")
        assert RoutePlanner(empty_graph).best_routes("A", "D") == []


class TestSingleSlopeRoute:
    def test_one_slope_route_maps_to_that_slope(self, empty_graph: ResortGraph, dem: MockDEMService) -> None:
        add_node(empty_graph, "Peak", 0.0, 0.0, 2000.0)
        add_node(empty_graph, "Base", 0.0, -1000 / M, 1400.0)
        add_slope(empty_graph, "SL1", top="Peak", bottom="Base")

        routes = _at_cap(RoutePlanner(empty_graph).best_routes("Peak", "Base"), BLACK)
        assert len(routes) == 1, "both criteria agree on the only route → one deduped result"
        route = routes[0]
        assert route.node_path == ("Peak", "Base")
        assert [s.entity_id for s in route.steps] == ["SL1"]
        assert route.lift_count == 0
        assert route.total_slope_drop_m == pytest.approx(600.0)
        assert set(route.criteria) == {RouteCriterion.FEWEST_LIFTS, RouteCriterion.SHORTEST_SLOPE}  # both P2P optima

    def test_interior_junction_maps_to_owning_slope(self, empty_graph: ResortGraph, dem: MockDEMService) -> None:
        """A route through a slope's mid-chain node still reads as one slope step (not per-segment)."""
        add_node(empty_graph, "Peak", 0.0, 0.0, 2000.0)
        add_node(empty_graph, "Mid", 0.0, -500 / M, 1700.0)
        add_node(empty_graph, "Base", 0.0, -1000 / M, 1400.0)
        add_slope(empty_graph, "SL1", top="Peak", bottom="Base", via=["Mid"])

        route = _route_for(RoutePlanner(empty_graph).best_routes("Peak", "Base"), RouteCriterion.FEWEST_LIFTS)
        assert route.node_path == ("Peak", "Mid", "Base")
        assert [s.entity_id for s in route.steps] == ["SL1"], "two segments of one slope collapse to one step"


class TestLiftRoutes:
    def test_bidirectional_gondola_usable_in_returning_route(
        self, empty_graph: ResortGraph, dem: MockDEMService
    ) -> None:
        """Top→Base is reachable ONLY via the gondola's reverse (down) edge — the exact route."""
        add_node(empty_graph, "Base", 0.0, 0.0, 1400.0)
        add_node(empty_graph, "Top", 0.0, 500 / M, 2000.0)
        empty_graph.add_lift(start_node_id="Base", end_node_id="Top", lift_type=LiftType.GONDOLA, dem=dem, name="Gondi")

        routes = _at_cap(RoutePlanner(empty_graph).best_routes("Top", "Base"), BLACK)
        assert len(routes) == 1, "one deduped route: gondola down"
        route = routes[0]
        assert route.node_path == ("Top", "Base")
        assert route.lift_count == 1
        assert [(s.is_lift, s.name) for s in route.steps] == [(True, "Gondi")]
        assert route.total_slope_length_m == 0.0, "no skiing on a gondola-only route"

    def test_bidirectional_lift_cycle_solves_fast_without_hanging(
        self, empty_graph: ResortGraph, dem: MockDEMService
    ) -> None:
        """A bidirectional lift up to a peak + a slope down forms a directed CYCLE (A→peak→B and
        peak→A). Every criterion must terminate quickly — a genuine shortest-path cost is cycle-safe.
        Regression: an invalid non-shortest cost once looped forever here.
        """
        add_node(empty_graph, "A", 0.0, 0.0, 1800.0)
        add_node(empty_graph, "B", 0.0, -2000 / M, 1000.0)
        add_slope(empty_graph, "Low", top="A", bottom="B")
        add_node(empty_graph, "Peak", 0.0, 500 / M, 2400.0)
        empty_graph.add_lift(start_node_id="A", end_node_id="Peak", lift_type=LiftType.GONDOLA, dem=dem, name="Up")
        add_slope(empty_graph, "Down", top="Peak", bottom="A")

        routes = _at_cap(RoutePlanner(empty_graph).best_routes("A", "B"), BLACK)
        won = {c for r in routes for c in r.criteria}
        assert won == {RouteCriterion.FEWEST_LIFTS, RouteCriterion.SHORTEST_SLOPE}, (
            "both P2P criteria resolve on a cycle"
        )
        fewest = _route_for(routes, RouteCriterion.FEWEST_LIFTS)
        assert fewest.node_path == ("A", "B") and fewest.lift_count == 0, "fewest-lifts takes the direct slope"

    def test_route_follows_lift_cable_geometry_not_straight(
        self, empty_graph: ResortGraph, dem: MockDEMService
    ) -> None:
        """A route crossing a lift traces the sagged cable_points, not a straight chord between stations."""
        add_node(empty_graph, "Base", 0.0, 0.0, 1400.0)
        add_node(empty_graph, "Top", 0.0, 500 / M, 2000.0)
        lift = empty_graph.add_lift(
            start_node_id="Base", end_node_id="Top", lift_type=LiftType.GONDOLA, dem=dem, name="Gondi"
        )
        assert len(lift.cable_points) > 2, "the cable has interior sag points to trace"

        route = _at_cap(RoutePlanner(empty_graph).best_routes("Top", "Base"), BLACK)[0]  # ride down
        pts = route.path_points
        # An interior cable point (elevation strictly between the two stations) must appear in the polyline.
        interior = [p for p in lift.cable_points if 1400.0 < p.elevation < 2000.0]
        assert interior, "cable has a mid point"
        assert any((p.lon, p.lat, p.elevation) in pts for p in interior), "route traces the cable, not a chord"
        # Riding Top→Base reverses the stored Base→Top order: descending elevation overall.
        assert pts[0][2] > pts[-1][2], "the ride descends Top→Base"


def _ladder_scc(graph: ResortGraph, dem: MockDEMService, n_lifts: int) -> None:
    """A base hub 'B' + n peaks, each reached by a chairlift UP and a slope back DOWN → one SCC through
    B holding n_lifts lifts (every lift reachable-and-returnable from B).
    """
    add_node(graph, "B", 0.0, 0.0, 1400.0)
    for i in range(1, n_lifts + 1):
        peak = f"P{i}"
        add_node(graph, peak, 0.0, (500 * i) / M, 2000.0)
        graph.add_lift(start_node_id="B", end_node_id=peak, lift_type=LiftType.CHAIRLIFT, dem=dem, name=f"L{i}")
        add_slope(graph, f"S{i}", top=peak, bottom="B")


class TestScenicTour:
    """Closed-walk 'visit every reachable lift' tours (start == end). Completeness is EXACT (every
    reachable lift is a TSP city); the ORDER is networkx's approximate ATSP (deterministic via seed).
    """

    def test_closed_tour_visits_every_reachable_lift_and_returns(
        self, empty_graph: ResortGraph, dem: MockDEMService
    ) -> None:
        _ladder_scc(empty_graph, dem, n_lifts=3)
        routes = RoutePlanner(empty_graph).best_routes("B", "B")  # closed loop
        assert routes, "a scenic tour exists"
        for r in routes:
            assert r.is_scenic
            assert r.node_path[0] == r.node_path[-1] == "B", "closed loop returns to start"
            assert r.scenic_lifts_visited == 3 == r.scenic_lifts_target, "every reachable lift ridden"
            ridden = {s.entity_id for s in r.steps if s.is_lift}
            assert ridden == {"L1", "L2", "L3"}, "all three lifts appear as ridden steps"

    def test_open_route_has_no_scenic_only_shortest(self, empty_graph: ResortGraph, dem: MockDEMService) -> None:
        _ladder_scc(empty_graph, dem, n_lifts=2)
        routes = RoutePlanner(empty_graph).best_routes("B", "P1")  # start != end
        assert routes, "a point-to-point route exists"
        assert not any(r.is_scenic for r in routes), "start != end never yields scenic tours"

    def test_scenic_excludes_a_disconnected_sub_resort(self, empty_graph: ResortGraph, dem: MockDEMService) -> None:
        """A lift in a disconnected pocket (no way back to the start's SCC) is NOT in the tour."""
        _ladder_scc(empty_graph, dem, n_lifts=1)  # B ⇄ P1 via L1
        # Disconnected pocket: X→Y lift with a down slope among themselves, unreachable from B.
        add_node(empty_graph, "X", 1.0, 0.0, 1400.0)
        add_node(empty_graph, "Y", 1.0, 500 / M, 2000.0)
        empty_graph.add_lift(start_node_id="X", end_node_id="Y", lift_type=LiftType.CHAIRLIFT, dem=dem, name="LX")
        add_slope(empty_graph, "SX", top="Y", bottom="X")

        routes = RoutePlanner(empty_graph).best_routes("B", "B")
        for r in routes:
            ridden = {s.entity_id for s in r.steps if s.is_lift}
            assert "LX" not in ridden, "the disconnected lift is unreachable → excluded"
            assert r.scenic_lifts_target == 1, "only L1 is reachable from B"

    def test_scenic_tour_is_deterministic(self, empty_graph: ResortGraph, dem: MockDEMService) -> None:
        _ladder_scc(empty_graph, dem, n_lifts=3)
        a = RoutePlanner(empty_graph).best_routes("B", "B")
        b = RoutePlanner(empty_graph).best_routes("B", "B")
        assert [r.node_path for r in a] == [r.node_path for r in b], "seeded ATSP → identical tours"

    def test_scenic_node_path_is_a_valid_edge_sequence(self, empty_graph: ResortGraph, dem: MockDEMService) -> None:
        _ladder_scc(empty_graph, dem, n_lifts=3)
        planner = RoutePlanner(empty_graph)
        for r in planner.best_routes("B", "B"):
            for a, b in zip(r.node_path, r.node_path[1:], strict=False):
                assert (a, b) in planner._owner, f"({a},{b}) must be a real graph edge"


class TestCriteriaDiverge:
    def test_shortest_slope_takes_lift_shortcut_while_fewest_lifts_stays_on_snow(
        self, empty_graph: ResortGraph, dem: MockDEMService
    ) -> None:
        """A long lift-free slope vs a shorter slope-lift-slope path. Shortest-slope takes the lift
        shortcut (less skiing); fewest-lifts stays on the long lift-free run — two DIFFERENT routes.
        """
        add_node(empty_graph, "A", 0.0, 0.0, 2000.0)
        add_node(empty_graph, "B", 0.0, -3000 / M, 1000.0)
        add_node(empty_graph, "Far", 2000 / M, -1500 / M, 1500.0)
        add_slope(empty_graph, "LongWay", top="A", bottom="B", via=["Far"])  # long, lift-free
        add_node(empty_graph, "Mid", 0.0, -300 / M, 1900.0)
        add_node(empty_graph, "Ledge", 0.0, -280 / M, 1950.0)
        add_slope(empty_graph, "Short1", top="A", bottom="Mid")
        empty_graph.add_lift(
            start_node_id="Mid", end_node_id="Ledge", lift_type=LiftType.CHAIRLIFT, dem=dem, name="Hop"
        )
        add_slope(empty_graph, "Short2", top="Ledge", bottom="B")

        routes = RoutePlanner(empty_graph).best_routes("A", "B")
        fewest = _route_for(routes, RouteCriterion.FEWEST_LIFTS)
        shortest = _route_for(routes, RouteCriterion.SHORTEST_SLOPE)
        assert fewest.node_path == ("A", "Far", "B"), "fewest-lifts stays on the lift-free run"
        assert fewest.lift_count == 0
        assert shortest.node_path == ("A", "Mid", "Ledge", "B"), "shortest-slope takes the lift shortcut"
        assert shortest.lift_count == 1
        assert shortest.total_slope_length_m < fewest.total_slope_length_m, "the shortcut skis less"
        assert fewest.node_path != shortest.node_path, "the two criteria genuinely diverge"

    def test_shortest_slope_prefers_less_skiing_over_a_down_lift(
        self, empty_graph: ResortGraph, dem: MockDEMService
    ) -> None:
        """Shortest-slope minimises skied distance (drop folded in lightly): skiing a short top then
        riding a down-gondola skis far less than the all-ski run, so shortest-slope takes the gondola.
        """
        add_node(empty_graph, "A", 0.0, 0.0, 2000.0)
        add_node(empty_graph, "B", 0.0, -3000 / M, 1000.0)
        add_slope(empty_graph, "AllSki", top="A", bottom="B")  # long ski
        add_node(empty_graph, "Mlow", 0.0, -200 / M, 1900.0)
        add_slope(empty_graph, "TopBit", top="A", bottom="Mlow")  # short ski
        empty_graph.add_lift(
            start_node_id="B", end_node_id="Mlow", lift_type=LiftType.GONDOLA, dem=dem, name="DownGondi"
        )

        routes = RoutePlanner(empty_graph).best_routes("A", "B")
        shortest = _route_for(routes, RouteCriterion.SHORTEST_SLOPE)
        fewest = _route_for(routes, RouteCriterion.FEWEST_LIFTS)
        assert shortest.node_path == ("A", "Mlow", "B"), "shortest-slope rides the gondola down"
        assert fewest.node_path == ("A", "B") and fewest.lift_count == 0
        assert shortest.total_slope_length_m < fewest.total_slope_length_m


class TestDifficultyCapIsHonest:
    """Difficulty is a COMPUTATION premise, not a post-filter: capping at green computes the shortest
    route over ONLY green/blue slopes (harder ones pruned), so a reachable green route is never hidden
    behind a "no path" — the exact dishonesty the redesign removes.
    """

    def test_green_cap_uses_the_gentle_detour_black_cap_uses_the_plunge(
        self, empty_graph: ResortGraph, dem: MockDEMService
    ) -> None:
        add_node(empty_graph, "A", 0.0, 0.0, 2000.0)
        add_node(empty_graph, "B", 0.0, -300 / M, 1400.0)  # 600m over 300m ⇒ black
        add_slope(empty_graph, "BlackRun", top="A", bottom="B")
        add_node(empty_graph, "W", -4000 / M, -150 / M, 1700.0)  # long, shallow legs ⇒ green
        add_slope(empty_graph, "GreenRun", top="A", bottom="B", via=["W"])
        assert empty_graph.slopes["BlackRun"].get_difficulty(segments=empty_graph.segments) == "black"
        assert empty_graph.slopes["GreenRun"].get_difficulty(segments=empty_graph.segments) == "green"

        routes = RoutePlanner(empty_graph).best_routes("A", "B")
        # Green cap: the black plunge is pruned, so only the green detour exists — HONEST, not "no path".
        green = _at_cap(routes, "green")
        assert green, "a green route exists and must be found under the green cap"
        assert all(r.node_path == ("A", "W", "B") and r.max_difficulty == "green" for r in green)
        assert all(r.difficulty_cap == "green" for r in green), "premise is recorded on each route"
        # Black cap: the short plunge is now the shortest slope route.
        shortest_black = _route_for(routes, RouteCriterion.SHORTEST_SLOPE, cap="black")
        assert shortest_black.node_path == ("A", "B") and shortest_black.max_difficulty == "black"

    def test_no_route_under_cap_but_a_harder_one_exists(self, empty_graph: ResortGraph, dem: MockDEMService) -> None:
        # Only a black run connects A→B. Under the green cap there is genuinely NO route; under black
        # there is. The per-cap result makes that distinction honestly.
        add_node(empty_graph, "A", 0.0, 0.0, 2000.0)
        add_node(empty_graph, "B", 0.0, -300 / M, 1400.0)  # steep ⇒ black
        add_slope(empty_graph, "BlackRun", top="A", bottom="B")
        assert empty_graph.slopes["BlackRun"].get_difficulty(segments=empty_graph.segments) == "black"

        routes = RoutePlanner(empty_graph).best_routes("A", "B")
        assert _at_cap(routes, "green") == [], "no green route exists"
        assert _at_cap(routes, "black"), "the black route exists under the black cap"


class TestRouteStats:
    def test_totals_aggregate_slopes_and_lifts(self, empty_graph: ResortGraph, dem: MockDEMService) -> None:
        """A lift then two slopes to a distinct base: lift_count and slope drop aggregate correctly."""
        add_node(empty_graph, "Start", 0.0, 0.0, 1400.0)
        add_node(empty_graph, "Top", 0.0, 500 / M, 2000.0)
        add_node(empty_graph, "Mid", 0.0, 250 / M, 1700.0)
        add_node(empty_graph, "Valley", 0.0, -500 / M, 1000.0)
        empty_graph.add_lift(start_node_id="Start", end_node_id="Top", lift_type=LiftType.CHAIRLIFT, dem=dem, name="Up")
        add_slope(empty_graph, "TopRun", top="Top", bottom="Mid")
        add_slope(empty_graph, "MidRun", top="Mid", bottom="Valley")

        route = _route_for(RoutePlanner(empty_graph).best_routes("Start", "Valley"), RouteCriterion.FEWEST_LIFTS)
        assert route.lift_count == 1
        assert route.total_slope_drop_m == pytest.approx(1000.0)  # 2000→1700→1000
        assert [(s.is_lift, s.name) for s in route.steps] == [(True, "Up"), (False, "TopRun"), (False, "MidRun")]


class TestRouteGeometry:
    """Route.path_points traces the ACTUAL slope geometry (bent segments), not straight node-to-node,
    and orients each segment to the direction the route skis it. Lifts stay straight node-to-node.
    """

    def _bent_slope(self, graph: ResortGraph, slid: str, top: str, bottom: str, bend: PathPoint) -> None:
        """A one-segment slope top→bottom whose geometry detours through `bend` (a non-collinear point)."""
        a, b = graph.nodes[top], graph.nodes[bottom]
        seg = PathSegment(
            id=f"{slid}_S0",
            name=f"{slid}_S0",
            start_node_id=top,
            end_node_id=bottom,
            kind=SegmentKind.SLOPE,
            points=[a.location, bend, b.location],
        )
        graph.segments[seg.id] = seg
        graph.slopes[slid] = Slope(id=slid, name=slid, segment_ids=[seg.id], start_node_id=top, end_node_id=bottom)

    def test_route_follows_bent_segment_geometry(self, empty_graph: ResortGraph) -> None:
        add_node(empty_graph, "A", 0.0, 0.0, 2000.0)
        add_node(empty_graph, "B", 0.0, -1000 / M, 1000.0)
        bend = PathPoint(lon=500 / M, lat=-500 / M, elevation=1500.0)  # off the straight A→B chord
        self._bent_slope(empty_graph, "Run", top="A", bottom="B", bend=bend)

        route = _at_cap(RoutePlanner(empty_graph).best_routes("A", "B"), BLACK)[0]
        pts = route.path_points
        assert pts[0] == (0.0, 0.0, 2000.0) and pts[-1] == (0.0, -1000 / M, 1000.0), "endpoints A→B"
        assert (bend.lon, bend.lat, bend.elevation) in pts, "the bend point is traced, not skipped"

    def test_segment_points_oriented_to_travel_direction(self, empty_graph: ResortGraph) -> None:
        # points are stored A→B; a route skiing A→B must emit them in descent order, not reversed.
        add_node(empty_graph, "A", 0.0, 0.0, 2000.0)
        add_node(empty_graph, "B", 0.0, -1000 / M, 1000.0)
        bend = PathPoint(lon=500 / M, lat=-500 / M, elevation=1500.0)
        self._bent_slope(empty_graph, "Run", top="A", bottom="B", bend=bend)  # stored A→B

        route = _at_cap(RoutePlanner(empty_graph).best_routes("A", "B"), BLACK)[0]
        elevs = [z for _, _, z in route.path_points]
        assert elevs == sorted(elevs, reverse=True), "points run in descent order A→bend→B, not reversed"


def _mk_route(*, difficulty_cap: str, max_difficulty: str = "green") -> Route:
    """A minimal Route carrying only what routes_for_cap inspects (its computation premise)."""
    return Route(
        node_path=("A", "B"),
        path_points=(),
        steps=(RouteStep(is_lift=False, entity_id="SL1", name="SL1", detail=max_difficulty),),
        total_slope_length_m=0.0,
        total_slope_drop_m=0.0,
        lift_count=0,
        max_difficulty=max_difficulty,
        difficulty_cap=difficulty_cap,
        criteria=(RouteCriterion.FEWEST_LIFTS,),
    )


class TestRoutesForCap:
    def test_selects_only_the_matching_premise(self) -> None:
        green = _mk_route(difficulty_cap="green")
        red = _mk_route(difficulty_cap="red")
        black = _mk_route(difficulty_cap="black")
        assert routes_for_cap([green, red, black], max_difficulty="red") == [red]


class TestRouteColor:
    """A route's overlay colour is keyed by its criterion, not its list position — so a colour always
    means the same metric regardless of ordering, and scenic tones stay in their metric's hue family.
    """

    @staticmethod
    def _route_with(criterion: RouteCriterion) -> Route:
        r = _mk_route(difficulty_cap="black")
        return replace(r, criteria=(criterion,))

    def test_color_is_keyed_by_criterion(self) -> None:
        for c in RouteCriterion:
            assert self._route_with(c).color == RoutePlannerConfig.ROUTE_COLORS[c.value]

    def test_scenic_shares_hue_family_with_its_base_metric(self) -> None:
        # Scenic is a darker tone of the same hue: the dominant RGB channel matches its base metric.
        for scenic, base in (
            (RouteCriterion.SCENIC_FEWEST_LIFTS, RouteCriterion.FEWEST_LIFTS),
            (RouteCriterion.SCENIC_SHORTEST_SLOPE, RouteCriterion.SHORTEST_SLOPE),
        ):
            sc = self._route_with(scenic).color
            bc = self._route_with(base).color
            assert sc[:3].index(max(sc[:3])) == bc[:3].index(max(bc[:3])), "same dominant channel = same hue"
            assert sum(sc[:3]) < sum(bc[:3]), "scenic is a DARKER tone than its shortest-path base"

    def test_empty_when_no_route_computed_for_that_cap(self) -> None:
        assert routes_for_cap([_mk_route(difficulty_cap="black")], max_difficulty="green") == []


class TestRouteStatsFieldsGone:
    def test_route_has_no_highest_elev_field(self) -> None:
        # The scenic-only stat was removed; Max Difficulty is the per-route headline stat now.
        assert not hasattr(_mk_route(difficulty_cap="black"), "highest_elev_m")

    def test_no_lift_type_filter_concept_remains(self) -> None:
        # The dishonest lift filter was removed entirely — Route no longer needs allowed-lift state.
        from skiresort_planner.ui import context

        assert not hasattr(context.RoutePlanContext(), "filter_lift_types")

"""Tests for the route planner: the four best-by-criterion A→B routes over the ski graph.

Topologies are built with the shared conftest builders (add_node/add_slope/add_slope_segment +
add_lift) so each route is exact. Tests assert concrete return values — the chosen node path, the
mapped entities, and the per-route totals — not the implementation.
"""

import pytest

from skiresort_planner.constants import LiftType, MapConfig
from skiresort_planner.model.resort_graph import ResortGraph
from skiresort_planner.model.routing import Route, RouteCriterion, RoutePlanner, RouteStep, filter_routes
from tests_workflow.conftest import MockDEMService, add_node, add_slope

M = MapConfig.METERS_PER_DEGREE_EQUATOR


@pytest.fixture
def dem() -> MockDEMService:
    return MockDEMService(base_elevation=2500.0, slope_ns_pct=20.0, slope_ew_pct=0.0)


def _route_for(routes: list[Route], criterion: RouteCriterion) -> Route:
    """The single route that wins `criterion` (each criterion maps to exactly one deduped route)."""
    for r in routes:
        if criterion in r.criteria:
            return r
    raise AssertionError(f"no route won {criterion}")


class TestFeasibility:
    def test_unknown_nodes_yield_no_routes(self, empty_graph: ResortGraph, dem: MockDEMService) -> None:
        add_node(empty_graph, "A", 0.0, 0.0, 2000.0)
        add_node(empty_graph, "B", 0.0, -0.001, 1900.0)
        add_slope(empty_graph, "SL1", top="A", bottom="B")
        planner = RoutePlanner(empty_graph)
        assert planner.best_routes("A", "NOPE") == []
        assert planner.best_routes("GHOST", "B") == []

    def test_unreachable_target_yields_no_routes(self, empty_graph: ResortGraph, dem: MockDEMService) -> None:
        """B sits in a separate pocket with no directed path from A → no routes (not a crash)."""
        add_node(empty_graph, "A", 0.0, 0.0, 2000.0)
        add_node(empty_graph, "B", 0.0, -0.001, 1900.0)
        add_slope(empty_graph, "SL1", top="A", bottom="B")
        # A disconnected pocket C→D, unreachable from A.
        add_node(empty_graph, "C", 0.01, 0.0, 2000.0)
        add_node(empty_graph, "D", 0.01, -0.001, 1900.0)
        add_slope(empty_graph, "SL2", top="C", bottom="D")
        assert RoutePlanner(empty_graph).best_routes("A", "D") == []


class TestSingleSlopeRoute:
    def test_one_slope_route_maps_to_that_slope(self, empty_graph: ResortGraph, dem: MockDEMService) -> None:
        add_node(empty_graph, "Peak", 0.0, 0.0, 2000.0)
        add_node(empty_graph, "Base", 0.0, -1000 / M, 1400.0)
        add_slope(empty_graph, "SL1", top="Peak", bottom="Base")

        routes = RoutePlanner(empty_graph).best_routes("Peak", "Base")
        assert len(routes) == 1, "all criteria agree on the only route → one deduped result"
        route = routes[0]
        assert route.node_path == ("Peak", "Base")
        assert [s.entity_id for s in route.steps] == ["SL1"]
        assert route.lift_count == 0
        assert route.total_slope_drop_m == pytest.approx(600.0)
        assert set(route.criteria) == set(RouteCriterion)  # wins every criterion

    def test_interior_junction_maps_to_owning_slope(self, empty_graph: ResortGraph, dem: MockDEMService) -> None:
        """A route through a slope's mid-chain node still reads as one slope step (not per-segment)."""
        add_node(empty_graph, "Peak", 0.0, 0.0, 2000.0)
        add_node(empty_graph, "Mid", 0.0, -500 / M, 1700.0)
        add_node(empty_graph, "Base", 0.0, -1000 / M, 1400.0)
        add_slope(empty_graph, "SL1", top="Peak", bottom="Base", via=["Mid"])

        route = RoutePlanner(empty_graph).best_routes("Peak", "Base")[0]
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

        routes = RoutePlanner(empty_graph).best_routes("Top", "Base")
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
        peak→A). Every criterion must terminate quickly on it — a genuine shortest/best-path cost is
        cycle-safe by construction. Regression: an invalid non-shortest cost once looped forever here.
        """
        add_node(empty_graph, "A", 0.0, 0.0, 1800.0)
        add_node(empty_graph, "B", 0.0, -2000 / M, 1000.0)
        add_slope(empty_graph, "Low", top="A", bottom="B")
        add_node(empty_graph, "Peak", 0.0, 500 / M, 2400.0)
        # A gondola is bidirectional → A⇄Peak, and the slope Peak→A closes a cycle through A.
        empty_graph.add_lift(start_node_id="A", end_node_id="Peak", lift_type=LiftType.GONDOLA, dem=dem, name="Up")
        add_slope(empty_graph, "Down", top="Peak", bottom="A")

        routes = RoutePlanner(empty_graph).best_routes("A", "B")
        won = {c for r in routes for c in r.criteria}
        assert won == set(RouteCriterion), "all four criteria resolve on a cyclic graph"
        fewest = _route_for(routes, RouteCriterion.FEWEST_LIFTS)
        assert fewest.node_path == ("A", "B") and fewest.lift_count == 0, "fewest-lifts takes the direct slope"


class TestCriteriaDiverge:
    def test_least_distance_takes_lift_shortcut_while_fewest_lifts_stays_on_snow(
        self, empty_graph: ResortGraph, dem: MockDEMService
    ) -> None:
        """A long lift-free slope vs a shorter slope-lift-slope path. Least-distance takes the lift
        shortcut; fewest-lifts stays on the long lift-free run — two DIFFERENT routes, each optimal.
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
        shortest = _route_for(routes, RouteCriterion.LEAST_DISTANCE)
        assert fewest.node_path == ("A", "Far", "B"), "fewest-lifts stays on the lift-free run"
        assert fewest.lift_count == 0
        assert shortest.node_path == ("A", "Mid", "Ledge", "B"), "least-distance takes the lift shortcut"
        assert shortest.lift_count == 1
        assert shortest.total_slope_length_m < fewest.total_slope_length_m, "the shortcut skis less"
        assert fewest.node_path != shortest.node_path, "the two criteria genuinely diverge"

    def test_least_drop_measures_skied_drop_and_favours_a_down_lift(
        self, empty_graph: ResortGraph, dem: MockDEMService
    ) -> None:
        """least-drop counts VERTICAL SKIED, not net height: an all-ski run drops 1000 m, while
        skiing a short top then riding a down-gondola drops only the skied 100 m. least-drop (and
        least-distance) take the gondola; fewest-lifts stays all-ski.
        """
        add_node(empty_graph, "A", 0.0, 0.0, 2000.0)
        add_node(empty_graph, "B", 0.0, -3000 / M, 1000.0)
        add_slope(empty_graph, "AllSki", top="A", bottom="B")  # 1000 m skied drop
        add_node(empty_graph, "Mlow", 0.0, -200 / M, 1900.0)
        add_slope(empty_graph, "TopBit", top="A", bottom="Mlow")  # 100 m skied drop
        empty_graph.add_lift(
            start_node_id="B", end_node_id="Mlow", lift_type=LiftType.GONDOLA, dem=dem, name="DownGondi"
        )

        routes = RoutePlanner(empty_graph).best_routes("A", "B")
        least_drop = _route_for(routes, RouteCriterion.LEAST_DROP)
        fewest = _route_for(routes, RouteCriterion.FEWEST_LIFTS)
        assert least_drop.node_path == ("A", "Mlow", "B"), "least-drop rides the gondola down"
        assert least_drop.total_slope_drop_m == pytest.approx(100.0)
        assert fewest.node_path == ("A", "B") and fewest.total_slope_drop_m == pytest.approx(1000.0)
        assert least_drop.total_slope_drop_m < fewest.total_slope_drop_m


class TestEasiest:
    def test_easiest_picks_gentle_green_over_short_black(self, empty_graph: ResortGraph, dem: MockDEMService) -> None:
        """A short BLACK direct plunge vs a long GENTLE green detour. Easiest picks the green route;
        every distance/drop/lift criterion picks the short black one.
        """
        add_node(empty_graph, "A", 0.0, 0.0, 2000.0)
        add_node(empty_graph, "B", 0.0, -300 / M, 1400.0)  # 600m over 300m ⇒ black
        add_slope(empty_graph, "BlackRun", top="A", bottom="B")
        add_node(empty_graph, "W", -4000 / M, -150 / M, 1700.0)  # long, shallow legs ⇒ green
        add_slope(empty_graph, "BlueRun", top="A", bottom="B", via=["W"])
        assert empty_graph.slopes["BlackRun"].get_difficulty(segments=empty_graph.segments) == "black"
        assert empty_graph.slopes["BlueRun"].get_difficulty(segments=empty_graph.segments) == "green"

        routes = RoutePlanner(empty_graph).best_routes("A", "B")
        easiest = _route_for(routes, RouteCriterion.EASIEST)
        shortest = _route_for(routes, RouteCriterion.LEAST_DISTANCE)
        assert easiest.node_path == ("A", "W", "B") and easiest.max_difficulty == "green"
        assert shortest.node_path == ("A", "B") and shortest.max_difficulty == "black"
        assert easiest.node_path != shortest.node_path, "easiest avoids the black plunge"


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

        route = RoutePlanner(empty_graph).best_routes("Start", "Valley")[0]  # lift up, then ski down
        assert route.lift_count == 1
        assert route.total_slope_drop_m == pytest.approx(1000.0)  # 2000→1700→1000
        assert [(s.is_lift, s.name) for s in route.steps] == [(True, "Up"), (False, "TopRun"), (False, "MidRun")]


def _mk_route(*, max_difficulty: str, lift_types: list[str]) -> Route:
    """A minimal Route carrying only what filter_routes inspects (difficulty band + lift step types)."""
    steps = tuple(RouteStep(is_lift=True, entity_id=f"L{i}", name=f"L{i}", detail=t) for i, t in enumerate(lift_types))
    return Route(
        node_path=("A", "B"),
        steps=steps,
        total_slope_length_m=0.0,
        total_slope_drop_m=0.0,
        lift_count=len(lift_types),
        max_difficulty=max_difficulty,
        highest_elev_m=0.0,
        criteria=(RouteCriterion.FEWEST_LIFTS,),
    )


class TestFilterRoutes:
    def test_no_filters_keeps_all(self) -> None:
        routes = [_mk_route(max_difficulty="black", lift_types=["chairlift"])]
        assert filter_routes(routes, max_difficulty=None, allowed_lift_types={"chairlift"}) == routes

    def test_difficulty_cap_hides_harder_routes(self) -> None:
        blue = _mk_route(max_difficulty="blue", lift_types=[])
        black = _mk_route(max_difficulty="black", lift_types=[])
        kept = filter_routes([blue, black], max_difficulty="red", allowed_lift_types=set(LiftType))
        assert kept == [blue], "black exceeds a red cap; blue passes"

    def test_lift_type_filter_hides_routes_using_disallowed_lift(self) -> None:
        chair = _mk_route(max_difficulty="green", lift_types=["chairlift"])
        drag = _mk_route(max_difficulty="green", lift_types=["surface_lift"])
        kept = filter_routes([chair, drag], max_difficulty=None, allowed_lift_types={"chairlift"})
        assert kept == [chair], "surface_lift route hidden when only chairlift is allowed"

    def test_difficulty_cap_at_exact_band_is_inclusive(self) -> None:
        red = _mk_route(max_difficulty="red", lift_types=[])
        assert filter_routes([red], max_difficulty="red", allowed_lift_types=set()) == [red]

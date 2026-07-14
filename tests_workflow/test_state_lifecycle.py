"""Unit tests for state lifecycle enter/exit handlers (ui/state_lifecycle.py).

Each state-machine state has an `enter_*`/`exit_*` handler that mutates the PlannerContext to set
up or tear down that state's UI. These are pure ctx mutators (no Streamlit, no graph), so we seed a
context with DIRTY state, call the handler, and assert the documented effect: which sub-contexts get
cleared, whether the info panel shows/hides, and which deferred flags fire. The "single point of
truth" contract (enter guarantees panel visibility/hiding regardless of source) is what these guard.
"""

from skiresort_planner.model.path_point import PathPoint
from skiresort_planner.ui import state_lifecycle as sl
from skiresort_planner.ui.context import PlannerContext


def _dirty_ctx() -> PlannerContext:
    """A context polluted with building/placement/viewing state, to prove handlers clean it up."""
    ctx = PlannerContext()
    ctx.slope_build.segments = ["S1"]
    ctx.slope_build.name = "Slope 3"
    ctx.road_build.segments = ["R1"]
    ctx.lift.start_node_id = "N9"
    ctx.lift.start_location = PathPoint(lon=0.0, lat=0.0, elevation=2000.0)
    ctx.custom_connect.force_mode = True
    ctx.custom_connect.start_node = "N1"
    ctx.selection.node_id = "N2"
    ctx.merge.node_ids = ["N3", "N4"]
    ctx.viewing.panel_visible = True
    ctx.deferred.osm_import_center_lon = 10.0
    ctx.deferred.osm_import_center_lat = 47.0
    return ctx


class TestEnterIdleReady:
    def test_clears_all_building_state_and_hides_panel(self) -> None:
        ctx = _dirty_ctx()
        sl.enter_idle_ready(ctx)
        assert ctx.slope_build.segments == [], "building segments cleared"
        assert ctx.road_build.segments == [], "road segments cleared"
        assert ctx.lift.start_node_id is None and ctx.lift.start_location is None, "lift placement cleared"
        assert ctx.custom_connect.force_mode is False, "custom-connect cleared"
        assert ctx.selection.node_id is None, "selection cleared"
        assert ctx.viewing.panel_visible is False, "viewing panel hidden"

    def test_preserves_user_preferences(self) -> None:
        # idle_ready must NOT reset map view or the segment-length preference.
        ctx = _dirty_ctx()
        ctx.map.zoom = 15
        ctx.segment_length_m = 450
        sl.enter_idle_ready(ctx)
        assert ctx.map.zoom == 15, "map view is a user preference, preserved across idle_ready"
        assert ctx.segment_length_m == 450, "segment-length setting is preserved"


class TestEnterViewingStates:
    def test_enter_viewing_slope_shows_panel_and_clears_build_state(self) -> None:
        ctx = _dirty_ctx()
        ctx.viewing.panel_visible = False  # start hidden, enter must force it visible
        sl.enter_idle_viewing_slope(ctx)
        assert ctx.viewing.panel_visible is True, "single point of truth: enter guarantees panel visible"
        assert ctx.slope_build.segments == [], "stale building state cleared defensively"
        assert ctx.lift.start_node_id is None

    def test_enter_viewing_lift_shows_panel(self) -> None:
        ctx = _dirty_ctx()
        ctx.viewing.panel_visible = False
        sl.enter_idle_viewing_lift(ctx)
        assert ctx.viewing.panel_visible is True
        assert ctx.road_build.segments == []

    def test_enter_viewing_road_shows_panel(self) -> None:
        ctx = _dirty_ctx()
        ctx.viewing.panel_visible = False
        sl.enter_idle_viewing_road(ctx)
        assert ctx.viewing.panel_visible is True
        assert ctx.slope_build.segments == []


class TestEnterBuildingStates:
    def test_enter_slope_starting_hides_panel(self) -> None:
        ctx = _dirty_ctx()
        sl.enter_slope_starting(ctx)
        assert ctx.viewing.panel_visible is False, "building mode hides the info panel"

    def test_enter_slope_building_preserves_committed_segments(self) -> None:
        # Unlike idle/viewing, slope_building must NOT clear the build context — it holds the
        # committed segments the user is extending.
        ctx = _dirty_ctx()
        sl.enter_slope_building(ctx)
        assert ctx.viewing.panel_visible is False, "panel hidden while building"
        assert ctx.slope_build.segments == ["S1"], "committed segments are preserved on enter_slope_building"

    def test_enter_road_building_preserves_road_segments(self) -> None:
        ctx = _dirty_ctx()
        sl.enter_road_building(ctx)
        assert ctx.viewing.panel_visible is False
        assert ctx.road_build.segments == ["R1"], "committed road segments preserved on enter_road_building"

    def test_enter_road_starting_hides_panel(self) -> None:
        ctx = _dirty_ctx()
        sl.enter_road_starting(ctx)
        assert ctx.viewing.panel_visible is False

    def test_enter_lift_placing_hides_panel(self) -> None:
        ctx = _dirty_ctx()
        sl.enter_lift_placing(ctx)
        assert ctx.viewing.panel_visible is False


class TestEnterSlopeCustomPath:
    def test_flags_deferred_custom_connect_generation(self) -> None:
        # The only job of enter_slope_custom_path is to trigger deferred proposal generation.
        ctx = PlannerContext()
        assert ctx.deferred.custom_connect is False
        sl.enter_slope_custom_path(ctx)
        assert ctx.deferred.custom_connect is True, "entering custom-path flags deferred generation"


class TestEnterPlacingStatesPreserveTheirScratch:
    def test_enter_import_placing_keeps_the_placed_center(self) -> None:
        # The self-loop re-enters on every retarget; enter must NOT wipe the placed center (set by
        # before_start_import) or the box would vanish on the next click.
        ctx = _dirty_ctx()
        sl.enter_import_placing(ctx)
        assert ctx.viewing.panel_visible is False
        assert ctx.deferred.osm_import_center_lon == 10.0, "enter must not clear the placed box center (self-loop)"
        assert ctx.deferred.osm_import_center_lat == 47.0

    def test_enter_merge_placing_keeps_the_selection(self) -> None:
        # Self-loop on every node toggle; enter must NOT wipe the accumulating selection.
        ctx = _dirty_ctx()
        sl.enter_merge_placing(ctx)
        assert ctx.viewing.panel_visible is False
        assert ctx.merge.node_ids == ["N3", "N4"], "enter must not clear the merge selection (self-loop)"


class TestExitHandlersCleanUpScratch:
    def test_exit_lift_placing_clears_lift_context(self) -> None:
        ctx = _dirty_ctx()
        sl.exit_lift_placing(ctx)
        assert ctx.lift.start_node_id is None and ctx.lift.start_location is None, "lift scratch cleared on exit"

    def test_exit_import_placing_clears_placed_center(self) -> None:
        ctx = _dirty_ctx()
        sl.exit_import_placing(ctx)
        assert ctx.deferred.osm_import_center_lon is None, "placed center cleared on exit"
        assert ctx.deferred.osm_import_center_lat is None

    def test_exit_import_placing_leaves_fetch_flag_alone(self) -> None:
        # A confirmed import sets osm_import just before exit; exit must NOT clear it (the deferred
        # handler consumes it). Only the center coordinates are cleared here.
        ctx = _dirty_ctx()
        ctx.deferred.osm_import = True
        sl.exit_import_placing(ctx)
        assert ctx.deferred.osm_import is True, "exit_import_placing must not consume the pending fetch flag"

    def test_exit_merge_placing_clears_selection(self) -> None:
        ctx = _dirty_ctx()
        sl.exit_merge_placing(ctx)
        assert ctx.merge.node_ids == [], "merge selection cleared on exit"


class TestNoopExitsAreSafe:
    """The 'destination controls cleanup' exits are intentional no-ops; they must not raise and must
    not mutate state (a regression that added cleanup here would break undo/self-loop flows).
    """

    def test_noop_exits_do_not_mutate_or_raise(self) -> None:
        for exit_fn in (
            sl.exit_idle_ready,
            sl.exit_idle_viewing_slope,
            sl.exit_idle_viewing_lift,
            sl.exit_idle_viewing_road,
            sl.exit_slope_starting,
            sl.exit_slope_building,
            sl.exit_slope_custom_path,
            sl.exit_road_starting,
            sl.exit_road_building,
        ):
            ctx = _dirty_ctx()
            exit_fn(ctx)
            # The committed segments / selection must survive these no-op exits (self-loops rely on it).
            assert ctx.slope_build.segments == ["S1"], f"{exit_fn.__name__} must not clear building state"
            assert ctx.merge.node_ids == ["N3", "N4"], f"{exit_fn.__name__} must not touch the merge selection"

    def test_exit_slope_building_preserves_proposals_for_undo_continue(self) -> None:
        # Explicitly documented invariant: undo_continue sets proposals BEFORE the transition, so
        # exit_slope_building must never clear them.
        from skiresort_planner.model.proposed_path import ProposedPathSegment

        ctx = PlannerContext()
        proposal = ProposedPathSegment(
            points=[
                PathPoint(lon=0.0, lat=0.0, elevation=2500.0),
                PathPoint(lon=0.0, lat=-0.01, elevation=2400.0),
            ],
            target_difficulty="blue",
        )
        ctx.proposals.paths = [proposal]
        sl.exit_slope_building(ctx)
        assert ctx.proposals.paths == [proposal], "exit_slope_building preserves proposals (undo_continue relies on it)"

"""Unit tests for state lifecycle enter/exit handlers (ui/state_lifecycle.py).

Each state-machine state has an `enter_*`/`exit_*` handler that mutates the PlannerContext to set
up or tear down that state's UI. These are pure ctx mutators (no Streamlit, no graph), so we seed a
context with DIRTY state, call the handler, and assert the documented effect: which sub-contexts get
cleared, whether the info panel shows/hides, and which deferred flags fire. The "single point of
truth" contract (enter guarantees panel visibility/hiding regardless of source) is what these guard.
"""

from collections.abc import Callable

import pytest

from skiresort_planner.model.path_point import PathPoint
from skiresort_planner.model.path_segment import SegmentKind
from skiresort_planner.ui import state_lifecycle as sl
from skiresort_planner.ui.context import PlannerContext


def _dirty_ctx() -> PlannerContext:
    """A context polluted with building/placement/viewing state, to prove handlers clean it up."""
    ctx = PlannerContext()
    ctx.build(SegmentKind.SLOPE).segments = ["S1"]
    ctx.build(SegmentKind.SLOPE).name = "Slope 3"
    ctx.build(SegmentKind.ROAD).segments = ["R1"]
    ctx.lift.first_node_id = "N9"
    ctx.lift.first_location = PathPoint(lon=0.0, lat=0.0, elevation=2000.0)
    ctx.custom_connect.target_location = (0.0, 0.0, 2100.0)  # force_mode derives from this
    ctx.custom_connect.start_node = "N1"
    ctx.selection.set(lon=5.0, lat=6.0, elevation=2100.0)
    ctx.merge.node_ids = ["N3", "N4"]
    ctx.viewing.panel_visible = True
    ctx.pending.osm_import_center_lon = 10.0
    ctx.pending.osm_import_center_lat = 47.0
    return ctx


class TestEnterIdleReady:
    def test_clears_all_building_state_and_hides_panel(self) -> None:
        ctx = _dirty_ctx()
        sl.enter_idle_ready(ctx)
        assert ctx.build(SegmentKind.SLOPE).segments == [], "building segments cleared"
        assert ctx.build(SegmentKind.ROAD).segments == [], "road segments cleared"
        assert ctx.lift.first_node_id is None and ctx.lift.first_location is None, "lift placement cleared"
        assert ctx.custom_connect.force_mode is False, "custom-connect cleared"
        assert not ctx.selection.has_selection(), "selection coordinates cleared"
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
        assert ctx.build(SegmentKind.SLOPE).segments == [], "stale building state cleared defensively"
        assert ctx.lift.first_node_id is None

    def test_enter_viewing_lift_shows_panel(self) -> None:
        ctx = _dirty_ctx()
        ctx.viewing.panel_visible = False
        sl.enter_idle_viewing_lift(ctx)
        assert ctx.viewing.panel_visible is True
        assert ctx.build(SegmentKind.ROAD).segments == []

    def test_enter_viewing_road_shows_panel(self) -> None:
        ctx = _dirty_ctx()
        ctx.viewing.panel_visible = False
        sl.enter_idle_viewing_road(ctx)
        assert ctx.viewing.panel_visible is True
        assert ctx.build(SegmentKind.SLOPE).segments == []


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
        assert ctx.build(SegmentKind.SLOPE).segments == ["S1"], (
            "committed segments are preserved on enter_slope_building"
        )

    def test_enter_road_building_preserves_road_segments(self) -> None:
        ctx = _dirty_ctx()
        sl.enter_road_building(ctx)
        assert ctx.viewing.panel_visible is False
        assert ctx.build(SegmentKind.ROAD).segments == ["R1"], (
            "committed road segments preserved on enter_road_building"
        )

    def test_enter_road_starting_hides_panel(self) -> None:
        ctx = _dirty_ctx()
        sl.enter_road_starting(ctx)
        assert ctx.viewing.panel_visible is False

    @pytest.mark.parametrize(
        ("enter_fn", "kind"),
        [
            (sl.enter_slope_starting, SegmentKind.SLOPE),
            (sl.enter_slope_building, SegmentKind.SLOPE),
            (sl.enter_road_starting, SegmentKind.ROAD),
            (sl.enter_road_building, SegmentKind.ROAD),
        ],
    )
    def test_enter_fan_state_arms_its_kind(self, enter_fn: Callable[[PlannerContext], None], kind: SegmentKind) -> None:
        # The shared _enter_fan_state contract: every fan-state entry arms the fan for THAT kind, so
        # the deferred pass regenerates proposals (first click, undo-back, cancel-custom-to-fan).
        ctx = PlannerContext()
        enter_fn(ctx)
        assert kind in ctx.pending.fan_generation, f"{enter_fn.__name__} must arm the {kind.value} fan"

    def test_enter_lift_placing_hides_panel(self) -> None:
        ctx = _dirty_ctx()
        sl.enter_lift_placing(ctx)
        assert ctx.viewing.panel_visible is False


class TestEnterSlopeCustomPath:
    def test_flags_deferred_custom_connect_generation(self) -> None:
        # The only job of enter_slope_custom_path is to trigger deferred proposal generation.
        ctx = PlannerContext()
        assert ctx.pending.custom_connect is False
        sl.enter_slope_custom_path(ctx)
        assert ctx.pending.custom_connect is True, "entering custom-path flags deferred generation"


class TestEnterPlacingStatesPreserveTheirScratch:
    def test_enter_import_placing_keeps_the_placed_center(self) -> None:
        # The self-loop re-enters on every retarget; enter must NOT wipe the placed center (set by
        # before_start_import) or the box would vanish on the next click.
        ctx = _dirty_ctx()
        sl.enter_import_placing(ctx)
        assert ctx.viewing.panel_visible is False
        assert ctx.pending.osm_import_center_lon == 10.0, "enter must not clear the placed box center (self-loop)"
        assert ctx.pending.osm_import_center_lat == 47.0

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
        assert ctx.lift.first_node_id is None and ctx.lift.first_location is None, "lift scratch cleared on exit"

    def test_exit_import_placing_clears_placed_center(self) -> None:
        ctx = _dirty_ctx()
        sl.exit_import_placing(ctx)
        assert ctx.pending.osm_import_center_lon is None, "placed center cleared on exit"
        assert ctx.pending.osm_import_center_lat is None

    def test_exit_import_placing_leaves_fetch_flag_alone(self) -> None:
        # A confirmed import sets osm_import_mode just before exit; exit must NOT clear it (the deferred
        # handler consumes it). Only the center coordinates are cleared here.
        from skiresort_planner.constants import OSMImportMode

        ctx = _dirty_ctx()
        ctx.pending.osm_import_mode = OSMImportMode.LIFTS_AND_SLOPES
        sl.exit_import_placing(ctx)
        assert ctx.pending.osm_import_mode is OSMImportMode.LIFTS_AND_SLOPES, (
            "exit_import_placing must not consume the pending fetch mode"
        )

    def test_exit_merge_placing_clears_selection(self) -> None:
        ctx = _dirty_ctx()
        sl.exit_merge_placing(ctx)
        assert ctx.merge.node_ids == [], "merge selection cleared on exit"


class TestNoOpExitsHaveNoHook:
    """States without real exit teardown must NOT have an on_exit_* hook (else a self-loop or undo
    would run cleanup that breaks undo_continue / retarget). Only lift/import/merge have exit hooks;
    everything else exits as a no-op by having no hook at all.
    """

    def test_only_real_cleanup_states_have_exit_hooks(self) -> None:
        from skiresort_planner.ui.state_machine import PlannerStateMachine

        on_exit_methods = {n.removeprefix("on_exit_") for n in dir(PlannerStateMachine) if n.startswith("on_exit_")}
        assert on_exit_methods == {"lift_placing"}, (
            "only lift_placing needs a library on_exit hook; import/merge clear via before-hooks, "
            f"the rest exit as no-ops. Got: {sorted(on_exit_methods)}"
        )


class TestEnterBuildAndCustomStatesClearMarkerDedup:
    """Re-clicking the SAME node must stay recognised while building/targeting.

    Node markers dedup by "node_<id>" with no map_version, so only clear_marker() frees a repeat
    click. Every enter hook that can be RE-reached with the same node still "seen" (a build/custom
    self-loop, or a commit/cancel that lands back in a build state) must clear the marker, or the
    second click on that node is silently swallowed. enter_*_starting already does this; the build
    and custom-path hooks must match.
    """

    @pytest.mark.parametrize(
        "enter_fn",
        [
            sl.enter_slope_building,
            sl.enter_road_building,
            sl.enter_slope_custom_path,
            sl.enter_road_custom_path,
        ],
    )
    def test_enter_clears_seen_node_marker(self, enter_fn: Callable[[PlannerContext], None]) -> None:
        ctx = _dirty_ctx()
        ctx.click_dedup.debounce_seconds = 0.0  # disable timing debounce so repeat clicks aren't time-suppressed
        # Simulate the node just clicked (targeting) still being the last-seen marker.
        assert ctx.click_dedup.is_new_click(coord=None, obj_id="node_N7") is True

        enter_fn(ctx)

        assert ctx.click_dedup.is_new_click(coord=None, obj_id="node_N7") is True, (
            f"{enter_fn.__name__} must clear the marker so the same node can be re-clicked (retarget / repeat)"
        )

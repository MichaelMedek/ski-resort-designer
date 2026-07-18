"""Unit tests for the UI context dataclasses (context.py).

Pure state logic — no Streamlit, no graph. Covers BuildMode helpers, the
ViewingContext mutual-exclusion setters, MapContext view setters, and the
ClickDeduplicationContext debounce/new-click logic.
"""

import pytest

from skiresort_planner.ui.context import (
    BuildMode,
    ClickDeduplicationContext,
    CustomConnectContext,
    LiftContext,
    MapContext,
    ViewingContext,
)


class TestBuildModeHelpers:
    @pytest.mark.parametrize(
        "mode, is_lift, is_road",
        [
            (BuildMode.SLOPE, False, False),
            (BuildMode.ROAD, False, True),
            (BuildMode.CHAIRLIFT, True, False),
            (BuildMode.GONDOLA, True, False),
            (BuildMode.SURFACE_LIFT, True, False),
            (BuildMode.AERIAL_TRAM, True, False),
        ],
    )
    def test_mode_predicates(self, mode: str, *, is_lift: bool, is_road: bool) -> None:
        assert BuildMode.is_lift(mode) is is_lift
        assert BuildMode.is_road(mode) is is_road

    @pytest.mark.parametrize(
        "mode",
        [
            BuildMode.SLOPE,
            BuildMode.ROAD,
            BuildMode.CHAIRLIFT,
            BuildMode.GONDOLA,
            BuildMode.SURFACE_LIFT,
            BuildMode.AERIAL_TRAM,
        ],
    )
    def test_display_name_and_icon_exist_for_every_mode(self, mode: str) -> None:
        assert BuildMode.display_name(mode)  # non-empty
        assert BuildMode.icon(mode)  # non-empty

    def test_display_name_rejects_unknown_mode(self) -> None:
        with pytest.raises(ValueError, match="Unknown build mode"):
            BuildMode.display_name("teleporter")

    def test_icon_rejects_unknown_mode(self) -> None:
        with pytest.raises(ValueError, match="Unknown build mode"):
            BuildMode.icon("teleporter")


class TestViewingContextSetters:
    """Each set_*_id keeps exactly one entity id and clears the other two."""

    def test_set_slope_id_clears_others(self) -> None:
        vc = ViewingContext(lift_id="L1", road_id="R1", panel_visible=True)
        vc.set_slope_id("SL2")
        assert (vc.slope_id, vc.lift_id, vc.road_id) == ("SL2", None, None)
        assert vc.is_viewing_slope() and not vc.is_viewing_lift() and not vc.is_viewing_road()

    def test_set_lift_id_clears_others(self) -> None:
        vc = ViewingContext(slope_id="SL1", road_id="R1", panel_visible=True)
        vc.set_lift_id("L2")
        assert (vc.slope_id, vc.lift_id, vc.road_id) == (None, "L2", None)
        assert vc.is_viewing_lift()

    def test_set_road_id_clears_others(self) -> None:
        vc = ViewingContext(slope_id="SL1", lift_id="L1", panel_visible=True)
        vc.set_road_id("R2")
        assert (vc.slope_id, vc.lift_id, vc.road_id) == (None, None, "R2")
        assert vc.is_viewing_road()

    def test_hide_panel_disables_3d(self) -> None:
        vc = ViewingContext(panel_visible=True, view_3d=True)
        vc.hide_panel()
        assert not vc.panel_visible and not vc.view_3d

    def test_clear_resets_all(self) -> None:
        vc = ViewingContext(slope_id="SL1", panel_visible=True, view_3d=True)
        vc.clear()
        assert vc.slope_id is None and not vc.panel_visible and not vc.view_3d


class TestMapContextViews:
    def test_reset_and_clear_restore_defaults(self) -> None:
        from skiresort_planner.constants import MapConfig

        mc = MapContext()
        mc.set_center(lon=1.0, lat=2.0)
        mc.pitch = 60.0  # non-default so reset_view is not vacuous
        mc.reset_view()
        assert mc.pitch == MapConfig.DEFAULT_PITCH and mc.bearing == MapConfig.DEFAULT_BEARING

        mc.clear()
        assert (mc.lon, mc.lat) == (MapConfig.START_CENTER_LON, MapConfig.START_CENTER_LAT)


class TestClickDeduplicationContext:
    def _ctx(self) -> ClickDeduplicationContext:
        # debounce 0 → timing never blocks, isolating the id/coord logic.
        return ClickDeduplicationContext(debounce_seconds=0.0)

    def test_no_data_is_not_a_new_click(self) -> None:
        assert self._ctx().is_new_click(coord=None, obj_id=None) is False

    def test_first_object_click_is_new_then_repeat_is_not(self) -> None:
        ctx = self._ctx()
        assert ctx.is_new_click(coord=(10.0, 46.0), obj_id="marker_slope_SL1") is True
        assert ctx.is_new_click(coord=(10.0, 46.0), obj_id="marker_slope_SL1") is False

    def test_different_object_id_is_new(self) -> None:
        ctx = self._ctx()
        ctx.is_new_click(coord=None, obj_id="A")
        assert ctx.is_new_click(coord=None, obj_id="B") is True

    def test_terrain_click_new_when_coord_changes(self) -> None:
        ctx = self._ctx()
        assert ctx.is_new_click(coord=(1.0, 2.0), obj_id=None) is True
        assert ctx.is_new_click(coord=(1.0, 2.0), obj_id=None) is False
        assert ctx.is_new_click(coord=(3.0, 4.0), obj_id=None) is True

    def test_debounce_blocks_rapid_repeat(self) -> None:
        # Large debounce + a fresh timestamp → the immediate next click is blocked.
        import time

        ctx = ClickDeduplicationContext(debounce_seconds=1000.0)
        ctx.last_click_timestamp = time.time()
        assert ctx.is_new_click(coord=(1.0, 2.0), obj_id="X") is False


class TestLiftContextClear:
    """LiftContext.clear() resets the placement scratch (first node/location) to a fresh state.

    The selected lift TYPE is deliberately NOT stored here — it lives in BuildModeContext.mode (the
    single source of truth), so clear() must restore only the first-endpoint fields.
    """

    def test_clear_resets_placement_fields(self) -> None:
        ctx = LiftContext()
        ctx.first_node_id = "N1"
        ctx.clear()
        assert ctx == LiftContext(), "clear() must restore a fresh context (first_node_id/first_location cleared)"


class TestCustomConnectForceModeIsDerived:
    """force_mode is a DERIVED property, not a stored flag: it must equal (target_location is not
    None) at all times. This pins the single-source-of-truth invariant so the two can never drift
    (the drift that caused the custom-path escape-button trap bug).
    """

    def test_force_mode_equals_target_location_presence(self) -> None:
        no_target = CustomConnectContext()
        assert no_target.force_mode is False, "no target → not in force mode"

        with_target = CustomConnectContext(target_location=(10.0, 47.0, 2000.0))
        assert with_target.force_mode is True, "a set target_location IS force mode (derived, not a flag)"

        with_target.clear()
        assert with_target.force_mode is False, "clear() drops the target → force_mode false again"

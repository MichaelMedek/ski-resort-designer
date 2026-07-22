"""Unit tests for user-facing toast messages with COMPUTED content (model/message.py).

Focuses on messages whose text is derived (formatted numbers, branches) rather than static,
so a formatting regression is caught. Static-text toasts need no test.
"""

import re

from skiresort_planner.model.message import (
    OSMImportErrorMessage,
    TargetNotDownhillMessage,
    TargetTooFarMessage,
    too_steep_detail,
)
from skiresort_planner.model.path_segment import SegmentKind


def _first_number(text: str) -> float:
    """First signed decimal number in the text."""
    m = re.search(r"-?\d+\.?\d*", text)
    assert m is not None, f"no number in {text!r}"
    return float(m.group())


def _concrete_subclasses(base: type) -> list[type]:
    """All concrete (instantiable) subclasses of `base` defined in model.message, recursively.

    Excludes the abstract intermediates (Message/ToastMessage/Info*/Warning*) so the list is exactly
    the real user-facing message classes — the set a completeness guard must cover.
    """
    from skiresort_planner.model import message as m

    abstract = {m.Message, m.ToastMessage, m.InfoMessage, m.WarningMessage, m.WarningToast}
    out: list[type] = []

    def walk(cls: type) -> None:
        for sub in cls.__subclasses__():
            if sub not in abstract:
                out.append(sub)
            walk(sub)

    walk(base)
    return sorted(set(out), key=lambda c: c.__name__)


def _make(cls: type):
    """Instantiate a message dataclass with dummy values for its required fields (defaults elsewhere).

    Covers the field types the messages actually use (str/float/int/bool/SegmentKind/OSMImportMode) so
    the hierarchy tests can construct EVERY class without hard-coding constructor args per class.
    """
    import dataclasses
    import typing

    from skiresort_planner.constants import OSMImportMode
    from skiresort_planner.model.path_segment import SegmentKind

    dummies: dict[type, object] = {
        str: "x",
        float: 1.0,
        int: 1,
        bool: True,
        SegmentKind: SegmentKind.SLOPE,
        OSMImportMode: OSMImportMode.LIFTS_ONLY,
    }
    hints = typing.get_type_hints(cls)  # resolves string annotations to real types
    kwargs = {}
    for f in dataclasses.fields(cls):
        if f.default is not dataclasses.MISSING or f.default_factory is not dataclasses.MISSING:
            continue  # has a default — leave it
        kwargs[f.name] = dummies[hints[f.name]]
    return cls(**kwargs)


class TestTooSteepDetail:
    """The shared "too steep" why-line (model.message.too_steep_detail), rendered in the right
    panel's No-Paths block. Roads use a ±band (two_sided=True) and "for a car road"; slopes use a
    single-sided ceiling (two_sided=False) and "to ski".
    """

    def test_road_just_over_cap_reads_strictly_above_limit(self) -> None:
        # A value just over the cap must NOT render "15%, over the ±15% limit" (a
        # self-contradiction). It shows a decimal strictly above 15.
        msg = too_steep_detail(gentlest_pct=15.02, max_grade_pct=15.0, subject="for a car road", two_sided=True)
        assert "±15% limit" in msg
        shown = _first_number(msg.split("gentlest possible is")[1])
        assert shown > 15.0, f"gentlest must read strictly above the ±15% cap: {msg}"

    def test_road_clearly_over_cap_message(self) -> None:
        msg = too_steep_detail(gentlest_pct=22.0, max_grade_pct=15.0, subject="for a car road", two_sided=True)
        assert "22.0%" in msg and "±15% limit" in msg and "car road" in msg

    def test_road_no_route_branch(self) -> None:
        msg = too_steep_detail(gentlest_pct=None, max_grade_pct=15.0, subject="for a car road", two_sided=True)
        assert "no route" in msg and "±15%" in msg

    def test_slope_uses_single_sided_ceiling_wording(self) -> None:
        # Slopes are one-sided: no ± prefix, wording "to ski".
        msg = too_steep_detail(gentlest_pct=80.0, max_grade_pct=70.0, subject="to ski", two_sided=False)
        assert "to ski" in msg and "70% limit" in msg and "±" not in msg
        assert "80.0%" in msg

    def test_slope_no_route_branch(self) -> None:
        msg = too_steep_detail(gentlest_pct=None, max_grade_pct=70.0, subject="to ski", two_sided=False)
        assert "no route" in msg and "under 70%" in msg and "±" not in msg


class TestDisconnectedEntityMessage:
    def test_names_entity_kind_and_core_lift(self) -> None:
        from skiresort_planner.model.message import DisconnectedEntityMessage

        msg = DisconnectedEntityMessage(entity_noun="slope", core_lift_name="5 (Summit Express)").message
        assert "This slope can't be reached from the core area" in msg
        assert "5 (Summit Express)" in msg


class TestNoReturnEntityMessage:
    def test_names_entity_kind_and_one_way(self) -> None:
        from skiresort_planner.model.message import NoReturnEntityMessage

        msg = NoReturnEntityMessage(entity_noun="lift").message
        assert "This lift is a one-way trip" in msg
        assert "back to ride it again" in msg


class TestTargetTooFarMessage:
    def test_just_over_max_reads_strictly_above(self) -> None:
        # Fires only when distance strictly exceeds the max; must not render "1000m (max: 1000m)".
        msg = TargetTooFarMessage(distance_m=1000.4, max_distance_m=1000.0).message
        assert "(max: 1000m)" in msg
        # Meters render as whole integers (no decimal), and strictly above the max: "1001m".
        assert msg == "Target Too Far — 1001m (max: 1000m)", msg
        shown = _first_number(msg.split("—")[1])
        assert shown > 1000.0, f"distance must read strictly above the max: {msg}"


class TestTargetNotDownhillMessage:
    def test_just_under_min_reads_strictly_below(self) -> None:
        # Fires only when drop is strictly under the minimum; must not render "drop: 5m, need at least 5m".
        msg = TargetNotDownhillMessage(start_elevation_m=2000.0, target_elevation_m=1995.4, min_drop_m=5.0).message
        assert "need at least 5m" in msg
        # Meters render as whole integers (no decimal), and strictly below the minimum: "4m".
        assert "drop: 4m," in msg, msg
        shown = _first_number(msg.split("drop:")[1])
        assert shown < 5.0, f"drop must read strictly below the minimum: {msg}"

    def test_uphill_target_adds_explainer(self) -> None:
        msg = TargetNotDownhillMessage(start_elevation_m=2000.0, target_elevation_m=2010.0, min_drop_m=5.0).message
        assert "above your current point" in msg


class TestLiftPlacingMessages:
    def test_context_message_labels_first_station(self) -> None:
        from skiresort_planner.model.message import LiftPlacingContextMessage

        msg = LiftPlacingContextMessage(first_node_id="N7", first_elevation_m=2500.0).message
        assert "First station" in msg and "Node **N7**" in msg

    def test_action_message_mentions_auto_orientation(self) -> None:
        from skiresort_planner.model.message import LiftActionMessage

        msg = LiftActionMessage().message
        assert "Second Station" in msg
        assert "low → high" in msg
        # The old bottom-elevation constraint wording is gone.
        assert "above" not in msg


class TestPathActionMessageTooSteep:
    """The empty-paths branch of PathActionMessage folds in the too-steep detail (this replaced the
    transient toast). The detail appears only when a cap was recorded; otherwise the generic block.
    """

    def test_no_paths_block_includes_too_steep_detail(self) -> None:
        from skiresort_planner.model.message import PathActionMessage

        msg = PathActionMessage(
            kind=SegmentKind.ROAD,
            is_custom_path=True,
            too_steep_cap_pct=15.0,
            too_steep_gentlest_pct=22.0,
            too_steep_subject="for a car road",
            too_steep_two_sided=True,
        ).message
        assert "No Paths Available" in msg
        assert "22.0%" in msg and "±15% limit" in msg, "the exact steepness detail is consolidated here"

    def test_no_paths_block_without_detail_is_generic(self) -> None:
        from skiresort_planner.model.message import PathActionMessage

        msg = PathActionMessage(kind=SegmentKind.SLOPE, is_custom_path=True).message
        assert "No Paths Available" in msg
        assert "Too steep" not in msg, "no cap recorded → no steepness line, just the generic guidance"

    def test_selecting_shows_magnitudes_for_a_climbing_road(self) -> None:
        # A climbing road has a negative drop/gradient in the backend's "downhill is positive" sign;
        # the selection message shows MAGNITUDES so it never reads a confusing "-5m / -3%" on a climb.
        from skiresort_planner.model.message import PathActionMessage

        msg = PathActionMessage(
            kind=SegmentKind.ROAD,
            is_selecting_path=True,
            num_paths=2,
            selected_path_idx=1,
            actual_gradient_pct=-3.0,
            target_gradient_pct=12.0,
            path_length_m=150.0,
            path_drop_m=-5.0,
            start_elevation_m=3438.0,
            end_elevation_m=3443.0,
        ).message
        assert "↕5m" in msg and "-5m" not in msg, "drop shown as magnitude"
        assert "3%" in msg and "-3%" not in msg, "gradient shown as magnitude"
        assert "3438m → 3443m" in msg, "direction still conveyed by the elevation line"


class TestOSMImportMessages:
    def test_error_message(self) -> None:
        msg = OSMImportErrorMessage(error="the current view is outside the terrain coverage").message
        assert "OSM import failed" in msg
        assert "outside the terrain coverage" in msg

    def test_loading_message_branches_by_mode(self) -> None:
        from skiresort_planner.constants import OSMImportMode
        from skiresort_planner.model.message import OSMImportLoadingMessage

        lifts = OSMImportLoadingMessage(mode=OSMImportMode.LIFTS_ONLY).message
        both = OSMImportLoadingMessage(mode=OSMImportMode.LIFTS_AND_SLOPES).message
        assert "lifts" in lifts.lower() and "slopes" not in lifts.lower(), "lifts-only text"
        assert "slopes" in both.lower(), "lifts+slopes text mentions slopes"


class TestMessageHierarchy:
    """Every concrete message inherits its level/icon from a base class by construction. These tests
    enumerate ALL message classes so a newly-added one that skips the InfoMessage/WarningMessage
    (inline) or WarningToast (transient) bases fails here (completeness guard), and assert the
    two-levels-only rule.
    """

    def test_only_info_and_warning_levels_exist(self) -> None:
        from skiresort_planner.model.message import MessageLevel

        assert set(MessageLevel) == {MessageLevel.INFO, MessageLevel.WARNING}, "no ERROR level (fail-fast)"

    def test_every_inline_message_is_info_or_warning(self) -> None:
        # Each concrete inline Message must subclass InfoMessage (blue) or WarningMessage (yellow) —
        # never Message directly — so its level is fixed by the base, not re-declared per class.
        from skiresort_planner.model import message as m

        for cls in _concrete_subclasses(m.Message):
            assert issubclass(cls, m.InfoMessage | m.WarningMessage), (
                f"{cls.__name__} must subclass InfoMessage or WarningMessage, not Message directly"
            )

    def test_every_toast_is_a_warning(self) -> None:
        # Toasts are always transient warnings — every concrete toast subclasses WarningToast so its
        # icon is fixed by the base (never ToastMessage directly).
        from skiresort_planner.model import message as m

        for cls in _concrete_subclasses(m.ToastMessage):
            assert issubclass(cls, m.WarningToast), (
                f"{cls.__name__} must subclass WarningToast, not ToastMessage directly"
            )

    def test_info_and_warning_inline_levels(self) -> None:
        from skiresort_planner.model import message as m
        from skiresort_planner.model.message import InfoMessage, MessageLevel

        for cls in _concrete_subclasses(m.Message):
            expected = MessageLevel.INFO if issubclass(cls, InfoMessage) else MessageLevel.WARNING
            assert _make(cls).level == expected, f"{cls.__name__} level"

    def test_warning_toast_icon(self) -> None:
        # Every concrete toast is a WarningToast and shares its ⚠️ icon.
        from skiresort_planner.model import message as m
        from skiresort_planner.model.message import WarningToast

        warn_icon = _make(next(c for c in _concrete_subclasses(WarningToast))).icon
        for cls in _concrete_subclasses(m.ToastMessage):
            assert _make(cls).icon == warn_icon, f"{cls.__name__} icon inherited from WarningToast"

    def test_clicking_disabled_in_3d_toast_text(self) -> None:
        from skiresort_planner.model.message import ClickingDisabledIn3DToast, WarningToast

        toast = ClickingDisabledIn3DToast()
        assert isinstance(toast, WarningToast)
        assert "3D" in toast.message


class TestImportSelectingMessages:
    def test_context_shows_center_and_area(self) -> None:
        from skiresort_planner.model.message import ImportSelectingContextMessage

        msg = ImportSelectingContextMessage(center_lat=47.05, center_lon=10.32, half_width_km=2.0).message
        assert "47.05" in msg and "10.32" in msg, "center coordinates shown"
        assert "4.0 × 4.0 km" in msg, "half-width 2.0 → 4×4 km area"

    def test_action_message_names_the_import_buttons(self) -> None:
        # The center dot is inert; only the panel buttons confirm, so the guidance must name them.
        from skiresort_planner.model.message import ImportActionMessage

        msg = ImportActionMessage().message
        assert "Import lifts + slopes" in msg and "Import lifts only" in msg
        assert "center dot" not in msg, "center dot no longer confirms — must not be advertised"


class TestNodeEditMessages:
    def test_context_zero_selected_prompts_to_click(self) -> None:
        from skiresort_planner.model.message import NodeEditContextMessage

        msg = NodeEditContextMessage(selected_count=0, span_m=0.0).message
        assert "Edit Nodes" in msg
        assert "Click node markers" in msg

    def test_context_shows_count_and_span(self) -> None:
        from skiresort_planner.model.message import NodeEditContextMessage

        msg = NodeEditContextMessage(selected_count=3, span_m=89.0).message
        assert "3 node" in msg
        assert "89m" in msg

    def test_action_lists_all_three_actions_every_count(self) -> None:
        from skiresort_planner.model.message import NodeEditActionMessage

        # The three actions are written once and shown at EVERY count (marked ✅/⬜ by availability).
        for n in (0, 1, 2, 3):
            msg = NodeEditActionMessage(selected_count=n).message
            assert "Merge" in msg and "Delete Direct Connection" in msg and "Delete" in msg, f"n={n}"

    def test_action_availability_markers_match_button_rules(self) -> None:
        from skiresort_planner.model.message import NodeEditActionMessage

        def marker(msg: str, action: str) -> str:
            line = next(ln for ln in msg.splitlines() if action in ln)
            return "✅" if "✅" in line else "⬜"

        # Distinguishing substrings: Merge, "Direct Connection" (the cut), "**Delete** —" (plain delete).
        one = NodeEditActionMessage(selected_count=1).message
        assert marker(one, "**Delete** —") == "✅", "1 selected enables plain Delete"
        assert marker(one, "**Merge**") == "⬜" and marker(one, "Direct Connection") == "⬜"

        two = NodeEditActionMessage(selected_count=2).message
        assert marker(two, "**Merge**") == "✅" and marker(two, "Direct Connection") == "✅"
        assert marker(two, "**Delete** —") == "✅"

        three = NodeEditActionMessage(selected_count=3).message
        assert marker(three, "**Merge**") == "✅", "3 still merges"
        assert marker(three, "Direct Connection") == "⬜", "cut needs EXACTLY 2"

    def test_unable_to_delete_names_the_reason(self) -> None:
        from skiresort_planner.model.message import UnableToDeleteMessage, WarningToast

        msg = UnableToDeleteMessage(reason="N5 is a lift station — delete the lift first")
        # Assert the exact wrapped format so a dropped prefix / wrong separator / omitted reason fails.
        assert msg.message == "Cannot delete — N5 is a lift station — delete the lift first"
        assert isinstance(msg, WarningToast), "a rejected delete is a warning toast (inherits its icon)"

    def test_too_far_reads_strictly_above_max(self) -> None:
        from skiresort_planner.model.message import MergeTooFarMessage

        # Fires only when the span strictly exceeds the max; the shown span must read above it.
        msg = MergeTooFarMessage(span_m=612.34, max_span_m=500.0).message
        assert "Too Far" in msg and "max: 500m" in msg
        # Meters render as whole integers (no decimal), rounded up: "613m".
        assert msg == "Nodes Too Far Apart — 613m (max: 500m)", msg
        shown = _first_number(msg)
        assert shown > 500.0, f"span must read strictly above the max: {msg}"


class TestPathActionNoPathsFallback:
    """The empty-proposals fallback must guide the RIGHT escape per mode: Undo for a fan-out dead
    end, but Cancel-Custom-Path when routing a too-steep custom target (Undo would be wrong).
    """

    def _msg(self, *, is_custom_path: bool) -> str:
        from skiresort_planner.model.message import PathActionMessage
        from skiresort_planner.model.path_segment import SegmentKind

        return PathActionMessage(kind=SegmentKind.ROAD, is_custom_path=is_custom_path).message

    def test_fanout_deadend_points_to_undo(self) -> None:
        msg = self._msg(is_custom_path=False)
        assert "No Paths Available" in msg and "Undo" in msg

    def test_custom_target_deadend_points_to_cancel_not_undo(self) -> None:
        msg = self._msg(is_custom_path=True)
        assert "No Paths Available" in msg
        assert "Cancel Custom Path" in msg, "a too-steep custom target must point at the escape, not Undo"
        assert "Undo" not in msg

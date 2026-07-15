"""Unit tests for user-facing toast messages with COMPUTED content (model/message.py).

Focuses on messages whose text is derived (formatted numbers, branches) rather than static,
so a formatting regression is caught. Static-text toasts need no test.
"""

import re

from skiresort_planner.model.message import (
    LiftMustGoUphillMessage,
    OSMImportErrorMessage,
    PathTooSteepMessage,
    TargetNotDownhillMessage,
    TargetTooFarMessage,
)


def _first_number(text: str) -> float:
    """First signed decimal number in the text."""
    m = re.search(r"-?\d+\.?\d*", text)
    assert m is not None, f"no number in {text!r}"
    return float(m.group())


class TestPathTooSteepMessage:
    """One unified refusal toast for both kinds. Roads use a ±band (two_sided=True) and "for a car
    road"; slopes use a single-sided ceiling (two_sided=False) and "to ski".
    """

    def test_road_just_over_cap_reads_strictly_above_limit(self) -> None:
        # Fires only when every route is strictly over the band. A value just over the cap
        # must NOT render "15%, over the ±15% limit" (a self-contradiction). It shows a
        # decimal strictly above 15.
        msg = PathTooSteepMessage(
            gentlest_pct=15.02, max_grade_pct=15.0, subject="for a car road", two_sided=True
        ).message
        assert "±15% limit" in msg
        shown = _first_number(msg.split("gentlest possible is")[1])
        assert shown > 15.0, f"gentlest must read strictly above the ±15% cap: {msg}"

    def test_road_clearly_over_cap_message(self) -> None:
        msg = PathTooSteepMessage(
            gentlest_pct=22.0, max_grade_pct=15.0, subject="for a car road", two_sided=True
        ).message
        assert "22.0%" in msg and "±15% limit" in msg and "car road" in msg

    def test_road_no_route_branch(self) -> None:
        msg = PathTooSteepMessage(
            gentlest_pct=None, max_grade_pct=15.0, subject="for a car road", two_sided=True
        ).message
        assert "no route" in msg and "±15%" in msg

    def test_slope_uses_single_sided_ceiling_wording(self) -> None:
        # Slopes are one-sided: no ± prefix, wording "to ski".
        msg = PathTooSteepMessage(gentlest_pct=80.0, max_grade_pct=70.0, subject="to ski", two_sided=False).message
        assert "to ski" in msg and "70% limit" in msg and "±" not in msg
        assert "80.0%" in msg

    def test_slope_no_route_branch(self) -> None:
        msg = PathTooSteepMessage(gentlest_pct=None, max_grade_pct=70.0, subject="to ski", two_sided=False).message
        assert "no route" in msg and "under 70%" in msg and "±" not in msg


class TestTargetTooFarMessage:
    def test_just_over_max_reads_strictly_above(self) -> None:
        # Fires only when distance strictly exceeds the max; must not render "1000m (max: 1000m)".
        msg = TargetTooFarMessage(distance_m=1000.4, max_distance_m=1000.0).message
        assert "(max: 1000m)" in msg
        shown = _first_number(msg.split("—")[1])
        assert shown > 1000.0, f"distance must read strictly above the max: {msg}"


class TestTargetNotDownhillMessage:
    def test_just_under_min_reads_strictly_below(self) -> None:
        # Fires only when drop is strictly under the minimum; must not render "drop: 5m, need at least 5m".
        msg = TargetNotDownhillMessage(start_elevation_m=2000.0, target_elevation_m=1995.4, min_drop_m=5.0).message
        assert "need at least 5m" in msg
        shown = _first_number(msg.split("drop:")[1])
        assert shown < 5.0, f"drop must read strictly below the minimum: {msg}"

    def test_uphill_target_adds_explainer(self) -> None:
        msg = TargetNotDownhillMessage(start_elevation_m=2000.0, target_elevation_m=2010.0, min_drop_m=5.0).message
        assert "above your current point" in msg


class TestLiftMustGoUphillMessage:
    def test_tiny_downhill_never_renders_negative_zero(self) -> None:
        # Fires when end <= start (diff <= 0). A -0.4m diff must not render a misleading "-0m".
        msg = LiftMustGoUphillMessage(start_elevation_m=2500.4, end_elevation_m=2500.0).message
        assert "-0m" not in msg and "+0m" not in msg, f"must not render a misleading zero diff: {msg}"
        diff_shown = _first_number(msg.split("(")[1])
        assert diff_shown < 0, f"a downhill lift must show a negative diff: {msg}"


class TestOSMImportMessages:
    def test_error_message(self) -> None:
        msg = OSMImportErrorMessage(error="the current view is outside the terrain coverage").message
        assert "OSM import failed" in msg
        assert "outside the terrain coverage" in msg


class TestImportPlacingMessages:
    def test_context_shows_center_and_area(self) -> None:
        from skiresort_planner.model.message import ImportPlacingContextMessage

        msg = ImportPlacingContextMessage(center_lat=47.05, center_lon=10.32, half_width_km=2.0).message
        assert "47.05" in msg and "10.32" in msg, "center coordinates shown"
        assert "4.0 × 4.0 km" in msg, "half-width 2.0 → 4×4 km area"

    def test_action_message_mentions_confirm(self) -> None:
        from skiresort_planner.model.message import ImportActionMessage

        msg = ImportActionMessage().message
        assert "Confirm" in msg or "center dot" in msg


class TestMergePlacingMessages:
    def test_context_zero_selected_prompts_to_click(self) -> None:
        from skiresort_planner.model.message import MergePlacingContextMessage

        msg = MergePlacingContextMessage(selected_count=0, span_m=0.0).message
        assert "Merge Nodes" in msg
        assert "Click node markers" in msg

    def test_context_shows_count_and_span(self) -> None:
        from skiresort_planner.model.message import MergePlacingContextMessage

        msg = MergePlacingContextMessage(selected_count=3, span_m=89.0).message
        assert "3 node" in msg
        assert "89m" in msg

    def test_action_under_two_asks_for_more(self) -> None:
        from skiresort_planner.model.message import MergeActionMessage

        msg = MergeActionMessage(selected_count=1).message
        assert "at least" in msg and "2" in msg

    def test_action_two_or_more_offers_confirm(self) -> None:
        from skiresort_planner.model.message import MergeActionMessage

        msg = MergeActionMessage(selected_count=2).message
        assert "Confirm Merge" in msg

    def test_too_far_reads_strictly_above_max(self) -> None:
        from skiresort_planner.model.message import MergeTooFarMessage

        # Fires only when the span strictly exceeds the max; the shown span must read above it.
        msg = MergeTooFarMessage(span_m=612.34, max_span_m=500.0).message
        assert "Too Far" in msg and "max: 500m" in msg
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

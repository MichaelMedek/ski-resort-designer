"""Unit tests for user-facing toast messages with COMPUTED content (model/message.py).

Focuses on messages whose text is derived (formatted numbers, branches) rather than static,
so a formatting regression is caught. Static-text toasts need no test.
"""

import re

from skiresort_planner.model.message import (
    LiftMustGoUphillMessage,
    OSMImportErrorMessage,
    OSMImportSummaryMessage,
    RoadTooSteepMessage,
    TargetNotDownhillMessage,
    TargetTooFarMessage,
)


def _first_number(text: str) -> float:
    """First signed decimal number in the text."""
    m = re.search(r"-?\d+\.?\d*", text)
    assert m is not None, f"no number in {text!r}"
    return float(m.group())


class TestRoadTooSteepMessage:
    def test_just_over_cap_reads_strictly_above_limit(self) -> None:
        # Fires only when every route is strictly over the band. A value just over the cap
        # must NOT render "15%, over the ±15% limit" (a self-contradiction). It shows a
        # decimal strictly above 15.
        msg = RoadTooSteepMessage(gentlest_pct=15.02, max_grade_pct=15.0).message
        assert "±15% limit" in msg
        shown = _first_number(msg.split("gentlest possible is")[1])
        assert shown > 15.0, f"gentlest must read strictly above the ±15% cap: {msg}"

    def test_clearly_over_cap_message(self) -> None:
        msg = RoadTooSteepMessage(gentlest_pct=22.0, max_grade_pct=15.0).message
        assert "22.0%" in msg and "±15% limit" in msg

    def test_no_route_branch(self) -> None:
        msg = RoadTooSteepMessage(gentlest_pct=None, max_grade_pct=15.0).message
        assert "no route" in msg and "±15%" in msg


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
    def test_summary_reports_counts(self) -> None:
        msg = OSMImportSummaryMessage(pistes=7, lifts=3, skipped=0, duplicates=0).message
        assert "7 pistes" in msg and "3 lifts" in msg
        assert "skipped" not in msg and "already imported" not in msg, "no extra clause when clean"

    def test_summary_states_skipped_when_present(self) -> None:
        msg = OSMImportSummaryMessage(pistes=5, lifts=2, skipped=4, duplicates=0).message
        assert "5 pistes" in msg and "2 lifts" in msg
        assert "4 skipped" in msg, "skipped ways must be visible (only full imports)"

    def test_summary_states_duplicates_when_present(self) -> None:
        msg = OSMImportSummaryMessage(pistes=0, lifts=0, skipped=0, duplicates=6).message
        assert "6 already imported" in msg, "duplicates must be visible and distinct from skipped"

    def test_summary_states_both_skipped_and_duplicates(self) -> None:
        msg = OSMImportSummaryMessage(pistes=1, lifts=1, skipped=2, duplicates=3).message
        assert "2 skipped" in msg and "3 already imported" in msg

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

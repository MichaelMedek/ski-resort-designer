"""Unit tests for user-facing toast messages with COMPUTED content (model/message.py).

Focuses on messages whose text is derived (formatted numbers, branches) rather than static,
so a formatting regression is caught. Static-text toasts need no test.
"""

import re

from skiresort_planner.model.message import (
    LiftMustGoUphillMessage,
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

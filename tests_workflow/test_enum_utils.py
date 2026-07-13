"""Unit tests for enum_utils.enum_eq — reload-safe Enum comparison.

enum_eq exists because Streamlit re-imports modules on rerun, creating a FRESH class per
Enum; values that persist across reruns are instances of the OLD class, so `is` (and, for
plain Enums, `==`) fail. enum_eq compares the stable `repr()` form instead.
"""

from enum import Enum

from skiresort_planner.enum_utils import enum_eq
from skiresort_planner.model.path_segment import SegmentKind
from skiresort_planner.ui.context import EntityKind


class TestEnumEq:
    def test_same_member_is_equal(self) -> None:
        assert enum_eq(SegmentKind.ROAD, SegmentKind.ROAD)
        assert enum_eq(EntityKind.SLOPE, EntityKind.SLOPE)

    def test_different_members_not_equal(self) -> None:
        assert not enum_eq(SegmentKind.ROAD, SegmentKind.SLOPE)
        assert not enum_eq(EntityKind.SLOPE, EntityKind.LIFT)

    def test_reloaded_class_same_member_is_equal(self) -> None:
        # Simulate a Streamlit reload: a fresh class with identical name + values.
        reloaded = Enum("SegmentKind", {"SLOPE": "slope", "ROAD": "road"}, type=str)  # type: ignore[misc]
        assert enum_eq(reloaded.ROAD, SegmentKind.ROAD)
        assert enum_eq(SegmentKind.SLOPE, reloaded.SLOPE)

    def test_reloaded_class_different_member_not_equal(self) -> None:
        reloaded = Enum("SegmentKind", {"SLOPE": "slope", "ROAD": "road"}, type=str)  # type: ignore[misc]
        assert not enum_eq(reloaded.ROAD, SegmentKind.SLOPE)

    def test_different_enum_classes_never_equal(self) -> None:
        # repr() form is class-qualified, so members of different enums never match
        # even if their .value strings collide ("slope" == "slope").
        assert not enum_eq(SegmentKind.SLOPE, EntityKind.SLOPE)

    def test_plain_enum_reload_safe(self) -> None:
        # Plain (non-str) Enums are the case where `==` itself fails across reloads;
        # enum_eq must still work by comparing the "<ClassName.MEMBER: value>" repr.
        class Color(Enum):
            RED = 1
            BLUE = 2

        reloaded = Enum("Color", {"RED": 1, "BLUE": 2})  # type: ignore[misc]
        assert enum_eq(Color.RED, reloaded.RED)
        assert not enum_eq(Color.RED, reloaded.BLUE)

"""Unit tests for ClickDetector parsing logic.

Tests all click type parsing in consolidated tests.
"""

from dataclasses import dataclass

from skiresort_planner.model.click_info import MapClickType, MarkerType
from skiresort_planner.ui.click_detector import ClickDetector


@dataclass
class MockDeduplicationContext:
    """Mock click deduplication context for testing."""

    _seen: set = None

    def __post_init__(self) -> None:
        self._seen = set()

    def is_new_click(self, coord: tuple | None, obj_id: str | None) -> bool:
        """Return True if this click hasn't been seen before."""
        key = (coord, obj_id)
        if key in self._seen:
            return False
        self._seen.add(key)
        return True

    def clear(self) -> None:
        """Clear seen clicks."""
        self._seen.clear()


class TestClickDetectorParsing:
    """Tests for click parsing from Pydeck events."""

    def test_parse_terrain_click_from_coordinate(self) -> None:
        """Terrain click detected from clicked_coordinate (no object).

        Tests:
        - Returns ClickInfo with TERRAIN type
        - Coordinates extracted correctly
        """
        dedup = MockDeduplicationContext()
        detector = ClickDetector(dedup=dedup)

        result = detector.detect(clicked_object=None, clicked_coordinate=[10.27, 46.97])

        assert result is not None, "Should return ClickInfo"
        assert result.click_type == MapClickType.TERRAIN
        assert abs(result.lon - 10.27) < 0.001
        assert abs(result.lat - 46.97) < 0.001

    def test_parse_terrain_click_from_invisible_layer(self) -> None:
        """Terrain click from invisible ScatterplotLayer grid."""
        dedup = MockDeduplicationContext()
        detector = ClickDetector(dedup=dedup)

        result = detector.detect(
            clicked_object={"type": "terrain", "lon": 10.5, "lat": 46.5},
            clicked_coordinate=None,
        )

        assert result is not None
        assert result.click_type == MapClickType.TERRAIN
        assert abs(result.lon - 10.5) < 0.001
        assert abs(result.lat - 46.5) < 0.001

    def test_parse_node_click(self) -> None:
        """Node click extracts node ID."""
        dedup = MockDeduplicationContext()
        detector = ClickDetector(dedup=dedup)

        result = detector.detect(
            clicked_object={"type": "node", "id": "N42"},
            clicked_coordinate=None,
        )

        assert result is not None
        assert result.click_type == MapClickType.MARKER
        assert result.marker_type == MarkerType.NODE
        assert result.node_id == "N42"

    def test_parse_slope_click(self) -> None:
        """Slope click extracts slope ID."""
        dedup = MockDeduplicationContext()
        detector = ClickDetector(dedup=dedup)

        result = detector.detect(
            clicked_object={"type": "slope", "id": "SL1"},
            clicked_coordinate=None,
        )

        assert result is not None
        assert result.marker_type == MarkerType.SLOPE
        assert result.slope_id == "SL1"

    def test_parse_lift_click(self) -> None:
        """Lift click extracts lift ID."""
        dedup = MockDeduplicationContext()
        detector = ClickDetector(dedup=dedup)

        result = detector.detect(
            clicked_object={"type": "lift", "id": "L5"},
            clicked_coordinate=None,
        )

        assert result is not None
        assert result.marker_type == MarkerType.LIFT
        assert result.lift_id == "L5"

    def test_parse_pylon_click(self) -> None:
        """Pylon click extracts lift ID and pylon index."""
        dedup = MockDeduplicationContext()
        detector = ClickDetector(dedup=dedup)

        result = detector.detect(
            clicked_object={"type": "pylon", "lift_id": "L1", "pylon_index": 3},
            clicked_coordinate=None,
        )

        assert result is not None
        assert result.marker_type == MarkerType.PYLON
        assert result.lift_id == "L1"
        assert result.pylon_index == 3

    def test_parse_proposal_endpoint_click(self) -> None:
        """Proposal endpoint click extracts proposal index."""
        dedup = MockDeduplicationContext()
        detector = ClickDetector(dedup=dedup)

        result = detector.detect(
            clicked_object={"type": "proposal_endpoint", "proposal_index": 2},
            clicked_coordinate=None,
        )

        assert result is not None
        assert result.marker_type == MarkerType.PROPOSAL_ENDPOINT
        assert result.proposal_index == 2

    def test_unknown_type_returns_none(self) -> None:
        """Unknown object type returns None."""
        dedup = MockDeduplicationContext()
        detector = ClickDetector(dedup=dedup)

        result = detector.detect(
            clicked_object={"type": "unknown_thing"},
            clicked_coordinate=None,
        )

        assert result is None, "Unknown type should return None"

    def test_missing_required_field_returns_none(self) -> None:
        """Object with missing required field returns None."""
        dedup = MockDeduplicationContext()
        detector = ClickDetector(dedup=dedup)

        # Node without ID
        result = detector.detect(
            clicked_object={"type": "node"},  # missing 'id'
            clicked_coordinate=None,
        )

        assert result is None, "Missing required field should return None"


class TestClickDeduplication:
    """Tests for click deduplication logic."""

    def test_duplicate_click_rejected(self) -> None:
        """Same click is rejected on second occurrence."""
        dedup = MockDeduplicationContext()
        detector = ClickDetector(dedup=dedup)

        obj = {"type": "node", "id": "N1"}

        result1 = detector.detect(clicked_object=obj, clicked_coordinate=None)
        result2 = detector.detect(clicked_object=obj, clicked_coordinate=None)

        assert result1 is not None, "First click should be accepted"
        assert result2 is None, "Duplicate click should be rejected"

    def test_different_clicks_accepted(self) -> None:
        """Different clicks are both accepted."""
        dedup = MockDeduplicationContext()
        detector = ClickDetector(dedup=dedup)

        result1 = detector.detect(
            clicked_object={"type": "node", "id": "N1"},
            clicked_coordinate=None,
        )
        result2 = detector.detect(
            clicked_object={"type": "node", "id": "N2"},
            clicked_coordinate=None,
        )

        assert result1 is not None, "First click should be accepted"
        assert result2 is not None, "Different object should also be accepted"

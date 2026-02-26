"""Unit tests for ClickDetector parsing logic.

Uses parametrize to test all click type parsing patterns.
"""

from dataclasses import dataclass, field

import pytest

from skiresort_planner.model.click_info import MapClickType, MarkerType
from skiresort_planner.ui.click_detector import ClickDetector


@dataclass
class MockDeduplicationContext:
    """Mock click deduplication context for testing."""

    _seen: set = field(default_factory=set)

    def is_new_click(self, coord: tuple | None, obj_id: str | None) -> bool:
        """Return True if this click hasn't been seen before."""
        key = (coord, obj_id)
        if key in self._seen:
            return False
        self._seen.add(key)
        return True


@pytest.fixture
def detector() -> ClickDetector:
    """Fresh ClickDetector with mock dedup for each test."""
    return ClickDetector(dedup=MockDeduplicationContext())


class TestClickDetectorParsing:
    """Parametrized tests for click parsing from Pydeck events."""

    @pytest.mark.parametrize(
        "clicked_object,clicked_coordinate,expected_lon,expected_lat",
        [
            pytest.param(None, [10.27, 46.97], 10.27, 46.97, id="from_coordinate"),
            pytest.param({"type": "terrain", "lon": 10.5, "lat": 46.5}, None, 10.5, 46.5, id="from_invisible_layer"),
        ],
    )
    def test_terrain_click_parsing(
        self,
        detector: ClickDetector,
        clicked_object: dict | None,
        clicked_coordinate: list | None,
        expected_lon: float,
        expected_lat: float,
    ) -> None:
        """Terrain clicks extract coordinates from coordinate or object."""
        result = detector.detect(clicked_object=clicked_object, clicked_coordinate=clicked_coordinate)
        assert result is not None
        assert result.click_type == MapClickType.TERRAIN
        assert abs(result.lon - expected_lon) < 0.001
        assert abs(result.lat - expected_lat) < 0.001

    @pytest.mark.parametrize(
        "clicked_object,expected_marker_type,expected_attrs",
        [
            pytest.param({"type": "node", "id": "N42"}, MarkerType.NODE, {"node_id": "N42"}, id="node"),
            pytest.param({"type": "slope", "id": "SL1"}, MarkerType.SLOPE, {"slope_id": "SL1"}, id="slope"),
            pytest.param({"type": "lift", "id": "L5"}, MarkerType.LIFT, {"lift_id": "L5"}, id="lift"),
            pytest.param(
                {"type": "pylon", "lift_id": "L1", "pylon_index": 3},
                MarkerType.PYLON,
                {"lift_id": "L1", "pylon_index": 3},
                id="pylon",
            ),
            pytest.param(
                {"type": "proposal_endpoint", "proposal_index": 2},
                MarkerType.PROPOSAL_ENDPOINT,
                {"proposal_index": 2},
                id="proposal",
            ),
        ],
    )
    def test_marker_click_parsing(
        self, detector: ClickDetector, clicked_object: dict, expected_marker_type: MarkerType, expected_attrs: dict
    ) -> None:
        """Marker clicks extract correct type and attributes."""
        result = detector.detect(clicked_object=clicked_object, clicked_coordinate=None)
        assert result is not None
        assert result.click_type == MapClickType.MARKER
        assert result.marker_type == expected_marker_type
        for attr, value in expected_attrs.items():
            assert getattr(result, attr) == value

    @pytest.mark.parametrize(
        "clicked_object",
        [
            pytest.param({"type": "unknown_thing"}, id="unknown_type"),
            pytest.param({"type": "node"}, id="node_missing_id"),
        ],
    )
    def test_invalid_object_returns_none(self, detector: ClickDetector, clicked_object: dict) -> None:
        """Unknown type or missing required field returns None."""
        result = detector.detect(clicked_object=clicked_object, clicked_coordinate=None)
        assert result is None


class TestClickDeduplication:
    """Tests for click deduplication logic."""

    def test_duplicate_click_rejected_different_accepted(self, detector: ClickDetector) -> None:
        """Same click rejected on second occurrence; different clicks accepted."""
        obj = {"type": "node", "id": "N1"}

        result1 = detector.detect(clicked_object=obj, clicked_coordinate=None)
        result_dup = detector.detect(clicked_object=obj, clicked_coordinate=None)
        result_diff = detector.detect(clicked_object={"type": "node", "id": "N2"}, clicked_coordinate=None)

        assert result1 is not None, "First click accepted"
        assert result_dup is None, "Duplicate rejected"
        assert result_diff is not None, "Different click accepted"

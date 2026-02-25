"""Tests for click_detector.py - Pydeck object click parsing.

Tests the ClickDetector class that converts Pydeck clicked object dicts
into ClickInfo objects. This is testable without Streamlit since it's
pure logic parsing dicts.
"""

import pytest

from skiresort_planner.constants import ClickConfig
from skiresort_planner.model.click_info import MapClickType, MarkerType
from skiresort_planner.ui.click_detector import ClickDetector
from skiresort_planner.ui.state_machine import ClickDeduplicationContext


@pytest.fixture
def detector() -> ClickDetector:
    """Fresh ClickDetector with clean dedup context (debounce disabled for testing)."""
    return ClickDetector(dedup=ClickDeduplicationContext(debounce_seconds=0))


class TestTerrainClicks:
    """Terrain clicks (no object picked)."""

    def test_terrain_click_returns_click_info(self, detector: ClickDetector) -> None:
        """Terrain click with coordinate creates TERRAIN ClickInfo."""
        result = detector.detect(
            clicked_object=None,
            clicked_coordinate=[10.27, 46.97],  # [lon, lat]
        )
        assert result is not None
        assert result.click_type == MapClickType.TERRAIN
        assert result.lat == 46.97
        assert result.lon == 10.27

    def test_no_object_no_coord_returns_none(self, detector: ClickDetector) -> None:
        """No object and no coordinate returns None."""
        result = detector.detect(clicked_object=None, clicked_coordinate=None)
        assert result is None

    def test_duplicate_terrain_click_rejected(self, detector: ClickDetector) -> None:
        """Same terrain coordinate twice is rejected."""
        coord = [10.27, 46.97]
        result1 = detector.detect(clicked_object=None, clicked_coordinate=coord)
        assert result1 is not None

        result2 = detector.detect(clicked_object=None, clicked_coordinate=coord)
        assert result2 is None


class TestInvisibleTerrainLayerClicks:
    """Terrain clicks via invisible ScatterplotLayer grid."""

    def test_terrain_layer_click_with_lon_lat(self, detector: ClickDetector) -> None:
        """Invisible terrain ScatterplotLayer click with lon/lat fields."""
        obj = {
            "type": ClickConfig.TYPE_TERRAIN,
            "id": "terrain_0",
            "lon": 10.27,
            "lat": 46.97,
        }
        result = detector.detect(clicked_object=obj, clicked_coordinate=None)

        assert result is not None
        assert result.click_type == MapClickType.TERRAIN
        assert result.lon == 10.27
        assert result.lat == 46.97

    def test_terrain_layer_click_no_coords_returns_none(self, detector: ClickDetector) -> None:
        """Terrain layer click without coordinates returns None."""
        obj = {
            "type": ClickConfig.TYPE_TERRAIN,
            "id": "terrain_0",
        }
        result = detector.detect(clicked_object=obj, clicked_coordinate=None)
        assert result is None

    def test_terrain_layer_deduplication(self, detector: ClickDetector) -> None:
        """Same terrain layer click position is deduplicated."""
        obj = {
            "type": ClickConfig.TYPE_TERRAIN,
            "id": "terrain_0",
            "lon": 10.27,
            "lat": 46.97,
        }
        result1 = detector.detect(clicked_object=obj, clicked_coordinate=None)
        assert result1 is not None

        result2 = detector.detect(clicked_object=obj, clicked_coordinate=None)
        assert result2 is None


class TestNodeClicks:
    """Node marker clicks."""

    def test_node_click_parses_correctly(self, detector: ClickDetector) -> None:
        """Node object is parsed to NODE marker click."""
        obj = {"type": ClickConfig.TYPE_NODE, "id": "N1", "position": [10.27, 46.97]}
        result = detector.detect(clicked_object=obj, clicked_coordinate=None)

        assert result is not None
        assert result.click_type == MapClickType.MARKER
        assert result.marker_type == MarkerType.NODE
        assert result.node_id == "N1"

    def test_node_click_missing_id_returns_none(self, detector: ClickDetector) -> None:
        """Node click without id returns None."""
        obj = {"type": ClickConfig.TYPE_NODE, "position": [10.27, 46.97]}
        result = detector.detect(clicked_object=obj, clicked_coordinate=None)
        assert result is None


class TestSlopeClicks:
    """Slope marker clicks."""

    def test_slope_click_parses_correctly(self, detector: ClickDetector) -> None:
        """Slope icon click returns SLOPE marker."""
        obj = {"type": ClickConfig.TYPE_SLOPE, "id": "slope_1"}
        result = detector.detect(clicked_object=obj, clicked_coordinate=None)

        assert result is not None
        assert result.click_type == MapClickType.MARKER
        assert result.marker_type == MarkerType.SLOPE
        assert result.slope_id == "slope_1"

    def test_segment_click_parses_correctly(self, detector: ClickDetector) -> None:
        """Segment click returns SEGMENT marker."""
        obj = {"type": ClickConfig.TYPE_SEGMENT, "id": "S1"}
        result = detector.detect(clicked_object=obj, clicked_coordinate=None)

        assert result is not None
        assert result.marker_type == MarkerType.SEGMENT
        assert result.segment_id == "S1"


class TestLiftClicks:
    """Lift and pylon clicks."""

    def test_lift_click_parses_correctly(self, detector: ClickDetector) -> None:
        """Lift click returns LIFT marker."""
        obj = {"type": ClickConfig.TYPE_LIFT, "id": "lift_1", "lift_type": "chairlift"}
        result = detector.detect(clicked_object=obj, clicked_coordinate=None)

        assert result is not None
        assert result.marker_type == MarkerType.LIFT
        assert result.lift_id == "lift_1"

    def test_pylon_click_parses_correctly(self, detector: ClickDetector) -> None:
        """Pylon click returns PYLON marker with lift_id and index."""
        obj = {
            "type": ClickConfig.TYPE_PYLON,
            "lift_id": "lift_1",
            "pylon_index": 2,
        }
        result = detector.detect(clicked_object=obj, clicked_coordinate=None)

        assert result is not None
        assert result.marker_type == MarkerType.PYLON
        assert result.lift_id == "lift_1"
        assert result.pylon_index == 2

    def test_pylon_click_missing_lift_id_returns_none(self, detector: ClickDetector) -> None:
        """Pylon without lift_id returns None."""
        obj = {"type": ClickConfig.TYPE_PYLON, "pylon_index": 0}
        result = detector.detect(clicked_object=obj, clicked_coordinate=None)
        assert result is None

    def test_pylon_click_missing_index_returns_none(self, detector: ClickDetector) -> None:
        """Pylon without pylon_index returns None."""
        obj = {"type": ClickConfig.TYPE_PYLON, "lift_id": "lift_1"}
        result = detector.detect(clicked_object=obj, clicked_coordinate=None)
        assert result is None


class TestProposalClicks:
    """Proposal endpoint and body clicks."""

    def test_proposal_endpoint_click(self, detector: ClickDetector) -> None:
        """Proposal endpoint click for committing."""
        obj = {"type": ClickConfig.TYPE_PROPOSAL_ENDPOINT, "proposal_index": 1}
        result = detector.detect(clicked_object=obj, clicked_coordinate=None)

        assert result is not None
        assert result.marker_type == MarkerType.PROPOSAL_ENDPOINT
        assert result.proposal_index == 1

    def test_proposal_body_click(self, detector: ClickDetector) -> None:
        """Proposal body click for selection."""
        obj = {"type": ClickConfig.TYPE_PROPOSAL_BODY, "proposal_index": 0}
        result = detector.detect(clicked_object=obj, clicked_coordinate=None)

        assert result is not None
        assert result.marker_type == MarkerType.PROPOSAL_BODY
        assert result.proposal_index == 0

    def test_proposal_missing_index_returns_none(self, detector: ClickDetector) -> None:
        """Proposal without index returns None."""
        obj = {"type": ClickConfig.TYPE_PROPOSAL_ENDPOINT}
        result = detector.detect(clicked_object=obj, clicked_coordinate=None)
        assert result is None


class TestUnknownTypes:
    """Handling of unknown object types."""

    def test_unknown_type_returns_none(self, detector: ClickDetector) -> None:
        """Unknown object type returns None (not crash)."""
        obj = {"type": "some_unknown_type", "id": "xyz"}
        result = detector.detect(clicked_object=obj, clicked_coordinate=None)
        assert result is None

    def test_object_without_type_returns_none(self, detector: ClickDetector) -> None:
        """Object without type field returns None."""
        obj = {"id": "test", "position": [10.0, 47.0]}
        result = detector.detect(clicked_object=obj, clicked_coordinate=None)
        assert result is None


class TestDeduplicationWithObjects:
    """Object click deduplication."""

    def test_same_object_click_rejected(self, detector: ClickDetector) -> None:
        """Clicking same object twice is rejected."""
        obj = {"type": ClickConfig.TYPE_NODE, "id": "N1"}
        result1 = detector.detect(clicked_object=obj, clicked_coordinate=None)
        assert result1 is not None

        result2 = detector.detect(clicked_object=obj, clicked_coordinate=None)
        assert result2 is None

    def test_different_objects_accepted(self, detector: ClickDetector) -> None:
        """Different objects are both accepted."""
        obj1 = {"type": ClickConfig.TYPE_NODE, "id": "N1"}
        obj2 = {"type": ClickConfig.TYPE_NODE, "id": "N2"}

        result1 = detector.detect(clicked_object=obj1, clicked_coordinate=None)
        result2 = detector.detect(clicked_object=obj2, clicked_coordinate=None)

        assert result1 is not None
        assert result2 is not None

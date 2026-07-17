"""Unit tests for ClickDetector parsing logic.

Uses parametrize to test all click type parsing patterns.
"""

from dataclasses import dataclass, field

import pytest

from skiresort_planner.model.click_info import MapClickType, MarkerType
from skiresort_planner.ui.click_detector import ClickDetector
from skiresort_planner.ui.context import ClickDeduplicationContext


@dataclass
class MockDeduplicationContext(ClickDeduplicationContext):
    """A real ClickDeduplicationContext whose dedup is a simple seen-set (no timing/debounce).

    Subclasses the production context so it satisfies ClickDetector's `dedup` type exactly, while
    overriding is_new_click with deterministic set-membership semantics for the parsing tests.
    """

    _seen: set[tuple[object, str | None]] = field(default_factory=set)

    def is_new_click(self, coord: tuple[float, ...] | None, obj_id: str | None) -> bool:
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
        clicked_object: dict[str, object] | None,
        clicked_coordinate: list[float] | None,
        expected_lon: float,
        expected_lat: float,
    ) -> None:
        """Terrain clicks extract coordinates from coordinate or object."""
        result = detector.detect(clicked_object=clicked_object, clicked_coordinate=clicked_coordinate)
        assert result is not None
        assert result.lon is not None and result.lat is not None
        assert result.click_type == MapClickType.TERRAIN
        assert abs(result.lon - expected_lon) < 0.001
        assert abs(result.lat - expected_lat) < 0.001

    @pytest.mark.parametrize(
        "clicked_object,expected_marker_type,expected_attrs",
        [
            pytest.param({"type": "node", "id": "N42"}, MarkerType.NODE, {"node_id": "N42"}, id="node"),
            pytest.param({"type": "slope", "id": "SL1"}, MarkerType.SLOPE, {"slope_id": "SL1"}, id="slope"),
            pytest.param({"type": "segment", "id": "S3"}, MarkerType.SEGMENT, {"segment_id": "S3"}, id="segment"),
            pytest.param({"type": "lift", "id": "L5"}, MarkerType.LIFT, {"lift_id": "L5"}, id="lift"),
            pytest.param({"type": "road", "id": "R2"}, MarkerType.ROAD, {"road_id": "R2"}, id="road"),
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
            pytest.param(
                {"type": "proposal_body", "proposal_index": 2},
                MarkerType.PROPOSAL_BODY,
                {"proposal_index": 2},
                id="proposal_body",
            ),
            pytest.param(
                {"type": "import_center"},
                MarkerType.IMPORT_CENTER,
                {},  # positionless confirm marker — no id fields
                id="import_center",
            ),
        ],
    )
    def test_marker_click_parsing(
        self,
        detector: ClickDetector,
        clicked_object: dict[str, object],
        expected_marker_type: MarkerType,
        expected_attrs: dict[str, object],
    ) -> None:
        """Marker clicks extract correct type and attributes."""
        result = detector.detect(clicked_object=clicked_object, clicked_coordinate=None)
        assert result is not None
        assert result.click_type == MapClickType.MARKER
        assert result.marker_type == expected_marker_type
        for attr, value in expected_attrs.items():
            assert getattr(result, attr) == value

    def test_segment_marker_carries_click_coordinate(self, detector: ClickDetector) -> None:
        """The SEGMENT (path belt) marker keeps the deck.gl coordinate so a path click adds a node."""
        result = detector.detect(clicked_object={"type": "segment", "id": "S1"}, clicked_coordinate=[10.27, 46.97])
        assert result is not None
        assert result.marker_type == MarkerType.SEGMENT
        assert result.lon == pytest.approx(10.27)
        assert result.lat == pytest.approx(46.97)

    @pytest.mark.parametrize(
        "clicked_object,expected_marker_type",
        [
            pytest.param({"type": "node", "id": "N1"}, MarkerType.NODE, id="node"),
            pytest.param({"type": "slope", "id": "SL1"}, MarkerType.SLOPE, id="slope_icon"),
            pytest.param({"type": "road", "id": "R1"}, MarkerType.ROAD, id="road_icon"),
        ],
    )
    def test_non_positioned_marker_ignores_click_coordinate(
        self, detector: ClickDetector, clicked_object: dict[str, object], expected_marker_type: MarkerType
    ) -> None:
        """NODE and the slope/road ICON markers stay position-less — only the SEGMENT belt is positioned."""
        result = detector.detect(clicked_object=clicked_object, clicked_coordinate=[10.27, 46.97])
        assert result is not None
        assert result.marker_type == expected_marker_type
        assert result.lat is None and result.lon is None

    @pytest.mark.parametrize(
        "clicked_object",
        [
            pytest.param({"type": "unknown_thing"}, id="unknown_type"),
            pytest.param({}, id="no_type_field"),
            pytest.param({"type": "terrain"}, id="terrain_missing_coords"),
            pytest.param({"type": "Feature", "properties": {}}, id="feature_without_properties_type"),
        ],
    )
    def test_unrecognized_object_returns_none(self, detector: ClickDetector, clicked_object: dict[str, object]) -> None:
        """Genuinely unlabeled/unknown picks (pydeck can emit these) are ignored, not crashed."""
        result = detector.detect(clicked_object=clicked_object, clicked_coordinate=None)
        assert result is None

    @pytest.mark.parametrize(
        "clicked_object",
        [
            pytest.param({"type": "node"}, id="node_missing_id"),
            pytest.param({"type": "segment"}, id="segment_missing_id"),
            pytest.param({"type": "slope"}, id="slope_missing_id"),
            pytest.param({"type": "lift"}, id="lift_missing_id"),
            pytest.param({"type": "road"}, id="road_missing_id"),
            pytest.param({"type": "pylon", "lift_id": "L1"}, id="pylon_missing_index"),
            pytest.param({"type": "pylon", "pylon_index": 0}, id="pylon_missing_lift_id"),
            pytest.param({"type": "proposal_endpoint"}, id="proposal_endpoint_missing_index"),
            pytest.param({"type": "proposal_body"}, id="proposal_body_missing_index"),
        ],
    )
    def test_marker_with_matching_type_but_missing_id_raises(
        self, detector: ClickDetector, clicked_object: dict[str, object]
    ) -> None:
        """A marker whose type matches but lacks its required id/index is a rendering bug → assert, not swallow.

        We always render these markers with their id (center_map.py), so a miss here
        can only be corrupted state — it must fail loudly, not silently return None.
        """
        with pytest.raises(AssertionError):
            detector.detect(clicked_object=clicked_object, clicked_coordinate=None)

    def test_geojson_feature_extracts_type_from_properties(self, detector: ClickDetector) -> None:
        """A GeoJSON Feature parses its marker type from nested properties (segment belt click)."""
        obj: dict[str, object] = {"type": "Feature", "properties": {"type": "segment", "id": "S3"}}
        result = detector.detect(clicked_object=obj, clicked_coordinate=None)
        assert result is not None
        assert result.marker_type == MarkerType.SEGMENT
        assert result.segment_id == "S3"

    def test_get_object_id_dedup_keys(self, detector: ClickDetector, fake_st) -> None:
        """Dedup keys are formatted per-type; proposals embed dedup_epoch so regenerated ones re-click.

        Guards the regression where dropping the epoch from the proposal key would make a freshly
        regenerated proposal collide with the previous generation's key and be swallowed.
        """
        fake_st.session_state["dedup_epoch"] = 5
        assert detector._get_object_id(obj={"type": "pylon", "lift_id": "L1", "pylon_index": 3}) == "pylon_L1_3"
        assert detector._get_object_id(obj={"type": "node", "id": "N42"}) == "node_N42"
        assert (
            detector._get_object_id(obj={"type": "proposal_endpoint", "proposal_index": 2}) == "proposal_endpoint_2_v5"
        )
        assert detector._get_object_id(obj={}) == ""


class TestClickDetectorDeduplication:
    """Tests for ClickDetector deduplication logic."""

    def test_duplicate_click_rejected_different_accepted(self, detector: ClickDetector) -> None:
        """Same click rejected on second occurrence; different clicks accepted."""
        obj: dict[str, object] = {"type": "node", "id": "N1"}

        result1 = detector.detect(clicked_object=obj, clicked_coordinate=None)
        result_dup = detector.detect(clicked_object=obj, clicked_coordinate=None)
        result_diff = detector.detect(clicked_object={"type": "node", "id": "N2"}, clicked_coordinate=None)

        assert result1 is not None, "First click accepted"
        assert result_dup is None, "Duplicate rejected"
        assert result_diff is not None, "Different click accepted"

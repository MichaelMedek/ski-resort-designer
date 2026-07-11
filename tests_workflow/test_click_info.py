"""Unit tests for ClickInfo dataclass - validation and formatting logic.

Uses parametrize to consolidate repetitive validation patterns.
"""

import pytest

from skiresort_planner.model.click_info import ClickInfo, MapClickType, MarkerType


class TestClickInfoValidation:
    """Parametrized tests for ClickInfo validation rules."""

    @pytest.mark.parametrize(
        "kwargs,error_match",
        [
            pytest.param(
                {"click_type": MapClickType.TERRAIN, "lon": 10.5},
                "TERRAIN click must have lat/lon",
                id="terrain_missing_lat",
            ),
            pytest.param(
                {"click_type": MapClickType.TERRAIN, "lat": 46.5},
                "TERRAIN click must have lat/lon",
                id="terrain_missing_lon",
            ),
            pytest.param(
                {"click_type": MapClickType.TERRAIN, "lat": 46.5, "lon": 10.5, "marker_type": MarkerType.NODE},
                "TERRAIN click must NOT have marker_type",
                id="terrain_with_marker",
            ),
            pytest.param(
                {"click_type": MapClickType.MARKER}, "MARKER click must have marker_type", id="marker_missing_type"
            ),
            pytest.param(
                {
                    "click_type": MapClickType.MARKER,
                    "marker_type": MarkerType.NODE,
                    "node_id": "N1",
                    "lat": 46.5,
                    "lon": 10.5,
                },
                "MARKER click must NOT have lat/lon",
                id="marker_with_coords",
            ),
            pytest.param(
                {"click_type": MapClickType.MARKER, "marker_type": MarkerType.NODE},
                "NODE marker must have node_id",
                id="node_missing_id",
            ),
            pytest.param(
                {"click_type": MapClickType.MARKER, "marker_type": MarkerType.SLOPE},
                "SLOPE marker must have slope_id",
                id="slope_missing_id",
            ),
            pytest.param(
                {"click_type": MapClickType.MARKER, "marker_type": MarkerType.SEGMENT},
                "SEGMENT marker must have segment_id",
                id="segment_missing_id",
            ),
            pytest.param(
                {"click_type": MapClickType.MARKER, "marker_type": MarkerType.LIFT},
                "LIFT marker must have lift_id",
                id="lift_missing_id",
            ),
            pytest.param(
                {"click_type": MapClickType.MARKER, "marker_type": MarkerType.ROAD},
                "ROAD marker must have road_id",
                id="road_missing_id",
            ),
            pytest.param(
                {"click_type": MapClickType.MARKER, "marker_type": MarkerType.PYLON},
                "PYLON marker must have lift_id and pylon_index",
                id="pylon_missing_both",
            ),
            pytest.param(
                {"click_type": MapClickType.MARKER, "marker_type": MarkerType.PYLON, "lift_id": "L1"},
                "PYLON marker must have lift_id and pylon_index",
                id="pylon_missing_index",
            ),
            pytest.param(
                {"click_type": MapClickType.MARKER, "marker_type": MarkerType.PYLON, "pylon_index": 2},
                "PYLON marker must have lift_id and pylon_index",
                id="pylon_missing_lift",
            ),
            pytest.param(
                {"click_type": MapClickType.MARKER, "marker_type": MarkerType.PROPOSAL_ENDPOINT},
                "PROPOSAL_ENDPOINT marker must have proposal_index",
                id="proposal_end_missing",
            ),
            pytest.param(
                {"click_type": MapClickType.MARKER, "marker_type": MarkerType.PROPOSAL_BODY},
                "PROPOSAL_BODY marker must have proposal_index",
                id="proposal_body_missing",
            ),
        ],
    )
    def test_invalid_construction_raises(self, kwargs: dict, error_match: str) -> None:
        """Invalid ClickInfo construction raises ValueError with expected message."""
        with pytest.raises(ValueError, match=error_match):
            ClickInfo(**kwargs)

    @pytest.mark.parametrize(
        "kwargs,check_field,expected_value",
        [
            pytest.param({"click_type": MapClickType.TERRAIN, "lat": 46.5, "lon": 10.5}, "lat", 46.5, id="terrain"),
            pytest.param(
                {"click_type": MapClickType.MARKER, "marker_type": MarkerType.NODE, "node_id": "N42"},
                "node_id",
                "N42",
                id="node",
            ),
            pytest.param(
                {"click_type": MapClickType.MARKER, "marker_type": MarkerType.SLOPE, "slope_id": "SL5"},
                "slope_id",
                "SL5",
                id="slope",
            ),
            pytest.param(
                {"click_type": MapClickType.MARKER, "marker_type": MarkerType.SEGMENT, "segment_id": "S3"},
                "segment_id",
                "S3",
                id="segment",
            ),
            pytest.param(
                {"click_type": MapClickType.MARKER, "marker_type": MarkerType.LIFT, "lift_id": "L2"},
                "lift_id",
                "L2",
                id="lift",
            ),
            pytest.param(
                {"click_type": MapClickType.MARKER, "marker_type": MarkerType.ROAD, "road_id": "R2"},
                "road_id",
                "R2",
                id="road",
            ),
            pytest.param(
                {"click_type": MapClickType.MARKER, "marker_type": MarkerType.PYLON, "lift_id": "L1", "pylon_index": 3},
                "pylon_index",
                3,
                id="pylon",
            ),
            pytest.param(
                {"click_type": MapClickType.MARKER, "marker_type": MarkerType.PROPOSAL_ENDPOINT, "proposal_index": 4},
                "proposal_index",
                4,
                id="proposal_end",
            ),
            pytest.param(
                {"click_type": MapClickType.MARKER, "marker_type": MarkerType.PROPOSAL_BODY, "proposal_index": 2},
                "proposal_index",
                2,
                id="proposal_body",
            ),
        ],
    )
    def test_valid_construction(self, kwargs: dict, check_field: str, expected_value: object) -> None:
        """Valid ClickInfo construction stores expected field values."""
        click = ClickInfo(**kwargs)
        assert getattr(click, check_field) == expected_value


class TestClickInfoDisplayName:
    """Parametrized tests for display_name property."""

    @pytest.mark.parametrize(
        "kwargs,expected_name",
        [
            pytest.param(
                {"click_type": MapClickType.TERRAIN, "lat": 46.5123, "lon": 10.9876},
                "Map terrain at (46.5123, 10.9876)",
                id="terrain",
            ),
            pytest.param(
                {"click_type": MapClickType.MARKER, "marker_type": MarkerType.NODE, "node_id": "N1"},
                "Junction N1",
                id="node",
            ),
            pytest.param(
                {"click_type": MapClickType.MARKER, "marker_type": MarkerType.SLOPE, "slope_id": "SL1"},
                "Slope SL1",
                id="slope",
            ),
            pytest.param(
                {"click_type": MapClickType.MARKER, "marker_type": MarkerType.SEGMENT, "segment_id": "S1"},
                "Segment S1",
                id="segment",
            ),
            pytest.param(
                {"click_type": MapClickType.MARKER, "marker_type": MarkerType.LIFT, "lift_id": "L1"},
                "Lift L1",
                id="lift",
            ),
            pytest.param(
                {"click_type": MapClickType.MARKER, "marker_type": MarkerType.ROAD, "road_id": "R1"},
                "Road R1",
                id="road",
            ),
            pytest.param(
                {"click_type": MapClickType.MARKER, "marker_type": MarkerType.PYLON, "lift_id": "L1", "pylon_index": 2},
                "Pylon 3 on Lift L1",
                id="pylon_1indexed",
            ),
            pytest.param(
                {"click_type": MapClickType.MARKER, "marker_type": MarkerType.PROPOSAL_ENDPOINT, "proposal_index": 0},
                "Path option 1 (endpoint)",
                id="proposal_end_1indexed",
            ),
            pytest.param(
                {"click_type": MapClickType.MARKER, "marker_type": MarkerType.PROPOSAL_BODY, "proposal_index": 4},
                "Path option 5",
                id="proposal_body",
            ),
        ],
    )
    def test_display_name_format(self, kwargs: dict, expected_name: str) -> None:
        """display_name property formats correctly for all click types."""
        click = ClickInfo(**kwargs)
        assert click.display_name == expected_name


class TestClickInfoConvenienceProperties:
    """Tests for 1-indexed number properties."""

    @pytest.mark.parametrize(
        "kwargs,prop,expected",
        [
            pytest.param(
                {"click_type": MapClickType.MARKER, "marker_type": MarkerType.PROPOSAL_BODY, "proposal_index": 0},
                "proposal_number",
                1,
                id="proposal_1indexed",
            ),
            pytest.param(
                {"click_type": MapClickType.MARKER, "marker_type": MarkerType.NODE, "node_id": "N1"},
                "proposal_number",
                None,
                id="proposal_none",
            ),
            pytest.param(
                {"click_type": MapClickType.MARKER, "marker_type": MarkerType.PYLON, "lift_id": "L1", "pylon_index": 0},
                "pylon_number",
                1,
                id="pylon_1indexed",
            ),
            pytest.param(
                {"click_type": MapClickType.MARKER, "marker_type": MarkerType.LIFT, "lift_id": "L1"},
                "pylon_number",
                None,
                id="pylon_none",
            ),
        ],
    )
    def test_number_properties(self, kwargs: dict, prop: str, expected: int | None) -> None:
        """1-indexed number properties return correct values or None."""
        click = ClickInfo(**kwargs)
        assert getattr(click, prop) == expected

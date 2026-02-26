"""Unit tests for ClickInfo dataclass - validation and formatting logic.

Tests the "Pure Data" validation in ClickInfo without requiring UI components.
These tests verify the strict contract enforcement in __post_init__.
"""

import pytest

from skiresort_planner.model.click_info import ClickInfo, MapClickType, MarkerType


class TestClickInfoTerrainValidation:
    """Tests for TERRAIN click validation."""

    def test_terrain_click_requires_coordinates(self) -> None:
        """TERRAIN click must have lat/lon set."""
        # Valid terrain click
        click = ClickInfo(click_type=MapClickType.TERRAIN, lat=46.5, lon=10.5)
        assert click.lat == 46.5
        assert click.lon == 10.5

    def test_terrain_click_rejects_missing_lat(self) -> None:
        """TERRAIN click raises ValueError if lat is missing."""
        with pytest.raises(ValueError, match="TERRAIN click must have lat/lon set"):
            ClickInfo(click_type=MapClickType.TERRAIN, lon=10.5)

    def test_terrain_click_rejects_missing_lon(self) -> None:
        """TERRAIN click raises ValueError if lon is missing."""
        with pytest.raises(ValueError, match="TERRAIN click must have lat/lon set"):
            ClickInfo(click_type=MapClickType.TERRAIN, lat=46.5)

    def test_terrain_click_rejects_marker_type(self) -> None:
        """TERRAIN click raises ValueError if marker_type is set."""
        with pytest.raises(ValueError, match="TERRAIN click must NOT have marker_type set"):
            ClickInfo(click_type=MapClickType.TERRAIN, lat=46.5, lon=10.5, marker_type=MarkerType.NODE)


class TestClickInfoMarkerValidation:
    """Tests for MARKER click validation."""

    def test_marker_click_requires_marker_type(self) -> None:
        """MARKER click must have marker_type set."""
        with pytest.raises(ValueError, match="MARKER click must have marker_type set"):
            ClickInfo(click_type=MapClickType.MARKER)

    def test_marker_click_rejects_coordinates(self) -> None:
        """MARKER click raises ValueError if lat/lon are set."""
        with pytest.raises(ValueError, match="MARKER click must NOT have lat/lon set"):
            ClickInfo(click_type=MapClickType.MARKER, marker_type=MarkerType.NODE, node_id="N1", lat=46.5, lon=10.5)

    def test_node_marker_requires_node_id(self) -> None:
        """NODE marker must have node_id set."""
        with pytest.raises(ValueError, match="NODE marker must have node_id set"):
            ClickInfo(click_type=MapClickType.MARKER, marker_type=MarkerType.NODE)

    def test_slope_marker_requires_slope_id(self) -> None:
        """SLOPE marker must have slope_id set."""
        with pytest.raises(ValueError, match="SLOPE marker must have slope_id set"):
            ClickInfo(click_type=MapClickType.MARKER, marker_type=MarkerType.SLOPE)

    def test_segment_marker_requires_segment_id(self) -> None:
        """SEGMENT marker must have segment_id set."""
        with pytest.raises(ValueError, match="SEGMENT marker must have segment_id set"):
            ClickInfo(click_type=MapClickType.MARKER, marker_type=MarkerType.SEGMENT)

    def test_lift_marker_requires_lift_id(self) -> None:
        """LIFT marker must have lift_id set."""
        with pytest.raises(ValueError, match="LIFT marker must have lift_id set"):
            ClickInfo(click_type=MapClickType.MARKER, marker_type=MarkerType.LIFT)

    def test_pylon_marker_requires_both_ids(self) -> None:
        """PYLON marker must have lift_id AND pylon_index set."""
        # Missing both
        with pytest.raises(ValueError, match="PYLON marker must have lift_id and pylon_index set"):
            ClickInfo(click_type=MapClickType.MARKER, marker_type=MarkerType.PYLON)

        # Missing pylon_index
        with pytest.raises(ValueError, match="PYLON marker must have lift_id and pylon_index set"):
            ClickInfo(click_type=MapClickType.MARKER, marker_type=MarkerType.PYLON, lift_id="L1")

        # Missing lift_id
        with pytest.raises(ValueError, match="PYLON marker must have lift_id and pylon_index set"):
            ClickInfo(click_type=MapClickType.MARKER, marker_type=MarkerType.PYLON, pylon_index=2)

    def test_proposal_endpoint_requires_proposal_index(self) -> None:
        """PROPOSAL_ENDPOINT marker must have proposal_index set."""
        with pytest.raises(ValueError, match="PROPOSAL_ENDPOINT marker must have proposal_index set"):
            ClickInfo(click_type=MapClickType.MARKER, marker_type=MarkerType.PROPOSAL_ENDPOINT)

    def test_proposal_body_requires_proposal_index(self) -> None:
        """PROPOSAL_BODY marker must have proposal_index set."""
        with pytest.raises(ValueError, match="PROPOSAL_BODY marker must have proposal_index set"):
            ClickInfo(click_type=MapClickType.MARKER, marker_type=MarkerType.PROPOSAL_BODY)


class TestClickInfoValidConstruction:
    """Tests for valid ClickInfo construction."""

    def test_valid_node_click(self) -> None:
        """Valid NODE marker click."""
        click = ClickInfo(click_type=MapClickType.MARKER, marker_type=MarkerType.NODE, node_id="N42")
        assert click.node_id == "N42"

    def test_valid_slope_click(self) -> None:
        """Valid SLOPE marker click."""
        click = ClickInfo(click_type=MapClickType.MARKER, marker_type=MarkerType.SLOPE, slope_id="SL5")
        assert click.slope_id == "SL5"

    def test_valid_segment_click(self) -> None:
        """Valid SEGMENT marker click."""
        click = ClickInfo(click_type=MapClickType.MARKER, marker_type=MarkerType.SEGMENT, segment_id="S3")
        assert click.segment_id == "S3"

    def test_valid_lift_click(self) -> None:
        """Valid LIFT marker click."""
        click = ClickInfo(click_type=MapClickType.MARKER, marker_type=MarkerType.LIFT, lift_id="L2")
        assert click.lift_id == "L2"

    def test_valid_pylon_click(self) -> None:
        """Valid PYLON marker click."""
        click = ClickInfo(click_type=MapClickType.MARKER, marker_type=MarkerType.PYLON, lift_id="L1", pylon_index=3)
        assert click.lift_id == "L1"
        assert click.pylon_index == 3

    def test_valid_proposal_endpoint_click(self) -> None:
        """Valid PROPOSAL_ENDPOINT marker click."""
        click = ClickInfo(click_type=MapClickType.MARKER, marker_type=MarkerType.PROPOSAL_ENDPOINT, proposal_index=4)
        assert click.proposal_index == 4

    def test_valid_proposal_body_click(self) -> None:
        """Valid PROPOSAL_BODY marker click."""
        click = ClickInfo(click_type=MapClickType.MARKER, marker_type=MarkerType.PROPOSAL_BODY, proposal_index=2)
        assert click.proposal_index == 2


class TestClickInfoDisplayName:
    """Tests for display_name property formatting."""

    def test_terrain_display_name(self) -> None:
        """Terrain click shows coordinates."""
        click = ClickInfo(click_type=MapClickType.TERRAIN, lat=46.5123, lon=10.9876)
        assert click.display_name == "Map terrain at (46.5123, 10.9876)"

    def test_node_display_name(self) -> None:
        """Node click shows ID."""
        click = ClickInfo(click_type=MapClickType.MARKER, marker_type=MarkerType.NODE, node_id="N1")
        assert click.display_name == "Junction N1"

    def test_slope_display_name(self) -> None:
        """Slope click shows ID."""
        click = ClickInfo(click_type=MapClickType.MARKER, marker_type=MarkerType.SLOPE, slope_id="SL1")
        assert click.display_name == "Slope SL1"

    def test_segment_display_name(self) -> None:
        """Segment click shows ID."""
        click = ClickInfo(click_type=MapClickType.MARKER, marker_type=MarkerType.SEGMENT, segment_id="S1")
        assert click.display_name == "Segment S1"

    def test_lift_display_name(self) -> None:
        """Lift click shows ID."""
        click = ClickInfo(click_type=MapClickType.MARKER, marker_type=MarkerType.LIFT, lift_id="L1")
        assert click.display_name == "Lift L1"

    def test_pylon_display_name_is_1_indexed(self) -> None:
        """Pylon display name uses 1-indexed number."""
        click = ClickInfo(click_type=MapClickType.MARKER, marker_type=MarkerType.PYLON, lift_id="L1", pylon_index=2)
        # Internal index=2 -> display "Pylon 3"
        assert click.display_name == "Pylon 3 on Lift L1"

    def test_proposal_endpoint_display_name_is_1_indexed(self) -> None:
        """Proposal endpoint uses 1-indexed number."""
        click = ClickInfo(click_type=MapClickType.MARKER, marker_type=MarkerType.PROPOSAL_ENDPOINT, proposal_index=0)
        # Internal index=0 -> display "Path option 1"
        assert click.display_name == "Path option 1 (endpoint)"

    def test_proposal_body_display_name(self) -> None:
        """Proposal body display name."""
        click = ClickInfo(click_type=MapClickType.MARKER, marker_type=MarkerType.PROPOSAL_BODY, proposal_index=4)
        assert click.display_name == "Path option 5"


class TestClickInfoDedupKey:
    """Tests for make_dedup_key method."""

    def test_terrain_dedup_key_format(self) -> None:
        """Terrain dedup key includes rounded coordinates."""
        click = ClickInfo(click_type=MapClickType.TERRAIN, lat=46.512345, lon=10.987654)
        key = click.make_dedup_key()
        assert key.startswith("terrain_")
        # Should contain rounded coordinates (6 decimal places)
        assert "46.512345" in key
        assert "10.987654" in key

    def test_node_dedup_key(self) -> None:
        """Node dedup key includes ID."""
        click = ClickInfo(click_type=MapClickType.MARKER, marker_type=MarkerType.NODE, node_id="N42")
        assert click.make_dedup_key() == "marker_node_N42"

    def test_slope_dedup_key(self) -> None:
        """Slope dedup key includes ID."""
        click = ClickInfo(click_type=MapClickType.MARKER, marker_type=MarkerType.SLOPE, slope_id="SL5")
        assert click.make_dedup_key() == "marker_slope_SL5"

    def test_pylon_dedup_key_includes_index_and_lift(self) -> None:
        """Pylon dedup key includes 0-indexed pylon_index and lift_id."""
        click = ClickInfo(click_type=MapClickType.MARKER, marker_type=MarkerType.PYLON, lift_id="L1", pylon_index=2)
        # Internal 0-indexed
        assert click.make_dedup_key() == "marker_pylon_2_L1"

    def test_proposal_endpoint_dedup_key(self) -> None:
        """Proposal endpoint dedup key includes 0-indexed index."""
        click = ClickInfo(click_type=MapClickType.MARKER, marker_type=MarkerType.PROPOSAL_ENDPOINT, proposal_index=4)
        assert click.make_dedup_key() == "marker_proposal_end_4"


class TestClickInfoConvenienceProperties:
    """Tests for proposal_number and pylon_number properties."""

    def test_proposal_number_returns_1_indexed(self) -> None:
        """proposal_number property returns 1-indexed value."""
        click = ClickInfo(click_type=MapClickType.MARKER, marker_type=MarkerType.PROPOSAL_BODY, proposal_index=0)
        assert click.proposal_number == 1

    def test_proposal_number_returns_none_for_non_proposal(self) -> None:
        """proposal_number returns None for non-proposal clicks."""
        click = ClickInfo(click_type=MapClickType.MARKER, marker_type=MarkerType.NODE, node_id="N1")
        assert click.proposal_number is None

    def test_pylon_number_returns_1_indexed(self) -> None:
        """pylon_number property returns 1-indexed value."""
        click = ClickInfo(click_type=MapClickType.MARKER, marker_type=MarkerType.PYLON, lift_id="L1", pylon_index=0)
        assert click.pylon_number == 1

    def test_pylon_number_returns_none_for_non_pylon(self) -> None:
        """pylon_number returns None for non-pylon clicks."""
        click = ClickInfo(click_type=MapClickType.MARKER, marker_type=MarkerType.LIFT, lift_id="L1")
        assert click.pylon_number is None

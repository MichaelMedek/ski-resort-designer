"""Tests for ClickInfo - unified click information.

Tests cover:
1. Factory method: from_tooltip parsing
2. Direct ID storage validation
3. Display properties: display_name, make_dedup_key
4. Convenience properties: proposal_number, pylon_number (1-indexed)

Naming Convention:
    - pylon_index: 0-indexed internal storage
    - pylon_number: 1-indexed for display (property)
    - proposal_index: 0-indexed internal storage
    - proposal_number: 1-indexed for display (property)
"""

import pytest
from skiresort_planner.model.click_info import ClickInfo, MapClickType, MarkerType


# =============================================================================
# TOOLTIP PARSING - ClickInfo.from_tooltip
# =============================================================================


class TestFromTooltip:
    """Tests for ClickInfo.from_tooltip - the unified tooltip parsing logic."""

    def test_node_tooltip(self) -> None:
        """'Build From Node N1' → NODE, node_id='N1'."""
        result = ClickInfo.from_tooltip(tooltip="Build From Node N1")
        assert result.click_type == MapClickType.MARKER
        assert result.marker_type == MarkerType.NODE
        assert result.node_id == "N1"

    def test_slope_tooltip(self) -> None:
        """'View Slope SL1' → SLOPE, slope_id='SL1'."""
        result = ClickInfo.from_tooltip(tooltip="View Slope SL1")
        assert result.click_type == MapClickType.MARKER
        assert result.marker_type == MarkerType.SLOPE
        assert result.slope_id == "SL1"

    def test_lift_tooltip(self) -> None:
        """'View Lift L1' → LIFT, lift_id='L1'."""
        result = ClickInfo.from_tooltip(tooltip="View Lift L1")
        assert result.click_type == MapClickType.MARKER
        assert result.marker_type == MarkerType.LIFT
        assert result.lift_id == "L1"

    def test_pylon_tooltip(self) -> None:
        """'View Pylon 3 on L1' → PYLON, pylon_index=2 (0-indexed), lift_id='L1'."""
        result = ClickInfo.from_tooltip(tooltip="View Pylon 3 on L1")
        assert result.click_type == MapClickType.MARKER
        assert result.marker_type == MarkerType.PYLON
        assert result.pylon_index == 2  # 0-indexed internally
        assert result.pylon_number == 3  # 1-indexed property
        assert result.lift_id == "L1"

    def test_proposal_body_tooltip(self) -> None:
        """'Select Proposal 1' → PROPOSAL_BODY, proposal_index=0 (0-indexed)."""
        result = ClickInfo.from_tooltip(tooltip="Select Proposal 1")
        assert result.click_type == MapClickType.MARKER
        assert result.marker_type == MarkerType.PROPOSAL_BODY
        assert result.proposal_index == 0  # 0-indexed internally
        assert result.proposal_number == 1  # 1-indexed property

    def test_proposal_endpoint_tooltip(self) -> None:
        """'Commit Proposal 5' → PROPOSAL_ENDPOINT, proposal_index=4 (0-indexed)."""
        result = ClickInfo.from_tooltip(tooltip="Commit Proposal 5")
        assert result.click_type == MapClickType.MARKER
        assert result.marker_type == MarkerType.PROPOSAL_ENDPOINT
        assert result.proposal_index == 4  # 0-indexed internally
        assert result.proposal_number == 5  # 1-indexed property

    def test_segment_tooltip(self) -> None:
        """'Segment S1' → SEGMENT, segment_id='S1'."""
        result = ClickInfo.from_tooltip(tooltip="Segment S1")
        assert result.click_type == MapClickType.MARKER
        assert result.marker_type == MarkerType.SEGMENT
        assert result.segment_id == "S1"

    def test_unknown_tooltip_raises(self) -> None:
        """Unknown tooltip raises ValueError."""
        with pytest.raises(ValueError, match="Unknown marker tooltip"):
            ClickInfo.from_tooltip(tooltip="Unknown Marker")


# =============================================================================
# DIRECT CONSTRUCTION VALIDATION
# =============================================================================


class TestDirectConstruction:
    """Tests for creating ClickInfo directly with IDs."""

    def test_terrain_click_valid(self) -> None:
        """TERRAIN click requires lat/lon."""
        click = ClickInfo(
            click_type=MapClickType.TERRAIN,
            lat=46.5,
            lon=10.5,
        )
        assert click.lat == 46.5
        assert click.lon == 10.5
        assert click.marker_type is None

    def test_terrain_click_missing_coords_raises(self) -> None:
        """TERRAIN click without lat/lon raises."""
        with pytest.raises(ValueError, match="TERRAIN click must have lat/lon"):
            ClickInfo(click_type=MapClickType.TERRAIN)

    def test_terrain_click_with_marker_type_raises(self) -> None:
        """TERRAIN click with marker_type raises."""
        with pytest.raises(ValueError, match="TERRAIN click must NOT have marker_type"):
            ClickInfo(
                click_type=MapClickType.TERRAIN,
                lat=46.5,
                lon=10.5,
                marker_type=MarkerType.NODE,
            )

    def test_node_click_valid(self) -> None:
        """NODE marker requires node_id."""
        click = ClickInfo(
            click_type=MapClickType.MARKER,
            marker_type=MarkerType.NODE,
            node_id="N1",
        )
        assert click.node_id == "N1"

    def test_node_click_missing_id_raises(self) -> None:
        """NODE marker without node_id raises."""
        with pytest.raises(ValueError, match="NODE marker must have node_id"):
            ClickInfo(
                click_type=MapClickType.MARKER,
                marker_type=MarkerType.NODE,
            )

    def test_pylon_click_valid(self) -> None:
        """PYLON marker requires both lift_id and pylon_index."""
        click = ClickInfo(
            click_type=MapClickType.MARKER,
            marker_type=MarkerType.PYLON,
            lift_id="L1",
            pylon_index=2,
        )
        assert click.lift_id == "L1"
        assert click.pylon_index == 2
        assert click.pylon_number == 3  # 1-indexed

    def test_pylon_click_missing_lift_id_raises(self) -> None:
        """PYLON marker without lift_id raises."""
        with pytest.raises(ValueError, match="PYLON marker must have lift_id and pylon_index"):
            ClickInfo(
                click_type=MapClickType.MARKER,
                marker_type=MarkerType.PYLON,
                pylon_index=2,
            )

    def test_pylon_click_missing_index_raises(self) -> None:
        """PYLON marker without pylon_index raises."""
        with pytest.raises(ValueError, match="PYLON marker must have lift_id and pylon_index"):
            ClickInfo(
                click_type=MapClickType.MARKER,
                marker_type=MarkerType.PYLON,
                lift_id="L1",
            )

    def test_marker_click_with_coords_raises(self) -> None:
        """MARKER click with lat/lon raises."""
        with pytest.raises(ValueError, match="MARKER click must NOT have lat/lon"):
            ClickInfo(
                click_type=MapClickType.MARKER,
                marker_type=MarkerType.NODE,
                node_id="N1",
                lat=46.5,
                lon=10.5,
            )


# =============================================================================
# DISPLAY NAME PROPERTY
# =============================================================================


class TestDisplayName:
    """Tests for display_name property."""

    def test_terrain_display_name(self) -> None:
        """TERRAIN shows user-friendly format with coordinates."""
        click = ClickInfo(
            click_type=MapClickType.TERRAIN,
            lat=46.5123,
            lon=10.9876,
        )
        assert click.display_name == "Map terrain at (46.5123, 10.9876)"

    def test_node_display_name(self) -> None:
        """NODE shows user-friendly junction format."""
        click = ClickInfo(
            click_type=MapClickType.MARKER,
            marker_type=MarkerType.NODE,
            node_id="N1",
        )
        assert click.display_name == "Junction N1"

    def test_slope_display_name(self) -> None:
        """SLOPE shows user-friendly slope format."""
        click = ClickInfo(
            click_type=MapClickType.MARKER,
            marker_type=MarkerType.SLOPE,
            slope_id="SL1",
        )
        assert click.display_name == "Slope SL1"

    def test_pylon_display_name(self) -> None:
        """PYLON shows user-friendly pylon format with 1-indexed number."""
        click = ClickInfo(
            click_type=MapClickType.MARKER,
            marker_type=MarkerType.PYLON,
            lift_id="L1",
            pylon_index=2,  # 0-indexed
        )
        assert click.display_name == "Pylon 3 on Lift L1"  # 1-indexed in display

    def test_proposal_endpoint_display_name(self) -> None:
        """PROPOSAL_ENDPOINT shows user-friendly path option format."""
        click = ClickInfo(
            click_type=MapClickType.MARKER,
            marker_type=MarkerType.PROPOSAL_ENDPOINT,
            proposal_index=4,  # 0-indexed
        )
        assert click.display_name == "Path option 5 (endpoint)"  # 1-indexed in display

    def test_proposal_body_display_name(self) -> None:
        """PROPOSAL_BODY shows user-friendly path option format."""
        click = ClickInfo(
            click_type=MapClickType.MARKER,
            marker_type=MarkerType.PROPOSAL_BODY,
            proposal_index=0,  # 0-indexed
        )
        assert click.display_name == "Path option 1"


# =============================================================================
# DEDUP KEY
# =============================================================================


class TestDedupKey:
    """Tests for make_dedup_key method."""

    def test_terrain_dedup_key(self) -> None:
        """TERRAIN key includes coordinates rounded to 6 decimals.

        Input with 10 decimal precision should be rounded to 6 decimals.
        46.5123456789 → "46.512346" (proper rounding, not truncation)
        """
        click = ClickInfo(
            click_type=MapClickType.TERRAIN,
            lat=46.5123456789,  # 10 decimals - tests rounding precision
            lon=10.9876543210,  # 10 decimals
        )
        # Verify proper rounding: 6789... rounds up the 5 to 6
        assert click.make_dedup_key() == "terrain_46.512346_10.987654"

    def test_node_dedup_key(self) -> None:
        """NODE key includes node_id."""
        click = ClickInfo(
            click_type=MapClickType.MARKER,
            marker_type=MarkerType.NODE,
            node_id="N1",
        )
        assert click.make_dedup_key() == "marker_node_N1"

    def test_pylon_dedup_key(self) -> None:
        """PYLON key includes 0-indexed index and lift_id."""
        click = ClickInfo(
            click_type=MapClickType.MARKER,
            marker_type=MarkerType.PYLON,
            lift_id="L1",
            pylon_index=2,  # 0-indexed
        )
        assert click.make_dedup_key() == "marker_pylon_2_L1"

    def test_proposal_endpoint_dedup_key(self) -> None:
        """PROPOSAL_ENDPOINT key includes 0-indexed index."""
        click = ClickInfo(
            click_type=MapClickType.MARKER,
            marker_type=MarkerType.PROPOSAL_ENDPOINT,
            proposal_index=4,  # 0-indexed
        )
        assert click.make_dedup_key() == "marker_proposal_end_4"


# =============================================================================
# CONVENIENCE NUMBER PROPERTIES (1-indexed)
# =============================================================================


class TestNumberProperties:
    """Tests for 1-indexed number properties."""

    def test_pylon_number_from_index(self) -> None:
        """pylon_number is pylon_index + 1."""
        click = ClickInfo(
            click_type=MapClickType.MARKER,
            marker_type=MarkerType.PYLON,
            lift_id="L1",
            pylon_index=0,  # First pylon
        )
        assert click.pylon_number == 1

    def test_proposal_number_from_index(self) -> None:
        """proposal_number is proposal_index + 1."""
        click = ClickInfo(
            click_type=MapClickType.MARKER,
            marker_type=MarkerType.PROPOSAL_BODY,
            proposal_index=4,  # Fifth proposal
        )
        assert click.proposal_number == 5

    def test_pylon_number_none_for_non_pylon(self) -> None:
        """Non-PYLON markers have None pylon_number."""
        click = ClickInfo(
            click_type=MapClickType.MARKER,
            marker_type=MarkerType.LIFT,
            lift_id="L1",
        )
        assert click.pylon_number is None

    def test_proposal_number_none_for_non_proposal(self) -> None:
        """Non-PROPOSAL markers have None proposal_number."""
        click = ClickInfo(
            click_type=MapClickType.MARKER,
            marker_type=MarkerType.NODE,
            node_id="N1",
        )
        assert click.proposal_number is None


# =============================================================================
# ID FIELD ACCESS FOR NON-MATCHING MARKERS
# =============================================================================


class TestIdFieldAccess:
    """Tests that ID fields return None for non-matching marker types."""

    def test_node_id_none_for_terrain(self) -> None:
        """TERRAIN clicks have no node_id."""
        click = ClickInfo(
            click_type=MapClickType.TERRAIN,
            lat=46.5,
            lon=10.5,
        )
        assert click.node_id is None

    def test_slope_id_none_for_node(self) -> None:
        """NODE markers have no slope_id."""
        click = ClickInfo(
            click_type=MapClickType.MARKER,
            marker_type=MarkerType.NODE,
            node_id="N1",
        )
        assert click.slope_id is None

    def test_lift_id_none_for_slope(self) -> None:
        """SLOPE markers have no lift_id."""
        click = ClickInfo(
            click_type=MapClickType.MARKER,
            marker_type=MarkerType.SLOPE,
            slope_id="SL1",
        )
        assert click.lift_id is None

    def test_segment_id_none_for_node(self) -> None:
        """NODE markers have no segment_id."""
        click = ClickInfo(
            click_type=MapClickType.MARKER,
            marker_type=MarkerType.NODE,
            node_id="N1",
        )
        assert click.segment_id is None

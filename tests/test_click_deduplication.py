"""Tests for simplified click deduplication.

With returned_objects limited to click fields only, pan/zoom don't trigger
reruns. Simple coordinate tracking prevents re-processing the same click.
"""

import pytest

from skiresort_planner.ui.state_machine import (
    ClickDeduplicationContext,
    ClickDetectionResult,
)


@pytest.fixture
def dedup() -> ClickDeduplicationContext:
    """Fresh dedup context for each test."""
    return ClickDeduplicationContext()


MARKER_A = {"lat": 46.990, "lng": 10.320}
MARKER_B = {"lat": 46.988, "lng": 10.318}
TOOLTIP_A = "View Slope A"
TOOLTIP_B = "View Lift B"
TERRAIN_A = {"lat": 46.980, "lng": 10.315}
TERRAIN_B = {"lat": 46.978, "lng": 10.312}


class TestClickDetectionResult:
    """Test the ClickDetectionResult dataclass."""

    def test_no_click_factory(self) -> None:
        """no_click() returns result with all None values."""
        result = ClickDetectionResult.no_click()
        assert result.click_type is None
        assert result.data is None
        assert result.tooltip is None
        assert not result.is_valid

    def test_is_valid_for_marker(self) -> None:
        """is_valid is True for marker clicks."""
        result = ClickDetectionResult(click_type="marker", data=MARKER_A, tooltip=TOOLTIP_A)
        assert result.is_valid
        assert result.is_marker
        assert not result.is_terrain

    def test_is_valid_for_terrain(self) -> None:
        """is_valid is True for terrain clicks."""
        result = ClickDetectionResult(click_type="terrain", data=TERRAIN_A, tooltip=None)
        assert result.is_valid
        assert result.is_terrain
        assert not result.is_marker


class TestFirstClicks:
    """Test that first clicks are always accepted."""

    def test_first_marker_click_accepted(self, dedup: ClickDeduplicationContext) -> None:
        """First marker click should be accepted."""
        result = dedup.detect_new_click(
            marker_data=MARKER_A,
            marker_tooltip=TOOLTIP_A,
            terrain_data=None,
        )
        assert result.click_type == "marker"
        assert result.data == MARKER_A
        assert result.tooltip == TOOLTIP_A

    def test_first_terrain_click_accepted(self, dedup: ClickDeduplicationContext) -> None:
        """First terrain click should be accepted."""
        result = dedup.detect_new_click(
            marker_data=None,
            marker_tooltip="",
            terrain_data=TERRAIN_A,
        )
        assert result.click_type == "terrain"
        assert result.data == TERRAIN_A

    def test_no_click_returns_none(self, dedup: ClickDeduplicationContext) -> None:
        """No click data returns None."""
        result = dedup.detect_new_click(
            marker_data=None,
            marker_tooltip="",
            terrain_data=None,
        )
        assert not result.is_valid


class TestNewCoordinates:
    """Test that new coordinates are accepted."""

    def test_marker_new_coords_accepted(self, dedup: ClickDeduplicationContext) -> None:
        """New marker coordinates should be accepted."""
        dedup.detect_new_click(marker_data=MARKER_A, marker_tooltip=TOOLTIP_A, terrain_data=None)
        result = dedup.detect_new_click(marker_data=MARKER_B, marker_tooltip=TOOLTIP_B, terrain_data=None)
        assert result.click_type == "marker"
        assert result.data == MARKER_B

    def test_terrain_new_coords_accepted(self, dedup: ClickDeduplicationContext) -> None:
        """New terrain coordinates should be accepted."""
        dedup.detect_new_click(marker_data=None, marker_tooltip="", terrain_data=TERRAIN_A)
        result = dedup.detect_new_click(marker_data=None, marker_tooltip="", terrain_data=TERRAIN_B)
        assert result.click_type == "terrain"
        assert result.data == TERRAIN_B


class TestSameCoordinatesRejected:
    """Test that same coordinates are rejected."""

    def test_same_marker_rejected(self, dedup: ClickDeduplicationContext) -> None:
        """Same marker coordinates should be rejected."""
        dedup.detect_new_click(marker_data=MARKER_A, marker_tooltip=TOOLTIP_A, terrain_data=None)
        result = dedup.detect_new_click(marker_data=MARKER_A, marker_tooltip=TOOLTIP_A, terrain_data=None)
        assert not result.is_valid

    def test_same_terrain_rejected(self, dedup: ClickDeduplicationContext) -> None:
        """Same terrain coordinates should be rejected."""
        dedup.detect_new_click(marker_data=None, marker_tooltip="", terrain_data=TERRAIN_A)
        result = dedup.detect_new_click(marker_data=None, marker_tooltip="", terrain_data=TERRAIN_A)
        assert not result.is_valid


class TestSeparateStreams:
    """Test that marker and terrain are separate streams."""

    def test_terrain_after_marker_accepted(self, dedup: ClickDeduplicationContext) -> None:
        """Terrain click after marker click should be accepted."""
        dedup.detect_new_click(marker_data=MARKER_A, marker_tooltip=TOOLTIP_A, terrain_data=None)
        result = dedup.detect_new_click(marker_data=MARKER_A, marker_tooltip=TOOLTIP_A, terrain_data=TERRAIN_A)
        assert result.click_type == "terrain"

    def test_marker_after_terrain_accepted(self, dedup: ClickDeduplicationContext) -> None:
        """Marker click after terrain click should be accepted."""
        dedup.detect_new_click(marker_data=None, marker_tooltip="", terrain_data=TERRAIN_A)
        result = dedup.detect_new_click(marker_data=MARKER_A, marker_tooltip=TOOLTIP_A, terrain_data=TERRAIN_A)
        assert result.click_type == "marker"


class TestMarkerPriority:
    """Test that marker takes priority over terrain when both have new coords."""

    def test_marker_wins_on_first_click(self, dedup: ClickDeduplicationContext) -> None:
        """When both streams have data on first click, marker wins."""
        result = dedup.detect_new_click(
            marker_data=MARKER_A,
            marker_tooltip=TOOLTIP_A,
            terrain_data=TERRAIN_A,
        )
        assert result.click_type == "marker"

    def test_marker_click_updates_terrain_stream(self, dedup: ClickDeduplicationContext) -> None:
        """Marker click should also update terrain stream to prevent ghost."""
        dedup.detect_new_click(
            marker_data=MARKER_A,
            marker_tooltip=TOOLTIP_A,
            terrain_data=TERRAIN_A,
        )
        # Same terrain should now be rejected (was updated by marker click)
        result = dedup.detect_new_click(
            marker_data=MARKER_A,
            marker_tooltip=TOOLTIP_A,
            terrain_data=TERRAIN_A,
        )
        assert not result.is_valid


class TestClear:
    """Test clear functionality."""

    def test_clear_allows_same_click(self, dedup: ClickDeduplicationContext) -> None:
        """After clear, same coordinates should be accepted again."""
        dedup.detect_new_click(marker_data=MARKER_A, marker_tooltip=TOOLTIP_A, terrain_data=None)
        dedup.clear()
        result = dedup.detect_new_click(marker_data=MARKER_A, marker_tooltip=TOOLTIP_A, terrain_data=None)
        assert result.click_type == "marker"

    def test_clear_resets_both_streams(self, dedup: ClickDeduplicationContext) -> None:
        """Clear should reset both marker and terrain tracking."""
        dedup.detect_new_click(marker_data=MARKER_A, marker_tooltip=TOOLTIP_A, terrain_data=TERRAIN_A)
        dedup.clear()
        assert dedup.last_marker_data is None
        assert dedup.last_terrain_data is None

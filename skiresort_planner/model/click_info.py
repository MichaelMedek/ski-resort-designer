"""Click detection types - unified click information for map interactions.

This module defines the canonical types for ALL click detection:
- MapClickType: Source of click (MARKER or TERRAIN)
- MarkerType: Type of marker clicked (or None for terrain)
- ClickInfo: Unified click information returned by ClickDetector

ID Storage:
    IDs are stored directly (node_id="N1", slope_id="SL1", etc.)
    Display names are generated via the display_name property
    No intermediate "element_id" - IDs are the source of truth

Index Convention:
    pylon_index and proposal_index are 0-indexed internally
    Display functions add 1 for user-facing text (1-indexed)

STRICT: All click detection flows through ClickInfo. Any deviation is a bug.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from skiresort_planner.constants import CoordinateConfig


class MapClickType(Enum):
    """Source of click on the map - EXACTLY one per interaction."""

    MARKER = "marker"  # Clicked on a marker (has tooltip)
    TERRAIN = "terrain"  # Clicked on empty map (raw coordinates)


class MarkerType(Enum):
    """Type of marker clicked. None for terrain clicks."""

    NODE = "node"
    SLOPE = "slope"
    SEGMENT = "segment"
    LIFT = "lift"
    PYLON = "pylon"
    PROPOSAL_ENDPOINT = "proposal_endpoint"
    PROPOSAL_BODY = "proposal_body"


@dataclass(frozen=True)
class ClickInfo:
    """Unified click information - the ONLY output from click detection.

    STRICT CONTRACT:
    - click_type is ALWAYS set (MARKER or TERRAIN)
    - For TERRAIN: lat/lon are REQUIRED
    - For MARKER: lat/lon are None (position not needed)
    - marker_type is set IFF click_type == MARKER
    - Exactly ONE ID field is set for each marker type

    ID Fields (exactly one set per marker type):
        node_id: "N1" for NODE markers
        slope_id: "SL1" for SLOPE markers
        segment_id: "S1" for SEGMENT markers
        lift_id: "L1" for LIFT markers, also set for PYLON markers
        pylon_index: 0-indexed for PYLON markers (display as 1-indexed)
        proposal_index: 0-indexed for PROPOSAL_* markers (display as 1-indexed)
    """

    click_type: MapClickType
    lat: Optional[float] = None
    lon: Optional[float] = None
    marker_type: Optional[MarkerType] = None

    # Direct ID storage - exactly ONE set for each marker type
    node_id: Optional[str] = None  # "N1" for NODE
    slope_id: Optional[str] = None  # "SL1" for SLOPE
    segment_id: Optional[str] = None  # "S1" for SEGMENT
    lift_id: Optional[str] = None  # "L1" for LIFT and PYLON
    pylon_index: Optional[int] = None  # 0-indexed (PYLON only)
    proposal_index: Optional[int] = None  # 0-indexed (PROPOSAL_* only)

    def __post_init__(self) -> None:
        """Validate invariants - STRICT: fail immediately on invalid state."""
        if self.click_type == MapClickType.TERRAIN:
            if self.lat is None or self.lon is None:
                raise ValueError("TERRAIN click must have lat/lon set")
            if self.marker_type is not None:
                raise ValueError("TERRAIN click must NOT have marker_type set")
        elif self.click_type == MapClickType.MARKER:
            if self.marker_type is None:
                raise ValueError("MARKER click must have marker_type set")
            if self.lat is not None or self.lon is not None:
                raise ValueError("MARKER click must NOT have lat/lon set")
            self._validate_marker_ids()
        else:
            raise RuntimeError(f"Unknown click_type: {self.click_type}")

    def _validate_marker_ids(self) -> None:
        """Validate that exactly the right ID fields are set for marker type."""
        mt = self.marker_type

        if mt == MarkerType.NODE:
            if self.node_id is None:
                raise ValueError("NODE marker must have node_id set")
        elif mt == MarkerType.SLOPE:
            if self.slope_id is None:
                raise ValueError("SLOPE marker must have slope_id set")
        elif mt == MarkerType.SEGMENT:
            if self.segment_id is None:
                raise ValueError("SEGMENT marker must have segment_id set")
        elif mt == MarkerType.LIFT:
            if self.lift_id is None:
                raise ValueError("LIFT marker must have lift_id set")
        elif mt == MarkerType.PYLON:
            if self.lift_id is None or self.pylon_index is None:
                raise ValueError("PYLON marker must have lift_id and pylon_index set")
        elif mt == MarkerType.PROPOSAL_ENDPOINT:
            if self.proposal_index is None:
                raise ValueError("PROPOSAL_ENDPOINT marker must have proposal_index set")
        elif mt == MarkerType.PROPOSAL_BODY:
            if self.proposal_index is None:
                raise ValueError("PROPOSAL_BODY marker must have proposal_index set")
        else:
            raise RuntimeError(f"Unknown marker_type: {mt}")

    # =========================================================================
    # DISPLAY PROPERTIES
    # =========================================================================

    @property
    def display_name(self) -> str:
        """Human-readable name for UI display and logging.

        User-friendly format:
            Map terrain at (46.5123, 10.9876)
            Junction N1
            Slope SL1
            Lift L1
            Pylon 3 on Lift L1
            Path option 5 (endpoint)
            Path option 5
            Segment S1
        """
        if self.click_type == MapClickType.TERRAIN:
            return f"Map terrain at ({self.lat:.4f}, {self.lon:.4f})"

        if self.click_type == MapClickType.MARKER:
            mt = self.marker_type
            if mt == MarkerType.NODE:
                return f"Junction {self.node_id}"
            elif mt == MarkerType.SLOPE:
                return f"Slope {self.slope_id}"
            elif mt == MarkerType.SEGMENT:
                return f"Segment {self.segment_id}"
            elif mt == MarkerType.LIFT:
                return f"Lift {self.lift_id}"
            elif mt == MarkerType.PYLON:
                assert self.pylon_index is not None
                return f"Pylon {self.pylon_index + 1} on Lift {self.lift_id}"
            elif mt == MarkerType.PROPOSAL_ENDPOINT:
                assert self.proposal_index is not None
                return f"Path option {self.proposal_index + 1} (endpoint)"
            elif mt == MarkerType.PROPOSAL_BODY:
                assert self.proposal_index is not None
                return f"Path option {self.proposal_index + 1}"
            else:
                raise RuntimeError(f"Unknown marker_type: {mt}")

        raise RuntimeError(f"Unknown click_type: {self.click_type}")

    @staticmethod
    def round_for_key(value: float) -> str:
        """Round coordinate to string for dedup key."""
        return f"{value:.{CoordinateConfig.DEDUP_KEY_DECIMALS}f}"

    def make_dedup_key(self) -> str:
        """Generate deduplication key for click tracking.

        Key format by type:
            TERRAIN: "terrain_{lat:.6f}_{lon:.6f}"
            NODE: "marker_node_N1"
            SLOPE: "marker_slope_SL1"
            SEGMENT: "marker_segment_S1"
            LIFT: "marker_lift_L1"
            PYLON: "marker_pylon_2_L1" (0-indexed)
            PROPOSAL_ENDPOINT: "marker_proposal_end_4" (0-indexed)
            PROPOSAL_BODY: "marker_proposal_body_4" (0-indexed)

        Returns:
            Unique string key for deduplication
        """
        if self.click_type == MapClickType.TERRAIN:
            assert self.lat is not None and self.lon is not None
            lat_key = self.round_for_key(self.lat)
            lon_key = self.round_for_key(self.lon)
            return f"terrain_{lat_key}_{lon_key}"

        if self.click_type == MapClickType.MARKER:
            mt = self.marker_type
            if mt == MarkerType.NODE:
                return f"marker_node_{self.node_id}"
            elif mt == MarkerType.SLOPE:
                return f"marker_slope_{self.slope_id}"
            elif mt == MarkerType.SEGMENT:
                return f"marker_segment_{self.segment_id}"
            elif mt == MarkerType.LIFT:
                return f"marker_lift_{self.lift_id}"
            elif mt == MarkerType.PYLON:
                return f"marker_pylon_{self.pylon_index}_{self.lift_id}"
            elif mt == MarkerType.PROPOSAL_ENDPOINT:
                return f"marker_proposal_end_{self.proposal_index}"
            elif mt == MarkerType.PROPOSAL_BODY:
                return f"marker_proposal_body_{self.proposal_index}"
            else:
                raise RuntimeError(f"Unknown marker_type: {mt}")

        raise RuntimeError(f"Unknown click_type: {self.click_type}")

    # =========================================================================
    # CONVENIENCE PROPERTIES
    # =========================================================================

    @property
    def proposal_number(self) -> Optional[int]:
        """Proposal number as 1-indexed for display. Returns None if not a proposal."""
        if self.proposal_index is not None:
            return self.proposal_index + 1
        return None

    @property
    def pylon_number(self) -> Optional[int]:
        """Pylon number as 1-indexed for display. Returns None if not a pylon."""
        if self.pylon_index is not None:
            return self.pylon_index + 1
        return None

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
    ROAD = "road"
    PYLON = "pylon"
    PROPOSAL_ENDPOINT = "proposal_endpoint"
    PROPOSAL_BODY = "proposal_body"
    IMPORT_CENTER = "import_center"


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
    lat: float | None = None
    lon: float | None = None
    marker_type: MarkerType | None = None

    # Direct ID storage - exactly ONE set for each marker type
    node_id: str | None = None  # "N1" for NODE
    slope_id: str | None = None  # "SL1" for SLOPE
    segment_id: str | None = None  # "S1" for SEGMENT
    lift_id: str | None = None  # "L1" for LIFT and PYLON
    road_id: str | None = None  # "R1" for ROAD
    pylon_index: int | None = None  # 0-indexed (PYLON only)
    proposal_index: int | None = None  # 0-indexed (PROPOSAL_* only)

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
        match self.marker_type:
            case MarkerType.NODE:
                if self.node_id is None:
                    raise ValueError("NODE marker must have node_id set")
            case MarkerType.SLOPE:
                if self.slope_id is None:
                    raise ValueError("SLOPE marker must have slope_id set")
            case MarkerType.SEGMENT:
                if self.segment_id is None:
                    raise ValueError("SEGMENT marker must have segment_id set")
            case MarkerType.LIFT:
                if self.lift_id is None:
                    raise ValueError("LIFT marker must have lift_id set")
            case MarkerType.ROAD:
                if self.road_id is None:
                    raise ValueError("ROAD marker must have road_id set")
            case MarkerType.PYLON:
                if self.lift_id is None or self.pylon_index is None:
                    raise ValueError("PYLON marker must have lift_id and pylon_index set")
            case MarkerType.PROPOSAL_ENDPOINT:
                if self.proposal_index is None:
                    raise ValueError("PROPOSAL_ENDPOINT marker must have proposal_index set")
            case MarkerType.PROPOSAL_BODY:
                if self.proposal_index is None:
                    raise ValueError("PROPOSAL_BODY marker must have proposal_index set")
            case MarkerType.IMPORT_CENTER:
                pass  # positionless confirm marker — carries no id
            case _:
                raise RuntimeError(f"Unknown marker_type: {self.marker_type}")

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
            match self.marker_type:
                case MarkerType.NODE:
                    return f"Junction {self.node_id}"
                case MarkerType.SLOPE:
                    return f"Slope {self.slope_id}"
                case MarkerType.SEGMENT:
                    return f"Segment {self.segment_id}"
                case MarkerType.LIFT:
                    return f"Lift {self.lift_id}"
                case MarkerType.ROAD:
                    return f"Road {self.road_id}"
                case MarkerType.PYLON:
                    assert self.pylon_index is not None
                    return f"Pylon {self.pylon_index + 1} on Lift {self.lift_id}"
                case MarkerType.PROPOSAL_ENDPOINT:
                    assert self.proposal_index is not None
                    return f"Path option {self.proposal_index + 1} (endpoint)"
                case MarkerType.PROPOSAL_BODY:
                    assert self.proposal_index is not None
                    return f"Path option {self.proposal_index + 1}"
                case MarkerType.IMPORT_CENTER:
                    return "Import area center"
                case _:
                    raise RuntimeError(f"Unknown marker_type: {self.marker_type}")

        raise RuntimeError(f"Unknown click_type: {self.click_type}")

    # =========================================================================
    # CONVENIENCE PROPERTIES
    # =========================================================================

    @property
    def proposal_number(self) -> int | None:
        """Proposal number as 1-indexed for display. Returns None if not a proposal."""
        if self.proposal_index is not None:
            return self.proposal_index + 1
        return None

    @property
    def pylon_number(self) -> int | None:
        """Pylon number as 1-indexed for display. Returns None if not a pylon."""
        if self.pylon_index is not None:
            return self.pylon_index + 1
        return None

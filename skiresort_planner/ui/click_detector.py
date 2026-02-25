"""Click detector - detects map clicks from Pydeck events.

Pydeck click events return picked object data directly, unlike Folium's
tooltip-based detection. Objects contain type and ID fields for identification.

Coordinate tracking prevents re-processing the same click on reruns.
"""

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from skiresort_planner.constants import ClickConfig
from skiresort_planner.model.click_info import ClickInfo, MapClickType, MarkerType

if TYPE_CHECKING:
    from skiresort_planner.ui.state_machine import ClickDeduplicationContext

logger = logging.getLogger(__name__)


@dataclass
class ClickDetector:
    """Detects clicks from Pydeck picked objects.

    Attributes:
        dedup: ClickDeduplicationContext for tracking last-seen clicks
    """

    dedup: "ClickDeduplicationContext"

    def detect(
        self,
        clicked_object: dict[str, Any] | None,
        clicked_coordinate: list[float] | None,
    ) -> ClickInfo | None:
        """Detect click from Pydeck event data.

        Args:
            clicked_object: The picked deck.gl object data (dict) or None
            clicked_coordinate: [lon, lat] of click location or None

        Returns:
            ClickInfo for new clicks, None otherwise
        """
        # Check for deduplication
        obj_id = self._get_object_id(obj=clicked_object)
        coord_tuple = tuple(clicked_coordinate) if clicked_coordinate else None

        if not self.dedup.is_new_click(coord=coord_tuple, obj_id=obj_id):
            return None

        # Object click (marker, segment, etc.)
        if clicked_object is not None:
            return self._parse_object_click(obj=clicked_object)

        # Terrain click (no object picked)
        if clicked_coordinate is not None:
            lon, lat = clicked_coordinate[0], clicked_coordinate[1]
            logger.debug(f"Terrain click at ({lat:.6f}, {lon:.6f})")
            return ClickInfo(
                click_type=MapClickType.TERRAIN,
                lat=lat,
                lon=lon,
            )

        return None

    def _get_object_id(self, obj: dict[str, Any] | None) -> str | None:
        """Generate unique ID for object for deduplication."""
        if obj is None:
            return None

        obj_type = obj.get("type", "")
        obj_id = obj.get("id", "")

        if obj_type == ClickConfig.TYPE_PYLON:
            lift_id = obj.get("lift_id", "")
            pylon_idx = obj.get("pylon_index", 0)
            return f"pylon_{lift_id}_{pylon_idx}"

        if obj_type in {ClickConfig.TYPE_PROPOSAL_BODY, ClickConfig.TYPE_PROPOSAL_ENDPOINT}:
            # Include map_version to make proposal IDs unique per generation
            # This ensures clicks work after proposals are regenerated
            import streamlit as st

            map_version = st.session_state.get("map_version", 0)
            proposal_idx = obj.get("proposal_index", 0)
            return f"{obj_type}_{proposal_idx}_v{map_version}"

        return f"{obj_type}_{obj_id}" if obj_id else obj_type

    def _parse_object_click(self, obj: dict[str, Any]) -> ClickInfo | None:
        """Parse clicked object to ClickInfo."""
        obj_type = obj.get("type")

        if not obj_type:
            logger.warning(f"Object click without type field: {obj}")
            return None

        # GeoJSON Feature: extract type from properties (for segment belts, etc.)
        if obj_type == "Feature":
            props = obj.get("properties", {})
            obj_type = props.get("type")
            if not obj_type:
                logger.debug(f"GeoJSON Feature without properties.type: {obj}")
                return None
            # Merge properties into obj for easier access
            obj = {**obj, **props}

        logger.debug(f"Object click: type={obj_type}, data={obj}")

        # TERRAIN click (invisible ScatterplotLayer grid for terrain click detection)
        if obj_type == ClickConfig.TYPE_TERRAIN:
            # ScatterplotLayer points have direct lon/lat fields
            lon = obj.get("lon")
            lat = obj.get("lat")
            if lon is None or lat is None:
                logger.debug(f"Terrain click missing lon/lat: {obj}")
                return None
            logger.debug(f"Terrain click at ({lat:.6f}, {lon:.6f})")
            return ClickInfo(
                click_type=MapClickType.TERRAIN,
                lat=lat,
                lon=lon,
            )

        # NODE click
        if obj_type == ClickConfig.TYPE_NODE:
            node_id = obj.get("id")
            if not node_id:
                logger.warning("Node click missing id")
                return None
            return ClickInfo(
                click_type=MapClickType.MARKER,
                marker_type=MarkerType.NODE,
                node_id=node_id,
            )

        # SEGMENT click
        if obj_type == ClickConfig.TYPE_SEGMENT:
            seg_id = obj.get("id")
            if not seg_id:
                logger.warning("Segment click missing id")
                return None
            return ClickInfo(
                click_type=MapClickType.MARKER,
                marker_type=MarkerType.SEGMENT,
                segment_id=seg_id,
            )

        # SLOPE click (icon marker)
        if obj_type == ClickConfig.TYPE_SLOPE:
            slope_id = obj.get("id")
            if not slope_id:
                logger.warning("Slope click missing id")
                return None
            return ClickInfo(
                click_type=MapClickType.MARKER,
                marker_type=MarkerType.SLOPE,
                slope_id=slope_id,
            )

        # LIFT click
        if obj_type == ClickConfig.TYPE_LIFT:
            lift_id = obj.get("id")
            if not lift_id:
                logger.warning("Lift click missing id")
                return None
            return ClickInfo(
                click_type=MapClickType.MARKER,
                marker_type=MarkerType.LIFT,
                lift_id=lift_id,
            )

        # PYLON click
        if obj_type == ClickConfig.TYPE_PYLON:
            lift_id = obj.get("lift_id")
            pylon_index = obj.get("pylon_index")
            if not lift_id or pylon_index is None:
                logger.warning(f"Pylon click missing lift_id or pylon_index: {obj}")
                return None
            return ClickInfo(
                click_type=MapClickType.MARKER,
                marker_type=MarkerType.PYLON,
                lift_id=lift_id,
                pylon_index=pylon_index,  # Already 0-indexed
            )

        # PROPOSAL ENDPOINT click (commit)
        if obj_type == ClickConfig.TYPE_PROPOSAL_ENDPOINT:
            proposal_index = obj.get("proposal_index")
            if proposal_index is None:
                logger.warning("Proposal endpoint click missing proposal_index")
                return None
            return ClickInfo(
                click_type=MapClickType.MARKER,
                marker_type=MarkerType.PROPOSAL_ENDPOINT,
                proposal_index=proposal_index,  # Already 0-indexed
            )

        # PROPOSAL BODY click (select)
        if obj_type == ClickConfig.TYPE_PROPOSAL_BODY:
            proposal_index = obj.get("proposal_index")
            if proposal_index is None:
                logger.warning("Proposal body click missing proposal_index")
                return None
            return ClickInfo(
                click_type=MapClickType.MARKER,
                marker_type=MarkerType.PROPOSAL_BODY,
                proposal_index=proposal_index,  # Already 0-indexed
            )

        # Unknown type - log and ignore
        logger.warning(f"Unknown object type: {obj_type}")
        return None

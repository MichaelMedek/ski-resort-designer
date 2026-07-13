"""Click detector - detects map clicks from Pydeck events.

Pydeck click events return picked object data directly, unlike Folium's
tooltip-based detection. Objects contain type and ID fields for identification.

Coordinate tracking prevents re-processing the same click on reruns.
"""

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import streamlit as st

from skiresort_planner.constants import ClickConfig
from skiresort_planner.model.click_info import ClickInfo, MapClickType, MarkerType

if TYPE_CHECKING:
    from skiresort_planner.ui.context import ClickDeduplicationContext

logger = logging.getLogger(__name__)

# Single source of truth guard: every MarkerType value must have a matching
# ClickConfig.TYPE_* tag. If the two drift, mapping silently breaks — fail loudly at import.
_CLICK_TYPE_TAGS = {v for k, v in vars(ClickConfig).items() if k.startswith("TYPE_")}
assert {m.value for m in MarkerType} <= _CLICK_TYPE_TAGS, (
    "MarkerType values must all have a matching ClickConfig.TYPE_* tag"
)


def _as_str(value: object) -> str | None:
    """Narrow a picked-object field to str, or None if absent/wrong type."""
    return value if isinstance(value, str) else None


def _as_int(value: object) -> int | None:
    """Narrow a picked-object field to int, or None if absent/wrong type."""
    return value if isinstance(value, int) else None


def _as_float(value: object) -> float | None:
    """Narrow a picked-object numeric field to float, or None if absent/wrong type."""
    return float(value) if isinstance(value, (int, float)) else None


@dataclass
class ClickDetector:
    """Detects clicks from Pydeck picked objects.

    Attributes:
        dedup: ClickDeduplicationContext for tracking last-seen clicks
    """

    dedup: "ClickDeduplicationContext"

    def detect(
        self,
        clicked_object: dict[str, object] | None,
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

    def _get_object_id(self, obj: dict[str, object] | None) -> str | None:
        """Generate unique ID for object for deduplication."""
        if obj is None:
            return None

        obj_type = _as_str(obj.get("type")) or ""
        obj_id = _as_str(obj.get("id")) or ""

        if obj_type == ClickConfig.TYPE_PYLON:
            lift_id = _as_str(obj.get("lift_id")) or ""
            pylon_idx = _as_int(obj.get("pylon_index")) or 0
            return f"pylon_{lift_id}_{pylon_idx}"

        if obj_type in {ClickConfig.TYPE_PROPOSAL_BODY, ClickConfig.TYPE_PROPOSAL_ENDPOINT}:
            # Include map_version to make proposal IDs unique per generation
            # This ensures clicks work after proposals are regenerated
            map_version = st.session_state.get("map_version", 0)
            proposal_idx = _as_int(obj.get("proposal_index")) or 0
            return f"{obj_type}_{proposal_idx}_v{map_version}"

        return f"{obj_type}_{obj_id}" if obj_id else obj_type

    def _parse_object_click(self, obj: dict[str, object]) -> ClickInfo | None:
        """Parse clicked object to ClickInfo."""
        obj_type = _as_str(obj.get("type"))

        if not obj_type:
            logger.warning(f"Object click without type field: {obj}")
            return None

        # GeoJSON Feature: extract type from properties (for segment belts, etc.)
        if obj_type == "Feature":
            props = obj.get("properties", {})
            if not isinstance(props, dict):
                logger.debug(f"GeoJSON Feature with non-dict properties: {obj}")
                return None
            obj_type = _as_str(props.get("type"))
            if not obj_type:
                logger.debug(f"GeoJSON Feature without properties.type: {obj}")
                return None
            # Merge properties into obj for easier access
            obj = {**obj, **props}

        logger.debug(f"Object click: type={obj_type}, data={obj}")

        # TERRAIN click (invisible ScatterplotLayer grid for terrain click detection)
        if obj_type == ClickConfig.TYPE_TERRAIN:
            # ScatterplotLayer points have direct lon/lat fields
            lon = _as_float(obj.get("lon"))
            lat = _as_float(obj.get("lat"))
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
            node_id = _as_str(obj.get("id"))
            assert node_id, f"node marker must carry an id (rendering bug): {obj}"
            return ClickInfo(
                click_type=MapClickType.MARKER,
                marker_type=MarkerType.NODE,
                node_id=node_id,
            )

        # SEGMENT click
        if obj_type == ClickConfig.TYPE_SEGMENT:
            seg_id = _as_str(obj.get("id"))
            assert seg_id, f"segment marker must carry an id (rendering bug): {obj}"
            return ClickInfo(
                click_type=MapClickType.MARKER,
                marker_type=MarkerType.SEGMENT,
                segment_id=seg_id,
            )

        # SLOPE click (icon marker)
        if obj_type == ClickConfig.TYPE_SLOPE:
            slope_id = _as_str(obj.get("id"))
            assert slope_id, f"slope marker must carry an id (rendering bug): {obj}"
            return ClickInfo(
                click_type=MapClickType.MARKER,
                marker_type=MarkerType.SLOPE,
                slope_id=slope_id,
            )

        # LIFT click
        if obj_type == ClickConfig.TYPE_LIFT:
            lift_id = _as_str(obj.get("id"))
            assert lift_id, f"lift marker must carry an id (rendering bug): {obj}"
            return ClickInfo(
                click_type=MapClickType.MARKER,
                marker_type=MarkerType.LIFT,
                lift_id=lift_id,
            )

        # ROAD click
        if obj_type == ClickConfig.TYPE_ROAD:
            road_id = _as_str(obj.get("id"))
            assert road_id, f"road marker must carry an id (rendering bug): {obj}"
            return ClickInfo(
                click_type=MapClickType.MARKER,
                marker_type=MarkerType.ROAD,
                road_id=road_id,
            )

        # PYLON click
        if obj_type == ClickConfig.TYPE_PYLON:
            lift_id = _as_str(obj.get("lift_id"))
            pylon_index = _as_int(obj.get("pylon_index"))
            assert lift_id and pylon_index is not None, f"pylon marker must carry lift_id + pylon_index: {obj}"
            return ClickInfo(
                click_type=MapClickType.MARKER,
                marker_type=MarkerType.PYLON,
                lift_id=lift_id,
                pylon_index=pylon_index,  # Already 0-indexed
            )

        # PROPOSAL ENDPOINT click (commit)
        if obj_type == ClickConfig.TYPE_PROPOSAL_ENDPOINT:
            proposal_index = _as_int(obj.get("proposal_index"))
            assert proposal_index is not None, f"proposal endpoint must carry proposal_index: {obj}"
            return ClickInfo(
                click_type=MapClickType.MARKER,
                marker_type=MarkerType.PROPOSAL_ENDPOINT,
                proposal_index=proposal_index,  # Already 0-indexed
            )

        # PROPOSAL BODY click (select)
        if obj_type == ClickConfig.TYPE_PROPOSAL_BODY:
            proposal_index = _as_int(obj.get("proposal_index"))
            assert proposal_index is not None, f"proposal body must carry proposal_index: {obj}"
            return ClickInfo(
                click_type=MapClickType.MARKER,
                marker_type=MarkerType.PROPOSAL_BODY,
                proposal_index=proposal_index,  # Already 0-indexed
            )

        # IMPORT CENTER click (confirm the placed import box). Carries no id — it is a positionless
        # confirm signal — so unlike the id-bearing markers above there is nothing to assert here.
        if obj_type == ClickConfig.TYPE_IMPORT_CENTER:
            return ClickInfo(
                click_type=MapClickType.MARKER,
                marker_type=MarkerType.IMPORT_CENTER,
            )

        # Unknown type — pydeck can pick unlabeled/basemap objects; ignore them.
        logger.warning(f"Unknown object type: {obj_type}")
        return None

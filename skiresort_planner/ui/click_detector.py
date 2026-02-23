"""Click detector - detects map clicks from st_folium output.

With returned_objects limited to click fields only, pan/zoom don't trigger
reruns. Simple coordinate tracking prevents re-processing the same click.
"""

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from skiresort_planner.ui.state_machine import ClickDeduplicationContext

from skiresort_planner.model.click_info import ClickInfo, MapClickType

logger = logging.getLogger(__name__)


@dataclass
class ClickDetector:
    """Detects map clicks from st_folium output.

    Attributes:
        dedup: ClickDeduplicationContext for tracking last-seen coordinates
    """

    dedup: "ClickDeduplicationContext"

    def detect(self, map_data: Optional[dict]) -> Optional[ClickInfo]:
        """Detect click from st_folium output.

        Args:
            map_data: Dictionary returned by st_folium

        Returns:
            ClickInfo for new clicks, None otherwise
        """
        if not map_data:
            return None

        marker_data = map_data.get("last_object_clicked")
        marker_tooltip = map_data.get("last_object_clicked_tooltip")
        terrain_data = map_data.get("last_clicked")

        result = self.dedup.detect_new_click(
            marker_data=marker_data,
            marker_tooltip=marker_tooltip,
            terrain_data=terrain_data,
        )

        if result.is_marker:
            # All our markers have tooltips - empty tooltip means ghost click
            if not result.tooltip:
                logger.warning(f"Marker click with empty tooltip, ignoring as likely ghost click: {result}")
                return None
            return ClickInfo.from_tooltip(tooltip=result.tooltip)
        elif result.is_terrain:
            return self._create_terrain_click_info(terrain_data=result.data)
        return None

    def _create_terrain_click_info(self, terrain_data: Optional[dict]) -> ClickInfo:
        """Create ClickInfo from terrain click coordinates."""
        if terrain_data is None or "lat" not in terrain_data or "lng" not in terrain_data:
            raise RuntimeError(f"last_clicked missing lat/lng: {terrain_data}")

        return ClickInfo(
            click_type=MapClickType.TERRAIN,
            lat=terrain_data["lat"],
            lon=terrain_data["lng"],
        )

"""Pydeck click handler using streamlit-deckgl for terrain click support.

Uses st_deckgl from streamlit-deckgl to capture ALL click events including
terrain clicks (empty space), not just object selections.

The key difference from st.pydeck_chart:
- st.pydeck_chart: Only returns object selections (pickable=True objects)
- st_deckgl: Returns full deck.gl onClick event with coordinate field for ALL clicks
"""

import logging
from dataclasses import dataclass
from typing import Any

import pydeck as pdk
import streamlit as st
from streamlit_deckgl import st_deckgl  # type: ignore[import-untyped]

from skiresort_planner.constants import ChartConfig

logger = logging.getLogger(__name__)


@dataclass
class PydeckClickResult:
    """Result from Pydeck click detection.

    Attributes:
        clicked_object: The picked deck.gl object data (dict) or None if terrain click
        clicked_coordinate: [lon, lat] of click location (always available for clicks)
        is_object_click: True if a pickable object was clicked
        is_terrain_click: True if terrain (no object) was clicked
    """

    clicked_object: dict[str, Any] | None
    clicked_coordinate: list[float] | None

    @property
    def is_object_click(self) -> bool:
        """True if a pickable object was clicked."""
        return self.clicked_object is not None

    @property
    def is_terrain_click(self) -> bool:
        """True if terrain (empty space) was clicked with valid coordinates."""
        return self.clicked_object is None and self.clicked_coordinate is not None

    @staticmethod
    def empty() -> "PydeckClickResult":
        """Return empty result (no click detected)."""
        return PydeckClickResult(clicked_object=None, clicked_coordinate=None)


def render_pydeck_map(
    deck: pdk.Deck,
    key: str,
    height: int = ChartConfig.PROFILE_HEIGHT_LARGE,
) -> PydeckClickResult:
    """Render Pydeck map with full click support including terrain clicks.

    Uses st_deckgl from streamlit-deckgl to capture ALL click events:
    - Object clicks: event contains 'object' with picked layer data
    - Terrain clicks: event contains 'coordinate' [lon, lat]

    Args:
        deck: Configured pydeck.Deck object
        key: Unique key for this component instance
        height: Height in pixels

    Returns:
        PydeckClickResult with click info (object and/or coordinate)
    """
    # Session state keys for click deduplication
    last_click_key = f"_deckgl_last_click_{key}"

    # Initialize session state
    if last_click_key not in st.session_state:
        st.session_state[last_click_key] = None

    # Render the map with st_deckgl (returns click event dict)
    # MUST pass events=['click'] to enable click detection!
    event = st_deckgl(deck, key=key, height=height, events=["click"])

    # No event = no click
    if not event:
        return PydeckClickResult.empty()

    # st_deckgl SPREADS object properties into the event dict (no "object" key!)
    # Event structure:
    # - Terrain click: {coordinate: [lon, lat], eventType: "click"}
    # - Object click: {type: ..., id: ..., position: [...], coordinate: [lon, lat], eventType: "click"}
    logger.debug(f"st_deckgl event keys: {list(event.keys()) if isinstance(event, dict) else type(event)}")

    # Extract click data from event
    clicked_object: dict[str, Any] | None = None
    clicked_coordinate: list[float] | None = None

    # Check for coordinate (always present for clicks)
    if "coordinate" in event and event["coordinate"]:
        coord = event["coordinate"]
        if isinstance(coord, (list, tuple)) and len(coord) >= 2:
            # st_deckgl returns [lon, lat] from deck.gl
            clicked_coordinate = [float(coord[0]), float(coord[1])]

    # Check for picked object by presence of "type" field (set by our layers)
    # Object properties are SPREAD into event, not under "object" key!
    if "type" in event and event["type"] and event["type"] != "click":
        # This is an object click - extract the object data
        clicked_object = {k: v for k, v in event.items() if k not in ("coordinate", "eventType")}
        logger.debug(f"Object click detected: type={event.get('type')}, id={event.get('id')}")

    # No useful click data
    if clicked_coordinate is None and clicked_object is None:
        return PydeckClickResult.empty()

    # Deduplication: check if this is the same click as before
    click_id = _get_click_id(obj=clicked_object, coord=clicked_coordinate)
    if click_id == st.session_state.get(last_click_key):
        return PydeckClickResult.empty()

    # New click - store for dedup and return
    st.session_state[last_click_key] = click_id

    logger.debug(f"Click detected: object={clicked_object is not None}, coord={clicked_coordinate}")

    return PydeckClickResult(
        clicked_object=clicked_object,
        clicked_coordinate=clicked_coordinate,
    )


def _get_click_id(obj: dict[str, Any] | None, coord: list[float] | None) -> str:
    """Generate unique ID for click deduplication."""
    parts = []

    if obj:
        obj_type = obj.get("type", "")
        obj_id = obj.get("id", "")
        if obj_type and obj_id:
            parts.append(f"{obj_type}_{obj_id}")
        else:
            # Use position as fallback
            pos = obj.get("position", [])
            if pos:
                parts.append(f"pos_{pos[0]:.6f}_{pos[1]:.6f}")

    if coord:
        # Round coordinates for dedup tolerance
        parts.append(f"coord_{coord[0]:.5f}_{coord[1]:.5f}")

    return "_".join(parts) if parts else ""

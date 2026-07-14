"""Free-text place search via the Nominatim (OpenStreetMap) geocoder.

Powers the sidebar search box: type a place name (a resort, a town, anything OSM knows),
and we return the top match's coordinates so the map can recenter on it. This is a
name→coordinates lookup only — unrelated to the Overpass lift/piste import.

Policy (https://operations.osmfoundation.org/policies/nominatim/): max 1 request/second
and a custom User-Agent are required; we send OSMConfig.USER_AGENT. We only ever search on
submit (Enter / button), never on keystroke — autocomplete is explicitly forbidden.
"""

import logging
from dataclasses import dataclass

import requests

from skiresort_planner.constants import OSMConfig

logger = logging.getLogger(__name__)


@dataclass
class GeocodeResult:
    """A single geocoded place: its coordinates and human-readable name."""

    lat: float
    lon: float
    display_name: str


def geocode(query: str) -> GeocodeResult | None:
    """Top Nominatim match for a free-text place query, or None if nothing/error.

    Returns None on an empty/whitespace query, no results, or any network error — the
    caller shows a message rather than crashing.
    """
    query = query.strip()
    if not query:
        return None

    try:
        response = requests.get(
            OSMConfig.NOMINATIM_URL,
            params={"q": query, "format": "json", "limit": "1"},
            headers={"User-Agent": OSMConfig.USER_AGENT},
            timeout=OSMConfig.NOMINATIM_TIMEOUT_S,
        )
        response.raise_for_status()
        results = response.json()
    except requests.RequestException as exc:
        logger.warning(f"Nominatim search for {query!r} failed: {exc}")
        return None

    if not results:
        logger.info(f"Nominatim found no place for {query!r}")
        return None

    top = results[0]
    return GeocodeResult(
        lat=float(top["lat"]),
        lon=float(top["lon"]),
        display_name=str(top["display_name"]),
    )

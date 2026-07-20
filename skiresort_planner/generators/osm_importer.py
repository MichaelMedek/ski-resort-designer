"""Import existing lifts from OpenStreetMap.

We take GEOMETRY ONLY from OSM — where the lifts are (the two stations of each aerialway).
Everything else is recomputed through our own pipeline: elevation from the DEM and pylons/catenary
from the terrain. OSM's difficulty / pylon / elevation tags are deliberately ignored.

The import region is a SQUARE bounding box the user picks, fetched in ONE Overpass query (a
lift/piste-only query is light — even a full-size box returns in a few seconds). Overpass gives a
few slots per IP; on a transient 429 (no slot) / 504 (busy) we wait for a free slot (from
/api/status) and retry once. Only FULL entities are imported — a way with any vertex outside the
box, or crossing a DEM nodata hole, is skipped entirely (counted), never half-imported. Only NAMED
lifts import: unnamed OSM ways are frequently outdated or duplicate, so we skip them (logged).

This module holds the SHARED importer base (`BaseOSMImporter`: Overpass fetch) plus `extract_lift_sections`
— the raw lift-section core (mid-station split incl.) both children reuse. Slope geometry is the
connected-graph builder's job — see the `LiftOnlyImporter` (lifts only, raw OSM) and `GraphImporter`
(lifts + slopes) children.
"""

import json
import logging
import math
import re
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, NamedTuple, TypedDict, cast
from urllib.parse import urlencode

import requests

from skiresort_planner.constants import EntitySource, MapConfig, OSMConfig
from skiresort_planner.core.dem_service import DEMService
from skiresort_planner.model.path_point import PathPoint

logger = logging.getLogger(__name__)

# The import region as a bounding box (min_lon, min_lat, max_lon, max_lat).
BBox = tuple[float, float, float, float]
# An OSM way vertex as (lon, lat).
Vertex = tuple[float, float]
# Progress reporter for a slow blocking action: (fraction 0..1, status text).
ProgressFn = Callable[[float, str], None]


def sub_progress(report: ProgressFn, lo: float, hi: float) -> ProgressFn:
    """Map a child's 0..1 progress onto the [lo, hi] slice of a parent progress bar."""
    return lambda frac, text: report(lo + (hi - lo) * max(0.0, min(1.0, frac)), text)


class OverpassVertex(TypedDict):
    """One inline geometry vertex from an Overpass `out geom;` way."""

    lon: float
    lat: float


class OverpassElement(TypedDict, total=False):
    """One Overpass element (a way or a node). `total=False`: any field may be absent in a raw
    response, so every access below still guards with .get()/defaults — the TypedDict only names the
    shape. A way carries `geometry` + a parallel `nodes` id list; a node carries `lat`/`lon` directly.
    """

    id: int
    type: str
    tags: dict[str, str]
    geometry: list[OverpassVertex]
    nodes: list[int]
    lat: float
    lon: float


def bbox_around(center_lon: float, center_lat: float, half_width_m: float) -> BBox:
    """The square bounding box of the given half-width (metres) centered on (lon, lat)."""
    m_per_deg = MapConfig.METERS_PER_DEGREE_EQUATOR
    dlat = half_width_m / m_per_deg
    dlon = dlat / max(math.cos(math.radians(center_lat)), 1e-6)
    return (center_lon - dlon, center_lat - dlat, center_lon + dlon, center_lat + dlat)


def _is_transient(exc: BaseException) -> bool:
    """True for errors worth retrying: rate limit (429), gateway timeout (504), or a network error.

    Response-bearing errors are transient only for 429/504; connection/timeout errors always are.
    """
    if not isinstance(exc, requests.RequestException):
        return False
    response = exc.response
    if response is None:
        return True  # connection reset / read timeout — retry
    return response.status_code in (429, 504)


@dataclass
class ImportResult:
    """What an OSM import produced, in the ONE shape every importer returns and the graph consumes.

    `lifts` are (bottom, top, lift_type, name) tuples; `slope_chains` is empty for a lift-only import
    and otherwise carries, per whole slope, a list of its segment point-lists plus the slope name.
    `source` tags every materialised entity (provenance + re-import dedup).
    """

    lifts: list[tuple[PathPoint, PathPoint, str, str | None]] = field(default_factory=list)
    slope_chains: list[tuple[list[list[PathPoint]], str | None]] = field(default_factory=list)
    source: str = EntitySource.OSM
    skipped: int = 0  # ways dropped because they reach outside the box / over nodata / too short


class OSMImportResult(NamedTuple):
    """Counts from one import_osm call (named so callers don't guess tuple positions)."""

    slopes_added: int
    lifts_added: int
    duplicates_skipped: int  # entities skipped because the graph already has that endpoint fingerprint


class BaseOSMImporter(ABC):
    """Shared OSM importer: fetches lifts & pistes in a bbox. Never used alone — a concrete subclass
    supplies `_assemble` (lift-only vs the connected slope graph).

    Pure fetch + parse; it does NOT mutate the resort graph (the caller does, so the whole import is
    one undoable batch).
    """

    SOURCE: ClassVar[str] = EntitySource.OSM

    def __init__(self, dem: DEMService, bbox: BBox) -> None:
        self.dem = dem
        self.bbox = bbox

    # -- public entry point ---------------------------------------------------

    def run(self, *, on_progress: ProgressFn, dump_dir: Path | None = None) -> ImportResult:
        """Fetch the box and assemble the import, reporting progress via `on_progress`: 0.1 before the
        (single, blocking) fetch so the bar moves immediately, 0.5 once it returns, then the assemble/build
        fills 0.5→1.0. If `dump_dir` is given, write reference artifacts (raw fetch + PNG) there.
        """
        on_progress(0.1, "Fetching from OpenStreetMap…")
        elements = self.fetch()
        on_progress(0.5, "Building…")
        result = self._assemble(elements, on_progress=sub_progress(on_progress, 0.5, 1.0))
        if dump_dir is not None:
            self._dump(elements, dump_dir)
        on_progress(1.0, "Done")
        return result

    @abstractmethod
    def _assemble(self, elements: list[OverpassElement], on_progress: ProgressFn) -> ImportResult:
        """Turn raw Overpass elements into an ImportResult (lifts, and slopes for the graph child)."""

    def _dump(self, elements: list[OverpassElement], dump_dir: Path) -> None:
        """Write the raw fetch to `dump_dir/osm_raw.json` for reference (never read back). Writes
        directly (like persistence.backup_store.save) — a failing OUTPUT_DIR write is a real problem.
        """
        dump_dir.mkdir(parents=True, exist_ok=True)
        out = dump_dir / "osm_raw.json"
        out.write_text(json.dumps(elements), encoding="utf-8")
        logger.debug(f"OSM import: wrote raw fetch ({len(elements)} elements) to {out}")

    # -- fetch ----------------------------------------------------------------

    def fetch(self) -> list[OverpassElement]:
        """Fetch all OSM lift/piste ways in the box with ONE Overpass query.

        On a transient 429/504 we wait for a free slot and retry once, then give up.
        """
        try:
            return self._query(self.bbox)
        except requests.RequestException as exc:
            if not _is_transient(exc):
                raise
            wait_s = _seconds_until_free_slot()
            logger.info(f"Overpass busy ({exc}); waiting {wait_s:.0f}s for a free slot, then retrying once")
            time.sleep(wait_s)
            return self._query(self.bbox)

    def _query(self, bbox: BBox) -> list[OverpassElement]:
        """POST one Overpass query for the box and return its ways (inline geometry) + station nodes.

        The `aerialway=station` nodes come back alongside the ways so a lift way with an interior
        station node can be split into per-section lifts (see `split_lift_way_at_stations`). One
        blocking POST (the response isn't streamed — see run() for the coarse fetch progress). Raises
        on any non-200.
        """
        min_lon, min_lat, max_lon, max_lat = bbox
        # Overpass bbox filter order is (south, west, north, east).
        area = f"({min_lat},{min_lon},{max_lat},{max_lon})"
        query = (
            f"[out:json][timeout:{OSMConfig.OVERPASS_TIMEOUT_S:d}];"
            + f'(way["aerialway"]{area};'
            + f'way["piste:type"]{area};'
            + f'node["aerialway"="station"]{area};);'
            + "out geom;"
        )
        response = requests.post(
            OSMConfig.OVERPASS_URL,
            data=urlencode({"data": query}),
            headers={"User-Agent": OSMConfig.USER_AGENT, "Content-Type": "application/x-www-form-urlencoded"},
            timeout=OSMConfig.OVERPASS_TIMEOUT_S,
        )
        response.raise_for_status()
        elements = cast(list[OverpassElement], response.json()["elements"])
        logger.info(f"Overpass returned {len(elements)} elements for bbox {bbox}")
        return elements


# -- lift-section extraction (shared core: both importers reuse it) -----------

LiftSection = tuple[list[Vertex], str, str | None]  # (vertices lon/lat, lift_type, name-or-None)


def extract_lift_sections(elements: list[OverpassElement], bbox: BBox) -> list[LiftSection]:
    """Raw skiable-lift sections shared by BOTH importers: keep mapped-aerialway ways fully in-box,
    split each at interior `aerialway=station` nodes, carry the per-section name. No DEM / min-length /
    dedup — those are each caller's policy (lift-only drops unnamed + DEM-drapes; graph hub-merges).
    """
    station_ids = station_node_ids(elements)
    sections: list[LiftSection] = []
    for el in elements:
        tags = el.get("tags", {})
        lift_type = OSMConfig.AERIALWAY_TO_LIFT_TYPE.get(tags.get("aerialway", ""))
        if lift_type is None:  # not a mapped skiable aerialway (station/pylon/zip_line/…)
            continue
        verts = [(v["lon"], v["lat"]) for v in el.get("geometry", [])]
        if len(verts) < 2 or not _fully_inside(vertices=verts, bbox=bbox):
            continue  # truncated by the box → skip the whole way (never half-import)
        parts = split_lift_way_at_stations(verts, el.get("nodes", []), station_ids)
        name = _lift_name(tags)
        for i, part in enumerate(parts):
            sections.append((part, str(lift_type), suffixed_name(name, i, len(parts))))
    return sections


def _fully_inside(vertices: list[Vertex], bbox: BBox) -> bool:
    """True if every vertex lies within the box (so the way is not truncated by it)."""
    min_lon, min_lat, max_lon, max_lat = bbox
    return all(min_lon <= lon <= max_lon and min_lat <= lat <= max_lat for lon, lat in vertices)


def _seconds_until_free_slot() -> float:
    """Seconds to wait for a free Overpass slot from /api/status, clamped to SLOT_WAIT_MAX_S.

    On any parse/network failure, falls back to a short fixed wait (status only tunes sleep length,
    it is not required for correctness).
    """
    try:
        text = requests.get(
            OSMConfig.OVERPASS_STATUS_URL, headers={"User-Agent": OSMConfig.USER_AGENT}, timeout=15
        ).text
    except requests.RequestException:
        return OSMConfig.SLOT_WAIT_FALLBACK_S
    if re.search(r"[1-9]\d* slots? available now", text):
        return 0.0
    waits = [int(m) for m in re.findall(r"in (\d+) seconds", text)]
    wait = min(waits) if waits else OSMConfig.SLOT_WAIT_FALLBACK_S
    return float(min(wait, OSMConfig.SLOT_WAIT_MAX_S))


def station_node_ids(elements: list[OverpassElement]) -> set[int]:
    """Ids of the `aerialway=station` node elements in the response (used to split lift ways)."""
    return {
        el["id"] for el in elements if el.get("type") == "node" and el.get("tags", {}).get("aerialway") == "station"
    }


def split_lift_way_at_stations(
    vertices: list[Vertex], node_ids: list[int], station_ids: set[int]
) -> list[list[Vertex]]:
    """Split a lift way into per-section polylines at every INTERIOR station node.

    A mid-station is an interior way vertex whose OSM node id is a station id (exact match, no
    tolerance). Consecutive sections share the station vertex (section k ends where k+1 begins), so
    N interior stations → N+1 sections. No interior station → the way unchanged as one section.

    Args:
        vertices: the way's (lon, lat) vertices.
        node_ids: the way's parallel OSM node ids (must align 1:1 with `vertices`).
        station_ids: ids of `aerialway=station` nodes fetched for the box.
    """
    assert len(vertices) == len(node_ids), "OSM way geometry and node ids must align 1:1"
    cuts = [i for i in range(1, len(vertices) - 1) if node_ids[i] in station_ids]
    if not cuts:
        return [vertices]
    bounds = [0, *cuts, len(vertices) - 1]
    return [vertices[bounds[k] : bounds[k + 1] + 1] for k in range(len(bounds) - 1)]


def suffixed_name(name: str | None, index: int, total: int) -> str | None:
    """Per-section display name: `"<name> (k)"` (1-based) when a name splits into `total` > 1 sections
    (lift mid-stations, or slope pieces sharing an OSM name); the bare name when it stays whole.
    """
    if name is None or total <= 1:
        return name
    return f"{name} ({index + 1})"


def _lift_name(tags: dict[str, str]) -> str | None:
    """OSM lift name resolution; None if unnamed (unnamed lifts are skipped on import)."""
    for key in ("name", "ref"):
        value = tags.get(key)
        if value:
            return str(value)
    return None

"""Import existing lifts & pistes from OpenStreetMap.

We take GEOMETRY ONLY from OSM — where the pistes and lifts are (raw lon/lat polylines and the
two lift stations). Everything else is recomputed through our own pipeline: elevation from the
DEM, difficulty from the DEM-derived max_slope_pct, and lift pylons/catenary from the terrain.
OSM's difficulty / pylon / elevation / WIDTH tags are deliberately ignored — belt width comes
entirely from our own PathSegment.width_m (adaptive to side slope).

The import region is a SQUARE bounding box the user picks. A large box in one Overpass query
times out (504), so we TILE it into a grid of smaller square boxes and fetch each separately,
merging elements deduped by OSM id. Only FULL entities are imported — a way with any vertex
outside the box, or crossing a DEM nodata hole, is skipped entirely (counted in the summary),
never half-imported. Only NAMED lifts and pistes import: unnamed OSM ways are frequently
outdated or duplicate, so we skip them (logged with the reason).
"""

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlencode

import requests
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from skiresort_planner.constants import OSMConfig
from skiresort_planner.core.dem_service import DEMService
from skiresort_planner.core.geo_calculator import GeoCalculator
from skiresort_planner.model.path_point import PathPoint

logger = logging.getLogger(__name__)

# The import region as a bounding box (min_lon, min_lat, max_lon, max_lat).
BBox = tuple[float, float, float, float]
# An OSM way vertex as (lon, lat).
Vertex = tuple[float, float]


def bbox_around(center_lon: float, center_lat: float, half_width_m: float) -> BBox:
    """The square bounding box of the given half-width (metres) centered on (lon, lat)."""
    m_per_deg = GeoCalculator.haversine_distance_m(lat1=0.0, lon1=0.0, lat2=1.0, lon2=0.0)
    dlat = half_width_m / m_per_deg
    dlon = dlat / max(math.cos(math.radians(center_lat)), 1e-6)
    return (center_lon - dlon, center_lat - dlat, center_lon + dlon, center_lat + dlat)


def _is_transient(exc: BaseException) -> bool:
    """True for errors worth retrying: rate limit (429), gateway timeout (504), or a network error.

    A response-bearing error is transient only for 429/504; a bad request (4xx like 406) is not.
    A connection/timeout error has no response and is always worth a retry. Non-request exceptions
    (which tenacity may pass in) are not retried.
    """
    if not isinstance(exc, requests.RequestException):
        return False
    response = exc.response
    if response is None:
        return True  # connection reset / read timeout — retry
    return response.status_code in (429, 504)


@dataclass(frozen=True)
class PisteImport:
    """One downhill piste ready to commit: DEM-sampled points + optional OSM name."""

    points: list[PathPoint]
    name: str | None


@dataclass(frozen=True)
class LiftImport:
    """One lift ready to add: its two stations (DEM-sampled), lift-type, + optional OSM name."""

    bottom: PathPoint
    top: PathPoint
    lift_type: str
    name: str | None


@dataclass
class ImportSummary:
    """What an import produced, so the UI can report full vs skipped counts."""

    pistes: list[PisteImport] = field(default_factory=list)
    lifts: list[LiftImport] = field(default_factory=list)
    skipped: int = 0  # ways dropped because they reach outside the box / over nodata / too short


class OSMImporter:
    """Fetches OSM lifts & pistes within a bounding box and converts them to import-ready geometry.

    Pure fetch + parse + DEM-drape; it does NOT mutate the resort graph (the caller does, so
    the whole import can be one undoable batch).
    """

    def __init__(self, dem: DEMService) -> None:
        self.dem = dem

    def fetch(self, bbox: BBox) -> list[dict[str, Any]]:
        """Fetch all OSM lift/piste ways in the box, tiling it into a grid of smaller boxes.

        One Overpass query over a large box times out (504), and firing every tile at once trips
        the public endpoint's rate limit (429). So we split the box into a grid of sub-tiles and
        fetch them PACED — a throttle between requests, each tile retried with exponential backoff —
        then merge the results deduped by OSM id. Reliable for a box of any size.
        """
        by_id: dict[int, dict[str, Any]] = {}
        tiles = _tile_bboxes(bbox)
        for i, tile in enumerate(tiles):
            if i > 0:
                time.sleep(OSMConfig.TILE_THROTTLE_S)  # pace requests so we don't trip the rate limit
            elements = self._fetch_tile(tile)
            for el in elements:
                by_id[el["id"]] = el
            logger.info(f"OSM tile {i + 1}/{len(tiles)}: {len(elements)} elements ({len(by_id)} unique so far)")
        merged = list(by_id.values())
        logger.info(f"Overpass returned {len(merged)} unique elements across {len(tiles)} tiles for bbox {bbox}")
        return merged

    @retry(
        retry=retry_if_exception(_is_transient),
        stop=stop_after_attempt(OSMConfig.TILE_RETRIES),
        wait=wait_exponential(multiplier=OSMConfig.TILE_RETRY_BACKOFF_S),
        before_sleep=before_sleep_log(logger, logging.INFO),
        reraise=True,
    )
    def _fetch_tile(self, tile: BBox) -> list[dict[str, Any]]:
        """POST an Overpass query for one sub-box, retrying transient failures with backoff.

        Uses Overpass's native bbox filter. tenacity retries a transient error (429 rate-limit,
        504 timeout, or a network error) up to TILE_RETRIES times with exponential backoff; a
        non-transient error (e.g. 406 missing User-Agent) is re-raised at once.
        """
        min_lon, min_lat, max_lon, max_lat = tile
        # Overpass bbox filter order is (south, west, north, east).
        area = f"({min_lat},{min_lon},{max_lat},{max_lon})"
        query = (
            "[out:json][timeout:%d];" % OSMConfig.OVERPASS_TIMEOUT_S
            + f'(way["aerialway"]{area};'
            + f'way["piste:type"]{area};);'
            + "out geom;"
        )
        response = requests.post(
            OSMConfig.OVERPASS_URL,
            data=urlencode({"data": query}),
            headers={"User-Agent": OSMConfig.USER_AGENT, "Content-Type": "application/x-www-form-urlencoded"},
            timeout=OSMConfig.OVERPASS_TIMEOUT_S,
        )
        response.raise_for_status()
        elements: list[dict[str, Any]] = response.json()["elements"]
        return elements

    def convert(self, bbox: BBox, elements: list[dict[str, Any]]) -> ImportSummary:
        """Turn raw Overpass elements into import-ready pistes + lifts for the given box.

        `bbox` is the region the user chose; only ways fully inside the box are kept. Every element
        that is NOT imported is logged with its reason, so a missing lift/piste can be traced
        (unnamed, too short, reaching outside the box, over nodata, or an unmapped aerialway value).
        """
        summary = ImportSummary()
        for el in elements:
            tags = el.get("tags", {})
            vertices = [(v["lon"], v["lat"]) for v in el.get("geometry", [])]
            osm_id = el.get("id", "?")
            if "aerialway" in tags:
                self._add_lift(tags, vertices, bbox, summary, osm_id)
            elif "piste:type" in tags:
                self._add_piste(tags, vertices, bbox, summary, osm_id)
        logger.info(f"Converted: {len(summary.pistes)} pistes, {len(summary.lifts)} lifts, {summary.skipped} skipped")
        return summary

    # -- pistes ---------------------------------------------------------------

    def _add_piste(
        self, tags: dict[str, Any], vertices: list[Vertex], bbox: BBox, summary: ImportSummary, osm_id: Any
    ) -> None:
        if tags.get("piste:type") != OSMConfig.PISTE_TYPE_DOWNHILL:
            return  # only alpine downhill runs; ignore connection/snow_park/playground/sled/yes
        name = _piste_name(tags)
        if name is None:
            logger.info(
                f"Skipped piste way/{osm_id}: unnamed (potentially outdated/duplicate — only named runs import)"
            )
            summary.skipped += 1
            return
        if len(vertices) < 2 or not _fully_inside(vertices, bbox):
            logger.info(f"Skipped piste '{name}' (way/{osm_id}): reaches outside the import area")
            summary.skipped += 1
            return
        length = _polyline_length_m(vertices)
        if length < OSMConfig.MIN_PISTE_LENGTH_M:
            logger.info(
                f"Skipped piste '{name}' (way/{osm_id}): {length:.0f}m < {OSMConfig.MIN_PISTE_LENGTH_M:.0f}m min"
            )
            summary.skipped += 1
            return
        points = self._resample(vertices)
        if points is None:
            logger.info(f"Skipped piste '{name}' (way/{osm_id}): over a DEM nodata hole")
            summary.skipped += 1
            return
        summary.pistes.append(PisteImport(points=points, name=name))

    # -- lifts ----------------------------------------------------------------

    def _add_lift(
        self, tags: dict[str, Any], vertices: list[Vertex], bbox: BBox, summary: ImportSummary, osm_id: Any
    ) -> None:
        aerialway = tags["aerialway"]
        lift_type = OSMConfig.AERIALWAY_TO_LIFT_TYPE.get(aerialway)
        if lift_type is None:
            # station/pylon/zip_line/magic_carpet/rope_tow/yes/… are not skiable lifts — not counted.
            logger.info(f"Ignored lift way/{osm_id}: unmapped aerialway='{aerialway}' (not a skiable lift)")
            return
        name = _lift_name(tags)
        if name is None:
            logger.info(
                f"Skipped {aerialway} way/{osm_id}: unnamed (potentially outdated/duplicate — only named lifts import)"
            )
            summary.skipped += 1
            return
        if len(vertices) < 2 or not _fully_inside(vertices, bbox):
            logger.info(f"Skipped lift '{name}' (way/{osm_id}): reaches outside the import area")
            summary.skipped += 1
            return
        length = _polyline_length_m(vertices)
        if length < OSMConfig.MIN_LIFT_LENGTH_M:
            logger.info(f"Skipped lift '{name}' (way/{osm_id}): {length:.0f}m < {OSMConfig.MIN_LIFT_LENGTH_M:.0f}m min")
            summary.skipped += 1
            return
        # Only the two stations matter; OSM intermediate pylons are dropped (we regenerate them).
        bottom = self._point(vertices[0])
        top = self._point(vertices[-1])
        if bottom is None or top is None:
            logger.info(f"Skipped lift '{name}' (way/{osm_id}): a station is over a DEM nodata hole")
            summary.skipped += 1
            return
        # A lift runs valley→mountain; orient bottom = lower station.
        if bottom.elevation > top.elevation:
            bottom, top = top, bottom
        summary.lifts.append(LiftImport(bottom=bottom, top=top, lift_type=lift_type, name=name))

    # -- geometry helpers -----------------------------------------------------

    def _resample(self, vertices: list[Vertex]) -> list[PathPoint] | None:
        """Linearly resample the polyline every RESAMPLE_STEP_M, DEM-sampling Z at each output
        point. Returns None if any sample falls on a DEM nodata cell.

        Linear (not the planner's cubic spline): real OSM pistes are already smooth and must not
        be over-smoothed. Whole-path finish smoothing still runs later via finish_slope.
        Callers gate on MIN_PISTE_LENGTH_M (>> step), so total is always > step here.
        """
        step = OSMConfig.RESAMPLE_STEP_M
        # Cumulative distance along the raw polyline.
        seg_len = [
            GeoCalculator.haversine_distance_m(
                lat1=vertices[i][1], lon1=vertices[i][0], lat2=vertices[i + 1][1], lon2=vertices[i + 1][0]
            )
            for i in range(len(vertices) - 1)
        ]

        out_lonlat: list[Vertex] = [vertices[0]]
        target = step
        walked = 0.0
        for i, length in enumerate(seg_len):
            (x0, y0), (x1, y1) = vertices[i], vertices[i + 1]
            while target <= walked + length and length > 0:
                bearing = GeoCalculator.initial_bearing_deg(lon1=x0, lat1=y0, lon2=x1, lat2=y1)
                out_lonlat.append(
                    GeoCalculator.destination(lon=x0, lat=y0, bearing_deg=bearing, distance_m=target - walked)
                )
                target += step
            walked += length
        out_lonlat.append(vertices[-1])

        return _drop_none([self._point(v) for v in out_lonlat])

    def _point(self, vertex: Vertex) -> PathPoint | None:
        lon, lat = vertex
        elev = self.dem.get_elevation(lon=lon, lat=lat)
        if elev is None:
            return None
        return PathPoint(lon=lon, lat=lat, elevation=elev)


def _fully_inside(vertices: list[Vertex], bbox: BBox) -> bool:
    """True if every vertex lies within the box (so the way is not truncated by it)."""
    min_lon, min_lat, max_lon, max_lat = bbox
    return all(min_lon <= lon <= max_lon and min_lat <= lat <= max_lat for lon, lat in vertices)


def _tile_bboxes(bbox: BBox) -> list[BBox]:
    """Split the box into a grid of sub-tiles no wider than 2·TILE_HALF_WIDTH_M (as bboxes).

    Overpass times out on a large bbox query, so we fetch tile-by-tile. The box is divided into an
    even grid whose cells each stay within the tile size; a box already within the tile size is a
    single tile equal to itself. Squares tile a square exactly — no overlap, no gaps.
    """
    min_lon, min_lat, max_lon, max_lat = bbox
    center_lat = (min_lat + max_lat) / 2.0
    m_per_deg = GeoCalculator.haversine_distance_m(lat1=0.0, lon1=0.0, lat2=1.0, lon2=0.0)
    tile_lat_deg = (2.0 * OSMConfig.TILE_HALF_WIDTH_M) / m_per_deg
    tile_lon_deg = tile_lat_deg / max(math.cos(math.radians(center_lat)), 1e-6)

    n_lon = max(1, math.ceil((max_lon - min_lon) / tile_lon_deg))
    n_lat = max(1, math.ceil((max_lat - min_lat) / tile_lat_deg))
    span_lon = (max_lon - min_lon) / n_lon
    span_lat = (max_lat - min_lat) / n_lat

    tiles: list[BBox] = []
    for iy in range(n_lat):
        for ix in range(n_lon):
            tiles.append(
                (
                    min_lon + ix * span_lon,
                    min_lat + iy * span_lat,
                    min_lon + (ix + 1) * span_lon,
                    min_lat + (iy + 1) * span_lat,
                )
            )
    return tiles


def _polyline_length_m(vertices: list[Vertex]) -> float:
    """Total ground length of the raw lon/lat polyline in metres (sum of haversine legs)."""
    return sum(
        GeoCalculator.haversine_distance_m(
            lat1=vertices[i][1], lon1=vertices[i][0], lat2=vertices[i + 1][1], lon2=vertices[i + 1][0]
        )
        for i in range(len(vertices) - 1)
    )


def _drop_none(points: list[PathPoint | None]) -> list[PathPoint] | None:
    """Return the points if all are present; None if ANY is a nodata miss (all-or-nothing)."""
    if any(p is None for p in points):
        return None
    return [p for p in points if p is not None]  # narrowed: no None remains


def _piste_name(tags: dict[str, Any]) -> str | None:
    """OSM name resolution order; None if the run is unnamed (unnamed runs are skipped on import)."""
    for key in ("name", "piste:name", "piste:ref", "ref"):
        value = tags.get(key)
        if value:
            return str(value)
    return None


def _lift_name(tags: dict[str, Any]) -> str | None:
    """OSM lift name resolution; None if unnamed (unnamed lifts are skipped on import)."""
    for key in ("name", "ref"):
        value = tags.get(key)
        if value:
            return str(value)
    return None

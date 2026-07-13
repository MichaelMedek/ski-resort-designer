"""Import existing lifts & pistes from OpenStreetMap.

We take GEOMETRY ONLY from OSM — where the pistes and lifts are (raw lon/lat polylines and the
two lift stations). Everything else is recomputed through our own pipeline: elevation from the
DEM, difficulty from the DEM-derived max_slope_pct, and lift pylons/catenary from the terrain.
OSM's difficulty / pylon / elevation / WIDTH tags are deliberately ignored — belt width comes
entirely from our own PathSegment.width_m (adaptive to side slope).

The import region is a SQUARE bounding box the user picks, fetched in ONE Overpass query (a
lift/piste-only query is light — even a full-size box returns in a few seconds). Overpass gives a
few slots per IP; on a transient 429 (no slot) / 504 (busy) we wait for a free slot (from
/api/status) and retry once. Only FULL entities are imported — a way with any vertex outside the
box, or crossing a DEM nodata hole, is skipped entirely (counted in the summary), never
half-imported. Pistes are trimmed to their descending run (a ski run only goes down). Only NAMED
lifts and pistes import: unnamed OSM ways are frequently outdated or duplicate, so we skip them
(logged with the reason).
"""

import logging
import math
import re
import time
from dataclasses import dataclass, field
from typing import TypedDict, cast
from urllib.parse import urlencode

import requests

from skiresort_planner.constants import OSMConfig, SlopeConfig
from skiresort_planner.core.dem_service import DEMService
from skiresort_planner.core.geo_calculator import GeoCalculator
from skiresort_planner.model.path_point import PathPoint

logger = logging.getLogger(__name__)

# The import region as a bounding box (min_lon, min_lat, max_lon, max_lat).
BBox = tuple[float, float, float, float]
# An OSM way vertex as (lon, lat).
Vertex = tuple[float, float]


class OverpassVertex(TypedDict):
    """One inline geometry vertex from an Overpass `out geom;` way."""

    lon: float
    lat: float


class OverpassElement(TypedDict, total=False):
    """One Overpass element (a way). `total=False`: any field may be absent in a raw response,
    so every access below still guards with .get()/defaults — the TypedDict only names the shape.
    """

    id: int
    tags: dict[str, str]
    geometry: list[OverpassVertex]


def bbox_around(center_lon: float, center_lat: float, half_width_m: float) -> BBox:
    """The square bounding box of the given half-width (metres) centered on (lon, lat)."""
    m_per_deg = GeoCalculator.haversine_distance_m(lat1=0.0, lon1=0.0, lat2=1.0, lon2=0.0)
    dlat = half_width_m / m_per_deg
    dlon = dlat / max(math.cos(math.radians(center_lat)), 1e-6)
    return (center_lon - dlon, center_lat - dlat, center_lon + dlon, center_lat + dlat)


def _is_transient(exc: BaseException) -> bool:
    """True for errors worth retrying: rate limit (429), gateway timeout (504), or a network error.

    A response-bearing error is transient only for 429/504; a bad request (4xx like 406) is not.
    A connection/timeout error has no response and is always worth a retry.
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

    def fetch(self, bbox: BBox) -> list[OverpassElement]:
        """Fetch all OSM lift/piste ways in the box with ONE Overpass query.

        A lift/piste-only query is light enough that even a full-size box returns in a few seconds,
        so no tiling is needed. On a transient 429 (no slot) / 504 (busy) we wait for a free slot
        and retry once, then give up (the caller shows an error toast).
        """
        try:
            return self._query(bbox)
        except requests.RequestException as exc:
            if not _is_transient(exc):
                raise
            wait_s = _seconds_until_free_slot()
            logger.info(f"Overpass busy ({exc}); waiting {wait_s:.0f}s for a free slot, then retrying once")
            time.sleep(wait_s)
            return self._query(bbox)

    def _query(self, bbox: BBox) -> list[OverpassElement]:
        """POST one Overpass query for the box and return its ways (with inline geometry).

        Uses Overpass's native bbox filter. Raises on any non-200 (the caller decides whether the
        error is transient and worth a retry).
        """
        min_lon, min_lat, max_lon, max_lat = bbox
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
        elements = cast(list[OverpassElement], response.json()["elements"])
        logger.info(f"Overpass returned {len(elements)} elements for bbox {bbox}")
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
        resampled = self._resample(vertices)
        if resampled is None:
            logger.info(f"Skipped piste '{name}' (way/{osm_id}): over a DEM nodata hole")
            summary.skipped += 1
            return
        # A ski run only goes down; trim an OSM out-and-back to its longest descending stretch
        # BEFORE the length gate, so the up-arm doesn't inflate a run that's really too short.
        points = _longest_descending_run(resampled)
        length = _polyline_length_m([(p.lon, p.lat) for p in points])
        if length < OSMConfig.MIN_PISTE_LENGTH_M:
            logger.info(
                f"Skipped piste '{name}' (way/{osm_id}): descending run {length:.0f}m "
                f"< {OSMConfig.MIN_PISTE_LENGTH_M:.0f}m min"
            )
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
        A polyline shorter than one step resamples to just its two endpoints.
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


def _seconds_until_free_slot() -> float:
    """Seconds to wait for a free Overpass slot, read from /api/status (clamped to SLOT_WAIT_MAX_S).

    /api/status reports either "N slots available now." (wait 0) or one "…in X seconds." line per
    busy slot (wait the soonest). On any parse/network failure, fall back to a short fixed wait so a
    retry still happens — status only tunes how long to sleep, it is not required for correctness.
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


def _longest_descending_run(points: list[PathPoint]) -> list[PathPoint]:
    """Trim a DEM-draped polyline to its longest DESCENDING run, oriented top→bottom.

    OSM mappers sometimes draw a piste as an out-and-back (up then down), which drapes to an
    up-and-down elevation profile — a 0 m net drop, 0% "slope". A ski run only goes down, so we keep
    the longest stretch that descends. To ignore point-level DEM noise we judge "descending" on
    elevations SMOOTHED over the rolling window (SlopeConfig.ROLLING_WINDOW_M): a real run with minor
    rolls survives, only a genuine sustained climb breaks a run. Both orientations are considered and
    the result is returned top→bottom (reversed if the descent runs end→start).
    """
    if len(points) < 2:
        return points

    elevs = [p.elevation for p in points]
    window_pts = max(1, round(SlopeConfig.ROLLING_WINDOW_M / OSMConfig.RESAMPLE_STEP_M))
    half = window_pts // 2
    smoothed: list[float] = []
    for i in range(len(elevs)):
        window = elevs[max(0, i - half) : min(len(elevs), i + half + 1)]
        smoothed.append(sum(window) / len(window))

    def longest_non_increasing(series: list[float]) -> tuple[int, int]:
        """(start, end) inclusive of the longest run where series never rises step-to-step."""
        best_start, best_end, start = 0, 0, 0
        for i in range(1, len(series)):
            if series[i] > series[i - 1] + 1e-9:  # a rise ends the current run
                start = i
            if i - start > best_end - best_start:
                best_start, best_end = start, i
        return best_start, best_end

    f0, f1 = longest_non_increasing(smoothed)
    r0, r1 = longest_non_increasing(smoothed[::-1])
    if (f1 - f0) >= (r1 - r0):
        return points[f0 : f1 + 1]
    n = len(points)
    return points[n - 1 - r1 : n - r0][::-1]  # map reversed indices back, orient top→bottom


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

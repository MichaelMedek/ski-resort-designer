"""Import existing lifts & pistes from OpenStreetMap.

We take GEOMETRY ONLY from OSM — where the pistes and lifts are (raw lon/lat polylines and the
two lift stations). Everything else is recomputed through our own pipeline: elevation from the
DEM, difficulty from the DEM-derived max_slope_pct, and lift pylons/catenary from the terrain.
OSM's difficulty / pylon / elevation tags are deliberately ignored.

The import region is a CIRCLE: a center point + radius the user picks. Only FULL entities are
imported — a way with any vertex outside the circle, or crossing a DEM nodata hole, is skipped
entirely (counted in the summary), never half-imported.
"""

import logging
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlencode

import requests

from skiresort_planner.constants import OSMConfig
from skiresort_planner.core.dem_service import DEMService
from skiresort_planner.core.geo_calculator import GeoCalculator
from skiresort_planner.model.path_point import PathPoint

logger = logging.getLogger(__name__)

# The import region as (center_lon, center_lat, radius_m).
Region = tuple[float, float, float]
# An OSM way vertex as (lon, lat).
Vertex = tuple[float, float]


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
    skipped: int = 0  # ways dropped because truncated by the circle / outside DEM / nodata


class OSMImporter:
    """Fetches OSM lifts & pistes within a circle and converts them to import-ready geometry.

    Pure fetch + parse + DEM-drape; it does NOT mutate the resort graph (the caller does, so
    the whole import can be one undoable batch).
    """

    def __init__(self, dem: DEMService) -> None:
        self.dem = dem

    def fetch(self, region: Region) -> list[dict[str, Any]]:
        """POST an Overpass query for the circle and return its `elements` (ways with inline geom).

        Uses Overpass's native (around:radius_m, lat, lon) filter — no bbox estimation. Raises on
        any non-200 response (e.g. 406 when the mandatory User-Agent is missing).
        """
        center_lon, center_lat, radius_m = region
        around = f"(around:{radius_m},{center_lat},{center_lon})"
        query = (
            "[out:json][timeout:%d];" % OSMConfig.OVERPASS_TIMEOUT_S
            + f'(way["aerialway"]{around};'
            + f'way["piste:type"]{around};);'
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
        logger.info(f"Overpass returned {len(elements)} elements for region {region}")
        return elements

    def convert(self, region: Region, elements: list[dict[str, Any]]) -> ImportSummary:
        """Turn raw Overpass elements into import-ready pistes + lifts for the given circle.

        `region` is the (center, radius) the user chose; only ways fully inside the circle are kept.
        """
        summary = ImportSummary()
        for el in elements:
            tags = el.get("tags", {})
            vertices = [(v["lon"], v["lat"]) for v in el.get("geometry", [])]
            if "aerialway" in tags:
                self._add_lift(tags, vertices, region, summary)
            elif "piste:type" in tags:
                self._add_piste(tags, vertices, region, summary)
        logger.info(f"Converted: {len(summary.pistes)} pistes, {len(summary.lifts)} lifts, {summary.skipped} skipped")
        return summary

    # -- pistes ---------------------------------------------------------------

    def _add_piste(self, tags: dict[str, Any], vertices: list[Vertex], region: Region, summary: ImportSummary) -> None:
        if tags.get("piste:type") != OSMConfig.PISTE_TYPE_DOWNHILL:
            return  # only alpine downhill runs; ignore connection/snow_park/playground/sled/yes
        if len(vertices) < 2 or not _fully_inside(vertices, region):
            summary.skipped += 1
            return
        if _polyline_length_m(vertices) < OSMConfig.MIN_PISTE_LENGTH_M:
            summary.skipped += 1  # stub run below the minimum importable length
            return
        points = self._resample(vertices)
        if points is None:
            summary.skipped += 1  # a nodata hole under the run — not fully importable
            return
        summary.pistes.append(PisteImport(points=points, name=_piste_name(tags)))

    # -- lifts ----------------------------------------------------------------

    def _add_lift(self, tags: dict[str, Any], vertices: list[Vertex], region: Region, summary: ImportSummary) -> None:
        lift_type = OSMConfig.AERIALWAY_TO_LIFT_TYPE.get(tags["aerialway"])
        if lift_type is None:
            return  # only mapped aerialway values import; everything else (station/pylon/…) is ignored
        if len(vertices) < 2 or not _fully_inside(vertices, region):
            summary.skipped += 1
            return
        if _polyline_length_m(vertices) < OSMConfig.MIN_LIFT_LENGTH_M:
            summary.skipped += 1  # nursery/kiddie lift below the minimum importable length
            return
        # Only the two stations matter; OSM intermediate pylons are dropped (we regenerate them).
        bottom = self._point(vertices[0])
        top = self._point(vertices[-1])
        if bottom is None or top is None:
            summary.skipped += 1
            return
        # A lift runs valley→mountain; orient bottom = lower station.
        if bottom.elevation > top.elevation:
            bottom, top = top, bottom
        summary.lifts.append(LiftImport(bottom=bottom, top=top, lift_type=lift_type, name=_lift_name(tags)))

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


def _fully_inside(vertices: list[Vertex], region: Region) -> bool:
    """True if every vertex lies within the circle (so the way is not truncated by it)."""
    center_lon, center_lat, radius_m = region
    return all(
        GeoCalculator.haversine_distance_m(lat1=center_lat, lon1=center_lon, lat2=lat, lon2=lon) <= radius_m
        for lon, lat in vertices
    )


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
    """OSM name resolution order; None if the run is unnamed (39% of real runs are)."""
    for key in ("name", "piste:name", "piste:ref", "ref"):
        value = tags.get(key)
        if value:
            return str(value)
    return None


def _lift_name(tags: dict[str, Any]) -> str | None:
    """OSM lift name resolution; None if unnamed (~80% of real lifts are)."""
    for key in ("name", "ref"):
        value = tags.get(key)
        if value:
            return str(value)
    return None

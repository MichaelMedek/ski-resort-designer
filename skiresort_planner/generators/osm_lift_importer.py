"""Lift-only OSM import: raw lifts exactly as OSM has them, no slopes, no connected-graph algorithm.

The fast, faithful import — fetch the box, extract the named skiable lifts (min-length/type filter),
and return them untouched. Use `GraphImporter` (osm_graph_builder.py) when you also want slopes and
the connected-graph preprocessing.

Lift EXTRACTION POLICY (drop unnamed, per-section min-length, DEM-drape stations, dedup) lives here:
only this child needs it — `GraphImporter` hub-merges the shared sections instead.
"""

import logging
from dataclasses import dataclass

from skiresort_planner.constants import OSMConfig
from skiresort_planner.core.geo_calculator import GeoCalculator
from skiresort_planner.generators.osm_importer import (
    BaseOSMImporter,
    ImportResult,
    OverpassElement,
    ProgressFn,
    Vertex,
    extract_lift_sections,
)
from skiresort_planner.model.path_point import PathPoint, endpoints_match

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LiftImport:
    """One lift ready to add: its two stations (DEM-sampled), lift-type, + optional OSM name."""

    bottom: PathPoint
    top: PathPoint
    lift_type: str
    name: str | None

    @property
    def length_m(self) -> float:
        """Ground length of the lift line (bottom station to top station)."""
        return self.bottom.distance_to(other=self.top)


class LiftOnlyImporter(BaseOSMImporter):
    """Imports ONLY lifts, as raw OSM geometry (the two stations). No pistes, no graph build."""

    def _assemble(self, elements: list[OverpassElement], on_progress: ProgressFn) -> ImportResult:
        on_progress(0.0, "Extracting lifts…")  # lift-only build is fast — one marker is enough
        lifts, skipped = self._extract_lifts(elements)
        logger.info(f"OSM lift-only import: {len(lifts)} lifts, {skipped} skipped")
        return ImportResult(
            lifts=[(lf.bottom, lf.top, lf.lift_type, lf.name) for lf in lifts],
            slope_chains=[],
            source=self.SOURCE,
            skipped=skipped,
        )

    def _extract_lifts(self, elements: list[OverpassElement]) -> tuple[list[LiftImport], int]:
        """Shared lift sections → raw-OSM LiftImports, with the LIFT-ONLY policy: drop unnamed,
        drop < MIN_LIFT_LENGTH_M, DEM-drape both stations (drop over a nodata hole), dedup coincident
        same-name duplicates. Returns (lifts, skipped count).
        """
        lifts: list[LiftImport] = []
        skipped = 0
        for vertices, lift_type, name in extract_lift_sections(elements, self.bbox):
            lift = self._lift_import(vertices=vertices, lift_type=lift_type, name=name)
            if lift is None:
                skipped += 1
            else:
                lifts.append(lift)
        lifts, dropped = _dedupe_coincident_per_name(lifts)
        return lifts, skipped + dropped

    def _lift_import(self, vertices: list[Vertex], lift_type: str, name: str | None) -> LiftImport | None:
        """One shared lift section → LiftImport, or None (logged) if the lift-only policy skips it."""
        if name is None:
            logger.debug("Skipped a lift section: unnamed (potentially outdated/duplicate — only named lifts import)")
            return None
        length = _polyline_length_m(vertices)
        if length < OSMConfig.MIN_LIFT_LENGTH_M:
            logger.debug(f"Skipped lift '{name}': {length:.0f}m < {OSMConfig.MIN_LIFT_LENGTH_M:.0f}m min")
            return None
        # A lift section's two stations are its endpoints; OSM intermediate pylons are dropped (we
        # regenerate them). Interior mid-STATIONS were already split out into their own sections.
        bottom = self._point(vertices[0])
        top = self._point(vertices[-1])
        if bottom is None or top is None:
            logger.debug(f"Skipped lift '{name}': a station is over a DEM nodata hole")
            return None
        # A lift runs valley→mountain; orient bottom = lower station.
        if bottom.elevation > top.elevation:
            bottom, top = top, bottom
        return LiftImport(bottom=bottom, top=top, lift_type=lift_type, name=name)

    def _point(self, vertex: Vertex) -> PathPoint | None:
        lon, lat = vertex
        elev = self.dem.get_elevation(lon=lon, lat=lat)
        if elev is None:
            return None
        return PathPoint(lon=lon, lat=lat, elevation=elev)


def _polyline_length_m(vertices: list[Vertex]) -> float:
    """Total ground length of the raw lon/lat polyline in metres (sum of haversine legs)."""
    return sum(
        GeoCalculator.haversine_distance_m(
            lat1=vertices[i][1], lon1=vertices[i][0], lat2=vertices[i + 1][1], lon2=vertices[i + 1][0]
        )
        for i in range(len(vertices) - 1)
    )


def _dedupe_coincident_per_name(items: list[LiftImport]) -> tuple[list[LiftImport], int]:
    """Drop same-name lifts that are GEOMETRICALLY COINCIDENT with a longer one (an old + a re-drawn
    OSM way share a name AND both endpoints); keep the longest of each such duplicate group.

    A same name alone is NOT a duplicate: consecutive mid-station sections (and separate same-named
    ways meeting end-to-end) share only ONE endpoint, so they are kept as distinct lifts.
    """
    kept: list[LiftImport] = []
    dropped = 0
    for item in sorted(items, key=lambda lf: lf.length_m, reverse=True):
        twin = next(
            (
                k
                for k in kept
                if k.name == item.name
                and endpoints_match(
                    pair_a=(k.bottom, k.top), pair_b=(item.bottom, item.top), tol_m=OSMConfig.OSM_DEDUP_TOL_M
                )
            ),
            None,
        )
        if twin is None:
            kept.append(item)
        else:
            dropped += 1
            logger.debug(
                f"Skipped lift '{item.name}': coincident duplicate, {item.length_m:.0f}m ≤ kept {twin.length_m:.0f}m"
            )
    return kept, dropped

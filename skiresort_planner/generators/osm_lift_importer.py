"""Lift-only OSM import: raw lifts exactly as OSM has them, no slopes, no connected-graph algorithm.

The fast, faithful import — fetch the box, extract the named skiable lifts (min-length/type filter),
and return them untouched. Use `GraphImporter` (osm_graph_builder.py) when you also want slopes and
the connected-graph preprocessing.
"""

import logging

from skiresort_planner.generators.osm_importer import BaseOSMImporter, ImportResult, OverpassElement

logger = logging.getLogger(__name__)


class LiftOnlyImporter(BaseOSMImporter):
    """Imports ONLY lifts, as raw OSM geometry (the two stations). No pistes, no graph build."""

    def _assemble(self, elements: list[OverpassElement]) -> ImportResult:
        lifts, skipped = self.extract_lifts(elements)
        logger.info(f"OSM lift-only import: {len(lifts)} lifts, {skipped} skipped")
        return ImportResult(
            lifts=[(lf.bottom, lf.top, lf.lift_type, lf.name) for lf in lifts],
            slope_chains=[],
            source=self.SOURCE,
            skipped=skipped,
        )
